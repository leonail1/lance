// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use ndarray::ArrayView2;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::{Eligibility, Error, PlaidIndex, Result};

/// Query-time controls for the PLAID candidate and reranking pipeline.
#[derive(Clone, Debug)]
pub struct PlaidSearchParams {
    /// Number of coarse centroids selected per query token.
    pub n_ivf_probe: usize,
    /// Number of approximate candidates retained before residual reranking.
    pub n_full_scores: usize,
    /// Number of final documents returned.
    pub top_k: usize,
    /// Minimum maximum query-token similarity required for a probed centroid.
    pub centroid_score_threshold: Option<f32>,
}

impl Default for PlaidSearchParams {
    fn default() -> Self {
        Self {
            n_ivf_probe: 8,
            n_full_scores: 4096,
            top_k: 10,
            centroid_score_threshold: Some(0.4),
        }
    }
}

/// Per-query phase counters and CPU durations returned by the PLAID kernel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlaidSearchStats {
    /// Nanoseconds spent scoring and selecting coarse centroids.
    pub centroid_probe_nanos: u64,
    /// Number of distinct coarse centroids probed.
    pub centroids_probed: u64,
    /// Nanoseconds spent expanding postings and applying eligibility.
    pub postings_nanos: u64,
    /// Posting entries visited before eligibility filtering.
    pub posting_entries_read: u64,
    /// Posting entries that survived eligibility filtering.
    pub posting_entries_eligible: u64,
    /// Number of unique eligible candidate documents.
    pub candidate_documents: u64,
    /// Nanoseconds spent computing code-only approximate scores.
    pub approximate_score_nanos: u64,
    /// Number of documents receiving approximate scores.
    pub approximate_documents: u64,
    /// Nanoseconds spent reconstructing residuals and computing MaxSim.
    pub exact_score_nanos: u64,
    /// Number of documents receiving reconstructed MaxSim scores.
    pub exact_documents: u64,
    /// Nanoseconds spent sorting approximate and final results.
    pub sort_nanos: u64,
    /// Total kernel wall-clock time in nanoseconds.
    pub total_nanos: u64,
}

/// One ranked database row returned by the PLAID kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    /// Dense document ordinal internal to the index segment.
    pub document_ordinal: u32,
    /// Database row address associated with the document.
    pub row_address: u64,
    /// Reconstructed MaxSim similarity, where larger is better.
    pub score: f32,
}

impl PlaidIndex {
    /// Runs centroid probing, posting expansion, code-only scoring, and
    /// quantized-residual MaxSim reranking for one multi-vector query.
    pub fn search(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
        eligibility: &dyn Eligibility,
    ) -> Result<(Vec<SearchHit>, PlaidSearchStats)> {
        self.validate_search(query, params)?;
        let total_started = Instant::now();
        let mut stats = PlaidSearchStats::default();

        let probe_started = Instant::now();
        let query_centroid_scores = query.dot(&self.centroids.t());
        let selected_centroids = select_centroids(
            query_centroid_scores.view(),
            params.n_ivf_probe,
            params.centroid_score_threshold,
        );
        stats.centroid_probe_nanos = duration_nanos(probe_started.elapsed());
        stats.centroids_probed = selected_centroids.len();

        let postings_started = Instant::now();
        let mut candidate_documents = RoaringBitmap::new();
        for centroid in selected_centroids {
            let posting_range = self.posting_range(centroid)?;
            stats.posting_entries_read = stats
                .posting_entries_read
                .saturating_add(posting_range.len() as u64);
            for document_ordinal in &self.postings[posting_range] {
                let row_address = self.row_addresses[*document_ordinal as usize];
                if eligibility.includes(*document_ordinal, row_address) {
                    stats.posting_entries_eligible =
                        stats.posting_entries_eligible.saturating_add(1);
                    candidate_documents.insert(*document_ordinal);
                }
            }
        }
        stats.candidate_documents = candidate_documents.len();
        stats.postings_nanos = duration_nanos(postings_started.elapsed());

        let approximate_started = Instant::now();
        let candidates = candidate_documents.iter().collect::<Vec<_>>();
        let mut approximate_scores = candidates
            .par_iter()
            .map(|document_ordinal| {
                let token_range = self.document_token_range(*document_ordinal)?;
                let score =
                    approximate_score(query_centroid_scores.view(), &self.token_codes[token_range]);
                Ok((*document_ordinal, score))
            })
            .collect::<Result<Vec<_>>>()?;
        stats.approximate_score_nanos = duration_nanos(approximate_started.elapsed());
        stats.approximate_documents = approximate_scores.len() as u64;

        let sort_started = Instant::now();
        approximate_scores.sort_unstable_by(|left, right| {
            compare_ranked(
                left.1,
                self.row_addresses[left.0 as usize],
                right.1,
                self.row_addresses[right.0 as usize],
            )
        });
        approximate_scores.truncate(params.n_full_scores.min(approximate_scores.len()));
        stats.sort_nanos = duration_nanos(sort_started.elapsed());

        let exact_count = (params.n_full_scores / 4)
            .max(params.top_k)
            .min(approximate_scores.len());
        let exact_started = Instant::now();
        let mut exact_scores = approximate_scores[..exact_count]
            .par_iter()
            .map(|(document_ordinal, _)| {
                Ok((
                    *document_ordinal,
                    self.quantized_maxsim(query, *document_ordinal)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        stats.exact_score_nanos = duration_nanos(exact_started.elapsed());
        stats.exact_documents = exact_scores.len() as u64;

        let sort_started = Instant::now();
        exact_scores.sort_unstable_by(|left, right| {
            compare_ranked(
                left.1,
                self.row_addresses[left.0 as usize],
                right.1,
                self.row_addresses[right.0 as usize],
            )
        });
        exact_scores.truncate(params.top_k.min(exact_scores.len()));
        stats.sort_nanos = stats
            .sort_nanos
            .saturating_add(duration_nanos(sort_started.elapsed()));

        let hits = exact_scores
            .into_iter()
            .map(|(document_ordinal, score)| SearchHit {
                document_ordinal,
                row_address: self.row_addresses[document_ordinal as usize],
                score,
            })
            .collect();
        stats.total_nanos = duration_nanos(total_started.elapsed());
        Ok((hits, stats))
    }

    fn validate_search(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
    ) -> Result<()> {
        if query.nrows() == 0 || query.ncols() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "query shape must be [positive, {}], got [{}, {}]",
                self.dimension,
                query.nrows(),
                query.ncols()
            )));
        }
        if query.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "query values must be finite".to_string(),
            ));
        }
        if params.n_ivf_probe == 0 || params.n_full_scores == 0 || params.top_k == 0 {
            return Err(Error::InvalidInput(format!(
                "n_ivf_probe, n_full_scores, and top_k must be positive, got {}, {}, {}",
                params.n_ivf_probe, params.n_full_scores, params.top_k
            )));
        }
        if let Some(threshold) = params.centroid_score_threshold
            && !threshold.is_finite()
        {
            return Err(Error::InvalidInput(
                "centroid_score_threshold must be finite when present".to_string(),
            ));
        }
        Ok(())
    }
}

fn select_centroids(
    query_centroid_scores: ArrayView2<'_, f32>,
    n_ivf_probe: usize,
    threshold: Option<f32>,
) -> RoaringBitmap {
    let mut selected = RoaringBitmap::new();
    for token_scores in query_centroid_scores.outer_iter() {
        let mut centroid_ids = (0..token_scores.len()).collect::<Vec<_>>();
        let keep = n_ivf_probe.min(centroid_ids.len());
        if keep < centroid_ids.len() {
            centroid_ids.select_nth_unstable_by(keep - 1, |left, right| {
                token_scores[*right]
                    .total_cmp(&token_scores[*left])
                    .then_with(|| left.cmp(right))
            });
        }
        for centroid in centroid_ids.into_iter().take(keep) {
            if threshold.is_none_or(|minimum| token_scores[centroid] >= minimum) {
                selected.insert(centroid as u32);
            }
        }
    }
    selected
}

fn approximate_score(query_centroid_scores: ArrayView2<'_, f32>, document_codes: &[u32]) -> f32 {
    let mut score = 0.0_f32;
    for query_token_scores in query_centroid_scores.outer_iter() {
        let mut maximum = f32::NEG_INFINITY;
        for centroid in document_codes {
            let similarity = query_token_scores[*centroid as usize];
            if similarity.total_cmp(&maximum).is_gt() {
                maximum = similarity;
            }
        }
        score += maximum;
    }
    score
}

fn compare_ranked(
    left_score: f32,
    left_row_address: u64,
    right_score: f32,
    right_row_address: u64,
) -> Ordering {
    right_score
        .total_cmp(&left_score)
        .then_with(|| left_row_address.cmp(&right_row_address))
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

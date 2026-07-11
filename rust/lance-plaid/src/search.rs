// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use ndarray::ArrayView2;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::{Eligibility, Error, PlaidIndex, Result};

// Building an eligible-centroid set touches every token code selected by the
// filter. Keep this deliberately conservative: the path is intended for the
// very selective predicates where posting expansion is otherwise dominant.
const ELIGIBLE_MAX_DOCUMENT_FRACTION_DENOMINATOR: usize = 32;
const ELIGIBLE_TOKEN_SCAN_SAFETY_FACTOR: u64 = 4;

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

/// Cost-model decision for filter-aware centroid selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EligibleCentroidDecision {
    /// The query has no structured filter, so the global centroid path is used.
    #[default]
    Unfiltered,
    /// The filter is not an enumerable allow-list (for example a block mask).
    NonEnumerable,
    /// The filter selects no indexed documents.
    Empty,
    /// The selected document fraction is too wide for an upfront token scan.
    TooWide,
    /// The estimated token-scan cost is not safely below posting expansion.
    ScanCostTooHigh,
    /// Only centroids referenced by eligible documents participate in probing.
    Enabled,
}

/// Reusable filter-aware centroid plan for one immutable PLAID segment.
///
/// The plan is query-shape and initial-probe specific, but independent of the
/// actual query values. It can therefore be built once and reused by all
/// incremental underfill probe rounds.
#[derive(Clone, Debug)]
pub struct EligibleCentroidPlan {
    index_instance_id: u64,
    decision: EligibleCentroidDecision,
    eligible_documents: u64,
    eligible_tokens: u64,
    eligible_token_codes_scanned: u64,
    eligible_centroids: RoaringBitmap,
    estimated_global_postings: u64,
    core_build_nanos: u64,
}

impl EligibleCentroidPlan {
    /// Cost-model outcome for this plan.
    pub fn decision(&self) -> EligibleCentroidDecision {
        self.decision
    }

    /// Whether probing is restricted to the plan's eligible centroid set.
    pub fn is_enabled(&self) -> bool {
        matches!(
            self.decision,
            EligibleCentroidDecision::Enabled | EligibleCentroidDecision::Empty
        )
    }

    /// Exact number of indexed documents in the enumerable filter.
    pub fn eligible_documents(&self) -> u64 {
        self.eligible_documents
    }

    /// Number of token codes inspected (or estimated before a rejected plan).
    pub fn eligible_tokens(&self) -> u64 {
        self.eligible_tokens
    }

    /// Number of token codes actually scanned to build the centroid bitmap.
    pub fn eligible_token_codes_scanned(&self) -> u64 {
        self.eligible_token_codes_scanned
    }

    /// Number of distinct centroids referenced by eligible documents.
    pub fn eligible_centroids(&self) -> u64 {
        self.eligible_centroids.len()
    }

    /// Estimated posting entries for the existing global probing path.
    pub fn estimated_global_postings(&self) -> u64 {
        self.estimated_global_postings
    }

    /// Core CPU wall time spent validating ordinals and scanning token codes.
    ///
    /// Database adapters may expose a wider total-plan metric that also
    /// includes address-to-ordinal materialization. This value is its nested
    /// core sub-phase.
    pub fn core_build_nanos(&self) -> u64 {
        self.core_build_nanos
    }

    fn selection_universe(&self) -> Option<&RoaringBitmap> {
        self.is_enabled().then_some(&self.eligible_centroids)
    }
}

/// Per-query phase counters and CPU durations returned by the PLAID kernel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlaidSearchStats {
    /// Nanoseconds spent scoring and selecting coarse centroids.
    pub centroid_probe_nanos: u64,
    /// Nanoseconds spent selecting within an eligible-centroid universe.
    pub eligible_centroid_selection_nanos: u64,
    /// Number of distinct coarse centroids probed.
    pub centroids_probed: u64,
    /// Number of centroid selection/expansion rounds.
    pub probe_rounds: u64,
    /// Number of underfill rounds after the initial probe.
    pub probe_retries: u64,
    /// Sum of configured per-token nprobe values across all rounds.
    pub configured_probes: u64,
    /// Final per-token nprobe reached by the adaptive search.
    pub final_nprobe: u64,
    /// Previously expanded centroids skipped by incremental retry rounds.
    pub incremental_centroids_reused: u64,
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
    /// Builds a conservative, reusable eligible-centroid plan.
    ///
    /// `eligible_document_ordinals` must be `Some` only for an exact,
    /// enumerable allow-list after database deletion visibility has been
    /// applied. `eligible_document_count_hint` lets an adapter reject a wide
    /// enumerable filter before allocating its ordinal vector.
    pub fn plan_eligible_centroids(
        &self,
        eligible_document_ordinals: Option<&[u32]>,
        eligible_document_count_hint: Option<usize>,
        filtered: bool,
        query_tokens: usize,
        initial_nprobe: usize,
    ) -> Result<EligibleCentroidPlan> {
        let started = Instant::now();
        let mut plan = EligibleCentroidPlan {
            index_instance_id: self.instance_id(),
            decision: if filtered {
                EligibleCentroidDecision::NonEnumerable
            } else {
                EligibleCentroidDecision::Unfiltered
            },
            eligible_documents: 0,
            eligible_tokens: 0,
            eligible_token_codes_scanned: 0,
            eligible_centroids: RoaringBitmap::new(),
            estimated_global_postings: 0,
            core_build_nanos: 0,
        };

        if !filtered {
            plan.core_build_nanos = duration_nanos(started.elapsed());
            return Ok(plan);
        }
        if query_tokens == 0 || initial_nprobe == 0 {
            return Err(Error::InvalidInput(format!(
                "query_tokens and initial_nprobe must be positive, got {query_tokens} and {initial_nprobe}"
            )));
        }
        if eligible_document_count_hint.is_some_and(|count| count > self.num_documents()) {
            return Err(Error::InvalidInput(format!(
                "eligible document count exceeds index size {}",
                self.num_documents()
            )));
        }
        plan.eligible_documents = eligible_document_count_hint.unwrap_or(0) as u64;
        let Some(document_ordinals) = eligible_document_ordinals else {
            if eligible_document_count_hint == Some(0) {
                plan.decision = EligibleCentroidDecision::Empty;
            } else if eligible_document_count_hint.is_some_and(|count| {
                count.saturating_mul(ELIGIBLE_MAX_DOCUMENT_FRACTION_DENOMINATOR)
                    > self.num_documents()
            }) {
                plan.decision = EligibleCentroidDecision::TooWide;
            }
            plan.core_build_nanos = duration_nanos(started.elapsed());
            return Ok(plan);
        };

        let mut document_ordinals = document_ordinals.to_vec();
        document_ordinals.sort_unstable();
        document_ordinals.dedup();
        if eligible_document_count_hint.is_some_and(|count| count != document_ordinals.len()) {
            return Err(Error::InvalidInput(format!(
                "eligible document count hint differs from {} unique ordinals",
                document_ordinals.len()
            )));
        }
        plan.eligible_documents = document_ordinals.len() as u64;
        if document_ordinals.is_empty() {
            plan.decision = EligibleCentroidDecision::Empty;
            plan.core_build_nanos = duration_nanos(started.elapsed());
            return Ok(plan);
        }

        if document_ordinals
            .len()
            .saturating_mul(ELIGIBLE_MAX_DOCUMENT_FRACTION_DENOMINATOR)
            > self.num_documents()
        {
            plan.decision = EligibleCentroidDecision::TooWide;
            plan.core_build_nanos = duration_nanos(started.elapsed());
            return Ok(plan);
        }

        for document_ordinal in &document_ordinals {
            let token_range = self.document_token_range(*document_ordinal)?;
            plan.eligible_tokens = plan
                .eligible_tokens
                .saturating_add(token_range.len() as u64);
        }
        let estimated_selected_centroids = query_tokens
            .saturating_mul(initial_nprobe)
            .min(self.num_centroids());
        plan.estimated_global_postings = (self.postings.len() as u64)
            .saturating_mul(estimated_selected_centroids as u64)
            .div_ceil(self.num_centroids() as u64);

        if plan
            .eligible_tokens
            .saturating_mul(ELIGIBLE_TOKEN_SCAN_SAFETY_FACTOR)
            > plan.estimated_global_postings
        {
            plan.decision = EligibleCentroidDecision::ScanCostTooHigh;
            plan.core_build_nanos = duration_nanos(started.elapsed());
            return Ok(plan);
        }

        for document_ordinal in document_ordinals {
            let token_range = self.document_token_range(document_ordinal)?;
            for centroid in &self.token_codes[token_range] {
                plan.eligible_centroids.insert(*centroid);
                plan.eligible_token_codes_scanned =
                    plan.eligible_token_codes_scanned.saturating_add(1);
            }
        }
        plan.decision = EligibleCentroidDecision::Enabled;
        plan.core_build_nanos = duration_nanos(started.elapsed());
        Ok(plan)
    }

    /// Scores an exact, sorted set of document ordinals directly with the
    /// quantized-residual MaxSim kernel.
    ///
    /// This is equivalent to the tail of [`Self::search_adaptive`] only when
    /// its caller has already proved that the legacy pipeline would retain and
    /// residual-score every supplied document. Returning `None` for an empty
    /// document lets database adapters conservatively fall back to that legacy
    /// pipeline: empty documents have no posting and are therefore not normal
    /// PLAID candidates.
    pub fn search_quantized_residuals(
        &self,
        query: ArrayView2<'_, f32>,
        document_ordinals: &[u32],
    ) -> Result<Option<(Vec<SearchHit>, PlaidSearchStats)>> {
        let total_started = Instant::now();
        self.validate_query(query)?;
        if document_ordinals.is_empty() {
            return Err(Error::InvalidInput(
                "direct quantized-residual search requires at least one document".to_string(),
            ));
        }
        if document_ordinals.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidInput(
                "direct quantized-residual ordinals must be sorted and unique".to_string(),
            ));
        }
        for document_ordinal in document_ordinals {
            if self.document_token_range(*document_ordinal)?.is_empty() {
                return Ok(None);
            }
        }

        let mut stats = PlaidSearchStats {
            candidate_documents: document_ordinals.len() as u64,
            ..Default::default()
        };
        let exact_started = Instant::now();
        let mut exact_scores = document_ordinals
            .par_iter()
            .map(|document_ordinal| {
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
        stats.sort_nanos = duration_nanos(sort_started.elapsed());
        let hits = exact_scores
            .into_iter()
            .map(|(document_ordinal, score)| SearchHit {
                document_ordinal,
                row_address: self.row_addresses[document_ordinal as usize],
                score,
            })
            .collect();
        stats.total_nanos = duration_nanos(total_started.elapsed());
        Ok(Some((hits, stats)))
    }

    /// Runs centroid probing, posting expansion, code-only scoring, and
    /// quantized-residual MaxSim reranking for one multi-vector query.
    pub fn search(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
        eligibility: &dyn Eligibility,
    ) -> Result<(Vec<SearchHit>, PlaidSearchStats)> {
        self.search_adaptive(query, params, params.n_ivf_probe, 0, eligibility, None)
    }

    /// Runs PLAID search with incremental nprobe expansion on underfill.
    ///
    /// Query-centroid scores, already expanded postings, and candidate
    /// documents are retained across retries. Each posting list is therefore
    /// visited at most once even when nprobe doubles multiple times.
    pub fn search_adaptive(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
        maximum_n_ivf_probe: usize,
        desired_candidates: usize,
        eligibility: &dyn Eligibility,
        eligible_centroid_plan: Option<&EligibleCentroidPlan>,
    ) -> Result<(Vec<SearchHit>, PlaidSearchStats)> {
        self.validate_search(query, params)?;
        if maximum_n_ivf_probe == 0 {
            return Err(Error::InvalidInput(
                "maximum_n_ivf_probe must be positive".to_string(),
            ));
        }
        if eligible_centroid_plan.is_some_and(|plan| plan.index_instance_id != self.instance_id()) {
            return Err(Error::InvalidInput(
                "eligible-centroid plan does not belong to this index".to_string(),
            ));
        }
        let total_started = Instant::now();
        let mut stats = PlaidSearchStats::default();
        if eligible_centroid_plan
            .is_some_and(|plan| plan.decision == EligibleCentroidDecision::Empty)
        {
            stats.total_nanos = duration_nanos(total_started.elapsed());
            return Ok((Vec::new(), stats));
        }

        let probe_started = Instant::now();
        let query_centroid_scores = query.dot(&self.centroids.t());
        stats.centroid_probe_nanos = duration_nanos(probe_started.elapsed());
        let selection_universe = eligible_centroid_plan.and_then(|plan| plan.selection_universe());
        let probe_ceiling = maximum_n_ivf_probe.min(
            selection_universe.map_or(self.num_centroids(), |centroids| centroids.len() as usize),
        );
        let mut current_nprobe = params.n_ivf_probe.min(probe_ceiling).max(1);
        let mut expanded_centroids = RoaringBitmap::new();
        let mut candidate_documents = RoaringBitmap::new();

        if probe_ceiling > 0 {
            loop {
                let selection_started = Instant::now();
                let selected_centroids = select_centroids(
                    query_centroid_scores.view(),
                    current_nprobe,
                    params.centroid_score_threshold,
                    selection_universe,
                );
                let selection_nanos = duration_nanos(selection_started.elapsed());
                stats.centroid_probe_nanos =
                    stats.centroid_probe_nanos.saturating_add(selection_nanos);
                if selection_universe.is_some() {
                    stats.eligible_centroid_selection_nanos = stats
                        .eligible_centroid_selection_nanos
                        .saturating_add(selection_nanos);
                }
                stats.probe_rounds = stats.probe_rounds.saturating_add(1);
                stats.configured_probes = stats
                    .configured_probes
                    .saturating_add(current_nprobe as u64);
                stats.final_nprobe = current_nprobe as u64;

                let postings_started = Instant::now();
                for centroid in selected_centroids {
                    if !expanded_centroids.insert(centroid) {
                        stats.incremental_centroids_reused =
                            stats.incremental_centroids_reused.saturating_add(1);
                        continue;
                    }
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
                stats.postings_nanos = stats
                    .postings_nanos
                    .saturating_add(duration_nanos(postings_started.elapsed()));

                if desired_candidates == 0
                    || candidate_documents.len() as usize >= desired_candidates
                    || current_nprobe >= probe_ceiling
                {
                    break;
                }
                let next_nprobe = current_nprobe.saturating_mul(2).min(probe_ceiling).max(1);
                if next_nprobe == current_nprobe {
                    break;
                }
                current_nprobe = next_nprobe;
                stats.probe_retries = stats.probe_retries.saturating_add(1);
            }
        }
        stats.centroids_probed = expanded_centroids.len();
        stats.candidate_documents = candidate_documents.len();

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

    fn validate_query(&self, query: ArrayView2<'_, f32>) -> Result<()> {
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
        Ok(())
    }

    fn validate_search(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
    ) -> Result<()> {
        self.validate_query(query)?;
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
    eligible_centroids: Option<&RoaringBitmap>,
) -> RoaringBitmap {
    let mut selected = RoaringBitmap::new();
    let centroid_universe = eligible_centroids.map(|centroids| {
        centroids
            .iter()
            .map(|centroid| centroid as usize)
            .collect::<Vec<_>>()
    });
    for token_scores in query_centroid_scores.outer_iter() {
        let mut centroid_ids = centroid_universe
            .clone()
            .unwrap_or_else(|| (0..token_scores.len()).collect::<Vec<_>>());
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

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::{Array2, ArrayView2};

use crate::{Error, ResidualQuantizer, Result, maxsim_naive};

// Runtime identity for exact eligible-centroid plan ownership checks.
static NEXT_PLAID_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn next_plaid_instance_id() -> u64 {
    NEXT_PLAID_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Exact values stored in document-ordinal order for database-native refinement.
#[derive(Clone, Debug)]
pub struct ExactStore {
    pub public_row_ids: Vec<u64>,
    pub raw_token_values: Vec<f32>,
}

/// Immutable PLAID data owned by one database index segment.
#[derive(Debug)]
pub struct PlaidIndex {
    instance_id: u64,
    pub(crate) dimension: usize,
    pub(crate) centroids: Array2<f32>,
    pub(crate) quantizer: ResidualQuantizer,
    pub(crate) row_addresses: Vec<u64>,
    pub(crate) document_offsets: Vec<u64>,
    pub(crate) token_codes: Vec<u32>,
    pub(crate) packed_residuals: Vec<u8>,
    pub(crate) posting_offsets: Vec<u64>,
    pub(crate) postings: Vec<u32>,
    pub(crate) exact_store: Option<ExactStore>,
}

impl Clone for PlaidIndex {
    fn clone(&self) -> Self {
        Self {
            instance_id: next_plaid_instance_id(),
            dimension: self.dimension,
            centroids: self.centroids.clone(),
            quantizer: self.quantizer.clone(),
            row_addresses: self.row_addresses.clone(),
            document_offsets: self.document_offsets.clone(),
            token_codes: self.token_codes.clone(),
            packed_residuals: self.packed_residuals.clone(),
            posting_offsets: self.posting_offsets.clone(),
            postings: self.postings.clone(),
            exact_store: self.exact_store.clone(),
        }
    }
}

impl PlaidIndex {
    /// Builds an immutable PLAID segment from already encoded document tokens.
    ///
    /// Row addresses must be strictly increasing so address-to-ordinal lookup
    /// remains allocation-free. `document_offsets` indexes token rows and must
    /// contain one more entry than `row_addresses`.
    pub fn try_new(
        centroids: Array2<f32>,
        quantizer: ResidualQuantizer,
        row_addresses: Vec<u64>,
        document_offsets: Vec<u64>,
        token_codes: Vec<u32>,
        packed_residuals: Vec<u8>,
    ) -> Result<Self> {
        let (posting_offsets, postings) =
            Self::build_postings(centroids.nrows(), &document_offsets, &token_codes)?;
        Self::try_from_parts(
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
        )
    }

    /// Builds an immutable PLAID segment with an index-resident exact store.
    ///
    /// `public_row_ids` and raw token rows are both aligned with the same dense
    /// document ordinals as `row_addresses` and `document_offsets`. The raw
    /// values are flattened row-major with exactly `num_tokens * dimension`
    /// finite `f32` values.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_exact_store(
        centroids: Array2<f32>,
        quantizer: ResidualQuantizer,
        row_addresses: Vec<u64>,
        document_offsets: Vec<u64>,
        token_codes: Vec<u32>,
        packed_residuals: Vec<u8>,
        public_row_ids: Vec<u64>,
        raw_token_values: Vec<f32>,
    ) -> Result<Self> {
        let (posting_offsets, postings) =
            Self::build_postings(centroids.nrows(), &document_offsets, &token_codes)?;
        Self::try_from_parts_with_exact_store(
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
            Some(ExactStore {
                public_row_ids,
                raw_token_values,
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_from_parts(
        centroids: Array2<f32>,
        quantizer: ResidualQuantizer,
        row_addresses: Vec<u64>,
        document_offsets: Vec<u64>,
        token_codes: Vec<u32>,
        packed_residuals: Vec<u8>,
        posting_offsets: Vec<u64>,
        postings: Vec<u32>,
    ) -> Result<Self> {
        Self::try_from_parts_with_exact_store(
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_from_parts_with_exact_store(
        centroids: Array2<f32>,
        quantizer: ResidualQuantizer,
        row_addresses: Vec<u64>,
        document_offsets: Vec<u64>,
        token_codes: Vec<u32>,
        packed_residuals: Vec<u8>,
        posting_offsets: Vec<u64>,
        postings: Vec<u32>,
        exact_store: Option<ExactStore>,
    ) -> Result<Self> {
        let dimension = centroids.ncols();
        let index = Self {
            instance_id: next_plaid_instance_id(),
            dimension,
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
            exact_store,
        };
        index.validate()?;
        Ok(index)
    }

    /// Vector dimension shared by centroids, query tokens, and document tokens.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub(crate) fn instance_id(&self) -> u64 {
        self.instance_id
    }

    /// Number of indexed documents.
    pub fn num_documents(&self) -> usize {
        self.row_addresses.len()
    }

    /// Number of indexed document tokens.
    pub fn num_tokens(&self) -> usize {
        self.token_codes.len()
    }

    /// Number of coarse centroids.
    pub fn num_centroids(&self) -> usize {
        self.centroids.nrows()
    }

    /// Coarse-centroid matrix with shape `[num_centroids, dimension]`.
    pub fn centroids(&self) -> ArrayView2<'_, f32> {
        self.centroids.view()
    }

    /// Residual quantizer used by this index segment.
    pub fn quantizer(&self) -> &ResidualQuantizer {
        &self.quantizer
    }

    /// Database row addresses in dense document-ordinal order.
    pub fn row_addresses(&self) -> &[u64] {
        &self.row_addresses
    }

    /// Whether this segment contains index-resident exact vectors.
    pub fn has_exact_store(&self) -> bool {
        self.exact_store.is_some()
    }

    /// Public row IDs in dense document-ordinal order, when present.
    pub fn exact_store_row_ids(&self) -> Option<&[u64]> {
        self.exact_store
            .as_ref()
            .map(|store| store.public_row_ids.as_slice())
    }

    /// Resolves a document ordinal to its exact-store public row ID.
    pub fn exact_public_row_id(&self, document_ordinal: u32) -> Option<u64> {
        self.exact_store
            .as_ref()?
            .public_row_ids
            .get(document_ordinal as usize)
            .copied()
    }

    /// Borrows one document's raw token matrix without copying.
    ///
    /// Returns `Ok(None)` for a V1 index without an exact store. An ordinal
    /// outside this segment is rejected even when no exact store is present.
    pub fn exact_document(&self, document_ordinal: u32) -> Result<Option<ArrayView2<'_, f32>>> {
        let token_range = self.document_token_range(document_ordinal)?;
        let Some(store) = &self.exact_store else {
            return Ok(None);
        };
        let value_start = token_range
            .start
            .checked_mul(self.dimension)
            .ok_or_else(|| {
                Error::InvalidInput("exact document value offset overflow".to_string())
            })?;
        let value_end = token_range.end.checked_mul(self.dimension).ok_or_else(|| {
            Error::InvalidInput("exact document value offset overflow".to_string())
        })?;
        let values = store
            .raw_token_values
            .get(value_start..value_end)
            .ok_or_else(|| {
                Error::InvalidInput("exact document value range is out of bounds".to_string())
            })?;
        Ok(Some(ArrayView2::from_shape(
            (token_range.len(), self.dimension),
            values,
        )?))
    }

    pub(crate) fn exact_store_parts(&self) -> Option<(&[u64], &[f32])> {
        self.exact_store.as_ref().map(|store| {
            (
                store.public_row_ids.as_slice(),
                store.raw_token_values.as_slice(),
            )
        })
    }

    /// Estimated in-memory bytes owned by the index's dense arrays.
    pub fn estimated_size_bytes(&self) -> usize {
        self.centroids.len() * std::mem::size_of::<f32>()
            + std::mem::size_of_val(self.quantizer.bucket_cutoffs())
            + std::mem::size_of_val(self.quantizer.bucket_weights())
            + self.row_addresses.len() * std::mem::size_of::<u64>()
            + self.document_offsets.len() * std::mem::size_of::<u64>()
            + self.token_codes.len() * std::mem::size_of::<u32>()
            + self.packed_residuals.len()
            + self.posting_offsets.len() * std::mem::size_of::<u64>()
            + self.postings.len() * std::mem::size_of::<u32>()
            + self
                .exact_store
                .as_ref()
                .map(|store| {
                    store.public_row_ids.len() * std::mem::size_of::<u64>()
                        + store.raw_token_values.len() * std::mem::size_of::<f32>()
                })
                .unwrap_or(0)
    }

    /// Resolves a dense document ordinal to its database row address.
    pub fn row_address(&self, document_ordinal: u32) -> Option<u64> {
        self.row_addresses.get(document_ordinal as usize).copied()
    }

    /// Resolves a database row address to its dense document ordinal.
    pub fn document_ordinal(&self, row_address: u64) -> Option<u32> {
        self.row_addresses
            .binary_search(&row_address)
            .ok()
            .and_then(|ordinal| u32::try_from(ordinal).ok())
    }

    /// Reconstructs all quantized token vectors for one document.
    pub fn reconstruct_document(&self, document_ordinal: u32) -> Result<Array2<f32>> {
        let token_range = self.document_token_range(document_ordinal)?;
        let num_tokens = token_range.end - token_range.start;
        let value_count = num_tokens.checked_mul(self.dimension).ok_or_else(|| {
            Error::InvalidInput("reconstructed document size overflow".to_string())
        })?;
        let mut values = Vec::with_capacity(value_count);
        let packed_len = self.quantizer.packed_len(self.dimension)?;

        for token_index in token_range {
            let centroid_index = self.token_codes[token_index] as usize;
            let centroid = self.centroids.row(centroid_index);
            let centroid = centroid.as_slice().ok_or_else(|| {
                Error::InvalidInput("centroid rows must be contiguous".to_string())
            })?;
            let packed_start = token_index.checked_mul(packed_len).ok_or_else(|| {
                Error::InvalidInput("packed residual offset overflow".to_string())
            })?;
            let mut token = vec![0.0_f32; self.dimension];
            self.quantizer.reconstruct_token(
                &self.packed_residuals[packed_start..packed_start + packed_len],
                centroid,
                &mut token,
            )?;
            values.extend(token);
        }
        Ok(Array2::from_shape_vec(
            (num_tokens, self.dimension),
            values,
        )?)
    }

    /// Computes MaxSim against one document after residual reconstruction.
    pub fn quantized_maxsim(
        &self,
        query: ArrayView2<'_, f32>,
        document_ordinal: u32,
    ) -> Result<f32> {
        if query.ncols() != self.dimension {
            return Err(Error::InvalidInput(format!(
                "query dimension must be {}, got {}",
                self.dimension,
                query.ncols()
            )));
        }
        let document = self.reconstruct_document(document_ordinal)?;
        Ok(maxsim_naive(query, document.view()))
    }

    pub(crate) fn document_token_range(
        &self,
        document_ordinal: u32,
    ) -> Result<std::ops::Range<usize>> {
        let ordinal = document_ordinal as usize;
        if ordinal >= self.num_documents() {
            return Err(Error::InvalidInput(format!(
                "document ordinal {document_ordinal} is out of range for {} documents",
                self.num_documents()
            )));
        }
        let start = usize::try_from(self.document_offsets[ordinal]).map_err(|_| {
            Error::InvalidInput("document token offset does not fit usize".to_string())
        })?;
        let end = usize::try_from(self.document_offsets[ordinal + 1]).map_err(|_| {
            Error::InvalidInput("document token offset does not fit usize".to_string())
        })?;
        Ok(start..end)
    }

    pub(crate) fn posting_range(&self, centroid: u32) -> Result<std::ops::Range<usize>> {
        let centroid = centroid as usize;
        if centroid >= self.num_centroids() {
            return Err(Error::InvalidInput(format!(
                "centroid {centroid} is out of range for {} centroids",
                self.num_centroids()
            )));
        }
        let start = usize::try_from(self.posting_offsets[centroid])
            .map_err(|_| Error::InvalidInput("posting offset does not fit usize".to_string()))?;
        let end = usize::try_from(self.posting_offsets[centroid + 1])
            .map_err(|_| Error::InvalidInput("posting offset does not fit usize".to_string()))?;
        Ok(start..end)
    }

    fn build_postings(
        num_centroids: usize,
        document_offsets: &[u64],
        token_codes: &[u32],
    ) -> Result<(Vec<u64>, Vec<u32>)> {
        let num_documents = document_offsets.len().checked_sub(1).ok_or_else(|| {
            Error::InvalidInput("document_offsets must contain at least one entry".to_string())
        })?;
        if num_documents > u32::MAX as usize {
            return Err(Error::InvalidInput(format!(
                "PLAID v1 supports at most {} documents, got {num_documents}",
                u32::MAX
            )));
        }
        let mut posting_lists = vec![Vec::<u32>::new(); num_centroids];
        for document_ordinal in 0..num_documents {
            let start = usize::try_from(document_offsets[document_ordinal]).map_err(|_| {
                Error::InvalidInput("document token offset does not fit usize".to_string())
            })?;
            let end = usize::try_from(document_offsets[document_ordinal + 1]).map_err(|_| {
                Error::InvalidInput("document token offset does not fit usize".to_string())
            })?;
            if start > end || end > token_codes.len() {
                return Err(Error::InvalidInput(format!(
                    "invalid token range {start}..{end} for document {document_ordinal}"
                )));
            }
            let mut unique_codes = token_codes[start..end].to_vec();
            unique_codes.sort_unstable();
            unique_codes.dedup();
            for centroid in unique_codes {
                let posting_list = posting_lists.get_mut(centroid as usize).ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "token code {centroid} is out of range for {num_centroids} centroids"
                    ))
                })?;
                posting_list.push(document_ordinal as u32);
            }
        }

        let total_postings = posting_lists.iter().try_fold(0_usize, |total, posting| {
            total
                .checked_add(posting.len())
                .ok_or_else(|| Error::InvalidInput("posting count overflowed usize".to_string()))
        })?;
        let mut posting_offsets = Vec::with_capacity(num_centroids + 1);
        let mut postings = Vec::with_capacity(total_postings);
        posting_offsets.push(0);
        for posting_list in posting_lists {
            postings.extend(posting_list);
            posting_offsets.push(postings.len() as u64);
        }
        Ok((posting_offsets, postings))
    }

    fn validate(&self) -> Result<()> {
        if self.dimension == 0 || self.num_centroids() == 0 {
            return Err(Error::InvalidInput(
                "centroid matrix must have non-zero rows and columns".to_string(),
            ));
        }
        if self.centroids.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "centroid values must be finite".to_string(),
            ));
        }
        if self.num_documents() > u32::MAX as usize {
            return Err(Error::InvalidInput(format!(
                "PLAID v1 supports at most {} documents, got {}",
                u32::MAX,
                self.num_documents()
            )));
        }
        if !self.row_addresses.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(Error::InvalidInput(
                "row addresses must be strictly increasing".to_string(),
            ));
        }
        if self.document_offsets.len() != self.num_documents() + 1
            || self.document_offsets.first() != Some(&0)
            || self.document_offsets.last().copied() != Some(self.num_tokens() as u64)
            || !self
                .document_offsets
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        {
            return Err(Error::InvalidInput(
                "document offsets must be monotonic, start at zero, and end at token count"
                    .to_string(),
            ));
        }
        if self
            .token_codes
            .iter()
            .any(|code| *code as usize >= self.num_centroids())
        {
            return Err(Error::InvalidInput(
                "token code exceeds centroid count".to_string(),
            ));
        }
        let packed_len = self.quantizer.packed_len(self.dimension)?;
        let expected_residual_len = self
            .num_tokens()
            .checked_mul(packed_len)
            .ok_or_else(|| Error::InvalidInput("packed residual size overflow".to_string()))?;
        if self.packed_residuals.len() != expected_residual_len {
            return Err(Error::InvalidInput(format!(
                "packed residual length must be {expected_residual_len}, got {}",
                self.packed_residuals.len()
            )));
        }
        if self.posting_offsets.len() != self.num_centroids() + 1
            || self.posting_offsets.first() != Some(&0)
            || self.posting_offsets.last().copied() != Some(self.postings.len() as u64)
            || !self
                .posting_offsets
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        {
            return Err(Error::InvalidInput(
                "posting offsets must be monotonic, start at zero, and end at posting count"
                    .to_string(),
            ));
        }
        if self
            .postings
            .iter()
            .any(|ordinal| *ordinal as usize >= self.num_documents())
        {
            return Err(Error::InvalidInput(
                "posting document ordinal exceeds document count".to_string(),
            ));
        }
        if let Some(store) = &self.exact_store {
            if store.public_row_ids.len() != self.num_documents() {
                return Err(Error::InvalidInput(format!(
                    "exact-store public row ID length must be {}, got {}",
                    self.num_documents(),
                    store.public_row_ids.len()
                )));
            }
            let expected_raw_len =
                self.num_tokens()
                    .checked_mul(self.dimension)
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "exact-store raw token value length overflow".to_string(),
                        )
                    })?;
            if store.raw_token_values.len() != expected_raw_len {
                return Err(Error::InvalidInput(format!(
                    "exact-store raw token value length must be {expected_raw_len}, got {}",
                    store.raw_token_values.len()
                )));
            }
            if store
                .raw_token_values
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(Error::InvalidInput(
                    "exact-store raw token values must be finite".to_string(),
                ));
            }
        }
        Ok(())
    }
}

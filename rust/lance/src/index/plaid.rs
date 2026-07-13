// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Database-native PLAID index build, persistence, and vector-index glue.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::DataType;
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance_arrow::{FixedSizeListArrayExt, RecordBatchExt};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ADDR, Result};
use lance_index::metrics::MetricsCollector;
use lance_index::pb::vector_index_details::{Compression, FlatCompression};
use lance_index::pb::{VectorIndexDetails, VectorMetricType};
use lance_index::prefilter::PreFilter;
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::flat::index::FlatQuantizer;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::quantizer::{QuantizationType, Quantizer};
use lance_index::vector::v3::subindex::SubIndexType;
use lance_index::vector::{Query, VECTOR_RESULT_SCHEMA, VectorIndex};
use lance_index::{INDEX_FILE_NAME, Index, IndexParams, IndexType};
use lance_linalg::distance::DistanceType;
use lance_plaid::{
    Eligibility, EligibleCentroidDecision, EligibleCentroidPlan, PlaidIndex, PlaidSearchParams,
    PlaidSearchStats, ResidualQuantizer,
};
use lance_select::RowAddrMask;
use lance_table::format::{IndexFile, IndexMetadata};
use ndarray::{Array2, ArrayView2};
use prost_types::Any as ProstAny;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use roaring::RoaringBitmap;
use serde_json::json;
use uuid::Uuid;

use super::vector::utils::{filter_finite_training_data, get_vector_type};
use crate::dataset::Dataset;

/// Built-in index family name used by [`PlaidIndexParams`].
pub const LANCE_PLAID_INDEX: &str = "PLAID";
/// Persisted `VectorIndexDetails.runtime_hints` discriminator.
pub const PLAID_RUNTIME_HINT: &str = "lance.plaid";
const PLAID_FORMAT_HINT: &str = "lance.plaid.format_version";
const PLAID_NBITS_HINT: &str = "lance.plaid.nbits";
const PLAID_CENTROIDS_HINT: &str = "lance.plaid.num_centroids";
const PLAID_TRAINING_SEED_HINT: &str = "lance.plaid.training_seed";
const PLAID_TRAINING_SEED: u64 = 42;
pub(crate) const PLAID_DEFAULT_N_FULL_SCORES: usize = 4096;
pub(crate) const PLAID_DEFAULT_DECOMPRESS_DOCUMENTS: usize = 1024;

/// Constructs the candidate budgets shared by the legacy and direct query
/// paths. Keeping this pure makes it possible for the direct gate to prove
/// that the legacy approximate and residual stages would not truncate F.
pub(crate) fn plaid_search_params(
    requested_candidates: usize,
    eligible_documents: usize,
    n_ivf_probe: usize,
) -> PlaidSearchParams {
    let core_residual_candidates = requested_candidates
        .max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS)
        .min(eligible_documents.max(1));
    PlaidSearchParams {
        n_ivf_probe,
        n_full_scores: PLAID_DEFAULT_N_FULL_SCORES.max(core_residual_candidates.saturating_mul(4)),
        top_k: core_residual_candidates,
        centroid_score_threshold: None,
    }
}

/// Build parameters for the database-native CPU PLAID index.
#[derive(Clone, Debug)]
pub struct PlaidIndexParams {
    /// Requested number of coarse centroids. Zero selects NextPlaid's automatic rule.
    pub num_centroids: usize,
    /// Residual bits per dimension. Version 1 supports 2 and 4.
    pub nbits: u8,
    /// Maximum KMeans iterations.
    pub max_iterations: u32,
    /// Number of sampled token vectors per requested centroid.
    pub sample_rate: usize,
}

impl Default for PlaidIndexParams {
    fn default() -> Self {
        Self {
            num_centroids: 0,
            nbits: 2,
            max_iterations: 20,
            sample_rate: 256,
        }
    }
}

impl PlaidIndexParams {
    fn validate(&self) -> Result<()> {
        if self.max_iterations == 0 || self.sample_rate == 0 {
            return Err(Error::invalid_input(format!(
                "PLAID max_iterations and sample_rate must be positive, got {} and {}",
                self.max_iterations, self.sample_rate
            )));
        }
        if !matches!(self.nbits, 2 | 4) {
            return Err(Error::invalid_input(format!(
                "PLAID nbits must be 2 or 4, got {}",
                self.nbits
            )));
        }
        Ok(())
    }
}

impl IndexParams for PlaidIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_name(&self) -> &str {
        LANCE_PLAID_INDEX
    }
}

/// Creates persisted vector index details that survive Dataset/Session reopen.
pub(crate) fn plaid_index_details(params: &PlaidIndexParams) -> Result<ProstAny> {
    let runtime_hints = HashMap::from([
        (PLAID_RUNTIME_HINT.to_string(), "true".to_string()),
        (PLAID_FORMAT_HINT.to_string(), "1".to_string()),
        (PLAID_NBITS_HINT.to_string(), params.nbits.to_string()),
        (
            PLAID_CENTROIDS_HINT.to_string(),
            if params.num_centroids == 0 {
                "auto".to_string()
            } else {
                params.num_centroids.to_string()
            },
        ),
        (
            PLAID_TRAINING_SEED_HINT.to_string(),
            PLAID_TRAINING_SEED.to_string(),
        ),
    ]);
    let details = VectorIndexDetails {
        metric_type: VectorMetricType::Dot.into(),
        target_partition_size: 0,
        hnsw_index_config: None,
        compression: Some(Compression::Flat(FlatCompression {})),
        runtime_hints,
    };
    ProstAny::from_msg(&details)
        .map_err(|error| Error::index(format!("failed to encode PLAID index details: {error}")))
}

/// Returns true when manifest-persisted vector details identify a PLAID segment.
pub(crate) fn is_plaid_index_metadata(metadata: &IndexMetadata) -> bool {
    metadata
        .index_details
        .as_ref()
        .and_then(|details| details.to_msg::<VectorIndexDetails>().ok())
        .and_then(|details| details.runtime_hints.get(PLAID_RUNTIME_HINT).cloned())
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
}

/// Builds one immutable PLAID segment by streaming the source multi-vector column.
pub(crate) async fn build_plaid_index(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    params: &PlaidIndexParams,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<Vec<IndexFile>> {
    params.validate()?;
    let (vector_type, element_type) = get_vector_type(dataset.schema(), column)?;
    let dimension = match vector_type {
        DataType::List(ref item) => match item.data_type() {
            DataType::FixedSizeList(_, dimension) if element_type == DataType::Float32 => {
                usize::try_from(*dimension).map_err(|_| {
                    Error::invalid_input("PLAID vector dimension does not fit usize".to_string())
                })?
            }
            _ => {
                return Err(Error::invalid_input(format!(
                    "PLAID requires List<FixedSizeList<Float32>>, got {vector_type}"
                )));
            }
        },
        _ => {
            return Err(Error::invalid_input(format!(
                "PLAID requires a multi-vector List column, got {vector_type}"
            )));
        }
    };
    if dimension == 0 || dimension.saturating_mul(usize::from(params.nbits)) % 8 != 0 {
        return Err(Error::invalid_input(format!(
            "PLAID requires dimension * nbits to be a positive multiple of 8, got dimension={dimension}, nbits={}",
            params.nbits
        )));
    }

    progress
        .stage_start("plaid_train", None, "token vectors")
        .await?;
    let requested_centroids = if params.num_centroids == 0 {
        next_plaid_num_centroids(count_plaid_tokens(dataset, column).await?)?
    } else {
        params.num_centroids
    };
    let sample_size = requested_centroids
        .checked_mul(params.sample_rate)
        .ok_or_else(|| Error::invalid_input("PLAID training sample size overflow".to_string()))?;
    let training =
        sample_plaid_training_data(dataset, column, dimension, sample_size, PLAID_TRAINING_SEED)
            .await?;
    let training = filter_finite_training_data(training)?;
    if training.is_empty() {
        return Err(Error::invalid_input(
            "PLAID cannot train on an empty multi-vector column".to_string(),
        ));
    }
    let num_centroids = requested_centroids.min(training.len());
    let raw_training_values = training.values().as_primitive::<Float32Type>().clone();
    // Match NextPlaid: train L2 KMeans on raw pooled embeddings, then
    // normalize only the learned centroids before assignment and residuals.
    let kmeans_params = lance_index::vector::kmeans::KMeansParams::new(
        None,
        params.max_iterations,
        1,
        DistanceType::L2,
    )
    .with_seed(PLAID_TRAINING_SEED);
    let model = lance_index::vector::kmeans::train_kmeans::<Float32Type>(
        &raw_training_values,
        kmeans_params,
        dimension,
        num_centroids,
        params.sample_rate,
    )?;
    let centroid_values = model
        .centroids
        .as_primitive::<Float32Type>()
        .values()
        .to_vec();
    let mut centroids = Array2::from_shape_vec((num_centroids, dimension), centroid_values)
        .map_err(|error| Error::index(format!("invalid PLAID centroid shape: {error}")))?;
    normalize_rows(&mut centroids)?;

    let mut sampled_residuals = Vec::with_capacity(training.len().saturating_mul(dimension));
    for token in raw_training_values.values().chunks_exact(dimension) {
        validate_token(token)?;
        // NextPlaid assigns against unit centroids but persists residuals from
        // the original pooled token, which may have norm below one.
        let code = nearest_centroid(token, centroids.view());
        append_raw_residual(
            token,
            centroids.row(code).as_slice().ok_or_else(|| {
                Error::internal("PLAID centroid row is not contiguous".to_string())
            })?,
            &mut sampled_residuals,
        )?;
    }
    let num_buckets = 1_usize << params.nbits;
    let cutoffs = (1..num_buckets)
        .map(|bucket| quantile(&sampled_residuals, bucket as f64 / num_buckets as f64))
        .collect::<Vec<_>>();
    let weights = (0..num_buckets)
        .map(|bucket| {
            quantile(
                &sampled_residuals,
                (bucket as f64 + 0.5) / num_buckets as f64,
            )
        })
        .collect::<Vec<_>>();
    let quantizer =
        ResidualQuantizer::try_new(params.nbits, cutoffs, weights).map_err(plaid_error)?;
    progress.stage_complete("plaid_train").await?;

    progress
        .stage_start("plaid_encode", None, "record batches")
        .await?;
    let mut scanner = dataset.scan();
    scanner.project(&[column])?.with_row_address();
    let mut stream = scanner.try_into_stream().await?;
    let mut row_addresses = Vec::new();
    let mut document_offsets = vec![0_u64];
    let mut token_codes = Vec::new();
    let mut packed_residuals = Vec::new();
    let mut batches = 0_u64;

    while let Some(batch) = stream.try_next().await? {
        let row_addrs = batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| Error::internal("PLAID build scan did not return _rowaddr".to_string()))?
            .as_primitive::<UInt64Type>();
        let documents = batch
            .column_by_qualified_name(column)
            .ok_or_else(|| {
                Error::invalid_input(format!("PLAID column {column} missing from batch"))
            })?
            .as_list::<i32>();
        if row_addrs.len() != documents.len() {
            return Err(Error::internal(format!(
                "PLAID row-address count {} differs from document count {}",
                row_addrs.len(),
                documents.len()
            )));
        }

        for row_index in 0..documents.len() {
            if documents.is_null(row_index) {
                continue;
            }
            let row_address = row_addrs.value(row_index);
            if row_addresses
                .last()
                .is_some_and(|previous| *previous >= row_address)
            {
                return Err(Error::index(
                    "PLAID build requires strictly increasing physical row addresses".to_string(),
                ));
            }
            let document = documents.value(row_index);
            let tokens = document.as_fixed_size_list();
            let mut document_residuals = Vec::with_capacity(tokens.len().saturating_mul(dimension));
            for token_index in 0..tokens.len() {
                if tokens.is_null(token_index) {
                    return Err(Error::invalid_input(
                        "PLAID does not support null token vectors".to_string(),
                    ));
                }
                let token = tokens.value(token_index);
                let token = token.as_primitive::<Float32Type>();
                validate_token(token.values())?;
                let code = nearest_centroid(token.values(), centroids.view());
                let code = u32::try_from(code)
                    .map_err(|_| Error::index("PLAID centroid ordinal exceeds u32".to_string()))?;
                token_codes.push(code);
                append_raw_residual(
                    token.values(),
                    centroids.row(code as usize).as_slice().ok_or_else(|| {
                        Error::internal("PLAID centroid row is not contiguous".to_string())
                    })?,
                    &mut document_residuals,
                )?;
            }
            let residuals =
                ArrayView2::from_shape((tokens.len(), dimension), document_residuals.as_slice())
                    .map_err(|error| {
                        Error::index(format!("invalid PLAID residual shape: {error}"))
                    })?;
            packed_residuals.extend(quantizer.quantize(residuals).map_err(plaid_error)?);
            row_addresses.push(row_address);
            document_offsets.push(
                u64::try_from(token_codes.len())
                    .map_err(|_| Error::index("PLAID token count exceeds u64".to_string()))?,
            );
        }
        batches = batches.saturating_add(1);
        progress.stage_progress("plaid_encode", batches).await?;
    }
    progress.stage_complete("plaid_encode").await?;

    if row_addresses.is_empty() {
        return Err(Error::invalid_input(
            "PLAID cannot build an index without non-null documents".to_string(),
        ));
    }
    let index = PlaidIndex::try_new(
        centroids,
        quantizer,
        row_addresses,
        document_offsets,
        token_codes,
        packed_residuals,
    )
    .map_err(plaid_error)?;
    let bytes = index.to_bytes().map_err(plaid_error)?;
    let path = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join(INDEX_FILE_NAME);
    dataset.object_store.put(&path, &bytes).await?;
    Ok(vec![IndexFile {
        path: INDEX_FILE_NAME.to_string(),
        size_bytes: bytes.len() as u64,
    }])
}

/// Opens a manifest-identified PLAID segment before LanceFile version parsing.
pub(crate) async fn open_plaid_index(
    dataset: &Dataset,
    metadata: &IndexMetadata,
) -> Result<Arc<dyn VectorIndex>> {
    if !is_plaid_index_metadata(metadata) {
        return Err(Error::index(format!(
            "index {} is not marked as PLAID in persisted VectorIndexDetails",
            metadata.uuid
        )));
    }
    let store = dataset.object_store_for_index(metadata).await?;
    let path = dataset
        .indice_files_dir(metadata)?
        .join(metadata.uuid.to_string())
        .join(INDEX_FILE_NAME);
    let bytes = store.read_one_all(&path).await?;
    let index =
        spawn_cpu(move || PlaidIndex::read_from_bytes(bytes.as_ref()).map_err(plaid_error)).await?;
    Ok(Arc::new(PlaidVectorIndex::try_new(index)?))
}

/// Loaded database-native PLAID segment.
#[derive(Debug)]
pub(crate) struct PlaidVectorIndex {
    core: Arc<PlaidIndex>,
    ivf_model: IvfModel,
    quantizer: FlatQuantizer,
}

#[derive(Clone, Debug)]
pub(crate) struct PlaidCandidatePlan {
    pub(crate) params: PlaidSearchParams,
    pub(crate) eligible_documents: usize,
    pub(crate) maximum_nprobes: usize,
    pub(crate) eligible_centroids: EligibleCentroidPlan,
    /// End-to-end plan time, including both mask/address passes and the nested
    /// storage-independent core build phase.
    pub(crate) total_plan_nanos: u64,
}

pub(crate) enum BoundedEligibleOrdinals {
    NonEnumerable,
    WithinLimit(Vec<u32>),
    OverLimit,
}

fn collect_bounded_document_ordinals(
    addresses: impl Iterator<Item = u64>,
    max_documents: usize,
    mut resolve: impl FnMut(u64) -> Option<u32>,
) -> BoundedEligibleOrdinals {
    let mut ordinals = Vec::new();
    let mut previous = None;
    for address in addresses {
        let Some(ordinal) = resolve(address) else {
            continue;
        };
        // RowAddrTreeMap is a set backed by ordered BTreeMap/Roaring iterators,
        // and PLAID row addresses are persisted strictly increasing. Their
        // address-to-ordinal mapping is therefore also sorted and unique, so no
        // O(F log F) sort/dedup pass is necessary here.
        debug_assert!(previous.is_none_or(|previous| previous < ordinal));
        if ordinals.len() == max_documents {
            // We resolved at most max_documents + 1 matching rows from this
            // segment and never allocate the over-limit row. The enclosing
            // global mask iterator may still advance past addresses belonging
            // to other segments before finding those matching rows.
            return BoundedEligibleOrdinals::OverLimit;
        }
        previous = Some(ordinal);
        ordinals.push(ordinal);
    }
    BoundedEligibleOrdinals::WithinLimit(ordinals)
}

impl PlaidVectorIndex {
    fn try_new(core: PlaidIndex) -> Result<Self> {
        let dimension = core.dimension();
        let values = Float32Array::from(core.centroids().iter().copied().collect::<Vec<_>>());
        let centroids = FixedSizeListArray::try_new_from_values(values, dimension as i32)?;
        Ok(Self {
            core: Arc::new(core),
            ivf_model: IvfModel::new(centroids, None),
            quantizer: FlatQuantizer::new(dimension, DistanceType::Dot),
        })
    }

    pub(crate) fn candidate_plan(
        &self,
        query: &Query,
        requested_candidates: usize,
        query_tokens: usize,
        mask: &RowAddrMask,
    ) -> Result<PlaidCandidatePlan> {
        let plan_started = Instant::now();
        let probe_ceiling = query
            .maximum_nprobes
            .unwrap_or(self.core.num_centroids())
            .min(self.core.num_centroids())
            .max(1);
        let base_nprobe = query.minimum_nprobes.max(8).min(probe_ceiling).max(1);
        // This is the same allocation-free segment count required by adaptive
        // nprobe before eligible-centroid probing. Wide filters stop here and
        // retain the prior global path without allocating or scanning codes.
        let segment_eligible = mask.iter_addrs().map(|addresses| {
            addresses
                .map(u64::from)
                .filter(|address| self.core.document_ordinal(*address).is_some())
                .count()
        });
        let eligible = segment_eligible.unwrap_or(self.core.num_documents());
        let adaptive_factor = if segment_eligible.is_some() && eligible > 0 {
            self.core.num_documents().div_ceil(eligible).max(1)
        } else {
            1
        };
        let n_ivf_probe = base_nprobe
            .saturating_mul(adaptive_factor)
            .min(probe_ceiling)
            .max(1);
        // The storage-independent core may residual-rerank a wider set than
        // the database layer will raw-refine.  These are quantized residual
        // candidates, not raw-vector fetches.
        let params = plaid_search_params(requested_candidates, eligible, n_ivf_probe);
        let filtered = !mask.is_select_all();
        let mut eligible_centroids = self
            .core
            .plan_eligible_centroids(None, segment_eligible, filtered, query_tokens, n_ivf_probe)
            .map_err(plaid_error)?;
        if filtered
            && segment_eligible.is_some()
            && eligible_centroids.decision() == EligibleCentroidDecision::NonEnumerable
        {
            // Only a cost-model-eligible narrow allow-list pays for a second
            // pass that materializes dense ordinals and then scans token codes.
            let addresses = mask.iter_addrs().ok_or_else(|| {
                Error::internal(
                    "PLAID enumerable filter changed while building its centroid plan".to_string(),
                )
            })?;
            let eligible_ordinals = addresses
                .map(u64::from)
                .filter_map(|address| self.core.document_ordinal(address))
                .collect::<Vec<_>>();
            eligible_centroids = self
                .core
                .plan_eligible_centroids(
                    Some(&eligible_ordinals),
                    segment_eligible,
                    filtered,
                    query_tokens,
                    n_ivf_probe,
                )
                .map_err(plaid_error)?;
        }
        Ok(PlaidCandidatePlan {
            params,
            eligible_documents: eligible,
            maximum_nprobes: probe_ceiling,
            eligible_centroids,
            total_plan_nanos: u64::try_from(plan_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        })
    }

    /// Maps an exact enumerable mask to at most `max_documents` sorted, unique
    /// ordinals for this segment. Wide masks stop on the first over-limit row
    /// matching this segment instead of fully allocating and sorting F before
    /// returning to the legacy path. A global mask can contain additional
    /// addresses owned by other segments, which do not count toward this cap.
    pub(crate) fn bounded_eligible_document_ordinals(
        &self,
        mask: &RowAddrMask,
        max_documents: usize,
    ) -> BoundedEligibleOrdinals {
        let Some(addresses) = mask.iter_addrs() else {
            return BoundedEligibleOrdinals::NonEnumerable;
        };
        collect_bounded_document_ordinals(addresses.map(u64::from), max_documents, |address| {
            self.core.document_ordinal(address)
        })
    }

    /// Directly residual-scores an exact ordinal set. `None` means at least
    /// one selected document is empty and the caller must use the legacy
    /// posting semantics instead.
    pub(crate) fn search_quantized_residuals(
        &self,
        query: ArrayView2<'_, f32>,
        document_ordinals: &[u32],
    ) -> Result<Option<(Vec<lance_plaid::SearchHit>, PlaidSearchStats)>> {
        self.core
            .search_quantized_residuals(query, document_ordinals)
            .map_err(plaid_error)
    }

    pub(crate) fn num_documents(&self) -> usize {
        self.core.num_documents()
    }

    /// Returns every indexed physical row address selected by `mask`.
    ///
    /// Allow lists are normally much smaller than the index, so probe them with
    /// the index's binary-search address lookup. Block lists cannot be
    /// enumerated directly and instead require one linear pass over this
    /// segment's dense address array.
    pub(crate) fn eligible_row_addresses(&self, mask: &RowAddrMask) -> Vec<u64> {
        if let Some(addresses) = mask.iter_addrs() {
            addresses
                .map(u64::from)
                .filter(|address| self.core.document_ordinal(*address).is_some())
                .collect()
        } else {
            self.core
                .row_addresses()
                .iter()
                .copied()
                .filter(|address| mask.selected(*address))
                .collect()
        }
    }

    pub(crate) fn search_candidates(
        &self,
        query: ArrayView2<'_, f32>,
        plan: &PlaidCandidatePlan,
        desired_candidates: usize,
        mask: &RowAddrMask,
    ) -> Result<(Vec<lance_plaid::SearchHit>, PlaidSearchStats)> {
        self.core
            .search_adaptive(
                query,
                &plan.params,
                plan.maximum_nprobes,
                desired_candidates,
                &MaskEligibility { mask },
                Some(&plan.eligible_centroids),
            )
            .map_err(plaid_error)
    }
}

impl DeepSizeOf for PlaidVectorIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.core.estimated_size_bytes() + self.ivf_model.deep_size_of_children(context)
    }
}

struct MaskEligibility<'a> {
    mask: &'a RowAddrMask,
}

impl Eligibility for MaskEligibility<'_> {
    fn includes(&self, _document_ordinal: u32, row_address: u64) -> bool {
        self.mask.selected(row_address)
    }
}

#[async_trait]
impl Index for PlaidVectorIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(json!({
            "index_type": LANCE_PLAID_INDEX,
            "documents": self.core.num_documents(),
            "tokens": self.core.num_tokens(),
            "centroids": self.core.num_centroids(),
            "dimension": self.core.dimension(),
            "nbits": self.core.quantizer().nbits(),
        }))
    }

    async fn prewarm(&self) -> Result<()> {
        Ok(())
    }

    fn index_type(&self) -> IndexType {
        IndexType::Vector
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(self
            .core
            .row_addresses()
            .iter()
            .map(|address| RowAddress::from(*address).fragment_id())
            .collect())
    }
}

#[async_trait]
impl VectorIndex for PlaidVectorIndex {
    async fn search(
        &self,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        let query_tokens = query_to_array(query, self.core.dimension())?;
        pre_filter.wait_for_ready().await?;
        let mask = pre_filter.mask();
        let plan = self.candidate_plan(query, query.k, query_tokens.nrows(), mask.as_ref())?;
        let desired_candidates = plan
            .eligible_documents
            .min(plan.params.top_k)
            .min(self.num_documents());
        let (hits, stats) = self.search_candidates(
            query_tokens.view(),
            &plan,
            desired_candidates,
            mask.as_ref(),
        )?;
        metrics.record_comparisons(
            usize::try_from(stats.exact_documents)
                .unwrap_or(usize::MAX)
                .saturating_mul(query_tokens.nrows()),
        );
        let distances = Float32Array::from(
            hits.iter()
                .take(query.k)
                .map(|hit| maxsim_distance(hit.score))
                .collect::<Vec<_>>(),
        );
        let row_addresses = UInt64Array::from(
            hits.iter()
                .take(query.k)
                .map(|hit| hit.row_address)
                .collect::<Vec<_>>(),
        );
        Ok(RecordBatch::try_new(
            VECTOR_RESULT_SCHEMA.clone(),
            vec![Arc::new(distances), Arc::new(row_addresses)],
        )?)
    }

    fn find_partitions(&self, _query: &Query) -> Result<(UInt32Array, Float32Array)> {
        Ok((
            UInt32Array::from(vec![0_u32]),
            Float32Array::from(vec![0.0_f32]),
        ))
    }

    fn total_partitions(&self) -> usize {
        1
    }

    async fn search_in_partition(
        &self,
        partition_id: usize,
        query: &Query,
        pre_filter: Arc<dyn PreFilter>,
        metrics: &dyn MetricsCollector,
    ) -> Result<RecordBatch> {
        if partition_id != 0 {
            return Err(Error::invalid_input(format!(
                "PLAID exposes one logical partition, got {partition_id}"
            )));
        }
        self.search(query, pre_filter, metrics).await
    }

    fn is_loadable(&self) -> bool {
        false
    }

    fn use_residual(&self) -> bool {
        false
    }

    async fn load(
        &self,
        _reader: Arc<dyn lance_io::traits::Reader>,
        _offset: usize,
        _length: usize,
    ) -> Result<Box<dyn VectorIndex>> {
        Err(Error::not_supported(
            "PLAID partition load is not supported; open the full versioned segment".to_string(),
        ))
    }

    async fn to_batch_stream(&self, _with_vector: bool) -> Result<SendableRecordBatchStream> {
        Err(Error::not_supported(
            "PLAID does not expose IVF partition batches".to_string(),
        ))
    }

    fn num_rows(&self) -> u64 {
        self.core.num_documents() as u64
    }

    fn row_ids(&self) -> Box<dyn Iterator<Item = &'_ u64> + '_> {
        Box::new(self.core.row_addresses().iter())
    }

    async fn remap(&mut self, _mapping: &RowAddrRemap) -> Result<()> {
        Err(Error::not_supported(
            "PLAID stores physical row addresses and refuses in-place remapping; drop and rebuild the PLAID index"
                .to_string(),
        ))
    }

    fn metric_type(&self) -> DistanceType {
        DistanceType::Dot
    }

    fn ivf_model(&self) -> &IvfModel {
        &self.ivf_model
    }

    fn quantizer(&self) -> Quantizer {
        Quantizer::Flat(self.quantizer.clone())
    }

    fn partition_size(&self, part_id: usize) -> usize {
        if part_id == 0 {
            self.core.num_documents()
        } else {
            0
        }
    }

    fn sub_index_type(&self) -> (SubIndexType, QuantizationType) {
        (SubIndexType::Flat, QuantizationType::Flat)
    }
}

pub(crate) fn maxsim_distance(maxsim: f32) -> f32 {
    1.0 - maxsim
}

pub(crate) fn query_to_array(query: &Query, dimension: usize) -> Result<Array2<f32>> {
    let values = query
        .key
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            Error::invalid_input("PLAID query must contain Float32 values".to_string())
        })?;
    if values.is_empty() || values.len() % dimension != 0 || values.null_count() != 0 {
        return Err(Error::invalid_input(format!(
            "PLAID query length {} must be a positive multiple of dimension {dimension} without nulls",
            values.len()
        )));
    }
    let query_tokens = Array2::from_shape_vec(
        (values.len() / dimension, dimension),
        values.values().to_vec(),
    )
    .map_err(|error| Error::invalid_input(format!("invalid PLAID query shape: {error}")))?;
    for row in query_tokens.outer_iter() {
        validate_token(
            row.as_slice()
                .ok_or_else(|| Error::internal("PLAID query row is not contiguous".to_string()))?,
        )?;
    }
    Ok(query_tokens)
}

async fn count_plaid_tokens(dataset: &Dataset, column: &str) -> Result<usize> {
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    let mut stream = scanner.try_into_stream().await?;
    let mut total = 0_usize;
    while let Some(batch) = stream.try_next().await? {
        let documents = batch
            .column_by_qualified_name(column)
            .ok_or_else(|| {
                Error::invalid_input(format!("PLAID column {column} missing from batch"))
            })?
            .as_list::<i32>();
        for row_index in 0..documents.len() {
            if !documents.is_null(row_index) {
                total = total
                    .checked_add(documents.value(row_index).len())
                    .ok_or_else(|| {
                        Error::invalid_input("PLAID token count overflow".to_string())
                    })?;
            }
        }
    }
    Ok(total)
}

async fn sample_plaid_training_data(
    dataset: &Dataset,
    column: &str,
    dimension: usize,
    sample_size: usize,
    seed: u64,
) -> Result<FixedSizeListArray> {
    let mut scanner = dataset.scan();
    scanner.project(&[column])?;
    let mut stream = scanner.try_into_stream().await?;
    let mut rng = SmallRng::seed_from_u64(seed);
    let initial_capacity = sample_size
        .min(4_096)
        .checked_mul(dimension)
        .ok_or_else(|| Error::invalid_input("PLAID sample capacity overflow".to_string()))?;
    let mut sampled = Vec::<f32>::with_capacity(initial_capacity);
    let mut seen = 0_usize;

    while let Some(batch) = stream.try_next().await? {
        let documents = batch
            .column_by_qualified_name(column)
            .ok_or_else(|| {
                Error::invalid_input(format!("PLAID column {column} missing from batch"))
            })?
            .as_list::<i32>();
        for row_index in 0..documents.len() {
            if documents.is_null(row_index) {
                continue;
            }
            let document = documents.value(row_index);
            let tokens = document.as_fixed_size_list();
            for token_index in 0..tokens.len() {
                if tokens.is_null(token_index) {
                    return Err(Error::invalid_input(
                        "PLAID does not support null token vectors".to_string(),
                    ));
                }
                let token = tokens.value(token_index);
                let token = token.as_primitive::<Float32Type>();
                validate_token(token.values())?;
                if seen < sample_size {
                    sampled.extend_from_slice(token.values());
                } else {
                    let replacement = rng.random_range(0..=seen);
                    if replacement < sample_size {
                        let start = replacement.checked_mul(dimension).ok_or_else(|| {
                            Error::invalid_input("PLAID sample offset overflow".to_string())
                        })?;
                        sampled[start..start + dimension].copy_from_slice(token.values());
                    }
                }
                seen = seen.checked_add(1).ok_or_else(|| {
                    Error::invalid_input("PLAID token count overflow".to_string())
                })?;
            }
        }
    }

    Ok(FixedSizeListArray::try_new_from_values(
        Float32Array::from(sampled),
        dimension as i32,
    )?)
}

fn next_plaid_num_centroids(total_tokens: usize) -> Result<usize> {
    if total_tokens == 0 {
        return Err(Error::invalid_input(
            "PLAID cannot train on an empty multi-vector column".to_string(),
        ));
    }
    let target = 16.0_f64 * (total_tokens as f64).sqrt();
    let exponent = target.log2().floor().max(0.0) as u32;
    let exponent = exponent.min(usize::BITS - 1);
    Ok((1_usize << exponent).min(total_tokens))
}

fn normalize_rows(rows: &mut Array2<f32>) -> Result<()> {
    for mut row in rows.outer_iter_mut() {
        let slice = row.as_slice_mut().ok_or_else(|| {
            Error::internal("PLAID row-major matrix unexpectedly became non-contiguous".to_string())
        })?;
        normalize_token(slice)?;
    }
    Ok(())
}

fn normalize_token(token: &mut [f32]) -> Result<()> {
    validate_token(token)?;
    let norm = token.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 1.0e-12 {
        return Err(Error::invalid_input(
            "PLAID token vectors must have non-zero norm".to_string(),
        ));
    }
    for value in token {
        *value /= norm;
    }
    Ok(())
}

fn validate_token(token: &[f32]) -> Result<()> {
    if token.is_empty() || token.iter().any(|value| !value.is_finite()) {
        return Err(Error::invalid_input(
            "PLAID token vectors must be non-empty and finite".to_string(),
        ));
    }
    Ok(())
}

fn append_raw_residual(token: &[f32], centroid: &[f32], output: &mut Vec<f32>) -> Result<()> {
    if token.len() != centroid.len() {
        return Err(Error::internal(format!(
            "PLAID token dimension {} differs from centroid dimension {}",
            token.len(),
            centroid.len()
        )));
    }
    output.extend(
        token
            .iter()
            .zip(centroid)
            .map(|(value, centroid)| value - centroid),
    );
    Ok(())
}

fn nearest_centroid(token: &[f32], centroids: ArrayView2<'_, f32>) -> usize {
    centroids
        .outer_iter()
        .enumerate()
        .map(|(index, centroid)| {
            let score = token
                .iter()
                .zip(centroid.iter())
                .map(|(left, right)| left * right)
                .sum::<f32>();
            (index, score)
        })
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn quantile(values: &[f32], quantile: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = (position - lower as f64) as f32;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn plaid_error(error: lance_plaid::Error) -> Error {
    Error::index(format!("PLAID index error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, ListBuilder};
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator, StringArray, StructArray};
    use arrow_schema::Field;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;

    use crate::DatasetBuilder;
    use crate::dataset::WriteParams;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
    use crate::io::exec::TakeExec;
    use crate::io::exec::plaid::PlaidTakeOptimizationTestGuard;
    use crate::session::Session;
    use lance_index::vector::DIST_COL;

    fn make_batch(ids: Vec<i32>, documents: Vec<Vec<[f32; 4]>>) -> RecordBatch {
        let token_builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
        let mut document_builder = ListBuilder::new(token_builder);
        for document in documents {
            for token in document {
                document_builder.values().values().append_slice(&token);
                document_builder.values().append(true);
            }
            document_builder.append(true);
        }
        RecordBatch::try_from_iter([
            ("id", Arc::new(Int32Array::from(ids)) as ArrayRef),
            ("mv", Arc::new(document_builder.finish()) as ArrayRef),
        ])
        .unwrap()
    }

    fn make_nested_batch(
        ids: Vec<i32>,
        languages: Vec<i32>,
        groups: Vec<i32>,
        documents: Vec<Vec<[f32; 4]>>,
    ) -> RecordBatch {
        let flat = make_batch(ids, documents);
        let payload = StructArray::new(
            vec![
                Arc::new(Field::new("lang", DataType::Int32, false)),
                Arc::new(Field::new("group", DataType::Int32, false)),
            ]
            .into(),
            vec![
                Arc::new(Int32Array::from(languages)) as ArrayRef,
                Arc::new(Int32Array::from(groups)) as ArrayRef,
            ],
            None,
        );
        RecordBatch::try_from_iter([
            ("id", flat["id"].clone()),
            ("mv", flat["mv"].clone()),
            ("payload", Arc::new(payload) as ArrayRef),
        ])
        .unwrap()
    }

    fn make_nullable_nested_batch(
        ids: Vec<i32>,
        strengths: Vec<f32>,
        languages: Vec<Option<i32>>,
        labels: Vec<Option<&'static str>>,
    ) -> RecordBatch {
        let documents = strengths
            .into_iter()
            .map(|strength| vec![[strength, 0.0, 0.0, 0.0], [0.0, strength, 0.0, 0.0]])
            .collect();
        let flat = make_batch(ids, documents);
        let payload = StructArray::new(
            vec![
                Arc::new(Field::new("lang", DataType::Int32, true)),
                Arc::new(Field::new("label", DataType::Utf8, true)),
            ]
            .into(),
            vec![
                Arc::new(Int32Array::from(languages)) as ArrayRef,
                Arc::new(StringArray::from(labels)) as ArrayRef,
            ],
            None,
        );
        RecordBatch::try_from_iter([
            ("id", flat["id"].clone()),
            ("mv", flat["mv"].clone()),
            ("payload", Arc::new(payload) as ArrayRef),
        ])
        .unwrap()
    }

    fn query() -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            4,
        )
        .unwrap()
    }

    fn count_take_execs(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> usize {
        usize::from(plan.as_any().is::<TakeExec>())
            + plan
                .children()
                .into_iter()
                .map(|child| count_take_execs(child.as_ref()))
                .sum::<usize>()
    }

    async fn search_ids(dataset: &Dataset, filter: Option<&str>, k: usize) -> RecordBatch {
        let mut scanner = dataset.scan();
        if let Some(filter) = filter {
            scanner.prefilter(true);
            scanner.filter(filter).unwrap();
        }
        scanner.nearest("mv", &query(), k).unwrap();
        scanner.refine(2);
        scanner.project(&["id"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    async fn grouped_semantic_search(
        dataset: &Dataset,
        fused: bool,
        grouped: bool,
        direct_winner_projection: bool,
        empty_bounds: bool,
    ) -> (RecordBatch, String, usize) {
        let _config = PlaidTakeOptimizationTestGuard::new_with_direct_winner_projection(
            fused,
            true,
            grouped,
            direct_winner_projection,
        );
        let mut scanner = dataset.scan();
        scanner.nearest("mv", &query(), 6).unwrap();
        scanner.refine(2);
        if empty_bounds {
            scanner.distance_range(Some(10.0), None);
        }
        scanner
            .project(&[lance_core::ROW_ID, "id", "payload.lang", "payload.label"])
            .unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let take_execs = count_take_execs(plan.as_ref());
        let analyzed = scanner.analyze_plan().await.unwrap();
        (
            scanner.try_into_batch().await.unwrap(),
            analyzed,
            take_execs,
        )
    }

    fn analyzed_metric_count(analyzed: &str, name: &str) -> usize {
        let needle = format!("{name}=");
        analyzed
            .match_indices(&needle)
            .map(|(start, _)| {
                analyzed[start + needle.len()..]
                    .chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .parse::<usize>()
                    .unwrap_or(0)
            })
            .sum()
    }

    async fn direct_projection_search(
        dataset: &Dataset,
        direct_winner_projection: bool,
        postfilter: Option<&str>,
    ) -> (RecordBatch, String, usize) {
        let _config = PlaidTakeOptimizationTestGuard::new_with_direct_winner_projection(
            true,
            true,
            true,
            direct_winner_projection,
        );
        let mut scanner = dataset.scan();
        if let Some(filter) = postfilter {
            scanner.prefilter(false);
            scanner.filter(filter).unwrap();
        }
        scanner.nearest("mv", &query(), 6).unwrap();
        scanner.refine(2);
        scanner.project(&[lance_core::ROW_ID, "id"]).unwrap();
        let plan = scanner.create_plan().await.unwrap();
        let take_execs = count_take_execs(plan.as_ref());
        let analyzed = scanner.analyze_plan().await.unwrap();
        (
            scanner.try_into_batch().await.unwrap(),
            analyzed,
            take_execs,
        )
    }

    #[test]
    fn automatic_centroids_match_next_plaid_rule() {
        assert_eq!(next_plaid_num_centroids(65_536).unwrap(), 4_096);
        assert_eq!(next_plaid_num_centroids(1_000_000).unwrap(), 8_192);
        assert_eq!(next_plaid_num_centroids(1).unwrap(), 1);
    }

    #[test]
    fn bounded_direct_ordinal_collection_stops_at_cap_plus_one() {
        let visited = std::cell::Cell::new(0_usize);
        let outcome = collect_bounded_document_ordinals(
            (0_u64..10_000).inspect(|_| visited.set(visited.get() + 1)),
            3,
            |address| Some(address as u32),
        );
        assert!(matches!(outcome, BoundedEligibleOrdinals::OverLimit));
        assert_eq!(visited.get(), 4);

        let visited = std::cell::Cell::new(0_usize);
        let outcome = collect_bounded_document_ordinals(
            (0_u64..10_000).inspect(|_| visited.set(visited.get() + 1)),
            0,
            |address| Some(address as u32),
        );
        assert!(matches!(outcome, BoundedEligibleOrdinals::OverLimit));
        assert_eq!(visited.get(), 1);

        let outcome =
            collect_bounded_document_ordinals([10_u64, 20, 30].into_iter(), 3, |address| {
                Some((address / 10) as u32)
            });
        let BoundedEligibleOrdinals::WithinLimit(ordinals) = outcome else {
            panic!("three rows at the cap should remain eligible");
        };
        assert_eq!(ordinals, vec![1, 2, 3]);

        let outcome = collect_bounded_document_ordinals([].into_iter(), 3, |_| Some(0));
        let BoundedEligibleOrdinals::WithinLimit(ordinals) = outcome else {
            panic!("an empty segment-local result must remain independently empty");
        };
        assert!(ordinals.is_empty());
    }

    #[test]
    fn pooled_token_residual_uses_raw_next_plaid_semantics() {
        let token = [0.5_f32, 0.0, 0.0, 0.0];
        let centroid = [1.0_f32, 0.0, 0.0, 0.0];
        let mut residual = Vec::new();
        append_raw_residual(&token, &centroid, &mut residual).unwrap();
        assert_eq!(residual, vec![-0.5, 0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn stable_row_ids_take_physical_addresses_across_two_fragments() {
        let directory = TempStrDir::default();
        let first = make_batch(
            vec![0, 1],
            vec![vec![[1.0, 0.0, 0.0, 0.0]], vec![[0.0, 1.0, 0.0, 0.0]]],
        );
        let schema = first.schema();
        let reader = RecordBatchIterator::new(vec![Ok(first)], schema.clone());
        let mut dataset = Dataset::write(
            reader,
            directory.as_ref(),
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let second = make_batch(
            vec![2, 3],
            vec![vec![[-1.0, 0.0, 0.0, 0.0]], vec![[0.0, -1.0, 0.0, 0.0]]],
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 4,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();

        // With one fixed probe, the positive centroid cannot discover the
        // negative document.  The small-filter exact path must still return
        // every visible row selected by the structured filter.
        let single_token_query = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 0.0, 0.0, 0.0]),
            4,
        )
        .unwrap();
        let mut filtered_scanner = dataset.scan();
        filtered_scanner.prefilter(true);
        filtered_scanner.filter("id IN (0, 1, 2, 3)").unwrap();
        filtered_scanner
            .nearest("mv", &single_token_query, 4)
            .unwrap();
        filtered_scanner.nprobes(1);
        filtered_scanner.project(&["id"]).unwrap();
        let filtered = filtered_scanner.try_into_batch().await.unwrap();
        assert_eq!(filtered.num_rows(), 4);
        let distances = filtered[DIST_COL].as_primitive::<Float32Type>();
        for (actual, expected) in distances.values().iter().zip([0.0_f32, 1.0, 1.0, 2.0]) {
            assert!((*actual - expected).abs() < 1.0e-5);
        }
        let mut filtered_ids = filtered["id"]
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values()
            .to_vec();
        filtered_ids.sort_unstable();
        assert_eq!(filtered_ids, &[0, 1, 2, 3]);

        let mut analyzed_scanner = dataset.scan();
        analyzed_scanner.prefilter(true);
        analyzed_scanner.filter("id IN (0, 1, 2, 3)").unwrap();
        analyzed_scanner
            .nearest("mv", &single_token_query, 4)
            .unwrap();
        analyzed_scanner.nprobes(1);
        analyzed_scanner.project(&["id"]).unwrap();
        let analyzed = analyzed_scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("plaid_filter_exact_small_filter_fallbacks=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_filter_exact_documents=4"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_index_only_queries=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_raw_vector_rows=4"),
            "unexpected analyzed plan:\n{analyzed}"
        );

        // A normal index-only query merges residual scores, keeps top-k, and
        // resolves only those stable row IDs without reading the vector column.
        let mut index_only_scanner = dataset.scan();
        index_only_scanner
            .nearest("mv", &single_token_query, 2)
            .unwrap();
        index_only_scanner.project(&["id"]).unwrap();
        let index_only_plan = index_only_scanner.explain_plan(false).await.unwrap();
        assert!(
            index_only_plan.contains("mode=index_only"),
            "unexpected index-only plan:\n{index_only_plan}"
        );
        assert!(
            index_only_plan.contains("raw_refinement_budget=0"),
            "unexpected index-only plan:\n{index_only_plan}"
        );
        let index_only_analyzed = index_only_scanner.analyze_plan().await.unwrap();
        assert!(
            index_only_analyzed.contains("plaid_index_only_queries=1"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );
        assert!(
            index_only_analyzed.contains("plaid_row_id_only_rows=2"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );
        assert!(
            index_only_analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );
        let index_only = index_only_scanner.try_into_batch().await.unwrap();
        assert_eq!(index_only.num_rows(), 2);
        assert_eq!(
            index_only["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[0, 1]
        );

        let mut bounded_index_only = dataset.scan();
        bounded_index_only
            .nearest("mv", &single_token_query, 4)
            .unwrap();
        bounded_index_only.distance_range(Some(0.5), Some(1.5));
        bounded_index_only.project(&["id"]).unwrap();
        let bounded_analyzed = bounded_index_only.analyze_plan().await.unwrap();
        assert!(
            bounded_analyzed.contains("plaid_row_id_only_rows=2"),
            "unexpected analyzed plan:\n{bounded_analyzed}"
        );
        assert!(
            bounded_analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{bounded_analyzed}"
        );
        let bounded_index_only = bounded_index_only.try_into_batch().await.unwrap();
        assert_eq!(
            bounded_index_only["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[1, 3]
        );

        // Explicit refinement reads no more than k * factor raw vectors; the
        // core is still free to residual-rerank its wider candidate budget.
        let mut exact_scanner = dataset.scan();
        exact_scanner.nearest("mv", &single_token_query, 2).unwrap();
        exact_scanner.refine(1);
        exact_scanner.project(&["id"]).unwrap();
        let exact_plan = exact_scanner.explain_plan(false).await.unwrap();
        assert!(
            exact_plan.contains("mode=exact"),
            "unexpected exact plan:\n{exact_plan}"
        );
        assert!(
            exact_plan.contains("raw_refinement_budget=2"),
            "unexpected exact plan:\n{exact_plan}"
        );
        let exact_analyzed = exact_scanner.analyze_plan().await.unwrap();
        assert!(
            exact_analyzed.contains("plaid_exact_refinement_queries=1"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );
        assert!(
            exact_analyzed.contains("plaid_raw_refinement_budget=2"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );
        assert!(
            exact_analyzed.contains("plaid_core_residual_budget=4"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );
        assert!(
            exact_analyzed.contains("plaid_raw_vector_rows=2"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );

        // F=4 is larger than the index-only requested budget k=2. One probe
        // finds only the negative document, so true underfill exact-scores all
        // four eligible rows and records the distinct fallback reason.
        let negative_query = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![-1.0, 0.0, 0.0, 0.0]),
            4,
        )
        .unwrap();
        let mut underfilled_scanner = dataset.scan();
        underfilled_scanner.prefilter(true);
        underfilled_scanner.filter("id IN (0, 1, 2, 3)").unwrap();
        underfilled_scanner
            .nearest("mv", &negative_query, 2)
            .unwrap();
        underfilled_scanner.nprobes(1);
        underfilled_scanner.project(&["id"]).unwrap();
        let underfilled_analyzed = underfilled_scanner.analyze_plan().await.unwrap();
        assert!(
            underfilled_analyzed.contains("plaid_filter_exact_underfilled_fallbacks=1"),
            "unexpected analyzed plan:\n{underfilled_analyzed}"
        );
        assert!(
            underfilled_analyzed.contains("plaid_raw_vector_rows=4"),
            "unexpected analyzed plan:\n{underfilled_analyzed}"
        );
        let underfilled = underfilled_scanner.try_into_batch().await.unwrap();
        assert_eq!(underfilled.num_rows(), 2);

        let second_fragment = search_ids(&dataset, Some("id = 2"), 1).await;
        assert_eq!(
            second_fragment["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[2]
        );
        dataset.delete("id = 2").await.unwrap();

        let mut filtered_after_delete = dataset.scan();
        filtered_after_delete.prefilter(true);
        filtered_after_delete.filter("id IN (0, 1, 2, 3)").unwrap();
        filtered_after_delete
            .nearest("mv", &single_token_query, 4)
            .unwrap();
        filtered_after_delete.nprobes(1);
        filtered_after_delete.project(&["id"]).unwrap();
        let filtered_after_delete = filtered_after_delete.try_into_batch().await.unwrap();
        assert_eq!(filtered_after_delete.num_rows(), 3);
        let mut filtered_ids = filtered_after_delete["id"]
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values()
            .to_vec();
        filtered_ids.sort_unstable();
        assert_eq!(filtered_ids, &[0, 1, 3]);

        let mut index_only_after_delete = dataset.scan();
        index_only_after_delete
            .nearest("mv", &single_token_query, 2)
            .unwrap();
        index_only_after_delete.project(&["id"]).unwrap();
        let after_delete_analyzed = index_only_after_delete.analyze_plan().await.unwrap();
        assert!(
            after_delete_analyzed.contains("plaid_row_id_only_rows=2"),
            "unexpected analyzed plan:\n{after_delete_analyzed}"
        );
        assert!(
            after_delete_analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{after_delete_analyzed}"
        );
        let index_only_after_delete = index_only_after_delete.try_into_batch().await.unwrap();
        assert_eq!(index_only_after_delete.num_rows(), 2);
        assert!(
            !index_only_after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&2)
        );

        let after_delete = search_ids(&dataset, None, 4).await;
        assert_eq!(after_delete.num_rows(), 3);
        assert!(
            !after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&2)
        );
    }

    #[tokio::test]
    async fn exact_take_optimizations_preserve_database_semantics_and_plan_boundaries() {
        let directory = TempStrDir::default();
        let batch = make_batch(
            vec![0, 1, 2, 3, 4, 5],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                vec![[0.9, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
                vec![[0.8, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]],
                vec![[0.7, 0.0, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0]],
                vec![[0.6, 0.0, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0]],
                vec![[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0]],
            ],
        );
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
            directory.as_ref(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 4,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();
        dataset.delete("id = 4").await.unwrap();

        let control_config = PlaidTakeOptimizationTestGuard::new(false, false);
        let mut control_scanner = dataset.scan();
        control_scanner.prefilter(false);
        control_scanner.filter("id != 1").unwrap();
        control_scanner.nearest("mv", &query(), 5).unwrap();
        control_scanner.refine(2);
        control_scanner.project(&["id"]).unwrap();
        let control_plan = control_scanner.create_plan().await.unwrap();
        assert!(count_take_execs(control_plan.as_ref()) >= 1);
        let control = control_scanner.try_into_batch().await.unwrap();

        drop(control_config);
        let _treatment_config = PlaidTakeOptimizationTestGuard::new_with_direct_winner_projection(
            true, true, true, true,
        );
        let mut treatment_scanner = dataset.scan();
        treatment_scanner.prefilter(false);
        treatment_scanner.filter("id != 1").unwrap();
        treatment_scanner.nearest("mv", &query(), 5).unwrap();
        treatment_scanner.refine(2);
        treatment_scanner.project(&["id"]).unwrap();
        let treatment_plan = treatment_scanner.create_plan().await.unwrap();
        assert_eq!(count_take_execs(treatment_plan.as_ref()), 0);
        let treatment_explain = treatment_scanner.explain_plan(false).await.unwrap();
        assert!(treatment_explain.contains("sorted_raw_take_mode=enabled"));
        assert!(treatment_explain.contains("grouped_refinement_mode=enabled"));
        assert!(treatment_explain.contains("fused_final_take_mode=enabled"));
        assert!(treatment_explain.contains("direct_winner_projection_mode=enabled"));
        assert!(treatment_explain.contains("fused_output_fields=1"));
        let treatment_analyzed = treatment_scanner.analyze_plan().await.unwrap();
        // The deletion mask activates the exact-filter fallback, which already
        // sorts physical addresses and therefore must not claim a second sort.
        assert!(!treatment_analyzed.contains("plaid_sorted_raw_take_queries=1"));
        assert!(!treatment_analyzed.contains("plaid_grouped_refinement_queries=1"));
        assert!(treatment_analyzed.contains("plaid_grouped_refinement_fallbacks=1"));
        assert!(treatment_analyzed.contains("plaid_fused_final_take_queries=1"));
        assert!(treatment_analyzed.contains("plaid_fused_final_take_candidate_rows=5"));
        assert!(treatment_analyzed.contains("plaid_fused_final_take_output_rows=5"));
        // A one-fragment read stays on ordinary TakeBuilder projection even
        // when the direct winner projector is enabled.
        assert_eq!(
            analyzed_metric_count(
                &treatment_analyzed,
                "plaid_fused_final_take_direct_projection_queries",
            ),
            0
        );
        assert_eq!(
            analyzed_metric_count(
                &treatment_analyzed,
                "plaid_fused_final_take_legacy_projection_queries",
            ),
            0
        );
        let treatment = treatment_scanner.try_into_batch().await.unwrap();
        assert_eq!(control, treatment);
        let treatment_ids = treatment["id"]
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values();
        assert!(!treatment_ids.contains(&1));
        assert!(!treatment_ids.contains(&4));

        // Asking for the stable row ID must not duplicate the system field in
        // the fused output schema.
        let mut row_id_scanner = dataset.scan();
        row_id_scanner.nearest("mv", &query(), 3).unwrap();
        row_id_scanner.refine(2);
        row_id_scanner.project(&[lance_core::ROW_ID, "id"]).unwrap();
        let row_id_batch = row_id_scanner.try_into_batch().await.unwrap();
        assert_eq!(
            row_id_batch
                .schema()
                .fields()
                .iter()
                .filter(|field| field.name() == lance_core::ROW_ID)
                .count(),
            1
        );

        // Index-only has no raw-vector Take to consolidate and must retain the
        // ordinary outer Take even when the experimental env is enabled.
        let mut index_only_scanner = dataset.scan();
        index_only_scanner.nearest("mv", &query(), 3).unwrap();
        index_only_scanner.project(&["id"]).unwrap();
        let index_only_plan = index_only_scanner.create_plan().await.unwrap();
        assert!(count_take_execs(index_only_plan.as_ref()) >= 1);
        let index_only_explain = index_only_scanner.explain_plan(false).await.unwrap();
        assert!(index_only_explain.contains("mode=index_only"));
        assert!(index_only_explain.contains("fused_output_fields=0"));

        // An unindexed append wraps PLAID in the stock combined-search tree.
        // Root-only fusion must decline it so both branches keep one schema.
        let appended = make_batch(
            vec![99],
            vec![vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]],
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(appended)], schema), None)
            .await
            .unwrap();
        let (append_control, append_control_analyzed, append_control_takes) =
            direct_projection_search(&dataset, false, None).await;
        let (append_direct, append_direct_analyzed, append_direct_takes) =
            direct_projection_search(&dataset, true, None).await;
        assert!(append_control_takes >= 1);
        assert!(append_direct_takes >= 1);
        assert!(append_control_analyzed.contains("fused_output_fields=0"));
        assert!(append_control_analyzed.contains("direct_winner_projection_mode=disabled"));
        assert!(append_direct_analyzed.contains("fused_output_fields=0"));
        assert!(append_direct_analyzed.contains("direct_winner_projection_mode=enabled"));
        for analyzed in [&append_control_analyzed, &append_direct_analyzed] {
            assert_eq!(
                analyzed_metric_count(analyzed, "plaid_fused_final_take_queries"),
                0
            );
            assert_eq!(
                analyzed_metric_count(analyzed, "plaid_fused_final_take_direct_projection_queries",),
                0
            );
            assert_eq!(
                analyzed_metric_count(analyzed, "plaid_fused_final_take_legacy_projection_queries",),
                0
            );
        }
        assert_eq!(append_direct.schema(), append_control.schema());
        assert_eq!(append_direct, append_control);
        assert!(
            append_direct["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&99)
        );
    }

    #[tokio::test]
    async fn grouped_refinement_preserves_multifragment_database_semantics() {
        for enable_stable_row_ids in [false, true] {
            let directory = TempStrDir::default();
            let fragments = vec![
                make_nullable_nested_batch(
                    vec![0, 1, 2],
                    vec![0.2, 1.0, 0.5],
                    vec![Some(10), None, Some(12)],
                    vec![Some("zero"), Some("one"), Some("two")],
                ),
                make_nullable_nested_batch(
                    vec![3, 4, 5],
                    vec![1.0, 0.3, 0.5],
                    vec![Some(13), Some(14), Some(15)],
                    vec![Some("three"), Some("four"), None],
                ),
                make_nullable_nested_batch(
                    vec![6, 7, 8],
                    vec![0.9, 0.8, 0.1],
                    vec![None, Some(17), Some(18)],
                    vec![Some("six"), Some("seven"), Some("eight")],
                ),
            ];
            let schema = fragments[0].schema();
            let reader = RecordBatchIterator::new(
                fragments
                    .into_iter()
                    .map(|batch| Ok::<RecordBatch, arrow_schema::ArrowError>(batch)),
                schema,
            );
            let mut dataset = Dataset::write(
                reader,
                directory.as_ref(),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    max_rows_per_group: 3,
                    enable_stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
            assert_eq!(dataset.get_fragments().len(), 3);
            dataset
                .create_index(
                    &["mv"],
                    IndexType::Vector,
                    Some("grouped_plaid_idx".to_string()),
                    &PlaidIndexParams {
                        num_centroids: 4,
                        nbits: 2,
                        max_iterations: 3,
                        sample_rate: 4,
                    },
                    false,
                )
                .await
                .unwrap();
            // Delete one row from the first, middle, and last fragment after
            // indexing. Every grouped batch must still line up with its public
            // row IDs and local batch ordinal.
            dataset.delete("id IN (0, 4, 8)").await.unwrap();

            let (fused_control, _, fused_control_takes) =
                grouped_semantic_search(&dataset, true, false, false, false).await;
            let (fused_grouped_legacy, legacy_analyzed, fused_grouped_legacy_takes) =
                grouped_semantic_search(&dataset, true, true, false, false).await;
            let (fused_grouped, fused_analyzed, fused_grouped_takes) =
                grouped_semantic_search(&dataset, true, true, true, false).await;
            assert_eq!(fused_control_takes, 0);
            assert_eq!(fused_grouped_legacy_takes, 0);
            assert_eq!(fused_grouped_takes, 0);
            assert_eq!(fused_grouped_legacy, fused_control);
            assert_eq!(fused_grouped, fused_grouped_legacy);
            assert!(legacy_analyzed.contains("direct_winner_projection_mode=disabled"));
            assert!(legacy_analyzed.contains("plaid_fused_final_take_legacy_projection_queries=1"));
            assert!(legacy_analyzed.contains("plaid_fused_final_take_legacy_projection_sub_time="));
            assert!(
                !legacy_analyzed.contains("plaid_fused_final_take_direct_projection_queries=1")
            );
            assert!(fused_analyzed.contains("direct_winner_projection_mode=enabled"));
            assert!(fused_analyzed.contains("plaid_fused_final_take_direct_projection_queries=1"));
            assert!(!fused_analyzed.contains("plaid_fused_final_take_legacy_projection_queries=1"));
            assert!(fused_analyzed.contains("plaid_grouped_refinement_queries=1"));
            assert!(fused_analyzed.contains("plaid_grouped_refinement_batches=3"));
            assert!(fused_analyzed.contains("plaid_grouped_refinement_rows=6"));
            assert!(!fused_analyzed.contains("plaid_grouped_refinement_fallbacks=1"));
            assert!(fused_analyzed.contains("plaid_fused_final_take_candidate_rows=6"));
            assert!(fused_analyzed.contains("plaid_fused_final_take_select_sub_time="));
            assert!(fused_analyzed.contains("plaid_fused_final_take_logical_projection_sub_time="));
            assert!(fused_analyzed.contains("plaid_fused_final_take_direct_projection_sub_time="));
            assert!(fused_analyzed.contains("plaid_fused_final_take_json_conversion_sub_time="));
            assert!(fused_analyzed.contains("plaid_fused_final_take_assembly_sub_time="));
            assert_eq!(
                fused_grouped
                    .schema()
                    .fields()
                    .iter()
                    .filter(|field| field.name() == lance_core::ROW_ID)
                    .count(),
                1
            );
            let ids = fused_grouped["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values();
            assert_eq!(ids.len(), 6);
            assert!(ids.iter().all(|id| ![0, 4, 8].contains(id)));
            assert!(
                fused_grouped
                    .column_by_name("payload.lang")
                    .unwrap()
                    .null_count()
                    > 0
            );
            assert!(
                fused_grouped
                    .column_by_name("payload.label")
                    .unwrap()
                    .null_count()
                    > 0
            );
            let distances = fused_grouped[DIST_COL].as_primitive::<Float32Type>();
            for (actual, expected) in distances
                .values()
                .iter()
                .zip([-1.0_f32, -1.0, -0.8, -0.6, 0.0, 0.0])
            {
                assert!(
                    (*actual - expected).abs() < 1.0e-5,
                    "actual distance {actual}, expected {expected}"
                );
            }
            let row_ids = fused_grouped[lance_core::ROW_ID].as_primitive::<UInt64Type>();
            for index in 1..distances.len() {
                if distances.value(index - 1).to_bits() == distances.value(index).to_bits() {
                    assert!(row_ids.value(index - 1) < row_ids.value(index));
                }
            }

            // Postfilter runs after PLAID top-k. The filter-only nested field is
            // carried through the fused read but must be removed from the final
            // projection; direct and legacy projector paths must be bit-exact.
            let (postfilter_legacy, postfilter_legacy_analyzed, postfilter_legacy_takes) =
                direct_projection_search(&dataset, false, Some("payload.lang >= 13")).await;
            let (postfilter_direct, postfilter_direct_analyzed, postfilter_direct_takes) =
                direct_projection_search(&dataset, true, Some("payload.lang >= 13")).await;
            assert_eq!(postfilter_legacy_takes, 0);
            assert_eq!(postfilter_direct_takes, 0);
            assert_eq!(
                analyzed_metric_count(
                    &postfilter_legacy_analyzed,
                    "plaid_fused_final_take_legacy_projection_queries",
                ),
                1
            );
            assert_eq!(
                analyzed_metric_count(
                    &postfilter_legacy_analyzed,
                    "plaid_fused_final_take_direct_projection_queries",
                ),
                0
            );
            assert_eq!(
                analyzed_metric_count(
                    &postfilter_direct_analyzed,
                    "plaid_fused_final_take_direct_projection_queries",
                ),
                1
            );
            assert_eq!(
                analyzed_metric_count(
                    &postfilter_direct_analyzed,
                    "plaid_fused_final_take_legacy_projection_queries",
                ),
                0
            );
            assert_eq!(postfilter_direct.schema(), postfilter_legacy.schema());
            assert_eq!(
                postfilter_direct["id"]
                    .as_primitive::<arrow::datatypes::Int32Type>()
                    .values(),
                postfilter_legacy["id"]
                    .as_primitive::<arrow::datatypes::Int32Type>()
                    .values()
            );
            assert_eq!(
                postfilter_direct[lance_core::ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values(),
                postfilter_legacy[lance_core::ROW_ID]
                    .as_primitive::<UInt64Type>()
                    .values()
            );
            let direct_distance_bits = postfilter_direct[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            let legacy_distance_bits = postfilter_legacy[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(direct_distance_bits, legacy_distance_bits);
            assert_eq!(postfilter_direct, postfilter_legacy);
            assert!(
                postfilter_direct
                    .schema()
                    .field_with_name("payload.lang")
                    .is_err()
            );
            let mut postfilter_ids = postfilter_direct["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .to_vec();
            postfilter_ids.sort_unstable();
            assert_eq!(postfilter_ids, [3, 5, 7]);

            let (nonfused_control, _, nonfused_control_takes) =
                grouped_semantic_search(&dataset, false, false, false, false).await;
            let (nonfused_grouped, nonfused_analyzed, nonfused_grouped_takes) =
                grouped_semantic_search(&dataset, false, true, false, false).await;
            assert!(nonfused_control_takes >= 1);
            assert!(nonfused_grouped_takes >= 1);
            assert_eq!(nonfused_grouped, nonfused_control);
            assert_eq!(nonfused_grouped, fused_grouped);
            assert!(nonfused_analyzed.contains("plaid_grouped_refinement_queries=1"));
            assert!(nonfused_analyzed.contains("plaid_grouped_refinement_batches=3"));

            let (empty, empty_analyzed, empty_takes) =
                grouped_semantic_search(&dataset, true, true, true, true).await;
            assert_eq!(empty_takes, 0);
            assert_eq!(empty.num_rows(), 0);
            assert!(empty_analyzed.contains("plaid_grouped_refinement_queries=1"));
            assert!(empty_analyzed.contains("plaid_grouped_refinement_batches=3"));
            assert!(empty_analyzed.contains("plaid_grouped_refinement_rows=6"));
        }
    }
    #[tokio::test]
    async fn fused_exact_take_projects_nested_filter_sibling_fields() {
        let directory = TempStrDir::default();
        let batch = make_nested_batch(
            vec![0, 1, 2, 3],
            vec![10, 20, 30, 40],
            vec![1, 2, 1, 2],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                vec![[0.9, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
                vec![[0.8, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]],
                vec![[0.7, 0.0, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0]],
            ],
        );
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            directory.as_ref(),
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("nested_plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 4,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();

        let control_config = PlaidTakeOptimizationTestGuard::new(false, false);
        let mut control_scanner = dataset.scan();
        control_scanner.prefilter(false);
        control_scanner.filter("payload.group = 1").unwrap();
        control_scanner.nearest("mv", &query(), 4).unwrap();
        control_scanner.refine(2);
        control_scanner.project(&["id", "payload.lang"]).unwrap();
        let control = control_scanner.try_into_batch().await.unwrap();

        drop(control_config);
        let _treatment_config = PlaidTakeOptimizationTestGuard::new(true, true);
        let mut treatment_scanner = dataset.scan();
        treatment_scanner.prefilter(false);
        treatment_scanner.filter("payload.group = 1").unwrap();
        treatment_scanner.nearest("mv", &query(), 4).unwrap();
        treatment_scanner.refine(2);
        treatment_scanner.project(&["id", "payload.lang"]).unwrap();
        let treatment_plan = treatment_scanner.create_plan().await.unwrap();
        assert_eq!(count_take_execs(treatment_plan.as_ref()), 0);
        let treatment_analyzed = treatment_scanner.analyze_plan().await.unwrap();
        assert!(treatment_analyzed.contains("plaid_sorted_raw_take_queries=1"));
        assert!(treatment_analyzed.contains("plaid_fused_final_take_queries=1"));
        let treatment = treatment_scanner.try_into_batch().await.unwrap();
        assert_eq!(control, treatment);
        assert_eq!(
            treatment["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[0, 2]
        );
        assert_eq!(
            treatment
                .column_by_name("payload.lang")
                .unwrap()
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[10, 30]
        );
    }

    #[tokio::test]
    async fn direct_residual_empty_document_falls_back_to_legacy() {
        let directory = TempStrDir::default();
        let batch = make_batch(
            vec![0, 1, 2, 3],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0]],
                Vec::new(),
                vec![[0.0, 1.0, 0.0, 0.0]],
                vec![[-1.0, 0.0, 0.0, 0.0]],
            ],
        );
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            directory.as_ref(),
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 2,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();

        let mut scanner = dataset.scan();
        scanner.prefilter(true);
        scanner.filter("id >= 0").unwrap();
        scanner.nearest("mv", &query(), 1).unwrap();
        scanner.project(&["id"]).unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_queries=0",
            "plaid_legacy_residual_queries=1",
            "plaid_legacy_residual_segments=1",
            "plaid_direct_residual_skipped_empty_tokens_segments=1",
            "plaid_direct_residual_time",
        ] {
            assert!(
                analyzed.contains(expected),
                "missing {expected} in empty-token analyzed plan:\n{analyzed}"
            );
        }
        assert!(
            !analyzed.contains("plaid_direct_residual_time=1ns"),
            "empty-token direct attempt time was not recorded:\n{analyzed}"
        );
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 1);
        assert_eq!(
            result["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[0]
        );
    }

    #[tokio::test]
    async fn adaptive_eligible_centroids_preserve_database_semantics_and_report_costs() {
        const DOCUMENTS: i32 = 512;
        const FILTERED: i32 = 16;
        let directory = TempStrDir::default();
        let documents = (0..DOCUMENTS)
            .map(|_| {
                vec![
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            })
            .collect::<Vec<_>>();
        let batch = make_batch((0..DOCUMENTS).collect(), documents);
        let schema = batch.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            directory.as_ref(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 4,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();
        let four_token_query = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ]),
            4,
        )
        .unwrap();

        // With no explicit probe ceiling, all 16 enumerable rows fit the
        // direct quantized-residual hard cap. The shortcut must bypass every
        // centroid/posting/approximate phase while retaining stable-row-ID
        // projection and deterministic top-k semantics.
        let mut direct = dataset.scan();
        direct.prefilter(true);
        direct.filter(&format!("id < {FILTERED}")).unwrap();
        direct.nearest("mv", &four_token_query, 10).unwrap();
        direct.project(&["id"]).unwrap();
        let direct_plan = direct.explain_plan(false).await.unwrap();
        assert!(
            direct_plan.contains("direct_residual_mode=enabled"),
            "unexpected direct plan:\n{direct_plan}"
        );
        assert!(
            direct_plan.contains("direct_residual_max_documents=1024"),
            "unexpected direct plan:\n{direct_plan}"
        );
        let direct_analyzed = direct.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_queries=1",
            "plaid_direct_residual_segments=1",
            "plaid_direct_residual_documents=16",
            "plaid_legacy_residual_queries=0",
            "plaid_legacy_residual_segments=0",
            "plaid_centroids_probed=0",
            "plaid_posting_entries=0",
            "plaid_approximate_documents=0",
            "plaid_candidate_documents=16",
            "plaid_residual_documents=16",
            "plaid_row_id_only_rows=10",
            "plaid_raw_vector_rows=0",
            "plaid_direct_residual_time",
        ] {
            assert!(
                direct_analyzed.contains(expected),
                "missing {expected} in direct analyzed plan:\n{direct_analyzed}"
            );
        }
        assert!(
            !direct_analyzed.contains("plaid_direct_residual_time=1ns"),
            "direct attempt time was not recorded:\n{direct_analyzed}"
        );
        let direct_indexed = direct.try_into_batch().await.unwrap();
        assert_eq!(direct_indexed.num_rows(), 10);
        assert_eq!(
            direct_indexed["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &(0..10).collect::<Vec<_>>()
        );

        // F/N = 1/32 and the eligible token scan is safely below the global
        // posting estimate, so the database-native eligible-centroid path is
        // selected when an explicit all-centroid ceiling forces legacy. Stable
        // row IDs require a row-id-only take, never a raw vector read.
        let mut index_only = dataset.scan();
        index_only.prefilter(true);
        index_only.filter(&format!("id < {FILTERED}")).unwrap();
        index_only.nearest("mv", &four_token_query, 10).unwrap();
        index_only.maximum_nprobes(4);
        index_only.project(&["id"]).unwrap();
        let analyzed = index_only.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("plaid_eligible_centroid_enabled_segments=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_documents=16"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_token_codes_scanned=64"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_plan_total_time"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_core_build_sub_time"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_selection_sub_time"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_probe_rounds=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_probe_retries=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_direct_residual_skipped_explicit_ceiling_segments=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_row_id_only_rows=10"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        let indexed = index_only.try_into_batch().await.unwrap();
        assert_eq!(indexed.num_rows(), 10);
        assert_eq!(
            indexed["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &(0..10).collect::<Vec<_>>()
        );

        // refine=5 requests 50 raw candidates while F=64. The direct path may
        // skip only candidate generation: it must still feed the same ordered
        // 50-address budget to the existing raw Take/exact MaxSim tail.
        let refined = |maximum_nprobes: Option<usize>| {
            let mut scanner = dataset.scan();
            scanner.prefilter(true);
            scanner.filter("id < 64").unwrap();
            scanner.nearest("mv", &four_token_query, 10).unwrap();
            scanner.refine(5);
            if let Some(maximum_nprobes) = maximum_nprobes {
                scanner.maximum_nprobes(maximum_nprobes);
            }
            scanner.project(&["id"]).unwrap();
            scanner
        };
        let refined_direct = refined(None);
        let refined_direct_analyzed = refined_direct.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_queries=1",
            "plaid_direct_residual_segments=1",
            "plaid_direct_residual_documents=64",
            "plaid_legacy_residual_segments=0",
            "plaid_approximate_documents=0",
            "plaid_candidate_documents=64",
            "plaid_residual_documents=64",
            "plaid_raw_refinement_budget=50",
            "plaid_raw_vector_rows=50",
        ] {
            assert!(
                refined_direct_analyzed.contains(expected),
                "missing {expected} in refined direct plan:\n{refined_direct_analyzed}"
            );
        }
        let refined_direct = refined_direct.try_into_batch().await.unwrap();

        let refined_legacy = refined(Some(4));
        let refined_legacy_analyzed = refined_legacy.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_queries=0",
            "plaid_legacy_residual_queries=1",
            "plaid_direct_residual_skipped_explicit_ceiling_segments=1",
            "plaid_raw_refinement_budget=50",
            "plaid_raw_vector_rows=50",
        ] {
            assert!(
                refined_legacy_analyzed.contains(expected),
                "missing {expected} in refined legacy plan:\n{refined_legacy_analyzed}"
            );
        }
        let refined_legacy = refined_legacy.try_into_batch().await.unwrap();
        assert_eq!(
            refined_direct["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            refined_legacy["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
        );
        assert_eq!(
            refined_direct[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|distance| distance.to_bits())
                .collect::<Vec<_>>(),
            refined_legacy[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|distance| distance.to_bits())
                .collect::<Vec<_>>()
        );

        // Exact refine=1 shares the same eligible-centroid candidate path and
        // reads exactly k raw rows. With zero residuals it agrees with a flat
        // structured-filter MaxSim scan.
        let mut exact = dataset.scan();
        exact.prefilter(true);
        exact.filter(&format!("id < {FILTERED}")).unwrap();
        exact.nearest("mv", &four_token_query, 10).unwrap();
        exact.maximum_nprobes(4);
        exact.refine(1);
        exact.project(&["id"]).unwrap();
        let exact_analyzed = exact.analyze_plan().await.unwrap();
        assert!(
            exact_analyzed.contains("plaid_eligible_centroid_enabled_segments=1"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );
        assert!(
            exact_analyzed.contains("plaid_raw_vector_rows=10"),
            "unexpected analyzed plan:\n{exact_analyzed}"
        );
        let exact_result = exact.try_into_batch().await.unwrap();

        let mut flat = dataset.scan();
        flat.use_index(false);
        flat.filter(&format!("id < {FILTERED}")).unwrap();
        flat.nearest("mv", &four_token_query, 10).unwrap();
        flat.project(&["id"]).unwrap();
        let flat_result = flat.try_into_batch().await.unwrap();
        assert_eq!(
            exact_result["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            flat_result["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
        );

        // A 50% predicate stays on the prior global probing path and does not
        // scan any token codes to construct an eligible-centroid bitmap.
        let mut wide = dataset.scan();
        wide.prefilter(true);
        wide.filter("id < 256").unwrap();
        wide.nearest("mv", &four_token_query, 10).unwrap();
        wide.maximum_nprobes(4);
        wide.project(&["id"]).unwrap();
        let wide_analyzed = wide.analyze_plan().await.unwrap();
        assert!(
            wide_analyzed.contains("plaid_eligible_centroid_skipped_wide_segments=1"),
            "unexpected analyzed plan:\n{wide_analyzed}"
        );
        assert!(
            wide_analyzed.contains("plaid_eligible_token_codes_scanned=0"),
            "unexpected analyzed plan:\n{wide_analyzed}"
        );

        // Deletion visibility is incorporated before dense ordinals and
        // eligible centroids are built; deleted stable row IDs cannot leak.
        dataset.delete("id = 0").await.unwrap();
        let mut after_delete = dataset.scan();
        after_delete.prefilter(true);
        after_delete.filter("id < 17").unwrap();
        after_delete.nearest("mv", &four_token_query, 10).unwrap();
        after_delete.maximum_nprobes(4);
        after_delete.project(&["id"]).unwrap();
        let after_delete_analyzed = after_delete.analyze_plan().await.unwrap();
        assert!(
            after_delete_analyzed.contains("plaid_eligible_documents=16"),
            "unexpected analyzed plan:\n{after_delete_analyzed}"
        );
        let after_delete = after_delete.try_into_batch().await.unwrap();
        assert_eq!(after_delete.num_rows(), 10);
        assert!(
            !after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&0)
        );

        let mut direct_after_delete = dataset.scan();
        direct_after_delete.prefilter(true);
        direct_after_delete.filter("id < 17").unwrap();
        direct_after_delete
            .nearest("mv", &four_token_query, 10)
            .unwrap();
        direct_after_delete.project(&["id"]).unwrap();
        let direct_after_delete_analyzed = direct_after_delete.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_queries=1",
            "plaid_direct_residual_segments=1",
            "plaid_direct_residual_documents=16",
            "plaid_legacy_residual_segments=0",
            "plaid_candidate_documents=16",
            "plaid_residual_documents=16",
        ] {
            assert!(
                direct_after_delete_analyzed.contains(expected),
                "missing {expected} after deletion:\n{direct_after_delete_analyzed}"
            );
        }
        let direct_after_delete = direct_after_delete.try_into_batch().await.unwrap();
        assert!(
            !direct_after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&0)
        );
        assert_eq!(
            direct_after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
        );
        assert_eq!(
            direct_after_delete[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|distance| distance.to_bits())
                .collect::<Vec<_>>(),
            after_delete[DIST_COL]
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .map(|distance| distance.to_bits())
                .collect::<Vec<_>>()
        );

        // Filters at or below the requested budget retain the exact fallback;
        // eligible-centroid construction is intentionally bypassed.
        let mut fallback = dataset.scan();
        fallback.prefilter(true);
        fallback.filter("id < 5").unwrap();
        fallback.nearest("mv", &four_token_query, 10).unwrap();
        fallback.project(&["id"]).unwrap();
        let fallback_analyzed = fallback.analyze_plan().await.unwrap();
        assert!(
            fallback_analyzed.contains("plaid_filter_exact_small_filter_fallbacks=1"),
            "unexpected analyzed plan:\n{fallback_analyzed}"
        );
        assert!(
            fallback_analyzed.contains("plaid_direct_residual_small_exact_precedence_queries=1"),
            "unexpected analyzed plan:\n{fallback_analyzed}"
        );
        assert!(
            fallback_analyzed.contains("plaid_raw_vector_rows=4"),
            "unexpected analyzed plan:\n{fallback_analyzed}"
        );
        let fallback = fallback.try_into_batch().await.unwrap();
        assert_eq!(fallback.num_rows(), 4);
        assert!(
            !fallback["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&0)
        );
    }

    #[tokio::test]
    async fn empty_filter_short_circuits_every_plaid_segment() {
        let directory = TempStrDir::default();
        let first = make_batch(
            (0..16).collect(),
            (0..16)
                .map(|_| vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
                .collect(),
        );
        let schema = first.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first)], schema.clone()),
            directory.as_ref(),
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &PlaidIndexParams {
                    num_centroids: 2,
                    nbits: 2,
                    max_iterations: 3,
                    sample_rate: 4,
                },
                false,
            )
            .await
            .unwrap();
        let second = make_batch(
            (16..32).collect(),
            (0..16)
                .map(|_| vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
                .collect(),
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        assert_eq!(
            dataset
                .load_indices_by_name("plaid_idx")
                .await
                .unwrap()
                .len(),
            2
        );

        let mut scanner = dataset.scan();
        scanner.prefilter(true);
        scanner.filter("id < 0").unwrap();
        scanner.nearest("mv", &query(), 10).unwrap();
        scanner.project(&["id"]).unwrap();
        let analyzed = scanner.analyze_plan().await.unwrap();
        assert!(
            analyzed.contains("plaid_empty_filter_queries=1"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_segments_searched=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_empty_segments=2"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_centroids_probed=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_probe_rounds=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_posting_entries=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_plan_total_time"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        assert!(
            analyzed.contains("plaid_eligible_centroid_core_build_sub_time"),
            "unexpected analyzed plan:\n{analyzed}"
        );
        let result = scanner.try_into_batch().await.unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[tokio::test]
    async fn plaid_compaction_is_refused_before_rewrite_for_all_row_id_modes() {
        for stable_row_ids in [false, true] {
            for defer_index_remap in [false, true] {
                let directory = TempStrDir::default();
                let first = make_batch(
                    vec![0, 1],
                    vec![vec![[1.0, 0.0, 0.0, 0.0]], vec![[0.0, 1.0, 0.0, 0.0]]],
                );
                let schema = first.schema();
                let reader = RecordBatchIterator::new(vec![Ok(first)], schema.clone());
                let mut dataset = Dataset::write(
                    reader,
                    directory.as_ref(),
                    Some(WriteParams {
                        max_rows_per_file: 2,
                        enable_stable_row_ids: stable_row_ids,
                        ..Default::default()
                    }),
                )
                .await
                .unwrap();
                let second = make_batch(
                    vec![2, 3],
                    vec![vec![[0.8, 0.0, 0.0, 0.0]], vec![[0.0, 0.8, 0.0, 0.0]]],
                );
                dataset
                    .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
                    .await
                    .unwrap();
                dataset
                    .create_index(
                        &["mv"],
                        IndexType::Vector,
                        Some("plaid_idx".to_string()),
                        &PlaidIndexParams {
                            num_centroids: 2,
                            nbits: 2,
                            max_iterations: 3,
                            sample_rate: 4,
                        },
                        false,
                    )
                    .await
                    .unwrap();

                let version = dataset.version().version;
                let fragments = dataset.fragments().as_ref().clone();
                let error = compact_files(
                    &mut dataset,
                    CompactionOptions {
                        target_rows_per_fragment: 100,
                        defer_index_remap,
                        ..Default::default()
                    },
                    None,
                )
                .await
                .unwrap_err();
                if stable_row_ids && defer_index_remap {
                    assert!(error.to_string().contains("stable row IDs"));
                } else {
                    assert!(error.to_string().contains("PLAID"));
                }
                assert_eq!(dataset.version().version, version);
                assert_eq!(dataset.fragments().as_ref(), &fragments);
            }
        }
    }

    #[tokio::test]
    async fn plaid_roundtrip_filter_delete_refine_ties_and_plan_shape() {
        let directory = TempStrDir::default();
        let first = make_batch(
            vec![0, 1],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ],
        );
        let schema = first.schema();
        let reader = RecordBatchIterator::new(vec![Ok(first)], schema.clone());
        let mut dataset = Dataset::write(reader, directory.as_ref(), None)
            .await
            .unwrap();
        let second = make_batch(
            vec![2, 3],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                vec![[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
            ],
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();

        let params = PlaidIndexParams {
            num_centroids: 2,
            nbits: 2,
            max_iterations: 3,
            sample_rate: 4,
        };
        dataset
            .create_index(
                &["mv"],
                IndexType::Vector,
                Some("plaid_idx".to_string()),
                &params,
                false,
            )
            .await
            .unwrap();
        let metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
        assert_eq!(metadata.len(), 1);
        assert!(is_plaid_index_metadata(&metadata[0]));
        let opened = dataset
            .open_vector_index("mv", &metadata[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        let plaid = opened.as_any().downcast_ref::<PlaidVectorIndex>().unwrap();
        assert_eq!(plaid.metric_type(), DistanceType::Dot);
        assert_eq!(plaid.core.row_addresses().len(), 4);
        assert_eq!(
            RowAddress::from(plaid.core.row_addresses()[0]).fragment_id(),
            0
        );
        assert_eq!(
            RowAddress::from(plaid.core.row_addresses()[2]).fragment_id(),
            1
        );

        drop(opened);
        drop(metadata);
        drop(dataset);
        let session = Arc::new(Session::default());
        let mut dataset = DatasetBuilder::from_uri(directory.as_ref())
            .with_session(session)
            .load()
            .await
            .unwrap();
        let metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
        assert!(is_plaid_index_metadata(&metadata[0]));
        dataset
            .open_vector_index("mv", &metadata[0].uuid, &NoOpMetricsCollector)
            .await
            .unwrap();

        let mut plan_scanner = dataset.scan();
        plan_scanner.nearest("mv", &query(), 3).unwrap();
        plan_scanner.project(&["id"]).unwrap();
        let plan = plan_scanner.explain_plan(false).await.unwrap();
        assert!(plan.contains("PlaidSearch"), "unexpected plan:\n{plan}");
        assert!(plan.contains("mode=index_only"), "unexpected plan:\n{plan}");
        assert!(
            plan.contains("raw_refinement_budget=0"),
            "unexpected plan:\n{plan}"
        );
        assert!(
            !plan.contains("MultivectorScoring"),
            "unexpected plan:\n{plan}"
        );
        assert!(!plan.contains("ANNSubIndex"), "unexpected plan:\n{plan}");
        let index_only_analyzed = plan_scanner.analyze_plan().await.unwrap();
        assert!(
            index_only_analyzed.contains("plaid_index_only_queries=1"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );
        assert!(
            index_only_analyzed.contains("plaid_row_id_only_rows=0"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );
        assert!(
            index_only_analyzed.contains("plaid_raw_vector_rows=0"),
            "unexpected analyzed plan:\n{index_only_analyzed}"
        );

        let result = search_ids(&dataset, None, 3).await;
        assert_eq!(
            result["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[0, 1, 2]
        );
        let distances = result[DIST_COL].as_primitive::<Float32Type>();
        assert!((distances.value(0) - -1.0).abs() < 1.0e-5);
        assert!((distances.value(1) - -1.0).abs() < 1.0e-5);
        assert!((distances.value(2) - 0.0).abs() < 1.0e-5);

        let mut bounded_scanner = dataset.scan();
        bounded_scanner.nearest("mv", &query(), 4).unwrap();
        bounded_scanner.refine(1);
        bounded_scanner.distance_range(Some(-0.5), Some(0.5));
        bounded_scanner.project(&["id"]).unwrap();
        let bounded = bounded_scanner.try_into_batch().await.unwrap();
        assert_eq!(
            bounded["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[2]
        );

        let allowed = search_ids(&dataset, Some("id IN (1, 2)"), 3).await;
        assert_eq!(
            allowed["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[1, 2]
        );
        let blocked = search_ids(&dataset, Some("id != 0"), 3).await;
        assert!(
            !blocked["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&0)
        );

        dataset.delete("id = 0").await.unwrap();
        let after_delete = search_ids(&dataset, None, 3).await;
        assert_eq!(
            after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .value(0),
            1
        );
        assert!(
            !after_delete["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values()
                .contains(&0)
        );

        let appended = make_batch(
            vec![4],
            vec![vec![[0.75, 0.0, 0.0, 0.0], [0.0, 0.75, 0.0, 0.0]]],
        );
        let appended_schema = appended.schema();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(appended)], appended_schema),
                None,
            )
            .await
            .unwrap();
        let with_flat_fallback = search_ids(&dataset, None, 4).await;
        let fallback_ids = with_flat_fallback["id"].as_primitive::<arrow::datatypes::Int32Type>();
        let appended_position = fallback_ids
            .values()
            .iter()
            .position(|id| *id == 4)
            .expect("unindexed appended row must be merged with PLAID results");
        let fallback_distances = with_flat_fallback[DIST_COL].as_primitive::<Float32Type>();
        assert!((fallback_distances.value(appended_position) - -0.5).abs() < 1.0e-5);

        let mut fallback_bounds = dataset.scan();
        fallback_bounds.nearest("mv", &query(), 4).unwrap();
        fallback_bounds.distance_range(Some(-0.75), Some(-0.25));
        fallback_bounds.project(&["id"]).unwrap();
        let fallback_bounds = fallback_bounds.try_into_batch().await.unwrap();
        assert_eq!(
            fallback_bounds["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[4]
        );
    }
}

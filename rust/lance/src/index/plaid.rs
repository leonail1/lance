// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Database-native PLAID index build, persistence, and vector-index glue.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

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
    Eligibility, PlaidIndex, PlaidSearchParams, PlaidSearchStats, ResidualQuantizer,
};
use lance_select::RowAddrMask;
use lance_table::format::{IndexFile, IndexMetadata};
use ndarray::{Array2, ArrayView2};
use prost_types::Any as ProstAny;
use roaring::RoaringBitmap;
use serde_json::json;
use uuid::Uuid;

use super::vector::utils::{
    filter_finite_training_data, get_vector_type, maybe_sample_training_data,
};
use crate::dataset::Dataset;

/// Built-in index family name used by [`PlaidIndexParams`].
pub const LANCE_PLAID_INDEX: &str = "PLAID";
/// Persisted `VectorIndexDetails.runtime_hints` discriminator.
pub const PLAID_RUNTIME_HINT: &str = "lance.plaid";
const PLAID_FORMAT_HINT: &str = "lance.plaid.format_version";
const PLAID_NBITS_HINT: &str = "lance.plaid.nbits";

/// Build parameters for the database-native CPU PLAID index.
#[derive(Clone, Debug)]
pub struct PlaidIndexParams {
    /// Requested number of coarse centroids. Small datasets use fewer.
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
            num_centroids: 1024,
            nbits: 2,
            max_iterations: 20,
            sample_rate: 256,
        }
    }
}

impl PlaidIndexParams {
    fn validate(&self) -> Result<()> {
        if self.num_centroids == 0 || self.max_iterations == 0 || self.sample_rate == 0 {
            return Err(Error::invalid_input(format!(
                "PLAID num_centroids, max_iterations, and sample_rate must be positive, got {}, {}, {}",
                self.num_centroids, self.max_iterations, self.sample_rate
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
    ]);
    let details = VectorIndexDetails {
        metric_type: VectorMetricType::Cosine.into(),
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
    let sample_size = params
        .num_centroids
        .checked_mul(params.sample_rate)
        .ok_or_else(|| Error::invalid_input("PLAID training sample size overflow".to_string()))?;
    let training = maybe_sample_training_data(dataset, column, sample_size, None).await?;
    let training = filter_finite_training_data(training)?;
    if training.is_empty() {
        return Err(Error::invalid_input(
            "PLAID cannot train on an empty multi-vector column".to_string(),
        ));
    }
    let num_centroids = params.num_centroids.min(training.len());
    let raw_training_values = training.values().as_primitive::<Float32Type>().clone();
    let mut normalized_training_values = raw_training_values.values().to_vec();
    for token in normalized_training_values.chunks_exact_mut(dimension) {
        normalize_token(token)?;
    }
    let normalized_training_values = Float32Array::from(normalized_training_values);
    // Lance KMeans lacks direct Cosine membership. L2 assignment on unit
    // vectors is equivalent to maximum dot-product assignment.
    let kmeans_params = lance_index::vector::kmeans::KMeansParams::new(
        None,
        params.max_iterations,
        1,
        DistanceType::L2,
    );
    let model = lance_index::vector::kmeans::train_kmeans::<Float32Type>(
        &normalized_training_values,
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
    let index = PlaidIndex::read_from_bytes(&bytes).map_err(plaid_error)?;
    Ok(Arc::new(PlaidVectorIndex::try_new(index)?))
}

/// Loaded database-native PLAID segment.
#[derive(Debug)]
pub(crate) struct PlaidVectorIndex {
    core: Arc<PlaidIndex>,
    ivf_model: IvfModel,
    quantizer: FlatQuantizer,
}

impl PlaidVectorIndex {
    fn try_new(core: PlaidIndex) -> Result<Self> {
        let dimension = core.dimension();
        let values = Float32Array::from(core.centroids().iter().copied().collect::<Vec<_>>());
        let centroids = FixedSizeListArray::try_new_from_values(values, dimension as i32)?;
        Ok(Self {
            core: Arc::new(core),
            ivf_model: IvfModel::new(centroids, None),
            quantizer: FlatQuantizer::new(dimension, DistanceType::Cosine),
        })
    }

    pub(crate) fn candidate_params(&self, query: &Query, top_k: usize) -> PlaidSearchParams {
        let n_ivf_probe = query
            .maximum_nprobes
            .unwrap_or_else(|| query.minimum_nprobes.max(8))
            .min(self.core.num_centroids())
            .max(1);
        PlaidSearchParams {
            n_ivf_probe,
            n_full_scores: top_k.saturating_mul(4).max(top_k),
            top_k,
            centroid_score_threshold: Some(0.4),
        }
    }

    pub(crate) fn search_candidates(
        &self,
        query: ArrayView2<'_, f32>,
        params: &PlaidSearchParams,
        mask: &RowAddrMask,
    ) -> Result<(Vec<lance_plaid::SearchHit>, PlaidSearchStats)> {
        self.core
            .search(query, params, &MaskEligibility { mask })
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
        let params = self.candidate_params(query, query.k);
        let (hits, stats) =
            self.search_candidates(query_tokens.view(), &params, &pre_filter.mask())?;
        metrics.record_comparisons(
            usize::try_from(stats.exact_documents)
                .unwrap_or(usize::MAX)
                .saturating_mul(query_tokens.nrows()),
        );
        let distances = Float32Array::from(
            hits.iter()
                .map(|hit| query_tokens.nrows() as f32 - hit.score)
                .collect::<Vec<_>>(),
        );
        let row_addresses =
            UInt64Array::from(hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>());
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
            "PLAID row-address remapping requires rebuilding the segment".to_string(),
        ))
    }

    fn metric_type(&self) -> DistanceType {
        DistanceType::Cosine
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
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::metrics::NoOpMetricsCollector;

    use crate::DatasetBuilder;
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
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

    fn query() -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            4,
        )
        .unwrap()
    }

    async fn search_ids(dataset: &Dataset, filter: Option<&str>, k: usize) -> RecordBatch {
        let mut scanner = dataset.scan();
        if let Some(filter) = filter {
            scanner.prefilter(true);
            scanner.filter(filter).unwrap();
        }
        scanner.nearest("mv", &query(), k).unwrap();
        scanner.project(&["id"]).unwrap();
        scanner.try_into_batch().await.unwrap()
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
        assert!(
            !plan.contains("MultivectorScoring"),
            "unexpected plan:\n{plan}"
        );
        assert!(!plan.contains("ANNSubIndex"), "unexpected plan:\n{plan}");

        let result = search_ids(&dataset, None, 3).await;
        assert_eq!(
            result["id"]
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[0, 1, 2]
        );
        let distances = result[DIST_COL].as_primitive::<Float32Type>();
        assert!((distances.value(0) - 0.0).abs() < 1.0e-5);
        assert!((distances.value(1) - 0.0).abs() < 1.0e-5);
        assert!((distances.value(2) - 1.0).abs() < 1.0e-5);

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
    }
}

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
use lance_table::format::{Fragment, IndexFile, IndexMetadata};
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

/// Classifies one logical index group before any maintenance writer is opened.
/// A PLAID discriminator on only some segments is a hard metadata-family error;
/// allowing that group into either maintenance implementation could reproduce
/// the format/details mismatch this gate is intended to prevent.
pub(crate) fn index_group_is_plaid(indices: &[&IndexMetadata]) -> Result<bool> {
    let plaid_segments = indices
        .iter()
        .filter(|metadata| is_plaid_index_metadata(metadata))
        .count();
    if plaid_segments > 0 && plaid_segments != indices.len() {
        return Err(Error::index(format!(
            "logical index '{}' mixes {plaid_segments} PLAID segment(s) with {} non-PLAID segment(s)",
            indices
                .first()
                .map(|metadata| metadata.name.as_str())
                .unwrap_or("<empty>"),
            indices.len() - plaid_segments
        )));
    }
    Ok(!indices.is_empty() && plaid_segments == indices.len())
}

/// Trained PLAID state that can be reused to encode immutable delta segments.
///
/// Append optimization intentionally reuses this state instead of training on
/// a small delta. This keeps quantized MaxSim scores comparable across all
/// physical segments of one logical index.
pub(crate) struct PlaidTrainedModel {
    centroids: Array2<f32>,
    quantizer: ResidualQuantizer,
}

/// Exact canonical identity of every trained value that affects PLAID encoding
/// and quantized MaxSim. Using IEEE-754 bits avoids tolerance-based acceptance
/// of subtly incompatible segment models.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PlaidModelFingerprint {
    dimension: usize,
    num_centroids: usize,
    nbits: u8,
    centroid_bits: Vec<u32>,
    cutoff_bits: Vec<u32>,
    weight_bits: Vec<u32>,
}

impl PlaidTrainedModel {
    pub(crate) fn fingerprint(&self) -> PlaidModelFingerprint {
        PlaidModelFingerprint {
            dimension: self.centroids.ncols(),
            num_centroids: self.centroids.nrows(),
            nbits: self.quantizer.nbits(),
            centroid_bits: self.centroids.iter().map(|value| value.to_bits()).collect(),
            cutoff_bits: self
                .quantizer
                .bucket_cutoffs()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
            weight_bits: self
                .quantizer
                .bucket_weights()
                .iter()
                .map(|value| value.to_bits())
                .collect(),
        }
    }
}

/// Opens and fully validates one persisted PLAID segment, then clones only its
/// trained state. Calling this for every existing segment before an append also
/// prevents a previously mislabeled generic vector file from being propagated.
pub(crate) async fn load_plaid_trained_model(
    dataset: &Dataset,
    metadata: &IndexMetadata,
) -> Result<PlaidTrainedModel> {
    if metadata.fields.len() != 1 {
        return Err(Error::index(format!(
            "PLAID segment {} must reference exactly one field, got {:?}",
            metadata.uuid, metadata.fields
        )));
    }
    let details = metadata
        .index_details
        .as_ref()
        .ok_or_else(|| {
            Error::index(format!(
                "PLAID segment {} is missing VectorIndexDetails",
                metadata.uuid
            ))
        })?
        .to_msg::<VectorIndexDetails>()
        .map_err(|error| {
            Error::index(format!(
                "PLAID segment {} has invalid VectorIndexDetails: {error}",
                metadata.uuid
            ))
        })?;
    if !details
        .runtime_hints
        .get(PLAID_RUNTIME_HINT)
        .is_some_and(|value| value.eq_ignore_ascii_case("true"))
    {
        return Err(Error::index(format!(
            "PLAID segment {} is missing the PLAID runtime discriminator",
            metadata.uuid
        )));
    }
    if details
        .runtime_hints
        .get(PLAID_FORMAT_HINT)
        .map(String::as_str)
        != Some("1")
    {
        return Err(Error::index(format!(
            "PLAID segment {} must use format version 1",
            metadata.uuid
        )));
    }
    let hinted_nbits = details
        .runtime_hints
        .get(PLAID_NBITS_HINT)
        .ok_or_else(|| {
            Error::index(format!(
                "PLAID segment {} is missing the persisted nbits hint",
                metadata.uuid
            ))
        })?
        .parse::<u8>()
        .map_err(|error| {
            Error::index(format!(
                "PLAID segment {} has an invalid nbits hint: {error}",
                metadata.uuid
            ))
        })?;
    if VectorMetricType::try_from(details.metric_type).ok() != Some(VectorMetricType::Dot)
        || !matches!(details.compression.as_ref(), Some(Compression::Flat(_)))
    {
        return Err(Error::index(format!(
            "PLAID segment {} must use Dot distance with Flat compression",
            metadata.uuid
        )));
    }

    let files = metadata.files.as_deref().ok_or_else(|| {
        Error::index(format!(
            "PLAID segment {} is missing its persisted file manifest",
            metadata.uuid
        ))
    })?;
    if files.len() != 1 || files[0].path != INDEX_FILE_NAME {
        return Err(Error::index(format!(
            "PLAID segment {} must declare exactly one '{}' file, got {:?}",
            metadata.uuid,
            INDEX_FILE_NAME,
            files
                .iter()
                .map(|file| file.path.as_str())
                .collect::<Vec<_>>()
        )));
    }
    let fragment_bitmap = metadata.fragment_bitmap.as_ref().ok_or_else(|| {
        Error::index(format!(
            "PLAID segment {} is missing fragment coverage",
            metadata.uuid
        ))
    })?;
    let store = dataset.object_store_for_index(metadata).await?;
    let path = dataset
        .indice_files_dir(metadata)?
        .join(metadata.uuid.to_string())
        .join(INDEX_FILE_NAME);
    let actual_size = store.size(&path).await?;
    if actual_size != files[0].size_bytes {
        return Err(Error::index(format!(
            "PLAID segment {} declares file size {}, actual size is {}",
            metadata.uuid, files[0].size_bytes, actual_size
        )));
    }
    let bytes = store.read_one_all(&path).await?;
    if u64::try_from(bytes.len()).ok() != Some(files[0].size_bytes) {
        return Err(Error::index(format!(
            "PLAID segment {} read length does not match its declared file size",
            metadata.uuid
        )));
    }
    const PLAID_MAGIC: &[u8; 8] = b"LPLDIDX\0";
    if !bytes.as_ref().starts_with(PLAID_MAGIC) {
        return Err(Error::index(format!(
            "PLAID segment {} has invalid magic bytes {:?}",
            metadata.uuid,
            bytes.as_ref().get(..8).unwrap_or(bytes.as_ref())
        )));
    }
    let index =
        spawn_cpu(move || PlaidIndex::read_from_bytes(bytes.as_ref()).map_err(plaid_error)).await?;
    if index.quantizer().nbits() != hinted_nbits {
        return Err(Error::index(format!(
            "PLAID segment {} persists nbits={}, but its core uses nbits={}",
            metadata.uuid,
            hinted_nbits,
            index.quantizer().nbits()
        )));
    }
    if let Some(row_address) =
        index.row_addresses().iter().copied().find(|row_address| {
            !fragment_bitmap.contains(RowAddress::from(*row_address).fragment_id())
        })
    {
        return Err(Error::index(format!(
            "PLAID segment {} contains row address {} outside its fragment bitmap",
            metadata.uuid, row_address
        )));
    }

    let field_path = dataset.schema().field_path(metadata.fields[0])?;
    let (vector_type, element_type) = get_vector_type(dataset.schema(), &field_path)?;
    let schema_dimension = match vector_type {
        DataType::List(ref item) => match item.data_type() {
            DataType::FixedSizeList(_, dimension) if element_type == DataType::Float32 => {
                usize::try_from(*dimension).map_err(|_| {
                    Error::invalid_input("PLAID vector dimension does not fit usize".to_string())
                })?
            }
            _ => {
                return Err(Error::index(format!(
                    "PLAID segment {} field '{}' is not List<FixedSizeList<Float32>>",
                    metadata.uuid, field_path
                )));
            }
        },
        _ => {
            return Err(Error::index(format!(
                "PLAID segment {} field '{}' is not a multi-vector List column",
                metadata.uuid, field_path
            )));
        }
    };
    if index.dimension() != schema_dimension {
        return Err(Error::index(format!(
            "PLAID segment {} dimension {} differs from field '{}' dimension {}",
            metadata.uuid,
            index.dimension(),
            field_path,
            schema_dimension
        )));
    }

    Ok(PlaidTrainedModel {
        centroids: index.centroids().to_owned(),
        quantizer: index.quantizer().clone(),
    })
}

/// Validates every immutable segment in one logical PLAID index before a new
/// object is written, and returns models in the same order as indices.
pub(crate) async fn load_compatible_plaid_models(
    dataset: &Dataset,
    indices: &[&IndexMetadata],
) -> Result<Vec<PlaidTrainedModel>> {
    let Some(reference) = indices.first() else {
        return Err(Error::index(
            "cannot validate an empty PLAID index group".to_string(),
        ));
    };
    let mut models = Vec::with_capacity(indices.len());
    let mut expected_fingerprint = None;
    for metadata in indices {
        if metadata.name != reference.name
            || metadata.fields != reference.fields
            || metadata.index_version != reference.index_version
        {
            return Err(Error::index(format!(
                "PLAID logical index '{}' has inconsistent segment metadata at {}",
                reference.name, metadata.uuid
            )));
        }
        let model = load_plaid_trained_model(dataset, metadata).await?;
        let fingerprint = model.fingerprint();
        match expected_fingerprint.as_ref() {
            Some(expected) if expected != &fingerprint => {
                return Err(Error::index(format!(
                    "PLAID logical index '{}' contains incompatible trained models; segment {} differs from the other segments",
                    reference.name, metadata.uuid
                )));
            }
            Some(_) => {}
            None => expected_fingerprint = Some(fingerprint),
        }
        models.push(model);
    }
    Ok(models)
}

/// Builds one append-only PLAID delta over exactly `fragments` using an existing
/// trained model. The scanner still reads physical row addresses when stable
/// row IDs are enabled; deletion vectors are applied by the dataset scan.
pub(crate) async fn build_plaid_delta_index(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    model: PlaidTrainedModel,
    fragments: Vec<Fragment>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<Vec<IndexFile>> {
    let fragment_bitmap = fragments
        .iter()
        .map(|fragment| fragment.id as u32)
        .collect();
    encode_plaid_segment(
        dataset,
        column,
        uuid,
        model.centroids,
        model.quantizer,
        Some(fragments),
        Some(fragment_bitmap),
        progress,
    )
    .await
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

    encode_plaid_segment(
        dataset, column, uuid, centroids, quantizer, None, None, progress,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn encode_plaid_segment(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    centroids: Array2<f32>,
    quantizer: ResidualQuantizer,
    fragments: Option<Vec<Fragment>>,
    expected_fragment_bitmap: Option<RoaringBitmap>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<Vec<IndexFile>> {
    let dimension = centroids.ncols();
    progress
        .stage_start("plaid_encode", None, "record batches")
        .await?;
    let mut scanner = dataset.scan();
    scanner.project(&[column])?.with_row_address();
    if let Some(fragments) = fragments {
        scanner.with_fragments(fragments);
    }
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

    let index = PlaidIndex::try_new(
        centroids,
        quantizer,
        row_addresses,
        document_offsets,
        token_codes,
        packed_residuals,
    )
    .map_err(plaid_error)?;
    let expected_row_addresses = index.row_addresses().to_vec();
    let bytes = index.to_bytes().map_err(plaid_error)?;
    let path = dataset
        .indices_dir()
        .join(uuid.to_string())
        .join(INDEX_FILE_NAME);
    dataset.object_store.put(&path, &bytes).await?;

    // Do not allow metadata to be committed until the object store returns the
    // exact PLAID object we just wrote and the complete decoder accepts it.
    let readback = dataset.object_store.read_one_all(&path).await?;
    if readback.as_ref() != bytes.as_slice() {
        return Err(Error::index(format!(
            "PLAID readback bytes differ from the object written for {uuid}"
        )));
    }
    let readback_size = readback.len();
    const PLAID_MAGIC: &[u8; 8] = b"LPLDIDX\0";
    if !readback.as_ref().starts_with(PLAID_MAGIC) {
        return Err(Error::index(format!(
            "PLAID readback for {uuid} has invalid magic bytes {:?}",
            readback.as_ref().get(..8).unwrap_or(readback.as_ref())
        )));
    }
    let decoded =
        spawn_cpu(move || PlaidIndex::read_from_bytes(readback.as_ref()).map_err(plaid_error))
            .await?;
    if decoded.row_addresses() != expected_row_addresses {
        return Err(Error::index(format!(
            "PLAID readback row-address mismatch for {uuid}: expected {} rows, decoded {}",
            expected_row_addresses.len(),
            decoded.row_addresses().len()
        )));
    }
    if let Some(expected_fragment_bitmap) = expected_fragment_bitmap.as_ref()
        && let Some(unexpected_address) = decoded.row_addresses().iter().copied().find(|address| {
            !expected_fragment_bitmap.contains(RowAddress::from(*address).fragment_id())
        })
    {
        return Err(Error::index(format!(
            "PLAID delta {uuid} contains row address {unexpected_address} outside its fragment bitmap"
        )));
    }
    let size_bytes = u64::try_from(readback_size)
        .map_err(|_| Error::index("PLAID file size does not fit u64".to_string()))?;
    if size_bytes != dataset.object_store.size(&path).await? {
        return Err(Error::index(format!(
            "PLAID readback file-size mismatch for {uuid}"
        )));
    }
    Ok(vec![IndexFile {
        path: INDEX_FILE_NAME.to_string(),
        size_bytes,
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
    pub(crate) fn try_new(core: PlaidIndex) -> Result<Self> {
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
    use arrow_array::{ArrayRef, Int32Array, RecordBatchIterator};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::optimize::OptimizeOptions;

    use crate::DatasetBuilder;
    use crate::dataset::WriteParams;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
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

    fn make_nullable_batch(ids: Vec<i32>, documents: Vec<Option<Vec<[f32; 4]>>>) -> RecordBatch {
        let token_builder = FixedSizeListBuilder::new(Float32Builder::new(), 4);
        let mut document_builder = ListBuilder::new(token_builder);
        for document in documents {
            if let Some(document) = document {
                for token in document {
                    document_builder.values().values().append_slice(&token);
                    document_builder.values().append(true);
                }
                document_builder.append(true);
            } else {
                document_builder.append(false);
            }
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
        scanner.refine(2);
        scanner.project(&["id"]).unwrap();
        scanner.try_into_batch().await.unwrap()
    }

    fn sorted_ids(batch: &RecordBatch) -> Vec<i32> {
        let mut ids = batch["id"]
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values()
            .to_vec();
        ids.sort_unstable();
        ids
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
    async fn plaid_append_delta_roundtrips_for_physical_and_stable_row_ids() {
        for stable_row_ids in [false, true] {
            let directory = TempStrDir::default();
            let first = make_batch(
                vec![0, 1, 2, 3],
                vec![
                    vec![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                    vec![[1.0, 0.0, 0.0, 0.0]],
                    vec![[0.0, 1.0, 0.0, 0.0]],
                    vec![[-1.0, 0.0, 0.0, 0.0]],
                ],
            );
            let schema = first.schema();
            let mut dataset = Dataset::write(
                RecordBatchIterator::new(vec![Ok(first)], schema.clone()),
                directory.as_ref(),
                Some(WriteParams {
                    max_rows_per_file: 4,
                    enable_stable_row_ids: stable_row_ids,
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
                vec![4, 5, 6, 7],
                vec![
                    vec![[0.9, 0.0, 0.0, 0.0], [0.0, 0.9, 0.0, 0.0]],
                    vec![[0.8, 0.0, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]],
                    vec![[1.0, 0.0, 0.0, 0.0]],
                    vec![[0.0, 1.0, 0.0, 0.0]],
                ],
            );
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(second)], schema.clone()),
                    None,
                )
                .await
                .unwrap();
            // The delta scanner must apply this deletion vector and never
            // persist physical row offset 1 from fragment 1.
            dataset.delete("id = 5").await.unwrap();
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();

            let mut metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
            metadata.sort_by_key(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .and_then(|bitmap| bitmap.iter().next())
                    .unwrap_or(u32::MAX)
            });
            assert_eq!(metadata.len(), 2);
            assert_eq!(
                metadata[0].fragment_bitmap,
                Some(RoaringBitmap::from_iter([0]))
            );
            assert_eq!(
                metadata[1].fragment_bitmap,
                Some(RoaringBitmap::from_iter([1]))
            );

            for segment in &metadata {
                assert!(is_plaid_index_metadata(segment));
                let files = segment.files.as_ref().unwrap();
                assert_eq!(files.len(), 1);
                assert_eq!(files[0].path, INDEX_FILE_NAME);
                let store = dataset.object_store_for_index(segment).await.unwrap();
                let path = dataset
                    .indice_files_dir(segment)
                    .unwrap()
                    .join(segment.uuid.to_string())
                    .join(INDEX_FILE_NAME);
                let bytes = store.read_one_all(&path).await.unwrap();
                assert!(bytes.as_ref().starts_with(b"LPLDIDX\0"));
                assert_eq!(files[0].size_bytes, bytes.len() as u64);
                PlaidIndex::read_from_bytes(bytes.as_ref()).unwrap();
            }

            let old_opened = open_plaid_index(&dataset, &metadata[0]).await.unwrap();
            let new_opened = open_plaid_index(&dataset, &metadata[1]).await.unwrap();
            let old_plaid = old_opened
                .as_any()
                .downcast_ref::<PlaidVectorIndex>()
                .unwrap();
            let new_plaid = new_opened
                .as_any()
                .downcast_ref::<PlaidVectorIndex>()
                .unwrap();
            assert_eq!(old_plaid.core.centroids(), new_plaid.core.centroids());
            assert_eq!(
                old_plaid.core.quantizer().bucket_cutoffs(),
                new_plaid.core.quantizer().bucket_cutoffs()
            );
            assert_eq!(
                old_plaid.core.quantizer().bucket_weights(),
                new_plaid.core.quantizer().bucket_weights()
            );
            assert_eq!(new_plaid.core.num_documents(), 3);
            let new_addresses = new_plaid
                .core
                .row_addresses()
                .iter()
                .copied()
                .map(RowAddress::from)
                .collect::<Vec<_>>();
            assert!(
                new_addresses
                    .iter()
                    .all(|address| address.fragment_id() == 1)
            );
            assert_eq!(
                new_addresses
                    .iter()
                    .map(RowAddress::row_offset)
                    .collect::<Vec<_>>(),
                vec![0, 2, 3]
            );

            // Index-only search must open and merge both true PLAID segments.
            let mut index_only = dataset.scan();
            index_only.nearest("mv", &query(), 7).unwrap();
            index_only.project(&["id"]).unwrap();
            let index_only = index_only.try_into_batch().await.unwrap();
            assert_eq!(sorted_ids(&index_only), vec![0, 1, 2, 3, 4, 6, 7]);

            // Filtered exact refinement exercises the database prefilter and
            // raw-vector tail over only the appended segment.
            let filtered = search_ids(&dataset, Some("id >= 4"), 4).await;
            assert_eq!(sorted_ids(&filtered), vec![4, 6, 7]);

            drop(old_opened);
            drop(new_opened);
            drop(metadata);
            drop(dataset);

            let session = Arc::new(Session::default());
            let mut dataset = DatasetBuilder::from_uri(directory.as_ref())
                .with_session(session)
                .load()
                .await
                .unwrap();
            let reopened = search_ids(&dataset, None, 7).await;
            assert_eq!(sorted_ids(&reopened), vec![0, 1, 2, 3, 4, 6, 7]);
            let reopened_filtered = search_ids(&dataset, Some("id >= 4"), 4).await;
            assert_eq!(sorted_ids(&reopened_filtered), vec![4, 6, 7]);

            // A second append creates a third immutable segment, while a
            // repeated append optimize at steady state is a strict no-op.
            let third = make_batch(
                vec![8, 9],
                vec![
                    vec![[0.95, 0.0, 0.0, 0.0], [0.0, 0.95, 0.0, 0.0]],
                    vec![[-1.0, 0.0, 0.0, 0.0]],
                ],
            );
            dataset
                .append(
                    RecordBatchIterator::new(vec![Ok(third)], schema.clone()),
                    None,
                )
                .await
                .unwrap();
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
            let before_noop_version = dataset.version().version;
            let before_noop_metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
            let mut expected_fingerprint = None;
            for segment in &before_noop_metadata {
                let fingerprint = load_plaid_trained_model(&dataset, segment)
                    .await
                    .unwrap()
                    .fingerprint();
                if let Some(expected_fingerprint) = expected_fingerprint.as_ref() {
                    assert_eq!(&fingerprint, expected_fingerprint);
                } else {
                    expected_fingerprint = Some(fingerprint);
                }
            }
            let mut before_noop_uuids = before_noop_metadata
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>();
            before_noop_uuids.sort_unstable();
            assert_eq!(before_noop_uuids.len(), 3);
            dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
            assert_eq!(dataset.version().version, before_noop_version);
            let mut after_noop_uuids = dataset
                .load_indices_by_name("plaid_idx")
                .await
                .unwrap()
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>();
            after_noop_uuids.sort_unstable();
            assert_eq!(after_noop_uuids, before_noop_uuids);
            let after_second_append = search_ids(&dataset, None, 9).await;
            assert_eq!(
                sorted_ids(&after_second_append),
                vec![0, 1, 2, 3, 4, 6, 7, 8, 9]
            );
            drop(dataset);
            let dataset = DatasetBuilder::from_uri(directory.as_ref())
                .with_session(Arc::new(Session::default()))
                .load()
                .await
                .unwrap();
            assert_eq!(
                sorted_ids(&search_ids(&dataset, None, 9).await),
                vec![0, 1, 2, 3, 4, 6, 7, 8, 9]
            );
            assert_eq!(
                sorted_ids(&search_ids(&dataset, Some("id >= 8"), 2).await),
                vec![8, 9]
            );
            // Prove the strict no-op does not even read trained model files:
            // remove every persisted core, reopen with a fresh session (no
            // index cache), and optimize the fully covered logical index.
            for segment in &before_noop_metadata {
                let store = dataset.object_store_for_index(segment).await.unwrap();
                let path = dataset
                    .indice_files_dir(segment)
                    .unwrap()
                    .join(segment.uuid.to_string())
                    .join(INDEX_FILE_NAME);
                store.delete(&path).await.unwrap();
            }
            drop(before_noop_metadata);
            drop(dataset);
            let mut no_read_dataset = DatasetBuilder::from_uri(directory.as_ref())
                .with_session(Arc::new(Session::default()))
                .load()
                .await
                .unwrap();
            let no_read_version = no_read_dataset.version().version;
            no_read_dataset
                .optimize_indices(&OptimizeOptions::append())
                .await
                .unwrap();
            assert_eq!(no_read_dataset.version().version, no_read_version);
            let mut no_read_uuids = no_read_dataset
                .load_indices_by_name("plaid_idx")
                .await
                .unwrap()
                .iter()
                .map(|segment| segment.uuid)
                .collect::<Vec<_>>();
            no_read_uuids.sort_unstable();
            assert_eq!(no_read_uuids, before_noop_uuids);
        }
    }

    #[tokio::test]
    async fn plaid_append_delta_indexes_only_live_non_null_documents() {
        let directory = TempStrDir::default();
        let first = make_nullable_batch(
            vec![0, 1, 99],
            vec![
                Some(vec![[1.0, 0.0, 0.0, 0.0]]),
                Some(vec![[0.0, 1.0, 0.0, 0.0]]),
                None,
            ],
        );
        let schema = first.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first)], schema),
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

        // A fragment containing only null documents is still covered by a
        // valid empty PLAID segment, so future append calls do not retry it.
        let all_null = make_nullable_batch(vec![2, 3], vec![None, None]);
        let all_null_schema = all_null.schema();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(all_null)], all_null_schema),
                None,
            )
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
        let empty_segment = metadata
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(1))
            })
            .unwrap();
        let empty_opened = open_plaid_index(&dataset, empty_segment).await.unwrap();
        let empty_plaid = empty_opened
            .as_any()
            .downcast_ref::<PlaidVectorIndex>()
            .unwrap();
        assert_eq!(empty_plaid.core.num_documents(), 0);
        assert_eq!(sorted_ids(&search_ids(&dataset, None, 2).await), vec![0, 1]);
        // Empty physical segments have one stable residual decision regardless
        // of the query gate that the non-empty segment takes.
        let mut unfiltered = dataset.scan();
        unfiltered.nearest("mv", &query(), 1).unwrap();
        unfiltered.project(&["id"]).unwrap();
        let unfiltered_analyzed = unfiltered.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_skipped_empty_segments=1",
            "plaid_legacy_residual_segments=1",
            "plaid_direct_residual_skipped_unfiltered_segments=1",
        ] {
            assert!(
                unfiltered_analyzed.contains(expected),
                "missing {expected} in unfiltered empty-segment plan:\n{unfiltered_analyzed}"
            );
        }

        let mut explicit_ceiling = dataset.scan();
        explicit_ceiling.prefilter(true);
        explicit_ceiling.filter("id >= 0").unwrap();
        explicit_ceiling.nearest("mv", &query(), 1).unwrap();
        explicit_ceiling.maximum_nprobes(1);
        explicit_ceiling.project(&["id"]).unwrap();
        let explicit_ceiling_analyzed = explicit_ceiling.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_skipped_empty_segments=1",
            "plaid_legacy_residual_segments=1",
            "plaid_direct_residual_skipped_explicit_ceiling_segments=1",
        ] {
            assert!(
                explicit_ceiling_analyzed.contains(expected),
                "missing {expected} in explicit-ceiling empty-segment plan:\n{explicit_ceiling_analyzed}"
            );
        }

        let mut small_exact = dataset.scan();
        small_exact.prefilter(true);
        small_exact.filter("id = 0").unwrap();
        small_exact.nearest("mv", &query(), 2).unwrap();
        small_exact.project(&["id"]).unwrap();
        let small_exact_analyzed = small_exact.analyze_plan().await.unwrap();
        for expected in [
            "plaid_direct_residual_skipped_empty_segments=1",
            "plaid_legacy_residual_segments=0",
            "plaid_direct_residual_small_exact_precedence_queries=1",
        ] {
            assert!(
                small_exact_analyzed.contains(expected),
                "missing {expected} in small-exact empty-segment plan:\n{small_exact_analyzed}"
            );
        }

        // A mixed fragment persists only its non-null document and retains its
        // physical fragment/offset identity.
        let mixed = make_nullable_batch(vec![4, 5], vec![Some(vec![[0.9, 0.0, 0.0, 0.0]]), None]);
        let mixed_schema = mixed.schema();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(mixed)], mixed_schema),
                None,
            )
            .await
            .unwrap();
        dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap();
        let metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
        let mixed_segment = metadata
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(2))
            })
            .unwrap();
        let mixed_opened = open_plaid_index(&dataset, mixed_segment).await.unwrap();
        let mixed_plaid = mixed_opened
            .as_any()
            .downcast_ref::<PlaidVectorIndex>()
            .unwrap();
        assert_eq!(mixed_plaid.core.num_documents(), 1);
        let address = RowAddress::from(mixed_plaid.core.row_addresses()[0]);
        assert_eq!(address.fragment_id(), 2);
        assert_eq!(address.row_offset(), 0);
        assert_eq!(
            sorted_ids(&search_ids(&dataset, None, 3).await),
            vec![0, 1, 4]
        );
    }

    #[tokio::test]
    async fn plaid_unsupported_optimize_modes_and_mixed_families_fail_before_mutation() {
        let directory = TempStrDir::default();
        let first = make_batch(
            vec![0, 1, 2, 3],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0]],
                vec![[0.0, 1.0, 0.0, 0.0]],
                vec![[-1.0, 0.0, 0.0, 0.0]],
                vec![[0.0, -1.0, 0.0, 0.0]],
            ],
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
            vec![4, 5],
            vec![vec![[0.9, 0.0, 0.0, 0.0]], vec![[0.0, 0.9, 0.0, 0.0]]],
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();

        let before_version = dataset.version().version;
        let before_metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
        assert_eq!(before_metadata.len(), 1);
        let before_segment = before_metadata[0].clone();
        let store = dataset
            .object_store_for_index(&before_segment)
            .await
            .unwrap();
        let path = dataset
            .indice_files_dir(&before_segment)
            .unwrap()
            .join(before_segment.uuid.to_string())
            .join(INDEX_FILE_NAME);
        let before_bytes = store.read_one_all(&path).await.unwrap();
        // Reusable models are accepted only after the persisted metadata and
        // object are cross-validated as one coherent segment.
        let mut wrong_bitmap = before_segment.clone();
        wrong_bitmap.fragment_bitmap = Some(RoaringBitmap::new());
        let error = load_plaid_trained_model(&dataset, &wrong_bitmap)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("outside its fragment bitmap"));

        let mut wrong_size = before_segment.clone();
        wrong_size.files.as_mut().unwrap()[0].size_bytes += 1;
        let error = load_plaid_trained_model(&dataset, &wrong_size)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("declares file size"));

        let mut wrong_nbits = before_segment.clone();
        let mut wrong_nbits_details = wrong_nbits
            .index_details
            .as_ref()
            .unwrap()
            .to_msg::<VectorIndexDetails>()
            .unwrap();
        wrong_nbits_details
            .runtime_hints
            .insert(PLAID_NBITS_HINT.to_string(), "4".to_string());
        wrong_nbits.index_details =
            Some(Arc::new(ProstAny::from_msg(&wrong_nbits_details).unwrap()));
        let error = load_plaid_trained_model(&dataset, &wrong_nbits)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("core uses nbits=2"));

        let mut wrong_fields = before_segment.clone();
        wrong_fields.fields.clear();
        let wrong_field_group = vec![&before_segment, &wrong_fields];
        let error = load_compatible_plaid_models(&dataset, &wrong_field_group)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("inconsistent segment metadata"));
        let wrong_dimension_uuid = Uuid::new_v4();
        let wrong_dimension_quantizer =
            ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
        let wrong_dimension_core = PlaidIndex::try_new(
            Array2::zeros((1, 8)),
            wrong_dimension_quantizer,
            Vec::new(),
            vec![0],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        let wrong_dimension_bytes = wrong_dimension_core.to_bytes().unwrap();
        let wrong_dimension_path = dataset
            .indices_dir()
            .join(wrong_dimension_uuid.to_string())
            .join(INDEX_FILE_NAME);
        dataset
            .object_store
            .put(&wrong_dimension_path, &wrong_dimension_bytes)
            .await
            .unwrap();
        let mut wrong_dimension = before_segment.clone();
        wrong_dimension.uuid = wrong_dimension_uuid;
        wrong_dimension.fragment_bitmap = Some(RoaringBitmap::new());
        wrong_dimension.files = Some(vec![IndexFile {
            path: INDEX_FILE_NAME.to_string(),
            size_bytes: wrong_dimension_bytes.len() as u64,
        }]);
        let error = load_plaid_trained_model(&dataset, &wrong_dimension)
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("dimension 8"));

        for (mode, options) in [
            ("default", OptimizeOptions::default()),
            ("merge", OptimizeOptions::merge(1)),
            ("retrain", OptimizeOptions::retrain()),
        ] {
            let error = dataset.optimize_indices(&options).await.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("supports only OptimizeOptions::append"),
                "unexpected {mode} error: {error}"
            );
            assert_eq!(dataset.version().version, before_version, "mode={mode}");
            let after_metadata = dataset.load_indices_by_name("plaid_idx").await.unwrap();
            assert_eq!(after_metadata.len(), 1, "mode={mode}");
            assert_eq!(after_metadata[0].uuid, before_segment.uuid, "mode={mode}");
            assert_eq!(
                after_metadata[0].fragment_bitmap, before_segment.fragment_bitmap,
                "mode={mode}"
            );
            let after_bytes = store.read_one_all(&path).await.unwrap();
            assert_eq!(after_bytes, before_bytes, "mode={mode}");
        }
        assert_eq!(
            dataset
                .unindexed_fragments("plaid_idx")
                .await
                .unwrap()
                .len(),
            1
        );

        // The lower-level maintenance API has the same guard and cannot bypass
        // Dataset::optimize_indices preflight.
        let unindexed = dataset.unindexed_fragments("plaid_idx").await.unwrap();
        let references = vec![&before_segment];
        let error = crate::index::append::merge_indices_with_unindexed_frags(
            Arc::new(dataset.clone()),
            &references,
            &unindexed,
            &OptimizeOptions::merge(1),
        )
        .await
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("supports only append optimization")
        );

        let mut generic = before_segment.clone();
        generic.uuid = Uuid::new_v4();
        generic.index_details = Some(Arc::new(crate::index::vector_index_details_default()));
        let mixed = vec![&before_segment, &generic];
        let error = index_group_is_plaid(&mixed).unwrap_err();
        assert!(error.to_string().contains("mixes 1 PLAID segment"));
        let error = crate::index::append::merge_indices_with_unindexed_frags(
            Arc::new(dataset.clone()),
            &mixed,
            &unindexed,
            &OptimizeOptions::append(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("mixes 1 PLAID segment"));
        assert_eq!(dataset.version().version, before_version);

        // Even when every metadata entry says PLAID and every file has valid
        // PLAID magic, independently trained models must not be combined by an
        // append. Build a valid one-centroid object outside the manifest and
        // verify model compatibility fails before the append writer is opened.
        let incompatible_uuid = Uuid::new_v4();
        let incompatible_params = PlaidIndexParams {
            num_centroids: 1,
            nbits: 2,
            max_iterations: 3,
            sample_rate: 4,
        };
        let incompatible_files = build_plaid_index(
            &dataset,
            "mv",
            incompatible_uuid,
            &incompatible_params,
            Arc::new(lance_index::progress::NoopIndexBuildProgress),
        )
        .await
        .unwrap();
        let mut incompatible = before_segment.clone();
        incompatible.uuid = incompatible_uuid;
        incompatible.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
        incompatible.index_details =
            Some(Arc::new(plaid_index_details(&incompatible_params).unwrap()));
        incompatible.files = Some(incompatible_files);
        let incompatible_references = vec![&before_segment, &incompatible];
        let error = crate::index::append::merge_indices_with_unindexed_frags(
            Arc::new(dataset.clone()),
            &incompatible_references,
            &unindexed,
            &OptimizeOptions::append(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("incompatible trained models"));
        assert_eq!(dataset.version().version, before_version);
        assert_eq!(
            dataset.load_indices_by_name("plaid_idx").await.unwrap()[0].uuid,
            before_segment.uuid
        );

        // A segment whose details claim PLAID but whose object has different
        // magic is rejected during the pre-write validation pass.
        let corrupt_uuid = Uuid::new_v4();
        let corrupt_path = dataset
            .indices_dir()
            .join(corrupt_uuid.to_string())
            .join(INDEX_FILE_NAME);
        let corrupt_bytes = vec![0_u8; 48];
        dataset
            .object_store
            .put(&corrupt_path, &corrupt_bytes)
            .await
            .unwrap();
        let mut corrupt = before_segment.clone();
        corrupt.uuid = corrupt_uuid;
        corrupt.fragment_bitmap = Some(RoaringBitmap::new());
        corrupt.files = Some(vec![IndexFile {
            path: INDEX_FILE_NAME.to_string(),
            size_bytes: corrupt_bytes.len() as u64,
        }]);
        let corrupt_references = vec![&before_segment, &corrupt];
        let error = crate::index::append::merge_indices_with_unindexed_frags(
            Arc::new(dataset.clone()),
            &corrupt_references,
            &unindexed,
            &OptimizeOptions::append(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("invalid magic bytes"));
        assert_eq!(dataset.version().version, before_version);
    }

    #[tokio::test]
    async fn plaid_decode_preflight_runs_before_any_other_index_writer() {
        let directory = TempStrDir::default();
        let first = make_batch(
            vec![0, 1, 2, 3],
            vec![
                vec![[1.0, 0.0, 0.0, 0.0]],
                vec![[0.0, 1.0, 0.0, 0.0]],
                vec![[-1.0, 0.0, 0.0, 0.0]],
                vec![[0.0, -1.0, 0.0, 0.0]],
            ],
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
        let scalar_params = lance_index::scalar::ScalarIndexParams::for_builtin(
            lance_index::scalar::BuiltinIndexType::BTree,
        );
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &scalar_params,
                false,
            )
            .await
            .unwrap();

        let second = make_batch(
            vec![4, 5],
            vec![vec![[0.9, 0.0, 0.0, 0.0]], vec![[0.0, 0.9, 0.0, 0.0]]],
        );
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();

        // Commit a manifest-visible PLAID-labeled segment with invalid bytes.
        // The scalar index also has unindexed data and would open a writer if
        // execution started before every PLAID group was decoded.
        let valid_plaid = dataset.load_indices_by_name("plaid_idx").await.unwrap()[0].clone();
        let corrupt_uuid = Uuid::new_v4();
        let corrupt_bytes = vec![0_u8; 48];
        let corrupt_path = dataset
            .indices_dir()
            .join(corrupt_uuid.to_string())
            .join(INDEX_FILE_NAME);
        dataset
            .object_store
            .put(&corrupt_path, &corrupt_bytes)
            .await
            .unwrap();
        let mut corrupt = valid_plaid;
        corrupt.uuid = corrupt_uuid;
        corrupt.fragment_bitmap = Some(RoaringBitmap::new());
        corrupt.files = Some(vec![IndexFile {
            path: INDEX_FILE_NAME.to_string(),
            size_bytes: corrupt_bytes.len() as u64,
        }]);
        let transaction = crate::dataset::transaction::Transaction::new(
            dataset.manifest.version,
            crate::dataset::transaction::Operation::CreateIndex {
                new_indices: vec![corrupt],
                removed_indices: Vec::new(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let before_version = dataset.version().version;
        let before_dirs = dataset
            .object_store
            .read_dir(dataset.indices_dir())
            .await
            .unwrap()
            .into_iter()
            .collect::<std::collections::HashSet<_>>();
        let before_scalar = dataset.load_indices_by_name("id_idx").await.unwrap();
        let error = dataset
            .optimize_indices(&OptimizeOptions::append())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("invalid magic bytes"));
        assert_eq!(dataset.version().version, before_version);
        assert_eq!(
            dataset
                .object_store
                .read_dir(dataset.indices_dir())
                .await
                .unwrap()
                .into_iter()
                .collect::<std::collections::HashSet<_>>(),
            before_dirs
        );
        assert_eq!(
            dataset.load_indices_by_name("id_idx").await.unwrap(),
            before_scalar
        );
        assert_eq!(
            dataset.unindexed_fragments("id_idx").await.unwrap().len(),
            1
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

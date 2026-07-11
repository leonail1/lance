// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Single-operator database-native PLAID multi-vector search.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::AsArray;
use arrow::datatypes::Float32Type;
use arrow_array::{Array, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion::common::stats::Precision;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, Count, ExecutionPlanMetricsSet, MetricsSet, Time,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::stream;
use lance_arrow::RecordBatchExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_datafusion::utils::ExecutionPlanMetricsSetExt;
use lance_index::prefilter::PreFilter;
use lance_index::vector::Query;
use lance_table::format::IndexMetadata;
use ndarray::{ArrayView1, ArrayView2};

use super::knn::KNN_INDEX_SCHEMA;
use super::utils::{IndexMetrics, PreFilterSource, build_prefilter};
use crate::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use crate::index::DatasetIndexInternalExt;
use crate::index::plaid::{
    PLAID_DEFAULT_DECOMPRESS_DOCUMENTS, PlaidVectorIndex, is_plaid_index_metadata,
    maxsim_distance, query_to_array,
};
use crate::{Error, Result};

const FILTER_TIME: &str = "plaid_filter_materialization_time";
const CENTROID_TIME: &str = "plaid_centroid_probe_time";
const POSTINGS_TIME: &str = "plaid_postings_time";
const CANDIDATE_TIME: &str = "plaid_candidate_total_time";
const APPROXIMATE_TIME: &str = "plaid_approximate_time";
const RESIDUAL_RERANK_TIME: &str = "plaid_residual_rerank_time";
const RAW_FETCH_TIME: &str = "plaid_raw_fetch_time";
const EXACT_TIME: &str = "plaid_exact_maxsim_time";
const SORT_TIME: &str = "plaid_sort_time";
const TOTAL_TIME: &str = "plaid_total_time";
const POSTINGS_COUNT: &str = "plaid_posting_entries";
const CANDIDATE_COUNT: &str = "plaid_candidate_documents";
const RAW_ROWS_COUNT: &str = "plaid_raw_rows";
const PROBE_RETRY_COUNT: &str = "plaid_probe_retries";
const CONFIGURED_PROBES_COUNT: &str = "plaid_configured_probes";
const N_FULL_BUDGET_COUNT: &str = "plaid_n_full_budget";
const RAW_BUDGET_COUNT: &str = "plaid_raw_candidate_budget";

/// One physical operator for the complete database-native PLAID query pipeline.
#[derive(Debug)]
pub struct PlaidSearchExec {
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    query: Query,
    prefilter_source: PreFilterSource,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl PlaidSearchExec {
    pub fn try_new(
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        query: Query,
        prefilter_source: PreFilterSource,
    ) -> Result<Self> {
        if indices.is_empty() {
            return Err(Error::invalid_input(
                "PlaidSearchExec requires at least one index segment".to_string(),
            ));
        }
        if indices.iter().any(|index| !is_plaid_index_metadata(index)) {
            return Err(Error::invalid_input(
                "PlaidSearchExec received a non-PLAID index segment".to_string(),
            ));
        }
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            dataset,
            indices,
            query,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for PlaidSearchExec {
    fn fmt_as(
        &self,
        format: DisplayFormatType,
        formatter: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                formatter,
                "PlaidSearch: name={}, k={}, segments={}, exact_refinement=true",
                self.indices[0].name,
                self.query.k,
                self.indices.len()
            ),
            DisplayFormatType::TreeRender => write!(
                formatter,
                "PlaidSearch\nname={}\nk={}\nsegments={}\nexact_refinement=true",
                self.indices[0].name,
                self.query.k,
                self.indices.len()
            ),
        }
    }
}

impl ExecutionPlan for PlaidSearchExec {
    fn name(&self) -> &str {
        "PlaidSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        KNN_INDEX_SCHEMA.clone()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        match &self.prefilter_source {
            PreFilterSource::None => Vec::new(),
            PreFilterSource::FilteredRowIds(child) | PreFilterSource::ScalarIndexQuery(child) => {
                vec![child]
            }
        }
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match (&self.prefilter_source, children.len()) {
            (PreFilterSource::None, 0) => PreFilterSource::None,
            (PreFilterSource::FilteredRowIds(_), 1) => {
                PreFilterSource::FilteredRowIds(children.pop().expect("length checked"))
            }
            (PreFilterSource::ScalarIndexQuery(_), 1) => {
                PreFilterSource::ScalarIndexQuery(children.pop().expect("length checked"))
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "PlaidSearchExec child count does not match its prefilter source".to_string(),
                ));
            }
        };
        Ok(Arc::new(Self::try_new(
            self.dataset.clone(),
            self.indices.clone(),
            self.query.clone(),
            prefilter_source,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let prefilter = build_prefilter(
            context,
            partition,
            &self.prefilter_source,
            self.dataset.clone(),
            &self.indices,
        )?;
        let metrics = Arc::new(PlaidExecMetrics::new(&self.metrics, partition));
        let dataset = self.dataset.clone();
        let indices = self.indices.clone();
        let query = self.query.clone();
        let stream = stream::once(async move {
            let total_started = Instant::now();
            let result = execute_search(dataset, indices, query, prefilter, metrics.clone())
                .await
                .map_err(DataFusionError::from);
            metrics.total.add_duration(total_started.elapsed());
            metrics.index.flush_io();
            metrics.baseline.done();
            result
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Statistics> {
        Ok(Statistics {
            num_rows: Precision::Inexact(self.query.k),
            ..Statistics::new_unknown(self.schema().as_ref())
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

struct PlaidExecMetrics {
    baseline: BaselineMetrics,
    index: IndexMetrics,
    filter: Time,
    centroid: Time,
    postings: Time,
    candidate: Time,
    approximate: Time,
    residual_rerank: Time,
    raw_fetch: Time,
    exact: Time,
    sort: Time,
    total: Time,
    postings_count: Count,
    candidate_count: Count,
    raw_rows_count: Count,
    probe_retry_count: Count,
    configured_probes_count: Count,
    n_full_budget_count: Count,
    raw_budget_count: Count,
}

impl PlaidExecMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            index: IndexMetrics::new(metrics, partition),
            filter: metrics.new_time(FILTER_TIME, partition),
            centroid: metrics.new_time(CENTROID_TIME, partition),
            postings: metrics.new_time(POSTINGS_TIME, partition),
            candidate: metrics.new_time(CANDIDATE_TIME, partition),
            approximate: metrics.new_time(APPROXIMATE_TIME, partition),
            residual_rerank: metrics.new_time(RESIDUAL_RERANK_TIME, partition),
            raw_fetch: metrics.new_time(RAW_FETCH_TIME, partition),
            exact: metrics.new_time(EXACT_TIME, partition),
            sort: metrics.new_time(SORT_TIME, partition),
            total: metrics.new_time(TOTAL_TIME, partition),
            postings_count: metrics.new_count(POSTINGS_COUNT, partition),
            candidate_count: metrics.new_count(CANDIDATE_COUNT, partition),
            raw_rows_count: metrics.new_count(RAW_ROWS_COUNT, partition),
            probe_retry_count: metrics.new_count(PROBE_RETRY_COUNT, partition),
            configured_probes_count: metrics.new_count(CONFIGURED_PROBES_COUNT, partition),
            n_full_budget_count: metrics.new_count(N_FULL_BUDGET_COUNT, partition),
            raw_budget_count: metrics.new_count(RAW_BUDGET_COUNT, partition),
        }
    }

    fn record_core(&self, stats: &lance_plaid::PlaidSearchStats) {
        self.centroid
            .add_duration(Duration::from_nanos(stats.centroid_probe_nanos));
        self.postings
            .add_duration(Duration::from_nanos(stats.postings_nanos));
        self.approximate
            .add_duration(Duration::from_nanos(stats.approximate_score_nanos));
        self.residual_rerank
            .add_duration(Duration::from_nanos(stats.exact_score_nanos));
        self.sort
            .add_duration(Duration::from_nanos(stats.sort_nanos));
        self.postings_count
            .add(usize::try_from(stats.posting_entries_read).unwrap_or(usize::MAX));
        self.candidate_count
            .add(usize::try_from(stats.candidate_documents).unwrap_or(usize::MAX));
    }
}

async fn execute_search(
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    query: Query,
    prefilter: Arc<crate::index::prefilter::DatasetPreFilter>,
    metrics: Arc<PlaidExecMetrics>,
) -> Result<RecordBatch> {
    let filter_started = Instant::now();
    prefilter.wait_for_ready().await?;
    metrics.filter.add_duration(filter_started.elapsed());

    let dimension = crate::index::vector::utils::get_vector_dim(dataset.schema(), &query.column)?;
    let query_tokens = query_to_array(&query, dimension)?;
    let default_refine = 4_usize;
    let refine = query
        .refine_factor
        .map(|factor| factor as usize)
        .unwrap_or(default_refine)
        .max(1);
    let requested_candidates = query.k.saturating_mul(refine).max(query.k);
    let candidate_limit = requested_candidates.max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS);
    let mask = prefilter.mask();
    let candidate_started = Instant::now();
    let mut candidates = HashMap::<u64, f32>::new();

    for metadata in &indices {
        let raw_index = dataset
            .open_vector_index(&query.column, &metadata.uuid, &metrics.index)
            .await?;
        let plaid = raw_index
            .as_any()
            .downcast_ref::<PlaidVectorIndex>()
            .ok_or_else(|| {
                Error::internal("persisted PLAID segment opened as another index type".to_string())
            })?;
        let (mut params, segment_eligible) =
            plaid.candidate_params(&query, requested_candidates, mask.as_ref());
        metrics.n_full_budget_count.add(params.n_full_scores);
        metrics.raw_budget_count.add(params.top_k);
        let desired_candidates = segment_eligible
            .min(params.top_k)
            .min(plaid.num_documents());
        let max_centroids = query
            .maximum_nprobes
            .unwrap_or(plaid.num_centroids())
            .min(plaid.num_centroids())
            .max(1);
        let hits = loop {
            metrics.configured_probes_count.add(params.n_ivf_probe);
            let query_for_cpu = query_tokens.clone();
            let mask_for_cpu = mask.clone();
            let index_for_cpu = raw_index.clone();
            let params_for_cpu = params.clone();
            let (hits, stats) = spawn_cpu(move || {
                let index = index_for_cpu
                    .as_any()
                    .downcast_ref::<PlaidVectorIndex>()
                    .ok_or_else(|| {
                        Error::internal("PLAID index downcast failed on CPU worker".to_string())
                    })?;
                index.search_candidates(
                    query_for_cpu.view(),
                    &params_for_cpu,
                    mask_for_cpu.as_ref(),
                )
            })
            .await?;
            metrics.record_core(&stats);
            if hits.len() >= desired_candidates || params.n_ivf_probe >= max_centroids {
                break hits;
            }
            params.n_ivf_probe = params
                .n_ivf_probe
                .saturating_mul(2)
                .min(max_centroids)
                .max(1);
            params.centroid_score_threshold = None;
            metrics.probe_retry_count.add(1);
        };
        for hit in hits {
            candidates
                .entry(hit.row_address)
                .and_modify(|score| {
                    if hit.score.total_cmp(score).is_gt() {
                        *score = hit.score;
                    }
                })
                .or_insert(hit.score);
        }
    }
    metrics.candidate.add_duration(candidate_started.elapsed());

    let sort_started = Instant::now();
    let mut candidates = candidates.into_iter().collect::<Vec<_>>();
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    candidates.truncate(candidate_limit.min(candidates.len()));
    metrics.sort.add_duration(sort_started.elapsed());

    if candidates.is_empty() {
        let batch = RecordBatch::new_empty(KNN_INDEX_SCHEMA.clone());
        metrics.baseline.record_output(0);
        return Ok(batch);
    }

    let row_addresses = candidates
        .iter()
        .map(|(row_address, _)| *row_address)
        .collect::<Vec<_>>();
    let projection = Arc::new(
        ProjectionRequest::Schema(Arc::new(
            dataset.schema().project(&[query.column.as_str()])?,
        ))
        .into_projection_plan(dataset.clone())?,
    );
    let raw_fetch_started = Instant::now();
    let batch = TakeBuilder::try_new_from_addresses(
        dataset.clone(),
        row_addresses.clone(),
        projection,
    )?
    .execute()
    .await?;
    metrics.raw_fetch.add_duration(raw_fetch_started.elapsed());
    metrics.raw_rows_count.add(batch.num_rows());
    if batch.num_rows() != row_addresses.len() {
        return Err(Error::internal(format!(
            "PLAID raw take returned {} rows for {} candidates",
            batch.num_rows(),
            row_addresses.len()
        )));
    }

    let exact_started = Instant::now();
    let column = query.column.clone();
    let query_for_cpu = query_tokens.clone();
    let mut exact = spawn_cpu(move || exact_scores(&batch, &column, query_for_cpu.view())).await?;
    metrics.exact.add_duration(exact_started.elapsed());
    for (hit, row_address) in exact.iter_mut().zip(row_addresses) {
        hit.row_address = row_address;
    }

    let sort_started = Instant::now();
    exact.retain(|hit| {
        query.lower_bound.is_none_or(|lower| hit.distance >= lower)
            && query.upper_bound.is_none_or(|upper| hit.distance < upper)
    });
    exact.sort_unstable_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.row_address.cmp(&right.row_address))
    });
    exact.truncate(query.k.min(exact.len()));
    metrics.sort.add_duration(sort_started.elapsed());

    let distances = Float32Array::from(exact.iter().map(|hit| hit.distance).collect::<Vec<_>>());
    let rows = UInt64Array::from(exact.iter().map(|hit| hit.row_address).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(
        KNN_INDEX_SCHEMA.clone(),
        vec![Arc::new(distances), Arc::new(rows)],
    )?;
    metrics.baseline.record_output(batch.num_rows());
    Ok(batch)
}

struct ExactHit {
    row_address: u64,
    distance: f32,
}

fn exact_scores(
    batch: &RecordBatch,
    column: &str,
    query: ArrayView2<'_, f32>,
) -> Result<Vec<ExactHit>> {
    let documents = batch
        .column_by_qualified_name(column)
        .ok_or_else(|| Error::internal(format!("PLAID raw batch is missing {column}")))?
        .as_list::<i32>();
    let mut hits = Vec::with_capacity(documents.len());
    for row_index in 0..documents.len() {
        if documents.is_null(row_index) {
            return Err(Error::internal(
                "PLAID candidate unexpectedly resolved to a null document".to_string(),
            ));
        }
        let document = documents.value(row_index);
        let tokens = document.as_fixed_size_list();
        let score = exact_maxsim(query, tokens)?;
        hits.push(ExactHit {
            row_address: 0,
            distance: maxsim_distance(score),
        });
    }
    Ok(hits)
}

fn exact_maxsim(
    query: ArrayView2<'_, f32>,
    document: &arrow_array::FixedSizeListArray,
) -> Result<f32> {
    if document.is_empty() {
        return Ok(f32::NAN);
    }
    let mut score = 0.0_f32;
    for query_token in query.outer_iter() {
        let mut maximum = f32::NEG_INFINITY;
        for token_index in 0..document.len() {
            if document.is_null(token_index) {
                return Err(Error::invalid_input(
                    "PLAID exact refinement does not support null token vectors".to_string(),
                ));
            }
            let token = document.value(token_index);
            let token = token.as_primitive::<Float32Type>();
            let similarity = raw_dot(query_token, token.values())?;
            maximum = maximum.max(similarity);
        }
        score += maximum;
    }
    Ok(score)
}

fn raw_dot(query: ArrayView1<'_, f32>, document: &[f32]) -> Result<f32> {
    if query.len() != document.len()
        || query.iter().any(|value| !value.is_finite())
        || document.iter().any(|value| !value.is_finite())
    {
        return Err(Error::invalid_input(format!(
            "PLAID exact token dimension {} does not match query dimension {} or contains non-finite values",
            document.len(),
            query.len()
        )));
    }
    Ok(query
        .iter()
        .zip(document)
        .map(|(left, right)| left * right)
        .sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, ListArray};
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_linalg::distance::{DistanceType, multivec_distance};
    use ndarray::array;

    #[test]
    fn exact_scores_equal_multivec_dot_distance() {
        let values = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.5, 0.0, 0.0, 0.0, 0.6, 0.8, 0.0, 0.0]),
            4,
        )
        .unwrap();
        let field = Arc::new(Field::new("item", values.data_type().clone(), false));
        let documents = ListArray::try_new(
            field,
            OffsetBuffer::from_lengths([1_usize, 1]),
            Arc::new(values),
            None,
        )
        .unwrap();
        let batch = RecordBatch::try_from_iter([(
            "mv",
            Arc::new(documents.clone()) as ArrayRef,
        )])
        .unwrap();
        let query = array![[1.0_f32, 0.0, 0.0, 0.0]];
        let actual = exact_scores(&batch, "mv", query.view()).unwrap();
        let query_values = Float32Array::from(query.iter().copied().collect::<Vec<_>>());
        let expected =
            multivec_distance(&query_values, &documents, DistanceType::Dot).unwrap();
        let actual = actual.iter().map(|hit| hit.distance).collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn exact_maxsim_uses_raw_pooled_token_dot_product_and_ranking() {
        let low_norm_exact_direction = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.5, 0.0, 0.0, 0.0]),
            4,
        )
        .unwrap();
        let higher_raw_dot = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.6, 0.8, 0.0, 0.0]),
            4,
        )
        .unwrap();
        let query = array![[1.0_f32, 0.0, 0.0, 0.0]];
        let low_norm_score = exact_maxsim(query.view(), &low_norm_exact_direction).unwrap();
        let higher_raw_score = exact_maxsim(query.view(), &higher_raw_dot).unwrap();
        assert!((low_norm_score - 0.5).abs() < 1.0e-6);
        assert!((higher_raw_score - 0.6).abs() < 1.0e-6);
        assert!(
            higher_raw_score > low_norm_score,
            "raw dot ranks the 0.6 token first; cosine would rank the 0.5 collinear token first"
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Single-operator database-native PLAID multi-vector search.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::AsArray;
use arrow::datatypes::{Float32Type, UInt64Type};
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
use lance_core::ROW_ID;
use lance_core::utils::tokio::spawn_cpu;
use lance_datafusion::utils::ExecutionPlanMetricsSetExt;
use lance_index::prefilter::PreFilter;
use lance_index::vector::Query;
use lance_plaid::EligibleCentroidDecision;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::IndexMetadata;
use ndarray::{ArrayView1, ArrayView2};

use super::knn::KNN_INDEX_SCHEMA;
use super::utils::{IndexMetrics, PreFilterSource, build_prefilter};
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use crate::index::DatasetIndexInternalExt;
use crate::index::plaid::{
    PLAID_DEFAULT_DECOMPRESS_DOCUMENTS, PlaidCandidatePlan, PlaidVectorIndex,
    is_plaid_index_metadata, maxsim_distance, query_to_array,
};
use crate::{Error, Result};

const FILTER_TIME: &str = "plaid_filter_materialization_time";
const ELIGIBLE_CENTROID_PLAN_TOTAL_TIME: &str = "plaid_eligible_centroid_plan_total_time";
const ELIGIBLE_CENTROID_CORE_BUILD_SUB_TIME: &str = "plaid_eligible_centroid_core_build_sub_time";
const ELIGIBLE_CENTROID_SELECTION_SUB_TIME: &str = "plaid_eligible_centroid_selection_sub_time";
const CENTROID_TIME: &str = "plaid_centroid_probe_time";
const POSTINGS_TIME: &str = "plaid_postings_time";
const CANDIDATE_TIME: &str = "plaid_candidate_total_time";
const APPROXIMATE_TIME: &str = "plaid_approximate_time";
const RESIDUAL_RERANK_TIME: &str = "plaid_residual_rerank_time";
const ROW_ID_FETCH_TIME: &str = "plaid_row_id_only_fetch_time";
const RAW_VECTOR_FETCH_TIME: &str = "plaid_raw_vector_fetch_time";
const EXACT_TIME: &str = "plaid_exact_maxsim_time";
const SORT_TIME: &str = "plaid_sort_time";
const TOTAL_TIME: &str = "plaid_total_time";
const POSTINGS_COUNT: &str = "plaid_posting_entries";
const CENTROIDS_PROBED_COUNT: &str = "plaid_centroids_probed";
const POSTINGS_ELIGIBLE_COUNT: &str = "plaid_posting_entries_eligible";
const CANDIDATE_COUNT: &str = "plaid_candidate_documents";
const APPROXIMATE_DOCUMENTS_COUNT: &str = "plaid_approximate_documents";
const RESIDUAL_DOCUMENTS_COUNT: &str = "plaid_residual_documents";
const FILTER_EXACT_SMALL_COUNT: &str = "plaid_filter_exact_small_filter_fallbacks";
const FILTER_EXACT_UNDERFILLED_COUNT: &str = "plaid_filter_exact_underfilled_fallbacks";
const FILTER_EXACT_DOCUMENTS_COUNT: &str = "plaid_filter_exact_documents";
const ROW_ID_ROWS_COUNT: &str = "plaid_row_id_only_rows";
const RAW_VECTOR_ROWS_COUNT: &str = "plaid_raw_vector_rows";
const INDEX_ONLY_QUERY_COUNT: &str = "plaid_index_only_queries";
const EXACT_QUERY_COUNT: &str = "plaid_exact_refinement_queries";
const EMPTY_FILTER_QUERY_COUNT: &str = "plaid_empty_filter_queries";
const SEGMENTS_SEARCHED_COUNT: &str = "plaid_segments_searched";
const PROBE_RETRY_COUNT: &str = "plaid_probe_retries";
const PROBE_ROUND_COUNT: &str = "plaid_probe_rounds";
const CONFIGURED_PROBES_COUNT: &str = "plaid_configured_probes";
const FINAL_PROBES_COUNT: &str = "plaid_final_probes";
const INCREMENTAL_CENTROIDS_REUSED_COUNT: &str = "plaid_incremental_centroids_reused";
const ELIGIBLE_CENTROID_ENABLED_COUNT: &str = "plaid_eligible_centroid_enabled_segments";
const ELIGIBLE_CENTROID_EMPTY_COUNT: &str = "plaid_eligible_centroid_empty_segments";
const ELIGIBLE_CENTROID_UNFILTERED_COUNT: &str =
    "plaid_eligible_centroid_skipped_unfiltered_segments";
const ELIGIBLE_CENTROID_NON_ENUMERABLE_COUNT: &str =
    "plaid_eligible_centroid_skipped_non_enumerable_segments";
const ELIGIBLE_CENTROID_WIDE_COUNT: &str = "plaid_eligible_centroid_skipped_wide_segments";
const ELIGIBLE_CENTROID_COST_COUNT: &str = "plaid_eligible_centroid_skipped_cost_segments";
const ELIGIBLE_DOCUMENTS_COUNT: &str = "plaid_eligible_documents";
const ELIGIBLE_TOKENS_COUNT: &str = "plaid_eligible_tokens";
const ELIGIBLE_TOKEN_CODES_SCANNED_COUNT: &str = "plaid_eligible_token_codes_scanned";
const ELIGIBLE_CENTROIDS_COUNT: &str = "plaid_eligible_centroids";
const ESTIMATED_GLOBAL_POSTINGS_COUNT: &str = "plaid_estimated_global_postings";
const CORE_APPROXIMATE_BUDGET_COUNT: &str = "plaid_core_approximate_budget";
const CORE_RESIDUAL_BUDGET_COUNT: &str = "plaid_core_residual_budget";
const RAW_REFINEMENT_BUDGET_COUNT: &str = "plaid_raw_refinement_budget";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaidExecutionMode {
    IndexOnly,
    Exact { raw_refinement_budget: usize },
}

impl PlaidExecutionMode {
    fn try_from_query(query: &Query) -> Result<Self> {
        match query.refine_factor {
            None => Ok(Self::IndexOnly),
            Some(0) => Err(Error::invalid_input(
                "PLAID refine factor must be positive".to_string(),
            )),
            Some(factor) => {
                let factor = usize::try_from(factor).map_err(|_| {
                    Error::invalid_input("PLAID refine factor does not fit usize".to_string())
                })?;
                let raw_refinement_budget = query.k.checked_mul(factor).ok_or_else(|| {
                    Error::invalid_input("PLAID raw refinement budget overflow".to_string())
                })?;
                Ok(Self::Exact {
                    raw_refinement_budget,
                })
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::IndexOnly => "index_only",
            Self::Exact { .. } => "exact",
        }
    }

    fn raw_refinement_budget(self) -> usize {
        match self {
            Self::IndexOnly => 0,
            Self::Exact {
                raw_refinement_budget,
            } => raw_refinement_budget,
        }
    }

    fn requested_candidates(self, top_k: usize) -> usize {
        self.raw_refinement_budget().max(top_k)
    }
}

/// One physical operator for the complete database-native PLAID query pipeline.
#[derive(Debug)]
pub struct PlaidSearchExec {
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    query: Query,
    mode: PlaidExecutionMode,
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
        let mode = PlaidExecutionMode::try_from_query(&query)?;
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
            mode,
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
                "PlaidSearch: name={}, k={}, segments={}, mode={}, core_residual_budget={}, raw_refinement_budget={}, filter_exact_fallback=enabled",
                self.indices[0].name,
                self.query.k,
                self.indices.len(),
                self.mode.name(),
                self.mode
                    .requested_candidates(self.query.k)
                    .max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS),
                self.mode.raw_refinement_budget(),
            ),
            DisplayFormatType::TreeRender => write!(
                formatter,
                "PlaidSearch\nname={}\nk={}\nsegments={}\nmode={}\ncore_residual_budget={}\nraw_refinement_budget={}\nfilter_exact_fallback=enabled",
                self.indices[0].name,
                self.query.k,
                self.indices.len(),
                self.mode.name(),
                self.mode
                    .requested_candidates(self.query.k)
                    .max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS),
                self.mode.raw_refinement_budget(),
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
        let mode = self.mode;
        let stream = stream::once(async move {
            let total_started = Instant::now();
            let result = execute_search(dataset, indices, query, mode, prefilter, metrics.clone())
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
    eligible_centroid_plan_total: Time,
    eligible_centroid_core_build_sub: Time,
    eligible_centroid_selection_sub: Time,
    centroid: Time,
    postings: Time,
    candidate: Time,
    approximate: Time,
    residual_rerank: Time,
    row_id_fetch: Time,
    raw_vector_fetch: Time,
    exact: Time,
    sort: Time,
    total: Time,
    postings_count: Count,
    centroids_probed_count: Count,
    postings_eligible_count: Count,
    candidate_count: Count,
    approximate_documents_count: Count,
    residual_documents_count: Count,
    filter_exact_small_count: Count,
    filter_exact_underfilled_count: Count,
    filter_exact_documents_count: Count,
    row_id_rows_count: Count,
    raw_vector_rows_count: Count,
    index_only_query_count: Count,
    exact_query_count: Count,
    empty_filter_query_count: Count,
    segments_searched_count: Count,
    probe_retry_count: Count,
    probe_round_count: Count,
    configured_probes_count: Count,
    final_probes_count: Count,
    incremental_centroids_reused_count: Count,
    eligible_centroid_enabled_count: Count,
    eligible_centroid_empty_count: Count,
    eligible_centroid_unfiltered_count: Count,
    eligible_centroid_non_enumerable_count: Count,
    eligible_centroid_wide_count: Count,
    eligible_centroid_cost_count: Count,
    eligible_documents_count: Count,
    eligible_tokens_count: Count,
    eligible_token_codes_scanned_count: Count,
    eligible_centroids_count: Count,
    estimated_global_postings_count: Count,
    core_approximate_budget_count: Count,
    core_residual_budget_count: Count,
    raw_refinement_budget_count: Count,
}

impl PlaidExecMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            index: IndexMetrics::new(metrics, partition),
            filter: metrics.new_time(FILTER_TIME, partition),
            eligible_centroid_plan_total: metrics
                .new_time(ELIGIBLE_CENTROID_PLAN_TOTAL_TIME, partition),
            eligible_centroid_core_build_sub: metrics
                .new_time(ELIGIBLE_CENTROID_CORE_BUILD_SUB_TIME, partition),
            eligible_centroid_selection_sub: metrics
                .new_time(ELIGIBLE_CENTROID_SELECTION_SUB_TIME, partition),
            centroid: metrics.new_time(CENTROID_TIME, partition),
            postings: metrics.new_time(POSTINGS_TIME, partition),
            candidate: metrics.new_time(CANDIDATE_TIME, partition),
            approximate: metrics.new_time(APPROXIMATE_TIME, partition),
            residual_rerank: metrics.new_time(RESIDUAL_RERANK_TIME, partition),
            row_id_fetch: metrics.new_time(ROW_ID_FETCH_TIME, partition),
            raw_vector_fetch: metrics.new_time(RAW_VECTOR_FETCH_TIME, partition),
            exact: metrics.new_time(EXACT_TIME, partition),
            sort: metrics.new_time(SORT_TIME, partition),
            total: metrics.new_time(TOTAL_TIME, partition),
            postings_count: metrics.new_count(POSTINGS_COUNT, partition),
            centroids_probed_count: metrics.new_count(CENTROIDS_PROBED_COUNT, partition),
            postings_eligible_count: metrics.new_count(POSTINGS_ELIGIBLE_COUNT, partition),
            candidate_count: metrics.new_count(CANDIDATE_COUNT, partition),
            approximate_documents_count: metrics.new_count(APPROXIMATE_DOCUMENTS_COUNT, partition),
            residual_documents_count: metrics.new_count(RESIDUAL_DOCUMENTS_COUNT, partition),
            filter_exact_small_count: metrics.new_count(FILTER_EXACT_SMALL_COUNT, partition),
            filter_exact_underfilled_count: metrics
                .new_count(FILTER_EXACT_UNDERFILLED_COUNT, partition),
            filter_exact_documents_count: metrics
                .new_count(FILTER_EXACT_DOCUMENTS_COUNT, partition),
            row_id_rows_count: metrics.new_count(ROW_ID_ROWS_COUNT, partition),
            raw_vector_rows_count: metrics.new_count(RAW_VECTOR_ROWS_COUNT, partition),
            index_only_query_count: metrics.new_count(INDEX_ONLY_QUERY_COUNT, partition),
            exact_query_count: metrics.new_count(EXACT_QUERY_COUNT, partition),
            empty_filter_query_count: metrics.new_count(EMPTY_FILTER_QUERY_COUNT, partition),
            segments_searched_count: metrics.new_count(SEGMENTS_SEARCHED_COUNT, partition),
            probe_retry_count: metrics.new_count(PROBE_RETRY_COUNT, partition),
            probe_round_count: metrics.new_count(PROBE_ROUND_COUNT, partition),
            configured_probes_count: metrics.new_count(CONFIGURED_PROBES_COUNT, partition),
            final_probes_count: metrics.new_count(FINAL_PROBES_COUNT, partition),
            incremental_centroids_reused_count: metrics
                .new_count(INCREMENTAL_CENTROIDS_REUSED_COUNT, partition),
            eligible_centroid_enabled_count: metrics
                .new_count(ELIGIBLE_CENTROID_ENABLED_COUNT, partition),
            eligible_centroid_empty_count: metrics
                .new_count(ELIGIBLE_CENTROID_EMPTY_COUNT, partition),
            eligible_centroid_unfiltered_count: metrics
                .new_count(ELIGIBLE_CENTROID_UNFILTERED_COUNT, partition),
            eligible_centroid_non_enumerable_count: metrics
                .new_count(ELIGIBLE_CENTROID_NON_ENUMERABLE_COUNT, partition),
            eligible_centroid_wide_count: metrics
                .new_count(ELIGIBLE_CENTROID_WIDE_COUNT, partition),
            eligible_centroid_cost_count: metrics
                .new_count(ELIGIBLE_CENTROID_COST_COUNT, partition),
            eligible_documents_count: metrics.new_count(ELIGIBLE_DOCUMENTS_COUNT, partition),
            eligible_tokens_count: metrics.new_count(ELIGIBLE_TOKENS_COUNT, partition),
            eligible_token_codes_scanned_count: metrics
                .new_count(ELIGIBLE_TOKEN_CODES_SCANNED_COUNT, partition),
            eligible_centroids_count: metrics.new_count(ELIGIBLE_CENTROIDS_COUNT, partition),
            estimated_global_postings_count: metrics
                .new_count(ESTIMATED_GLOBAL_POSTINGS_COUNT, partition),
            core_approximate_budget_count: metrics
                .new_count(CORE_APPROXIMATE_BUDGET_COUNT, partition),
            core_residual_budget_count: metrics.new_count(CORE_RESIDUAL_BUDGET_COUNT, partition),
            raw_refinement_budget_count: metrics.new_count(RAW_REFINEMENT_BUDGET_COUNT, partition),
        }
    }

    fn record_mode(&self, mode: PlaidExecutionMode) {
        match mode {
            PlaidExecutionMode::IndexOnly => self.index_only_query_count.add(1),
            PlaidExecutionMode::Exact { .. } => self.exact_query_count.add(1),
        }
        self.raw_refinement_budget_count
            .add(mode.raw_refinement_budget());
    }

    fn record_eligible_centroid_plan(&self, candidate_plan: &PlaidCandidatePlan) {
        let plan = &candidate_plan.eligible_centroids;
        self.eligible_centroid_plan_total
            .add_duration(Duration::from_nanos(candidate_plan.total_plan_nanos));
        self.eligible_centroid_core_build_sub
            .add_duration(Duration::from_nanos(plan.core_build_nanos()));
        self.eligible_documents_count
            .add(usize::try_from(plan.eligible_documents()).unwrap_or(usize::MAX));
        self.eligible_tokens_count
            .add(usize::try_from(plan.eligible_tokens()).unwrap_or(usize::MAX));
        self.eligible_token_codes_scanned_count
            .add(usize::try_from(plan.eligible_token_codes_scanned()).unwrap_or(usize::MAX));
        self.eligible_centroids_count
            .add(usize::try_from(plan.eligible_centroids()).unwrap_or(usize::MAX));
        self.estimated_global_postings_count
            .add(usize::try_from(plan.estimated_global_postings()).unwrap_or(usize::MAX));
        match plan.decision() {
            EligibleCentroidDecision::Enabled => self.eligible_centroid_enabled_count.add(1),
            EligibleCentroidDecision::Empty => self.eligible_centroid_empty_count.add(1),
            EligibleCentroidDecision::Unfiltered => self.eligible_centroid_unfiltered_count.add(1),
            EligibleCentroidDecision::NonEnumerable => {
                self.eligible_centroid_non_enumerable_count.add(1)
            }
            EligibleCentroidDecision::TooWide => self.eligible_centroid_wide_count.add(1),
            EligibleCentroidDecision::ScanCostTooHigh => self.eligible_centroid_cost_count.add(1),
        }
    }

    fn record_core(&self, stats: &lance_plaid::PlaidSearchStats) {
        self.centroid
            .add_duration(Duration::from_nanos(stats.centroid_probe_nanos));
        self.eligible_centroid_selection_sub
            .add_duration(Duration::from_nanos(
                stats.eligible_centroid_selection_nanos,
            ));
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
        self.centroids_probed_count
            .add(usize::try_from(stats.centroids_probed).unwrap_or(usize::MAX));
        self.postings_eligible_count
            .add(usize::try_from(stats.posting_entries_eligible).unwrap_or(usize::MAX));
        self.candidate_count
            .add(usize::try_from(stats.candidate_documents).unwrap_or(usize::MAX));
        self.approximate_documents_count
            .add(usize::try_from(stats.approximate_documents).unwrap_or(usize::MAX));
        self.residual_documents_count
            .add(usize::try_from(stats.exact_documents).unwrap_or(usize::MAX));
        self.probe_retry_count
            .add(usize::try_from(stats.probe_retries).unwrap_or(usize::MAX));
        self.probe_round_count
            .add(usize::try_from(stats.probe_rounds).unwrap_or(usize::MAX));
        self.configured_probes_count
            .add(usize::try_from(stats.configured_probes).unwrap_or(usize::MAX));
        self.final_probes_count
            .add(usize::try_from(stats.final_nprobe).unwrap_or(usize::MAX));
        self.incremental_centroids_reused_count
            .add(usize::try_from(stats.incremental_centroids_reused).unwrap_or(usize::MAX));
    }
}

fn small_filter_exact_fallback(filter_max_len: Option<u64>, requested_candidates: usize) -> bool {
    filter_max_len.is_some_and(|count| {
        count > 0 && count <= u64::try_from(requested_candidates).unwrap_or(u64::MAX)
    })
}

fn underfilled_filter_exact_fallback(
    filtered_query: bool,
    ann_candidates: usize,
    top_k: usize,
    eligible_documents: usize,
) -> bool {
    filtered_query && ann_candidates < top_k.min(eligible_documents)
}

async fn execute_search(
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    query: Query,
    mode: PlaidExecutionMode,
    prefilter: Arc<crate::index::prefilter::DatasetPreFilter>,
    metrics: Arc<PlaidExecMetrics>,
) -> Result<RecordBatch> {
    metrics.record_mode(mode);
    let filter_started = Instant::now();
    prefilter.wait_for_ready().await?;
    metrics.filter.add_duration(filter_started.elapsed());

    let dimension = crate::index::vector::utils::get_vector_dim(dataset.schema(), &query.column)?;
    let query_tokens = query_to_array(&query, dimension)?;
    let requested_candidates = mode.requested_candidates(query.k);
    let mask = plaid_address_mask(dataset.as_ref(), prefilter.mask()).await?;
    let candidate_started = Instant::now();
    if mask.max_len() == Some(0) {
        // Record one logical Empty decision per immutable segment without
        // opening an index or entering the Q x C kernel.
        metrics.empty_filter_query_count.add(1);
        metrics.eligible_centroid_empty_count.add(indices.len());
        metrics.candidate.add_duration(candidate_started.elapsed());
        let batch = RecordBatch::new_empty(KNN_INDEX_SCHEMA.clone());
        metrics.baseline.record_output(0);
        return Ok(batch);
    }
    // Proactive exact scoring must stay within the work already requested for
    // result refinement. This captures very selective filters without turning a
    // 1% or 10% predicate into an accidental full-filter scan.
    let small_filter_exact = small_filter_exact_fallback(mask.max_len(), requested_candidates);
    let filtered_query = !mask.is_select_all();
    let mut candidates = HashMap::<u64, f32>::new();
    let mut opened_indices = Vec::with_capacity(indices.len());

    for metadata in &indices {
        metrics.segments_searched_count.add(1);
        let raw_index = dataset
            .open_vector_index(&query.column, &metadata.uuid, &metrics.index)
            .await?;
        raw_index
            .as_any()
            .downcast_ref::<PlaidVectorIndex>()
            .ok_or_else(|| {
                Error::internal("persisted PLAID segment opened as another index type".to_string())
            })?;
        opened_indices.push(raw_index.clone());
        if small_filter_exact {
            continue;
        }
        let query_for_cpu = query_tokens.clone();
        let mask_for_cpu = mask.clone();
        let index_for_cpu = raw_index.clone();
        let query_settings = query.clone();
        let (hits, stats, plan) = spawn_cpu(move || {
            let index = index_for_cpu
                .as_any()
                .downcast_ref::<PlaidVectorIndex>()
                .ok_or_else(|| {
                    Error::internal("PLAID index downcast failed on CPU worker".to_string())
                })?;
            let plan = index.candidate_plan(
                &query_settings,
                requested_candidates,
                query_for_cpu.nrows(),
                mask_for_cpu.as_ref(),
            )?;
            let desired_candidates = plan
                .eligible_documents
                .min(plan.params.top_k)
                .min(index.num_documents());
            let (hits, stats) = index.search_candidates(
                query_for_cpu.view(),
                &plan,
                desired_candidates,
                mask_for_cpu.as_ref(),
            )?;
            Ok::<_, Error>((hits, stats, plan))
        })
        .await?;
        metrics
            .core_approximate_budget_count
            .add(plan.params.n_full_scores);
        metrics.core_residual_budget_count.add(plan.params.top_k);
        metrics.record_eligible_centroid_plan(&plan);
        metrics.record_core(&stats);
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
    let collect_eligible_addresses = || -> Result<Vec<u64>> {
        let mut addresses = Vec::new();
        for raw_index in &opened_indices {
            let plaid = raw_index
                .as_any()
                .downcast_ref::<PlaidVectorIndex>()
                .ok_or_else(|| {
                    Error::internal(
                        "PLAID index downcast failed during exact filter fallback".to_string(),
                    )
                })?;
            addresses.extend(plaid.eligible_row_addresses(mask.as_ref()));
        }
        addresses.sort_unstable();
        addresses.dedup();
        Ok(addresses)
    };
    let mut exact_addresses = None;
    if small_filter_exact {
        exact_addresses = Some(collect_eligible_addresses()?);
    } else if filtered_query && candidates.len() < query.k {
        // Larger filters remain on the ANN path unless it cannot produce
        // min(k, eligible) rows. Only then pay O(F) I/O and exact MaxSim work
        // to preserve database top-k completeness.
        let eligible_addresses = collect_eligible_addresses()?;
        if underfilled_filter_exact_fallback(
            filtered_query,
            candidates.len(),
            query.k,
            eligible_addresses.len(),
        ) {
            exact_addresses = Some(eligible_addresses);
        }
    }
    let exact_fallback = exact_addresses.is_some();
    if let Some(addresses) = exact_addresses {
        candidates.clear();
        for row_address in addresses {
            candidates.insert(row_address, 0.0);
        }
        if small_filter_exact {
            metrics.filter_exact_small_count.add(1);
        } else {
            metrics.filter_exact_underfilled_count.add(1);
        }
        metrics.filter_exact_documents_count.add(candidates.len());
    }
    metrics.candidate.add_duration(candidate_started.elapsed());

    if candidates.is_empty() {
        let batch = RecordBatch::new_empty(KNN_INDEX_SCHEMA.clone());
        metrics.baseline.record_output(0);
        return Ok(batch);
    }

    if mode == PlaidExecutionMode::IndexOnly && !exact_fallback {
        let mut hits = candidates
            .into_iter()
            .map(|(row_address, score)| ExactHit {
                row_address,
                distance: maxsim_distance(score),
            })
            .collect::<Vec<_>>();
        let sort_started = Instant::now();
        // Physical row address makes HashMap iteration irrelevant and gives a
        // deterministic cutoff before stable IDs are available. Equal-score
        // rows at the top-k boundary are semantically interchangeable.
        rank_hits(&mut hits, &query);
        metrics.sort.add_duration(sort_started.elapsed());

        let row_addresses = hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>();
        let result_row_ids =
            index_only_result_row_ids(dataset.clone(), &row_addresses, metrics.as_ref()).await?;
        for (hit, row_id) in hits.iter_mut().zip(result_row_ids) {
            hit.row_address = row_id;
        }
        let sort_started = Instant::now();
        hits.sort_unstable_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.row_address.cmp(&right.row_address))
        });
        let batch = hits_to_batch(&hits)?;
        metrics.sort.add_duration(sort_started.elapsed());
        metrics.baseline.record_output(batch.num_rows());
        return Ok(batch);
    }

    let sort_started = Instant::now();
    let mut candidates = candidates.into_iter().collect::<Vec<_>>();
    if exact_fallback {
        candidates.sort_unstable_by_key(|(row_address, _)| *row_address);
    } else {
        candidates.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        candidates.truncate(requested_candidates.min(candidates.len()));
    }
    metrics.sort.add_duration(sort_started.elapsed());

    let row_addresses = candidates
        .iter()
        .map(|(row_address, _)| *row_address)
        .collect::<Vec<_>>();
    let projection = Arc::new(
        ProjectionRequest::from_columns([query.column.as_str(), ROW_ID], dataset.schema())
            .into_projection_plan(dataset.clone())?,
    );
    let raw_vector_fetch_started = Instant::now();
    let batch =
        TakeBuilder::try_new_from_addresses(dataset.clone(), row_addresses.clone(), projection)?
            .execute()
            .await?;
    metrics
        .raw_vector_fetch
        .add_duration(raw_vector_fetch_started.elapsed());
    metrics.raw_vector_rows_count.add(batch.num_rows());
    if batch.num_rows() != row_addresses.len() {
        return Err(Error::internal(format!(
            "PLAID raw-vector take returned {} rows for {} candidates",
            batch.num_rows(),
            row_addresses.len()
        )));
    }

    let result_row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::internal("PLAID address take did not return _rowid".to_string()))?
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec();
    let exact_started = Instant::now();
    let column = query.column.clone();
    let query_for_cpu = query_tokens.clone();
    let mut hits = spawn_cpu(move || exact_scores(&batch, &column, query_for_cpu.view())).await?;
    metrics.exact.add_duration(exact_started.elapsed());
    for (hit, row_id) in hits.iter_mut().zip(result_row_ids) {
        hit.row_address = row_id;
    }

    let sort_started = Instant::now();
    let batch = finalize_hits(hits, &query)?;
    metrics.sort.add_duration(sort_started.elapsed());
    metrics.baseline.record_output(batch.num_rows());
    Ok(batch)
}

async fn index_only_result_row_ids(
    dataset: Arc<Dataset>,
    row_addresses: &[u64],
    metrics: &PlaidExecMetrics,
) -> Result<Vec<u64>> {
    if !dataset.manifest().uses_stable_row_ids() {
        // In legacy row-address mode the physical address is the public row ID.
        return Ok(row_addresses.to_vec());
    }

    let projection = Arc::new(
        ProjectionRequest::from_columns([ROW_ID], dataset.schema())
            .into_projection_plan(dataset.clone())?,
    );
    let row_id_fetch_started = Instant::now();
    let batch = TakeBuilder::try_new_from_addresses(dataset, row_addresses.to_vec(), projection)?
        .execute()
        .await?;
    metrics
        .row_id_fetch
        .add_duration(row_id_fetch_started.elapsed());
    metrics.row_id_rows_count.add(batch.num_rows());
    if batch.num_rows() != row_addresses.len() {
        return Err(Error::internal(format!(
            "PLAID row-id-only take returned {} rows for {} candidates",
            batch.num_rows(),
            row_addresses.len()
        )));
    }
    Ok(batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::internal("PLAID row-id-only take omitted _rowid".to_string()))?
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec())
}

fn rank_hits(hits: &mut Vec<ExactHit>, query: &Query) {
    hits.retain(|hit| {
        query.lower_bound.is_none_or(|lower| hit.distance >= lower)
            && query.upper_bound.is_none_or(|upper| hit.distance < upper)
    });
    hits.sort_unstable_by(|left, right| {
        left.distance
            .total_cmp(&right.distance)
            .then_with(|| left.row_address.cmp(&right.row_address))
    });
    hits.truncate(query.k.min(hits.len()));
}

fn hits_to_batch(hits: &[ExactHit]) -> Result<RecordBatch> {
    let distances = Float32Array::from(hits.iter().map(|hit| hit.distance).collect::<Vec<_>>());
    let rows = UInt64Array::from(hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>());
    Ok(RecordBatch::try_new(
        KNN_INDEX_SCHEMA.clone(),
        vec![Arc::new(distances), Arc::new(rows)],
    )?)
}

fn finalize_hits(mut hits: Vec<ExactHit>, query: &Query) -> Result<RecordBatch> {
    rank_hits(&mut hits, query);
    hits_to_batch(&hits)
}

async fn plaid_address_mask(dataset: &Dataset, mask: Arc<RowAddrMask>) -> Result<Arc<RowAddrMask>> {
    if !dataset.manifest().uses_stable_row_ids() {
        return Ok(mask);
    }
    let row_id_index = get_row_id_index(dataset)
        .await?
        .ok_or_else(|| Error::internal("stable-row-id dataset has no row-id index".to_string()))?;
    let translate_enumerable = |row_ids: &RowAddrTreeMap| -> Option<RowAddrTreeMap> {
        let row_ids = row_ids.row_addrs()?.map(u64::from).collect::<Vec<_>>();
        Some(RowAddrTreeMap::from_iter(
            row_id_index
                .get_many(&row_ids)
                .into_iter()
                .flatten()
                .map(u64::from),
        ))
    };
    // Full-prefix stable-ID markers cannot be enumerated from the mask alone.
    // Evaluate them over the finite live RowIdIndex instead. Returning a
    // physical allow-list for this fallback also ensures tombstoned rows stay
    // invisible. An explicit block list uses the same live fallback whenever
    // the dataset has deletion files; otherwise translating only its blocked
    // IDs is the cheaper equivalent representation.
    let translate_live_selection = || {
        RowAddrTreeMap::from_iter(row_id_index.iter().filter_map(|(row_id, row_address)| {
            mask.selected(row_id).then_some(u64::from(row_address))
        }))
    };
    let has_deletions = dataset
        .manifest()
        .fragments
        .iter()
        .any(|fragment| fragment.deletion_file.is_some());
    let translated = match mask.as_ref() {
        RowAddrMask::AllowList(row_ids) => translate_enumerable(row_ids)
            .map(RowAddrMask::from_allowed)
            .unwrap_or_else(|| RowAddrMask::from_allowed(translate_live_selection())),
        RowAddrMask::BlockList(row_ids) if !has_deletions => translate_enumerable(row_ids)
            .map(RowAddrMask::from_block)
            .unwrap_or_else(|| RowAddrMask::from_allowed(translate_live_selection())),
        RowAddrMask::BlockList(_) => RowAddrMask::from_allowed(translate_live_selection()),
    };
    Ok(Arc::new(translated))
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
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatchIterator,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::address::RowAddress;
    use lance_linalg::distance::{DistanceType, multivec_distance};
    use ndarray::array;

    use crate::dataset::WriteParams;

    #[test]
    fn exact_filter_fallback_policy_tracks_budget_and_true_underfill() {
        assert_eq!(PlaidExecutionMode::IndexOnly.requested_candidates(10), 10);
        assert_eq!(PlaidExecutionMode::IndexOnly.raw_refinement_budget(), 0);
        let exact = PlaidExecutionMode::Exact {
            raw_refinement_budget: 50,
        };
        assert_eq!(exact.requested_candidates(10), 50);
        assert_eq!(exact.raw_refinement_budget(), 50);

        assert!(small_filter_exact_fallback(Some(10), 50));
        assert!(small_filter_exact_fallback(Some(50), 50));
        assert!(!small_filter_exact_fallback(Some(0), 50));
        assert!(!small_filter_exact_fallback(Some(51), 50));
        assert!(!small_filter_exact_fallback(Some(1_000), 50));
        assert!(!small_filter_exact_fallback(None, 50));

        assert!(underfilled_filter_exact_fallback(true, 3, 10, 10));
        assert!(underfilled_filter_exact_fallback(true, 4, 10, 5));
        assert!(!underfilled_filter_exact_fallback(true, 5, 10, 5));
        assert!(!underfilled_filter_exact_fallback(true, 10, 10, 100));
        assert!(!underfilled_filter_exact_fallback(false, 3, 10, 10));
    }

    #[tokio::test]
    async fn stable_full_prefix_masks_translate_across_fragments_and_deletions() {
        let directory = lance_core::utils::tempfile::TempStrDir::default();
        let first = RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef,
        )])
        .unwrap();
        let schema = first.schema();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(first)], schema.clone()),
            directory.as_ref(),
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let second = RecordBatch::try_from_iter([(
            "id",
            Arc::new(Int32Array::from(vec![2, 3])) as ArrayRef,
        )])
        .unwrap();
        dataset
            .append(RecordBatchIterator::new(vec![Ok(second)], schema), None)
            .await
            .unwrap();
        dataset.delete("id = 1").await.unwrap();

        let addresses = [
            RowAddress::new_from_parts(0, 0),
            RowAddress::new_from_parts(0, 1),
            RowAddress::new_from_parts(1, 0),
            RowAddress::new_from_parts(1, 1),
        ];
        // All generated stable IDs are in high-32-bit prefix zero even though
        // their current physical homes span two fragments.
        let mut full_zero = RowAddrTreeMap::new();
        full_zero.insert_fragment(0);
        let allowed = plaid_address_mask(
            &dataset,
            Arc::new(RowAddrMask::from_allowed(full_zero.clone())),
        )
        .await
        .unwrap();
        assert!(allowed.iter_addrs().is_some());
        assert!(allowed.selected(addresses[0].into()));
        assert!(!allowed.selected(addresses[1].into()));
        assert!(allowed.selected(addresses[2].into()));
        assert!(allowed.selected(addresses[3].into()));

        let blocked = plaid_address_mask(&dataset, Arc::new(RowAddrMask::from_block(full_zero)))
            .await
            .unwrap();
        assert!(
            addresses
                .iter()
                .all(|address| !blocked.selected((*address).into()))
        );

        // Prefix one contains no stable IDs. A full-prefix block therefore
        // leaves every live row selected while the deleted physical row stays
        // invisible.
        let mut absent_prefix = RowAddrTreeMap::new();
        absent_prefix.insert_fragment(1);
        let block_absent =
            plaid_address_mask(&dataset, Arc::new(RowAddrMask::from_block(absent_prefix)))
                .await
                .unwrap();
        assert!(block_absent.selected(addresses[0].into()));
        assert!(!block_absent.selected(addresses[1].into()));
        assert!(block_absent.selected(addresses[2].into()));
        assert!(block_absent.selected(addresses[3].into()));
    }

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
        let batch =
            RecordBatch::try_from_iter([("mv", Arc::new(documents.clone()) as ArrayRef)]).unwrap();
        let query = array![[1.0_f32, 0.0, 0.0, 0.0]];
        let actual = exact_scores(&batch, "mv", query.view()).unwrap();
        let query_values = Float32Array::from(query.iter().copied().collect::<Vec<_>>());
        let expected = multivec_distance(&query_values, &documents, DistanceType::Dot).unwrap();
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

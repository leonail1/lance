// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Single-operator database-native PLAID multi-vector search.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::AsArray;
use arrow::datatypes::UInt64Type;
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
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
use lance_arrow::interleave_batches;
use lance_arrow::json::convert_lance_json_to_arrow;
use lance_core::ROW_ID;
use lance_core::datatypes::{OnMissing, Projection};
use lance_core::utils::tokio::spawn_cpu;
use lance_datafusion::projection::ProjectionPlan;
use lance_datafusion::utils::ExecutionPlanMetricsSetExt;
use lance_index::prefilter::PreFilter;
use lance_index::vector::{DIST_COL, Query};
use lance_linalg::distance::dot_f32;
use lance_plaid::{EligibleCentroidDecision, PlaidSearchParams};
use lance_select::{RowAddrMask, RowAddrTreeMap};
use lance_table::format::IndexMetadata;
use ndarray::ArrayView2;

use super::knn::KNN_INDEX_SCHEMA;
use super::take::TakeExec;
use super::utils::{IndexMetrics, PreFilterSource, build_prefilter};
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use crate::index::DatasetIndexInternalExt;
use crate::index::plaid::{
    BoundedEligibleOrdinals, PLAID_DEFAULT_DECOMPRESS_DOCUMENTS, PlaidCandidatePlan,
    PlaidVectorIndex, is_plaid_index_metadata, maxsim_distance, plaid_search_params,
    query_to_array,
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
// Nested candidate sub-time for a direct scoring attempt. It includes direct
// planning, mask/ordinal enumeration, query and token-range validation, score,
// and sort, so it overlaps residual/sort and the executor's end-to-end time.
const DIRECT_RESIDUAL_TIME: &str = "plaid_direct_residual_time";
const DIRECT_RESIDUAL_QUERY_COUNT: &str = "plaid_direct_residual_queries";
const DIRECT_RESIDUAL_SEGMENT_COUNT: &str = "plaid_direct_residual_segments";
const DIRECT_RESIDUAL_DOCUMENT_COUNT: &str = "plaid_direct_residual_documents";
const LEGACY_RESIDUAL_QUERY_COUNT: &str = "plaid_legacy_residual_queries";
const LEGACY_RESIDUAL_SEGMENT_COUNT: &str = "plaid_legacy_residual_segments";
const DIRECT_RESIDUAL_SMALL_EXACT_PRECEDENCE_COUNT: &str =
    "plaid_direct_residual_small_exact_precedence_queries";
const DIRECT_RESIDUAL_DISABLED_COUNT: &str = "plaid_direct_residual_skipped_disabled_segments";
const DIRECT_RESIDUAL_EXPLICIT_CEILING_COUNT: &str =
    "plaid_direct_residual_skipped_explicit_ceiling_segments";
const DIRECT_RESIDUAL_UNFILTERED_COUNT: &str = "plaid_direct_residual_skipped_unfiltered_segments";
const DIRECT_RESIDUAL_NON_ENUMERABLE_COUNT: &str =
    "plaid_direct_residual_skipped_non_enumerable_segments";
const DIRECT_RESIDUAL_EMPTY_SEGMENT_COUNT: &str = "plaid_direct_residual_skipped_empty_segments";
const DIRECT_RESIDUAL_OVER_LIMIT_COUNT: &str = "plaid_direct_residual_skipped_over_limit_segments";
const DIRECT_RESIDUAL_BUDGET_MISMATCH_COUNT: &str =
    "plaid_direct_residual_skipped_budget_mismatch_segments";
const DIRECT_RESIDUAL_EMPTY_TOKENS_COUNT: &str =
    "plaid_direct_residual_skipped_empty_tokens_segments";
const FUSED_FINAL_TAKE_TIME: &str = "plaid_fused_final_take_time";
const FUSED_FINAL_TAKE_SELECT_TIME: &str = "plaid_fused_final_take_select_sub_time";
const FUSED_FINAL_TAKE_LOGICAL_PROJECTION_TIME: &str =
    "plaid_fused_final_take_logical_projection_sub_time";
// Direct and legacy projection timers are mutually exclusive nested sub-times
// of FUSED_FINAL_TAKE_LOGICAL_PROJECTION_TIME, not additive top-level phases.
const FUSED_FINAL_TAKE_DIRECT_PROJECTION_TIME: &str =
    "plaid_fused_final_take_direct_projection_sub_time";
const FUSED_FINAL_TAKE_LEGACY_PROJECTION_TIME: &str =
    "plaid_fused_final_take_legacy_projection_sub_time";
const FUSED_FINAL_TAKE_JSON_CONVERSION_TIME: &str =
    "plaid_fused_final_take_json_conversion_sub_time";
const FUSED_FINAL_TAKE_ASSEMBLY_TIME: &str = "plaid_fused_final_take_assembly_sub_time";
const FUSED_FINAL_TAKE_QUERY_COUNT: &str = "plaid_fused_final_take_queries";
const FUSED_FINAL_TAKE_DIRECT_PROJECTION_QUERY_COUNT: &str =
    "plaid_fused_final_take_direct_projection_queries";
const FUSED_FINAL_TAKE_LEGACY_PROJECTION_QUERY_COUNT: &str =
    "plaid_fused_final_take_legacy_projection_queries";
const FUSED_FINAL_TAKE_CANDIDATE_ROWS_COUNT: &str = "plaid_fused_final_take_candidate_rows";
const FUSED_FINAL_TAKE_OUTPUT_ROWS_COUNT: &str = "plaid_fused_final_take_output_rows";
const FUSED_FINAL_TAKE_OUTPUT_BYTES_COUNT: &str = "plaid_fused_final_take_output_batch_bytes";
// This is a nested sub-time of PLAID_SORT_TIME, not an additive phase.
const SORTED_RAW_TAKE_TIME: &str = "plaid_sorted_raw_take_sub_time";
const SORTED_RAW_TAKE_QUERY_COUNT: &str = "plaid_sorted_raw_take_queries";
const SORTED_RAW_TAKE_ROWS_COUNT: &str = "plaid_sorted_raw_take_rows";
const RAW_VECTOR_BYTES_COUNT: &str = "plaid_raw_vector_batch_bytes";
const GROUPED_REFINEMENT_TIME: &str = "plaid_grouped_refinement_take_sub_time";
const GROUPED_REFINEMENT_HELPER_WALL_TIME: &str = "plaid_grouped_refinement_helper_wall_sub_time";
// The one-time plan and grouping timers are disjoint wall-clock sub-phases of
// GROUPED_REFINEMENT_TIME. Fanout collection and row-offset injection are also
// one-time wall sub-phases. Explicit shared-scheduler creation is another
// one-time wall sub-phase (zero/unavailable for the strict control). The
// per-fragment open/read/total timers are concurrent child-work sums (plus a
// max critical-path proxy), so they are not additive with the parent wall.
// Control open includes its internal per-file scheduler creation; treatment
// open does not include the separately reported one-time shared creation.
const GROUPED_REFINEMENT_PLAN_TIME: &str = "plaid_grouped_refinement_plan_sub_time";
const GROUPED_REFINEMENT_GROUPING_TIME: &str = "plaid_grouped_refinement_grouping_sub_time";
const GROUPED_REFINEMENT_FANOUT_COLLECT_WALL_TIME: &str =
    "plaid_grouped_refinement_fanout_collect_wall_sub_time";
const GROUPED_REFINEMENT_ROW_OFFSET_INJECTION_WALL_TIME: &str =
    "plaid_grouped_refinement_row_offset_injection_wall_sub_time";
const GROUPED_REFINEMENT_SCHEDULER_CREATE_WALL_TIME: &str =
    "plaid_grouped_refinement_scheduler_create_wall_sub_time";
const GROUPED_REFINEMENT_FRAGMENT_OPEN_SUM_TIME: &str =
    "plaid_grouped_refinement_fragment_open_sum_time";
const GROUPED_REFINEMENT_FRAGMENT_OPEN_MAX_TIME: &str =
    "plaid_grouped_refinement_fragment_open_max_time";
const GROUPED_REFINEMENT_FRAGMENT_READ_SUM_TIME: &str =
    "plaid_grouped_refinement_fragment_read_sum_time";
const GROUPED_REFINEMENT_FRAGMENT_READ_MAX_TIME: &str =
    "plaid_grouped_refinement_fragment_read_max_time";
const GROUPED_REFINEMENT_FRAGMENT_TOTAL_ELAPSED_SUM_TIME: &str =
    "plaid_grouped_refinement_fragment_total_elapsed_sum_time";
const GROUPED_REFINEMENT_FRAGMENT_TOTAL_ELAPSED_MAX_TIME: &str =
    "plaid_grouped_refinement_fragment_total_elapsed_max_time";
const GROUPED_REFINEMENT_QUERY_COUNT: &str = "plaid_grouped_refinement_queries";
const GROUPED_REFINEMENT_BATCH_COUNT: &str = "plaid_grouped_refinement_batches";
const GROUPED_REFINEMENT_ROWS_COUNT: &str = "plaid_grouped_refinement_rows";
const GROUPED_REFINEMENT_FANOUT_CONCURRENCY_LIMIT_COUNT: &str =
    "plaid_grouped_refinement_fanout_concurrency_limit";
const GROUPED_REFINEMENT_BATCH_BYTES_COUNT: &str = "plaid_grouped_refinement_batch_bytes";
const GROUPED_REFINEMENT_ROWS_PER_FRAGMENT_MIN_COUNT: &str =
    "plaid_grouped_refinement_rows_per_fragment_min_per_query";
const GROUPED_REFINEMENT_ROWS_PER_FRAGMENT_MAX_COUNT: &str =
    "plaid_grouped_refinement_rows_per_fragment_max_per_query";
// Scoped scheduler counters include only physical ranges submitted through
// the explicit default-base V2 scheduler(s). They exclude stable-row-ID and
// deletion side reads as well as legacy/non-default-base file I/O.
const GROUPED_REFINEMENT_SCHEDULER_SCOPED_IOPS_COUNT: &str =
    "plaid_grouped_refinement_scheduler_scoped_iops";
const GROUPED_REFINEMENT_SCHEDULER_SCOPED_REQUESTS_COUNT: &str =
    "plaid_grouped_refinement_scheduler_scoped_requests";
const GROUPED_REFINEMENT_SCHEDULER_SCOPED_BYTES_READ_COUNT: &str =
    "plaid_grouped_refinement_scheduler_scoped_bytes_read";
const GROUPED_REFINEMENT_SCHEDULER_STATS_COVERED_FRAGMENTS_COUNT: &str =
    "plaid_grouped_refinement_scheduler_stats_covered_fragments";
const GROUPED_REFINEMENT_FALLBACK_COUNT: &str = "plaid_grouped_refinement_fallbacks";
const GROUPED_SHARED_SCHEDULER_QUERY_COUNT: &str = "plaid_grouped_shared_scheduler_queries";
const GROUPED_SHARED_SCHEDULER_FRAGMENT_COUNT: &str = "plaid_grouped_shared_scheduler_fragments";
const GROUPED_PER_FRAGMENT_SCHEDULER_QUERY_COUNT: &str =
    "plaid_grouped_per_fragment_scheduler_queries";
const GROUPED_PER_FRAGMENT_SCHEDULER_FRAGMENT_COUNT: &str =
    "plaid_grouped_per_fragment_scheduler_fragments";
const GROUPED_SHARED_SCHEDULER_FALLBACK_QUERY_COUNT: &str =
    "plaid_grouped_shared_scheduler_fallback_queries";
const GROUPED_SHARED_SCHEDULER_FALLBACK_LEGACY_FRAGMENT_COUNT: &str =
    "plaid_grouped_shared_scheduler_fallback_legacy_fragments";
const GROUPED_SHARED_SCHEDULER_FALLBACK_NONPRIMARY_FRAGMENT_COUNT: &str =
    "plaid_grouped_shared_scheduler_fallback_nonprimary_fragments";
const GROUPED_SHARED_SCHEDULER_FALLBACK_UNSUPPORTED_FRAGMENT_COUNT: &str =
    "plaid_grouped_shared_scheduler_fallback_unsupported_fragments";
const DATA_FILE_READER_CACHE_LOOKUP_SUM_TIME: &str = "plaid_data_file_reader_cache_lookup_sum_time";
const DATA_FILE_READER_CACHE_GET_OR_OPEN_SUM_TIME: &str =
    "plaid_data_file_reader_cache_get_or_open_sum_time";
const DATA_FILE_READER_CACHE_COALESCED_WAIT_SUM_TIME: &str =
    "plaid_data_file_reader_cache_coalesced_wait_sum_time";
const DATA_FILE_READER_CACHE_ACQUIRE_SUM_TIME: &str =
    "plaid_data_file_reader_cache_acquire_sum_time";
const DATA_FILE_READER_CACHE_PHYSICAL_OPEN_SUM_TIME: &str =
    "plaid_data_file_reader_cache_physical_open_sum_time";
const DATA_FILE_READER_CACHE_BIND_SUM_TIME: &str = "plaid_data_file_reader_cache_bind_sum_time";
const DATA_FILE_READER_CACHE_QUERY_COUNT: &str = "plaid_data_file_reader_cache_queries";
const DATA_FILE_READER_CACHE_ELIGIBLE_FILE_COUNT: &str =
    "plaid_data_file_reader_cache_eligible_files";
const DATA_FILE_READER_CACHE_LOOKUP_FILE_COUNT: &str = "plaid_data_file_reader_cache_lookup_files";
const DATA_FILE_READER_CACHE_HIT_FILE_COUNT: &str = "plaid_data_file_reader_cache_hit_files";
const DATA_FILE_READER_CACHE_MISS_OPEN_FILE_COUNT: &str =
    "plaid_data_file_reader_cache_miss_open_files";
const DATA_FILE_READER_CACHE_COALESCED_FILE_COUNT: &str =
    "plaid_data_file_reader_cache_coalesced_files";
const DATA_FILE_READER_CACHE_BYPASS_FILE_COUNT: &str = "plaid_data_file_reader_cache_bypass_files";
const DATA_FILE_READER_CACHE_FALLBACK_OPEN_FILE_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_open_files";
const DATA_FILE_READER_CACHE_OPEN_FAILURE_COUNT: &str =
    "plaid_data_file_reader_cache_open_failures";
const DATA_FILE_READER_CACHE_FD_BUDGET_REJECTION_COUNT: &str =
    "plaid_data_file_reader_cache_fd_budget_rejections";
const DATA_FILE_READER_CACHE_RESIDENT_START_COUNT: &str =
    "plaid_data_file_reader_cache_resident_entries_start_approx";
const DATA_FILE_READER_CACHE_RESIDENT_END_COUNT: &str =
    "plaid_data_file_reader_cache_resident_entries_end_approx";
const DATA_FILE_READER_CACHE_CAPACITY_COUNT: &str = "plaid_data_file_reader_cache_capacity";
const DATA_FILE_READER_CACHE_FD_SOFT_LIMIT_COUNT: &str =
    "plaid_data_file_reader_cache_fd_soft_limit";
const DATA_FILE_READER_CACHE_FALLBACK_QUERY_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_queries";
const DATA_FILE_READER_CACHE_PARTIAL_FALLBACK_QUERY_COUNT: &str =
    "plaid_data_file_reader_cache_partial_fallback_queries";
const DATA_FILE_READER_CACHE_FALLBACK_LEGACY_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_legacy_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_NONPRIMARY_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_nonprimary_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_NONLOCAL_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_nonlocal_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_UNKNOWN_SIZE_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_unknown_size_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_SMALL_FILE_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_small_file_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_UNSUPPORTED_FRAGMENT_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_unsupported_fragments";
const DATA_FILE_READER_CACHE_FALLBACK_CAPACITY_QUERY_COUNT: &str =
    "plaid_data_file_reader_cache_fallback_capacity_queries";

const DIRECT_RESIDUAL_ENABLED_ENV: &str = "LANCE_PLAID_DIRECT_RESIDUAL_ENABLED";
const DIRECT_RESIDUAL_MAX_DOCUMENTS_ENV: &str = "LANCE_PLAID_DIRECT_RESIDUAL_MAX_DOCUMENTS";
const DEFAULT_DIRECT_RESIDUAL_MAX_DOCUMENTS: usize = 1024;
const FUSED_FINAL_TAKE_ENABLED_ENV: &str = "LANCE_PLAID_FUSED_FINAL_TAKE_ENABLED";
const SORTED_RAW_TAKE_ENABLED_ENV: &str = "LANCE_PLAID_SORTED_RAW_TAKE_ENABLED";
const GROUPED_REFINEMENT_ENABLED_ENV: &str = "LANCE_PLAID_GROUPED_REFINEMENT_ENABLED";
const GROUPED_SHARED_SCHEDULER_ENABLED_ENV: &str = "LANCE_PLAID_GROUPED_SHARED_SCHEDULER_ENABLED";
const DATA_FILE_READER_CACHE_ENABLED_ENV: &str = "LANCE_PLAID_DATA_FILE_READER_CACHE_ENABLED";
const DIRECT_WINNER_PROJECTION_ENABLED_ENV: &str = "LANCE_PLAID_DIRECT_WINNER_PROJECTION_ENABLED";

#[cfg(test)]
thread_local! {
    static TAKE_OPT_TEST_OVERRIDE: std::cell::Cell<Option<(bool, bool, bool, bool, bool, bool)>> = const {
        std::cell::Cell::new(None)
    };
}

#[cfg(test)]
pub(crate) struct PlaidTakeOptimizationTestGuard {
    previous: Option<(bool, bool, bool, bool, bool, bool)>,
}

#[cfg(test)]
impl PlaidTakeOptimizationTestGuard {
    pub(crate) fn new(fused: bool, sorted: bool) -> Self {
        let previous =
            TAKE_OPT_TEST_OVERRIDE.replace(Some((fused, sorted, false, false, false, false)));
        Self { previous }
    }

    pub(crate) fn new_with_direct_winner_projection(
        fused: bool,
        sorted: bool,
        grouped: bool,
        direct_winner_projection: bool,
    ) -> Self {
        let previous = TAKE_OPT_TEST_OVERRIDE.replace(Some((
            fused,
            sorted,
            grouped,
            direct_winner_projection,
            false,
            false,
        )));
        Self { previous }
    }

    pub(crate) fn new_with_grouped_shared_scheduler(
        fused: bool,
        sorted: bool,
        grouped: bool,
        direct_winner_projection: bool,
        grouped_shared_scheduler: bool,
    ) -> Self {
        let previous = TAKE_OPT_TEST_OVERRIDE.replace(Some((
            fused,
            sorted,
            grouped,
            direct_winner_projection,
            grouped_shared_scheduler,
            false,
        )));
        Self { previous }
    }
}

#[cfg(test)]
impl Drop for PlaidTakeOptimizationTestGuard {
    fn drop(&mut self) {
        TAKE_OPT_TEST_OVERRIDE.set(self.previous);
    }
}

#[cfg(test)]
fn take_optimization_test_override() -> Option<(bool, bool, bool, bool, bool, bool)> {
    TAKE_OPT_TEST_OVERRIDE.get()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FusedFinalTakeConfig {
    enabled: bool,
}

impl FusedFinalTakeConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((enabled, _, _, _, _, _)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(FUSED_FINAL_TAKE_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {FUSED_FINAL_TAKE_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SortedRawTakeConfig {
    enabled: bool,
}

impl SortedRawTakeConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((_, enabled, _, _, _, _)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(SORTED_RAW_TAKE_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {SORTED_RAW_TAKE_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GroupedRefinementConfig {
    enabled: bool,
}

impl GroupedRefinementConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((_, _, enabled, _, _, _)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(GROUPED_REFINEMENT_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {GROUPED_REFINEMENT_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GroupedSharedSchedulerConfig {
    enabled: bool,
}

impl GroupedSharedSchedulerConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((_, _, _, _, enabled, _)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(GROUPED_SHARED_SCHEDULER_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {GROUPED_SHARED_SCHEDULER_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DataFileReaderCacheConfig {
    enabled: bool,
}

impl DataFileReaderCacheConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((_, _, _, _, _, enabled)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(DATA_FILE_READER_CACHE_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {DATA_FILE_READER_CACHE_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DirectWinnerProjectionConfig {
    enabled: bool,
}

impl DirectWinnerProjectionConfig {
    fn from_env() -> Result<Self> {
        #[cfg(test)]
        if let Some((_, _, _, enabled, _, _)) = take_optimization_test_override() {
            return Ok(Self { enabled });
        }
        let enabled = read_utf8_env(DIRECT_WINNER_PROJECTION_ENABLED_ENV)?;
        Self::from_value(enabled.as_deref())
    }

    fn from_value(enabled: Option<&str>) -> Result<Self> {
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {DIRECT_WINNER_PROJECTION_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(false);
        Ok(Self { enabled })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectResidualConfig {
    enabled: bool,
    max_documents: usize,
}

impl Default for DirectResidualConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_documents: DEFAULT_DIRECT_RESIDUAL_MAX_DOCUMENTS,
        }
    }
}

impl DirectResidualConfig {
    fn from_env() -> Result<Self> {
        let enabled = read_utf8_env(DIRECT_RESIDUAL_ENABLED_ENV)?;
        let max_documents = read_utf8_env(DIRECT_RESIDUAL_MAX_DOCUMENTS_ENV)?;
        Self::from_values(enabled.as_deref(), max_documents.as_deref())
    }

    fn from_values(enabled: Option<&str>, max_documents: Option<&str>) -> Result<Self> {
        let default = Self::default();
        let enabled = enabled
            .map(|value| {
                parse_bool(value).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "invalid {DIRECT_RESIDUAL_ENABLED_ENV}={value:?}; expected true/false"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(default.enabled);
        let max_documents = max_documents
            .map(|value| {
                value.parse::<usize>().map_err(|_| {
                    Error::invalid_input(format!(
                        "invalid {DIRECT_RESIDUAL_MAX_DOCUMENTS_ENV}={value:?}; expected a non-negative integer"
                    ))
                })
            })
            .transpose()?
            .unwrap_or(default.max_documents);
        Ok(Self {
            enabled,
            max_documents,
        })
    }

    fn mode_name(self) -> &'static str {
        if self.enabled { "enabled" } else { "disabled" }
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn read_utf8_env(name: &str) -> Result<Option<String>> {
    match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::invalid_input(format!(
            "environment variable {name} is not valid UTF-8"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectResidualDecision {
    Direct,
    Disabled,
    ExplicitProbeCeiling,
    Unfiltered,
    NonEnumerable,
    EmptySegment,
    OverLimit,
    BudgetMismatch,
    EmptyTokens,
}

enum DirectResidualPlan {
    Direct {
        document_ordinals: Vec<u32>,
        params: PlaidSearchParams,
    },
    Empty,
    Legacy(DirectResidualDecision),
}

enum SegmentSearchOutcome {
    Direct {
        hits: Vec<lance_plaid::SearchHit>,
        stats: lance_plaid::PlaidSearchStats,
        documents: usize,
        n_full_scores: usize,
        attempt_nanos: u64,
    },
    Empty,
    Legacy {
        hits: Vec<lance_plaid::SearchHit>,
        stats: lance_plaid::PlaidSearchStats,
        plan: PlaidCandidatePlan,
        direct_decision: DirectResidualDecision,
        direct_attempt_nanos: Option<u64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidualSegmentPath {
    Direct,
    Legacy,
    Empty,
}

#[derive(Default)]
struct ResidualQueryUsage {
    used_direct: bool,
    used_legacy: bool,
}

impl ResidualQueryUsage {
    fn observe(&mut self, path: ResidualSegmentPath) {
        match path {
            ResidualSegmentPath::Direct => self.used_direct = true,
            ResidualSegmentPath::Legacy => self.used_legacy = true,
            ResidualSegmentPath::Empty => {}
        }
    }

    fn record_query_metrics(&self, metrics: &PlaidExecMetrics) {
        if self.used_direct {
            metrics.direct_residual_query_count.add(1);
        }
        if self.used_legacy {
            metrics.legacy_residual_query_count.add(1);
        }
    }
}

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
    direct_residual_config: DirectResidualConfig,
    fused_final_take_config: FusedFinalTakeConfig,
    sorted_raw_take_config: SortedRawTakeConfig,
    grouped_refinement_config: GroupedRefinementConfig,
    grouped_shared_scheduler_config: GroupedSharedSchedulerConfig,
    data_file_reader_cache_config: DataFileReaderCacheConfig,
    direct_winner_projection_config: DirectWinnerProjectionConfig,
    fused_output_projection: Option<Projection>,
    output_schema: SchemaRef,
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
        Self::try_new_with_configs(
            dataset,
            indices,
            query,
            prefilter_source,
            DirectResidualConfig::from_env()?,
            FusedFinalTakeConfig::from_env()?,
            SortedRawTakeConfig::from_env()?,
            GroupedRefinementConfig::from_env()?,
            GroupedSharedSchedulerConfig::from_env()?,
            DataFileReaderCacheConfig::from_env()?,
            DirectWinnerProjectionConfig::from_env()?,
            None,
        )
    }

    fn try_new_with_configs(
        dataset: Arc<Dataset>,
        indices: Vec<IndexMetadata>,
        query: Query,
        prefilter_source: PreFilterSource,
        direct_residual_config: DirectResidualConfig,
        fused_final_take_config: FusedFinalTakeConfig,
        sorted_raw_take_config: SortedRawTakeConfig,
        grouped_refinement_config: GroupedRefinementConfig,
        grouped_shared_scheduler_config: GroupedSharedSchedulerConfig,
        data_file_reader_cache_config: DataFileReaderCacheConfig,
        direct_winner_projection_config: DirectWinnerProjectionConfig,
        fused_output_projection: Option<Projection>,
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
        if fused_output_projection.is_some()
            && (!fused_final_take_config.enabled
                || !matches!(mode, PlaidExecutionMode::Exact { .. }))
        {
            return Err(Error::invalid_input(
                "PLAID fused final take requires enabled exact refinement".to_string(),
            ));
        }
        let output_schema = if let Some(projection) = fused_output_projection.as_ref() {
            // Match TakeExec exactly: fields already produced by PLAID (most
            // importantly _rowid) must not be added a second time.
            let missing_projection = projection
                .clone()
                .subtract_arrow_schema(KNN_INDEX_SCHEMA.as_ref(), OnMissing::Ignore)?;
            let output_schema = TakeExec::calculate_output_schema(
                dataset.schema(),
                KNN_INDEX_SCHEMA.as_ref(),
                &missing_projection,
            );
            Arc::new(ArrowSchema::from(&output_schema))
        } else {
            KNN_INDEX_SCHEMA.clone()
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(output_schema.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            dataset,
            indices,
            query,
            mode,
            direct_residual_config,
            fused_final_take_config,
            sorted_raw_take_config,
            grouped_refinement_config,
            grouped_shared_scheduler_config,
            data_file_reader_cache_config,
            direct_winner_projection_config,
            fused_output_projection,
            output_schema,
            prefilter_source,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// Return an exact-search operator that carries dataset output columns in
    /// the existing raw-vector take. Index-only queries and the default-off
    /// control path keep the ordinary outer [`TakeExec`].
    pub(crate) fn try_with_fused_output_projection(
        &self,
        projection: Projection,
    ) -> Result<Option<Self>> {
        if !self.fused_final_take_config.enabled
            || !matches!(self.mode, PlaidExecutionMode::Exact { .. })
            || !projection.has_data_fields()
        {
            return Ok(None);
        }
        Ok(Some(Self::try_new_with_configs(
            self.dataset.clone(),
            self.indices.clone(),
            self.query.clone(),
            self.prefilter_source.clone(),
            self.direct_residual_config,
            self.fused_final_take_config,
            self.sorted_raw_take_config,
            self.grouped_refinement_config,
            self.grouped_shared_scheduler_config,
            self.data_file_reader_cache_config,
            self.direct_winner_projection_config,
            Some(projection),
        )?))
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
                "PlaidSearch: name={}, k={}, segments={}, mode={}, core_residual_budget={}, raw_refinement_budget={}, filter_exact_fallback=enabled, direct_residual_mode={}, direct_residual_max_documents={}, sorted_raw_take_mode={}, grouped_refinement_mode={}, grouped_shared_scheduler_mode={}, data_file_reader_cache_mode={}, fused_final_take_mode={}, direct_winner_projection_mode={}, fused_output_fields={}",
                self.indices[0].name,
                self.query.k,
                self.indices.len(),
                self.mode.name(),
                self.mode
                    .requested_candidates(self.query.k)
                    .max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS),
                self.mode.raw_refinement_budget(),
                self.direct_residual_config.mode_name(),
                self.direct_residual_config.max_documents,
                self.sorted_raw_take_config.mode_name(),
                self.grouped_refinement_config.mode_name(),
                self.grouped_shared_scheduler_config.mode_name(),
                self.data_file_reader_cache_config.mode_name(),
                self.fused_final_take_config.mode_name(),
                self.direct_winner_projection_config.mode_name(),
                self.fused_output_projection
                    .as_ref()
                    .map(|projection| projection.to_bare_schema().fields.len())
                    .unwrap_or(0),
            ),
            DisplayFormatType::TreeRender => write!(
                formatter,
                "PlaidSearch\nname={}\nk={}\nsegments={}\nmode={}\ncore_residual_budget={}\nraw_refinement_budget={}\nfilter_exact_fallback=enabled\ndirect_residual_mode={}\ndirect_residual_max_documents={}\nsorted_raw_take_mode={}\ngrouped_refinement_mode={}\ngrouped_shared_scheduler_mode={}\ndata_file_reader_cache_mode={}\nfused_final_take_mode={}\ndirect_winner_projection_mode={}\nfused_output_fields={}",
                self.indices[0].name,
                self.query.k,
                self.indices.len(),
                self.mode.name(),
                self.mode
                    .requested_candidates(self.query.k)
                    .max(PLAID_DEFAULT_DECOMPRESS_DOCUMENTS),
                self.mode.raw_refinement_budget(),
                self.direct_residual_config.mode_name(),
                self.direct_residual_config.max_documents,
                self.sorted_raw_take_config.mode_name(),
                self.grouped_refinement_config.mode_name(),
                self.grouped_shared_scheduler_config.mode_name(),
                self.data_file_reader_cache_config.mode_name(),
                self.fused_final_take_config.mode_name(),
                self.direct_winner_projection_config.mode_name(),
                self.fused_output_projection
                    .as_ref()
                    .map(|projection| projection.to_bare_schema().fields.len())
                    .unwrap_or(0),
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
        self.output_schema.clone()
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
        Ok(Arc::new(Self::try_new_with_configs(
            self.dataset.clone(),
            self.indices.clone(),
            self.query.clone(),
            prefilter_source,
            self.direct_residual_config,
            self.fused_final_take_config,
            self.sorted_raw_take_config,
            self.grouped_refinement_config,
            self.grouped_shared_scheduler_config,
            self.data_file_reader_cache_config,
            self.direct_winner_projection_config,
            self.fused_output_projection.clone(),
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
        let direct_residual_config = self.direct_residual_config;
        let sorted_raw_take_config = self.sorted_raw_take_config;
        let grouped_refinement_config = self.grouped_refinement_config;
        let grouped_shared_scheduler_config = self.grouped_shared_scheduler_config;
        let data_file_reader_cache_config = self.data_file_reader_cache_config;
        let direct_winner_projection_config = self.direct_winner_projection_config;
        let fused_output_projection = self.fused_output_projection.clone();
        let output_schema = self.output_schema.clone();
        let stream = stream::once(async move {
            let total_started = Instant::now();
            let result = execute_search(
                dataset,
                indices,
                query,
                mode,
                direct_residual_config,
                sorted_raw_take_config,
                grouped_refinement_config,
                grouped_shared_scheduler_config,
                data_file_reader_cache_config,
                direct_winner_projection_config,
                fused_output_projection,
                output_schema,
                prefilter,
                metrics.clone(),
            )
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
    direct_residual: Time,
    row_id_fetch: Time,
    raw_vector_fetch: Time,
    exact: Time,
    sorted_raw_take: Time,
    grouped_refinement: Time,
    grouped_refinement_helper_wall: Time,
    grouped_refinement_plan: Time,
    grouped_refinement_grouping: Time,
    grouped_refinement_fanout_collect_wall: Time,
    grouped_refinement_row_offset_injection_wall: Time,
    grouped_refinement_scheduler_create_wall: Time,
    grouped_refinement_fragment_open_sum: Time,
    grouped_refinement_fragment_open_max: Time,
    grouped_refinement_fragment_read_sum: Time,
    grouped_refinement_fragment_read_max: Time,
    grouped_refinement_fragment_total_elapsed_sum: Time,
    grouped_refinement_fragment_total_elapsed_max: Time,
    data_file_reader_cache_lookup_sum: Time,
    data_file_reader_cache_get_or_open_sum: Time,
    data_file_reader_cache_coalesced_wait_sum: Time,
    data_file_reader_cache_acquire_sum: Time,
    data_file_reader_cache_physical_open_sum: Time,
    data_file_reader_cache_bind_sum: Time,
    fused_final_take: Time,
    fused_final_take_select: Time,
    fused_final_take_logical_projection: Time,
    fused_final_take_direct_projection: Time,
    fused_final_take_legacy_projection: Time,
    fused_final_take_json_conversion: Time,
    fused_final_take_assembly: Time,
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
    raw_vector_bytes_count: Count,
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
    direct_residual_query_count: Count,
    direct_residual_segment_count: Count,
    direct_residual_document_count: Count,
    legacy_residual_query_count: Count,
    legacy_residual_segment_count: Count,
    direct_residual_small_exact_precedence_count: Count,
    direct_residual_disabled_count: Count,
    direct_residual_explicit_ceiling_count: Count,
    direct_residual_unfiltered_count: Count,
    direct_residual_non_enumerable_count: Count,
    direct_residual_empty_segment_count: Count,
    direct_residual_over_limit_count: Count,
    direct_residual_budget_mismatch_count: Count,
    direct_residual_empty_tokens_count: Count,
    fused_final_take_query_count: Count,
    fused_final_take_direct_projection_query_count: Count,
    fused_final_take_legacy_projection_query_count: Count,
    fused_final_take_candidate_rows_count: Count,
    fused_final_take_output_rows_count: Count,
    fused_final_take_output_bytes_count: Count,
    sorted_raw_take_query_count: Count,
    sorted_raw_take_rows_count: Count,
    grouped_refinement_query_count: Count,
    grouped_refinement_batch_count: Count,
    grouped_refinement_rows_count: Count,
    grouped_refinement_fanout_concurrency_limit_count: Count,
    grouped_refinement_batch_bytes_count: Count,
    grouped_refinement_rows_per_fragment_min_count: Count,
    grouped_refinement_rows_per_fragment_max_count: Count,
    grouped_refinement_scheduler_scoped_iops_count: Count,
    grouped_refinement_scheduler_scoped_requests_count: Count,
    grouped_refinement_scheduler_scoped_bytes_read_count: Count,
    grouped_refinement_scheduler_stats_covered_fragments_count: Count,
    grouped_refinement_fallback_count: Count,
    grouped_shared_scheduler_query_count: Count,
    grouped_shared_scheduler_fragment_count: Count,
    grouped_per_fragment_scheduler_query_count: Count,
    grouped_per_fragment_scheduler_fragment_count: Count,
    grouped_shared_scheduler_fallback_query_count: Count,
    grouped_shared_scheduler_fallback_legacy_fragment_count: Count,
    grouped_shared_scheduler_fallback_nonprimary_fragment_count: Count,
    grouped_shared_scheduler_fallback_unsupported_fragment_count: Count,
    data_file_reader_cache_query_count: Count,
    data_file_reader_cache_eligible_file_count: Count,
    data_file_reader_cache_lookup_file_count: Count,
    data_file_reader_cache_hit_file_count: Count,
    data_file_reader_cache_miss_open_file_count: Count,
    data_file_reader_cache_coalesced_file_count: Count,
    data_file_reader_cache_bypass_file_count: Count,
    data_file_reader_cache_fallback_open_file_count: Count,
    data_file_reader_cache_open_failure_count: Count,
    data_file_reader_cache_fd_budget_rejection_count: Count,
    data_file_reader_cache_resident_start_count: Count,
    data_file_reader_cache_resident_end_count: Count,
    data_file_reader_cache_capacity_count: Count,
    data_file_reader_cache_fd_soft_limit_count: Count,
    data_file_reader_cache_fallback_query_count: Count,
    data_file_reader_cache_partial_fallback_query_count: Count,
    data_file_reader_cache_fallback_legacy_fragment_count: Count,
    data_file_reader_cache_fallback_nonprimary_fragment_count: Count,
    data_file_reader_cache_fallback_nonlocal_fragment_count: Count,
    data_file_reader_cache_fallback_unknown_size_fragment_count: Count,
    data_file_reader_cache_fallback_small_file_fragment_count: Count,
    data_file_reader_cache_fallback_unsupported_fragment_count: Count,
    data_file_reader_cache_fallback_capacity_query_count: Count,
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
            direct_residual: metrics.new_time(DIRECT_RESIDUAL_TIME, partition),
            row_id_fetch: metrics.new_time(ROW_ID_FETCH_TIME, partition),
            raw_vector_fetch: metrics.new_time(RAW_VECTOR_FETCH_TIME, partition),
            exact: metrics.new_time(EXACT_TIME, partition),
            sorted_raw_take: metrics.new_time(SORTED_RAW_TAKE_TIME, partition),
            grouped_refinement: metrics.new_time(GROUPED_REFINEMENT_TIME, partition),
            grouped_refinement_helper_wall: metrics
                .new_time(GROUPED_REFINEMENT_HELPER_WALL_TIME, partition),
            grouped_refinement_plan: metrics.new_time(GROUPED_REFINEMENT_PLAN_TIME, partition),
            grouped_refinement_grouping: metrics
                .new_time(GROUPED_REFINEMENT_GROUPING_TIME, partition),
            grouped_refinement_fanout_collect_wall: metrics
                .new_time(GROUPED_REFINEMENT_FANOUT_COLLECT_WALL_TIME, partition),
            grouped_refinement_row_offset_injection_wall: metrics
                .new_time(GROUPED_REFINEMENT_ROW_OFFSET_INJECTION_WALL_TIME, partition),
            grouped_refinement_scheduler_create_wall: metrics
                .new_time(GROUPED_REFINEMENT_SCHEDULER_CREATE_WALL_TIME, partition),
            grouped_refinement_fragment_open_sum: metrics
                .new_time(GROUPED_REFINEMENT_FRAGMENT_OPEN_SUM_TIME, partition),
            grouped_refinement_fragment_open_max: metrics
                .new_time(GROUPED_REFINEMENT_FRAGMENT_OPEN_MAX_TIME, partition),
            grouped_refinement_fragment_read_sum: metrics
                .new_time(GROUPED_REFINEMENT_FRAGMENT_READ_SUM_TIME, partition),
            grouped_refinement_fragment_read_max: metrics
                .new_time(GROUPED_REFINEMENT_FRAGMENT_READ_MAX_TIME, partition),
            grouped_refinement_fragment_total_elapsed_sum: metrics.new_time(
                GROUPED_REFINEMENT_FRAGMENT_TOTAL_ELAPSED_SUM_TIME,
                partition,
            ),
            grouped_refinement_fragment_total_elapsed_max: metrics.new_time(
                GROUPED_REFINEMENT_FRAGMENT_TOTAL_ELAPSED_MAX_TIME,
                partition,
            ),
            data_file_reader_cache_lookup_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_LOOKUP_SUM_TIME, partition),
            data_file_reader_cache_get_or_open_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_GET_OR_OPEN_SUM_TIME, partition),
            data_file_reader_cache_coalesced_wait_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_COALESCED_WAIT_SUM_TIME, partition),
            data_file_reader_cache_acquire_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_ACQUIRE_SUM_TIME, partition),
            data_file_reader_cache_physical_open_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_PHYSICAL_OPEN_SUM_TIME, partition),
            data_file_reader_cache_bind_sum: metrics
                .new_time(DATA_FILE_READER_CACHE_BIND_SUM_TIME, partition),
            fused_final_take: metrics.new_time(FUSED_FINAL_TAKE_TIME, partition),
            fused_final_take_select: metrics.new_time(FUSED_FINAL_TAKE_SELECT_TIME, partition),
            fused_final_take_logical_projection: metrics
                .new_time(FUSED_FINAL_TAKE_LOGICAL_PROJECTION_TIME, partition),
            fused_final_take_direct_projection: metrics
                .new_time(FUSED_FINAL_TAKE_DIRECT_PROJECTION_TIME, partition),
            fused_final_take_legacy_projection: metrics
                .new_time(FUSED_FINAL_TAKE_LEGACY_PROJECTION_TIME, partition),
            fused_final_take_json_conversion: metrics
                .new_time(FUSED_FINAL_TAKE_JSON_CONVERSION_TIME, partition),
            fused_final_take_assembly: metrics.new_time(FUSED_FINAL_TAKE_ASSEMBLY_TIME, partition),
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
            raw_vector_bytes_count: metrics.new_count(RAW_VECTOR_BYTES_COUNT, partition),
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
            direct_residual_query_count: metrics.new_count(DIRECT_RESIDUAL_QUERY_COUNT, partition),
            direct_residual_segment_count: metrics
                .new_count(DIRECT_RESIDUAL_SEGMENT_COUNT, partition),
            direct_residual_document_count: metrics
                .new_count(DIRECT_RESIDUAL_DOCUMENT_COUNT, partition),
            legacy_residual_query_count: metrics.new_count(LEGACY_RESIDUAL_QUERY_COUNT, partition),
            legacy_residual_segment_count: metrics
                .new_count(LEGACY_RESIDUAL_SEGMENT_COUNT, partition),
            direct_residual_small_exact_precedence_count: metrics
                .new_count(DIRECT_RESIDUAL_SMALL_EXACT_PRECEDENCE_COUNT, partition),
            direct_residual_disabled_count: metrics
                .new_count(DIRECT_RESIDUAL_DISABLED_COUNT, partition),
            direct_residual_explicit_ceiling_count: metrics
                .new_count(DIRECT_RESIDUAL_EXPLICIT_CEILING_COUNT, partition),
            direct_residual_unfiltered_count: metrics
                .new_count(DIRECT_RESIDUAL_UNFILTERED_COUNT, partition),
            direct_residual_non_enumerable_count: metrics
                .new_count(DIRECT_RESIDUAL_NON_ENUMERABLE_COUNT, partition),
            direct_residual_empty_segment_count: metrics
                .new_count(DIRECT_RESIDUAL_EMPTY_SEGMENT_COUNT, partition),
            direct_residual_over_limit_count: metrics
                .new_count(DIRECT_RESIDUAL_OVER_LIMIT_COUNT, partition),
            direct_residual_budget_mismatch_count: metrics
                .new_count(DIRECT_RESIDUAL_BUDGET_MISMATCH_COUNT, partition),
            direct_residual_empty_tokens_count: metrics
                .new_count(DIRECT_RESIDUAL_EMPTY_TOKENS_COUNT, partition),
            fused_final_take_query_count: metrics
                .new_count(FUSED_FINAL_TAKE_QUERY_COUNT, partition),
            fused_final_take_direct_projection_query_count: metrics
                .new_count(FUSED_FINAL_TAKE_DIRECT_PROJECTION_QUERY_COUNT, partition),
            fused_final_take_legacy_projection_query_count: metrics
                .new_count(FUSED_FINAL_TAKE_LEGACY_PROJECTION_QUERY_COUNT, partition),
            fused_final_take_candidate_rows_count: metrics
                .new_count(FUSED_FINAL_TAKE_CANDIDATE_ROWS_COUNT, partition),
            fused_final_take_output_rows_count: metrics
                .new_count(FUSED_FINAL_TAKE_OUTPUT_ROWS_COUNT, partition),
            fused_final_take_output_bytes_count: metrics
                .new_count(FUSED_FINAL_TAKE_OUTPUT_BYTES_COUNT, partition),
            sorted_raw_take_query_count: metrics.new_count(SORTED_RAW_TAKE_QUERY_COUNT, partition),
            sorted_raw_take_rows_count: metrics.new_count(SORTED_RAW_TAKE_ROWS_COUNT, partition),
            grouped_refinement_query_count: metrics
                .new_count(GROUPED_REFINEMENT_QUERY_COUNT, partition),
            grouped_refinement_batch_count: metrics
                .new_count(GROUPED_REFINEMENT_BATCH_COUNT, partition),
            grouped_refinement_rows_count: metrics
                .new_count(GROUPED_REFINEMENT_ROWS_COUNT, partition),
            grouped_refinement_fanout_concurrency_limit_count: metrics
                .new_count(GROUPED_REFINEMENT_FANOUT_CONCURRENCY_LIMIT_COUNT, partition),
            grouped_refinement_batch_bytes_count: metrics
                .new_count(GROUPED_REFINEMENT_BATCH_BYTES_COUNT, partition),
            grouped_refinement_rows_per_fragment_min_count: metrics
                .new_count(GROUPED_REFINEMENT_ROWS_PER_FRAGMENT_MIN_COUNT, partition),
            grouped_refinement_rows_per_fragment_max_count: metrics
                .new_count(GROUPED_REFINEMENT_ROWS_PER_FRAGMENT_MAX_COUNT, partition),
            grouped_refinement_scheduler_scoped_iops_count: metrics
                .new_count(GROUPED_REFINEMENT_SCHEDULER_SCOPED_IOPS_COUNT, partition),
            grouped_refinement_scheduler_scoped_requests_count: metrics.new_count(
                GROUPED_REFINEMENT_SCHEDULER_SCOPED_REQUESTS_COUNT,
                partition,
            ),
            grouped_refinement_scheduler_scoped_bytes_read_count: metrics.new_count(
                GROUPED_REFINEMENT_SCHEDULER_SCOPED_BYTES_READ_COUNT,
                partition,
            ),
            grouped_refinement_scheduler_stats_covered_fragments_count: metrics.new_count(
                GROUPED_REFINEMENT_SCHEDULER_STATS_COVERED_FRAGMENTS_COUNT,
                partition,
            ),
            grouped_refinement_fallback_count: metrics
                .new_count(GROUPED_REFINEMENT_FALLBACK_COUNT, partition),
            grouped_shared_scheduler_query_count: metrics
                .new_count(GROUPED_SHARED_SCHEDULER_QUERY_COUNT, partition),
            grouped_shared_scheduler_fragment_count: metrics
                .new_count(GROUPED_SHARED_SCHEDULER_FRAGMENT_COUNT, partition),
            grouped_per_fragment_scheduler_query_count: metrics
                .new_count(GROUPED_PER_FRAGMENT_SCHEDULER_QUERY_COUNT, partition),
            grouped_per_fragment_scheduler_fragment_count: metrics
                .new_count(GROUPED_PER_FRAGMENT_SCHEDULER_FRAGMENT_COUNT, partition),
            grouped_shared_scheduler_fallback_query_count: metrics
                .new_count(GROUPED_SHARED_SCHEDULER_FALLBACK_QUERY_COUNT, partition),
            grouped_shared_scheduler_fallback_legacy_fragment_count: metrics.new_count(
                GROUPED_SHARED_SCHEDULER_FALLBACK_LEGACY_FRAGMENT_COUNT,
                partition,
            ),
            grouped_shared_scheduler_fallback_nonprimary_fragment_count: metrics.new_count(
                GROUPED_SHARED_SCHEDULER_FALLBACK_NONPRIMARY_FRAGMENT_COUNT,
                partition,
            ),
            grouped_shared_scheduler_fallback_unsupported_fragment_count: metrics.new_count(
                GROUPED_SHARED_SCHEDULER_FALLBACK_UNSUPPORTED_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_query_count: metrics
                .new_count(DATA_FILE_READER_CACHE_QUERY_COUNT, partition),
            data_file_reader_cache_eligible_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_ELIGIBLE_FILE_COUNT, partition),
            data_file_reader_cache_lookup_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_LOOKUP_FILE_COUNT, partition),
            data_file_reader_cache_hit_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_HIT_FILE_COUNT, partition),
            data_file_reader_cache_miss_open_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_MISS_OPEN_FILE_COUNT, partition),
            data_file_reader_cache_coalesced_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_COALESCED_FILE_COUNT, partition),
            data_file_reader_cache_bypass_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_BYPASS_FILE_COUNT, partition),
            data_file_reader_cache_fallback_open_file_count: metrics
                .new_count(DATA_FILE_READER_CACHE_FALLBACK_OPEN_FILE_COUNT, partition),
            data_file_reader_cache_open_failure_count: metrics
                .new_count(DATA_FILE_READER_CACHE_OPEN_FAILURE_COUNT, partition),
            data_file_reader_cache_fd_budget_rejection_count: metrics
                .new_count(DATA_FILE_READER_CACHE_FD_BUDGET_REJECTION_COUNT, partition),
            data_file_reader_cache_resident_start_count: metrics
                .new_count(DATA_FILE_READER_CACHE_RESIDENT_START_COUNT, partition),
            data_file_reader_cache_resident_end_count: metrics
                .new_count(DATA_FILE_READER_CACHE_RESIDENT_END_COUNT, partition),
            data_file_reader_cache_capacity_count: metrics
                .new_count(DATA_FILE_READER_CACHE_CAPACITY_COUNT, partition),
            data_file_reader_cache_fd_soft_limit_count: metrics
                .new_count(DATA_FILE_READER_CACHE_FD_SOFT_LIMIT_COUNT, partition),
            data_file_reader_cache_fallback_query_count: metrics
                .new_count(DATA_FILE_READER_CACHE_FALLBACK_QUERY_COUNT, partition),
            data_file_reader_cache_partial_fallback_query_count: metrics.new_count(
                DATA_FILE_READER_CACHE_PARTIAL_FALLBACK_QUERY_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_legacy_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_LEGACY_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_nonprimary_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_NONPRIMARY_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_nonlocal_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_NONLOCAL_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_unknown_size_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_UNKNOWN_SIZE_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_small_file_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_SMALL_FILE_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_unsupported_fragment_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_UNSUPPORTED_FRAGMENT_COUNT,
                partition,
            ),
            data_file_reader_cache_fallback_capacity_query_count: metrics.new_count(
                DATA_FILE_READER_CACHE_FALLBACK_CAPACITY_QUERY_COUNT,
                partition,
            ),
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

    fn record_direct_residual_decision(&self, decision: DirectResidualDecision) {
        match decision {
            DirectResidualDecision::Direct => self.direct_residual_segment_count.add(1),
            DirectResidualDecision::Disabled => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_disabled_count.add(1);
            }
            DirectResidualDecision::ExplicitProbeCeiling => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_explicit_ceiling_count.add(1);
            }
            DirectResidualDecision::Unfiltered => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_unfiltered_count.add(1);
            }
            DirectResidualDecision::NonEnumerable => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_non_enumerable_count.add(1);
            }
            DirectResidualDecision::EmptySegment => {
                self.direct_residual_empty_segment_count.add(1);
            }
            DirectResidualDecision::OverLimit => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_over_limit_count.add(1);
            }
            DirectResidualDecision::BudgetMismatch => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_budget_mismatch_count.add(1);
            }
            DirectResidualDecision::EmptyTokens => {
                self.legacy_residual_segment_count.add(1);
                self.direct_residual_empty_tokens_count.add(1);
            }
        }
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

fn direct_residual_precheck(
    config: DirectResidualConfig,
    maximum_nprobes: Option<usize>,
    filtered_query: bool,
) -> Option<DirectResidualDecision> {
    if !config.enabled {
        return Some(DirectResidualDecision::Disabled);
    }
    // Even a ceiling equal to the current centroid count is an explicit user
    // contract. The experimental shortcut must never silently ignore it.
    if maximum_nprobes.is_some() {
        return Some(DirectResidualDecision::ExplicitProbeCeiling);
    }
    if !filtered_query {
        return Some(DirectResidualDecision::Unfiltered);
    }
    None
}

fn direct_residual_budget_params(
    requested_candidates: usize,
    eligible_documents: usize,
) -> Option<PlaidSearchParams> {
    // nprobe does not affect candidate truncation. The real candidate plan
    // derives it from the query/mask; one is sufficient to construct the same
    // shared budgets and threshold here.
    let params = plaid_search_params(requested_candidates, eligible_documents, 1);
    let desired_candidates = eligible_documents.min(params.top_k);
    let approximate_retained = eligible_documents.min(params.n_full_scores);
    let residual_scored = (params.n_full_scores / 4)
        .max(params.top_k)
        .min(approximate_retained);
    let final_retained = params.top_k.min(residual_scored);
    (params.centroid_score_threshold.is_none()
        && params.top_k == eligible_documents
        && desired_candidates == eligible_documents
        && approximate_retained == eligible_documents
        && residual_scored == eligible_documents
        && final_retained == eligible_documents)
        .then_some(params)
}

fn direct_residual_count_decision(
    config: DirectResidualConfig,
    requested_candidates: usize,
    enumerable_documents: Option<usize>,
) -> DirectResidualDecision {
    let Some(eligible_documents) = enumerable_documents else {
        return DirectResidualDecision::NonEnumerable;
    };
    if eligible_documents == 0 {
        return DirectResidualDecision::EmptySegment;
    }
    if eligible_documents > config.max_documents {
        return DirectResidualDecision::OverLimit;
    }

    if direct_residual_budget_params(requested_candidates, eligible_documents).is_none() {
        return DirectResidualDecision::BudgetMismatch;
    }
    DirectResidualDecision::Direct
}

fn direct_residual_plan(
    config: DirectResidualConfig,
    query: &Query,
    requested_candidates: usize,
    filtered_query: bool,
    index: &PlaidVectorIndex,
    mask: &RowAddrMask,
) -> DirectResidualPlan {
    if let Some(decision) = direct_residual_precheck(config, query.maximum_nprobes, filtered_query)
    {
        return DirectResidualPlan::Legacy(decision);
    }
    let document_ordinals =
        match index.bounded_eligible_document_ordinals(mask, config.max_documents) {
            BoundedEligibleOrdinals::NonEnumerable => {
                return DirectResidualPlan::Legacy(DirectResidualDecision::NonEnumerable);
            }
            BoundedEligibleOrdinals::OverLimit => {
                return DirectResidualPlan::Legacy(DirectResidualDecision::OverLimit);
            }
            BoundedEligibleOrdinals::WithinLimit(document_ordinals) => document_ordinals,
        };
    match direct_residual_count_decision(
        config,
        requested_candidates,
        Some(document_ordinals.len()),
    ) {
        DirectResidualDecision::EmptySegment => DirectResidualPlan::Empty,
        DirectResidualDecision::Direct => {
            match direct_residual_budget_params(requested_candidates, document_ordinals.len()) {
                Some(params) => DirectResidualPlan::Direct {
                    document_ordinals,
                    params,
                },
                None => DirectResidualPlan::Legacy(DirectResidualDecision::BudgetMismatch),
            }
        }
        decision => DirectResidualPlan::Legacy(decision),
    }
}

fn underfilled_filter_exact_fallback(
    filtered_query: bool,
    ann_candidates: usize,
    top_k: usize,
    eligible_documents: usize,
) -> bool {
    filtered_query && ann_candidates < top_k.min(eligible_documents)
}

fn elapsed_nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn merge_segment_hits(
    candidates: &mut HashMap<u64, f32>,
    hits: impl IntoIterator<Item = lance_plaid::SearchHit>,
) {
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

fn direct_winner_projection_eligible(
    config: DirectWinnerProjectionConfig,
    grouped_physical_batches: bool,
    projection: &ProjectionPlan,
    batch: &RecordBatch,
) -> bool {
    config.enabled
        && grouped_physical_batches
        && projection.has_output_cols()
        && (!projection.must_add_row_offset
            || batch.column_by_name(lance_core::ROW_OFFSET).is_some())
        && projection.is_identity_column_projection()
}

async fn execute_search(
    dataset: Arc<Dataset>,
    indices: Vec<IndexMetadata>,
    query: Query,
    mode: PlaidExecutionMode,
    direct_residual_config: DirectResidualConfig,
    sorted_raw_take_config: SortedRawTakeConfig,
    grouped_refinement_config: GroupedRefinementConfig,
    grouped_shared_scheduler_config: GroupedSharedSchedulerConfig,
    data_file_reader_cache_config: DataFileReaderCacheConfig,
    direct_winner_projection_config: DirectWinnerProjectionConfig,
    fused_output_projection: Option<Projection>,
    output_schema: SchemaRef,
    prefilter: Arc<crate::index::prefilter::DatasetPreFilter>,
    metrics: Arc<PlaidExecMetrics>,
) -> Result<RecordBatch> {
    metrics.record_mode(mode);
    if fused_output_projection.is_some() {
        metrics.fused_final_take_query_count.add(1);
    }
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
        metrics
            .direct_residual_empty_segment_count
            .add(indices.len());
        metrics.candidate.add_duration(candidate_started.elapsed());
        let batch = RecordBatch::new_empty(output_schema.clone());
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
    let mut residual_usage = ResidualQueryUsage::default();
    if small_filter_exact {
        metrics.direct_residual_small_exact_precedence_count.add(1);
    }

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
        let outcome = spawn_cpu(move || {
            let index = index_for_cpu
                .as_any()
                .downcast_ref::<PlaidVectorIndex>()
                .ok_or_else(|| {
                    Error::internal("PLAID index downcast failed on CPU worker".to_string())
                })?;
            let direct_attempt_started = Instant::now();
            let (direct_decision, direct_attempt_nanos) = match direct_residual_plan(
                direct_residual_config,
                &query_settings,
                requested_candidates,
                filtered_query,
                index,
                mask_for_cpu.as_ref(),
            ) {
                DirectResidualPlan::Direct {
                    document_ordinals,
                    params,
                } => {
                    let documents = document_ordinals.len();
                    let n_full_scores = params.n_full_scores;
                    match index
                        .search_quantized_residuals(query_for_cpu.view(), &document_ordinals)?
                    {
                        Some((hits, stats)) => {
                            let attempt_nanos = elapsed_nanos(direct_attempt_started);
                            return Ok::<_, Error>(SegmentSearchOutcome::Direct {
                                hits,
                                stats,
                                documents,
                                n_full_scores,
                                attempt_nanos,
                            });
                        }
                        None => (
                            DirectResidualDecision::EmptyTokens,
                            Some(elapsed_nanos(direct_attempt_started)),
                        ),
                    }
                }
                DirectResidualPlan::Empty => return Ok::<_, Error>(SegmentSearchOutcome::Empty),
                DirectResidualPlan::Legacy(decision) => (decision, None),
            };
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
            Ok::<_, Error>(SegmentSearchOutcome::Legacy {
                hits,
                stats,
                plan,
                direct_decision,
                direct_attempt_nanos,
            })
        })
        .await?;
        let hits = match outcome {
            SegmentSearchOutcome::Direct {
                hits,
                stats,
                documents,
                n_full_scores,
                attempt_nanos,
            } => {
                residual_usage.observe(ResidualSegmentPath::Direct);
                metrics.record_direct_residual_decision(DirectResidualDecision::Direct);
                metrics.direct_residual_document_count.add(documents);
                metrics
                    .direct_residual
                    .add_duration(Duration::from_nanos(attempt_nanos));
                metrics.core_approximate_budget_count.add(n_full_scores);
                metrics.core_residual_budget_count.add(documents);
                metrics.record_core(&stats);
                hits
            }
            SegmentSearchOutcome::Empty => {
                residual_usage.observe(ResidualSegmentPath::Empty);
                metrics.record_direct_residual_decision(DirectResidualDecision::EmptySegment);
                Vec::new()
            }
            SegmentSearchOutcome::Legacy {
                hits,
                stats,
                plan,
                direct_decision,
                direct_attempt_nanos,
            } => {
                residual_usage.observe(ResidualSegmentPath::Legacy);
                if let Some(direct_attempt_nanos) = direct_attempt_nanos {
                    metrics
                        .direct_residual
                        .add_duration(Duration::from_nanos(direct_attempt_nanos));
                }
                metrics.record_direct_residual_decision(direct_decision);
                metrics
                    .core_approximate_budget_count
                    .add(plan.params.n_full_scores);
                metrics.core_residual_budget_count.add(plan.params.top_k);
                metrics.record_eligible_centroid_plan(&plan);
                metrics.record_core(&stats);
                hits
            }
        };
        merge_segment_hits(&mut candidates, hits);
    }
    residual_usage.record_query_metrics(metrics.as_ref());
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
        let batch = RecordBatch::new_empty(output_schema.clone());
        metrics.baseline.record_output(0);
        return Ok(batch);
    }

    if mode == PlaidExecutionMode::IndexOnly && !exact_fallback {
        let mut hits = candidates
            .into_iter()
            .map(|(row_address, score)| ExactHit {
                row_address,
                distance: maxsim_distance(score),
                batch_ordinal: 0,
                batch_index: 0,
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
    // Exact fallback already established physical-address order above.
    if sorted_raw_take_config.enabled && !exact_fallback {
        let sorted_take_started = Instant::now();
        candidates.sort_unstable_by_key(|(row_address, _)| *row_address);
        metrics
            .sorted_raw_take
            .add_duration(sorted_take_started.elapsed());
        metrics.sorted_raw_take_query_count.add(1);
        metrics.sorted_raw_take_rows_count.add(candidates.len());
    }
    metrics.sort.add_duration(sort_started.elapsed());

    let row_addresses = candidates
        .iter()
        .map(|(row_address, _)| *row_address)
        .collect::<Vec<_>>();
    // Exact refinement already performs one address-based read for every raw
    // candidate.  In the opt-in fused mode carry the final/filter data fields
    // through that same read, then select the winning rows in memory.  This
    // removes the second database TakeExec without changing candidate search.
    let raw_projection = fused_output_projection
        .clone()
        .unwrap_or_else(|| dataset.empty_projection())
        .union_column(&query.column, OnMissing::Error)?
        .with_row_id();
    let mut projection = ProjectionRequest::from_schema(raw_projection.to_schema())
        .into_projection_plan(dataset.clone())?;
    // ProjectionRequest is schema based and would otherwise reset non-schema
    // read policy such as BlobHandling.  Preserve the scanner's physical
    // projection contract in the consolidated read.
    projection.physical_projection = raw_projection;
    let projection = Arc::new(projection);

    let addresses_are_sorted = row_addresses.windows(2).all(|pair| pair[0] <= pair[1]);
    let fragment_count = addresses_are_sorted.then(|| {
        1 + row_addresses
            .windows(2)
            .filter(|pair| pair[0] >> 32 != pair[1] >> 32)
            .count()
    });
    // Preserve physical fragment batches and defer logical projection until
    // top-k only when cross-fragment layout work exists. Keep the ordinary
    // TakeBuilder path for a single fragment until benchmarks justify otherwise.
    let grouped_attempt = grouped_refinement_config.enabled
        && fragment_count.is_some_and(|fragment_count| fragment_count > 1);
    let raw_vector_fetch_started = Instant::now();
    // Preserve the historical grouped timer boundary: it starts before
    // TakeBuilder construction and includes a failed specialization attempt.
    let grouped_started = grouped_attempt.then(Instant::now);
    let grouped_read = if grouped_attempt {
        let builder = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            row_addresses.clone(),
            projection.clone(),
        )?;
        match (
            grouped_shared_scheduler_config.enabled,
            data_file_reader_cache_config.enabled,
        ) {
            (false, false) => builder.read_sorted_physical_by_fragment().await?,
            (true, false) => {
                builder
                    .read_sorted_physical_by_fragment_with_shared_scheduler(true)
                    .await?
            }
            (shared_scheduler_enabled, true) => {
                builder
                    .read_sorted_physical_by_fragment_with_options(shared_scheduler_enabled, true)
                    .await?
            }
        }
    } else {
        None
    };
    if let Some(grouped_started) = grouped_started {
        metrics
            .grouped_refinement
            .add_duration(grouped_started.elapsed());
    }
    let (batches, grouped_physical_batches) = if let Some(grouped_read) = grouped_read {
        let grouped_stats = grouped_read.stats;
        metrics
            .grouped_refinement_helper_wall
            .add_duration(Duration::from_nanos(grouped_stats.parent_wall_nanos));
        metrics
            .grouped_refinement_plan
            .add_duration(Duration::from_nanos(grouped_stats.plan_nanos));
        metrics
            .grouped_refinement_grouping
            .add_duration(Duration::from_nanos(grouped_stats.grouping_nanos));
        metrics
            .grouped_refinement_fanout_collect_wall
            .add_duration(Duration::from_nanos(
                grouped_stats.fanout_collect_wall_nanos,
            ));
        metrics
            .grouped_refinement_row_offset_injection_wall
            .add_duration(Duration::from_nanos(
                grouped_stats.row_offset_injection_wall_nanos,
            ));
        metrics
            .grouped_refinement_scheduler_create_wall
            .add_duration(Duration::from_nanos(
                grouped_stats.scheduler_create_wall_nanos,
            ));
        metrics
            .grouped_refinement_fragment_open_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.fragment_open_aggregate_nanos,
            ));
        metrics
            .grouped_refinement_fragment_open_max
            .add_duration(Duration::from_nanos(grouped_stats.fragment_open_max_nanos));
        metrics
            .grouped_refinement_fragment_read_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.fragment_read_aggregate_nanos,
            ));
        metrics
            .grouped_refinement_fragment_read_max
            .add_duration(Duration::from_nanos(grouped_stats.fragment_read_max_nanos));
        metrics
            .grouped_refinement_fragment_total_elapsed_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.fragment_total_elapsed_aggregate_nanos,
            ));
        metrics
            .grouped_refinement_fragment_total_elapsed_max
            .add_duration(Duration::from_nanos(
                grouped_stats.fragment_total_elapsed_max_nanos,
            ));
        metrics.grouped_refinement_query_count.add(1);
        metrics
            .grouped_refinement_batch_count
            .add(grouped_stats.fragments);
        metrics
            .grouped_refinement_rows_count
            .add(grouped_stats.rows);
        metrics
            .grouped_refinement_fanout_concurrency_limit_count
            .add(grouped_stats.fanout_concurrency_limit);
        metrics
            .grouped_refinement_batch_bytes_count
            .add(grouped_stats.batch_bytes);
        metrics
            .grouped_refinement_rows_per_fragment_min_count
            .add(grouped_stats.rows_per_fragment_min);
        metrics
            .grouped_refinement_rows_per_fragment_max_count
            .add(grouped_stats.rows_per_fragment_max);
        metrics
            .grouped_refinement_scheduler_scoped_iops_count
            .add(usize::try_from(grouped_stats.scheduler_scoped_iops).unwrap_or(usize::MAX));
        metrics
            .grouped_refinement_scheduler_scoped_requests_count
            .add(usize::try_from(grouped_stats.scheduler_scoped_requests).unwrap_or(usize::MAX));
        metrics
            .grouped_refinement_scheduler_scoped_bytes_read_count
            .add(usize::try_from(grouped_stats.scheduler_scoped_bytes_read).unwrap_or(usize::MAX));
        metrics
            .grouped_refinement_scheduler_stats_covered_fragments_count
            .add(grouped_stats.scheduler_stats_covered_fragments);
        metrics
            .grouped_shared_scheduler_query_count
            .add(grouped_stats.shared_scheduler_queries);
        metrics
            .grouped_shared_scheduler_fragment_count
            .add(grouped_stats.shared_scheduler_fragments);
        metrics
            .grouped_per_fragment_scheduler_query_count
            .add(grouped_stats.per_fragment_scheduler_queries);
        metrics
            .grouped_per_fragment_scheduler_fragment_count
            .add(grouped_stats.per_fragment_scheduler_fragments);
        metrics
            .grouped_shared_scheduler_fallback_query_count
            .add(grouped_stats.shared_scheduler_fallback_queries);
        metrics
            .grouped_shared_scheduler_fallback_legacy_fragment_count
            .add(grouped_stats.shared_scheduler_fallback_legacy_fragments);
        metrics
            .grouped_shared_scheduler_fallback_nonprimary_fragment_count
            .add(grouped_stats.shared_scheduler_fallback_nonprimary_fragments);
        metrics
            .grouped_shared_scheduler_fallback_unsupported_fragment_count
            .add(grouped_stats.shared_scheduler_fallback_unsupported_fragments);
        metrics
            .data_file_reader_cache_lookup_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.reader_cache_lookup_nanos,
            ));
        metrics
            .data_file_reader_cache_get_or_open_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.reader_cache_get_or_open_nanos,
            ));
        metrics
            .data_file_reader_cache_coalesced_wait_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.reader_cache_coalesced_wait_nanos,
            ));
        metrics
            .data_file_reader_cache_acquire_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.reader_cache_acquire_nanos,
            ));
        metrics
            .data_file_reader_cache_physical_open_sum
            .add_duration(Duration::from_nanos(
                grouped_stats.reader_cache_physical_open_nanos,
            ));
        metrics
            .data_file_reader_cache_bind_sum
            .add_duration(Duration::from_nanos(grouped_stats.reader_cache_bind_nanos));
        metrics
            .data_file_reader_cache_query_count
            .add(grouped_stats.reader_cache_queries);
        metrics
            .data_file_reader_cache_eligible_file_count
            .add(usize::try_from(grouped_stats.reader_cache_eligible_files).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_lookup_file_count
            .add(usize::try_from(grouped_stats.reader_cache_lookup_files).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_hit_file_count
            .add(usize::try_from(grouped_stats.reader_cache_hit_files).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_miss_open_file_count
            .add(usize::try_from(grouped_stats.reader_cache_miss_open_files).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_coalesced_file_count
            .add(usize::try_from(grouped_stats.reader_cache_coalesced_files).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_bypass_file_count
            .add(usize::try_from(grouped_stats.reader_cache_bypass_files).unwrap_or(usize::MAX));
        metrics.data_file_reader_cache_fallback_open_file_count.add(
            usize::try_from(grouped_stats.reader_cache_fallback_open_files).unwrap_or(usize::MAX),
        );
        metrics
            .data_file_reader_cache_open_failure_count
            .add(usize::try_from(grouped_stats.reader_cache_open_failures).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_fd_budget_rejection_count
            .add(
                usize::try_from(grouped_stats.reader_cache_fd_budget_rejections)
                    .unwrap_or(usize::MAX),
            );
        metrics.data_file_reader_cache_resident_start_count.add(
            usize::try_from(grouped_stats.reader_cache_resident_entries_start_approx)
                .unwrap_or(usize::MAX),
        );
        metrics.data_file_reader_cache_resident_end_count.add(
            usize::try_from(grouped_stats.reader_cache_resident_entries_end_approx)
                .unwrap_or(usize::MAX),
        );
        metrics
            .data_file_reader_cache_capacity_count
            .add(usize::try_from(grouped_stats.reader_cache_capacity).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_fd_soft_limit_count
            .add(usize::try_from(grouped_stats.reader_cache_fd_soft_limit).unwrap_or(usize::MAX));
        metrics
            .data_file_reader_cache_fallback_query_count
            .add(grouped_stats.reader_cache_fallback_queries);
        metrics
            .data_file_reader_cache_partial_fallback_query_count
            .add(grouped_stats.reader_cache_partial_fallback_queries);
        metrics
            .data_file_reader_cache_fallback_legacy_fragment_count
            .add(grouped_stats.reader_cache_fallback_legacy_fragments);
        metrics
            .data_file_reader_cache_fallback_nonprimary_fragment_count
            .add(grouped_stats.reader_cache_fallback_nonprimary_fragments);
        metrics
            .data_file_reader_cache_fallback_nonlocal_fragment_count
            .add(grouped_stats.reader_cache_fallback_nonlocal_fragments);
        metrics
            .data_file_reader_cache_fallback_unknown_size_fragment_count
            .add(grouped_stats.reader_cache_fallback_unknown_size_fragments);
        metrics
            .data_file_reader_cache_fallback_small_file_fragment_count
            .add(grouped_stats.reader_cache_fallback_small_file_fragments);
        metrics
            .data_file_reader_cache_fallback_unsupported_fragment_count
            .add(grouped_stats.reader_cache_fallback_unsupported_fragments);
        metrics
            .data_file_reader_cache_fallback_capacity_query_count
            .add(grouped_stats.reader_cache_fallback_capacity_queries);
        (grouped_read.batches, true)
    } else {
        if grouped_refinement_config.enabled {
            metrics.grouped_refinement_fallback_count.add(1);
        }
        (
            vec![
                TakeBuilder::try_new_from_addresses(
                    dataset.clone(),
                    row_addresses.clone(),
                    projection.clone(),
                )?
                .execute()
                .await?,
            ],
            false,
        )
    };
    metrics
        .raw_vector_fetch
        .add_duration(raw_vector_fetch_started.elapsed());
    let raw_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let raw_bytes = batches
        .iter()
        .map(RecordBatch::get_array_memory_size)
        .sum::<usize>();
    metrics.raw_vector_rows_count.add(raw_rows);
    metrics.raw_vector_bytes_count.add(raw_bytes);
    if fused_output_projection.is_some() {
        metrics.fused_final_take_candidate_rows_count.add(raw_rows);
    }
    if raw_rows != row_addresses.len() {
        return Err(Error::internal(format!(
            "PLAID raw-vector take returned {} rows for {} candidates",
            raw_rows,
            row_addresses.len()
        )));
    }

    let mut result_row_ids = Vec::with_capacity(raw_rows);
    for batch in &batches {
        result_row_ids.extend_from_slice(
            batch
                .column_by_name(ROW_ID)
                .ok_or_else(|| {
                    Error::internal("PLAID address take did not return _rowid".to_string())
                })?
                .as_primitive::<UInt64Type>()
                .values(),
        );
    }
    let exact_started = Instant::now();
    let column = query.column.clone();
    let query_for_cpu = query_tokens.clone();
    let scoring_batches = batches.clone();
    let mut hits =
        spawn_cpu(move || exact_scores_batches(&scoring_batches, &column, query_for_cpu.view()))
            .await?;
    metrics.exact.add_duration(exact_started.elapsed());
    for (hit, row_id) in hits.iter_mut().zip(result_row_ids) {
        hit.row_address = row_id;
    }

    let sort_started = Instant::now();
    rank_hits(&mut hits, &query);
    metrics.sort.add_duration(sort_started.elapsed());

    let output_batch = if fused_output_projection.is_some() {
        let fused_started = Instant::now();
        let output_batch = if hits.is_empty() {
            RecordBatch::new_empty(output_schema.clone())
        } else {
            let select_started = Instant::now();
            let selected = select_fused_candidate_rows(&hits, &batches)?;
            metrics
                .fused_final_take_select
                .add_duration(select_started.elapsed());
            // Grouped take returns physical batches. Apply the consolidated
            // logical projection and JSON conversion only to selected top-k
            // rows. The fallback batch already has ordinary TakeBuilder
            // projection semantics and must not be projected twice.
            let selected = if grouped_physical_batches {
                let projection_started = Instant::now();
                let selected = if direct_winner_projection_eligible(
                    direct_winner_projection_config,
                    grouped_physical_batches,
                    projection.as_ref(),
                    &selected,
                ) {
                    metrics
                        .fused_final_take_direct_projection_query_count
                        .add(1);
                    let direct_started = Instant::now();
                    let selected = projection.project_batch_direct(selected)?;
                    metrics
                        .fused_final_take_direct_projection
                        .add_duration(direct_started.elapsed());
                    selected
                } else {
                    metrics
                        .fused_final_take_legacy_projection_query_count
                        .add(1);
                    let legacy_started = Instant::now();
                    let selected = projection.project_batch(selected).await?;
                    metrics
                        .fused_final_take_legacy_projection
                        .add_duration(legacy_started.elapsed());
                    selected
                };
                metrics
                    .fused_final_take_logical_projection
                    .add_duration(projection_started.elapsed());
                let conversion_started = Instant::now();
                let selected = convert_lance_json_to_arrow(&selected)?;
                metrics
                    .fused_final_take_json_conversion
                    .add_duration(conversion_started.elapsed());
                selected
            } else {
                selected
            };
            let assembly_started = Instant::now();
            let output = fused_selected_hits_to_batch(&hits, &selected, output_schema)?;
            metrics
                .fused_final_take_assembly
                .add_duration(assembly_started.elapsed());
            output
        };
        metrics
            .fused_final_take
            .add_duration(fused_started.elapsed());
        metrics
            .fused_final_take_output_rows_count
            .add(output_batch.num_rows());
        metrics
            .fused_final_take_output_bytes_count
            .add(output_batch.get_array_memory_size());
        output_batch
    } else {
        hits_to_batch(&hits)?
    };
    metrics.baseline.record_output(output_batch.num_rows());
    Ok(output_batch)
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

#[cfg(test)]
fn fused_hits_to_batch(
    hits: &[ExactHit],
    candidate_batch: &RecordBatch,
    output_schema: SchemaRef,
) -> Result<RecordBatch> {
    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(output_schema));
    }
    let selected = select_fused_candidate_rows(hits, std::slice::from_ref(candidate_batch))?;
    fused_selected_hits_to_batch(hits, &selected, output_schema)
}

fn select_fused_candidate_rows(
    hits: &[ExactHit],
    candidate_batches: &[RecordBatch],
) -> Result<RecordBatch> {
    let first_batch = candidate_batches.first().ok_or_else(|| {
        Error::internal("PLAID fused final take received no candidate batches".to_string())
    })?;
    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(first_batch.schema()));
    }
    let indices = hits
        .iter()
        .map(|hit| {
            let batch = candidate_batches.get(hit.batch_ordinal).ok_or_else(|| {
                Error::internal(format!(
                    "PLAID fused final take batch ordinal {} exceeds {} candidate batches",
                    hit.batch_ordinal,
                    candidate_batches.len()
                ))
            })?;
            if hit.batch_index >= batch.num_rows() {
                return Err(Error::internal(format!(
                    "PLAID fused final take row {} exceeds batch {} length {}",
                    hit.batch_index,
                    hit.batch_ordinal,
                    batch.num_rows()
                )));
            }
            Ok((hit.batch_ordinal, hit.batch_index))
        })
        .collect::<Result<Vec<_>>>()?;
    // Interleave only the final top-k rows in rank order. Candidate-sized
    // nested arrays are never concatenated across fragments.
    interleave_batches(candidate_batches, &indices).map_err(Error::from)
}

fn fused_selected_hits_to_batch(
    hits: &[ExactHit],
    selected: &RecordBatch,
    output_schema: SchemaRef,
) -> Result<RecordBatch> {
    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(output_schema));
    }
    // The raw projection is a union of vector, filter, and final fields.  For
    // nested columns that can make the selected top-level StructArray wider
    // than the final output (for example payload.{mv,lang} vs payload.lang).
    // Apply the final nested projection before assembling the KNN metadata.
    let selected_output_schema = ArrowSchema::new(
        output_schema
            .fields()
            .iter()
            .filter(|field| field.name() != DIST_COL && field.name() != ROW_ID)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let selected = selected.project_by_schema(&selected_output_schema)?;
    let distances: ArrayRef = Arc::new(Float32Array::from(
        hits.iter().map(|hit| hit.distance).collect::<Vec<_>>(),
    ));
    let row_ids: ArrayRef = Arc::new(UInt64Array::from(
        hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>(),
    ));
    let columns = output_schema
        .fields()
        .iter()
        .map(|field| match field.name().as_str() {
            DIST_COL => Ok(distances.clone()),
            ROW_ID => Ok(row_ids.clone()),
            name => selected.column_by_name(name).cloned().ok_or_else(|| {
                Error::internal(format!(
                    "PLAID fused final take batch is missing output field {name}"
                ))
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(output_schema, columns)?)
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
    batch_index: usize,
    batch_ordinal: usize,
}

#[cfg(test)]
fn exact_scores(
    batch: &RecordBatch,
    column: &str,
    query: ArrayView2<'_, f32>,
) -> Result<Vec<ExactHit>> {
    exact_scores_batches(std::slice::from_ref(batch), column, query)
}

fn exact_scores_batches(
    batches: &[RecordBatch],
    column: &str,
    query: ArrayView2<'_, f32>,
) -> Result<Vec<ExactHit>> {
    let mut hits = Vec::with_capacity(batches.iter().map(RecordBatch::num_rows).sum());
    // Validate the query once per database query, not once per fragment.
    let scorer = ExactMaxsimScorer::try_new(query)?;
    for (batch_ordinal, batch) in batches.iter().enumerate() {
        let documents = batch
            .column_by_qualified_name(column)
            .ok_or_else(|| Error::internal(format!("PLAID raw batch is missing {column}")))?
            .as_list::<i32>();
        for batch_index in 0..documents.len() {
            if documents.is_null(batch_index) {
                return Err(Error::internal(
                    "PLAID candidate unexpectedly resolved to a null document".to_string(),
                ));
            }
            let document = documents.value(batch_index);
            let tokens = document.as_fixed_size_list();
            let score = scorer.score(tokens)?;
            hits.push(ExactHit {
                row_address: 0,
                distance: maxsim_distance(score),
                batch_index,
                batch_ordinal,
            });
        }
    }
    Ok(hits)
}

/// A row-major, finite multi-vector query validated once for a batch of exact
/// refinement candidates.
///
/// `query_to_array` already establishes these invariants for a PLAID query.
/// Keeping the checks at this boundary makes `exact_scores` robust without
/// repeating them for every query-token/document-token pair.
#[derive(Debug)]
struct ExactMaxsimScorer<'a> {
    query_values: &'a [f32],
    dimension: usize,
}

impl<'a> ExactMaxsimScorer<'a> {
    fn try_new(query: ArrayView2<'a, f32>) -> Result<Self> {
        let dimension = query.ncols();
        if dimension == 0 {
            return Err(Error::invalid_input(
                "PLAID exact query dimension must be positive".to_string(),
            ));
        }
        let query_values = query.to_slice().ok_or_else(|| {
            Error::invalid_input(
                "PLAID exact query matrix must be row-major contiguous".to_string(),
            )
        })?;
        if query_values.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid_input(
                "PLAID exact query contains non-finite values".to_string(),
            ));
        }
        Ok(Self {
            query_values,
            dimension,
        })
    }

    fn score(&self, document: &arrow_array::FixedSizeListArray) -> Result<f32> {
        // Preserve the established multi-vector convention: an empty document
        // has an undefined MaxSim score, represented as NaN.
        if document.is_empty() {
            return Ok(f32::NAN);
        }

        let document_dimension = usize::try_from(document.value_length()).map_err(|_| {
            Error::invalid_input("PLAID exact document dimension does not fit usize".to_string())
        })?;
        if document_dimension != self.dimension {
            return Err(Error::invalid_input(format!(
                "PLAID exact document dimension {document_dimension} does not match query dimension {}",
                self.dimension
            )));
        }
        if document.null_count() != 0 {
            return Err(Error::invalid_input(
                "PLAID exact refinement does not support null token vectors".to_string(),
            ));
        }

        // FixedSizeList stores every token in one contiguous primitive child.
        // Validate its null/finite invariants once per document, then slice it
        // directly instead of allocating an Arrow value for every token.
        let document_values = document
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
                Error::invalid_input(
                    "PLAID exact refinement requires Float32 document token values".to_string(),
                )
            })?;
        if document_values.null_count() != 0 {
            return Err(Error::invalid_input(
                "PLAID exact refinement does not support null token values".to_string(),
            ));
        }
        let document_values = document_values.values();
        if document_values.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid_input(
                "PLAID exact document contains non-finite values".to_string(),
            ));
        }
        if document_values.len()
            != document.len().checked_mul(self.dimension).ok_or_else(|| {
                Error::invalid_input("PLAID exact document shape overflow".to_string())
            })?
        {
            return Err(Error::invalid_input(
                "PLAID exact document values do not match its token shape".to_string(),
            ));
        }

        let mut score = 0.0_f32;
        for query_token in self.query_values.chunks_exact(self.dimension) {
            let mut maximum = f32::NEG_INFINITY;
            for document_token in document_values.chunks_exact(self.dimension) {
                // Runtime dispatch selects AVX-512 on capable CPUs and the
                // portable SIMD-friendly implementation everywhere else.
                // Its FMA/reduction order may differ by a few ULPs from the old
                // scalar iterator; exact refinement guarantees numerical, not
                // bitwise, equivalence.
                maximum = maximum.max(dot_f32(query_token, document_token));
            }
            score += maximum;
        }
        Ok(score)
    }
}

#[cfg(test)]
fn exact_maxsim(
    query: ArrayView2<'_, f32>,
    document: &arrow_array::FixedSizeListArray,
) -> Result<f32> {
    ExactMaxsimScorer::try_new(query)?.score(document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::Float32Type;
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float32Array, Int32Array, ListArray, RecordBatchIterator,
        StringArray, StructArray,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer};
    use arrow_schema::{DataType, Field};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::utils::address::RowAddress;
    use lance_linalg::distance::{DistanceType, multivec_distance};
    use ndarray::{Array2, ArrayView1, array};

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

    #[test]
    fn direct_residual_config_values_are_reproducible_and_fail_closed() {
        assert_eq!(
            DirectResidualConfig::from_values(None, None).unwrap(),
            DirectResidualConfig::default()
        );
        for enabled in ["1", "true", "YES", "on"] {
            assert_eq!(
                DirectResidualConfig::from_values(Some(enabled), Some("64")).unwrap(),
                DirectResidualConfig {
                    enabled: true,
                    max_documents: 64,
                }
            );
        }
        for disabled in ["0", "false", "No", "OFF"] {
            assert_eq!(
                DirectResidualConfig::from_values(Some(disabled), Some("0")).unwrap(),
                DirectResidualConfig {
                    enabled: false,
                    max_documents: 0,
                }
            );
        }
        for (enabled, maximum) in [(Some("maybe"), None), (None, Some("-1"))] {
            let error = DirectResidualConfig::from_values(enabled, maximum).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("invalid LANCE_PLAID_DIRECT_RESIDUAL")
            );
        }
    }

    #[test]
    fn take_optimization_configs_default_off_and_fail_closed() {
        assert_eq!(
            FusedFinalTakeConfig::from_value(None).unwrap(),
            FusedFinalTakeConfig::default()
        );
        assert_eq!(
            SortedRawTakeConfig::from_value(None).unwrap(),
            SortedRawTakeConfig::default()
        );
        assert_eq!(
            GroupedRefinementConfig::from_value(None).unwrap(),
            GroupedRefinementConfig::default()
        );
        assert_eq!(
            GroupedSharedSchedulerConfig::from_value(None).unwrap(),
            GroupedSharedSchedulerConfig::default()
        );
        assert_eq!(
            DataFileReaderCacheConfig::from_value(None).unwrap(),
            DataFileReaderCacheConfig::default()
        );
        assert_eq!(
            DirectWinnerProjectionConfig::from_value(None).unwrap(),
            DirectWinnerProjectionConfig::default()
        );
        for enabled in ["1", "true", "YES", "on"] {
            assert_eq!(
                FusedFinalTakeConfig::from_value(Some(enabled)).unwrap(),
                FusedFinalTakeConfig { enabled: true }
            );
            assert_eq!(
                SortedRawTakeConfig::from_value(Some(enabled)).unwrap(),
                SortedRawTakeConfig { enabled: true }
            );
            assert_eq!(
                GroupedRefinementConfig::from_value(Some(enabled)).unwrap(),
                GroupedRefinementConfig { enabled: true }
            );
            assert_eq!(
                GroupedSharedSchedulerConfig::from_value(Some(enabled)).unwrap(),
                GroupedSharedSchedulerConfig { enabled: true }
            );
            assert_eq!(
                DataFileReaderCacheConfig::from_value(Some(enabled)).unwrap(),
                DataFileReaderCacheConfig { enabled: true }
            );
            assert_eq!(
                DirectWinnerProjectionConfig::from_value(Some(enabled)).unwrap(),
                DirectWinnerProjectionConfig { enabled: true }
            );
        }
        for disabled in ["0", "false", "No", "OFF"] {
            assert_eq!(
                FusedFinalTakeConfig::from_value(Some(disabled)).unwrap(),
                FusedFinalTakeConfig { enabled: false }
            );
            assert_eq!(
                SortedRawTakeConfig::from_value(Some(disabled)).unwrap(),
                SortedRawTakeConfig { enabled: false }
            );
            assert_eq!(
                GroupedRefinementConfig::from_value(Some(disabled)).unwrap(),
                GroupedRefinementConfig { enabled: false }
            );
            assert_eq!(
                GroupedSharedSchedulerConfig::from_value(Some(disabled)).unwrap(),
                GroupedSharedSchedulerConfig { enabled: false }
            );
            assert_eq!(
                DataFileReaderCacheConfig::from_value(Some(disabled)).unwrap(),
                DataFileReaderCacheConfig { enabled: false }
            );
            assert_eq!(
                DirectWinnerProjectionConfig::from_value(Some(disabled)).unwrap(),
                DirectWinnerProjectionConfig { enabled: false }
            );
        }
        assert!(
            FusedFinalTakeConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(FUSED_FINAL_TAKE_ENABLED_ENV)
        );
        assert!(
            SortedRawTakeConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(SORTED_RAW_TAKE_ENABLED_ENV)
        );
        assert!(
            GroupedRefinementConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(GROUPED_REFINEMENT_ENABLED_ENV)
        );
        assert!(
            GroupedSharedSchedulerConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(GROUPED_SHARED_SCHEDULER_ENABLED_ENV)
        );
        assert!(
            DataFileReaderCacheConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(DATA_FILE_READER_CACHE_ENABLED_ENV)
        );
        assert!(
            DirectWinnerProjectionConfig::from_value(Some("maybe"))
                .unwrap_err()
                .to_string()
                .contains(DIRECT_WINNER_PROJECTION_ENABLED_ENV)
        );
    }

    #[test]
    fn fused_hits_select_candidate_rows_and_preserve_ranked_metadata() {
        let candidate_batch = RecordBatch::try_from_iter([
            (
                "doc_id",
                Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            ),
            (
                ROW_ID,
                Arc::new(UInt64Array::from(vec![100, 200, 300])) as ArrayRef,
            ),
        ])
        .unwrap();
        let output_schema = Arc::new(ArrowSchema::new(vec![
            Arc::new(Field::new(DIST_COL, DataType::Float32, false)),
            Arc::new(Field::new(ROW_ID, DataType::UInt64, false)),
            Arc::new(Field::new("doc_id", DataType::Int32, false)),
        ]));
        let hits = vec![
            ExactHit {
                row_address: 3_000,
                distance: -3.0,
                batch_ordinal: 0,
                batch_index: 2,
            },
            ExactHit {
                row_address: 1_000,
                distance: -1.0,
                batch_ordinal: 0,
                batch_index: 0,
            },
        ];
        let actual = fused_hits_to_batch(&hits, &candidate_batch, output_schema.clone()).unwrap();
        assert_eq!(actual.schema(), output_schema);
        assert_eq!(
            actual
                .column_by_name(DIST_COL)
                .unwrap()
                .as_primitive::<Float32Type>()
                .values(),
            &[-3.0, -1.0]
        );
        assert_eq!(
            actual
                .column_by_name(ROW_ID)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values(),
            &[3_000, 1_000]
        );
        assert_eq!(
            actual
                .column_by_name("doc_id")
                .unwrap()
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[30, 10]
        );

        let empty = fused_hits_to_batch(&[], &candidate_batch, actual.schema()).unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_eq!(empty.schema(), actual.schema());
    }

    #[test]
    fn fused_hits_use_batch_ordinal_and_reject_invalid_locations() {
        let batches = [
            RecordBatch::try_from_iter([
                (
                    "doc_id",
                    Arc::new(Int32Array::from(vec![10, 11])) as ArrayRef,
                ),
                (
                    ROW_ID,
                    Arc::new(UInt64Array::from(vec![100, 110])) as ArrayRef,
                ),
            ])
            .unwrap(),
            RecordBatch::try_from_iter([
                (
                    "doc_id",
                    Arc::new(Int32Array::from(vec![20, 21])) as ArrayRef,
                ),
                (
                    ROW_ID,
                    Arc::new(UInt64Array::from(vec![200, 210])) as ArrayRef,
                ),
            ])
            .unwrap(),
        ];
        let output_schema = Arc::new(ArrowSchema::new(vec![
            Arc::new(Field::new(DIST_COL, DataType::Float32, false)),
            Arc::new(Field::new(ROW_ID, DataType::UInt64, false)),
            Arc::new(Field::new("doc_id", DataType::Int32, false)),
        ]));
        let hits = [
            ExactHit {
                row_address: 2_100,
                distance: -3.0,
                batch_ordinal: 1,
                batch_index: 1,
            },
            ExactHit {
                row_address: 1_000,
                distance: -2.0,
                batch_ordinal: 0,
                batch_index: 0,
            },
            ExactHit {
                row_address: 2_000,
                distance: -1.0,
                batch_ordinal: 1,
                batch_index: 0,
            },
        ];
        let selected = select_fused_candidate_rows(&hits, &batches).unwrap();
        let actual = fused_selected_hits_to_batch(&hits, &selected, output_schema).unwrap();
        assert_eq!(
            actual["doc_id"]
                .as_primitive::<arrow_array::types::Int32Type>()
                .values(),
            &[21, 10, 20]
        );
        assert_eq!(
            actual[ROW_ID].as_primitive::<UInt64Type>().values(),
            &[2_100, 1_000, 2_000]
        );

        let mut invalid = hits;
        invalid[0].batch_ordinal = 2;
        assert!(
            select_fused_candidate_rows(&invalid, &batches)
                .unwrap_err()
                .to_string()
                .contains("batch ordinal")
        );
        invalid[0].batch_ordinal = 1;
        invalid[0].batch_index = 2;
        assert!(
            select_fused_candidate_rows(&invalid, &batches)
                .unwrap_err()
                .to_string()
                .contains("exceeds batch")
        );
    }

    #[test]
    fn fused_hits_apply_nested_output_projection() {
        let mv_field = Arc::new(Field::new("mv", DataType::Int32, false));
        let lang_field = Arc::new(Field::new("lang", DataType::Utf8, false));
        let payload = StructArray::from(vec![
            (
                mv_field,
                Arc::new(Int32Array::from(vec![10, 20])) as ArrayRef,
            ),
            (
                lang_field.clone(),
                Arc::new(StringArray::from(vec!["en", "zh"])) as ArrayRef,
            ),
        ]);
        let candidate_batch = RecordBatch::try_from_iter([
            ("payload", Arc::new(payload) as ArrayRef),
            (
                ROW_ID,
                Arc::new(UInt64Array::from(vec![100, 200])) as ArrayRef,
            ),
        ])
        .unwrap();
        let output_schema = Arc::new(ArrowSchema::new(vec![
            Arc::new(Field::new(DIST_COL, DataType::Float32, false)),
            Arc::new(Field::new(ROW_ID, DataType::UInt64, false)),
            Arc::new(Field::new(
                "payload",
                DataType::Struct(vec![lang_field].into()),
                false,
            )),
        ]));
        let hits = [ExactHit {
            row_address: 2_000,
            distance: -2.0,
            batch_ordinal: 0,
            batch_index: 1,
        }];
        let actual = fused_hits_to_batch(&hits, &candidate_batch, output_schema.clone()).unwrap();
        assert_eq!(actual.schema(), output_schema);
        let payload = actual.column_by_name("payload").unwrap().as_struct();
        assert_eq!(payload.num_columns(), 1);
        assert_eq!(payload.fields()[0].name(), "lang");
        assert_eq!(payload.column(0).as_string::<i32>().value(0), "zh");
    }

    #[test]
    fn direct_residual_policy_covers_every_gate_and_hard_cap_boundary() {
        let default = DirectResidualConfig::default();
        assert_eq!(
            direct_residual_precheck(
                DirectResidualConfig {
                    enabled: false,
                    ..default
                },
                None,
                true,
            ),
            Some(DirectResidualDecision::Disabled)
        );
        assert_eq!(
            direct_residual_precheck(default, Some(1), true),
            Some(DirectResidualDecision::ExplicitProbeCeiling)
        );
        assert_eq!(
            direct_residual_precheck(default, None, false),
            Some(DirectResidualDecision::Unfiltered)
        );
        assert_eq!(direct_residual_precheck(default, None, true), None);

        assert_eq!(
            direct_residual_count_decision(default, 10, None),
            DirectResidualDecision::NonEnumerable
        );
        assert_eq!(
            direct_residual_count_decision(default, 10, Some(0)),
            DirectResidualDecision::EmptySegment
        );
        assert_eq!(
            direct_residual_count_decision(default, 10, Some(1024)),
            DirectResidualDecision::Direct
        );
        assert_eq!(
            direct_residual_count_decision(default, 10, Some(1025)),
            DirectResidualDecision::OverLimit
        );

        let cap64 = DirectResidualConfig {
            enabled: true,
            max_documents: 64,
        };
        assert_eq!(
            direct_residual_count_decision(cap64, 5_000, Some(64)),
            DirectResidualDecision::Direct
        );
        assert_eq!(
            direct_residual_count_decision(cap64, 5_000, Some(65)),
            DirectResidualDecision::OverLimit
        );
        assert_eq!(
            direct_residual_count_decision(
                DirectResidualConfig {
                    enabled: true,
                    max_documents: 0,
                },
                5_000,
                Some(1),
            ),
            DirectResidualDecision::OverLimit
        );
        assert_eq!(
            direct_residual_count_decision(
                DirectResidualConfig {
                    enabled: true,
                    max_documents: 2_048,
                },
                10,
                Some(1_025),
            ),
            DirectResidualDecision::BudgetMismatch
        );
    }

    #[test]
    fn empty_segment_metrics_do_not_claim_direct_or_legacy_query_work() {
        fn assert_usage(
            paths: &[ResidualSegmentPath],
            expected_direct_queries: usize,
            expected_legacy_queries: usize,
            expected_direct_segments: usize,
            expected_legacy_segments: usize,
            expected_empty_segments: usize,
        ) {
            let metrics_set = ExecutionPlanMetricsSet::new();
            let metrics = PlaidExecMetrics::new(&metrics_set, 0);
            let mut usage = ResidualQueryUsage::default();
            for path in paths {
                usage.observe(*path);
                metrics.record_direct_residual_decision(match path {
                    ResidualSegmentPath::Direct => DirectResidualDecision::Direct,
                    ResidualSegmentPath::Legacy => DirectResidualDecision::Disabled,
                    ResidualSegmentPath::Empty => DirectResidualDecision::EmptySegment,
                });
            }
            usage.record_query_metrics(&metrics);

            assert_eq!(
                metrics.direct_residual_query_count.value(),
                expected_direct_queries
            );
            assert_eq!(
                metrics.legacy_residual_query_count.value(),
                expected_legacy_queries
            );
            assert_eq!(
                metrics.direct_residual_segment_count.value(),
                expected_direct_segments
            );
            assert_eq!(
                metrics.legacy_residual_segment_count.value(),
                expected_legacy_segments
            );
            assert_eq!(
                metrics.direct_residual_empty_segment_count.value(),
                expected_empty_segments
            );
        }

        assert_usage(
            &[ResidualSegmentPath::Empty, ResidualSegmentPath::Empty],
            0,
            0,
            0,
            0,
            2,
        );
        assert_usage(
            &[ResidualSegmentPath::Empty, ResidualSegmentPath::Direct],
            1,
            0,
            1,
            0,
            1,
        );
        assert_usage(
            &[ResidualSegmentPath::Empty, ResidualSegmentPath::Legacy],
            0,
            1,
            0,
            1,
            1,
        );
        assert_usage(
            &[
                ResidualSegmentPath::Empty,
                ResidualSegmentPath::Direct,
                ResidualSegmentPath::Legacy,
            ],
            1,
            1,
            1,
            1,
            1,
        );
    }

    #[test]
    fn mixed_segment_hit_merge_keeps_best_score_and_deduplicates_addresses() {
        let mut candidates = HashMap::new();
        merge_segment_hits(
            &mut candidates,
            [
                lance_plaid::SearchHit {
                    document_ordinal: 0,
                    row_address: 10,
                    score: 1.0,
                },
                lance_plaid::SearchHit {
                    document_ordinal: 1,
                    row_address: 20,
                    score: 0.5,
                },
            ],
        );
        merge_segment_hits(
            &mut candidates,
            [
                // A lower score for an overlapping address must not replace
                // the first segment's candidate.
                lance_plaid::SearchHit {
                    document_ordinal: 7,
                    row_address: 10,
                    score: 0.75,
                },
                // A higher score from either a direct or legacy segment wins.
                lance_plaid::SearchHit {
                    document_ordinal: 8,
                    row_address: 20,
                    score: 0.875,
                },
                lance_plaid::SearchHit {
                    document_ordinal: 9,
                    row_address: 30,
                    score: 0.25,
                },
            ],
        );
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[&10].to_bits(), 1.0_f32.to_bits());
        assert_eq!(candidates[&20].to_bits(), 0.875_f32.to_bits());
        assert_eq!(candidates[&30].to_bits(), 0.25_f32.to_bits());
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

    /// The pre-optimization kernel, retained only as an oracle and benchmark
    /// baseline. In particular, it recreates an Arrow slice and rescans both
    /// inputs for every query-token/document-token pair.
    fn scalar_exact_maxsim_reference(
        query: ArrayView2<'_, f32>,
        document: &FixedSizeListArray,
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
                        "reference does not support null token vectors".to_string(),
                    ));
                }
                let token = document.value(token_index);
                let token = token.as_primitive::<Float32Type>();
                let similarity = scalar_checked_dot(query_token, token.values())?;
                maximum = maximum.max(similarity);
            }
            score += maximum;
        }
        Ok(score)
    }

    fn scalar_checked_dot(query: ArrayView1<'_, f32>, document: &[f32]) -> Result<f32> {
        if query.len() != document.len()
            || query.iter().any(|value| !value.is_finite())
            || document.iter().any(|value| !value.is_finite())
        {
            return Err(Error::invalid_input(
                "reference dimension mismatch or non-finite value".to_string(),
            ));
        }
        Ok(query
            .iter()
            .zip(document)
            .map(|(left, right)| left * right)
            .sum())
    }

    fn deterministic_values(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                // xorshift64*: deterministic and sufficient for numerical
                // coverage without coupling tests to a rand crate version.
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
                let unit = ((bits >> 40) as u32) as f32 / ((1_u32 << 24) - 1) as f32;
                (unit - 0.5) * 0.5
            })
            .collect()
    }

    fn assert_numerically_equivalent(actual: f32, expected: f32, context: &str) {
        if expected.is_nan() {
            assert!(actual.is_nan(), "{context}: expected NaN, got {actual}");
            return;
        }
        let tolerance = 2.0e-5_f32 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: SIMD={actual} (0x{:08x}), scalar={expected} (0x{:08x}), tolerance={tolerance}",
            actual.to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn exact_maxsim_simd_matches_scalar_reference_across_shapes() {
        let dimensions = [1_usize, 2, 3, 7, 15, 16, 17, 31, 32, 64, 127, 128, 129];
        for case in 0..39_usize {
            let dimension = dimensions[case % dimensions.len()];
            let query_tokens = 1 + (case * 5) % 9;
            let document_tokens = 1 + (case * 17) % 73;
            let query = Array2::from_shape_vec(
                (query_tokens, dimension),
                deterministic_values(
                    query_tokens * dimension,
                    0x1234_5678_9abc_def0 ^ case as u64,
                ),
            )
            .unwrap();
            let document = FixedSizeListArray::try_new_from_values(
                Float32Array::from(deterministic_values(
                    document_tokens * dimension,
                    0xfedc_ba98_7654_3210 ^ case as u64,
                )),
                dimension as i32,
            )
            .unwrap();
            let expected = scalar_exact_maxsim_reference(query.view(), &document).unwrap();
            let actual = exact_maxsim(query.view(), &document).unwrap();
            assert_numerically_equivalent(
                actual,
                expected,
                &format!("case={case}, q={query_tokens}, d={document_tokens}, dim={dimension}"),
            );
        }

        for (label, query_tokens, document_tokens, dimension, seed) in [
            (
                "canonical 13x64x96 shape",
                13_usize,
                64_usize,
                96_usize,
                41_u64,
            ),
            (
                "stress/ColBERT-style 32x180x128 shape",
                32_usize,
                180_usize,
                128_usize,
                43_u64,
            ),
        ] {
            let query = Array2::from_shape_vec(
                (query_tokens, dimension),
                deterministic_values(query_tokens * dimension, seed),
            )
            .unwrap();
            let document = FixedSizeListArray::try_new_from_values(
                Float32Array::from(deterministic_values(document_tokens * dimension, seed + 1)),
                dimension as i32,
            )
            .unwrap();
            assert_numerically_equivalent(
                exact_maxsim(query.view(), &document).unwrap(),
                scalar_exact_maxsim_reference(query.view(), &document).unwrap(),
                label,
            );
        }
    }

    #[test]
    fn exact_maxsim_uses_only_the_sliced_document_values() {
        let all_tokens = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![
                100.0, 100.0, 100.0, 100.0, // excluded prefix
                0.5, 0.0, 0.0, 0.0, // first included token
                0.0, 0.25, 0.0, 0.0, // second included token
                200.0, 200.0, 200.0, 200.0, // excluded suffix
            ]),
            4,
        )
        .unwrap();
        let document = all_tokens.slice(1, 2);
        let query = array![[1.0_f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let actual = exact_maxsim(query.view(), &document).unwrap();
        assert_eq!(actual.to_bits(), 0.75_f32.to_bits());
        assert_numerically_equivalent(
            actual,
            scalar_exact_maxsim_reference(query.view(), &document).unwrap(),
            "sliced FixedSizeList child",
        );
    }

    #[test]
    fn exact_maxsim_preserves_empty_and_rejects_invalid_shapes() {
        let query = array![[1.0_f32, 2.0]];
        let empty =
            FixedSizeListArray::try_new_from_values(Float32Array::from(Vec::<f32>::new()), 2)
                .unwrap();
        assert!(exact_maxsim(query.view(), &empty).unwrap().is_nan());

        let wrong_dimension =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0, 3.0]), 3)
                .unwrap();
        let error = exact_maxsim(query.view(), &wrong_dimension).unwrap_err();
        assert!(error.to_string().contains("dimension 3"));

        let zero_dimension_query = Array2::<f32>::zeros((1, 0));
        let error = ExactMaxsimScorer::try_new(zero_dimension_query.view()).unwrap_err();
        assert!(error.to_string().contains("dimension must be positive"));

        let query_storage = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let error = ExactMaxsimScorer::try_new(query_storage.t()).unwrap_err();
        assert!(error.to_string().contains("row-major contiguous"));

        let wrong_type =
            FixedSizeListArray::try_new_from_values(Int32Array::from(vec![1, 2]), 2).unwrap();
        let error = exact_maxsim(query.view(), &wrong_type).unwrap_err();
        assert!(error.to_string().contains("requires Float32"));
    }

    #[test]
    fn exact_maxsim_rejects_null_and_non_finite_values() {
        let query = array![[1.0_f32, 2.0]];
        let float_field = Arc::new(Field::new("item", arrow_schema::DataType::Float32, true));
        let null_token = FixedSizeListArray::try_new(
            float_field.clone(),
            2,
            Arc::new(Float32Array::from(vec![1.0, 2.0])),
            Some(NullBuffer::from(vec![false])),
        )
        .unwrap();
        let error = exact_maxsim(query.view(), &null_token).unwrap_err();
        assert!(error.to_string().contains("null token vectors"));

        let null_value = FixedSizeListArray::try_new(
            float_field,
            2,
            Arc::new(Float32Array::from(vec![Some(1.0), None])),
            None,
        )
        .unwrap();
        let error = exact_maxsim(query.view(), &null_value).unwrap_err();
        assert!(error.to_string().contains("null token values"));

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let invalid_document =
                FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, invalid]), 2)
                    .unwrap();
            let error = exact_maxsim(query.view(), &invalid_document).unwrap_err();
            assert!(error.to_string().contains("non-finite"));

            let invalid_query = array![[1.0_f32, invalid]];
            let finite_document =
                FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0]), 2)
                    .unwrap();
            let error = exact_maxsim(invalid_query.view(), &finite_document).unwrap_err();
            assert!(error.to_string().contains("non-finite"));
        }
    }

    #[test]
    fn exact_scores_batches_rejects_a_null_document_in_later_batch() {
        let tokens =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0_f32, 2.0]), 2)
                .unwrap();
        let field = Arc::new(Field::new("item", tokens.data_type().clone(), true));
        let valid_documents = ListArray::try_new(
            field.clone(),
            OffsetBuffer::from_lengths([1_usize]),
            Arc::new(tokens.clone()),
            None,
        )
        .unwrap();
        let valid_batch =
            RecordBatch::try_from_iter([("mv", Arc::new(valid_documents) as ArrayRef)]).unwrap();

        let null_documents = ListArray::try_new(
            field,
            OffsetBuffer::from_lengths([1_usize]),
            Arc::new(tokens),
            Some(NullBuffer::from(vec![false])),
        )
        .unwrap();
        let null_batch =
            RecordBatch::try_from_iter([("mv", Arc::new(null_documents) as ArrayRef)]).unwrap();
        let query = array![[1.0_f32, 2.0]];
        let error = match exact_scores_batches(&[valid_batch, null_batch], "mv", query.view()) {
            Ok(_) => panic!("null document unexpectedly produced exact scores"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("null document"));
    }

    #[test]
    fn exact_equal_scores_keep_row_id_tie_order() {
        let document = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.5_f32, 0.25, -0.5, 0.125]),
            4,
        )
        .unwrap();
        let query_values = vec![1.0_f32, 0.0, 0.0, 0.0];
        let query_matrix = Array2::from_shape_vec((1, 4), query_values.clone()).unwrap();
        let first = exact_maxsim(query_matrix.view(), &document).unwrap();
        let second = exact_maxsim(query_matrix.view(), &document).unwrap();
        assert_eq!(first.to_bits(), second.to_bits());

        let mut hits = vec![
            ExactHit {
                row_address: 9,
                distance: maxsim_distance(first),
                batch_ordinal: 0,
                batch_index: 0,
            },
            ExactHit {
                row_address: 3,
                distance: maxsim_distance(second),
                batch_ordinal: 0,
                batch_index: 1,
            },
            ExactHit {
                row_address: 7,
                distance: maxsim_distance(first),
                batch_ordinal: 0,
                batch_index: 2,
            },
        ];
        let query = Query {
            column: "mv".to_string(),
            key: Arc::new(Float32Array::from(query_values)),
            k: 2,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::Dot),
            use_index: true,
            query_parallelism: 1,
            dist_q_c: 0.0,
            approx_mode: lance_index::vector::ApproxMode::Normal,
        };
        rank_hits(&mut hits, &query);
        assert_eq!(
            hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>(),
            vec![3, 7]
        );
    }

    /// Run manually with an optimized build and one pinned CPU, for example:
    /// `taskset -c 0 cargo +1.95.0 test -p lance --release
    /// exact_maxsim_kernel_microbenchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "deterministic CPU microbenchmark; run explicitly with --release"]
    fn exact_maxsim_kernel_microbenchmark() {
        use std::hint::black_box;

        fn run_profile(
            label: &str,
            candidates: usize,
            query_tokens: usize,
            document_tokens: usize,
            dimension: usize,
            samples: usize,
        ) {
            let query = Array2::from_shape_vec(
                (query_tokens, dimension),
                deterministic_values(query_tokens * dimension, 0x5eed),
            )
            .unwrap();
            let documents = (0..candidates)
                .map(|candidate| {
                    FixedSizeListArray::try_new_from_values(
                        Float32Array::from(deterministic_values(
                            document_tokens * dimension,
                            0xc001_d00d ^ candidate as u64,
                        )),
                        dimension as i32,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let scorer = ExactMaxsimScorer::try_new(query.view()).unwrap();

            let run_scalar = || {
                documents
                    .iter()
                    .map(|document| {
                        scalar_exact_maxsim_reference(black_box(query.view()), black_box(document))
                            .unwrap()
                    })
                    .sum::<f32>()
            };
            let run_simd = || {
                documents
                    .iter()
                    .map(|document| scorer.score(black_box(document)).unwrap())
                    .sum::<f32>()
            };
            let scalar_result = black_box(run_scalar());
            let simd_result = black_box(run_simd());
            assert_numerically_equivalent(
                simd_result,
                scalar_result,
                &format!("{label} microbenchmark checksum"),
            );

            let mut scalar_samples = Vec::with_capacity(samples);
            let mut simd_samples = Vec::with_capacity(samples);
            for sample in 0..samples {
                let measure_scalar = || {
                    let started = Instant::now();
                    black_box(run_scalar());
                    started.elapsed()
                };
                let measure_simd = || {
                    let started = Instant::now();
                    black_box(run_simd());
                    started.elapsed()
                };
                let (scalar, simd) = if sample % 2 == 0 {
                    (measure_scalar(), measure_simd())
                } else {
                    let simd = measure_simd();
                    let scalar = measure_scalar();
                    (scalar, simd)
                };
                scalar_samples.push(scalar);
                simd_samples.push(simd);
            }
            scalar_samples.sort_unstable();
            simd_samples.sort_unstable();
            let scalar = scalar_samples[samples / 2];
            let simd = simd_samples[samples / 2];
            println!(
                "exact_maxsim profile={label} candidates={candidates} query_tokens={query_tokens} document_tokens={document_tokens} dimension={dimension}: scalar_median={scalar:?}, simd_median={simd:?}, speedup={:.3}x",
                scalar.as_secs_f64() / simd.as_secs_f64()
            );
        }

        // Canonical shape from the current LOTTE smoke workload. This is the
        // representative number used for performance claims.
        run_profile("canonical", 50, 13, 64, 96, 11);
        // Retain a larger shape as a stress/throughput diagnostic only.
        run_profile("stress", 50, 32, 180, 128, 7);
    }
}

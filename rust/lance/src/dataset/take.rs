// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::BTreeMap, collections::HashMap, ops::Range, pin::Pin, sync::Arc, time::Instant,
};

use crate::dataset::fragment::{
    FragReadConfig, FragmentReaderCacheEligibility, FragmentSharedSchedulerEligibility,
    FragmentTakePhaseStats,
};
use crate::dataset::rowids::get_row_id_index;
use crate::io::exec::AddRowOffsetExec;
use crate::{Error, Result};
use arrow::{compute::concat_batches, datatypes::UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, UInt64Array};
use arrow_buffer::{ArrowNativeType, BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::Field as ArrowField;
use datafusion::common::Column;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_expr::Expr;
use futures::{Future, Stream, StreamExt, TryStreamExt};
use lance_arrow::RecordBatchExt;
use lance_arrow::json::convert_lance_json_to_arrow;
use lance_core::datatypes::Schema;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::OffsetMapper;
use lance_core::{ROW_ADDR, ROW_OFFSET};
use lance_datafusion::projection::{OutputColumn, ProjectionPlan};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

use super::ProjectionRequest;
use super::{Dataset, fragment::FileFragment, scanner::DatasetRecordBatchStream};

/// Convert a list of row offsets to a list of row addresses
///
/// A row offset is a 64-bit integer in the range [0, num_rows_in_dataset]
///
/// For example, if there are two fragments, each with 100 rows, then the row offset
/// 150 maps to the address (1, 50), the 50th row in the second fragment
///
/// Row offsets are useful for sampling because you don't need to know the ids / addresses
/// up front (you can just use the range [0, num_rows_in_dataset]) and they can be cheaply
/// converted to row addresses in a single pass through fragment sizes (which is what this method does)
///
/// This method accounts for deletions.  If there is one fragment with 100 rows and rows 50-59 are
/// deleted then the row offset 70 will map to the address (0, 80) and the row offset 100 will map
/// to the address (1, 10), assuming the second fragment starts with 10 undeleted rows.
///
/// If any offsets are beyond the end of the dataset, they will be mapped to a tombstone row address.
pub async fn row_offsets_to_row_addresses(
    fragments: &[FileFragment],
    row_indices: &[u64],
) -> Result<Vec<u64>> {
    let mut perm = permutation::sort(row_indices);
    let sorted_offsets = perm.apply_slice(row_indices);

    let mut frag_iter = fragments.iter();
    let mut cur_frag = frag_iter.next();
    let mut cur_frag_rows = if let Some(cur_frag) = cur_frag {
        cur_frag.count_rows(None).await? as u64
    } else {
        0
    };
    let mut offset_mapper = if let Some(cur_frag) = cur_frag {
        let deletion_vector = cur_frag.get_deletion_vector().await?;
        deletion_vector.map(OffsetMapper::new)
    } else {
        None
    };
    let mut frag_offset = 0;

    let mut addrs: Vec<u64> = Vec::with_capacity(sorted_offsets.len());
    for sorted_offset in sorted_offsets.into_iter() {
        while cur_frag.is_some() && sorted_offset >= frag_offset + cur_frag_rows {
            frag_offset += cur_frag_rows;
            cur_frag = frag_iter.next();
            cur_frag_rows = if let Some(cur_frag) = cur_frag {
                cur_frag.count_rows(None).await? as u64
            } else {
                0
            };
            offset_mapper = if let Some(cur_frag) = cur_frag {
                let deletion_vector = cur_frag.get_deletion_vector().await?;
                deletion_vector.map(OffsetMapper::new)
            } else {
                None
            };
        }
        let Some(cur_frag) = cur_frag else {
            addrs.push(RowAddress::TOMBSTONE_ROW);
            continue;
        };

        let mut local_offset = (sorted_offset - frag_offset) as u32;
        if let Some(offset_mapper) = &mut offset_mapper {
            local_offset = offset_mapper.map_offset(local_offset);
        };
        let row_addr = RowAddress::new_from_parts(cur_frag.id() as u32, local_offset);
        addrs.push(u64::from(row_addr));
    }

    // Restore the original order
    perm.apply_inv_slice_in_place(&mut addrs);
    Ok(addrs)
}

pub async fn take(
    dataset: &Dataset,
    offsets: &[u64],
    projection: ProjectionRequest,
) -> Result<RecordBatch> {
    let projection = projection.into_projection_plan(Arc::new(dataset.clone()))?;
    if offsets.is_empty() {
        return to_logical_json_batch(RecordBatch::new_empty(Arc::new(
            projection.output_schema()?,
        )));
    }

    // First, convert the dataset offsets into row addresses
    let fragments = dataset.get_fragments();
    let addrs = row_offsets_to_row_addresses(&fragments, offsets).await?;

    let builder = TakeBuilder::try_new_from_addresses(
        Arc::new(dataset.clone()),
        addrs,
        Arc::new(projection),
    )?;

    take_rows(builder).await
}

/// Take rows by the internal ROW ids.
#[allow(clippy::needless_question_mark)]
async fn do_take_rows(
    mut builder: TakeBuilder,
    projection: Arc<ProjectionPlan>,
) -> Result<RecordBatch> {
    // If we need row addresses in output, add to projection's output expressions
    let projection = if builder.with_row_address {
        let mut proj = (*projection).clone();
        // Add _rowaddr to output if not already present
        if !proj
            .requested_output_expr
            .iter()
            .any(|c| c.name == ROW_ADDR)
        {
            proj.requested_output_expr.push(OutputColumn {
                expr: Expr::Column(Column::from_name(ROW_ADDR)),
                name: ROW_ADDR.to_string(),
            });
        }
        Arc::new(proj)
    } else {
        projection
    };

    let with_row_id_in_projection = projection.physical_projection.with_row_id;
    let with_row_addr_in_projection = projection.physical_projection.with_row_addr;
    let with_row_created_at_version_in_projection =
        projection.physical_projection.with_row_created_at_version;
    let with_row_last_updated_at_version_in_projection = projection
        .physical_projection
        .with_row_last_updated_at_version;

    let row_addrs = builder.get_row_addrs().await?.clone();

    if row_addrs.is_empty() {
        // It is possible that `row_id_index` returns None when a fragment has been wholly deleted
        let empty_batch = RecordBatch::new_empty(Arc::new(builder.projection.output_schema()?));
        // If row addresses were requested, add an empty row address column.
        // This ensures callers that expect the _rowaddr column don't panic.
        if builder.with_row_address {
            let row_addr_col = Arc::new(UInt64Array::from(Vec::<u64>::new()));
            let row_addr_field =
                ArrowField::new(ROW_ADDR, arrow::datatypes::DataType::UInt64, false);
            return to_logical_json_batch(
                empty_batch.try_with_column(row_addr_field, row_addr_col)?,
            );
        }
        return to_logical_json_batch(empty_batch);
    }

    let row_addr_stats = check_row_addrs(&row_addrs);

    // This method is mostly to annotate the send bound to avoid the
    // higher-order lifetime error.
    // manually implemented async for Send bound
    #[allow(clippy::manual_async_fn)]
    fn do_take(
        fragment: FileFragment,
        row_offsets: Vec<u32>,
        projection: Arc<Schema>,
        with_row_id: bool,
        with_row_addresses: bool,
        with_row_created_at_version: bool,
        with_row_last_updated_at_version: bool,
    ) -> impl Future<Output = Result<RecordBatch>> + Send {
        async move {
            fragment
                .take_rows(
                    &row_offsets,
                    projection.as_ref(),
                    with_row_id,
                    with_row_addresses,
                    with_row_created_at_version,
                    with_row_last_updated_at_version,
                )
                .await
        }
    }

    let physical_schema = Arc::new(projection.physical_projection.to_bare_schema());
    let mut batch = if row_addr_stats.contiguous {
        // Fastest path: Can use `read_range` directly
        let start = row_addrs.first().expect("empty range passed to take_rows");
        let fragment_id = (start >> 32) as usize;
        let range_start = *start as u32 as usize;
        let range_end = *row_addrs.last().expect("empty range passed to take_rows") as u32 as usize;
        let range = range_start..(range_end + 1);

        let fragment = builder.dataset.get_fragment(fragment_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "rowaddr start: {} belongs to non-existent fragment: {}",
                start, fragment_id
            ))
        })?;

        let read_config = FragReadConfig::default()
            .with_row_id(with_row_id_in_projection)
            .with_row_address(with_row_addr_in_projection)
            .with_row_created_at_version(with_row_created_at_version_in_projection)
            .with_row_last_updated_at_version(with_row_last_updated_at_version_in_projection);
        let reader = fragment.open(&physical_schema, read_config).await?;
        reader.legacy_read_range_as_batch(range).await
    } else if row_addr_stats.sorted {
        // Don't need to re-arrange data, just concatenate
        let mut batches: Vec<_> = Vec::new();
        let mut current_fragment = row_addrs[0] >> 32;
        let mut current_start = 0;
        let mut row_addr_iter = row_addrs.iter().enumerate();
        'outer: loop {
            let (fragment_id, range) = loop {
                if let Some((i, row_addr)) = row_addr_iter.next() {
                    let fragment_id = row_addr >> 32;
                    if fragment_id != current_fragment {
                        let next = (current_fragment, current_start..i);
                        current_fragment = fragment_id;
                        current_start = i;
                        break next;
                    }
                } else if current_start != row_addrs.len() {
                    let next = (current_fragment, current_start..row_addrs.len());
                    current_start = row_addrs.len();
                    break next;
                } else {
                    break 'outer;
                }
            };

            let fragment = builder
                .dataset
                .get_fragment(fragment_id as usize)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "rowaddr {} belongs to non-existent fragment: {}",
                        row_addrs[range.start], fragment_id
                    ))
                })?;
            let row_offsets: Vec<u32> = row_addrs[range].iter().map(|x| *x as u32).collect();

            let batch_fut = do_take(
                fragment,
                row_offsets,
                physical_schema.clone(),
                with_row_id_in_projection,
                with_row_addr_in_projection,
                with_row_created_at_version_in_projection,
                with_row_last_updated_at_version_in_projection,
            );
            batches.push(batch_fut);
        }
        let batches: Vec<RecordBatch> = futures::stream::iter(batches)
            .buffered(builder.dataset.object_store.io_parallelism())
            .try_collect()
            .await?;
        Ok(concat_batches(&batches[0].schema(), &batches)?)
    } else {
        // Slow case: need to re-map data into expected order
        let mut sorted_row_addrs = row_addrs.clone();
        sorted_row_addrs.sort();
        // Go ahead and dedup, we will reinsert duplicates during the remapping
        sorted_row_addrs.dedup();
        // Group ROW Ids by the fragment
        let mut row_addrs_per_fragment: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        sorted_row_addrs.iter().for_each(|row_addr| {
            let row_addr = RowAddress::from(*row_addr);
            let fragment_id = row_addr.fragment_id();
            let offset = row_addr.row_offset();
            row_addrs_per_fragment
                .entry(fragment_id)
                .and_modify(|v| v.push(offset))
                .or_insert_with(|| vec![offset]);
        });

        let fragments = builder.dataset.get_fragments();
        let fragment_and_indices = fragments.into_iter().filter_map(|f| {
            let row_offset = row_addrs_per_fragment.remove(&(f.id() as u32))?;
            Some((f, row_offset))
        });

        let mut batches = futures::stream::iter(fragment_and_indices)
            .map(|(fragment, indices)| {
                do_take(
                    fragment,
                    indices,
                    physical_schema.clone(),
                    with_row_id_in_projection,
                    true,
                    with_row_created_at_version_in_projection,
                    with_row_last_updated_at_version_in_projection,
                )
            })
            .buffered(builder.dataset.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;
        let one_batch = if batches.len() > 1 {
            concat_batches(&batches[0].schema(), &batches)?
        } else {
            batches.pop().unwrap()
        };
        // Note: one_batch may contains fewer rows than the number of requested
        // row ids because some rows may have been deleted. Because of this, we
        // get the results with row ids so that we can re-order the results
        // to match the requested order.

        let returned_row_addr = one_batch
            .column_by_name(ROW_ADDR)
            .ok_or_else(|| Error::internal("_rowaddr column not found"))?
            .as_primitive::<UInt64Type>()
            .values();

        let addr_to_pos: HashMap<u64, u64> = returned_row_addr
            .iter()
            .enumerate()
            .map(|(i, addr)| (*addr, i as u64))
            .collect();
        let remapping_index: UInt64Array = row_addrs
            .iter()
            .filter_map(|o| addr_to_pos.get(o).copied())
            .collect();

        // remapping_index may be greater than the number of rows in one_batch
        // if there are duplicates in the requested row ids. This is expected.
        debug_assert!(remapping_index.len() >= one_batch.num_rows());

        // There's a bug in arrow_select::take::take, that it doesn't handle empty struct correctly,
        // so we need to handle it manually here.
        // TODO: remove this once the bug is fixed.
        let struct_arr: StructArray = one_batch.into();
        let reordered = take_struct_array(&struct_arr, &remapping_index)?;
        Ok(reordered.into())
    }?;

    if builder.with_row_address || projection.must_add_row_offset {
        // compile `ROW_ADDR` column
        if batch.num_rows() != row_addrs.len() {
            return Err(Error::not_supported_source(format!(
                "Expected {} rows, got {}.  A take operation that includes row addresses must not target deleted rows.",
                row_addrs.len(),
                batch.num_rows()
            ).into()));
        }

        let row_addr_col: ArrayRef = Arc::new(UInt64Array::from(row_addrs));

        if projection.must_add_row_offset {
            // compile and inject `ROW_OFFSET` column
            let row_offset_col =
                AddRowOffsetExec::compute_row_offset_array(&row_addr_col, builder.dataset).await?;
            let row_offset_field =
                ArrowField::new(ROW_OFFSET, arrow::datatypes::DataType::UInt64, false);
            if batch.schema().column_with_name(ROW_OFFSET).is_none() {
                batch = batch.try_with_column(row_offset_field, row_offset_col)?;
            }
        }

        if builder.with_row_address {
            // inject `ROW_ADDR` column
            let row_addr_field =
                ArrowField::new(ROW_ADDR, arrow::datatypes::DataType::UInt64, false);
            if batch.schema().column_with_name(ROW_ADDR).is_none() {
                batch = batch.try_with_column(row_addr_field, row_addr_col)?;
            }
        }
    }

    to_logical_json_batch(projection.project_batch(batch).await?)
}

async fn take_rows(builder: TakeBuilder) -> Result<RecordBatch> {
    if builder.is_empty() {
        return to_logical_json_batch(RecordBatch::new_empty(Arc::new(
            builder.projection.output_schema()?,
        )));
    }

    let projection = builder.projection.clone();

    do_take_rows(builder, projection).await
}

fn to_logical_json_batch(batch: RecordBatch) -> Result<RecordBatch> {
    Ok(convert_lance_json_to_arrow(&batch)?)
}

/// Get a stream of batches based on iterator of ranges of row numbers.
///
/// This is an experimental API. It may change at any time.
pub fn take_scan(
    dataset: &Dataset,
    row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
    projection: Arc<Schema>,
    batch_readahead: usize,
) -> DatasetRecordBatchStream {
    let arrow_schema = Arc::new(projection.as_ref().into());
    let dataset = Arc::new(dataset.clone());
    let batch_stream = row_ranges
        .map(move |res| {
            let dataset = dataset.clone();
            let projection = projection.clone();
            let fut = async move {
                let range = res.map_err(|err| DataFusionError::External(Box::new(err)))?;
                let row_pos: Vec<u64> = (range.start..range.end).collect();
                dataset
                    .take(&row_pos, ProjectionRequest::Schema(projection.clone()))
                    .await
                    .map_err(|err| DataFusionError::External(Box::new(err)))
            };
            async move { tokio::task::spawn(fut).await.unwrap() }
        })
        .buffered(batch_readahead);

    DatasetRecordBatchStream::new(Box::pin(RecordBatchStreamAdapter::new(
        arrow_schema,
        batch_stream,
    )))
}

struct RowAddressStats {
    sorted: bool,
    contiguous: bool,
}

/// Profile for one successful grouped physical read.
///
/// `parent_wall_nanos` encloses the complete helper. Plan, grouping, optional
/// explicit-scheduler creation, fanout collection, and row-offset injection are
/// one-time wall-clock phases. Fragment open and read are disjoint within each
/// fragment but are aggregated across concurrently processed fragments; those
/// aggregate work timers must not be added to the parent wall time.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GroupedPhysicalReadStats {
    pub parent_wall_nanos: u64,
    pub plan_nanos: u64,
    pub grouping_nanos: u64,
    /// One-time wall for an explicitly managed scheduler. The strict control
    /// leaves scheduler ownership inside fragment open and reports zero.
    pub scheduler_create_wall_nanos: u64,
    pub fragment_open_aggregate_nanos: u64,
    pub fragment_open_max_nanos: u64,
    pub fragment_read_aggregate_nanos: u64,
    pub fragment_read_max_nanos: u64,
    pub fragment_total_elapsed_aggregate_nanos: u64,
    pub fragment_total_elapsed_max_nanos: u64,
    pub fanout_collect_wall_nanos: u64,
    pub fanout_concurrency_limit: usize,
    pub row_offset_injection_wall_nanos: u64,
    pub fragments: usize,
    pub rows: usize,
    pub batch_bytes: usize,
    pub rows_per_fragment_min: usize,
    pub rows_per_fragment_max: usize,
    /// Post-coalescing physical I/O submitted through explicit schedulers.
    /// This excludes deletion/stable-row-ID side reads and any legacy or
    /// non-default-base file whose scheduler is managed internally.
    pub scheduler_scoped_iops: u64,
    pub scheduler_scoped_requests: u64,
    pub scheduler_scoped_bytes_read: u64,
    pub scheduler_stats_covered_fragments: usize,
    pub shared_scheduler_queries: usize,
    pub shared_scheduler_fragments: usize,
    pub per_fragment_scheduler_queries: usize,
    pub per_fragment_scheduler_fragments: usize,
    pub shared_scheduler_fallback_queries: usize,
    pub shared_scheduler_fallback_legacy_fragments: usize,
    pub shared_scheduler_fallback_nonprimary_fragments: usize,
    pub shared_scheduler_fallback_unsupported_fragments: usize,
    pub reader_cache_queries: usize,
    pub reader_cache_eligible_files: u64,
    pub reader_cache_lookup_files: u64,
    pub reader_cache_hit_files: u64,
    pub reader_cache_miss_open_files: u64,
    pub reader_cache_coalesced_files: u64,
    pub reader_cache_bypass_files: u64,
    pub reader_cache_fallback_open_files: u64,
    pub reader_cache_open_failures: u64,
    pub reader_cache_resident_entries_start_approx: u64,
    pub reader_cache_resident_entries_end_approx: u64,
    pub reader_cache_capacity: u64,
    pub reader_cache_fd_soft_limit: u64,
    pub reader_cache_lookup_nanos: u64,
    pub reader_cache_acquire_nanos: u64,
    pub reader_cache_physical_open_nanos: u64,
    pub reader_cache_bind_nanos: u64,
    pub reader_cache_fallback_queries: usize,
    pub reader_cache_fallback_legacy_fragments: usize,
    pub reader_cache_fallback_nonprimary_fragments: usize,
    pub reader_cache_fallback_nonlocal_fragments: usize,
    pub reader_cache_fallback_unknown_size_fragments: usize,
    pub reader_cache_fallback_small_file_fragments: usize,
    pub reader_cache_fallback_unsupported_fragments: usize,
    pub reader_cache_fallback_capacity_queries: usize,
}

#[derive(Debug)]
pub(crate) struct GroupedPhysicalRead {
    pub batches: Vec<RecordBatch>,
    pub stats: GroupedPhysicalReadStats,
}

struct FragmentTakeOutcome {
    batch: RecordBatch,
    phases: FragmentTakePhaseStats,
    total_elapsed_nanos: u64,
}

struct FragmentTakePlan {
    fragment: FileFragment,
    row_offsets: Vec<u32>,
    reader_priority: u32,
    shared_scheduler_eligibility: FragmentSharedSchedulerEligibility,
    reader_cache_eligibility: FragmentReaderCacheEligibility,
}

fn elapsed_nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn check_row_addrs(row_addrs: &[u64]) -> RowAddressStats {
    let mut sorted = true;
    let mut contiguous = true;

    if row_addrs.is_empty() {
        return RowAddressStats { sorted, contiguous };
    }

    let mut last_offset = row_addrs[0];
    let first_fragment_id = row_addrs[0] >> 32;

    for addr in row_addrs.iter().skip(1) {
        sorted &= *addr >= last_offset;
        contiguous &= *addr == last_offset + 1;
        // Contiguous also requires the fragment ids are all the same
        contiguous &= (*addr >> 32) == first_fragment_id;
        last_offset = *addr;
    }

    RowAddressStats { sorted, contiguous }
}

/// Builder for the `take` operation.
#[derive(Clone, Debug)]
pub struct TakeBuilder {
    dataset: Arc<Dataset>,
    row_ids: Option<Vec<u64>>,
    row_addrs: Option<Vec<u64>>,
    projection: Arc<ProjectionPlan>,
    with_row_address: bool,
}

impl TakeBuilder {
    /// Create a new `TakeBuilder` for taking by id
    pub fn try_new_from_ids(
        dataset: Arc<Dataset>,
        row_ids: Vec<u64>,
        projection: ProjectionRequest,
    ) -> Result<Self> {
        Ok(Self {
            row_ids: Some(row_ids),
            row_addrs: None,
            projection: Arc::new(projection.into_projection_plan(dataset.clone())?),
            dataset,
            with_row_address: false,
        })
    }

    /// Create a new `TakeBuilder` for taking by address
    pub fn try_new_from_addresses(
        dataset: Arc<Dataset>,
        addresses: Vec<u64>,
        projection: Arc<ProjectionPlan>,
    ) -> Result<Self> {
        Ok(Self {
            row_ids: None,
            row_addrs: Some(addresses),
            projection,
            dataset,
            with_row_address: false,
        })
    }

    /// Adds row addresses to the output
    pub fn with_row_address(mut self, with_row_address: bool) -> Self {
        self.with_row_address = with_row_address;
        self
    }

    /// Execute the take operation and return a single batch
    pub async fn execute(self) -> Result<RecordBatch> {
        take_rows(self).await
    }

    /// Read a physically sorted, address-based take as one physical batch per
    /// fragment, without concatenating or logically projecting the batches.
    ///
    /// This internal specialization computes dataset row offsets once for the
    /// complete address list. The caller must apply the builder's logical
    /// projection after selecting or concatenating rows. `None` means the
    /// request needs ID translation, address reordering, row-address injection,
    /// or empty-output schema recovery and must use `execute`.
    pub(crate) async fn read_sorted_physical_by_fragment(
        self,
    ) -> Result<Option<GroupedPhysicalRead>> {
        self.read_sorted_physical_by_fragment_with_options(false, false)
            .await
    }

    /// Variant of [`Self::read_sorted_physical_by_fragment`] that may share one
    /// scheduler across the complete grouped read.
    ///
    /// The opt-in path is whole-query conservative: all projection-matching
    /// files in all participating fragments must be default-base V2 files.
    /// Otherwise no shared scheduler is created and the complete request uses
    /// the strict per-file control path.
    pub(crate) async fn read_sorted_physical_by_fragment_with_shared_scheduler(
        self,
        shared_scheduler_enabled: bool,
    ) -> Result<Option<GroupedPhysicalRead>> {
        self.read_sorted_physical_by_fragment_with_options(shared_scheduler_enabled, false)
            .await
    }

    /// Internal grouped-read options used by PLAID experiments.  Both
    /// optimizations are whole-query gated and default off.
    pub(crate) async fn read_sorted_physical_by_fragment_with_options(
        self,
        shared_scheduler_enabled: bool,
        reader_cache_enabled: bool,
    ) -> Result<Option<GroupedPhysicalRead>> {
        let parent_started = Instant::now();
        if self.row_ids.is_some() || self.with_row_address {
            return Ok(None);
        }
        let Some(row_addrs) = self.row_addrs.as_ref() else {
            return Ok(None);
        };
        if row_addrs.is_empty() {
            return Ok(None);
        }
        if !check_row_addrs(row_addrs).sorted {
            return Ok(None);
        }

        let projection = self.projection.clone();
        let with_row_id = projection.physical_projection.with_row_id;
        let with_row_address = projection.physical_projection.with_row_addr;
        let with_row_created_at_version =
            projection.physical_projection.with_row_created_at_version;
        let with_row_last_updated_at_version = projection
            .physical_projection
            .with_row_last_updated_at_version;
        let physical_schema = Arc::new(projection.physical_projection.to_bare_schema());
        let plan_nanos = elapsed_nanos(parent_started);

        // Keep this helper local to make its Send bound explicit while the
        // independent fragment reads are buffered. `buffered` below preserves
        // fragment order even when later reads complete first.
        #[allow(clippy::manual_async_fn)]
        fn take_fragment(
            fragment: FileFragment,
            row_offsets: Vec<u32>,
            projection: Arc<Schema>,
            shared_scheduler: Option<Arc<ScanScheduler>>,
            reader_cache_query_stats: Option<
                Arc<crate::session::data_file_reader_cache::ReaderCacheQueryStats>,
            >,
            reader_priority: u32,
            with_row_id: bool,
            with_row_address: bool,
            with_row_created_at_version: bool,
            with_row_last_updated_at_version: bool,
        ) -> impl Future<Output = Result<FragmentTakeOutcome>> + Send {
            async move {
                let fragment_started = Instant::now();
                // Keep the control path's scheduler creation exactly where it
                // was: FileFragment::open creates one internally for each
                // projection-matching default-base V2 data file. The scoped
                // scheduler fields remain unavailable until the opt-in shared
                // path supplies an Arc.
                let mut read_config = FragReadConfig::default()
                    .with_row_id(with_row_id)
                    .with_row_address(with_row_address)
                    .with_row_created_at_version(with_row_created_at_version)
                    .with_row_last_updated_at_version(with_row_last_updated_at_version);
                if let Some(shared_scheduler) = shared_scheduler {
                    read_config = read_config
                        .with_scan_scheduler(shared_scheduler)
                        .with_reader_priority(reader_priority);
                }
                if let Some(reader_cache_query_stats) = reader_cache_query_stats {
                    read_config =
                        read_config.with_reader_cache_query_stats(reader_cache_query_stats);
                }
                let mut phases = FragmentTakePhaseStats::default();
                let batch = fragment
                    .take_rows_with_config(
                        &row_offsets,
                        projection.as_ref(),
                        read_config,
                        Some(&mut phases),
                    )
                    .await?;
                Ok(FragmentTakeOutcome {
                    batch,
                    phases,
                    total_elapsed_nanos: elapsed_nanos(fragment_started),
                })
            }
        }

        let grouping_started = Instant::now();
        let mut plans = Vec::new();
        let mut start = 0;
        while start < row_addrs.len() {
            let fragment_id = row_addrs[start] >> 32;
            let mut end = start + 1;
            while end < row_addrs.len() && row_addrs[end] >> 32 == fragment_id {
                end += 1;
            }
            let fragment = self
                .dataset
                .get_fragment(fragment_id as usize)
                .ok_or_else(|| {
                    Error::invalid_input(format!(
                        "rowaddr {} belongs to non-existent fragment: {}",
                        row_addrs[start], fragment_id
                    ))
                })?;
            let row_offsets = row_addrs[start..end]
                .iter()
                .map(|address| *address as u32)
                .collect();
            let reader_priority = u32::try_from(plans.len()).unwrap_or(u32::MAX);
            let mut shared_scheduler_eligibility = FragmentSharedSchedulerEligibility::Eligible;
            if shared_scheduler_enabled {
                shared_scheduler_eligibility = fragment
                    .grouped_shared_scheduler_eligibility(physical_schema.as_ref())
                    .unwrap_or(FragmentSharedSchedulerEligibility::Unsupported);
                if reader_priority == u32::MAX && plans.len() != u32::MAX as usize {
                    shared_scheduler_eligibility = FragmentSharedSchedulerEligibility::Unsupported;
                }
            }
            let reader_cache_eligibility = if reader_cache_enabled {
                fragment
                    .grouped_reader_cache_eligibility(physical_schema.as_ref())
                    .unwrap_or(FragmentReaderCacheEligibility::Unsupported)
            } else {
                FragmentReaderCacheEligibility::Eligible { files: 0 }
            };
            plans.push(FragmentTakePlan {
                fragment,
                row_offsets,
                reader_priority,
                shared_scheduler_eligibility,
                reader_cache_eligibility,
            });
            start = end;
        }
        let grouping_nanos = elapsed_nanos(grouping_started);

        let shared_scheduler_fallback_legacy_fragments = plans
            .iter()
            .filter(|plan| {
                plan.shared_scheduler_eligibility == FragmentSharedSchedulerEligibility::Legacy
            })
            .count();
        let shared_scheduler_fallback_nonprimary_fragments = plans
            .iter()
            .filter(|plan| {
                plan.shared_scheduler_eligibility
                    == FragmentSharedSchedulerEligibility::NonPrimaryBase
            })
            .count();
        let shared_scheduler_fallback_unsupported_fragments = plans
            .iter()
            .filter(|plan| {
                plan.shared_scheduler_eligibility == FragmentSharedSchedulerEligibility::Unsupported
            })
            .count();
        let all_fragments_eligible = plans.iter().all(|plan| {
            plan.shared_scheduler_eligibility == FragmentSharedSchedulerEligibility::Eligible
        });
        let use_shared_scheduler = shared_scheduler_enabled && all_fragments_eligible;
        let scheduler_create_started = use_shared_scheduler.then(Instant::now);
        let shared_scheduler = use_shared_scheduler.then(|| {
            let object_store = self.dataset.object_store.clone();
            ScanScheduler::new(
                object_store.clone(),
                SchedulerConfig::max_bandwidth(&object_store),
            )
        });
        let scheduler_create_wall_nanos = scheduler_create_started.map(elapsed_nanos).unwrap_or(0);

        let reader_cache_fallback_legacy_fragments = plans
            .iter()
            .filter(|plan| plan.reader_cache_eligibility == FragmentReaderCacheEligibility::Legacy)
            .count();
        let reader_cache_fallback_nonprimary_fragments = plans
            .iter()
            .filter(|plan| {
                plan.reader_cache_eligibility == FragmentReaderCacheEligibility::NonPrimaryBase
            })
            .count();
        let reader_cache_fallback_nonlocal_fragments = plans
            .iter()
            .filter(|plan| {
                plan.reader_cache_eligibility == FragmentReaderCacheEligibility::NonLocalStore
            })
            .count();
        let reader_cache_fallback_unknown_size_fragments = plans
            .iter()
            .filter(|plan| {
                plan.reader_cache_eligibility == FragmentReaderCacheEligibility::UnknownSize
            })
            .count();
        let reader_cache_fallback_small_file_fragments = plans
            .iter()
            .filter(|plan| {
                plan.reader_cache_eligibility == FragmentReaderCacheEligibility::SmallFile
            })
            .count();
        let reader_cache_fallback_unsupported_fragments = plans
            .iter()
            .filter(|plan| {
                plan.reader_cache_eligibility == FragmentReaderCacheEligibility::Unsupported
            })
            .count();
        let reader_cache_eligible_files = plans
            .iter()
            .map(|plan| match plan.reader_cache_eligibility {
                FragmentReaderCacheEligibility::Eligible { files } => files,
                _ => 0,
            })
            .sum::<usize>();
        let all_fragments_reader_cache_eligible = plans.iter().all(|plan| {
            matches!(
                plan.reader_cache_eligibility,
                FragmentReaderCacheEligibility::Eligible { .. }
            )
        });
        let reader_cache_capacity_ok = reader_cache_eligible_files
            <= self
                .dataset
                .session
                .data_file_reader_cache
                .max_eligible_files_per_query();
        let use_reader_cache = reader_cache_enabled
            && all_fragments_reader_cache_eligible
            && reader_cache_eligible_files > 0
            && reader_cache_capacity_ok;
        let reader_cache_query_stats = use_reader_cache.then(|| {
            self.dataset
                .session
                .data_file_reader_cache
                .begin_query(reader_cache_eligible_files)
        });

        let reads = plans
            .into_iter()
            .map(|plan| {
                take_fragment(
                    plan.fragment,
                    plan.row_offsets,
                    physical_schema.clone(),
                    shared_scheduler.clone(),
                    reader_cache_query_stats.clone(),
                    plan.reader_priority,
                    with_row_id,
                    with_row_address,
                    with_row_created_at_version,
                    with_row_last_updated_at_version,
                )
            })
            .collect::<Vec<_>>();

        let io_parallelism = self.dataset.object_store.io_parallelism();
        let fanout_concurrency_limit = reads.len().min(io_parallelism);
        let fanout_started = Instant::now();
        let outcomes = futures::stream::iter(reads)
            .buffered(io_parallelism)
            .try_collect::<Vec<_>>()
            .await?;
        let fanout_collect_wall_nanos = elapsed_nanos(fanout_started);
        let fragment_open_aggregate_nanos = outcomes.iter().fold(0_u64, |total, outcome| {
            total.saturating_add(outcome.phases.open_nanos)
        });
        let fragment_open_max_nanos = outcomes
            .iter()
            .map(|outcome| outcome.phases.open_nanos)
            .max()
            .unwrap_or(0);
        let fragment_read_aggregate_nanos = outcomes.iter().fold(0_u64, |total, outcome| {
            total.saturating_add(outcome.phases.read_nanos)
        });
        let fragment_read_max_nanos = outcomes
            .iter()
            .map(|outcome| outcome.phases.read_nanos)
            .max()
            .unwrap_or(0);
        let fragment_total_elapsed_aggregate_nanos =
            outcomes.iter().fold(0_u64, |total, outcome| {
                total.saturating_add(outcome.total_elapsed_nanos)
            });
        let fragment_total_elapsed_max_nanos = outcomes
            .iter()
            .map(|outcome| outcome.total_elapsed_nanos)
            .max()
            .unwrap_or(0);
        let mut batches = outcomes
            .into_iter()
            .map(|outcome| outcome.batch)
            .collect::<Vec<_>>();
        let mut row_offset_injection_wall_nanos = 0;
        if projection.must_add_row_offset {
            let row_offset_started = Instant::now();
            let returned_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
            if returned_rows != row_addrs.len() {
                return Err(Error::not_supported_source(format!(
                    "Expected {} rows, got {}.  A take operation that includes row addresses must not target deleted rows.",
                    row_addrs.len(),
                    returned_rows
                ).into()));
            }
            let row_addr_col: ArrayRef = Arc::new(UInt64Array::from(row_addrs.clone()));
            let row_offset_col =
                AddRowOffsetExec::compute_row_offset_array(&row_addr_col, self.dataset.clone())
                    .await?;
            let mut start = 0;
            for batch in &mut batches {
                let len = batch.num_rows();
                if batch.schema().column_with_name(ROW_OFFSET).is_none() {
                    *batch = batch.clone().try_with_column(
                        ArrowField::new(ROW_OFFSET, arrow::datatypes::DataType::UInt64, false),
                        row_offset_col.slice(start, len),
                    )?;
                }
                start += len;
            }
            row_offset_injection_wall_nanos = elapsed_nanos(row_offset_started);
        }

        let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        let batch_bytes = batches
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum::<usize>();
        let rows_per_fragment_min = batches.iter().map(RecordBatch::num_rows).min().unwrap_or(0);
        let rows_per_fragment_max = batches.iter().map(RecordBatch::num_rows).max().unwrap_or(0);
        // All fragment futures have completed before this snapshot.  The
        // scheduler remains owned here, so its root cannot be dropped while
        // any covered reader still has pending I/O.
        let scoped_scheduler_stats = shared_scheduler
            .as_ref()
            .map(|scheduler| scheduler.stats())
            .unwrap_or_default();
        let fragments = batches.len();
        let shared_scheduler_queries = usize::from(use_shared_scheduler);
        let shared_scheduler_fragments = if use_shared_scheduler { fragments } else { 0 };
        let per_fragment_scheduler_queries = usize::from(!use_shared_scheduler);
        let per_fragment_scheduler_fragments = if use_shared_scheduler { 0 } else { fragments };
        let shared_scheduler_fallback_queries =
            usize::from(shared_scheduler_enabled && !use_shared_scheduler);
        let reader_cache_snapshot = reader_cache_query_stats.as_ref().map(|query_stats| {
            self.dataset
                .session
                .data_file_reader_cache
                .finish_query(query_stats)
        });
        let reader_cache_fallback_queries = usize::from(reader_cache_enabled && !use_reader_cache);
        let reader_cache_fallback_capacity_queries = usize::from(
            reader_cache_enabled
                && all_fragments_reader_cache_eligible
                && reader_cache_eligible_files > 0
                && !reader_cache_capacity_ok,
        );
        let stats = GroupedPhysicalReadStats {
            parent_wall_nanos: elapsed_nanos(parent_started),
            plan_nanos,
            grouping_nanos,
            scheduler_create_wall_nanos,
            fragment_open_aggregate_nanos,
            fragment_open_max_nanos,
            fragment_read_aggregate_nanos,
            fragment_read_max_nanos,
            fragment_total_elapsed_aggregate_nanos,
            fragment_total_elapsed_max_nanos,
            fanout_collect_wall_nanos,
            fanout_concurrency_limit,
            row_offset_injection_wall_nanos,
            fragments,
            rows,
            batch_bytes,
            rows_per_fragment_min,
            rows_per_fragment_max,
            scheduler_scoped_iops: scoped_scheduler_stats.iops,
            scheduler_scoped_requests: scoped_scheduler_stats.requests,
            scheduler_scoped_bytes_read: scoped_scheduler_stats.bytes_read,
            scheduler_stats_covered_fragments: shared_scheduler_fragments,
            shared_scheduler_queries,
            shared_scheduler_fragments,
            per_fragment_scheduler_queries,
            per_fragment_scheduler_fragments,
            shared_scheduler_fallback_queries,
            shared_scheduler_fallback_legacy_fragments,
            shared_scheduler_fallback_nonprimary_fragments,
            shared_scheduler_fallback_unsupported_fragments,
            reader_cache_queries: usize::from(use_reader_cache),
            reader_cache_eligible_files: reader_cache_snapshot
                .map(|snapshot| snapshot.eligible_files)
                .unwrap_or(0),
            reader_cache_lookup_files: reader_cache_snapshot
                .map(|snapshot| snapshot.lookup_files)
                .unwrap_or(0),
            reader_cache_hit_files: reader_cache_snapshot
                .map(|snapshot| snapshot.hit_files)
                .unwrap_or(0),
            reader_cache_miss_open_files: reader_cache_snapshot
                .map(|snapshot| snapshot.miss_open_files)
                .unwrap_or(0),
            reader_cache_coalesced_files: reader_cache_snapshot
                .map(|snapshot| snapshot.coalesced_files)
                .unwrap_or(0),
            reader_cache_bypass_files: reader_cache_snapshot
                .map(|snapshot| snapshot.bypass_files)
                .unwrap_or(0),
            reader_cache_fallback_open_files: reader_cache_snapshot
                .map(|snapshot| snapshot.fallback_open_files)
                .unwrap_or(0),
            reader_cache_open_failures: reader_cache_snapshot
                .map(|snapshot| snapshot.open_failures)
                .unwrap_or(0),
            reader_cache_resident_entries_start_approx: reader_cache_snapshot
                .map(|snapshot| snapshot.resident_entries_start_approx)
                .unwrap_or(0),
            reader_cache_resident_entries_end_approx: reader_cache_snapshot
                .map(|snapshot| snapshot.resident_entries_end_approx)
                .unwrap_or(0),
            reader_cache_capacity: reader_cache_snapshot
                .map(|snapshot| snapshot.capacity)
                .unwrap_or(0),
            reader_cache_fd_soft_limit: reader_cache_snapshot
                .map(|snapshot| snapshot.fd_soft_limit)
                .unwrap_or(0),
            reader_cache_lookup_nanos: reader_cache_snapshot
                .map(|snapshot| snapshot.lookup_nanos)
                .unwrap_or(0),
            reader_cache_acquire_nanos: reader_cache_snapshot
                .map(|snapshot| snapshot.acquire_nanos)
                .unwrap_or(0),
            reader_cache_physical_open_nanos: reader_cache_snapshot
                .map(|snapshot| snapshot.physical_open_nanos)
                .unwrap_or(0),
            reader_cache_bind_nanos: reader_cache_snapshot
                .map(|snapshot| snapshot.bind_nanos)
                .unwrap_or(0),
            reader_cache_fallback_queries,
            reader_cache_fallback_legacy_fragments,
            reader_cache_fallback_nonprimary_fragments,
            reader_cache_fallback_nonlocal_fragments,
            reader_cache_fallback_unknown_size_fragments,
            reader_cache_fallback_small_file_fragments,
            reader_cache_fallback_unsupported_fragments,
            reader_cache_fallback_capacity_queries,
        };

        Ok(Some(GroupedPhysicalRead { batches, stats }))
    }

    pub fn is_empty(&self) -> bool {
        match (self.row_ids.as_ref(), self.row_addrs.as_ref()) {
            (Some(ids), _) => ids.is_empty(),
            (_, Some(addrs)) => addrs.is_empty(),
            _ => unreachable!(),
        }
    }

    async fn get_row_addrs(&mut self) -> Result<&Vec<u64>> {
        if self.row_addrs.is_none() {
            let row_ids = self
                .row_ids
                .as_ref()
                .expect("row_ids must be set if row_addrs is not");
            let addrs = if let Some(row_id_index) = get_row_id_index(&self.dataset).await? {
                row_id_index
                    .get_many(row_ids)
                    .into_iter()
                    .filter_map(|opt| opt.map(|address| address.into()))
                    .collect::<Vec<_>>()
            } else {
                row_ids.clone()
            };
            self.row_addrs = Some(addrs);
        }
        Ok(self.row_addrs.as_ref().unwrap())
    }
}

fn take_struct_array(array: &StructArray, indices: &UInt64Array) -> Result<StructArray> {
    let nulls = array.nulls().map(|nulls| {
        let is_valid = indices.iter().map(|index| {
            if let Some(index) = index {
                nulls.is_valid(index.to_usize().unwrap())
            } else {
                false
            }
        });
        NullBuffer::new(BooleanBuffer::new(
            Buffer::from_iter(is_valid),
            0,
            indices.len(),
        ))
    });

    if array.fields().is_empty() {
        return Ok(StructArray::new_empty_fields(indices.len(), nulls));
    }

    let arrays = array
        .columns()
        .iter()
        .map(|array| {
            let array = match array.data_type() {
                arrow::datatypes::DataType::Struct(_) => {
                    Arc::new(take_struct_array(array.as_struct(), indices)?)
                }
                _ => arrow_select::take::take(array, indices, None)?,
            };
            Ok(array)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(StructArray::new(array.fields().clone(), arrays, nulls))
}

#[cfg(test)]
mod test {
    use arrow_array::{
        Int32Array, LargeBinaryArray, ListArray, RecordBatchIterator, StringArray, StructArray,
    };
    use arrow_buffer::{OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Fields, Schema as ArrowSchema};
    use lance_arrow::ARROW_EXT_NAME_KEY;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, is_arrow_json_field};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{ROW_ADDR, ROW_ADDR_FIELD, ROW_ID, ROW_ID_FIELD, ROW_OFFSET};
    use lance_file::version::LanceFileVersion;
    use pretty_assertions::assert_eq;
    use rstest::rstest;
    use std::collections::HashMap;

    use crate::dataset::{WriteParams, scanner::test_dataset::TestVectorDataset};

    use super::*;

    // Used to validate that futures returned are Send.
    fn require_send<T: Send>(t: T) -> T {
        t
    }

    fn test_batch(i_range: Range<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("i", DataType::Int32, false),
            ArrowField::new("s", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(i_range.clone())),
                Arc::new(StringArray::from_iter_values(
                    i_range.map(|i| format!("str-{}", i)),
                )),
            ],
        )
        .unwrap()
    }

    fn nested_arrow_json_batch() -> RecordBatch {
        let uri_field = Arc::new(ArrowField::new("uri", DataType::Utf8, false));
        let mut metadata = HashMap::new();
        metadata.insert(
            ARROW_EXT_NAME_KEY.to_string(),
            ARROW_JSON_EXT_NAME.to_string(),
        );
        let extra_field =
            Arc::new(ArrowField::new("extra", DataType::Utf8, true).with_metadata(metadata));
        let item_fields = Fields::from(vec![uri_field, extra_field]);
        let values = StructArray::new(
            item_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec![Some("a.jpg"), Some("b.jpg")])) as ArrayRef,
                Arc::new(StringArray::from(vec![
                    Some(r#"{"codec":"h264"}"#),
                    None::<&str>,
                ])) as ArrayRef,
            ],
            None,
        );
        let item = Arc::new(ArrowField::new("item", DataType::Struct(item_fields), true));
        let media = ListArray::new(
            item,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1, 2])),
            Arc::new(values),
            None,
        );
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "media",
            media.data_type().clone(),
            true,
        )]));

        RecordBatch::try_new(schema, vec![Arc::new(media) as ArrayRef]).unwrap()
    }

    fn assert_nested_arrow_json_schema(batch: &RecordBatch) {
        let schema = batch.schema();
        let DataType::List(item) = schema.field(0).data_type() else {
            panic!("expected list field");
        };
        let DataType::Struct(fields) = item.data_type() else {
            panic!("expected struct item");
        };
        assert!(is_arrow_json_field(&fields[1]));
    }

    fn assert_first_nested_json_value(batch: &RecordBatch) {
        let media: &ListArray = batch.column(0).as_list();
        let values = media.values().as_struct();
        let uri = values
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let extra = values
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(uri.value(0), "a.jpg");
        assert!(extra.value(0).contains("h264"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_take(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    200, // 200
                    199, // 199
                    39,  // 39
                    40,  // 40
                    199, // 40
                    40,  // 40
                    125, // 125
                ],
                projection,
            )
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([
                        200, 199, 39, 40, 199, 40, 125
                    ])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 199, 40, 125]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_nested_arrow_json_returns_logical_schema(
        #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2, LanceFileVersion::V2_3)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = nested_arrow_json_batch();
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: false,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();

        let values = dataset.take(&[0], projection.clone()).await.unwrap();
        assert_nested_arrow_json_schema(&values);
        assert_first_nested_json_value(&values);

        let empty = dataset.take(&[], projection.clone()).await.unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_nested_arrow_json_schema(&empty);

        let values = dataset.take_rows(&[0], projection.clone()).await.unwrap();
        assert_nested_arrow_json_schema(&values);
        assert_first_nested_json_value(&values);

        let empty = dataset.take_rows(&[], projection).await.unwrap();
        assert_eq!(empty.num_rows(), 0);
        assert_nested_arrow_json_schema(&empty);
    }

    #[rstest]
    #[tokio::test]
    async fn grouped_physical_take_preserves_system_columns_and_nested_json(
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = nested_arrow_json_batch();
        let schema = data.schema();
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(data)], schema),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 1,
                    max_rows_per_group: 1,
                    enable_stable_row_ids,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert_eq!(dataset.get_fragments().len(), 2);
        let addresses = vec![
            u64::from(RowAddress::new_from_parts(0, 0)),
            u64::from(RowAddress::new_from_parts(1, 0)),
        ];
        let projection = Arc::new(
            ProjectionRequest::from_columns(
                ["media", ROW_ID, ROW_ADDR, ROW_OFFSET],
                dataset.schema(),
            )
            .into_projection_plan(dataset.clone())
            .unwrap(),
        );

        let control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .execute()
        .await
        .unwrap();
        let physical = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let stats = physical.stats;
        assert_eq!(stats.fragments, 2);
        assert_eq!(stats.rows, 2);
        assert!(stats.batch_bytes > 0);
        assert_eq!(stats.rows_per_fragment_min, 1);
        assert_eq!(stats.rows_per_fragment_max, 1);
        // The strict control keeps scheduler ownership inside Fragment::open,
        // so no explicit scheduler is available for scoped I/O snapshots.
        assert_eq!(stats.scheduler_stats_covered_fragments, 0);
        assert_eq!(stats.scheduler_scoped_iops, 0);
        assert_eq!(stats.scheduler_scoped_requests, 0);
        assert_eq!(stats.scheduler_scoped_bytes_read, 0);
        assert_eq!(stats.shared_scheduler_queries, 0);
        assert_eq!(stats.shared_scheduler_fragments, 0);
        assert_eq!(stats.per_fragment_scheduler_queries, 1);
        assert_eq!(stats.per_fragment_scheduler_fragments, 2);
        assert_eq!(stats.shared_scheduler_fallback_queries, 0);
        assert_eq!(stats.shared_scheduler_fallback_legacy_fragments, 0);
        assert_eq!(stats.shared_scheduler_fallback_nonprimary_fragments, 0);
        assert_eq!(stats.shared_scheduler_fallback_unsupported_fragments, 0);
        assert!(stats.parent_wall_nanos > 0);
        assert!(stats.plan_nanos > 0);
        assert!(stats.grouping_nanos > 0);
        assert_eq!(stats.scheduler_create_wall_nanos, 0);
        assert!(stats.fragment_open_aggregate_nanos > 0);
        assert!(stats.fragment_open_max_nanos > 0);
        assert!(stats.fragment_read_aggregate_nanos > 0);
        assert!(stats.fragment_read_max_nanos > 0);
        assert!(stats.fragment_total_elapsed_aggregate_nanos > 0);
        assert!(stats.fragment_total_elapsed_max_nanos > 0);
        assert!(stats.fanout_collect_wall_nanos > 0);
        assert_eq!(stats.fanout_concurrency_limit, 2);
        assert!(stats.row_offset_injection_wall_nanos > 0);
        assert!(stats.plan_nanos.saturating_add(stats.grouping_nanos) <= stats.parent_wall_nanos);
        assert!(stats.fragment_total_elapsed_max_nanos <= stats.parent_wall_nanos);

        let first_fragment = dataset.get_fragment(0).unwrap();
        let physical_projection = projection.physical_projection.to_bare_schema();
        assert_eq!(
            first_fragment
                .grouped_shared_scheduler_eligibility(&physical_projection)
                .unwrap(),
            FragmentSharedSchedulerEligibility::Eligible
        );
        let mut nonprimary_metadata = first_fragment.metadata().clone();
        nonprimary_metadata.files[0].base_id = Some(7);
        let nonprimary_fragment = FileFragment::new(dataset.clone(), nonprimary_metadata);
        assert_eq!(
            nonprimary_fragment
                .grouped_shared_scheduler_eligibility(&physical_projection)
                .unwrap(),
            FragmentSharedSchedulerEligibility::NonPrimaryBase
        );
        let mut malformed_metadata = first_fragment.metadata().clone();
        malformed_metadata.files[0].file_major_version = u32::MAX;
        let malformed_fragment = FileFragment::new(dataset.clone(), malformed_metadata);
        assert!(
            malformed_fragment
                .grouped_shared_scheduler_eligibility(&physical_projection)
                .is_err()
        );
        assert_eq!(
            first_fragment
                .grouped_shared_scheduler_eligibility(&Schema::default())
                .unwrap(),
            FragmentSharedSchedulerEligibility::Unsupported
        );
        let mut old_metadata = first_fragment.metadata().clone();
        old_metadata.physical_rows = None;
        let old_fragment = FileFragment::new(dataset.clone(), old_metadata);
        assert_eq!(
            old_fragment
                .grouped_shared_scheduler_eligibility(&physical_projection)
                .unwrap(),
            FragmentSharedSchedulerEligibility::Unsupported
        );
        let mut old_dataset = dataset.as_ref().clone();
        Arc::make_mut(&mut old_dataset.manifest).writer_version = None;
        let old_writer_fragment =
            FileFragment::new(Arc::new(old_dataset), first_fragment.metadata().clone());
        assert_eq!(
            old_writer_fragment
                .grouped_shared_scheduler_eligibility(&physical_projection)
                .unwrap(),
            FragmentSharedSchedulerEligibility::Unsupported
        );

        let shared = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_shared_scheduler(true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(shared.batches, physical.batches);
        assert_eq!(shared.stats.shared_scheduler_queries, 1);
        assert_eq!(shared.stats.shared_scheduler_fragments, 2);
        assert_eq!(shared.stats.per_fragment_scheduler_queries, 0);
        assert_eq!(shared.stats.per_fragment_scheduler_fragments, 0);
        assert_eq!(shared.stats.shared_scheduler_fallback_queries, 0);
        assert_eq!(shared.stats.scheduler_stats_covered_fragments, 2);
        assert!(shared.stats.scheduler_create_wall_nanos > 0);
        assert!(shared.stats.scheduler_scoped_iops > 0);
        assert!(shared.stats.scheduler_scoped_requests > 0);
        assert!(shared.stats.scheduler_scoped_bytes_read > 0);

        let system_projection = Arc::new(
            ProjectionRequest::from_columns([ROW_ID], dataset.schema())
                .into_projection_plan(dataset.clone())
                .unwrap(),
        );
        let system_control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            system_projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let system_treatment = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            system_projection,
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_shared_scheduler(true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(system_treatment.batches, system_control.batches);
        assert_eq!(system_treatment.stats.shared_scheduler_queries, 0);
        assert_eq!(system_treatment.stats.shared_scheduler_fragments, 0);
        assert_eq!(system_treatment.stats.per_fragment_scheduler_queries, 1);
        assert_eq!(system_treatment.stats.per_fragment_scheduler_fragments, 2);
        assert_eq!(system_treatment.stats.shared_scheduler_fallback_queries, 1);
        assert_eq!(
            system_treatment
                .stats
                .shared_scheduler_fallback_unsupported_fragments,
            2
        );
        assert_eq!(system_treatment.stats.scheduler_create_wall_nanos, 0);
        assert_eq!(system_treatment.stats.scheduler_stats_covered_fragments, 0);
        assert_eq!(system_treatment.stats.scheduler_scoped_iops, 0);
        assert_eq!(system_treatment.stats.scheduler_scoped_requests, 0);
        assert_eq!(system_treatment.stats.scheduler_scoped_bytes_read, 0);

        let mut mixed_dataset = dataset.as_ref().clone();
        let mixed_manifest = Arc::make_mut(&mut mixed_dataset.manifest);
        Arc::make_mut(&mut mixed_manifest.fragments)[0].physical_rows = None;
        let mixed_dataset = Arc::new(mixed_dataset);
        let mixed_control = TakeBuilder::try_new_from_addresses(
            mixed_dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let mixed_treatment = TakeBuilder::try_new_from_addresses(
            mixed_dataset,
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_shared_scheduler(true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(mixed_treatment.batches, mixed_control.batches);
        assert_eq!(mixed_treatment.stats.shared_scheduler_queries, 0);
        assert_eq!(mixed_treatment.stats.shared_scheduler_fragments, 0);
        assert_eq!(mixed_treatment.stats.per_fragment_scheduler_queries, 1);
        assert_eq!(mixed_treatment.stats.per_fragment_scheduler_fragments, 2);
        assert_eq!(mixed_treatment.stats.shared_scheduler_fallback_queries, 1);
        assert_eq!(
            mixed_treatment
                .stats
                .shared_scheduler_fallback_unsupported_fragments,
            1
        );
        assert_eq!(mixed_treatment.stats.scheduler_create_wall_nanos, 0);
        assert_eq!(mixed_treatment.stats.scheduler_stats_covered_fragments, 0);
        assert_eq!(mixed_treatment.stats.scheduler_scoped_iops, 0);
        assert_eq!(mixed_treatment.stats.scheduler_scoped_requests, 0);
        assert_eq!(mixed_treatment.stats.scheduler_scoped_bytes_read, 0);
        let physical = physical.batches;
        assert_eq!(physical.len(), 2);
        assert!(physical.iter().all(|batch| batch.num_rows() == 1));
        assert!(physical.iter().all(|batch| {
            batch.column_by_name(ROW_ID).is_some()
                && batch.column_by_name(ROW_ADDR).is_some()
                && batch.column_by_name(ROW_OFFSET).is_some()
        }));
        let combined = concat_batches(&physical[0].schema(), &physical).unwrap();
        let projected = projection.project_batch(combined).await.unwrap();
        let logical = to_logical_json_batch(projected).unwrap();
        assert_eq!(logical, control);
        assert_nested_arrow_json_schema(&logical);
        assert_first_nested_json_value(&logical);
        assert_eq!(
            logical[ROW_OFFSET].as_primitive::<UInt64Type>().values(),
            &[0, 1]
        );
        assert_eq!(
            logical[ROW_ADDR].as_primitive::<UInt64Type>().values(),
            addresses.as_slice()
        );

        let unsorted = vec![addresses[1], addresses[0]];
        assert!(
            TakeBuilder::try_new_from_addresses(dataset.clone(), unsorted, projection.clone())
                .unwrap()
                .read_sorted_physical_by_fragment()
                .await
                .unwrap()
                .is_none()
        );
        let unsorted = vec![addresses[1], addresses[0]];
        assert!(
            TakeBuilder::try_new_from_addresses(dataset.clone(), unsorted, projection.clone())
                .unwrap()
                .read_sorted_physical_by_fragment_with_shared_scheduler(true)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            TakeBuilder::try_new_from_addresses(
                dataset.clone(),
                addresses.clone(),
                projection.clone(),
            )
            .unwrap()
            .with_row_address(true)
            .read_sorted_physical_by_fragment()
            .await
            .unwrap()
            .is_none()
        );
        assert!(
            TakeBuilder::try_new_from_addresses(dataset.clone(), Vec::new(), projection.clone(),)
                .unwrap()
                .read_sorted_physical_by_fragment()
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            TakeBuilder::try_new_from_addresses(dataset.clone(), Vec::new(), projection.clone(),)
                .unwrap()
                .read_sorted_physical_by_fragment_with_shared_scheduler(true)
                .await
                .unwrap()
                .is_none()
        );
        let invalid = vec![u64::from(RowAddress::new_from_parts(99, 0))];
        let error = TakeBuilder::try_new_from_addresses(dataset, invalid, projection)
            .unwrap()
            .read_sorted_physical_by_fragment()
            .await
            .unwrap_err();
        assert!(error.to_string().contains("non-existent fragment"));
    }

    #[tokio::test]
    async fn grouped_shared_scheduler_falls_back_for_legacy_fragments() {
        let data = test_batch(0..4);
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(data.clone())], data.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 2,
                    max_rows_per_group: 2,
                    data_storage_version: Some(LanceFileVersion::Legacy),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        assert_eq!(dataset.get_fragments().len(), 2);
        let addresses = vec![
            u64::from(RowAddress::new_from_parts(0, 0)),
            u64::from(RowAddress::new_from_parts(1, 0)),
        ];
        let projection = Arc::new(
            ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema())
                .into_projection_plan(dataset.clone())
                .unwrap(),
        );
        let control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let treatment = TakeBuilder::try_new_from_addresses(dataset, addresses, projection)
            .unwrap()
            .read_sorted_physical_by_fragment_with_shared_scheduler(true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(treatment.batches, control.batches);
        assert_eq!(treatment.stats.shared_scheduler_queries, 0);
        assert_eq!(treatment.stats.shared_scheduler_fragments, 0);
        assert_eq!(treatment.stats.per_fragment_scheduler_queries, 1);
        assert_eq!(treatment.stats.per_fragment_scheduler_fragments, 2);
        assert_eq!(treatment.stats.shared_scheduler_fallback_queries, 1);
        assert_eq!(
            treatment.stats.shared_scheduler_fallback_legacy_fragments,
            2
        );
        assert_eq!(
            treatment
                .stats
                .shared_scheduler_fallback_nonprimary_fragments,
            0
        );
        assert_eq!(
            treatment
                .stats
                .shared_scheduler_fallback_unsupported_fragments,
            0
        );
        assert_eq!(treatment.stats.scheduler_create_wall_nanos, 0);
        assert_eq!(treatment.stats.scheduler_stats_covered_fragments, 0);
        assert_eq!(treatment.stats.scheduler_scoped_iops, 0);
        assert_eq!(treatment.stats.scheduler_scoped_requests, 0);
        assert_eq!(treatment.stats.scheduler_scoped_bytes_read, 0);
    }

    #[tokio::test]
    async fn grouped_data_file_reader_cache_cold_warm_and_fresh_deletions() {
        let test_dir = TempStrDir::default();
        let data = test_batch(0..2_000);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 1_000,
                max_rows_per_group: 128,
                data_storage_version: Some(LanceFileVersion::V2_1),
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        let addresses = vec![
            u64::from(RowAddress::new_from_parts(0, 0)),
            u64::from(RowAddress::new_from_parts(1, 0)),
        ];
        let dataset = Arc::new(dataset);
        let projection = Arc::new(
            ProjectionRequest::from_columns(["i", "s", ROW_ID], dataset.schema())
                .into_projection_plan(dataset.clone())
                .unwrap(),
        );
        let physical_projection = projection.physical_projection.to_bare_schema();
        for fragment in dataset.get_fragments() {
            assert_eq!(
                fragment
                    .grouped_reader_cache_eligibility(&physical_projection)
                    .unwrap(),
                FragmentReaderCacheEligibility::Eligible { files: 1 }
            );
        }

        let control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        assert_eq!(control.stats.reader_cache_queries, 0);

        let cold = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_options(false, true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(cold.batches, control.batches);
        assert_eq!(cold.stats.reader_cache_queries, 1);
        assert_eq!(cold.stats.reader_cache_eligible_files, 2);
        assert_eq!(cold.stats.reader_cache_lookup_files, 2);
        assert_eq!(cold.stats.reader_cache_hit_files, 0);
        assert_eq!(cold.stats.reader_cache_miss_open_files, 2);
        assert_eq!(cold.stats.reader_cache_coalesced_files, 0);
        assert_eq!(cold.stats.reader_cache_bypass_files, 0);
        assert_eq!(cold.stats.reader_cache_fallback_open_files, 0);
        assert_eq!(cold.stats.reader_cache_open_failures, 0);
        assert_eq!(cold.stats.reader_cache_fallback_queries, 0);
        assert!(cold.stats.reader_cache_physical_open_nanos > 0);

        let warm = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_options(false, true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(warm.batches, control.batches);
        assert_eq!(warm.stats.reader_cache_queries, 1);
        assert_eq!(warm.stats.reader_cache_lookup_files, 2);
        assert_eq!(warm.stats.reader_cache_hit_files, 2);
        assert_eq!(warm.stats.reader_cache_miss_open_files, 0);
        assert_eq!(warm.stats.reader_cache_physical_open_nanos, 0);

        // Mutating the manifest must not make snapshot-scoped deletion state
        // stale even though the underlying immutable data-file reader is hot.
        let mut deleted_dataset = dataset.as_ref().clone();
        deleted_dataset.delete("i = 0").await.unwrap();
        let deleted_dataset = Arc::new(deleted_dataset);
        let deleted_projection = Arc::new(
            ProjectionRequest::from_columns(["i", "s", ROW_ID], deleted_dataset.schema())
                .into_projection_plan(deleted_dataset.clone())
                .unwrap(),
        );
        let deleted_control = TakeBuilder::try_new_from_addresses(
            deleted_dataset.clone(),
            addresses.clone(),
            deleted_projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let deleted_cached = TakeBuilder::try_new_from_addresses(
            deleted_dataset.clone(),
            addresses,
            deleted_projection,
        )
        .unwrap()
        .read_sorted_physical_by_fragment_with_options(false, true)
        .await
        .unwrap()
        .unwrap();
        assert_eq!(deleted_cached.batches, deleted_control.batches);
        assert_eq!(
            deleted_cached
                .batches
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>(),
            1
        );
        assert_eq!(deleted_cached.stats.reader_cache_hit_files, 2);

        let first_fragment = deleted_dataset.get_fragment(0).unwrap();
        let mut unknown_size_metadata = first_fragment.metadata().clone();
        unknown_size_metadata.files[0].file_size_bytes = lance_io::utils::CachedFileSize::unknown();
        assert_eq!(
            FileFragment::new(deleted_dataset.clone(), unknown_size_metadata)
                .grouped_reader_cache_eligibility(&physical_projection)
                .unwrap(),
            FragmentReaderCacheEligibility::UnknownSize
        );
        let mut small_file_metadata = first_fragment.metadata().clone();
        small_file_metadata.files[0].file_size_bytes = lance_io::utils::CachedFileSize::new(1);
        assert_eq!(
            FileFragment::new(deleted_dataset, small_file_metadata)
                .grouped_reader_cache_eligibility(&physical_projection)
                .unwrap(),
            FragmentReaderCacheEligibility::SmallFile
        );
    }

    #[tokio::test]
    async fn grouped_data_file_reader_cache_falls_back_for_memory_store() {
        let data = test_batch(0..4);
        let dataset = Arc::new(
            Dataset::write(
                RecordBatchIterator::new([Ok(data.clone())], data.schema()),
                "memory://",
                Some(WriteParams {
                    max_rows_per_file: 2,
                    max_rows_per_group: 2,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        let addresses = vec![
            u64::from(RowAddress::new_from_parts(0, 0)),
            u64::from(RowAddress::new_from_parts(1, 0)),
        ];
        let projection = Arc::new(
            ProjectionRequest::from_columns(["i", ROW_ID], dataset.schema())
                .into_projection_plan(dataset.clone())
                .unwrap(),
        );
        let control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let fallback = TakeBuilder::try_new_from_addresses(dataset, addresses, projection)
            .unwrap()
            .read_sorted_physical_by_fragment_with_options(false, true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fallback.batches, control.batches);
        assert_eq!(fallback.stats.reader_cache_queries, 0);
        assert_eq!(fallback.stats.reader_cache_fallback_queries, 1);
        assert_eq!(fallback.stats.reader_cache_fallback_nonlocal_fragments, 2);
        assert_eq!(fallback.stats.reader_cache_lookup_files, 0);
    }

    #[tokio::test]
    async fn grouped_shared_scheduler_preserves_deletions_and_stable_row_ids() {
        let data = test_batch(0..6);
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(data.clone())], data.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                max_rows_per_group: 2,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.delete("i IN (1, 4)").await.unwrap();
        let dataset = Arc::new(dataset);
        assert_eq!(dataset.get_fragments().len(), 3);
        let addresses = vec![
            u64::from(RowAddress::new_from_parts(0, 0)),
            u64::from(RowAddress::new_from_parts(1, 1)),
            u64::from(RowAddress::new_from_parts(2, 1)),
        ];
        let projection = Arc::new(
            ProjectionRequest::from_columns(["i", ROW_ID, ROW_ADDR, ROW_OFFSET], dataset.schema())
                .into_projection_plan(dataset.clone())
                .unwrap(),
        );
        let control = TakeBuilder::try_new_from_addresses(
            dataset.clone(),
            addresses.clone(),
            projection.clone(),
        )
        .unwrap()
        .read_sorted_physical_by_fragment()
        .await
        .unwrap()
        .unwrap();
        let treatment = TakeBuilder::try_new_from_addresses(dataset, addresses, projection)
            .unwrap()
            .read_sorted_physical_by_fragment_with_shared_scheduler(true)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(treatment.batches, control.batches);
        assert_eq!(treatment.stats.shared_scheduler_queries, 1);
        assert_eq!(treatment.stats.shared_scheduler_fragments, 3);
        assert_eq!(treatment.stats.scheduler_stats_covered_fragments, 3);
        assert_eq!(treatment.stats.shared_scheduler_fallback_queries, 0);
        assert!(treatment.stats.scheduler_scoped_iops > 0);
        assert!(treatment.stats.scheduler_scoped_requests > 0);
        assert!(treatment.stats.scheduler_scoped_bytes_read > 0);
    }

    #[tokio::test]
    async fn test_take_with_deletion() {
        let data = test_batch(0..120);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (40, 77, 78, 79)").await.unwrap();

        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let values = dataset
            .take(
                &[
                    0,   // 0
                    39,  // 39
                    40,  // 41
                    75,  // 76
                    76,  // 80
                    77,  // 81
                    115, // 119
                ],
                projection,
            )
            .await
            .unwrap();

        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([0, 39, 41, 76, 80, 81, 119])),
                    Arc::new(StringArray::from_iter_values(
                        [0, 39, 41, 76, 80, 81, 119]
                            .iter()
                            .map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_with_projection(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![("foo", "i"), ("bar", "i*2")]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset.take_rows(&[10, 50, 100], projection).await.unwrap();
        assert_eq!(values, values2);
    }

    #[tokio::test]
    async fn test_reject_legacy_blob_schema_on_v2_2() {
        let mut metadata = HashMap::new();
        metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(metadata),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeBinaryArray::from(vec![Some(
                b"hello".as_slice(),
            )]))],
        )
        .unwrap();

        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let err = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Legacy blob columns"));
        assert!(msg.contains("lance.blob.v2"));
    }

    #[tokio::test]
    async fn test_take_blob_v2_from_blob_v2_struct_on_v2_2() {
        let schema = Arc::new(ArrowSchema::new(vec![crate::blob::blob_field(
            "blob", true,
        )]));
        let mut builder = crate::blob::BlobArrayBuilder::new(1);
        builder.push_bytes(b"hello").unwrap();
        let array = builder.finish().unwrap();

        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = crate::dataset::write::InsertBuilder::new("memory://")
            .with_params(&write_params)
            .execute_stream(batches)
            .await
            .unwrap();

        let proj = ProjectionRequest::from_columns(["blob"], dataset.schema());
        let values = dataset.take(&[0u64], proj).await.unwrap();

        let struct_arr = values.column(0).as_struct();
        assert_eq!(struct_arr.fields().len(), 5);
        assert_eq!(struct_arr.fields()[0].name(), "kind");
        assert_eq!(struct_arr.fields()[1].name(), "position");
        assert_eq!(struct_arr.fields()[2].name(), "size");
        assert_eq!(struct_arr.fields()[3].name(), "blob_id");
        assert_eq!(struct_arr.fields()[4].name(), "blob_uri");
    }

    #[tokio::test]
    async fn test_projection_plan_accepts_unloaded_legacy_blob_schema() {
        let mut metadata = HashMap::new();
        metadata.insert(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string());
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("blob", DataType::LargeBinary, true).with_metadata(metadata),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeBinaryArray::from(vec![Some(
                b"hello".as_slice(),
            )]))],
        )
        .unwrap();
        let write_params = WriteParams {
            data_storage_version: Some(LanceFileVersion::Legacy),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        let mut projection = dataset.schema().project(&["blob"]).unwrap();
        projection.fields[0].unloaded_mut();

        let projection = ProjectionRequest::from_schema(projection)
            .into_projection_plan(Arc::new(dataset))
            .unwrap();

        let output_schema = projection.output_schema().unwrap();
        let blob_field = output_schema.field_with_name("blob").unwrap();
        let DataType::Struct(fields) = blob_field.data_type() else {
            panic!("expected blob output schema to be a struct, got {blob_field:?}");
        };
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), "position");
        assert_eq!(fields[1].name(), "size");
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_enable_stable_row_ids_projection_from_sql(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![
            ("foo", "i"),
            ("bar", "i*2"),
            ("_rowid", "_rowid"),
            ("_rowaddr", "_rowaddr"),
        ]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                    Arc::new(UInt64Array::from_iter_values([10, 50, 100])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset.take_rows(&[10, 50, 100], projection).await.unwrap();
        assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_enable_stable_row_ids_projection_from_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection =
            ProjectionRequest::from_columns(["_rowid", "_rowaddr", "i"], dataset.schema());

        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
            ArrowField::new("i", DataType::Int32, false),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from_iter_values([10, 50, 100])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset
            .take_rows(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        assert_eq!(values, values2);

        let values3 = dataset
            .take(&[50, 100, 10], projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(UInt64Array::from_iter_values([50, 100, 10])),
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(Int32Array::from_iter_values([50, 100, 10])),
                ],
            )
            .unwrap(),
            values3
        );
        let values4 = dataset.take_rows(&[50, 100, 10], projection).await.unwrap();
        assert_eq!(values3, values4);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_disable_stable_row_ids_projection_from_sql(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: false,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = ProjectionRequest::from_sql(vec![
            ("foo", "i"),
            ("bar", "i*2"),
            ("_rowid", "_rowid"),
            ("_rowaddr", "_rowaddr"),
        ]);
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();
        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("foo", DataType::Int32, false),
            ArrowField::new("bar", DataType::Int32, false),
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
        ]));
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                    Arc::new(Int32Array::from_iter_values([20, 100, 200])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                ],
            )
            .unwrap(),
            values
        );

        let values2 = dataset
            .take_rows(&[10, 4294967296, 8589934592], projection)
            .await
            .unwrap();
        assert_eq!(values, values2);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rowid_rowaddr_with_projection_disable_stable_row_ids_projection_from_columns(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: false,
            max_rows_per_file: 50,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);

        let projection =
            ProjectionRequest::from_columns(["_rowid", "_rowaddr", "i"], dataset.schema());
        let values = dataset
            .take(&[10, 50, 100], projection.clone())
            .await
            .unwrap();

        let expected_schema = Arc::new(ArrowSchema::new(vec![
            ROW_ID_FIELD.clone(),
            ROW_ADDR_FIELD.clone(),
            ArrowField::new("i", DataType::Int32, false),
        ]));

        assert_eq!(
            RecordBatch::try_new(
                expected_schema.clone(),
                vec![
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(UInt64Array::from_iter_values([10, 4294967296, 8589934592])),
                    Arc::new(Int32Array::from_iter_values([10, 50, 100])),
                ],
            )
            .unwrap(),
            values
        );
        let values2 = dataset
            .take_rows(&[10, 4294967296, 8589934592], projection.clone())
            .await
            .unwrap();
        assert_eq!(values, values2);

        let values3 = dataset
            .take(&[50, 100, 10], projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                expected_schema,
                vec![
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(UInt64Array::from_iter_values([4294967296, 8589934592, 10])),
                    Arc::new(Int32Array::from_iter_values([50, 100, 10])),
                ],
            )
            .unwrap(),
            values3
        );
        let values4 = dataset
            .take_rows(&[4294967296, 8589934592, 10], projection)
            .await
            .unwrap();
        assert_eq!(values3, values4);
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_out_of_bound(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // a dataset with 1 fragment and 400 rows
        let test_ds = TestVectorDataset::new(data_storage_version, false)
            .await
            .unwrap();
        let ds = test_ds.dataset;

        // take the last row of first fragment
        // this triggers the contiguous branch
        let indices = &[(1 << 32) - 1];
        let fut = require_send(ds.take_rows(indices, ds.schema().clone()));
        let err = fut.await.unwrap_err();
        assert!(
            err.to_string().contains("Invalid read params"),
            "{}",
            err.to_string()
        );

        // this triggers the sorted branch, but not contiguous
        let indices = &[(1 << 32) - 3, (1 << 32) - 1];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );

        // this triggers the catch all branch
        let indices = &[(1 << 32) - 1, (1 << 32) - 3];
        let err = ds
            .take_rows(indices, ds.schema().clone())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("Invalid read params Indices(4294967293,4294967295)"),
            "{}",
            err.to_string()
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..400);
        let write_params = WriteParams {
            max_rows_per_file: 40,
            max_rows_per_group: 10,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.count_rows(None).await.unwrap(), 400);
        let projection = Schema::try_from(data.schema().as_ref()).unwrap();
        let indices = &[
            5_u64 << 32,        // 200
            (4_u64 << 32) + 39, // 199
            39,                 // 39
            1_u64 << 32,        // 40
            (2_u64 << 32) + 20, // 100
        ];
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 199, 39, 40, 100])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 199, 39, 40, 100].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Delete some rows from a fragment
        dataset.delete("i in (199, 100)").await.unwrap();
        dataset.validate().await.unwrap();
        let values = dataset
            .take_rows(indices, projection.clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values([200, 39, 40])),
                    Arc::new(StringArray::from_iter_values(
                        [200, 39, 40].iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            values
        );

        // Take an empty selection.
        let values = dataset.take_rows(&[], projection).await.unwrap();
        assert_eq!(RecordBatch::new_empty(data.schema()), values);
    }

    #[rstest]
    #[tokio::test]
    async fn take_scan_dataset(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        use arrow::datatypes::Int32Type;

        let data = test_batch(1..5);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        let projection = Arc::new(dataset.schema().project(&["i"]).unwrap());
        let ranges = [0_u64..3, 1..4, 0..1];
        let range_stream = futures::stream::iter(ranges).map(Ok).boxed();
        let results = dataset
            .take_scan(range_stream, projection.clone(), 10)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let expected_schema = projection.as_ref().into();
        for batch in &results {
            assert_eq!(batch.schema().as_ref(), &expected_schema);
        }
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].column(0).as_primitive::<Int32Type>().values(),
            &[1, 2, 3],
        );
        assert_eq!(
            results[1].column(0).as_primitive::<Int32Type>().values(),
            &[2, 3, 4],
        );
        assert_eq!(
            results[2].column(0).as_primitive::<Int32Type>().values(),
            &[1],
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_take_rows_with_row_ids(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let data = test_batch(0..8);
        let write_params = WriteParams {
            max_rows_per_group: 2,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let batches = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        let mut dataset = Dataset::write(batches, "memory://", Some(write_params))
            .await
            .unwrap();

        dataset.delete("i in (1, 2, 3, 7)").await.unwrap();

        let indices = &[0, 4, 6, 5];
        let result = dataset
            .take_rows(indices, dataset.schema().clone())
            .await
            .unwrap();
        assert_eq!(
            RecordBatch::try_new(
                data.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values(
                        indices.iter().map(|x| *x as i32)
                    )),
                    Arc::new(StringArray::from_iter_values(
                        indices.iter().map(|v| format!("str-{v}"))
                    )),
                ],
            )
            .unwrap(),
            result
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Session-local cache for already-open Lance data-file readers.
//!
//! This cache deliberately stops at [`Reader`].  Query-scoped schedulers,
//! projections, deletion vectors, and stable row-id sequences are rebuilt for
//! every fragment open, so keeping a reader here cannot make snapshot state
//! stale.  The cache is currently used only by the opt-in PLAID exact
//! refinement path and only for ordinary local files.

use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures::future::BoxFuture;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::traits::{ByteStream, Reader};
use moka::future::Cache;
use object_store::path::Path;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const DEFAULT_MAX_ENTRIES: u64 = 128;
const DEFAULT_MAX_IN_FLIGHT_OPENS: usize = 64;
const DEFAULT_PROCESS_MAX_CACHED_READERS: usize = 4096;
const DEFAULT_TIME_TO_IDLE: Duration = Duration::from_secs(5 * 60);

/// One process-wide budget prevents multiple Sessions plus query-held evicted
/// entries from multiplying cache-owned file descriptors without bound.  A
/// failed non-blocking admission simply falls back to Lance's ordinary open
/// path, leaving at least three quarters of RLIMIT_NOFILE for transient I/O and
/// unrelated process state.
#[derive(Debug)]
struct GlobalReaderFdBudget {
    permits: Arc<Semaphore>,
    max_cached_readers: usize,
    fd_soft_limit: u64,
}

fn global_reader_fd_budget() -> &'static GlobalReaderFdBudget {
    static BUDGET: OnceLock<GlobalReaderFdBudget> = OnceLock::new();
    BUDGET.get_or_init(|| {
        let fd_soft_limit = fd_soft_limit();
        let fd_limited_readers = if fd_soft_limit == u64::MAX {
            DEFAULT_PROCESS_MAX_CACHED_READERS
        } else {
            usize::try_from(fd_soft_limit / 4).unwrap_or(DEFAULT_PROCESS_MAX_CACHED_READERS)
        };
        let max_cached_readers = DEFAULT_PROCESS_MAX_CACHED_READERS.min(fd_limited_readers);
        GlobalReaderFdBudget {
            permits: Arc::new(Semaphore::new(max_cached_readers)),
            max_cached_readers,
            fd_soft_limit,
        }
    })
}

/// Delegating reader that keeps one process-wide cached-FD permit alive for
/// exactly as long as either the cache or an in-flight query retains the reader.
#[derive(Debug)]
struct FdBudgetedReader {
    inner: Arc<dyn Reader>,
    _fd_permit: OwnedSemaphorePermit,
}

impl DeepSizeOf for FdBudgetedReader {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        // The inner reader owns shared object-store tracking state and this
        // cache is entry-count bounded rather than byte weighted.
        0
    }
}

impl Reader for FdBudgetedReader {
    fn path(&self) -> &Path {
        self.inner.path()
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        self.inner.size()
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        self.inner.get_range(range)
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        self.inner.get_all()
    }

    fn get_stream(&self) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        self.inner.get_stream()
    }

    fn get_range_stream(
        &self,
        range: Range<usize>,
    ) -> BoxFuture<'_, object_store::Result<ByteStream>> {
        self.inner.get_range_stream(range)
    }
}

/// An immutable identity for one physical Lance data file.
///
/// Retaining the object-store `Arc` is intentional.  A local reader captures
/// store-specific I/O tracking state, so two stores with the same textual
/// prefix must not accidentally share a reader.
#[derive(Debug, Clone)]
pub(crate) struct DataFileReaderCacheKey {
    store: Arc<ObjectStore>,
    path: Path,
    known_size: usize,
    file_major_version: u32,
    file_minor_version: u32,
}

impl DataFileReaderCacheKey {
    pub(crate) fn new(
        store: Arc<ObjectStore>,
        path: Path,
        known_size: usize,
        file_major_version: u32,
        file_minor_version: u32,
    ) -> Self {
        Self {
            store,
            path,
            known_size,
            file_major_version,
            file_minor_version,
        }
    }
}

impl PartialEq for DataFileReaderCacheKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.store, &other.store)
            && self.path == other.path
            && self.known_size == other.known_size
            && self.file_major_version == other.file_major_version
            && self.file_minor_version == other.file_minor_version
    }
}

impl Eq for DataFileReaderCacheKey {}

impl Hash for DataFileReaderCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.store) as usize).hash(state);
        self.path.hash(state);
        self.known_size.hash(state);
        self.file_major_version.hash(state);
        self.file_minor_version.hash(state);
    }
}

/// Per-query counters shared by concurrently-opened fragments.
#[derive(Debug)]
pub(crate) struct ReaderCacheQueryStats {
    resident_entries_start_approx: u64,
    lookup_files: AtomicU64,
    hit_files: AtomicU64,
    miss_open_files: AtomicU64,
    coalesced_files: AtomicU64,
    bypass_files: AtomicU64,
    fallback_open_files: AtomicU64,
    open_failures: AtomicU64,
    fd_budget_rejections: AtomicU64,
    lookup_nanos: AtomicU64,
    get_or_open_nanos: AtomicU64,
    coalesced_wait_nanos: AtomicU64,
    acquire_nanos: AtomicU64,
    physical_open_nanos: AtomicU64,
    bind_nanos: AtomicU64,
}

impl ReaderCacheQueryStats {
    fn new(resident_entries_start_approx: u64) -> Self {
        Self {
            resident_entries_start_approx,
            lookup_files: AtomicU64::new(0),
            hit_files: AtomicU64::new(0),
            miss_open_files: AtomicU64::new(0),
            coalesced_files: AtomicU64::new(0),
            bypass_files: AtomicU64::new(0),
            fallback_open_files: AtomicU64::new(0),
            open_failures: AtomicU64::new(0),
            fd_budget_rejections: AtomicU64::new(0),
            lookup_nanos: AtomicU64::new(0),
            get_or_open_nanos: AtomicU64::new(0),
            coalesced_wait_nanos: AtomicU64::new(0),
            acquire_nanos: AtomicU64::new(0),
            physical_open_nanos: AtomicU64::new(0),
            bind_nanos: AtomicU64::new(0),
        }
    }

    pub(crate) fn add_bind_duration(&self, duration: Duration) {
        self.bind_nanos
            .fetch_add(duration_nanos(duration), Ordering::Relaxed);
    }

    pub(crate) fn record_bypass(&self) {
        self.bypass_files.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_fallback_open(&self) {
        self.fallback_open_files.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self, resident_entries_end_approx: u64) -> ReaderCacheQueryStatsSnapshot {
        ReaderCacheQueryStatsSnapshot {
            resident_entries_start_approx: self.resident_entries_start_approx,
            resident_entries_end_approx,
            lookup_files: self.lookup_files.load(Ordering::Relaxed),
            hit_files: self.hit_files.load(Ordering::Relaxed),
            miss_open_files: self.miss_open_files.load(Ordering::Relaxed),
            coalesced_files: self.coalesced_files.load(Ordering::Relaxed),
            bypass_files: self.bypass_files.load(Ordering::Relaxed),
            fallback_open_files: self.fallback_open_files.load(Ordering::Relaxed),
            open_failures: self.open_failures.load(Ordering::Relaxed),
            fd_budget_rejections: self.fd_budget_rejections.load(Ordering::Relaxed),
            lookup_nanos: self.lookup_nanos.load(Ordering::Relaxed),
            get_or_open_nanos: self.get_or_open_nanos.load(Ordering::Relaxed),
            coalesced_wait_nanos: self.coalesced_wait_nanos.load(Ordering::Relaxed),
            acquire_nanos: self.acquire_nanos.load(Ordering::Relaxed),
            physical_open_nanos: self.physical_open_nanos.load(Ordering::Relaxed),
            bind_nanos: self.bind_nanos.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReaderCacheQueryStatsSnapshot {
    /// Moka's eventually-maintained diagnostic entry count at query start.
    pub resident_entries_start_approx: u64,
    /// Moka's eventually-maintained diagnostic entry count at query finish.
    pub resident_entries_end_approx: u64,
    pub lookup_files: u64,
    pub hit_files: u64,
    pub miss_open_files: u64,
    pub coalesced_files: u64,
    pub bypass_files: u64,
    pub fallback_open_files: u64,
    pub open_failures: u64,
    pub fd_budget_rejections: u64,
    pub lookup_nanos: u64,
    pub get_or_open_nanos: u64,
    pub coalesced_wait_nanos: u64,
    pub acquire_nanos: u64,
    pub physical_open_nanos: u64,
    pub bind_nanos: u64,
}

/// Bounded, session-local cache of bare data-file readers.
#[derive(Clone)]
pub(crate) struct DataFileReaderCache {
    inner: Cache<DataFileReaderCacheKey, Arc<dyn Reader>>,
    open_permits: Arc<Semaphore>,
    fd_permits: Arc<Semaphore>,
    capacity: u64,
    fd_soft_limit: u64,
}

impl std::fmt::Debug for DataFileReaderCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DataFileReaderCache")
            .field("entries", &self.inner.entry_count())
            .field("capacity", &self.capacity)
            .field("fd_soft_limit", &self.fd_soft_limit)
            .finish()
    }
}

impl Default for DataFileReaderCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DataFileReaderCache {
    pub(crate) fn new() -> Self {
        let fd_budget = global_reader_fd_budget();
        let capacity = DEFAULT_MAX_ENTRIES.min(fd_budget.max_cached_readers as u64);
        let max_in_flight = DEFAULT_MAX_IN_FLIGHT_OPENS.min(capacity as usize).max(1);
        Self {
            inner: Cache::builder()
                .max_capacity(capacity)
                .time_to_idle(DEFAULT_TIME_TO_IDLE)
                .support_invalidation_closures()
                .build(),
            open_permits: Arc::new(Semaphore::new(max_in_flight)),
            fd_permits: fd_budget.permits.clone(),
            capacity,
            fd_soft_limit: fd_budget.fd_soft_limit,
        }
    }

    pub(crate) fn resident_entries_approx(&self) -> u64 {
        self.inner.entry_count()
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(crate) fn fd_soft_limit(&self) -> u64 {
        self.fd_soft_limit
    }

    /// Conservative whole-query admission threshold, not a concurrency limit.
    pub(crate) fn max_eligible_files_per_query(&self) -> usize {
        DEFAULT_MAX_IN_FLIGHT_OPENS.min(self.capacity as usize)
    }

    pub(crate) fn begin_query(&self) -> Arc<ReaderCacheQueryStats> {
        Arc::new(ReaderCacheQueryStats::new(self.resident_entries_approx()))
    }

    pub(crate) fn finish_query(
        &self,
        stats: &ReaderCacheQueryStats,
    ) -> ReaderCacheQueryStatsSnapshot {
        stats.snapshot(self.resident_entries_approx())
    }

    /// Stop the cache itself from retaining a file successfully removed by
    /// dataset cleanup.  An in-flight query may keep its Arc alive until that
    /// read completes, which is required for safe Unix unlink semantics.
    pub(crate) async fn invalidate_store_path(
        &self,
        store: &Arc<ObjectStore>,
        path: &Path,
    ) -> Result<()> {
        let store = store.clone();
        let path = path.clone();
        self.inner
            .invalidate_entries_if(move |key, _reader| {
                Arc::ptr_eq(&key.store, &store) && key.path == path
            })
            .map_err(|error| {
                Error::internal(format!(
                    "failed to invalidate cleaned data-file reader: {error}"
                ))
            })?;
        self.inner.run_pending_tasks().await;
        Ok(())
    }

    /// Return a cached reader or single-flight one physical local-file open.
    ///
    /// Callers deliberately fall back to the ordinary database open path on
    /// error, so the cloned Moka loader error is diagnostic rather than the
    /// final user-visible failure.
    pub(crate) async fn get_or_open(
        &self,
        key: DataFileReaderCacheKey,
        stats: &Arc<ReaderCacheQueryStats>,
    ) -> Result<Arc<dyn Reader>> {
        let total_started = Instant::now();
        stats.lookup_files.fetch_add(1, Ordering::Relaxed);
        let lookup_started = Instant::now();
        if let Some(reader) = self.inner.get(&key).await {
            stats
                .lookup_nanos
                .fetch_add(elapsed_nanos(lookup_started), Ordering::Relaxed);
            stats.hit_files.fetch_add(1, Ordering::Relaxed);
            stats
                .get_or_open_nanos
                .fetch_add(elapsed_nanos(total_started), Ordering::Relaxed);
            return Ok(reader);
        }
        stats
            .lookup_nanos
            .fetch_add(elapsed_nanos(lookup_started), Ordering::Relaxed);

        let loader_ran = Arc::new(AtomicBool::new(false));
        let permit_acquire_nanos = Arc::new(AtomicU64::new(0));
        let physical_open_nanos = Arc::new(AtomicU64::new(0));
        let load_store = key.store.clone();
        let load_path = key.path.clone();
        let known_size = key.known_size;
        let open_permits = self.open_permits.clone();
        let fd_permits = self.fd_permits.clone();
        let loader_ran_for_future = loader_ran.clone();
        let acquire_nanos_for_future = permit_acquire_nanos.clone();
        let open_nanos_for_future = physical_open_nanos.clone();
        let stats_for_future = stats.clone();
        let coordination_started = Instant::now();
        let result = self
            .inner
            .try_get_with(key, async move {
                loader_ran_for_future.store(true, Ordering::Relaxed);
                let acquire_started = Instant::now();
                let _open_permit = open_permits.acquire_owned().await.map_err(|_| {
                    Error::internal("data-file reader cache admission semaphore closed")
                })?;
                let fd_permit = match fd_permits.try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        stats_for_future
                            .fd_budget_rejections
                            .fetch_add(1, Ordering::Relaxed);
                        return Err(Error::internal(
                            "process-wide data-file reader cache FD budget exhausted",
                        ));
                    }
                };
                acquire_nanos_for_future.store(elapsed_nanos(acquire_started), Ordering::Relaxed);
                let open_started = Instant::now();
                let result = load_store.open_with_size(&load_path, known_size).await;
                open_nanos_for_future.store(elapsed_nanos(open_started), Ordering::Relaxed);
                let reader = Arc::<dyn Reader>::from(result?);
                Ok(Arc::new(FdBudgetedReader {
                    inner: reader,
                    _fd_permit: fd_permit,
                }) as Arc<dyn Reader>)
            })
            .await;
        let coordination_nanos = elapsed_nanos(coordination_started);
        stats
            .get_or_open_nanos
            .fetch_add(elapsed_nanos(total_started), Ordering::Relaxed);
        let loader_ran = loader_ran.load(Ordering::Relaxed);
        if loader_ran {
            stats.miss_open_files.fetch_add(1, Ordering::Relaxed);
            stats.acquire_nanos.fetch_add(
                permit_acquire_nanos.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
            stats.physical_open_nanos.fetch_add(
                physical_open_nanos.load(Ordering::Relaxed),
                Ordering::Relaxed,
            );
        } else {
            stats.coalesced_files.fetch_add(1, Ordering::Relaxed);
            stats
                .coalesced_wait_nanos
                .fetch_add(coordination_nanos, Ordering::Relaxed);
        }
        result.map_err(|error: Arc<Error>| {
            if loader_ran {
                stats.open_failures.fetch_add(1, Ordering::Relaxed);
            }
            Error::cloned(error.to_string())
        })
    }
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn elapsed_nanos(started: Instant) -> u64 {
    duration_nanos(started.elapsed())
}

#[cfg(unix)]
fn fd_soft_limit() -> u64 {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` points to writable storage for one `rlimit` value and
    // `RLIMIT_NOFILE` is valid on every Unix target supported by libc.
    let status = unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) };
    if status == 0 {
        u64::try_from(limit.rlim_cur).unwrap_or(u64::MAX)
    } else {
        u64::MAX
    }
}

#[cfg(not(unix))]
fn fd_soft_limit() -> u64 {
    u64::MAX
}

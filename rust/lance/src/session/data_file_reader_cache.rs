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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use lance_core::{Error, Result};
use lance_io::object_store::ObjectStore;
use lance_io::traits::Reader;
use moka::future::Cache;
use object_store::path::Path;
use tokio::sync::Semaphore;

const DEFAULT_MAX_ENTRIES: u64 = 128;
const DEFAULT_MAX_IN_FLIGHT_OPENS: usize = 64;
const DEFAULT_TIME_TO_IDLE: Duration = Duration::from_secs(5 * 60);

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
    eligible_files: u64,
    resident_entries_start: u64,
    capacity: u64,
    fd_soft_limit: u64,
    lookup_files: AtomicU64,
    hit_files: AtomicU64,
    miss_open_files: AtomicU64,
    coalesced_files: AtomicU64,
    bypass_files: AtomicU64,
    fallback_open_files: AtomicU64,
    open_failures: AtomicU64,
    lookup_nanos: AtomicU64,
    acquire_nanos: AtomicU64,
    physical_open_nanos: AtomicU64,
    bind_nanos: AtomicU64,
}

impl ReaderCacheQueryStats {
    fn new(
        eligible_files: usize,
        resident_entries_start: u64,
        capacity: u64,
        fd_soft_limit: u64,
    ) -> Self {
        Self {
            eligible_files: u64::try_from(eligible_files).unwrap_or(u64::MAX),
            resident_entries_start,
            capacity,
            fd_soft_limit,
            lookup_files: AtomicU64::new(0),
            hit_files: AtomicU64::new(0),
            miss_open_files: AtomicU64::new(0),
            coalesced_files: AtomicU64::new(0),
            bypass_files: AtomicU64::new(0),
            fallback_open_files: AtomicU64::new(0),
            open_failures: AtomicU64::new(0),
            lookup_nanos: AtomicU64::new(0),
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

    fn snapshot(&self, resident_entries_end: u64) -> ReaderCacheQueryStatsSnapshot {
        ReaderCacheQueryStatsSnapshot {
            eligible_files: self.eligible_files,
            resident_entries_start: self.resident_entries_start,
            resident_entries_end,
            capacity: self.capacity,
            fd_soft_limit: self.fd_soft_limit,
            lookup_files: self.lookup_files.load(Ordering::Relaxed),
            hit_files: self.hit_files.load(Ordering::Relaxed),
            miss_open_files: self.miss_open_files.load(Ordering::Relaxed),
            coalesced_files: self.coalesced_files.load(Ordering::Relaxed),
            bypass_files: self.bypass_files.load(Ordering::Relaxed),
            fallback_open_files: self.fallback_open_files.load(Ordering::Relaxed),
            open_failures: self.open_failures.load(Ordering::Relaxed),
            lookup_nanos: self.lookup_nanos.load(Ordering::Relaxed),
            acquire_nanos: self.acquire_nanos.load(Ordering::Relaxed),
            physical_open_nanos: self.physical_open_nanos.load(Ordering::Relaxed),
            bind_nanos: self.bind_nanos.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReaderCacheQueryStatsSnapshot {
    pub eligible_files: u64,
    pub resident_entries_start: u64,
    pub resident_entries_end: u64,
    pub capacity: u64,
    pub fd_soft_limit: u64,
    pub lookup_files: u64,
    pub hit_files: u64,
    pub miss_open_files: u64,
    pub coalesced_files: u64,
    pub bypass_files: u64,
    pub fallback_open_files: u64,
    pub open_failures: u64,
    pub lookup_nanos: u64,
    pub acquire_nanos: u64,
    pub physical_open_nanos: u64,
    pub bind_nanos: u64,
}

/// Bounded, process-local cache of bare data-file readers.
#[derive(Clone)]
pub(crate) struct DataFileReaderCache {
    inner: Cache<DataFileReaderCacheKey, Arc<dyn Reader>>,
    open_permits: Arc<Semaphore>,
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
        let fd_soft_limit = fd_soft_limit();
        let fd_limited_capacity = if fd_soft_limit == u64::MAX {
            DEFAULT_MAX_ENTRIES
        } else {
            (fd_soft_limit / 4).max(1)
        };
        let capacity = DEFAULT_MAX_ENTRIES.min(fd_limited_capacity);
        let max_in_flight = DEFAULT_MAX_IN_FLIGHT_OPENS.min(capacity as usize).max(1);
        Self {
            inner: Cache::builder()
                .max_capacity(capacity)
                .time_to_idle(DEFAULT_TIME_TO_IDLE)
                .build(),
            open_permits: Arc::new(Semaphore::new(max_in_flight)),
            capacity,
            fd_soft_limit,
        }
    }

    pub(crate) fn resident_entries(&self) -> u64 {
        self.inner.entry_count()
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(crate) fn fd_soft_limit(&self) -> u64 {
        self.fd_soft_limit
    }

    pub(crate) fn max_files_per_query(&self) -> usize {
        DEFAULT_MAX_IN_FLIGHT_OPENS.min(self.capacity as usize)
    }

    pub(crate) fn begin_query(&self, eligible_files: usize) -> Arc<ReaderCacheQueryStats> {
        Arc::new(ReaderCacheQueryStats::new(
            eligible_files,
            self.resident_entries(),
            self.capacity,
            self.fd_soft_limit,
        ))
    }

    pub(crate) fn finish_query(
        &self,
        stats: &ReaderCacheQueryStats,
    ) -> ReaderCacheQueryStatsSnapshot {
        stats.snapshot(self.resident_entries())
    }

    /// Return a cached reader or single-flight one physical local-file open.
    pub(crate) async fn get_or_open(
        &self,
        key: DataFileReaderCacheKey,
        stats: &Arc<ReaderCacheQueryStats>,
    ) -> Result<Arc<dyn Reader>> {
        stats.lookup_files.fetch_add(1, Ordering::Relaxed);
        let lookup_started = Instant::now();
        if let Some(reader) = self.inner.get(&key).await {
            stats
                .lookup_nanos
                .fetch_add(elapsed_nanos(lookup_started), Ordering::Relaxed);
            stats.hit_files.fetch_add(1, Ordering::Relaxed);
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
        let loader_ran_for_future = loader_ran.clone();
        let acquire_nanos_for_future = permit_acquire_nanos.clone();
        let open_nanos_for_future = physical_open_nanos.clone();
        let result = self
            .inner
            .try_get_with(key, async move {
                loader_ran_for_future.store(true, Ordering::Relaxed);
                let acquire_started = Instant::now();
                let _permit = open_permits.acquire_owned().await.map_err(|_| {
                    Error::internal("data-file reader cache admission semaphore closed")
                })?;
                acquire_nanos_for_future.store(elapsed_nanos(acquire_started), Ordering::Relaxed);
                let open_started = Instant::now();
                let result = load_store.open_with_size(&load_path, known_size).await;
                open_nanos_for_future.store(elapsed_nanos(open_started), Ordering::Relaxed);
                result.map(Arc::<dyn Reader>::from)
            })
            .await;
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

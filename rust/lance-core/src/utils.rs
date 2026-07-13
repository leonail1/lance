// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod address;
pub mod aimd;
pub mod assume;
pub mod backoff;
pub mod bit;
pub mod blob;
pub mod bloomfilter;
pub mod cpu;
pub mod deletion;
pub mod futures;
pub mod hash;
pub mod io_stats;
pub mod parse;
pub mod path;
pub mod row_addr_remap;
pub mod tempfile;
pub mod testing;
pub mod tokio;
pub mod tracing;

/// Return the process soft limit for open file descriptors.
///
/// A failure to query the platform limit is represented as unbounded so callers
/// can retain their own conservative fixed ceiling.
#[cfg(unix)]
pub fn file_descriptor_soft_limit() -> u64 {
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
pub fn file_descriptor_soft_limit() -> u64 {
    u64::MAX
}

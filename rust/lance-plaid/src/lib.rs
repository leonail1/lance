// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! CPU-only PLAID primitives for database-native late-interaction search.
//!
//! The codec and search pipeline are derived from NextPlaid commit
//! `2dbc95f152244a95c8175d163a77a832d8c8c97d`. This crate keeps the search
//! kernel independent from Lance storage and DataFusion so database adapters
//! can supply row visibility without pulling in SQLite, model, or GPU code.

mod codec;
mod eligibility;
mod error;
mod format;
mod index;
mod maxsim;
mod search;

pub use codec::ResidualQuantizer;
pub use eligibility::{AddressEligibility, AllEligible, Eligibility};
pub use error::{Error, Result};
pub use format::PlaidFormatVersion;
pub use index::PlaidIndex;
pub use maxsim::maxsim_naive;
pub use search::{PlaidSearchParams, PlaidSearchStats, SearchHit};

/// On-disk file name used by the first version of the native PLAID format.
pub const PLAID_DATA_FILE: &str = "plaid.bin";

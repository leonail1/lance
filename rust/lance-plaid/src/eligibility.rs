// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use roaring::RoaringTreemap;

/// Supplies database row visibility to the storage-independent PLAID kernel.
///
/// Both the dense document ordinal and the durable database row address are
/// supplied. Lance's adapter uses the row address for predicate and deletion
/// masks while tests and other engines may use either identifier.
pub trait Eligibility: Send + Sync {
    /// Returns true when a document may participate in the query.
    fn includes(&self, document_ordinal: u32, row_address: u64) -> bool;

    /// Returns an exact selected-document count when cheaply available.
    fn selected_count_hint(&self) -> Option<u64> {
        None
    }
}

/// Eligibility implementation that admits every indexed document.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllEligible;

impl Eligibility for AllEligible {
    fn includes(&self, _document_ordinal: u32, _row_address: u64) -> bool {
        true
    }
}

/// Address-based allow or block mask backed by a compressed 64-bit bitmap.
#[derive(Clone, Debug)]
pub enum AddressEligibility {
    /// Only addresses present in the bitmap are selected.
    Allow(RoaringTreemap),
    /// Addresses present in the bitmap are excluded.
    Block(RoaringTreemap),
}

impl AddressEligibility {
    /// Creates an address allow-list.
    pub fn allow(addresses: impl IntoIterator<Item = u64>) -> Self {
        Self::Allow(addresses.into_iter().collect())
    }

    /// Creates an address block-list.
    pub fn block(addresses: impl IntoIterator<Item = u64>) -> Self {
        Self::Block(addresses.into_iter().collect())
    }
}

impl Eligibility for AddressEligibility {
    fn includes(&self, _document_ordinal: u32, row_address: u64) -> bool {
        match self {
            Self::Allow(addresses) => addresses.contains(row_address),
            Self::Block(addresses) => !addresses.contains(row_address),
        }
    }

    fn selected_count_hint(&self) -> Option<u64> {
        match self {
            Self::Allow(addresses) => Some(addresses.len()),
            Self::Block(_) => None,
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use approx::assert_abs_diff_eq;
use lance_plaid::{
    AddressEligibility, AllEligible, PlaidIndex, PlaidSearchParams, ResidualQuantizer, maxsim_naive,
};
use ndarray::{Array2, array};

fn test_index(nbits: u8) -> PlaidIndex {
    let quantizer = match nbits {
        2 => ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0, 0.0, 0.0, 0.0]).unwrap(),
        4 => ResidualQuantizer::try_new(
            4,
            (-7..8).map(|value| value as f32 / 100.0).collect(),
            vec![0.0; 16],
        )
        .unwrap(),
        _ => unreachable!(),
    };
    let centroids = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
    let residuals = Array2::<f32>::zeros((3, 4));
    let packed_residuals = quantizer.quantize(residuals.view()).unwrap();
    PlaidIndex::try_new(
        centroids,
        quantizer,
        vec![7, 42, 1001],
        vec![0, 1, 2, 3],
        vec![0, 1, 0],
        packed_residuals,
    )
    .unwrap()
}

fn exhaustive_params(top_k: usize) -> PlaidSearchParams {
    PlaidSearchParams {
        n_ivf_probe: 2,
        n_full_scores: 3,
        top_k,
        centroid_score_threshold: None,
    }
}

#[test]
fn naive_and_quantized_maxsim_agree_for_exact_centroids() {
    let index = test_index(2);
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let reconstructed = index.reconstruct_document(0).unwrap();
    assert_abs_diff_eq!(
        maxsim_naive(query.view(), reconstructed.view()),
        index.quantized_maxsim(query.view(), 0).unwrap(),
        epsilon = 1.0e-6
    );
    assert_abs_diff_eq!(
        index.quantized_maxsim(query.view(), 0).unwrap(),
        1.0,
        epsilon = 1.0e-6
    );
}

#[test]
fn maps_non_contiguous_row_addresses_both_directions() {
    let index = test_index(2);
    assert_eq!(index.row_address(0), Some(7));
    assert_eq!(index.row_address(2), Some(1001));
    assert_eq!(index.document_ordinal(7), Some(0));
    assert_eq!(index.document_ordinal(1001), Some(2));
    assert_eq!(index.document_ordinal(8), None);
}

#[test]
fn allow_and_block_masks_filter_postings() {
    let index = test_index(2);
    let query = array![[1.0, 0.0, 0.0, 0.0]];

    let allow = AddressEligibility::allow([1001]);
    let (allowed, allowed_stats) = index
        .search(query.view(), &exhaustive_params(1), &allow)
        .unwrap();
    assert_eq!(allowed.len(), 1);
    assert_eq!(allowed[0].row_address, 1001);
    assert_eq!(allowed_stats.candidate_documents, 1);

    let block = AddressEligibility::block([7]);
    let (unblocked, _) = index
        .search(query.view(), &exhaustive_params(1), &block)
        .unwrap();
    assert_eq!(unblocked.len(), 1);
    assert_eq!(unblocked[0].row_address, 1001);
}

#[test]
fn stable_ties_use_ascending_row_address() {
    let index = test_index(4);
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let (hits, stats) = index
        .search(query.view(), &exhaustive_params(2), &AllEligible)
        .unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>(),
        vec![7, 1001]
    );
    assert_eq!(stats.centroids_probed, 2);
    assert_eq!(stats.candidate_documents, 3);
    assert_eq!(stats.exact_documents, 2);
    assert!(stats.total_nanos > 0);
}

#[test]
fn versioned_file_round_trips_both_quantizers() {
    for nbits in [2, 4] {
        let original = test_index(nbits);
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(format!("plaid-{nbits}.bin"));
        original.write_to_path(&path).unwrap();
        let restored = PlaidIndex::read_from_path(&path).unwrap();
        let bytes = original.to_bytes().unwrap();
        let restored_from_bytes = PlaidIndex::read_from_bytes(&bytes).unwrap();
        assert_eq!(restored.dimension(), original.dimension());
        assert_eq!(
            restored_from_bytes.row_addresses(),
            original.row_addresses()
        );
        assert_eq!(restored.num_documents(), original.num_documents());
        assert_eq!(restored.num_tokens(), original.num_tokens());
        assert_eq!(restored.quantizer(), original.quantizer());

        let query = array![[1.0, 0.0, 0.0, 0.0]];
        let (expected, _) = original
            .search(query.view(), &exhaustive_params(2), &AllEligible)
            .unwrap();
        let (actual, _) = restored
            .search(query.view(), &exhaustive_params(2), &AllEligible)
            .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn versioned_bytes_reject_unknown_version() {
    let index = test_index(2);
    let mut bytes = index.to_bytes().unwrap();
    bytes[8..10].copy_from_slice(&2_u16.to_le_bytes());
    let error = PlaidIndex::read_from_bytes(&bytes).unwrap_err();
    assert!(error.to_string().contains("unsupported format version 2"));
}

#[test]
fn versioned_bytes_reject_unbounded_header_counts_before_allocation() {
    let index = test_index(2);
    let mut bytes = index.to_bytes().unwrap();
    bytes[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
    let error = PlaidIndex::read_from_bytes(&bytes).unwrap_err();
    assert!(
        error.to_string().contains("overflow")
            || error.to_string().contains("encoded length mismatch")
    );

    let valid_bytes = index.to_bytes().unwrap();
    let truncated = &valid_bytes[..47];
    let error = PlaidIndex::read_from_bytes(truncated).unwrap_err();
    assert!(error.to_string().contains("shorter than the 48-byte header"));
}

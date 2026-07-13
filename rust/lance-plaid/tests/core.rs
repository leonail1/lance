// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use approx::assert_abs_diff_eq;
use lance_plaid::{
    AddressEligibility, AllEligible, EligibleCentroidDecision, PlaidIndex, PlaidSearchParams,
    ResidualQuantizer, maxsim_naive,
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

fn exact_test_index(nbits: u8) -> PlaidIndex {
    let quantizer = match nbits {
        2 => ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap(),
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
    PlaidIndex::try_new_with_exact_store(
        centroids,
        quantizer,
        vec![7, 42, 1001],
        vec![0, 2, 2, 3],
        vec![0, 1, 0],
        packed_residuals,
        vec![9001, 7007, 8008],
        (1..=12).map(|value| value as f32).collect(),
    )
    .unwrap()
}

fn legacy_v1_fixture_bytes() -> Vec<u8> {
    fn push_f32s(bytes: &mut Vec<u8>, values: &[f32]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fn push_u32s(bytes: &mut Vec<u8>, values: &[u32]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fn push_u64s(bytes: &mut Vec<u8>, values: &[u64]) {
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LPLDIDX\0");
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.push(2);
    bytes.push(0);
    bytes.extend_from_slice(&0x0102_0304_u32.to_le_bytes());
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&3_u64.to_le_bytes());
    bytes.extend_from_slice(&3_u64.to_le_bytes());
    bytes.extend_from_slice(&3_u64.to_le_bytes());
    push_f32s(&mut bytes, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    push_f32s(&mut bytes, &[-0.1, 0.0, 0.1]);
    push_f32s(&mut bytes, &[0.0; 4]);
    push_u64s(&mut bytes, &[7, 42, 1001]);
    push_u64s(&mut bytes, &[0, 1, 2, 3]);
    push_u32s(&mut bytes, &[0, 1, 0]);
    bytes.extend_from_slice(&[0xaa; 3]);
    push_u64s(&mut bytes, &[0, 2, 3]);
    push_u32s(&mut bytes, &[0, 2, 1]);
    bytes
}

fn exhaustive_params(top_k: usize) -> PlaidSearchParams {
    PlaidSearchParams {
        n_ivf_probe: 2,
        n_full_scores: 3,
        top_k,
        centroid_score_threshold: None,
    }
}

fn filter_cost_index(tokens_per_document: usize) -> PlaidIndex {
    const DOCUMENTS: usize = 128;
    let quantizer = ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
    let centroids = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
    let total_tokens = DOCUMENTS * tokens_per_document;
    let residuals = Array2::<f32>::zeros((total_tokens, 4));
    let packed_residuals = quantizer.quantize(residuals.view()).unwrap();
    let mut document_offsets = Vec::with_capacity(DOCUMENTS + 1);
    let mut token_codes = Vec::with_capacity(total_tokens);
    document_offsets.push(0);
    for document in 0..DOCUMENTS {
        token_codes.extend(std::iter::repeat_n(
            (document % 2) as u32,
            tokens_per_document,
        ));
        document_offsets.push(token_codes.len() as u64);
    }
    PlaidIndex::try_new(
        centroids,
        quantizer,
        (0..DOCUMENTS)
            .map(|ordinal| 1_000 + ordinal as u64)
            .collect(),
        document_offsets,
        token_codes,
        packed_residuals,
    )
    .unwrap()
}

fn tie_index() -> PlaidIndex {
    let quantizer = ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
    let centroids = array![
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ];
    let residuals = Array2::<f32>::zeros((4, 4));
    let packed_residuals = quantizer.quantize(residuals.view()).unwrap();
    PlaidIndex::try_new(
        centroids,
        quantizer,
        vec![100, 101, 102, 103],
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 2, 3],
        packed_residuals,
    )
    .unwrap()
}

fn empty_document_index() -> PlaidIndex {
    let quantizer = ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
    let centroids = array![[1.0, 0.0, 0.0, 0.0]];
    let packed_residuals = quantizer
        .quantize(Array2::<f32>::zeros((1, 4)).view())
        .unwrap();
    PlaidIndex::try_new(
        centroids,
        quantizer,
        vec![10, 20],
        vec![0, 1, 1],
        vec![0],
        packed_residuals,
    )
    .unwrap()
}

fn retry_ceiling_index() -> PlaidIndex {
    const DOCUMENTS: usize = 128;
    const CENTROIDS: usize = 8;
    let quantizer = ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
    let centroids = Array2::from_shape_fn((CENTROIDS, 4), |(row, column)| {
        if column == 0 {
            (CENTROIDS - row) as f32
        } else {
            0.0
        }
    });
    let residuals = Array2::<f32>::zeros((DOCUMENTS, 4));
    let packed_residuals = quantizer.quantize(residuals.view()).unwrap();
    PlaidIndex::try_new(
        centroids,
        quantizer,
        (0..DOCUMENTS)
            .map(|ordinal| 2_000 + ordinal as u64)
            .collect(),
        (0..=DOCUMENTS as u64).collect(),
        (0..DOCUMENTS)
            .map(|ordinal| (ordinal % CENTROIDS) as u32)
            .collect(),
        packed_residuals,
    )
    .unwrap()
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
fn direct_quantized_residuals_are_bitwise_identical_to_exhaustive_legacy() {
    for nbits in [2, 4] {
        let index = test_index(nbits);
        let query = array![[1.0_f32, 0.0, 0.0, 0.0], [0.0_f32, 1.0, 0.0, 0.0]];
        let eligibility = AddressEligibility::allow([7, 42, 1001]);
        let (legacy, legacy_stats) = index
            .search(query.view(), &exhaustive_params(3), &eligibility)
            .unwrap();
        let (direct, direct_stats) = index
            .search_quantized_residuals(query.view(), &[0, 1, 2])
            .unwrap()
            .expect("all documents have tokens");

        assert_eq!(
            direct
                .iter()
                .map(|hit| (hit.document_ordinal, hit.row_address, hit.score.to_bits()))
                .collect::<Vec<_>>(),
            legacy
                .iter()
                .map(|hit| (hit.document_ordinal, hit.row_address, hit.score.to_bits()))
                .collect::<Vec<_>>()
        );
        assert_eq!(direct_stats.candidate_documents, 3);
        assert_eq!(direct_stats.approximate_documents, 0);
        assert_eq!(direct_stats.exact_documents, 3);
        assert_eq!(legacy_stats.exact_documents, 3);
    }
}

#[test]
fn direct_quantized_residuals_share_legacy_tie_breaking() {
    let index = tie_index();
    let query = array![[1.0_f32, 0.0, 0.0, 0.0], [0.0_f32, 1.0, 0.0, 0.0]];
    let params = PlaidSearchParams {
        n_ivf_probe: 4,
        n_full_scores: 16,
        top_k: 4,
        centroid_score_threshold: None,
    };
    let (legacy, _) = index.search(query.view(), &params, &AllEligible).unwrap();
    let (direct, _) = index
        .search_quantized_residuals(query.view(), &[0, 1, 2, 3])
        .unwrap()
        .expect("all documents have tokens");
    assert_eq!(direct, legacy);
    assert_eq!(
        direct.iter().map(|hit| hit.row_address).collect::<Vec<_>>(),
        vec![100, 101, 102, 103]
    );
}

#[test]
fn direct_quantized_residuals_reject_noncanonical_ordinals_and_fallback_on_empty_tokens() {
    let index = empty_document_index();
    let query = array![[1.0_f32, 0.0, 0.0, 0.0]];
    assert!(
        index
            .search_quantized_residuals(query.view(), &[0, 1])
            .unwrap()
            .is_none()
    );
    for ordinals in [&[1_u32, 0][..], &[0_u32, 0][..]] {
        let error = index
            .search_quantized_residuals(query.view(), ordinals)
            .unwrap_err();
        assert!(error.to_string().contains("sorted and unique"));
    }
}

#[test]
fn eligible_centroid_cost_model_enables_only_selective_cheap_filters() {
    let index = filter_cost_index(1);

    let unfiltered = index
        .plan_eligible_centroids(None, None, false, 1, 1)
        .unwrap();
    assert_eq!(unfiltered.decision(), EligibleCentroidDecision::Unfiltered);
    assert!(!unfiltered.is_enabled());

    let non_enumerable = index
        .plan_eligible_centroids(None, None, true, 1, 1)
        .unwrap();
    assert_eq!(
        non_enumerable.decision(),
        EligibleCentroidDecision::NonEnumerable
    );
    assert!(!non_enumerable.is_enabled());

    let empty = index
        .plan_eligible_centroids(Some(&[]), Some(0), true, 1, 1)
        .unwrap();
    assert_eq!(empty.decision(), EligibleCentroidDecision::Empty);
    assert!(empty.is_enabled());

    let selective = index
        .plan_eligible_centroids(Some(&[0, 1]), Some(2), true, 1, 1)
        .unwrap();
    assert_eq!(selective.decision(), EligibleCentroidDecision::Enabled);
    assert_eq!(selective.eligible_documents(), 2);
    assert_eq!(selective.eligible_tokens(), 2);
    assert_eq!(selective.eligible_token_codes_scanned(), 2);
    assert_eq!(selective.eligible_centroids(), 2);
    assert_eq!(selective.estimated_global_postings(), 64);

    let wide_ordinals = (0_u32..5).collect::<Vec<_>>();
    let wide = index
        .plan_eligible_centroids(Some(&wide_ordinals), Some(5), true, 1, 1)
        .unwrap();
    assert_eq!(wide.decision(), EligibleCentroidDecision::TooWide);
    assert!(!wide.is_enabled());
    assert_eq!(wide.eligible_token_codes_scanned(), 0);
    let wide_count_only = index
        .plan_eligible_centroids(None, Some(5), true, 1, 1)
        .unwrap();
    assert_eq!(
        wide_count_only.decision(),
        EligibleCentroidDecision::TooWide
    );
    assert_eq!(wide_count_only.eligible_documents(), 5);
    assert_eq!(wide_count_only.eligible_tokens(), 0);

    let token_heavy = filter_cost_index(16)
        .plan_eligible_centroids(Some(&[0, 1]), Some(2), true, 1, 1)
        .unwrap();
    assert_eq!(
        token_heavy.decision(),
        EligibleCentroidDecision::ScanCostTooHigh
    );
    assert!(!token_heavy.is_enabled());
    assert_eq!(token_heavy.eligible_token_codes_scanned(), 0);
}

#[test]
fn eligible_centroid_retry_expands_postings_incrementally() {
    let index = filter_cost_index(1);
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let plan = index
        .plan_eligible_centroids(Some(&[0, 1]), Some(2), true, 1, 1)
        .unwrap();
    let eligibility = AddressEligibility::allow([1_000, 1_001]);
    let params = PlaidSearchParams {
        n_ivf_probe: 1,
        n_full_scores: 128,
        top_k: 2,
        centroid_score_threshold: None,
    };
    let (incremental, stats) = index
        .search_adaptive(query.view(), &params, 2, 2, &eligibility, Some(&plan))
        .unwrap();

    let mut exhaustive = params;
    exhaustive.n_ivf_probe = 2;
    let (expected, expected_stats) = index
        .search(query.view(), &exhaustive, &eligibility)
        .unwrap();
    assert_eq!(incremental, expected);
    assert_eq!(stats.probe_rounds, 2);
    assert_eq!(stats.probe_retries, 1);
    assert_eq!(stats.configured_probes, 3);
    assert_eq!(stats.final_nprobe, 2);
    assert_eq!(stats.centroids_probed, 2);
    assert_eq!(stats.incremental_centroids_reused, 1);
    assert_eq!(stats.posting_entries_read, 128);
    assert_eq!(stats.posting_entries_eligible, 2);
    assert_eq!(
        stats.posting_entries_read,
        expected_stats.posting_entries_read
    );
    assert!(stats.eligible_centroid_selection_nanos > 0);
}

#[test]
fn empty_plan_skips_gemm_and_plan_identity_is_exact() {
    let index = filter_cost_index(1);
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let params = PlaidSearchParams {
        n_ivf_probe: 1,
        n_full_scores: 16,
        top_k: 2,
        centroid_score_threshold: None,
    };
    let empty_plan = index
        .plan_eligible_centroids(Some(&[]), Some(0), true, 1, 1)
        .unwrap();
    let (hits, stats) = index
        .search_adaptive(query.view(), &params, 2, 2, &AllEligible, Some(&empty_plan))
        .unwrap();
    assert!(hits.is_empty());
    assert_eq!(stats.centroid_probe_nanos, 0);
    assert_eq!(stats.probe_rounds, 0);
    assert_eq!(stats.posting_entries_read, 0);
    assert_eq!(stats.approximate_documents, 0);
    assert_eq!(stats.exact_documents, 0);

    let other_index = filter_cost_index(1);
    let error = other_index
        .search_adaptive(query.view(), &params, 2, 2, &AllEligible, Some(&empty_plan))
        .unwrap_err();
    assert!(error.to_string().contains("does not belong"));

    let cloned_index = index.clone();
    let error = cloned_index
        .search_adaptive(query.view(), &params, 2, 2, &AllEligible, Some(&empty_plan))
        .unwrap_err();
    assert!(error.to_string().contains("does not belong"));
    let (original_hits, _) = index
        .search_adaptive(query.view(), &params, 2, 2, &AllEligible, Some(&empty_plan))
        .unwrap();
    assert!(original_hits.is_empty());
}

#[test]
fn multi_token_ties_are_deterministic_and_thresholded() {
    let index = tie_index();
    let query = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
    let params = PlaidSearchParams {
        n_ivf_probe: 1,
        n_full_scores: 4,
        top_k: 2,
        centroid_score_threshold: Some(0.5),
    };
    let (hits, stats) = index.search(query.view(), &params, &AllEligible).unwrap();
    assert_eq!(
        hits.iter().map(|hit| hit.row_address).collect::<Vec<_>>(),
        vec![100, 102]
    );
    assert_eq!(stats.centroids_probed, 2);
    let (again, _) = index.search(query.view(), &params, &AllEligible).unwrap();
    assert_eq!(again, hits);

    let mut above_ties = params;
    above_ties.centroid_score_threshold = Some(1.1);
    let (none, stats) = index
        .search(query.view(), &above_ties, &AllEligible)
        .unwrap();
    assert!(none.is_empty());
    assert_eq!(stats.centroids_probed, 0);
}

#[test]
fn maximum_probe_ceiling_stops_incremental_multi_round_retry() {
    let index = retry_ceiling_index();
    let query = array![[1.0, 0.0, 0.0, 0.0]];
    let plan = index
        .plan_eligible_centroids(Some(&[0, 1, 2, 3]), Some(4), true, 1, 1)
        .unwrap();
    assert_eq!(plan.decision(), EligibleCentroidDecision::Enabled);
    let eligibility = AddressEligibility::allow([2_000, 2_001, 2_002, 2_003]);
    let params = PlaidSearchParams {
        n_ivf_probe: 1,
        n_full_scores: 128,
        top_k: 4,
        centroid_score_threshold: None,
    };
    let (ceiling_hits, ceiling_stats) = index
        .search_adaptive(query.view(), &params, 3, 4, &eligibility, Some(&plan))
        .unwrap();
    assert_eq!(ceiling_hits.len(), 3);
    assert_eq!(ceiling_stats.probe_rounds, 3);
    assert_eq!(ceiling_stats.probe_retries, 2);
    assert_eq!(ceiling_stats.configured_probes, 6);
    assert_eq!(ceiling_stats.final_nprobe, 3);
    assert_eq!(ceiling_stats.centroids_probed, 3);
    assert_eq!(ceiling_stats.incremental_centroids_reused, 3);
    assert_eq!(ceiling_stats.posting_entries_read, 48);

    let (complete_hits, complete_stats) = index
        .search_adaptive(query.view(), &params, 4, 4, &eligibility, Some(&plan))
        .unwrap();
    assert_eq!(complete_hits.len(), 4);
    assert_eq!(complete_stats.probe_rounds, 3);
    assert_eq!(complete_stats.probe_retries, 2);
    assert_eq!(complete_stats.final_nprobe, 4);
    assert_eq!(complete_stats.posting_entries_read, 64);
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
        assert!(!restored.has_exact_store());
        assert!(restored.exact_store_row_ids().is_none());
        assert!(restored.exact_document(0).unwrap().is_none());

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
fn v1_encoding_matches_the_frozen_legacy_fixture() {
    let bytes = test_index(2).to_bytes().unwrap();
    assert_eq!(bytes, legacy_v1_fixture_bytes());
    assert_eq!(&bytes[8..10], &1_u16.to_le_bytes());
    assert_eq!(bytes[11], 0);

    let restored = PlaidIndex::read_from_bytes(&bytes).unwrap();
    assert!(!restored.has_exact_store());
    assert_eq!(restored.to_bytes().unwrap(), bytes);
}

#[test]
fn v2_exact_store_round_trips_and_borrows_ordinal_aligned_documents() {
    for nbits in [2, 4] {
        let original = exact_test_index(nbits);
        let base_size = test_index(nbits).estimated_size_bytes();
        assert!(original.has_exact_store());
        assert_eq!(
            original.exact_store_row_ids(),
            Some(&[9001, 7007, 8008][..])
        );
        assert_eq!(original.exact_public_row_id(0), Some(9001));
        assert_eq!(original.exact_public_row_id(2), Some(8008));
        assert_eq!(original.exact_public_row_id(3), None);
        assert_eq!(
            original.estimated_size_bytes() - base_size,
            3 * std::mem::size_of::<u64>() + 12 * std::mem::size_of::<f32>()
        );

        let first = original.exact_document(0).unwrap().unwrap();
        let empty = original.exact_document(1).unwrap().unwrap();
        let third = original.exact_document(2).unwrap().unwrap();
        assert_eq!(first.shape(), &[2, 4]);
        assert_eq!(
            first.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(empty.shape(), &[0, 4]);
        assert!(empty.as_slice().unwrap().is_empty());
        assert_eq!(third.as_slice().unwrap(), &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(
            third.as_ptr(),
            first.as_ptr().wrapping_add(first.len()),
            "adjacent views must borrow the contiguous exact-store allocation"
        );
        assert!(original.exact_document(3).is_err());

        let cloned = original.clone();
        assert!(cloned.has_exact_store());
        assert_eq!(
            cloned
                .exact_document(2)
                .unwrap()
                .unwrap()
                .as_slice()
                .unwrap(),
            &[9.0, 10.0, 11.0, 12.0]
        );

        let bytes = original.to_bytes().unwrap();
        assert_eq!(&bytes[8..10], &2_u16.to_le_bytes());
        assert_eq!(bytes[11], 1);
        let restored = PlaidIndex::read_from_bytes(&bytes).unwrap();
        assert!(restored.has_exact_store());
        assert_eq!(
            restored.exact_store_row_ids(),
            original.exact_store_row_ids()
        );
        for ordinal in 0..original.num_documents() as u32 {
            assert_eq!(
                restored
                    .exact_document(ordinal)
                    .unwrap()
                    .unwrap()
                    .as_slice(),
                original
                    .exact_document(ordinal)
                    .unwrap()
                    .unwrap()
                    .as_slice()
            );
        }
        assert_eq!(restored.to_bytes().unwrap(), bytes);

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(format!("plaid-v2-{nbits}.bin"));
        original.write_to_path(&path).unwrap();
        assert!(PlaidIndex::read_from_path(path).unwrap().has_exact_store());
    }
}

#[test]
fn exact_store_constructor_rejects_invalid_lengths_and_non_finite_values() {
    let make = |row_ids: Vec<u64>, raw: Vec<f32>| {
        let quantizer = ResidualQuantizer::try_new(2, vec![-0.1, 0.0, 0.1], vec![0.0; 4]).unwrap();
        let packed_residuals = quantizer
            .quantize(Array2::<f32>::zeros((3, 4)).view())
            .unwrap();
        PlaidIndex::try_new_with_exact_store(
            array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            quantizer,
            vec![7, 42, 1001],
            vec![0, 1, 2, 3],
            vec![0, 1, 0],
            packed_residuals,
            row_ids,
            raw,
        )
    };

    let error = make(vec![1, 2], vec![0.0; 12]).unwrap_err();
    assert!(error.to_string().contains("public row ID length"));
    let error = make(vec![1, 2, 3], vec![0.0; 11]).unwrap_err();
    assert!(error.to_string().contains("raw token value length"));
    let mut non_finite = vec![0.0; 12];
    non_finite[7] = f32::INFINITY;
    let error = make(vec![1, 2, 3], non_finite).unwrap_err();
    assert!(error.to_string().contains("must be finite"));
}

#[test]
fn versioned_bytes_reject_unknown_version() {
    let index = test_index(2);
    let mut bytes = index.to_bytes().unwrap();
    bytes[8..10].copy_from_slice(&3_u16.to_le_bytes());
    let error = PlaidIndex::read_from_bytes(&bytes).unwrap_err();
    assert!(error.to_string().contains("unsupported format version 3"));
}

#[test]
fn versioned_bytes_reject_invalid_flags_truncation_trailing_and_non_finite_exact_values() {
    let mut v1_flags = test_index(2).to_bytes().unwrap();
    v1_flags[11] = 1;
    assert!(
        PlaidIndex::read_from_bytes(&v1_flags)
            .unwrap_err()
            .to_string()
            .contains("V1 flags must be zero")
    );

    let valid_v2 = exact_test_index(2).to_bytes().unwrap();
    for flags in [0, 2, 3] {
        let mut invalid = valid_v2.clone();
        invalid[11] = flags;
        let message = PlaidIndex::read_from_bytes(&invalid)
            .unwrap_err()
            .to_string();
        assert!(message.contains("exact-store flag") || message.contains("unknown flags"));
    }

    let mut truncated = valid_v2.clone();
    truncated.pop();
    assert!(
        PlaidIndex::read_from_bytes(&truncated)
            .unwrap_err()
            .to_string()
            .contains("encoded length mismatch")
    );
    let mut trailing = valid_v2.clone();
    trailing.push(0);
    assert!(
        PlaidIndex::read_from_bytes(&trailing)
            .unwrap_err()
            .to_string()
            .contains("encoded length mismatch")
    );

    let mut non_finite = valid_v2;
    let raw_start = non_finite.len() - 12 * std::mem::size_of::<f32>();
    non_finite[raw_start..raw_start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        PlaidIndex::read_from_bytes(&non_finite)
            .unwrap_err()
            .to_string()
            .contains("must be finite")
    );
}

#[test]
fn v2_header_rejects_exact_value_length_overflow_before_allocation() {
    let mut bytes = exact_test_index(2).to_bytes().unwrap();
    bytes[16..20].copy_from_slice(&8_u32.to_le_bytes());
    let overflowing_tokens = (usize::MAX / 8 + 1) as u64;
    bytes[32..40].copy_from_slice(&overflowing_tokens.to_le_bytes());
    let error = PlaidIndex::read_from_bytes(&bytes).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exact-store value count overflow")
    );
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
    assert!(
        error
            .to_string()
            .contains("shorter than the 48-byte header")
    );
}

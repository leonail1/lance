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
    assert!(
        error
            .to_string()
            .contains("shorter than the 48-byte header")
    );
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use ndarray::ArrayView2;

/// Computes ColBERT MaxSim with a deterministic scalar implementation.
///
/// For each query token, this finds the maximum dot product with any document
/// token and sums those maxima. Empty documents score negative infinity.
pub fn maxsim_naive(query: ArrayView2<'_, f32>, document: ArrayView2<'_, f32>) -> f32 {
    if document.nrows() == 0 {
        return f32::NEG_INFINITY;
    }
    let mut score = 0.0_f32;
    for query_token in query.outer_iter() {
        let mut maximum = f32::NEG_INFINITY;
        for document_token in document.outer_iter() {
            let similarity = query_token.dot(&document_token);
            if similarity.total_cmp(&maximum).is_gt() {
                maximum = similarity;
            }
        }
        score += maximum;
    }
    score
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    use super::*;

    #[test]
    fn sums_per_query_token_maxima() {
        let query = array![[1.0, 0.0], [0.0, 1.0]];
        let document = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
        assert_abs_diff_eq!(
            maxsim_naive(query.view(), document.view()),
            2.0,
            epsilon = 1.0e-6
        );
    }

    #[test]
    fn empty_document_has_no_finite_score() {
        let query = array![[1.0, 0.0]];
        let document = ndarray::Array2::<f32>::zeros((0, 2));
        assert_eq!(
            maxsim_naive(query.view(), document.view()),
            f32::NEG_INFINITY
        );
    }
}

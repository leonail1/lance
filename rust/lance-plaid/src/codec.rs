// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! NextPlaid-compatible 2-bit and 4-bit residual packing.

use ndarray::ArrayView2;

use crate::{Error, Result};

/// Scalar residual quantizer used after coarse centroid assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualQuantizer {
    nbits: u8,
    bucket_cutoffs: Vec<f32>,
    bucket_weights: Vec<f32>,
}

impl ResidualQuantizer {
    /// Creates a residual quantizer.
    ///
    /// `nbits` must be 2 or 4. The cutoff count must be `2^nbits - 1` and
    /// the reconstruction-weight count must be `2^nbits`.
    pub fn try_new(nbits: u8, bucket_cutoffs: Vec<f32>, bucket_weights: Vec<f32>) -> Result<Self> {
        if !matches!(nbits, 2 | 4) {
            return Err(Error::InvalidInput(format!(
                "nbits must be 2 or 4, got {nbits}"
            )));
        }
        let num_buckets = 1_usize << nbits;
        if bucket_cutoffs.len() != num_buckets - 1 {
            return Err(Error::InvalidInput(format!(
                "bucket_cutoffs length must be {}, got {}",
                num_buckets - 1,
                bucket_cutoffs.len()
            )));
        }
        if bucket_weights.len() != num_buckets {
            return Err(Error::InvalidInput(format!(
                "bucket_weights length must be {num_buckets}, got {}",
                bucket_weights.len()
            )));
        }
        if bucket_cutoffs.iter().any(|value| !value.is_finite())
            || bucket_weights.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InvalidInput(
                "bucket cutoffs and weights must be finite".to_string(),
            ));
        }
        if !bucket_cutoffs.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(Error::InvalidInput(
                "bucket_cutoffs must be sorted in ascending order".to_string(),
            ));
        }
        Ok(Self {
            nbits,
            bucket_cutoffs,
            bucket_weights,
        })
    }

    /// Number of bits used for each residual dimension.
    pub fn nbits(&self) -> u8 {
        self.nbits
    }

    /// Ordered residual bucket boundaries.
    pub fn bucket_cutoffs(&self) -> &[f32] {
        &self.bucket_cutoffs
    }

    /// Representative reconstruction value for each residual bucket.
    pub fn bucket_weights(&self) -> &[f32] {
        &self.bucket_weights
    }

    /// Number of packed bytes required for one residual vector.
    pub fn packed_len(&self, dimension: usize) -> Result<usize> {
        let bit_count = dimension
            .checked_mul(usize::from(self.nbits))
            .ok_or_else(|| {
                Error::InvalidInput("residual bit count overflowed usize".to_string())
            })?;
        if !bit_count.is_multiple_of(8) {
            return Err(Error::InvalidInput(format!(
                "dimension * nbits must be divisible by 8, got dimension={dimension}, nbits={}",
                self.nbits
            )));
        }
        Ok(bit_count / 8)
    }

    /// Quantizes and packs a row-major residual matrix.
    ///
    /// Bits follow NextPlaid's order: each bucket is emitted least-significant
    /// bit first, while successive bits occupy each byte from MSB to LSB.
    pub fn quantize(&self, residuals: ArrayView2<'_, f32>) -> Result<Vec<u8>> {
        if residuals.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidInput(
                "residual values must be finite".to_string(),
            ));
        }
        let packed_len = self.packed_len(residuals.ncols())?;
        let output_len = residuals.nrows().checked_mul(packed_len).ok_or_else(|| {
            Error::InvalidInput("packed residual size overflowed usize".to_string())
        })?;
        let mut output = vec![0_u8; output_len];
        for (row_index, row) in residuals.outer_iter().enumerate() {
            let start = row_index * packed_len;
            self.quantize_row(
                row.as_slice().ok_or_else(|| {
                    Error::InvalidInput("residual rows must be contiguous".to_string())
                })?,
                &mut output[start..start + packed_len],
            );
        }
        Ok(output)
    }

    pub(crate) fn reconstruct_token(
        &self,
        packed: &[u8],
        centroid: &[f32],
        output: &mut [f32],
    ) -> Result<()> {
        if output.len() != centroid.len() {
            return Err(Error::InvalidInput(format!(
                "output length {} does not match centroid dimension {}",
                output.len(),
                centroid.len()
            )));
        }
        let expected_len = self.packed_len(centroid.len())?;
        if packed.len() != expected_len {
            return Err(Error::InvalidInput(format!(
                "packed residual length must be {expected_len}, got {}",
                packed.len()
            )));
        }

        let mut squared_norm = 0.0_f32;
        for dimension in 0..centroid.len() {
            let bucket = self.decode_bucket(packed, dimension);
            let value = centroid[dimension] + self.bucket_weights[bucket];
            output[dimension] = value;
            squared_norm += value * value;
        }
        let norm = squared_norm.sqrt().max(1.0e-12);
        for value in output {
            *value /= norm;
        }
        Ok(())
    }

    fn quantize_row(&self, residual: &[f32], output: &mut [u8]) {
        let mut bit_index = 0_usize;
        for value in residual {
            let bucket = self.bucket_cutoffs.partition_point(|cutoff| value > cutoff);
            for bucket_bit in 0..self.nbits {
                let bit = ((bucket >> bucket_bit) & 1) as u8;
                let byte_index = bit_index / 8;
                let bit_position = 7 - (bit_index % 8);
                output[byte_index] |= bit << bit_position;
                bit_index += 1;
            }
        }
    }

    fn decode_bucket(&self, packed: &[u8], dimension: usize) -> usize {
        let mut bucket = 0_usize;
        let first_bit = dimension * usize::from(self.nbits);
        for bucket_bit in 0..self.nbits {
            let bit_index = first_bit + usize::from(bucket_bit);
            let byte_index = bit_index / 8;
            let bit_position = 7 - (bit_index % 8);
            let bit = (packed[byte_index] >> bit_position) & 1;
            bucket |= usize::from(bit) << bucket_bit;
        }
        bucket
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn two_bit_packing_matches_next_plaid_order() {
        let quantizer =
            ResidualQuantizer::try_new(2, vec![-1.0, 0.0, 1.0], vec![-2.0, -0.5, 0.5, 2.0])
                .unwrap();
        let packed = quantizer
            .quantize(array![[-2.0, -0.5, 0.5, 2.0]].view())
            .unwrap();
        assert_eq!(packed, vec![0x27]);
    }

    #[test]
    fn four_bit_packing_matches_next_plaid_order() {
        let quantizer = ResidualQuantizer::try_new(
            4,
            (0..15).map(|value| value as f32).collect(),
            (0..16).map(|value| value as f32).collect(),
        )
        .unwrap();
        let packed = quantizer.quantize(array![[-1.0, 100.0]].view()).unwrap();
        assert_eq!(packed, vec![0x0f]);
    }
}

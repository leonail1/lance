// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array2;

use crate::{Error, PlaidIndex, ResidualQuantizer, Result};

const MAGIC: [u8; 8] = *b"LPLDIDX\0";
const ENDIAN_MARKER: u32 = 0x0102_0304;

/// Version discriminator for the stable local PLAID file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum PlaidFormatVersion {
    /// Initial format with fixed little-endian dense sections.
    V1 = 1,
}

impl TryFrom<u16> for PlaidFormatVersion {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::V1),
            _ => Err(Error::CorruptFile(format!(
                "unsupported format version {value}"
            ))),
        }
    }
}

impl PlaidIndex {
    /// Writes the index to the versioned local PLAID format.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&MAGIC)?;
        writer.write_u16::<LittleEndian>(PlaidFormatVersion::V1 as u16)?;
        writer.write_u8(self.quantizer.nbits())?;
        writer.write_u8(0)?;
        writer.write_u32::<LittleEndian>(ENDIAN_MARKER)?;
        writer.write_u32::<LittleEndian>(u32_len(self.dimension, "dimension")?)?;
        writer.write_u32::<LittleEndian>(u32_len(self.num_centroids(), "centroid count")?)?;
        writer.write_u64::<LittleEndian>(self.num_documents() as u64)?;
        writer.write_u64::<LittleEndian>(self.num_tokens() as u64)?;
        writer.write_u64::<LittleEndian>(self.postings.len() as u64)?;

        write_f32s(
            &mut writer,
            self.centroids.as_slice().ok_or_else(|| {
                Error::InvalidInput("centroid matrix must be contiguous".to_string())
            })?,
        )?;
        write_f32s(&mut writer, self.quantizer.bucket_cutoffs())?;
        write_f32s(&mut writer, self.quantizer.bucket_weights())?;
        write_u64s(&mut writer, &self.row_addresses)?;
        write_u64s(&mut writer, &self.document_offsets)?;
        write_u32s(&mut writer, &self.token_codes)?;
        writer.write_all(&self.packed_residuals)?;
        write_u64s(&mut writer, &self.posting_offsets)?;
        write_u32s(&mut writer, &self.postings)?;
        writer.flush()?;
        Ok(())
    }

    /// Loads and validates an index from the versioned local PLAID format.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(Error::CorruptFile(format!("invalid magic bytes {magic:?}")));
        }
        let _version = PlaidFormatVersion::try_from(reader.read_u16::<LittleEndian>()?)?;
        let nbits = reader.read_u8()?;
        if !matches!(nbits, 2 | 4) {
            return Err(Error::CorruptFile(format!(
                "nbits must be 2 or 4, got {nbits}"
            )));
        }
        let reserved = reader.read_u8()?;
        if reserved != 0 {
            return Err(Error::CorruptFile(format!(
                "reserved header byte must be zero, got {reserved}"
            )));
        }
        let marker = reader.read_u32::<LittleEndian>()?;
        if marker != ENDIAN_MARKER {
            return Err(Error::CorruptFile(format!(
                "invalid endian marker {marker:#010x}"
            )));
        }

        let dimension = usize::try_from(reader.read_u32::<LittleEndian>()?)
            .map_err(|_| Error::CorruptFile("dimension does not fit usize".to_string()))?;
        let num_centroids = usize::try_from(reader.read_u32::<LittleEndian>()?)
            .map_err(|_| Error::CorruptFile("centroid count does not fit usize".to_string()))?;
        let num_documents = usize_len(reader.read_u64::<LittleEndian>()?, "document count")?;
        let num_tokens = usize_len(reader.read_u64::<LittleEndian>()?, "token count")?;
        let num_postings = usize_len(reader.read_u64::<LittleEndian>()?, "posting count")?;
        let num_buckets = 1_usize
            .checked_shl(u32::from(nbits))
            .ok_or_else(|| Error::CorruptFile(format!("invalid nbits value {nbits}")))?;

        let centroid_values = read_f32s(
            &mut reader,
            num_centroids
                .checked_mul(dimension)
                .ok_or_else(|| Error::CorruptFile("centroid size overflow".to_string()))?,
        )?;
        let bucket_cutoffs = read_f32s(
            &mut reader,
            num_buckets
                .checked_sub(1)
                .ok_or_else(|| Error::CorruptFile("bucket count underflow".to_string()))?,
        )?;
        let bucket_weights = read_f32s(&mut reader, num_buckets)?;
        let row_addresses = read_u64s(&mut reader, num_documents)?;
        let document_offsets = read_u64s(
            &mut reader,
            num_documents
                .checked_add(1)
                .ok_or_else(|| Error::CorruptFile("document offset count overflow".to_string()))?,
        )?;
        let token_codes = read_u32s(&mut reader, num_tokens)?;
        let quantizer = ResidualQuantizer::try_new(nbits, bucket_cutoffs, bucket_weights)
            .map_err(|error| Error::CorruptFile(error.to_string()))?;
        let packed_len = quantizer
            .packed_len(dimension)
            .map_err(|error| Error::CorruptFile(error.to_string()))?;
        let residual_len = num_tokens
            .checked_mul(packed_len)
            .ok_or_else(|| Error::CorruptFile("residual size overflow".to_string()))?;
        let mut packed_residuals = vec![0_u8; residual_len];
        reader.read_exact(&mut packed_residuals)?;
        let posting_offsets = read_u64s(
            &mut reader,
            num_centroids
                .checked_add(1)
                .ok_or_else(|| Error::CorruptFile("posting offset count overflow".to_string()))?,
        )?;
        let postings = read_u32s(&mut reader, num_postings)?;
        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(Error::CorruptFile(
                "unexpected trailing bytes after posting section".to_string(),
            ));
        }

        let centroids = Array2::from_shape_vec((num_centroids, dimension), centroid_values)
            .map_err(|error| Error::CorruptFile(error.to_string()))?;
        Self::try_from_parts(
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
        )
        .map_err(|error| Error::CorruptFile(error.to_string()))
    }
}

fn u32_len(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::InvalidInput(format!("{name} {value} exceeds u32 format limit")))
}

fn usize_len(value: u64, name: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::CorruptFile(format!("{name} {value} does not fit usize")))
}

fn write_f32s(writer: &mut impl Write, values: &[f32]) -> Result<()> {
    for value in values {
        writer.write_f32::<LittleEndian>(*value)?;
    }
    Ok(())
}

fn write_u32s(writer: &mut impl Write, values: &[u32]) -> Result<()> {
    for value in values {
        writer.write_u32::<LittleEndian>(*value)?;
    }
    Ok(())
}

fn write_u64s(writer: &mut impl Write, values: &[u64]) -> Result<()> {
    for value in values {
        writer.write_u64::<LittleEndian>(*value)?;
    }
    Ok(())
}

fn read_f32s(reader: &mut impl Read, len: usize) -> Result<Vec<f32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(reader.read_f32::<LittleEndian>()?);
    }
    Ok(values)
}

fn read_u32s(reader: &mut impl Read, len: usize) -> Result<Vec<u32>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(reader.read_u32::<LittleEndian>()?);
    }
    Ok(values)
}

fn read_u64s(reader: &mut impl Read, len: usize) -> Result<Vec<u64>> {
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(reader.read_u64::<LittleEndian>()?);
    }
    Ok(values)
}

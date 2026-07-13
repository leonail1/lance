// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::Array2;

use crate::index::ExactStore;
use crate::{Error, PlaidIndex, ResidualQuantizer, Result};

const MAGIC: [u8; 8] = *b"LPLDIDX\0";
const ENDIAN_MARKER: u32 = 0x0102_0304;
const HEADER_LEN: usize = 48;
const EXACT_STORE_FLAG: u8 = 0x01;

/// Version discriminator for the stable local PLAID file format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum PlaidFormatVersion {
    /// Initial format with fixed little-endian dense sections.
    V1 = 1,
    /// Adds public row IDs and raw token values for exact refinement.
    V2 = 2,
}

impl TryFrom<u16> for PlaidFormatVersion {
    type Error = Error;

    fn try_from(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
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
        self.write_to(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Serializes the index into the versioned PLAID format.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        self.write_to(&mut bytes)?;
        Ok(bytes)
    }

    fn write_to(&self, writer: &mut impl Write) -> Result<()> {
        let (version, flags) = if self.has_exact_store() {
            (PlaidFormatVersion::V2, EXACT_STORE_FLAG)
        } else {
            (PlaidFormatVersion::V1, 0)
        };
        writer.write_all(&MAGIC)?;
        writer.write_u16::<LittleEndian>(version as u16)?;
        writer.write_u8(self.quantizer.nbits())?;
        writer.write_u8(flags)?;
        writer.write_u32::<LittleEndian>(ENDIAN_MARKER)?;
        writer.write_u32::<LittleEndian>(u32_len(self.dimension, "dimension")?)?;
        writer.write_u32::<LittleEndian>(u32_len(self.num_centroids(), "centroid count")?)?;
        writer.write_u64::<LittleEndian>(self.num_documents() as u64)?;
        writer.write_u64::<LittleEndian>(self.num_tokens() as u64)?;
        writer.write_u64::<LittleEndian>(self.postings.len() as u64)?;

        write_f32s(
            writer,
            self.centroids.as_slice().ok_or_else(|| {
                Error::InvalidInput("centroid matrix must be contiguous".to_string())
            })?,
        )?;
        write_f32s(writer, self.quantizer.bucket_cutoffs())?;
        write_f32s(writer, self.quantizer.bucket_weights())?;
        write_u64s(writer, &self.row_addresses)?;
        write_u64s(writer, &self.document_offsets)?;
        write_u32s(writer, &self.token_codes)?;
        writer.write_all(&self.packed_residuals)?;
        write_u64s(writer, &self.posting_offsets)?;
        write_u32s(writer, &self.postings)?;
        if let Some((public_row_ids, raw_token_values)) = self.exact_store_parts() {
            write_u64s(writer, public_row_ids)?;
            write_f32s(writer, raw_token_values)?;
        }
        Ok(())
    }

    /// Loads and validates an index from the versioned local PLAID format.
    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::read_from_bytes(&bytes)
    }

    /// Loads and validates an index from in-memory versioned PLAID bytes.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self> {
        validate_encoded_len(bytes)?;
        Self::read_from(&mut Cursor::new(bytes))
    }

    fn read_from(reader: &mut impl Read) -> Result<Self> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(Error::CorruptFile(format!("invalid magic bytes {magic:?}")));
        }
        let version = PlaidFormatVersion::try_from(reader.read_u16::<LittleEndian>()?)?;
        let nbits = reader.read_u8()?;
        if !matches!(nbits, 2 | 4) {
            return Err(Error::CorruptFile(format!(
                "nbits must be 2 or 4, got {nbits}"
            )));
        }
        let flags = reader.read_u8()?;
        let has_exact_store = validate_flags(version, flags)?;
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
            reader,
            num_centroids
                .checked_mul(dimension)
                .ok_or_else(|| Error::CorruptFile("centroid size overflow".to_string()))?,
        )?;
        let bucket_cutoffs = read_f32s(
            reader,
            num_buckets
                .checked_sub(1)
                .ok_or_else(|| Error::CorruptFile("bucket count underflow".to_string()))?,
        )?;
        let bucket_weights = read_f32s(reader, num_buckets)?;
        let row_addresses = read_u64s(reader, num_documents)?;
        let document_offsets = read_u64s(
            reader,
            num_documents
                .checked_add(1)
                .ok_or_else(|| Error::CorruptFile("document offset count overflow".to_string()))?,
        )?;
        let token_codes = read_u32s(reader, num_tokens)?;
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
            reader,
            num_centroids
                .checked_add(1)
                .ok_or_else(|| Error::CorruptFile("posting offset count overflow".to_string()))?,
        )?;
        let postings = read_u32s(reader, num_postings)?;
        let exact_store = if has_exact_store {
            let public_row_ids = read_u64s(reader, num_documents)?;
            let raw_value_count = num_tokens.checked_mul(dimension).ok_or_else(|| {
                Error::CorruptFile("exact-store value count overflow".to_string())
            })?;
            let raw_token_values = read_f32s(reader, raw_value_count)?;
            Some(ExactStore {
                public_row_ids,
                raw_token_values,
            })
        } else {
            None
        };
        let mut trailing = [0_u8; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(Error::CorruptFile(
                "unexpected trailing bytes after final PLAID section".to_string(),
            ));
        }

        let centroids = Array2::from_shape_vec((num_centroids, dimension), centroid_values)
            .map_err(|error| Error::CorruptFile(error.to_string()))?;
        Self::try_from_parts_with_exact_store(
            centroids,
            quantizer,
            row_addresses,
            document_offsets,
            token_codes,
            packed_residuals,
            posting_offsets,
            postings,
            exact_store,
        )
        .map_err(|error| Error::CorruptFile(error.to_string()))
    }
}

fn validate_encoded_len(bytes: &[u8]) -> Result<()> {
    if bytes.len() < HEADER_LEN {
        return Err(Error::CorruptFile(format!(
            "file is shorter than the {HEADER_LEN}-byte header: {} bytes",
            bytes.len()
        )));
    }
    let mut reader = Cursor::new(bytes);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(Error::CorruptFile(format!("invalid magic bytes {magic:?}")));
    }
    let version = PlaidFormatVersion::try_from(reader.read_u16::<LittleEndian>()?)?;
    let nbits = reader.read_u8()?;
    if !matches!(nbits, 2 | 4) {
        return Err(Error::CorruptFile(format!(
            "nbits must be 2 or 4, got {nbits}"
        )));
    }
    let flags = reader.read_u8()?;
    let has_exact_store = validate_flags(version, flags)?;
    let marker = reader.read_u32::<LittleEndian>()?;
    if marker != ENDIAN_MARKER {
        return Err(Error::CorruptFile(format!(
            "invalid endian marker {marker:#010x}"
        )));
    }

    let dimension = usize_len(u64::from(reader.read_u32::<LittleEndian>()?), "dimension")?;
    let num_centroids = usize_len(
        u64::from(reader.read_u32::<LittleEndian>()?),
        "centroid count",
    )?;
    let num_documents = usize_len(reader.read_u64::<LittleEndian>()?, "document count")?;
    let num_tokens = usize_len(reader.read_u64::<LittleEndian>()?, "token count")?;
    let num_postings = usize_len(reader.read_u64::<LittleEndian>()?, "posting count")?;
    if dimension == 0 || num_centroids == 0 {
        return Err(Error::CorruptFile(
            "dimension and centroid count must be positive".to_string(),
        ));
    }
    if num_documents > u32::MAX as usize {
        return Err(Error::CorruptFile(format!(
            "document count {num_documents} exceeds PLAID v1 limit {}",
            u32::MAX
        )));
    }
    if num_postings > num_tokens {
        return Err(Error::CorruptFile(format!(
            "posting count {num_postings} exceeds token count {num_tokens}"
        )));
    }
    let residual_bits = dimension
        .checked_mul(usize::from(nbits))
        .ok_or_else(|| Error::CorruptFile("residual bit width overflow".to_string()))?;
    if residual_bits % 8 != 0 {
        return Err(Error::CorruptFile(format!(
            "dimension * nbits must be divisible by 8, got {dimension} * {nbits}"
        )));
    }
    let num_buckets = 1_usize << nbits;
    let mut expected = HEADER_LEN;
    expected = checked_section_len(
        expected,
        num_centroids
            .checked_mul(dimension)
            .ok_or_else(|| Error::CorruptFile("centroid count overflow".to_string()))?,
        std::mem::size_of::<f32>(),
        "centroids",
    )?;
    expected = checked_section_len(
        expected,
        num_buckets - 1,
        std::mem::size_of::<f32>(),
        "bucket cutoffs",
    )?;
    expected = checked_section_len(
        expected,
        num_buckets,
        std::mem::size_of::<f32>(),
        "bucket weights",
    )?;
    expected = checked_section_len(
        expected,
        num_documents,
        std::mem::size_of::<u64>(),
        "row addresses",
    )?;
    expected = checked_section_len(
        expected,
        num_documents
            .checked_add(1)
            .ok_or_else(|| Error::CorruptFile("document offset count overflow".to_string()))?,
        std::mem::size_of::<u64>(),
        "document offsets",
    )?;
    expected = checked_section_len(
        expected,
        num_tokens,
        std::mem::size_of::<u32>(),
        "token codes",
    )?;
    expected = checked_section_len(expected, num_tokens, residual_bits / 8, "packed residuals")?;
    expected = checked_section_len(
        expected,
        num_centroids
            .checked_add(1)
            .ok_or_else(|| Error::CorruptFile("posting offset count overflow".to_string()))?,
        std::mem::size_of::<u64>(),
        "posting offsets",
    )?;
    expected = checked_section_len(
        expected,
        num_postings,
        std::mem::size_of::<u32>(),
        "postings",
    )?;
    if has_exact_store {
        expected = checked_section_len(
            expected,
            num_documents,
            std::mem::size_of::<u64>(),
            "exact-store public row IDs",
        )?;
        let raw_value_count = num_tokens
            .checked_mul(dimension)
            .ok_or_else(|| Error::CorruptFile("exact-store value count overflow".to_string()))?;
        expected = checked_section_len(
            expected,
            raw_value_count,
            std::mem::size_of::<f32>(),
            "exact-store raw token values",
        )?;
    }
    if expected != bytes.len() {
        return Err(Error::CorruptFile(format!(
            "encoded length mismatch: header requires {expected} bytes, file has {}",
            bytes.len()
        )));
    }
    Ok(())
}

fn validate_flags(version: PlaidFormatVersion, flags: u8) -> Result<bool> {
    match version {
        PlaidFormatVersion::V1 if flags == 0 => Ok(false),
        PlaidFormatVersion::V1 => Err(Error::CorruptFile(format!(
            "PLAID V1 flags must be zero, got {flags:#04x}"
        ))),
        PlaidFormatVersion::V2 if flags == EXACT_STORE_FLAG => Ok(true),
        PlaidFormatVersion::V2 if flags & !EXACT_STORE_FLAG != 0 => Err(Error::CorruptFile(
            format!("PLAID V2 contains unknown flags {flags:#04x}"),
        )),
        PlaidFormatVersion::V2 => Err(Error::CorruptFile(
            "PLAID V2 requires the exact-store flag".to_string(),
        )),
    }
}

fn checked_section_len(current: usize, count: usize, width: usize, name: &str) -> Result<usize> {
    let bytes = count
        .checked_mul(width)
        .ok_or_else(|| Error::CorruptFile(format!("{name} byte length overflow")))?;
    current
        .checked_add(bytes)
        .ok_or_else(|| Error::CorruptFile(format!("encoded length overflow at {name}")))
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

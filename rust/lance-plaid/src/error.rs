// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt::{Display, Formatter};

/// Errors returned by the PLAID codec, storage, and search pipeline.
#[derive(Debug)]
pub enum Error {
    /// An API input or index-construction argument is invalid.
    InvalidInput(String),
    /// A persisted PLAID file violates the versioned format.
    CorruptFile(String),
    /// A local file operation failed.
    Io(std::io::Error),
    /// An ndarray could not be constructed with the requested shape.
    Shape(ndarray::ShapeError),
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(formatter, "invalid PLAID input: {message}"),
            Self::CorruptFile(message) => write!(formatter, "corrupt PLAID file: {message}"),
            Self::Io(source) => write!(formatter, "PLAID I/O error: {source}"),
            Self::Shape(source) => write!(formatter, "invalid PLAID array shape: {source}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Shape(source) => Some(source),
            Self::InvalidInput(_) | Self::CorruptFile(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(source: ndarray::ShapeError) -> Self {
        Self::Shape(source)
    }
}

/// Result type for the PLAID crate.
pub type Result<T> = std::result::Result<T, Error>;

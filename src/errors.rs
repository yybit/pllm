use std::io;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum RlmError {
    #[error(transparent)]
    Io(#[from] io::Error),

    #[error("Invalid ggml type: {0}")]
    InvalidGgmlType(u32),

    #[error("Invalid gguf metadata value type: {0}")]
    InvalidGgufMetadataValueType(u32),

    #[error("{0}")]
    Other(String),
}

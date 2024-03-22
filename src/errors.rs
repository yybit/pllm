use std::io;

use thiserror::Error;

use crate::gguf::GgmlType;

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

    #[error("Invalid gguf metadata key: {0}")]
    InvalidGgufMetadataKey(String),

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Ggml type not support: {0:?}")]
    GgmlTypeNotSupport(GgmlType),
}

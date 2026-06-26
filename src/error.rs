use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("config: {0}")]
    Config(String),
    #[error("embedding endpoint: {0}")]
    Embed(String),
    #[error("index: {0}")]
    Index(String),
    #[error("chunking: {0}")]
    Chunk(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

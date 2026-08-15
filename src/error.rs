use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("config: {0}")]
    Config(String),
    #[error("embedding endpoint: {0}")]
    Embed(String),
    /// The endpoint rejected the request for authentication, not connectivity.
    /// Retrying cannot help, and every other file will fail the same way.
    #[error("embedding endpoint auth: {0}")]
    EmbedAuth(String),
    /// A transient endpoint or transport problem — timeout, overload, reset.
    /// Worth retrying; not a property of the file.
    #[error("embedding endpoint (transient): {0}")]
    EmbedTransient(String),
    /// The endpoint rejected an input for exceeding its context window. Carries
    /// the server's own numbers, which are authoritative even when `/props` is
    /// unavailable — the engine uses `n_ctx` to re-split and retry the file.
    #[error(
        "embedding input too large: {n_prompt_tokens} tokens exceeds the endpoint's \
         {n_ctx}-token context window"
    )]
    EmbedContextExceeded {
        n_prompt_tokens: usize,
        n_ctx: usize,
    },
    #[error("index: {0}")]
    Index(String),
    #[error("chunking: {0}")]
    Chunk(String),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Whether retrying the same request could plausibly succeed.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(self, Error::EmbedTransient(_) | Error::Timeout(_))
    }

    /// Whether this condition dooms every remaining file, so the reconcile
    /// should stop rather than repeat it once per file.
    #[must_use]
    pub fn is_fatal_for_run(&self) -> bool {
        matches!(self, Error::EmbedAuth(_))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

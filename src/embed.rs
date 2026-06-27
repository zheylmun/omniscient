//! Embeddings: Embedder trait + llama.cpp HTTP backend (/v1/embeddings) + mock.
use crate::config::EmbedderConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::process::Stdio;
use std::time::{Duration, Instant};

/// Bounds for splitting a list of texts into `embed()` batches. A batch is flushed
/// before adding an item that would exceed either bound (a single item larger than
/// `max_bytes` is sent alone — we never split an item).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLimits {
    pub max_chunks: usize,
    /// Byte budget per batch, measured as the sum of `String::len()` (UTF-8 bytes).
    pub max_bytes: usize,
}

#[async_trait]
pub trait Embedder: Send + Sync {
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed `texts` in order, splitting into batches bounded by `limits`, calling
    /// `embed()` once per batch. Returns exactly `texts.len()` vectors in input order.
    /// Serial by design: batches run one after another.
    async fn embed_batched(&self, texts: &[String], limits: BatchLimits) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut start = 0;
        let mut cur_bytes = 0usize;
        let mut cur_len = 0usize;
        for (i, t) in texts.iter().enumerate() {
            let would_overflow = cur_len > 0
                && (cur_len >= limits.max_chunks
                    || cur_bytes.saturating_add(t.len()) > limits.max_bytes);
            if would_overflow {
                let batch = &texts[start..i];
                let vecs = self.embed(batch).await?;
                if vecs.len() != batch.len() {
                    return Err(Error::Embed(format!(
                        "embedder returned {} vectors for a batch of {}",
                        vecs.len(),
                        batch.len()
                    )));
                }
                out.extend(vecs);
                start = i;
                cur_bytes = 0;
                cur_len = 0;
            }
            cur_bytes = cur_bytes.saturating_add(t.len());
            cur_len += 1;
        }
        if cur_len > 0 {
            let batch = &texts[start..];
            let vecs = self.embed(batch).await?;
            if vecs.len() != batch.len() {
                return Err(Error::Embed(format!(
                    "embedder returned {} vectors for a batch of {}",
                    vecs.len(),
                    batch.len()
                )));
            }
            out.extend(vecs);
        }
        Ok(out)
    }
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub async fn build_embedder(cfg: &EmbedderConfig) -> Result<Box<dyn Embedder>> {
    // Always prefer an already-running endpoint: connect first, and only fall
    // through to spawning when that fails AND auto_start is enabled. This means a
    // user-managed server is used as-is and never spawned over.
    match LlamaCppEmbedder::connect(cfg.base_url.clone(), cfg.model.clone()).await {
        Ok(e) => return Ok(Box::new(e)),
        Err(e) if !cfg.auto_start => return Err(e),
        Err(e) => {
            tracing::info!(
                "embeddings endpoint unreachable ({e}); auto_start is on, launching llama.cpp"
            );
        }
    }
    let server = ManagedServer::spawn(cfg)?;
    Ok(Box::new(connect_managed(server, cfg).await?))
}

/// Parse the `(host, port)` to bind/spawn on from a base URL. Uses the URL's
/// known default port (80/443) when none is given.
fn parse_host_port(base_url: &str) -> Result<(String, u16)> {
    let url = reqwest::Url::parse(base_url)
        .map_err(|e| Error::Embed(format!("invalid base_url {base_url}: {e}")))?;
    let host = url
        .host_str()
        .ok_or_else(|| Error::Embed(format!("base_url has no host: {base_url}")))?
        .to_string();
    let port = url
        .port_or_known_default()
        .ok_or_else(|| Error::Embed(format!("base_url has no port: {base_url}")))?;
    Ok((host, port))
}

/// Whether `host` names the local machine — the only place we can launch a
/// process. Auto-starting a server for a remote `base_url` is nonsensical.
fn is_local_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "0.0.0.0" | "::1")
}

/// The argument vector for `llama serve …`, mirroring the documented manual
/// command. Factored out so it can be unit-tested without spawning.
fn server_args(cfg: &EmbedderConfig, port: u16) -> Vec<String> {
    vec![
        "serve".into(),
        "-hf".into(),
        cfg.hf_repo.clone(),
        "--port".into(),
        port.to_string(),
        "--embedding".into(),
        "--pooling".into(),
        cfg.pooling.clone(),
    ]
}

/// A llama.cpp server process owned by omniscient. Killed when dropped
/// (`kill_on_drop`), so the child never outlives the server that spawned it.
#[derive(Debug)]
pub struct ManagedServer {
    child: tokio::process::Child,
    bin: String,
}

impl ManagedServer {
    fn spawn(cfg: &EmbedderConfig) -> Result<Self> {
        let (host, port) = parse_host_port(&cfg.base_url)?;
        if !is_local_host(&host) {
            return Err(Error::Embed(format!(
                "[embedder] auto_start can only launch a local server, but base_url host is \
                 {host:?}; start llama.cpp manually or point base_url at localhost"
            )));
        }
        let mut cmd = tokio::process::Command::new(&cfg.llama_bin);
        cmd.args(server_args(cfg, port))
            .stdin(Stdio::null())
            // Keep OUR stdout clean — it is reserved for the MCP protocol. The
            // child's stderr is inherited so model download/load progress is
            // visible to the user.
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        let child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::Embed(format!(
                    "[embedder] auto_start is enabled but `{}` was not found. Install \
                     llama.cpp (https://github.com/ggml-org/llama.cpp) and ensure the `llama` \
                     CLI is on PATH, or set [embedder] llama_bin to its full path.",
                    cfg.llama_bin
                ))
            } else {
                Error::Embed(format!("failed to spawn `{}`: {e}", cfg.llama_bin))
            }
        })?;
        Ok(Self {
            child,
            bin: cfg.llama_bin.clone(),
        })
    }
}

/// Poll the endpoint until the spawned server answers (or `auto_start_timeout_secs`
/// elapses), then hand the server's lifetime to the returned embedder. Bails early
/// if the child exits before becoming ready (bad flags, missing model, …).
async fn connect_managed(
    mut server: ManagedServer,
    cfg: &EmbedderConfig,
) -> Result<LlamaCppEmbedder> {
    let deadline = Instant::now() + Duration::from_secs(cfg.auto_start_timeout_secs);
    let mut waited_secs = 0u64;
    loop {
        // If the process already died, looping until timeout would just hide the
        // real failure — surface it now (details are on the inherited stderr).
        if let Ok(Some(status)) = server.child.try_wait() {
            return Err(Error::Embed(format!(
                "`{}` exited before becoming ready ({status}); check the flags/model above",
                server.bin
            )));
        }
        match LlamaCppEmbedder::connect(cfg.base_url.clone(), cfg.model.clone()).await {
            Ok(e) => {
                tracing::info!("llama.cpp embeddings server is ready");
                return Ok(e.with_server(server));
            }
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(Error::Embed(format!(
                        "llama.cpp server did not become ready within {}s: {e}",
                        cfg.auto_start_timeout_secs
                    )));
                }
                if waited_secs.is_multiple_of(10) {
                    tracing::info!(
                        "waiting for llama.cpp to become ready (the model is downloaded on first run, which can take a while)…"
                    );
                }
                tokio::time::sleep(Duration::from_secs(1)).await;
                waited_secs += 1;
            }
        }
    }
}

// ---- Deterministic test embedder ----
pub struct MockEmbedder {
    id: String,
    dim: usize,
}
impl MockEmbedder {
    pub fn new(id: &str, dim: usize) -> Self {
        Self { id: id.into(), dim }
    }
}
#[async_trait]
impl Embedder for MockEmbedder {
    fn id(&self) -> &str {
        &self.id
    }
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0f32; self.dim];
                for (i, b) in t.bytes().enumerate() {
                    // weight is in 1..=7, so the u8 conversion never truncates
                    let weight = f32::from(u8::try_from(i % 7 + 1).unwrap());
                    v[i % self.dim] += (f32::from(b) + 1.0) * weight;
                }
                l2_normalize(&mut v);
                v
            })
            .collect())
    }
}

// ---- llama.cpp HTTP backend ----
pub struct LlamaCppEmbedder {
    base_url: String,
    model: String,
    dim: usize,
    client: reqwest::Client,
    /// A llama.cpp server omniscient spawned itself (`auto_start`), kept alive for
    /// the embedder's lifetime and killed on drop. `None` when connecting to a
    /// user-managed endpoint. Held purely for its `Drop` side effect.
    #[allow(dead_code)]
    server: Option<ManagedServer>,
}

impl LlamaCppEmbedder {
    pub async fn connect(base_url: String, model: String) -> Result<Self> {
        let mut e = Self {
            base_url,
            model,
            dim: 0,
            client: reqwest::Client::new(),
            server: None,
        };
        let probe = e.embed_raw(&["probe".to_string()]).await?;
        e.dim = probe.first().map_or(0, std::vec::Vec::len);
        if e.dim == 0 {
            return Err(Error::Embed(
                "embeddings endpoint returned an empty vector (dim 0)".into(),
            ));
        }
        Ok(e)
    }

    /// Attach a spawned server so its lifetime is tied to this embedder.
    fn with_server(mut self, server: ManagedServer) -> Self {
        self.server = Some(server);
        self
    }

    async fn embed_raw(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(serde::Serialize)]
        struct Req<'a> {
            model: &'a str,
            input: &'a [String],
        }
        #[derive(serde::Deserialize)]
        struct Item {
            embedding: Vec<f32>,
        }
        #[derive(serde::Deserialize)]
        struct Resp {
            data: Vec<Item>,
        }
        let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .json(&Req {
                model: &self.model,
                input: texts,
            })
            .send()
            .await
            .map_err(|e| {
                Error::Embed(format!(
                    "POST {url} failed: {e}. Is llama.cpp serving the embedding model?"
                ))
            })?
            .error_for_status()
            .map_err(|e| Error::Embed(e.to_string()))?
            .json::<Resp>()
            .await
            .map_err(|e| Error::Embed(e.to_string()))?;
        Ok(resp.data.into_iter().map(|it| it.embedding).collect())
    }
}

#[async_trait]
impl Embedder for LlamaCppEmbedder {
    fn id(&self) -> &str {
        &self.model
    }
    fn dim(&self) -> usize {
        self.dim
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let mut rows = self.embed_raw(texts).await?;
        for r in &mut rows {
            l2_normalize(r);
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_embeds_are_normalized_and_stable() {
        let e = MockEmbedder::new("mock-v1", 16);
        let a = e.embed(&["hello".into()]).await.unwrap();
        let b = e.embed(&["hello".into()]).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(a[0].len(), 16);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn distinct_texts_differ() {
        let e = MockEmbedder::new("mock-v1", 16);
        let v = e.embed(&["alpha".into(), "beta".into()]).await.unwrap();
        assert_ne!(v[0], v[1]);
    }

    #[test]
    fn id_and_dim() {
        let e = MockEmbedder::new("mock-v1", 16);
        assert_eq!(e.id(), "mock-v1");
        assert_eq!(e.dim(), 16);
    }

    #[test]
    fn parses_host_and_port() {
        assert_eq!(
            parse_host_port("http://localhost:8080").unwrap(),
            ("localhost".into(), 8080)
        );
        assert_eq!(
            parse_host_port("http://127.0.0.1:11434").unwrap(),
            ("127.0.0.1".into(), 11434)
        );
        // missing port falls back to the scheme's default
        assert_eq!(
            parse_host_port("https://example.com").unwrap(),
            ("example.com".into(), 443)
        );
        assert!(parse_host_port("not a url").is_err());
    }

    #[test]
    fn local_host_detection() {
        assert!(is_local_host("localhost"));
        assert!(is_local_host("127.0.0.1"));
        assert!(is_local_host("::1"));
        assert!(!is_local_host("example.com"));
        assert!(!is_local_host("10.0.0.5"));
    }

    #[test]
    fn server_args_mirror_documented_command() {
        // The defaults already match the documented command.
        let cfg = EmbedderConfig::default();
        assert_eq!(
            server_args(&cfg, 8080),
            vec![
                "serve",
                "-hf",
                "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M",
                "--port",
                "8080",
                "--embedding",
                "--pooling",
                "last",
            ]
        );
    }

    #[tokio::test]
    async fn auto_start_refuses_remote_host() {
        // The locality guard runs before any spawn, so this needs no binary.
        let cfg = EmbedderConfig {
            base_url: "http://embeddings.example.com:8080".into(),
            ..Default::default()
        };
        let err = ManagedServer::spawn(&cfg).unwrap_err();
        assert!(
            matches!(&err, Error::Embed(m) if m.contains("local")),
            "expected a 'local server only' error, got {err:?}"
        );
    }

    // A spy embedder that records the length of every embed() batch it receives,
    // delegating the actual vectors to a MockEmbedder.
    struct SpyEmbedder {
        inner: MockEmbedder,
        calls: std::sync::Mutex<Vec<usize>>,
    }
    impl SpyEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                inner: MockEmbedder::new("spy", dim),
                calls: std::sync::Mutex::new(vec![]),
            }
        }
    }
    #[async_trait]
    impl Embedder for SpyEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.lock().unwrap().push(texts.len());
            self.inner.embed(texts).await
        }
    }

    fn texts(parts: &[&str]) -> Vec<String> {
        parts.iter().map(std::string::ToString::to_string).collect()
    }

    #[tokio::test]
    async fn batched_packs_by_count() {
        let e = SpyEmbedder::new(16);
        let t = texts(&["a", "b", "c", "d", "e"]);
        let limits = BatchLimits {
            max_chunks: 2,
            max_bytes: 1_000_000,
        };
        let out = e.embed_batched(&t, limits).await.unwrap();
        assert_eq!(out.len(), 5, "one vector per input, in order");
        assert_eq!(
            *e.calls.lock().unwrap(),
            vec![2, 2, 1],
            "5 items, cap 2 -> 2+2+1"
        );
    }

    #[tokio::test]
    async fn batched_packs_by_bytes() {
        let e = SpyEmbedder::new(16);
        // each item is 4 bytes; max_bytes=8 -> 2 per batch
        let t = texts(&["aaaa", "bbbb", "cccc"]);
        let limits = BatchLimits {
            max_chunks: 1000,
            max_bytes: 8,
        };
        let out = e.embed_batched(&t, limits).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(
            *e.calls.lock().unwrap(),
            vec![2, 1],
            "8-byte budget -> 4+4, then 4"
        );
    }

    #[tokio::test]
    async fn batched_oversized_single_chunk_goes_alone() {
        let e = SpyEmbedder::new(16);
        let t = texts(&["aa", "bbbbbbbbbb", "cc"]); // middle item is 10 bytes > budget 4
        let limits = BatchLimits {
            max_chunks: 1000,
            max_bytes: 4,
        };
        let out = e.embed_batched(&t, limits).await.unwrap();
        assert_eq!(out.len(), 3);
        // "aa"(2) fits; adding "bbbbbbbbbb" would exceed -> flush [aa]; the big one is
        // alone (it exceeds the budget by itself); "cc" follows in its own batch.
        assert_eq!(*e.calls.lock().unwrap(), vec![1, 1, 1]);
        // no batch ever exceeds the count limit
        assert!(e.calls.lock().unwrap().iter().all(|&n| n <= 1000));
    }

    #[tokio::test]
    async fn batched_empty_input_makes_no_calls() {
        let e = SpyEmbedder::new(16);
        let out = e
            .embed_batched(
                &[],
                BatchLimits {
                    max_chunks: 4,
                    max_bytes: 100,
                },
            )
            .await
            .unwrap();
        assert!(out.is_empty());
        assert!(
            e.calls.lock().unwrap().is_empty(),
            "no embed() calls for empty input"
        );
    }

    #[tokio::test]
    async fn batched_equals_unbatched_elementwise() {
        let e = MockEmbedder::new("mock-v1", 16);
        let t = texts(&["alpha", "beta", "gamma", "delta", "epsilon"]);
        let whole = e.embed(&t).await.unwrap();
        let batched = e
            .embed_batched(
                &t,
                BatchLimits {
                    max_chunks: 2,
                    max_bytes: 7,
                },
            )
            .await
            .unwrap();
        assert_eq!(whole, batched, "batching must not change vectors or order");
    }
}

#[cfg(test)]
mod live {
    use super::*;
    #[tokio::test]
    #[ignore = "requires a running llama.cpp embeddings server"]
    async fn live_embed_dim_and_norm() {
        let e =
            LlamaCppEmbedder::connect("http://localhost:8080".into(), "qwen3-embedding-4b".into())
                .await
                .unwrap();
        assert!(e.dim() > 0);
        let v = e.embed(&["fn main() {}".into()]).await.unwrap();
        let n: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }
}

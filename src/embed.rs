//! Embeddings: Embedder trait + llama.cpp HTTP backend (/v1/embeddings) + mock.
use crate::caps::{CapsSource, ServerCaps, parse_models, parse_props};
use crate::config::EmbedderConfig;
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::process::Stdio;
use std::sync::Arc;
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

    /// Tokens the endpoint accepts in a single input, when it told us. `None`
    /// means unknown — callers fall back to configuration rather than guessing.
    fn max_input_tokens(&self) -> Option<usize> {
        None
    }

    /// How many embedding requests the endpoint can serve concurrently, when it
    /// says. `None` means unknown — callers stay serial rather than guessing.
    fn max_concurrent_requests(&self) -> Option<usize> {
        None
    }

    /// Which probe produced `max_input_tokens`, for diagnostics reporting.
    fn caps_source(&self) -> CapsSource {
        CapsSource::None
    }

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

pub async fn build_embedder(cfg: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
    let api_key = cfg.resolved_api_key()?;
    // Always prefer an already-running endpoint: connect first, and only fall
    // through to spawning when that fails AND auto_start is enabled. This means a
    // user-managed server is used as-is and never spawned over.
    match LlamaCppEmbedder::connect(
        cfg.base_url.clone(),
        cfg.model.clone(),
        cfg.request_timeout_secs,
        api_key.clone(),
    )
    .await
    {
        Ok(e) => return Ok(Arc::new(e)),
        Err(e) if !cfg.auto_start => return Err(e),
        Err(e) => {
            // `connect()` failing doesn't always mean the port is free: a server
            // could be bound but misconfigured (wrong model, not in embedding
            // mode, bad response). Spawning then would just collide on the port,
            // breaking the "never spawn over a running endpoint" invariant — so
            // only spawn when nothing is actually listening; otherwise surface the
            // real error.
            if endpoint_listening(&cfg.base_url).await {
                return Err(e);
            }
            tracing::info!(
                "embeddings endpoint unreachable ({e}); auto_start is on, launching llama.cpp"
            );
        }
    }
    let server = ManagedServer::spawn(cfg)?;
    Ok(Arc::new(connect_managed(server, cfg, api_key).await?))
}

/// Build a capability-probe URL, optionally scoping it to a model id.
///
/// The id is appended as a real query parameter rather than interpolated, because
/// llama.cpp ids routinely contain `/` and `:` (`Qwen/Qwen3-Embedding-8B-GGUF:Q8_0`)
/// and must be percent-encoded to survive the query string. `None` when `base_url`
/// does not parse, so a malformed config degrades to "no caps" like every other
/// probe failure.
fn probe_url(base_url: &str, path: &str, model: Option<&str>) -> Option<String> {
    let raw = format!("{}{path}", base_url.trim_end_matches('/'));
    let mut url = reqwest::Url::parse(&raw).ok()?;
    if let Some(m) = model {
        url.query_pairs_mut().append_pair("model", m);
    }
    Some(url.into())
}

/// Whether something is accepting TCP connections at `base_url`'s host:port. Used
/// to distinguish "nothing is bound here" (safe to spawn) from "a server is up but
/// answered badly" (must not spawn over it). A malformed URL or DNS failure counts
/// as not listening.
pub(crate) async fn endpoint_listening(base_url: &str) -> bool {
    let Ok((host, port)) = parse_host_port(base_url) else {
        return false;
    };
    matches!(
        tokio::time::timeout(
            Duration::from_secs(2),
            tokio::net::TcpStream::connect((host.as_str(), port)),
        )
        .await,
        Ok(Ok(_))
    )
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
/// `0.0.0.0` is excluded: it's a wildcard *bind* address, not a client-facing
/// destination, so it has no business in a `base_url`.
fn is_local_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// Smallest context/batch size we will ever ask the spawned server for, so a tiny
/// `max_batch_bytes` can't shrink it to something uselessly small. llama.cpp's own
/// historical default; comfortably below any embedding model's trained context.
const MIN_SERVER_CTX: usize = 2048;

/// Context/batch size for the spawned server, derived from our own batching budget.
///
/// Only **per-item** size is actually enforced by llama.cpp, and that is what
/// `enforce_byte_budget` bounds. Two constraints once documented here were
/// measured false against build `b9821` (4B and 8B): a pooled embedding sequence
/// does *not* have to fit whole in one ubatch (llama.cpp splits it across
/// ubatches), and batch *totals* are not bounded by `n_ctx` (20 inputs totalling
/// 60020 tokens succeeded on a server reporting `n_ctx: 40960`).
///
/// So this sizing is deliberately over-conservative rather than load-bearing: a
/// token is always ≥ 1 byte, which makes `max_batch_bytes` a safe upper bound on
/// the tokens in any single request, and tying the two to one knob means they
/// cannot drift. It costs KV/compute memory on a spawned server but cannot cause
/// the "starts fine, dies mid-use" failure. Right-sizing it is a perf change, not
/// a correctness one.
fn server_ctx_size(cfg: &EmbedderConfig) -> usize {
    cfg.max_batch_bytes.max(MIN_SERVER_CTX)
}

/// The argument vector for `llama serve …`, mirroring the documented manual
/// command. Factored out so it can be unit-tested without spawning.
///
/// `--alias <model>` is what keeps the spawned server addressable. `llama serve`
/// is a router: it registers a `-hf`-loaded model under the repo id and answers
/// any *other* id with `400 model '<id>' not found`, including on `/props?model=`.
/// Without the alias, a `model` that differs from `hf_repo` — for any reason, a
/// typo included — makes every embeddings request fail against a server we just
/// started ourselves, and because that 400 never resolves by waiting, the
/// readiness poll would sit there until `auto_start_timeout_secs`. Aliasing the
/// server to the id we are going to ask for makes the two agree by construction.
fn server_args(cfg: &EmbedderConfig, port: u16) -> Vec<String> {
    // ctx == batch == ubatch: one conservative bound for all three, for the
    // reasons (and with the caveats) in `server_ctx_size`.
    let ctx = server_ctx_size(cfg).to_string();
    vec![
        "serve".into(),
        "-hf".into(),
        cfg.hf_repo.clone(),
        "--alias".into(),
        cfg.model.clone(),
        "--port".into(),
        port.to_string(),
        "--embedding".into(),
        "--pooling".into(),
        cfg.pooling.clone(),
        "--ctx-size".into(),
        ctx.clone(),
        "--batch-size".into(),
        ctx.clone(),
        "--ubatch-size".into(),
        ctx,
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
///
/// Two things keep a misconfiguration from masquerading as a slow first-run
/// download, which is the failure this generous timeout exists to tolerate:
///
/// - **A rejected API key aborts immediately.** `EmbedAuth` is the one error that
///   provably never resolves by waiting — the server is up and answering, it just
///   refuses us — so polling it for ten minutes only delays a message the user
///   needs now.
/// - **Every other error is still polled, but no longer silently.** A 400 from
///   the router *can* be a startup race (the model is registered only once its
///   GGUF has been fetched and loaded), so treating non-retryable errors as fatal
///   would trade a slow failure for a flaky one. Instead the actual error is
///   logged alongside the "still waiting" line, so a wait that is never going to
///   end says why on its face.
async fn connect_managed(
    mut server: ManagedServer,
    cfg: &EmbedderConfig,
    api_key: Option<String>,
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
        match LlamaCppEmbedder::connect(
            cfg.base_url.clone(),
            cfg.model.clone(),
            cfg.request_timeout_secs,
            api_key.clone(),
        )
        .await
        {
            Ok(e) => {
                tracing::info!("llama.cpp embeddings server is ready");
                return Ok(e.with_server(server));
            }
            // A rejected key is the one failure that waiting cannot fix.
            Err(e) if e.is_fatal_for_run() => return Err(e),
            Err(e) => {
                if Instant::now() >= deadline {
                    return Err(Error::Embed(format!(
                        "llama.cpp server did not become ready within {}s: {e}",
                        cfg.auto_start_timeout_secs
                    )));
                }
                if waited_secs.is_multiple_of(10) {
                    tracing::info!(
                        error = %e,
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
    max_input_tokens: Option<usize>,
    /// Simulated context window. When set, oversized inputs fail with the same
    /// structured error llama.cpp returns, so the adaptive budget loop is
    /// exercisable offline.
    context_limit: Option<usize>,
    /// Bytes per simulated token, so a test can pick a ratio the budget must
    /// discover.
    mock_bytes_per_token: usize,
    /// Remaining simulated transport blips. Shared + atomic so `&self` can
    /// decrement it.
    fail_remaining: Arc<std::sync::atomic::AtomicUsize>,
    /// Total `embed` calls made, so a test can assert that a caller gave up
    /// early rather than grinding through every file.
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl MockEmbedder {
    pub fn new(id: &str, dim: usize) -> Self {
        Self {
            id: id.into(),
            dim,
            max_input_tokens: None,
            context_limit: None,
            mock_bytes_per_token: 1,
            fail_remaining: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            calls: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Fail the next `n` embed calls with a retryable transport error, then
    /// succeed. Exercises the retry ladder without a server.
    #[must_use]
    pub fn failing_times(mut self, n: usize) -> Self {
        self.fail_remaining = Arc::new(std::sync::atomic::AtomicUsize::new(n));
        self
    }

    /// Fail *every* embed call with a retryable transport error — a wholly-down
    /// endpoint, as opposed to `failing_times`' recoverable blip. The two are
    /// indistinguishable per-request, which is the whole reason the reconcile
    /// loop needs a consecutive-failure circuit breaker.
    #[must_use]
    pub fn always_transiently_failing(self) -> Self {
        self.failing_times(usize::MAX)
    }

    /// How many `embed` calls have been made against this embedder.
    pub fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// A handle sharing this embedder's call counter, for reading the count after
    /// the embedder itself has been moved into an `Engine`.
    #[must_use]
    pub fn call_counter(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.calls)
    }

    /// Declare a per-input token budget, as a probed real server would.
    #[must_use]
    pub fn with_max_input_tokens(mut self, tokens: usize) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Simulate a real context window: inputs whose byte length implies more
    /// than `n_ctx` tokens (at `bytes_per_token` bytes/token) fail with
    /// `Error::EmbedContextExceeded`, matching llama.cpp's structured 400.
    #[must_use]
    pub fn with_context_limit(mut self, n_ctx: usize, bytes_per_token: usize) -> Self {
        self.context_limit = Some(n_ctx);
        self.mock_bytes_per_token = bytes_per_token.max(1);
        self
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
    fn max_input_tokens(&self) -> Option<usize> {
        self.max_input_tokens
    }
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use std::sync::atomic::Ordering;
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.fail_remaining.load(Ordering::Relaxed) > 0 {
            self.fail_remaining.fetch_sub(1, Ordering::Relaxed);
            return Err(Error::EmbedTransient("simulated blip".into()));
        }
        if let Some(n_ctx) = self.context_limit {
            for t in texts {
                // Ceiling division: a partial token still counts.
                let n_prompt_tokens = t.len().div_ceil(self.mock_bytes_per_token);
                if n_prompt_tokens > n_ctx {
                    return Err(Error::EmbedContextExceeded {
                        n_prompt_tokens,
                        n_ctx,
                    });
                }
            }
        }
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
    /// Limits the server reported at connect time. Empty when neither `/props`
    /// nor `/v1/models` was informative — a non-llama.cpp endpoint exposes
    /// neither, and a `llama serve` supervisor answers `/props` about itself.
    caps: ServerCaps,
    client: reqwest::Client,
    /// Bearer token for the embeddings endpoint, resolved once at connect time.
    /// `None` for an unauthenticated endpoint.
    api_key: Option<String>,
    /// A llama.cpp server omniscient spawned itself (`auto_start`), kept alive for
    /// the embedder's lifetime and killed on drop. `None` when connecting to a
    /// user-managed endpoint. Held purely for its `Drop` side effect.
    #[allow(dead_code)]
    server: Option<ManagedServer>,
}

/// Attach `Authorization: Bearer <key>` when a key is configured. Factored out
/// so the header decision is unit-testable without a live server.
fn apply_auth(rb: reqwest::RequestBuilder, api_key: Option<&str>) -> reqwest::RequestBuilder {
    match api_key {
        Some(key) => rb.bearer_auth(key),
        None => rb,
    }
}

/// llama.cpp's oversize rejection, which is machine-readable rather than prose:
/// `{"error":{"type":"exceed_context_size_error","n_prompt_tokens":N,"n_ctx":M}}`.
/// Returns `(n_prompt_tokens, n_ctx)` only when the type matches AND `n_ctx` is
/// usable — a body without real numbers gives us nothing to retry against, so it
/// stays a generic error.
fn context_exceeded(body: &str) -> Option<(usize, usize)> {
    #[derive(serde::Deserialize)]
    struct Envelope {
        error: Body,
    }
    #[derive(serde::Deserialize)]
    struct Body {
        #[serde(default, rename = "type")]
        kind: String,
        #[serde(default)]
        n_prompt_tokens: usize,
        #[serde(default)]
        n_ctx: usize,
    }
    let env: Envelope = serde_json::from_str(body).ok()?;
    if env.error.kind != "exceed_context_size_error" || env.error.n_ctx == 0 {
        return None;
    }
    Some((env.error.n_prompt_tokens, env.error.n_ctx))
}

/// Build a helpful error for a non-success embeddings response. A `401`/`403`
/// means the request reached the server but was *rejected* — an auth problem,
/// not connectivity — so point at `[embedder] api_key` rather than the
/// "is it serving the model?" hint that fits a transport failure. Factored out
/// so the status→message mapping is unit-testable without a live server.
fn embed_http_error(status: reqwest::StatusCode, url: &str, body: &str) -> Error {
    use reqwest::StatusCode;
    // Bound the body: an OpenAI-compatible router may answer with a full HTML
    // error page, not the small JSON llama.cpp returns.
    let body = body.trim();
    let detail = if body.is_empty() {
        String::new()
    } else {
        let mut b = body.chars().take(300).collect::<String>();
        if body.chars().count() > 300 {
            b.push('…');
        }
        format!(": {b}")
    };
    if let Some((n_prompt_tokens, n_ctx)) = context_exceeded(body) {
        return Error::EmbedContextExceeded {
            n_prompt_tokens,
            n_ctx,
        };
    }
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Error::EmbedAuth(format!(
            "POST {url} rejected with {status}{detail}. The embeddings endpoint requires \
             authentication — set `[embedder] api_key` in omniscient.toml (a literal token, \
             or `${{VAR}}` to read one from the environment) to a valid key."
        )),
        StatusCode::TOO_MANY_REQUESTS => {
            Error::EmbedTransient(format!("POST {url} rate-limited with {status}{detail}"))
        }
        s if s.is_server_error() => {
            Error::EmbedTransient(format!("POST {url} failed with {status}{detail}"))
        }
        _ => Error::Embed(format!("POST {url} failed with {status}{detail}")),
    }
}

impl LlamaCppEmbedder {
    pub async fn connect(
        base_url: String,
        model: String,
        request_timeout_secs: u64,
        api_key: Option<String>,
    ) -> Result<Self> {
        let client = reqwest::ClientBuilder::new()
            .timeout(Duration::from_secs(request_timeout_secs.max(1)))
            .build()
            .map_err(|e| Error::Embed(format!("failed to build HTTP client: {e}")))?;
        let mut e = Self {
            base_url,
            model,
            dim: 0,
            caps: ServerCaps::default(),
            client,
            api_key,
            server: None,
        };
        let probe = e.embed_raw(&["probe".to_string()]).await?;
        e.dim = probe.first().map_or(0, std::vec::Vec::len);
        if e.dim == 0 {
            return Err(Error::Embed(
                "embeddings endpoint returned an empty vector (dim 0)".into(),
            ));
        }
        e.caps = e.probe_caps().await;
        if let Some(t) = e.caps.max_input_tokens {
            tracing::debug!("embeddings endpoint reports a {t}-token context window");
        } else {
            tracing::debug!(
                "embeddings endpoint did not report a context window; \
                 falling back to [embedder] max_chunk_tokens"
            );
        }
        Ok(e)
    }

    /// Best-effort capability discovery, tried in order of how much each probe
    /// can tell us, and accumulated rather than replaced.
    ///
    /// 1. `GET /props?model=<id>` — the best answer, and the one that works in
    ///    both deployment shapes. A bare `llama-server` ignores the unknown query
    ///    parameter; a `llama serve` supervisor uses it to proxy through to the
    ///    backend process instead of describing itself. It is the only probe that
    ///    reports `total_slots` *and* the context window together.
    /// 2. `GET /props` — the unscoped form, for a server that rejects the
    ///    parameter outright.
    /// 3. `GET /v1/models` — `meta.n_ctx` on the loaded model. Never carries a
    ///    slot count, which is why the results are merged with `or` rather than
    ///    replacing one another: discarding a discovered `total_slots` here would
    ///    silently drop embed concurrency to serial.
    ///
    /// Each step runs only while something is still unknown. Every failure path —
    /// transport error, non-2xx, unreadable or unexpected body — contributes empty
    /// caps: a server that cannot describe itself must still be usable.
    async fn probe_caps(&self) -> ServerCaps {
        let mut caps = self.props_caps(Some(&self.model)).await;
        if caps.max_input_tokens.is_none() || caps.total_slots.is_none() {
            caps = caps.or(self.props_caps(None).await);
        }
        if caps.max_input_tokens.is_none() {
            caps = caps.or(self
                .get_text("/v1/models", None)
                .await
                .map(|b| parse_models(&b, &self.model))
                .unwrap_or_default());
        }
        caps
    }

    /// `GET /props`, optionally scoped to a model id.
    async fn props_caps(&self, model: Option<&str>) -> ServerCaps {
        self.get_text("/props", model)
            .await
            .map(|b| parse_props(&b))
            .unwrap_or_default()
    }

    /// `GET {base_url}{path}`, returning the body only on a 2xx. `None` for any
    /// failure, so callers can treat "absent" and "unparseable" identically.
    async fn get_text(&self, path: &str, model: Option<&str>) -> Option<String> {
        let url = probe_url(&self.base_url, path, model)?;
        let resp = apply_auth(self.client.get(&url), self.api_key.as_deref())
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.text().await.ok()
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
        let resp = apply_auth(self.client.post(&url), self.api_key.as_deref())
            .json(&Req {
                model: &self.model,
                input: texts,
            })
            .send()
            .await
            .map_err(|e| {
                // A `send()` failure is transport-level — a reset, refused
                // connection or timeout — so it is worth retrying rather than
                // costing the file its place in the index.
                Error::EmbedTransient(format!(
                    "POST {url} failed: {e}. Is llama.cpp serving the embedding model?"
                ))
            })?;
        let status = resp.status();
        if !status.is_success() {
            // Read the body before erroring: llama.cpp/OpenAI-compatible servers
            // put the real reason (e.g. "Invalid API Key") in the response body.
            let body = resp.text().await.unwrap_or_default();
            return Err(embed_http_error(status, &url, &body));
        }
        let resp = resp
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
    fn max_input_tokens(&self) -> Option<usize> {
        self.caps.max_input_tokens
    }
    fn max_concurrent_requests(&self) -> Option<usize> {
        self.caps.total_slots
    }
    fn caps_source(&self) -> CapsSource {
        self.caps.source
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
    fn probe_url_percent_encodes_the_model_id() {
        // Real llama.cpp model ids carry '/' and ':'. Interpolated raw they would
        // corrupt the query string, and the router would fall back to describing
        // itself — the exact failure the scoped probe exists to avoid.
        let u = probe_url(
            "http://127.0.0.1:8080",
            "/props",
            Some("Qwen/Qwen3-Embedding-8B-GGUF:Q8_0"),
        )
        .unwrap();
        assert_eq!(
            u,
            "http://127.0.0.1:8080/props?model=Qwen%2FQwen3-Embedding-8B-GGUF%3AQ8_0"
        );
    }

    #[test]
    fn probe_url_without_a_model_is_unscoped() {
        assert_eq!(
            probe_url("http://127.0.0.1:8080/", "/props", None).unwrap(),
            "http://127.0.0.1:8080/props"
        );
    }

    #[test]
    fn probe_url_rejects_an_unparseable_base() {
        assert_eq!(probe_url("not a url", "/props", None), None);
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
        // 0.0.0.0 is a wildcard bind address, not a client destination.
        assert!(!is_local_host("0.0.0.0"));
    }

    #[tokio::test]
    async fn endpoint_listening_detects_open_and_closed_ports() {
        // A bound listener: the probe must see it.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(endpoint_listening(&format!("http://127.0.0.1:{port}")).await);
        // Once it's closed, the same port reads as not listening.
        drop(listener);
        assert!(!endpoint_listening(&format!("http://127.0.0.1:{port}")).await);
        // A malformed base_url counts as not listening (so auto_start can proceed).
        assert!(!endpoint_listening("not a url").await);
    }

    #[test]
    fn server_args_mirror_documented_command() {
        // The defaults already match the documented command. ctx/batch/ubatch are
        // sized to max_batch_bytes (32000) so the spawned server can hold the
        // largest request omniscient will send, not llama.cpp's tiny defaults.
        let cfg = EmbedderConfig::default();
        assert_eq!(
            server_args(&cfg, 8080),
            vec![
                "serve",
                "-hf",
                "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
                "--alias",
                "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
                "--port",
                "8080",
                "--embedding",
                "--pooling",
                "last",
                "--ctx-size",
                "32000",
                "--batch-size",
                "32000",
                "--ubatch-size",
                "32000",
            ]
        );
    }

    #[test]
    fn the_spawned_server_is_aliased_to_the_configured_model_id() {
        // `llama serve` routes by model id and answers an unknown one with
        // `400 model '<id>' not found` — on /v1/embeddings AND on /props?model=.
        // A `model` that differs from `hf_repo` would therefore make every request
        // to a server we just started ourselves fail, and since that 400 never
        // resolves by waiting, the readiness poll would burn the whole
        // auto_start_timeout_secs before saying so. `--alias` must carry `model`,
        // not `hf_repo`, so the id we ask for is the id the server registers.
        let cfg = EmbedderConfig {
            model: "my-embedder".into(),
            hf_repo: "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M".into(),
            ..Default::default()
        };
        let args = server_args(&cfg, 8080);
        let alias = args
            .iter()
            .position(|a| a == "--alias")
            .map(|i| args[i + 1].clone());
        assert_eq!(
            alias.as_deref(),
            Some("my-embedder"),
            "--alias must carry `model` so the server registers under the id we send"
        );
        let hf = args
            .iter()
            .position(|a| a == "-hf")
            .map(|i| args[i + 1].clone());
        assert_eq!(
            hf.as_deref(),
            Some("Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M"),
            "-hf still selects which GGUF to fetch"
        );
    }

    #[test]
    fn the_default_model_id_matches_the_default_gguf() {
        // The out-of-the-box config must work against the documented
        // `llama serve -hf <hf_repo>` command. `llama serve` registers a
        // -hf-loaded model under the repo id, so a `model` that differs from
        // `hf_repo` 400s on the very first connect probe — which is exactly what
        // the previous default ("qwen3-embedding-4b") did.
        let cfg = EmbedderConfig::default();
        assert_eq!(
            cfg.model, cfg.hf_repo,
            "the default model id must be the id the default hf_repo registers as"
        );
    }

    #[test]
    fn server_ctx_floors_a_tiny_batch_budget() {
        // A small max_batch_bytes must not shrink the server below the floor.
        let cfg = EmbedderConfig {
            max_batch_bytes: 100,
            ..Default::default()
        };
        assert_eq!(server_ctx_size(&cfg), MIN_SERVER_CTX);
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

    #[test]
    fn apply_auth_sets_bearer_when_present() {
        let client = reqwest::Client::new();
        let req = apply_auth(client.post("http://x.invalid/"), Some("sk-test"))
            .build()
            .unwrap();
        assert_eq!(
            req.headers().get(reqwest::header::AUTHORIZATION).unwrap(),
            "Bearer sk-test"
        );
    }

    #[test]
    fn apply_auth_omits_header_when_absent() {
        let client = reqwest::Client::new();
        let req = apply_auth(client.post("http://x.invalid/"), None)
            .build()
            .unwrap();
        assert!(req.headers().get(reqwest::header::AUTHORIZATION).is_none());
    }

    #[test]
    fn http_error_401_points_at_api_key_and_includes_body() {
        let Error::EmbedAuth(m) = embed_http_error(
            reqwest::StatusCode::UNAUTHORIZED,
            "http://host:8080/v1/embeddings",
            "{\"error\":{\"message\":\"Invalid API Key\"}}",
        ) else {
            unreachable!("a 401 is an auth error")
        };
        assert!(m.contains("401"), "message should name the status: {m}");
        assert!(
            m.contains("api_key"),
            "message should point at api_key: {m}"
        );
        assert!(
            m.contains("Invalid API Key"),
            "message should include the server body: {m}"
        );
    }

    #[test]
    fn http_error_403_also_points_at_api_key() {
        let Error::EmbedAuth(m) = embed_http_error(
            reqwest::StatusCode::FORBIDDEN,
            "http://host:8080/v1/embeddings",
            "",
        ) else {
            unreachable!("a 403 is an auth error")
        };
        assert!(m.contains("api_key"), "403 should point at api_key: {m}");
    }

    #[test]
    fn http_error_500_is_generic_not_auth() {
        let Error::EmbedTransient(m) = embed_http_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            "http://host:8080/v1/embeddings",
            "boom",
        ) else {
            unreachable!("a 5xx is a transient error")
        };
        assert!(m.contains("500"), "message should name the status: {m}");
        assert!(
            !m.contains("api_key"),
            "a non-auth failure must not mislead toward api_key: {m}"
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

    #[test]
    fn embedder_reports_no_budget_by_default() {
        let e = MockEmbedder::new("m", 8);
        assert_eq!(
            e.max_input_tokens(),
            None,
            "an embedder that never probed must report an unknown budget"
        );
    }

    #[test]
    fn mock_embedder_can_declare_a_budget() {
        let e = MockEmbedder::new("m", 8).with_max_input_tokens(2048);
        assert_eq!(e.max_input_tokens(), Some(2048));
    }

    #[tokio::test]
    async fn mock_rejects_oversized_input_like_llama_cpp() {
        // Simulates llama.cpp's structured 400 so the adaptive budget loop is
        // testable without a server.
        let e = MockEmbedder::new("m", 8).with_context_limit(100, 4);
        // 400 bytes / 4 = 100 tokens: exactly at the limit, accepted.
        let ok = e.embed(&["x".repeat(400)]).await;
        assert!(ok.is_ok(), "at-limit input must be accepted");
        // 404 bytes / 4 = 101 tokens: over.
        match e.embed(&["x".repeat(404)]).await {
            Err(Error::EmbedContextExceeded {
                n_prompt_tokens,
                n_ctx,
            }) => {
                assert_eq!(n_prompt_tokens, 101);
                assert_eq!(n_ctx, 100);
            }
            other => panic!("expected EmbedContextExceeded, got {other:?}"),
        }
    }

    #[test]
    fn structured_context_error_is_typed() {
        let body = r#"{"error":{"code":400,"message":"request (15001 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15001,"n_ctx":2048}}"#;
        let err = embed_http_error(
            reqwest::StatusCode::BAD_REQUEST,
            "http://x/v1/embeddings",
            body,
        );
        match err {
            Error::EmbedContextExceeded {
                n_prompt_tokens,
                n_ctx,
            } => {
                assert_eq!(n_prompt_tokens, 15001);
                assert_eq!(n_ctx, 2048);
            }
            other => panic!("expected EmbedContextExceeded, got {other:?}"),
        }
    }

    #[test]
    fn unrelated_400_stays_generic() {
        let body = r#"{"error":{"code":400,"message":"bad input","type":"invalid_request_error"}}"#;
        let err = embed_http_error(
            reqwest::StatusCode::BAD_REQUEST,
            "http://x/v1/embeddings",
            body,
        );
        assert!(
            matches!(err, Error::Embed(_)),
            "a non-context 400 must not be reported as a context overflow"
        );
    }

    #[test]
    fn context_error_without_n_ctx_stays_generic() {
        // A router could echo the type without the numbers; without n_ctx there is
        // nothing to retry against, so it must not become a typed retry signal.
        let body = r#"{"error":{"type":"exceed_context_size_error"}}"#;
        let err = embed_http_error(
            reqwest::StatusCode::BAD_REQUEST,
            "http://x/v1/embeddings",
            body,
        );
        assert!(matches!(err, Error::Embed(_)));
    }

    #[test]
    fn classifies_http_status_into_error_kinds() {
        let url = "http://localhost:8080/v1/embeddings";
        // Auth failure: the request reached the server and was rejected. Retrying
        // cannot help and the whole run is doomed.
        let e = embed_http_error(reqwest::StatusCode::UNAUTHORIZED, url, "");
        assert!(matches!(e, Error::EmbedAuth(_)));
        assert!(e.is_fatal_for_run());
        assert!(!e.is_retryable());

        // Overload: transient, worth retrying.
        for status in [
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            reqwest::StatusCode::SERVICE_UNAVAILABLE,
            reqwest::StatusCode::BAD_GATEWAY,
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        ] {
            let e = embed_http_error(status, url, "");
            assert!(e.is_retryable(), "{status} should be retryable");
            assert!(!e.is_fatal_for_run());
        }

        // A structured overflow is a budget signal, not a failure of either kind.
        let body =
            r#"{"error":{"type":"exceed_context_size_error","n_prompt_tokens":9000,"n_ctx":8192}}"#;
        let e = embed_http_error(reqwest::StatusCode::BAD_REQUEST, url, body);
        assert!(matches!(e, Error::EmbedContextExceeded { .. }));
        assert!(!e.is_retryable());
        assert!(!e.is_fatal_for_run());
    }

    #[test]
    fn auth_errors_are_unchanged() {
        let err = embed_http_error(
            reqwest::StatusCode::UNAUTHORIZED,
            "http://x/v1/embeddings",
            "",
        );
        let msg = err.to_string();
        assert!(msg.contains("api_key"), "auth hint must survive: {msg}");
    }
}

#[cfg(test)]
mod live {
    use super::*;
    #[tokio::test]
    #[ignore = "requires a running llama.cpp embeddings server"]
    async fn live_embed_dim_and_norm() {
        let e = LlamaCppEmbedder::connect(
            "http://localhost:8080".into(),
            "qwen3-embedding-4b".into(),
            30,
            None,
        )
        .await
        .unwrap();
        assert!(e.dim() > 0);
        let v = e.embed(&["fn main() {}".into()]).await.unwrap();
        let n: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4);
    }

    /// Exercises the real discovery path against the machine's actual configured
    /// endpoint, credentials included — hence the full `Config::load` rather than
    /// `EmbedderConfig::default()`, which carries no `api_key` and would only ever
    /// prove that a 401 is a 401. Against a `llama serve` supervisor this
    /// necessarily resolves via `/v1/models`, since its `/props` reports `n_ctx: 0`.
    #[tokio::test]
    #[ignore = "requires a running llama.cpp embeddings server"]
    async fn live_probe_reports_context_window() {
        let cfg = crate::config::Config::load(None, std::env::current_dir().unwrap())
            .expect("load config")
            .embedder;
        let e = LlamaCppEmbedder::connect(
            cfg.base_url.clone(),
            cfg.model.clone(),
            cfg.request_timeout_secs,
            cfg.resolved_api_key().unwrap(),
        )
        .await
        .expect("connect");
        let tokens = e
            .max_input_tokens()
            .expect("server should report its context window via /props or /v1/models");
        assert!(tokens >= 512, "implausible context window: {tokens}");
    }
}

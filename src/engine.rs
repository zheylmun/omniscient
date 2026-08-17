//! Engine: ties freshness + embed + index + distill into the always-fresh search path.
use crate::budget::{ChunkBudget, Tightened};
use crate::chunk::{chunk_file, language_for_path, split_for_embedding};
use crate::config::Config;
use crate::distill::{ContextEntry, distill_context};
use crate::embed::{Embedder, build_embedder};
use crate::error::{Error, Result};
use crate::freshness::{diff, exclude_matcher, is_excluded, resolve_excludes, scan};
use crate::index::{Index, StoredChunk};
use crate::refresh::RefreshState;
use futures::stream::{self, StreamExt};
use ignore::gitignore::Gitignore;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::OnceCell;

const MAX_WINDOW_LINES: usize = 80;

/// Backstop on re-split rounds after a reported context overflow. Not the real
/// terminator — each round's budget is strictly below the previous round's, so
/// the loop converges on its own; this only bounds a server whose reported
/// numbers are self-inconsistent. Shared by the indexing and focus-read paths.
const MAX_OVERFLOW_ROUNDS: usize = 8;

/// Stand-in "path" recorded when a reconcile aborted as a whole rather than
/// failing on individual files, so `diagnostics` can render the reason with the
/// same `path: error` shape it uses for per-file failures.
const ABORTED_RUN: &str = "<reconcile aborted>";

/// Consecutive retryable per-file failures after which the endpoint — not the
/// files — is treated as the problem, and the reconcile is abandoned.
///
/// Per-file isolation is right for a bad file and for a blip, but a refused
/// connection also classifies as retryable, and rediscovering a down endpoint
/// once per changed file costs the full retry ladder each time. Five is high
/// enough that an unlucky cluster of genuine blips does not trip it (each of the
/// five already failed three attempts of its own) and low enough that a
/// thousand-file reconcile against a dead endpoint gives up in seconds.
const CONSECUTIVE_TRANSIENT_LIMIT: usize = 5;

/// A single file's work item for parallel reconcile: chunks + metadata.
#[derive(Clone)]
struct WorkItem {
    path: String,
    chunks: Vec<crate::chunk::Chunk>,
    file_hash: String,
}

/// What one reconcile pass accomplished. Errors are kept as strings so the
/// summary is `Clone` and can be handed to `diagnostics` after the fact.
#[derive(Debug, Clone, Default)]
pub struct ReconcileSummary {
    /// Files whose chunks were embedded and written to the index.
    pub indexed: usize,
    /// `(path, error)` for every file that could not be indexed this pass. Their
    /// hashes are unwritten, so `diff` offers them again next reconcile.
    pub failed: Vec<(String, String)>,
}

impl ReconcileSummary {
    /// A pass is clean only when nothing failed — the exact condition under
    /// which it is safe to clear `dirty` and let `search` skip its scan.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failed.is_empty()
    }

    /// One-line-per-file description of the failures, for an error message or the CLI.
    #[must_use]
    pub fn failure_report(&self) -> String {
        self.failed
            .iter()
            .map(|(p, e)| format!("  {p}: {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Restores the `dirty` flag on drop unless disarmed. A reconcile clears `dirty` before
/// scanning; if it is then cancelled (dropped tool future, panic) or errors, the guard
/// re-marks `dirty` so the next search re-scans instead of serving a clean+partial index.
struct DirtyGuard<'a> {
    state: &'a RefreshState,
    armed: bool,
}

impl<'a> DirtyGuard<'a> {
    fn new(state: &'a RefreshState) -> Self {
        state.clear_dirty();
        Self { state, armed: true }
    }
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for DirtyGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.state.mark_dirty();
        }
    }
}

pub struct Engine {
    config: Config,
    embedder: Arc<dyn Embedder>,
    index: Index,
    refresh: Arc<RefreshState>,
    /// Per-chunk byte budget, seeded from the endpoint's reported context
    /// window and corrected by any overflow it reports. Shared across the whole
    /// reconcile so one file's correction benefits every later file.
    budget: ChunkBudget,
    /// Effective exclude globs (lock files + tests + user), resolved once. Drives
    /// both the index-time `scan` and the read-time search filter via `matcher`.
    excludes: Vec<String>,
    matcher: Gitignore,
    /// The most recent reconcile's outcome, for `diagnostics` to report. A
    /// `std::sync::Mutex` is sound here because the guard is only ever taken for
    /// a single non-async assignment/clone and never held across an `.await`.
    last_summary: std::sync::Mutex<ReconcileSummary>,
}

impl Engine {
    pub async fn new(config: Config) -> Result<Engine> {
        let embedder = build_embedder(&config.embedder).await?;
        Self::new_with_embedder(config, embedder).await
    }

    pub async fn new_with_embedder(config: Config, embedder: Arc<dyn Embedder>) -> Result<Engine> {
        Self::with_refresh_state(config, embedder, Arc::new(RefreshState::standalone())).await
    }

    pub async fn with_refresh_state(
        config: Config,
        embedder: Arc<dyn Embedder>,
        refresh: Arc<RefreshState>,
    ) -> Result<Engine> {
        let dir = config.repo_root.join(".omniscient");
        let index = Index::open(
            &dir,
            embedder.id(),
            embedder.dim().max(1),
            crate::chunk::CHUNKER_VERSION,
        )
        .await?;
        let excludes = resolve_excludes(&config.exclude, config.index_tests);
        let matcher = exclude_matcher(&config.repo_root, &excludes)?;
        let budget = ChunkBudget::from_probe(
            embedder.max_input_tokens(),
            config.embedder.chunk_budget_tokens(),
        );
        Ok(Engine {
            config,
            embedder,
            index,
            refresh,
            budget,
            excludes,
            matcher,
            last_summary: std::sync::Mutex::new(ReconcileSummary::default()),
        })
    }

    pub fn embedder_id(&self) -> &str {
        self.embedder.id()
    }

    /// The live per-chunk byte budget, for diagnostics.
    pub fn chunk_budget(&self) -> &ChunkBudget {
        &self.budget
    }

    /// Which probe produced the endpoint's reported context window, for diagnostics.
    #[must_use]
    pub fn caps_source(&self) -> crate::caps::CapsSource {
        self.embedder.caps_source()
    }

    /// The endpoint's reported request-slot count, for diagnostics.
    #[must_use]
    pub fn max_concurrent_requests(&self) -> Option<usize> {
        self.embedder.max_concurrent_requests()
    }

    pub fn refresh_state(&self) -> &Arc<RefreshState> {
        &self.refresh
    }

    pub async fn stats(&self) -> Result<(usize, usize)> {
        Ok((
            self.index.file_hashes().await?.len(),
            self.index.chunk_count().await?,
        ))
    }

    /// Skip the scan entirely when a healthy watcher guarantees freshness;
    /// otherwise reconcile. This is the only search-path change vs. always-scan.
    pub async fn ensure_fresh(&self) -> Result<()> {
        // pre-lock fast path; re-checked under the lock in reconcile()
        if self.refresh.can_skip_scan() {
            return Ok(());
        }
        self.reconcile().await
    }

    /// Single-flight reconcile. Clears `dirty` BEFORE scanning so an event arriving
    /// mid-scan re-sets it (costing at most one redundant reconcile, never a lost update).
    /// `DirtyGuard` restores `dirty` if `reconcile_inner` errors, if any file failed, OR
    /// if the future is cancelled (dropped) before it finishes — otherwise a
    /// failed/cancelled reconcile would leave the state clean+active and let `search` skip
    /// the scan and serve stale.
    ///
    /// `skip_if_fresh` distinguishes the search path (which may bail when a healthy
    /// watcher already guarantees freshness) from the explicit commands (which must
    /// always do the work).
    async fn reconcile_guarded(&self, skip_if_fresh: bool) -> Result<ReconcileSummary> {
        let _guard = self.refresh.lock.lock().await;
        if skip_if_fresh && self.refresh.can_skip_scan() {
            return Ok(ReconcileSummary::default()); // another reconcile beat us
        }
        let dirty_guard = DirtyGuard::new(&self.refresh);
        // Record the outcome either way. A run that aborts (a rejected API key
        // dooms every remaining file, an index write means the store is unhealthy)
        // is the *most* important thing for `diagnostics` to show; storing the
        // summary only on the success path leaves it reporting "no failures"
        // immediately after a run in which nothing was indexed at all.
        let summary = match self.reconcile_inner().await {
            Ok(summary) => summary,
            Err(e) => {
                self.record_summary(ReconcileSummary {
                    indexed: 0,
                    failed: vec![(ABORTED_RUN.to_string(), e.to_string())],
                });
                return Err(e);
            }
        };
        if summary.is_clean() {
            dirty_guard.disarm();
        }
        self.record_summary(summary.clone());
        Ok(summary)
    }

    fn record_summary(&self, summary: ReconcileSummary) {
        *self
            .last_summary
            .lock()
            .expect("reconcile summary lock poisoned") = summary;
    }

    /// Search-path reconcile: tolerant. A partial index plus a set `dirty` flag
    /// beats a `search` that fails outright, and `ensure_fresh` propagates with
    /// `?`, so this must not turn a bad file into a broken tool call.
    pub async fn reconcile(&self) -> Result<()> {
        let summary = self.reconcile_guarded(true).await?;
        if !summary.is_clean() {
            tracing::warn!(
                failed = summary.failed.len(),
                "reconcile finished with failures; index left marked dirty"
            );
        }
        Ok(())
    }

    /// Explicit reconcile (`reindex`, `status`, tests): reports failures, so a
    /// rebuild that did not happen cannot look like one that did.
    pub async fn refresh(&self) -> Result<()> {
        let summary = self.reconcile_guarded(false).await?;
        if summary.is_clean() {
            return Ok(());
        }
        Err(Error::Embed(format!(
            "{} file(s) could not be indexed:\n{}",
            summary.failed.len(),
            summary.failure_report()
        )))
    }

    /// The most recent reconcile's per-file failures, for `diagnostics`.
    #[must_use]
    pub fn last_failures(&self) -> Vec<(String, String)> {
        self.last_summary
            .lock()
            .expect("reconcile summary lock poisoned")
            .failed
            .clone()
    }

    /// Reconcile the index against the working tree. Returns a summary naming
    /// every file whose embedding failed and was therefore skipped: those files
    /// are not in the index and their hashes are unwritten, so the caller leaves
    /// the state dirty and `search` keeps scanning rather than trusting a partial
    /// index (the "embedder was down" case of the always-fresh invariant).
    async fn reconcile_inner(&self) -> Result<ReconcileSummary> {
        let current = scan(&self.config.repo_root, &self.excludes)?;
        let stored = self.index.file_hashes().await?;
        let delta = diff(&current, &stored);
        let hash_of: std::collections::HashMap<&str, &str> = current
            .iter()
            .map(|s| (s.path.as_str(), s.hash.as_str()))
            .collect();

        // Phase 1: decide scope only — which paths are in the configured languages.
        // Deliberately does NOT read or chunk: reading here would buffer the whole
        // changed set (O(repo) memory) or throw the contents away and re-read them
        // in phase 2, and chunking against one budget snapshot taken before any
        // embedding would leave every file holding chunks built against a value
        // that file A's overflow may already have corrected, so each would pay its
        // own rejected round-trip anyway.
        let mut work: Vec<(String, String)> = Vec::new();
        for path in &delta.changed {
            // Gate on the language whitelist: skip files not in the configured languages.
            let detected = language_for_path(Path::new(path));
            if !self.config.is_language_allowed(Path::new(path), detected) {
                continue;
            }
            let file_hash = hash_of
                .get(path.as_str())
                .copied()
                .unwrap_or("")
                .to_string();
            work.push((path.clone(), file_hash));
        }

        // Phase 2: read, chunk and embed in parallel (bounded), write serially.
        // buffer_unordered runs up to `concurrency` embeds at once but yields to a
        // single serial consumer, so index commits never overlap (no LanceDB commit
        // conflicts) and only `concurrency` files are held in flight (O(concurrency)
        // memory, not O(repo)). Each file reads `self.budget.bytes()` here, at the
        // moment it is chunked, so it already reflects a correction made moments
        // earlier by a concurrently-embedding file.
        let concurrency = resolve_concurrency(
            self.config.embedder.embed_concurrency,
            self.embedder.max_concurrent_requests(),
        );
        let mut results = stream::iter(work)
            .map(|(path, file_hash)| async move {
                // Read here, inside the bounded stage, so only `concurrency` file
                // contents are ever live. A path that is not readable as UTF-8 is
                // skipped rather than failed — it is not a broken file, just not
                // one we index — so it must not make `refresh()` exit non-zero.
                let abs = self.config.repo_root.join(&path);
                let Ok(source) = std::fs::read_to_string(&abs) else {
                    return (path, file_hash, None);
                };
                let result = self
                    .embed_file_with_overflow_retry(&path, &file_hash, &source)
                    .await;
                (path, file_hash, Some(result))
            })
            .buffer_unordered(concurrency);

        let mut summary = ReconcileSummary::default();
        let mut consecutive_transient = 0usize;
        while let Some((path, file_hash, result)) = results.next().await {
            // Not readable as UTF-8: not indexed, but not a failure either.
            let Some(result) = result else { continue };
            match result {
                // Index write failures stay fatal: they mean the store itself is
                // unhealthy, not that one file is unembeddable.
                Ok(stored) => {
                    self.index.upsert_file(&path, &file_hash, stored).await?;
                    summary.indexed += 1;
                    // The endpoint answered, so whatever failed before it was not
                    // the endpoint being down.
                    consecutive_transient = 0;
                }
                // A condition that dooms every remaining file (a rejected API key)
                // is not worth repeating once per file: stop the run. The guard in
                // `reconcile` leaves `dirty` set, so nothing is trusted as fresh.
                Err(e) if e.is_fatal_for_run() => return Err(e),
                // Any other embed failure is scoped to this file. Its hash is
                // deliberately not written, so `diff` offers it again next
                // reconcile — correct for a transient outage, and harmless now that
                // chunks are split to fit. The rest of the repo still commits.
                Err(e) => {
                    // ...but "transient" describes one request, not the endpoint.
                    // A refused connection classifies as `EmbedTransient` (right:
                    // a blip must not cost a file its place in the index), and it
                    // is not `is_fatal_for_run`, so a wholly-down endpoint would
                    // otherwise be discovered once per changed file — each paying
                    // the full retry ladder of 3 attempts plus 500 ms of backoff.
                    // On a branch switch that changes thousands of files that turns
                    // a `search` into a multi-minute stall, because `ensure_fresh`
                    // deliberately sits outside `search_timeout_secs`. Enough
                    // consecutive retryable failures is evidence about the endpoint
                    // rather than about the files, so stop and say so.
                    if e.is_retryable() {
                        consecutive_transient += 1;
                        if consecutive_transient >= CONSECUTIVE_TRANSIENT_LIMIT {
                            tracing::warn!(
                                failures = consecutive_transient,
                                error = %e,
                                "embeddings endpoint appears to be down; abandoning this reconcile"
                            );
                            return Err(Error::EmbedTransient(format!(
                                "embeddings endpoint failed on {consecutive_transient} consecutive \
                                 files and is treated as unavailable; last error: {e}"
                            )));
                        }
                    } else {
                        // A per-file problem says nothing about the endpoint.
                        consecutive_transient = 0;
                    }
                    tracing::warn!(path = %path, error = %e, "skipping file: embedding failed");
                    summary.failed.push((path, e.to_string()));
                }
            }
        }
        for path in &delta.deleted {
            self.index.delete_file(path).await?;
        }
        Ok(summary)
    }

    /// Embed a single string, enforcing the embedder contract that exactly one
    /// vector comes back (so a misbehaving endpoint errors instead of panicking).
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut vs = self.embedder.embed(&[text.to_string()]).await?;
        if vs.len() != 1 {
            return Err(Error::Embed(format!(
                "embedder returned {} vectors for 1 input",
                vs.len()
            )));
        }
        Ok(vs.remove(0))
    }

    /// Chunk, embed, and re-split one file until the endpoint stops rejecting it
    /// as too long. Each rejection tightens the shared budget from the server's
    /// own numbers, and tightening strictly decreases until the 1 byte/token
    /// floor, so this loop converges; the bound is a backstop against a server
    /// whose reported numbers are inconsistent.
    ///
    /// The budget is re-read at the top of every round, so a correction another
    /// file made while this one was in flight is already in force.
    async fn embed_file_with_overflow_retry(
        &self,
        path: &str,
        file_hash: &str,
        source: &str,
    ) -> Result<Vec<StoredChunk>> {
        let mut budget = self.budget.bytes();
        let mut rounds_left = MAX_OVERFLOW_ROUNDS;
        loop {
            let (chunks, largest) = chunks_for_embedding(path, source, budget)?;
            let result = self.embed_chunks(path, file_hash, chunks).await;
            rounds_left -= 1;
            let Err(Error::EmbedContextExceeded {
                n_prompt_tokens,
                n_ctx,
            }) = result
            else {
                return result;
            };
            if rounds_left == 0 {
                return result;
            }
            // Correct from the largest chunk's *real* size, not the budget it was
            // split against. The budget is only an upper bound, and a chunk well
            // under it makes the measured bytes-per-token ratio look far more
            // generous than it is — each round would then shrink the budget by
            // only the headroom factor, so a file that splits perfectly well
            // would exhaust the round bound before converging.
            budget = self.next_budget_after_overflow(path, largest, n_prompt_tokens, n_ctx)?;
            tracing::info!(
                path,
                n_prompt_tokens,
                n_ctx,
                bytes = budget,
                "context overflow: re-splitting the file against the corrected chunk budget"
            );
        }
    }

    /// Fold one endpoint overflow into the shared budget and decide what to
    /// re-split against, or fail with a diagnosis of why no smaller budget helps.
    ///
    /// Shared by the indexing path and `read_file`'s focus branch so the floor
    /// test exists once.
    ///
    /// `attempted` is the **largest** byte length in the batch the endpoint
    /// rejected — not the budget it was split against, which both callers used to
    /// pass and which inflates the measured ratio badly whenever chunks land well
    /// under the budget (the common case). It is still an upper bound rather than
    /// the rejected chunk's own size: a batch reports one `n_prompt_tokens`, so
    /// which member overflowed is not knowable without re-sending singly. That
    /// residual only makes the correction less aggressive, never unsafe — the
    /// worst case is a round that shrinks the budget by little more than the
    /// headroom factor, which `MAX_OVERFLOW_ROUNDS` bounds and the shared,
    /// monotonic budget carries into the next reconcile rather than losing.
    fn next_budget_after_overflow(
        &self,
        path: &str,
        attempted: usize,
        n_prompt_tokens: usize,
        n_ctx: usize,
    ) -> Result<usize> {
        match self.budget.tighten(attempted, n_prompt_tokens, n_ctx) {
            Tightened::To(bytes) => Ok(bytes),
            // `Unchanged` alone does NOT mean we are at the floor: a
            // concurrently-embedding file may have tightened past our target
            // between this attempt being chunked and now, in which case its
            // correction is the better one and there is nothing to learn here.
            // The floor is a property of the budget's *state* — the 1 byte/token
            // floor is `n_ctx` bytes — so test that, not the return value, or a
            // benign race is misreported as an unembeddable file.
            Tightened::Unchanged => {
                let current = self.budget.bytes();
                if current < attempted {
                    // Someone else's correction is already tighter than what this
                    // rejection measures; retry against it.
                    Ok(current)
                } else if current <= n_ctx {
                    // Genuinely out of room: this attempt was made at the 1
                    // byte/token floor (`n_ctx` bytes always fits, whatever the
                    // content) and the endpoint still rejected it.
                    Err(Error::Embed(format!(
                        "{path}: endpoint rejected a {n_prompt_tokens}-token input against \
                         its {n_ctx}-token window even at the minimum chunk size"
                    )))
                } else {
                    // Above the floor, yet no smaller budget is on offer: the
                    // server's numbers are inconsistent (an overflow whose own
                    // measurement implies the input fits). Retrying the same size
                    // would spin, so fail — on the indexing path the file stays
                    // unhashed and is offered again next reconcile.
                    Err(Error::Embed(format!(
                        "{path}: endpoint reported a context overflow \
                         ({n_prompt_tokens} tokens against a {n_ctx}-token window) that \
                         does not reduce the {current}-byte chunk budget"
                    )))
                }
            }
        }
    }

    /// Embed one file's already-split chunks.
    async fn embed_chunks(
        &self,
        path: &str,
        file_hash: &str,
        chunks: Vec<crate::chunk::Chunk>,
    ) -> Result<Vec<StoredChunk>> {
        embed_with_retry(
            self.embedder.as_ref(),
            WorkItem {
                path: path.to_string(),
                chunks,
                file_hash: file_hash.to_string(),
            },
            self.config.embedder.batch_limits(),
        )
        .await
    }

    fn query_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.config.search.search_timeout_secs.max(1))
    }

    fn query_timeout_err(&self) -> Error {
        Error::Timeout(format!(
            "query timed out after {}s — embed + index query exceeded the limit \
             (set [search] search_timeout_secs)",
            self.config.search.search_timeout_secs
        ))
    }

    pub async fn search(&self, query: &str, k: Option<usize>) -> Result<Vec<ContextEntry>> {
        // ensure_fresh may run a legitimately slow reconcile (cold/large index); it is NOT
        // under the timeout. Its hang guard is the per-request embedder timeout. The timeout
        // here bounds only the interactive query so a hung endpoint can't block the tool.
        self.ensure_fresh().await?;
        let k = k.unwrap_or(self.config.search.max_results).max(1);
        let hits = tokio::time::timeout(self.query_timeout(), async {
            let qv = self.embed_one(query).await?;
            self.index.search(&qv, k).await
        })
        .await
        .map_err(|_| self.query_timeout_err())??;
        // Enforce the exclude policy at read time too: even if the index hasn't reconciled
        // away an excluded file yet (lag window), never surface it.
        let hits: Vec<_> = hits
            .into_iter()
            .filter(|h| !is_excluded(&self.matcher, &h.chunk.path))
            .collect();
        Ok(distill_context(
            hits,
            self.config.strip_banner_comments,
            self.config.search.token_budget,
            self.config.search.relevance_ratio,
        ))
    }

    pub async fn read_file(&self, path: &str, focus: Option<&str>) -> Result<Vec<ContextEntry>> {
        let abs = self.config.repo_root.join(path);
        let source = std::fs::read_to_string(&abs)?;
        let chunks = chunk_file(Path::new(path), &source, MAX_WINDOW_LINES)?;
        match focus {
            None => Ok(outline_entries(path, chunks)),
            Some(f) => self.focus_entries(path, f, chunks).await,
        }
    }

    /// Rank a file's chunks against a focus string and return the best few.
    async fn focus_entries(
        &self,
        path: &str,
        f: &str,
        chunks: Vec<crate::chunk::Chunk>,
    ) -> Result<Vec<ContextEntry>> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }
        // Ranking means embedding, so the same per-input budget that binds
        // the indexing path binds here: an unsplit chunk from a minified or
        // vendored file would be rejected with a 400 and fail the tool call.
        // Splitting is applied to the focus branch ONLY — the outline must
        // stay structural, one entry per definition.
        //
        // An overflow here is folded back into the shared budget exactly as
        // on the indexing path. Without that, a rejection on a cold index —
        // where the budget is still at its optimistic starting point —
        // would fail this call and teach the budget nothing, so the very
        // next identical call would fail identically. The whole retry loop
        // sits inside ONE timeout so re-splitting cannot multiply the
        // caller's deadline by the round count.
        //
        // Each structural chunk is split on its own so a piece knows
        // whether it is a fragment of something larger: pieces of a split
        // definition share one line span (a minified file is a single
        // line), so the entry says so rather than presenting a fragment as
        // the whole definition.
        let (pieces, fv, cvs) = tokio::time::timeout(self.query_timeout(), async {
            // Embed the focus query ONCE, outside the re-split loop. It is
            // caller-supplied and is never split, so an overflow here is a
            // property of the argument, not a measurement of how large a
            // chunk may be. Feeding it to the shared budget would be a
            // category error that `tighten`'s monotonicity makes permanent,
            // and re-splitting the file's chunks could never make the query
            // itself fit anyway.
            let fv = self.embed_one(f).await.map_err(|e| match e {
                Error::EmbedContextExceeded {
                    n_prompt_tokens,
                    n_ctx,
                } => Error::Embed(format!(
                    "{path}: the focus query is too large to embed \
                             ({n_prompt_tokens} tokens against the endpoint's \
                             {n_ctx}-token window) — use a shorter focus"
                )),
                other => other,
            })?;
            let mut attempted = self.budget.bytes();
            for _ in 0..MAX_OVERFLOW_ROUNDS {
                let pieces = split_focus_pieces(&chunks, attempted);
                let texts: Vec<String> = pieces.iter().map(|(c, _)| c.text.clone()).collect();
                match self
                    .embedder
                    .embed_batched(&texts, self.config.embedder.batch_limits())
                    .await
                {
                    Ok(cvs) => {
                        if cvs.len() != texts.len() {
                            return Err(Error::Embed(format!(
                                "embedder returned {} vectors for {} inputs",
                                cvs.len(),
                                texts.len()
                            )));
                        }
                        return Ok((pieces, fv, cvs));
                    }
                    Err(Error::EmbedContextExceeded {
                        n_prompt_tokens,
                        n_ctx,
                    }) => {
                        // Correct from the largest piece's real size, as
                        // the indexing path does — the budget it was split
                        // against is only an upper bound.
                        let largest = texts.iter().map(String::len).max().unwrap_or(0);
                        attempted =
                            self.next_budget_after_overflow(path, largest, n_prompt_tokens, n_ctx)?;
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(Error::Embed(format!(
                "{path}: focus read still exceeded the endpoint's context window after \
                         {MAX_OVERFLOW_ROUNDS} re-splits"
            )))
        })
        .await
        .map_err(|_| self.query_timeout_err())??;
        let mut scored: Vec<(f32, &(crate::chunk::Chunk, bool))> = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (dot(&fv, &cvs[i]), p))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(5)
            .map(|(score, (c, partial))| ContextEntry {
                path: path.to_string(),
                start_line: c.start_line,
                end_line: c.end_line,
                language: c.language.clone(),
                symbol: c.symbol.clone(),
                code: c.text.clone(),
                score,
                why_matched: if *partial {
                    format!(
                        "focus similarity {score:.3} (partial: split to fit the embedding window)"
                    )
                } else {
                    format!("focus similarity {score:.3}")
                },
            })
            .collect())
    }
}

/// Chunk one file for embedding, reporting the largest piece's byte length
/// alongside the pieces.
///
/// That length is the honest `sent_bytes` for a budget correction. The budget the
/// pieces were split against is only an upper bound on their true size, and using
/// it inflates the measured bytes-per-token ratio whenever a chunk lands well
/// under the budget — which is the common case, since most definitions are far
/// smaller than the window.
fn chunks_for_embedding(
    path: &str,
    source: &str,
    budget: usize,
) -> Result<(Vec<crate::chunk::Chunk>, usize)> {
    let chunks = chunk_file(Path::new(path), source, MAX_WINDOW_LINES)?;
    let chunks = split_for_embedding(chunks, budget);
    let largest = chunks.iter().map(|c| c.text.len()).max().unwrap_or(0);
    Ok((chunks, largest))
}

/// One entry per structural definition, carrying its signature line only.
///
/// Deliberately NOT split for embedding: showing whole definitions is the
/// outline's contract, and splitting here is what once produced entries whose
/// "signature" was a fragment of some body.
fn outline_entries(path: &str, chunks: Vec<crate::chunk::Chunk>) -> Vec<ContextEntry> {
    chunks
        .into_iter()
        .map(|c| ContextEntry {
            path: path.to_string(),
            start_line: c.start_line,
            end_line: c.end_line,
            language: c.language,
            symbol: c.symbol,
            code: c.text.lines().next().unwrap_or("").to_string(),
            score: 0.0,
            why_matched: "outline".into(),
        })
        .collect()
}

/// Split each structural chunk against `budget` on its own, tagging every piece
/// with whether its parent actually split. Per-chunk rather than in bulk because
/// the tag is what lets a focus entry say it is a fragment: pieces of one split
/// definition share a line span, so the span alone cannot distinguish them.
fn split_focus_pieces(
    chunks: &[crate::chunk::Chunk],
    budget: usize,
) -> Vec<(crate::chunk::Chunk, bool)> {
    let mut pieces = Vec::new();
    for c in chunks {
        let parts = split_for_embedding(vec![c.clone()], budget);
        let partial = parts.len() > 1;
        pieces.extend(parts.into_iter().map(|p| (p, partial)));
    }
    pieces
}

/// Concurrency policy: an explicit config value wins, then what the server
/// reports, then serial. The derived value is capped because `total_slots` on a
/// large server can far exceed what saturating it usefully requires.
pub(crate) fn resolve_concurrency(configured: Option<usize>, reported: Option<usize>) -> usize {
    const MAX_DERIVED: usize = 8;
    configured
        .or_else(|| reported.map(|s| s.clamp(1, MAX_DERIVED)))
        .unwrap_or(1)
        .max(1)
}

/// Retry one file's embed through transient endpoint failures. Bounded, with
/// backoff: a blip must not cost a file its place in the index, but a
/// persistently failing endpoint must not be hammered either. A transient
/// failure needs the *same* chunks re-sent, so this retries `embed_work_item`
/// directly rather than re-reading and re-chunking the file.
async fn embed_with_retry(
    embedder: &dyn Embedder,
    item: WorkItem,
    batch_limits: crate::embed::BatchLimits,
) -> Result<Vec<StoredChunk>> {
    const BACKOFF_MS: [u64; 2] = [100, 400];
    let path = item.path.clone();
    let mut last = embed_work_item(embedder, item.clone(), batch_limits).await;
    for delay in BACKOFF_MS {
        match &last {
            Err(e) if e.is_retryable() => {
                tracing::warn!(path, error = %e, "transient embed failure; retrying");
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                last = embed_work_item(embedder, item.clone(), batch_limits).await;
            }
            _ => return last,
        }
    }
    last
}

/// Embed one file's chunks into rows ready for the index.
async fn embed_work_item(
    embedder: &dyn Embedder,
    item: WorkItem,
    batch_limits: crate::embed::BatchLimits,
) -> Result<Vec<StoredChunk>> {
    let WorkItem {
        path,
        chunks,
        file_hash,
    } = item;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let vectors = embedder.embed_batched(&texts, batch_limits).await?;
    if vectors.len() != texts.len() {
        return Err(Error::Embed(format!(
            "embedder returned {} vectors for {} inputs (file {path})",
            vectors.len(),
            texts.len(),
        )));
    }
    Ok(chunks
        .into_iter()
        .zip(vectors)
        .enumerate()
        .map(|(chunk_index, (c, v))| StoredChunk {
            path: path.clone(),
            start_line: c.start_line,
            end_line: c.end_line,
            chunk_index,
            language: c.language,
            symbol: c.symbol,
            text: c.text,
            file_hash: file_hash.clone(),
            vector: v,
        })
        .collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Lazily initialized engine — constructed on first `get()`, not at server startup,
/// so `tools/list` works even when the embedder endpoint is down and a failed init
/// is retryable. Shares one `RefreshState` with the watcher.
#[derive(Clone)]
pub struct LazyEngine {
    config: Config,
    state: Arc<RefreshState>,
    inner: Arc<OnceCell<Arc<Engine>>>,
}

impl LazyEngine {
    pub fn new(config: Config, state: Arc<RefreshState>) -> Self {
        Self {
            config,
            state,
            inner: Arc::new(OnceCell::new()),
        }
    }

    /// Test/seam constructor: a `LazyEngine` whose cell is already filled, so `get()`
    /// never builds a real embedder.
    pub fn from_engine(config: Config, state: Arc<RefreshState>, engine: Arc<Engine>) -> Self {
        let cell = OnceCell::new();
        let _ = cell.set(engine);
        Self {
            config,
            state,
            inner: Arc::new(cell),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub async fn get(&self) -> std::result::Result<Arc<Engine>, String> {
        self.inner
            .get_or_try_init(|| async {
                let embedder = build_embedder(&self.config.embedder)
                    .await
                    .map_err(|e| e.to_string())?;
                Engine::with_refresh_state(self.config.clone(), embedder, self.state.clone())
                    .await
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .await
            .map(Arc::clone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embed::MockEmbedder;
    use async_trait::async_trait;
    use std::fs;
    use tempfile::tempdir;

    async fn engine_for(root: std::path::PathBuf) -> Engine {
        let cfg = Config::default_for(root);
        Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap()
    }

    /// Build an engine over `root` with a caller-supplied embedder, so a test can
    /// control exactly which inputs the endpoint rejects.
    async fn build_test_engine(root: &std::path::Path, embedder: Arc<dyn Embedder>) -> Engine {
        let cfg = Config::default_for(root.to_path_buf());
        Engine::new_with_embedder(cfg, embedder).await.unwrap()
    }

    /// Fails only for texts containing a marker, so one file in a repo can be made
    /// to fail while its neighbours succeed.
    struct PoisonEmbedder {
        inner: MockEmbedder,
        marker: String,
    }

    #[async_trait]
    impl Embedder for PoisonEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if texts.iter().any(|t| t.contains(&self.marker)) {
                return Err(Error::EmbedContextExceeded {
                    n_prompt_tokens: 99_999,
                    n_ctx: 2048,
                });
            }
            self.inner.embed(texts).await
        }
    }

    #[tokio::test]
    async fn one_failing_file_does_not_block_the_others() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("good_a.rs"), "fn good_a() { let a = 1; }\n").unwrap();
        fs::write(root.join("bad.rs"), "fn bad() { let POISON = 1; }\n").unwrap();
        fs::write(root.join("good_b.rs"), "fn good_b() { let b = 2; }\n").unwrap();

        let embedder = Arc::new(PoisonEmbedder {
            inner: MockEmbedder::new("mock", 8),
            marker: "POISON".to_string(),
        });
        let engine = build_test_engine(root, embedder).await;

        // Must not error: one bad file cannot fail the whole reconcile.
        engine
            .reconcile()
            .await
            .expect("reconcile must survive one bad file");

        let hashes = engine.index.file_hashes().await.unwrap();
        assert!(
            hashes.contains_key("good_a.rs"),
            "good file must be indexed"
        );
        assert!(
            hashes.contains_key("good_b.rs"),
            "good file must be indexed"
        );
        assert!(
            !hashes.contains_key("bad.rs"),
            "failed file must not record a hash, so it is retried once fixed"
        );
    }

    #[tokio::test]
    async fn failed_file_is_retried_on_the_next_reconcile() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::write(root.join("bad.rs"), "fn bad() { let POISON = 1; }\n").unwrap();

        let embedder = Arc::new(PoisonEmbedder {
            inner: MockEmbedder::new("mock", 8),
            marker: "POISON".to_string(),
        });
        let engine = build_test_engine(root, embedder).await;
        engine.reconcile().await.unwrap();
        assert!(
            !engine
                .index
                .file_hashes()
                .await
                .unwrap()
                .contains_key("bad.rs")
        );

        // Fix the file; the next reconcile must pick it up.
        fs::write(root.join("bad.rs"), "fn bad() { let ok = 1; }\n").unwrap();
        engine.reconcile().await.unwrap();
        assert!(
            engine
                .index
                .file_hashes()
                .await
                .unwrap()
                .contains_key("bad.rs"),
            "a repaired file must be indexed on the next pass"
        );
    }

    /// Rejects any single text over `limit` bytes with the server's structured
    /// overflow error, exactly as llama.cpp does — so a too-generous byte budget
    /// fails on the first attempt and must succeed after a re-split. The reported
    /// `n_ctx` is in tokens, so it is `limit / 3` to match the bytes-per-token
    /// relationship a real server exhibits.
    struct ByteLimitEmbedder {
        inner: MockEmbedder,
        limit: usize,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl Embedder for ByteLimitEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_input_tokens(&self) -> Option<usize> {
            // Claim a budget far larger than `limit` allows, so the first pass
            // produces an oversized chunk and the retry path is exercised.
            Some(100_000)
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if let Some(t) = texts.iter().find(|t| t.len() > self.limit) {
                return Err(Error::EmbedContextExceeded {
                    n_prompt_tokens: t.len() / 3,
                    n_ctx: self.limit / 3,
                });
            }
            self.inner.embed(texts).await
        }
    }

    #[tokio::test]
    async fn context_overflow_triggers_one_resplit_retry() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let body = "    let x = 1;\n".repeat(500);
        fs::write(root.join("huge.rs"), format!("fn huge() {{\n{body}}}\n")).unwrap();

        // First pass: the claimed 100_000-token budget yields a 400_000-byte chunk
        // budget, so the file's single ~7000-byte chunk is sent whole and rejected.
        // Each rejection folds the server's own n_ctx/n_prompt_tokens back into the
        // shared budget, which strictly decreases until a split lands under `limit`.
        let embedder = Arc::new(ByteLimitEmbedder {
            inner: MockEmbedder::new("mock", 8),
            limit: 2000,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let engine = build_test_engine(root, embedder.clone()).await;
        engine.reconcile().await.unwrap();

        assert!(
            engine
                .index
                .file_hashes()
                .await
                .unwrap()
                .contains_key("huge.rs"),
            "the file must be indexed after re-splitting against the server's n_ctx"
        );
        assert!(
            embedder.calls.load(std::sync::atomic::Ordering::SeqCst) > 1,
            "the retry path must have run"
        );
    }

    #[tokio::test]
    async fn focus_read_recovers_from_an_overflow_and_teaches_the_budget() {
        // A focus read embeds every chunk to rank it, so it hits the same per-input
        // limit as indexing. Before the budget was fed from this path, a rejection
        // on a cold index failed the tool call AND taught the budget nothing, so
        // the identical next call failed identically. It must now re-split, return
        // results, and leave the correction in force for everyone else.
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("big.rs"),
            format!("fn big() {{\n{}}}\n", "    let x = 1;\n".repeat(400)),
        )
        .unwrap();
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(2048)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;

        let hits = engine.read_file("big.rs", Some("let x")).await.unwrap();

        assert!(!hits.is_empty(), "focus read must survive an overflow");
        assert!(
            engine.chunk_budget().was_tightened(),
            "the overflow must correct the shared budget, not be discarded"
        );
        assert!(engine.chunk_budget().bytes() <= 2048);
        assert!(
            hits.iter().any(|h| h.why_matched.contains("partial")),
            "a piece of a split definition must say so"
        );
    }

    #[tokio::test]
    async fn a_file_with_no_chunks_settles_instead_of_churning_every_pass() {
        // An empty (or whitespace-only) file yields zero chunks, which reaches
        // `upsert_file` with an empty vec — that path deletes the file's rows and
        // never records a hash, so `diff` offers the path again on every single
        // reconcile and repeats the deletes forever.
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("empty.rs"), "").unwrap();
        fs::write(repo.path().join("a.rs"), "fn a() {}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;

        engine.refresh().await.unwrap();
        let second = engine
            .reconcile_guarded(false)
            .await
            .expect("second reconcile must succeed");

        assert_eq!(
            second.indexed, 0,
            "nothing changed on disk, so no file should be re-indexed"
        );
        assert!(second.is_clean(), "a settled pass must be clean");
    }

    #[tokio::test]
    async fn an_oversized_focus_query_does_not_poison_the_chunk_budget() {
        // The focus string is caller-supplied and is never split, so its token
        // count says nothing about how large a *chunk* may be. Folding it into the
        // shared budget is a category error with lasting consequences: `tighten`
        // is monotonic, so one oversized focus argument would ratchet the budget
        // to its floor for every file indexed afterwards, and those undersized
        // chunks are persisted without a CHUNKER_VERSION bump to undo them.
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "fn a() {}\n").unwrap();
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(2048)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;
        let before = engine.chunk_budget().bytes();

        let huge_focus = "x".repeat(10_000);
        let err = engine
            .read_file("a.rs", Some(&huge_focus))
            .await
            .expect_err("a focus query larger than the window must fail the call");

        assert_eq!(
            engine.chunk_budget().bytes(),
            before,
            "the focus query's size must not tighten the chunk budget"
        );
        assert!(
            !engine.chunk_budget().was_tightened(),
            "no chunk overflowed, so nothing should be recorded as a correction"
        );
        assert!(
            err.to_string().contains("focus"),
            "the error must name the focus query as the oversized input, got: {err}"
        );
    }

    #[tokio::test]
    async fn unreadable_files_are_skipped_without_failing_the_reconcile() {
        // A non-UTF-8 file is not something we index, but it is not a failure
        // either: it must not land in `failed` and so must not make `refresh()`
        // exit non-zero for `reindex`/`status`.
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("ok.rs"), "fn ok() {}\n").unwrap();
        fs::write(repo.path().join("bad.rs"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;

        engine.refresh().await.unwrap();

        assert!(engine.last_failures().is_empty(), "skipped != failed");
        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(files, 1, "only the readable file is indexed");
    }

    #[tokio::test]
    async fn overflow_tightens_the_budget_and_indexes_the_file() {
        // The server's structured 400 is authoritative. Re-split against it and the
        // file must land in the index — overflow is a budget signal, not a failure.
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("big.rs"),
            format!("fn big() {{\n{}}}\n", "    let x = 1;\n".repeat(400)),
        )
        .unwrap();
        // Reports a 2048-token window but really packs 1 byte/token, so the
        // optimistic 4 bytes/token start is guaranteed to overflow first.
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(2048)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;

        engine.refresh().await.unwrap();

        let (files, chunks) = engine.stats().await.unwrap();
        assert_eq!(files, 1, "the file must be indexed, not skipped");
        assert!(chunks > 1, "it must have been split");
        assert!(engine.chunk_budget().was_tightened());
        assert!(engine.chunk_budget().bytes() <= 2048);
    }

    #[tokio::test]
    async fn a_legitimate_file_indexes_without_exhausting_the_retry_rounds() {
        // A chunk modestly over the window but far *under* the budget. Correcting
        // from the budget rather than the chunk's real size makes the measured
        // ratio wildly optimistic, so each round shrinks by only the 15/16
        // headroom: convergence needs ~14 rounds against a cap of 8, and `refresh`
        // reports failure on a file that splits perfectly well. The chunk is 2118
        // bytes against a 2048-token window at 1 byte/token, with the budget
        // starting at 2048 * 4 = 8192.
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("stubborn.rs"),
            format!("fn stubborn() {{\n{}}}\n", "    let x = 1;\n".repeat(140)),
        )
        .unwrap();
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(2048)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;

        engine
            .refresh()
            .await
            .expect("a file this size must index, not exhaust the retry rounds");

        let (files, chunks) = engine.stats().await.unwrap();
        assert_eq!(files, 1, "the file must be indexed");
        assert!(chunks > 1, "it must have been split to fit");
    }

    #[tokio::test]
    async fn refresh_reports_failure_but_reconcile_tolerates_it() {
        // reindex/status must not print success over a failed rebuild, while
        // `search` must keep working against a partial index.
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "fn a() {}\n").unwrap();
        // Fails far more times than the retry ladder will absorb.
        let embedder = std::sync::Arc::new(MockEmbedder::new("m", 8).failing_times(999));
        let engine = build_test_engine(repo.path(), embedder).await;

        let err = engine.refresh().await.unwrap_err();
        assert!(
            err.to_string().contains("a.rs"),
            "the failure must name the file, got: {err}"
        );
        assert_eq!(
            engine.last_failures().len(),
            1,
            "the failure must be recorded for diagnostics"
        );

        // The search path stays tolerant, and leaves dirty set so it re-scans.
        engine
            .reconcile()
            .await
            .expect("reconcile must not propagate");
        assert!(
            !engine.refresh_state().can_skip_scan(),
            "a failed reconcile must leave the index marked dirty"
        );
    }

    #[tokio::test]
    async fn a_clean_reconcile_still_succeeds() {
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "fn a() {}\n").unwrap();
        let engine =
            build_test_engine(repo.path(), std::sync::Arc::new(MockEmbedder::new("m", 8))).await;
        engine.refresh().await.unwrap();
        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(files, 1);
        assert!(engine.last_failures().is_empty());
    }

    #[tokio::test]
    async fn transient_failures_are_retried_not_skipped() {
        // A blip must not cost the file its place in the index.
        let repo = tempfile::tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "fn a() {}\n").unwrap();
        // Fails the first two attempts, then succeeds.
        let embedder = std::sync::Arc::new(MockEmbedder::new("m", 8).failing_times(2));
        let engine = build_test_engine(repo.path(), embedder).await;

        engine.refresh().await.unwrap();

        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(files, 1, "a transient failure must be retried, not skipped");
    }

    #[tokio::test]
    async fn auth_failure_stops_the_whole_run() {
        // An unusable API key dooms every file: stop rather than repeating the
        // same rejection once per file, and leave the state dirty.
        struct AuthEmbedder;
        #[async_trait]
        impl Embedder for AuthEmbedder {
            fn id(&self) -> &'static str {
                "auth"
            }
            fn dim(&self) -> usize {
                8
            }
            async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
                Err(Error::EmbedAuth("401".into()))
            }
        }
        let repo = tempdir().unwrap();
        for name in ["a.rs", "b.rs"] {
            fs::write(repo.path().join(name), "fn f() {}\n").unwrap();
        }
        let engine = build_test_engine(repo.path(), Arc::new(AuthEmbedder)).await;

        let err = engine.refresh().await.unwrap_err();
        assert!(matches!(err, Error::EmbedAuth(_)), "got {err:?}");
        assert!(
            engine.refresh_state().is_dirty(),
            "an aborted reconcile must leave the index dirty"
        );
        // The abort must reach `diagnostics` too. Recording the summary only on
        // the success path leaves it reporting "no failures" straight after a run
        // in which every single file was doomed.
        let failures = engine.last_failures();
        assert!(
            !failures.is_empty(),
            "an aborted run must be visible to diagnostics, not reported as clean"
        );
        assert!(
            failures.iter().any(|(_, e)| e.contains("401")),
            "the recorded failure must carry the reason, got {failures:?}"
        );
    }

    #[tokio::test]
    async fn a_down_endpoint_gives_up_instead_of_grinding_through_every_file() {
        // Per-file isolation is right for a bad file, but a refused connection is
        // ALSO classified retryable — correctly, since one request cannot tell a
        // blip from an outage. Without a circuit breaker each of N changed files
        // rediscovers the same dead endpoint through the full retry ladder (3
        // attempts + 500 ms of backoff), and because `ensure_fresh` deliberately
        // sits outside `search_timeout_secs`, a branch switch that touches
        // thousands of files turns a single `search` into a multi-minute stall.
        const FILES: usize = 40;

        let repo = tempdir().unwrap();
        for i in 0..FILES {
            fs::write(repo.path().join(format!("f{i}.rs")), "fn f() {}\n").unwrap();
        }
        let embedder = Arc::new(MockEmbedder::new("m", 8).always_transiently_failing());
        let calls = embedder.call_counter();
        let engine = build_test_engine(repo.path(), embedder).await;

        let err = engine.refresh().await.unwrap_err();
        assert!(
            err.to_string().contains("consecutive"),
            "the abort must name the endpoint as the cause, not a file; got {err}"
        );

        // The point of the breaker: it stops early. Each file burns the whole
        // ladder, so the bound is the breaker limit's worth of files, not all 40.
        let attempted = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            attempted < FILES,
            "must abandon the run rather than attempt all {FILES} files, attempted {attempted} embeds"
        );

        // Failing fast must not be mistaken for succeeding: nothing indexed, the
        // state stays dirty, and diagnostics can see why.
        assert!(
            engine.refresh_state().is_dirty(),
            "an abandoned reconcile must leave the index dirty"
        );
        assert!(
            !engine.last_failures().is_empty(),
            "the abort must be visible to diagnostics"
        );
    }

    #[tokio::test]
    async fn transient_failures_below_the_limit_stay_isolated_to_their_files() {
        // The counterpart: the breaker must not turn per-file isolation into a
        // hair trigger. `failing_times` is a run-wide budget of blips, and the
        // per-file ladder is 3 attempts, so 9 blips exhaust exactly 3 files —
        // under CONSECUTIVE_TRANSIENT_LIMIT. Those 3 must be reported as skipped
        // files while every other file still commits, exactly as before the
        // breaker existed.
        const FILES: usize = 10;
        const DOOMED: usize = 3;
        // Compile-time: this test only means anything below the breaker's
        // threshold, so raising the limit must not silently invalidate it.
        const _: () = assert!(DOOMED < CONSECUTIVE_TRANSIENT_LIMIT);

        let repo = tempdir().unwrap();
        for i in 0..FILES {
            fs::write(repo.path().join(format!("f{i}.rs")), "fn f() {}\n").unwrap();
        }
        let embedder = Arc::new(MockEmbedder::new("m", 8).failing_times(DOOMED * 3));
        let engine = build_test_engine(repo.path(), embedder).await;

        // `refresh` reports any per-file failure, so an error is expected here —
        // what matters is WHICH error: the per-file list, not the breaker.
        let err = engine.refresh().await.unwrap_err();
        assert!(
            !err.to_string().contains("consecutive"),
            "failures below the limit must stay per-file, not abort the run; got {err}"
        );
        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(
            files,
            FILES - DOOMED,
            "every file the endpoint did answer for must still be indexed"
        );
    }

    #[tokio::test]
    async fn the_correction_applies_to_every_later_file() {
        // The whole point of a shared budget: file B must not repeat file A's
        // rejected round-trip. After a reconcile that tightened, the budget is
        // already correct, so a second reconcile embeds with no overflow at all.
        let repo = tempdir().unwrap();
        for name in ["a.rs", "b.rs", "c.rs"] {
            fs::write(
                repo.path().join(name),
                format!("fn f() {{\n{}}}\n", "    let x = 1;\n".repeat(400)),
            )
            .unwrap();
        }
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(2048)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;

        engine.refresh().await.unwrap();
        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(files, 3, "every file must be indexed");

        let settled = engine.chunk_budget().bytes();
        assert!(settled <= 2048);
        // A second pass over unchanged files does no work, but the budget must not
        // have drifted back up — tightening is one-way.
        engine.refresh().await.unwrap();
        assert_eq!(engine.chunk_budget().bytes(), settled);
    }

    /// Number of files raced against each other in the concurrent-overflow test.
    /// It is also the barrier width and the reported slot count, so every file's
    /// first (oversized) request is in flight at the same instant.
    const RACERS: usize = 6;

    /// Rejects anything over `limit` bytes, and holds every oversized request at a
    /// barrier so all files in flight overflow simultaneously — the interleaving in
    /// which one file's correction lands while another is deciding what its own
    /// rejection means.
    struct RacingOverflowEmbedder {
        inner: MockEmbedder,
        limit: usize,
        gate: tokio::sync::Barrier,
    }

    #[async_trait]
    impl Embedder for RacingOverflowEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_input_tokens(&self) -> Option<usize> {
            // Far more generous than `limit` allows, so the first pass overflows.
            Some(4096)
        }
        fn max_concurrent_requests(&self) -> Option<usize> {
            Some(RACERS)
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            if let Some(t) = texts.iter().find(|t| t.len() > self.limit) {
                // Timeout so a mis-sized barrier fails the test instead of hanging it.
                let _ =
                    tokio::time::timeout(std::time::Duration::from_secs(5), self.gate.wait()).await;
                return Err(Error::EmbedContextExceeded {
                    // 1 byte/token, so the reported window is `limit` tokens.
                    n_prompt_tokens: t.len(),
                    n_ctx: self.limit,
                });
            }
            self.inner.embed(texts).await
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_overflows_do_not_report_a_false_floor() {
        // Every file overflows at the same moment, so each one's `tighten` races
        // the others'. A loser sees `Unchanged` — its measured target is no smaller
        // than the correction that already landed — which is NOT the 1 byte/token
        // floor and must not be reported as one. Every file must still be indexed.
        let repo = tempdir().unwrap();
        for i in 0..RACERS {
            fs::write(
                repo.path().join(format!("f{i}.rs")),
                format!("fn f{i}() {{\n{}}}\n", "    let x = 1;\n".repeat(1000)),
            )
            .unwrap();
        }
        let embedder = Arc::new(RacingOverflowEmbedder {
            inner: MockEmbedder::new("m", 8),
            limit: 4000,
            gate: tokio::sync::Barrier::new(RACERS),
        });
        let engine = build_test_engine(repo.path(), embedder).await;

        // refresh() surfaces per-file failures, so a false floor error fails here.
        engine
            .refresh()
            .await
            .expect("a concurrent correction is not a floor condition");
        assert!(
            engine.last_failures().is_empty(),
            "no file may be failed by a benign race: {:?}",
            engine.last_failures()
        );
        let (files, _) = engine.stats().await.unwrap();
        assert_eq!(files, RACERS, "every raced file must be indexed");
        assert!(engine.chunk_budget().bytes() <= 4000);
    }

    /// Rejects every input whatever its size, always naming the same window. No
    /// budget makes this file fit, so the loop must stop at the floor.
    struct AlwaysOverflowEmbedder {
        inner: MockEmbedder,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl Embedder for AlwaysOverflowEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        fn max_input_tokens(&self) -> Option<usize> {
            Some(2048)
        }
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Err(Error::EmbedContextExceeded {
                n_prompt_tokens: 99_999,
                n_ctx: 2048,
            })
        }
    }

    #[tokio::test]
    async fn a_file_that_cannot_fit_at_the_floor_surfaces_an_error() {
        // The other side of the floor test: when the budget really has bottomed out
        // at `n_ctx` bytes and the endpoint still says no, the file must fail loudly
        // — not spin to MAX_OVERFLOW_ROUNDS and not vanish silently.
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("stubborn.rs"),
            format!("fn stubborn() {{\n{}}}\n", "    let x = 1;\n".repeat(400)),
        )
        .unwrap();
        let embedder = Arc::new(AlwaysOverflowEmbedder {
            inner: MockEmbedder::new("m", 8),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let engine = build_test_engine(repo.path(), embedder.clone()).await;

        let err = engine.refresh().await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("stubborn.rs"),
            "must name the file, got: {msg}"
        );
        assert!(
            msg.contains("minimum chunk size"),
            "must diagnose the floor, got: {msg}"
        );
        assert_eq!(
            engine.chunk_budget().bytes(),
            2048 - crate::budget::SPECIAL_TOKEN_SLACK,
            "the budget must have bottomed out at the floor"
        );
        assert!(
            embedder.calls.load(std::sync::atomic::Ordering::SeqCst) <= 3,
            "it must stop at the floor, not run the full round bound"
        );
        // Always-fresh: a failed file keeps no hash and leaves the state dirty.
        assert!(
            !engine
                .index
                .file_hashes()
                .await
                .unwrap()
                .contains_key("stubborn.rs")
        );
        assert!(engine.refresh_state().is_dirty());
    }

    #[tokio::test]
    async fn focus_read_of_a_minified_file_fits_the_context_window() {
        // read_file is deliberately not exclude-filtered, so a vendored/minified
        // file is exactly what a focus read gets pointed at. The focus branch
        // embeds to rank, and there is no overflow retry on that path — so its
        // inputs must be split to the same per-chunk budget the indexer uses.
        let repo = tempdir().unwrap();
        fs::create_dir_all(repo.path().join("dist")).unwrap();
        // One declaration on one line: no newline to split on, so only a
        // char-boundary split can bring it under the budget.
        let minified = format!(
            "var AUTH_BUNDLE=\"{}auth{}\";\n",
            "x".repeat(20_000),
            "y".repeat(20_000)
        );
        fs::write(repo.path().join("dist/bundle.min.js"), &minified).unwrap();

        // Budget: 400 reported tokens x 4 bytes = 1600 bytes per input. The endpoint
        // enforces 2048 bytes, so the unsplit 40 KB chunk is rejected outright and
        // only a split read can succeed.
        let embedder = Arc::new(
            MockEmbedder::new("m", 8)
                .with_max_input_tokens(400)
                .with_context_limit(2048, 1),
        );
        let engine = build_test_engine(repo.path(), embedder).await;

        let focus = engine
            .read_file("dist/bundle.min.js", Some("auth"))
            .await
            .expect("a focus read must not send an oversized input");
        assert!(!focus.is_empty());
        assert!(
            focus.iter().all(|e| e.code.len() <= 1600),
            "every ranked piece must be within the budget"
        );
        assert!(
            focus.iter().any(|e| e.why_matched.contains("partial")),
            "a piece of a split definition must say so: {:?}",
            focus.iter().map(|e| &e.why_matched).collect::<Vec<_>>()
        );

        // The outline branch stays structural: one whole definition, unsplit.
        let outline = engine.read_file("dist/bundle.min.js", None).await.unwrap();
        assert_eq!(
            outline.len(),
            1,
            "one entry per definition, got {outline:#?}"
        );
        assert!(outline.iter().all(|e| e.why_matched == "outline"));
        assert!(
            outline[0].code.len() > 2048,
            "the outline must not be split into fragments"
        );
    }

    /// Misbehaving embedder that returns no vectors — exercises the `embed_one` guard.
    struct ZeroEmbedder;
    #[async_trait]
    impl Embedder for ZeroEmbedder {
        fn id(&self) -> &'static str {
            "zero"
        }
        fn dim(&self) -> usize {
            64
        }
        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn search_errors_when_embedder_returns_no_vectors() {
        // Empty repo: refresh embeds nothing, so this isolates embed_one(query).
        let repo = tempdir().unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Engine::new_with_embedder(cfg, Arc::new(ZeroEmbedder))
            .await
            .unwrap();
        let err = engine.search("anything", Some(3)).await.unwrap_err();
        assert!(
            matches!(err, Error::Embed(_)),
            "expected Error::Embed, got {err:?}"
        );
    }

    #[tokio::test]
    async fn indexes_and_finds_relevant_file() {
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("auth.rs"),
            "pub fn renew_credentials() -> Token {\n    refresh_token()\n}\n",
        )
        .unwrap();
        fs::write(
            repo.path().join("math.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
        )
        .unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        let entries = engine.search("renew_credentials", Some(3)).await.unwrap();
        assert!(entries.iter().any(|e| e.path == "auth.rs"));
    }

    #[tokio::test]
    async fn test_files_are_not_indexed_by_default() {
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("auth.rs"),
            "pub fn renew_credentials() -> Token {\n    refresh_token()\n}\n",
        )
        .unwrap();
        fs::create_dir(repo.path().join("tests")).unwrap();
        // Same content under tests/: must be excluded, so it never competes in results.
        fs::write(
            repo.path().join("tests").join("auth_test.rs"),
            "pub fn renew_credentials() -> Token {\n    refresh_token()\n}\n",
        )
        .unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        let entries = engine.search("renew_credentials", Some(5)).await.unwrap();
        assert!(entries.iter().any(|e| e.path == "auth.rs"));
        assert!(
            !entries.iter().any(|e| e.path.starts_with("tests/")),
            "tests/ files must be excluded from the index; got {:?}",
            entries.iter().map(|e| &e.path).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn refresh_picks_up_new_and_deleted_files() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "fn a(){}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        engine.refresh().await.unwrap();
        fs::write(repo.path().join("b.rs"), "fn b(){}\n").unwrap();
        fs::remove_file(repo.path().join("a.rs")).unwrap();
        let entries = engine.search("b", Some(5)).await.unwrap();
        assert!(entries.iter().any(|e| e.path == "b.rs"));
        assert!(!entries.iter().any(|e| e.path == "a.rs"));
    }

    #[tokio::test]
    async fn newly_excluded_file_is_purged_on_reconcile() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("keep.rs"), "pub fn keep() {}\n").unwrap();
        fs::write(repo.path().join("data.txt"), "noise noise noise\n").unwrap();

        // First engine: no excludes, no language filter -> indexes both files.
        {
            let mut cfg = Config::default_for(repo.path().to_path_buf());
            cfg.languages = vec![]; // allow all languages
            cfg.index_tests = true;
            let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
                .await
                .unwrap();
            engine.refresh().await.unwrap();
            let stored = engine.index.file_hashes().await.unwrap();
            assert!(
                stored.contains_key("data.txt"),
                "precondition: data.txt must be indexed first"
            );
        }

        // Second engine on the SAME index dir, now excluding data.txt. A reconcile
        // must purge the previously-indexed file (this is what should happen when a
        // new built-in/config exclusion ships and the daemon reconciles).
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.languages = vec![]; // allow all languages
        cfg.index_tests = true;
        cfg.exclude = vec!["**/data.txt".to_string()];
        let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap();
        engine.refresh().await.unwrap();
        let stored = engine.index.file_hashes().await.unwrap();
        assert!(
            !stored.contains_key("data.txt"),
            "newly-excluded file must be purged on reconcile; got {:?}",
            stored.keys().collect::<Vec<_>>()
        );
        assert!(stored.contains_key("keep.rs"), "kept file stays indexed");
        // And it must no longer surface in search results (the user-visible symptom).
        let hits = engine.search("noise", Some(10)).await.unwrap();
        assert!(
            !hits.iter().any(|e| e.path == "data.txt"),
            "excluded file must not be returned by search; got {:?}",
            hits.iter().map(|e| &e.path).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn search_filters_excluded_files_when_index_is_stale() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("keep.rs"), "pub fn keep() {}\n").unwrap();
        fs::write(repo.path().join("data.txt"), "noise noise noise\n").unwrap();

        // Index both with no excludes — simulates an index built before the exclusion.
        {
            let mut cfg = Config::default_for(repo.path().to_path_buf());
            cfg.languages = vec![]; // allow all languages
            cfg.index_tests = true;
            let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
                .await
                .unwrap();
            engine.refresh().await.unwrap();
        }

        // New engine excludes data.txt, but force the "watcher caught up" state so
        // ensure_fresh SKIPS reconcile — the stale row stays in the index. This is
        // the lag window the read-time filter exists to cover.
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.languages = vec![]; // allow all languages
        cfg.index_tests = true;
        cfg.exclude = vec!["**/data.txt".to_string()];
        let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap();
        engine.refresh_state().set_watch_active(true);
        engine.refresh_state().clear_dirty();
        assert!(
            engine.refresh.can_skip_scan(),
            "precondition: reconcile skipped"
        );
        assert!(
            engine
                .index
                .file_hashes()
                .await
                .unwrap()
                .contains_key("data.txt"),
            "precondition: stale row still in the index (reconcile was skipped)"
        );

        // Search must not surface the excluded file even though it's still indexed.
        let hits = engine.search("noise", Some(10)).await.unwrap();
        assert!(
            !hits.iter().any(|e| e.path == "data.txt"),
            "read-time filter must drop excluded files from a stale index; got {:?}",
            hits.iter().map(|e| &e.path).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn read_file_outline_and_focus() {
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("lib.rs"),
            "pub fn alpha() {}\npub fn beta() {}\n",
        )
        .unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;

        // Outline (no focus): one entry per chunk, why_matched == "outline".
        let outline = engine.read_file("lib.rs", None).await.unwrap();
        assert!(!outline.is_empty());
        assert!(outline.iter().all(|e| e.why_matched == "outline"));
        assert!(outline.iter().any(|e| e.symbol.as_deref() == Some("alpha")));

        // Focus: ranked by similarity, why_matched mentions the focus.
        let focus = engine.read_file("lib.rs", Some("alpha")).await.unwrap();
        assert!(!focus.is_empty());
        assert!(
            focus
                .iter()
                .all(|e| e.why_matched.contains("focus similarity"))
        );
    }

    #[tokio::test]
    async fn outline_gives_one_entry_per_definition() {
        // read_file's outline is a structural summary of signatures. Splitting it
        // to an embedding budget turned one big function into several entries whose
        // "signature" was a body fragment — breaking the contract for exactly the
        // large files an outline exists to summarize.
        let repo = tempdir().unwrap();
        let src = format!(
            "fn small() {{}}\n\nfn big() {{\n{}}}\n",
            "    let x = 1;\n".repeat(500)
        );
        fs::write(repo.path().join("m.rs"), &src).unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;

        let outline = engine.read_file("m.rs", None).await.unwrap();
        assert_eq!(
            outline.len(),
            2,
            "one entry per definition, got {outline:#?}"
        );
        assert!(outline.iter().any(|e| e.code.contains("fn small")));
        assert!(outline.iter().any(|e| e.code.contains("fn big")));
        assert!(
            !outline.iter().any(|e| e.code.trim() == "let x = 1;"),
            "no entry may be a body fragment"
        );
    }

    #[tokio::test]
    async fn search_skips_scan_when_watch_active_and_clean() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        // Prime the index, then simulate a healthy, caught-up watcher.
        engine.refresh().await.unwrap();
        engine.refresh_state().set_watch_active(true);
        engine.refresh_state().clear_dirty();

        // Change the tree WITHOUT marking dirty: a skipped scan must not see it.
        fs::write(repo.path().join("b.rs"), "pub fn beta() {}\n").unwrap();
        let entries = engine.search("beta", Some(5)).await.unwrap();
        assert!(
            !entries.iter().any(|e| e.path == "b.rs"),
            "active+clean must skip the scan; got {:?}",
            entries.iter().map(|e| &e.path).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn failed_reconcile_keeps_state_dirty_for_retry() {
        // A watcher marks dirty then triggers a reconcile that fails to embed.
        // The failure must NOT leave the state clean+active, or the next search
        // takes the skip-scan path and serves stale results forever.
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Engine::new_with_embedder(cfg, Arc::new(ZeroEmbedder))
            .await
            .unwrap();
        engine.refresh_state().set_watch_active(true);
        engine.refresh_state().mark_dirty(); // watcher would do this before reconciling

        // A per-file embed failure no longer aborts the reconcile — the file is
        // skipped so the rest of the repo commits — but a reconcile that skipped
        // anything must still leave the state dirty.
        engine.reconcile().await.unwrap();

        assert!(
            engine.refresh_state().is_dirty(),
            "a failed reconcile must re-mark dirty"
        );
        assert!(
            !engine.refresh_state().can_skip_scan(),
            "a failed reconcile must keep search on the scanning path"
        );
    }

    #[tokio::test]
    async fn search_reconciles_when_dirty() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let engine = engine_for(repo.path().to_path_buf()).await;
        engine.refresh().await.unwrap();
        engine.refresh_state().set_watch_active(true);
        engine.refresh_state().clear_dirty();

        fs::write(repo.path().join("b.rs"), "pub fn beta() {}\n").unwrap();
        engine.refresh_state().mark_dirty(); // watcher would do this
        let entries = engine.search("beta", Some(5)).await.unwrap();
        assert!(
            entries.iter().any(|e| e.path == "b.rs"),
            "dirty => reconcile picks up b.rs"
        );
    }

    /// Verify that parallel reconcile indexes all changed files concurrently.
    #[tokio::test]
    async fn reconcile_embeds_multiple_files_in_parallel() {
        let repo = tempdir().unwrap();
        // Create several files that will all need embedding on first reconcile.
        for i in 0..12 {
            fs::write(
                repo.path().join(format!("mod_{i}.rs")),
                format!("pub fn func_{i}() {{ /* body {i} */ }}\n"),
            )
            .unwrap();
        }
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        // Unset embed_concurrency derives to 1 (serial) against a MockEmbedder, which
        // reports no total_slots; force an explicit override so this test actually
        // drives the concurrent embed path (buffer_unordered with overlapping embeds
        // feeding a single serial index writer) rather than silently running serially.
        cfg.embedder.embed_concurrency = Some(4);
        let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap();
        engine.refresh().await.unwrap();
        let hashes = engine.index.file_hashes().await.unwrap();
        for i in 0..12 {
            assert!(
                hashes.contains_key(&format!("mod_{i}.rs")),
                "mod_{i}.rs should be indexed"
            );
        }
        let count = engine.index.chunk_count().await.unwrap();
        assert_eq!(count, 12, "exactly 12 chunks from 12 files");

        // A second reconcile with no changes must not duplicate anything (idempotency).
        engine.refresh().await.unwrap();
        assert_eq!(
            engine.index.chunk_count().await.unwrap(),
            12,
            "re-reconcile must not duplicate chunks"
        );
    }

    #[tokio::test]
    async fn lazy_engine_from_engine_returns_ready_instance() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let state = Arc::new(RefreshState::standalone());
        let engine = Arc::new(
            Engine::with_refresh_state(
                cfg.clone(),
                Arc::new(MockEmbedder::new("mock-v1", 64)),
                state.clone(),
            )
            .await
            .unwrap(),
        );
        let lazy = LazyEngine::from_engine(cfg, state.clone(), engine.clone());
        let got = lazy.get().await.unwrap();
        assert!(
            Arc::ptr_eq(&got, &engine),
            "from_engine yields the pre-built engine"
        );
    }

    /// Records every `embed()` batch length, delegating vectors to a `MockEmbedder`, so we
    /// can assert the engine honored the batch cap on a real reconcile.
    struct CountingEmbedder {
        inner: MockEmbedder,
        calls: std::sync::Mutex<Vec<usize>>,
    }
    impl CountingEmbedder {
        fn new(dim: usize) -> Self {
            Self {
                inner: MockEmbedder::new("mock-v1", dim),
                calls: std::sync::Mutex::new(vec![]),
            }
        }
    }
    #[async_trait]
    impl Embedder for CountingEmbedder {
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

    #[tokio::test]
    async fn reconcile_respects_batch_cap() {
        // Engine takes Box<dyn Embedder>; wrap the Arc so we keep a handle to the spy.
        struct Shared(std::sync::Arc<CountingEmbedder>);
        #[async_trait]
        impl Embedder for Shared {
            fn id(&self) -> &str {
                self.0.id()
            }
            fn dim(&self) -> usize {
                self.0.dim()
            }
            async fn embed(&self, t: &[String]) -> Result<Vec<Vec<f32>>> {
                self.0.embed(t).await
            }
        }

        let repo = tempdir().unwrap();
        // One file with 5 top-level fns -> 5 chunks.
        fs::write(
            repo.path().join("many.rs"),
            "pub fn a() {}\npub fn b() {}\npub fn c() {}\npub fn d() {}\npub fn e() {}\n",
        )
        .unwrap();

        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.embedder.max_batch_chunks = 2; // force splitting
        cfg.embedder.max_batch_bytes = 1_000_000;

        let embedder = std::sync::Arc::new(CountingEmbedder::new(64));
        let engine = Engine::new_with_embedder(cfg, Arc::new(Shared(embedder.clone())))
            .await
            .unwrap();
        engine.refresh().await.unwrap();

        let calls = embedder.calls.lock().unwrap().clone();
        // 5 chunks under max_batch_chunks=2 must split deterministically into 2+2+1,
        // never one batch of 5 and never a batch over the cap.
        assert_eq!(
            calls,
            vec![2, 2, 1],
            "5 chunks, cap 2 -> 2+2+1; got {calls:?}"
        );
    }

    #[tokio::test]
    async fn index_persists_across_engine_reopen() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let (files, chunks) = {
            let engine = engine_for(repo.path().to_path_buf()).await;
            engine.refresh().await.unwrap();
            engine.stats().await.unwrap()
        };
        assert!(files >= 1 && chunks >= 1);

        // Reopen a fresh Engine over the same repo dir (same embedder id) — the
        // LanceDB index must persist, and re-refreshing unchanged files must not
        // duplicate rows.
        let engine2 = engine_for(repo.path().to_path_buf()).await;
        assert_eq!(
            engine2.stats().await.unwrap(),
            (files, chunks),
            "index persisted"
        );
        engine2.refresh().await.unwrap();
        assert_eq!(
            engine2.stats().await.unwrap(),
            (files, chunks),
            "unchanged files not re-indexed"
        );
    }

    #[tokio::test]
    async fn language_whitelist_gates_indexing() {
        let repo = tempdir().unwrap();
        fs::write(
            repo.path().join("lib.rs"),
            "pub fn rust_fn() -> i32 { 42 }\n",
        )
        .unwrap();
        fs::write(repo.path().join("lib.py"), "def py_fn(): return 42\n").unwrap();

        // Engine with languages = ["rust"] should only index .rs files.
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.languages = vec!["rust".to_string()];
        cfg.index_tests = true; // don't let test excludes interfere
        let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap();
        engine.refresh().await.unwrap();

        let stored = engine.index.file_hashes().await.unwrap();
        assert!(stored.contains_key("lib.rs"), "rust file should be indexed");
        assert!(
            !stored.contains_key("lib.py"),
            "python file should be skipped by language whitelist; got {:?}",
            stored.keys().collect::<Vec<_>>()
        );

        // Search for python content should not surface the .py file.
        let hits = engine.search("py_fn", Some(10)).await.unwrap();
        assert!(
            !hits.iter().any(|e| e.path == "lib.py"),
            "python file must not appear in search results"
        );

        // Rust content should still be found.
        let hits = engine.search("rust_fn", Some(10)).await.unwrap();
        assert!(
            hits.iter().any(|e| e.path == "lib.rs"),
            "rust file should appear in search results"
        );
    }

    /// Slow embedder that sleeps on every call — used to verify search timeout fires.
    struct SlowEmbedder {
        inner: MockEmbedder,
    }
    #[async_trait]
    impl Embedder for SlowEmbedder {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn dim(&self) -> usize {
            self.inner.dim()
        }
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            self.inner.embed(texts).await
        }
    }

    #[tokio::test]
    async fn search_times_out_when_embedder_is_slow() {
        // Empty repo on purpose: ensure_fresh's reconcile finds no files to embed and
        // returns fast (it is deliberately NOT under the query timeout), so the only slow
        // embed is embed_one(query) — which is what the query timeout must bound. With an
        // indexed file here, the reconcile itself would sleep 10s before the timeout path
        // was ever reached, and the test would no longer isolate the query timeout.
        let repo = tempdir().unwrap();

        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.search.search_timeout_secs = 1; // very short timeout
        let engine = Engine::new_with_embedder(
            cfg,
            Arc::new(SlowEmbedder {
                inner: MockEmbedder::new("mock-v1", 64),
            }),
        )
        .await
        .unwrap();
        // ensure_fresh returns Ok fast (nothing to embed); embed_one(query) then sleeps
        // 10s but the overall query timeout is 1s.
        let err = engine.search("anything", Some(3)).await.unwrap_err();
        assert!(
            matches!(err, Error::Timeout(_)),
            "expected Error::Timeout, got {err:?}"
        );
    }

    #[tokio::test]
    async fn cancelled_reconcile_leaves_dirty_set() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Engine::new_with_embedder(
            cfg,
            Arc::new(SlowEmbedder {
                inner: MockEmbedder::new("mock-v1", 64),
            }),
        )
        .await
        .unwrap();
        // Drop the reconcile future while it is mid-embed (SlowEmbedder sleeps 10s).
        let _ =
            tokio::time::timeout(std::time::Duration::from_millis(50), engine.reconcile()).await;
        assert!(
            engine.refresh_state().is_dirty(),
            "a cancelled reconcile must leave the index marked dirty"
        );
    }

    #[tokio::test]
    async fn read_file_focus_times_out_when_embedder_is_slow() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.search.search_timeout_secs = 1;
        let engine = Engine::new_with_embedder(
            cfg,
            Arc::new(SlowEmbedder {
                inner: MockEmbedder::new("mock-v1", 64),
            }),
        )
        .await
        .unwrap();
        let err = engine
            .read_file("a.rs", Some("anything"))
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::Timeout(_)),
            "read_file focus must honor the query timeout, got {err:?}"
        );
    }

    #[test]
    fn concurrency_prefers_config_then_server_then_one() {
        assert_eq!(resolve_concurrency(Some(2), Some(4)), 2, "config wins");
        assert_eq!(
            resolve_concurrency(None, Some(4)),
            4,
            "server is asked next"
        );
        assert_eq!(
            resolve_concurrency(None, None),
            1,
            "unknown server stays serial"
        );
        assert_eq!(
            resolve_concurrency(None, Some(64)),
            8,
            "derived value is capped"
        );
        assert_eq!(resolve_concurrency(Some(0), None), 1, "never zero");
    }
}

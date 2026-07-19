//! Engine: ties freshness + embed + index + distill into the always-fresh search path.
use crate::chunk::{chunk_file, language_for_path};
use crate::config::Config;
use crate::distill::{ContextEntry, distill_context};
use crate::embed::{Embedder, build_embedder};
use crate::error::{Error, Result};
use crate::freshness::{diff, exclude_matcher, is_excluded, resolve_excludes, scan};
use crate::index::{Index, StoredChunk};
use crate::refresh::RefreshState;
use futures::stream::{self, StreamExt, TryStreamExt};
use ignore::gitignore::Gitignore;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::OnceCell;

const MAX_WINDOW_LINES: usize = 80;

/// A single file's work item for parallel reconcile: chunks + metadata.
struct WorkItem {
    path: String,
    chunks: Vec<crate::chunk::Chunk>,
    file_hash: String,
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
    /// Effective exclude globs (lock files + tests + user), resolved once. Drives
    /// both the index-time `scan` and the read-time search filter via `matcher`.
    excludes: Vec<String>,
    matcher: Gitignore,
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
        Ok(Engine {
            config,
            embedder,
            index,
            refresh,
            excludes,
            matcher,
        })
    }

    pub fn embedder_id(&self) -> &str {
        self.embedder.id()
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
    /// `DirtyGuard` restores `dirty` if `reconcile_inner` errors OR the future is
    /// cancelled (dropped) before it finishes — otherwise a failed/cancelled reconcile
    /// would leave the state clean+active and let `search` skip the scan and serve stale.
    pub async fn reconcile(&self) -> Result<()> {
        let _guard = self.refresh.lock.lock().await;
        if self.refresh.can_skip_scan() {
            return Ok(());
        } // another reconcile beat us
        let dirty_guard = DirtyGuard::new(&self.refresh);
        self.reconcile_inner().await?;
        dirty_guard.disarm();
        Ok(())
    }

    /// Force a full reconcile regardless of flags (used by `reindex` and tests).
    pub async fn refresh(&self) -> Result<()> {
        let _guard = self.refresh.lock.lock().await;
        let dirty_guard = DirtyGuard::new(&self.refresh);
        self.reconcile_inner().await?;
        dirty_guard.disarm();
        Ok(())
    }

    async fn reconcile_inner(&self) -> Result<()> {
        let current = scan(&self.config.repo_root, &self.excludes)?;
        let stored = self.index.file_hashes().await?;
        let delta = diff(&current, &stored);
        let hash_of: std::collections::HashMap<&str, &str> = current
            .iter()
            .map(|s| (s.path.as_str(), s.hash.as_str()))
            .collect();

        // Phase 1: collect work items (sequential, I/O + tree-sitter is fast).
        let mut work: Vec<WorkItem> = Vec::new();
        for path in &delta.changed {
            // Gate on the language whitelist: skip files not in the configured languages.
            let detected = language_for_path(Path::new(path));
            if !self.config.is_language_allowed(Path::new(path), detected) {
                continue;
            }
            let abs = self.config.repo_root.join(path);
            let Ok(source) = std::fs::read_to_string(&abs) else {
                continue;
            };
            let chunks = chunk_file(Path::new(path), &source, MAX_WINDOW_LINES)?;
            if chunks.is_empty() {
                continue;
            }
            let file_hash = hash_of
                .get(path.as_str())
                .copied()
                .unwrap_or("")
                .to_string();
            work.push(WorkItem {
                path: path.clone(),
                chunks,
                file_hash,
            });
        }

        // Phase 2: embed in parallel (bounded), write serially. buffer_unordered runs up
        // to `concurrency` embeds at once but yields to a single serial consumer, so index
        // commits never overlap (no LanceDB commit conflicts) and only `concurrency` files
        // are held in flight (O(concurrency) memory, not O(repo)).
        let concurrency = self.config.embedder.embed_concurrency.max(1);
        let batch_limits = self.config.embedder.batch_limits();
        let embedder = &self.embedder;
        let index = &self.index;
        stream::iter(work)
            .map(|item| async move {
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
                let stored: Vec<StoredChunk> = chunks
                    .into_iter()
                    .zip(vectors)
                    .map(|(c, v)| StoredChunk {
                        path: path.clone(),
                        start_line: c.start_line,
                        end_line: c.end_line,
                        language: c.language,
                        symbol: c.symbol,
                        text: c.text,
                        file_hash: file_hash.clone(),
                        vector: v,
                    })
                    .collect();
                Ok::<(String, Vec<StoredChunk>), Error>((path, stored))
            })
            .buffer_unordered(concurrency)
            .try_for_each(|(path, stored)| async move { index.upsert_file(&path, stored).await })
            .await?;
        for path in &delta.deleted {
            self.index.delete_file(path).await?;
        }
        Ok(())
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
            None => Ok(chunks
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
                .collect()),
            Some(f) => {
                if chunks.is_empty() {
                    return Ok(vec![]);
                }
                let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
                let (fv, cvs) = tokio::time::timeout(self.query_timeout(), async {
                    let fv = self.embed_one(f).await?;
                    let cvs = self
                        .embedder
                        .embed_batched(&texts, self.config.embedder.batch_limits())
                        .await?;
                    Ok::<_, Error>((fv, cvs))
                })
                .await
                .map_err(|_| self.query_timeout_err())??;
                if cvs.len() != texts.len() {
                    return Err(Error::Embed(format!(
                        "embedder returned {} vectors for {} inputs",
                        cvs.len(),
                        texts.len()
                    )));
                }
                let mut scored: Vec<(f32, &crate::chunk::Chunk)> = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (dot(&fv, &cvs[i]), c))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored
                    .into_iter()
                    .take(5)
                    .map(|(score, c)| ContextEntry {
                        path: path.to_string(),
                        start_line: c.start_line,
                        end_line: c.end_line,
                        language: c.language.clone(),
                        symbol: c.symbol.clone(),
                        code: c.text.clone(),
                        score,
                        why_matched: format!("focus similarity {score:.3}"),
                    })
                    .collect())
            }
        }
    }
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

        let err = engine.reconcile().await.unwrap_err();
        assert!(
            matches!(err, Error::Embed(_)),
            "expected embed failure, got {err:?}"
        );

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
        // The default embed_concurrency is 1 (serial); force >1 so this test actually
        // drives the concurrent embed path (buffer_unordered with overlapping embeds
        // feeding a single serial index writer) rather than silently running serially.
        cfg.embedder.embed_concurrency = 4;
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
}

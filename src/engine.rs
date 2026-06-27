//! Engine: ties freshness + embed + index + distill into the always-fresh search path.
use crate::chunk::chunk_file;
use crate::config::Config;
use crate::distill::{ContextEntry, distill_context};
use crate::embed::{Embedder, build_embedder};
use crate::error::{Error, Result};
use crate::freshness::{diff, resolve_excludes, scan};
use crate::index::{Index, StoredChunk};
use crate::refresh::RefreshState;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::OnceCell;

const MAX_WINDOW_LINES: usize = 80;

pub struct Engine {
    config: Config,
    embedder: Box<dyn Embedder>,
    index: Index,
    refresh: Arc<RefreshState>,
}

impl Engine {
    pub async fn new(config: Config) -> Result<Engine> {
        let embedder = build_embedder(&config.embedder).await?;
        Self::new_with_embedder(config, embedder).await
    }

    pub async fn new_with_embedder(config: Config, embedder: Box<dyn Embedder>) -> Result<Engine> {
        Self::with_refresh_state(config, embedder, Arc::new(RefreshState::standalone())).await
    }

    pub async fn with_refresh_state(
        config: Config,
        embedder: Box<dyn Embedder>,
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
        Ok(Engine {
            config,
            embedder,
            index,
            refresh,
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
    /// If `reconcile_inner` fails, `dirty` is restored — otherwise a failed reconcile
    /// would leave the state clean+active and let `search` skip the scan and serve stale.
    pub async fn reconcile(&self) -> Result<()> {
        let _guard = self.refresh.lock.lock().await;
        if self.refresh.can_skip_scan() {
            return Ok(());
        } // another reconcile beat us
        self.refresh.clear_dirty();
        self.reconcile_inner()
            .await
            .inspect_err(|_| self.refresh.mark_dirty())
    }

    /// Force a full reconcile regardless of flags (used by `reindex` and tests).
    pub async fn refresh(&self) -> Result<()> {
        let _guard = self.refresh.lock.lock().await;
        self.refresh.clear_dirty();
        self.reconcile_inner()
            .await
            .inspect_err(|_| self.refresh.mark_dirty())
    }

    async fn reconcile_inner(&self) -> Result<()> {
        let excludes = resolve_excludes(&self.config.exclude, self.config.index_tests);
        let current = scan(&self.config.repo_root, &excludes)?;
        let stored = self.index.file_hashes().await?;
        let delta = diff(&current, &stored);
        let hash_of: std::collections::HashMap<&str, &str> = current
            .iter()
            .map(|s| (s.path.as_str(), s.hash.as_str()))
            .collect();

        for path in &delta.changed {
            let abs = self.config.repo_root.join(path);
            let Ok(source) = std::fs::read_to_string(&abs) else {
                continue;
            };
            let chunks = chunk_file(Path::new(path), &source, MAX_WINDOW_LINES)?;
            if chunks.is_empty() {
                continue;
            }
            let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
            let vectors = self
                .embedder
                .embed_batched(&texts, self.config.embedder.batch_limits())
                .await?;
            if vectors.len() != texts.len() {
                return Err(Error::Embed(format!(
                    "embedder returned {} vectors for {} inputs (file {path})",
                    vectors.len(),
                    texts.len()
                )));
            }
            let file_hash = hash_of
                .get(path.as_str())
                .copied()
                .unwrap_or("")
                .to_string();
            let stored_chunks: Vec<StoredChunk> = chunks
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
            self.index.upsert_file(path, stored_chunks).await?;
        }
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

    pub async fn search(&self, query: &str, k: Option<usize>) -> Result<Vec<ContextEntry>> {
        self.ensure_fresh().await?;
        let k = k.unwrap_or(self.config.search.default_k);
        let qv = self.embed_one(query).await?;
        let hits = self.index.search(&qv, k).await?;
        Ok(distill_context(
            hits,
            self.config.strip_comments,
            self.config.search.token_budget,
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
                let fv = self.embed_one(f).await?;
                let cvs = self
                    .embedder
                    .embed_batched(&texts, self.config.embedder.batch_limits())
                    .await?;
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
            .map_err(|e| e.clone())
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
        Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64)))
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
        let engine = Engine::new_with_embedder(cfg, Box::new(ZeroEmbedder))
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

        // First engine: no excludes -> indexes both files.
        {
            let mut cfg = Config::default_for(repo.path().to_path_buf());
            cfg.index_tests = true;
            let engine = Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64)))
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
        cfg.index_tests = true;
        cfg.exclude = vec!["**/data.txt".to_string()];
        let engine = Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64)))
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
        let engine = Engine::new_with_embedder(cfg, Box::new(ZeroEmbedder))
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

    #[tokio::test]
    async fn lazy_engine_from_engine_returns_ready_instance() {
        let repo = tempdir().unwrap();
        fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let state = Arc::new(RefreshState::standalone());
        let engine = Arc::new(
            Engine::with_refresh_state(
                cfg.clone(),
                Box::new(MockEmbedder::new("mock-v1", 64)),
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
        let engine = Engine::new_with_embedder(cfg, Box::new(Shared(embedder.clone())))
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
}

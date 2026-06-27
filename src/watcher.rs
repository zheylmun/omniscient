//! Background filesystem watcher: debounced FS events trigger an index reconcile.

use crate::config::WatchConfig;
use crate::engine::LazyEngine;
use crate::error::{Error, Result};
use crate::refresh::RefreshState;
use notify_debouncer_full::{new_debouncer, DebounceEventResult};
use notify_debouncer_full::notify::RecursiveMode;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// What the debouncer's (sync, off-runtime) handler sends to the async reconcile task.
enum Tick { Events, Error }

/// Keeps the OS watcher alive and aborts the reconcile task when dropped.
pub struct WatchGuard {
    _debouncer: Box<dyn std::any::Any + Send>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for WatchGuard {
    fn drop(&mut self) { self.task.abort(); }
}

/// Spawn a debounced recursive watcher over `repo_root`. Events under `.omniscient/`
/// (our own index writes) are filtered out to avoid a write→event→reconcile loop.
pub fn spawn(
    repo_root: PathBuf,
    cfg: &WatchConfig,
    lazy: LazyEngine,
    state: Arc<RefreshState>,
) -> Result<WatchGuard> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Tick>();
    let omni = repo_root.join(".omniscient");

    let handler = move |res: DebounceEventResult| match res {
        Ok(events) => {
            let relevant = events
                .iter()
                .any(|ev| ev.paths.iter().any(|p| !p.starts_with(&omni)));
            if relevant { let _ = tx.send(Tick::Events); }
        }
        Err(_errors) => { let _ = tx.send(Tick::Error); }
    };

    let mut debouncer = new_debouncer(Duration::from_millis(cfg.debounce_ms), None, handler)
        .map_err(|e| Error::Other(anyhow::anyhow!("watcher init: {e}")))?;
    debouncer
        .watch(&repo_root, RecursiveMode::Recursive)
        .map_err(|e| Error::Other(anyhow::anyhow!("watcher watch: {e}")))?;

    let task = tokio::spawn(async move {
        // Startup warm-up: a successful reconcile means the watch is live.
        if reconcile_once(&lazy).await {
            state.set_watch_active(true);
        }
        while let Some(tick) = rx.recv().await {
            match tick {
                Tick::Events => {
                    state.mark_dirty();
                    // A real event proves the watch is live; enable skip-scan on success.
                    if reconcile_once(&lazy).await {
                        state.set_watch_active(true);
                    }
                }
                Tick::Error => {
                    // OS watch may be dead — stay in fallback (search keeps scanning)
                    // until a real Tick::Events proves liveness. Reconcile best-effort
                    // but do NOT re-enable watch_active.
                    state.set_watch_active(false);
                    state.mark_dirty();
                    let _ = reconcile_once(&lazy).await;
                }
            }
        }
    });

    Ok(WatchGuard { _debouncer: Box::new(debouncer), task })
}

/// Best-effort reconcile through the lazy engine. Returns true iff the reconcile
/// succeeded. Does NOT touch watch_active — the caller decides, because a successful
/// reconcile does not prove the OS watch is still live.
async fn reconcile_once(lazy: &LazyEngine) -> bool {
    match lazy.get().await {
        Ok(engine) => match engine.reconcile().await {
            Ok(()) => true,
            Err(e) => { tracing::warn!("watcher reconcile failed: {e}"); false }
        },
        Err(e) => { tracing::warn!("watcher: engine init deferred: {e}"); false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, WatchConfig};
    use crate::embed::MockEmbedder;
    use crate::engine::{Engine, LazyEngine};
    use crate::refresh::RefreshState;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    async fn search_finds(engine: &Engine, query: &str, path: &str) -> bool {
        engine.search(query, Some(5)).await.unwrap().iter().any(|e| e.path == path)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watcher_reconciles_on_file_write() {
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let state = Arc::new(RefreshState::standalone());
        let engine = Arc::new(
            Engine::with_refresh_state(cfg.clone(), Box::new(MockEmbedder::new("mock-v1", 64)), state.clone())
                .await.unwrap(),
        );
        let lazy = LazyEngine::from_engine(cfg.clone(), state.clone(), engine.clone());
        let wcfg = WatchConfig { enabled: true, debounce_ms: 50 };
        let _guard = spawn(repo.path().to_path_buf(), &wcfg, lazy, state.clone()).unwrap();

        // New file appears; the watcher must reconcile it in within a bounded wait.
        std::fs::write(repo.path().join("b.rs"), "pub fn beta() {}\n").unwrap();

        let mut found = false;
        for _ in 0..100 { // up to ~5s, condition-based (not a fixed sleep)
            if search_finds(&engine, "beta", "b.rs").await { found = true; break; }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(found, "watcher should have reconciled b.rs into the index");
    }
}

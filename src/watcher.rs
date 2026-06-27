//! Background filesystem watcher: debounced FS events trigger an index reconcile.

use crate::config::WatchConfig;
use crate::engine::LazyEngine;
use crate::error::{Error, Result};
use crate::refresh::RefreshState;
use notify_debouncer_full::{new_debouncer, DebounceEventResult};
use notify_debouncer_full::notify::RecursiveMode;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// What the debouncer's (sync, off-runtime) handler sends to the async reconcile task.
/// The tick is only a wakeup — the freshness flags are updated synchronously in the
/// handler — so a single queued tick suffices to coalesce a burst of events.
enum Tick { Events, Error }

/// Mark the index dirty iff any event touches a path outside `.omniscient/`, and
/// report whether it did. Called from the (sync) debouncer handler so `dirty` flips
/// the instant a relevant event is observed — even while a long reconcile is still
/// in flight — which is what keeps `search` from skipping the scan past that event.
/// Events confined to `.omniscient/` are our own index writes and must be ignored to
/// avoid a write -> event -> reconcile feedback loop.
fn mark_if_relevant<'a>(
    mut paths: impl Iterator<Item = &'a Path>,
    omni: &Path,
    state: &RefreshState,
) -> bool {
    let relevant = paths.any(|p| !p.starts_with(omni));
    if relevant { state.mark_dirty(); }
    relevant
}

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
    repo_root: &Path,
    cfg: &WatchConfig,
    lazy: LazyEngine,
    state: Arc<RefreshState>,
) -> Result<WatchGuard> {
    // Capacity-1: the tick is just a wakeup and the freshness flags are set in this
    // handler, so a full channel already has a pending wakeup. `try_send` then drops
    // the redundant tick instead of letting a burst queue up behind a long reconcile.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Tick>(1);
    let omni = repo_root.join(".omniscient");
    let handler_state = state.clone();

    let handler = move |res: DebounceEventResult| match res {
        Ok(events) => {
            let paths = events.iter().flat_map(|ev| ev.paths.iter().map(std::path::PathBuf::as_path));
            if mark_if_relevant(paths, &omni, &handler_state) {
                let _ = tx.try_send(Tick::Events);
            }
        }
        Err(_errors) => {
            // The OS watch may be dead: drop to fallback (search resumes scanning)
            // and mark dirty so the next reconcile re-absorbs anything missed — both
            // synchronously here, so dropping the tick below never loses the signal.
            handler_state.set_watch_active(false);
            handler_state.mark_dirty();
            let _ = tx.try_send(Tick::Error);
        }
    };

    let mut debouncer = new_debouncer(Duration::from_millis(cfg.debounce_ms), None, handler)
        .map_err(|e| Error::Other(anyhow::anyhow!("watcher init: {e}")))?;
    debouncer
        .watch(repo_root, RecursiveMode::Recursive)
        .map_err(|e| Error::Other(anyhow::anyhow!("watcher watch: {e}")))?;

    let task = tokio::spawn(async move {
        // Startup warm-up: a successful reconcile means the watch is live.
        if reconcile_once(&lazy).await {
            state.set_watch_active(true);
        }
        while let Some(tick) = rx.recv().await {
            match tick {
                Tick::Events => {
                    // `dirty` was already set by the handler. Reconcile and, on
                    // success, (re)confirm the watch is live so search can skip-scan.
                    if reconcile_once(&lazy).await {
                        state.set_watch_active(true);
                    }
                }
                Tick::Error => {
                    // `watch_active` is already false + `dirty` set by the handler, so
                    // search keeps scanning. Reconcile best-effort but do NOT re-enable
                    // watch_active until a real Tick::Events proves liveness again.
                    let _ = reconcile_once(&lazy).await;
                }
            }
        }
    });

    Ok(WatchGuard { _debouncer: Box::new(debouncer), task })
}

/// Best-effort reconcile through the lazy engine. Returns true iff the reconcile
/// succeeded. Does NOT touch `watch_active` — the caller decides, because a successful
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
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn relevant_event_marks_dirty_synchronously() {
        // The debouncer handler must mark dirty the instant a relevant event is
        // seen — not when the async task later pops the tick — so a long in-flight
        // reconcile can't leave search on the skip-scan path past the event.
        let state = RefreshState::standalone();
        state.clear_dirty(); // simulate a caught-up watcher
        let omni = PathBuf::from("/repo/.omniscient");
        let touched = PathBuf::from("/repo/src/main.rs");
        assert!(mark_if_relevant([touched.as_path()].into_iter(), &omni, &state));
        assert!(state.is_dirty(), "a relevant event must mark dirty immediately");
    }

    #[test]
    fn omniscient_only_events_do_not_mark_dirty() {
        // Our own index writes under .omniscient/ must not mark dirty, or the
        // watcher would chase its own tail (write -> event -> reconcile -> write).
        let state = RefreshState::standalone();
        state.clear_dirty();
        let omni = PathBuf::from("/repo/.omniscient");
        let ours = PathBuf::from("/repo/.omniscient/lance/data.lance");
        assert!(!mark_if_relevant([ours.as_path()].into_iter(), &omni, &state));
        assert!(!state.is_dirty(), "our own index writes must not mark dirty");
    }

    #[test]
    fn mixed_events_are_relevant_when_any_path_is_outside_omniscient() {
        let state = RefreshState::standalone();
        state.clear_dirty();
        let omni = PathBuf::from("/repo/.omniscient");
        let ours = PathBuf::from("/repo/.omniscient/lance/data.lance");
        let touched = PathBuf::from("/repo/src/main.rs");
        assert!(mark_if_relevant([ours.as_path(), touched.as_path()].into_iter(), &omni, &state));
        assert!(state.is_dirty());
    }

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
        let _guard = spawn(repo.path(), &wcfg, lazy, state.clone()).unwrap();

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

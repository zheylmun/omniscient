//! Shared freshness flags coordinating the watcher and search-time reconcile.

use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

/// Coordinates freshness between the background watcher and search-time reconcile.
/// `dirty` means "an FS change has occurred that the index has not yet absorbed".
/// `watch_active` is true only while a healthy watcher is running; search may skip
/// its filesystem scan only when a watcher is active AND nothing is dirty.
pub struct RefreshState {
    dirty: AtomicBool,
    watch_active: AtomicBool,
    /// Serializes all reconciles (search-triggered and watcher-triggered) so the
    /// non-atomic delete-then-add in `Index::upsert_file` cannot interleave.
    pub lock: Mutex<()>,
}

impl RefreshState {
    /// Default state for engines with no watcher: dirty (forces a first scan) and
    /// inactive (so `can_skip_scan` is always false => always scan).
    pub fn standalone() -> Self {
        Self {
            dirty: AtomicBool::new(true),
            watch_active: AtomicBool::new(false),
            lock: Mutex::new(()),
        }
    }

    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
    pub fn clear_dirty(&self) {
        self.dirty.store(false, Ordering::Release);
    }
    pub fn set_watch_active(&self, v: bool) {
        self.watch_active.store(v, Ordering::Release);
    }
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }
    pub fn is_watch_active(&self) -> bool {
        self.watch_active.load(Ordering::Acquire)
    }

    /// Search may skip its scan only when a healthy watcher guarantees freshness.
    pub fn can_skip_scan(&self) -> bool {
        self.is_watch_active() && !self.is_dirty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standalone_is_dirty_and_inactive() {
        let s = RefreshState::standalone();
        assert!(s.is_dirty());
        assert!(!s.is_watch_active());
        assert!(!s.can_skip_scan(), "no watcher => never skip the scan");
    }

    #[test]
    fn skip_only_when_active_and_clean() {
        let s = RefreshState::standalone();
        s.set_watch_active(true);
        assert!(!s.can_skip_scan(), "still dirty => must scan");
        s.clear_dirty();
        assert!(s.can_skip_scan(), "active + clean => skip");
        s.mark_dirty();
        assert!(!s.can_skip_scan(), "new event => scan again");
        s.clear_dirty();
        s.set_watch_active(false);
        assert!(!s.can_skip_scan(), "inactive => scan even when clean");
    }
}

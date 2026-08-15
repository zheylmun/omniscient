//! The adaptive per-chunk byte budget.
//!
//! omniscient must never send the embeddings endpoint an input larger than its
//! context window: llama.cpp rejects it with a 400, and because a failed file's
//! hash is never stored, an unsplittable chunk would fail every subsequent
//! reconcile and wedge `search` permanently.
//!
//! Rather than estimate conservatively, this type *measures*. Every
//! `exceed_context_size_error` reports `n_prompt_tokens` for a payload whose
//! byte length we know, which is an exact bytes-per-token ratio for real
//! content. Three properties make that safe to rely on:
//!
//! - **It converges.** Each correction is a measurement, not a guess.
//! - **It terminates.** A token is never smaller than a byte, so a budget of
//!   `n_ctx` bytes always fits whatever the content — CJK, base64, minified JS —
//!   less `SPECIAL_TOKEN_SLACK` for the BOS/EOS the server adds on top. That is
//!   the floor. Note that reaching it is a property of the budget's *state*, not
//!   of `tighten`'s return value: callers must compare the budget against `n_ctx`
//!   rather than read `Tightened::Unchanged` as "at the floor".
//! - **It strictly decreases.** An overflow means `n_prompt_tokens > n_ctx`, so
//!   the recomputed target is necessarily below the budget that produced it.
//!
//! Starting optimistic is therefore the right posture: over-estimating costs a
//! few rejected requests once and self-heals, while under-estimating splits
//! large-but-legal definitions forever and degrades retrieval on every query.
//!
//! **The self-correction runs in one direction only.** An overflow is the only
//! signal there is, and it fires only when a chunk is too *big*, so the budget
//! ratchets down and never back up. A seed that is too high costs a couple of
//! rejected round-trips and then converges; a seed that is too low generates no
//! overflows at all, so nothing ever corrects it and the under-split is
//! permanent. That asymmetry is why the probe matters and why the optimistic
//! start is deliberate — see `caps` for how the window is discovered.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Starting bytes-per-token assumption. Real code measures 4.0–4.4 against
/// Qwen3-Embedding; 4 is the low end of that range and self-corrects if wrong.
pub const OPTIMISTIC_BYTES_PER_TOKEN: usize = 4;

/// Headroom applied to a *measured* correction. The wide margin a guess needs
/// exists to absorb estimation error; once the ratio is measured, the margin
/// only has to cover request framing overhead.
const HEADROOM_NUM: usize = 15;
const HEADROOM_DEN: usize = 16;

/// Tokens reserved at the floor for the server's own additions.
///
/// The floor rests on "a token is never smaller than a byte", which is true of
/// content but says nothing about the BOS/EOS (and any template) the server
/// prepends. Without this slack a chunk of exactly `n_ctx` bytes of
/// 1-token-per-byte content tokenizes to `n_ctx + 2`, is rejected while already at
/// the floor, and becomes a file that fails every reconcile forever — keeping
/// `dirty` set and defeating `can_skip_scan()` permanently.
pub(crate) const SPECIAL_TOKEN_SLACK: usize = 4;

/// Outcome of feeding an overflow back into the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tightened {
    /// The budget was lowered to this many bytes; re-split and retry.
    To(usize),
    /// This overflow offers no target smaller than the budget already in force.
    ///
    /// This does **not** on its own mean the floor was reached. A file
    /// embedding concurrently may have tightened past this measurement between
    /// the rejected chunk being split and this report arriving — a benign race
    /// whose correct response is to retry against the current, tighter budget.
    /// To distinguish the two, inspect the budget's state: the 1 byte/token
    /// floor is `n_ctx` bytes, so only an attempt made at or below that is
    /// genuinely out of room. Treating every `Unchanged` as the floor fails
    /// perfectly embeddable files.
    Unchanged,
}

/// The effective per-chunk byte budget, shared across a reconcile so one file's
/// correction benefits every later file.
#[derive(Debug)]
pub struct ChunkBudget {
    bytes: AtomicUsize,
    probed: Option<usize>,
    tightened: AtomicBool,
}

impl ChunkBudget {
    /// Start from the endpoint's reported window when it has one, otherwise the
    /// configured fallback. A reported `0` (what a `llama serve` supervisor
    /// answers about itself) is not a usable budget and counts as unknown.
    pub fn from_probe(probed: Option<usize>, fallback_tokens: usize) -> Self {
        let probed = probed.filter(|t| *t > 0);
        let tokens = probed.unwrap_or(fallback_tokens).max(1);
        Self {
            bytes: AtomicUsize::new(tokens.saturating_mul(OPTIMISTIC_BYTES_PER_TOKEN)),
            probed,
            tightened: AtomicBool::new(false),
        }
    }

    /// The budget the chunker should split against right now.
    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// The endpoint's reported context window, or `None` if it reported none.
    pub fn probed_tokens(&self) -> Option<usize> {
        self.probed
    }

    /// Whether an overflow has corrected the budget since startup. Reported by
    /// `diagnostics` so a misconfigured router is visible rather than inferred.
    pub fn was_tightened(&self) -> bool {
        self.tightened.load(Ordering::Relaxed)
    }

    /// Fold an `exceed_context_size_error` back into the shared budget.
    ///
    /// `sent_bytes` must be the **rejected chunk's actual byte length**, not the
    /// budget it was split against. The budget is only an upper bound, and passing
    /// it inflates the measured bytes-per-token ratio whenever a chunk lands well
    /// under it — which is the common case. The correction then shrinks the budget
    /// by little more than the headroom factor per round, and a file that splits
    /// perfectly well can exhaust the caller's round bound before converging.
    /// `Engine::chunks_for_embedding` reports that length alongside the pieces.
    pub fn tighten(&self, sent_bytes: usize, n_prompt_tokens: usize, n_ctx: usize) -> Tightened {
        // A token is never smaller than a byte, so this many bytes of *content*
        // always fits — less the slack the server needs for its own special
        // tokens.
        let floor = n_ctx.saturating_sub(SPECIAL_TOKEN_SLACK).max(1);
        let target = n_ctx
            .saturating_mul(sent_bytes)
            .checked_div(n_prompt_tokens)
            .map_or(floor, |measured| {
                (measured.saturating_mul(HEADROOM_NUM) / HEADROOM_DEN).max(floor)
            });
        let mut cur = self.bytes.load(Ordering::Relaxed);
        loop {
            if target >= cur {
                return Tightened::Unchanged;
            }
            match self.bytes.compare_exchange_weak(
                cur,
                target,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.tightened.store(true, Ordering::Relaxed);
                    return Tightened::To(target);
                }
                Err(actual) => cur = actual,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimistic_start_from_probe() {
        // 2048 tokens x 4 bytes/token. Optimistic relative to the old 3, and
        // safe because a wrong guess self-corrects on the first rejection.
        let b = ChunkBudget::from_probe(Some(2048), 999);
        assert_eq!(b.bytes(), 8192);
        assert_eq!(b.probed_tokens(), Some(2048));
        assert!(!b.was_tightened());
    }

    #[test]
    fn falls_back_when_unprobed() {
        // A non-llama.cpp endpoint reports nothing; max_chunk_tokens applies.
        let b = ChunkBudget::from_probe(None, 2048);
        assert_eq!(b.bytes(), 8192);
        assert_eq!(b.probed_tokens(), None);
    }

    #[test]
    fn zero_probe_is_treated_as_unknown() {
        // `llama serve`'s router reports n_ctx: 0. Never adopt it as a budget.
        let b = ChunkBudget::from_probe(Some(0), 1024);
        assert_eq!(b.bytes(), 4096);
        assert_eq!(b.probed_tokens(), None);
    }

    #[test]
    fn tighten_uses_the_measured_ratio() {
        // Sent a 32768-byte payload; server says it was 12000 tokens against an
        // 8192-token window. Measured ratio 32768/12000, so a chunk that fits
        // 8192 tokens is 8192 * 32768 / 12000 = 22369 bytes, less 15/16 headroom.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        assert_eq!(b.bytes(), 32768);
        let expected = (8192 * 32768 / 12000) * 15 / 16;
        assert_eq!(b.tighten(32768, 12000, 8192), Tightened::To(expected));
        assert_eq!(b.bytes(), expected);
        assert!(b.was_tightened());
    }

    #[test]
    fn tighten_is_monotonic() {
        // A later file reporting a laxer limit must not undo an earlier
        // correction — the budget only ever ratchets down.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        let Tightened::To(tight) = b.tighten(32768, 24000, 8192) else {
            panic!("expected a tightening");
        };
        assert_eq!(b.tighten(32768, 9000, 8192), Tightened::Unchanged);
        assert_eq!(b.bytes(), tight);
    }

    #[test]
    fn converges_to_the_one_byte_per_token_floor() {
        // Worst-case content (1 token per byte). Repeated tightening must reach
        // n_ctx bytes and then stop — that budget always fits, because a token
        // is never smaller than a byte.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        let mut rounds = 0;
        loop {
            let sent = b.bytes();
            // Pathological: every byte is its own token.
            match b.tighten(sent, sent, 8192) {
                Tightened::To(_) => rounds += 1,
                Tightened::Unchanged => break,
            }
            assert!(rounds < 32, "tighten must terminate, not spin");
        }
        assert_eq!(
            b.bytes(),
            8192 - SPECIAL_TOKEN_SLACK,
            "floor is 1 byte/token less the server's special-token slack"
        );
    }

    #[test]
    fn tighten_strictly_decreases_on_every_overflow() {
        // The convergence guarantee: because n_prompt_tokens > n_ctx is what
        // caused the 400, the new target is always below the current budget.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        let before = b.bytes();
        let Tightened::To(after) = b.tighten(before, 8193, 8192) else {
            panic!("an overflow must always tighten until the floor is reached");
        };
        assert!(after < before, "{after} must be below {before}");
    }

    #[test]
    fn the_floor_leaves_room_for_the_servers_special_tokens() {
        // "a token is never smaller than a byte" holds for *content*, but the
        // server prepends its own BOS/EOS. A chunk of exactly `n_ctx` bytes of
        // 1-token-per-byte content therefore tokenizes to `n_ctx + 2` and is
        // rejected while sitting exactly at the floor — a file that can never be
        // embedded, failing every reconcile forever, keeping `dirty` set and so
        // defeating `can_skip_scan()` permanently. That is the exact wedge this
        // design exists to prevent, so the floor must sit *below* `n_ctx`.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        while let Tightened::To(_) = b.tighten(b.bytes(), b.bytes(), 8192) {}
        assert!(
            b.bytes() < 8192,
            "the floor must leave headroom for special tokens, got {}",
            b.bytes()
        );
    }

    #[test]
    fn zero_prompt_tokens_falls_back_to_the_floor() {
        // A malformed 400 with no usable token count: drop straight to the
        // always-safe floor rather than dividing by zero.
        let b = ChunkBudget::from_probe(Some(8192), 8192);
        assert_eq!(
            b.tighten(32768, 0, 8192),
            Tightened::To(8192 - SPECIAL_TOKEN_SLACK)
        );
    }
}

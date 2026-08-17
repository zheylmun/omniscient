//! Distillation: deterministic context extraction (no LLM).
use crate::index::Hit;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub symbol: Option<String>,
    pub code: String,
    /// Raw relevance score (cosine similarity). Authoritative for ordering;
    /// `why_matched` is a human-facing rendering of it.
    pub score: f32,
    pub why_matched: String,
}

pub fn approx_tokens(s: &str) -> usize {
    (s.chars().count() / 4).max(1)
}

fn strip_banner(text: &str, strip: bool) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    if strip {
        let mut i = 0;
        while i < lines.len() {
            let t = lines[i].trim_start();
            let banner = t.is_empty()
                || t.starts_with("//")
                || t.starts_with('#')
                || t.starts_with("/*")
                || t.starts_with('*');
            if banner {
                i += 1;
            } else {
                break;
            }
        }
        lines.drain(0..i);
    }
    lines
        .iter()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

struct Merged {
    s: usize,
    e: usize,
    /// `start_line` of the hit appended most recently, which is what decides
    /// whether the *next* hit needs a newline in front of it. See the separator
    /// comment in `distill_context`: neither `s` nor `e` can answer that question.
    last_s: usize,
    score: f32,
    text: String,
    language: String,
    symbol: Option<String>,
}

/// Merge per-file hits into entries, then select by *shape*: keep every entry
/// scoring at least `relevance_ratio` of the top entry's score, so a sharp query
/// (one clear match) returns few results and a broad one returns many — rather
/// than a fixed k that is too coarse for both. The token budget is a hard ceiling
/// on top of that, and the single best match is always returned.
pub fn distill_context(
    hits: Vec<Hit>,
    strip_comments: bool,
    token_budget: usize,
    relevance_ratio: f32,
) -> Vec<ContextEntry> {
    let mut by_file: HashMap<String, Vec<Hit>> = HashMap::new();
    for h in hits {
        by_file.entry(h.chunk.path.clone()).or_default().push(h);
    }

    let mut entries: Vec<ContextEntry> = Vec::new();
    for (path, mut group) in by_file {
        group.sort_by_key(|h| (h.chunk.start_line, h.chunk.chunk_index));
        let mut cur: Option<Merged> = None;
        for h in group {
            let (s, e) = (h.chunk.start_line, h.chunk.end_line);
            match &mut cur {
                Some(m) if s <= m.e + 2 => {
                    // Append unconditionally: each hit is a distinct row, and
                    // pieces of one source line share a range, so gating on
                    // `e > m.e` would drop every piece after the first.
                    //
                    // The separator, though, is conditional. Only a piece that
                    // genuinely starts on a later line gets a newline: pieces of
                    // ONE physical line (a minified bundle is a single line) must
                    // be concatenated, or the distilled body stops matching the
                    // file it claims to quote — while the entry's line numbers
                    // still say it is all one line.
                    //
                    // The comparison is against the PREVIOUS HIT's start_line, and
                    // neither endpoint of the merged span can stand in for it.
                    // Against `m.e` (the running max end_line) the overlapping
                    // windows `line_windows` emits — `1-80`, `65-144`, ~20% overlap
                    // by construction — look like sub-line pieces, because `65 <= 80`,
                    // and window A's last line gets glued onto window B's first.
                    // That path is taken for any file tree-sitter yields no
                    // definitions for. Against `m.s` every piece after the first
                    // would compare to the span's *original* start, which says
                    // nothing about the piece immediately before it.
                    if s > m.last_s {
                        m.text.push('\n');
                    }
                    m.last_s = s;
                    m.e = m.e.max(e);
                    m.text.push_str(&h.chunk.text);
                    if h.score > m.score {
                        m.score = h.score;
                    }
                }
                _ => {
                    if let Some(m) = cur.take() {
                        entries.push(finish(&path, m, strip_comments));
                    }
                    cur = Some(Merged {
                        s,
                        e,
                        last_s: s,
                        score: h.score,
                        text: h.chunk.text.clone(),
                        language: h.chunk.language.clone(),
                        symbol: h.chunk.symbol.clone(),
                    });
                }
            }
        }
        if let Some(m) = cur.take() {
            entries.push(finish(&path, m, strip_comments));
        }
    }

    entries.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });

    // Entries are sorted by score desc, so the relevance floor is a cut point:
    // once one entry falls below it, every later one does too. A non-positive top
    // score means even the best match is weak — the floor would reject everything,
    // so the always-keep-the-first rule (out.is_empty()) carries it instead.
    let floor = entries.first().map_or(0.0, |e| e.score) * relevance_ratio.clamp(0.0, 1.0);

    let mut out = Vec::new();
    let mut used = 0usize;
    for e in entries {
        if !out.is_empty() && e.score < floor {
            break;
        }
        let cost = approx_tokens(&e.code);
        if out.is_empty() || used + cost <= token_budget {
            used += cost;
            out.push(e);
        }
    }
    out
}

fn finish(path: &str, m: Merged, strip_comments: bool) -> ContextEntry {
    ContextEntry {
        path: path.to_string(),
        start_line: m.s,
        end_line: m.e,
        language: m.language,
        symbol: m.symbol,
        code: strip_banner(&m.text, strip_comments),
        score: m.score,
        why_matched: format!("similarity {:.3}", m.score),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Hit, StoredChunk};
    fn hit(path: &str, s: usize, e: usize, score: f32, text: &str) -> Hit {
        Hit {
            score,
            chunk: StoredChunk {
                path: path.into(),
                start_line: s,
                end_line: e,
                chunk_index: 0,
                language: "rust".into(),
                symbol: None,
                text: text.into(),
                file_hash: "h".into(),
                vector: vec![],
            },
        }
    }

    #[test]
    fn merges_overlapping_same_file_hits() {
        let out = distill_context(
            vec![
                hit("a.rs", 1, 5, 0.9, "fn a(){}\n"),
                hit("a.rs", 6, 8, 0.8, "fn b(){}\n"),
                hit("b.rs", 1, 2, 0.7, "fn c(){}\n"),
            ],
            false,
            100_000,
            0.0,
        );
        let a: Vec<_> = out.iter().filter(|e| e.path == "a.rs").collect();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].start_line, 1);
        assert_eq!(a[0].end_line, 8);
    }

    #[test]
    fn strips_banner_when_enabled() {
        let out = distill_context(
            vec![hit(
                "a.rs",
                1,
                4,
                0.9,
                "// Copyright 2026\n// SPDX: MIT\npub fn a() {}\n",
            )],
            true,
            100_000,
            0.0,
        );
        assert!(!out[0].code.contains("Copyright"));
        assert!(out[0].code.contains("pub fn a"));
    }

    #[test]
    fn respects_token_budget_but_keeps_at_least_one() {
        let big = "x".repeat(10_000);
        let out = distill_context(
            vec![hit("a.rs", 1, 1, 0.9, &big), hit("b.rs", 1, 1, 0.8, &big)],
            false,
            100,
            0.0,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a.rs");
    }

    #[test]
    fn why_matched_reports_score() {
        let out = distill_context(
            vec![hit("a.rs", 1, 1, 0.876, "fn a(){}")],
            false,
            100_000,
            0.0,
        );
        assert!(out[0].why_matched.contains("0.87") || out[0].why_matched.contains("0.876"));
        assert!((out[0].score - 0.876).abs() < 1e-6);
    }

    #[test]
    fn equal_scores_order_deterministically_by_path() {
        // Equal scores across files must produce a stable, path-then-line ordering
        // (not the nondeterministic HashMap iteration order). Inputs are deliberately
        // out of order to prove the tiebreak, not insertion order, decides it.
        let order = || {
            distill_context(
                vec![
                    hit("z.rs", 1, 1, 0.5, "fn z(){}"),
                    hit("a.rs", 1, 1, 0.5, "fn a(){}"),
                    hit("m.rs", 1, 1, 0.5, "fn m(){}"),
                ],
                false,
                100_000,
                0.0,
            )
            .into_iter()
            .map(|e| e.path)
            .collect::<Vec<_>>()
        };
        assert_eq!(order(), vec!["a.rs", "m.rs", "z.rs"]);
        assert_eq!(order(), order()); // stable run-to-run
    }

    #[test]
    fn relevance_ratio_keeps_similar_drops_falloff() {
        // Distinct files so nothing merges; only the shape filter decides inclusion.
        // ratio 0.75, top 1.0 -> floor 0.75: keep 1.0/0.9/0.8, drop 0.5.
        let out = distill_context(
            vec![
                hit("a.rs", 1, 1, 1.0, "fn a(){}"),
                hit("b.rs", 1, 1, 0.9, "fn b(){}"),
                hit("c.rs", 1, 1, 0.8, "fn c(){}"),
                hit("d.rs", 1, 1, 0.5, "fn d(){}"),
            ],
            false,
            100_000,
            0.75,
        );
        let paths: Vec<_> = out.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(
            paths,
            vec!["a.rs", "b.rs", "c.rs"],
            "0.5 is below the floor"
        );
    }

    #[test]
    fn relevance_ratio_one_match_returns_just_the_best() {
        // A sharp query: one strong hit, the rest far below. floor 0.75*0.9=0.675.
        let out = distill_context(
            vec![
                hit("a.rs", 1, 1, 0.9, "fn a(){}"),
                hit("b.rs", 1, 1, 0.3, "fn b(){}"),
                hit("c.rs", 1, 1, 0.2, "fn c(){}"),
            ],
            false,
            100_000,
            0.75,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a.rs");
    }

    #[test]
    fn relevance_ratio_always_keeps_best_even_when_weak() {
        // Non-positive top score: the floor can't admit anything, but the best
        // match is still returned rather than an empty result.
        let out = distill_context(
            vec![
                hit("a.rs", 1, 1, -0.1, "fn a(){}"),
                hit("b.rs", 1, 1, -0.4, "fn b(){}"),
            ],
            false,
            100_000,
            0.75,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a.rs");
    }

    #[test]
    fn same_line_pieces_are_all_kept() {
        // A minified file is one long line split into sub-line pieces, so every
        // piece carries the same (start_line, end_line). Merging on line range
        // alone would keep only the first and silently drop the rest — returning
        // the wrong body for the exact case sub-line splitting exists to handle.
        let hit = |idx: usize, text: &str, score: f32| Hit {
            score,
            chunk: StoredChunk {
                path: "bundle.min.js".into(),
                start_line: 1,
                end_line: 1,
                chunk_index: idx,
                language: "javascript".into(),
                symbol: None,
                text: text.into(),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        let out = distill_context(
            vec![hit(2, "PIECE_TWO", 0.9), hit(0, "PIECE_ZERO", 0.8)],
            false,
            10_000,
            0.0,
        );
        assert_eq!(out.len(), 1, "pieces of one file merge into one entry");
        let code = &out[0].code;
        assert!(
            code.contains("PIECE_ZERO"),
            "first piece missing from {code}"
        );
        assert!(
            code.contains("PIECE_TWO"),
            "matched piece missing from {code}"
        );
        assert!(
            code.find("PIECE_ZERO") < code.find("PIECE_TWO"),
            "pieces must reassemble in chunk_index order, got {code}"
        );
    }

    #[test]
    fn sub_line_pieces_reassemble_without_inventing_newlines() {
        // Pieces of ONE physical line must be concatenated, not joined with '\n'.
        // A minified bundle is a single line, so a newline between pieces makes
        // the distilled body differ from the file it claims to quote — and the
        // line numbers on the entry say it is all one line.
        let hit = |idx: usize, text: &str| Hit {
            score: 0.9,
            chunk: StoredChunk {
                path: "bundle.min.js".into(),
                start_line: 1,
                end_line: 1,
                chunk_index: idx,
                language: "javascript".into(),
                symbol: None,
                text: text.into(),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        let out = distill_context(
            vec![hit(0, "let a=1;"), hit(1, "let b=2;")],
            false,
            10_000,
            0.0,
        );
        assert_eq!(out[0].code, "let a=1;let b=2;");
    }

    #[test]
    fn pieces_on_separate_lines_still_get_a_newline() {
        // The counterpart: genuinely distinct lines must not be run together.
        let hit = |idx: usize, line: usize, text: &str| Hit {
            score: 0.9,
            chunk: StoredChunk {
                path: "a.rs".into(),
                start_line: line,
                end_line: line,
                chunk_index: idx,
                language: "rust".into(),
                symbol: None,
                text: text.into(),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        let out = distill_context(
            vec![hit(0, 1, "let a=1;"), hit(1, 2, "let b=2;")],
            false,
            10_000,
            0.0,
        );
        assert_eq!(out[0].code, "let a=1;\nlet b=2;");
    }

    #[test]
    fn overlapping_line_windows_keep_their_newline() {
        // The multi-line counterpart, and the case a `s > m.e` separator test gets
        // wrong. `chunk::line_windows` emits windows overlapping by ~20% by
        // construction (`step = win - win/5`), so consecutive windows look like
        // `1-80` then `65-144`. Comparing the second hit's start_line against the
        // running max end_line gives `65 <= 80` — indistinguishable from a sub-line
        // piece — and window A's last line gets concatenated onto window B's first,
        // producing a body that does not appear anywhere in the file.
        //
        // This is not an exotic path: line windows are what every file
        // tree-sitter finds no definitions in falls back to (a Python script of
        // top-level statements, a Rust file of only `const`/`static`/
        // `macro_rules!`, or anything at all when `languages = []`).
        let hit = |idx: usize, s: usize, e: usize, text: &str| Hit {
            score: 0.9,
            chunk: StoredChunk {
                path: "script.py".into(),
                start_line: s,
                end_line: e,
                chunk_index: idx,
                language: "python".into(),
                symbol: None,
                text: text.into(),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        let out = distill_context(
            vec![
                hit(0, 1, 80, "first = 1\nlast_of_a = 80"),
                hit(1, 65, 144, "first_of_b = 65\nlast = 144"),
            ],
            false,
            10_000,
            0.0,
        );
        assert_eq!(out.len(), 1, "adjacent windows merge into one entry");
        assert!(
            !out[0].code.contains("last_of_a = 80first_of_b = 65"),
            "overlapping windows must not be spliced into one line, got:\n{}",
            out[0].code
        );
        assert_eq!(
            out[0].code,
            "first = 1\nlast_of_a = 80\nfirst_of_b = 65\nlast = 144"
        );
    }
}

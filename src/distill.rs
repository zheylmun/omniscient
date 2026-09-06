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
                    // What to append is decided by `merge_fragment`: verbatim
                    // shared lines (overlapping `line_windows`) are dropped so
                    // the body quotes each file line once, while sub-line
                    // fragments — whose texts don't byte-match — are appended
                    // whole so a split chunk still reassembles. Gating on
                    // `e > m.e` instead would drop every fragment after the
                    // first, since pieces of one source line share a range.
                    //
                    // The separator is conditional too. Only a fragment that
                    // genuinely starts on a later line gets a newline: pieces of
                    // ONE physical line (a minified bundle is a single line) must
                    // be concatenated, or the distilled body stops matching the
                    // file it claims to quote — while the entry's line numbers
                    // still say it is all one line. The comparison is against
                    // the PREVIOUS append's start line (`last_s`): `m.e` would
                    // misclassify a window continuation as a sub-line piece,
                    // and `m.s` says nothing about the piece just before.
                    let (fragment, fragment_start) = merge_fragment(m, s, e, &h.chunk.text);
                    if !fragment.is_empty() {
                        if fragment_start > m.last_s {
                            m.text.push('\n');
                        }
                        m.last_s = fragment_start;
                        m.text.push_str(fragment);
                    }
                    m.e = m.e.max(e);
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

/// Decide what part of an overlapping hit's text to append, and the line it
/// starts on. Hits whose spans overlap the merged span AND quote the file
/// verbatim (adjacent `line_windows` share ~20% of their lines by construction)
/// would repeat the shared lines if appended whole, so:
///
/// - a hit contained in the merged span whose text matches the merged text at
///   the corresponding line offset contributes nothing;
/// - a hit overhanging the end whose leading lines byte-match the merged
///   text's trailing lines contributes only the remainder.
///
/// The byte-match guard is what keeps sub-line fragments safe: pieces of one
/// split chunk share a line span too, but their texts differ, so they fall
/// through to plain concatenation — reassembling the split exactly as before.
fn merge_fragment<'a>(m: &Merged, s: usize, e: usize, incoming: &'a str) -> (&'a str, usize) {
    if s > m.e {
        return (incoming, s); // no shared lines to worry about
    }
    let shared = e.min(m.e) - s + 1;
    if e <= m.e {
        // Fully contained: skip only if the merged text really holds this
        // exact text where lines s..=e should sit.
        if line_region(&m.text, s - m.s, shared) == Some(incoming) {
            return ("", s);
        }
        return (incoming, s);
    }
    // Overhangs the end: drop the shared prefix if it matches the merged tail.
    match (
        tail_lines(&m.text, shared),
        first_lines_len(incoming, shared),
    ) {
        (Some(tail), Some(plen)) if tail == &incoming[..plen] => {
            let rest = &incoming[plen..];
            (rest.strip_prefix('\n').unwrap_or(rest), m.e + 1)
        }
        _ => (incoming, s),
    }
}

/// Byte length of the first `n` lines of `text` (without the trailing
/// newline), or `None` if `text` has fewer than `n` lines.
fn first_lines_len(text: &str, n: usize) -> Option<usize> {
    let mut len = 0usize;
    let mut count = 0usize;
    for line in text.split('\n') {
        if count > 0 {
            len += 1; // the '\n' before this line
        }
        len += line.len();
        count += 1;
        if count == n {
            return Some(len);
        }
    }
    None
}

/// The sub-slice of `text` covering `take` lines starting after `skip` lines,
/// or `None` if `text` has too few lines.
fn line_region(text: &str, skip: usize, take: usize) -> Option<&str> {
    let start = if skip == 0 {
        0
    } else {
        first_lines_len(text, skip)? + 1 // step past the '\n'
    };
    if start > text.len() {
        return None;
    }
    let region = &text[start..];
    let len = first_lines_len(region, take)?;
    Some(&region[..len])
}

/// The last `n` lines of `text`, or `None` if it has fewer.
fn tail_lines(text: &str, n: usize) -> Option<&str> {
    let mut newlines = 0usize;
    for (i, _) in text.rmatch_indices('\n') {
        newlines += 1;
        if newlines == n {
            return Some(&text[i + 1..]);
        }
    }
    (newlines + 1 == n).then_some(text)
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
        // The non-verbatim fallback: these two hits claim overlapping spans
        // (`1-80` then `65-144`) but their texts do NOT byte-match on the
        // shared lines, so the dedup guard in `merge_fragment` must stand
        // aside and both texts must be appended in full — separated by a
        // newline, not spliced into one line. (Verbatim overlap, the case
        // real `line_windows` produce, is covered by
        // `overlapping_windows_do_not_duplicate_shared_lines`.)
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

    /// Windows built by `chunk::line_windows` quote the file verbatim, so the
    /// overlap region (~20% by construction) appears in BOTH windows' texts.
    /// Merging must not repeat it: the entry claims one line span, and a body
    /// with the shared lines twice matches nothing in the file — and pays for
    /// the duplicate lines in the caller's token budget.
    #[test]
    fn overlapping_windows_do_not_duplicate_shared_lines() {
        let file_lines: Vec<String> = (1..=8).map(|i| format!("line {i}")).collect();
        let window = |idx: usize, s: usize, e: usize| Hit {
            score: 0.9,
            chunk: StoredChunk {
                path: "notes.md".into(),
                start_line: s,
                end_line: e,
                chunk_index: idx,
                language: "text".into(),
                symbol: None,
                text: file_lines[s - 1..e].join("\n"),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        // Windows 1-5 and 4-8: lines 4-5 are quoted by both.
        let out = distill_context(vec![window(0, 1, 5), window(1, 4, 8)], false, 10_000, 0.0);
        assert_eq!(out.len(), 1, "overlapping windows merge into one entry");
        assert_eq!((out[0].start_line, out[0].end_line), (1, 8));
        assert_eq!(
            out[0].code,
            file_lines.join("\n"),
            "the merged body must quote lines 1-8 exactly once each"
        );
    }

    /// A hit whose span is wholly inside the merged span, quoting the same
    /// text, adds nothing — it must be skipped, not appended a second time.
    #[test]
    fn contained_duplicate_window_is_not_appended() {
        let file_lines: Vec<String> = (1..=8).map(|i| format!("line {i}")).collect();
        let window = |idx: usize, s: usize, e: usize, score: f32| Hit {
            score,
            chunk: StoredChunk {
                path: "notes.md".into(),
                start_line: s,
                end_line: e,
                chunk_index: idx,
                language: "text".into(),
                symbol: None,
                text: file_lines[s - 1..e].join("\n"),
                file_hash: "h".into(),
                vector: vec![],
            },
        };
        // 1-8 first, then 4-6 (fully contained, verbatim). The contained hit's
        // higher score must still win the entry's score.
        let out = distill_context(
            vec![window(0, 1, 8, 0.7), window(1, 4, 6, 0.95)],
            false,
            10_000,
            0.0,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].code, file_lines.join("\n"));
        assert!(
            (out[0].score - 0.95).abs() < 1e-6,
            "score is the max of merged hits"
        );
    }
}

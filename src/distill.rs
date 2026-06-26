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
    pub why_matched: String,
}

pub fn approx_tokens(s: &str) -> usize { (s.chars().count() / 4).max(1) }

fn strip_noise(text: &str, strip_comments: bool) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    if strip_comments {
        let mut i = 0;
        while i < lines.len() {
            let t = lines[i].trim_start();
            let banner = t.is_empty() || t.starts_with("//") || t.starts_with('#')
                || t.starts_with("/*") || t.starts_with('*');
            if banner { i += 1; } else { break; }
        }
        lines.drain(0..i);
    }
    lines.iter().map(|l| l.trim_end()).collect::<Vec<_>>().join("\n")
}

struct Merged { s: usize, e: usize, score: f32, text: String, language: String, symbol: Option<String> }

pub fn distill_context(hits: Vec<Hit>, strip_comments: bool, token_budget: usize) -> Vec<ContextEntry> {
    let mut by_file: HashMap<String, Vec<Hit>> = HashMap::new();
    for h in hits { by_file.entry(h.chunk.path.clone()).or_default().push(h); }

    let mut entries: Vec<ContextEntry> = Vec::new();
    for (path, mut group) in by_file {
        group.sort_by_key(|h| h.chunk.start_line);
        let mut cur: Option<Merged> = None;
        for h in group {
            let (s, e) = (h.chunk.start_line, h.chunk.end_line);
            match &mut cur {
                Some(m) if s <= m.e + 2 => {
                    if e > m.e { m.e = e; m.text.push('\n'); m.text.push_str(&h.chunk.text); }
                    if h.score > m.score { m.score = h.score; }
                }
                _ => {
                    if let Some(m) = cur.take() { entries.push(finish(&path, m, strip_comments)); }
                    cur = Some(Merged { s, e, score: h.score, text: h.chunk.text.clone(),
                                        language: h.chunk.language.clone(), symbol: h.chunk.symbol.clone() });
                }
            }
        }
        if let Some(m) = cur.take() { entries.push(finish(&path, m, strip_comments)); }
    }

    entries.sort_by(|a, b| score_of(b).partial_cmp(&score_of(a)).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::new();
    let mut used = 0usize;
    for e in entries {
        let cost = approx_tokens(&e.code);
        if out.is_empty() || used + cost <= token_budget { used += cost; out.push(e); }
    }
    out
}

fn finish(path: &str, m: Merged, strip_comments: bool) -> ContextEntry {
    ContextEntry {
        path: path.to_string(), start_line: m.s, end_line: m.e,
        language: m.language, symbol: m.symbol,
        code: strip_noise(&m.text, strip_comments),
        why_matched: format!("similarity {:.3}", m.score),
    }
}

fn score_of(e: &ContextEntry) -> f32 {
    e.why_matched.rsplit(' ').next().and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Hit, StoredChunk};
    fn hit(path: &str, s: usize, e: usize, score: f32, text: &str) -> Hit {
        Hit { score, chunk: StoredChunk {
            path: path.into(), start_line: s, end_line: e, language: "rust".into(),
            symbol: None, text: text.into(), file_hash: "h".into(), vector: vec![] } }
    }

    #[test]
    fn merges_overlapping_same_file_hits() {
        let out = distill_context(vec![
            hit("a.rs",1,5,0.9,"fn a(){}\n"),
            hit("a.rs",6,8,0.8,"fn b(){}\n"),
            hit("b.rs",1,2,0.7,"fn c(){}\n"),
        ], false, 100_000);
        let a: Vec<_> = out.iter().filter(|e| e.path == "a.rs").collect();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].start_line, 1);
        assert_eq!(a[0].end_line, 8);
    }

    #[test]
    fn strips_leading_license_comments_when_enabled() {
        let out = distill_context(vec![hit("a.rs",1,4,0.9,
            "// Copyright 2026\n// SPDX: MIT\npub fn a() {}\n")], true, 100_000);
        assert!(!out[0].code.contains("Copyright"));
        assert!(out[0].code.contains("pub fn a"));
    }

    #[test]
    fn respects_token_budget_but_keeps_at_least_one() {
        let big = "x".repeat(10_000);
        let out = distill_context(vec![hit("a.rs",1,1,0.9,&big), hit("b.rs",1,1,0.8,&big)], false, 100);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a.rs");
    }

    #[test]
    fn why_matched_reports_score() {
        let out = distill_context(vec![hit("a.rs",1,1,0.876,"fn a(){}")], false, 100_000);
        assert!(out[0].why_matched.contains("0.87") || out[0].why_matched.contains("0.876"));
    }
}

//! Chunking: file -> semantic chunks (tree-sitter) or line-windows (fallback).
use crate::error::{Error, Result};
use std::path::Path;

/// Version of the chunking logic. Bump this whenever a change to how files are
/// split into chunks would alter the stored chunks for unchanged source (e.g.
/// new language support, different definition kinds, the test-skipping rules),
/// or whenever the on-disk chunk schema changes (a new column an old table lacks).
/// The index records it in `meta.json`; a mismatch forces a full reindex, the
/// same way a changed embedder id does — otherwise already-indexed files keep
/// their pre-change chunks until their content hash happens to change.
/// v3: added the `chunk_index` column to the chunks table.
/// v4: chunks are split to fit the embedding endpoint's context window.
/// 5: byte-budget splitting moved off the read path onto the embedding path,
/// and sub-line pieces are now identified by `chunk_index`.
pub const CHUNKER_VERSION: u32 = 5;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub start_line: usize, // 1-based inclusive
    pub end_line: usize,   // 1-based inclusive
    pub language: String,
    pub symbol: Option<String>,
}

pub fn language_for_path(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => Some("rust"),
        Some("py") => Some("python"),
        Some("js" | "jsx") => Some("javascript"),
        Some("ts") => Some("typescript"),
        Some("tsx") => Some("tsx"),
        _ => None,
    }
}

pub fn chunk_file(path: &Path, source: &str, max_window_lines: usize) -> Result<Vec<Chunk>> {
    chunk_source(language_for_path(path), source, max_window_lines)
}

/// Chunk `source` into structural pieces: one chunk per top-level definition
/// (tree-sitter), or line windows for unsupported languages. Purely structural
/// — no size budget is applied here, so a single oversized definition stays one
/// chunk. Splitting to fit an embedding endpoint's context window is a separate
/// concern; see `split_for_embedding`.
pub fn chunk_source(
    language: Option<&str>,
    source: &str,
    max_window_lines: usize,
) -> Result<Vec<Chunk>> {
    let chunks = if let Some(lang) = language {
        match treesitter_chunks(lang, source) {
            Ok(c) if !c.is_empty() => c,
            Ok(_) => line_windows(source, lang, max_window_lines),
            Err(e) => {
                tracing::warn!("tree-sitter parse failed for {lang}: {e}; using line windows");
                line_windows(source, lang, max_window_lines)
            }
        }
    } else {
        line_windows(source, "text", max_window_lines)
    };
    Ok(chunks)
}

/// Split chunks so none exceeds the endpoint's per-input byte budget.
///
/// This is an *embedding* concern, not a chunking one: only the indexing path
/// calls it. Keeping it off `chunk_file` is what stops `read_file`'s outline
/// from being broken into body fragments.
///
/// Covers all three ways a chunk gets too big — an oversized tree-sitter
/// definition (emitted whole, never recursed into), an oversized line window,
/// and a single oversized line (minified/generated files have no newline to
/// split on, so only a UTF-8 char-boundary split handles them).
pub fn split_for_embedding(chunks: Vec<Chunk>, max_bytes: usize) -> Vec<Chunk> {
    chunks
        .into_iter()
        .flat_map(|c| enforce_byte_budget(c, max_bytes))
        .collect()
}

fn line_windows(source: &str, language: &str, max_window_lines: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return vec![];
    }
    let win = max_window_lines.max(1);
    let step = (win - win / 5).max(1); // ~20% overlap
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + win).min(lines.len());
        chunks.push(Chunk {
            text: lines[start..end].join("\n"),
            start_line: start + 1,
            end_line: end,
            language: language.to_string(),
            symbol: None,
        });
        if end == lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

/// One piece of a split chunk. Pieces inherit the parent's language and symbol
/// so a split definition still retrieves under its own name.
fn piece(parent: &Chunk, text: String, start_line: usize, end_line: usize) -> Chunk {
    Chunk {
        text,
        start_line,
        end_line: end_line.max(start_line),
        language: parent.language.clone(),
        symbol: parent.symbol.clone(),
    }
}

/// Split a single line that is itself over budget, on UTF-8 char boundaries.
/// This is the only thing that handles minified/generated files, where the whole
/// file can be one line and there is no newline to split on.
fn split_long_line(line: &str, max: usize) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < line.len() {
        let mut end = (start + max).min(line.len());
        while end > start && !line.is_char_boundary(end) {
            end -= 1;
        }
        if end == start {
            // The budget is narrower than a single char; emit that whole char so
            // the loop always advances. Over budget by design — better than
            // slicing mid-codepoint or spinning forever.
            end = line[start..]
                .char_indices()
                .nth(1)
                .map_or(line.len(), |(i, _)| start + i);
        }
        parts.push(&line[start..end]);
        start = end;
    }
    parts
}

/// Split `chunk` so no piece exceeds `max_bytes`, preserving the text exactly
/// (concatenating the pieces reproduces the input). This is the single mechanism
/// covering all three ways a chunk gets too big: an oversized tree-sitter
/// definition, an oversized line window, and a single oversized line.
fn enforce_byte_budget(chunk: Chunk, max_bytes: usize) -> Vec<Chunk> {
    let max = max_bytes.max(1);
    if chunk.text.len() <= max {
        return vec![chunk];
    }
    let mut out: Vec<Chunk> = Vec::new();
    let mut buf = String::new();
    let mut buf_start = chunk.start_line;
    let mut line_no = chunk.start_line;
    // split_inclusive keeps the trailing newline, so the pieces rejoin losslessly.
    for line in chunk.text.split_inclusive('\n') {
        if line.len() > max {
            if !buf.is_empty() {
                let text = std::mem::take(&mut buf);
                out.push(piece(&chunk, text, buf_start, line_no.saturating_sub(1)));
            }
            for part in split_long_line(line, max) {
                out.push(piece(&chunk, part.to_string(), line_no, line_no));
            }
            line_no += 1;
            buf_start = line_no;
            continue;
        }
        if !buf.is_empty() && buf.len() + line.len() > max {
            let text = std::mem::take(&mut buf);
            out.push(piece(&chunk, text, buf_start, line_no.saturating_sub(1)));
            buf_start = line_no;
        }
        buf.push_str(line);
        line_no += 1;
    }
    if !buf.is_empty() {
        out.push(piece(&chunk, buf, buf_start, line_no.saturating_sub(1)));
    }
    out
}

fn ts_language(lang: &str) -> Option<tree_sitter::Language> {
    match lang {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "javascript" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        _ => None,
    }
}

fn def_kinds(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &[
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
        ],
        "python" => &["function_definition", "class_definition"],
        "javascript" => &[
            "function_declaration",
            "class_declaration",
            "method_definition",
            "lexical_declaration",
            "variable_declaration",
        ],
        "typescript" | "tsx" => &[
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "method_definition",
            "lexical_declaration",
        ],
        _ => &[],
    }
}

fn treesitter_chunks(lang: &str, source: &str) -> Result<Vec<Chunk>> {
    let language =
        ts_language(lang).ok_or_else(|| Error::Chunk(format!("no grammar for {lang}")))?;
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&language)
        .map_err(|e| Error::Chunk(e.to_string()))?;
    // non-fatal: caller falls through to line_windows
    let Some(tree) = parser.parse(source, None) else {
        return Ok(vec![]);
    };
    let kinds = def_kinds(lang);
    let bytes = source.as_bytes();

    let mut chunks = Vec::new();
    walk_children(tree.root_node(), kinds, bytes, &mut chunks, lang);
    Ok(chunks)
}

/// Walk a parent node's children, emitting one chunk per top-level def-kind node.
///
/// Items gated by a test attribute (`#[cfg(test)]`, `#[test]`, `#[bench]`,
/// `#[<path>::test]`) are skipped entirely — neither emitted nor recursed into — so
/// inline test code never enters the index. Non-test code keeps the original behavior:
/// a def-kind node emits one chunk and is not recursed into (nested defs are part of
/// it); any other node is recursed into so real modules still contribute their defs.
fn walk_children(
    parent: tree_sitter::Node,
    kinds: &[&str],
    bytes: &[u8],
    chunks: &mut Vec<Chunk>,
    lang: &str,
) {
    let mut cursor = parent.walk();
    let mut test_gated = false;
    for child in parent.children(&mut cursor) {
        match child.kind() {
            "attribute_item" => {
                // Attributes stack; any test attribute gates the item that follows.
                if is_test_attribute(child, bytes) {
                    test_gated = true;
                }
                continue;
            }
            // Comments between an attribute and its item must not consume the gate.
            "line_comment" | "block_comment" => continue,
            _ => {}
        }

        let gated = test_gated;
        test_gated = false;
        if gated {
            continue; // skip the gated item entirely: no chunk, no recursion
        }

        if kinds.contains(&child.kind()) {
            let text = child.utf8_text(bytes).unwrap_or("").to_string();
            let symbol = child
                .child_by_field_name("name")
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(std::string::ToString::to_string);
            chunks.push(Chunk {
                text,
                start_line: child.start_position().row + 1,
                end_line: child.end_position().row + 1,
                language: lang.to_string(),
                symbol,
            });
            // Do NOT recurse into a matched def — nested definitions (methods inside
            // impl/class) are part of this chunk and must not be emitted again.
        } else {
            walk_children(child, kinds, bytes, chunks, lang);
        }
    }
}

/// True if an `attribute_item` is a test marker: `#[test]`, `#[bench]`,
/// `#[<path>::test]` (e.g. `#[tokio::test]`), or a `#[cfg(...)]` whose predicate
/// contains the bare `test` cfg — `cfg(test)`, `cfg(all(test, …))`,
/// `cfg(any(test, …))` — but not when `test` is negated (`cfg(not(test))`).
fn is_test_attribute(attr_item: tree_sitter::Node, bytes: &[u8]) -> bool {
    let mut cursor = attr_item.walk();
    let Some(attribute) = attr_item
        .children(&mut cursor)
        .find(|n| n.kind() == "attribute")
    else {
        return false;
    };
    match attribute_name_last_segment(attribute, bytes).as_deref() {
        Some("test" | "bench") => true,
        Some("cfg") => attribute_has_cfg_test(attribute, bytes),
        _ => false,
    }
}

/// Last path segment of an attribute's name: `test` for `#[test]`, `cfg` for
/// `#[cfg(...)]`, `test` for `#[tokio::test]` (a `scoped_identifier`).
fn attribute_name_last_segment(attribute: tree_sitter::Node, bytes: &[u8]) -> Option<String> {
    let mut cursor = attribute.walk();
    let name = attribute
        .children(&mut cursor)
        .find(|n| matches!(n.kind(), "identifier" | "scoped_identifier"))?;
    match name.kind() {
        "identifier" => name
            .utf8_text(bytes)
            .ok()
            .map(std::string::ToString::to_string),
        "scoped_identifier" => name
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(std::string::ToString::to_string),
        _ => None,
    }
}

/// True if a `cfg(...)` attribute's argument list contains the bare `test` cfg
/// identifier (handles `cfg(test)`, `cfg(all(test, …))`, `cfg(any(test, …))`).
/// Matches the `identifier` AST node, so `cfg(feature = "test-utils")` (a string
/// literal) is not a false positive. A `test` nested under `not(...)` is ignored,
/// so `#[cfg(not(test))]` (production-only code) is not treated as a test gate.
fn attribute_has_cfg_test(attribute: tree_sitter::Node, bytes: &[u8]) -> bool {
    let mut cursor = attribute.walk();
    match attribute
        .children(&mut cursor)
        .find(|n| n.kind() == "token_tree")
    {
        Some(tt) => token_tree_has_test(tt, bytes),
        None => false,
    }
}

/// Search a cfg `token_tree` for a bare `test` predicate identifier, descending
/// into nested groups but skipping any group that is the argument of `not(...)`
/// (a negated `test` means the item is compiled when *not* testing).
fn token_tree_has_test(tt: tree_sitter::Node, bytes: &[u8]) -> bool {
    let mut cursor = tt.walk();
    let mut prev_was_not = false;
    for child in tt.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                let text = child.utf8_text(bytes).ok();
                if text == Some("test") {
                    return true;
                }
                prev_was_not = text == Some("not");
            }
            "token_tree" => {
                // Descend unless this group is the argument of `not(...)`.
                if !prev_was_not && token_tree_has_test(child, bytes) {
                    return true;
                }
                prev_was_not = false;
            }
            _ => prev_was_not = false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    fn read(p: &str) -> String {
        std::fs::read_to_string(p).unwrap()
    }

    fn parent(text: &str) -> Chunk {
        Chunk {
            text: text.to_string(),
            start_line: 10,
            end_line: 10 + text.lines().count().saturating_sub(1),
            language: "rust".into(),
            symbol: Some("big_fn".into()),
        }
    }

    #[test]
    fn chunk_file_does_not_split_on_size() {
        // The chunker's job is structure. A single oversized definition stays one
        // chunk; only the embedding path is allowed to break it up.
        let src = format!("fn big() {{\n{}}}\n", "    let x = 1;\n".repeat(500));
        let chunks = chunk_file(Path::new("big.rs"), &src, 80).unwrap();
        assert_eq!(chunks.len(), 1, "one definition is one structural chunk");
        assert!(chunks[0].text.len() > 5000);
    }

    #[test]
    fn split_for_embedding_is_lossless() {
        // Concatenating the pieces must reproduce the input exactly, or the index
        // would hold text that does not appear in the file.
        let src = format!("fn big() {{\n{}}}\n", "    let x = 1;\n".repeat(500));
        let chunks = chunk_file(Path::new("big.rs"), &src, 80).unwrap();
        let original: String = chunks.iter().map(|c| c.text.clone()).collect();
        let pieces = split_for_embedding(chunks, 512);
        assert!(pieces.len() > 1, "an oversized chunk must actually split");
        assert!(pieces.iter().all(|p| p.text.len() <= 512));
        let rejoined: String = pieces.iter().map(|c| c.text.clone()).collect();
        assert_eq!(rejoined, original);
    }

    #[test]
    fn split_for_embedding_handles_a_single_long_line() {
        // Minified/generated files have no newline to split on; only a UTF-8
        // char-boundary split handles them.
        //
        // Deviation from the brief: compare against the chunker's own output
        // (as `split_for_embedding_is_lossless` above does) rather than the raw
        // `src` literal. tree-sitter node byte ranges never include the file's
        // trailing newline after the matched top-level statement, so a chunk_file
        // + split_for_embedding round trip does not reproduce `src` exactly when
        // `src` ends in "\n" — a pre-existing property of the structural chunker,
        // unrelated to this task's change, confirmed present before it too.
        let src = format!("const D=\"{}\";\n", "x".repeat(4000));
        let chunks = chunk_file(Path::new("bundle.min.js"), &src, 80).unwrap();
        let original: String = chunks.iter().map(|c| c.text.clone()).collect();
        let pieces = split_for_embedding(chunks, 256);
        assert!(pieces.iter().all(|p| p.text.len() <= 256));
        let rejoined: String = pieces.iter().map(|c| c.text.clone()).collect();
        assert_eq!(rejoined, original);
    }

    #[test]
    fn oversized_definition_is_split_by_split_for_embedding() {
        // One top-level fn far larger than the budget. tree-sitter emits it as a
        // single chunk and does not recurse into it, so without splitting this
        // would be one enormous chunk headed to the embedder.
        let body = "    let x = 1;\n".repeat(2000);
        let source = format!("fn huge() {{\n{body}}}\n");
        let structural = chunk_file(Path::new("a.rs"), &source, 80).unwrap();
        let out = split_for_embedding(structural, 1000);
        assert!(out.len() > 1, "oversized definition must be split");
        for c in &out {
            assert!(c.text.len() <= 1000, "chunk of {} bytes", c.text.len());
        }
        assert!(
            out.iter().any(|c| c.symbol.as_deref() == Some("huge")),
            "split pieces must keep the definition's symbol"
        );
    }

    #[test]
    fn minified_single_line_file_is_split_by_split_for_embedding() {
        // No newlines at all: the case line-window fallback cannot handle.
        let source = format!("const DATA=\"{}\";", "a".repeat(50_000));
        let structural = chunk_file(Path::new("data.ts"), &source, 80).unwrap();
        let out = split_for_embedding(structural, 1000);
        for c in &out {
            assert!(c.text.len() <= 1000, "chunk of {} bytes", c.text.len());
        }
    }

    #[test]
    fn generous_budget_leaves_normal_files_unsplit() {
        let source = "fn a() {}\nfn b() {}\n";
        let structural = chunk_file(Path::new("a.rs"), source, 80).unwrap();
        let out = split_for_embedding(structural, 100_000);
        assert_eq!(out.len(), 2, "normal code must not be split by this change");
    }

    #[test]
    fn under_budget_chunk_passes_through_untouched() {
        let c = parent("fn a() {}\n");
        let out = enforce_byte_budget(c, 1000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "fn a() {}\n");
        assert_eq!(out[0].start_line, 10);
    }

    #[test]
    fn oversized_chunk_splits_on_line_boundaries_within_budget() {
        // 100 lines x 10 bytes = 1000 bytes, budget 100 -> ~10 pieces.
        let text = "0123456789\n".repeat(100);
        let out = enforce_byte_budget(parent(&text), 100);
        assert!(out.len() > 1, "must split");
        for p in &out {
            assert!(
                p.text.len() <= 100,
                "piece of {} bytes exceeds the budget",
                p.text.len()
            );
        }
        // Reassembly is lossless: split_inclusive keeps the newlines.
        let rejoined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(rejoined, text);
    }

    #[test]
    fn split_pieces_inherit_symbol_and_language() {
        let out = enforce_byte_budget(parent(&"x\n".repeat(500)), 100);
        assert!(out.len() > 1);
        for p in &out {
            assert_eq!(p.symbol.as_deref(), Some("big_fn"));
            assert_eq!(p.language, "rust");
        }
    }

    #[test]
    fn split_pieces_have_ascending_nonoverlapping_line_ranges() {
        let out = enforce_byte_budget(parent(&"x\n".repeat(500)), 100);
        assert_eq!(
            out[0].start_line, 10,
            "first piece starts at the parent line"
        );
        for w in out.windows(2) {
            assert!(w[0].end_line >= w[0].start_line, "range must not invert");
            assert!(
                w[1].start_line > w[0].end_line,
                "pieces must not overlap: {:?} then {:?}",
                (w[0].start_line, w[0].end_line),
                (w[1].start_line, w[1].end_line)
            );
        }
    }

    #[test]
    fn single_enormous_line_is_split_on_byte_boundaries() {
        // The minified/generated case: one line, no newline to split on. This is
        // the input that line-based splitting cannot handle at all.
        let text = "a".repeat(10_000);
        let out = enforce_byte_budget(parent(&text), 100);
        assert!(out.len() >= 100, "expected many pieces, got {}", out.len());
        for p in &out {
            assert!(p.text.len() <= 100);
            assert_eq!(p.start_line, p.end_line, "a sub-line piece spans one line");
        }
        let rejoined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(rejoined, text);
    }

    #[test]
    fn multibyte_line_splits_on_char_boundaries() {
        // Every char is 4 bytes; a naive byte slice would panic mid-codepoint.
        let text = "😀".repeat(1000);
        let out = enforce_byte_budget(parent(&text), 10);
        for p in &out {
            assert!(p.text.len() <= 10);
        }
        let rejoined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(rejoined, text, "no codepoint may be lost or corrupted");
    }

    #[test]
    fn budget_smaller_than_one_char_still_terminates() {
        // A 4-byte char against a 1-byte budget: the piece must exceed the budget
        // rather than loop forever producing empty slices.
        let out = enforce_byte_budget(parent("😀😀"), 1);
        assert_eq!(out.len(), 2);
        let rejoined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(rejoined, "😀😀");
    }

    #[test]
    fn mixed_long_and_short_lines_all_fit() {
        let text = format!("short\n{}\nshort\n", "z".repeat(5000));
        let out = enforce_byte_budget(parent(&text), 200);
        for p in &out {
            assert!(p.text.len() <= 200, "piece of {} bytes", p.text.len());
        }
        let rejoined: String = out.iter().map(|p| p.text.as_str()).collect();
        assert_eq!(rejoined, text);
    }

    #[test]
    fn rust_chunks_on_definitions() {
        let src = read("tests/fixtures/sample.rs");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.rs"), &src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.clone()).collect();
        assert!(symbols.contains(&"alpha".to_string()));
        assert!(symbols.contains(&"Point".to_string()));
        assert!(chunks.iter().all(|c| c.language == "rust"));
        let alpha = chunks
            .iter()
            .find(|c| c.symbol.as_deref() == Some("alpha"))
            .unwrap();
        assert!(alpha.text.contains("pub fn alpha"));
    }

    #[test]
    fn python_ts_js_tsx_jsx_recognized() {
        let py = read("tests/fixtures/sample.py");
        let c = chunk_file(Path::new("tests/fixtures/sample.py"), &py, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("alpha")));
        let ts = read("tests/fixtures/sample.ts");
        let c = chunk_file(Path::new("tests/fixtures/sample.ts"), &ts, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("alpha")));
        assert!(c.iter().all(|c| c.language == "typescript"));
        let js = read("tests/fixtures/sample.js");
        let c = chunk_file(Path::new("tests/fixtures/sample.js"), &js, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("alpha")));
        assert!(c.iter().all(|c| c.language == "javascript"));
        // TSX uses the dedicated TSX grammar, not the TS grammar.
        let tsx = read("tests/fixtures/sample.tsx");
        let c = chunk_file(Path::new("tests/fixtures/sample.tsx"), &tsx, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("Alpha")));
        assert!(c.iter().all(|c| c.language == "tsx"));
        // JSX uses the JavaScript grammar (no separate JSX grammar exists).
        let jsx = read("tests/fixtures/sample.jsx");
        let c = chunk_file(Path::new("tests/fixtures/sample.jsx"), &jsx, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("Alpha")));
        assert!(c.iter().all(|c| c.language == "javascript"));
    }

    #[test]
    fn unknown_language_uses_line_windows() {
        let md = read("tests/fixtures/sample.md");
        let c = chunk_file(Path::new("tests/fixtures/sample.md"), &md, 2).unwrap();
        assert!(!c.is_empty());
        assert!(c.iter().all(|c| c.language == "text" && c.symbol.is_none()));
        assert!(c.iter().all(|c| c.end_line - c.start_line < 2));
    }

    #[test]
    fn line_windows_cover_full_range() {
        let src = (1..=10)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let c = chunk_source(None, &src, 4).unwrap();
        assert_eq!(c.first().unwrap().start_line, 1);
        assert_eq!(c.last().unwrap().end_line, 10);
    }

    // Regression: nested definitions must not appear as standalone chunks (over-chunking).

    #[test]
    fn rust_no_over_chunking() {
        let src = read("tests/fixtures/sample.rs");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.rs"), &src, 100).unwrap();
        // beta is a method inside `impl Point` — it must NOT appear as a standalone chunk
        assert!(
            !chunks.iter().any(|c| c.symbol.as_deref() == Some("beta")),
            "beta should not be emitted as a standalone chunk; chunks: {chunks:?}"
        );
        // Exactly 3 top-level definitions: alpha (fn), Point (struct), impl Point (symbol=None)
        assert_eq!(
            chunks.len(),
            3,
            "expected exactly 3 chunks, got {}: {chunks:?}",
            chunks.len()
        );
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"alpha"), "alpha missing from {symbols:?}");
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
    }

    #[test]
    fn python_no_over_chunking() {
        let src = read("tests/fixtures/sample.py");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.py"), &src, 100).unwrap();
        // beta is a method inside class Point — must NOT be a standalone chunk
        assert!(
            !chunks.iter().any(|c| c.symbol.as_deref() == Some("beta")),
            "beta should not be emitted as a standalone chunk; chunks: {chunks:?}"
        );
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"alpha"), "alpha missing from {symbols:?}");
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
        assert_eq!(
            chunks.len(),
            2,
            "expected exactly 2 chunks (alpha, Point), got {}: {chunks:?}",
            chunks.len()
        );
    }

    #[test]
    fn typescript_no_over_chunking() {
        let src = read("tests/fixtures/sample.ts");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.ts"), &src, 100).unwrap();
        // beta is a method inside class Point — must NOT be a standalone chunk
        assert!(
            !chunks.iter().any(|c| c.symbol.as_deref() == Some("beta")),
            "beta should not be emitted as a standalone chunk; chunks: {chunks:?}"
        );
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        // alpha is exported (wrapped in export_statement) — must still be captured
        assert!(
            symbols.contains(&"alpha"),
            "alpha (exported fn) missing from {symbols:?}"
        );
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
        assert_eq!(
            chunks.len(),
            2,
            "expected exactly 2 chunks (alpha, Point), got {}: {chunks:?}",
            chunks.len()
        );
    }

    #[test]
    fn tsx_no_over_chunking() {
        let src = read("tests/fixtures/sample.tsx");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.tsx"), &src, 100).unwrap();
        // render is a method inside class Point — must NOT be a standalone chunk
        assert!(
            !chunks.iter().any(|c| c.symbol.as_deref() == Some("render")),
            "render should not be emitted as a standalone chunk; chunks: {chunks:?}"
        );
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"Alpha"), "Alpha missing from {symbols:?}");
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
        assert!(chunks.iter().all(|c| c.language == "tsx"));
        assert_eq!(
            chunks.len(),
            2,
            "expected exactly 2 chunks (Alpha, Point), got {}: {chunks:?}",
            chunks.len()
        );
    }

    #[test]
    fn javascript_no_over_chunking() {
        let src = read("tests/fixtures/sample.js");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.js"), &src, 100).unwrap();
        // beta is a method inside class Point — must NOT be a standalone chunk
        assert!(
            !chunks.iter().any(|c| c.symbol.as_deref() == Some("beta")),
            "beta should not be emitted as a standalone chunk; chunks: {chunks:?}"
        );
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"alpha"), "alpha missing from {symbols:?}");
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
        assert_eq!(
            chunks.len(),
            2,
            "expected exactly 2 chunks (alpha, Point), got {}: {chunks:?}",
            chunks.len()
        );
    }

    #[test]
    fn rust_skips_test_code() {
        let src = read("tests/fixtures/sample_tests.rs");
        let chunks = chunk_file(Path::new("tests/fixtures/sample_tests.rs"), &src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        // production code is kept
        assert!(
            symbols.contains(&"production_fn"),
            "production_fn missing from {symbols:?}"
        );
        assert!(
            symbols.contains(&"Widget"),
            "Widget missing from {symbols:?}"
        );
        // every flavor of test code is dropped
        for banned in [
            "test_helper",
            "checks_widget",
            "checks_production_fn",
            "standalone_test",
        ] {
            assert!(
                !symbols.contains(&banned),
                "{banned} should be skipped; chunks: {chunks:?}"
            );
        }
        // exactly the two production defs survive
        assert_eq!(
            chunks.len(),
            2,
            "expected exactly 2 production chunks, got {}: {chunks:?}",
            chunks.len()
        );
    }

    #[test]
    fn cfg_feature_with_test_substring_not_skipped() {
        // `cfg(feature = "test-utils")` must NOT be treated as a test gate:
        // "test-utils" is a string literal, not the `test` cfg identifier.
        let src = r#"
#[cfg(feature = "test-utils")]
pub fn util_fn() -> i32 { 1 }

pub fn always() -> i32 { 2 }
"#;
        let chunks = chunk_source(Some("rust"), src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(
            symbols.contains(&"util_fn"),
            "util_fn wrongly skipped: {symbols:?}"
        );
        assert!(symbols.contains(&"always"), "always missing: {symbols:?}");
    }

    #[test]
    fn cfg_not_test_is_kept() {
        // `#[cfg(not(test))]` is production-only code (compiled when NOT testing):
        // a `test` nested under `not(...)` must NOT be treated as a test gate.
        let src = r"
#[cfg(not(test))]
pub fn only_in_prod() -> i32 { 1 }

#[cfg(all(unix, not(test)))]
pub fn unix_prod() -> i32 { 2 }

pub fn always() -> i32 { 3 }
";
        let chunks = chunk_source(Some("rust"), src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(
            symbols.contains(&"only_in_prod"),
            "only_in_prod wrongly skipped: {symbols:?}"
        );
        assert!(
            symbols.contains(&"unix_prod"),
            "unix_prod wrongly skipped: {symbols:?}"
        );
        assert!(symbols.contains(&"always"), "always missing: {symbols:?}");
    }

    #[test]
    fn tokio_test_attribute_is_skipped() {
        // `#[tokio::test]` (scoped path ending in `test`) is also a test marker.
        let src = r"
pub fn keep_me() -> i32 { 1 }

#[tokio::test]
async fn async_test() {}
";
        let chunks = chunk_source(Some("rust"), src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"keep_me"), "keep_me missing: {symbols:?}");
        assert!(
            !symbols.contains(&"async_test"),
            "async_test should be skipped: {symbols:?}"
        );
    }
}

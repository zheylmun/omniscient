//! Chunking: file -> semantic chunks (tree-sitter) or line-windows (fallback).
use crate::error::{Error, Result};
use std::path::Path;

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
        Some("ts") | Some("tsx") => Some("typescript"),
        _ => None,
    }
}

pub fn chunk_file(path: &Path, source: &str, max_window_lines: usize) -> Result<Vec<Chunk>> {
    chunk_source(language_for_path(path), source, max_window_lines)
}

pub fn chunk_source(language: Option<&str>, source: &str, max_window_lines: usize) -> Result<Vec<Chunk>> {
    if let Some(lang) = language {
        match treesitter_chunks(lang, source) {
            Ok(chunks) if !chunks.is_empty() => return Ok(chunks),
            Ok(_) => {}
            Err(e) => tracing::warn!("tree-sitter parse failed for {lang}: {e}; using line windows"),
        }
        return Ok(line_windows(source, lang, max_window_lines));
    }
    Ok(line_windows(source, "text", max_window_lines))
}

fn line_windows(source: &str, language: &str, max_window_lines: usize) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() { return vec![]; }
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
        if end == lines.len() { break; }
        start += step;
    }
    chunks
}

fn ts_language(lang: &str) -> Option<tree_sitter::Language> {
    match lang {
        "rust" => Some(tree_sitter_rust::LANGUAGE.into()),
        "python" => Some(tree_sitter_python::LANGUAGE.into()),
        "typescript" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        _ => None,
    }
}

fn def_kinds(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &["function_item", "struct_item", "enum_item", "trait_item", "impl_item"],
        "python" => &["function_definition", "class_definition"],
        "typescript" => &["function_declaration", "class_declaration",
                          "interface_declaration", "method_definition", "lexical_declaration"],
        _ => &[],
    }
}

fn treesitter_chunks(lang: &str, source: &str) -> Result<Vec<Chunk>> {
    let language = ts_language(lang).ok_or_else(|| Error::Chunk(format!("no grammar for {lang}")))?;
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language).map_err(|e| Error::Chunk(e.to_string()))?;
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return Ok(vec![]), // non-fatal: caller falls through to line_windows
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
                .map(|s| s.to_string());
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
/// `#[<path>::test]` (e.g. `#[tokio::test]`), or `#[cfg(test)]` / `cfg(all(test, …))`.
fn is_test_attribute(attr_item: tree_sitter::Node, bytes: &[u8]) -> bool {
    let mut cursor = attr_item.walk();
    let attribute = match attr_item.children(&mut cursor).find(|n| n.kind() == "attribute") {
        Some(a) => a,
        None => return false,
    };
    match attribute_name_last_segment(attribute, bytes).as_deref() {
        Some("test") | Some("bench") => true,
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
        "identifier" => name.utf8_text(bytes).ok().map(|s| s.to_string()),
        "scoped_identifier" => name
            .child_by_field_name("name")
            .and_then(|n| n.utf8_text(bytes).ok())
            .map(|s| s.to_string()),
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
    match attribute.children(&mut cursor).find(|n| n.kind() == "token_tree") {
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
    fn read(p: &str) -> String { std::fs::read_to_string(p).unwrap() }

    #[test]
    fn rust_chunks_on_definitions() {
        let src = read("tests/fixtures/sample.rs");
        let chunks = chunk_file(Path::new("tests/fixtures/sample.rs"), &src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.clone()).collect();
        assert!(symbols.contains(&"alpha".to_string()));
        assert!(symbols.contains(&"Point".to_string()));
        assert!(chunks.iter().all(|c| c.language == "rust"));
        let alpha = chunks.iter().find(|c| c.symbol.as_deref() == Some("alpha")).unwrap();
        assert!(alpha.text.contains("pub fn alpha"));
    }

    #[test]
    fn python_and_ts_recognized() {
        let py = read("tests/fixtures/sample.py");
        let c = chunk_file(Path::new("tests/fixtures/sample.py"), &py, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("alpha")));
        let ts = read("tests/fixtures/sample.ts");
        let c = chunk_file(Path::new("tests/fixtures/sample.ts"), &ts, 100).unwrap();
        assert!(c.iter().any(|c| c.symbol.as_deref() == Some("alpha")));
        assert!(c.iter().all(|c| c.language == "typescript"));
    }

    #[test]
    fn unknown_language_uses_line_windows() {
        let md = read("tests/fixtures/sample.md");
        let c = chunk_file(Path::new("tests/fixtures/sample.md"), &md, 2).unwrap();
        assert!(!c.is_empty());
        assert!(c.iter().all(|c| c.language == "text" && c.symbol.is_none()));
        assert!(c.iter().all(|c| c.end_line - c.start_line + 1 <= 2));
    }

    #[test]
    fn line_windows_cover_full_range() {
        let src = (1..=10).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n");
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
        assert_eq!(chunks.len(), 3, "expected exactly 3 chunks, got {}: {chunks:?}", chunks.len());
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
        assert_eq!(chunks.len(), 2, "expected exactly 2 chunks (alpha, Point), got {}: {chunks:?}", chunks.len());
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
        assert!(symbols.contains(&"alpha"), "alpha (exported fn) missing from {symbols:?}");
        assert!(symbols.contains(&"Point"), "Point missing from {symbols:?}");
        assert_eq!(chunks.len(), 2, "expected exactly 2 chunks (alpha, Point), got {}: {chunks:?}", chunks.len());
    }

    #[test]
    fn rust_skips_test_code() {
        let src = read("tests/fixtures/sample_tests.rs");
        let chunks = chunk_file(Path::new("tests/fixtures/sample_tests.rs"), &src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        // production code is kept
        assert!(symbols.contains(&"production_fn"), "production_fn missing from {symbols:?}");
        assert!(symbols.contains(&"Widget"), "Widget missing from {symbols:?}");
        // every flavor of test code is dropped
        for banned in ["test_helper", "checks_widget", "checks_production_fn", "standalone_test", "tests"] {
            assert!(
                !symbols.contains(&banned),
                "{banned} should be skipped; chunks: {chunks:?}"
            );
        }
        // exactly the two production defs survive
        assert_eq!(chunks.len(), 2, "expected exactly 2 production chunks, got {}: {chunks:?}", chunks.len());
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
        assert!(symbols.contains(&"util_fn"), "util_fn wrongly skipped: {symbols:?}");
        assert!(symbols.contains(&"always"), "always missing: {symbols:?}");
    }

    #[test]
    fn cfg_not_test_is_kept() {
        // `#[cfg(not(test))]` is production-only code (compiled when NOT testing):
        // a `test` nested under `not(...)` must NOT be treated as a test gate.
        let src = r#"
#[cfg(not(test))]
pub fn only_in_prod() -> i32 { 1 }

#[cfg(all(unix, not(test)))]
pub fn unix_prod() -> i32 { 2 }

pub fn always() -> i32 { 3 }
"#;
        let chunks = chunk_source(Some("rust"), src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"only_in_prod"), "only_in_prod wrongly skipped: {symbols:?}");
        assert!(symbols.contains(&"unix_prod"), "unix_prod wrongly skipped: {symbols:?}");
        assert!(symbols.contains(&"always"), "always missing: {symbols:?}");
    }

    #[test]
    fn tokio_test_attribute_is_skipped() {
        // `#[tokio::test]` (scoped path ending in `test`) is also a test marker.
        let src = r#"
pub fn keep_me() -> i32 { 1 }

#[tokio::test]
async fn async_test() {}
"#;
        let chunks = chunk_source(Some("rust"), src, 100).unwrap();
        let symbols: Vec<_> = chunks.iter().filter_map(|c| c.symbol.as_deref()).collect();
        assert!(symbols.contains(&"keep_me"), "keep_me missing: {symbols:?}");
        assert!(!symbols.contains(&"async_test"), "async_test should be skipped: {symbols:?}");
    }
}

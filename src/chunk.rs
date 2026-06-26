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
    let root = tree.root_node();
    let kinds = def_kinds(lang);
    let bytes = source.as_bytes();

    let mut chunks = Vec::new();
    let mut cursor = root.walk();

    fn walk_tree(node: tree_sitter::Node, kinds: &[&str], bytes: &[u8], chunks: &mut Vec<Chunk>, lang: &str) {
        if kinds.contains(&node.kind()) {
            let text = node.utf8_text(bytes).unwrap_or("").to_string();
            let symbol = node.child_by_field_name("name")
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(|s| s.to_string());
            chunks.push(Chunk {
                text,
                start_line: node.start_position().row + 1,
                end_line: node.end_position().row + 1,
                language: lang.to_string(),
                symbol,
            });
            // Do NOT recurse into children of a matched node — nested definitions
            // (e.g. methods inside impl/class) are part of this chunk and must not
            // be emitted again as standalone chunks.
            return;
        }

        let mut child_cursor = node.walk();
        for child in node.children(&mut child_cursor) {
            walk_tree(child, kinds, bytes, chunks, lang);
        }
    }

    for node in root.children(&mut cursor) {
        walk_tree(node, kinds, bytes, &mut chunks, lang);
    }

    Ok(chunks)
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
}

//! Freshness: walk repo (gitignore-aware), hash files, compute delta vs stored hashes.
use crate::error::{Error, Result};
use ignore::overrides::OverrideBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Built-in glob patterns for test/fixture files that are skipped during
/// indexing unless `index_tests` is set. `**/` prefixes so they match in
/// workspace members and nested packages, not just the repo root. `examples/`
/// is deliberately absent — examples are real, runnable code worth searching.
pub const DEFAULT_TEST_EXCLUDES: &[&str] = &[
    "**/tests/**",      // Rust integration tests + fixtures live here
    "**/benches/**",    // Rust benchmarks
    "**/__tests__/**",  // JS/TS
    "**/*.test.*",      // JS/TS
    "**/*.spec.*",      // JS/TS
    "**/*_test.*",      // Python *_test.py, Go *_test.go, ...
    "**/test_*.py",     // pytest
    "**/conftest.py",   // pytest fixtures
];

/// Resolve the effective exclude patterns: the built-in test excludes (unless
/// `index_tests`) followed by the user's extra patterns.
pub fn resolve_excludes(user_exclude: &[String], index_tests: bool) -> Vec<String> {
    let mut out: Vec<String> = if index_tests {
        Vec::new()
    } else {
        DEFAULT_TEST_EXCLUDES.iter().map(std::string::ToString::to_string).collect()
    };
    out.extend(user_exclude.iter().cloned());
    out
}

#[derive(Debug, Clone)]
pub struct FileState { pub path: String, pub hash: String }

#[derive(Debug, Clone, Default)]
pub struct Delta { pub changed: Vec<String>, pub deleted: Vec<String> }

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

pub fn scan(repo_root: &Path, excludes: &[String]) -> Result<Vec<FileState>> {
    // Apply excludes as ignore-style overrides (each as a `!glob`), so the
    // walker skips matching files in the same pass as .gitignore. With only
    // negated globs and no whitelist globs, every non-matching file is kept.
    let mut ob = OverrideBuilder::new(repo_root);
    for pat in excludes {
        ob.add(&format!("!{pat}"))
            .map_err(|e| Error::Config(format!("invalid exclude pattern {pat:?}: {e}")))?;
    }
    let overrides = ob.build().map_err(|e| Error::Config(format!("building excludes: {e}")))?;

    let mut out = Vec::new();
    for entry in ignore::WalkBuilder::new(repo_root).overrides(overrides).build() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) { continue; }
        let path = entry.path();
        let Ok(bytes) = std::fs::read(path) else { continue };
        out.push(FileState { path: rel(repo_root, path), hash: blake3::hash(&bytes).to_hex().to_string() });
    }
    Ok(out)
}

pub fn diff<S: std::hash::BuildHasher>(current: &[FileState], stored: &HashMap<String, String, S>) -> Delta {
    let mut delta = Delta::default();
    let current_paths: std::collections::HashSet<&str> = current.iter().map(|s| s.path.as_str()).collect();
    for s in current {
        match stored.get(&s.path) {
            Some(h) if *h == s.hash => {}
            _ => delta.changed.push(s.path.clone()),
        }
    }
    for path in stored.keys() {
        if !current_paths.contains(path.as_str()) { delta.deleted.push(path.clone()); }
    }
    delta.changed.sort();
    delta.deleted.sort();
    delta
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn scan_respects_gitignore_and_hashes() {
        let d = tempdir().unwrap();
        let git_dir = d.path().join(".git");
        fs::create_dir(&git_dir).unwrap();
        fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(git_dir.join("config"), "[core]\n\trepositoryformatversion = 0\n").unwrap();
        fs::write(d.path().join("keep.rs"), "fn a(){}").unwrap();
        fs::write(d.path().join(".gitignore"), "ignored.rs\n").unwrap();
        fs::write(d.path().join("ignored.rs"), "fn b(){}").unwrap();
        let states = scan(d.path(), &[]).unwrap();
        let paths: Vec<_> = states.iter().map(|s| s.path.clone()).collect();
        assert!(paths.contains(&"keep.rs".to_string()));
        assert!(!paths.iter().any(|p| p == "ignored.rs"));
        assert!(states.iter().all(|s| !s.hash.is_empty()));
        // Regression guard: .git/ internals must never appear in the scan results.
        assert!(
            !paths.iter().any(|p| p.starts_with(".git/")),
            "scan must not emit git-internal files; found: {:?}",
            paths.iter().filter(|p| p.starts_with(".git/")).collect::<Vec<_>>()
        );
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    fn paths_in(root: &Path, excludes: &[String]) -> Vec<String> {
        scan(root, excludes).unwrap().into_iter().map(|s| s.path).collect()
    }

    #[test]
    fn scan_excludes_known_test_files_by_default() {
        let d = tempdir().unwrap();
        write(d.path(), "src/lib.rs", "fn a(){}");
        write(d.path(), "tests/integration.rs", "fn t(){}");
        write(d.path(), "benches/bench.rs", "fn b(){}");
        write(d.path(), "pkg/test_foo.py", "def t(): pass");
        write(d.path(), "web/app.test.ts", "test('x',()=>{})");
        write(d.path(), "web/app.spec.ts", "test('y',()=>{})");
        write(d.path(), "web/__tests__/helper.ts", "export {}");
        write(d.path(), "examples/demo.rs", "fn main(){}"); // real code: kept

        let excludes = resolve_excludes(&[], false);
        let paths = paths_in(d.path(), &excludes);

        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&"examples/demo.rs".to_string()), "examples are real code, kept");
        for excluded in [
            "tests/integration.rs", "benches/bench.rs", "pkg/test_foo.py",
            "web/app.test.ts", "web/app.spec.ts", "web/__tests__/helper.ts",
        ] {
            assert!(!paths.iter().any(|p| p == excluded), "{excluded} should be excluded; got {paths:?}");
        }
    }

    #[test]
    fn user_exclude_extends_builtins() {
        let d = tempdir().unwrap();
        write(d.path(), "src/lib.rs", "fn a(){}");
        write(d.path(), "vendor/dep.rs", "fn v(){}");

        let excludes = resolve_excludes(&["vendor/**".to_string()], false);
        let paths = paths_in(d.path(), &excludes);

        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(!paths.iter().any(|p| p == "vendor/dep.rs"), "user exclude must apply");
    }

    #[test]
    fn index_tests_true_keeps_test_files() {
        let d = tempdir().unwrap();
        write(d.path(), "src/lib.rs", "fn a(){}");
        write(d.path(), "tests/integration.rs", "fn t(){}");

        let excludes = resolve_excludes(&[], true);
        let paths = paths_in(d.path(), &excludes);

        assert!(paths.contains(&"tests/integration.rs".to_string()), "index_tests=true re-includes tests");
    }

    #[test]
    fn diff_detects_changed_new_deleted() {
        let current = vec![
            FileState { path: "a.rs".into(), hash: "h1".into() },
            FileState { path: "b.rs".into(), hash: "NEW".into() },
            FileState { path: "c.rs".into(), hash: "h3".into() },
        ];
        let mut stored = HashMap::new();
        stored.insert("a.rs".to_string(), "h1".to_string());
        stored.insert("b.rs".to_string(), "h2".to_string());
        stored.insert("d.rs".to_string(), "h4".to_string());
        let delta = diff(&current, &stored);
        assert!(delta.changed.contains(&"b.rs".to_string()));
        assert!(delta.changed.contains(&"c.rs".to_string()));
        assert!(!delta.changed.contains(&"a.rs".to_string()));
        assert_eq!(delta.deleted, vec!["d.rs".to_string()]);
    }
}

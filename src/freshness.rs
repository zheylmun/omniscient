//! Freshness: walk repo (gitignore-aware), hash files, compute delta vs stored hashes.
use crate::error::{Error, Result};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use ignore::overrides::OverrideBuilder;
use std::collections::HashMap;
use std::path::Path;

/// Built-in glob patterns that are *always* skipped, regardless of `index_tests`:
/// dependency lock files. They are large, machine-generated, and almost pure noise
/// for semantic search. `**/` prefixes so they match in workspace members and nested
/// packages, not just the repo root. Note `go.sum` (checksums) is excluded while
/// `go.mod` is kept — the latter is meaningful, hand-edited dependency declaration.
pub const DEFAULT_EXCLUDES: &[&str] = &[
    "**/Cargo.lock",          // Rust
    "**/package-lock.json",   // npm
    "**/npm-shrinkwrap.json", // npm
    "**/yarn.lock",           // Yarn
    "**/pnpm-lock.yaml",      // pnpm
    "**/bun.lock",            // Bun (text)
    "**/bun.lockb",           // Bun (binary)
    "**/poetry.lock",         // Poetry
    "**/Pipfile.lock",        // pipenv
    "**/uv.lock",             // uv
    "**/Gemfile.lock",        // Bundler
    "**/composer.lock",       // Composer
    "**/go.sum",              // Go module checksums
    "**/mix.lock",            // Elixir
    "**/flake.lock",          // Nix
    "**/packages.lock.json",  // NuGet
];

/// Built-in glob patterns for test/fixture files that are skipped during
/// indexing unless `index_tests` is set. `**/` prefixes so they match in
/// workspace members and nested packages, not just the repo root. `examples/`
/// is deliberately absent — examples are real, runnable code worth searching.
pub const DEFAULT_TEST_EXCLUDES: &[&str] = &[
    "**/tests/**",     // Rust integration tests + fixtures live here
    "**/benches/**",   // Rust benchmarks
    "**/__tests__/**", // JS/TS
    "**/*.test.*",     // JS/TS
    "**/*.spec.*",     // JS/TS
    "**/*_test.*",     // Python *_test.py, Go *_test.go, ...
    "**/test_*.py",    // pytest
    "**/conftest.py",  // pytest fixtures
];

/// Resolve the effective exclude patterns: the always-on built-in excludes (lock
/// files), then the test excludes unless `index_tests`, then the user's extras.
pub fn resolve_excludes(user_exclude: &[String], index_tests: bool) -> Vec<String> {
    let mut out: Vec<String> = DEFAULT_EXCLUDES
        .iter()
        .map(std::string::ToString::to_string)
        .collect();
    if !index_tests {
        out.extend(
            DEFAULT_TEST_EXCLUDES
                .iter()
                .map(std::string::ToString::to_string),
        );
    }
    out.extend(user_exclude.iter().cloned());
    out
}

/// Build a matcher over `excludes` so the same patterns `scan` applies at index
/// time can also be enforced at read time (search). Keeping both paths driven by
/// one `resolve_excludes` list means they cannot drift apart. Patterns use
/// gitignore glob semantics, matching how `scan` interprets them.
pub fn exclude_matcher(repo_root: &Path, excludes: &[String]) -> Result<Gitignore> {
    let mut b = GitignoreBuilder::new(repo_root);
    for pat in excludes {
        b.add_line(None, pat)
            .map_err(|e| Error::Config(format!("invalid exclude pattern {pat:?}: {e}")))?;
    }
    b.build()
        .map_err(|e| Error::Config(format!("building exclude matcher: {e}")))
}

/// True if `rel_path` (repo-relative, forward-slashed — as stored on `Hit`s) is
/// excluded by `matcher`. Paths are always files here, never directories.
pub fn is_excluded(matcher: &Gitignore, rel_path: &str) -> bool {
    matcher.matched(rel_path, false).is_ignore()
}

#[derive(Debug, Clone)]
pub struct FileState {
    pub path: String,
    pub hash: String,
}

#[derive(Debug, Clone, Default)]
pub struct Delta {
    pub changed: Vec<String>,
    pub deleted: Vec<String>,
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root)
        .unwrap_or(p)
        .to_string_lossy()
        .replace('\\', "/")
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
    let overrides = ob
        .build()
        .map_err(|e| Error::Config(format!("building excludes: {e}")))?;

    let mut out = Vec::new();
    for entry in ignore::WalkBuilder::new(repo_root)
        .overrides(overrides)
        .build()
    {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        out.push(FileState {
            path: rel(repo_root, path),
            hash: blake3::hash(&bytes).to_hex().to_string(),
        });
    }
    Ok(out)
}

pub fn diff<S: std::hash::BuildHasher>(
    current: &[FileState],
    stored: &HashMap<String, String, S>,
) -> Delta {
    let mut delta = Delta::default();
    let current_paths: std::collections::HashSet<&str> =
        current.iter().map(|s| s.path.as_str()).collect();
    for s in current {
        match stored.get(&s.path) {
            Some(h) if *h == s.hash => {}
            _ => delta.changed.push(s.path.clone()),
        }
    }
    for path in stored.keys() {
        if !current_paths.contains(path.as_str()) {
            delta.deleted.push(path.clone());
        }
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
        fs::write(
            git_dir.join("config"),
            "[core]\n\trepositoryformatversion = 0\n",
        )
        .unwrap();
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
            paths
                .iter()
                .filter(|p| p.starts_with(".git/"))
                .collect::<Vec<_>>()
        );
    }

    fn write(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    fn paths_in(root: &Path, excludes: &[String]) -> Vec<String> {
        scan(root, excludes)
            .unwrap()
            .into_iter()
            .map(|s| s.path)
            .collect()
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
        assert!(
            paths.contains(&"examples/demo.rs".to_string()),
            "examples are real code, kept"
        );
        for excluded in [
            "tests/integration.rs",
            "benches/bench.rs",
            "pkg/test_foo.py",
            "web/app.test.ts",
            "web/app.spec.ts",
            "web/__tests__/helper.ts",
        ] {
            assert!(
                !paths.iter().any(|p| p == excluded),
                "{excluded} should be excluded; got {paths:?}"
            );
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
        assert!(
            !paths.iter().any(|p| p == "vendor/dep.rs"),
            "user exclude must apply"
        );
    }

    #[test]
    fn index_tests_true_keeps_test_files() {
        let d = tempdir().unwrap();
        write(d.path(), "src/lib.rs", "fn a(){}");
        write(d.path(), "tests/integration.rs", "fn t(){}");

        let excludes = resolve_excludes(&[], true);
        let paths = paths_in(d.path(), &excludes);

        assert!(
            paths.contains(&"tests/integration.rs".to_string()),
            "index_tests=true re-includes tests"
        );
    }

    #[test]
    fn scan_always_excludes_lock_files() {
        let d = tempdir().unwrap();
        write(d.path(), "src/lib.rs", "fn a(){}");
        write(d.path(), "Cargo.lock", "# generated\n");
        write(d.path(), "web/package-lock.json", "{}");
        write(d.path(), "web/yarn.lock", "# yarn\n");
        write(d.path(), "py/poetry.lock", "# poetry\n");
        write(d.path(), "go.sum", "mod h1:...\n");
        write(d.path(), "go.mod", "module x\n"); // meaningful: kept

        // Lock files are not gated by index_tests; assert both settings exclude them.
        for index_tests in [false, true] {
            let paths = paths_in(d.path(), &resolve_excludes(&[], index_tests));
            assert!(paths.contains(&"src/lib.rs".to_string()));
            assert!(
                paths.contains(&"go.mod".to_string()),
                "go.mod is meaningful, kept (index_tests={index_tests})"
            );
            for locked in [
                "Cargo.lock",
                "web/package-lock.json",
                "web/yarn.lock",
                "py/poetry.lock",
                "go.sum",
            ] {
                assert!(
                    !paths.iter().any(|p| p == locked),
                    "{locked} must be excluded (index_tests={index_tests}); got {paths:?}"
                );
            }
        }
    }

    #[test]
    fn exclude_matcher_agrees_with_scan_patterns() {
        // The read-time matcher must exclude exactly what scan excludes, or search
        // and indexing would disagree. Drive both from the same resolve_excludes set.
        let excludes = resolve_excludes(&["vendor/**".to_string()], false);
        let m = exclude_matcher(Path::new("/repo"), &excludes).unwrap();
        for excluded in [
            "Cargo.lock",
            "web/package-lock.json",
            "go.sum",
            "tests/it.rs",
            "a/__tests__/x.ts",
            "web/app.test.ts",
            "vendor/dep.rs",
        ] {
            assert!(is_excluded(&m, excluded), "{excluded} should be excluded");
        }
        for kept in ["src/lib.rs", "go.mod", "examples/demo.rs"] {
            assert!(!is_excluded(&m, kept), "{kept} should be kept");
        }
    }

    #[test]
    fn diff_detects_changed_new_deleted() {
        let current = vec![
            FileState {
                path: "a.rs".into(),
                hash: "h1".into(),
            },
            FileState {
                path: "b.rs".into(),
                hash: "NEW".into(),
            },
            FileState {
                path: "c.rs".into(),
                hash: "h3".into(),
            },
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

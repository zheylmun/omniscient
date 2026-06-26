//! Freshness: walk repo (gitignore-aware), hash files, compute delta vs stored hashes.
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct FileState { pub path: String, pub hash: String }

#[derive(Debug, Clone, Default)]
pub struct Delta { pub changed: Vec<String>, pub deleted: Vec<String> }

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

pub fn scan(repo_root: &Path) -> Result<Vec<FileState>> {
    let mut out = Vec::new();
    for entry in ignore::WalkBuilder::new(repo_root).build() {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) { continue; }
        let path = entry.path();
        let bytes = match std::fs::read(path) { Ok(b) => b, Err(_) => continue };
        out.push(FileState { path: rel(repo_root, path), hash: blake3::hash(&bytes).to_hex().to_string() });
    }
    Ok(out)
}

pub fn diff(current: &[FileState], stored: &HashMap<String, String>) -> Delta {
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
        let states = scan(d.path()).unwrap();
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

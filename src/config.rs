//! Configuration: omniscient.toml -> Config, with defaults.
use crate::embed::BatchLimits;
use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig {
    pub base_url: String,
    pub model: String,
    pub max_batch_chunks: usize,
    pub max_batch_chars: usize,
}
impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
            model: "qwen3-embedding-4b".into(),
            max_batch_chunks: 64,
            max_batch_chars: 32000,
        }
    }
}
impl EmbedderConfig {
    pub fn batch_limits(&self) -> BatchLimits {
        BatchLimits { max_chunks: self.max_batch_chunks, max_chars: self.max_batch_chars }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig { pub default_k: usize, pub token_budget: usize }
impl Default for SearchConfig {
    fn default() -> Self { Self { default_k: 8, token_budget: 4000 } }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WatchConfig { pub enabled: bool, pub debounce_ms: u64 }
impl Default for WatchConfig {
    fn default() -> Self { Self { enabled: true, debounce_ms: 200 } }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(skip)]
    pub repo_root: PathBuf,
    pub embedder: EmbedderConfig,
    pub search: SearchConfig,
    pub watch: WatchConfig,
    pub languages: Vec<String>,
    pub strip_comments: bool,
    /// Extra glob patterns to skip when indexing, unioned with the built-in
    /// test/fixture excludes (see `freshness`). Matched against repo-relative paths.
    pub exclude: Vec<String>,
    /// When true, the built-in test/fixture excludes are not applied (so test
    /// files are indexed). The `exclude` list still applies. Defaults to false.
    pub index_tests: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            repo_root: PathBuf::new(),
            embedder: EmbedderConfig::default(),
            search: SearchConfig::default(),
            watch: WatchConfig::default(),
            languages: vec!["rust".into(), "python".into(), "typescript".into()],
            strip_comments: true,
            exclude: Vec::new(),
            index_tests: false,
        }
    }
}

impl Config {
    pub fn default_for(repo_root: PathBuf) -> Config { Config { repo_root, ..Default::default() } }

    pub fn from_toml_str(s: &str, repo_root: PathBuf) -> Result<Config> {
        let mut c: Config = toml::from_str(s).map_err(|e| Error::Config(e.to_string()))?;
        c.repo_root = repo_root;
        Ok(c)
    }

    pub fn load(path: Option<&Path>, repo_root: PathBuf) -> Result<Config> {
        let candidate = path.map(PathBuf::from).unwrap_or_else(|| repo_root.join("omniscient.toml"));
        match std::fs::read_to_string(&candidate) {
            Ok(s) => Config::from_toml_str(&s, repo_root),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default_for(repo_root)),
            Err(e) => Err(Error::Config(format!("reading {}: {e}", candidate.display()))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn defaults_are_sane() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
        assert_eq!(c.embedder.base_url, "http://localhost:8080");
        assert_eq!(c.search.default_k, 8);
        assert!(c.search.token_budget > 0);
        assert_eq!(c.languages, vec!["rust","python","typescript"]);
    }

    #[test]
    fn parses_partial_toml_over_defaults() {
        let toml = r#"
            languages = ["rust"]
            [embedder]
            model = "bge-code"
            [search]
            default_k = 5
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.model, "bge-code");
        assert_eq!(c.embedder.base_url, "http://localhost:8080"); // defaulted
        assert_eq!(c.search.default_k, 5);
        assert_eq!(c.languages, vec!["rust".to_string()]);
    }

    #[test]
    fn exclude_and_index_tests_default_and_parse() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert!(c.exclude.is_empty(), "exclude defaults to empty");
        assert!(!c.index_tests, "index_tests defaults to false");

        let toml = r#"
            index_tests = true
            exclude = ["vendor/**", "**/*.gen.rs"]
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert!(c.index_tests);
        assert_eq!(c.exclude, vec!["vendor/**".to_string(), "**/*.gen.rs".to_string()]);
    }

    #[test]
    fn missing_file_yields_defaults() {
        let c = Config::load(Some(Path::new("/nonexistent.toml")), PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }

    #[test]
    fn unreadable_config_surfaces_error_not_defaults() {
        // Pointing the config path at a directory makes read_to_string fail with a
        // non-NotFound error; that must surface, not silently fall back to defaults.
        let dir = tempfile::tempdir().unwrap();
        let res = Config::load(Some(dir.path()), PathBuf::from("/repo"));
        assert!(res.is_err(), "a non-NotFound IO error must not yield defaults");
    }

    #[test]
    fn watch_config_defaults_and_parse() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert!(c.watch.enabled, "watching defaults to on");
        assert_eq!(c.watch.debounce_ms, 200);

        let toml = r#"
            [watch]
            enabled = false
            debounce_ms = 500
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert!(!c.watch.enabled);
        assert_eq!(c.watch.debounce_ms, 500);
    }

    #[test]
    fn embedder_batch_defaults() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert_eq!(c.embedder.max_batch_chunks, 64);
        assert_eq!(c.embedder.max_batch_chars, 32000);
        let limits = c.embedder.batch_limits();
        assert_eq!(limits.max_chunks, 64);
        assert_eq!(limits.max_chars, 32000);
    }

    #[test]
    fn embedder_batch_overrides_parse() {
        let toml = r#"
            [embedder]
            max_batch_chunks = 16
            max_batch_chars = 8000
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.max_batch_chunks, 16);
        assert_eq!(c.embedder.max_batch_chars, 8000);
        // unspecified embedder fields keep their defaults
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }
}

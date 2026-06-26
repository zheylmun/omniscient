//! Configuration: omniscient.toml -> Config, with defaults.
use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig { pub base_url: String, pub model: String }
impl Default for EmbedderConfig {
    fn default() -> Self {
        Self { base_url: "http://localhost:8080".into(), model: "qwen3-embedding-4b".into() }
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
pub struct Config {
    #[serde(skip)]
    pub repo_root: PathBuf,
    pub embedder: EmbedderConfig,
    pub search: SearchConfig,
    pub languages: Vec<String>,
    pub strip_comments: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            repo_root: PathBuf::new(),
            embedder: EmbedderConfig::default(),
            search: SearchConfig::default(),
            languages: vec!["rust".into(), "python".into(), "typescript".into()],
            strip_comments: true,
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
            Err(_) => Ok(Config::default_for(repo_root)),
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
    fn missing_file_yields_defaults() {
        let c = Config::load(Some(Path::new("/nonexistent.toml")), PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }
}

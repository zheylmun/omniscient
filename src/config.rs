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
    pub max_batch_bytes: usize,
    /// When true and `base_url` is unreachable at startup, omniscient launches a
    /// local llama.cpp server (`llama serve …`) itself and waits for it to come
    /// up, instead of erroring. Off by default — an already-running endpoint is
    /// always used as-is and never spawned over.
    pub auto_start: bool,
    /// The llama.cpp CLI binary to spawn for `auto_start` (the unified `llama`
    /// command; omniscient always passes the `serve` subcommand). Resolved on
    /// PATH unless an absolute path is given.
    pub llama_bin: String,
    /// The `-hf` argument passed to `llama serve`: a Hugging Face GGUF repo with
    /// an optional `:QUANT` tag. The GGUF is downloaded on first run.
    pub hf_repo: String,
    /// The `--pooling` strategy for the spawned server. Qwen3-Embedding (a
    /// decoder LLM) needs `last`; BERT-family embedders need `mean`.
    pub pooling: String,
    /// How long to wait (seconds) for an `auto_start`ed server to become ready
    /// before giving up. Generous by default because the first run downloads the
    /// model.
    pub auto_start_timeout_secs: u64,
}
impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
            model: "qwen3-embedding-4b".into(),
            max_batch_chunks: 64,
            max_batch_bytes: 32000,
            auto_start: false,
            llama_bin: "llama".into(),
            hf_repo: "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M".into(),
            pooling: "last".into(),
            auto_start_timeout_secs: 600,
        }
    }
}
impl EmbedderConfig {
    /// Batch limits for embedding. A `0` in either knob is clamped to 1 so a
    /// fat-fingered config degrades to one-chunk-per-request rather than being
    /// rejected — and never produces an empty/looping batch.
    pub fn batch_limits(&self) -> BatchLimits {
        BatchLimits {
            max_chunks: self.max_batch_chunks.max(1),
            max_bytes: self.max_batch_bytes.max(1),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Upper bound on candidates fetched from the index and results returned. A
    /// safety ceiling, not a target — relevance-shape selection (see
    /// `relevance_ratio`) usually returns fewer. The MCP `k` argument overrides
    /// it per call.
    pub max_results: usize,
    /// Keep every result scoring at least this fraction of the top result's
    /// cosine similarity, so the result count tracks the *shape* of the score
    /// distribution instead of a fixed k. 0.75 = "within 75% of the best match".
    /// Clamped to `[0.0, 1.0]`; the best match is always returned.
    pub relevance_ratio: f32,
    pub token_budget: usize,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 25,
            relevance_ratio: 0.75,
            token_budget: 4000,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WatchConfig {
    pub enabled: bool,
    pub debounce_ms: u64,
}
impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            debounce_ms: 200,
        }
    }
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
    pub fn default_for(repo_root: PathBuf) -> Config {
        Config {
            repo_root,
            ..Default::default()
        }
    }

    pub fn from_toml_str(s: &str, repo_root: PathBuf) -> Result<Config> {
        let mut c: Config = toml::from_str(s).map_err(|e| Error::Config(e.to_string()))?;
        c.repo_root = repo_root;
        Ok(c)
    }

    pub fn load(path: Option<&Path>, repo_root: PathBuf) -> Result<Config> {
        let candidate = path.map_or_else(|| repo_root.join("omniscient.toml"), PathBuf::from);
        match std::fs::read_to_string(&candidate) {
            Ok(s) => Config::from_toml_str(&s, repo_root),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Ok(Config::default_for(repo_root))
            }
            Err(e) => Err(Error::Config(format!(
                "reading {}: {e}",
                candidate.display()
            ))),
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
        assert_eq!(c.search.max_results, 25);
        assert!((c.search.relevance_ratio - 0.75).abs() < 1e-6);
        assert!(c.search.token_budget > 0);
        assert_eq!(c.languages, vec!["rust", "python", "typescript"]);
    }

    #[test]
    fn parses_partial_toml_over_defaults() {
        let toml = r#"
            languages = ["rust"]
            [embedder]
            model = "bge-code"
            [search]
            max_results = 5
            relevance_ratio = 0.5
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.model, "bge-code");
        assert_eq!(c.embedder.base_url, "http://localhost:8080"); // defaulted
        assert_eq!(c.search.max_results, 5);
        assert!((c.search.relevance_ratio - 0.5).abs() < 1e-6);
        assert_eq!(c.search.token_budget, 4000); // defaulted
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
        assert_eq!(
            c.exclude,
            vec!["vendor/**".to_string(), "**/*.gen.rs".to_string()]
        );
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
        assert!(
            res.is_err(),
            "a non-NotFound IO error must not yield defaults"
        );
    }

    #[test]
    fn watch_config_defaults_and_parse() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert!(c.watch.enabled, "watching defaults to on");
        assert_eq!(c.watch.debounce_ms, 200);

        let toml = r"
            [watch]
            enabled = false
            debounce_ms = 500
        ";
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert!(!c.watch.enabled);
        assert_eq!(c.watch.debounce_ms, 500);
    }

    #[test]
    fn embedder_batch_defaults() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert_eq!(c.embedder.max_batch_chunks, 64);
        assert_eq!(c.embedder.max_batch_bytes, 32000);
        let limits = c.embedder.batch_limits();
        assert_eq!(limits.max_chunks, 64);
        assert_eq!(limits.max_bytes, 32000);
    }

    #[test]
    fn embedder_batch_overrides_parse() {
        let toml = r"
            [embedder]
            max_batch_chunks = 16
            max_batch_bytes = 8000
        ";
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.max_batch_chunks, 16);
        assert_eq!(c.embedder.max_batch_bytes, 8000);
        // unspecified embedder fields keep their defaults
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }

    #[test]
    fn auto_start_defaults_and_parse() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert!(!c.embedder.auto_start, "auto_start defaults to off");
        assert_eq!(c.embedder.llama_bin, "llama");
        assert_eq!(c.embedder.hf_repo, "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M");
        assert_eq!(c.embedder.pooling, "last");
        assert_eq!(c.embedder.auto_start_timeout_secs, 600);

        let toml = r#"
            [embedder]
            auto_start = true
            llama_bin = "/opt/llama/llama"
            hf_repo = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
            pooling = "mean"
            auto_start_timeout_secs = 120
        "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert!(c.embedder.auto_start);
        assert_eq!(c.embedder.llama_bin, "/opt/llama/llama");
        assert_eq!(c.embedder.hf_repo, "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0");
        assert_eq!(c.embedder.pooling, "mean");
        assert_eq!(c.embedder.auto_start_timeout_secs, 120);
        // unspecified embedder fields keep their defaults
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }

    #[test]
    fn batch_limits_clamp_zero_to_one() {
        let toml = r"
            [embedder]
            max_batch_chunks = 0
            max_batch_bytes = 0
        ";
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        // raw fields keep the user's value; batch_limits() clamps to a safe minimum
        assert_eq!(c.embedder.max_batch_chunks, 0);
        assert_eq!(c.embedder.max_batch_bytes, 0);
        let limits = c.embedder.batch_limits();
        assert_eq!(limits.max_chunks, 1);
        assert_eq!(limits.max_bytes, 1);
    }
}

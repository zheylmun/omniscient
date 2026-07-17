//! Configuration cascade: defaults → global → repo → `--config`.
//!
//! Each level is partial — fields not set at a higher level inherit from the
//! lower one. Vector fields (`languages`, `exclude`) are **replaced** entirely
//! by the higher level (not merged).
use crate::embed::BatchLimits;
use crate::error::{Error, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Where a config level came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    /// `~/.config/omniscient/omniscient.toml`
    Global { path: PathBuf },
    /// `<repo_root>/omniscient.toml`
    Repo { path: PathBuf },
    /// `--config <path>`
    Cli { path: PathBuf },
}

/// One resolved level in the config cascade.
#[derive(Debug, Clone)]
pub struct ConfigLevel {
    /// Where this level was loaded from.
    pub source: ConfigSource,
    /// The raw parsed config at this level (before merge into the final result).
    pub config: Config,
}

/// Resolve the global config path.
///
/// Returns `dirs::config_dir().join("omniscient/omniscient.toml")`. On macOS
/// this is `~/Library/Application Support/omniscient/omniscient.toml`; on
/// Linux `~/.config/omniscient/omniscient.toml`.
pub fn global_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("omniscient/omniscient.toml"))
}

/// Merge `overlay` onto `base`. For all fields the overlay wins when it
/// differs from its default value. Vector fields (`languages`, `exclude`) are
/// replaced entirely (not merged). `repo_root` is preserved from `base`.
pub fn merge(base: Config, overlay: Config) -> Config {
    let def = Config::default();
    Config {
        repo_root: base.repo_root,
        embedder: merge_embedder(base.embedder, overlay.embedder),
        search: merge_search(&base.search, &overlay.search),
        watch: merge_watch(&base.watch, &overlay.watch),
        languages: if overlay.languages == def.languages {
            base.languages
        } else {
            overlay.languages
        },
        strip_banner_comments: if overlay.strip_banner_comments == def.strip_banner_comments {
            base.strip_banner_comments
        } else {
            overlay.strip_banner_comments
        },
        exclude: if overlay.exclude == def.exclude {
            base.exclude
        } else {
            overlay.exclude
        },
        index_tests: if overlay.index_tests == def.index_tests {
            base.index_tests
        } else {
            overlay.index_tests
        },
        cascade: base.cascade, // cascade metadata stays on the base (accumulator)
    }
}

fn merge_embedder(base: EmbedderConfig, overlay: EmbedderConfig) -> EmbedderConfig {
    let def = EmbedderConfig::default();
    EmbedderConfig {
        base_url: if overlay.base_url == def.base_url {
            base.base_url
        } else {
            overlay.base_url
        },
        model: if overlay.model == def.model {
            base.model
        } else {
            overlay.model
        },
        max_batch_chunks: if overlay.max_batch_chunks == def.max_batch_chunks {
            base.max_batch_chunks
        } else {
            overlay.max_batch_chunks
        },
        max_batch_bytes: if overlay.max_batch_bytes == def.max_batch_bytes {
            base.max_batch_bytes
        } else {
            overlay.max_batch_bytes
        },
        embed_concurrency: if overlay.embed_concurrency == def.embed_concurrency {
            base.embed_concurrency
        } else {
            overlay.embed_concurrency
        },
        auto_start: if overlay.auto_start == def.auto_start {
            base.auto_start
        } else {
            overlay.auto_start
        },
        llama_bin: if overlay.llama_bin == def.llama_bin {
            base.llama_bin
        } else {
            overlay.llama_bin
        },
        hf_repo: if overlay.hf_repo == def.hf_repo {
            base.hf_repo
        } else {
            overlay.hf_repo
        },
        pooling: if overlay.pooling == def.pooling {
            base.pooling
        } else {
            overlay.pooling
        },
        auto_start_timeout_secs: if overlay.auto_start_timeout_secs == def.auto_start_timeout_secs {
            base.auto_start_timeout_secs
        } else {
            overlay.auto_start_timeout_secs
        },
        request_timeout_secs: if overlay.request_timeout_secs == def.request_timeout_secs {
            base.request_timeout_secs
        } else {
            overlay.request_timeout_secs
        },
        api_key: if overlay.api_key == def.api_key {
            base.api_key
        } else {
            overlay.api_key
        },
    }
}

fn merge_search(base: &SearchConfig, overlay: &SearchConfig) -> SearchConfig {
    let def = SearchConfig::default();
    SearchConfig {
        max_results: if overlay.max_results == def.max_results {
            base.max_results
        } else {
            overlay.max_results
        },
        relevance_ratio: if (overlay.relevance_ratio - def.relevance_ratio).abs() <= 1e-6 {
            base.relevance_ratio
        } else {
            overlay.relevance_ratio
        },
        token_budget: if overlay.token_budget == def.token_budget {
            base.token_budget
        } else {
            overlay.token_budget
        },
        search_timeout_secs: if overlay.search_timeout_secs == def.search_timeout_secs {
            base.search_timeout_secs
        } else {
            overlay.search_timeout_secs
        },
    }
}

fn merge_watch(base: &WatchConfig, overlay: &WatchConfig) -> WatchConfig {
    let def = WatchConfig::default();
    WatchConfig {
        enabled: if overlay.enabled == def.enabled {
            base.enabled
        } else {
            overlay.enabled
        },
        debounce_ms: if overlay.debounce_ms == def.debounce_ms {
            base.debounce_ms
        } else {
            overlay.debounce_ms
        },
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbedderConfig {
    pub base_url: String,
    pub model: String,
    pub max_batch_chunks: usize,
    pub max_batch_bytes: usize,
    /// Maximum number of concurrent embedding requests during reconcile.
    /// Each request covers one file's chunks (batched by `max_batch_chunks/bytes`).
    /// Higher values saturate the endpoint faster on large initial indexes.
    pub embed_concurrency: usize,
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
    /// Timeout in seconds for each HTTP request to the embeddings endpoint.
    /// Guards against a hanging llama.cpp server blocking the MCP tool call.
    pub request_timeout_secs: u64,
    /// Bearer token for an authenticated embeddings endpoint (a llama.cpp server
    /// started with `--api-key`, or an OpenAI-compatible router). Sent as
    /// `Authorization: Bearer <key>`. `None`/absent → no auth header.
    ///
    /// A whole-string `${VAR}` or `$VAR` value is expanded from the environment
    /// at connect time (errors if the var is unset), keeping the secret out of
    /// the config file. Any other value is used literally.
    pub api_key: Option<String>,
}
impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
            model: "qwen3-embedding-4b".into(),
            max_batch_chunks: 64,
            max_batch_bytes: 32000,
            embed_concurrency: 4,
            auto_start: false,
            llama_bin: "llama".into(),
            hf_repo: "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M".into(),
            pooling: "last".into(),
            auto_start_timeout_secs: 600,
            request_timeout_secs: 30,
            api_key: None,
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

    /// Resolve `api_key` into the literal bearer token to send (see `resolve_key`).
    /// Called once on the connect path; the expanded secret is never stored.
    pub fn resolved_api_key(&self) -> Result<Option<String>> {
        resolve_key(self.api_key.as_deref(), |n| std::env::var(n).ok())
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
    /// Timeout in seconds for the entire `search` operation (`ensure_fresh` + embed
    /// + index query + distill). Guards against any step hanging indefinitely.
    pub search_timeout_secs: u64,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 25,
            relevance_ratio: 0.75,
            token_budget: 4000,
            search_timeout_secs: 60,
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

/// Parse the `api_key` form and produce the literal secret to send.
/// Pure: env access is injected via `lookup` so the logic is testable without
/// touching process env.
///
/// - `None` / blank: `Ok(None)` (no auth header)
/// - whole-string `${NAME}`/`$NAME` (NAME = `[A-Za-z_][A-Za-z0-9_]*`): `lookup(NAME)`;
///   `Err` if it returns `None`
/// - anything else: `Ok(Some(literal))`
fn resolve_key(
    raw: Option<&str>,
    lookup: impl Fn(&str) -> Option<String>,
) -> Result<Option<String>> {
    let Some(val) = raw.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if let Some(name) = env_var_ref(val) {
        let Some(resolved) = lookup(name) else {
            return Err(Error::Config(format!(
                "embedder.api_key references ${{{name}}} but that environment variable is not set"
            )));
        };
        let trimmed = resolved.trim();
        return Ok((!trimmed.is_empty()).then(|| trimmed.to_string()));
    }
    Ok(Some(val.to_string()))
}

/// If `s` is a whole-string environment reference (`${NAME}` or `$NAME` with a
/// valid shell-identifier `NAME`), return `NAME`; otherwise `None` (literal).
fn env_var_ref(s: &str) -> Option<&str> {
    let inner = s
        .strip_prefix("${")
        .and_then(|r| r.strip_suffix('}'))
        .or_else(|| s.strip_prefix('$'))?;
    let valid = inner
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    valid.then_some(inner)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(skip)]
    pub repo_root: PathBuf,
    /// Per-level snapshots of the cascade (global → repo → cli).
    /// Populated by `Config::load` so diagnostics can report overrides.
    #[serde(skip)]
    pub cascade: Vec<ConfigLevel>,
    pub embedder: EmbedderConfig,
    pub search: SearchConfig,
    pub watch: WatchConfig,
    pub languages: Vec<String>,
    /// Strip leading banner comments (license headers, module doc comments) from
    /// the code returned in search results. Does NOT strip inline or trailing
    /// comments — those carry implementation intent.
    #[serde(alias = "strip_comments")]
    pub strip_banner_comments: bool,
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
            cascade: Vec::new(),
            embedder: EmbedderConfig::default(),
            search: SearchConfig::default(),
            watch: WatchConfig::default(),
            languages: vec!["rust".into(), "python".into(), "typescript".into()],
            strip_banner_comments: true,
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

    /// Load with cascade: defaults → global → repo/`--config`.
    ///
    /// `path` is `Some` when `--config` is given; it replaces the repo config
    /// slot (not additive with it). Returns the merged config with
    /// `cascade` populated so diagnostics can report overrides.
    pub fn load(path: Option<&Path>, repo_root: PathBuf) -> Result<Config> {
        let mut base = Config::default_for(repo_root.clone());

        // Level 1: global config
        if let Some(ref global) = global_path().filter(|p| p.exists()) {
            let s = std::fs::read_to_string(global)
                .map_err(|e| Error::Config(format!("reading {}: {e}", global.display())))?;
            let parsed = Config::from_toml_str(&s, repo_root.clone())?;
            base.cascade.push(ConfigLevel {
                source: ConfigSource::Global {
                    path: global.clone(),
                },
                config: parsed.clone(),
            });
            base = merge(base, parsed);
        }

        // Level 2: repo config or --config (mutually exclusive)
        let local_candidate = path.map_or_else(|| repo_root.join("omniscient.toml"), PathBuf::from);
        match std::fs::read_to_string(&local_candidate) {
            Ok(s) => {
                let source = if path.is_some() {
                    ConfigSource::Cli {
                        path: local_candidate.clone(),
                    }
                } else {
                    ConfigSource::Repo {
                        path: local_candidate.clone(),
                    }
                };
                let parsed = Config::from_toml_str(&s, repo_root.clone())?;
                base.cascade.push(ConfigLevel {
                    source,
                    config: parsed.clone(),
                });
                base = merge(base, parsed);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // no local file — use whatever we have so far (defaults ± global)
            }
            Err(e) => {
                return Err(Error::Config(format!(
                    "reading {}: {e}",
                    local_candidate.display()
                )));
            }
        }

        // Ensure repo_root is set on the final merged config
        base.repo_root = repo_root;
        Ok(base)
    }

    /// Variant of `load` that lets tests inject a custom global path.
    #[cfg(test)]
    pub fn load_with_global(
        path: Option<&Path>,
        repo_root: PathBuf,
        global: Option<&Path>,
    ) -> Result<Config> {
        let mut base = Config::default_for(repo_root.clone());

        if let Some(global_p) = global {
            let global_path_buf = PathBuf::from(global_p);
            if global_path_buf.exists() {
                let s = std::fs::read_to_string(&global_path_buf).map_err(|e| {
                    Error::Config(format!("reading {}: {e}", global_path_buf.display()))
                })?;
                let parsed = Config::from_toml_str(&s, repo_root.clone())?;
                base.cascade.push(ConfigLevel {
                    source: ConfigSource::Global {
                        path: global_path_buf,
                    },
                    config: parsed.clone(),
                });
                base = merge(base, parsed);
            }
        }

        let local_candidate = path.map_or_else(|| repo_root.join("omniscient.toml"), PathBuf::from);
        match std::fs::read_to_string(&local_candidate) {
            Ok(s) => {
                let source = if path.is_some() {
                    ConfigSource::Cli {
                        path: local_candidate.clone(),
                    }
                } else {
                    ConfigSource::Repo {
                        path: local_candidate.clone(),
                    }
                };
                let parsed = Config::from_toml_str(&s, repo_root.clone())?;
                base.cascade.push(ConfigLevel {
                    source,
                    config: parsed.clone(),
                });
                base = merge(base, parsed);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(Error::Config(format!(
                    "reading {}: {e}",
                    local_candidate.display()
                )));
            }
        }

        base.repo_root = repo_root;
        Ok(base)
    }

    /// Check if a file should be indexed based on the language whitelist.
    ///
    /// Empty `languages` → allow all. Otherwise, matches both the chunker's
    /// detected language name ("rust", "python", "typescript") and the raw
    /// extension ("rs", "py", "ts", "tsx"), so a user specifying either works.
    pub fn is_language_allowed(&self, path: &Path, detected_language: Option<&str>) -> bool {
        if self.languages.is_empty() {
            return true;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let ext_matches = ext
            .as_deref()
            .is_some_and(|e| self.languages.iter().any(|l| l.eq_ignore_ascii_case(e)));
        let lang_matches = detected_language.is_some_and(|l| {
            self.languages
                .iter()
                .any(|lang| lang.eq_ignore_ascii_case(l))
        });
        ext_matches || lang_matches
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
        assert!(c.cascade.is_empty(), "defaults have no cascade levels");
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

    // — Merge unit tests —

    #[test]
    fn merge_scalar_override() {
        let base = Config::default_for(PathBuf::from("/repo"));
        let mut overlay = Config::default_for(PathBuf::from("/other"));
        overlay.search.max_results = 5;

        let merged = merge(base, overlay);
        assert_eq!(merged.search.max_results, 5, "overlay scalar wins");
        assert!(
            (merged.search.relevance_ratio - 0.75).abs() < 1e-6,
            "unspecified overlay field keeps base value (got {})",
            merged.search.relevance_ratio
        );
    }

    #[test]
    fn is_language_allowed_empty_whitelist_allows_all() {
        let c = Config {
            languages: vec![],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(c.is_language_allowed(Path::new("lib.rs"), Some("rust")));
        assert!(c.is_language_allowed(Path::new("lib.py"), Some("python")));
        assert!(c.is_language_allowed(Path::new("README.md"), None));
        assert!(c.is_language_allowed(Path::new("Cargo.toml"), None));
    }

    #[test]
    fn is_language_allowed_matches_by_language_name() {
        let c = Config {
            languages: vec!["rust".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(
            c.is_language_allowed(Path::new("lib.rs"), Some("rust")),
            "rust name matches rust"
        );
        assert!(
            !c.is_language_allowed(Path::new("lib.py"), Some("python")),
            "python should be blocked"
        );
        assert!(
            !c.is_language_allowed(Path::new("README.md"), None),
            "unknown language with no extension match should be blocked"
        );
    }

    #[test]
    fn is_language_allowed_matches_by_extension() {
        let c = Config {
            languages: vec!["md".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(
            c.is_language_allowed(Path::new("README.md"), None),
            "md extension should match"
        );
        assert!(
            !c.is_language_allowed(Path::new("lib.rs"), Some("rust")),
            "rust should be blocked"
        );
    }

    #[test]
    fn is_language_allowed_case_insensitive() {
        let c = Config {
            languages: vec!["Rust".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(
            c.is_language_allowed(Path::new("lib.rs"), Some("rust")),
            "Rust config should match rust language (name match)"
        );
        assert!(
            !c.is_language_allowed(Path::new("lib.rs"), None),
            "Rust config should NOT match .rs extension (different strings)"
        );
        // Extension match works when config uses the extension form
        let c2 = Config {
            languages: vec!["rs".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(
            c2.is_language_allowed(Path::new("lib.rs"), None),
            "rs config should match .rs extension"
        );
    }

    #[test]
    fn is_language_allowed_typescript_matches_both_ts_and_tsx() {
        let c = Config {
            languages: vec!["typescript".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(
            c.is_language_allowed(Path::new("app.ts"), Some("typescript")),
            ".ts with typescript language should match"
        );
        assert!(
            c.is_language_allowed(Path::new("App.tsx"), Some("typescript")),
            ".tsx with typescript language should match"
        );
    }

    #[test]
    fn is_language_allowed_multiple_languages() {
        let c = Config {
            languages: vec!["rust".into(), "python".into()],
            ..Config::default_for(PathBuf::from("/repo"))
        };
        assert!(c.is_language_allowed(Path::new("lib.rs"), Some("rust")));
        assert!(c.is_language_allowed(Path::new("main.py"), Some("python")));
        assert!(
            !c.is_language_allowed(Path::new("app.ts"), Some("typescript")),
            "typescript should be blocked"
        );
    }

    #[test]
    fn timeout_defaults_and_parse() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert_eq!(
            c.embedder.request_timeout_secs, 30,
            "embedder timeout defaults to 30s"
        );
        assert_eq!(
            c.search.search_timeout_secs, 60,
            "search timeout defaults to 60s"
        );

        let toml = r"
            [embedder]
            request_timeout_secs = 10
            [search]
            search_timeout_secs = 120
        ";
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.request_timeout_secs, 10);
        assert_eq!(c.search.search_timeout_secs, 120);
    }

    #[test]
    fn strip_banner_aliases_old_key() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert!(
            c.strip_banner_comments,
            "strip_banner_comments defaults to true"
        );
        // Old key name 'strip_comments' must still work via #[serde(alias = ...)].
        let c = Config::from_toml_str("strip_comments = false", PathBuf::from("/repo")).unwrap();
        assert!(!c.strip_banner_comments, "old key must be aliased");
        // New key name works too.
        let c =
            Config::from_toml_str("strip_banner_comments = false", PathBuf::from("/repo")).unwrap();
        assert!(!c.strip_banner_comments, "new key must work");
    }

    #[test]
    fn merge_vec_replacement() {
        let base = Config::default_for(PathBuf::from("/repo"));
        let mut overlay = Config::default_for(PathBuf::from("/other"));
        overlay.languages = vec!["go".into()];

        let merged = merge(base, overlay);
        assert_eq!(
            merged.languages,
            vec!["go"],
            "overlay Vec replaces base entirely, not merged"
        );
        assert_eq!(
            merged.exclude,
            Vec::<String>::new(),
            "empty overlay exclude keeps base empty"
        );
    }

    #[test]
    fn merge_default_vec_keeps_base() {
        // When overlay's vec equals the global default, base wins.
        // This covers the real-world case where the overlay TOML doesn't
        // mention `languages` / `exclude` and they fall back to defaults.
        let mut base = Config::default_for(PathBuf::from("/repo"));
        base.languages = vec!["rust".into(), "python".into()];
        base.exclude = vec!["vendor/**".into()];
        let overlay = Config::default_for(PathBuf::from("/other")); // default vecs

        let merged = merge(base, overlay);
        assert_eq!(
            merged.languages,
            vec!["rust", "python"],
            "overlay default Vec preserves base"
        );
        assert_eq!(
            merged.exclude,
            vec!["vendor/**"],
            "overlay default empty exclude preserves base"
        );
    }

    #[test]
    fn merge_repo_root_from_base() {
        let base = Config::default_for(PathBuf::from("/base-repo"));
        let overlay = Config::default_for(PathBuf::from("/overlay-repo"));
        let merged = merge(base, overlay);
        assert_eq!(
            merged.repo_root,
            PathBuf::from("/base-repo"),
            "repo_root always comes from base"
        );
    }

    // — Cascade integration tests —

    #[test]
    fn cascade_defaults_only() {
        let repo = tempfile::tempdir().unwrap();
        // No global, no repo file
        let c = Config::load_with_global(
            None,
            repo.path().to_path_buf(),
            Some(Path::new("/no/global")),
        )
        .unwrap();
        assert!(c.cascade.is_empty(), "no files → no cascade levels");
        assert_eq!(c.embedder.model, "qwen3-embedding-4b");
    }

    #[test]
    fn cascade_global_only() {
        let repo = tempfile::tempdir().unwrap();
        let global_dir = tempfile::tempdir().unwrap();
        let global_file = global_dir.path().join("omniscient.toml");
        std::fs::write(&global_file, "[embedder]\nmodel = \"global-model\"\n").unwrap();

        let c =
            Config::load_with_global(None, repo.path().to_path_buf(), Some(global_file.as_ref()))
                .unwrap();

        assert_eq!(c.cascade.len(), 1);
        assert!(matches!(&c.cascade[0].source, ConfigSource::Global { .. }));
        assert_eq!(c.embedder.model, "global-model");
    }

    #[test]
    fn cascade_repo_only() {
        let repo = tempfile::tempdir().unwrap();
        let repo_file = repo.path().join("omniscient.toml");
        std::fs::write(&repo_file, "[search]\nmax_results = 10\n").unwrap();

        let c = Config::load_with_global(
            None,
            repo.path().to_path_buf(),
            Some(Path::new("/no/global")),
        )
        .unwrap();

        assert_eq!(c.cascade.len(), 1);
        assert!(matches!(&c.cascade[0].source, ConfigSource::Repo { .. }));
        assert_eq!(c.search.max_results, 10);
    }

    #[test]
    fn cascade_global_and_repo_repo_overrides() {
        let repo = tempfile::tempdir().unwrap();
        let global_dir = tempfile::tempdir().unwrap();
        let global_file = global_dir.path().join("omniscient.toml");
        std::fs::write(
            &global_file,
            r#"
            [embedder]
            model = "global-model"
            [search]
            max_results = 20
            "#,
        )
        .unwrap();

        let repo_file = repo.path().join("omniscient.toml");
        std::fs::write(&repo_file, "[search]\nmax_results = 5\n").unwrap();

        let c =
            Config::load_with_global(None, repo.path().to_path_buf(), Some(global_file.as_ref()))
                .unwrap();

        assert_eq!(c.cascade.len(), 2);
        assert!(matches!(&c.cascade[0].source, ConfigSource::Global { .. }));
        assert!(matches!(&c.cascade[1].source, ConfigSource::Repo { .. }));
        assert_eq!(
            c.embedder.model, "global-model",
            "global model preserved (repo didn't set it)"
        );
        assert_eq!(c.search.max_results, 5, "repo max_results overrides global");
    }

    #[test]
    fn cascade_cli_replaces_repo() {
        let repo = tempfile::tempdir().unwrap();
        // Repo config exists
        let repo_file = repo.path().join("omniscient.toml");
        std::fs::write(&repo_file, "[search]\nmax_results = 10\n").unwrap();

        // CLI config exists
        let cli_dir = tempfile::tempdir().unwrap();
        let cli_file = cli_dir.path().join("cli.toml");
        std::fs::write(&cli_file, "[search]\nmax_results = 99\n").unwrap();

        let c = Config::load_with_global(
            Some(cli_file.as_ref()),
            repo.path().to_path_buf(),
            Some(Path::new("/no/global")),
        )
        .unwrap();

        assert_eq!(c.cascade.len(), 1);
        assert!(matches!(&c.cascade[0].source, ConfigSource::Cli { .. }));
        assert_eq!(c.search.max_results, 99);
    }

    #[test]
    fn cascade_vec_replacement_across_levels() {
        let repo = tempfile::tempdir().unwrap();
        let global_dir = tempfile::tempdir().unwrap();
        let global_file = global_dir.path().join("omniscient.toml");
        std::fs::write(
            &global_file,
            r#"
            languages = ["rust", "python", "typescript"]
            exclude = ["vendor/**"]
            "#,
        )
        .unwrap();

        let repo_file = repo.path().join("omniscient.toml");
        std::fs::write(&repo_file, "languages = [\"rust\"]\n").unwrap();

        let c =
            Config::load_with_global(None, repo.path().to_path_buf(), Some(global_file.as_ref()))
                .unwrap();

        assert_eq!(
            c.languages,
            vec!["rust"],
            "repo languages replaces global (not appended)"
        );
        assert_eq!(
            c.exclude,
            vec!["vendor/**"],
            "global exclude preserved (repo didn't set it)"
        );
    }

    #[test]
    fn resolve_key_none_and_empty() {
        assert_eq!(resolve_key(None, |_| None).unwrap(), None);
        assert_eq!(resolve_key(Some(""), |_| None).unwrap(), None);
        assert_eq!(resolve_key(Some("   "), |_| None).unwrap(), None);
    }

    #[test]
    fn resolve_key_literal() {
        assert_eq!(
            resolve_key(Some("sk-abc123"), |_| None).unwrap(),
            Some("sk-abc123".to_string())
        );
    }

    #[test]
    fn resolve_key_braced_env() {
        let out = resolve_key(Some("${MY_KEY}"), |n| {
            (n == "MY_KEY").then(|| "secret-val".to_string())
        })
        .unwrap();
        assert_eq!(out, Some("secret-val".to_string()));
    }

    #[test]
    fn resolve_key_unbraced_env() {
        let out = resolve_key(Some("$MY_KEY"), |n| {
            (n == "MY_KEY").then(|| "secret-val".to_string())
        })
        .unwrap();
        assert_eq!(out, Some("secret-val".to_string()));
    }

    #[test]
    fn resolve_key_referenced_but_unset_errors() {
        let err = resolve_key(Some("${MISSING}"), |_| None).unwrap_err();
        assert!(
            matches!(&err, Error::Config(m) if m.contains("MISSING")),
            "error must name the missing var, got {err:?}"
        );
    }

    #[test]
    fn resolve_key_env_set_but_empty_is_none() {
        // A referenced var that is set but empty/whitespace → no auth (not an empty Bearer).
        assert_eq!(
            resolve_key(Some("${MY_KEY}"), |n| (n == "MY_KEY").then(String::new)).unwrap(),
            None
        );
        assert_eq!(
            resolve_key(Some("$MY_KEY"), |n| (n == "MY_KEY")
                .then(|| "   ".to_string()))
            .unwrap(),
            None
        );
    }

    #[test]
    fn resolve_key_env_value_is_trimmed() {
        assert_eq!(
            resolve_key(Some("${MY_KEY}"), |n| (n == "MY_KEY")
                .then(|| "  sk-xyz  ".to_string()))
            .unwrap(),
            Some("sk-xyz".to_string())
        );
    }

    #[test]
    fn resolve_key_malformed_dollar_is_literal() {
        // No valid var name after $ → treated as a literal key, not an env ref.
        assert_eq!(
            resolve_key(Some("${BAD-NAME}"), |_| None).unwrap(),
            Some("${BAD-NAME}".to_string())
        );
        assert_eq!(
            resolve_key(Some("$"), |_| None).unwrap(),
            Some("$".to_string())
        );
    }

    #[test]
    fn api_key_defaults_none_and_parses() {
        let c = Config::default_for(PathBuf::from("/repo"));
        assert_eq!(c.embedder.api_key, None, "api_key defaults to None");

        let toml = r#"
        [embedder]
        api_key = "${OMNISCIENT_API_KEY}"
    "#;
        let c = Config::from_toml_str(toml, PathBuf::from("/repo")).unwrap();
        assert_eq!(c.embedder.api_key.as_deref(), Some("${OMNISCIENT_API_KEY}"));
    }

    #[test]
    fn merge_api_key_overlay_wins_and_inherits() {
        // Overlay sets it → overlay wins.
        let base = Config::default_for(PathBuf::from("/repo"));
        let mut overlay = Config::default_for(PathBuf::from("/other"));
        overlay.embedder.api_key = Some("repo-key".into());
        let merged = merge(base, overlay);
        assert_eq!(merged.embedder.api_key.as_deref(), Some("repo-key"));

        // Overlay unset (None = default) → base value inherited.
        let mut base = Config::default_for(PathBuf::from("/repo"));
        base.embedder.api_key = Some("global-key".into());
        let overlay = Config::default_for(PathBuf::from("/other"));
        let merged = merge(base, overlay);
        assert_eq!(merged.embedder.api_key.as_deref(), Some("global-key"));
    }

    #[test]
    fn global_path_uses_dirs_config_dir() {
        let p = global_path();
        // On any supported platform this should resolve to something
        // ending in `omniscient/omniscient.toml`
        assert!(p.is_some(), "global_path should resolve on this platform");
        let path = p.unwrap();
        assert_eq!(path.file_name().unwrap(), "omniscient.toml");
        assert_eq!(path.parent().unwrap().file_name().unwrap(), "omniscient");
    }
}

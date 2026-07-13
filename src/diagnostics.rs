//! End-to-end health checks shared by the `diagnostics` MCP tool and the
//! `doctor` CLI. `run` is given an ALREADY-RESOLVED engine (never builds one),
//! so the MCP path reuses the lazy engine and cannot spawn a second auto-start
//! server.
use crate::config::Config;
use crate::embed::endpoint_listening;
use crate::engine::Engine;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
}

pub struct Check {
    pub name: String,
    pub status: Status,
    pub detail: String,
}

pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    /// Overall status is the worst child: any Fail => Fail, else any Warn => Warn.
    pub fn overall(&self) -> Status {
        if self.checks.iter().any(|c| c.status == Status::Fail) {
            Status::Fail
        } else if self.checks.iter().any(|c| c.status == Status::Warn) {
            Status::Warn
        } else {
            Status::Pass
        }
    }

    pub fn render(&self) -> String {
        use std::fmt::Write;
        let header = match self.overall() {
            Status::Pass => "PASS".to_string(),
            Status::Warn => "WARN".to_string(),
            Status::Fail => {
                let fails = self
                    .checks
                    .iter()
                    .filter(|c| c.status == Status::Fail)
                    .count();
                format!("FAIL — {fails} of {} checks failed", self.checks.len())
            }
        };
        let mut out = format!("omniscient diagnostics: {header}\n");
        for c in &self.checks {
            let tag = match c.status {
                Status::Pass => "PASS",
                Status::Warn => "WARN",
                Status::Fail => "FAIL",
            };
            let _ = writeln!(out, "  [{tag}] {}: {}", c.name, c.detail);
        }
        out
    }
}

/// Run every check we can and classify each — never abort on the first failure,
/// because the whole point is to report *which* stage is broken.
pub async fn run(config: &Config, engine: std::result::Result<&Arc<Engine>, &str>) -> Report {
    let mut checks = vec![build_check(), config_check(config)];
    checks.push(embedder_check(config, engine).await);
    checks.extend(index_and_query_checks(engine).await);
    checks.push(watcher_check(config, engine));
    Report { checks }
}

/// Build/version — always informational.
fn build_check() -> Check {
    Check {
        name: "build".into(),
        status: Status::Pass,
        detail: format!("omniscient {}", env!("CARGO_PKG_VERSION")),
    }
}

/// Resolved repo root + config cascade (global → repo → cli) with override details.
fn config_check(config: &Config) -> Check {
    use crate::config::ConfigSource;
    use std::fmt::Write;

    let mut detail = format!("{}", config.repo_root.display());

    if config.cascade.is_empty() {
        detail.push_str(" — built-in defaults (no config files)");
        return Check {
            name: "config".into(),
            status: Status::Pass,
            detail,
        };
    }

    // List each level
    for (i, level) in config.cascade.iter().enumerate() {
        let (tag, path_ref) = match &level.source {
            ConfigSource::Global { path } => ("global", path),
            ConfigSource::Repo { path } => ("repo", path),
            ConfigSource::Cli { path } => ("cli", path),
        };
        let _ = writeln!(detail, "\n  [{i}] {tag}: {}", path_ref.display());
    }

    // Show overrides between levels
    if config.cascade.len() >= 2 {
        let _ = writeln!(detail, "\n  overrides:");
        for i in 1..config.cascade.len() {
            let prev = &config.cascade[i - 1];
            let curr = &config.cascade[i];
            let prev_tag = match &prev.source {
                ConfigSource::Global { .. } => "global",
                ConfigSource::Repo { .. } => "repo",
                ConfigSource::Cli { .. } => "cli",
            };
            let curr_tag = match &curr.source {
                ConfigSource::Global { .. } => "global",
                ConfigSource::Repo { .. } => "repo",
                ConfigSource::Cli { .. } => "cli",
            };

            let overrides = collect_overrides(&prev.config, &curr.config);
            if overrides.is_empty() {
                let _ = writeln!(
                    detail,
                    "    (none — {curr_tag} config uses {prev_tag} values for all fields)"
                );
            } else {
                for field_name in &overrides {
                    let _ = writeln!(
                        detail,
                        "    {field_name}: ({prev_tag}) → ({curr_tag})  [OVERRIDDEN]"
                    );
                }
            }
        }
        // Note about vector replacement semantics
        let _ = writeln!(
            detail,
            "    note: vector fields (languages, exclude) are replaced entirely, not merged"
        );
    }

    Check {
        name: "config".into(),
        status: Status::Pass,
        detail,
    }
}

/// Return field names where `overlay` differs from `base` (non-default values).
fn collect_overrides(base: &Config, overlay: &Config) -> Vec<String> {
    let def = Config::default();
    let mut names = Vec::new();

    // Embedder fields
    if overlay.embedder.model != def.embedder.model && overlay.embedder.model != base.embedder.model
    {
        names.push("embedder.model".into());
    }
    if overlay.embedder.base_url != def.embedder.base_url
        && overlay.embedder.base_url != base.embedder.base_url
    {
        names.push("embedder.base_url".into());
    }
    if overlay.embedder.max_batch_chunks != def.embedder.max_batch_chunks
        && overlay.embedder.max_batch_chunks != base.embedder.max_batch_chunks
    {
        names.push("embedder.max_batch_chunks".into());
    }
    if overlay.embedder.max_batch_bytes != def.embedder.max_batch_bytes
        && overlay.embedder.max_batch_bytes != base.embedder.max_batch_bytes
    {
        names.push("embedder.max_batch_bytes".into());
    }
    if overlay.embedder.auto_start != def.embedder.auto_start
        && overlay.embedder.auto_start != base.embedder.auto_start
    {
        names.push("embedder.auto_start".into());
    }
    if overlay.embedder.llama_bin != def.embedder.llama_bin
        && overlay.embedder.llama_bin != base.embedder.llama_bin
    {
        names.push("embedder.llama_bin".into());
    }
    if overlay.embedder.hf_repo != def.embedder.hf_repo
        && overlay.embedder.hf_repo != base.embedder.hf_repo
    {
        names.push("embedder.hf_repo".into());
    }
    if overlay.embedder.pooling != def.embedder.pooling
        && overlay.embedder.pooling != base.embedder.pooling
    {
        names.push("embedder.pooling".into());
    }
    if overlay.embedder.auto_start_timeout_secs != def.embedder.auto_start_timeout_secs
        && overlay.embedder.auto_start_timeout_secs != base.embedder.auto_start_timeout_secs
    {
        names.push("embedder.auto_start_timeout_secs".into());
    }

    // Search fields
    if overlay.search.max_results != def.search.max_results
        && overlay.search.max_results != base.search.max_results
    {
        names.push("search.max_results".into());
    }
    if (overlay.search.relevance_ratio - def.search.relevance_ratio).abs() > 1e-6
        && (overlay.search.relevance_ratio - base.search.relevance_ratio).abs() > 1e-6
    {
        names.push("search.relevance_ratio".into());
    }
    if overlay.search.token_budget != def.search.token_budget
        && overlay.search.token_budget != base.search.token_budget
    {
        names.push("search.token_budget".into());
    }

    // Watch fields
    if overlay.watch.enabled != def.watch.enabled && overlay.watch.enabled != base.watch.enabled {
        names.push("watch.enabled".into());
    }
    if overlay.watch.debounce_ms != def.watch.debounce_ms
        && overlay.watch.debounce_ms != base.watch.debounce_ms
    {
        names.push("watch.debounce_ms".into());
    }

    // Top-level fields
    if !overlay.languages.is_empty() && overlay.languages != base.languages {
        names.push("languages".into());
    }
    if overlay.strip_banner_comments != def.strip_banner_comments
        && overlay.strip_banner_comments != base.strip_banner_comments
    {
        names.push("strip_banner_comments".into());
    }
    if !overlay.exclude.is_empty() && overlay.exclude != base.exclude {
        names.push("exclude".into());
    }
    if overlay.index_tests != def.index_tests && overlay.index_tests != base.index_tests {
        names.push("index_tests".into());
    }

    names
}

/// Embedder connectivity. On engine-init failure, a TCP probe of the endpoint
/// disambiguates "endpoint down" from other init errors and drives the hint.
async fn embedder_check(config: &Config, engine: std::result::Result<&Arc<Engine>, &str>) -> Check {
    match engine {
        Ok(e) => Check {
            name: "embedder".into(),
            status: Status::Pass,
            detail: format!("{} @ {}", e.embedder_id(), config.embedder.base_url),
        },
        Err(err) => {
            let hint = if endpoint_listening(&config.embedder.base_url).await {
                "endpoint is listening but init failed"
            } else {
                "endpoint not listening — start llama.cpp or set [embedder] auto_start = true"
            };
            Check {
                name: "embedder".into(),
                status: Status::Fail,
                detail: format!("{} — {err} ({hint})", config.embedder.base_url),
            }
        }
    }
}

/// Index population + a live end-to-end sample query. Only meaningful when the
/// engine is up; both are FAIL-skipped otherwise.
async fn index_and_query_checks(engine: std::result::Result<&Arc<Engine>, &str>) -> [Check; 2] {
    let Ok(e) = engine else {
        return [
            Check {
                name: "index".into(),
                status: Status::Fail,
                detail: "skipped — engine unavailable".into(),
            },
            Check {
                name: "query".into(),
                status: Status::Fail,
                detail: "skipped — engine unavailable".into(),
            },
        ];
    };
    // Run the sample query FIRST: search() reconciles via ensure_fresh(), so a
    // cold / not-yet-reconciled index gets populated before we read stats().
    // stats() does not reconcile, so reading it first would FAIL a healthy
    // server whose index just hasn't been built yet (the cold-start case this
    // tool is meant to be called in).
    let query_result = e.search("function definition", Some(1)).await;
    let index = match e.stats().await {
        Ok((files, 0)) => Check {
            name: "index".into(),
            status: Status::Fail,
            detail: format!("{files} files, 0 chunks — nothing indexed"),
        },
        Ok((files, chunks)) => Check {
            name: "index".into(),
            status: Status::Pass,
            detail: format!("{files} files, {chunks} chunks"),
        },
        Err(err) => Check {
            name: "index".into(),
            status: Status::Fail,
            detail: err.to_string(),
        },
    };
    let query = match query_result {
        Ok(hits) if hits.is_empty() => Check {
            name: "query".into(),
            status: Status::Warn,
            detail: "sample query returned no matches (index may be empty or query too specific)"
                .into(),
        },
        Ok(hits) => Check {
            name: "query".into(),
            status: Status::Pass,
            detail: format!("sample query returned {} result(s)", hits.len()),
        },
        Err(err) => Check {
            name: "query".into(),
            status: Status::Fail,
            detail: format!("sample query failed: {err}"),
        },
    };
    [index, query]
}

/// Watcher state — informational (enabled-but-warming is expected on cold start).
fn watcher_check(config: &Config, engine: std::result::Result<&Arc<Engine>, &str>) -> Check {
    let detail = if config.watch.enabled {
        match engine {
            Ok(e) if e.refresh_state().is_watch_active() => "enabled, active".into(),
            Ok(_) => "enabled, warming up (scan-on-search until caught up)".into(),
            Err(_) => "enabled (status unknown — engine unavailable)".into(),
        }
    } else {
        "disabled".to_string()
    };
    Check {
        name: "watcher".into(),
        status: Status::Pass,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::embed::MockEmbedder;
    use crate::engine::Engine;
    use std::sync::Arc;
    use tempfile::tempdir;

    async fn healthy_engine(root: std::path::PathBuf) -> Arc<Engine> {
        let cfg = Config::default_for(root);
        let engine = Engine::new_with_embedder(cfg, Arc::new(MockEmbedder::new("mock-v1", 64)))
            .await
            .unwrap();
        engine.refresh().await.unwrap();
        Arc::new(engine)
    }

    #[tokio::test]
    async fn healthy_pipeline_reports_pass() {
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = healthy_engine(repo.path().to_path_buf()).await;

        let report = run(&cfg, Ok(&engine)).await;

        assert_eq!(
            report.overall(),
            Status::Pass,
            "render:\n{}",
            report.render()
        );
        assert!(report.render().contains("omniscient diagnostics: PASS"));
        // embedder id is surfaced
        assert!(report.render().contains("mock-v1"));
    }

    #[tokio::test]
    async fn engine_down_reports_fail_with_hint() {
        let repo = tempdir().unwrap();
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        // A definitely-closed port so endpoint_listening is deterministically false.
        cfg.embedder.base_url = "http://127.0.0.1:1".into();

        let report = run(&cfg, Err("connect: connection refused")).await;

        assert_eq!(report.overall(), Status::Fail);
        let text = report.render();
        assert!(text.contains("FAIL"));
        assert!(text.contains("connection refused"));
        assert!(
            text.contains("auto_start"),
            "should hint remediation:\n{text}"
        );
    }

    #[tokio::test]
    async fn empty_index_reports_fail() {
        // Engine built but nothing indexed (no refresh, empty repo).
        let repo = tempdir().unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Arc::new(
            Engine::new_with_embedder(cfg.clone(), Arc::new(MockEmbedder::new("mock-v1", 64)))
                .await
                .unwrap(),
        );

        let report = run(&cfg, Ok(&engine)).await;

        assert_eq!(
            report.overall(),
            Status::Fail,
            "render:\n{}",
            report.render()
        );
        assert!(report.render().contains("nothing indexed"));
    }

    #[tokio::test]
    async fn cold_index_is_reconciled_before_stats_check() {
        // A freshly-built engine whose on-disk index is not yet populated must NOT
        // report FAIL: the sample query reconciles via ensure_fresh(), so the index
        // check must see the reconciled chunks rather than a spurious 0. Regression
        // for the stats()-before-search() ordering bug.
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        // NOTE: intentionally NO engine.refresh() — the on-disk index starts empty.
        let engine = Arc::new(
            Engine::new_with_embedder(cfg.clone(), Arc::new(MockEmbedder::new("mock-v1", 64)))
                .await
                .unwrap(),
        );

        let report = run(&cfg, Ok(&engine)).await;

        assert_eq!(
            report.overall(),
            Status::Pass,
            "cold index must reconcile via the sample query, not FAIL:\n{}",
            report.render()
        );
    }
}

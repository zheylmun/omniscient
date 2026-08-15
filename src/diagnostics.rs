//! End-to-end health checks shared by the `diagnostics` MCP tool and the
//! `doctor` CLI. `run` is given an ALREADY-RESOLVED engine (never builds one),
//! so the MCP path reuses the lazy engine and cannot spawn a second auto-start
//! server.
use crate::caps::CapsSource;
use crate::config::Config;
use crate::embed::endpoint_listening;
use crate::engine::{Engine, resolve_concurrency};
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
    // AFTER the sample query: that query reconciles, so it is what populates the
    // failure list and can tighten the budget. Reading either first would report
    // a startup snapshot rather than the state a search actually runs against.
    checks.push(limits_check(config, engine));
    checks.push(reconcile_check(engine));
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
///
/// Split by section purely to keep each function readable; `note` is what makes
/// adding a field a one-liner, so a new knob is less likely to be forgotten here
/// (which is how `embed_concurrency`, `request_timeout_secs` and
/// `search_timeout_secs` went unreported). `every_overridable_field_is_reported_as_an_override`
/// pins the full list.
fn collect_overrides(base: &Config, overlay: &Config) -> Vec<String> {
    let def = Config::default();
    let mut names = Vec::new();
    embedder_overrides(&mut names, base, overlay, &def);
    search_and_watch_overrides(&mut names, base, overlay, &def);
    top_level_overrides(&mut names, base, overlay, &def);
    names
}

/// Record `field` when the overlay set it to something that is neither the
/// built-in default (i.e. it was actually specified) nor what the level below
/// already had (i.e. it actually changes something).
fn note<T: PartialEq>(names: &mut Vec<String>, field: &str, base: &T, overlay: &T, def: &T) {
    if overlay != def && overlay != base {
        names.push(field.to_string());
    }
}

fn embedder_overrides(names: &mut Vec<String>, base: &Config, o: &Config, def: &Config) {
    let (base, over, dflt) = (&base.embedder, &o.embedder, &def.embedder);
    note(
        names,
        "embedder.model",
        &base.model,
        &over.model,
        &dflt.model,
    );
    note(
        names,
        "embedder.base_url",
        &base.base_url,
        &over.base_url,
        &dflt.base_url,
    );
    note(
        names,
        "embedder.max_batch_chunks",
        &base.max_batch_chunks,
        &over.max_batch_chunks,
        &dflt.max_batch_chunks,
    );
    note(
        names,
        "embedder.max_batch_bytes",
        &base.max_batch_bytes,
        &over.max_batch_bytes,
        &dflt.max_batch_bytes,
    );
    note(
        names,
        "embedder.max_chunk_tokens",
        &base.max_chunk_tokens,
        &over.max_chunk_tokens,
        &dflt.max_chunk_tokens,
    );
    note(
        names,
        "embedder.embed_concurrency",
        &base.embed_concurrency,
        &over.embed_concurrency,
        &dflt.embed_concurrency,
    );
    note(
        names,
        "embedder.request_timeout_secs",
        &base.request_timeout_secs,
        &over.request_timeout_secs,
        &dflt.request_timeout_secs,
    );
    note(
        names,
        "embedder.auto_start",
        &base.auto_start,
        &over.auto_start,
        &dflt.auto_start,
    );
    note(
        names,
        "embedder.llama_bin",
        &base.llama_bin,
        &over.llama_bin,
        &dflt.llama_bin,
    );
    note(
        names,
        "embedder.hf_repo",
        &base.hf_repo,
        &over.hf_repo,
        &dflt.hf_repo,
    );
    note(
        names,
        "embedder.pooling",
        &base.pooling,
        &over.pooling,
        &dflt.pooling,
    );
    note(
        names,
        "embedder.auto_start_timeout_secs",
        &base.auto_start_timeout_secs,
        &over.auto_start_timeout_secs,
        &dflt.auto_start_timeout_secs,
    );
    // By NAME only — never the value. `api_key` may hold a literal secret.
    note(
        names,
        "embedder.api_key",
        &base.api_key,
        &over.api_key,
        &dflt.api_key,
    );
}

fn search_and_watch_overrides(names: &mut Vec<String>, base: &Config, o: &Config, def: &Config) {
    let (base_s, over_s, dflt_s) = (&base.search, &o.search, &def.search);
    note(
        names,
        "search.max_results",
        &base_s.max_results,
        &over_s.max_results,
        &dflt_s.max_results,
    );
    // The one float: compared by epsilon rather than `note`'s equality.
    if (over_s.relevance_ratio - dflt_s.relevance_ratio).abs() > 1e-6
        && (over_s.relevance_ratio - base_s.relevance_ratio).abs() > 1e-6
    {
        names.push("search.relevance_ratio".to_string());
    }
    note(
        names,
        "search.token_budget",
        &base_s.token_budget,
        &over_s.token_budget,
        &dflt_s.token_budget,
    );
    note(
        names,
        "search.search_timeout_secs",
        &base_s.search_timeout_secs,
        &over_s.search_timeout_secs,
        &dflt_s.search_timeout_secs,
    );

    let (base_w, over_w, dflt_w) = (&base.watch, &o.watch, &def.watch);
    note(
        names,
        "watch.enabled",
        &base_w.enabled,
        &over_w.enabled,
        &dflt_w.enabled,
    );
    note(
        names,
        "watch.debounce_ms",
        &base_w.debounce_ms,
        &over_w.debounce_ms,
        &dflt_w.debounce_ms,
    );
}

fn top_level_overrides(names: &mut Vec<String>, base: &Config, o: &Config, def: &Config) {
    // Vec fields use emptiness rather than the default as the "unset" test: an
    // empty list is how a level declines to replace the one below it.
    if !o.languages.is_empty() && o.languages != base.languages {
        names.push("languages".to_string());
    }
    note(
        names,
        "strip_banner_comments",
        &base.strip_banner_comments,
        &o.strip_banner_comments,
        &def.strip_banner_comments,
    );
    if !o.exclude.is_empty() && o.exclude != base.exclude {
        names.push("exclude".to_string());
    }
    note(
        names,
        "index_tests",
        &base.index_tests,
        &o.index_tests,
        &def.index_tests,
    );
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

/// How the effective context window was arrived at. Named explicitly rather than
/// inferred, because "not reported" is the misconfiguration users hit: a
/// `llama serve` router answers `/props` about itself with `n_ctx: 0`, so the
/// configured fallback applies silently and chunks get split far smaller than the
/// backend could take.
fn context_window_detail(
    probed: Option<usize>,
    source: CapsSource,
    fallback_tokens: usize,
) -> String {
    match (probed, source) {
        (Some(t), CapsSource::Props) => format!("context window: {t} tokens (from /props)"),
        (Some(t), CapsSource::Models) => format!("context window: {t} tokens (from /v1/models)"),
        (Some(t), CapsSource::None) => format!("context window: {t} tokens (reported)"),
        (None, _) => format!(
            "context window: not reported — falling back to [embedder] max_chunk_tokens = {fallback_tokens}"
        ),
    }
}

/// The auto-detected limits the pipeline is actually running with. Informational,
/// but the only place a user can audit values that are otherwise chosen silently.
/// Numbers and field names only — never a configured value, so no secret can ride
/// along here.
fn limits_check(config: &Config, engine: std::result::Result<&Arc<Engine>, &str>) -> Check {
    use std::fmt::Write;

    let Ok(e) = engine else {
        return Check {
            name: "limits".into(),
            status: Status::Fail,
            detail: "skipped — engine unavailable".into(),
        };
    };
    let budget = e.chunk_budget();
    let mut detail = context_window_detail(
        budget.probed_tokens(),
        e.caps_source(),
        config.embedder.chunk_budget_tokens(),
    );
    let _ = write!(
        detail,
        "\n  chunk budget: {} bytes{}",
        budget.bytes(),
        if budget.was_tightened() {
            " (tightened by an endpoint overflow)"
        } else {
            ""
        }
    );
    let reported = e.max_concurrent_requests();
    let slots = match reported {
        Some(n) => format!("server reports {n} slot(s)"),
        None => "slots not reported".to_string(),
    };
    let _ = write!(
        detail,
        "\n  embed concurrency: {} ({slots})",
        resolve_concurrency(config.embedder.embed_concurrency, reported)
    );
    Check {
        name: "limits".into(),
        status: Status::Pass,
        detail,
    }
}

/// Per-file failures from the most recent reconcile. A skipped file leaves the
/// index partial but usable, so this warns rather than fails.
fn reconcile_check(engine: std::result::Result<&Arc<Engine>, &str>) -> Check {
    use std::fmt::Write;

    let Ok(e) = engine else {
        return Check {
            name: "reconcile".into(),
            status: Status::Fail,
            detail: "skipped — engine unavailable".into(),
        };
    };
    let failures = e.last_failures();
    if failures.is_empty() {
        return Check {
            name: "reconcile".into(),
            status: Status::Pass,
            detail: "no failures".into(),
        };
    }
    let mut detail = format!(
        "{} file(s) failed and were skipped — they are retried on the next pass",
        failures.len()
    );
    for (path, err) in &failures {
        let _ = write!(detail, "\n  {path}: {err}");
    }
    Check {
        name: "reconcile".into(),
        status: Status::Warn,
        detail,
    }
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

    #[test]
    fn config_check_reports_api_key_override_without_leaking_value() {
        use crate::config::{Config, ConfigLevel, ConfigSource};
        use std::path::PathBuf;

        let repo = PathBuf::from("/repo");
        let mut global = Config::default_for(repo.clone());
        global.embedder.api_key = None;
        let mut local = Config::default_for(repo.clone());
        local.embedder.api_key = Some("super-secret-key".into());

        let mut config = Config::default_for(repo.clone());
        config.embedder.api_key = Some("super-secret-key".into());
        config.cascade = vec![
            ConfigLevel {
                source: ConfigSource::Global {
                    path: PathBuf::from("/g/omniscient.toml"),
                },
                config: global,
            },
            ConfigLevel {
                source: ConfigSource::Repo {
                    path: repo.join("omniscient.toml"),
                },
                config: local,
            },
        ];

        let check = config_check(&config);
        assert!(
            check.detail.contains("embedder.api_key"),
            "override report must name the field, got:\n{}",
            check.detail
        );
        assert!(
            !check.detail.contains("super-secret-key"),
            "the secret VALUE must never appear in diagnostics, got:\n{}",
            check.detail
        );
    }

    #[test]
    fn every_overridable_field_is_reported_as_an_override() {
        // `collect_overrides` is a hand-maintained field list, so a knob added to
        // `Config` is silently absent from diagnostics until someone remembers to
        // wire it in — which is exactly what happened to embed_concurrency,
        // request_timeout_secs and search_timeout_secs. This test sets every
        // overridable field to a non-default value at once and pins the expected
        // names, so adding a field without listing it here fails loudly.
        use std::path::PathBuf;

        let repo = PathBuf::from("/repo");
        let base = Config::default_for(repo.clone());
        let mut o = Config::default_for(repo);
        o.embedder.model = "other-model".into();
        o.embedder.base_url = "http://elsewhere:9999".into();
        o.embedder.max_batch_chunks = 7;
        o.embedder.max_batch_bytes = 777;
        o.embedder.max_chunk_tokens = 77;
        o.embedder.embed_concurrency = Some(3);
        o.embedder.request_timeout_secs = 11;
        o.embedder.auto_start = true;
        o.embedder.llama_bin = "/opt/llama".into();
        o.embedder.hf_repo = "some/repo:Q2_K".into();
        o.embedder.pooling = "mean".into();
        o.embedder.auto_start_timeout_secs = 12;
        o.embedder.api_key = Some("k".into());
        o.search.max_results = 3;
        o.search.relevance_ratio = 0.25;
        o.search.token_budget = 99;
        o.search.search_timeout_secs = 13;
        o.watch.enabled = false;
        o.watch.debounce_ms = 999;
        o.languages = vec!["go".into()];
        o.strip_banner_comments = false;
        o.exclude = vec!["vendor/**".into()];
        o.index_tests = true;

        let mut got = collect_overrides(&base, &o);
        got.sort();
        let mut want = vec![
            "embedder.model",
            "embedder.base_url",
            "embedder.max_batch_chunks",
            "embedder.max_batch_bytes",
            "embedder.max_chunk_tokens",
            "embedder.embed_concurrency",
            "embedder.request_timeout_secs",
            "embedder.auto_start",
            "embedder.llama_bin",
            "embedder.hf_repo",
            "embedder.pooling",
            "embedder.auto_start_timeout_secs",
            "embedder.api_key",
            "search.max_results",
            "search.relevance_ratio",
            "search.token_budget",
            "search.search_timeout_secs",
            "watch.enabled",
            "watch.debounce_ms",
            "languages",
            "strip_banner_comments",
            "exclude",
            "index_tests",
        ];
        want.sort_unstable();
        assert_eq!(
            got, want,
            "every overridable config field must be reported by name"
        );
    }

    #[tokio::test]
    async fn reports_the_discovered_limits() {
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let engine = Arc::new(
            Engine::new_with_embedder(
                cfg.clone(),
                Arc::new(MockEmbedder::new("mock-v1", 64).with_max_input_tokens(4096)),
            )
            .await
            .unwrap(),
        );

        let report = run(&cfg, Ok(&engine)).await;
        let text = report.render();

        assert!(
            text.contains("4096"),
            "the probed window must appear:\n{text}"
        );
        assert!(
            text.contains("chunk budget"),
            "the effective byte budget must appear:\n{text}"
        );
        assert!(
            text.contains("embed concurrency"),
            "the derived concurrency must appear:\n{text}"
        );
        assert!(
            text.contains("no failures"),
            "a clean reconcile must be stated:\n{text}"
        );
    }

    #[test]
    fn context_window_detail_names_its_source() {
        assert!(context_window_detail(Some(2048), CapsSource::Props, 512).contains("/props"));
        assert!(context_window_detail(Some(40960), CapsSource::Models, 512).contains("/v1/models"));
        assert!(context_window_detail(Some(4096), CapsSource::None, 512).contains("4096"));
        // The router case CLAUDE.md warns about: nothing probed, fallback applied.
        let fallback = context_window_detail(None, CapsSource::None, 2048);
        assert!(
            fallback.contains("not reported") && fallback.contains("2048"),
            "the fallback must be visible as a fallback, got: {fallback}"
        );
    }

    #[tokio::test]
    async fn limits_report_never_carries_a_configured_secret() {
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn alpha() {}\n").unwrap();
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.embedder.api_key = Some("super-secret-key".into());
        let engine = healthy_engine(repo.path().to_path_buf()).await;

        let text = run(&cfg, Ok(&engine)).await.render();

        assert!(
            !text.contains("super-secret-key"),
            "no diagnostics check may print a secret VALUE, got:\n{text}"
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

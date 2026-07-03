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
    let mut checks = Vec::new();

    // Build — always informational.
    checks.push(Check {
        name: "build".into(),
        status: Status::Pass,
        detail: format!("omniscient {}", env!("CARGO_PKG_VERSION")),
    });

    // Repo + config source.
    let cfg_file = config.repo_root.join("omniscient.toml");
    let source = if cfg_file.exists() {
        format!("config {}", cfg_file.display())
    } else {
        "built-in defaults (no omniscient.toml)".into()
    };
    checks.push(Check {
        name: "repo".into(),
        status: Status::Pass,
        detail: format!("{} — {source}", config.repo_root.display()),
    });

    // Embedder.
    match engine {
        Ok(e) => checks.push(Check {
            name: "embedder".into(),
            status: Status::Pass,
            detail: format!("{} @ {}", e.embedder_id(), config.embedder.base_url),
        }),
        Err(err) => {
            let listening = endpoint_listening(&config.embedder.base_url).await;
            let hint = if listening {
                "endpoint is listening but init failed"
            } else {
                "endpoint not listening — start llama.cpp or set [embedder] auto_start = true"
            };
            checks.push(Check {
                name: "embedder".into(),
                status: Status::Fail,
                detail: format!("{} — {err} ({hint})", config.embedder.base_url),
            });
        }
    }

    // Index + end-to-end query (only meaningful when the engine is up).
    match engine {
        Ok(e) => {
            match e.stats().await {
                Ok((files, chunks)) if chunks == 0 => checks.push(Check {
                    name: "index".into(),
                    status: Status::Fail,
                    detail: format!("{files} files, 0 chunks — nothing indexed"),
                }),
                Ok((files, chunks)) => checks.push(Check {
                    name: "index".into(),
                    status: Status::Pass,
                    detail: format!("{files} files, {chunks} chunks"),
                }),
                Err(err) => checks.push(Check {
                    name: "index".into(),
                    status: Status::Fail,
                    detail: err.to_string(),
                }),
            }
            match e.search("function definition", Some(1)).await {
                Ok(hits) if hits.is_empty() => checks.push(Check {
                    name: "query".into(),
                    status: Status::Warn,
                    detail: "sample query returned no matches (index may be empty or query too specific)".into(),
                }),
                Ok(hits) => checks.push(Check {
                    name: "query".into(),
                    status: Status::Pass,
                    detail: format!("sample query returned {} result(s)", hits.len()),
                }),
                Err(err) => checks.push(Check {
                    name: "query".into(),
                    status: Status::Fail,
                    detail: format!("sample query failed: {err}"),
                }),
            }
        }
        Err(_) => {
            checks.push(Check {
                name: "index".into(),
                status: Status::Fail,
                detail: "skipped — engine unavailable".into(),
            });
            checks.push(Check {
                name: "query".into(),
                status: Status::Fail,
                detail: "skipped — engine unavailable".into(),
            });
        }
    }

    // Watcher — informational.
    let watch_detail = if !config.watch.enabled {
        "disabled".to_string()
    } else {
        match engine {
            Ok(e) if e.refresh_state().is_watch_active() => "enabled, active".into(),
            Ok(_) => "enabled, warming up (scan-on-search until caught up)".into(),
            Err(_) => "enabled (status unknown — engine unavailable)".into(),
        }
    };
    checks.push(Check {
        name: "watcher".into(),
        status: Status::Pass,
        detail: watch_detail,
    });

    Report { checks }
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
        let engine = Engine::new_with_embedder(cfg, Box::new(MockEmbedder::new("mock-v1", 64)))
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
            Engine::new_with_embedder(cfg.clone(), Box::new(MockEmbedder::new("mock-v1", 64)))
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
}

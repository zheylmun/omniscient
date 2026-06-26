//! CLI: serve (MCP) + status/reindex (human debugging).
use crate::config::Config;
use crate::engine::Engine;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "omniscient")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    repo: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd { Serve, Status, Reindex }

fn load(cli: &Cli) -> anyhow::Result<Config> {
    let repo = cli.repo.clone().map(Ok).unwrap_or_else(std::env::current_dir)?;
    // Normalize to an absolute path so the index dir and scan are stable regardless
    // of the invocation cwd; keep the original if the path doesn't exist yet, but
    // surface any other error (permission denied, symlink loop, …) instead of
    // silently using a non-canonical path.
    let repo = match repo.canonicalize() {
        Ok(canonical) => canonical,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => repo,
        Err(e) => return Err(anyhow::anyhow!("failed to canonicalize repo path {}: {e}", repo.display())),
    };
    Ok(Config::load(cli.config.as_deref(), repo)?)
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let rt = tokio::runtime::Runtime::new()?;
    match cli.cmd {
        Cmd::Serve => {
            tracing_subscriber::fmt().with_writer(std::io::stderr).init();
            rt.block_on(crate::mcp::serve(load(&cli)?))?;
        }
        Cmd::Status => {
            tracing_subscriber::fmt().with_writer(std::io::stderr).init();
            rt.block_on(async {
                let engine = Engine::new(load(&cli)?).await?;
                engine.refresh().await?;
                let (files, chunks) = engine.stats().await?;
                println!("embedder: {}", engine.embedder_id());
                println!("files indexed: {files}");
                println!("chunks indexed: {chunks}");
                Ok::<_, anyhow::Error>(())
            })?;
        }
        Cmd::Reindex => {
            tracing_subscriber::fmt().with_writer(std::io::stderr).init();
            let cfg = load(&cli)?;
            let _ = std::fs::remove_dir_all(cfg.repo_root.join(".omniscient"));
            rt.block_on(async {
                let engine = Engine::new(cfg).await?;
                engine.refresh().await?;
                println!("reindex complete");
                Ok::<_, anyhow::Error>(())
            })?;
        }
    }
    Ok(())
}

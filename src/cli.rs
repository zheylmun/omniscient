//! CLI entrypoint.
use crate::config::Config;
use crate::mcp;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "omniscient", about = "Semantic code search")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the MCP stdio server
    Serve {
        /// Path to the repository root
        #[arg(long, default_value = ".")]
        repo: PathBuf,
        /// Path to config file (defaults to <repo>/omniscient.toml)
        #[arg(long)]
        config: Option<PathBuf>,
    },
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Serve { repo, config } => {
            let repo_root = repo.canonicalize().unwrap_or(repo);
            let cfg = Config::load(config.as_deref(), repo_root)?;
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(mcp::serve(cfg))?;
        }
    }
    Ok(())
}

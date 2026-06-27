//! CLI: serve (MCP) + status/reindex (human debugging).
use crate::config::Config;
use crate::engine::Engine;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

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

/// First ancestor of `start` (inclusive) that contains a `.git` entry. `.git` is
/// a directory in a normal clone but a file in worktrees/submodules, so we only
/// test for existence.
fn find_git_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|a| a.join(".git").exists())
        .map(Path::to_path_buf)
}

/// Normalize to an absolute path so the index dir and scan are stable regardless
/// of the invocation cwd; keep the original if the path doesn't exist yet, but
/// surface any other error (permission denied, symlink loop, …) instead of
/// silently using a non-canonical path.
fn canonicalize_repo(repo: PathBuf) -> anyhow::Result<PathBuf> {
    match repo.canonicalize() {
        Ok(canonical) => Ok(canonical),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(repo),
        Err(e) => Err(anyhow::anyhow!("failed to canonicalize repo path {}: {e}", repo.display())),
    }
}

fn resolve_repo(cli: &Cli) -> anyhow::Result<PathBuf> {
    // An explicit --repo is taken verbatim: the caller has named the tree, so we
    // don't second-guess it (and tests / non-git dirs stay usable).
    if let Some(repo) = cli.repo.clone() {
        return canonicalize_repo(repo);
    }
    // No --repo: index the git repo enclosing the launch directory. This is what
    // makes a single user-scope MCP registration work across every repo — the
    // client spawns `serve` with cwd set to the project. We refuse to guess when
    // there's no enclosing repo rather than silently indexing (and writing
    // .omniscient/ into) a stray directory like $HOME.
    let cwd = canonicalize_repo(std::env::current_dir()?)?;
    find_git_root(&cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "no git repository found at or above the current directory ({}); \
             run omniscient from inside a repository, or pass --repo <path>",
            cwd.display()
        )
    })
}

fn load(cli: &Cli) -> anyhow::Result<Config> {
    let repo = resolve_repo(cli)?;
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

#[cfg(test)]
mod tests {
    use super::find_git_root;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn finds_root_from_nested_subdir() {
        let tmp = tempdir().unwrap();
        let root = tmp.path();
        fs::create_dir(root.join(".git")).unwrap();
        let nested = root.join("crates/core/src");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_git_root(&nested).as_deref(), Some(root));
        assert_eq!(find_git_root(root).as_deref(), Some(root));
    }

    #[test]
    fn none_when_no_repo_above() {
        let tmp = tempdir().unwrap();
        let nested = tmp.path().join("a/b");
        fs::create_dir_all(&nested).unwrap();
        assert_eq!(find_git_root(&nested), None);
    }
}

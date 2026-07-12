//! MCP server over stdio: exposes `search` and `read_file`.
use crate::config::Config;
use crate::distill::ContextEntry;
use crate::engine::LazyEngine;
use crate::error::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Implementation, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::transport::stdio;
use rmcp::{RoleServer, ServerHandler, ServiceExt, tool, tool_router};

#[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
struct SearchParams {
    /// Natural-language or code query. Matches by meaning, not literal tokens, so
    /// it finds relevant code even when your query words never appear in it.
    query: String,
    /// Optional ceiling on how many results come back (it overrides the configured
    /// `max_results` for this call) — NOT a target. Results are selected by
    /// relevance shape: every hit within a set ratio of the top hit is returned, so
    /// a sharp query yields a few and a broad one more. Omit it to let the relevance
    /// distribution decide; set it only to cap an over-broad query.
    #[serde(default)]
    k: Option<u32>,
}

#[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
struct ReadFileParams {
    /// Repo-relative path of the file to read. Read live from disk, so it reflects
    /// uncommitted edits.
    path: String,
    /// Optional natural-language description of what you're looking for. Omit for a
    /// structural outline (every definition's signature + line range, bodies elided);
    /// provide it to get back only the chunks of the file most relevant to it.
    #[serde(default)]
    focus: Option<String>,
}

#[derive(Clone)]
struct Server {
    engine: LazyEngine,
    tool_router: ToolRouter<Server>,
}

#[tool_router]
impl Server {
    fn new(engine: LazyEngine) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Semantic code search over the indexed implementation + docs corpus. \
Given a natural-language or code query, returns ranked, distilled context (file:line + code \
body, each with a relevance note) by meaning rather than literal tokens — it finds relevant \
code even when your query words never appear in it.\n\
\n\
Result count follows the relevance shape, not a fixed number: every hit scoring within a set \
ratio of the top hit is returned, so a sharp query yields a few results and a broad one yields \
more. The optional `k` is a ceiling on how many results come back (it overrides the configured \
max), not a target — omit it and let the relevance distribution decide.\n\
\n\
Corpus: implementation source, docs/markdown, build scripts, and config. By design it EXCLUDES \
test code (#[cfg(test)] modules, tests/, benches/, **/*.test.*, **/*.spec.*, **/*_test.*) and \
dependency lock files (Cargo.lock, package-lock.json, go.sum, ...) — their absence is intended, \
not a relevance bug. examples/ IS indexed. For test code, call sites inside tests, or any \
exhaustive 'every occurrence of X' sweep, use a grep/text tool — those lines are not in this \
index and cannot be returned.\n\
\n\
Use when: 'where does concept X live', 'how does X work', locating code by behavior, or pulling \
design rationale from docs. Avoid when: you need every call site (results are relevance-ranked and \
capped, not exhaustive), an exact symbol you already know (grep is faster and literal), or results that must \
reflect the very latest working-tree edits — the index refreshes on each call but can briefly \
lag disk if the embedding backend is unreachable or a refresh is mid-flight, so cross-check grep \
when freshness is load-bearing.\n\
\n\
Results are heuristic similarity, not ground truth — verify before acting. Bulk data files may \
surface as low-relevance noise."
    )]
    async fn search(
        &self,
        Parameters(SearchParams { query, k }): Parameters<SearchParams>,
    ) -> String {
        match self.engine.get().await {
            Err(e) => format!("omniscient error: engine init failed: {e}"),
            Ok(engine) => match engine.search(&query, k.map(|v| v as usize)).await {
                Ok(entries) => render(&entries),
                Err(e) => format!("omniscient error: {e}"),
            },
        }
    }

    #[tool(
        description = "View of one file, read live from disk. Leading banner comments (license \
headers, module doc comments) are stripped from search results when configured; inline comments \
are preserved. Without `focus`: a structural outline — every type/impl/fn signature with its \
line range, bodies elided — a cheap way to grasp a large file's shape before reading it in \
full. With `focus` (a natural-language description): returns only the parts relevant to it.\n\
\n\
Use when: orienting in an unfamiliar or large file (outline), or extracting the relevant slice \
of a big file without paying to read the whole thing (focus). Avoid when: you need exact, \
complete file contents — the outline elides bodies and focus is selective; use a full file read \
for that. Unlike search, this reads current disk content, so it reflects uncommitted edits."
    )]
    async fn read_file(
        &self,
        Parameters(ReadFileParams { path, focus }): Parameters<ReadFileParams>,
    ) -> String {
        match self.engine.get().await {
            Err(e) => format!("omniscient error: engine init failed: {e}"),
            Ok(engine) => match engine.read_file(&path, focus.as_deref()).await {
                Ok(entries) => render(&entries),
                Err(e) => format!("omniscient error: {e}"),
            },
        }
    }

    #[tool(
        description = "Self-test the omniscient server end-to-end and return a PASS/FAIL \
report: embedder connectivity, index population, and a live sample query. Call this once before \
relying on `search` — if it reports FAIL, tell the user what failed instead of silently skipping \
omniscient. Takes no arguments."
    )]
    async fn diagnostics(&self) -> String {
        let engine = self.engine.get().await;
        let report = crate::diagnostics::run(
            self.engine.config(),
            engine.as_ref().map_err(String::as_str),
        )
        .await;
        report.render()
    }
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("omniscient", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "Local semantic code search (omniscient). Tools: search, read_file, diagnostics.\n\
                 Before relying on search, call `diagnostics` once to confirm the server is \
                 healthy. If it reports FAIL, tell the user what failed instead of skipping \
                 omniscient — do not silently ignore search errors.",
            )
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<CallToolResult, rmcp::ErrorData>>
    + Send
    + '_ {
        let ctx = ToolCallContext::new(self, request, context);
        self.tool_router.call(ctx)
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<ListToolsResult, rmcp::ErrorData>>
    + Send
    + '_ {
        let tools = self.tool_router.list_all();
        std::future::ready(Ok(ListToolsResult {
            tools,
            ..Default::default()
        }))
    }
}

fn render(entries: &[ContextEntry]) -> String {
    use std::fmt::Write;
    if entries.is_empty() {
        return "No matches.".into();
    }
    let mut out = String::new();
    for e in entries {
        let sym = e
            .symbol
            .as_deref()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        let _ = write!(
            out,
            "{}:{}-{}{} ({})\n```{}\n{}\n```\n\n",
            e.path, e.start_line, e.end_line, sym, e.why_matched, e.language, e.code
        );
    }
    out
}

/// Holds the live filesystem watcher (once its deferred setup finishes). Dropping
/// it stops watching and aborts the reconcile task, so `serve` keeps it to shutdown.
type WatcherSlot = std::sync::Arc<std::sync::Mutex<Option<crate::watcher::WatchGuard>>>;

/// Set up the file watcher WITHOUT blocking the caller. `notify`'s recursive watch
/// builds its file-id map by walking the whole tree synchronously — and it is
/// gitignore-blind, so it descends into `target/`, `node_modules/`, `.git/`, etc.
/// On a large repo that walk can outlast an MCP client's connect timeout, so we run
/// it on a blocking thread and return immediately, letting `serve` reach the stdio
/// handshake right away. Until setup lands the guard in the slot and a first reconcile
/// flips `watch_active`, `RefreshState::can_skip_scan()` stays false and `search`
/// takes the full-scan path — so results are never stale during the warm-up window.
fn spawn_watcher_deferred(
    config: &Config,
    lazy: LazyEngine,
    state: std::sync::Arc<crate::refresh::RefreshState>,
) -> WatcherSlot {
    let slot: WatcherSlot = std::sync::Arc::new(std::sync::Mutex::new(None));
    if !config.watch.enabled {
        return slot;
    }
    let slot_for_task = slot.clone();
    let repo = config.repo_root.clone();
    let watch_cfg = config.watch.clone();
    tokio::spawn(async move {
        // The recursive watch walk is blocking I/O; keep it off the async workers.
        match tokio::task::spawn_blocking(move || {
            crate::watcher::spawn(&repo, &watch_cfg, lazy, state)
        })
        .await
        {
            Ok(Ok(guard)) => *slot_for_task.lock().unwrap() = Some(guard),
            Ok(Err(e)) => tracing::warn!("file watcher disabled: {e}"),
            Err(e) => tracing::warn!("file watcher setup task failed: {e}"),
        }
    });
    slot
}

pub async fn serve(config: Config) -> Result<()> {
    let state = std::sync::Arc::new(crate::refresh::RefreshState::standalone());
    let lazy = LazyEngine::new(config.clone(), state.clone());

    // Set up the watcher off the handshake path so a large repo's recursive watch
    // walk can't stall the MCP connect. Held until shutdown; dropping the slot stops
    // watching and aborts the reconcile task.
    let _watch_slot = spawn_watcher_deferred(&config, lazy.clone(), state.clone());

    let server = Server::new(lazy);
    let running = server
        .serve(stdio())
        .await
        .map_err(|e| crate::error::Error::Other(anyhow::anyhow!("mcp serve: {e}")))?;
    running
        .waiting()
        .await
        .map_err(|e| crate::error::Error::Other(anyhow::anyhow!("mcp wait: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, WatchConfig};
    use crate::embed::MockEmbedder;
    use crate::engine::{Engine, LazyEngine};
    use crate::refresh::RefreshState;
    use std::sync::Arc;
    use std::time::Duration;
    use tempfile::tempdir;

    async fn lazy_for(cfg: &Config, state: &Arc<RefreshState>) -> LazyEngine {
        let engine = Arc::new(
            Engine::with_refresh_state(
                cfg.clone(),
                Arc::new(MockEmbedder::new("mock-v1", 64)),
                state.clone(),
            )
            .await
            .unwrap(),
        );
        LazyEngine::from_engine(cfg.clone(), state.clone(), engine)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deferred_watcher_setup_activates_the_watcher() {
        // The watcher is set up on a background blocking task (so a huge repo's
        // recursive watch walk never blocks the MCP handshake), yet it must still
        // come up and mark itself active once that setup finishes.
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.watch = WatchConfig {
            enabled: true,
            debounce_ms: 50,
        };
        let state = Arc::new(RefreshState::standalone());
        let lazy = lazy_for(&cfg, &state).await;

        let slot = spawn_watcher_deferred(&cfg, lazy, state.clone());

        // Condition-based wait (no fixed sleep): setup + first reconcile flips active.
        let mut active = false;
        for _ in 0..100 {
            if state.is_watch_active() {
                active = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(active, "deferred watcher setup should activate the watcher");
        assert!(
            slot.lock().unwrap().is_some(),
            "the live watcher guard must land in the slot"
        );
        drop(slot);
    }

    #[tokio::test]
    async fn disabled_watch_yields_empty_slot_and_inactive_state() {
        let repo = tempdir().unwrap();
        let mut cfg = Config::default_for(repo.path().to_path_buf());
        cfg.watch.enabled = false;
        let state = Arc::new(RefreshState::standalone());
        let lazy = lazy_for(&cfg, &state).await;

        let slot = spawn_watcher_deferred(&cfg, lazy, state.clone());
        assert!(
            slot.lock().unwrap().is_none(),
            "no watcher should be created when watching is disabled"
        );
        assert!(!state.is_watch_active());
    }

    #[tokio::test]
    async fn server_info_reports_omniscient_version() {
        let repo = tempdir().unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let state = Arc::new(RefreshState::standalone());
        let lazy = lazy_for(&cfg, &state).await;
        let server = Server::new(lazy);
        let info = server.get_info();
        assert_eq!(info.server_info.name, "omniscient");
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn diagnostics_tool_listed_and_returns_report() {
        let repo = tempdir().unwrap();
        std::fs::write(repo.path().join("a.rs"), "pub fn a() {}\n").unwrap();
        let cfg = Config::default_for(repo.path().to_path_buf());
        let state = Arc::new(RefreshState::standalone());
        let lazy = lazy_for(&cfg, &state).await;
        let server = Server::new(lazy);

        // Tool is advertised.
        let names: Vec<_> = server
            .tool_router
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        assert!(
            names.contains(&"diagnostics".to_string()),
            "tools: {names:?}"
        );

        // And it produces a report string.
        let out = server.diagnostics().await;
        assert!(out.contains("omniscient diagnostics:"), "out:\n{out}");
    }
}

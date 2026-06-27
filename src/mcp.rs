//! MCP server over stdio: exposes `search` and `read_file`.
use crate::config::Config;
use crate::distill::ContextEntry;
use crate::engine::LazyEngine;
use crate::error::Result;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{tool, tool_router, RoleServer, ServerHandler, ServiceExt};
use rmcp::transport::stdio;

#[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
struct SearchParams {
    query: String,
    #[serde(default)]
    k: Option<u32>,
}

#[derive(serde::Deserialize, rmcp::schemars::JsonSchema)]
struct ReadFileParams {
    path: String,
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
        Self { engine, tool_router: Self::tool_router() }
    }

    #[tool(description = "Semantic code search. Returns distilled, relevant code context (file:line + code) for a natural-language or code query.")]
    async fn search(&self, Parameters(SearchParams { query, k }): Parameters<SearchParams>) -> String {
        match self.engine.get().await {
            Err(e) => format!("omniscient error: engine init failed: {e}"),
            Ok(engine) => match engine.search(&query, k.map(|v| v as usize)).await {
                Ok(entries) => render(&entries),
                Err(e) => format!("omniscient error: {e}"),
            },
        }
    }

    #[tool(description = "Return a noise-stripped view of one file. With `focus`, returns the most relevant parts; without it, a structural outline.")]
    async fn read_file(&self, Parameters(ReadFileParams { path, focus }): Parameters<ReadFileParams>) -> String {
        match self.engine.get().await {
            Err(e) => format!("omniscient error: engine init failed: {e}"),
            Ok(engine) => match engine.read_file(&path, focus.as_deref()).await {
                Ok(entries) => render(&entries),
                Err(e) => format!("omniscient error: {e}"),
            },
        }
    }
}

impl ServerHandler for Server {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Local semantic code search (omniscient). Tools: search, read_file.")
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<CallToolResult, rmcp::ErrorData>> + Send + '_ {
        let ctx = ToolCallContext::new(self, request, context);
        self.tool_router.call(ctx)
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = std::result::Result<ListToolsResult, rmcp::ErrorData>> + Send + '_ {
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
        let sym = e.symbol.as_deref().map(|s| format!(" [{s}]")).unwrap_or_default();
        let _ = write!(
            out,
            "{}:{}-{}{} ({})\n```{}\n{}\n```\n\n",
            e.path, e.start_line, e.end_line, sym, e.why_matched, e.language, e.code
        );
    }
    out
}

pub async fn serve(config: Config) -> Result<()> {
    let state = std::sync::Arc::new(crate::refresh::RefreshState::standalone());
    let lazy = LazyEngine::new(config.clone(), state.clone());

    // Held until shutdown; dropping it stops watching and aborts the reconcile task.
    let _watch_guard = if config.watch.enabled {
        match crate::watcher::spawn(&config.repo_root, &config.watch, lazy.clone(), state.clone()) {
            Ok(guard) => Some(guard),
            Err(e) => { tracing::warn!("file watcher disabled: {e}"); None }
        }
    } else {
        None
    };

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`omniscient` is a local MCP server (single Rust binary, edition 2024) that gives MCP clients **semantic, distilled** code search. It indexes a repo into a local LanceDB vector store and exposes exactly two stdio tools — `search(query, k?)` and `read_file(path, focus?)`. Embeddings are computed by an **external** local llama.cpp `/v1/embeddings` endpoint; there is no in-process inference.

## Commands

```bash
cargo build                 # debug build
cargo build --release       # -> target/release/omniscient
cargo test                  # full suite; no network/server needed (uses MockEmbedder)
cargo test chunk::          # run one module's tests (chunk|config|embed|index|freshness|distill|engine)
cargo test --test integration   # the end-to-end integration test
cargo test embed::live -- --ignored   # the ONE network test: needs a running llama.cpp embeddings server

# Run the server / debug the index (require a reachable embeddings endpoint to build the Engine):
cargo run -- serve --repo <path>      # stdio MCP server
cargo run -- status --repo <path>     # embedder id + file/chunk counts
cargo run -- reindex --repo <path>    # delete .omniscient/ then rebuild
```

The whole suite runs offline because tests inject `embed::MockEmbedder` (deterministic, in `src/embed.rs`). Only the `#[ignore]`d `embed::live` test hits a real server.

## Architecture

The pipeline is a one-directional chain, wired only inside `src/engine.rs` (`Engine`):

```
freshness::scan  →  chunk::chunk_file  →  embed::Embedder  →  index (LanceDB)  →  distill::distill_context
```

- **`engine`** — the only module that composes the others. `Engine::search` is the single entry point used by both the MCP tools and the CLI.
- **`freshness`** — walks the repo (gitignore-aware) and blake3-hashes files; `diff` against stored hashes yields changed/deleted.
- **`chunk`** — tree-sitter for Rust/Python/TypeScript (one chunk per top-level definition), line-window fallback for everything else.
- **`embed`** — `Embedder` trait. Real impl is `LlamaCppEmbedder` (HTTP); `MockEmbedder` is the test seam. `build_embedder` is `async` (it probes the embedding dimension at connect time).
- **`index`** — LanceDB table + a `meta.json` sidecar recording the embedder id and dim.
- **`distill`** — deterministic, NO LLM: merges overlapping hits, strips noise, trims to a token budget.
- **`mcp`** — rmcp stdio server. **`cli`** — clap (`serve`/`status`/`reindex`).

### Invariants you must not break

- **Always-fresh (watcher-aware):** `Engine::search` calls `ensure_fresh()` before querying. It may skip the filesystem scan *only* when a healthy watcher guarantees the index already reflects the working tree (`RefreshState::can_skip_scan()`); in every other case — watching disabled, watcher not yet started, embedder was down, a watch error — it falls back to a full `reconcile()`. Never add a search path that can return stale results when the watcher is not known-caught-up.
- **Embedder id keys the index:** `index/meta.json` stores the embedder id + dim. On mismatch (e.g. config `model` changed) `Index::open` drops the table and forces a full rebuild. The index lives in `<repo_root>/.omniscient/` (excluded from the scan because dotfiles are skipped).
- **No LLM in `search`:** distillation is deterministic. v1 deliberately has NO generative "answer" mode, NO in-process/candle embeddings, and NO device (Metal/CUDA) policy. The `Embedder` trait is the seam for adding in-process embeddings later — keep that boundary clean rather than reintroducing those features inline.
- **stdout is reserved for the MCP protocol.** All logging goes to stderr (`tracing_subscriber` writer is stderr). Never `println!` in the `serve` path; `status`/`reindex` print to stdout only because they are human CLI commands.
- **Vectors are L2-normalized** (by `embed`), so cosine distance == dot product. `read_file`'s focus ranking relies on this.
- **`arrow-array`/`arrow-schema` are pinned to `58` to match `lancedb` 0.30**, which does not re-export arrow. Do not bump them independently or add a second arrow version — verify with `cargo tree | grep arrow-array` (must be a single line). `reqwest` is rustls-only (`default-features = false`); don't let a dep pull OpenSSL back in.

### rmcp specifics

`src/mcp.rs` implements `ServerHandler` **manually** (routing through `ToolRouter::call`/`list_all`) rather than via `#[tool_handler]`, because the macro's generated code collides with this crate's `Result` alias. The `Engine` is initialized **lazily** on first tool call via `tokio::sync::OnceCell::get_or_try_init` — so `serve` starts and `tools/list` works even when the embeddings endpoint is down, and a failed init is retryable (not poisoned).

## Configuration

`omniscient.toml` at the repo root (see `omniscient.example.toml`): `[embedder] base_url/model`, `[search] default_k/token_budget`, `languages`, `strip_comments`, `index_tests`, `exclude`. Defaults apply if the file is absent.

- **Test/fixture files are excluded from indexing** by built-in globs (`tests/`, `benches/`, `**/*.test.*`, `**/*.spec.*`, `**/*_test.*`, `**/test_*.py`, `**/conftest.py`, `**/__tests__/`; `examples/` is kept). `index_tests = true` disables the built-ins; `exclude = [...]` adds more globs. Applied in `freshness::scan` via `ignore`'s `OverrideBuilder`. Exclusion changes need NO reindex — they shift file *membership*, which the always-fresh `diff` already reconciles (unlike `CHUNKER_VERSION`, which changes chunk *content*).

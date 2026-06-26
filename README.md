# omniscient

A local semantic and distilled code search MCP server for Claude Code and other MCP clients. omniscient indexes your repository with tree-sitter-based chunking and vector embeddings, then exposes two tools over stdio: `search` and `read_file`.

## Tools

- **`search(query, k?)`** — Semantic search over your codebase. Returns up to `k` distilled code snippets most relevant to `query`. Always refreshes the index before searching, so results are always up to date with your working tree.
- **`read_file(path, focus?)`** — Return the outline (all chunks) of a file, or, if `focus` is given, the chunks of that file most relevant to `focus`.

## Always-Fresh Guarantee

Before every `search` call, omniscient computes a delta between the on-disk file hashes and the stored hashes, re-embeds only changed or new files, and deletes stale entries. You never see results from deleted or overwritten code.

## Embeddings

omniscient does **not** do in-process inference. Embeddings are served by a local [llama.cpp](https://github.com/ggml-org/llama.cpp) instance via its `/v1/embeddings` endpoint. You point omniscient at it with `base_url` in your config.

To start llama.cpp serving an embedding model (for example Qwen3-Embedding-4B):

```bash
llama-server \
  --model qwen3-embedding-4b-q4_k_m.gguf \
  --port 8080 \
  --embedding \
  --pooling mean
```

The `Embedder` trait in `src/embed.rs` is the seam for adding in-process embedding support in a future version.

> **v1 scope note:** omniscient v1 does not support in-process embeddings or a device policy (no Metal/CUDA selection). It also has no generative `answer` mode — search returns distilled code, not a generated summary. Both are planned future work.

## Configuration

Copy `omniscient.example.toml` to `omniscient.toml` in your repo root and edit:

```toml
languages = ["rust", "python", "typescript"]
strip_comments = true

[embedder]
base_url = "http://localhost:8080"
model = "qwen3-embedding-4b"

[search]
default_k = 8
token_budget = 4000
```

Changing the `model` field triggers an automatic full reindex on the next run — the index records the embedder id and rebuilds when it detects a mismatch.

## Build

```bash
cargo build --release
```

The resulting binary is at `target/release/omniscient`.

## Register with Claude Code

Add omniscient as a stdio MCP server in your Claude Code project config (`.claude/settings.json` or `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "omniscient": {
      "command": "/path/to/omniscient",
      "args": ["serve", "--repo", "/path/to/your/repo"]
    }
  }
}
```

Or using the Claude Code CLI:

```bash
claude mcp add omniscient /path/to/omniscient -- serve --repo /path/to/your/repo
```

## Debugging

```bash
# Show index status (file count, chunk count, embedder id)
omniscient status --repo /path/to/your/repo

# Force a full reindex
omniscient reindex --repo /path/to/your/repo
```

Both commands write progress to stderr and keep stdout reserved for MCP protocol output.

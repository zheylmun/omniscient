# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1]

### Fixed

- **Auto-started llama.cpp no longer crashes mid-use.** The spawned server
  inherited llama.cpp's small default context/batch sizes (ctx ~4096, ubatch
  512), far below the embedding requests omniscient sends (up to `max_batch_bytes`
  of text per request). A pooled embedding model must fit each sequence whole in
  one ubatch and within the context, so an under-sized server passed the startup
  probe and then aborted on the first real reconcile batch — appearing to "start
  fine, then silently die." `auto_start` now sets `--ctx-size`/`--batch-size`/`--ubatch-size`
  to `max(2048, max_batch_bytes)`, a safe ceiling since a token is always ≥ 1
  byte. The documented manual `llama serve` command carries the same flags.

### Changed

- **`search` MCP tool description** now explains relevance-shape result selection
  and documents the `k` argument as a ceiling (not a target), matching the 0.1.0
  behavior the tool description had not been updated to reflect.

## [0.1.0]

Initial release. A local MCP server (single Rust binary) that gives MCP clients
semantic, distilled code search over a repository.

### Added

- **Two stdio MCP tools:** `search(query, k?)` for semantic, distilled code
  search and `read_file(path, focus?)` for a noise-stripped, live-from-disk view
  of one file (structural outline by default, focus-ranked chunks with `focus`).
- **Relevance-shape result selection:** `search` returns every hit scoring at
  least `relevance_ratio` (default 0.75) of the top hit, so result count follows
  the score distribution — sharp queries return few results, broad ones more.
  `max_results` and `token_budget` are caps; the single best match is always
  returned. The MCP `k` argument overrides `max_results` per call.
- **Always-fresh index:** each `search` reconciles on-disk file hashes against
  stored hashes, re-embedding only changed/new files and deleting stale entries.
- **File watcher (default on):** debounced filesystem events reconcile the index
  proactively; `search` skips its scan only when a healthy watcher guarantees the
  index already reflects the tree, otherwise it falls back to a full scan.
- **External embeddings via llama.cpp:** embeddings come from a local llama.cpp
  `/v1/embeddings` endpoint (no in-process inference). Vectors are L2-normalized.
- **Opt-in auto-start:** with `[embedder] auto_start = true`, omniscient launches
  a local llama.cpp server when `base_url` is unreachable and ties its lifetime to
  the embedder. An already-running endpoint is always reused, never spawned over.
- **Tree-sitter chunking** for Rust, Python, and TypeScript (one chunk per
  top-level definition), with a line-window fallback for other languages.
- **Built-in exclusions:** dependency lock files and test/fixture files are
  excluded from indexing by default (`examples/` is kept), enforced at both index
  time and read time. Configurable via `index_tests` and `exclude`.
- **CLI:** `serve`, `status`, and `reindex` subcommands. Repo resolution walks up
  to the enclosing `.git` root, so one user-scope MCP registration works across
  every repository.
- **Configuration** via `omniscient.toml` (`[embedder]`, `[search]`, `[watch]`,
  plus `languages`, `strip_comments`, `index_tests`, `exclude`); sensible defaults
  apply when the file is absent.

[0.1.1]: https://github.com/zheylmun/omniscient/releases/tag/v0.1.1
[0.1.0]: https://github.com/zheylmun/omniscient/releases/tag/v0.1.0

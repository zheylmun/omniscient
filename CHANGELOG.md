# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Chunks are split to fit the endpoint's context window, and the budget is
  measured rather than guessed.** A single oversized input is rejected with a 400,
  and because a failed file's hash is never stored, a chunk that could never be
  split small enough failed *every* subsequent reconcile — wedging `search`
  permanently. omniscient now discovers the server's real limit at startup, starts
  optimistic from it, and folds each `exceed_context_size_error` back in as an
  exact bytes-per-token measurement. The correction is shared across a reconcile
  (so one file's rejection spares every later file), monotonic, strictly
  decreasing, and floors at 1 byte/token less slack for the server's special
  tokens — which always fits, whatever the content.
- **Server capability discovery** via `GET /props?model=<id>`, plain `GET /props`,
  then `GET /v1/models`, accumulated rather than replaced. The `?model=` scoping is
  what makes the first probe work behind a `llama serve` router, which otherwise
  answers `/props` about *itself* with `n_ctx: 0` and no slot count.
- **Embed concurrency is derived from the server's reported `total_slots`**
  (capped at 8), instead of always running serially. `[embedder] embed_concurrency`
  overrides it.
- **`diagnostics` / `doctor` report the discovered limits**: the context window and
  which probe answered it, the effective chunk budget and whether an overflow has
  tightened it, and the embed concurrency with the slot count it came from — so a
  misconfigured router is visible rather than inferred.
- `[embedder] max_chunk_tokens` seeds the budget when the endpoint reports no
  window at all. Both it and `embed_concurrency` are overrides of an auto-detected
  value and are best left unset.

### Changed

- **The default embedding model is now Qwen3-Embedding-0.6B (Q8_0, ~640 MB)**,
  down from 4B. It runs on CPU or an integrated GPU, so the out-of-the-box
  configuration works on an ordinary developer machine rather than requiring a
  large discrete GPU. To keep the previous model, pin `[embedder] model` **and**
  `hf_repo` to `Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M`. The model id keys the index,
  so anyone who does not pin it gets one full rebuild on the next run.
- CLI commands no longer print dependency logs over their own output. The default
  filter is `warn,omniscient=info`; `RUST_LOG` overrides it in full.

### Fixed

- **The default configuration did not work with the documented setup.**
  `[embedder] model` defaulted to a nickname (`qwen3-embedding-4b`) while `hf_repo`
  named a real GGUF. `llama serve` routes by model id and answers an unknown one
  with `400 model '<id>' not found`, so the shipped defaults failed on the first
  connect probe. `model` and `hf_repo` now default to the same string, and
  `auto_start` passes `--alias <model>` so a spawned server registers under the id
  omniscient asks for whatever `hf_repo` says.
- **A model-id mismatch under `auto_start` no longer hangs for ten minutes.** The
  readiness poll retried a permanent 400 until `auto_start_timeout_secs`. It now
  aborts immediately on an auth rejection and logs the underlying error with each
  "still waiting" line, so a doomed wait explains itself.
- **A down endpoint no longer stalls `search` for minutes.** A refused connection
  classifies as transient (correctly — one request cannot tell an outage from a
  blip), so every changed file rediscovered it through the full retry ladder.
  `reconcile_inner` now abandons the run after 5 consecutive retryable failures and
  names the endpoint as the cause. Measured on 40 files: 20.1 s → 2.5 s.
- **Merged search hits no longer splice two lines together.** Distillation dropped
  the newline between overlapping line-window chunks, so the last line of one
  window was concatenated onto the first line of the next and the distilled body
  did not appear anywhere in the file. Affected any file tree-sitter found no
  definitions in.
- One bad file can no longer block the index: embed failures are classified
  (`EmbedAuth` aborts the run, transient failures retry with backoff, overflow
  re-splits, anything else is scoped to one file) and reported through
  `diagnostics` instead of being swallowed.
- A file that yields no chunks now has its hash recorded, so `diff` stops
  reporting it as changed on every single reconcile forever.
- An aborted reconcile is recorded too, so `diagnostics` cannot report "no
  failures" straight after a run in which a rejected key doomed every file.
- `Index::open` rebuilds when `meta.json` is missing or unreadable.
- `[embedder] max_chunk_tokens = 0` is treated as unset rather than obeyed (it
  would clamp to a 4-byte budget and shatter every file in the repo).
- `embed_concurrency`, `request_timeout_secs` and `search_timeout_secs` are now
  reported by the `diagnostics` config-override listing, which previously omitted
  them.

## [0.2.0] - 2026-07-03

### Added

- `omniscient --version` / `-V` and a corrected MCP `serverInfo` (previously
  advertised as `rmcp`/`1.8.0`).
- `diagnostics` MCP tool and `omniscient doctor` CLI command: an end-to-end
  self-test (embedder connectivity, index population, live sample query) that
  reports PASS/FAIL. Server instructions now direct agents to run `diagnostics`
  before relying on `search` and to surface failures instead of silently
  dropping the tool.

### Fixed

- **MCP connect no longer times out while indexing a large repo.** The filesystem
  watcher was set up synchronously in `serve` before the stdio handshake, and
  `notify`'s recursive watch walks the entire tree (gitignore-blind, so it descends
  into `target/`, `node_modules/`, `.git/`) to build its file-id map. On a large
  repo, cold, that walk outlasted the client's 30-second connect timeout. Watcher
  setup now runs on a background blocking task, so `serve` reaches the handshake
  immediately; the watcher activates once setup finishes and `search` scans until
  then, so results are never stale during warm-up.

### Changed

- **`search`/`read_file` arguments now carry per-parameter schema descriptions.**
  The `k` ceiling-not-a-target framing (and `query`/`path`/`focus` docs) previously
  lived only in the tool's overall description; clients that render an argument list
  now see it at the parameter level too.

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

[Unreleased]: https://github.com/zheylmun/omniscient/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/zheylmun/omniscient/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/zheylmun/omniscient/releases/tag/v0.1.1
[0.1.0]: https://github.com/zheylmun/omniscient/releases/tag/v0.1.0

# omniscient

A local semantic and distilled code search MCP server for Claude Code and other MCP clients. omniscient indexes your repository with tree-sitter-based chunking and vector embeddings, then exposes two tools over stdio: `search` and `read_file`.

## Tools

- **`search(query, k?)`** — Semantic search over your codebase by meaning, not literal tokens. Returns ranked, distilled snippets (`file:line` + code body + a relevance note), selected by the *shape* of the relevance scores rather than a fixed count: every hit within `relevance_ratio` of the top hit is returned, so a sharp query yields a few results and a broad one yields more. `k` is an optional ceiling on candidates (it overrides `max_results` for the call), not a target. The index is refreshed before each search (see below). The corpus is implementation source, docs, build scripts, and config — it **excludes** test code and dependency lock files by design (`examples/` is kept); use a grep/text tool for exhaustive "every occurrence" sweeps or known-symbol lookups.
- **`read_file(path, focus?)`** — A noise-stripped view of one file, read live from disk (so it reflects uncommitted edits). Without `focus`, a structural outline: every definition's signature and line range with bodies elided. With `focus` (a natural-language description), only the chunks of that file most relevant to it.

## Always-Fresh Guarantee

Before every `search` call, omniscient computes a delta between the on-disk file hashes and the stored hashes, re-embeds only changed or new files, and deletes stale entries. You never see results from deleted or overwritten code.

With the file watcher enabled (the default), omniscient also reconciles the index
proactively as files change, and `search` skips its filesystem scan when the watcher
guarantees the index already reflects the tree. This trades strict per-search rescan
for a sub-second freshness window on the happy path; whenever the watcher is disabled,
not yet started, or unhealthy, `search` falls back to a full scan — so results are
never stale outside that narrow window.

## Embeddings

omniscient does **not** do in-process inference. Embeddings are served by a local [llama.cpp](https://github.com/ggml-org/llama.cpp) instance via its `/v1/embeddings` endpoint. You point omniscient at it with `base_url` in your config.

To start llama.cpp serving the default embedding model, letting it download the GGUF from Hugging Face:

```bash
llama serve \
  -hf Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --alias Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --port 8080 \
  --embedding \
  --pooling last \
  --ctx-size 32000 \
  --batch-size 32000 \
  --ubatch-size 32000
```

> **The model id must match.** `llama serve` is a router: it registers a
> `-hf`-loaded model under the repo id and answers any other id with
> `400 model '<id>' not found` — on `/v1/embeddings` and on the `/props?model=`
> capability probe alike. So `[embedder] model` has to be a string the server
> actually serves. The defaults are set up for this (`model` and `hf_repo` are the
> same string, and the `--alias` above is what pins it), but if you point
> omniscient at a server you started yourself, check `curl localhost:8080/v1/models`
> and use an `id` from that list. `omniscient doctor` reports the id it connected as.

**Scaling up.** The default is Qwen3-Embedding-**0.6B** (~640 MB, runs on CPU or an
integrated GPU) so that omniscient works out of the box on an ordinary machine. If
you have GPU memory to spare, the 4B and 8B models in the same family retrieve
noticeably better — swap **both** `model` and `hf_repo` to
`Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M` or `Qwen/Qwen3-Embedding-8B-GGUF:Q8_0`
(`pooling = "last"` is correct for all three). Changing the model changes the
embedder id, which keys the index, so the next run rebuilds it from scratch.

> **Pooling:** Qwen3-Embedding is a decoder/LLM-based embedder and uses **last-token** pooling (`--pooling last`), not mean pooling. BERT-family embedders (BGE, jina, nomic) use `--pooling mean`. If you omit `--pooling`, llama.cpp falls back to the model's metadata default. Wrong pooling produces degenerate embeddings, so it's the first thing to check if search quality looks off.

> **Context/batch size:** what llama.cpp actually enforces is the size of a single
> *input*, not the total of a batch. Sizing `--ctx-size` to your `max_batch_bytes`
> (32000) is therefore a deliberately generous ceiling — a token is always ≥ 1 byte,
> so no single chunk omniscient sends can exceed it — rather than a hard requirement.
> `--batch-size`/`--ubatch-size` are matched to it for simplicity; they can be left
> at llama.cpp's defaults if you would rather save the memory, since llama.cpp splits
> a pooled embedding sequence across ubatches on its own. omniscient additionally
> discovers the server's real limit at startup and splits chunks to fit, correcting
> itself from the server's own rejections, so a smaller context is safe — it just
> means larger definitions get split more. The `auto_start` path applies the
> generous sizing for you.

### Let omniscient start the server for you

Don't want to manage a separate process? Set `auto_start = true` under `[embedder]`
and omniscient will launch llama.cpp itself whenever `base_url` is unreachable —
running exactly the command above, derived from your config (`hf_repo`, `pooling`,
the port from `base_url`, and context/batch sizes from `max_batch_bytes`). It waits for the server to come up (the first run
downloads the GGUF, which can take a while) and shuts it down when omniscient
exits. An endpoint you started yourself is always reused as-is and never spawned
over.

The spawned server is always started with `--alias <model>`, so it registers under
whatever `[embedder] model` says regardless of `hf_repo`. On this path the two
therefore cannot drift: change either one and auto-start still connects.

This is opt-in and off by default. It requires the unified `llama` CLI on your
`PATH` (set `llama_bin` to an absolute path otherwise); if the binary is missing
omniscient fails with an install pointer rather than spawning. It only manages a
**local** server — a remote `base_url` is an error, since omniscient can't start a
process on another machine.

The `Embedder` trait in `src/embed.rs` is the seam for adding in-process embedding support in a future version.

> **v1 scope note:** omniscient v1 does not support in-process embeddings or a device policy (no Metal/CUDA selection). It also has no generative `answer` mode — search returns distilled code, not a generated summary. Both are planned future work.

## Configuration

Copy `omniscient.example.toml` to `omniscient.toml` in your repo root and edit:

```toml
# Language whitelist. Empty (the default) indexes every language: tree-sitter
# chunking for rust/python/javascript/typescript/tsx, line windows for the rest.
# Entries match a language name ("tsx"), a family ("typescript" also admits
# .tsx), or a raw extension ("rs").
languages = []
strip_comments = true

# Test/fixture files (tests/, benches/, **/*.test.*, **/*.spec.*, **/*_test.*,
# **/test_*.py, **/conftest.py, **/__tests__/) are skipped by default; examples/
# is kept. Set true to index test files too.
index_tests = false
# Extra glob patterns to skip, on top of the built-in test excludes.
exclude = []   # e.g. ["vendor/**", "**/*.generated.rs"]

[embedder]
base_url = "http://localhost:8080"
model = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"   # must be an id the server serves (see /v1/models)
max_batch_chunks = 64     # max chunks per embed request
max_batch_bytes = 32000   # max total bytes per embed request (~8k tokens)

# Optionally let omniscient launch llama.cpp itself when base_url is unreachable.
auto_start = false                                  # opt-in; off by default
llama_bin = "llama"                                 # the unified llama.cpp CLI (PATH or absolute path)
hf_repo = "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"    # passed to `llama serve -hf`; aliased to `model`
pooling = "last"                                    # last for Qwen3-Embedding, mean for BERT-family
auto_start_timeout_secs = 600                       # max wait for readiness (first run downloads the model)

[search]
# Results track the shape of the relevance scores: every hit within
# relevance_ratio of the top hit is returned (max_results / token_budget cap it).
relevance_ratio = 0.75
max_results = 25
token_budget = 4000

[watch]
enabled = true      # set false to disable the filesystem watcher
debounce_ms = 200   # coalesce bursts of FS events into one reconcile
```

Changing the `model` field triggers an automatic full reindex on the next run — the index records the embedder id and rebuilds when it detects a mismatch.

## Install

Install (or update to) the latest build into Cargo's bin directory
(`$CARGO_HOME/bin`, default `~/.cargo/bin`, which should be on your `PATH`) with
the bundled script:

```bash
./install.sh
```

That's a thin wrapper over `cargo install --path . --force`; re-run it any time to
pick up new commits. If you just want a local build instead, `cargo build --release`
leaves the binary at `target/release/omniscient`.

### Git hooks (CI parity)

Local git hooks are managed by [prek](https://github.com/j178/prek) (a fast,
Rust-native drop-in for the pre-commit framework) from `.pre-commit-config.yaml`,
so the checks that gate a PR also run as you work. Stages mirror
`.github/workflows/ci.yml`:

- **pre-commit** — `cargo fmt` and `dprint fmt` (markdown); both cheap, run on
  every commit
- **pre-push** — `cargo clippy --all-targets --all-features` and `cargo test
  --all-features`, both with `RUSTFLAGS=-Dwarnings` like CI

`install.sh` runs `prek install` for you once prek is on your `PATH`. Both tools
are Rust binaries: `cargo install --locked prek dprint`. Markdown formatting is
configured in `dprint.json` (matches the repo's existing style — no churn).
Bypass a run with `git commit -n` / `git push --no-verify`, or
`SKIP=fmt,clippy git commit` to skip specific hooks.

## Register with Claude Code

Register **once** at user scope and it works in every repository:

```bash
claude mcp add -s user omniscient omniscient -- serve
```

With no `--repo`, omniscient indexes the git repository enclosing the directory
Claude is launched from (it walks up from the current directory to find the `.git`
root). So a single global registration follows you from project to project — each
repo gets its own `.omniscient/` index. If the launch directory isn't inside a git
repo, omniscient refuses to guess and exits with an error rather than indexing a
stray directory.

To pin a specific repository regardless of launch directory, pass `--repo`
explicitly (useful for a per-project registration):

```bash
claude mcp add omniscient omniscient -- serve --repo /path/to/your/repo
```

Either form maps to this entry in `.claude/settings.json` / `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "omniscient": {
      "command": "omniscient",
      "args": ["serve"]
    }
  }
}
```

(Use an absolute `command` path if Cargo's bin directory — often `$CARGO_HOME/bin`,
default `~/.cargo/bin` — isn't on the PATH seen by your MCP client.)

## Debugging

```bash
# Show index status (file count, chunk count, embedder id)
omniscient status --repo /path/to/your/repo

# Force a full reindex
omniscient reindex --repo /path/to/your/repo
```

Both commands write progress to stderr and keep stdout reserved for MCP protocol output.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option. Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in this work by you, as
defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.

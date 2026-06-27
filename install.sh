#!/usr/bin/env bash
# Build and install the latest omniscient binary from this checkout into Cargo's
# bin dir ($CARGO_HOME/bin, default ~/.cargo/bin, which should be on your PATH).
# Re-run any time to update to the current source.
set -euo pipefail

cd "$(dirname "$0")"

# Install the prek-managed git hooks (rustfmt + markdown on commit; clippy +
# tests on push — see .pre-commit-config.yaml). Safe to re-run. Best-effort: this
# requires a git checkout, and a failure here must never block `cargo install`.
if [ -f .pre-commit-config.yaml ] && [ -e .git ]; then
  if command -v prek >/dev/null 2>&1; then
    prek install || echo "Note: 'prek install' failed; local git hooks not enabled." >&2
  else
    echo "Note: 'prek' not found. Install it with 'cargo install --locked prek'," >&2
    echo "      then run 'prek install' to enable the local hooks." >&2
  fi
  command -v dprint >/dev/null 2>&1 || \
    echo "Note: 'dprint' not found (markdown hook). Install: 'cargo install --locked dprint'." >&2
fi

cargo install --path . --force

bin="$(command -v omniscient || true)"
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
if [ -n "$bin" ]; then
  installed="$bin"
else
  installed="omniscient (ensure Cargo's bin dir, $cargo_bin, is on your PATH)"
fi
cat <<EOF

Installed: $installed

Register once with Claude Code so it works in every repository — the server
indexes whichever git repo Claude is launched in:

  claude mcp add -s user omniscient omniscient -- serve

Then in any repo, optionally drop an omniscient.toml (see omniscient.example.toml)
to point at your embeddings endpoint. Verify the index with:

  omniscient status
EOF

#!/usr/bin/env bash
# Build and install the latest omniscient binary from this checkout into Cargo's
# bin dir ($CARGO_HOME/bin, default ~/.cargo/bin, which should be on your PATH).
# Re-run any time to update to the current source.
set -euo pipefail

cd "$(dirname "$0")"

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

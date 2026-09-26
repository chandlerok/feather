#!/usr/bin/env bash
#
# One-time setup for a fresh clone or worktree: install the pinned toolchain,
# sync the Python environment, and install the git hooks.
#
# Everything after the first step is already declared in mise.toml; this script
# only exists to bootstrap the part mise cannot do for itself.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if ! command -v mise >/dev/null 2>&1; then
  cat >&2 <<'EOF'
mise is not installed. Install it first:
  https://mise.jdx.dev/getting-started.html
EOF
  exit 1
fi

# An untrusted mise.toml does not run the tasks below, and the error it prints
# never mentions trust. Sharing across worktrees usually makes this a no-op.
mise trust

# rust 1.98.1 (with rustfmt and clippy), uv, hk, ruff, maturin, dprint, bacon.
mise install

# `uv sync --all-groups` then `hk install --mise`.
mise run setup

cat <<'EOF'
Done. Next:
  mise run check        # everything CI runs (slow: builds bundled DuckDB)
  mise run valkey:up    # before the integration tests
EOF

#!/usr/bin/env bash
#
# One-time setup for a fresh clone or worktree: install the pinned toolchain,
# sync the Python environment, and install the git hooks.
#
# The human path is `mise run setup`. This is the same thing for an agent, with
# one deliberate difference: the Python environment is installed without the
# project, so the release build of the extension is skipped. See the note on
# the uv sync line below for the measurement behind that.
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

# Installs rust 1.98.1 (with rustfmt and clippy), ccache, uv, hk, ruff, maturin,
# dprint, bacon. ccache is not optional: `.cargo/config.toml` points CC and CXX
# at it, so a build without it fails.
mise install

# The two halves of `mise run setup`, except that the Python environment is
# installed without the project. `mise run setup` builds the extension, which is
# a release build with LTO: over eight minutes and 800MB of target/ per worktree
# on this machine. The lint and type-check path does not need that extension,
# and AGENTS.md calls out the trap that `uv run` without `--no-sync` builds it
# anyway. `mise run setup` remains the human path, and the one to use when the
# extension itself is what you need to exercise.
# `mise x --` runs each command with this repo's tools on PATH, which is what
# the `mise run setup` task was doing implicitly. Calling the binaries directly
# fails: neither uv nor hk is on PATH in a plain shell.
mise x -- uv sync --all-groups --no-install-project
mise x -- hk install --mise

cat <<'EOF'
Done. Next:
  cargo fmt --all --check && ruff check python tests && dprint check
  mise run check        # what CI runs; slow, so let a push run it instead
  mise run valkey:up    # before the integration tests
EOF

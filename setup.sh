#!/usr/bin/env bash
#
# One-time setup for a fresh clone or worktree: install the pinned toolchain,
# install the Python environment, and install the git hooks.
#
# The human path is `mise run setup`. This is the same work for an agent, with
# one deliberate difference: the project itself is not installed, which skips the
# extension build. See the note on the uv sync line for why that is not a
# smaller version of the same thing.
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

# Installs rust 1.98.1 (with rustfmt and clippy), uv, hk, ruff, maturin, dprint, bacon.
mise install

# The two halves of `mise run setup`, except that the project is not installed.
# Installing it builds the extension, and `feather-py`'s default features include
# `offline`, so that build compiles bundled DuckDB from source. That is the 10 to
# 15 minutes and the disk pressure the next section of AGENTS.md is about, once
# per worktree, and it is the build this repository tells agents not to run.
# What this does provision is the path agents use: mise's tools, the format and
# lint commands, and a venv that `uv run --no-sync pyrefly check` works against.
# It does not provision the Python test suite, which imports the compiled
# extension (tests/test_online.py imports MaterializeReport), so pytest needs
# that DuckDB build and is CI's job. `mise run setup` remains the human path, and
# the one to use when the extension itself is what you need to exercise.
#
# `mise x --` puts this repository's tools on PATH for each command, which the
# `mise run setup` task did implicitly. Calling the binaries directly fails:
# neither uv nor hk is on PATH in a plain shell.
mise x -- uv sync --all-groups --no-install-project
mise x -- hk install --mise

cat <<'EOF'
Done. Next:
  cargo fmt --all --check && ruff check python tests && dprint check
  uv run --no-sync pyrefly check
  mise run check        # what CI runs; slow, so let a push run it instead
  mise run valkey:up    # before the integration tests
EOF

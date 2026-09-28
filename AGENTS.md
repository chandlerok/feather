# Agent rules

Rules for coding agents in this repository. Each rule below has already been broken here at least
once: the cost was thrown-away work or a full disk.

## Bootstrap the tree before you start

A fresh worktree has no toolchain, no Python environment, and no git hooks, and none of the other
rules below work until it has them. Run this once, before any other command:

```bash
./setup.sh
```

It installs the pinned tools through mise (`mise trust`, `mise install`), installs the Python
environment, and installs the git hooks. It takes about a second. It does not build the extension:
no DuckDB, no cargo build, no wheel.

`paseo.json` declares this as the worktree setup command, so paseo runs it during worktree
bootstrap and you should not have to. It is not automatic in two cases: the operator has to allow
it once per workspace (`paseo workspace setup <workspace-id>`), and a `local` isolation workspace
has no worktree bootstrap at all. If `setup.sh` is missing, the tree is older than this section,
and the toolchain is not installed. If the run fails, say which step failed rather than working
around it; a half-set-up tree produces errors that look like code defects.

One trap, because it costs eight minutes and looks like a hang. `uv run` syncs the project before
it runs anything, so plain `uv run pytest` rebuilds the extension that `./setup.sh` skipped. Pass
`--no-sync` to `uv run` and the environment is used as installed. The build is a release build
with LTO: over eight minutes and 800MB of `target/` in a fresh worktree on this machine. It is
also what `mise run test` triggers, which is one more reason that task is not an agent's to run.

This is bootstrap, not a check, so it does not conflict with the next section. The prohibition is
on building to decide whether a change is correct.

## Give each tree its own target directory

This section does not say whether to use worktrees; it covers the builds either way. Two trees get
two `target/` directories unless something tells cargo otherwise, and nothing here does: no
`.cargo/config.toml` in the tree sets a `[build]` `target-dir`, and `mise.toml` declares no
`CARGO_TARGET_DIR`. A `.cargo/config.toml` that only carries an `[env]` table does not change
this, because cargo reads a target directory from `[build]`, not from the environment it hands
the build. Keep it that way. Pointing
several trees at one `CARGO_TARGET_DIR` does share the expensive part, because `libduckdb-sys` is a
registry dependency and its artifact name does not depend on where the tree is. It also merges the
trees' own crates, and that is not safe. Cargo leaves the absolute path out of the metadata hash on
purpose, so `feather-core` in two worktrees is one artifact: one
`libfeather_core-<hash>.rlib`, one
fingerprint, one dep-info file. Whichever tree built last owns it, and the other reports itself fresh
and links the wrong source. Two trees differing in one function, built in both orders, each produced
the other's value with no warning. It is
[cargo#12516](https://github.com/rust-lang/cargo/issues/12516), open since 2023, and two checkouts
of a workspace with path members reproduce it. Sharing is safe only while every tree sits at the same
commit, which is the moment before a lane's first edit, so it buys the DuckDB build at the price of
tests that run against another lane's code. Per-tree directories cost one duplicate build, the
cheaper failure.
The C++ half of that build is already shared without sharing a target directory, which is what
`.cargo/config.toml` is for: it points `CC` and `CXX` at ccache, whose directory is per user rather
than per tree, so a second tree relinks `libduckdb.a` instead of recompiling it.

Units are the other half of the disk, and sharing does not touch them. The record is one `target/` at
43G holding 12 debug and 8 release `libduckdb-sys` units; the cleanup that freed 15G kept the two
newest per profile, which was the wrong rule, as the next paragraph explains. Units accumulate
through repeated build configurations, which per-tree duplication cannot account for. The recorded
worst case was within 2.4Gi of full (#43). When free space drops below about 4Gi, keep the two
largest `libduckdb-sys` units per profile of that tree's directory and delete the rest, not the two
newest: a finished unit is gigabytes, an aborted build leaves a megabyte stub with a fresh mtime, and
newest-first therefore keeps the stubs and deletes the finished builds. Name the directory rather
than a relative `target/`, and confirm nothing is building against it first, because deleting units
under a running build breaks it. `pgrep -l cargo` does not confirm that on its own: a DuckDB build
here runs `cc1plus` with its cwd in the registry source directory, so a check that watches only
`cargo` misses the compiler writing the unit. A live build holds a descriptor under the target
directory, so watch the descriptors and it does not matter how the build was started:

```bash
tree=/path/to/the/tree                          # each tree has its own; never share one
target=$tree/target
if ls -l /proc/[0-9]*/cwd /proc/[0-9]*/fd 2>/dev/null | grep -q "$target"; then echo "a build is using $target; wait"; else
  for p in debug release; do du -sh "$target/$p/build/libduckdb-sys-"*/ 2>/dev/null | sort -rh | tail -n +3 | cut -f2- | xargs rm -rf; done
fi
```

`DUCKDB_DOWNLOAD_LIB=1` is not the way out of any of this. It was measured on the pinned
`libduckdb-sys` 1.10505.0 and the result is recorded in `crates/feather-core/Cargo.toml`: the
prebuilt path leaves the binary with no RUNPATH, and the static archive in the same zip does not
link. Read that comment before spending a run on it.

## Do not build locally to check a change; CI is the gate

`mise run check` is the expensive one: `mise.toml` defines it as format:check, lint and test, where
lint runs `cargo clippy --workspace --all-targets --all-features -- -D warnings` plus ruff and
pyrefly, and test runs `cargo test --workspace --all-features` plus pytest, all of it waiting on the
bundled DuckDB build. So do not run `cargo check`, `cargo clippy`, `cargo test`, `uv sync`,
`pytest`, `mise run` or Docker to check a change: the static archive alone is 1.8GB on the CI runner
(`.github/workflows/check.yml`), and a cold build is roughly 10 to 15 minutes
(`crates/feather-core/Cargo.toml`). The README's `mise run check` and `mise run setup` are for a
human with a machine to spare; this rule governs agents. Reproducing a documented measurement
(`docs/engine-and-format-decisions.md`, `docs/rust_feature_store_architecture.md`) is the one
exception, and only when the number is the claim under review.

None of these builds, so they are cheap enough to run in a worktree:

```bash
cargo fmt --all --check          # the formatter CI runs; it formats without compiling
ruff check python tests          # the paths mise and CI pass
ruff format --check python tests
dprint check <files>             # hk hands it the staged files; mise runs it over the tree
uv run --no-sync pyrefly check   # --no-sync, so nothing builds
```

Only `cargo fmt`, `ruff` and `dprint` work in a fresh worktree, and they need mise's tools on PATH
(`mise x --` or an activated shell); `pyrefly` needs the `.venv`, which `./setup.sh` creates. That
is the one environment an agent is expected to install, and only through `./setup.sh`, once, before
starting work. dprint owns each non-Rust, non-Python file type it has a plugin for: JSON, TOML,
YAML, Markdown and Dockerfile (`dprint.json`); `Cargo.lock`, `uv.lock` and `hk.pkl` are outside it.
Do not wait on a build or a CI run in the foreground; watch it in the background.

## Pushing and the CI gate

Push the branch and open the pull request; the checks on it (`lint`, `rust`, `python 3.11`,
`python 3.14`, and `wheel`) are the gate. A `cold measure` job also appears on the pull request while
the ccache branch is open: it is temporary scaffolding, it times three cold builds, and it is not
part of the gate. `check.yml` triggers on pushes to `main` only, which the
pre-push hook blocks, so a push to a branch with no pull request runs nothing; a pull request runs
against its merge ref, so it validates against current `main`. `main` takes no direct pushes; the
[README](README.md) "Contributing" section explains the hook. A new worktree's `mise.toml` is
untrusted, so its `hk` pre-push hook fails with a mise error that never names mise until
`mise trust` is run there once. Run `mise trust` in your own worktree, or push by sha from a tree
that is already trusted: `git push origin <sha>:<branch>`.

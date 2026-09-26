# Agent rules

Rules for coding agents in this repository. Each one has already been broken here at least once.

## Work in your own worktree

Every top-level agent works in a new worktree, unless it has been told otherwise. Run this from
the repository root: from a subdirectory the worktree lands inside the repository, where the
parent lane can stage it as a nested repository.

```bash
git fetch origin
git worktree add --detach ../feather--<lane> origin/main
cd ../feather--<lane>
git checkout -b <branch>
```

A subagent is exempt and shares the worktree of the agent that spawned it. "Told otherwise" means
someone named a path for you, and that tree must then be yours alone: run `git worktree list` and
stop and ask if another writer has it. Two writers never share a tree, including a tree you were
pointed at and including two subagents of one parent.

The root checkout belongs to whoever is already working in it, often mid-edit or holding conflict
markers, so no state-changing git command there: a branch switch, commit, reset, restore, stash,
clean, rebase, merge, or `git worktree add`, `git worktree remove`, `git worktree prune`.
`git worktree remove` deletes another lane's entire tree and `git clean` deletes its untracked
scratch files.

Read it only with commands that cannot write (`git show`, `git log`, `git diff`, plain file
reads). Run no build, test, watcher (`bacon.toml` defaults to clippy), language server
(`.pi-lens.json` starts pyrefly) or state-changing git command in it, because a watcher or a
language server starts the build the next section bans. Run those in your own worktree, and remove
it when you are done.

## Point worktrees at one shared target directory

A worktree gets its own `target/` unless something tells cargo otherwise, so N worktrees run the
bundled DuckDB build N times. Nothing overrides that today: no `.cargo/config.toml`, no `[env]`
table in `mise.toml`. Set it durably there, in `[env]` or in `[build] target-dir`, because an
`export CARGO_TARGET_DIR=...` reaches only the processes that inherit that shell, so it covers
neither a subagent nor your next shell.

Sharing removes the duplication, not the units. The record is one `target/` at 43G holding 12 debug
and 8 release `libduckdb-sys` units: 20 units in a single directory, which private per-worktree
directories cannot produce. Units accumulate in a directory through repeated build configurations,
and `mise.toml` names one source of churn, that rust-cache keys on the resolved rustc version.
Whether sharing is safe across worktrees holding different source trees is unresolved, and #43
exists to settle it.

Free space has fallen to within 2.4Gi of full (#43). When it drops below about 4Gi, keep the two
newest `libduckdb-sys` units in each profile of the shared directory and delete the rest, which
reclaimed 15G the last time it was needed. Name that directory rather than a relative `target/`,
which resolves to nothing once the rule above is in effect, and check that no build is running
against it first, because deleting units under a running build breaks it:

```bash
target=/path/to/a/shared/target
pgrep -l cargo                                  # a build here means wait
for p in debug release; do ls -1td "$target/$p/build/libduckdb-sys-"* | tail -n +3 | xargs rm -rf; done
```

## Do not build locally to check a change; CI is the gate

`mise run check` is the expensive one: `mise.toml` defines it as format:check, lint and test, where
lint runs `cargo clippy --workspace --all-targets --all-features` and test runs
`cargo test --workspace --all-features` plus pytest, all of it waiting on the bundled DuckDB build.
So do not run `cargo check`, `cargo clippy`, `cargo test`, `uv sync`, `pytest`, `mise run` or
Docker to check a change: the static archive alone is 1.8GB (`.github/workflows/check.yml`), the
cache entry is 2.3GB (`mise.toml`), and a cold build is roughly 10 to 15 minutes
(`crates/feather-core/Cargo.toml`). Reproducing a documented measurement
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
(`mise x --` or an activated shell); `pyrefly` needs the project `.venv` and its dev dependencies,
whose creation is the `uv sync` this section bans, so a fresh worktree has only some of these
checks. dprint owns each non-Rust, non-Python file type it has a plugin for:
JSON, TOML, YAML, Markdown and Dockerfile (`dprint.json`); `Cargo.lock`, `uv.lock` and `hk.pkl` are
outside it. Do not wait on a build or a CI run in the foreground; watch it in the background.

## Pushing and the CI gate

Push the branch and open the pull request; the four checks on it (`lint`, `rust`, `python 3.11`,
`python 3.14`) are the gate. `check.yml` triggers on pushes to `main` only, which the pre-push hook
blocks, so a branch push runs nothing, and a pull request runs against its merge ref, so it
validates against current `main`.

`main` takes no direct pushes; the [README](README.md) "Contributing" section explains the hook. A
new worktree's `mise.toml` is untrusted, so its `hk` pre-push hook fails with a mise error that
never names mise until `mise trust` is run there once. Run `mise trust` in your own worktree, or
push by sha from a tree that is already trusted: `git push origin <sha>:<branch>`.

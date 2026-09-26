# Agent rules

Rules for coding agents in this repository. Each rule below has already been broken here at least
once: the cost was thrown-away work or a full disk.

## Work in your own worktree

Every top-level agent works in a new worktree, unless it has been told otherwise. Run this from the
repository root: from a subdirectory the worktree lands inside the repository, where the parent lane
can stage it as a nested repository.

```bash
git fetch origin
git worktree add --detach ../feather--<lane> origin/main
cd ../feather--<lane>
git checkout -b <branch>
git worktree lock --reason <lane> ../feather--<lane>
```

One lane is one worktree with one writer at a time; two top-level lanes never share a tree. A parent
and the subagents it spawns are one lane, and a subagent works in its parent's tree: subagents there
may read concurrently, but only one writes at a time and the parent stops editing while one works.

A tree you were pointed at is occupied unless whoever sent you confirms it is yours alone; if the
sender cannot confirm, report back to them. Git has no liveness check, so the only signal is a lock
you declare yourself: `git worktree lock --reason <lane> <worktree>` when you create your tree;
check `locked` in `git worktree list --porcelain` before you join another;
`git worktree unlock <worktree>` before you remove one. A marker is a claim, not a detection. If
your path or branch is taken, pick another; never reuse or remove another lane's tree. Reading
another lane's tree with commands that cannot write is fine, and is how review lanes work; writing
in it is what the rule forbids.

The root checkout belongs to whoever is already working in it, often mid-edit or holding conflict
markers, so no state-changing git command there: a branch switch, commit, reset, restore, stash,
clean, rebase, merge, or `git worktree add`, `git worktree remove`, which deletes another lane's
entire tree, or `git worktree prune`, which drops a lane's registration, as `git clean` deletes its
untracked scratch files.

Read it only with commands that cannot write (`git show`, `git log`, `git diff`, plain reads). Run
no build, test, watcher (`bacon.toml` defaults to clippy) or language server (`.pi-lens.json` starts
pyrefly) in it: a watcher starts the build `Do not build locally` bans, and a language server writes
state into a tree that is not yours and needs the `.venv` that rule bars creating. Run those in your
own worktree, and remove it when you are done.

## Point worktrees at one shared target directory

A worktree gets its own `target/` unless something tells cargo otherwise, so N worktrees run the
bundled DuckDB build N times. Nothing overrides that today: there is no cargo configuration file in
the tree and no `[env]` table in `mise.toml`. Set the shared path durably, in a committed
`.cargo/config.toml` under `[build] target-dir` or in `mise.toml` under `[env]` as
`CARGO_TARGET_DIR`; an `export` covers neither a subagent nor your next shell, and the path is
machine-specific, so it is a repository-level change, not one a lane commits for itself.

Sharing removes the duplication, not the units. The record is one `target/` at 43G holding 12 debug
and 8 release `libduckdb-sys` units; keeping the two newest per profile and deleting the rest freed
15G. Units accumulate through repeated build configurations, which per-worktree duplication cannot
account for. Whether sharing is safe across worktrees with different sources is unresolved; #43
settles it. The recorded worst case was within 2.4Gi of full (#43). When free space drops below
about 4Gi, keep the two largest `libduckdb-sys` units per profile of the shared directory and delete
the rest, because finished units are gigabytes and aborted ones are megabytes. Name that directory,
not a relative `target/`, which resolves to nothing once the rule above is in effect, and check that
no build is running against it first, because deleting units under a running build breaks it:

```bash
target=/path/to/a/shared/target
pgrep -l cargo                                  # a build here means wait
for p in debug release; do du -sh "$target/$p/build/libduckdb-sys-"*/ 2>/dev/null | sort -rh | tail -n +3 | cut -f2- | xargs rm -rf; done
```

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
(`mise x --` or an activated shell); `pyrefly` needs the `.venv` whose creation is the `uv sync`
this section bans. dprint owns each non-Rust, non-Python file type it has a plugin for: JSON, TOML,
YAML, Markdown and Dockerfile (`dprint.json`); `Cargo.lock`, `uv.lock` and `hk.pkl` are outside it.
Do not wait on a build or a CI run in the foreground; watch it in the background.

## Pushing and the CI gate

Push the branch and open the pull request; the four checks on it (`lint`, `rust`, `python 3.11`,
`python 3.14`) are the gate. `check.yml` triggers on pushes to `main` only, which the pre-push hook
blocks, so a push to a branch with no pull request runs nothing; a pull request runs against its
merge ref, so it validates against current `main`. `main` takes no direct pushes; the
[README](README.md) "Contributing" section explains the hook. A new worktree's `mise.toml` is
untrusted, so its `hk` pre-push hook fails with a mise error that never names mise until
`mise trust` is run there once. Run `mise trust` in your own worktree, or push by sha from a tree
that is already trusted: `git push origin <sha>:<branch>`.

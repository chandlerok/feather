# Agent rules

Rules for coding agents working in this repository. Each one is here because it has already
been broken here, and the cost was thrown-away work or a full disk.

## Work in your own worktree

Every top-level agent works in a new worktree, unless it has been told otherwise:

```bash
git worktree add --detach ../feather--<lane> origin/main
cd ../feather--<lane>
git checkout -b <branch>
```

The checkout at the repository root belongs to whoever is already working in it, and it is
often mid-edit, mid-rebase, or holding unresolved conflict markers. Read from it freely.
Switching its branch, committing in it, or resetting it destroys another lane's work, and
neither side can tell what the other was doing.

A subagent is exempt from this rule. It shares the worktree of the agent that spawned it,
which is the reason to spawn it there. The exemption does not reach a top-level agent, and two
writers never share one tree.

### Point the worktree at a shared target directory

A worktree gets its own `target/` unless it is told otherwise, so eight worktrees run the
bundled DuckDB build eight times. Export one shared directory in every worktree:

```bash
export CARGO_TARGET_DIR=/path/to/a/shared/target
```

That omission is how `target/` reached 43G here, holding 12 debug and 8 release
`libduckdb-sys` units, and how the volume reached 100% with 2.4Gi free.

## Do not build locally; CI is the execution gate

`mise run check` runs clippy over the whole workspace, and bundled DuckDB has to build before
clippy does anything: roughly 3GB and 10 to 15 minutes, against a volume that stays near full.
Do not run `cargo check`, `cargo clippy`, `cargo test`, `uv sync`, `pytest`, `mise run`, or
Docker in order to check a change. Push it, and let the four CI jobs (`lint`, `rust`,
`python 3.11`, `python 3.14`) be the gate. A pull request runs against its merge ref, so a push
already validates the change against current `main`.

These are cheap enough to run locally in a worktree:

```bash
dprint check <files>              # via mise; dprint owns every non-Rust, non-Python file (see hk.pkl)
ruff check
ruff format --check
rustfmt --edition 2024 --check <file>
uv run --no-sync pyrefly check    # --no-sync, so nothing builds
```

Do not wait on a build or a CI run in the foreground. Watch it in the background and keep
working.

## Recover the disk before it fills

When free space drops below about 4Gi, keep the two newest `libduckdb-sys` units in each
profile and delete the rest. That reclaimed 15G the last time it was needed:

```bash
for p in debug release; do ls -1td target/$p/build/libduckdb-sys-* | tail -n +3 | xargs rm -rf; done
```

## Pushing from a fresh worktree

`main` takes no direct pushes; the [README](README.md) "Contributing" section explains the
hook. A new worktree's `mise.toml` is untrusted, so its `hk` pre-push hook fails with a mise
error that never names mise, until `mise trust` is run there once. Pushing by sha from a tree
that is already trusted avoids the trap:

```bash
git push origin <sha>:<branch>
```

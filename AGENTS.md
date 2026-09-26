# Agent rules

Rules for coding agents in this repository. Each rule below has already been broken here at least
once: the cost was thrown-away work or a full disk.

## Share one target directory if you work in several trees

This section does not say whether to use worktrees; it covers the builds either way. Two trees get
two `target/` directories unless something tells cargo otherwise, so the bundled DuckDB build runs
once per tree. Nothing overrides that today: there is no cargo configuration file in the tree and no
`[env]` table in `mise.toml`. Set the shared path durably, in a committed `.cargo/config.toml` under
`[build] target-dir` or in `mise.toml` under `[env]` as `CARGO_TARGET_DIR`; an `export` covers
neither a subagent nor your next shell, and the path is machine-specific, so it is a
repository-level change, not one a tree commits for itself. Whether sharing is safe across trees
with different sources is unresolved; #43 settles it.

Sharing removes the duplication, not the units. The record is one `target/` at 43G holding 12 debug
and 8 release `libduckdb-sys` units; keeping the two largest per profile and deleting the rest freed
15G. Units accumulate through repeated build configurations, which per-tree duplication cannot
account for. The recorded worst case was within 2.4Gi of full (#43). When free space drops below
about 4Gi, keep the two largest `libduckdb-sys` units per profile of that directory and delete the
rest, because finished units are gigabytes and aborted ones are megabytes. Name that directory, not
a relative `target/`, and check that no build is running against it first, because deleting units
under a running build breaks it:

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

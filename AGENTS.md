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

One trap, because it pauses long enough to look like a hang. `uv run` syncs the project before it
runs anything, so plain `uv run pytest` rebuilds the extension that `./setup.sh` skipped. Pass
`--no-sync` to `uv run` and the environment is used as installed. That build is a release build
with thin LTO, and DuckDB used to be the bulk of it: 62s for a cold
`cargo build -p feather-core --features offline` and 4m23s for the release wheel, measured by the
author of the prebuilt-library change, against over eight minutes and 800MB of `target/` in a fresh
worktree before it. The pause is much shorter than it was, and it is still a pause, which is why
the flag is here. It is also what `mise run test` triggers, which is one more reason that task is
not an agent's to run.

This is bootstrap, not a check, so it does not conflict with the next section. The prohibition is
on building to decide whether a change is correct.

## Give each tree its own target directory

This section does not say whether to use worktrees; it covers the builds either way. Two trees get
two `target/` directories unless something tells cargo otherwise, and nothing here does: no
`.cargo/config.toml` in the tree sets a `[build]` `target-dir`, and `mise.toml` declares no
`CARGO_TARGET_DIR`. A `.cargo/config.toml` that only carries an `[env]` table does not change
this, because cargo reads a target directory from `[build]`, not from the environment it hands
the build. Keep it that way. Pointing
several trees at one `CARGO_TARGET_DIR` saves one download of the prebuilt DuckDB library and
nothing else, because there is no C++ compile left to share. What it also does is merge the
trees' own crates, and that is not safe. Cargo leaves the absolute path out of the metadata hash on
purpose, so `feather-core` in two worktrees is one artifact: one
`libfeather_core-<hash>.rlib`, one
fingerprint, one dep-info file. Whichever tree built last owns it, and the other reports itself fresh
and links the wrong source. Two trees differing in one function, built in both orders, each produced
the other's value with no warning. It is
[cargo#12516](https://github.com/rust-lang/cargo/issues/12516), open since 2023, and two checkouts
of a workspace with path members reproduce it. Sharing is safe only while every tree sits at the same
commit, which is the moment before a lane's first edit, so it buys one download at the price of
tests that run against another lane's code. Per-tree directories cost one duplicate build, the
cheaper failure.

Units are the other half of the disk, and sharing does not touch them. The record is one `target/` at
43G holding 12 debug and 8 release `libduckdb-sys` units; the cleanup that freed 15G kept the two
newest per profile, which was the wrong rule, as the next paragraph explains. Units accumulate
through repeated build configurations, which per-tree duplication cannot account for. The recorded
worst case was within 2.4Gi of full (#43). Nothing in that build graph compiles DuckDB any more, so
the directory worth clearing today is `target/duckdb-download/`, roughly 40MB per target: it is a
download cache, safe to delete outright, and the next build re-fetches it. When free space drops
below about 4Gi, keep the two largest entries under it and delete the rest, not the two newest: a
finished download is tens of megabytes, an aborted one leaves a stub with a fresh mtime, and
newest-first therefore keeps the stubs and deletes the finished downloads. Name the directory rather
than a relative `target/`, and confirm nothing is building against it first, because deleting files
under a running build breaks it. `pgrep -l cargo` does not confirm that on its own: the compiler is
a child process, and a `rustc` here writes into the target directory, so a check that watches only
`cargo` can miss a unit still being written. A live build holds a descriptor under the target
directory, so watch the descriptors and it does not matter how the build was started:

```bash
tree=/path/to/the/tree                          # each tree has its own; never share one
target=$tree/target
if ls -l /proc/[0-9]*/cwd /proc/[0-9]*/fd 2>/dev/null | grep -q "$target"; then echo "a build is using $target; wait"; else
  du -sh "$target/duckdb-download/"*/ 2>/dev/null | sort -rh | tail -n +3 | cut -f2- | xargs -r rm -rf
fi
```

## Do not build locally to check a change; CI is the gate

`mise run check` is the expensive one: `mise.toml` defines it as format:check, lint and test, where
lint runs `cargo clippy --workspace --all-targets --all-features -- -D warnings` plus ruff and
pyrefly, and test runs `cargo test --workspace --all-features` plus pytest, all of it behind a build
of the extension and of every crate behind it. So do not run `cargo check`, `cargo clippy`, `cargo
test`, `uv sync`, `pytest`, `mise run` or Docker to check a change. The prebuilt library took the
DuckDB compile out of that build, but it is still minutes of work for a question CI answers anyway,
and a build now also needs network egress to fetch the library, which is a second reason not to
trigger one casually. The README's `mise run check` and `mise run setup` are for a human with a
machine to spare; this rule governs agents. Reproducing a documented measurement
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

Push the branch and open the pull request; the four checks on it (`lint`, `rust`, `python 3.11`,
`python 3.14`) are the gate. `check.yml` triggers on pushes to `main` only, which the pre-push hook
blocks, so a push to a branch with no pull request runs nothing; a pull request runs against its
merge ref, so it validates against current `main`. `main` takes no direct pushes; the
[README](README.md) "Contributing" section explains the hook. A new worktree's `mise.toml` is
untrusted, so its `hk` pre-push hook fails with a mise error that never names mise until
`mise trust` is run there once. Run `mise trust` in your own worktree, or push by sha from a tree
that is already trusted: `git push origin <sha>:<branch>`.

# Feather

An opinionated feature store with a Rust core and a native Python API. Online serving reads
from an embedded store, and `FeatureStore.serve()` turns the calling process into a feature
server over Arrow Flight. Historical point-in-time joins run in an embedded DuckDB engine over
Arrow.

> **Status: early implementation.** The definition layer, entity key encoding, value codec,
> and the online serving layer are built and measured, and materialization writes a view's
> values into the online store from its source. `FeatureStore.serve()` turns the calling
> process into a feature server over Arrow Flight. The point-in-time join is built over local
> Parquet, object storage, and a Postgres table, and `FeatureStore` exposes the join, the
> refresh, and the online read to Python. Warehouse sources read through an Iceberg REST
> catalog are design only, and the served endpoint does not authenticate yet. The `feather`
> command is not on PyPI either, so `pip install feather-py polars` is a line that will work
> rather than one that works today; everything after it runs.

## Design goals

- **Single configuration file.** One `feather.toml`, validated by the Rust core when it
  is loaded, so misconfiguration fails before any engine starts.
- **Zero-copy input handoff.** Frames cross the PyO3 boundary through the Arrow PyCapsule
  interface, so Rust reads a Polars or PyArrow table's own buffers without serializing them.
  A pandas frame has to be converted first, which copies. The join result is materialized as
  a new Arrow table; only the input transfer is zero-copy.
- **Point-in-time joins in-process.** An embedded DuckDB engine computes `ASOF` joins over
  local Parquet, object storage, or a table in a configured database. No separate compute
  cluster is required for local or medium-scale workloads.
- **An embedded store, and a server that owns it.** Online serving reads from a store inside the
  process, with a read-time TTL check, and `serve()` serves it over Arrow Flight. The store is a
  database that one process opens at a time, which is why the server is the only thing that opens
  it, and why a refresh and a server cannot run at once. A point read costs single-digit
  microseconds warm where a networked store pays a round trip; see
  [`docs/serving-transport.md`](docs/serving-transport.md).
- **No registry or lockfile.** Feature definitions are Python modules that both the offline
  and serving paths import directly. Git versions them; there is no generated artifact to
  drift.
- **A library, with a server you can point it at.** The language that defines the features is
  the language that reads them, in the same process, and the read path stays a single
  in-process call. `serve()` is a thin wrapper over that call rather than a second
  implementation of it, so a project in one process never pays for a hop, and a project that
  needs an RPC surface for another language gets one without changing how features are read.

## Quickstart

```bash
pip install feather-py polars
feather init my_feature_store
cd my_feature_store
feather demo
```

`feather init` generates this workspace:

```text
my_feature_store/
├── feather.toml            # infrastructure config, validated at load
└── definitions/
    └── user_clicks.py      # entities, sources, and feature views
```

`feather demo` writes the two Parquet files the view reads, into `data/`. It is a separate
command because a project that shipped with data in it would have that data deleted before
the real thing was loaded. Its rows are dated from today rather than fixed, since the view
below expires values after 30 days.

Define features in `definitions/user_clicks.py`. A field's dtype is a type argument and its
name is the attribute it is assigned to, so a reference to a feature is a checked attribute
access rather than a string lookup:

```python
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
user_stats_source = FileSource(path="data/user_stats.parquet")


@feature_view(
    name="user_clicks",
    entity=user_entity,
    source=user_stats_source,
    ttl_days=30,
)
class UserClicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)
```

Definition modules depend only on `feather` and contain no logic at import time, so serving
can load them without pulling in a dataframe library or running user code. `feather.toml`
lists the modules explicitly rather than discovering them by scanning the directory.

Run a historical join. The entity frame is passed by reference through Arrow, and the
temporal join executes inside DuckDB. The feature list takes the declared fields, so a typo
is a type error rather than a missing-column failure at runtime.

```python
import polars as pl
from feather import FeatureStore
from definitions.user_clicks import UserClicks

store = FeatureStore("feather.toml")
entity_df = pl.read_parquet("data/training_labels.parquet")

training_data = pl.DataFrame(
    store.get_historical_features(
        entity_df=entity_df,
        features=[UserClicks.click_count, UserClicks.purchase_count],
    )
)
```

The result is an Arrow table: the label set's own columns, then one column per requested
feature. `pl.DataFrame` above converts it. A label row whose value is missing or older than
`ttl_days` keeps its place holding a null, and `on_missing="drop"` removes those rows
instead. Anything exporting Arrow buffers is accepted, which a pandas frame is not until it
has been converted.

Both files are yours to supply; `feather demo` writes one plausible set, and both are
ordinary Parquet you can replace.

- `data/user_stats.parquet` is the **feature table** the view reads: one row per user per
  timestamp, holding `click_count` and `purchase_count`.
- `data/training_labels.parquet` is the **label set**: one row per user per timestamp you
  want features for, plus the column you are training on. Each label row takes the newest
  feature row for the same `user_id` at or before its own timestamp. The demo's target is
  whether the user purchased the next day, which is not one of the columns carried back,
  so the join cannot trivially give it away.

The source is declared on the view, next to the fields it feeds, and it is a file or a table in
a Postgres database named in `feather.toml`. A file source may name the format it is read as,
which defaults to Parquet, and the choice is per source, so one project can read two formats
side by side. The label set decides which rows exist and carries the target, so it is passed
per call and changes with every experiment.

## Serving the project

```bash
feather refresh
```

That writes each view's values into the online store, where `get_online_features` reads
them. It is a refresh and not an apply: there is no apply step to expose, because the core
reads `feather.toml` and the definition modules are imported rather than applied to
anything. A full refresh overwrites every value, so a run that fails can simply be repeated,
and a view a previous refresh declared and this one does not is retired by the run, which
removes its fields from every entity that still carries them. Name views to refresh a
subset: `feather refresh user_clicks`. That refreshes the project you are standing in, so
from somewhere else pass `-C`: `feather refresh -C my_feature_store user_clicks`.

A generated project declares no `[store]`, so its online store is a database at
`.feather/online` beside the `feather.toml`, which `serve()` opens. A deployment can put a
different path under `[store]`, and the same command writes there instead. A project
refreshed under the earlier behaviour has no database on disk yet, so run `feather refresh`
once after upgrading or `serve()` will open an empty store and answer null for every column.

To serve the project, name a feature service and hand that name to `serve()`:

```python
from feather import FeatureService
from definitions.user_clicks import UserClicks

ranking = FeatureService(name="ranking_v3", features=[UserClicks])

store = FeatureStore("feather.toml")
store.serve("ranking_v3", addr="127.0.0.1:8815")
```

That blocks the calling process and never returns. The process becomes the feature server: it
opens the store, resolves the service's field set once at startup, and answers Arrow Flight
requests from the same read path `get_online_features` uses. A client in any language reads the
response with its own Arrow library, so there is no generated client and no row-oriented format
on the way. Resolution happens at boot, so a service naming a view or a feature the project
does not declare fails there rather than on the first request, and a request cannot make the
server resolve metadata per call.

Two things to know before pointing it at anything. The store is a database one process opens at
a time, so the serving process is the only thing that opens it and a refresh cannot run while it
holds the directory; a deployment schedules that before anything else. And the endpoint does not
authenticate yet, so `serve()` binds loopback by default and that is the only thing protecting
it.

## Going to production

The same code runs against remote stores; only `feather.toml` changes. The configuration
schema, the query routing rules, the storage layout, and the shape of a production deployment
are described in [the architecture document](docs/rust_feature_store_architecture.md). The
engine and format calls, with the measurements behind them, are in
[the engine and format decisions](docs/engine-and-format-decisions.md), and the serving
transport in [its own record](docs/serving-transport.md).

## Contributing

`main` takes no direct pushes. Every change reaches it through a pull request, and a
pre-push hook refuses a push that would update `refs/heads/main`:

```text
$ git push origin main
Refusing to push to main. Open a pull request instead.
```

The hook is declared in `hk.pkl` and reads the refs Git hands the hook, so
`git push origin feature:main` is caught along with `git push origin main`. `mise run setup`
installs it; `mise run check` runs what CI runs. The Rust toolchain is pinned in `mise.toml`
rather than in a `rust-toolchain.toml`, so run cargo through mise, in an activated shell or as
`mise run ...`, to build with the compiler CI uses. A newer rustup default rebuilds everything
under `target/` and can accept code the pinned compiler rejects, and the resolved compiler
version is what the CI build cache is keyed on.

The hook is local. It guards only a clone where it has been installed, and
`git push --no-verify` skips it. The backstop is server-side, and it is the reason this
repository is public rather than private: GitHub offers branch protection and branch rulesets
to private repositories only on a paid plan, so while it was private there was no way to
require a check before `main` could change. `main` is now a protected branch and a pull
request has to clear all five checks (`lint`, `rust`, `python 3.11`, `python 3.14`, and
`wheel`) with the branch up to date, and force pushes and branch deletion are off. A push
that skips the hook still cannot land; it just fails after the Rust build instead of in a
second, which is what the hook is for.

## Releasing

Two artifacts are published from this repository, and only one of them is automated.

### The wheel, on PyPI (automated)

Bump `version` in `pyproject.toml`, land that, then tag `v<that same version>` and push the
tag. `.github/workflows/publish.yml` builds the four wheels and uploads them. The job checks
the version in the built wheel rather than in either file, so it holds whichever one is
behind. PyPI refuses to reuse a version, so a re-tag is not a retry: a failed upload is fixed
and released as a new version. `workflow_dispatch` builds all four wheels and skips the
upload, which is how to check the matrix without spending a tag on it.

### `feather-core`, on crates.io (manual)

The binding crates deliver a wheel, not a crate fetch, so only `feather-core` is published and
`cargo publish` is run by hand. The Cargo version is inherited from `[workspace.package]` in
the root `Cargo.toml`, and that is the version crates.io receives.

```bash
# 1. Bump [workspace.package] version in Cargo.toml and land that.
#    This is not a feather-core-only edit: crates/feather-py also inherits it. It is
#    harmless for the wheel, because maturin prefers pyproject.toml and pyproject still
#    wins, so the two can legitimately carry different versions.
# 2. Dry run first, so a packaging error costs nothing.
cargo publish -p feather-core --locked --dry-run
# 3. Publish for real.
cargo publish -p feather-core --locked
# 4. Tag the same version, which is what runs the PyPI job above.
git tag v<version> && git push origin v<version>
```

Two things about this that are easy to get wrong. crates.io does not let a name be
re-published at a version already claimed, exactly as PyPI does not, so **the first publish
claims `feather-core 0.0.1` permanently** and a failed upload cannot be retried at the same
version either. And a successful publish takes the name `feather-core` permanently, so
confirm the name is still the one you want immediately before step 3 rather than assuming it.

## License

Apache 2.0. See [LICENSE](LICENSE).

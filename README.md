# Feather

An opinionated feature store with a Rust core and a native Python API. Online serving reads
from Valkey over async `tokio` I/O; the in-process `moka` L1 cache that will sit in front of it
is design only. Historical point-in-time joins run in an embedded DuckDB engine over Arrow.

> **Status: early implementation.** The definition layer, entity key encoding, value codec,
> and the online serving layer are built and measured, and materialization writes a view's
> values into the online store from its source. The point-in-time join is built over local
> Parquet, object storage, and a Postgres table, and `FeatureStore` exposes the join, the
> refresh, and the online read to Python. Snowflake sources and Arrow Flight serving are
> design only, and the L1 cache is not built. There is no published package and no Helm chart
> yet, so the install and `init` steps below describe the intended interface; the code after
> them runs.

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
- **Two-tier online reads.** An in-process `moka` cache (L1) absorbs hot keys, invalidated
  by Valkey client-side caching rather than a fixed TTL; Valkey (L2) serves the rest over
  async `tokio` I/O. Performance targets are stated in the architecture document and are not
  yet benchmarked.
- **No registry or lockfile.** Feature definitions are Python modules that both the offline
  and serving paths import directly. Git versions them; there is no generated artifact to
  drift.

## Quickstart

```bash
pip install feather-py
feather init my_feature_store
cd my_feature_store
```

`feather init` generates this workspace:

```text
my_feature_store/
├── feather.toml            # infrastructure config, validated at load
└── definitions/
    └── user_clicks.py      # entities, sources, and feature views
```

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

Both files are yours to supply; `init` does not create sample data.

- `data/user_stats.parquet` is the **feature table** the view reads: one row per user per
  timestamp, holding `click_count` and `purchase_count`.
- `data/training_labels.parquet` is the **label set**: one row per user per timestamp you
  want features for, plus the column you are training on. Each label row takes the newest
  feature row for the same `user_id` at or before its own timestamp.

The source is declared on the view, next to the fields it feeds, and it is a Parquet file or
a table in a Postgres database named in `feather.toml`. The label set decides which
rows exist and carries the target, so it is passed per call and changes with every
experiment.

## Going to production

The same code runs against remote stores; only `feather.toml` changes. The
configuration schema, the query routing rules, and the deployment topology are described in
[the architecture document](docs/rust_feature_store_architecture.md).

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
`git push --no-verify` skips it. Server-side enforcement is not available on this
repository: it is private, and GitHub offers branch protection and branch rulesets for
private repositories only on a paid plan. Making the repository public, or upgrading to
GitHub Pro, is what would close that gap.

## License

Apache 2.0. See [LICENSE](LICENSE).

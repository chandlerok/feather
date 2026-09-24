# Feather

An opinionated feature store with a Rust core and a native Python API. Online serving
uses an in-process `moka` cache over Valkey. Historical point-in-time joins run in an
embedded DuckDB engine over Arrow.

> **Status: early implementation.** The definition layer, entity key encoding, value codec,
> and the two-tier online read path are built and measured; the point-in-time join over local
> Parquet is built and covered by a conformance suite. Warehouse sources, Arrow Flight
> serving, and materialization are design only. There is no published package and no Helm
> chart yet, so the quickstart below describes the intended interface.

## Design goals

- **Single configuration file.** One `feather.toml`, validated by the Rust core when it
  is loaded, so misconfiguration fails before any engine starts.
- **Zero-copy input handoff.** DataFrames cross the PyO3 boundary through the Arrow
  PyCapsule interface, so Rust reads Polars or pandas buffers without serializing them.
  The join result is materialized as a new Arrow table; only the input transfer is
  zero-copy.
- **Point-in-time joins in-process.** An embedded DuckDB engine computes `ASOF` joins over
  local Parquet, object storage, or a warehouse. No separate compute cluster is required
  for local or medium-scale workloads.
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

Run a historical join. The entity DataFrame is passed by reference through Arrow, and the
temporal join executes inside DuckDB. The feature list takes the declared fields, so a typo
is a type error rather than a missing-column failure at runtime.

```python
import polars as pl
from feather import FeatureStore
from definitions.user_clicks import UserClicks

store = FeatureStore("feather.toml")
entity_df = pl.read_parquet("data/training_labels.parquet")

training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[UserClicks.click_count, UserClicks.purchase_count],
)
```

Both files are yours to supply; `init` does not create sample data.

- `data/user_stats.parquet` is the **feature table** the view reads: one row per user per
  timestamp, holding `click_count` and `purchase_count`.
- `data/training_labels.parquet` is the **label set**: one row per user per timestamp you
  want features for, plus the column you are training on. Each label row takes the newest
  feature row for the same `user_id` at or before its own timestamp.

The source is declared on the view, next to the fields it feeds. The label set decides which
rows exist and carries the target, so it is passed per call and changes with every
experiment.

## Going to production

The same code runs against remote stores; only `feather.toml` changes. The
configuration schema, the query routing rules, and the deployment topology are described in
[the architecture document](docs/rust_feature_store_architecture.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

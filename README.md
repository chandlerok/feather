# Feather

An opinionated feature store with a Rust core and a native Python API. Online serving
uses an in-process `moka` cache over Valkey. Historical point-in-time joins run in an
embedded DuckDB engine over Arrow.

> **Status: design phase.** This repository currently contains the design documents only.
> There is no implementation, no published package, and no Helm chart yet. The quickstart
> below describes the intended interface.

## Design goals

- **Single configuration file.** One `feature_store.yaml`, validated by Pydantic v2 at the
  Python boundary, so misconfiguration fails before any engine starts.
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
pip install feather-store
feather init my_feature_store
cd my_feature_store
```

`feather init` generates this workspace:

```text
my_feature_store/
├── feature_store.yaml      # infrastructure config, validated by Pydantic v2
└── definitions/
    └── user_clicks.py      # entities, sources, and feature views
```

Define features in `definitions/user_clicks.py`. Fields carry real dtypes, so the schema is
explicit rather than inferred from bare strings:

```python
from feather import Entity, FeatureView, Field, FileSource
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
user_stats_source = FileSource(path="data/user_stats.parquet")

user_clicks_view = FeatureView(
    name="user_clicks",
    entities=[user_entity],
    source=user_stats_source,
    features=[
        Field("click_count", dtype=Int64),
        Field("purchase_count", dtype=Int64),
    ],
    ttl_days=30,
)
```

Definition modules depend only on `feather` and contain no logic at import time, so serving
can load them without pulling in a dataframe library or running user code. `feature_store.yaml`
lists the modules explicitly rather than discovering them by scanning the directory.

Run a historical join. The entity DataFrame is passed by reference through Arrow, and the
temporal join executes inside DuckDB.

```python
import polars as pl
from feather import FeatureStore

store = FeatureStore("feature_store.yaml")
entity_df = pl.read_parquet("data/training_labels.parquet")

training_data = store.get_historical_features(
    entity_df=entity_df,
    features=["user_clicks:click_count", "user_clicks:purchase_count"],
)
```

`data/user_stats.parquet` and `data/training_labels.parquet` are yours to supply; `init`
does not create sample data.

## Going to production

The same code runs against remote stores; only `feature_store.yaml` changes. The
configuration schema, the query routing rules, and the deployment topology are described in
[the architecture document](docs/rust_feature_store_architecture.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

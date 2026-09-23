# 🪶 Feather

**The featherweight, ultra-fast feature store built for the modern data stack.**

Feather is an opinionated, lightning-fast open-source feature store implemented in **Rust** with a native **Python API**. By stripping away the architectural bloat of legacy feature stores, Feather delivers sub-millisecond online serving and blazing-fast historical joins by leveraging **DuckDB** and **Valkey** natively.

[![PyPI Version](https://shields.io)](https://pypi.org)
[![License](https://shields.io)](https://opensource.org)

## ⚡ Why Feather?

Traditional feature stores force you to manage heavy JVM infrastructure, write complex configuration abstractions, and suffer from high Python serving latency. Feather changes the paradigm:

* **Zero-Copy Performance:** Driven by a core Rust engine using the Apache Arrow C Data Interface. Dataframes move between Python and Rust with zero memory copying.
* **Streamlined Offline Joins:** Uses an embedded **DuckDB** engine to compute historically accurate point-in-time (`ASOF`) joins on Parquet, CSV, or remote data lakes (Snowflake, BigQuery, MotherDuck) over high-speed ADBC.
* **Sub-Millisecond Serving:** Built for production workloads using a dual-tier cache—an in-process Rust `moka` cache (L1) coupled with **Valkey** (L2) over asynchronous `tokio` threads.
* **Type-Safe Ergonomics:** Configured entirely via a single `feature_store.yaml` file validated instantly at the Python boundary using **Pydantic v2**.

---

## 🚀 3-Minute Local Setup

Feather requires zero external infrastructure to get started locally. Pre-compiled Rust wheels mean you don't even need a Rust toolchain installed.

### 1. Install Feather
```bash
pip install feather-store
```

### 2. Initialize a Project
```bash
feather init my_feature_store
cd my_feature_store
```

This generates a lean, declarative workspace:
```text
my_feature_store/
├── feature_store.yaml      # Unified Pydantic configuration file
└── features.py             # Feature views and data sources
```

### 3. Define Your Features (`features.py`)
```python
from feather import Entity, FeatureView, FileSource

user_entity = Entity(name="user_id", join_key="user_id")
user_stats_source = FileSource(path="data/user_stats.parquet")

user_clicks_view = FeatureView(
    name="user_clicks",
    entities=[user_entity],
    source=user_stats_source,
    features=["click_count", "purchase_count"],
    ttl_days=30
)
```

### 4. Run a Historical Join (Offline)
Pass your training labels directly. Feather shares memory with your DataFrame via Arrow and executes the join inside DuckDB instantly.

```python
import polars as pl
from feather import FeatureStore

store = FeatureStore("feature_store.yaml")
entity_df = pl.read_parquet("data/training_labels.parquet") 

# Blazing-fast temporal join executed in Rust
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=["user_clicks:click_count", "user_clicks:purchase_count"]
)
```

---

## ☸️ Scaling to Production

Moving from a local script to a hardened cloud environment requires **zero code changes**. Simply swap your target configurations in `feature_store.yaml`:

```yaml
project: ad_recommendations
registry: md:prod_feature_registry  # Centrally stored in MotherDuck

offline_store:
  type: snowflake
  config:
    account: "xy12345.us-east-1"
    warehouse: "PROD_WH"

online_store:
  type: valkey
  config:
    endpoint: "valkey-cluster.internal.svc:6379"
    l1_cache:
      max_capacity_mb: 2048
      ttl_seconds: 300
```

### Cloud Architecture Overview

Feather ships with native Kubernetes manifests and Helm charts to deploy two optimized workload profiles automatically:

1. **The Offline Worker Pool:** Heavy CPU/Memory ephemeral containers that trigger on-demand or cron-scheduled `materialization` batches (syncing historical sources to Valkey).
2. **The Online Serving Service:** Lightweight, auto-scaling microservices running Feather's high-performance gRPC/REST server to supply real-time vectors to your inference models under heavy traffic load.

---

## 📄 License
Feather is open-source software licensed under the [Apache 2.0 License](LICENSE).

# Architectural Design Document: Opinionated, Ultra-Performant Rust Feature Store

This document outlines the production-ready technical architecture for an opinionated, high-performance open-source alternative to Feast. By stripping away multi-provider abstractions, this system leverages a specialized, native **Rust core** paired with **DuckDB** and **Valkey** to maximize throughput, minimize infrastructure friction, and deliver exceptional developer ergonomics.

---

## 🏛️ System Architecture Overview

The platform uses a unified codebase but enforces a strict division of labor between heavy, asynchronous historical processing (**Offline Engine**) and low-latency, real-time feature lookup (**Online Serving Layer**).

```
                      ┌──────────────────────────────────────────┐
                      │          Python Client Layer             │
                      │   (Polars / Pandas / PyArrow / Python)   │
                      └────────────────────┬─────────────────────┘
                                           │ (Zero-copy Arrow PyCapsule)
                      ┌────────────────────▼─────────────────────┐
                      │       Maturin / PyO3 API Bindings        │
                      └────────────────────┬─────────────────────┘
                                           │
                      ┌────────────────────▼─────────────────────┐
                      │               Rust Core                  │
                      └───────────┬──────────────────┬───────────┘
                                  │                  │
    (Tiered Offline Queries)      │                  │   (Async Batch Write / Read)
      ┌───────────────────────────▼──┐            ┌──▼───────────────────────────┐
      │        DuckDB Engine         │            │   Two-Tier Online Store      │
      │  (Parquet / GCS / S3 / MD)   │            │   - L1: Rust moka Cache      │
      │  & Native Warehouse Drivers  │            │   - L2: Distributed Valkey   │
      └──────────────────────────────┘            └──────────────────────────────┘
```

### Core Design Tenets
1. **Performance First:** Built using native system languages (Rust) and vectorized memory layouts (Apache Arrow).
2. **Ease of Setup:** Zero-infrastructure local mode. A user can run full point-in-time joins and online lookups out of the box with zero external dependencies.
3. **Opinionated Simplification:** Strict standardization around Valkey for online serving, DuckDB for local/medium cloud data compute, and Apache Arrow as the universal internal language.

---

## 🎛️ Execution Layer Details

### 1. Python & Rust Boundary (Maturin + Pydantic v2)
The user interface is exposed as a native Python extension package compiled via **Maturin** and **PyO3**.
* **Type Safety:** **Pydantic v2** models validate all user input, environment schemas, and `feature_store.yaml` files inside the Python runtime. Misconfigured settings trigger immediate, readable errors before spinning up system resources.
* **Serialization Handoff:** Python objects pass data schemas directly to the Rust core using Pydantic’s high-speed Rust serialization engine via `.model_dump_json()`. The Rust backend deserializes it instantly using `serde_json`.
* **Zero-Copy Memory:** When DataFrames are submitted for Point-in-Time (PIT) joins, data is passed across the FFI boundary using the **Arrow PyCapsule Interface**. This allows Rust to read Polars or Pandas memory addresses natively with **0% serialization overhead and a 0MB memory allocation footprint**.

### 2. The Online Serving Layer (Two-Tier Hybrid Cache)
To support ultra-low latency serving at scale, the online engine splits lookups into an in-process and a distributed tier:
* **L1 Cache (In-Process):** Implemented directly inside the Rust client library using the **`moka`** crate. It features a concurrent TinyLFU eviction policy providing **sub-microsecond** feature lookups for hot keys, fully bypassing network bounds.
* **L2 Cache (Distributed online store):** Powered by **Valkey** (the open-source successor to Redis). On an L1 cache miss, the Rust client asynchronously reaches out to Valkey using non-blocking I/O driven by the **`tokio`** async runtime, populates L1, and returns the vector.

### 3. The Tiered Offline Engine (Smart Query Routing)
To circumvent the network egress and computing limitations of single-node data streaming, the Rust engine automatically routes historical queries into three distinct execution tiers based on data scale:

#### 🔹 Tier 1: Local / Medium Cloud Scale (<100GB)
* **Engine:** Local DuckDB instance via the `duckdb` Rust crate.
* **Mechanism:** Leverages DuckDB extensions (`snowflake`, `bigquery`) using Apache Arrow ADBC drivers. DuckDB pushes down column projections and basic filter predicates (`WHERE country = 'US'`) to the data warehouse. The filtered Arrow record batches stream directly into local memory/NVMe swap space where DuckDB processes the high-speed `ASOF JOIN`.

#### 🔹 Tier 2: Warehouse Native Scale (>100GB, Single Cloud Warehouse)
* **Engine:** Native Target Database Compute (Snowflake / BigQuery / Azure).
* **Mechanism:** If the Rust query planner detects that both the entity source dataframe and the historical feature views reside natively in the same cloud warehouse, **local DuckDB processing is bypassed**. The Rust core uses direct warehouse client drivers to compile and submit native temporal SQL statements. The join is computed entirely on the remote enterprise cluster, downloading *only the final, highly reduced training dataset*.

#### 🔹 Tier 3: Open Data Lakehouse Scale (Apache Iceberg / Delta Lake)
* **Engine:** DuckDB + Cloud Object Storage.
* **Mechanism:** For modern data lakes (Parquet/Iceberg arrays on AWS S3, GCS, or Azure Blob), the engine uses DuckDB's native `iceberg` and `httpfs` extensions. It utilizes HTTP range requests to selectively scan files in the storage buckets, executing client-side execution without incurring data warehouse compute charges.

---

## 🛠️ Configuration & Developer Ergonomics

The entire runtime is defined using a single, unified declarative file.

### Production Manifest Specification (`feature_store.yaml`)
```yaml
project: ad_recommendations
registry: md:prod_feature_registry  # Centrally stored metadata in MotherDuck

offline_store:
  type: snowflake
  config:
    account: "xy12345.us-east-1"
    warehouse: "PROD_WH"
    username: "fs_runner"

online_store:
  type: valkey
  config:
    endpoint: "valkey-cluster.internal.svc:6379"
    l1_cache:
      enabled: true
      max_capacity_mb: 2048
      ttl_seconds: 300
```

---

## ☸️ Infrastructure & Deployment

To balance smooth developer onboarding with hardened enterprise scaling, the system abstracts configuration across environment targets:

1. **Local Development (Zero-Infra Loop):** A developer installs the package via `pip install native-feature-store`. The pipeline runs purely in-process using local Parquet files, an in-memory DuckDB engine, and an embedded L1 memory cache.
2. **Production Deployment (Kubernetes Native):** System architecture files ship with pre-baked **Helm charts** and **Kustomize manifests**. 
   * **The Scaling Blueprint:** When pushed to Kubernetes, the system deploys a StatefulSet of **Valkey cluster nodes** alongside a horizontally scaled deployment of **Rust API pods** exposing gRPC/REST serving boundaries. Historical materialization jobs are dispatched cleanly as ephemeral, resource-isolated **Kubernetes Jobs**.
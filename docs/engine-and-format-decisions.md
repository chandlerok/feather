# Engine and format decisions

This file records what was evaluated for the offline engine and the source formats, with the
numbers behind each call, so the same ground is not covered twice. The design itself is in
[the architecture document](./rust_feature_store_architecture.md); this is the evidence and the
rules that follow from it.

Everything here is dated. DuckDB, the extension tiers, and the newer formats all move, so a
claim below carries the version it was checked against.

## The engine stays DuckDB

Four alternatives were taken far enough to run feather's own conformance cases: the twelve in
the architecture document's "Conformance" section, plus the materialization reduction and a
Parquet source read.

| Option                           | Cases    | Verdict                                         |
| -------------------------------- | -------- | ----------------------------------------------- |
| DuckDB (current)                 | 12 of 12 | Kept.                                           |
| Apache DataFusion                | not run  | `ASOF JOIN` is unreleased.                      |
| Polars 0.55.2                    | 21 of 21 | Rejected: no arrow-rs boundary.                 |
| Embedded ClickHouse (chDB 4.4.0) | 22 of 22 | Rejected: size, no Delta, experimental binding. |
| Hand-rolled join over Arrow      | 23 of 23 | Viable, not adopted.                            |

### Apache DataFusion

`ASOF JOIN` was requested in 2021 and closed on 2026-09-15, but it is not in a release. The
newest published crate is 55.1.0 (2026-09-11), and ASOF is slated for 56.0.0 in October 2026.

More importantly, the documented execution strategy requires the **complete right input to fit
in memory**, and spilling and repartitioned ASOF execution are not supported. In feather's join
the right side is the feature source, which is the large side. That is a regression against a
problem the architecture document already records, on the operator the product is built around.

Two secondary problems. There is no Snowflake or BigQuery table provider, so a warehouse source
would have to be written. And the Arrow majors do not line up: DataFusion 55.1.0 wants arrow
`^59.2`, 56 moves to 60, and feather is pinned to 58 by `duckdb-rs` and `pyo3-arrow`.

### Polars

Polars passes, and its API is a closer fit than DuckDB's. `join_asof_by` takes
`left_on`, `right_on`, `left_by`, `right_by`, `AsofStrategy::Backward`, `tolerance`, `allow_eq`,
and `check_sortedness`, where `tolerance` is the TTL window and `allow_eq = false` is `strict`.
The three things that exist in feather only because SQL's `ASOF JOIN` permits one inequality,
the `QUALIFY row_number()` pre-dedup, the post-join TTL predicate, and the `CASE WHEN` that nulls
expired values, all collapse into arguments. Grouped ASOF in the streaming engine landed in
April 2026.

The blocker is the boundary, not the join. **Polars 0.55 has no arrow-rs dependency.** Its
`arrow` dependency is `package = "polars-arrow"`, its own implementation, and there is no
`from_arrow` or `to_arrow` on `DataFrame`. Interop is `polars_arrow::ffi`, the Arrow C Data
Interface. Feather's zero-copy arrow-rs handoff becomes a C-Data-Interface hop, which is a
direct cost against a stated design goal. The build is also 1.6 GB of target, a 156 MB debug
binary, and about 9.5 minutes cold.

### Embedded ClickHouse (chDB)

chDB passes everything, including the materialization reduction and a Parquet read.

Two problems. **`join_use_nulls` defaults to 0**, so an unmatched `ASOF LEFT JOIN` column comes
back as the type default. An unmatched label silently becomes `0` rather than `NULL`, which for
a feature store is a wrong feature value rather than a missing one. Verified on the raw join:
`[0]` with the default and `[None]` with `SETTINGS join_use_nulls = 1`. The setting is mandatory,
and forgetting it produces no error.

**There is no `deltaLake` table function in the embedded build.** `system.table_functions` lists
`s3`, `url`, `postgresql`, `mysql`, `sqlite`, `iceberg`, `icebergS3`, `icebergLocal`, `hudi`,
`paimon`, `azureBlobStorage`, `gcs`, `mongodb`, `odbc`, `jdbc`, `redis`, `arrowstream` and
`input`, but Delta, which is half of the tier 2 requirement, is absent.

The artifact is 326 MB of `libchdb.so` dynamically or 490 MB statically. `chdb-rust` is
self-labelled experimental, and its build dependencies download `libchdb` over the network.

### Hand-rolled join over Arrow

The join is smaller than expected. Two dependencies (`arrow` and `parquet`), a 14 MB release
binary, and 241 lines covering source preparation, the merge, the point-in-time join, and the
materialization reduction.

Measured on the same machine and the same shape as `examples/duckdb_input.rs` (500k labels
against a 2M-row feature table):

|                                                        | Time    |
| ------------------------------------------------------ | ------- |
| DuckDB, `count(*)` over the ASOF join                  | 34.4 ms |
| Hand-rolled merge plus materializing the output column | 27.9 ms |

DuckDB's figure excludes materializing result columns, so the operator is competitive rather
than merely comparable. End to end the hand-rolled version is slower, at about 210 ms, and the
reason is the sort: source preparation is 153 ms, the label sort 26 ms, and sorting the 2M-row
source costs 37 ms for one column against 117 ms for two. DuckDB's hash-based ASOF avoids that
global sort.

The gaps are real: no out-of-core path, only a Parquet reader, and narrow type coverage
(`Int64`/`Int32`/`Utf8` keys, `Int64`/`Timestamp(us)` timestamps).

### Why DuckDB anyway

Because the operator is not what DuckDB is buying. The operator is 241 lines and can be
replaced; the readers cannot be replaced cheaply. Every alternative above either loses a
source, loses the Arrow boundary, or costs an engine's worth of binary and build time. Keeping
DuckDB keeps Parquet, S3, Postgres, Iceberg, Delta, DuckLake and the warehouse path reachable
from one dependency.

This is a decision about breadth, and it should be revisited if the source list in the
architecture document ever shrinks on purpose.

## Format rules

### Support tiers decide the default

DuckDB publishes a support tier per core extension. Primary extensions are covered by community
support; Secondary extensions are best-effort, still bugfixed and shipped with each release. The
tier is what makes a format safe to default to, more than any benchmark.

| Tier                   | Extensions                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Primary**            | `parquet`, `httpfs`, `icu`, `json`                                                                                      |
| Secondary              | `postgres`, `iceberg`, `delta`, `ducklake`, `azure`, `mysql`, `sqlite`, `unity_catalog`, `avro`, `vss`, `tpch`, `tpcds` |
| Third-party maintained | `vortex`, `lance`, `motherduck`                                                                                         |

So **Parquet on object storage, `parquet` plus `httpfs`, is the only fully supported read path**,
and it is the default for anything feather writes or suggests. Every other format is offered,
but marked for what it is.

### Layout beats format

Sort and partition a feature source by `(entity key, event_timestamp)`. That is what lets DuckDB
skip row groups, or storage segments under Vortex, for the key range instead of scanning
everything. **A sorted Parquet beats an unsorted Vortex.** This costs nothing in dependencies,
is format-independent, and is the highest-leverage change available without changing engines.

If feather recommends one thing to users about how to land feature data, it should be this.

### Iceberg and Delta

Interop for users who already have a lakehouse, not defaults. Both are Secondary. Iceberg
supports snapshot pinning and full writes through a REST catalog; Delta's writes in DuckDB are
blind append only.

### DuckLake

[DuckLake](https://ducklake.select/) 1.0 shipped 2026-04-13. It keeps metadata in a SQL database
(PostgreSQL, MySQL, SQLite or DuckDB) and data in plain Parquet on object storage, with no
catalog server to run. It provides snapshots, time travel, schema evolution, partitioning, ACID
over multi-table operations, and statistics-based filter pushdown.

It is the recommendation for anyone currently pointing a feature view at a Postgres table, and
its Parquet files are Iceberg-compatible, so it is not a one-way door. It is Secondary tier, but
maintained by the DuckDB team.

### Vortex

Opt-in, not a default. A DuckDB core extension since January 2026, with `read_vortex(...)` and
`COPY ... TO 'x' (FORMAT vortex)`. Its mechanism is late decompression and evaluating filters on
compressed data, which suits a scan-heavy point-in-time join because the key and timestamp
predicates can be applied before anything is decompressed.

DuckDB's own TPC-H SF100 measurement puts it **18% ahead of Parquet v2 and 35% ahead of v1**
(geometric mean, Mac M1, 10 cores, 32 GB, single file per table).

The reason it is opt-in: it is third-party maintained rather than tiered, pre-1.0 (0.86.1 in
September 2026 against 0.56.0 in January), and it pins to the DuckDB release like every
extension.

### Lance

Not a candidate. Third-party maintained, no `osx_amd64` build, and shaped for vector and
multimodal data rather than key-and-timestamp scans.

## Warehouses are read as Iceberg

Snowflake, BigQuery, Databricks, Amazon S3 Tables and Fabric all expose an **Iceberg REST
catalog** now, so a warehouse is reached through the open table format rather than a
vendor-specific driver.

### One integration, not three

```sql
INSTALL iceberg;
CREATE SECRET iceberg_secret (
    TYPE iceberg,
    CLIENT_ID '...',
    CLIENT_SECRET '...',
    OAUTH2_SERVER_URI 'https://catalog.example.com/v1/oauth/tokens'
);
ATTACH 'warehouse' AS my_catalog (
    TYPE iceberg,
    SECRET iceberg_secret,
    ENDPOINT 'https://catalog.example.com'
);
SELECT count(*) FROM my_catalog.default.events;
```

The catalog then behaves like any other DuckDB database. The endpoints that matter:

- **BigQuery**: the BigLake metastore speaks the REST catalog protocol and is generally
  available at `https://biglake.googleapis.com/iceberg/v1/restcatalog`.
- **Snowflake**: Iceberg tables with a remote Iceberg REST catalog (AWS Glue, Snowflake Open
  Catalog), plus catalog-linked databases that sync namespaces and tables from a remote catalog.
- **Amazon S3 Tables**: `ATTACH '<arn>' AS cat (TYPE iceberg, ENDPOINT_TYPE s3_tables)`.

A single source kind, `Source::Iceberg { endpoint, secret_ref, table }`, covers all of them
instead of three vendor-shaped kinds. It is also the integration tier 2 already needs.

### What it gives us

`iceberg_snapshots(my_catalog.default.events)` works on catalog tables, so a warehouse source
can record a snapshot id like every other source. That closes the one source kind that had no
pin, which is the requirement issue #2 already states.

### The performance play

```sql
ATTACH 'ducklake:my_ducklake.ducklake' AS my_ducklake;
CALL iceberg_to_ducklake('my_catalog', 'my_ducklake');
```

`iceberg_to_ducklake` is a metadata-only copy of an attached Iceberg catalog into a DuckLake
catalog. No data moves. The warehouse's tables become queryable as DuckLake tables, so repeated
reads inherit DuckLake's statistics-based filter pushdown and snapshot semantics instead of
paying the remote catalog's latency on every request.

The shape is: warehouse is the system of record, Iceberg is the wire, DuckLake is the local
read path.

### Fallbacks

The vendor extensions remain, for warehouses and tables with no Iceberg surface: `snowflake`
through ADBC, `bigquery`, `onelake` for Fabric, and the generic `adbc` extension for anything
with an ADBC driver. All of them are **community extensions**, so they are rebuilt per DuckDB
release and only the latest release is listed. An extension that lags fails to load.

One concrete gotcha to document: DuckDB against Snowflake Open Catalog requires credential
vending to be enabled, or `ACCESS_DELEGATION_MODE 'none'` plus an S3 secret of your own.
Namespace depth, AWS region and secret URL-encoding are all fiddly.

### Azure

The weakest of the three and worth being plain about. The `azure` extension is Blob Storage
filesystem access, which makes `az://` paths work, and it is not a warehouse. Synapse and
Databricks on Azure go through Iceberg or ADBC. Fabric goes through the `onelake` community
extension. Azure support therefore means ADLS paths plus Iceberg, not a Synapse connector.

### Not doing

**No warehouse-native execution tier.** Reading a warehouse's Iceberg is not running the join on
the warehouse. Compute stays local, which is the point, and BigQuery having no `ASOF JOIN`
remains the standing proof that the join rules cannot be pushed there without a second
implementation.

**No warehouse as a default source.** Every warehouse path is Secondary or community tier, so a
warehouse source is best-effort, with the snapshot id recorded and a metadata-only drift check.

## Reproducing these

Each spike was a scratch crate or virtualenv outside the repository, so none of them are
committed. They are small enough to rebuild when a claim needs rechecking.

- **DuckDB**: in this repository, `ENTITIES=500000 cargo run --release -p feather-core
  --features offline --example duckdb_input`. That is where the 34.4 ms figure comes from.
- **Hand-rolled join**: a crate depending only on `arrow` 58 and `parquet` 58, with the merge as
  a single monotonic cursor over a source sorted by `(key, ts, created)`. The `arrow` API details
  that cost the most time were `lexsort_to_indices(&[SortColumn], limit)` and the fact that
  `take` treats a null index as a null row.
- **Polars**: a crate depending on `polars` 0.55.2 with the `lazy`, `asof_join`, `streaming` and
  `parquet` features, running the cases through `join_asof_by(..., AsofStrategy::Backward,
  tolerance, allow_eq, check_sortedness)`. `DataFrame::unique` needs a turbofish because its
  generic parameters are unused; `unique_stable` does not.
- **chDB**: `pip install chdb` (4.4.0, engine 26.7.2), with each case as a query over the
  `values(...)` table function and `SETTINGS join_use_nulls = 1` on every one.
- **DataFusion**: not built, and the row above says so. That finding comes from the tracker, the
  release notes, and the published crate metadata rather than from a run.

The extension tiers, the Vortex TPC-H result, and the Iceberg REST endpoints all come from
DuckDB's and the vendors' own documentation, not from a measurement here, and they are the parts
most likely to have moved. Recheck them against the release actually pinned.

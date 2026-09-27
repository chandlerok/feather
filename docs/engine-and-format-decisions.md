# Engine and format decisions

This file records what was evaluated for the offline engine and the source formats, with the
numbers behind each call, so the same ground is not covered twice. The design itself is in
[the architecture document](./rust_feature_store_architecture.md); this is the evidence and the
rules that follow from it.

Everything here is dated. DuckDB, the extension tiers, and the newer formats all move, so a
claim below carries the version it was checked against, or says it comes from documentation
rather than a check here.

## The engine stays DuckDB

Three alternatives were taken far enough to run feather's own conformance cases, the seven the
architecture document's "Conformance" section names. DataFusion was not built; its row is
assessed from its issue tracker, release notes, and published crate metadata.

| Option                                          | Cases   | Verdict                                         |
| ----------------------------------------------- | ------- | ----------------------------------------------- |
| DuckDB (current, the pinned `duckdb` 1.10505.0) | 7 of 7  | Kept.                                           |
| Apache DataFusion                               | not run | `ASOF JOIN` is unreleased.                      |
| Polars 0.55.2                                   | 7 of 7  | Rejected: no arrow-rs boundary.                 |
| Embedded ClickHouse (chDB 4.4.0)                | 7 of 7  | Rejected: size, no Delta, experimental binding. |
| Hand-rolled join over Arrow                     | 7 of 7  | Viable, not adopted.                            |

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

chDB passes all seven, and was also run on the materialization reduction and a Parquet read.

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

Measured on an Apple M2 with 8 cores and 8 GiB, the same machine and shape as
`examples/duckdb_input.rs` (500k labels against a 2M-row feature table):

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
tier is what makes a format safe to default to, more than any benchmark. The table below is
taken from DuckDB's published extension documentation rather than checked here.

| Tier                   | Extensions                                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Primary**            | `parquet`, `httpfs`, `icu`, `json`                                                                                      |
| Secondary              | `postgres`, `iceberg`, `delta`, `ducklake`, `azure`, `mysql`, `sqlite`, `unity_catalog`, `avro`, `vss`, `tpch`, `tpcds` |
| Third-party maintained | `vortex`, `lance`, `motherduck`                                                                                         |

So within DuckDB's tiers, **Parquet on object storage, `parquet` plus `httpfs`, is the only
fully supported read path**, and it is the format a source gets when it does not name one.
The Postgres reader is a table scan rather than a file format (see "Backend coverage" in the
architecture document), so a format never applies to it.

### The format is a per-source choice

A `File` source carries an optional `format`, and it resolves independently per source, so one
project can read Parquet and Vortex side by side. The default is the Primary format, Parquet, and
that is the only reason it is the default: the tier table above is what makes it defensible
rather than a preference, and a format added later cannot become the default by being added.

| `format`  | Read by             | Tier                 | Extension                   |
| --------- | ------------------- | -------------------- | --------------------------- |
| omitted   | `read_parquet(...)` | Primary              | compiled into the build     |
| `parquet` | `read_parquet(...)` | Primary              | compiled into the build     |
| `vortex`  | `read_vortex(...)`  | Third-party, pre-1.0 | installed on the first read |

`Engine::relation` is where the format becomes a table function, and
`Engine::ensure_source_loaded` is where its extension is installed, so a format added later
changes those two and nothing else. Vortex is installed on the first read rather than at open,
on the same deferral as `httpfs` and `postgres`, so a project that names no non-default format
never touches the extension repository. Nothing in Feather is compiled against it.

**The tier is declared in the code, on the format, and in neither `feather.toml` nor the
definition module.** It is a fact about DuckDB's extension rather than about a project: the
answer is the same everywhere, so both files would only be somewhere for a project to assert a
tier that could be wrong. `feather.toml` holds what a deployment is made of, and a support tier
is not infrastructure. The definition module holds what the data is, and a Vortex file is a
Vortex file whatever tier its reader carries. `FileFormat::tier` in
`crates/feather-core/src/definitions.rs` is the single place, and a load failure for a
non-Primary format names it so an operator reading the error knows they are on a best-effort
or third-party path.

An unknown format is a load error, not a fallback:

```text
view `user_clicks` reads source `data/clicks.vortex` in format `vortx`, which Feather does
not read; the formats are parquet, vortex
```

The alternative, silently reading it as Parquet, fails much later as a decode error with
nothing near the name that caused it. The error names the view, the path, and the value, which
is why `format` is carried on the source as a string and resolved by the loader rather than
being a typed field that serde would reject as a bare "unknown variant".

### Layout beats format

**Sort and partition a feature source by `(entity key, event_timestamp)`.** The recommendation
stands, and it is the one thing feather should tell users about how to land feature data: it
costs nothing in dependencies, it is format-independent, and it makes the point-in-time join
read 31.5% fewer bytes. The byte counts in the table below are measured; the join times are
not, and the table marks which is which.

**The reason the first draft gave for it was wrong**, and the correction matters to anyone
choosing a layout. The claim was that sorting lets DuckDB skip row groups for the key range.
It does not, for this join. With 8 of 16 row groups wholly outside the label keys, the sorted
file's bytes read are the same _to the byte_ as when the range covers every key. An `ASOF
JOIN` reads its build side before it knows any label keys, so a dynamic filter arrives after
the scan has already read.

Measured with `examples/layout.rs`: one generated 2M-row feature table over 1M distinct keys,
written twice with the same writer and the same 131072-row row groups, differing only in row
order. `unsorted` is a deterministic permutation of the same rows, so no row group has local
key structure; `sorted` is ordered by `(user_id, event_timestamp)`. 500k labels, minimum of
seven runs, DuckDB 1.105.05. `half` labels are over the lower half of the key space (stride 1)
and `full` over every other key (stride 2), so `full` reaches the top of the key space without
covering all of it.

| Figure                              | unsorted     | sorted       | sorted wins by |
| ----------------------------------- | ------------ | ------------ | -------------- |
| file size                           | 20,887,564 B | 14,495,033 B | 30.6%          |
| bytes read by the join              | 20,545,535 B | 14,068,158 B | 31.5%          |
| row groups wholly out of range      | 0 of 16      | 8 of 16      | —              |
| join, labels over half the keys †   | 174.4 ms     | 138.9 ms     | not measured † |
| join, labels over every other key † | 178.9 ms     | 142.4 ms     | not measured † |

**† The two join times are stale and must be re-measured before they are quoted.** The run
that produced them gave the label frame timestamps that fell below the feature rows' timestamps
for their key, so a single label of 500,000 matched anything: the figure is a
decode-and-build-side-sort ratio, not a point-in-time join. No run of the fixed program has been
made on this machine, so no replacement figure exists yet and the ratio is not stated anywhere
in this document. The bytes and the file sizes are decode-side and do not depend on the
predicate, which is why they stand; the example now places the label timestamps inside the
feature range and asserts that every label matched, so the next run measures a join that joins.

Every byte count here reproduced exactly across three runs, and these times come from a run on
an otherwise idle machine, which is why the absolute times are not comparable with anything
measured elsewhere. That run is the stale one above, so there is no join ratio to quote from it:
the layout's effect on join time is the open question this table leaves, and the next run of
`examples/layout.rs` answers it. Quote the byte counts, which are what stands.

What the layout actually buys is compression, not pruning. Clustered keys are the same value
over and over, which is what dictionary encoding is for, so the file is 30.6% smaller and there
are 31.5% fewer bytes to decode. That is what the byte counts show on their own: the two label
ranges read the same amount from each file, and the sorted file's row groups are not skipped
from the join at all, which is the opposite of what row group skipping would produce.

**Pruning is real, and it needs a static predicate rather than a join.** The same sorted file
under `WHERE user_id <= 499999` reads 3,150,108 B against the unsorted file's 10,008,816 B,
3.18x, which is about what skipping 8 of 16 row groups predicts. The unsorted file can skip
none, because every one of its row groups spans the whole key space. Both figures reproduced
exactly. So for a scan that filters on the key, sorting is worth even more than the table above
says, and it is the same recommendation for the same underlying reason.

**A sorted Parquet beats an unsorted Vortex**, still, but for the reason above: clustered keys
compress, and 30% fewer bytes beats any format's decoder.

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

**Status: intended, not built.** No warehouse reader exists today: `Source` has `File` and
`Postgres`, no Iceberg feature is compiled, and the architecture document's "Backend coverage"
paragraph is the record of what is built. What follows is the intended mechanism, written in
the future tense.

Snowflake, BigQuery, Databricks and Amazon S3 Tables all expose an **Iceberg REST catalog**
now, so a warehouse would be reached through the open table format rather than a
vendor-specific driver.

### One integration, not four

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

The catalog would then behave like any other DuckDB database. The endpoints that matter:

- **BigQuery**: the BigLake metastore speaks the REST catalog protocol and is generally
  available at `https://biglake.googleapis.com/iceberg/v1/restcatalog`.
- **Snowflake**: Iceberg tables with a remote Iceberg REST catalog (AWS Glue, Snowflake Open
  Catalog), plus catalog-linked databases that sync namespaces and tables from a remote catalog.
- **Amazon S3 Tables**: `ATTACH '<arn>' AS cat (TYPE iceberg, ENDPOINT_TYPE s3_tables)`.

A single source kind, `Source::Iceberg { endpoint, secret_ref, table }`, would cover all of them
instead of four vendor-shaped kinds. It is also the integration tier 2 already needs.

### What it gives us

`iceberg_snapshots(my_catalog.default.events)` would work on catalog tables, so a warehouse
source could record a snapshot id like every other source. That would close the one source kind
that had no pin, which is the requirement issue #2 already states.

### The performance play

```sql
ATTACH 'ducklake:my_ducklake.ducklake' AS my_ducklake;
CALL iceberg_to_ducklake('my_catalog', 'my_ducklake');
```

`iceberg_to_ducklake` would be a metadata-only copy of an attached Iceberg catalog into a
DuckLake catalog. No data would move. The warehouse's tables would become queryable as DuckLake
tables, so repeated reads would inherit DuckLake's statistics-based filter pushdown and snapshot
semantics instead of paying the remote catalog's latency on every request.

The shape would be: warehouse as the system of record, Iceberg as the wire, DuckLake as the
local read path.

### Fallbacks

The vendor extensions would remain, for warehouses and tables with no Iceberg surface:
`snowflake` through ADBC, `bigquery`, `onelake` for Fabric, and the generic `adbc` extension for
anything with an ADBC driver. All of them are **community extensions**, so they are rebuilt per
DuckDB release and only the latest release is listed. An extension that lags fails to load.

One concrete gotcha to document: DuckDB against Snowflake Open Catalog requires credential
vending to be enabled, or `ACCESS_DELEGATION_MODE 'none'` plus an S3 secret of your own.
Namespace depth, AWS region and secret URL-encoding are all fiddly.

### Azure

Worth being plain about. The `azure` extension is Blob Storage
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
committed, with one exception: the layout measurement is `examples/layout.rs`, because it is a
claim worth rechecking rather than a one-off. They are small enough to rebuild when a claim needs
rechecking.

- **DuckDB**: in this repository, `ENTITIES=500000 cargo run --release -p feather-core
  --features offline --example duckdb_input`. That is where the 34.4 ms figure comes from.
  The same caveat as the layout table's join rows applies to it: that example's label
  timestamps fall below the feature rows' for their key, so the figure is a decode and
  build-side-sort cost, not a point-in-time join.
- **Layout**: in this repository, `cargo run --release -p feather-core --features offline
  --example layout`. It writes the same generated feature table twice, ordered by
  `(user_id, event_timestamp)` and by a deterministic permutation of the same rows
  (`(i * 2654435761) % 2000000`, deliberately rather than the order `range()` produced,
  so no row group has local key structure to begin with), and joins each with the same
  `ASOF LEFT JOIN`, so the layout is the only variable. Bytes read are the kernel's
  `rchar` counter, sampled either side of the timed join, because DuckDB's own profile
  cannot answer it; the row-group counts are read from the file's footer. The program
  prints a matched-row count and asserts it equals the label count, so the figure is a
  join that returned features rather than one that returned nulls. `cargo test` does not run
  examples, so the same invariant is also asserted as an ordinary test, over three label rows
  and no timing, in `crates/feather-core/src/offline.rs`.
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

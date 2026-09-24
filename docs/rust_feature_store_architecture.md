# Architectural design document: Rust feature store

This document describes the intended architecture for an opinionated, high-performance
open-source alternative to Feast. It strips out multi-provider abstraction and specializes
on a native Rust core, with DuckDB for historical computation and Valkey for online serving.

Scope note: this is a design document. Nothing described here is implemented. Performance
figures are targets, not measurements; no benchmarks have been published.

---

## System architecture overview

A single codebase, with a strict division of labor between historical processing (the
offline engine) and low-latency feature lookup (the online serving layer).

```text
     ┌──────────────────────────────────────────────────────────────┐
     │  Python layer                                                │
     │  feature_store.yaml   infrastructure config (Pydantic v2)    │
     │  definitions/*.py     entities, sources, feature views       │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ Maturin / PyO3
     ┌───────────────────────────▼──────────────────────────────────┐
     │  Rust core                                                   │
     └────────┬─────────────────────────────────────┬───────────────┘
              │ offline                             │ online
┌─────────────▼──────────────────┐    ┌─────────────▼─────────────────────┐
│  DuckDB                        │    │  L1: moka, invalidated by         │
│  Parquet / S3 / Iceberg        │    │      Valkey CLIENT TRACKING       │
│  warehouse sources (community  │    │  L2: Valkey, one hash per entity  │
│  extensions)                   │    │      field per feature view       │
│  ASOF join, Arrow out          │    │                                   │
└────────────────────────────────┘    └───────────────────────────────────┘
```

Note what is absent: there is no registry, no lockfile, and no compiled definition
artifact. See "Definitions are imported, not compiled" for why, and "Recorded decisions"
for the evidence behind it.

### Design tenets

1. **Performance first.** Native code (Rust) and vectorized memory layouts (Apache Arrow),
   with no per-row Python and no serialization format on the hot path.
2. **Zero-infrastructure local mode.** Point-in-time joins and online lookups work out of
   the box with no external services.
3. **Opinionated simplification.** One online store (Valkey), one local compute engine
   (DuckDB), one internal representation (Arrow).
4. **Additive upgrade paths.** Every capability deferred from v1 has a documented retrofit
   that does not require changing the storage format or the serving path.

### Non-goals

Explicitly out of scope, so that the "opinionated" claim has content:

- No streaming ingestion or real-time feature computation. Features are materialized in
  batches.
- No Spark or Flink execution backend.
- No pluggable online store interface. Valkey is the only L2.
- No feature transformations expressed as arbitrary user code in the serving path.
- No registry or lockfile. Definitions are versioned by git and imported directly.
- No large vectors. The online store holds small scalar values. Embeddings are a different
  access pattern (batched, row-oriented, bandwidth-dominated) and a different cache problem
  (kilobyte values instead of hundreds of bytes), so mixing them penalizes both. This is a
  scoping decision, not a permanent one: if large vectors are ever added they belong in a
  separate store, and quantization must happen before the feature log is written, because the
  stored form has to be the trained form.
- No incremental materialization in v1. See "Materialization".
- No warehouse-native execution tier. Point-in-time joins always run in the local DuckDB
  engine. Running them on the warehouse's own compute would mean uploading the entity frame
  and reimplementing the join semantics in warehouse SQL. BigQuery has no `ASOF JOIN`, and
  Snowflake's takes its own `MATCH_CONDITION` form, so this would be a second implementation
  of the rules in "Point-in-time join semantics" for a routing decision that cannot be made
  safely.

---

## Execution layers

### 1. Python and Rust boundary (Maturin + Pydantic v2)

The user interface is a native Python extension compiled with Maturin and PyO3.

- **Type safety.** Pydantic v2 models validate `feature_store.yaml`, entity definitions,
  data sources, and feature views. Misconfiguration raises before any engine or connection
  is created.
- **Configuration handoff.** Validated config is serialized with `model_dump_json()` and
  deserialized in Rust with `serde_json`. This runs once per process, so it is a correctness
  boundary, not a performance boundary.
- **Data handoff.** DataFrames submitted for point-in-time joins cross the boundary through
  the Arrow PyCapsule interface, so Rust reads the existing Polars or pandas buffers instead
  of receiving a serialized copy. This makes the **input transfer** zero-copy.

  Two caveats, stated because an earlier draft overstated this:
  - The join result is a new relation that DuckDB materializes. It is not zero-allocation.
  - pandas DataFrames are converted to Arrow before the boundary, which copies. Only inputs
    that already expose Arrow buffers avoid the copy entirely.

#### Definitions are imported, not compiled

Feature definitions are Python modules. Both the offline path and the serving path import
them directly. There is no `apply`-generated artifact, no registry, and no lockfile.

The consequence to accept: serving executes user code at import. Two rules keep that
tractable.

1. **The definition surface depends only on `feather`.** A feature view declares
   `Field("click_count", dtype=Int64)`, where `Int64` comes from `feather.types`. It does not
   import Polars, PyArrow, or a warehouse client. This keeps the serving image thin and makes
   the definitions importable in any environment that can import `feather`.
2. **Definitions are declarative data.** No module-level I/O, no network calls, no
   computation at import. A definition module is a list of constructor calls. CI imports
   every declared module and fails on import errors, which is the backstop for rule 2.

`feature_store.yaml` lists the definition modules explicitly:

```yaml
definitions:
  - definitions/user_clicks.py
  - definitions/user_stats.py
```

Explicit listing rather than recursive discovery. Feast reads every `.py` file under the
feature repository, including files that contain no definitions, which is a documented
footgun. An explicit list also tells the serving image exactly what it must contain.

#### Why there is no registry

A registry exists to hold data that the definitions cannot express. Because the definitions
are Python data with real dtypes, the only irreducible candidate is pinning _external_
sources, and that is covered by read-time validation instead.

For a registry to be worth its cost it would have to hold something git does not already
provide. Definitions are in git, so pinning them is redundant. Source schemas are validated
at read time against the declared dtypes, so they need no stored fingerprint. Version stamps
for training sets are computed as a content hash of the definitions rather than stored.

The failure modes this avoids are documented in "Recorded decisions".

#### Feature services

A feature service is a **named projection**: a name and a list of view-and-feature references.
Nothing else.

```python
FeatureService(
    name="ranking_v3",
    features=["user_clicks:click_count", "user_stats:ltv"],
)
```

- **No entity declaration.** Entities come from the referenced views, so declaring them again
  would be a second source of truth.
- **No versioning, no logging config, no infrastructure.** Feast's feature service bundles all
  of that, which makes it registry-shaped, and this design has no registry.
- **The name is what the serving request carries**, so serving resolves the field set once at
  startup instead of accepting a raw field list per request. That is what makes the
  "no per-request metadata resolution" rule in "Read path" enforceable.
- **A raw field list stays available** as an escape hatch for ad-hoc and debug requests. The
  hot path uses a name.

If logged features are ever adopted (see "Open design questions"), this is where their
configuration belongs.

### 2. Online serving layer

#### Storage layout

One Valkey hash per entity. All feature views that share an entity live in that single hash,
which means a cluster colocates them automatically, with no hash tags.

```text
key:    {project}:{entity_name}:{encoded_entity_key}
fields: v:{view}   the view's encoded feature vector, all its features in one blob
        f:{view}   event timestamp of the last write for this view (int64 micros)
```

#### Entity key encoding

`encoded_entity_key` is the entity tuple encoded so that the result is unambiguous without
escaping:

```text
{len}:{value}|{len}:{value}
```

Each component carries its own byte length, so a `|` inside a value cannot be mistaken for a
separator: the parser reads the length, consumes exactly that many bytes, then expects the
separator. That leaves no escaping rule to get wrong and no restriction on key contents,
while staying readable in `valkey-cli`.

Three consequences:

- **All of a view's entities must be supplied.** Partial-key lookups are rejected at
  validation time rather than producing a key that can never match.
- **Component length is capped** at 512 bytes, so one pathological key cannot produce an
  unbounded Valkey key.
- **No cluster hash tag, permanently.** One hash is already one slot, so every field of an
  entity is colocated by construction. A project-level hash tag would force every entity in
  the project into a single slot, which is the opposite of what a cluster is for. Reading N
  entities fans out across N slots, and the client handles that by pipelining per node. This
  is not a v1 shortcut: the access pattern is single-hash by design, so the key format never
  needs a tag.

The leading segment is a namespace, reserved so that later capabilities are additive rather
than a key-format change:

- `v:` one view's encoded feature vector. What v1 writes. One field per view rather
  than per feature, because the value codec writes a whole vector in a single
  fixed-stride blob; a per-feature field would either repeat the schema tag per feature
  or force the codec down to one column at a time and lose the stride.
- `f:` freshness metadata. Used for the read-time TTL check and for staleness reporting.
- `t:` pre-aggregated tiles. Reserved for the tiling upgrade path, see "Scalability".

Values are self-describing: each encoded value carries a small schema tag. Serving treats a
value whose tag does not match the expected dtype as missing rather than attempting to
decode it. This is what makes a dtype change a null window instead of a decode failure, and
it is what allows the encoding to change later without a migration.

#### Value byte layout

A field value is one view's feature vector for one entity:

```text
[tag: u8][flags: u8][null bitmap][fixed-width columns, declared order][variable-width tail]
```

The property that matters is that **fixed-width columns are concatenated with no per-value
length prefix**, so locating column _k_ is arithmetic rather than parsing. Variable-width
values (strings, lists) live in a trailing section behind an offset table, so a reader that
wants only fixed-width columns never touches them.

`tag` identifies the view's schema version: field names, dtypes, and declaration order. A
mismatch means the value is treated as missing rather than decoded.

One field per view, not per feature. That is a deliberate reversal of this document's earlier
naming, which was inconsistent with the codec: the codec writes a whole view's vector as a
single fixed-stride blob, so a per-feature field name would either repeat the schema tag per
feature or force the codec down to one column at a time and lose the stride. Measured on the
benchmark schema, one field per view is 66 bytes against roughly 108 for per-feature fields,
because a schema tag on a single `Int64` is most of its payload.

The cost is that requesting two features from a wide view decodes the whole vector. That is
currently free: requesting 3 of 8 features measures p99 0.741ms, where the round trip dominates.
The layout does make the fix available when it stops being free, since the byte offset of any
fixed-width column is computable from the schema, so a decoder could read only the columns the
caller asked for.

Rejected encodings, and why:

- **MessagePack or JSON.** Self-describing, so every cell is parsed individually. That is
  exactly the cost documented in "Recorded decisions".
- **Arrow IPC per value.** Larger than protobuf for a single row, because each message carries
  its schema, and it still serializes per cell.
- **A per-value length prefix on fixed-width columns.** Doubles the metadata for no benefit,
  since the view's schema already fixes the stride.

Because the server resolves the schema once at startup, decoding needs no per-value type
dispatch. That is the point of the layout.

#### Read path

For a request covering any number of feature views on the same entity, serving issues **one
`HMGET` per entity**, requesting one field per view plus one freshness field. View count does
not multiply round trips. The caller's requested features are projected out after decoding,
so reading two features from a wide view costs decode time for the columns it did not ask for.

Feast issues one `read_from_online_store` call per feature view despite advertising entity
collocation; a user with 11 feature views reported "abysmal" retrieval times
([#3596](https://github.com/feast-dev/feast/issues/3596)). The same issue notes all reads are
synchronous, and
[#2247](https://github.com/feast-dev/feast/issues/2247) reports the same for DynamoDB. The
rule here is absolute: never one read per view.

Requested fields are resolved from the feature service definition **once at process start**
and cached. Feast resolves registry metadata on every request, which profiling shows is more
than half of `get_online_features` execution time
([#4710](https://github.com/feast-dev/feast/issues/4710)).

Batch within a request, not across requests. A staff-level design for a comparable system
measured this and rejected cross-request micro-batching: at 1M requests/second across 2000
hosts, a 1ms batching window captures about 0.5 requests, which is not enough to justify the
added latency. The useful batching is within one request's entity list, which the
one-`HMGET`-per-entity rule already provides. Do not build a cross-request aggregator.

TTL is enforced at read time by comparing `f:{view}` against the view's `ttl`. An expired
value is returned as null, which is the documented contract. The freshness field is read as
part of the same `HMGET`, so this check costs no additional round trip.

#### TTL and reclamation

Two mechanisms, belt and braces:

- **Read-time check (authoritative, portable).** `f:{view}` plus the declared `ttl`. Works on
  any Valkey. This is the correctness path.
- **Native field expiration (reclamation, Valkey 9.0+ / Redis 8.0+).** Write fields with
  `HSETEX`/`HEXPIREAT`, setting an absolute expiry derived from the event timestamp plus the
  TTL, not from wall-clock write time. Expired fields are then reclaimed by the server's
  periodic job instead of lingering.

Using the event timestamp for the absolute expiry matters. Setting a relative TTL at write
time would give a 40-day-old value another 30 days of life under a 30-day TTL.

The version floor is real: field expiration landed in Valkey 9.0 and Redis 8.0, with a
compatible API. On older servers the read-time check still returns correct results, but
expired fields are not reclaimed until the hash is rewritten or deleted.

Feast has neither mechanism. Its Redis adapter can only expire whole entities, so its own
documented principle ("if you request a feature ... older than its TTL, you should get a
null") is violated in practice
([#3596](https://github.com/feast-dev/feast/issues/3596)).

#### Missingness

Three states exist, and collapsing them would lie to the consumer:

| State     | Cause                                                          |
| --------- | -------------------------------------------------------------- |
| `null`    | The source had a null, and that null was materialized.         |
| `missing` | The field was never written, or its schema tag does not match. |
| `expired` | `f:{view}` is older than the view's `ttl`.                     |

`expired` and `missing` both mean "unknown"; the difference is diagnostic, and both are
reported as `missing`.

In the transport, **both `null` and `missing` surface as Arrow nulls by default**. Callers
that need to distinguish them, for example to fill with learned per-feature defaults rather
than a single sentinel, pass `include_state_mask=True` and receive one `uint8` state column
per feature.

The default is nulls-only because Arrow's null bitmap cannot separate the two cases, so
exactness has to be a separate column, and paying for it unconditionally would widen every
response for a distinction most consumers ignore. A caller that fills missing values needs
the mask; a caller that reads present values does not.

#### Freshness contract

What the store promises:

```text
max_staleness = refresh_interval + invalidation_propagation
```

- `refresh_interval` is the deployment's schedule, so this number is per deployment rather
  than a constant.
- `invalidation_propagation` is sub-second on a healthy tracking connection.
- The L1 fallback TTL is the hard worst case, and TTL expiry caps staleness by construction:
  an expired value reads as missing rather than as a stale value.

The store exposes observed lag per view, derived from `f:{view}`, so consumers alert on
reality rather than on the formula. The metric is the alert signal; the formula is what gets
written down.

#### Two-tier cache

- **L1, in-process.** A `moka` cache in the Rust client. The default `moka` eviction policy
  is TinyLFU, which suits skewed key popularity. L1 hits avoid all network I/O.
- **L2, distributed.** Valkey, over non-blocking I/O on the `tokio` runtime.

L1 invalidation uses **Valkey client-side caching** in broadcasting mode:
`CLIENT TRACKING on REDIRECT <id> BCAST PREFIX <project>:`. The server pushes invalidation
messages when a tracked key changes, so invalidation is exact and push-based rather than
TTL-bounded.

Broadcasting is chosen over the default per-key mode for three reasons:

- **No server-side invalidation table.** Valkey's own guidance is to "use the BCAST mode that
  consumes no memory at all on the Valkey side", whereas per-key tracking costs memory
  "proportional both to the number of keys tracked and the number of clients requesting such
  keys". A feature store has few serving clients and a very large keyspace, which is exactly
  the wrong shape for per-key tracking.
- **Server cost scales with clients times prefixes**, not with keyspace size.
- **It avoids four open server bugs in the per-key path**, all concerning the invalidation
  table that this mode does not have:
  [#4143](https://github.com/valkey-io/valkey/issues/4143) (table not cleaned after client
  disconnect), [#4736](https://github.com/valkey-io/valkey/issues/4736) and
  [#4775](https://github.com/valkey-io/valkey/issues/4775) (unbounded eviction effort), and
  [#4416](https://github.com/valkey-io/valkey/issues/4416) (redirect client not notified when
  the tracking source disconnects).

Constraint to respect: **no two prefixes may overlap.** Valkey rejects `foo` and `foob`
together because both would match `foobar`. One prefix per project is sufficient.

Operational requirements, all owned by the Rust core: RESP3, a dedicated redirect connection
for invalidation pushes, and a full L1 flush on reconnect, since the client cannot know what
changed while it was disconnected.

This is a work item, not a configuration flag. valkey-glide implements client-side caching
for Node, Python, Go, and Java, but not for Rust
([#6918](https://github.com/valkey-io/valkey-glide/issues/6918) is still open), so Feather
implements RESP3 push handling and reconnect logic itself.

**The fallback TTL is always on, not conditional on tracking being unavailable.** Push
invalidation is the fast path; the TTL is the guarantee. A bug in push handling then degrades
staleness to a bounded interval instead of silently breaking invalidation, which is the
hardest class of failure to notice. This costs a few lines and removes the whole
"invalidation quietly stopped working" incident.

Subscription detail: invalidations arrive on `__valkey__:invalidate`. Valkey renamed it from
`__redis__:invalidate` and keeps the old name only for Redis 7.2 compatibility, so subscribe
to the Valkey name.

Measured, on a container limited to 2 CPUs and 512MiB with `maxmemory` 384MiB and
`allkeys-lru`, over 100k entities across 4 views of 8 features each, with 66-byte vectors:

| Measurement                  | Result                                    |
| ---------------------------- | ----------------------------------------- |
| read, 1 entity x 4 views     | p50 0.327ms, p95 0.411ms, p99 0.741ms     |
| read, 100 entities x 4 views | p50 2.36ms, p99 5.10ms, 24.2us per entity |
| write, materialization shape | 193k vectors/s, 386k field writes/s       |

Single-entity p99 is sub-millisecond across four views, which is the one-`HMGET`-per-entity
rule and the fixed-stride encoding doing their job. The 17.7ms tail maximum on single reads is
unexplained and worth investigating.

Reproduce with `mise run bench:load`, which prints hardware, key cardinality, and payload size
alongside the numbers so a figure cannot be quoted without them. Compare against
`mise run bench:valkey` for the server's own ceiling on the same container.

These numbers are from one host and one container shape. Treat them as an order of magnitude,
not a guarantee.

### 3. Serving API (Arrow Flight)

The serving transport is **Arrow Flight**, not protobuf over gRPC and not JSON over REST.
Flight runs on gRPC and carries Arrow IPC as its payload, so the gRPC transport claim holds
while protobuf stays out of the data path, consistent with the encoding decision.

The precedent is direct. Hopsworks built this same combination for a feature store, DuckDB
plus Arrow Flight plus Python clients, and reported up to 45x throughput over their previous
REST API, with "zero-copy data transfer on server- as well as on client-side". Independent
benchmarks put Flight up to 30x over ODBC for Arrow data.

What this buys, in order of importance:

- **No decode step for the client.** A Python client receives Arrow record batches and hands
  them to Polars, pandas, or PyArrow directly. The alternative deserializes every value before
  a DataFrame can exist.
- **Batch semantics come free.** Flight streams record batches, so a request covering N
  entities is one streamed response rather than N round trips.
- **Auth has a defined home.** Flight supports token and handshake-based auth handlers, so
  authentication is a supported extension point rather than a bolt-on.
- **Columnar throughout.** The offline path already produces Arrow, so there is one
  representation end to end.

Feast's community asked for this and did not get it
([#2013](https://github.com/feast-dev/feast/issues/2013), 29 comments). That thread is also
where the real cost of the alternative is documented; see "Recorded decisions".

A request must carry a feature service reference plus entity keys, not a raw field list, so
that serving can resolve the field set once at startup. The feature service model itself is
not yet defined; see "Open design questions".

### 4. Offline engine (query routing)

Historical queries are routed between two execution tiers. Both execute in the local DuckDB
engine; they differ in where the source data lives and how it is reached.

#### Tier 1: local files and warehouse sources

- **Engine.** Local DuckDB via the `duckdb` Rust crate.
- **Mechanism.** DuckDB's `snowflake` and `bigquery` extensions read from those warehouses.
  Both are **community extensions**: contributed and maintained outside DuckLabs, and
  installed from the community repository (`INSTALL snowflake FROM community`). They are not
  core extensions and are not covered by the core support tier. The Snowflake extension
  connects through an Arrow ADBC driver.
- **Operational consequence.** Community extensions are rebuilt per DuckDB release. Pin the
  DuckDB version and the extension version together; an extension that lags the DuckDB
  release will fail to load. DuckDB's generic `adbc` extension is an alternative for any
  database with an ADBC driver.
- **Spill behavior.** When a join exceeds `memory_limit`, DuckDB spills to its temp
  directory, which should be pointed at local SSD. It does not use swap. The temp directory
  location and `memory_limit` are the two knobs that matter for large joins.

#### Tier 2: open table formats on object storage

- **Engine.** DuckDB plus cloud object storage.
- **Mechanism.** The `iceberg` and `delta` extensions read table formats; `httpfs` provides
  S3 and generic HTTP access, and the `azure` extension covers Blob Storage. `httpfs` issues
  HTTP range requests, so a scan fetches only the byte ranges it needs rather than whole
  files.
- **Consequence.** Compute runs locally, so there is no warehouse compute charge. Metadata
  operations (manifest reads, snapshot resolution) still hit the catalog and are the usual
  source of latency on high-file-count tables.

#### Source schema validation

Read-time validation is the drift check, since there is no stored fingerprint (see "Why
there is no registry"). Validating on every read is too expensive for a warehouse source, so:

- Validate **once per process per source** at startup, and cache the resolved schema.
- Re-validate on a schedule in long-lived processes.
- Record the source's snapshot identity alongside the result, where the source has one: a
  Parquet file's metadata, an Iceberg or Delta snapshot id, or a warehouse table version.
  That is the closest thing to a source pin available without a registry, and it is what
  makes a drift report actionable rather than just "something changed".

Metadata-only checks keep this cheap. DuckDB's `DESCRIBE` on Parquet reads only the footer,
and table-format catalogs expose schema without scanning data.

---

## Point-in-time join semantics

This is the part of a feature store that is easiest to get subtly wrong, so the rules are
specified rather than left to the engine.

### Column conventions

- The entity DataFrame carries a label timestamp column, configured as
  `entity_df_timestamp` (default `event_timestamp`).
- Each feature view declares a `timestamp_field` (default `event_timestamp`) and an optional
  `created_timestamp_field`.

### Match rule

For each label row with entity key `k` and label time `t`, the join selects the feature row
with key `k` whose timestamp is the greatest value satisfying `fv.timestamp <= t`. A feature
row at exactly `t` is included by default; `strict=True` switches to `fv.timestamp < t`.

### TTL

DuckDB's `ASOF JOIN` permits exactly one inequality condition, on the ordering column.
Every other condition must be an equality or `NOT DISTINCT`. TTL therefore **cannot** be a
second range predicate inside the join. Two workable shapes:

- **Pre-filter the right side.** Only possible when every label time falls in a known
  window. Otherwise the filter depends on `t` per row and a plain range join replaces the
  ASOF join, losing its performance advantage.
- **Join, then apply TTL to the matched timestamp.** The row survives the join, and the
  TTL predicate is evaluated afterwards.

Feather uses the second shape. That makes the handling of expired and unmatched rows an
explicit choice rather than an accident of query planning:

- `on_missing: "null"` (default): the row is kept and its feature values are null.
- `on_missing: "drop"`: the row is removed from the result.

The default is `null` because dropping rows silently changes the size and class balance of
the training set, and a training job that shrinks without saying so is worse than one that
carries explicit nulls. Callers that want Feast-style row dropping opt in.

### Duplicate timestamps

If two feature rows share the same entity key and the same maximal timestamp, `ASOF JOIN`
picks one without a documented rule. Feather resolves this deterministically before the
join: if `created_timestamp_field` is declared, the row with the greatest
`created_timestamp` wins; if it is not declared, the join fails with an error rather than
returning a planner-dependent value.

### Null keys

SQL equality never matches `NULL`, so label rows with a null entity key get no match and
follow the `on_missing` rule. This is intentional and documented rather than special-cased.

### Row order

`ASOF JOIN` does not preserve the input order of the left side. Feather attaches a stable
row index to the entity DataFrame before the join and sorts by it afterwards, so the output
training set stays aligned row-for-row with the input labels. Without this step, labels and
features silently misalign, which is the most damaging failure mode available here.

### Time zones

All timestamps are normalized to UTC. A naive `TIMESTAMP` and a `TIMESTAMP WITH TIME ZONE`
in the same join raise an error rather than being coerced, because an implicit coercion is
how off-by-hours bugs enter a training set.

### Conformance

Both execution tiers share one DuckDB code path, so there is a single implementation of
these rules. The conformance suite covers: match at exact timestamp, TTL boundary at
exactly `ttl_days`, no match, duplicate timestamps, null keys, timezone mixing, and output
row alignment.

---

## Materialization

Materialization computes feature values from offline sources and writes them to Valkey.

### Full refresh, no watermarks

v1 does a full refresh: recompute every value and overwrite it in place. There is no
watermark, no materialization interval, and no progress state anywhere.

This is a correctness decision before it is a simplicity decision. **Full refresh is
self-healing.** A crashed or partial run leaves some values from the previous run, which are
still within TTL, and the next run overwrites everything. There is no partial state to
reconcile.

Incremental materialization is not self-healing. It requires the watermark advance to be
atomic with the data write, and getting that wrong means either permanently skipping a range
or reprocessing it. Feast has shipped this bug at least twice, in independent
implementations:

- [#3789](https://github.com/feast-dev/feast/issues/3789): "The last materialized date ...
  was being updated before materialization had even run. This meant that if materialization
  had failed for some reason, many updates could be ignored, because the next run would
  continue from beyond the end date of the failed run."
- [#3787](https://github.com/feast-dev/feast/issues/3787): the same defect in the Bytewax
  materializer.
- [#4222](https://github.com/feast-dev/feast/issues/4222): a far-future end date "breaks
  incremental materializations up to that [date]", permanently.

Watermarks also make schema evolution a migration problem. Feast needed an opt-in migration
flag for incremental materialization after a schema change
([#2307](https://github.com/feast-dev/feast/issues/2307)). With full refresh, a dtype change
is fixed by the next refresh, because every value is rewritten with the new encoding.
Re-materialization is the migration.

### Arrow-native pipeline

The write path never leaves Arrow and never builds a row-oriented intermediate:

1. DuckDB computes the values and streams Arrow record batches out.
2. Each batch is encoded directly into Valkey write commands.
3. Commands are pipelined, bounded by an in-flight byte budget.

No step materializes the full dataset. This is the specific difference from Feast, which
converts the entire Arrow table into a Python list of protobuf objects before writing:
Arrow to NumPy to `List[ValueProto]` to row-wise, three full copies, at "~200 bytes overhead
per value vs 4 to 8 bytes raw", OOMing at hundreds of thousands of rows
([#6160](https://github.com/feast-dev/feast/issues/6160)), and requiring the whole dataset in
memory ([#2594](https://github.com/feast-dev/feast/issues/2594)). Protobuf also costs 80% of
online read execution time in Feast ([#3649](https://github.com/feast-dev/feast/issues/3649)).
There is no protobuf anywhere in Feather's data path.

### Latest-per-entity pushdown

A full refresh of _current_ feature values needs only the most recent row per entity, not the
full history. That reduction happens inside DuckDB (`arg_max`, or
`QUALIFY row_number() OVER (PARTITION BY entity ORDER BY ts DESC) = 1`), never in Python.

The scan is therefore bounded by the widest feature window rather than all history. Feast
regressed on exactly this, moving from "latest values only" to pulling everything and
deduplicating in pandas
([#5707](https://github.com/feast-dev/feast/issues/5707)), and later added deduplication
pushdown to fix it ([#5709](https://github.com/feast-dev/feast/issues/5709)).

### Parallelism

Work is parallelized across feature views and across key ranges within a view. Feature views
are independent, since each writes its own fields. Feast materializes views sequentially
([#2594](https://github.com/feast-dev/feast/issues/2594)).

### Scheduling

A refresh is a Kubernetes `CronJob` per view group, with `concurrencyPolicy: Forbid` and a
non-zero `startingDeadlineSeconds` to tolerate scheduler clock skew. This is the same recipe
Feast's operator uses, and it needs no new component.

**The lock is an efficiency guard, not a correctness one.** Feast needs a distributed lock
plus job-state tracking because its incremental materialization is not idempotent, so a
duplicate run can corrupt the watermark. A full refresh is idempotent by construction, so a
duplicate run wastes work and produces the same state. Feather therefore takes a
non-blocking lock (`SET NX PX`, TTL set to the expected maximum runtime) and exits 0 when it
is already held. No job-state table, and no already-running detection.

Freshness is surfaced from `f:{view}`, which the write path already maintains. The maximum
lag across views is the metric to alert on, exported as a gauge rather than discovered by a
consumer noticing stale features.

### Garbage collection

Full refresh only writes fields for currently defined views, so fields belonging to renamed
or removed views would otherwise persist forever. Feast has exactly this leak, growing the
online store indefinitely because it only deletes data when the last view for an entity is
gone ([#3596](https://github.com/feast-dev/feast/issues/3596)).

Feather deletes the affected fields on `apply`: `HDEL` the `v:{view}:*` and `f:{view}`
fields for a removed view. This has to be idempotent and safe to re-run. Note that it makes
`apply` no longer a pure function, since it touches the online store. That is acceptable
because the definitions artifact itself is still never written at runtime.

### Ceiling

Full refresh cost scales with total entity-view pairs, not new data. With one hash per entity
and one field per view, a refresh writes `entities x views` field values. Measured at 386k
field writes per second on the constrained container described under "Two-tier cache", tens of
millions of entity-view pairs is about a minute and low hundreds of millions is minutes. The
earlier estimate in this document, which said hours for the second case, was several times too
pessimistic.

That is the v1 ceiling, and it is deliberate. See "Scalability" for the documented path past
it.

---

## Scalability and the upgrade path

### The two decisions that keep retrofits additive

Everything deferred from v1 is additive or a rewrite depending on two choices, both of which
are cheap now and structural later:

1. **Key layout: one hash per entity, one field per feature view.** Incremental updates later
   become "`HSET` a few fields on a few hashes". Tiling later becomes "add `t:` fields to the
   same hash". Choosing key-per-feature, or one opaque blob per entity, makes both a storage
   format rewrite.
2. **Self-describing, namespaced values.** The schema tag lets the encoding change without a
   migration, and the field namespace prefix lets a new field kind be introduced without
   renaming existing fields.

### Tiles: the designed-for next step

The scalable answer for aggregation features is pre-aggregation, not incremental
materialization. [Chronon's tiled architecture](https://www.chronon.ai/Tiled_Architecture.html)
is the reference design.

The problem it solves: an untiled read path fetches and aggregates O(events) per request. At
10 events/second with a 12-hour window, a single request fetches and aggregates 432,000
events. The tiled write path pre-aggregates into fixed windows, so the same request fetches
and merges 12 or 13 hourly tiles instead. Stripe measured a 33% latency reduction.

The properties matter more than the speedup:

- **Idempotent by window.** Recomputing hour H overwrites the same key. There is no progress
  state to corrupt.
- **Late data is handled by recomputation.** A late event invalidates one window, which is
  recomputed. No watermark, no backfill logic.
- **Bounded read cost.** O(windows) rather than O(events), regardless of stream volume.

Chronon requires Flink for this. Feather does not: the same tiles can be computed in DuckDB
during batch materialization. Same idempotency, same bounded reads, no streaming runtime and
no new process.

### Tile encoding: partial aggregates, never final values

A tile stores a **partial aggregate**, not the finished feature. This is the rule that makes
merging correct, and it is the mistake Feast names explicitly: recomputing from raw data is
slow, while storing final aggregated values per tile is fast but "often incorrect when
merging".

The v1 set is closed and small, covering the aggregates that merge exactly:

| Aggregate | Stored in tile    | Merge                      | Storage  |
| --------- | ----------------- | -------------------------- | -------- |
| `sum`     | sum               | `sum(tile_sums)`           | 1 value  |
| `count`   | count             | `sum(tile_counts)`         | 1 value  |
| `min`     | min               | `min(tile_mins)`           | 1 value  |
| `max`     | max               | `max(tile_maxes)`          | 1 value  |
| `mean`    | sum, count        | sum both, then divide      | 2 values |
| `stddev`  | sum, sumsq, count | sum all three, then derive | 3 values |

> `ponytail:` `stddev` is stored as sum, sumsq, and count and derived as
> `sumsq/n - (sum/n)^2`. That form cancels catastrophically when values are large relative to
> their variance: with values near 1e9, both `sumsq/n` and `(sum/n)^2` land near 1e18, and
> float64 resolution at 1e18 is roughly 100, so a variance of 1 is lost entirely and the
> result is noise. Ceiling: correct for well-scaled features, wrong for large-magnitude ones.
> Upgrade path: store `(count, mean, M2)` instead, the same three values, and merge with
> Chan's parallel formula, which is numerically stable at identical storage cost. `mean` is
> unaffected, since `sum/count` is well-conditioned.

Anything that does not merge exactly is out of scope rather than approximated. Tecton stores
"a complex data structure (such as those used for approximate algorithms like HyperLogLog)"
in a tile, and BigQuery ships mergeable Data Sketches, so sketches are the documented upgrade
path for cardinality and quantiles. They are not v1, because an approximate feature needs an
explicit accuracy contract rather than an inherited one.

The schema tag already distinguishes an IR encoding from a plain value, which is why adding
tiles does not change existing field names.

### Sawtooth windowing

Tiles alone are not enough for a _relative_ window, because the freshest data is always
younger than the newest tile boundary. Tecton's answer is sawtooth windowing, and it is worth
copying: the compaction job writes "a few number of small tiles at the tail (i.e. the oldest
edge) of the aggregation window in addition to a larger compacted tile". Their example: "for
a 30-day window, the large compacted tile could be 28 days with 48 one-hour sawtooth tiles."

A read then merges a few large tiles, a few small tail tiles, and recent raw events at the
head. Bounded read cost, fresh data, and no recomputation from full history.

Feather's version has no streaming head, so the head is simply the part of the window younger
than the last materialization, computed in DuckDB at read time. That keeps the whole
mechanism batch-only.

### Compaction is materialization

Tecton's compaction job is "a batch job [that] will read the offline data for the full
aggregation window, perform the aggregations, and update the Online Store for each entity",
on a cadence of typically daily. That is exactly Feather's full refresh. No separate
compaction subsystem is needed: a full refresh already rewrites tiles, and tile idempotency
is what makes that safe.

Two limits, restated: tiling only helps aggregation features, since lookup features remain
latest-value. And the read path gains a merge step, which is why the value encoding is
versioned and self-describing.

Chronon's own guidance is the right trigger: "If your hottest keys don't exceed a few
thousand events per day, the untiled approach may still be sufficient."

### Zero-downtime schema changes

v1 accepts a null window during a dtype change: serving sees a value whose schema tag does not
match and returns null until the refresh completes. Correct, never a decode failure, but not
zero-downtime for large refreshes.

The upgrade is double-buffered generations: write the new encoding under a new generation,
flip a pointer atomically, then let the old generation expire via native field expiration.
This is additive, because values are already self-describing and fields are already
namespaced. It costs 2x peak storage during a refresh, which is why it is not the v1 default.

### Watermarks as a last resort

Incremental materialization with watermarks is the last option, not the second. It is only
justified for non-aggregation features that need source-side change detection, and it brings
the atomicity problem documented in "Materialization".

Triggers for revisiting, stated concretely rather than as "at scale":

- Refresh wall-clock exceeds roughly half the schedule interval, so hourly jobs over about 30
  minutes.
- A single view's source scan exceeds a few hundred GB, so the offline side dominates.
- A feature is expensive to recompute and cheap to update from a source-side diff, and the
  diff is cheaper than the recompute.

Any of these, and the answer is tiles first, watermarks only if tiles do not apply.

---

## Configuration

`feature_store.yaml` holds infrastructure. Python modules hold definitions. Both are
validated by Pydantic v2, and this section is the single source of truth for the schema; the
README links here instead of restating it.

```yaml
project: ad_recommendations

definitions:
  - definitions/user_clicks.py
  - definitions/user_stats.py

offline_store:
  type: snowflake
  config:
    account: "xy12345.us-east-1"
    warehouse: "PROD_WH"
    username: "fs_runner"
    password: ${SNOWFLAKE_PASSWORD}

online_store:
  type: valkey
  config:
    endpoint: "valkey-cluster.internal.svc:6379"
    tls: true
    # Requires Valkey 9.0+ / Redis 8.0+ for native field expiration.
    # Older servers fall back to read-time TTL checks only.
    field_expiration: true
    l1_cache:
      enabled: true
      max_capacity_mb: 2048
      # Only used when CLIENT TRACKING is unavailable. This is the
      # staleness contract in that case.
      fallback_ttl_seconds: 30
```

### Secrets

`${VAR}` in any config value is interpolated from the process environment at load time. A
referenced variable that is unset is a load error, not an empty string. `feature_store.yaml`
is committed, so it must never contain a literal credential: use environment interpolation,
or a secret manager reference. A literal secret in this file is a bug.

### Backend coverage

The schema above shows Snowflake and Valkey. The same `type` plus `config` shape applies to
the other backends, but only the two shown have a settled schema. BigQuery, Azure, and the
object-storage tier still need their `config` keys specified before they can be documented.

---

## Deployment

1. **Local development.** `pip install feather-store`. Everything runs in-process: local
   Parquet, an in-memory DuckDB, and the L1 memory cache. No external services.
2. **Production.** Valkey runs as a StatefulSet; Rust API pods run as a horizontally scaled
   Deployment behind gRPC and REST; materialization runs as resource-isolated Kubernetes
   Jobs on a schedule or on demand.

The Helm charts and Kustomize manifests are not written yet. When they exist, this section
should link to them rather than describe them.

---

## Naming

The distribution is `feather-store` and the import name is `feather`. The import name
collides with the pre-existing PyPI package `feather` (the Feather dataframe format), so an
environment must not contain both. If that becomes a problem, the import name should change
before the first release, since renaming after publication is far more expensive.

---

## Recorded decisions and rejected alternatives

Recorded so they are not re-litigated. Each cites the evidence that drove it.

### No registry, no lockfile

**Rejected:** a generated registry artifact (a lockfile) holding compiled definitions.

Feast's registry holds both definitions and mutable runtime state, and that conflation is
destructive:

- [#2905](https://github.com/feast-dev/feast/issues/2905): running `feast apply` before
  `feast materialize-incremental` loses the tracked incremental date, so the incremental run
  silently does nothing. The reporter's conclusion: "the feature store cannot be touched once
  it is used in production." Regenerating the artifact destroys runtime state, because they
  are the same file.
- [#6780](https://github.com/feast-dev/feast/issues/6780) (open): materialize sets the view
  to `MATERIALIZING` in the shared registry, and `get_online_features` then refuses to serve
  it. This "turns routine materialization into recurring online-serving outages."

Feather has no runtime writer for definitions, so neither failure is reachable. Source schema
drift is caught by read-time validation rather than a stored fingerprint.

### Definitions are imported, not compiled

**Rejected:** a compile step from Python definitions to a serialized artifact.

A compile step is only necessary when the authoring language is a program that must become
data. It buys decoupling from user code at the cost of an artifact, a drift check, and the
conflation above. The alternative is to keep the definition surface declarative and thin, and
import it.

### One hash per entity, one field per view

**Rejected:** one key per (view, entity), and one opaque blob per entity.

The hash gives cluster colocation for free and makes a request for N views a single `HMGET`
per entity. The rejected alternatives either reintroduce per-view round trips or make
incremental updates and tiling structural rewrites. The tradeoff accepted is that per-view
TTL needs field expiration rather than key expiration, which is why the Valkey version floor
exists.

### Read-time TTL as the correctness path

**Rejected:** relying on native field expiration alone.

Field expiration is version-gated and its reclamation is a periodic job, so it is not a
correctness guarantee. The read-time check is authoritative and portable; native expiration
is a reclamation optimization.

### Full refresh, no watermarks

See "Materialization". The evidence is three independent Feast bugs in watermark handling and
one permanent-corruption variant.

### Arrow end to end, no protobuf

**Rejected:** protobuf as the value encoding.

It costs three full copies on the write path and 80% of read execution time in Feast, and it
requires the whole dataset in memory. Arrow is already the internal representation, so
encoding directly from Arrow batches removes a format rather than adding one.

### Arrow Flight as the serving transport

**Rejected:** protobuf over gRPC, and JSON over REST.

Flight is gRPC with Arrow IPC as the payload, so it keeps the transport while removing the
serialization cost from the data path. Hopsworks shipped the same combination for a feature
store and measured up to 45x throughput over their REST API. Feast's community requested the
switch and it was not made ([#2013](https://github.com/feast-dev/feast/issues/2013)).

### Serialize at vector granularity, not per cell

**Rejected:** protobuf per feature value, and Arrow IPC per feature value.

Both are wrong for the same reason, and the reasoning is worth preserving because Arrow IPC
per value is the intuitive fix. In Feast's own investigation, writing 5M rows by 30 columns
took over an hour, at roughly 1e-5 seconds per feature, single-threaded, because
`convert_arrow_to_proto` serializes each cell independently. A maintainer confirmed both the
timing and the single-core behavior.

Switching the per-cell format to Arrow IPC does not fix it. A reviewer on the same issue
noted that you would "still have to serialize every individual cell of the table
separately", and another noted that Arrow IPC encoded per single row is _larger_ than
protobuf, because each message carries its schema.

The fix is granularity. Feather serializes one vector per entity per view, or better, encodes
a batch of vectors into a single buffer and slices it per entity. The existing layout already
turns N x M x F cell serializations into N x M vector serializations.

### Broadcasting mode for client-side caching

**Rejected:** per-key `CLIENT TRACKING`, and TTL-only invalidation.

Per-key tracking costs server memory proportional to keys times clients, which is the wrong
shape for a large keyspace with few clients, and it carries four open server bugs. TTL-only
invalidation reintroduces the staleness window that exact invalidation removes. See
"Two-tier cache".

### Partial aggregates in tiles

**Rejected:** storing final aggregate values per tile.

A finished value makes merging incorrect: a mean of means is not a mean, and a stddev of
stddevs is not a stddev. Tiles store the mergeable partial state instead, and aggregates that
do not merge exactly are out of scope for v1 rather than approximated. See "Tile encoding".

### Length-prefixed entity keys

**Rejected:** escaping a separator, hashing the entity tuple, and cluster hash tags.

Escaping needs a rule that must be right in two places and is easy to get subtly wrong.
Hashing the tuple makes keys unreadable, so `valkey-cli` stops being useful for debugging.
Hash tags are unnecessary because the access pattern is single-hash: one hash is one slot, so
a tag buys nothing for reads and would force a whole project into a single slot.

### Fixed-stride value encoding

**Rejected:** MessagePack, JSON, and Arrow IPC per value.

The first two are self-describing, so every cell is parsed individually, which is exactly the
cost documented in "Serialize at vector granularity". Arrow IPC per value is larger than
protobuf for a single row because each message carries its schema, and it does not avoid
per-cell work either. A fixed-stride layout makes column location arithmetic instead of
parsing.

### Missingness as a distinct state

**Rejected:** collapsing null, missing, and expired into a single null.

A consumer that fills missing values with learned per-feature defaults cannot do that
correctly if "we know this is null" and "we do not know" are the same value. The transport
defaults to nulls because that is cheap, and exposes a state mask on request because the
distinction is real.

---

## Open design questions

Two items remain open. Two further capabilities are deferred with triggers rather than left
open, and they are documented where they belong: approximate aggregates in "Tile encoding",
and incremental materialization in "Watermarks as a last resort".

### Needs a product decision

- **Logged features.** Write the exact served feature vector to durable storage at request
  time, and train on those logs rather than on a recomputed historical join. A staff-level
  design for a comparable system does this and calls it the thing that "eliminates
  offline/online computation skew by construction for served features". Feather already holds
  the served vector, so the marginal cost is a write.

  The recommendation is to take the hook and defer the pipeline: expose the served vector
  through a return value or a logger callback, document the pattern, and let the user's
  platform own the sink. Owning a durable log with its own retention, privacy, and volume
  story is a platform concern, not a feature store concern.

  Record the trap when it is adopted: **logged vectors only exist for entities that were
  actually served.** Training only on served vectors inherits the serving policy's selection
  bias, which makes this unusable as the only training path.

  Trigger for revisiting: a concrete consumer for the log, not just interest in the idea.

### Needs an operational answer

- **Observability.** The metric set is named here so that instrumentation lands with the code
  rather than after it. The backend and any dashboards are not chosen.

  Read latency histogram split by resolution tier (L1 hit, L2 hit, missing); L1 hit rate and
  entry count; invalidation messages received and L1 flushes; refresh duration, rows written,
  and failures; freshness lag per view as a gauge; decode errors and schema-tag mismatches.

  The last two are the failure modes that are otherwise silent, so they matter more than the
  latency numbers. Prometheus text exposition, optional OTLP traces.

  Trigger for revisiting: the first working read path.

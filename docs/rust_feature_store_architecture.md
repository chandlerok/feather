# Architectural design document: Rust feature store

This document describes the intended architecture for an opinionated, high-performance
open-source alternative to Feast. It strips out multi-provider abstraction and specializes
on a native Rust core, with DuckDB for historical computation and an embedded online store that
a feature server owns.

Scope note: this is a design document with a partial implementation. The definition layer,
entity key encoding, value codec, and the online serving layer are built and measured, and so is
materialization, which computes a view's values from its source and writes them to the online
store. The offline engine is built over local Parquet, object storage, and a Postgres table, and
is design only for the source kinds with no reader. `FeatureStore.serve()` is built: it makes the
calling process an Arrow Flight feature server over the embedded store. What is still design only
is everything a deployment runs around that: the schedule, the per-view parallelism, the lock, and
the Kubernetes objects. Nothing in this repository issues any of them. Figures that are
measurements say so and carry their hardware and cardinality; the rest are targets.

Sections describing a tier that no longer exists say so where they begin. Their prose is kept,
because the trade each one records is one a reader of the current store still has to make, but
nothing below should be read as a description of the code.

---

## System architecture overview

A single codebase, with a strict division of labor between historical processing (the
offline engine) and low-latency feature lookup (the online serving layer).

```text
     ┌──────────────────────────────────────────────────────────────┐
     │  Python layer                                                │
     │  feather.toml         infrastructure config (Rust core)      │
     │  definitions/*.py     entities, sources, feature views       │
     └───────────────────────────┬──────────────────────────────────┘
                                 │ Maturin / PyO3
     ┌───────────────────────────▼──────────────────────────────────┐
     │  Rust core                                                   │
     │  feather-serve: Arrow Flight, the whole request path         │
     └────────┬─────────────────────────────────────┬───────────────┘
              │ offline                             │ online
┌─────────────▼──────────────────┐    ┌─────────────▼─────────────────────┐
│  DuckDB                        │    │  Embedded LSM (fjall), in a       │
│  Parquet / S3 / Iceberg        │    │  directory one process opens      │
│  warehouse sources (Iceberg    │    │  one key per entity, all of its    │
│  REST catalogs)                │    │  fields in one contiguous run     │
│  ASOF join, Arrow out          │    │  one store read per entity        │
│                                │    │  read-time TTL check              │
└────────────────────────────────┘    └───────────────────────────────────┘
```

Note what is absent: there is no registry, no lockfile, and no compiled definition
artifact. See "Definitions are imported, not compiled" for why, and "Recorded decisions"
for the evidence behind it.

### Design tenets

1. **Performance first.** Native code (Rust) and vectorized memory layouts (Apache Arrow),
   with no per-row Python and no serialization format on the hot path.
2. **Zero-infrastructure local mode.** Point-in-time joins and online lookups work out of
   the box with no external services. The online store is a directory in the project, and
   the process that serves it is the process that opened it.
3. **Opinionated simplification.** One online store, an embedded LSM whose position is set
   by a `path` in `feather.toml`, one local compute engine (DuckDB), one internal
   representation (Arrow), and one serving transport (Arrow Flight).
4. **Additive upgrade paths.** Every capability deferred from v1 has a documented retrofit
   that does not require changing the storage format or the serving path.

### Non-goals

Explicitly out of scope, so that the "opinionated" claim has content:

- No streaming ingestion or real-time feature computation. Features are materialized in
  batches.
- No Spark or Flink execution backend.
- No networkable online store. The store is a directory one process opens, and the server
  is how a second process reads it. A store that several processes write is a different
  product; see "Open design questions".
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

### 1. Python and Rust boundary (Maturin + PyO3)

The user interface is a native Python extension compiled with Maturin and PyO3.

- **Type safety.** Two layers, and the core is the authority. Pydantic validates a definition as
  it is written, so a mistake is reported in the language it was written in, before the process
  does anything else. The core validates what it accepts, so a binding that skips its own checks
  still cannot hand over a value an engine cannot represent, and a second binding has nothing
  new to implement.
- **Configuration handoff.** The core reads `feather.toml` itself: it parses the TOML, resolves
  `${VAR}`, and validates the schema. A binding passes a path and receives the validated result,
  so the file format and its rules have one implementation rather than one per language.
- **Data handoff.** DataFrames submitted for point-in-time joins cross the boundary through
  the Arrow PyCapsule interface, so Rust reads the existing Polars or pandas buffers instead
  of receiving a serialized copy. This makes the **input transfer** zero-copy.

  Two caveats, stated because an earlier draft overstated this:
  - The join result is a new relation that DuckDB materializes. It is not zero-allocation.
  - pandas DataFrames are converted to Arrow before the boundary, which copies. Only inputs
    that already expose Arrow buffers avoid the copy entirely.

#### Declared views compile to the wire models

Definitions are declared as classes and compiled to the Pydantic models above, so there are
two layers rather than one:

- **Authoring** (`feather.definitions`): a feature is a typed class attribute, as in
  `click_count = Field(Int64)`, inside a class whose configuration comes from a decorator.
- **The wire** (`feather._wire`): the Pydantic models that serialize and cross into Rust.
  Private, because the shape is a contract rather than a user interface. These constraints bound
  the shape on the Python side and the core validates what it receives, so they are a duplicate
  of the core's rules rather than the rule itself.

The split exists because a reference to a feature has to be checkable. A string cannot be
checked, and neither can attribute access on an object whose fields are populated at runtime:
a checker only knows attributes declared in a class body, so `fv.click_count` would resolve
through `__getattr__` and catch no typo at all. Only moving the declaration into the class body
makes the reference a checked attribute access.

Two consequences, both measured rather than assumed:

- **A dtype is a class, not an enum member.** It is used as a type argument, `Field[Int64]`. The
  enum-alias form `Int64 = DType.INT64` is a value, and a checker reports "not valid as a type"
  while runtime accepts the subscript silently, which is the worst combination.
- **Configuration is decorator arguments, not class attributes.** Every config key in the body
  would take that name away from the schema, so a feature could never be called `name`,
  `source`, or `ttl_days`. Putting configuration on the decorator also makes a typo in it a
  static error with a suggestion. The base class is empty for the same reason, and a test pins
  that it declares nothing but dunders.

Rust is unaffected: the compiled JSON is identical to what the models produced before this layer
existed.

> `ponytail:` the frame returned by a historical join is not schema-typed, so
> `frame["click_count"]` is unchecked. The output schema is a function of a list argument, and
> a checker cannot turn `list[Field[Int64]]` into a typed frame. Polars has no
> `DataFrame[Schema]` yet ([#22119](https://github.com/pola-rs/polars/issues/22119), open since
> April 2025), so the options today are a user-declared `TypedDict` that restates the field
> names, or a third-party layer such as pandera. Ceiling: a misspelled output column is a
> runtime `KeyError` rather than a static error. Upgrade path: `DataFrame[Schema]` when Polars
> ships it, which needs no Feather-owned wrapper.

#### The historical read path

`FeatureStore(path)` reads and validates `feather.toml` through the core, imports the modules
it lists, and hands the compiled definitions back for the core to validate again. Nothing is
registered and nothing is generated.

`store.get_historical_features(entity_df=..., features=[...])` returns an Arrow table: the
entity frame's own columns in the surviving row order, then one column per requested feature
in request order. The frame's columns are reattached by row index, which is why the engine
returns `feather_row` rather than the labels it was handed.

One join runs per distinct view, however many of that view's features were requested. Under
`on_missing: "null"` every view covers every label row, so the result is row-for-row aligned
with the input. Under `"drop"` a row survives only if **every** requested view produced a
fresh value, so a row that `drop` kept never holds a null it was asked to remove. The engine
cannot express that rule because it joins one view at a time; the binding applies it.

The result is an Arrow table rather than a Polars or pandas one, so `feather` depends on no
dataframe library. `pl.DataFrame(result)` accepts it through the same PyCapsule interface the
input used. A pandas frame does not export Arrow buffers, so it has to be converted before the
call, which copies.

#### Definitions are imported, not compiled

Feature definitions are Python modules. Both the offline path and the serving path import
them directly. There is no `apply`-generated artifact, no registry, and no lockfile.

The consequence to accept: serving executes user code at import. Two rules keep that
tractable.

1. **The definition surface depends only on `feather`.** A feature view declares
   `click_count = Field(Int64)`, where `Int64` comes from `feather.types`. It does not
   import Polars, PyArrow, or a warehouse client. This keeps the serving image thin and makes
   the definitions importable in any environment that can import `feather`.
2. **Definitions are declarative data.** No module-level I/O, no network calls, no
   computation at import. A definition module is a list of constructor calls. CI imports
   every declared module and fails on import errors, which is the backstop for rule 2.

`feather.toml` lists the definition modules explicitly:

```toml
definitions = ["definitions/user_clicks.py", "definitions/user_stats.py"]
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

A feature service is a **named projection**: a name and a list of declared fields, or a
whole view to mean all of its fields. Nothing else.

```python
FeatureService(
    name="ranking_v3",
    features=[UserClicks.click_count, UserStats.ltv],
)
```

- **No entity declaration.** Entities come from the referenced views, so declaring them again
  would be a second source of truth.
- **No versioning, no logging config, no infrastructure.** Feast's feature service bundles all
  of that, which makes it registry-shaped, and this design has no registry.
- **What consumes it now.** The name is the wire form of a serving request: a `DoGet` ticket
  carries a feature service name, and `ResolvedService::resolve` turns it into a field set
  once at startup. In-process, a checked module-level list gives the same protection against a
  training script and a serving handler listing different features, so the name was doing
  nothing before the server existed. It is load-bearing across a process boundary now, and a
  service that names nothing is a boot failure rather than an empty response.
- **A raw field list stays available** as an escape hatch for ad-hoc and debug requests.

If logged features are ever adopted (see "Open design questions"), this is where their
configuration belongs.

#### Other language bindings

The language that defines the features is the language that reads them. A Go binding is what a
Go project needs to declare its views in Go and read them in its own process, the same way the
Python binding does. There is no Go binding yet: the workspace has three Rust members,
`feather-core`, `feather-py` and `feather-serve`, and no Go source. When one is written it will
link `feather-core` the way the PyO3 crate does, so the key encoding, the value codec, the TTL
check, and the point-in-time join stay one implementation rather than becoming one per language.

That is still the fastest arrangement: a read is a call in the same process, with no server to
reach and no hop to pay for.

Reading from another language no longer requires building anything. The server is built, and it
speaks Arrow Flight, so a Go service reads features the Python process declared by opening a
Flight client and decoding the Arrow response. What it costs is a network hop and a client
library, and what it buys is that the Go process never has to import the definitions.

**A client is not a second declaration, and the difference matters.** A client names a feature
service and gets back the columns that service resolved at startup. That is safe. What is not
safe is a second _process_ declaring the same views itself, which is a second source of truth
for the schema tag: the tag covers field names, dtypes, and declaration order, and does not
cover `ttl_days`. Two declarations that agree on fields and disagree on the TTL compute the same
tag, so the same stored value reads as fresh in one and expired in the other, and nothing
detects it. Harmless while one declaration exists, because the read-time check uses that one.
Mixing `ttl_days` into the hash, announced by the reserved `flags` byte, is the fix, and it is
cheap while nothing published depends on the current tag.

### 2. Online serving layer

#### Storage layout

One key per entity. All feature views that share an entity live under it, so one store read
covers every view an entity has values for.

```text
key:    {project}:{entity_name}:{encoded_entity_key}
fields: v:{view}   the view's encoded feature vector, all its features in one blob
        f:{view}   event timestamp of the last write for this view (int64 micros)
```

Both stores put an entity's fields in one contiguous run and read them together. The in-process
map holds a sorted slice of `(name, value)` pairs per entity. The embedded LSM has no hash type,
so it prefixes each record with the entity key's length, which makes an entity's records
adjacent and turns a per-entity read into one range rather than a point get per field. The
reasoning for that encoding, and the one cost it takes on, are in
[`online/fjall.rs`](../crates/feather-core/src/online/fjall.rs).

One key holds no entity's values: the project registry, `{project}:views`. Its fields are the view
names a refresh has declared — a view that every refresh so far has found no rows for has no
field — and its values are each view's newest event timestamp, encoded the way a freshness field
is. A refresh reads it to learn which views the previous refresh declared, writes it last, and
garbage collection diffs the two sets. Losing it is a worse failure than a crash before it is
written: a directory restored from an older copy, or handed to a fresh project, leaves entity
keys and no registry, and every later refresh then reads nothing, retires nothing, and leaves a
renamed view's fields in the store for good.

#### Entity key encoding

`encoded_entity_key` is the entity tuple encoded so that the result is unambiguous without
escaping:

```text
{len}:{value}|{len}:{value}
```

Each component carries its own byte length, so a `|` inside a value cannot be mistaken for a
separator: the parser reads the length, consumes exactly that many bytes, then expects the
separator. That leaves no escaping rule to get wrong and no restriction on key contents.

Four consequences:

- **All of a view's entities must be supplied.** Partial-key lookups are rejected at
  validation time rather than producing a key that can never match.
- **Component length is capped** at 512 bytes, so one pathological key cannot produce an
  unbounded store key.
- **A colon is refused in the project name and in every entity name.** Reading a project back
  out of a key means splitting on the first two colons, so a colon inside either name is
  ambiguous rather than escapable: garbage collection would fail to recognise those keys and
  skip them, leaking the fields it was meant to reclaim. The loader refuses such a name instead
  of writing keys it cannot read back. That is a breaking change for a `project` value that
  previously loaded. **A view name is refused as well, for a different reason:** a reference is
  `view:feature` split on the first colon, so a view named `a:b` is indistinguishable from view
  `a`'s field `b:count`, and the reference `a:b:count` resolves to whichever of the two came
  first. A colon in a _feature_ name is safe, because everything after the first separator is
  the feature.
- **No cluster hash tag.** The store is a directory one process opens, so there is no cluster
  to spread entities across and a project-level tag would only concentrate them. The question
  was worth answering before a networked tier was built; that tier was removed, and the
  question went with it.

The leading segment is a namespace, reserved so that later capabilities are additive rather
than a key-format change:

- `v:` one view's encoded feature vector. What v1 writes. One field per view rather
  than per feature, because the value codec writes a whole vector in a single
  fixed-stride blob; a per-feature field would either repeat the schema tag per feature
  or force the codec down to one column at a time and lose the stride.
- `f:` freshness metadata. Used for the read-time TTL check and for staleness reporting.
- `t:` pre-aggregated tiles. Reserved for the tiling upgrade path, see "Scalability".

Values carry a small schema tag, which **verifies** a schema rather than carrying one. The tag
is a hash of the field names, dtypes, and declaration order, and the reader supplies the schema
it expects. Serving treats a value whose tag does not match as missing rather than attempting to
decode it. This is what makes a dtype change a null window instead of a decode failure, and what
allows the encoding to change later without a migration. It is also what lets an independent
implementation agree with this one without a shared artifact.

#### Value byte layout

A field value is one view's feature vector for one entity:

```text
[tag: u32 LE][flags: u8][null bitmap][fixed-width columns, declared order][variable-width tail]
```

The property that matters is that **fixed-width columns are concatenated with no per-value
length prefix**, so locating column _k_ is arithmetic rather than parsing. Variable-width
values (strings, lists) live in a trailing section behind an offset table, so a reader that
wants only fixed-width columns never touches them.

`tag` identifies the view's schema version: field names, dtypes, and declaration order. A
mismatch means the value is treated as missing rather than decoded. `flags` is reserved and
written as zero, so an encoding-level change has somewhere to announce itself without moving
anything.

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

Because the reader supplies the declared schema with each read, rather than the value carrying
it, decoding needs no per-value type dispatch. That is the point of the layout.

#### Read path

For a request covering any number of feature views on the same entity, serving issues **one
store read per entity**, asking for one field per view plus one freshness field. View count does
not multiply reads. The caller's requested features are projected out after decoding, so reading
two features from a wide view costs decode time for the columns it did not ask for.

Feast issues one `read_from_online_store` call per feature view despite advertising entity
collocation; a user with 11 feature views reported "abysmal" retrieval times
([#3596](https://github.com/feast-dev/feast/issues/3596)). The same issue notes all reads are
synchronous, and
[#2247](https://github.com/feast-dev/feast/issues/2247) reports the same for DynamoDB. The
rule here is absolute: never one read per view. `online/mod.rs` has a test that counts store
reads for a request naming four views and asserts the count is one, because the rule is a
property of the layout, and a property nothing checks is a property that erodes.

Requested fields are resolved against already-loaded in-process definitions, so resolution is a
table lookup rather than I/O. Over Flight the resolution happens once at startup rather than
per request, and the resolved field set is what the ticket's service name selects. Feast
resolves registry metadata on every request, which profiling shows is more than half of
`get_online_features` execution time
([#4710](https://github.com/feast-dev/feast/issues/4710)); that cost does not exist here,
because there is no registry and no separate process to resolve against.

Batch within a request, not across requests. A staff-level design for a comparable system
measured this and rejected cross-request micro-batching: at 1M requests/second across 2000
hosts, a 1ms batching window captures about 0.5 requests, which is not enough to justify the
added latency. The useful batching is within one request's entity list, which the
one-read-per-entity rule already provides. Do not build a cross-request aggregator.

TTL is enforced at read time by comparing `f:{view}` against the view's `ttl`. An expired
value is returned as null, which is the documented contract. The freshness field is read in the
same call, so this check costs no additional read.

#### TTL and reclamation

**One mechanism: the read-time check.** `f:{view}` plus the declared `ttl`. This is the
correctness path, and it is the only one: a value is served because the read path decided to
serve it, never because something else removed it.

Nothing reclaims. `FjallStore` drops the `expires_at_unix_secs` a write carries, and
`MemoryStore` records it without acting on it, because neither has a clock-driven expiry to
offer and the read-time check makes reclamation a disk-space question rather than a correctness
one. The cost is that an expired value holds its key until the next refresh rewrites that
entity or the view is retired. A full refresh rewrites every value anyway, so the exposure is
bounded by how long a view goes unrefreshed.

The write path still computes the absolute expiry and passes it through, because it is the
right value to pass and re-deriving it later would be a second rule to keep in step.
`materialize` rounds it up, so a store that did reclaim would err late rather than early; the
comment on `value_expiry_unix_secs` carries the argument, including the clock skew that no
rounding direction covers.

The freshness field would not be expired even by a store that reclaimed. The read path tells
`expired` from `never written` by comparing it against the TTL, so reclaiming it would collapse
two states the contract keeps apart.

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

The distinction is preserved in the core: `Missing` is an enum with a variant per cause, and
`read_entities` returns which one applies. **The transport flattens all three to an Arrow
null.** A client cannot tell a value that was never written from one that expired, or from a
materialized null, and the online contract is nulls.

The default is nulls-only because Arrow's null bitmap cannot separate the two cases, so
exactness would have to be a separate column, and paying for it unconditionally would widen
every response for a distinction most consumers ignore. Nothing returns that column yet.
`Missing::as_str` exists and the enum is exhaustive, so adding a state column is a field on
the response and a column per feature; what it needs is a decision about whether the online
contract is nulls or a state mask, because a client that starts depending on the mask cannot be
given nulls alone afterwards.

#### Freshness contract

What the store promises:

```text
max_staleness = refresh_interval
```

- `refresh_interval` is the deployment's schedule, so this number is per deployment rather
  than a constant. Both stores return the newest value materialized into them, so both are as
  stale as that interval; they differ in whose writes they can see, since an in-process read
  sees only what this process's refresh wrote and a read against the directory sees any
  process's.
- TTL expiry caps staleness by construction: an expired value reads as missing rather than as
  a stale value.

The store exposes observed lag per view, derived from `f:{view}`, so consumers alert on
reality rather than on the formula. The metric is the alert signal; the formula is what gets
written down.

#### Where the store lives

The store is embedded and in-process. `feather.toml` with no `[store]` table serves from an
in-process map that dies with the process; a `[store]` table names a directory, and both a
materialization and `FeatureStore.serve()` open it. It is a database that one process opens at a
time, so the server is the only thing that opens it, and that is what replaces a shared tier: a
reader in any language goes to the server, not to the store. A point read costs single-digit
microseconds warm, where a networked KV store pays a round trip per entity.

The relationship is substitution rather than layering. The store is not a cache in front of
something else, and there is no second copy to keep correct. It is the trade SQLite and Postgres
present, with the embedded store on one side and the server on the other.

Both stores are the same trait. `read_entities<S: OnlineStore>` is the only caller in the
serving path, so which store is behind it does not touch the Python surface or the offline path,
and choosing between them is configuration rather than a code change.

Two things differ, and both matter:

- **Where writes land.** `MemoryStore` writes into the one map this process holds, which dies
  with it. `FjallStore` writes into a directory that outlives the process. Both satisfy `write`
  and `delete_fields`, and both are mutated where they stand: a refresh is not staged into a copy
  and swapped in.
- **How stale a read can be.** `MemoryStore` serves what the last refresh in this process wrote,
  so staleness is bounded by the process's own lifetime. `FjallStore` serves what any refresh
  wrote, so staleness is bounded by the refresh interval across processes. Writes take
  `&mut self` and reads take `&self`, and the Python binding holds its store behind one lock for
  a whole refresh, so a read sees a completed refresh rather than a partial one. That is
  exclusion rather than an atomic swap, so the bound is the interval and not zero.

The store is not responsible for expiry. `OnlineStore::write` documents that a store which
cannot honour an expiry "writes the value anyway: the read-time TTL check in `read_entities` is
what decides whether a value is served, so an unexpired leftover costs reclamation and never
correctness." Neither store needs a per-field TTL, which is what lets a store with no clock sit
under the serving path.

The in-process store is the embedded LSM, `FjallStore`, and `MemoryStore` is the test double.
The two are not both production positions: an earlier decision made the in-process store the
default to keep one implementation of these semantics honest, and adding the LSM under that rule
means demoting `MemoryStore` rather than joining it. See
[`embedded-online-store.md`](./embedded-online-store.md) and
[`serving-transport.md`](./serving-transport.md).

#### Serving measurements

**Superseded.** The figures below were measured against a shared Valkey, which is no longer a
position. The serving path is now an embedded store behind Arrow Flight, and its measurements,
including the read cost and the cost of hosting the server in Python, are in
[`serving-transport.md`](./serving-transport.md). They are kept as the record of the
networked-store numbers, which are the baseline the embedded store is measured against, and for
the version-floor argument under "TTL and reclamation", which described a store that reclaimed.

Measured against Valkey directly, on a container limited to 2 CPUs and 512MiB with `maxmemory`
384MiB and `allkeys-lru`, over 100k entities across 4 views of 8 features each, with 66-byte
vectors:

| Measurement                  | Result                                    |
| ---------------------------- | ----------------------------------------- |
| read, 1 entity x 4 views     | p50 0.327ms, p95 0.411ms, p99 0.741ms     |
| read, 100 entities x 4 views | p50 2.36ms, p99 5.10ms, 24.2us per entity |
| write, materialization shape | 193k vectors/s, 386k field writes/s       |

Single-entity p99 is sub-millisecond across four views, which is the one-read-per-entity rule
and the fixed-stride encoding doing their job. The 17.7ms tail maximum on single reads is
unexplained and was never investigated.

These were reproduced with `mise run bench:load` against a Valkey container, which the removal
of the tier took with it. The task name survives and now builds the `feather-serve` examples;
[`serving-transport.md`](./serving-transport.md) records how to run it and which parts of its
harness ship.

These numbers are from one host and one container shape. Treat them as an order of magnitude,
not a guarantee.

### 3. Offline engine (query routing)

Historical queries are routed between two execution tiers. Both execute in the local DuckDB
engine; they differ in where the source data lives and how it is reached. The engine choice and
the format rules are recorded, with the evidence behind them, in
[`engine-and-format-decisions.md`](./engine-and-format-decisions.md).

#### Tier 1: local files and database tables

- **Engine.** Local DuckDB via the `duckdb` Rust crate.
- **Entity frame input.** The caller's entity frame is an Arrow record batch. It is appended to
  a temporary table with DuckDB's Arrow appender (`appender-arrow`) and joined as the left side
  of the `ASOF` join. Columnar, no per-row work, no serialization, and no file to clean up. None
  of the three routes costs build time the others do not, since the engine is a prebuilt library.

  Measured on an Apple M2 with 8 cores and 8 GiB, 500k entity rows against a 2M-row feature
  table, minimum of 3 repetitions, three runs:

  | Route          | Load, 500k rows  | Join         |
  | -------------- | ---------------- | ------------ |
  | Arrow appender | 3.2ms to 3.3ms   | 33ms to 35ms |
  | Row appender   | 29.0ms to 29.6ms | 33ms to 35ms |
  | Temp Parquet   | 21.3ms to 23.3ms | 39ms to 40ms |

  The appender is about 9x the row appender and about 7x the Parquet route on load. Parquet also
  pays for the decode inside the join, which is why it is the only route above 35ms, and it
  leaves a file behind. The join is `count(*)` over the `ASOF` join, so these figures exclude
  materializing the result columns.

  `ANALYZE` on the appended table is not required. The join moved by under 3% in either
  direction across runs, including one run that was 4.1ms slower after `ANALYZE`, which is
  inside the roughly 15% run-to-run spread on the join itself. The concern that a table filled
  through the appender carries no statistics did not produce a measurable planning penalty at
  this size.

  `crates/feather-core/examples/duckdb_input.rs` runs the comparison:
  `ENTITIES=500000 cargo run --release -p feather-core --features offline --example duckdb_input`.
- **Spill behavior.** DuckDB spills automatically: temporary storage is on by default, and a
  query that exceeds `memory_limit` writes its intermediates to the temp directory rather than
  failing. It does not use swap. Spilling is a safety net and not a plan, and the ceiling is
  real: DuckDB cancels the query with an out-of-memory error when the spill directory exceeds
  `max_temp_directory_size`, so the budget is one local disk rather than "unlimited".

  Four things make that budget usable, all set when the engine opens in
  `crates/feather-core/src/offline.rs`:

  - **The spill directory is private to the engine.** DuckDB removes it when the instance shuts
    down, so a shared one lets the first engine to finish delete another's live spill files.
    It is created at open and removed when the engine is dropped.
  - **Its parents are created, not assumed.** DuckDB creates the directory it spills into but
    not its parents, so a missing parent otherwise fails with an IO error at the first spill,
    under load, rather than at startup.
  - **`memory_limit` is derived from the container.** DuckDB's default is 80% of the machine's
    RAM, which inside a container can exceed what the container may use, so the kernel kills
    the process before DuckDB ever spills. A cgroup limit, when there is one, becomes the
    ceiling at 80% of it.
  - **Its size is capped.** DuckDB's default is 90% of free disk, which an engine embedded in
    someone else's process has no business taking.

  The directory must be local disk. Every spill read and write is synchronous, so on a network
  mount each block becomes a round trip, and object storage cannot be a spill target at all:
  spilling goes through the local filesystem.

#### Tier 2: open table formats on object storage

- **Engine.** DuckDB plus cloud object storage.
- **Mechanism.** The `iceberg` and `delta` extensions read table formats; `httpfs` provides
  S3 and generic HTTP access, and the `azure` extension covers Blob Storage. `httpfs` issues
  HTTP range requests, so a scan fetches only the byte ranges it needs rather than whole
  files. All of them are loadable rather than compiled in; see "Loadable extensions".
- **Consequence.** Compute runs locally, so there is no warehouse compute charge. Metadata
  operations (manifest reads, snapshot resolution) still hit the catalog and are the usual
  source of latency on high-file-count tables.
- **Warehouses.** A warehouse will be read through its Iceberg REST catalog rather than a
  vendor-specific driver, so Snowflake, BigQuery, Databricks and Amazon S3 Tables would be one
  source kind instead of four. `iceberg_snapshots(...)` would then supply the source pin, and
  `iceberg_to_ducklake(...)` would make a metadata-only copy into a DuckLake catalog so
  repeated reads would get DuckLake's filter pushdown without copying data. The vendor extensions
  (`snowflake`, `bigquery`, `onelake`) would stay as fallbacks for warehouses with no Iceberg
  surface, and are community extensions. Azure is Blob Storage filesystem access plus Iceberg
  or ADBC, not a Synapse connector.
- **What is not a default.** `ducklake` is the recommendation for a source currently held in
  Postgres. `vortex` is a core extension with filters evaluated on compressed data, measured
  by DuckDB at 18% ahead of Parquet v2 on TPC-H SF100, but it is third-party maintained and
  pre-1.0, so it is selectable per source and opt-in. Only `parquet`, `httpfs`, `icu` and
  `json` are Primary tier, so Parquet on object storage is the only fully supported read path.
  The per-source `format` that makes `vortex` selectable is in place; the table formats are
  not implemented. The layout measurement in
  [`engine-and-format-decisions.md`](./engine-and-format-decisions.md) is why that order is
  the honest one: clustered keys compressed a feature source 30.6% smaller and cut the bytes
  a point-in-time join read by 31.5%, which is a larger and cheaper win than any format
  change. It also corrected the reason, which had been row group skipping: for this join the
  reader skips nothing, and the win is compression.

#### Source schema validation

Read-time validation is the drift check, since there is no stored fingerprint (see "Why
there is no registry"). Validating on every read is too expensive for a warehouse source, so:

- Validate **once per process per source** at startup, and cache the resolved schema.
- Re-validate on a schedule in long-lived processes.
- Record the source's snapshot identity alongside the result, where the source has one: a
  Parquet file's metadata, an Iceberg, Delta or DuckLake snapshot, or — for a warehouse with no
  Iceberg surface — a warehouse table version. A warehouse reached through its Iceberg REST
  catalog would yield a snapshot id from `iceberg_snapshots(...)`, so it is the same mechanism
  rather than a vendor-specific one.
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
returning a planner-dependent value. A row with a null `created_timestamp` sorts last, so
it loses to a row that carries one.

The tie-break has one column, so rows that share the created timestamp as well are
undecidable too, and a third such row is no more decidable than a second. There is no
further key to order by: the view declares the entity, the features and the two timestamps
and nothing else, and a row's position in the file is not something a Postgres table or an
object-storage prefix offers at all. Those rows are therefore refused with the same error as
the undeclared case, naming the created timestamp they are tied on. A refresh refuses them
before writing anything of that view's, so serving never answers from a source training
would not read.

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

Materialization computes feature values from offline sources and writes them to the online
store: the embedded LSM where `[store]` names a directory, the in-process map otherwise.

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
2. Each batch is encoded into the store's own records and written before the next batch is
   pulled, one key per entity holding that view's vector and freshness field.
3. Writes are batched. One `OnlineStore::write` call carries up to `WRITE_CHUNK` (4096) field
   records, which is a ceiling rather than a typical size: a call usually carries one streamed
   Arrow batch, which is far fewer rows. It exists so that a caller handing over a very large
   slice does not build one enormous journal entry.

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
duplicate run can corrupt the watermark. A full refresh is idempotent by construction for an
unchanged declared set, so a duplicate run wastes work and produces the same state. Two runs
with _different_ declared sets are the exception: each computes its retired set from the
registry it read, and the two write the same fields, so interleaving them can leave the
registry holding a timestamp from the older read while each report counts the same rows. A
serialized run is what removes that case, and serialization belongs to the deployment rather
than to the refresh, which takes no lock of its own. Two deployment mechanisms do it, both
design only, since nothing in this repository schedules a refresh or issues a lock: the
`concurrencyPolicy: Forbid` above, or, for a scheduler without it, a non-blocking lock around
the run (`SET NX PX`, TTL set to the expected maximum runtime) that exits 0 when the lock is
already held. No job-state table, and no already-running detection.

Freshness is surfaced from `f:{view}`, which the write path already maintains. The maximum
lag across views is the metric to alert on, exported as a gauge rather than discovered by a
consumer noticing stale features.

### Garbage collection

Full refresh only writes fields for currently defined views, so fields belonging to renamed or
removed views would otherwise persist forever. Feast has exactly this leak, growing the online
store indefinitely because it only deletes data when the last view for an entity is gone
([#3596](https://github.com/feast-dev/feast/issues/3596)).

The refresh knows which views retired, because it reads the project registry before it writes
anything and diffs it against every view the project declares — not against the selection, which
is why refreshing a subset retires nothing. What it does not know is where
their fields are. A retired view's entities do not have to appear in any source any more, and
their keys carry no trace of which view put a field in them, so the view set names the orphan
but not its location: the project's keyspace is the only complete list. When, and only when, a
view retires, the refresh therefore walks that keyspace and removes the retired views' `v:{view}`
and `f:{view}` fields from every entity key it finds, leaving every other field alone. A key not
shaped like an entity key is skipped, because everything else under the prefix belongs to
something the retired view never wrote.

On the embedded LSM that walk is a full keyspace scan, not a prefix scan. The length prefix
that makes an entity's records adjacent also means the entity key does not start the record, so
the scan parses every record and filters by project. That is affordable here because the only
caller is this one, and a full refresh that retires a view is already writing every value. The
`ponytail:` marker on `scan_entity_keys` names the upgrade: a second keyspace holding one record
per entity key, which turns the walk into a prefix scan at the cost of a second write per
entity.

The walk is rare by construction, which is the point of the registry: a project whose declared
views have not changed retires nothing and walks nothing. It is idempotent as well, because
removing a field that is already gone is a no-op, and the registry is written after the walk
rather than before it, so a run that dies partway leaves a state the next run repairs — as long
as that next run declares the same view set, since the retired set it computes comes from the
registry rather than from the run it is repairing. This is the reason a refresh is not a pure
function of its inputs, and it is the only part of it that is not.

### Ceiling

Full refresh cost scales with total entity-view pairs, not new data. With one key per entity and
one field per view, a refresh writes `entities x views` field values, encoded and streamed one
Arrow batch at a time. There is no write-throughput figure for the embedded store: the
`bench:load` task builds the serving examples, and nothing here measures a refresh's write
rate. The Valkey measurement recorded under "Serving measurements" is 386k field writes per
second, on a container limited to 2 CPUs, and it is a different store on different hardware.

So the ceiling is stated as a shape rather than a number: a refresh is bounded by how fast the
engine can reduce the source to one row per entity and by how fast the store can absorb the
encoded batches, and neither has been measured against a dataset large enough to matter. What is
measured is the join (see the offline engine section) and the read. See "Scalability" for the
documented path past this.

---

## Scalability and the upgrade path

### The two decisions that keep retrofits additive

Everything deferred from v1 is additive or a rewrite depending on two choices, both of which
are cheap now and structural later:

1. **Key layout: one key per entity, one field per feature view.** Incremental updates later
   become "set a few fields on a few keys". Tiling later becomes "add `t:` fields beside the
   existing ones". Choosing a key per feature, or one opaque blob per entity, makes both a
   storage format rewrite.
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

Chronon requires Flink for this, and Feast ships the same idea as tiling with intermediate
representations, which in its streaming form needs Spark or Ray. Feather's version needs neither:
the same tiles are computed in DuckDB during batch materialization, with the same idempotency and
the same bounded reads. That is a deployment difference rather than a design difference, and this
document should not claim more than that.

### Tile encoding: partial aggregates, never final values

A tile stores a **partial aggregate**, not the finished feature. This is the rule that makes
merging correct. Feast states it and implements it: storing final aggregated values per tile is
fast but "often incorrect when merging", which is why Feast's tiling stores intermediate
representations too. The two designs converge here, so the choice of stored aggregates is not a
differentiator.

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
versioned and the tag acts as the check.

Chronon's own guidance is the right trigger: "If your hottest keys don't exceed a few
thousand events per day, the untiled approach may still be sufficient."

### Zero-downtime schema changes

v1 accepts a null window during a dtype change: serving sees a value whose schema tag does not
match and returns null until the refresh completes. Correct, never a decode failure, but not
zero-downtime for large refreshes.

The upgrade is double-buffered generations: write the new encoding under a new generation,
flip a pointer atomically, then let the old generation go. This is additive, because values
already carry a schema tag and fields are already namespaced. It costs 2x peak storage during a
refresh, which is why it is not the v1 default. The old generation cannot be reclaimed by
expiry, because nothing reclaims; a refresh has to overwrite or the view has to be retired.

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

`feather.toml` holds infrastructure. Python modules hold definitions. The core owns the file:
`crates/feather-core/src/settings.rs` is the schema, and it parses the TOML, resolves `${VAR}`,
and validates the keys. `python/feather/settings.py` is a typed view of the validated result, so
Python code gets autocomplete without a second implementation of the schema. The README links
here rather than restating the keys.

```toml
project = "ad_recommendations"
definitions = ["definitions/user_clicks.py", "definitions/user_stats.py"]

[connections.snowflake_prod]
type = "snowflake"
account = "xy12345.us-east-1"
warehouse = "PROD_WH"
username = "fs_runner"
password = "${SNOWFLAKE_PASSWORD}"

[connections.s3_lake]
type = "s3"
region = "us-east-1"
key_id = "${AWS_ACCESS_KEY_ID}"
secret = "${AWS_SECRET_ACCESS_KEY}"
# An S3-compatible store rather than AWS. The key pair comes from the same environment
# variables, but the endpoint is not AWS's, and DuckDB addresses S3 virtual hosted style
# over TLS by default, which a server reached at a host and a port does not answer.
endpoint = "minio.internal.svc:9000"
use_ssl = false
url_style = "path"

[connections.pg_prod]
type = "postgres"
host = "pg.internal.svc"
port = 5432
database = "warehouse"
user = "fs_runner"
password = "${POSTGRES_PASSWORD}"
# Optional. Absent leaves libpq's own behaviour, which already prefers TLS when the
# server offers it, so this is only for a deployment that will not settle for that.
ssl_mode = "verify-full"

[store]
# The embedded LSM's directory. One process opens it at a time, so in a
# deployment `serve()` is the process that opens it and a refresh runs
# against the same directory from a job that opens it while no server is
# running. Absent means the in-process map, which dies with the process.
path = "var/feather-store"
# Resident block cache, in bytes. Default 64MiB.
cache_bytes = 67108864
# Write buffer, in bytes. Default 64MiB, which is fjall's own and the size
# its measurements point at: across scattered updates a 4MiB memtable
# wrote 5.3x the logical bytes and a 64MiB one wrote 1.1x.
memtable_bytes = 67108864
```

An `endpoint` is what makes the entry an S3-compatible store rather than AWS, and it brings
`use_ssl` and `url_style` with it, because DuckDB's defaults are AWS's. Its S3 client
addresses buckets virtual hosted style and speaks TLS, while a server reached at a host and a
port answers path style in the clear, so an `endpoint` implies path style and is paired with an
explicit `use_ssl`. Both are written out in the example rather than left to that implication,
since the implication exists to make the common local case short, not to make the two settings
invisible. The integration test in `crates/feather-core/tests/s3_integration.rs` runs against
RustFS, an S3-compatible server, and joins the same Parquet rows read from a local file and
from `s3://` to check that the two agree.

The example's `pg_prod` is a deployment's own database, on the port that server listens on.
`docker-compose.yml` publishes the Postgres the integration test uses on 5433, because 5432 is
the port a locally installed server already holds, so a `feather.toml` pointed at the compose
server writes `host = "127.0.0.1"` and `port = 5433` instead, with `user = "feathertest"`,
`password = "feathertest"`, and `database = "feathertest"`, which are the three the compose file
sets. A connection needs all five, and a note giving only two of them cannot be used.

Three absences are deliberate.

- **No `offline_store`.** A source belongs to the view it feeds and is declared on that view,
  so there is no deployment-wide offline store to name. See "Offline engine".
- **No `type` on `[store]`.** There is one online store, and a pluggable one is a non-goal.
  The table name carries the kind, so a key that could only hold one value is not written down.
  Its absence is a position rather than an omission: no `[store]` table means the in-process
  map, which is the right default for a project read from the same process that refreshes it.
- **No compute engine.** Every source runs on DuckDB, so a local file, an object-storage
  prefix, and a database table differ only in how the source is declared, not in what
  executes it.

`connections` holds the credentials a source must not carry. A Postgres source names one with
`connection`, and that name is checked where definitions and settings first meet, so a source
naming a connection the project does not have, or one of the wrong kind, is a startup error
rather than a surprise at the first read. The kinds with no consumer yet are declared for the
same reason anyway: a credential belongs in this file rather than in a committed definition
module.

A view's source is tagged with its kind. A file source is
`{"type": "file", "path": "data/user_stats.parquet"}`, and a Postgres source is
`{"type": "postgres", "connection": "pg_prod", "schema": "public", "table": "user_stats"}`.
The tag selects the variant, so a path is not read as a table because a key happened to be
present, and a key belonging to the other kind is rejected rather than ignored. Python writes
`FileSource(path=...)` and `PostgresSource(connection=...)`; the wire shape itself is the
core's.

Which kind a view declares changes nothing downstream. A Postgres table is read with
`postgres_scan`, an object-storage prefix through `httpfs`, and a local file off the disk, and
all three are then the same `ASOF` join. `crates/feather-core/tests/postgres_integration.rs`
checks exactly that by joining the same rows once from a local Parquet file and once from a
table.

### Secrets

`${VAR}` in any config value is interpolated from the process environment at load time, by the
core rather than by a binding, so every language resolves it the same way. A referenced variable
that is unset is a load error, not an empty string. `feather.toml` is committed, so it must never
contain a literal credential. Either interpolate the value or have the platform inject the
variable before load. A literal secret in this file is a bug.

A `[connections]` entry is the only place a credential is expected, and it is the only
consumer of `${VAR}` today. Credentials are never declared in a definition module, because
those are committed and every process reads them.

A configured entry is handed to DuckDB as a secret when the engine opens, which is what makes an
`s3://` path work. With no entry, DuckDB's own credential chain applies instead, so a cloud
instance role or the standard environment variables are found with nothing configured at all.
That is the intended shape in the cloud: no secret in the file, nothing to rotate there.

### Loadable extensions

The compiled-in set is DuckDB's own release configuration rather than something this repository
chooses, because the engine is a prebuilt library and nothing here builds it. The library Feather
links carries the Parquet reader, JSON, ICU and TPCH, which is the configuration DuckDB publishes;
it does not carry `httpfs`, `postgres`, `iceberg`, or `delta`, so those are loadable extensions
rather than libraries. Nothing in this repository's Cargo manifest can turn them on, because a
manifest selects crate features rather than a DuckDB build:

- `httpfs` is loaded on the first read of a URI path, and a configured object-store connection
  loads it at open, because the `s3` secret type comes from the extension itself. A project
  reading only local files never loads it.
- `postgres` is loaded on the first read of a Postgres source. Nothing loads it at open, because
  a Postgres connection travels in the relation as libpq's own connection string rather than as
  a secret.
- `vortex` is loaded on the first read of a source whose `format` names it. It is loadable
  because it is third-party maintained and pre-1.0, which is also why nothing in Feather is
  compiled against it. See "The format is a per-source choice" in
  [`engine-and-format-decisions.md`](./engine-and-format-decisions.md).
- An image that cannot reach the extension repository bakes the files in and points
  `Limits::extension_directory` at them, so `INSTALL` is a no-op and `LOAD` finds them locally.

Support tiers decide what may be a default. Primary extensions are covered by community
support; Secondary ones are best-effort, still bugfixed and shipped with each release. Only
`parquet`, `httpfs`, `icu` and `json` are Primary. `iceberg`, `delta`, `ducklake`, `postgres`,
`azure`, `mysql`, `sqlite` and `unity_catalog` are Secondary, and `vortex`, `lance` and
`motherduck` are third-party maintained. Parquet on object storage is therefore the only fully
supported read path in DuckDB's tiers, and it is the format a file source gets when it does not
name one.

A `File` source may name its own `format`, so the choice is per source rather than per project
and one project can read two formats side by side. The tier a format belongs to is declared in
the core on the format itself, not in `feather.toml` and not in a definition module: it is a
property of DuckDB's extension rather than of a deployment or of the data, so a file carrying
it could only hold an assertion that could be wrong.

The cost of that design is the same one the warehouse extensions carry: an extension is built for
one DuckDB version and one platform, and the path it lives at names both. An extension that lags
the engine fails to load rather than degrading.

### Backend coverage

`[connections]` shows the three kinds with a settled schema: Snowflake, S3, and Postgres. Each
declares `type` plus its own keys, and a kind is added when a source needs it. The kinds that read
from a server are covered by an integration test against a real one, RustFS for the S3-compatible
case and a Postgres container for the table, because a reader that only ever meets a mock is not
known to work. The `file` kind is covered locally instead: it is the control inside both of those
tests, so a file read and a server read are joined to the same rows and compared, and a local
Parquet file is not a server to start. Snowflake declares a connection but has no reader yet, so
nothing reads through it and no test does.

The intended warehouse kind is Iceberg rather than one kind per vendor, since an Iceberg REST
catalog reaches Snowflake, BigQuery, Databricks and Amazon S3 Tables through one attach. That
would make a `bigquery` or `azure` connection kind unnecessary for reading a warehouse, and it
narrows the open work in issue #7. Azure Blob Storage stays a filesystem concern rather than a
warehouse one.

---

## Deployment

1. **Local development.** `pip install feather-py`. Everything runs in-process: local
   Parquet, an in-memory DuckDB, and the in-process online store, so nothing external is
   needed. `feather refresh` and `get_online_features` share one process's store, and the values
   are gone when it exits.
2. **Serving.** `[store]` names a directory. `FeatureStore.serve()` opens it and blocks, so the
   serving process is the store's owner, and a client in any language reads over Arrow Flight.
   The ceiling is that directory's disk, and the staleness bound is the refresh interval.
3. **Refreshing a served project.** The database is opened by one process at a time, so a
   refresh cannot run while a server holds it. This is the constraint a deployment has to
   schedule around, and it is the reason the scheduling below is design only rather than a
   recipe: the obvious shape is a `CronJob` per view group that runs while the server is down,
   which trades serving availability for refresh correctness, and a project that needs both has
   to run the server against a store that permits a second writer.

There is no Helm chart and no Kustomize manifest, and there will not be one before v1. What
ships is a library, a `serve()` that turns the calling process into a server, and a refresh. A
deployment wires those three together; this repository does not.

The single-writer rule is the constraint a deployment has to schedule around before anything
else, and nothing in this repository resolves it.

---

## Naming

The distribution is `feather-py`, because `feather` is already registered on PyPI. The import
name is `feather`.

That pairing is what creates the collision this section exists to name. Installing
`feather-py` puts a top-level `feather` on the path, and the PyPI package `feather` (the
Feather dataframe format) provides the same module name. An environment must not contain
both, because which one an import resolves to depends on install order. This is accepted for
now rather than resolved. If it becomes a problem, the import name should change before the
first release, since renaming after publication is far more expensive.

`crates/feather-py` is the PyO3 binding crate and shares the name. It is internal and is not
published.

---

## Recorded decisions and rejected alternatives

Recorded so they are not re-litigated. Each cites the evidence that drove it. The engine and
format calls, and the alternatives rejected for them, are in
[`engine-and-format-decisions.md`](./engine-and-format-decisions.md).

### The core validates, not the binding

**Rejected:** one validator per language, with the core trusting whatever it deserializes.

The first version of this design had Python validate `feather.toml`, the entity definitions, and
the feature views, and had Rust deserialize the result without re-checking it. Two facts made
that untenable. The core already re-checked part of it, because `Definitions::validate` has to
resolve cross-references Pydantic cannot see, so there were two validators rather than one and
the "does not re-validate" claim was false. And a second binding would have had to reimplement
the schema, the `${VAR}` rule, and the reference checks, which is a third and fourth
implementation of rules that only need one.

The core is now the authority for both halves: `crates/feather-core/src/settings.rs` reads and
validates `feather.toml`, and `Definitions::validate` owns the definition rules. A binding keeps
its own layer for authoring ergonomics and for reporting a mistake as early as its language can,
and that layer is a duplicate rather than the rule.

The cost is that a binding which wants the best error message does the work twice. The benefit is
that adding a language means writing an authoring surface, not a second validator.

### The join materializes its result rather than panicking

**Rejected:** reading the join's Arrow batches straight out of the statement.

`duckdb`'s Arrow iterator panics on a step error rather than yielding one:

```rust
Err(err) => panic!("Failed to fetch Arrow record batch: {err}"),
```

This library is called across FFI, so a panic here aborts the host process instead of
returning a bad result, and the caller cannot catch it. `CREATE TEMPORARY TABLE ... AS`
reports the same failure as a value, so the join is executed into a table and only a plain
scan is read through the Arrow path.

The cost is materializing the result. The ordering is applied to the scan rather than at
materialization, because a sort performed while building the table would not order a later
scan, and the row order is the thing that keeps labels and features aligned.

### The online store is embedded

**Rejected:** Valkey as the only store, with the in-process `moka` cache as an optimization in
front of it. That tier was built, measured, and then removed; the reasoning below is why it was
built, and the store it was rejected for is what shipped.

A two-tier design has to keep the L1 correct, which for Valkey means client-side caching in
broadcasting mode, a RESP3 redirect connection that was implemented here because valkey-glide has
no Rust support, reconnect handling with a full flush, a non-overlapping-prefix constraint, and
an always-on fallback TTL. All of that existed because there are two copies of the data.

Inverting the tiers removes it. With one copy there is nothing to invalidate, the store need not
be Redis-family at all, and `OnlineStore::write` already permits a store that ignores expiry,
because the read-time TTL check is the authoritative path. The cost is that the store is opened
by one process at a time, which is the constraint a deployment schedules around, and that a read
can be as stale as the refresh interval.

Evidence: [`embedded-online-store.md`](./embedded-online-store.md) and
[`serving-transport.md`](./serving-transport.md). The rejected cache design is retained,
unprioritized, in [#25](https://github.com/chandlerok/feather/issues/25).

### Spill is local, capped, and private

**Rejected:** leaving `temp_directory` and `memory_limit` at their defaults, or at literals.

DuckDB's own defaults are 80% of the machine's RAM and 90% of free disk. Both are wrong for an
engine embedded in someone else's process:

- 80% of the machine is more than a container may use, so the kernel kills the process before
  DuckDB reaches the point of spilling at all. The limit is derived from the cgroup when there
  is one.
- 90% of free disk means a spill can fill the host. There is a cap, and exceeding it cancels the
  query rather than growing.
- A shared spill directory is unsafe, because DuckDB removes the directory when the instance
  shuts down. Each engine takes a private subdirectory.

Measured before these were set, with a one-million-row source and 100k labels: a join at 8, 16,
and 24 MB spilled megabytes and still failed, and succeeded at 32 MB while spilling 7.3 MB, and
at 64 MB while spilling 1.7 MB. Spilling is adaptive rather than a capacity guarantee. That is
the reason the scaling answer is partitioning the work, not relying on the spill to absorb it.

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

### Typed references, not strings

**Rejected:** `"view:feature"` strings as the reference a caller writes.

Feast is the cautionary case. Its `my_fv[[selections]]` syntax looks object-shaped, but it
builds a `FeatureViewProjection` that exists "for 1 second", after which "everything goes back
to `feature_ref` strings" ([#1907](https://github.com/feast-dev/feast/issues/1907)). The object
form is surface sugar over a string interior, so nothing downstream is checkable and a
misspelling is a runtime lookup failure. Tecton's `my_feature_view[["count"]]` has the same
shape: objects for the definition, strings for the selection.

References are therefore objects, and the string form exists only as a derived rendering:

- `features=[UserClicks.click_count, UserStats.ltv]` selects a subset.
- `features=[UserClicks]` selects every field on a view.

One unchecked entry point is kept on purpose. A raw `"user_clicks:click_count"` string is
useful for ad-hoc and debug work, so it lives in its own argument rather than in the same list,
which is what keeps the checked path checked.

Rejected alternatives:

- **Module-level attribute objects**, as Tecton declares with `Attribute`. Checkable, but it
  drops the view namespace, so a reference is `click_count` rather than
  `UserClicks.click_count`, and a field shared by two views is ambiguous.
- **Configuration in the class body**, as `name = ...`, `source = ...`, `ttl_days = ...`
  alongside the fields. Every one of those names becomes undeclarable as a feature.
- **Configuration as class keyword arguments**, as `class V(FeatureView, name=...)`. The body
  stays clean, but an unexpected keyword falls into `**kwargs` and is silently ignored, so a
  configuration typo is not caught. Verified against the type checker rather than reasoned
  about.
- **Integer feature ids**, as Fennel assigns, unique within a featureset. They buy rename-safety
  at the cost of a registry-shaped artifact, and this design has no registry. A rename is a
  rewrite under full refresh, which is the migration path anyway.

### One key per entity, one field per view

**Rejected:** one key per (view, entity), and one opaque blob per entity.

Colocating an entity's views under one key is what makes a request for N views a single store
read. The rejected alternatives either reintroduce per-view reads or make incremental updates and
tiling structural rewrites. The trade accepted is that the layout does not need the store to have
a hash type, which is what let the embedded LSM be the store without a different layout.

### Read-time TTL as the correctness path

**Superseded in its original form.** The rejected alternative was relying on a networked store's
native field expiration alone, which was version-gated and reclaimed on a periodic job, so it was
never a correctness guarantee. The decision that survives is the positive one: a value is served
because the read path decided to serve it, never because something removed it. Neither store
reclaims at all, which makes the point without needing the version floor that made it necessary.

### Full refresh, no watermarks

See "Materialization". The evidence is three independent Feast bugs in watermark handling and
one permanent-corruption variant.

### Arrow end to end, no protobuf

**Rejected:** protobuf as the value encoding.

It costs three full copies on the write path and 80% of read execution time in Feast, and it
requires the whole dataset in memory. Arrow is already the internal representation, so
encoding directly from Arrow batches removes a format rather than adding one.

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

### Partial aggregates in tiles

**Rejected:** storing final aggregate values per tile.

A finished value makes merging incorrect: a mean of means is not a mean, and a stddev of
stddevs is not a stddev. Tiles store the mergeable partial state instead, and aggregates that
do not merge exactly are out of scope for v1 rather than approximated. See "Tile encoding".

### Length-prefixed entity keys

**Rejected:** escaping a separator, and hashing the entity tuple.

Escaping needs a rule that must be right in two places and is easy to get subtly wrong.
Hashing the tuple makes keys unreadable, and an unreadable key cannot be diagnosed by looking
at it, which is what a length prefix is for.

A cluster hash tag was a third rejected option, on the access pattern rather than the encoding:
one key is already the unit of a read, so a tag would buy nothing and would concentrate every
entity in a project into one slot. That tier was built and then removed, and the tag question
went with it.

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

  Read latency histogram split by outcome (hit, missing); refresh duration, rows written, and
  failures; freshness lag per view as a gauge; decode errors and schema-tag mismatches.

  The last two are the failure modes that are otherwise silent, so they matter more than the
  latency numbers. Prometheus text exposition, optional OTLP traces.

  The trigger has fired: the read path and the refresh are built and measured. What is missing is
  the instrumentation, not the reason to add it.

### The serving surface: what shipped and what did not

The RPC surface is built. `feather-serve` is an Arrow Flight service, `FeatureStore.serve()`
binds it, and a client in any language reads the response with its own Arrow library because the
response is Arrow IPC inside gRPC. That was the transport the research below chose, and the
reasoning is kept because it is the reasoning the choice still rests on.

A `DoGet` ticket carries a feature service name and the entity keys to read. The field set is
not in the ticket: `ResolvedService::resolve` resolves the service against the project's
definitions once at startup, so a service naming a view or a feature that does not exist fails
at boot rather than on the first request, and a request cannot make the serving path resolve
metadata per call. That is where the `FeatureService` name stopped being decorative.

**Authentication is the gap, and it is a real one.** `feather serve` does not authenticate. The
`handshake` call answers `unimplemented` rather than accepting anything, which is the honest
answer, and `serve()` binds loopback by default, so loopback is the only thing protecting the
endpoint. The Flight specification is explicit that a token validated only at connection time is
not safe behind a layer-7 load balancer, so the answer is per-call validation or mTLS, and
neither is written. Tracked in
[issue #5](https://github.com/chandlerok/feather/issues/5).

What is also not there: `DoPut`, `DoExchange`, and `do_action` all answer `unimplemented`, and
`ListFlights` returns nothing. A write path over the transport is a different feature from
serving reads, and there is no attempt at it.

**The Python-host measurement does not reproduce from this repository.** The rows in
[`serving-transport.md`](./serving-transport.md) labelled "Python (pyo3, GIL released)" came
from a throwaway module built against this crate by path, living outside the tree. What ships is
`serve_rust`, which makes the same `feather_serve::serve` call a Python `FeatureStore.serve()`
makes, differing only in the host process. A harness built the same way belongs in the tree
rather than in `/tmp`; until one lands, treat that column as a reading from a prototype.

Arrow Flight is the transport that keeps the data path Arrow-native, being gRPC with Arrow IPC
as the payload, so it keeps a mainstream RPC transport while keeping protobuf out of the data
path. Hopsworks shipped that combination for a feature store and reported up to 45x throughput
over their REST API, and independent benchmarks put Flight up to 30x over ODBC. Feast's
community requested the switch and it was not made
([#2013](https://github.com/feast-dev/feast/issues/2013), 29 comments).

`arrow-flight` tracks the workspace's Arrow major and brings `tonic` and `prost` with it, and it
ships no auth handler. The serving surface adds an auth handler rather than inheriting one.

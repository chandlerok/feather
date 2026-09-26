# The embedded online store

**Status: decided.** The in-process store is the online store, and Valkey is the same store
shared across processes. This records why, what it removes, and the one assumption it rests on.

## The decision

`feather.toml` with no `[valkey]` table serves from an in-process store.

The relationship is substitution, not layering. This is SQLite against Postgres: an embedded
store a project runs inside itself, and a server it moves to when it needs one. The in-process
store is not a cache in front of Valkey, and Valkey is not the store of record with a local copy
in front of it. A deployment picks one, and the trait makes that a configuration change rather
than a rewrite.

The distinction matters because the two designs look alike and are not. A cache in front of a
store has to be kept correct, so something has to invalidate it. A store that _is_ the store has
nothing to invalidate, because there is no second copy. The previous design was the first kind,
and most of its machinery existed to serve that obligation.

The in-process store was already reachable, but as a fallback: `MemoryStore` documented itself as
something which "makes no attempt to be fast" and "exists to make the read path exercisable". It
is the default now, and it is written to be served from.

A deployment moves to Valkey at the scale where one process stops being enough: the dataset no
longer fits that process's memory, or several processes need to share one writable store. It is
the same trade SQLite and Postgres present.

## Why one copy deletes work

The previous design was a `moka` L1 in front of a Valkey L2, with the L1 kept correct by server
pushed invalidation. That is a correct design, and most of the machinery around it existed only
because there are two copies:

- a dedicated RESP3 redirect connection that this project implemented itself, because
  valkey-glide has no Rust client-side caching (its
  [#6918](https://github.com/valkey-io/valkey-glide/issues/6918) is open);
- reconnect handling and a full L1 flush on reconnect, since a client cannot know what changed
  while it was disconnected;
- the no-overlapping-prefixes constraint, because Valkey rejects `foo` and `foob` together;
- an always-on fallback TTL, so a bug in push handling degrades staleness rather than silently
  breaking invalidation;
- an eviction policy that can drop an entry, which in a store of record is a wrong answer rather
  than a slow one.

With one copy of the data there is nothing to invalidate. All five go away, and so does the
requirement that the store be Redis-family, which is what made the store choice hard.

That design is rejected as the default rather than deleted: [issue
#25](https://github.com/chandlerok/feather/issues/25) stays open as an unprioritized spike for
the shared tier, triggered by a measured serving latency floor the Valkey read path cannot meet,
and it is the record for those five items.

## What the store has to do

The seam already existed and it is small. From `crates/feather-core/src/online/mod.rs`:

- `read(&[ReadRequest])`, where a request is a byte key plus field names like `v:{view}` and
  `f:{view}`, returning values in request order.
- `write(&[WriteBatch])`, where fields not mentioned are left alone, so a partial refresh is
  safe.
- `delete_fields`.
- `ProjectScan::hash_fields` and `ProjectScan::scan_entity_keys`.

**The store is not responsible for expiry**, and this is what makes the rest tractable. `write`
documents that a store which cannot honour an expiry "writes the value anyway: the read-time TTL
check in `read_entities` is what decides whether a value is served, so an unexpired leftover
costs reclamation and never correctness." An in-process store therefore needs no per-field TTL,
which was the single feature that made the Redis family uniquely suitable.

**The read path is already generic.** `read_entities<S: OnlineStore>` is the only caller in the
serving path, so which store is behind it does not touch the Python surface or the offline
path.

## What was built

`MemoryStore` was promoted rather than joined by a second in-process store, because two
implementations of the same semantics drift and the drift is invisible. What it mirrors is the
server's `HSET` behaviour, which
`rewriting_a_field_without_an_expiry_clears_the_servers_expiry` pins against a real Valkey in
`tests/valkey_integration.rs`. That test never constructs `MemoryStore`: it observes the server's
own reclamation, and the mirror of the one rule the two tests share is
`a_write_without_an_expiry_clears_a_recorded_one` in
`crates/feather-core/src/online/memory.rs`. That store is the only in-process implementation
there is to keep honest.

The representation changed from `HashMap<Vec<u8>, HashMap<String, Vec<u8>>>` plus a second map
for expiries, to one sorted slice of entries per entity. Every value is still its own allocation,
so what the change buys is the index rather than the value bytes:

- **One allocation for the field index per entity** instead of a `HashMap` per entity, with the
  entries ordered by name. A request asks for one or two fields per view on one entity, and that
  index is the structure it searches.
- **Field names are stored once.** The old layout held a name twice, in the values map and in the
  expiries map, but only for fields written with an expiry, and it held the entity key twice
  wherever that happened.
- **A read is a binary search over the names** rather than a hash per name, which is why the
  entries are kept in name order. At the few fields an entity holds this is reasoning rather
  than a measurement: a couple of string comparisons against one hash is a trade nobody here has
  timed, so the win is asserted rather than shown.

## The next step, gated on a measurement

When the dataset outgrows one process, the generation becomes an mmap'd, sorted Arrow artifact
with columns `(entity_key, field_name, value)`. A point read is one binary search plus a scan of
a short contiguous run.

- **Bounded memory falls out.** The file is mmap'd, so resident memory is what the operating
  system keeps hot, and a cold read costs a page fault rather than a heap cap.
- **Durability falls out.** A restart is a remap, not a refresh.
- **`arrow-ipc` 58 already supports mmap**, and `memmap2` would be the only new dependency.

The trigger is not a date. It is resident memory per process against the dataset size, or a p99
read latency that page faults rather than lookups explain.

## What the generation model would remove

- **Eviction.** A generation is a complete snapshot, so there is no capacity policy and no way to
  serve a missing feature because a cache decided to evict.
- **`delete_fields`.** A retired view is absent from the next generation, so delete is implied.
- **`scan_entity_keys`.** Its documentation explains that it exists because "the keyspace itself
  is the only complete list" of entity hashes a retired view's fields may still sit in. A
  generation built from current declarations never contains those fields. `hash_fields` becomes a
  manifest read, since the project registry is metadata rather than a hash.

## The upgrade to Valkey

A configuration change, because `read_entities` is generic over the trait. Two things differ, and
both are documented in the architecture document:

- **Where writes land.** In-process writes into the one map this process holds, under the
  store's own `&mut self`; Valkey writes into shared state. Both satisfy `write` and
  `delete_fields`: fields a batch does not mention are left alone, and a delete removes.
- **How stale a read can be.** In-process serves what the last refresh in this process wrote, so
  staleness is bounded by the refresh interval. Writes take `&mut self` and reads take `&self`,
  and the Python binding holds its store behind one lock for a whole refresh, so a read sees a
  completed refresh rather than a partial one. That is exclusion rather than an atomic swap, so
  the bound is the interval and not zero.

## The assumption it rests on

**Materialization stays a full refresh.** Immutable generations and watermark-driven incremental
updates pull against each other. They can be reconciled LSM-style, with delta generations that
readers overlay on a base, but that is real complexity and it is not proposed here.

This is the assumption most likely to be overturned, and the thing that would overturn it is
issue [#9](https://github.com/chandlerok/feather/issues/9). If incremental materialization
becomes a near-term goal, the mutable in-process store becomes the long-term shape and the
mmap'd artifact is demoted to a cold tier.

## Costs

- **Memory is the limit, and it is per process.** State it as a number rather than discovering
  it: entities times bytes per entity, and the point at which that stops fitting. The mmap'd
  artifact moves the ceiling from RAM to local disk, not to infinity.
- **Staleness is bounded rather than zero**, as above.
- **The async trait drags a runtime into the default build.** `read` is `async fn`, and for an
  in-memory map that is a poll and a future for work that never yields. `valkey` is currently a
  default feature in `feather-py`, so `tokio` arrives with it. If the in-process store is the
  default, either the default build carries a runtime for nothing, or the trait becomes
  synchronous. A third option is cheap and already available: `futures` is a dependency, and
  `futures::executor::block_on` can drive the async trait without `tokio`. Not decided here.

## What this does not change

The offline path, the point-in-time join, the materialization semantics, the Arrow
representation, and the Python surface. It changes where a served value is read from, and it
removes a service from the default deployment.

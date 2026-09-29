//! Online storage and the read path.
//!
//! The store trait is deliberately narrow: it moves opaque field values in and
//! out of hashes. Encoding, TTL, and missingness live above it, so the same
//! logic runs against the in-memory store and against Valkey.
//!
//! There is no cache in the store.

pub mod memory;

#[cfg(feature = "valkey")]
pub mod valkey;

use std::collections::BTreeMap;
use std::time::Duration;

use arrow::array::ArrayRef;

use crate::definitions::FeatureView;
use crate::error::{Error, Result};
use crate::key::{encode_entity_key, entity_hash_key, freshness_field, value_field};
use crate::value::{SchemaTag, decode_batch, encode_batch};

/// One field of a write: its value, and when it stops being readable.
///
/// The expiry rides with the value rather than in a call of its own because the two belong
/// together: a field written without the expiry that is meant to reclaim it is a field the
/// server keeps until its hash is rewritten or deleted. The unit is named in the field rather
/// than left to the reader, because a bare `i64` expiry inside a codec that otherwise counts
/// microseconds is a bug waiting to happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenField {
    pub name: String,
    pub value: Vec<u8>,
    /// Absolute expiry as seconds since the Unix epoch, or `None` for a field that never
    /// expires.
    ///
    /// Absolute rather than a TTL: a value that is already older than its TTL has to expire at
    /// once, and a relative TTL measured from the write would give it a second lease on life.
    pub expires_at_unix_secs: Option<i64>,
}

impl WrittenField {
    /// A field, with an absolute expiry when it has one.
    ///
    /// Args:
    ///     name: The field name.
    ///     value: The encoded value.
    ///     expires_at_unix_secs: When the field stops being readable, or `None` to leave it to
    ///         the read-time check alone.
    ///
    /// Returns:
    ///     The field.
    pub fn new(name: impl Into<String>, value: Vec<u8>, expires_at_unix_secs: Option<i64>) -> Self {
        Self {
            name: name.into(),
            value,
            expires_at_unix_secs,
        }
    }
}

/// The precondition a guarded write carries: land only if nothing fresher is already there.
///
/// A view's freshness field, `f:{view}`, is the newest event timestamp any write has recorded
/// for that view and entity. Comparing the timestamp a write is claiming against it is what
/// makes a refresh and a push composable: a refresh recomputes from a source that may lag, and
/// without the comparison it would overwrite a value that is newer than anything the source
/// holds. The same comparison is what makes a push safe to retry, since a repeat of the same
/// push is refused rather than rewriting a byte-identical value.
///
/// The guard is on the whole batch rather than on each field because the thing being compared
/// is per view and entity, which is the unit a batch is written at. A batch without one always
/// lands, which is what the project registry and a first write need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshnessGuard {
    /// The freshness field to compare, which is `f:{view}`.
    pub field: String,
    /// The event timestamp this write is claiming, in microseconds since the epoch.
    pub event_micros: i64,
}

impl FreshnessGuard {
    /// A guard on one view's freshness, for a write claiming `event_micros`.
    ///
    /// Args:
    ///     view: The view whose freshness is being guarded.
    ///     event_micros: The event timestamp the write is claiming.
    ///
    /// Returns:
    ///     The guard.
    pub fn for_view(view: &str, event_micros: i64) -> Self {
        Self {
            field: freshness_field(view),
            event_micros,
        }
    }
}

/// One write: a hash key and the fields to set in it.
///
/// "Set" rather than "insert", because a field already in the hash is overwritten. That is
/// what makes a full refresh safe to re-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteBatch {
    pub key: Vec<u8>,
    pub fields: Vec<WrittenField>,
    /// The condition this write has to beat, or `None` to write unconditionally.
    pub guard: Option<FreshnessGuard>,
}

impl WriteBatch {
    /// A write that lands whenever it is issued.
    ///
    /// Args:
    ///     key: The hash key.
    ///     fields: The fields to set in it. Fields not named are left alone.
    ///
    /// Returns:
    ///     The batch.
    pub fn new(key: Vec<u8>, fields: Vec<WrittenField>) -> Self {
        Self {
            key,
            fields,
            guard: None,
        }
    }

    /// A write that lands only if nothing fresher is already recorded.
    ///
    /// Args:
    ///     key: The hash key.
    ///     fields: The fields to set in it. Fields not named are left alone.
    ///     guard: The condition to beat, from [`FreshnessGuard::for_view`].
    ///
    /// Returns:
    ///     The batch.
    pub fn guarded(key: Vec<u8>, fields: Vec<WrittenField>, guard: FreshnessGuard) -> Self {
        Self {
            key,
            fields,
            guard: Some(guard),
        }
    }

    /// Whether this write has to beat a recorded freshness.
    pub const fn is_guarded(&self) -> bool {
        self.guard.is_some()
    }
}

/// One `HMGET`: a hash key and the fields to read from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadRequest {
    pub key: Vec<u8>,
    pub fields: Vec<String>,
}

/// A minimal hash store. Async because the real implementation is, and because
/// pretending otherwise would make the serving path look cheaper than it is.
#[allow(async_fn_in_trait)]
pub trait OnlineStore {
    /// Write fields into hashes. Fields not mentioned are left alone, which is
    /// what makes a full refresh safe to re-run.
    ///
    /// A field may carry an absolute expiry. A store that cannot honour one writes the value
    /// anyway: the read-time TTL check in [`read_entities`] is what decides whether a value is
    /// served, so an unexpired leftover costs reclamation and never correctness.
    ///
    /// A batch carrying a [`FreshnessGuard`] lands only if the guard's field is absent or holds
    /// an older timestamp, and the store has to decide that atomically with the write: a guard
    /// read and then written is a lost update, which is the one failure the guard exists to
    /// prevent. A store that cannot do this cannot honour the guard, and says so rather than
    /// writing anyway.
    ///
    /// Returns:
    ///     One entry per batch, in the order given, saying whether it landed. A refused batch
    ///     wrote nothing, and a batch without a guard always lands.
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<Vec<bool>>;

    /// Read fields from hashes, one `HMGET` per request, in request order.
    ///
    /// The outer `Vec` is per request and the inner is per requested field, in
    /// the order requested. A field that is absent yields `None`, which is how a
    /// TTL-expired or never-written value arrives here.
    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>>;

    /// Remove fields, used when a view is retired.
    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()>;
}

/// The reads that look at a project's whole keyspace rather than at one hash in it.
///
/// Deliberately separate from [`OnlineStore`] rather than a method on it. `read_entities` bounds
/// its store by that trait, and the serving path has no business reaching for a keyspace walk:
/// the narrowness of that trait is what keeps the cost of a read explicable. A refresh is the
/// only caller.
#[allow(async_fn_in_trait)]
pub trait ProjectScan {
    /// Every field of one hash, as `(name, value)` pairs, and empty when there is no such key.
    ///
    /// A whole hash rather than named fields, which is what [`OnlineStore::read`] takes, because
    /// the project registry is read to learn which fields it holds. Asking for them one at a
    /// time would mean knowing the answer in advance.
    ///
    /// Args:
    ///     key: The hash key.
    ///
    /// Returns:
    ///     Each field with its value. An absent key yields an empty list rather than an error,
    ///     which is the first refresh of a project.
    async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>>;

    /// Every key in a project's namespace except `exclude`, which is compared exactly.
    ///
    /// Used to find the entity hashes a retired view's fields may still sit in. An entity that
    /// the retired view used to cover does not have to appear in the source any more, so there
    /// is no way to enumerate them from the declarations: the keyspace itself is the only
    /// complete list.
    ///
    /// Args:
    ///     project: The project whose keys are wanted.
    ///     exclude: A key to leave out. The caller names the project registry, which lives
    ///         under the same prefix and is not an entity hash.
    ///
    /// Returns:
    ///     The keys, in no particular order. The caller filters them further; this is a key
    ///     listing, not a promise that every key in it is an entity hash.
    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>>;
}

/// The key prefix every key of a project shares.
///
/// One spelling of the prefix, so a walk and a filter cannot disagree about where a project's
/// keys start.
pub fn project_key_prefix(project: &str) -> Vec<u8> {
    let mut prefix = Vec::with_capacity(project.len() + 1);
    prefix.extend_from_slice(project.as_bytes());
    prefix.push(b':');
    prefix
}

/// A project name escaped for a `SCAN` pattern.
///
/// `MATCH` reads `*`, `?`, `[` and `\` as pattern syntax, so a project named `ads*` would
/// otherwise walk another project's keys. Backslash is the escape for all four.
pub fn glob_escape(project: &str) -> String {
    let mut escaped = String::with_capacity(project.len());
    for character in project.chars() {
        if matches!(character, '*' | '?' | '[' | ']' | '\\') {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

/// Why a view's values are not usable for an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Missing {
    /// The freshness field is absent, so the view was never written for this
    /// entity.
    NeverWritten,
    /// Freshness is present but older than the view's TTL.
    Expired,
    /// A value carried a schema tag that does not match the current definition.
    SchemaMismatch,
}

impl Missing {
    /// The reason, as it appears in the response state mask.
    pub const fn as_str(self) -> &'static str {
        match self {
            Missing::NeverWritten => "missing",
            Missing::Expired => "expired",
            Missing::SchemaMismatch => "schema_mismatch",
        }
    }
}

/// What a read produced for one view of one entity.
#[derive(Debug)]
pub enum ViewValues {
    Present {
        columns: Vec<ArrayRef>,
        freshness_micros: i64,
    },
    Missing(Missing),
}

/// One view's worth of a read request.
#[derive(Debug, Clone)]
pub struct ViewRequest {
    pub view: String,
    /// Feature names, in the order the caller wants them back.
    pub fields: Vec<String>,
}

/// One entity's worth of a read request.
#[derive(Debug, Clone)]
pub struct EntityRequest {
    /// The encoded entity key, as produced by [`crate::key::encode_entity_key`].
    pub encoded_key: Vec<u8>,
    pub views: Vec<ViewRequest>,
}

/// Read a batch of entities.
///
/// Issues exactly one store read per entity regardless of how many views are
/// requested. That is the rule the whole layout exists to make possible: Feast
/// issues one call per feature view, and a user with 11 views reported "abysmal"
/// retrieval times (feast-dev/feast#3596).
///
/// One field per view is requested, holding the view's whole encoded vector, and
/// the caller's requested features are projected out after decoding. Reading a
/// subset therefore costs decode time for the columns it did not ask for, which
/// is the trade recorded on [`crate::key::value_field`].
///
/// `now_micros` is a parameter rather than a clock read so TTL behaviour is
/// testable without sleeping.
pub async fn read_entities<S: OnlineStore>(
    store: &S,
    project: &str,
    views: &BTreeMap<String, FeatureView>,
    entities: &[EntityRequest],
    now_micros: i64,
) -> Result<Vec<Vec<ViewValues>>> {
    /// Where one view's value and freshness sit in the flat field list, and which
    /// of the view's columns the caller actually asked for.
    struct Plan<'a> {
        view: &'a FeatureView,
        /// Output column order: index into the view's `features`.
        requested: Vec<usize>,
        value_at: usize,
        freshness_at: usize,
    }

    let mut requests = Vec::with_capacity(entities.len());
    let mut plans: Vec<Vec<Plan<'_>>> = Vec::with_capacity(entities.len());

    for entity in entities {
        let mut fields = Vec::new();
        let mut per_view = Vec::with_capacity(entity.views.len());

        for request in &entity.views {
            let view = views
                .get(&request.view)
                .ok_or_else(|| Error::UnknownView(request.view.clone()))?;

            let mut requested = Vec::with_capacity(request.fields.len());
            for name in &request.fields {
                let index = view
                    .features
                    .iter()
                    .position(|f| &f.name == name)
                    .ok_or_else(|| Error::UnknownFeature {
                        view: view.name.clone(),
                        name: name.clone(),
                    })?;
                requested.push(index);
            }

            let value_at = fields.len();
            fields.push(value_field(&view.name));
            let freshness_at = fields.len();
            fields.push(freshness_field(&view.name));

            per_view.push(Plan {
                view,
                requested,
                value_at,
                freshness_at,
            });
        }

        requests.push(ReadRequest {
            key: crate::key::entity_hash_key(
                project,
                shared_entity_name(&entity.views, views)?,
                &entity.encoded_key,
            ),
            fields,
        });
        plans.push(per_view);
    }

    let raw = store.read(&requests).await?;

    let mut out = Vec::with_capacity(entities.len());
    for (entity_index, per_view) in plans.iter().enumerate() {
        let values = &raw[entity_index];
        let mut results = Vec::with_capacity(per_view.len());

        for plan in per_view {
            let view = plan.view;
            let tag = SchemaTag::of(&view.features);

            let Some(freshness_bytes) = values.get(plan.freshness_at).and_then(|v| v.as_deref())
            else {
                results.push(ViewValues::Missing(Missing::NeverWritten));
                continue;
            };
            let freshness_micros = read_micros(freshness_bytes)?;

            if let Some(ttl) = view.ttl_days {
                let ttl_micros = i64::from(ttl) * 86_400 * 1_000_000;
                if now_micros.saturating_sub(freshness_micros) > ttl_micros {
                    results.push(ViewValues::Missing(Missing::Expired));
                    continue;
                }
            }

            let Some(blob) = values.get(plan.value_at).and_then(|v| v.as_deref()) else {
                results.push(ViewValues::Missing(Missing::NeverWritten));
                continue;
            };

            match decode_batch(&view.features, tag, &[blob]) {
                Ok(all_columns) => {
                    let columns = plan
                        .requested
                        .iter()
                        .map(|i| all_columns[*i].clone())
                        .collect();
                    results.push(ViewValues::Present {
                        columns,
                        freshness_micros,
                    });
                }
                // A tag mismatch is the documented null window during a dtype
                // change, not an error. Anything else is real.
                Err(Error::SchemaTagMismatch { .. }) => {
                    results.push(ViewValues::Missing(Missing::SchemaMismatch));
                }
                Err(other) => return Err(other),
            }
        }

        out.push(results);
    }

    Ok(out)
}

/// Write one entity's whole vector for one view into the online store.
///
/// A push is the write a refresh would have made later, for one entity, by a caller that
/// already holds the values. It covers a view in full rather than a subset of it, and that is
/// what keeps it one atomic write: a view's features share a single field, so a partial write
/// would be a read-modify-write of that field, and a read-modify-write races the refresh
/// writing the same field. A caller holding a subset reads the stored vector, patches it, and
/// pushes the whole result, which is the same two round trips it would have cost anyway.
///
/// The write is guarded on [`FreshnessGuard`], so a push is refused rather than overwriting
/// something newer, and a refresh reaching the same entity with an older event timestamp is
/// refused in turn. The source stays the authority over everything it has actually seen, and a
/// push is the authority for the window before the source catches up.
///
/// Only the online store is written. A pushed value is therefore not in the training data until
/// a refresh of the same source reaches it, which is a training-serving skew for a feature that
/// is only ever pushed. Closing that would mean appending to the offline side as well, and the
/// appendable source and its point-in-time correctness case are not built.
///
/// Args:
///     store: The store to write to.
///     project: The project name, which namespaces the key.
///     view: The view being pushed. Its features in declaration order are `columns`.
///     entity_component: The entity's join-key value, as
///         [`crate::key::entity_key_component`] produces it.
///     columns: One single-value column per declared feature, in declaration order.
///     event_micros: The event timestamp these values describe, in microseconds since the
///         epoch.
///
/// Returns:
///     `true` if the values were written, and `false` if something newer was already recorded.
///
/// Raises:
///     Error: If the view did not declare itself [`FeatureView::pushable`], which is the check
///         that makes the guard a property of the definition rather than of the caller. If the
///         view declares no single entity, or `columns` is not one column per declared feature,
///         or a column's type is not the one its feature declares.
pub async fn push_record<S: OnlineStore>(
    store: &mut S,
    project: &str,
    view: &FeatureView,
    entity_component: &[u8],
    columns: &[ArrayRef],
    event_micros: i64,
) -> Result<bool> {
    // Refused rather than written unguarded. A push to a view whose refresh is not guarded
    // would be reverted by the next refresh, so accepting it would hand back a `true` for a
    // write that does not survive; the capability is the view's to declare, not the caller's to
    // assume.
    if !view.pushable {
        return Err(Error::ViewNotPushable {
            view: view.name.clone(),
        });
    }
    let entity = view.entity()?;
    // One row is asked for, so this is a single whole-vector encode: the same blob, in the same
    // layout, with the same schema tag, that a refresh would have written for this entity.
    let encoded = encode_batch(&view.features, columns, 0..1)?;
    let blob = encoded
        .row(0)
        .map(<[u8]>::to_vec)
        .ok_or(Error::RowOutOfRange {
            row: 0,
            len: encoded.len(),
        })?;

    let key = entity_hash_key(
        project,
        &entity.name,
        &encode_entity_key(&[entity_component])?,
    );
    let fields = vec![
        WrittenField::new(
            value_field(&view.name),
            blob,
            value_expiry_unix_secs(view, event_micros),
        ),
        // The freshness field is deliberately left unexpired, for the same reason a refresh
        // leaves it unexpired: reclaiming it would collapse `expired` and `never written`,
        // which the read path keeps apart.
        WrittenField::new(
            freshness_field(&view.name),
            encode_freshness(event_micros),
            None,
        ),
    ];

    let mut applied = store
        .write(&[WriteBatch::guarded(
            key,
            fields,
            FreshnessGuard::for_view(&view.name, event_micros),
        )])
        .await?;
    Ok(applied.pop().unwrap_or(false))
}

/// v1 supports exactly one entity per view, and a single request must resolve to a
/// single entity name because that name is part of the hash key.
fn shared_entity_name<'a>(
    requests: &[ViewRequest],
    views: &'a BTreeMap<String, FeatureView>,
) -> Result<&'a str> {
    let mut resolved: Option<&'a str> = None;
    for request in requests {
        let view = views
            .get(&request.view)
            .ok_or_else(|| Error::UnknownView(request.view.clone()))?;
        let name = view.entity()?.name.as_str();
        match resolved {
            None => resolved = Some(name),
            Some(existing) if existing == name => {}
            Some(existing) => {
                return Err(Error::MixedEntities {
                    first: existing.to_owned(),
                    second: name.to_owned(),
                });
            }
        }
    }
    resolved.ok_or_else(|| Error::MalformedFeatureReference {
        reference: "(no views requested)".to_owned(),
    })
}

/// Read back a freshness timestamp the way [`encode_freshness`] writes one.
///
/// The inverse of that function, and the reason a store can compare a guarded write's claim
/// against what it already holds. It is crate-visible rather than public because nothing
/// outside the crate decodes a freshness field, and a store that does is a server-side script
/// rather than Rust.
pub(crate) fn read_micros(bytes: &[u8]) -> Result<i64> {
    let array: [u8; 8] = bytes.try_into().map_err(|_| Error::MalformedValue {
        field: "freshness".to_owned(),
        reason: format!("expected 8 bytes, got {}", bytes.len()),
    })?;
    Ok(i64::from_le_bytes(array))
}

/// The instant a value stops being reclaimable by the server, in Unix seconds.
///
/// Measured from the winning row's event timestamp rather than from the time of the write. A
/// 40-day-old value under a 30-day TTL has already expired, and an expiry measured from the
/// write would give it another 30 days of life; the read-time check would then disagree with
/// what the server holds. `None` for a view that declares no TTL.
///
/// Rounded up, so rounding is not what makes the server reclaim a field while the read-time
/// check would still call it fresh. Rounding down opens a window just under a second wide where
/// a value the read path would have served is already gone, which reports it as never written
/// instead. The two sides read different clocks — the server's, against this absolute instant,
/// and the caller's `now`, against the recorded freshness — so a skew between them is the one
/// way left for the server to reclaim early, and no rounding direction can cover it.
pub(crate) fn value_expiry_unix_secs(view: &FeatureView, event_micros: i64) -> Option<i64> {
    let ttl_days = view.ttl_days?;
    // Rounded up rather than down. `i64::div_ceil` is still unstable in this toolchain, so the
    // half second of arithmetic is spelled out rather than imported behind a feature gate.
    let remainder = event_micros.rem_euclid(1_000_000) != 0;
    let seconds = event_micros.div_euclid(1_000_000) + i64::from(remainder);
    Some(seconds + i64::from(ttl_days) * 86_400)
}

/// Encode a freshness timestamp the way the read path expects it.
pub fn encode_freshness(micros: i64) -> Vec<u8> {
    micros.to_le_bytes().to_vec()
}

/// Convert a TTL in days to a duration.
pub fn ttl_duration(ttl_days: u32) -> Duration {
    Duration::from_secs(u64::from(ttl_days) * 86_400)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definitions::{DType, Entity, Field, Source};
    use crate::online::memory::MemoryStore;
    use crate::value::encode_batch;
    use arrow::array::{ArrayRef, Int64Array};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const NOW: i64 = 1_700_000_000_000_000;

    fn view(name: &str, ttl_days: Option<u32>) -> FeatureView {
        FeatureView {
            name: name.to_owned(),
            entities: vec![Entity::new("user_id", "user_id")],
            source: Source::file("data/x.parquet"),
            features: vec![Field::new("count", DType::Int64)],
            ttl_days,
            timestamp_field: None,
            created_timestamp_field: None,
            // Pushable, because this module's tests are the push tests and `push_record`
            // refuses a view that did not declare it. The refusal itself is pinned below.
            pushable: true,
        }
    }

    fn views(names: &[(&str, Option<u32>)]) -> BTreeMap<String, FeatureView> {
        names
            .iter()
            .map(|(n, ttl)| ((*n).to_owned(), view(n, *ttl)))
            .collect()
    }

    fn encoded_vector(tag_source: &FeatureView, value: i64) -> Vec<u8> {
        let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![value]))];
        encode_batch(&tag_source.features, &columns, 0..1)
            .unwrap()
            .row(0)
            .unwrap()
            .to_vec()
    }

    /// An entity key as the read path is given it: the encoded form the binding builds, not the
    /// raw component. A raw component is a key the encoder would never produce, so it is also a
    /// key garbage collection skips, and a fixture built from one cannot show what a real key
    /// does.
    fn encoded_key(value: &str) -> Vec<u8> {
        crate::key::encode_entity_key(&[value.as_bytes()]).expect("encode the key")
    }

    /// `seed` and `request` take the join key value and encode it, so the hash a fixture writes
    /// and the hash a read looks up cannot be spelled differently.
    ///
    /// Guarded, like every other writer: a fixture that wrote unconditionally would let a later
    /// stale write land, and the guard is exactly what the push tests are here to pin.
    async fn seed(
        store: &mut MemoryStore,
        project: &str,
        view: &FeatureView,
        entity: &str,
        value: i64,
        freshness: i64,
    ) {
        let key = crate::key::entity_hash_key(project, "user_id", &encoded_key(entity));
        store
            .write(&[WriteBatch::guarded(
                key,
                vec![
                    WrittenField::new(value_field(&view.name), encoded_vector(view, value), None),
                    WrittenField::new(
                        freshness_field(&view.name),
                        encode_freshness(freshness),
                        None,
                    ),
                ],
                FreshnessGuard::for_view(&view.name, freshness),
            )])
            .await
            .unwrap();
    }

    /// One single-value column per declared feature, which is the shape a push is given.
    fn columns(values: &[i64]) -> Vec<ArrayRef> {
        values
            .iter()
            .map(|value| Arc::new(Int64Array::from(vec![*value])) as ArrayRef)
            .collect()
    }

    /// Push one entity's vector, and report what the guard decided.
    async fn push(
        store: &mut MemoryStore,
        project: &str,
        view: &FeatureView,
        entity: &str,
        value: i64,
        event_micros: i64,
    ) -> bool {
        push_record(
            store,
            project,
            view,
            entity.as_bytes(),
            &columns(&[value]),
            event_micros,
        )
        .await
        .unwrap()
    }
    #[tokio::test]
    async fn a_view_that_is_not_pushable_is_refused() {
        // The rule that makes opting in safe. An unguarded push would be reverted by the next
        // refresh, so accepting one would report a success for a write that does not survive.
        let mut view = view("clicks", Some(30));
        view.pushable = false;
        let mut store = MemoryStore::new();

        let refused = push_record(&mut store, "ads", &view, b"u1", &columns(&[1]), NOW).await;

        assert!(
            matches!(refused, Err(Error::ViewNotPushable { .. })),
            "expected ViewNotPushable, got {refused:?}"
        );
        assert!(
            store
                .fields(&crate::key::entity_hash_key(
                    "ads",
                    "user_id",
                    &encoded_key("u1")
                ))
                .is_none(),
            "a refused push writes nothing at all"
        );
    }

    #[tokio::test]
    async fn a_push_is_read_back_by_the_serving_read() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();

        assert!(push(&mut store, "ads", view, "u1", 99, NOW).await);

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        match &out[0][0] {
            ViewValues::Present {
                columns,
                freshness_micros,
            } => {
                assert_eq!(*freshness_micros, NOW);
                assert_eq!(
                    columns[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0),
                    99
                );
            }
            other => panic!("expected Present, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_push_writes_the_expiry_the_ttl_asks_for() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        push(&mut store, "ads", view, "u1", 1, NOW).await;

        let key = crate::key::entity_hash_key("ads", "user_id", &encoded_key("u1"));
        // Thirty days after the event, not thirty days after the write, so an old push does not
        // buy itself another lease.
        let expected = value_expiry_unix_secs(view, NOW);
        assert_eq!(store.field_expiry(&key, &value_field("clicks")), expected);
    }

    #[tokio::test]
    async fn a_push_at_the_stored_timestamp_is_refused() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 1, NOW).await;

        // The same claim a second time, so a retry of a push that already landed writes nothing.
        assert!(!push(&mut store, "ads", view, "u1", 2, NOW).await);

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        match &out[0][0] {
            ViewValues::Present { columns, .. } => assert_eq!(
                columns[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0),
                1
            ),
            other => panic!("expected Present, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_push_older_than_what_is_stored_is_refused() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 1, NOW).await;

        // A producer replaying an old event, which is what a lagging source looks like from here.
        assert!(!push(&mut store, "ads", view, "u1", 2, NOW - 1_000).await);
    }

    #[tokio::test]
    async fn a_push_newer_than_what_is_stored_lands() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 1, NOW).await;

        assert!(push(&mut store, "ads", view, "u1", 2, NOW + 1_000).await);

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        match &out[0][0] {
            ViewValues::Present { columns, .. } => assert_eq!(
                columns[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(0),
                2
            ),
            other => panic!("expected Present, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_push_only_touches_its_own_views_fields() {
        // One hash holds every view of an entity, so a push writing one view must leave the
        // others alone. That is what `OnlineStore::write` promising to leave unmentioned fields
        // alone is for, and a guard that failed to check before writing would break it.
        let views = views(&[("a", Some(30)), ("b", Some(30))]);
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", views.get("a").unwrap(), "u1", 1, NOW).await;
        seed(&mut store, "ads", views.get("b").unwrap(), "u1", 2, NOW).await;

        assert!(
            push(
                &mut store,
                "ads",
                views.get("a").unwrap(),
                "u1",
                3,
                NOW + 1_000
            )
            .await
        );

        for (name, expected) in [("a", 3), ("b", 2)] {
            let out = read_entities(
                &store,
                "ads",
                &views,
                &[EntityRequest {
                    encoded_key: encoded_key("u1"),
                    views: vec![ViewRequest {
                        view: name.to_owned(),
                        fields: vec!["count".to_owned()],
                    }],
                }],
                NOW,
            )
            .await
            .unwrap();
            match &out[0][0] {
                ViewValues::Present { columns, .. } => assert_eq!(
                    columns[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(0),
                    expected,
                    "view {name}"
                ),
                other => panic!("expected Present for {name}, got {other:?}"),
            }
        }
    }

    fn request(view_name: &str, entity: &str) -> EntityRequest {
        EntityRequest {
            encoded_key: encoded_key(entity),
            views: vec![ViewRequest {
                view: view_name.to_owned(),
                fields: vec!["count".to_owned()],
            }],
        }
    }

    #[tokio::test]
    async fn reads_back_a_written_vector() {
        let views = views(&[("clicks", Some(30))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 42, NOW).await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        match &out[0][0] {
            ViewValues::Present {
                columns,
                freshness_micros,
            } => {
                assert_eq!(*freshness_micros, NOW);
                let counts = columns[0].as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(counts.value(0), 42);
            }
            other => panic!("expected Present, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn an_expired_value_is_missing_not_stale() {
        let views = views(&[("clicks", Some(1))]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        // Written two days ago against a one-day TTL.
        seed(
            &mut store,
            "ads",
            view,
            "u1",
            1,
            NOW - 2 * 86_400 * 1_000_000,
        )
        .await;

        let out = read_entities(&store, "ads", &views, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        assert!(matches!(out[0][0], ViewValues::Missing(Missing::Expired)));
    }

    #[tokio::test]
    async fn a_never_written_entity_is_missing() {
        let views = views(&[("clicks", Some(30))]);
        let store = MemoryStore::new();
        let out = read_entities(&store, "ads", &views, &[request("clicks", "nobody")], NOW)
            .await
            .unwrap();
        assert!(matches!(
            out[0][0],
            ViewValues::Missing(Missing::NeverWritten)
        ));
    }

    #[tokio::test]
    async fn a_schema_change_reads_as_missing_rather_than_garbage() {
        let views = views(&[("clicks", Some(30))]);
        let old = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", old, "u1", 7, NOW).await;

        // The definition changes dtype; the stored tag no longer matches.
        let mut changed = views.clone();
        changed.get_mut("clicks").unwrap().features = vec![Field::new("count", DType::Float64)];

        let out = read_entities(&store, "ads", &changed, &[request("clicks", "u1")], NOW)
            .await
            .unwrap();
        assert!(matches!(
            out[0][0],
            ViewValues::Missing(Missing::SchemaMismatch)
        ));
    }

    /// Counts store reads, to prove the one-read-per-entity rule holds.
    struct CountingStore {
        inner: MemoryStore,
        reads: Arc<AtomicUsize>,
    }

    impl OnlineStore for CountingStore {
        async fn write(&mut self, batches: &[WriteBatch]) -> Result<Vec<bool>> {
            self.inner.write(batches).await
        }

        async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
            self.reads.fetch_add(requests.len(), Ordering::SeqCst);
            self.inner.read(requests).await
        }

        async fn delete_fields(&mut self, keys: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
            self.inner.delete_fields(keys).await
        }
    }

    #[tokio::test]
    async fn reads_once_per_entity_however_many_views() {
        let many = views(&[
            ("a", Some(30)),
            ("b", Some(30)),
            ("c", Some(30)),
            ("d", Some(30)),
        ]);
        let mut inner = MemoryStore::new();
        for name in ["a", "b", "c", "d"] {
            seed(&mut inner, "ads", many.get(name).unwrap(), "u1", 1, NOW).await;
        }

        let reads = Arc::new(AtomicUsize::new(0));
        let store = CountingStore {
            inner,
            reads: Arc::clone(&reads),
        };

        let mut entity = request("a", "u1");
        for name in ["b", "c", "d"] {
            entity.views.push(ViewRequest {
                view: name.to_owned(),
                fields: vec!["count".to_owned()],
            });
        }

        let out = read_entities(&store, "ads", &many, &[entity], NOW)
            .await
            .unwrap();
        assert_eq!(out[0].len(), 4);
        assert_eq!(
            reads.load(Ordering::SeqCst),
            1,
            "four views on one entity must still be one store read"
        );
    }

    #[tokio::test]
    async fn views_disagreeing_on_their_entity_are_rejected() {
        let mut many = views(&[("a", None), ("b", None)]);
        many.get_mut("b").unwrap().entities = vec![Entity::new("item_id", "item_id")];
        let store = MemoryStore::new();

        let entity = EntityRequest {
            encoded_key: encoded_key("u1"),
            views: vec![
                ViewRequest {
                    view: "a".to_owned(),
                    fields: vec!["count".to_owned()],
                },
                ViewRequest {
                    view: "b".to_owned(),
                    fields: vec!["count".to_owned()],
                },
            ],
        };

        assert!(matches!(
            read_entities(&store, "ads", &many, &[entity], NOW).await,
            Err(Error::MixedEntities { .. })
        ));
    }

    #[tokio::test]
    async fn a_keyspace_scan_returns_the_entity_hashes_and_not_the_registry() {
        use crate::key::views_registry_key;
        let mut store = MemoryStore::new();
        let registry = views_registry_key("ads");
        let mut keys = vec![registry.clone()];
        for entity in ["u1", "u2"] {
            let key = crate::key::entity_hash_key("ads", "user_id", &encoded_key(entity));
            store
                .write(&[WriteBatch::new(
                    key.clone(),
                    vec![WrittenField::new(
                        crate::key::value_field("clicks"),
                        vec![1],
                        None,
                    )],
                )])
                .await
                .unwrap();
            keys.push(key);
        }
        store
            .write(&[WriteBatch::new(
                registry.clone(),
                vec![WrittenField::new("clicks", vec![2], None)],
            )])
            .await
            .unwrap();

        let mut scanned = store.scan_entity_keys("ads", &registry).await.unwrap();
        scanned.sort();
        keys.remove(0);
        keys.sort();
        assert_eq!(scanned, keys);

        // Another project shares no prefix with this one.
        assert!(
            store
                .scan_entity_keys("other", b"other:views")
                .await
                .unwrap()
                .is_empty()
        );

        let fields = store.hash_fields(&registry).await.unwrap();
        assert_eq!(fields, vec![("clicks".to_owned(), vec![2])]);
    }

    #[tokio::test]
    async fn a_retired_view_can_be_deleted() {
        let views = views(&[("clicks", None)]);
        let view = views.get("clicks").unwrap();
        let mut store = MemoryStore::new();
        seed(&mut store, "ads", view, "u1", 1, NOW).await;
        assert_eq!(store.hash_count(), 1);

        let key = crate::key::entity_hash_key("ads", "user_id", &encoded_key("u1"));
        store
            .delete_fields(&[(key, vec![value_field("clicks")])])
            .await
            .unwrap();
        assert!(
            !store
                .fields(&crate::key::entity_hash_key(
                    "ads",
                    "user_id",
                    &encoded_key("u1"),
                ))
                .unwrap()
                .contains_key(&value_field("clicks"))
        );
    }
}

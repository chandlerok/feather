//! In-process [`OnlineStore`], and the store a project serves from until it needs a shared one.
//!
//! Local mode runs on this: `feather.toml` with no `[valkey]` table. It is the default rather
//! than a fallback, so it is written to be served from rather than merely to be correct.
//!
//! Shape: one allocation for the entity's field index, its entries in name order; each field's
//! name and value are still separate allocations, as they were. A request asks for one or two
//! fields per view on one entity, so searching that entity's index is the access pattern that
//! matters, and a binary search over the names is expected to beat a hash per name, a win
//! asserted rather than shown. A field name appears once,
//! which the previous two-map layout did not manage for a field written with an expiry: the
//! expiries map held that name a second time, and the entity key with it.
//!
//! A field's expiry is remembered and never acted on: this store has no clock and reclaims
//! nothing, and the read-time TTL check is the authoritative path in any case. Keeping the
//! expiry visible is what lets a test assert what the write path asked for without a server.
//!
//! A write that carries no expiry drops any expiry this store already holds for that field,
//! because that is what the server does: a field with no expiry is written as a plain `HSET`, and
//! `HSET` clears whatever TTL the server holds for that field, the same way `SET` clears a key's
//! TTL. The consequence is user-visible and intended: a view that drops its `ttl_days` stops
//! asking for an expiry, the rewrite clears the old one, and its values stop expiring
//! server-side. `rewriting_a_field_without_an_expiry_clears_the_servers_expiry` in
//! `tests/valkey_integration.rs` pins that against a real Valkey.

use std::collections::HashMap;

use super::{OnlineStore, ProjectScan, ReadRequest, WriteBatch, read_micros};
use crate::error::Result;

/// One field's bytes and the expiry the write asked for.
#[derive(Debug, Clone)]
struct Stored {
    value: Box<[u8]>,
    /// Unix seconds. Recorded and never acted on; see the module documentation.
    expires_at_unix_secs: Option<i64>,
}

/// One entity's fields, in field-name order.
#[derive(Debug, Default, Clone)]
pub struct Fields {
    entries: Box<[(Box<str>, Stored)]>,
}

impl Fields {
    /// The value stored for one field.
    pub fn get(&self, field: &str) -> Option<&[u8]> {
        self.position(field)
            .map(|at| self.entries[at].1.value.as_ref())
    }

    /// Whether the entity has this field.
    pub fn contains_key(&self, field: &str) -> bool {
        self.position(field).is_some()
    }

    /// How many fields the entity has.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the entity has no fields.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// When one field was asked to expire, as Unix seconds, or `None` when nothing asked.
    ///
    /// For assertions. Nothing in the store consults it.
    pub fn expiry(&self, field: &str) -> Option<i64> {
        self.position(field)
            .and_then(|at| self.entries[at].1.expires_at_unix_secs)
    }

    /// Where a field sits. The entries are kept in name order, so this is a binary search.
    fn position(&self, field: &str) -> Option<usize> {
        self.entries
            .binary_search_by(|(name, _)| name.as_ref().cmp(field))
            .ok()
    }
}

/// An in-process [`OnlineStore`].
#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    /// One entry per entity hash, with that entity's fields in name order.
    hashes: HashMap<Box<[u8]>, Fields>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of hashes held. Useful in assertions.
    pub fn hash_count(&self) -> usize {
        self.hashes.len()
    }

    /// One entity's fields, or `None` when the entity has no hash.
    pub fn fields(&self, key: &[u8]) -> Option<&Fields> {
        self.hashes.get(key)
    }

    /// When one field was asked to expire, as Unix seconds, or `None` when nothing asked.
    ///
    /// For assertions. Nothing in the store consults it.
    pub fn field_expiry(&self, key: &[u8], field: &str) -> Option<i64> {
        self.fields(key)?.expiry(field)
    }
}

impl OnlineStore for MemoryStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<Vec<bool>> {
        let mut applied = Vec::with_capacity(batches.len());
        for batch in batches {
            // A guard is decided here, against this store's own map, which is what makes it
            // atomic here: nothing else can write between the comparison and the merge below,
            // because both happen under the one `&mut self` the call already holds. The server
            // has to earn the same property with a script.
            if let Some(guard) = &batch.guard {
                let stored = self
                    .hashes
                    .get(batch.key.as_slice())
                    .and_then(|fields| fields.get(&guard.field))
                    .map(read_micros)
                    .transpose()?;
                if stored.is_some_and(|recorded| recorded >= guard.event_micros) {
                    applied.push(false);
                    continue;
                }
            }

            // Taken out and put back so the fields can be re-sorted in place: one merge per
            // batch, not one per field. A batch names one entity but may carry only some of that
            // entity's fields, which is what `materialize` does when it writes one batch per row
            // per view, because `OnlineStore::write` leaves the fields a batch does not mention
            // alone.
            let mut entries = match self.hashes.remove(batch.key.as_slice()) {
                Some(fields) => fields.entries.into_vec(),
                None => Vec::new(),
            };
            for field in &batch.fields {
                let stored = Stored {
                    value: field.value.as_slice().into(),
                    // Carried through as written, including `None`, which clears whatever the
                    // store held. That is the server's `HSET` behaviour the module documents.
                    expires_at_unix_secs: field.expires_at_unix_secs,
                };
                match entries.binary_search_by(|(name, _)| name.as_ref().cmp(field.name.as_str())) {
                    Ok(at) => entries[at].1 = stored,
                    Err(at) => entries.insert(at, (Box::from(field.name.as_str()), stored)),
                }
            }
            // A batch with no fields still leaves a hash, which is what this store did before it
            // was compacted and what `materialize` relies on when it decides not to write an
            // empty registry.
            self.hashes.insert(
                batch.key.clone().into_boxed_slice(),
                Fields {
                    entries: entries.into_boxed_slice(),
                },
            );
            applied.push(true);
        }
        Ok(applied)
    }

    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
        Ok(requests
            .iter()
            .map(|request| {
                let fields = self.hashes.get(request.key.as_slice());
                request
                    .fields
                    .iter()
                    .map(|field| {
                        fields
                            .and_then(|fields| fields.get(field))
                            .map(<[u8]>::to_vec)
                    })
                    .collect()
            })
            .collect())
    }

    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
        for (key, fields) in keys_and_fields {
            let Some(existing) = self.hashes.remove(key.as_slice()) else {
                continue;
            };
            let mut entries = existing.entries.into_vec();
            entries.retain(|(name, _)| !fields.iter().any(|field| field.as_str() == name.as_ref()));
            // A hash whose last field went is removed, which is what the server does when the
            // last field of a key is deleted.
            if !entries.is_empty() {
                self.hashes.insert(
                    key.clone().into_boxed_slice(),
                    Fields {
                        entries: entries.into_boxed_slice(),
                    },
                );
            }
        }
        Ok(())
    }
}

impl ProjectScan for MemoryStore {
    async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
        Ok(self
            .hashes
            .get(key)
            .map(|fields| {
                fields
                    .entries
                    .iter()
                    .map(|(name, stored)| (name.to_string(), stored.value.to_vec()))
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>> {
        let prefix = crate::online::project_key_prefix(project);
        Ok(self
            .hashes
            .keys()
            .filter(|key| key.starts_with(&prefix) && &key[..] != exclude)
            .map(|key| key.to_vec())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::MemoryStore;
    use crate::online::{OnlineStore, WriteBatch, WrittenField};

    /// A field written with no expiry loses the expiry the store already recorded for it,
    /// because that is what the server does: the write is a plain `HSET`, which sets the value
    /// and clears the field's TTL. A view whose definition loses its `ttl_days` writes `v:{view}`
    /// this way and its values stop expiring server-side, which
    /// `rewriting_a_field_without_an_expiry_clears_the_servers_expiry` in
    /// `tests/valkey_integration.rs` pins against a real Valkey. The read path is then the only
    /// thing that stops serving the value once it is stale.
    #[tokio::test]
    async fn a_write_without_an_expiry_clears_a_recorded_one() {
        let key = b"ads:user_id:2:u1".to_vec();
        let mut store = MemoryStore::new();

        store
            .write(&[WriteBatch::new(
                key.clone(),
                vec![WrittenField::new(
                    "v:clicks",
                    b"one".to_vec(),
                    Some(1_700_000_000),
                )],
            )])
            .await
            .expect("a write with an expiry");
        assert_eq!(store.field_expiry(&key, "v:clicks"), Some(1_700_000_000));

        store
            .write(&[WriteBatch::new(
                key.clone(),
                vec![WrittenField::new("v:clicks", b"two".to_vec(), None)],
            )])
            .await
            .expect("a write with no expiry");

        assert_eq!(
            store.fields(&key).and_then(|fields| fields.get("v:clicks")),
            Some(b"two".as_slice()),
            "the value is overwritten either way"
        );
        assert_eq!(
            store.field_expiry(&key, "v:clicks"),
            None,
            "the server's `HSET` clears the field's expiry, so the double drops it too"
        );
    }

    /// Writing a field twice replaces it rather than appending a second entry, and the entries
    /// stay in name order however they were written. Both are invariants of holding one sorted
    /// slice per entity, and neither is visible from the trait.
    #[tokio::test]
    async fn a_field_written_twice_is_stored_once_in_name_order() {
        let key = b"ads:user_id:2:u1".to_vec();
        let mut store = MemoryStore::new();

        store
            .write(&[WriteBatch::new(
                key.clone(),
                vec![
                    WrittenField::new("v:zeta", b"z".to_vec(), None),
                    WrittenField::new("v:alpha", b"a".to_vec(), None),
                ],
            )])
            .await
            .expect("a write out of order");
        store
            .write(&[WriteBatch::new(
                key.clone(),
                vec![WrittenField::new("v:zeta", b"z2".to_vec(), None)],
            )])
            .await
            .expect("a second write");

        let fields = store.fields(&key).expect("the entity keeps its hash");
        assert_eq!(
            fields.len(),
            2,
            "the second write replaces rather than appends"
        );
        assert_eq!(fields.get("v:zeta"), Some(b"z2".as_slice()));
        assert_eq!(fields.get("v:alpha"), Some(b"a".as_slice()));
    }

    /// A batch with no fields still leaves a hash, which is a deliberate divergence: the server
    /// queues nothing at all for an empty batch. `materialize` leans on this store creating the
    /// hash for every key it is handed when it decides not to write an empty registry, so what
    /// is pinned here is the hash being there rather than the fields being absent.
    #[tokio::test]
    async fn an_empty_batch_still_leaves_a_hash() {
        let key = b"ads:user_id:2:u1".to_vec();
        let mut store = MemoryStore::new();

        store
            .write(&[WriteBatch::new(key.clone(), Vec::new())])
            .await
            .expect("a write with no fields");

        assert_eq!(store.hash_count(), 1, "the key is a hash all the same");
        let fields = store.fields(&key).expect("the key is a hash all the same");
        assert!(fields.is_empty(), "and that hash holds no fields");
    }

    /// Deleting an entity's last field drops the entity rather than leaving an empty hash, which
    /// is what the server does when the last field of a key is deleted. The hash count is per
    /// entity and `scan_entity_keys` walks the same map, so an empty hash left behind would be a
    /// key the server does not have.
    #[tokio::test]
    async fn deleting_the_last_field_drops_the_entity() {
        let key = b"ads:user_id:2:u1".to_vec();
        let mut store = MemoryStore::new();

        store
            .write(&[WriteBatch::new(
                key.clone(),
                vec![
                    WrittenField::new("v:clicks", b"one".to_vec(), None),
                    WrittenField::new("v:views", b"two".to_vec(), None),
                ],
            )])
            .await
            .expect("a two-field write");

        store
            .delete_fields(&[(key.clone(), vec!["v:clicks".to_owned()])])
            .await
            .expect("a delete that leaves one field");
        assert!(
            store.fields(&key).is_some_and(|fields| fields.len() == 1),
            "one field is left, so the entity stays"
        );

        store
            .delete_fields(&[(key.clone(), vec!["v:views".to_owned()])])
            .await
            .expect("a delete of the last field");

        assert_eq!(
            store.hash_count(),
            0,
            "the entity's hash went with its last field"
        );
        assert!(store.fields(&key).is_none(), "so the entity is gone");
    }
}

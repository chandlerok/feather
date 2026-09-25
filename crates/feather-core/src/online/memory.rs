//! In-memory [`OnlineStore`].
//!
//! The default for tests, so unit tests need no container. It is not a cache and
//! makes no attempt to be fast; it exists to make the read path exercisable.
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

use super::{OnlineStore, ProjectScan, ReadRequest, WriteBatch};
use crate::error::Result;

#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    hashes: HashMap<Vec<u8>, HashMap<String, Vec<u8>>>,
    expiries: HashMap<Vec<u8>, HashMap<String, i64>>,
}

impl MemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of hashes held. Useful in assertions.
    pub fn hash_count(&self) -> usize {
        self.hashes.len()
    }

    /// Fields in one hash, for assertions.
    pub fn fields(&self, key: &[u8]) -> Option<&HashMap<String, Vec<u8>>> {
        self.hashes.get(key)
    }

    /// When one field was asked to expire, as Unix seconds, or `None` when nothing asked.
    ///
    /// For assertions. Nothing in the store consults it.
    pub fn field_expiry(&self, key: &[u8], field: &str) -> Option<i64> {
        self.expiries.get(key)?.get(field).copied()
    }
}

impl OnlineStore for MemoryStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
        for batch in batches {
            let hash = self.hashes.entry(batch.key.clone()).or_default();
            for field in &batch.fields {
                hash.insert(field.name.clone(), field.value.clone());
                match field.expires_at_unix_secs {
                    // Recorded when the write carries one.
                    Some(at) => {
                        self.expiries
                            .entry(batch.key.clone())
                            .or_default()
                            .insert(field.name.clone(), at);
                    }
                    // Cleared when it does not, which is what the server does: a field written
                    // with no expiry is a plain `HSET`, and `HSET` clears any TTL the server
                    // holds for that field. A view that drops its `ttl_days` therefore stops
                    // expiring server-side, and the read-time TTL check is left as the only
                    // thing that stops serving its values. Keeping the recorded expiry here
                    // would show a test a state the server never reaches.
                    None => {
                        if let Some(expiries) = self.expiries.get_mut(&batch.key) {
                            expiries.remove(&field.name);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
        Ok(requests
            .iter()
            .map(|request| {
                let hash = self.hashes.get(&request.key);
                request
                    .fields
                    .iter()
                    .map(|field| hash.and_then(|h| h.get(field)).cloned())
                    .collect()
            })
            .collect())
    }

    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
        for (key, fields) in keys_and_fields {
            if let Some(hash) = self.hashes.get_mut(key) {
                for field in fields {
                    hash.remove(field);
                }
                if hash.is_empty() {
                    self.hashes.remove(key);
                }
            }
            if let Some(expiries) = self.expiries.get_mut(key) {
                for field in fields {
                    expiries.remove(field);
                }
                if expiries.is_empty() {
                    self.expiries.remove(key);
                }
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
            .map(|hash| {
                hash.iter()
                    .map(|(field, value)| (field.clone(), value.clone()))
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>> {
        let prefix = crate::online::project_key_prefix(project);
        Ok(self
            .hashes
            .keys()
            .filter(|key| key.starts_with(&prefix) && key.as_slice() != exclude)
            .cloned()
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
            .write(&[WriteBatch {
                key: key.clone(),
                fields: vec![WrittenField::new(
                    "v:clicks",
                    b"one".to_vec(),
                    Some(1_700_000_000),
                )],
            }])
            .await
            .expect("a write with an expiry");
        assert_eq!(store.field_expiry(&key, "v:clicks"), Some(1_700_000_000));

        store
            .write(&[WriteBatch {
                key: key.clone(),
                fields: vec![WrittenField::new("v:clicks", b"two".to_vec(), None)],
            }])
            .await
            .expect("a write with no expiry");

        assert_eq!(
            store.fields(&key).and_then(|hash| hash.get("v:clicks")),
            Some(&b"two".to_vec()),
            "the value is overwritten either way"
        );
        assert_eq!(
            store.field_expiry(&key, "v:clicks"),
            None,
            "the server's `HSET` clears the field's expiry, so the double drops it too"
        );
    }
}

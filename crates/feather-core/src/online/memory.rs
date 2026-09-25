//! In-memory [`OnlineStore`].
//!
//! The default for tests, so unit tests need no container. It is not a cache and
//! makes no attempt to be fast; it exists to make the read path exercisable.
//!
//! A field's expiry is remembered and never acted on: this store has no clock and reclaims
//! nothing, and the read-time TTL check is the authoritative path in any case. Keeping the
//! expiry visible is what lets a test assert what the write path asked for without a server.

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
                    Some(at) => {
                        self.expiries
                            .entry(batch.key.clone())
                            .or_default()
                            .insert(field.name.clone(), at);
                    }
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
}

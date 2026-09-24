//! In-memory [`OnlineStore`].
//!
//! The default for tests, so unit tests need no container. It is not a cache and
//! makes no attempt to be fast; it exists to make the read path exercisable.

use std::collections::HashMap;

use super::{OnlineStore, ReadRequest, WriteBatch};
use crate::error::Result;

#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    hashes: HashMap<Vec<u8>, HashMap<String, Vec<u8>>>,
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
}

impl OnlineStore for MemoryStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
        for batch in batches {
            let hash = self.hashes.entry(batch.key.clone()).or_default();
            for (field, value) in &batch.fields {
                hash.insert(field.clone(), value.clone());
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
        }
        Ok(())
    }
}

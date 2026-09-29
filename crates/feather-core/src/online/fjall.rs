//! Embedded LSM-backed [`OnlineStore`].
//!
//! fjall has no hash type, so the one-hash-per-entity layout that a Valkey read is a single
//! `HMGET` against is here a run of adjacent keys. That run is what the key encoding below is
//! for, and getting it right is most of what this module does.
//!
//! **Why the key is length-prefixed.** [`crate::key::entity_hash_key`] is
//! `project:entity_name:` followed by the output of `encode_entity_key`, and the field names are
//! `v:{view}` and `f:{view}`, which contain a colon themselves. So no byte is a usable separator:
//! the encoded key is arbitrary bytes and can contain any of them. The record is therefore
//!
//! ```text
//! [key_len:u32 big-endian][entity_hash_key][field_name]
//! ```
//!
//! which is unambiguous in both directions. It also groups one entity's fields into a single
//! contiguous run, because the prefix is fixed for a given key, so a per-entity read is one range
//! rather than one point get per field. That is what replaces the `HMGET`.
//!
//! **The one thing this encoding gives up.** A project scan cannot be a prefix scan, because the
//! length prefix comes first and it varies per entity. [`ProjectScan::scan_entity_keys`] is
//! therefore a full keyspace walk that parses each record's key and filters by project. That is
//! the right trade: the only caller is retiring a view, which is rare, whereas the serving path
//! is a point read and must be exact. See `ponytail:` on that method.
//!
//! **Durability.** [`OnlineStore::write`] is called once per materialization with every batch
//! (`materialize.rs`), so one `SyncData` per call is one fsync per refresh, which is what a
//! refresh wants. It is configurable because a push path that calls `write` per update would
//! otherwise pay an fsync per update; see [`FjallStore::with_persist_mode`].
//!
//! **Blocking.** The trait is async because the Valkey store is. This one is not, and it is
//! deliberately not wrapped in `spawn_blocking`: a read is a point get measured in single-digit
//! microseconds when the working set is cached, so blocking a runtime worker for that is cheaper
//! than a task hop. The ceiling is the cold case, where a read that misses the block cache and
//! the page cache costs ~150us; past that, or past a slow device, a serving path would want
//! `spawn_blocking` around this.

use std::collections::HashSet;
use std::path::Path;

use super::{OnlineStore, ProjectScan, ReadRequest, WriteBatch};
use crate::error::Result;
use crate::online::project_key_prefix;

/// Records per journal entry on the write path.
///
/// Large enough that the fsync amortises, small enough that one commit does not pin a
/// multi-megabyte buffer. A materialization arrives as many single-entity batches, so this is
/// what turns one call into a handful of commits rather than one per row.
const WRITE_CHUNK: usize = 4096;

/// An LSM store over a directory this process owns.
///
/// A database may be opened by one process at a time and only by its owner, so this is the
/// embedded tier: fast point reads with no second service, at the cost of being single-writer.
pub struct FjallStore {
    /// Held for its `Drop`, which is what flushes and closes the database. The keyspace borrows
    /// nothing from it, so the order here is only about keeping the owner alive.
    _db: fjall::Database,
    keyspace: fjall::Keyspace,
    persist_mode: fjall::PersistMode,
}

impl FjallStore {
    /// Open, or create, the store at `path`.
    ///
    /// `cache_bytes` bounds the resident set. fjall's own guidance is 20-25% of available memory,
    /// or more if the dataset fits; below the working set, reads miss to the page cache and then
    /// to the device, which is the ~150us case rather than the ~5us one.
    ///
    /// `memtable_bytes` is how much is buffered in memory before a flush. It is the one knob
    /// that matters for the write path: measured across scattered updates, a 4MiB memtable wrote
    /// 5.3x the logical bytes and a 64MiB one wrote 1.1x, because a small memtable flushes often
    /// and every flush leaves an overlapping run for compaction to rewrite.
    pub fn open(path: impl AsRef<Path>, cache_bytes: u64, memtable_bytes: u64) -> Result<Self> {
        let path = path.as_ref();
        let db = fjall::Database::builder(path)
            .cache_size(cache_bytes)
            .worker_threads(2)
            .open()?;
        let keyspace = db.keyspace(Self::KEYSPACE, || {
            fjall::KeyspaceCreateOptions::default().max_memtable_size(memtable_bytes)
        })?;
        Ok(Self {
            _db: db,
            keyspace,
            persist_mode: fjall::PersistMode::SyncData,
        })
    }

    /// Change what a write costs in durability.
    ///
    /// `SyncData` is the default and is right for a materialization, which is idempotent and
    /// re-runnable. A push path calling [`OnlineStore::write`] once per update would want
    /// [`fjall::PersistMode::Buffer`] and its own durability policy, at the cost of losing
    /// updates written since the last flush on a hard kill.
    #[must_use]
    pub fn with_persist_mode(mut self, mode: fjall::PersistMode) -> Self {
        self.persist_mode = mode;
        self
    }

    /// The keyspace the online store keeps its records in.
    ///
    /// One keyspace rather than one per project, because a project is a key prefix and fjall already
    /// has prefix iteration. Splitting them would buy per-project compaction at the cost of a
    /// keyspace per project and a lookup to find it.
    const KEYSPACE: &str = "online";

    /// The key one record is stored under.
    fn record(key: &[u8], field: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + key.len() + field.len());
        out.extend_from_slice(&(key.len() as u32).to_be_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(field.as_bytes());
        out
    }

    /// The prefix every field of one entity is stored under.
    ///
    /// Unique to that entity: another entity's records can only share it if the same bytes and
    /// the same length appear at the front, which means the same key.
    fn entity_prefix(key: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + key.len());
        out.extend_from_slice(&(key.len() as u32).to_be_bytes());
        out.extend_from_slice(key);
        out
    }

    /// The entity key a record belongs to, or `None` if the record is not one of ours.
    fn parse_key(record: &[u8]) -> Option<&[u8]> {
        let len = u32::from_be_bytes(record.get(..4)?.try_into().ok()?) as usize;
        record.get(4..4 + len)
    }
}

impl OnlineStore for FjallStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
        // `expires_at_unix_secs` is dropped, which is not a loss. The trait documents that a
        // store which cannot honour an expiry writes the value anyway, because the read-time TTL
        // check in `read_entities` is what decides whether a value is served: an unexpired
        // leftover costs reclamation and never correctness. This store has no reclamation at
        // all, so what it would cost is disk, bounded by a compaction rewriting the key.
        for chunk in batches.chunks(WRITE_CHUNK) {
            let mut batch = fjall::OwnedWriteBatch::with_capacity(self._db.clone(), chunk.len());
            for write in chunk {
                for field in &write.fields {
                    batch.insert(
                        &self.keyspace,
                        Self::record(&write.key, &field.name),
                        field.value.clone(),
                    );
                }
            }
            batch.durability(Some(self.persist_mode)).commit()?;
        }
        Ok(())
    }

    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
        let mut out = Vec::with_capacity(requests.len());
        for request in requests {
            let prefix = Self::entity_prefix(&request.key);
            let mut row: Vec<Option<Vec<u8>>> = vec![None; request.fields.len()];

            // One range per entity rather than a point get per field: fjall has no `multi_get`
            // (upstream #229), and a request's fields are the `v:` and `f:` pair of each view, so
            // scanning the entity's short run and matching against the requested names is fewer
            // operations than looking each one up. The name list is short, so it is matched
            // linearly rather than through a map built per request.
            for entry in self.keyspace.prefix(&prefix) {
                let (record, value) = entry.into_inner()?;
                let record: &[u8] = &record;
                let Some(name) = record.get(prefix.len()..) else {
                    continue;
                };
                if let Some(at) = request.fields.iter().position(|f| f.as_bytes() == name) {
                    let value: &[u8] = &value;
                    row[at] = Some(value.to_vec());
                }
            }
            out.push(row);
        }
        Ok(out)
    }

    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
        // Removing a field is a record delete, not a hash delete, so a retired view's fields go
        // without touching the fields of the views still on the entity.
        let mut batch =
            fjall::OwnedWriteBatch::with_capacity(self._db.clone(), keys_and_fields.len());
        for (key, fields) in keys_and_fields {
            for field in fields {
                batch.remove(&self.keyspace, Self::record(key, field));
            }
        }
        batch.durability(Some(self.persist_mode)).commit()?;
        Ok(())
    }
}

impl ProjectScan for FjallStore {
    async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
        let prefix = Self::entity_prefix(key);
        let mut out = Vec::new();
        for entry in self.keyspace.prefix(&prefix) {
            let (record, value) = entry.into_inner()?;
            let record: &[u8] = &record;
            if let Some(name) = record.get(prefix.len()..) {
                let value: &[u8] = &value;
                out.push((String::from_utf8_lossy(name).into_owned(), value.to_vec()));
            }
        }
        Ok(out)
    }

    /// Every entity key in a project, which is a full keyspace walk.
    ///
    /// `ponytail:` the length prefix that makes the per-entity encoding unambiguous also makes a
    /// project scan unable to be a prefix scan, so this walks everything and filters. Ceiling:
    /// it costs a full scan, which is fine for retiring a view and wrong for anything on a
    /// serving path. Upgrade path: a second keyspace holding one record per entity key, written
    /// alongside the fields, which turns this into a prefix scan at the cost of a second write
    /// per entity and a second thing to keep consistent.
    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>> {
        let prefix = project_key_prefix(project);
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        for entry in self.keyspace.iter() {
            let (record, _) = entry.into_inner()?;
            let Some(key) = Self::parse_key(&record) else {
                continue;
            };
            if key.starts_with(&prefix) && key != exclude && seen.insert(key.to_vec()) {
                // Returned inside the loop so `seen` grows with the walk rather than after it.
            }
        }
        Ok(seen
            .into_iter()
            .filter(|key| key.as_slice() != exclude)
            .collect())
    }
}

#[cfg(all(test, feature = "fjall"))]
mod tests {
    use super::*;
    use crate::key::{encode_entity_key, entity_hash_key, value_field};
    use crate::online::{WriteBatch, WrittenField, encode_freshness};

    /// A store in a temporary directory, closed when it drops.
    ///
    /// A database holds an exclusive lock while open, so a test that leaked one would fail the
    /// next test with an error about the directory rather than about its own assertion.
    fn store() -> (tempfile::TempDir, FjallStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = FjallStore::open(dir.path(), 8 << 20, 8 << 20).expect("open");
        (dir, store)
    }

    fn key(project: &str, entity: &str, id: &str) -> Vec<u8> {
        let encoded = encode_entity_key(&[id.as_bytes()]).expect("encode");
        entity_hash_key(project, entity, &encoded)
    }

    fn write_one(key: &[u8], fields: &[(&str, &[u8])]) -> WriteBatch {
        WriteBatch {
            key: key.to_vec(),
            fields: fields
                .iter()
                .map(|(name, value)| WrittenField::new(*name, value.to_vec(), None))
                .collect(),
        }
    }

    #[tokio::test]
    async fn a_written_field_reads_back() {
        let (_dir, mut store) = store();
        let k = key("ads", "user", "u1");
        store
            .write(&[write_one(&k, &[("v:view0", b"hello")])])
            .await
            .expect("write");

        let out = store
            .read(&[ReadRequest {
                key: k.clone(),
                fields: vec!["v:view0".into(), "f:view0".into()],
            }])
            .await
            .expect("read");

        assert_eq!(out[0][0].as_deref(), Some(&b"hello"[..]));
        // A field never written arrives as None rather than as an error, which is how a
        // TTL-expired value reaches the read path too.
        assert_eq!(out[0][1], None);
    }

    #[tokio::test]
    async fn a_partial_write_leaves_the_other_fields_alone() {
        let (_dir, mut store) = store();
        let k = key("ads", "user", "u1");
        store
            .write(&[write_one(&k, &[("v:a", b"1"), ("v:b", b"2")])])
            .await
            .expect("write");
        store
            .write(&[write_one(&k, &[("v:a", b"9")])])
            .await
            .expect("rewrite");

        let out = store
            .read(&[ReadRequest {
                key: k.clone(),
                fields: vec!["v:a".into(), "v:b".into()],
            }])
            .await
            .expect("read");
        assert_eq!(
            out[0][0].as_deref(),
            Some(&b"9"[..]),
            "the named field changed"
        );
        assert_eq!(
            out[0][1].as_deref(),
            Some(&b"2"[..]),
            "the other field did not"
        );
    }

    /// The reason the key is length-prefixed rather than joined.
    ///
    /// `u1` and `u10` are different entities whose encoded keys share a byte prefix, and
    /// `encode_entity_key` length-prefixes each component, so this is the shape a real project
    /// has. Reading one must not see the other's fields.
    #[tokio::test]
    async fn an_entity_is_not_confused_with_one_whose_key_extends_it() {
        let (_dir, mut store) = store();
        let short = key("ads", "user", "u1");
        let long = key("ads", "user", "u10");
        store
            .write(&[write_one(&short, &[("v:view0", b"short")])])
            .await
            .expect("write short");
        store
            .write(&[write_one(&long, &[("v:view0", b"long")])])
            .await
            .expect("write long");

        let out = store
            .read(&[
                ReadRequest {
                    key: short.clone(),
                    fields: vec!["v:view0".into()],
                },
                ReadRequest {
                    key: long,
                    fields: vec!["v:view0".into()],
                },
            ])
            .await
            .expect("read");
        assert_eq!(out[0][0].as_deref(), Some(&b"short"[..]));
        assert_eq!(out[1][0].as_deref(), Some(&b"long"[..]));
    }

    #[tokio::test]
    async fn a_field_with_no_entity_reads_as_all_none() {
        let (_dir, store) = store();
        let out = store
            .read(&[ReadRequest {
                key: key("ads", "user", "nobody"),
                fields: vec!["v:view0".into()],
            }])
            .await
            .expect("read");
        assert_eq!(out[0][0], None);
    }

    #[tokio::test]
    async fn delete_removes_only_the_named_field() {
        let (_dir, mut store) = store();
        let k = key("ads", "user", "u1");
        store
            .write(&[write_one(&k, &[("v:a", b"1"), ("v:b", b"2")])])
            .await
            .expect("write");
        store
            .delete_fields(&[(k.clone(), vec!["v:a".into()])])
            .await
            .expect("delete");

        let out = store
            .read(&[ReadRequest {
                key: k,
                fields: vec!["v:a".into(), "v:b".into()],
            }])
            .await
            .expect("read");
        assert_eq!(out[0][0], None, "the deleted field is gone");
        assert_eq!(
            out[0][1].as_deref(),
            Some(&b"2"[..]),
            "its neighbour is not"
        );
    }

    #[tokio::test]
    async fn hash_fields_returns_a_whole_entity() {
        let (_dir, mut store) = store();
        let k = key("ads", "user", "u1");
        store
            .write(&[write_one(&k, &[("v:a", b"1"), ("f:a", b"2")])])
            .await
            .expect("write");

        let mut fields = store.hash_fields(&k).await.expect("hash_fields");
        fields.sort();
        assert_eq!(
            fields,
            vec![
                ("f:a".to_string(), b"2".to_vec()),
                ("v:a".to_string(), b"1".to_vec())
            ]
        );
        assert!(
            store
                .hash_fields(b"not an entity")
                .await
                .expect("absent")
                .is_empty(),
            "an absent key yields an empty list, not an error, which is a first refresh"
        );
    }

    #[tokio::test]
    async fn scan_finds_a_projects_entities_and_skips_the_registry() {
        let (_dir, mut store) = store();
        let mine = key("ads", "user", "u1");
        let theirs = key("other", "user", "u2");
        let registry = crate::key::views_registry_key("ads");
        store
            .write(&[
                write_one(&mine, &[("v:a", b"1")]),
                write_one(&theirs, &[("v:a", b"2")]),
                write_one(&registry, &[("x", b"r")]),
            ])
            .await
            .expect("write");

        let mut found = store
            .scan_entity_keys("ads", &registry)
            .await
            .expect("scan");
        found.sort();
        assert_eq!(
            found,
            vec![mine],
            "only this project's entity, not the registry"
        );
    }

    #[tokio::test]
    async fn a_write_then_restart_keeps_the_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let k = key("ads", "user", "u1");
        {
            let mut store = FjallStore::open(dir.path(), 8 << 20, 8 << 20).expect("open");
            store
                .write(&[write_one(
                    &k,
                    &[("v:view0", b"persisted"), ("f:view0", &encode_freshness(1))],
                )])
                .await
                .expect("write");
        }
        // Reopened rather than reused: the claim under test is that a restart is a remap.
        let store = FjallStore::open(dir.path(), 8 << 20, 8 << 20).expect("reopen");
        let out = store
            .read(&[ReadRequest {
                key: k,
                fields: vec![value_field("view0")],
            }])
            .await
            .expect("read");
        assert_eq!(out[0][0].as_deref(), Some(&b"persisted"[..]));
    }
}

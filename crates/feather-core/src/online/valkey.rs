//! Valkey-backed [`OnlineStore`].
//!
//! Commands are pipelined: one round trip per chunk of work rather than one per
//! entity. Combined with the one-`HMGET`-per-entity rule in
//! [`crate::online::read_entities`], a request for N entities over M views costs
//! one round trip and N commands, not N x M.
//!
//! > `ponytail:` fields are written with plain `HSET`, so an expired value is
//! > reclaimed only when its hash is rewritten. The read-time TTL check in
//! > `read_entities` is the authoritative path and is unaffected, so this is a
//! > reclamation gap rather than a correctness one. The upgrade is `HSETEX` with
//! > an absolute `PXAT` derived from the event timestamp plus the TTL, which
//! > needs the write batch to carry a per-view expiry. Verified available on
//! > Valkey 9.1.2; see "TTL and reclamation" in the architecture document.

use redis::RedisResult;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};

use super::{OnlineStore, ReadRequest, WriteBatch};
use crate::error::Result;

/// Commands per pipeline flush. Large enough to amortise the round trip, small
/// enough that one flush does not hold a multi-megabyte request buffer.
const DEFAULT_CHUNK: usize = 1024;

pub struct ValkeyStore {
    connection: ConnectionManager,
    chunk: usize,
}

impl ValkeyStore {
    /// Connect and enable RESP3.
    ///
    /// RESP3 is not optional: client-side caching and its invalidation pushes
    /// only work over it. A URL without `?protocol=resp3` is upgraded rather
    /// than silently falling back to RESP2.
    pub async fn connect(url: &str) -> Result<Self> {
        let url = if url.contains("protocol=") {
            url.to_owned()
        } else if url.contains('?') {
            format!("{url}&protocol=resp3")
        } else {
            format!("{url}?protocol=resp3")
        };

        let client = redis::Client::open(url)?;
        let connection = client.get_connection_manager().await?;
        Ok(Self {
            connection,
            chunk: DEFAULT_CHUNK,
        })
    }

    /// Override the pipeline chunk size.
    pub fn with_chunk(mut self, chunk: usize) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    /// Connect with a push sender for invalidation messages, and register a
    /// broadcasting prefix so the server reports changes under it.
    ///
    /// `CLIENT TRACKING` is issued here rather than left to the client library:
    /// broadcasting mode with a prefix is what keeps server memory flat, and the
    /// library's own cache helper does not use it.
    pub async fn connect_tracking<S>(url: &str, prefix: &str, sender: S) -> Result<Self>
    where
        S: redis::aio::AsyncPushSender + Send + Sync + 'static,
    {
        let url = if url.contains("protocol=") {
            url.to_owned()
        } else if url.contains('?') {
            format!("{url}&protocol=resp3")
        } else {
            format!("{url}?protocol=resp3")
        };

        let client = redis::Client::open(url)?;
        let config = ConnectionManagerConfig::new().set_push_sender(sender);
        let mut connection = client.get_connection_manager_with_config(config).await?;

        // No two prefixes may overlap; Valkey rejects `foo` and `foob` together
        // because both would match `foobar`.
        let mut command = redis::cmd("CLIENT");
        command
            .arg("TRACKING")
            .arg("ON")
            .arg("BCAST")
            .arg("PREFIX")
            .arg(prefix);
        let _: RedisResult<()> = command.query_async(&mut connection).await?;

        Ok(Self {
            connection,
            chunk: DEFAULT_CHUNK,
        })
    }
}

impl OnlineStore for ValkeyStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
        for group in batches.chunks(self.chunk) {
            let mut pipeline = redis::pipe();
            for batch in group {
                let mut command = redis::cmd("HSET");
                command.arg(&batch.key);
                for (field, value) in &batch.fields {
                    command.arg(field).arg(value);
                }
                pipeline.add_command(command);
            }
            let _: redis::Value = pipeline.query_async(&mut self.connection).await?;
        }
        Ok(())
    }

    async fn read(&self, requests: &[ReadRequest]) -> Result<Vec<Vec<Option<Vec<u8>>>>> {
        let mut connection = self.connection.clone();
        let mut out = Vec::with_capacity(requests.len());

        for group in requests.chunks(self.chunk) {
            let mut pipeline = redis::pipe();
            for request in group {
                let mut command = redis::cmd("HMGET");
                command.arg(&request.key);
                for field in &request.fields {
                    command.arg(field);
                }
                pipeline.add_command(command);
            }
            let rows: Vec<Vec<Option<Vec<u8>>>> = pipeline.query_async(&mut connection).await?;
            out.extend(rows);
        }

        Ok(out)
    }

    async fn delete_fields(&mut self, keys_and_fields: &[(Vec<u8>, Vec<String>)]) -> Result<()> {
        for group in keys_and_fields.chunks(self.chunk) {
            let mut pipeline = redis::pipe();
            for (key, fields) in group {
                let mut command = redis::cmd("HDEL");
                command.arg(key);
                for field in fields {
                    command.arg(field);
                }
                pipeline.add_command(command);
            }
            let _: redis::Value = pipeline.query_async(&mut self.connection).await?;
        }
        Ok(())
    }
}

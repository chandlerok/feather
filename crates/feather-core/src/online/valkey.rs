//! Valkey-backed [`OnlineStore`].
//!
//! Commands are pipelined: one round trip per chunk of work rather than one per
//! entity. Combined with the one-`HMGET`-per-entity rule in
//! [`crate::online::read_entities`], a request for N entities over M views costs
//! one round trip and N commands, not N x M.
//!
//! A field that carries an absolute expiry is written with a server-side expiry, so the server
//! reclaims it without waiting for its hash to be rewritten. Which command does that is decided
//! once per connection from what the server answers, because the family is younger than the
//! servers in the field: `HEXPIREAT` is a Redis 7.4 addition and `HSETEX` is a Redis 8.0 one.
//! The read-time TTL check in `read_entities` remains the authoritative path, so a server
//! without either command stores the same bytes and only reclaims them later. See "TTL and
//! reclamation" in the architecture document.

use std::collections::BTreeMap;

use redis::RedisResult;
use redis::aio::{ConnectionManager, ConnectionManagerConfig};

use super::{OnlineStore, ProjectScan, ReadRequest, WriteBatch, WrittenField};
use crate::error::Result;

/// Commands per pipeline flush. Large enough to amortise the round trip, small
/// enough that one flush does not hold a multi-megabyte request buffer.
const DEFAULT_CHUNK: usize = 1024;

/// How a server sets an absolute expiry on a hash field, decided once per connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldExpiry {
    /// `HSETEX`, which sets a value and its expiry in one command. A Redis 8.0 addition.
    Setex,
    /// `HSET` followed by `HEXPIREAT`. Redis 7.4 and later, so it is the wider floor.
    Expireat,
    /// Neither, so a field is reclaimed only when its hash is rewritten or deleted.
    None,
}

/// Detect how a server expires a hash field by asking it about the commands.
///
/// A capability probe rather than a version comparison, because the versions do not line up:
/// Valkey and Redis number their releases differently, and the two commands arrived in
/// different ones. `COMMAND INFO` answers with a one-element array holding nil for a command
/// the server does not know.
async fn detect_field_expiry(connection: &mut ConnectionManager) -> FieldExpiry {
    if server_knows(connection, "HSETEX").await {
        FieldExpiry::Setex
    } else if server_knows(connection, "HEXPIREAT").await {
        FieldExpiry::Expireat
    } else {
        FieldExpiry::None
    }
}

/// Whether a server knows a command, asked once when a connection is opened.
async fn server_knows(connection: &mut ConnectionManager, command: &str) -> bool {
    let reply: RedisResult<redis::Value> = redis::cmd("COMMAND")
        .arg("INFO")
        .arg(command)
        .query_async(connection)
        .await;
    match reply {
        Ok(redis::Value::Array(entries)) => entries
            .first()
            .is_some_and(|entry| !matches!(entry, redis::Value::Nil)),
        _ => false,
    }
}

pub struct ValkeyStore {
    connection: ConnectionManager,
    chunk: usize,
    field_expiry: FieldExpiry,
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
        let mut connection = client.get_connection_manager().await?;
        let field_expiry = detect_field_expiry(&mut connection).await;
        Ok(Self {
            connection,
            chunk: DEFAULT_CHUNK,
            field_expiry,
        })
    }

    /// Override the pipeline chunk size.
    pub fn with_chunk(mut self, chunk: usize) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    /// Turn server-side field expiry off, whatever this server supports.
    ///
    /// The settings default it off (`valkey.field_expiration`), because what gets written to
    /// the store should not change with a server's version unless an operator said so. The
    /// commands themselves are still chosen by what the server answers.
    pub fn with_field_expiration(mut self, enabled: bool) -> Self {
        if !enabled {
            self.field_expiry = FieldExpiry::None;
        }
        self
    }

    /// How this store expires a hash field on the server it is connected to.
    pub fn field_expiry(&self) -> FieldExpiry {
        self.field_expiry
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

        let field_expiry = detect_field_expiry(&mut connection).await;
        Ok(Self {
            connection,
            chunk: DEFAULT_CHUNK,
            field_expiry,
        })
    }
}

impl OnlineStore for ValkeyStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<()> {
        for group in batches.chunks(self.chunk) {
            let mut pipeline = redis::pipe();
            for batch in group {
                queue_write(&mut pipeline, batch, self.field_expiry);
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

/// Queue the commands that write one batch.
///
/// The fields are grouped by their expiry, because `HSETEX` and `HEXPIREAT` each take one
/// time for every field they list. A batch normally holds one expiring field, the view's
/// value, and one field that never expires, its freshness, so a view with a TTL costs one more
/// command than one without. Both commands are queued before the pipeline is flushed, so no
/// reader ever sees the value without its expiry, and the extra command is not an extra round
/// trip.
fn queue_write(pipeline: &mut redis::Pipeline, batch: &WriteBatch, expiry: FieldExpiry) {
    let mut groups: BTreeMap<Option<i64>, Vec<&WrittenField>> = BTreeMap::new();
    for field in &batch.fields {
        groups
            .entry(field.expires_at_unix_secs)
            .or_default()
            .push(field);
    }

    for (expires_at, fields) in groups {
        match (expires_at, expiry) {
            (Some(at), FieldExpiry::Setex) => {
                let mut command = redis::cmd("HSETEX");
                command
                    .arg(&batch.key)
                    .arg("EXAT")
                    .arg(at)
                    .arg("FIELDS")
                    .arg(fields.len());
                for field in fields {
                    command.arg(&field.name).arg(&field.value);
                }
                pipeline.add_command(command);
            }
            (Some(at), FieldExpiry::Expireat) => {
                queue_hset(pipeline, &batch.key, &fields);
                let mut command = redis::cmd("HEXPIREAT");
                command
                    .arg(&batch.key)
                    .arg(at)
                    .arg("FIELDS")
                    .arg(fields.len());
                for field in fields {
                    command.arg(&field.name);
                }
                pipeline.add_command(command);
            }
            // Either the field never expires or the server cannot expire a field. Without
            // server support the value is still correct; it is reclaimed when its hash is
            // rewritten, and the read-time TTL check is what decides whether it is served.
            (None, _) | (Some(_), FieldExpiry::None) => {
                queue_hset(pipeline, &batch.key, &fields);
            }
        }
    }
}

/// Queue one `HSET` for a set of fields.
fn queue_hset(pipeline: &mut redis::Pipeline, key: &[u8], fields: &[&WrittenField]) {
    if fields.is_empty() {
        return;
    }
    let mut command = redis::cmd("HSET");
    command.arg(key);
    for field in fields {
        command.arg(&field.name).arg(&field.value);
    }
    pipeline.add_command(command);
}

impl ProjectScan for ValkeyStore {
    async fn hash_fields(&self, key: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
        let mut connection = self.connection.clone();
        // Typed as a map so the reply is read as field-value pairs whether the connection is
        // RESP2 or RESP3; the store asks for RESP3, where `HGETALL` is a map, and a RESP2 reply
        // is a flat array that converts to the same thing. An absent key answers nil, which
        // converts to an empty map.
        let fields: BTreeMap<String, Vec<u8>> = redis::cmd("HGETALL")
            .arg(key)
            .query_async(&mut connection)
            .await?;
        Ok(fields.into_iter().collect())
    }
}

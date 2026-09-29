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
use redis::aio::ConnectionManager;

use super::{OnlineStore, ProjectScan, ReadRequest, WriteBatch, WrittenField, encode_freshness};
use crate::error::Result;

/// Commands per pipeline flush. Large enough to amortise the round trip, small
/// enough that one flush does not hold a multi-megabyte request buffer.
const DEFAULT_CHUNK: usize = 1024;

/// How many keys one `SCAN` step is asked to return. A hint to the server, not a bound on the
/// reply, and large enough that a project of any size takes few steps.
const SCAN_COUNT: usize = 1024;

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
    /// RESP3 is requested for the client-side caching #25 holds the design for, not for the
    /// read path: `hash_fields` reads a RESP2 reply into the same map. A URL without
    /// `?protocol=resp3` is upgraded rather than silently falling back to RESP2.
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
}

impl OnlineStore for ValkeyStore {
    async fn write(&mut self, batches: &[WriteBatch]) -> Result<Vec<bool>> {
        let mut applied = vec![false; batches.len()];
        let mut offset = 0;
        for group in batches.chunks(self.chunk) {
            // Guarded and unguarded writes go down two pipelines rather than one, so that a
            // reply is attributable to a batch by position. A refresh writes only guarded
            // batches and the project registry is the only unguarded write, so one of the two
            // is empty in practice and this costs a single round trip either way.
            let mut guarded: Vec<usize> = Vec::new();
            let mut plain: Vec<usize> = Vec::new();
            for (at, batch) in group.iter().enumerate() {
                if batch.is_guarded() {
                    guarded.push(at);
                } else {
                    plain.push(at);
                }
            }

            if !plain.is_empty() {
                let mut pipeline = redis::pipe();
                for at in &plain {
                    queue_write(&mut pipeline, &group[*at], self.field_expiry);
                }
                let _: redis::Value = pipeline.query_async(&mut self.connection).await?;
                for at in plain {
                    applied[offset + at] = true;
                }
            }

            if !guarded.is_empty() {
                let mut pipeline = redis::pipe();
                for at in &guarded {
                    queue_guarded_write(&mut pipeline, &group[*at], self.field_expiry);
                }
                // The script answers 1 for a write it made and 0 for one the guard refused, and
                // it is the only command in this pipeline, so the replies line up with the
                // guarded batches in order.
                let replies: Vec<i64> = pipeline.query_async(&mut self.connection).await?;
                for (at, reply) in guarded.iter().zip(replies) {
                    applied[offset + at] = reply != 0;
                }
            }

            offset += group.len();
        }
        Ok(applied)
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

/// Queue the commands that write one batch, guarded or not.
///
/// A guarded batch goes down as a single [`GUARDED_WRITE_SCRIPT`] evaluation, because the
/// freshness comparison and the write have to be one step. Reading the field first and then
/// writing is a lost update, which is the one failure the guard exists to prevent, and a
/// pipeline does not prevent it: `redis::pipe()` batches commands into one write without
/// wrapping them in `MULTI`/`EXEC`, so another connection can write between the read and the
/// write. A script is atomic, so the check and the write cannot be interleaved with anything.
///
/// `EVAL` rather than `EVALSHA` because the difference is a lookup: the server keys its
/// compiled-script cache by the body's SHA-1, which `EVAL` computes, so a repeated body is
/// served from that cache and `EVALSHA` only saves the computation. That is not worth a script
/// load on connect, a `NOSCRIPT` recovery path, and a retry of the whole chunk.
fn queue_guarded_write(pipeline: &mut redis::Pipeline, batch: &WriteBatch, expiry: FieldExpiry) {
    let Some(guard) = &batch.guard else {
        queue_write(pipeline, batch, expiry);
        return;
    };

    let mut command = redis::cmd("EVAL");
    command
        .arg(GUARDED_WRITE_SCRIPT)
        .arg(1)
        .arg(&batch.key)
        .arg(&guard.field)
        // The claim goes over the wire in the encoding the read path decodes, so the script
        // compares like for like rather than being handed a second spelling of the timestamp.
        .arg(encode_freshness(guard.event_micros))
        .arg(expiry_mode(expiry))
        .arg(batch.fields.len());
    for field in &batch.fields {
        command.arg(&field.name).arg(&field.value);
        match field.expires_at_unix_secs {
            Some(at) => {
                command.arg(at);
            }
            // An empty string rather than a sentinel, because a field that never expires and a
            // field whose expiry the server cannot set take different paths in the script and
            // both have to be sayable.
            None => {
                command.arg("");
            }
        }
    }
    pipeline.add_command(command);
}

/// The number the script reads as "how to write a field that carries an expiry".
const fn expiry_mode(expiry: FieldExpiry) -> usize {
    match expiry {
        FieldExpiry::None => 0,
        FieldExpiry::Expireat => 1,
        FieldExpiry::Setex => 2,
    }
}

/// Write a batch only if nothing fresher is recorded, in one step.
///
/// The check and the write are inside the same script, which is what makes the guard atomic.
/// See [`queue_guarded_write`] for why a pipeline cannot stand in for it.
const GUARDED_WRITE_SCRIPT: &str = r#"
local key = KEYS[1]
local mode = tonumber(ARGV[3])
local count = tonumber(ARGV[4])

local recorded = redis.call('HMGET', key, ARGV[1])
local current = recorded and recorded[1]
-- A freshness field that is absent, or that is not the eight bytes a freshness is written as,
-- is not something this guard can compare, and the write stands rather than being refused for
-- good by a value it cannot read. Every writer goes through this script, so a field is either
-- absent or exactly eight bytes.
if current and #current == 8 then
  local refused = true
  -- Little-endian, so the most significant byte is the last one, and comparing from the back is
  -- a numeric comparison of non-negative i64. A negative timestamp would compare as its two's
  -- complement, and no source produces one.
  for i = 8, 1, -1 do
    local mine = string.byte(ARGV[2], i)
    local theirs = string.byte(current, i)
    if mine ~= theirs then
      refused = mine < theirs
      break
    end
  end
  -- Every byte equal leaves `refused` set, so a repeat of the same write is refused rather
  -- than rewriting bytes the store already holds.
  if refused then
    return 0
  end
end

local at = 4
for _ = 1, count do
  local name = ARGV[at + 1]
  local value = ARGV[at + 2]
  local expires_at = ARGV[at + 3]
  at = at + 3
  if mode == 2 and expires_at ~= '' then
    redis.call('HSETEX', key, 'EXAT', expires_at, 'FIELDS', 1, name, value)
  else
    -- A plain HSET clears whatever expiry the server holds for that field, which is the
    -- behaviour `queue_write` documents and the memory store mirrors.
    redis.call('HSET', key, name, value)
    if mode == 1 and expires_at ~= '' then
      -- `HEXPIREAT key unix-time-seconds [NX|XX|GT|LT] FIELDS numfields field`, so the
      -- timestamp sits between the key and `FIELDS`. The same order `queue_write` uses.
      redis.call('HEXPIREAT', key, expires_at, 'FIELDS', 1, name)
    end
  end
end

return 1
"#;

/// Queue the commands that write one batch.
///
/// The fields are grouped by their expiry, because `HSETEX` and `HEXPIREAT` each take one
/// time for every field they list. A batch normally holds one expiring field, the view's
/// value, and one field that never expires, its freshness, so a view with a TTL costs one more
/// command than one without. Both commands are queued before the pipeline is flushed, so the
/// pair costs one round trip rather than two.
///
/// The two are not atomic. `redis::pipe()` batches the commands into one write; it does not wrap
/// them in `MULTI`/`EXEC`, so a reader on another connection can see the value in the window
/// between the two. The value's bytes are correct either way and the read-time TTL check is what
/// decides whether it is served, so what the window costs is a field briefly stored without the
/// expiry meant to reclaim it.
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
            //
            // A plain `HSET` also clears any TTL the server holds for that field, the same way
            // `SET` clears a key's TTL without `KEEPTTL`, which is why `HSETEX` has that option
            // and `HSET` does not. That is the behaviour to match rather than a detail to work
            // around: a view that drops its `ttl_days` stops asking for an expiry, this rewrite
            // clears the old one, and its values stop expiring server-side. `MemoryStore` drops
            // its recorded expiry here for the same reason, so a test sees what the server
            // would, and `rewriting_a_field_without_an_expiry_clears_the_servers_expiry` in
            // `tests/valkey_integration.rs` pins it against a real server.
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

    async fn scan_entity_keys(&self, project: &str, exclude: &[u8]) -> Result<Vec<Vec<u8>>> {
        let mut connection = self.connection.clone();
        let pattern = format!("{}*", crate::online::glob_escape(project));
        let mut keys = Vec::new();
        let mut cursor = 0u64;
        loop {
            // `SCAN` with a bounded `COUNT`, never `KEYS`: a walk that blocks the server for its
            // whole duration is not something a refresh may do to a serving instance. The count
            // is a hint, so the reply's length is not the bound; the loop ends when the server
            // returns cursor 0.
            let (next, batch): (u64, Vec<Vec<u8>>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(SCAN_COUNT)
                .query_async(&mut connection)
                .await?;
            for key in batch {
                if key.as_slice() != exclude {
                    keys.push(key);
                }
            }
            cursor = next;
            if cursor == 0 {
                return Ok(keys);
            }
        }
    }
}

//! Feather: an opinionated feature store.
//!
//! This crate is everything that is not Python. The validating definition models
//! live on the Python side (Pydantic v2) and are handed across as JSON, so the
//! types here mirror them rather than duplicating their validation.
//!
//! The encoding layer ([`key`], [`value`]) is always available. The engines are
//! behind features because bundled DuckDB takes minutes to build:
//!
//! - `offline`: DuckDB point-in-time joins
//! - `valkey`: the real Valkey online store
//!
//! Run `cargo test --all-features` to exercise everything.

pub mod definitions;
pub mod error;
pub mod key;
pub mod online;
pub mod value;

pub use definitions::{DType, Definitions, Entity, FeatureService, FeatureView, Field, FileSource};
pub use error::{Error, Result};
pub use key::{
    MAX_COMPONENT_LEN, decode_entity_key, encode_entity_key, entity_hash_key, freshness_field,
    parse_entity_hash_key, value_field,
};
#[cfg(feature = "valkey")]
pub use online::valkey::ValkeyStore;
pub use online::{
    EntityRequest, Missing, OnlineStore, ReadRequest, ViewRequest, ViewValues, WriteBatch,
    read_entities,
};
pub use value::{EncodedBatch, SchemaTag, arrow_type, decode_batch, encode_batch};

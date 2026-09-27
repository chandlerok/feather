//! Feather: an opinionated feature store.
//!
//! This crate is everything that is not a language binding. The binding layer
//! offers authoring ergonomics and the earliest possible errors, but the contract
//! is here: definition and settings types are validated on ingest, so a binding
//! that skips its own checks still cannot hand over something the engines cannot
//! represent, and a second binding has nothing new to implement.
//!
//! The encoding layer ([`key`], [`value`]) and the settings loader are always
//! available. The engines are behind features because bundled DuckDB takes minutes
//! to build:
//!
//! - `offline`: DuckDB point-in-time joins and materialization
//! - `valkey`: the real Valkey online store
//!
//! Run `cargo test --all-features` to exercise everything.

pub mod definitions;
pub mod error;
pub mod key;
#[cfg(feature = "offline")]
pub mod materialize;
#[cfg(feature = "offline")]
pub mod offline;
pub mod online;
pub mod settings;
pub mod value;

pub use definitions::{
    DType, Definitions, Entity, FeatureService, FeatureView, Field, FileFormat, Source, SupportTier,
};
pub use error::{Error, Result};
pub use key::{
    MAX_COMPONENT_LEN, decode_entity_key, encode_entity_key, entity_hash_key, entity_key_component,
    freshness_field, is_entity_hash_key, parse_entity_hash_key, value_field, views_registry_key,
};
#[cfg(feature = "offline")]
pub use materialize::{MaterializeReport, ViewRefresh, materialize};
#[cfg(feature = "offline")]
pub use offline::{Engine as OfflineEngine, JoinOptions, Limits, OnMissing};
#[cfg(feature = "valkey")]
pub use online::valkey::ValkeyStore;
pub use online::{
    EntityRequest, Missing, OnlineStore, ProjectScan, ReadRequest, ViewRequest, ViewValues,
    WriteBatch, WrittenField, read_entities,
};
pub use settings::{Connection, Secret, Settings, Valkey, load_settings, parse_settings};
pub use value::{EncodedBatch, SchemaTag, arrow_type, decode_batch, encode_batch};

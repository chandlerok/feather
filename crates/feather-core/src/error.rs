#[cfg(feature = "offline")]
use std::fmt;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("entity `{entity}` has no key components")]
    EmptyEntityKey { entity: String },

    #[error("entity `{entity}` component {index} is {len} bytes, over the {max}-byte limit")]
    EntityKeyComponentTooLong {
        entity: String,
        index: usize,
        len: usize,
        max: usize,
    },

    #[error("malformed encoded entity key: {reason}")]
    MalformedEntityKey { reason: String },

    #[error("an entity key value is null, and a null addresses no entity")]
    NullEntityKey,

    #[error("an entity key of type {dtype} cannot be encoded; a key is an integer or a string")]
    UnsupportedKeyType { dtype: String },

    #[error("value is truncated: need {need} bytes, have {have}")]
    TruncatedValue { need: usize, have: usize },

    #[error("field `{field}` value is {len} bytes, over the {max}-byte limit")]
    ValueTooLong {
        field: String,
        len: usize,
        max: usize,
    },

    #[error("malformed value for field `{field}`: {reason}")]
    MalformedValue { field: String, reason: String },

    #[error(
        "value schema tag {found:#010x} does not match the expected {expected:#010x}; \
         treating the value as missing"
    )]
    SchemaTagMismatch { found: u32, expected: u32 },

    #[error("view `{view}` has {fields} fields but {columns} columns were supplied")]
    ColumnCountMismatch {
        view: String,
        fields: usize,
        columns: usize,
    },

    #[error("field `{name}` is declared {declared} but the column is {actual}")]
    ColumnTypeMismatch {
        name: String,
        declared: String,
        actual: String,
    },

    #[error("row index {row} is out of range for a column of length {len}")]
    RowOutOfRange { row: usize, len: usize },

    #[error("field `{name}` is not part of view `{view}`")]
    UnknownFeature { view: String, name: String },

    #[error("view `{view}` declares an unsupported number of entities: {count}")]
    UnsupportedEntityCount { view: String, count: usize },

    #[error("view `{view}` {reason}")]
    MalformedView { view: String, reason: String },

    #[error("malformed definitions: {reason}")]
    MalformedDefinitions { reason: String },

    /// A source naming a format Feather has no reader for.
    ///
    /// Reported against the view and the path rather than as a deserialization failure,
    /// because a definition that is wrong here is wrong about one source rather than
    /// about the file, and a name is what tells the two apart.
    #[error(
        "view `{view}` reads source `{path}` in format `{format}`, which Feather does not \
         read; the formats are {known}"
    )]
    UnknownSourceFormat {
        view: String,
        path: String,
        format: String,
        known: String,
    },

    #[error(
        "one request mixed entities `{first}` and `{second}`; all views in a request must \
         share the same entity"
    )]
    MixedEntities { first: String, second: String },

    #[error("malformed feature reference `{reference}`, expected `view:feature`")]
    MalformedFeatureReference { reference: String },

    #[error("no feature view named `{0}`")]
    UnknownView(String),

    #[error("view `{view}` names connection `{connection}`, which is not configured")]
    UnknownConnection { view: String, connection: String },

    #[error(
        "view `{view}` reads a `{source_kind}` source through connection `{connection}`, \
         which is a `{connection_kind}` connection"
    )]
    SourceConnectionKind {
        view: String,
        source_kind: String,
        connection: String,
        connection_kind: String,
    },

    #[error("no feature service named `{0}`")]
    UnknownService(String),

    #[error("could not read `{path}`: {source}")]
    SettingsIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("malformed settings: {reason}")]
    MalformedSettings { reason: String },

    #[error("{field} must not be empty")]
    EmptySettingsField { field: String },

    #[error("{key}: environment variable {variable} is not set")]
    MissingEnvironmentVariable { key: String, variable: String },

    #[error("{key}: environment variable {variable} is not valid UTF-8")]
    NonUnicodeEnvironmentVariable { key: String, variable: String },

    #[error("{key}: ${{{reference}}} is not a valid environment variable name")]
    MalformedEnvironmentReference { key: String, reference: String },

    #[error("arrow: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[cfg(feature = "offline")]
    #[error("duckdb: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[cfg(feature = "offline")]
    #[error("could not create the spill directory `{path}`: {source}")]
    SpillDirectory {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[cfg(feature = "offline")]
    #[error("{location} has no column `{column}`")]
    MissingColumn { location: String, column: String },

    #[cfg(feature = "offline")]
    #[error(
        "unsupported {role} type `{dtype}`; an entity key is an integer or a string, and a timestamp is integer microseconds or microsecond precision"
    )]
    UnsupportedOfflineType { role: String, dtype: String },

    #[cfg(feature = "offline")]
    #[error(
        "the entity frame's `{label}` is {label_type} and the source's `{source_column}` is {source_type}, \
         which cannot be compared without coercing one of them"
    )]
    OfflineTypeMismatch {
        label: String,
        label_type: String,
        source_column: String,
        source_type: String,
    },

    #[cfg(feature = "offline")]
    #[error(
        "view `{view}` has {rows} rows for `{key_column}` = `{key}` at timestamp {timestamp}, and \
         declares no created_timestamp_field, so which one wins would depend on the query plan"
    )]
    AmbiguousTimestamp {
        view: String,
        key_column: String,
        key: String,
        timestamp: i64,
        rows: i64,
    },

    #[cfg(feature = "offline")]
    #[error("could not read `{path}`: {source}")]
    Source {
        path: String,
        #[source]
        source: duckdb::Error,
    },

    #[cfg(feature = "offline")]
    #[error("view `{view}` failed while its source was scanned: {reason}")]
    StreamInterrupted { view: String, reason: String },

    /// A source that could not be read, attributed to the view that reads it.
    ///
    /// The scanner's own text is carried in the message rather than replaced, since it is
    /// the only description of what went wrong; naming the view and the source around it is
    /// what makes one failing Postgres view distinguishable from another.
    #[cfg(feature = "offline")]
    #[error("view `{view}` could not read {location}: {reason}")]
    UnreadableSource {
        view: String,
        location: String,
        reason: String,
    },

    /// An extension that could not be installed or loaded.
    ///
    /// The tier travels with it because that is the whole reason a non-Primary format
    /// is opt-in: an operator reading this needs to know they are on a best-effort or
    /// third-party path, and the extension's own message names neither the view nor the
    /// format. It travels only for the format's own reader, because `httpfs` reads no
    /// format and has no tier, and an `s3://` Parquet source that cannot reach the
    /// extension repository must not be told that `httpfs` reads Parquet.
    #[cfg(feature = "offline")]
    #[error("view `{view}` could not load the `{extension}` extension{reader}: {source}")]
    ExtensionUnavailable {
        view: String,
        extension: &'static str,
        reader: ExtensionReader,
        #[source]
        source: duckdb::Error,
    },

    #[cfg(feature = "valkey")]
    #[error("valkey: {0}")]
    Valkey(#[from] redis::RedisError),
}

/// What a failed extension load says about the format the extension reads.
///
/// Empty for an extension that is not a format's reader, which is the whole point: the
/// same [`Error::ExtensionUnavailable`] covers `httpfs` and a format's own extension,
/// and only the second has a format and a tier to name.
#[cfg(feature = "offline")]
#[derive(Debug)]
pub struct ExtensionReader(Option<(&'static str, &'static str)>);

#[cfg(feature = "offline")]
impl fmt::Display for ExtensionReader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((format, tier)) = self.0 {
            write!(formatter, ", which reads {format} and is {tier} tier")?;
        }
        Ok(())
    }
}

#[cfg(feature = "offline")]
impl Error {
    /// A `duckdb` failure installing or loading `extension` for `view`.
    ///
    /// `reader` is the format's `(name, tier)` when `extension` is that format's own
    /// reader, and `None` for anything else, so the message attributes a format and a
    /// tier only to the extension that has them.
    pub fn extension_unavailable(
        view: String,
        extension: &'static str,
        reader: Option<(&'static str, &'static str)>,
        source: duckdb::Error,
    ) -> Self {
        Self::ExtensionUnavailable {
            view,
            extension,
            reader: ExtensionReader(reader),
            source,
        }
    }
}

#[cfg(all(test, feature = "offline"))]
mod tests {
    use super::ExtensionReader;

    #[test]
    fn a_format_readers_failure_clause_names_the_format_and_the_tier() {
        // The whole reason the clause exists: an operator reading a load failure for a
        // non-Primary format has to be told they are on a third-party path. Rendered here
        // rather than only through a real failed install, because provoking one needs an
        // extension repository this test must not depend on.
        assert_eq!(
            ExtensionReader(Some(("vortex", "third-party maintained"))).to_string(),
            ", which reads vortex and is third-party maintained tier"
        );
    }

    #[test]
    fn a_filesystems_failure_clause_is_nothing_at_all() {
        // `httpfs` reads no format and is no tier, so a clause naming one is not a
        // smaller claim, it is the wrong claim. The message reads
        // "could not load the `httpfs` extension: <duckdb's own text>" instead.
        assert_eq!(ExtensionReader(None).to_string(), "");
    }
}

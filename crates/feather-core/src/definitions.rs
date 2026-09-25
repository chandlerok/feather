//! Definition types: the language-neutral contract for what a view is.
//!
//! A binding's definition layer offers authoring ergonomics and reports a mistake
//! as early as that language can, but the contract is here. These types validate on
//! ingest, so a binding that skips its own checks still cannot hand over something
//! the engines cannot represent, and a second binding has nothing new to implement.
//!
//! Validation covers what an engine depends on: exactly one entity per view, a source
//! that carries the values its own kind needs, the field names a view declares, and
//! every service reference resolving to a view and a field that both exist.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::settings::Connection;

/// The storage type of a feature.
///
/// Deliberately a closed set. Every variant is either fixed-width, which the
/// value encoding writes with no per-value prefix, or UTF-8, which is the only
/// variable-width case v1 supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Int64,
    Float64,
    Boolean,
    Utf8,
    /// Microseconds since the Unix epoch, stored as an `i64`.
    TimestampMicros,
}

impl DType {
    /// Bytes this type occupies in the fixed-width section, or `None` if it is
    /// variable-width and lives in the tail.
    pub const fn fixed_width(self) -> Option<usize> {
        match self {
            DType::Int64 | DType::Float64 | DType::TimestampMicros => Some(8),
            DType::Boolean => Some(1),
            DType::Utf8 => None,
        }
    }

    pub const fn is_fixed_width(self) -> bool {
        self.fixed_width().is_some()
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            DType::Int64 => "int64",
            DType::Float64 => "float64",
            DType::Boolean => "boolean",
            DType::Utf8 => "utf8",
            DType::TimestampMicros => "timestamp_micros",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub dtype: DType,
}

impl Field {
    pub fn new(name: impl Into<String>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            dtype,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub join_key: String,
}

impl Entity {
    pub fn new(name: impl Into<String>, join_key: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            join_key: join_key.into(),
        }
    }
}

/// Where a view's features are read from.
///
/// Discriminated on `type` in the wire form, so each kind carries only its own keys
/// and a key lifted from another kind is rejected rather than ignored. A new kind is
/// added here when a reader exists for it, which is why there are two rather than a
/// path with a scheme.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Source {
    /// A Parquet file, local or reached through a URI scheme such as `s3://`.
    File { path: String },
    /// A table in a Postgres database.
    ///
    /// The connection is named rather than carried, because credentials belong in
    /// `feather.toml` and a definition module is committed.
    Postgres {
        connection: String,
        schema: String,
        table: String,
    },
}

impl Source {
    /// A local or remote Parquet file.
    pub fn file(path: impl Into<String>) -> Self {
        Self::File { path: path.into() }
    }

    /// A table read from the named Postgres connection.
    pub fn postgres(
        connection: impl Into<String>,
        schema: impl Into<String>,
        table: impl Into<String>,
    ) -> Self {
        Self::Postgres {
            connection: connection.into(),
            schema: schema.into(),
            table: table.into(),
        }
    }

    /// This kind's wire name, which is what `type` holds.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::File { .. } => "file",
            Self::Postgres { .. } => "postgres",
        }
    }

    /// The `[connections]` entry this source reads through, if it names one.
    pub fn connection_name(&self) -> Option<&str> {
        match self {
            Self::File { .. } => None,
            Self::Postgres { connection, .. } => Some(connection),
        }
    }

    /// How this source is named in an error about it.
    pub fn description(&self) -> String {
        match self {
            Self::File { path } => path.clone(),
            Self::Postgres {
                connection,
                schema,
                table,
            } => format!("{connection}.{schema}.{table}"),
        }
    }

    /// This kind's string fields, as `(key, value)` pairs.
    ///
    /// The keys are the wire names, so an error about one names the key the author
    /// wrote rather than the variant it selects.
    fn string_fields(&self) -> Vec<(&'static str, &str)> {
        match self {
            Self::File { path } => vec![("path", path)],
            Self::Postgres {
                connection,
                schema,
                table,
            } => vec![
                ("connection", connection),
                ("schema", schema),
                ("table", table),
            ],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureView {
    pub name: String,
    pub entities: Vec<Entity>,
    pub source: Source,
    pub features: Vec<Field>,
    /// Retention. `None` means no expiry.
    #[serde(default)]
    pub ttl_days: Option<u32>,
    #[serde(default)]
    pub timestamp_field: Option<String>,
    #[serde(default)]
    pub created_timestamp_field: Option<String>,
}

impl FeatureView {
    pub const DEFAULT_TIMESTAMP_FIELD: &'static str = "event_timestamp";

    pub fn timestamp_field(&self) -> &str {
        self.timestamp_field
            .as_deref()
            .unwrap_or(Self::DEFAULT_TIMESTAMP_FIELD)
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.features.iter().find(|f| f.name == name)
    }

    /// The single entity this view is keyed on.
    ///
    /// v1 requires exactly one. Composite keys are designed for in the key
    /// encoding but no multi-entity view is accepted yet, so this returns an
    /// error rather than silently using the first entity.
    pub fn entity(&self) -> Result<&Entity> {
        match self.entities.as_slice() {
            [one] => Ok(one),
            other => Err(Error::UnsupportedEntityCount {
                view: self.name.clone(),
                count: other.len(),
            }),
        }
    }
}

/// A named projection: a name and a list of `view:feature` references.
///
/// Deliberately not a registry object. It carries no entities, no version, and
/// no infrastructure, because entities come from the referenced views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureService {
    pub name: String,
    pub features: Vec<String>,
}

impl FeatureService {
    /// Parse the references into `(view, feature)` pairs.
    pub fn references(&self) -> Result<Vec<(&str, &str)>> {
        self.features
            .iter()
            .map(|r| {
                r.split_once(':')
                    .filter(|(v, f)| !v.is_empty() && !f.is_empty())
                    .ok_or_else(|| Error::MalformedFeatureReference {
                        reference: r.clone(),
                    })
            })
            .collect()
    }
}

/// Everything the engines need to know about a project's feature definitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Definitions {
    pub project: String,
    #[serde(default)]
    pub views: Vec<FeatureView>,
    #[serde(default)]
    pub services: Vec<FeatureService>,
}

impl Definitions {
    pub fn new(project: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            views: Vec::new(),
            services: Vec::new(),
        }
    }

    pub fn view(&self, name: &str) -> Result<&FeatureView> {
        self.views
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| Error::UnknownView(name.to_owned()))
    }

    pub fn service(&self, name: &str) -> Result<&FeatureService> {
        self.services
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| Error::UnknownService(name.to_owned()))
    }

    /// Reject a definition the engines cannot work from.
    ///
    /// This is the authority, not a second opinion. A binding may reject the same
    /// thing earlier and in its own words, but a value that reaches here is checked
    /// again, and references that cross objects are checked only here.
    pub fn validate(&self) -> Result<()> {
        for view in &self.views {
            // Surfaces the multi-entity error at validation time rather than at
            // the first read.
            view.entity()?;
            // Emptiness, not whitespace, because that is the rule the Python mirror states
            // with its non-empty string type and this module is the authority. Left empty, a
            // source field reaches the scanner as an opaque read failure.
            for (key, value) in view.source.string_fields() {
                if value.is_empty() {
                    return Err(Error::MalformedView {
                        view: view.name.clone(),
                        reason: format!("declares an empty `{key}` in its source"),
                    });
                }
            }
            if view.ttl_days == Some(0) {
                return Err(Error::MalformedView {
                    view: view.name.clone(),
                    reason: "declares ttl_days of 0, which can never expire a value".to_owned(),
                });
            }
            if view.timestamp_field.is_some()
                && view.timestamp_field == view.created_timestamp_field
            {
                return Err(Error::MalformedView {
                    view: view.name.clone(),
                    reason: format!(
                        "uses `{}` as both the event timestamp and the created timestamp",
                        view.timestamp_field(),
                    ),
                });
            }
            let mut seen = HashSet::new();
            for field in &view.features {
                if field.name.is_empty() {
                    return Err(Error::MalformedView {
                        view: view.name.clone(),
                        reason: "declares an unnamed field".to_owned(),
                    });
                }
                if !seen.insert(field.name.as_str()) {
                    return Err(Error::MalformedView {
                        view: view.name.clone(),
                        reason: format!("declares field `{}` twice", field.name),
                    });
                }
            }
        }
        for service in &self.services {
            for (view_name, feature_name) in service.references()? {
                let view = self.view(view_name)?;
                if view.field(feature_name).is_none() {
                    return Err(Error::UnknownFeature {
                        view: view_name.to_owned(),
                        name: feature_name.to_owned(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Reject a view whose source names a connection the project cannot use.
    ///
    /// This is where definitions and settings first meet, so an unknown connection
    /// name or a connection of the wrong kind fails here, at startup, rather than at
    /// the first read. The join resolves the same source again as a backstop.
    ///
    /// Args:
    ///     connections: The project's `[connections]`, keyed by the name a source refers to.
    ///
    /// Returns:
    ///     `Ok(())` when every view's source resolves.
    ///
    /// Raises:
    ///     [`Error::UnknownConnection`] if a source names a name that is not configured.
    ///     [`Error::SourceConnectionKind`] if the named connection is another kind.
    pub fn validate_sources(&self, connections: &BTreeMap<String, Connection>) -> Result<()> {
        for view in &self.views {
            let Some(name) = view.source.connection_name() else {
                continue;
            };
            let configured = connections
                .get(name)
                .ok_or_else(|| Error::UnknownConnection {
                    view: view.name.clone(),
                    connection: name.to_owned(),
                })?;
            if configured.kind() != view.source.kind() {
                return Err(Error::SourceConnectionKind {
                    view: view.name.clone(),
                    source_kind: view.source.kind().to_owned(),
                    connection: name.to_owned(),
                    connection_kind: configured.kind().to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Deserialize and validate in one step.
    ///
    /// A binding hands over JSON and cannot get a `Definitions` back without it
    /// having been checked, which is what makes this module the authority rather
    /// than one opinion among several.
    pub fn from_json(json: &str) -> Result<Self> {
        let definitions: Self =
            serde_json::from_str(json).map_err(|source| Error::MalformedDefinitions {
                reason: source.to_string(),
            })?;
        definitions.validate()?;
        Ok(definitions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One valid view, with `mutate` applied before it goes into the project.
    fn project(mutate: impl FnOnce(&mut FeatureView)) -> Definitions {
        let mut view = FeatureView {
            name: "user_clicks".to_owned(),
            entities: vec![Entity::new("user_id", "user_id")],
            source: Source::file("data/user_stats.parquet"),
            features: vec![
                Field::new("click_count", DType::Int64),
                Field::new("purchase_count", DType::Int64),
            ],
            ttl_days: None,
            timestamp_field: None,
            created_timestamp_field: None,
        };
        mutate(&mut view);
        Definitions {
            project: "ad_recommendations".to_owned(),
            views: vec![view],
            services: Vec::new(),
        }
    }

    #[test]
    fn a_valid_view_passes() {
        project(|_| {}).validate().expect("valid");
    }

    /// A configured connection, built through the shape the settings loader produces.
    fn connection(json: &str) -> Connection {
        serde_json::from_str(json).expect("connection")
    }

    #[test]
    fn a_file_source_validates_with_no_connections() {
        // Which is what local mode looks like: the check is per source kind, so a project
        // of file sources needing credentials would be the bug.
        project(|_| {})
            .validate_sources(&BTreeMap::new())
            .expect("valid");
    }

    #[test]
    fn a_source_naming_an_unknown_connection_is_rejected() {
        let definitions = project(|v| {
            v.source = Source::postgres("pg_prod", "public", "user_stats");
        });

        let error = definitions
            .validate_sources(&BTreeMap::new())
            .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` names connection `pg_prod`, which is not configured"
        );
    }

    #[test]
    fn a_source_and_a_connection_of_different_kinds_are_rejected() {
        let definitions = project(|v| {
            v.source = Source::postgres("s3_lake", "public", "user_stats");
        });
        let connections = BTreeMap::from([(
            "s3_lake".to_owned(),
            connection(r#"{"type":"s3","region":"us-east-1","key_id":"k","secret":"s"}"#),
        )]);

        let error = definitions
            .validate_sources(&connections)
            .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` reads a `postgres` source through connection `s3_lake`, \
             which is a `s3` connection"
        );
    }

    #[test]
    fn an_unnamed_field_is_rejected() {
        let error = project(|v| v.features[1].name.clear())
            .validate()
            .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` declares an unnamed field"
        );
    }

    #[test]
    fn a_duplicate_field_is_rejected() {
        let error = project(|v| v.features[1].name = "click_count".to_owned())
            .validate()
            .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` declares field `click_count` twice"
        );
    }

    #[test]
    fn a_zero_ttl_is_rejected() {
        let error = project(|v| v.ttl_days = Some(0))
            .validate()
            .expect_err("must fail");

        assert!(error.to_string().contains("ttl_days of 0"), "{error}");
    }

    #[test]
    fn a_timestamp_collision_is_rejected() {
        let error = project(|v| {
            v.timestamp_field = Some("ts".to_owned());
            v.created_timestamp_field = Some("ts".to_owned());
        })
        .validate()
        .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` uses `ts` as both the event timestamp and the created timestamp"
        );
    }

    #[test]
    fn a_multi_entity_view_is_rejected() {
        let error = project(|v| v.entities.push(Entity::new("other", "other")))
            .validate()
            .expect_err("must fail");

        assert!(
            error.to_string().contains("unsupported number of entities"),
            "{error}"
        );
    }

    #[test]
    fn a_service_naming_an_unknown_view_is_rejected() {
        let mut definitions = project(|_| {});
        definitions.services.push(FeatureService {
            name: "ranking_v3".to_owned(),
            features: vec!["nope:click_count".to_owned()],
        });

        let error = definitions.validate().expect_err("must fail");

        assert!(
            error.to_string().contains("no feature view named `nope`"),
            "{error}"
        );
    }

    #[test]
    fn a_service_naming_an_unknown_feature_is_rejected() {
        let mut definitions = project(|_| {});
        definitions.services.push(FeatureService {
            name: "ranking_v3".to_owned(),
            features: vec!["user_clicks:click_cout".to_owned()],
        });

        let error = definitions.validate().expect_err("must fail");

        assert!(
            error
                .to_string()
                .contains("`click_cout` is not part of view"),
            "{error}"
        );
    }

    #[test]
    fn a_resolvable_service_passes() {
        let mut definitions = project(|_| {});
        definitions.services.push(FeatureService {
            name: "ranking_v3".to_owned(),
            features: vec!["user_clicks:click_count".to_owned()],
        });

        definitions.validate().expect("valid");
    }

    #[test]
    fn from_json_validates_what_it_reads() {
        let json = serde_json::to_string(&project(|_| {})).expect("serialize");
        Definitions::from_json(&json).expect("valid");

        // A field declared twice is only caught by `validate`, so reaching it
        // through `from_json` is what proves the two are wired together.
        let broken = project(|v| v.features[1].name = "click_count".to_owned());
        let json = serde_json::to_string(&broken).expect("serialize");
        let error = Definitions::from_json(&json).expect_err("must fail");

        assert!(
            error
                .to_string()
                .contains("declares field `click_count` twice"),
            "{error}"
        );
    }

    #[test]
    fn from_json_rejects_json_that_is_not_definitions() {
        let error = Definitions::from_json(r#"{"project": 1}"#).expect_err("must fail");

        assert!(
            error.to_string().contains("malformed definitions"),
            "{error}"
        );
    }
}

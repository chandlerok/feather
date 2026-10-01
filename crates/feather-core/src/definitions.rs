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

/// DuckDB's published support tier for the extension a file format needs.
///
/// A fact about the extension rather than about the project, which is why it is
/// declared on the format and lives in neither `feather.toml` nor a definition module:
/// the answer is the same for every project, so either file would only be somewhere
/// for a project to assert a tier that could be wrong. It is also the reason a format
/// is allowed to exist here at all without being the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportTier {
    /// Community supported.
    Primary,
    /// Best effort: still bugfixed and shipped with each release.
    Secondary,
    /// Maintained outside the DuckDB team, so it is rebuilt per release and an
    /// extension that lags the engine fails to load rather than degrading.
    ThirdParty,
}

impl SupportTier {
    /// This tier's name, as the error a failed load reports it.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "Primary",
            Self::Secondary => "Secondary",
            Self::ThirdParty => "third-party maintained",
        }
    }
}

/// The file format a `File` source is read as.
///
/// Closed rather than free, because each variant is a reader that exists and a tier
/// that says whether relying on it is safe. One source picks one, so a project can
/// read several formats side by side, and the default is the Primary one rather than
/// whichever was added most recently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FileFormat {
    /// DuckDB's `read_parquet`. Compiled into the engine, and the only Primary-tier
    /// file format DuckDB publishes.
    #[default]
    Parquet,
    /// DuckDB's `read_vortex`. Loadable rather than compiled in, third-party
    /// maintained, and pre-1.0, so a source naming it opts into an extension Feather
    /// neither ships nor builds against.
    Vortex,
}

impl FileFormat {
    /// Every format's wire name, in the order readers are added.
    const ALL: [Self; 2] = [Self::Parquet, Self::Vortex];

    /// This format's wire name, which is what a source's `format` holds.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Parquet => "parquet",
            Self::Vortex => "vortex",
        }
    }

    /// This format's support tier, which is what makes it safe to default to.
    pub const fn tier(self) -> SupportTier {
        match self {
            Self::Parquet => SupportTier::Primary,
            Self::Vortex => SupportTier::ThirdParty,
        }
    }

    /// A format from the name a source declares, or `None` if Feather has no reader for it.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|format| format.as_str() == name)
    }

    /// Every format's wire name, for an error that has to name them.
    pub fn names() -> impl Iterator<Item = &'static str> {
        Self::ALL.into_iter().map(|format| format.as_str())
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
    /// A file, local or reached through a URI scheme such as `s3://`.
    File {
        path: String,
        /// The format the file is read as. Absent means [`FileFormat::default`], so a
        /// project that only reads Parquet never writes the key.
        ///
        /// A `String` rather than a [`FileFormat`] because the value has to survive
        /// deserialization to be reported properly: `serde` cannot reach the sibling
        /// `path` from a field deserializer, and an unknown format has to name the
        /// source that declared it rather than fail as a bare "unknown variant".
        /// [`FeatureView::source_format`] is where it is resolved.
        #[serde(default)]
        format: Option<String>,
    },
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
    /// A local or remote file in the default format.
    pub fn file(path: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            format: None,
        }
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
            Self::File { path, .. } => path.clone(),
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
    /// wrote rather than the variant it selects. An absent optional contributes no
    /// pair: there is no value to be empty, and an absent `format` is the default
    /// rather than a declaration.
    fn string_fields(&self) -> Vec<(&'static str, &str)> {
        match self {
            Self::File { path, format } => {
                let mut fields = vec![("path", path.as_str())];
                if let Some(format) = format {
                    fields.push(("format", format.as_str()));
                }
                fields
            }
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

    /// The format this view's source is read as, or an error naming the source that
    /// declared one Feather has no reader for.
    ///
    /// Checked here, at load, because an unknown format is a mistake in the
    /// definition rather than a failure at the first read, and the source is named
    /// either way. The engine resolves the same format again as a backstop.
    pub fn source_format(&self) -> Result<FileFormat> {
        let Source::File { path, format } = &self.source else {
            // A Postgres table is a table scan rather than a file format, so there is
            // nothing for a format to decide and the default stands.
            return Ok(FileFormat::default());
        };
        let Some(declared) = format else {
            return Ok(FileFormat::default());
        };
        FileFormat::parse(declared).ok_or_else(|| Error::UnknownSourceFormat {
            view: self.name.clone(),
            path: path.clone(),
            format: declared.clone(),
            known: FileFormat::names().collect::<Vec<_>>().join(", "),
        })
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

/// One requested feature, split into the view it comes from and its name there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reference {
    pub view: String,
    pub feature: String,
}

/// Split `view:feature` references, rejecting the shapes an engine cannot use.
///
/// This is the one spelling of what a request may name, so every binding refuses
/// the same inputs with the same reasons. An empty list is refused because a read
/// with nothing to return is a caller's mistake, not an empty result; a repeated
/// reference is refused because the result's column order would carry it twice and
/// the two copies could not be told apart.
pub fn parse_references<S: AsRef<str>>(features: &[S]) -> Result<Vec<Reference>> {
    if features.is_empty() {
        return Err(Error::NoFeaturesRequested);
    }
    let mut references: Vec<Reference> = Vec::with_capacity(features.len());
    for reference in features {
        let reference = reference.as_ref();
        let (view, feature) = reference
            .split_once(':')
            .filter(|(view, feature)| !view.is_empty() && !feature.is_empty())
            .ok_or_else(|| Error::MalformedFeatureReference {
                reference: reference.to_owned(),
            })?;
        if references
            .iter()
            .any(|seen| seen.view == view && seen.feature == feature)
        {
            return Err(Error::DuplicateFeatureReference {
                reference: reference.to_owned(),
            });
        }
        references.push(Reference {
            view: view.to_owned(),
            feature: feature.to_owned(),
        });
    }
    Ok(references)
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
        // A key is `{project}:{entity}:{encoded}`, and the parser finds its two separators by
        // position, so a colon in either name produces a key that cannot be taken apart again.
        // Nothing would be deleted that should not be, because garbage collection checks a key's
        // shape before touching it, but that project's keys would be skipped silently and for
        // good. A name the key layout cannot carry is refused here rather than skipped later.
        if self.project.contains(':') {
            return Err(Error::MalformedDefinitions {
                reason: format!(
                    "project `{}` contains a colon, which is the key layout's separator",
                    self.project
                ),
            });
        }
        // A view name reaches the reference syntax as well as the registry and the field names,
        // and `view:feature` splits on the first colon, so a view named `a:b` would be
        // indistinguishable from view `a`'s field `b:count` and `a:b:count` would resolve to
        // whichever of the two the declarations happened to list first. A colon in a *feature*
        // name is safe, because everything after the first separator is the feature, so only the
        // view is refused here.
        let mut view_names = HashSet::new();
        for view in &self.views {
            if !view_names.insert(view.name.as_str()) {
                return Err(Error::MalformedDefinitions {
                    reason: format!("declares view `{}` twice", view.name),
                });
            }
            if view.name.contains(':') {
                return Err(Error::MalformedView {
                    view: view.name.clone(),
                    reason: "has a name containing a colon, which is the separator between a \
                             view and a feature in a reference"
                        .to_owned(),
                });
            }
            // Surfaces the multi-entity error at validation time rather than at
            // the first read.
            view.entity()?;
            for entity in &view.entities {
                if entity.name.contains(':') {
                    return Err(Error::MalformedView {
                        view: view.name.clone(),
                        reason: format!(
                            "declares entity `{}`, whose name contains a colon, which is the key \
                             layout's separator",
                            entity.name
                        ),
                    });
                }
            }
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
            // An unreadable format is refused at load rather than reaching a table
            // function that does not exist, and `source_format` is what names the view.
            view.source_format()?;
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

    /// One project whose view's source is `source`, and nothing else set.
    fn project_with_source(source: Source) -> Definitions {
        project(|view| view.source = source)
    }

    #[test]
    fn a_file_source_with_no_format_is_parquet() {
        let view = project_with_source(Source::file("data/user_stats.parquet"))
            .views
            .remove(0);

        assert_eq!(view.source_format().expect("format"), FileFormat::Parquet);
        // The default is the Primary format rather than a preference, which is the only
        // reason adding a format later cannot quietly change what an old project reads.
        assert_eq!(FileFormat::default(), FileFormat::Parquet);
    }

    #[test]
    fn a_file_source_naming_a_format_resolves_to_it() {
        let view = project_with_source(Source::File {
            path: "data/clicks.vortex".to_owned(),
            format: Some("vortex".to_owned()),
        })
        .views
        .remove(0);

        assert_eq!(view.source_format().expect("format"), FileFormat::Vortex);
    }

    #[test]
    fn a_source_read_as_several_formats_resolves_independently() {
        // One project, two file sources, two formats, so the choice is per source rather
        // than a setting somewhere that applies to all of them at once.
        let mut definitions = project_with_source(Source::file("data/user_stats.parquet"));
        let mut other = definitions.views[0].clone();
        other.name = "user_clicks_vortex".to_owned();
        other.source = Source::File {
            path: "s3://lake/clicks.vortex".to_owned(),
            format: Some("vortex".to_owned()),
        };
        definitions.views.push(other);
        definitions.validate().expect("valid");

        assert_eq!(
            definitions
                .view("user_clicks")
                .expect("view")
                .source_format()
                .expect("parquet"),
            FileFormat::Parquet
        );
        assert_eq!(
            definitions
                .view("user_clicks_vortex")
                .expect("view")
                .source_format()
                .expect("vortex"),
            FileFormat::Vortex
        );
    }

    #[test]
    fn an_unknown_format_is_rejected_at_load_naming_the_source() {
        let definitions = project_with_source(Source::File {
            path: "data/clicks.vortex".to_owned(),
            format: Some("vortx".to_owned()),
        });

        let error = definitions.validate().expect_err("must fail");

        // Naming the view, the path, and the value is the whole point: a bare "unknown
        // variant" from serde would say none of them, and a file that is really Parquet
        // read through the wrong reader fails much later and nowhere near the mistake.
        assert_eq!(
            error.to_string(),
            "view `user_clicks` reads source `data/clicks.vortex` in format `vortx`, which \
             Feather does not read; the formats are parquet, vortex"
        );
    }

    #[test]
    fn an_unknown_format_naming_an_unlisted_one_says_what_exists() {
        // A user reaching for a table format is the likely mistake, so the message has to
        // say which formats do exist rather than only that this one does not.
        let definitions = project_with_source(Source::File {
            path: "data/events".to_owned(),
            format: Some("iceberg".to_owned()),
        });

        let error = definitions.validate().expect_err("must fail");

        assert!(error.to_string().contains("parquet, vortex"), "{error}");
    }

    #[test]
    fn a_format_is_a_tier_duckdb_publishes() {
        // The tier is what makes a format safe to default to, and it is a fact about the
        // extension rather than something a project declares, so it is checked here
        // against the table the decision document carries.
        assert_eq!(FileFormat::Parquet.tier(), SupportTier::Primary);
        assert_eq!(FileFormat::Vortex.tier(), SupportTier::ThirdParty);
        assert_eq!(FileFormat::Parquet.tier().as_str(), "Primary");
        assert_eq!(FileFormat::Vortex.tier().as_str(), "third-party maintained");
        assert_eq!(
            FileFormat::names().collect::<Vec<_>>(),
            ["parquet", "vortex"]
        );
        assert_eq!(FileFormat::parse("parquet"), Some(FileFormat::Parquet));
        assert_eq!(FileFormat::parse("PARQUET"), None);
    }

    #[test]
    fn an_absent_format_key_means_the_default() {
        // What a binding that leaves an unset optional out produces, which is the other
        // spelling of the same thing the Python layer writes as a null.
        let definitions = Definitions::from_json(
            r#"{"project":"ads","views":[{"name":"user_clicks",
                "entities":[{"name":"user_id","join_key":"user_id"}],
                "source":{"type":"file","path":"data/user_stats.parquet"},
                "features":[{"name":"click_count","dtype":"int64"}]}]}"#,
        )
        .expect("valid");

        assert_eq!(
            definitions
                .view("user_clicks")
                .expect("view")
                .source_format()
                .expect("parquet"),
            FileFormat::Parquet
        );
    }

    #[test]
    fn an_empty_format_is_refused_when_the_json_is_read() {
        // The core is the authority on the rule that no source field is empty, and
        // Python enforces it with `NonEmptyStr`. A binding that does not would
        // otherwise get the reader's error instead of the sibling one, and the sibling
        // one is what names the field the author wrote. `from_json` validates, so the
        // binding is refused here rather than handed a `Definitions` it has to
        // remember to check; reaching this through `from_json` is what pins that.
        let error = Definitions::from_json(
            r#"{"project":"ads","views":[{"name":"user_clicks",
                "entities":[{"name":"user_id","join_key":"user_id"}],
                "source":{"type":"file","path":"data/user_stats.parquet","format":""},
                "features":[{"name":"click_count","dtype":"int64"}]}]}"#,
        )
        .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "view `user_clicks` declares an empty `format` in its source"
        );
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
    #[test]
    fn a_project_name_the_key_layout_cannot_carry_is_refused() {
        let mut definitions = project(|_| {});
        definitions.project = "ad:s".to_owned();
        let error = definitions.validate().expect_err("must fail");
        assert!(error.to_string().contains("ad:s"), "{error}");
    }

    #[test]
    fn an_entity_name_the_key_layout_cannot_carry_is_refused() {
        let error = project(|view| {
            view.entities = vec![Entity::new("user:id", "user_id")];
        })
        .validate()
        .expect_err("must fail");
        assert!(error.to_string().contains("user:id"), "{error}");
    }

    #[test]
    fn a_view_name_that_would_re_split_a_reference_is_refused() {
        // `view:feature` splits on the first colon, so a view named `clicks:raw` would be
        // indistinguishable from view `clicks`'s field `raw:count`, and the reference
        // `clicks:raw:count` would resolve to whichever of the two the declarations listed
        // first. Nothing else in the key layout sees the colon, which is why this refusal is a
        // separate one from the project and entity names'.
        let error = project(|view| view.name = "clicks:raw".to_owned())
            .validate()
            .expect_err("must fail");

        assert!(error.to_string().contains("clicks:raw"), "{error}");
        assert!(error.to_string().contains("colon"), "{error}");
    }

    #[test]
    fn a_feature_name_containing_a_colon_is_accepted() {
        // Everything after the first separator is the feature, so a colon there cannot be
        // mistaken for the view's end, and the reference still resolves.
        let mut definitions = project(|view| view.features[0].name = "count:raw".to_owned());
        definitions.services.push(FeatureService {
            name: "ranking_v3".to_owned(),
            features: vec!["user_clicks:count:raw".to_owned()],
        });

        definitions.validate().expect("valid");
    }

    #[test]
    fn a_view_declared_twice_is_refused() {
        // Two views sharing a name make the registry write last-write-wins, so it can hold a
        // timestamp older than the refresh that ran, and the two views write the same fields.
        let mut definitions = project(|_| {});
        let mut duplicate = definitions.views[0].clone();
        duplicate.features = vec![Field::new("other", DType::Int64)];
        definitions.views.push(duplicate);

        let error = definitions.validate().expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "malformed definitions: declares view `user_clicks` twice"
        );
    }

    #[test]
    fn a_well_formed_reference_splits_into_its_view_and_feature() {
        let references = parse_references(&["user_clicks:click_count"]).expect("valid");

        assert_eq!(references.len(), 1);
        assert_eq!(references[0].view, "user_clicks");
        assert_eq!(references[0].feature, "click_count");
    }

    #[test]
    fn a_reference_keeps_its_request_order() {
        let references = parse_references(&["b:two", "a:one", "b:one"]).expect("valid");

        let pairs: Vec<(&str, &str)> = references
            .iter()
            .map(|r| (r.view.as_str(), r.feature.as_str()))
            .collect();
        assert_eq!(pairs, [("b", "two"), ("a", "one"), ("b", "one")]);
    }

    #[test]
    fn an_empty_reference_list_is_refused() {
        let error = parse_references::<String>(&[]).expect_err("must fail");

        assert_eq!(error.to_string(), "features must name at least one feature");
    }

    #[test]
    fn a_reference_without_a_colon_is_refused() {
        let error = parse_references(&["click_count"]).expect_err("must fail");

        assert!(
            matches!(error, Error::MalformedFeatureReference { .. }),
            "{error}"
        );
    }

    #[test]
    fn a_reference_with_an_empty_side_is_refused() {
        // Both halves are refused, because an empty view or feature names nothing.
        for reference in [":count", "clicks:", ":"] {
            let error = parse_references(&[reference]).expect_err("must fail");
            assert!(
                matches!(error, Error::MalformedFeatureReference { .. }),
                "{reference} produced {error}"
            );
        }
    }

    #[test]
    fn a_repeated_reference_is_refused() {
        let error = parse_references(&["clicks:count", "clicks:count"]).expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "the reference `clicks:count` is requested twice"
        );
    }

    #[test]
    fn the_same_feature_under_two_views_is_not_a_duplicate() {
        // The namespace is what makes the pair unique, so these are two different
        // columns and both are kept.
        let references = parse_references(&["a:count", "b:count"]).expect("valid");

        assert_eq!(references.len(), 2);
    }
}

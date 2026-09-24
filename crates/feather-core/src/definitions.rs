//! Definition types, mirroring the Pydantic v2 models on the Python side.
//!
//! Python is the validating authority: it parses `feature_store.yaml` and the
//! definition modules, and hands the result across the boundary as JSON. These
//! types therefore deserialize that JSON and do not re-validate it. The one
//! exception is [`Definitions::validate`], which checks the cross-references
//! Pydantic cannot see (a feature service naming a view that does not exist).

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

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

/// A file-backed offline source.
///
/// v1 has exactly one source kind. The object-storage and warehouse tiers are
/// the same type with a different path scheme, so no enum is needed yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSource {
    pub path: String,
}

impl FileSource {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureView {
    pub name: String,
    pub entities: Vec<Entity>,
    pub source: FileSource,
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

    /// Check the references Pydantic cannot see, because they cross objects.
    pub fn validate(&self) -> Result<()> {
        for view in &self.views {
            // Surfaces the multi-entity error at validation time rather than at
            // the first read.
            view.entity()?;
            for field in &view.features {
                if field.name.is_empty() {
                    return Err(Error::MalformedEntityKey {
                        reason: format!("view `{}` declares an unnamed field", view.name),
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
}

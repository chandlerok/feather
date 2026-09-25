//! Project settings, read from `feather.toml`.
//!
//! This is the validating authority for the settings file. A language binding may
//! parse the file early to report a mistake sooner, but nothing here trusts that:
//! every binding eventually hands its values to this module, and this module is
//! what decides whether they are valid.
//!
//! The file holds what the deployment is made of, and nothing about what the data
//! means. Offline sources are declared on the views they feed, in a definition
//! module, so there is no offline store here. What is here is the credentials a
//! source must not carry, since a definition module is committed.
//!
//! `${VAR}` is resolved from the process environment while reading, and a variable
//! that is unset is an error rather than an empty string. That is the only way the
//! environment reaches a value, so there is one mechanism rather than two that can
//! disagree about precedence.

use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroU64;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// How a variable is looked up while resolving `${VAR}`.
///
/// Taking the lookup as an argument rather than reading the environment directly
/// keeps resolution a pure function, so it is tested without mutating the process
/// environment, which is unsafe in edition 2024 and racy between tests.
type Lookup<'a> = &'a dyn Fn(&str) -> std::result::Result<String, std::env::VarError>;

/// A credential, kept out of `Debug` output.
///
/// A `String` here would print in full the first time anything debug-logs a
/// `Settings`, which is the most common way a credential escapes.
#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    /// The value, revealed at the one place it is used.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(..)")
    }
}

/// A named credential set a source can read from, discriminated on `type`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Connection {
    /// A Snowflake account.
    Snowflake {
        account: String,
        warehouse: String,
        username: String,
        password: Secret,
    },
    /// An S3 or S3-compatible object store.
    ///
    /// `key_id` is not a secret and stays readable, so a misconfigured key is
    /// diagnosable from a log.
    S3 {
        region: String,
        key_id: String,
        secret: Secret,
        /// A non-AWS endpoint, such as `localhost:9000` for a local S3-compatible server.
        #[serde(default)]
        endpoint: Option<String>,
        /// Whether the endpoint speaks HTTPS. DuckDB's own default is true.
        #[serde(default = "enabled_by_default")]
        use_ssl: bool,
        /// `path`, `vhost`, or absent to let the engine decide.
        ///
        /// Absent means path style when an `endpoint` is set, because an S3-compatible
        /// server reached at a host and port is addressed path-style, and DuckDB's own
        /// default for S3 is vhost.
        #[serde(default)]
        url_style: Option<String>,
    },
}

impl Connection {
    /// Reject a credential set missing a value the engine needs.
    ///
    /// Serde cannot express a non-empty string, so the empty cases are checked
    /// here rather than being left to fail at the first read.
    fn validate(&self, name: &str) -> Result<()> {
        match self {
            Self::Snowflake {
                account,
                warehouse,
                username,
                ..
            } => {
                check_present(account, &format!("connections.{name}.account"))?;
                check_present(warehouse, &format!("connections.{name}.warehouse"))?;
                check_present(username, &format!("connections.{name}.username"))?;
            }
            Self::S3 {
                region,
                key_id,
                url_style,
                ..
            } => {
                check_present(region, &format!("connections.{name}.region"))?;
                check_present(key_id, &format!("connections.{name}.key_id"))?;
                if let Some(style) = url_style {
                    // Checked here rather than at the secret, so a typo names the key it is in
                    // instead of surfacing as a DuckDB parser error.
                    if style != "path" && style != "vhost" {
                        return Err(Error::MalformedSettings {
                            reason: format!(
                                "connections.{name}.url_style must be `path` or `vhost`, not `{style}`"
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// The in-process cache in front of Valkey.
///
/// Configured whether or not Valkey is, because local mode has an L1 and no L2.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct L1Cache {
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// `None` leaves the capacity to the engine's default.
    #[serde(default)]
    pub max_capacity_mb: Option<NonZeroU64>,
    /// `None` leaves the interval to the engine's default.
    ///
    /// Always on, not conditional on push invalidation being available: it bounds
    /// staleness when an invalidation message is missed.
    #[serde(default)]
    pub fallback_ttl_seconds: Option<NonZeroU64>,
}

fn enabled_by_default() -> bool {
    true
}

/// The Valkey connection.
///
/// Valkey is the only L2, so no `type` discriminates it. `tls` has no default: a
/// security flag guessed wrong is worse than one an operator had to write down.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Valkey {
    pub endpoint: String,
    pub tls: bool,
    /// Needs Valkey 9.0+ or Redis 8.0+ for native field expiration. `false` is the
    /// conservative default, since older servers fall back to read-time TTL checks.
    #[serde(default)]
    pub field_expiration: bool,
}

/// A validated `feather.toml`.
///
/// Unknown keys are rejected, so a typo is a load error rather than a silently
/// ignored setting. `valkey` and `l1_cache` absent is local mode: an in-process
/// DuckDB over local files, and the engine's own cache defaults.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub project: String,
    pub definitions: Vec<String>,
    /// Declared ahead of its first consumer. No source kind references a
    /// connection yet, since v1 declares a source as a path, so these entries are
    /// inert until one does.
    #[serde(default)]
    pub connections: BTreeMap<String, Connection>,
    #[serde(default)]
    pub valkey: Option<Valkey>,
    #[serde(default)]
    pub l1_cache: Option<L1Cache>,
}

impl Settings {
    /// Reject settings the engines cannot work from.
    ///
    /// Returns:
    ///     `Ok(())` when every value the engines read is present and usable.
    ///
    /// Raises:
    ///     [`Error::EmptySettingsField`] for a present-but-empty value.
    ///     [`Error::MalformedSettings`] for an empty list where at least one entry
    ///     is required.
    pub fn validate(&self) -> Result<()> {
        check_present(&self.project, "project")?;
        if self.definitions.is_empty() {
            return Err(Error::MalformedSettings {
                reason: "definitions must list at least one module".to_owned(),
            });
        }
        for (index, module) in self.definitions.iter().enumerate() {
            check_present(module, &format!("definitions.{index}"))?;
        }
        for (name, connection) in &self.connections {
            check_present(name, "a connection name")?;
            connection.validate(name)?;
        }
        if let Some(valkey) = &self.valkey {
            check_present(&valkey.endpoint, "valkey.endpoint")?;
        }
        Ok(())
    }
}

/// Reject an empty or whitespace-only value.
///
/// Args:
///     value: The value as written.
///     field: The dotted key path, used in the error.
///
/// Returns:
///     `Ok(())` when the value carries something.
fn check_present(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::EmptySettingsField {
            field: field.to_owned(),
        });
    }
    Ok(())
}

/// Read and validate a `feather.toml`.
///
/// Args:
///     path: The file to read.
///
/// Returns:
///     The validated settings.
///
/// Raises:
///     [`Error::SettingsIo`] if the file cannot be read.
///     [`Error::MalformedSettings`] if it is not valid TOML, or does not match the
///     schema, including an unknown key.
///     [`Error::MissingEnvironmentVariable`] or
///     [`Error::MalformedEnvironmentReference`] for a `${VAR}` that cannot be
///     resolved.
///     [`Error::EmptySettingsField`] for a present-but-empty value.
pub fn load_settings(path: impl AsRef<Path>) -> Result<Settings> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| Error::SettingsIo {
        path: path.display().to_string(),
        source,
    })?;
    parse_settings(&text)
}

/// Parse and validate `feather.toml` text.
///
/// Args:
///     text: The file contents.
///
/// Returns:
///     The validated settings.
///
/// Raises:
///     As [`load_settings`], minus the read failure.
pub fn parse_settings(text: &str) -> Result<Settings> {
    parse_settings_with(text, &|name| std::env::var(name))
}

/// Parse and validate, resolving `${VAR}` through `env`.
///
/// Args:
///     text: The file contents.
///     env: The variable lookup. Taken as an argument so resolution is testable
///         without writing to the process environment.
///
/// Returns:
///     The validated settings.
///
/// Raises:
///     As [`parse_settings`].
fn parse_settings_with(text: &str, env: Lookup<'_>) -> Result<Settings> {
    let mut value: toml::Value =
        toml::from_str(text).map_err(|source| Error::MalformedSettings {
            reason: source.to_string(),
        })?;
    interpolate(&mut value, &mut Vec::new(), env)?;
    let settings = Settings::deserialize(value).map_err(|source: toml::de::Error| {
        Error::MalformedSettings {
            reason: source.to_string(),
        }
    })?;
    settings.validate()?;
    Ok(settings)
}

/// Resolve `${VAR}` in every string, in place.
///
/// Args:
///     value: A node of the parsed document.
///     path: The keys walked to reach it, used to say where a bad reference was.
///     env: The variable lookup.
///
/// Returns:
///     `Ok(())` once every node has been visited.
fn interpolate(value: &mut toml::Value, path: &mut Vec<String>, env: Lookup<'_>) -> Result<()> {
    match value {
        toml::Value::String(text) => {
            let resolved = interpolate_text(text, &key_of(path), env)?;
            *text = resolved;
        }
        toml::Value::Table(table) => {
            for (name, item) in table.iter_mut() {
                path.push(name.clone());
                interpolate(item, path, env)?;
                path.pop();
            }
        }
        toml::Value::Array(items) => {
            for (index, item) in items.iter_mut().enumerate() {
                path.push(index.to_string());
                interpolate(item, path, env)?;
                path.pop();
            }
        }
        _ => {}
    }
    Ok(())
}

/// The dotted key path for an error message.
///
/// Args:
///     path: The keys walked so far.
///
/// Returns:
///     The joined path, or the file name at the root.
fn key_of(path: &[String]) -> String {
    if path.is_empty() {
        "feather.toml".to_owned()
    } else {
        path.join(".")
    }
}

/// Resolve every `${VAR}` in one string.
///
/// Args:
///     text: The value as written.
///     key: The dotted key path, used in the error.
///     env: The variable lookup.
///
/// Returns:
///     The value with every reference replaced.
///
/// Raises:
///     [`Error::MalformedEnvironmentReference`] if the braces do not hold a usable
///     variable name.
///     [`Error::MissingEnvironmentVariable`] if the variable is not set.
///     [`Error::NonUnicodeEnvironmentVariable`] if it is set to non-UTF-8 bytes.
fn interpolate_text(text: &str, key: &str, env: Lookup<'_>) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let opened = &rest[start + 2..];
        // No closing brace, so this is not a reference. Keep it as written rather
        // than guessing where it was meant to end.
        let Some(end) = opened.find('}') else {
            out.push_str(&rest[start..]);
            return Ok(out);
        };
        let name = &opened[..end];
        if !is_environment_name(name) {
            return Err(Error::MalformedEnvironmentReference {
                key: key.to_owned(),
                reference: name.to_owned(),
            });
        }
        match env(name) {
            Ok(value) => out.push_str(&value),
            Err(std::env::VarError::NotPresent) => {
                return Err(Error::MissingEnvironmentVariable {
                    key: key.to_owned(),
                    variable: name.to_owned(),
                });
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(Error::NonUnicodeEnvironmentVariable {
                    key: key.to_owned(),
                    variable: name.to_owned(),
                });
            }
        }
        rest = &opened[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Whether `name` can be an environment variable name.
///
/// Args:
///     name: The text between the braces.
///
/// Returns:
///     `true` for a leading letter or underscore followed by word characters.
fn is_environment_name(name: &str) -> bool {
    let mut characters = name.chars();
    match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' => {}
        _ => return false,
    }
    characters.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCAL: &str = r#"
project = "ad_recommendations"
definitions = ["definitions/user_clicks.py"]
"#;

    const FULL: &str = r#"
project = "ad_recommendations"
definitions = ["definitions/user_clicks.py", "definitions/user_stats.py"]

[connections.snowflake_prod]
type = "snowflake"
account = "xy12345.us-east-1"
warehouse = "PROD_WH"
username = "fs_runner"
password = "${SNOWFLAKE_PASSWORD}"

[connections.s3_lake]
type = "s3"
region = "us-east-1"
key_id = "AKIAEXAMPLE"
secret = "shhh"

[valkey]
endpoint = "valkey-cluster.internal.svc:6379"
tls = true
field_expiration = true

[l1_cache]
enabled = true
max_capacity_mb = 2048
fallback_ttl_seconds = 30
"#;

    /// A lookup that resolves only `SNOWFLAKE_PASSWORD`.
    fn with_password(name: &str) -> std::result::Result<String, std::env::VarError> {
        match name {
            "SNOWFLAKE_PASSWORD" => Ok("hunter2".to_owned()),
            _ => Err(std::env::VarError::NotPresent),
        }
    }

    fn resolve(text: &str) -> Result<String> {
        interpolate_text(text, "a.b", &|name| match name {
            "SET" => Ok("value".to_owned()),
            _ => Err(std::env::VarError::NotPresent),
        })
    }

    #[test]
    fn local_mode_needs_only_a_project() {
        let settings = parse_settings(LOCAL).expect("valid");

        assert_eq!(settings.project, "ad_recommendations");
        assert_eq!(settings.definitions, ["definitions/user_clicks.py"]);
        assert!(settings.connections.is_empty());
        assert!(settings.valkey.is_none());
        assert!(settings.l1_cache.is_none());
    }

    #[test]
    fn a_full_file_parses() {
        let settings = parse_settings_with(FULL, &with_password).expect("valid");

        match &settings.connections["snowflake_prod"] {
            Connection::Snowflake {
                account, password, ..
            } => {
                assert_eq!(account, "xy12345.us-east-1");
                assert_eq!(password.expose(), "hunter2");
            }
            other => panic!("expected snowflake, got {other:?}"),
        }
        match &settings.connections["s3_lake"] {
            Connection::S3 { key_id, secret, .. } => {
                assert_eq!(key_id, "AKIAEXAMPLE");
                assert_eq!(secret.expose(), "shhh");
            }
            other => panic!("expected s3, got {other:?}"),
        }

        let valkey = settings.valkey.expect("valkey");
        assert_eq!(valkey.endpoint, "valkey-cluster.internal.svc:6379");
        assert!(valkey.tls);
        assert!(valkey.field_expiration);

        let l1 = settings.l1_cache.expect("l1");
        assert!(l1.enabled);
        assert_eq!(l1.max_capacity_mb.map(NonZeroU64::get), Some(2048));
        assert_eq!(l1.fallback_ttl_seconds.map(NonZeroU64::get), Some(30));
    }

    #[test]
    fn the_l1_cache_is_configured_without_valkey() {
        let settings = parse_settings(&format!("{LOCAL}\n[l1_cache]\nmax_capacity_mb = 512\n"))
            .expect("valid");

        assert!(settings.valkey.is_none());
        assert_eq!(
            settings
                .l1_cache
                .expect("l1")
                .max_capacity_mb
                .map(NonZeroU64::get),
            Some(512)
        );
    }

    #[test]
    fn the_l1_cache_defaults_to_enabled() {
        let settings = parse_settings(&format!("{LOCAL}\n[l1_cache]\n")).expect("valid");

        assert!(settings.l1_cache.expect("l1").enabled);
    }

    #[test]
    fn secrets_are_not_leaked_by_debug() {
        let settings = parse_settings_with(FULL, &with_password).expect("valid");
        let rendered = format!("{settings:?}");

        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(!rendered.contains("shhh"), "{rendered}");
        assert!(rendered.contains("AKIAEXAMPLE"), "{rendered}");
    }

    #[test]
    fn a_reference_is_resolved() {
        assert_eq!(
            resolve("before ${SET} after").expect("resolved"),
            "before value after"
        );
        assert_eq!(resolve("${SET}").expect("resolved"), "value");
        assert_eq!(resolve("no reference").expect("resolved"), "no reference");
    }

    #[test]
    fn an_unset_variable_is_an_error_naming_the_key_and_the_variable() {
        let error = resolve("${UNSET_FOR_THE_TEST}").expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "a.b: environment variable UNSET_FOR_THE_TEST is not set"
        );
    }

    #[test]
    fn a_malformed_reference_is_an_error() {
        let error = resolve("${1BAD}").expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "a.b: ${1BAD} is not a valid environment variable name"
        );
    }

    #[test]
    fn an_unclosed_brace_is_left_alone() {
        assert_eq!(resolve("${UNCLOSED").expect("resolved"), "${UNCLOSED");
    }

    #[test]
    fn a_non_utf8_variable_is_an_error() {
        let error = interpolate_text("${BAD}", "a.b", &|_| {
            Err(std::env::VarError::NotUnicode("x".into()))
        })
        .expect_err("must fail");

        assert_eq!(
            error.to_string(),
            "a.b: environment variable BAD is not valid UTF-8"
        );
    }

    #[test]
    fn a_nested_key_path_appears_in_the_error() {
        let text =
            format!("{LOCAL}\n[valkey]\nendpoint = \"${{UNSET_FOR_THE_TEST}}\"\ntls = true\n");
        let error = parse_settings_with(&text, &|_| Err(std::env::VarError::NotPresent))
            .expect_err("must fail");

        assert!(error.to_string().contains("valkey.endpoint"), "{error}");
    }

    #[test]
    fn an_unknown_top_level_key_is_rejected() {
        let error = parse_settings(&format!("{LOCAL}\nnope = 1\n")).expect_err("must fail");

        assert!(error.to_string().contains("nope"), "{error}");
    }

    #[test]
    fn an_offline_store_table_is_rejected() {
        let text = format!("{LOCAL}\n[offline_store]\ntype = \"snowflake\"\n");

        assert!(parse_settings(&text).is_err());
    }

    #[test]
    fn an_unknown_key_in_valkey_is_rejected() {
        let text = format!("{LOCAL}\n[valkey]\nendpoint = \"e\"\ntls = true\nnope = 1\n");
        let error = parse_settings(&text).expect_err("must fail");

        assert!(error.to_string().contains("nope"), "{error}");
    }

    #[test]
    fn an_unknown_key_in_a_connection_is_rejected() {
        let text = format!(
            "{LOCAL}\n[connections.s3_lake]\ntype = \"s3\"\nregion = \"us-east-1\"\n\
             key_id = \"k\"\nsecret = \"s\"\nnope = 1\n"
        );

        assert!(parse_settings(&text).is_err());
    }

    #[test]
    fn a_connection_kind_must_be_known() {
        let text = format!("{LOCAL}\n[connections.mystery]\ntype = \"bigquery\"\n");

        assert!(parse_settings(&text).is_err());
    }

    #[test]
    fn a_connection_missing_a_key_is_rejected() {
        let text =
            format!("{LOCAL}\n[connections.s3_lake]\ntype = \"s3\"\nregion = \"us-east-1\"\n");

        assert!(parse_settings(&text).is_err());
    }

    #[test]
    fn an_empty_required_field_is_rejected() {
        let text = format!("{LOCAL}\n[valkey]\nendpoint = \"\"\ntls = true\n");
        let error = parse_settings(&text).expect_err("must fail");

        assert_eq!(error.to_string(), "valkey.endpoint must not be empty");
    }

    #[test]
    fn a_zero_capacity_is_rejected() {
        let text = format!("{LOCAL}\n[l1_cache]\nmax_capacity_mb = 0\n");

        assert!(parse_settings(&text).is_err());
    }

    #[test]
    fn a_missing_project_is_rejected() {
        assert!(parse_settings("definitions = [\"a.py\"]\n").is_err());
    }

    #[test]
    fn an_empty_definitions_list_is_rejected() {
        let text = "project = \"p\"\ndefinitions = []\n";
        let error = parse_settings(text).expect_err("must fail");

        assert!(error.to_string().contains("at least one module"), "{error}");
    }

    #[test]
    fn a_missing_file_reports_the_path() {
        let error = load_settings("/nonexistent/feather.toml").expect_err("must fail");

        assert!(
            error.to_string().contains("/nonexistent/feather.toml"),
            "{error}"
        );
    }
}

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
    /// A Postgres database.
    ///
    /// The values are libpq's, not a URL, so one connection string is assembled from
    /// them in [`Connection::conninfo`] rather than a URL being parsed.
    Postgres {
        host: String,
        #[serde(default = "default_postgres_port")]
        port: u16,
        database: String,
        user: String,
        password: Secret,
        /// Any of libpq's `sslmode` values: `disable`, `allow`, `prefer`, `require`,
        /// `verify-ca`, or `verify-full`. Absent leaves libpq's own default, which already
        /// prefers TLS when the server offers it.
        #[serde(default)]
        ssl_mode: Option<String>,
    },
}

impl Connection {
    /// Reject a credential set the engine cannot work from.
    ///
    /// Serde cannot express a non-empty string or a closed set of strings, so the empty and
    /// unlisted cases are checked here rather than being left to fail at the first read.
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
            Self::Postgres {
                host,
                database,
                user,
                ssl_mode,
                ..
            } => {
                check_present(host, &format!("connections.{name}.host"))?;
                check_present(database, &format!("connections.{name}.database"))?;
                check_present(user, &format!("connections.{name}.user"))?;
                if let Some(mode) = ssl_mode {
                    // Checked here rather than at the connection, so a typo names the key it is
                    // in instead of surfacing as a libpq error at the first read. The set is
                    // libpq's own, not the `disable`/`require`/`verify-*` subset a locked
                    // down deployment writes: `allow` and `prefer` are real values, and
                    // refusing them would reject a working connection. The comparison is exact
                    // because libpq's is; see `POSTGRES_SSL_MODES`.
                    if !POSTGRES_SSL_MODES.contains(&mode.as_str()) {
                        return Err(Error::MalformedSettings {
                            reason: format!(
                                "connections.{name}.ssl_mode must be one of {}, not `{mode}`",
                                POSTGRES_SSL_MODES.join(", ")
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// This kind's wire name, which is what `type` holds.
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Snowflake { .. } => "snowflake",
            Self::S3 { .. } => "s3",
            Self::Postgres { .. } => "postgres",
        }
    }

    /// The libpq connection string for a Postgres connection.
    ///
    /// `None` for another kind, which connects some other way. Every value is quoted
    /// by [`conninfo_value`], which is what makes a password holding a space, a quote,
    /// or a backslash survive both this layer and the SQL literal it is embedded in.
    pub fn conninfo(&self) -> Option<String> {
        let Self::Postgres {
            host,
            port,
            database,
            user,
            password,
            ssl_mode,
        } = self
        else {
            return None;
        };
        let mut parts = vec![
            format!("host={}", conninfo_value(host)),
            format!("port={}", conninfo_value(&port.to_string())),
            format!("dbname={}", conninfo_value(database)),
            format!("user={}", conninfo_value(user)),
            format!("password={}", conninfo_value(password.expose())),
        ];
        if let Some(mode) = ssl_mode {
            parts.push(format!("sslmode={}", conninfo_value(mode)));
        }
        Some(parts.join(" "))
    }
}

/// The default for a boolean setting whose default is `true`.
///
/// Named so `serde` can reference it, since `#[serde(default)]` on a `bool` gives `false`.
fn enabled_by_default() -> bool {
    true
}

/// The `sslmode` values libpq accepts, in the one case it accepts them in.
///
/// The whole set, not the subset a locked down deployment writes: `allow` and `prefer` are
/// real values, so a check against the subset would reject a connection that works. Lower case
/// is not a convention here, it is the only spelling that works: libpq compares the value
/// exactly, so `sslmode=REQUIRE` is refused while `require` reaches the TLS negotiation.
/// Verified against the compose server rather than read out of libpq's documentation, because
/// the question was what this libpq accepts:
///
/// ```text
/// $ psql "host=127.0.0.1 port=5433 dbname=feathertest user=feathertest sslmode=REQUIRE" -c 'select 1'
/// psql: error: invalid sslmode value: "REQUIRE"
/// ```
///
/// So the check below stays exact. Accepting upper case and normalising it would make Feather
/// more permissive than the client it hands the value to, which is the same defect in the other
/// direction as a check that refuses a value the client would have taken.
const POSTGRES_SSL_MODES: [&str; 6] = [
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
];

/// The port a Postgres connection uses when it does not say.
fn default_postgres_port() -> u16 {
    5432
}

/// One value in a libpq connection string.
///
/// Every value is single-quoted, and inside the quotes libpq reads a backslash as an
/// escape, so `\` becomes `\\` and `'` becomes `\'`. This is not SQL quoting and must not
/// be replaced by it: doubling the quote (`fea''ther`) is the SQL spelling, and libpq
/// answers it with "missing = after" rather than a connection. The two layers are both
/// required, because the conninfo is itself embedded in a SQL literal one level up:
/// this escapes a value for libpq, and `quote_literal` in the engine escapes the
/// resulting string for DuckDB. Removing either one breaks a password with a quote.
fn conninfo_value(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// The Valkey connection.
///
/// Valkey is the only _shared_ store that can be configured, so no `type` discriminates it.
/// `tls` has no default: a security flag guessed wrong is worse than one an operator had to
/// write down.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Valkey {
    pub endpoint: String,
    pub tls: bool,
    /// Needs a server that can expire a hash field: `HEXPIREAT` landed in Redis 7.4 and
    /// `HSETEX` in Redis 8.0, and Valkey carries the family from 9.0. `false` is the
    /// conservative default, since a server without either falls back to the read-time TTL
    /// check, which is authoritative either way.
    #[serde(default)]
    pub field_expiration: bool,
}

/// A validated `feather.toml`.
///
/// Unknown keys are rejected, so a typo is a load error rather than a silently
/// ignored setting. An absent `valkey` is local mode: an in-process DuckDB over
/// local files, served from the in-process online store.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub project: String,
    pub definitions: Vec<String>,
    /// Declared ahead of its first consumer. A Postgres source names an entry in
    /// here, and the definitions-and-settings seam rejects a name that is not
    /// configured, so a credential has a home that is not a committed definition
    /// module.
    #[serde(default)]
    pub connections: BTreeMap<String, Connection>,
    #[serde(default)]
    pub valkey: Option<Valkey>,
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
///         without writing to the process environment, which is unsafe in edition
///         2024 and racy between tests. Public for the same reason: a caller that
///         owns the environment parses through here rather than around the loader.
///
/// Returns:
///     The validated settings.
///
/// Raises:
///     As [`parse_settings`].
pub fn parse_settings_with(text: &str, env: Lookup<'_>) -> Result<Settings> {
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

[connections.pg_prod]
type = "postgres"
host = "pg.internal.svc"
database = "warehouse"
user = "fs_runner"
password = "${POSTGRES_PASSWORD}"
ssl_mode = "verify-full"

[valkey]
endpoint = "valkey-cluster.internal.svc:6379"
tls = true
field_expiration = true
"#;

    /// A lookup that resolves only `SNOWFLAKE_PASSWORD` and `POSTGRES_PASSWORD`.
    fn with_password(name: &str) -> std::result::Result<String, std::env::VarError> {
        match name {
            "SNOWFLAKE_PASSWORD" => Ok("hunter2".to_owned()),
            "POSTGRES_PASSWORD" => Ok("pgsecret".to_owned()),
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
        match &settings.connections["pg_prod"] {
            Connection::Postgres {
                host,
                port,
                database,
                user,
                password,
                ssl_mode,
            } => {
                assert_eq!(host, "pg.internal.svc");
                // Absent in the file, so the default is what a psql default would be.
                assert_eq!(*port, 5432);
                assert_eq!(database, "warehouse");
                assert_eq!(user, "fs_runner");
                assert_eq!(password.expose(), "pgsecret");
                assert_eq!(ssl_mode.as_deref(), Some("verify-full"));
            }
            other => panic!("expected postgres, got {other:?}"),
        }

        let valkey = settings.valkey.expect("valkey");
        assert_eq!(valkey.endpoint, "valkey-cluster.internal.svc:6379");
        assert!(valkey.tls);
        assert!(valkey.field_expiration);
    }

    #[test]
    fn secrets_are_not_leaked_by_debug() {
        let settings = parse_settings_with(FULL, &with_password).expect("valid");
        let rendered = format!("{settings:?}");

        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(!rendered.contains("shhh"), "{rendered}");
        assert!(!rendered.contains("pgsecret"), "{rendered}");
        // The key id is not a secret and stays readable, so a misconfigured key is
        // diagnosable from a log.
        assert!(rendered.contains("AKIAEXAMPLE"), "{rendered}");
    }

    /// A Postgres connection whose password is the given value.
    fn postgres(password: &str) -> Connection {
        serde_json::from_value(serde_json::json!({
            "type": "postgres",
            "host": "127.0.0.1",
            "port": 5433,
            "database": "feathertest",
            "user": "feathertest",
            "password": password,
        }))
        .expect("connection")
    }

    #[test]
    fn a_postgres_connection_assembles_a_libpq_conninfo() {
        let conninfo = postgres("feathertest").conninfo().expect("conninfo");

        assert_eq!(
            conninfo,
            "host='127.0.0.1' port='5433' dbname='feathertest' user='feathertest' \
             password='feathertest'"
        );
        // No ssl_mode, so libpq's own default applies rather than a guess.
        assert!(!conninfo.contains("sslmode"), "{conninfo}");
    }

    #[test]
    fn a_conninfo_value_is_quoted_and_backslash_escaped_for_libpq() {
        // The trap: libpq reads `\` as an escape inside its quotes, so a backslash is
        // doubled and a quote is backslashed. SQL's doubled quote is the wrong spelling
        // here and libpq refuses it, so `''` must never appear in a conninfo value.
        let cases = [
            ("fea ther", "'fea ther'"),
            ("fea'ther", r"'fea\'ther'"),
            (r"fea\ther", r"'fea\\ther'"),
        ];

        for (password, expected) in cases {
            let conninfo = postgres(password).conninfo().expect("conninfo");
            assert!(
                conninfo.contains(&format!("password={expected}")),
                "password {password:?} became {conninfo}"
            );
            assert!(!conninfo.contains("''"), "{password:?} became {conninfo}");
        }
    }

    #[test]
    fn a_postgres_ssl_mode_is_carried_into_the_conninfo() {
        let configured: Connection = serde_json::from_value(serde_json::json!({
            "type": "postgres",
            "host": "h",
            "database": "d",
            "user": "u",
            "password": "p",
            "ssl_mode": "require",
        }))
        .expect("connection");

        assert!(
            configured
                .conninfo()
                .expect("conninfo")
                .ends_with("sslmode='require'"),
            "{:?}",
            configured.conninfo()
        );
    }

    #[test]
    fn another_connection_kind_has_no_conninfo() {
        let configured = connection_kind("s3");

        assert!(configured.conninfo().is_none());
        assert_eq!(configured.kind(), "s3");
    }

    /// A minimal valid connection of the named kind.
    fn connection_kind(kind: &str) -> Connection {
        let json = match kind {
            "s3" => serde_json::json!({
                "type": "s3", "region": "us-east-1", "key_id": "k", "secret": "s"
            }),
            "snowflake" => serde_json::json!({
                "type": "snowflake", "account": "a", "warehouse": "w", "username": "u", "password": "p"
            }),
            other => panic!("unknown kind {other}"),
        };
        serde_json::from_value(json).expect("connection")
    }

    #[test]
    fn an_empty_postgres_host_is_rejected_naming_the_key() {
        let text = format!(
            "{LOCAL}\n[connections.pg]\ntype = \"postgres\"\nhost = \"\"\n\
             database = \"d\"\nuser = \"u\"\npassword = \"p\"\n"
        );
        let error = parse_settings(&text).expect_err("must fail");

        assert_eq!(error.to_string(), "connections.pg.host must not be empty");
    }

    /// A Postgres connection entry whose `ssl_mode` is the given value.
    fn postgres_with_ssl_mode(mode: &str) -> String {
        format!(
            "{LOCAL}\n[connections.pg]\ntype = \"postgres\"\nhost = \"h\"\ndatabase = \"d\"\n\
             user = \"u\"\npassword = \"p\"\nssl_mode = \"{mode}\"\n"
        )
    }

    #[test]
    fn an_unknown_postgres_ssl_mode_is_rejected_naming_the_key() {
        // Named here rather than at the connection, for the same reason `url_style` is:
        // otherwise a typo reaches libpq and comes back as a connection failure.
        let error = parse_settings(&postgres_with_ssl_mode("verfiy-full")).expect_err("must fail");

        assert!(
            error.to_string().contains("connections.pg.ssl_mode"),
            "{error}"
        );
        assert!(error.to_string().contains("verfiy-full"), "{error}");
    }

    #[test]
    fn every_ssl_mode_libpq_accepts_is_accepted() {
        // The comment above the set used to list only the strict subset, so this pins the
        // whole set: `allow` and `prefer` are what a check copied from that list would
        // wrongly refuse.
        for mode in POSTGRES_SSL_MODES {
            parse_settings(&postgres_with_ssl_mode(mode))
                .unwrap_or_else(|error| panic!("{mode}: {error}"));
        }
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

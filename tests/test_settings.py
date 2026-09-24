"""The settings binding: what crosses from the core, and how failures surface.

The core owns parsing, ``${VAR}`` resolution, and the schema, so those rules are
pinned by the Rust tests beside `crates/feather-core/src/settings.rs`. What is
pinned here is the binding contract: that delegation reaches the core, and that a
failure arrives as the exception a Python caller would catch.
"""

from pathlib import Path

import pydantic
import pytest

from feather import FeatherSettings, load_settings
from feather.settings import DEFAULT_PATH, S3Connection, SnowflakeConnection

LOCAL = """
project = "ad_recommendations"
definitions = ["definitions/user_clicks.py"]
"""

FULL = """
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
"""


def write(tmp_path: Path, body: str) -> Path:
    """Write a TOML body to a temp file.

    Args:
        tmp_path: The pytest temp directory.
        body: The TOML text.

    Returns:
        The path written.
    """
    path = tmp_path / "feather.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_local_mode_needs_nothing_but_a_project(tmp_path: Path) -> None:
    settings = load_settings(write(tmp_path, LOCAL))

    assert settings.project == "ad_recommendations"
    assert settings.definitions == ["definitions/user_clicks.py"]
    assert settings.connections == {}
    assert settings.valkey is None
    assert settings.l1_cache is None


def test_a_full_file_arrives_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")

    settings = load_settings(write(tmp_path, FULL))

    snowflake = settings.connections["snowflake_prod"]
    assert isinstance(snowflake, SnowflakeConnection)
    assert snowflake.account == "xy12345.us-east-1"
    assert snowflake.password.get_secret_value() == "hunter2"

    s3 = settings.connections["s3_lake"]
    assert isinstance(s3, S3Connection)
    assert s3.key_id == "AKIAEXAMPLE"
    assert s3.secret.get_secret_value() == "shhh"

    assert settings.valkey is not None
    assert settings.valkey.endpoint == "valkey-cluster.internal.svc:6379"
    assert settings.valkey.tls is True
    assert settings.valkey.field_expiration is True

    assert settings.l1_cache is not None
    assert settings.l1_cache.max_capacity_mb == 2048
    assert settings.l1_cache.fallback_ttl_seconds == 30


def test_secrets_are_not_leaked_by_repr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "hunter2")

    rendered = repr(load_settings(write(tmp_path, FULL)))

    assert "hunter2" not in rendered
    assert "shhh" not in rendered
    assert "AKIAEXAMPLE" in rendered


def test_the_l1_cache_is_configured_without_valkey(tmp_path: Path) -> None:
    settings = load_settings(write(tmp_path, LOCAL + "\n[l1_cache]\nmax_capacity_mb = 512\n"))

    assert settings.valkey is None
    assert settings.l1_cache is not None
    assert settings.l1_cache.max_capacity_mb == 512
    assert settings.l1_cache.enabled is True


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.toml"):
        load_settings(tmp_path / "absent.toml")


def test_an_unset_variable_raises_and_names_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)

    with pytest.raises(ValueError) as raised:
        load_settings(write(tmp_path, FULL))

    message = str(raised.value)
    assert "SNOWFLAKE_PASSWORD" in message
    assert "connections.snowflake_prod.password" in message


def test_an_unknown_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_settings(write(tmp_path, LOCAL + "\nnope = 1\n"))


def test_an_offline_store_table_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_settings(write(tmp_path, LOCAL + '\n[offline_store]\ntype = "snowflake"\n'))


def test_a_connection_kind_must_be_known(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_settings(write(tmp_path, LOCAL + '\n[connections.mystery]\ntype = "bigquery"\n'))


def test_the_default_path_is_feather_toml() -> None:
    assert DEFAULT_PATH.name == "feather.toml"


def test_settings_can_be_built_without_a_file() -> None:
    settings = FeatherSettings(project="p", definitions=["definitions/a.py"])

    assert settings.valkey is None


def test_the_model_rejects_a_shape_the_core_would_not_produce() -> None:
    with pytest.raises(pydantic.ValidationError):
        FeatherSettings.model_validate({"project": "p", "definitions": []})

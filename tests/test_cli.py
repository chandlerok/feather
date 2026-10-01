"""The `feather` command: the tree it writes, and the path it has to make runnable.

The generated project is the README's example, so the two are pinned against each
other rather than against a copy of the expected text. A change to either that
breaks the other fails here instead of in a newcomer's terminal.
"""

import contextlib
import importlib.util
import os
import re
import resource
import signal
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from feather import FeatureStore, load_settings
from feather.cli import DEFINITION, SETTINGS_NAME, demo, init, main, project_name, refresh
from feather.definitions import view_fields

README = Path(__file__).resolve().parent.parent / "README.md"
"""The document the generated project is written to match."""

_CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
"""A fenced Python block, which is how the README writes every example."""

DAY = 86_400_000_000
"""Microseconds in a day, which the demo data is laid out in."""

SECOND_DEFINITION = """\
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

other_entity = Entity(name="user_session", join_key="user_id")
other_source = FileSource(path="data/user_stats.parquet")


@feature_view(
    name="user_totals",
    entity=other_entity,
    source=other_source,
    ttl_days=30,
)
class UserTotals(FeatureView):
    purchase_count = Field(Int64)
"""
"""A second view over the same source, so a refresh has something to leave out.

The entity name and the field are both deliberately different from the
generated module's. A second view redeclaring `user_id` as an entity, or
`click_count` as a field, would be testing the store's duplicate handling rather
than the refresh's view selection, and would make these two tests fail for a
reason that has nothing to do with what they claim to pin.
"""

_BLOCK_EXTENSION = """
import sys


class Blocked:
    def find_spec(self, name, path=None, target=None):
        if name == "feather._core":
            raise ImportError("feather._core is blocked for this test")
        return None


sys.meta_path.insert(0, Blocked())
from feather.cli import main

raise SystemExit(main(["--help"]))
"""
"""A fresh interpreter in which the extension cannot be imported, however it is installed."""


def readme_example(marker: str) -> str:
    """Return the first Python block the README puts after `marker`.

    Args:
        marker: The sentence introducing the block.

    Returns:
        The block's body, without the fence.

    Raises:
        AssertionError: If the marker is gone, or the block that followed it is not
            there any more, which is itself the drift this exists to catch.
    """
    text = README.read_text(encoding="utf-8")
    _, _, rest = text.partition(marker)
    assert rest, f"the README no longer introduces an example with {marker!r}"
    block = _CODE_BLOCK.search(rest)
    assert block is not None, f"the README has no Python block after {marker!r}"
    return block.group(1)


@contextlib.contextmanager
def inside(project: Path) -> Iterator[None]:
    """Run the body with `project` as the working directory.

    A source is declared as a path relative to the working directory, so a project
    is read from its own root, which is what the CLI and the README both do.

    Args:
        project: The directory to make current.

    Yields:
        Nothing. The working directory is restored on the way out.
    """
    previous = Path.cwd()
    os.chdir(project)
    try:
        yield
    finally:
        os.chdir(previous)


def generated_view(project: Path) -> Any:
    """Load the view class out of the module `init` wrote.

    Loaded by path, the way the store loads it, so the test reads the file the CLI
    produced rather than a name the test made up itself.

    Args:
        project: The project directory.

    Returns:
        The generated `UserClicks` class.
    """
    path = project / DEFINITION
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UserClicks


def add_second_view(project: Path) -> None:
    """Add a second view to a generated project and register it in the settings.

    The generated project declares exactly one view, so a refresh naming that view
    and a refresh naming none produce the same output and a test cannot tell them
    apart. A second view is what makes the difference observable.

    Args:
        project: The project directory.
    """
    other = "definitions/user_totals.py"
    (project / other).write_text(SECOND_DEFINITION, encoding="utf-8")
    settings = project / SETTINGS_NAME
    settings.write_text(
        settings.read_text(encoding="utf-8").replace(
            f'definitions = ["{DEFINITION}"]', f'definitions = ["{DEFINITION}", "{other}"]'
        ),
        encoding="utf-8",
    )


@contextlib.contextmanager
def write_fails_past(limit: int) -> Iterator[None]:
    """Make every write that would take a file past `limit` bytes fail.

    RLIMIT_FSIZE is the one way to fail a write part-way through from outside
    the process making it, and the extension is loaded in-process here, so there
    is no second process to fill a disk for it. The failure is a real one rather
    than a simulated one: the kernel returns the same EFBIG a full disk raises,
    so the writer takes the path it would take there.

    Past the limit the kernel also sends SIGXFSZ, whose default action is to
    kill the process, so the signal is ignored for the duration. Ignoring it is
    what leaves the error for the writer to report instead of a dead test run.

    Args:
        limit: The largest a file may get, in bytes.

    Yields:
        Nothing. The limit and the previous handler are put back on the way out.
    """
    previous = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, previous[1]))
    ignored = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGXFSZ, ignored)
        resource.setrlimit(resource.RLIMIT_FSIZE, previous)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A generated project with the demo data written into it.

    Args:
        tmp_path: The pytest temp directory.

    Returns:
        The project directory.
    """
    generated = tmp_path / "store"
    init(generated)
    demo(generated)
    return generated


def source_rows(project: Path) -> dict[tuple[int, int], tuple[int, int]]:
    """Index the feature table by user and event timestamp.

    Args:
        project: The project directory.

    Returns:
        Each row's click and purchase counts, keyed by the user and the timestamp
        of the row.
    """
    table = pl.read_parquet(project / "data" / "user_stats.parquet")
    return {
        (row["user_id"], row["event_timestamp"]): (row["click_count"], row["purchase_count"])
        for row in table.iter_rows(named=True)
    }


def test_init_writes_exactly_the_tree_the_readme_documents(tmp_path: Path) -> None:
    generated = tmp_path / "my_feature_store"
    init(generated)
    written = {str(path.relative_to(generated)) for path in generated.rglob("*") if path.is_file()}
    assert written == {SETTINGS_NAME, DEFINITION}


def test_the_generated_module_is_the_readme_example(tmp_path: Path) -> None:
    generated = tmp_path / "store"
    init(generated)
    assert (generated / DEFINITION).read_text(encoding="utf-8") == readme_example(
        f"Define features in `{DEFINITION}`."
    )


def test_the_generated_settings_are_valid(tmp_path: Path) -> None:
    generated = tmp_path / "store"
    init(generated)
    settings = load_settings(generated / SETTINGS_NAME)
    assert settings.project == "store"
    assert list(settings.definitions) == [DEFINITION]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("my_feature_store", "my_feature_store"),
        ("my store", "my_store"),
        ("a.b", "a_b"),
        ("UPPER", "UPPER"),
        ("!!!", "___"),
    ],
)
def test_a_project_name_survives_an_awkward_directory(
    tmp_path: Path, name: str, expected: str
) -> None:
    directory = tmp_path / name
    directory.mkdir()
    assert project_name(directory) == expected


def test_a_name_that_sanitizes_to_nothing_still_writes_a_project() -> None:
    """The root directory is the one that has no name to work from.

    An empty `project` key is rejected by the core, so there is a fallback rather
    than a settings file that fails to load.
    """
    assert project_name(Path("/")) == "feather_project"


def test_init_refuses_to_overwrite_and_force_does_not(tmp_path: Path) -> None:
    generated = tmp_path / "store"
    init(generated)
    (generated / SETTINGS_NAME).write_text("# mine\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        init(generated)
    init(generated, force=True)
    assert "# mine" not in (generated / SETTINGS_NAME).read_text(encoding="utf-8")


def test_init_leaves_the_rest_of_the_directory_alone(tmp_path: Path) -> None:
    generated = tmp_path / "store"
    generated.mkdir()
    (generated / "notes.md").write_text("mine\n", encoding="utf-8")
    init(generated, force=True)
    assert (generated / "notes.md").read_text(encoding="utf-8") == "mine\n"


def test_main_reports_a_failure_as_a_message_and_a_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generated = tmp_path / "store"
    init(generated)
    code = main(["init", str(generated)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("feather init: ")
    assert "Traceback" not in captured.err


def test_an_argument_argparse_rejects_is_still_its_own_exit_code() -> None:
    """The catch around a command is broad, and argparse's exit has to stay outside it.

    `main` parses before it enters the try, so an argument it rejects is
    argparse's own SystemExit(2) rather than a reported failure. This pins that
    separation: widening the catch to cover a user's definition module must not
    turn a mistyped flag into exit 1 with a message, which is what would happen
    if parsing moved inside the try or SystemExit were caught.
    """
    with pytest.raises(SystemExit) as exit:
        main(["refresh", "--no-such-flag"])
    assert exit.value.code == 2


def test_the_quickstart_runs_as_the_readme_writes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with inside(tmp_path):
        assert main(["init", "my_feature_store"]) == 0
    with inside(tmp_path / "my_feature_store"):
        assert main(["demo"]) == 0
    assert (tmp_path / "my_feature_store" / "data" / "user_stats.parquet").is_file()


def test_demo_refuses_to_overwrite_and_force_does_not(project: Path) -> None:
    written = sorted(path.name for path in (project / "data").iterdir())
    assert written == ["training_labels.parquet", "user_stats.parquet"]
    with pytest.raises(FileExistsError, match="--force"):
        demo(project)
    demo(project, force=True)


def test_a_demo_write_that_fails_leaves_no_partial_parquet(tmp_path: Path) -> None:
    """A failed write leaves `data/` empty, because a file is renamed into place whole.

    The writer used to create `user_stats.parquet` and then write into it, so a
    disk that filled up part-way through left a truncated Parquet under the name
    the generated view reads. That is the file the next `feather demo` refuses to
    overwrite and the file a refresh cannot read, and a truncated Parquet is
    indistinguishable from a project that was created wrong.
    """
    generated = tmp_path / "store"
    init(generated)
    # Well under the feature table, which is 14 days of 250 users, and well over
    # the Parquet header, so the failure lands in the middle of the data rather
    # than at the first byte.
    with write_fails_past(limit=4096), pytest.raises(OSError):
        demo(generated)
    assert list((generated / "data").iterdir()) == []


def test_a_demo_that_fails_on_its_second_file_keeps_the_first_one_whole(tmp_path: Path) -> None:
    """The atomicity is per file, and this is what choosing that looks like.

    Two files cannot be renamed as a pair, so a `demo` that fails on the label
    set leaves the feature table written and the labels not there. That is the
    trade `write_parquet` makes deliberately, and the temporary file is gone
    either way, so the failure leaves no file the project would try to read.
    """
    generated = tmp_path / "store"
    init(generated)
    # A directory where the label file goes, so the write itself succeeds and
    # the rename is what cannot land. That is the other failure the cleanup
    # covers, and `demo` checks for the name before it writes anything, so this
    # also takes the `--force` path.
    (generated / "data" / "training_labels.parquet").mkdir(parents=True)
    with pytest.raises(OSError):
        demo(generated, force=True)
    features = pl.read_parquet(generated / "data" / "user_stats.parquet")
    # 14 days of 250 users, so a feature table the writer did not finish fails
    # here rather than reading back as a valid file with fewer rows in it.
    assert features.height == 250 * 14
    assert sorted(path.name for path in (generated / "data").iterdir()) == [
        "training_labels.parquet",
        "user_stats.parquet",
    ]


def test_the_demo_files_hold_the_columns_the_generated_view_declares(project: Path) -> None:
    features = pl.read_parquet(project / "data" / "user_stats.parquet")
    labels = pl.read_parquet(project / "data" / "training_labels.parquet")
    assert features.columns == ["user_id", "event_timestamp", "click_count", "purchase_count"]
    assert labels.columns == ["user_id", "event_timestamp", "label"]
    # Microseconds as an integer, so the point-in-time window the core builds is
    # integer arithmetic on both sides rather than a date subtraction.
    assert features.schema["event_timestamp"] == pl.Int64
    assert features.height > labels.height


def test_the_demo_target_is_the_day_after_the_row_a_label_reads(project: Path) -> None:
    """A label is half a day after the features it carries, and a day before its target."""
    features = source_rows(project)
    labels = pl.read_parquet(project / "data" / "training_labels.parquet")
    targets = set()
    for row in labels.iter_rows(named=True):
        read_at = row["event_timestamp"] - DAY // 2
        target_at = read_at + DAY
        assert (row["user_id"], read_at) in features
        assert (row["user_id"], target_at) in features
        assert row["label"] == int(features[(row["user_id"], target_at)][1] > 0)
        targets.add(row["label"])
    # Both classes, or the target is a constant and nothing was demonstrated.
    assert targets == {0, 1}


def test_refresh_reports_what_it_wrote(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    refresh(project, [])
    out = capsys.readouterr().out
    written = re.search(r"refreshed user_clicks: (\d+) rows", out)
    assert written is not None, out
    # One row per entity, because a refresh takes the newest source row for each
    # key. The demo has 250 users and 14 days each, so 3500 here would mean it
    # wrote a day that the join can no longer reach.
    assert int(written.group(1)) == 250


def test_refresh_takes_one_view_by_name(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Naming a view refreshes that view and leaves the project's others alone.

    Driven through `main` rather than the `refresh` function, so the argument
    order in the parser is part of what is pinned: a positional directory ahead
    of the views would bind `user_clicks` to the directory and refresh both.
    """
    add_second_view(project)
    assert main(["refresh", "-C", str(project), "user_clicks"]) == 0
    names = re.findall(r"^refreshed (\w+):", capsys.readouterr().out, re.MULTILINE)
    assert names == ["user_clicks"]


def test_refresh_with_no_view_named_refreshes_every_view(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the pair: naming none is what reaches the second view."""
    add_second_view(project)
    assert main(["refresh", "-C", str(project)]) == 0
    names = re.findall(r"^refreshed (\w+):", capsys.readouterr().out, re.MULTILINE)
    assert sorted(names) == ["user_clicks", "user_totals"]


def test_the_refreshed_values_are_served_from_the_same_store(project: Path) -> None:
    """One store, not two.

    The embedded store is single-owner and takes an exclusive lock, so a second
    `FeatureStore` over the same directory cannot be opened until the first is
    released. `test_a_renamed_view_is_retired_and_leaves_the_registry` says the
    same at `tests/test_online.py:346`, and
    `a_second_open_over_the_same_directory_is_refused_until_the_first_is_closed`
    pins it in Rust at `crates/feather-core/src/online/fjall.rs:541`. This test
    wants a read against the store that just wrote, so it holds that one store
    rather than opening a second.
    """
    view = generated_view(project)
    entities = pl.DataFrame({"user_id": [1, 2, 3]})
    with inside(project):
        store = FeatureStore(SETTINGS_NAME)
        before = pl.DataFrame(store.get_online_features(entities, [view.click_count]))
        assert before["click_count"].null_count() == 3

        store.materialize(None)

        after = pl.DataFrame(store.get_online_features(entities, [view.click_count]))
    assert after.columns == ["user_id", "click_count"]
    assert after["click_count"].null_count() == 0

    # The newest feature row for each user, which is a refresh's last write per
    # key. Anything else would be a refresh that kept the wrong day's value.
    features = source_rows(project)
    newest = max(timestamp for _, timestamp in features)
    assert after["click_count"].to_list() == [features[(user, newest)][0] for user in (1, 2, 3)]


def test_a_refreshed_project_leaves_a_database_the_next_process_reads(project: Path) -> None:
    """A refresh writes a database on disk, so what it wrote outlives the process.

    The generated project declares no `[store]`, so an absent one resolves to
    `.feather/online` beside the `feather.toml`, and both the write path and
    `serve()` open that same directory.

    This is the test that fails if that resolution is reverted. Before it, the
    write path used an in-process map: `.feather/online` was never created, so the
    first assertion below fails, and the store opened afterwards is a fresh
    in-memory one that knows nothing the first wrote, so every column reads null
    and the second assertion fails. `serve()` would have answered null for every
    column with no error and no log line.

    The first store is released before the second is opened because the embedded
    store is single-owner and takes an exclusive lock over the directory, so a
    second `FeatureStore` cannot be opened until the first is gone.
    """
    view = generated_view(project)
    entities = pl.DataFrame({"user_id": [1, 2, 3]})

    def materialize() -> None:
        with inside(project):
            FeatureStore(SETTINGS_NAME).materialize(None)

    # The store this creates is dropped as the call returns, which is what releases
    # the lock the second `FeatureStore` below needs.
    materialize()

    assert (project / ".feather" / "online").is_dir(), "a refresh writes a database on disk"

    with inside(project):
        reopened = FeatureStore(SETTINGS_NAME)
        values = pl.DataFrame(reopened.get_online_features(entities, [view.click_count]))

    assert values["click_count"].null_count() == 0, "a second store reads what the first wrote"


def test_refresh_of_an_unknown_view_is_a_message_not_a_traceback(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["refresh", "-C", str(project), "not_a_view"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("feather refresh: ")
    assert "Traceback" not in captured.err


def test_a_mistyped_definition_module_is_a_message_not_a_traceback(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`init` writes a module the user then edits, so a typo in it is the likely first failure.

    A definition module is the user's own Python and the store imports it with
    exec_module, so a misspelled name or a missing colon reaches the command as a
    SyntaxError or a NameError. Neither is an OSError, an ImportError or a
    ValueError, so a narrow catch let it out as a raw traceback, against the
    contract `main` documents.
    """
    (project / DEFINITION).write_text("class UserClicks(:\n    pass\n", encoding="utf-8")
    assert main(["refresh", "-C", str(project)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("feather refresh: ")
    assert "Traceback" not in captured.err


def test_a_name_error_in_a_definition_module_is_a_message_not_a_traceback(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A module that parses but names something undefined, which is the other way a typo escapes."""
    (project / DEFINITION).write_text(
        "from feather import Entity, FeatureView, Field, feature_view\n"
        "user_entity = Entity(name='user_id', join_key='user_id')\n"
        "missing = Field(Int64)\n",
        encoding="utf-8",
    )
    assert main(["refresh", "-C", str(project)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("feather refresh: ")
    assert "Traceback" not in captured.err


def test_the_generated_project_reaches_a_training_set_on_its_own(project: Path) -> None:
    """The ten-minute path, end to end: init, demo, then the README's join.

    Every label row has to carry the values of the newest feature row at or before
    it, which for this data is that morning's row rather than yesterday's or
    tomorrow's. The demo dates from today so the generated view's 30-day TTL still
    holds, and each label sits half a day after a feature row, so a null or a
    neighbouring day's value means one of those two facts stopped being true.
    """
    view = generated_view(project)
    labels = pl.read_parquet(project / "data" / "training_labels.parquet")
    with inside(project):
        store = FeatureStore(SETTINGS_NAME)
        joined = pl.DataFrame(
            store.get_historical_features(entity_df=labels, features=list(view_fields(view)))
        )
    assert joined.columns == [
        "user_id",
        "event_timestamp",
        "label",
        "click_count",
        "purchase_count",
    ]
    assert joined.height == labels.height

    features = source_rows(project)
    for row in joined.iter_rows(named=True):
        assert (row["click_count"], row["purchase_count"]) == features[
            (row["user_id"], row["event_timestamp"] - DAY // 2)
        ]


def test_the_two_declared_versions_agree() -> None:
    """pyproject.toml and the workspace Cargo.toml are both a place a version is written.

    maturin builds the wheel from the Python metadata and the extension reports
    the Cargo one, so a release that bumped only one of them would publish a
    package whose `feather.__version__` and `feather._core.__version__` disagree,
    and nothing in CI compared them.
    """
    root = README.parent
    packaged = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    declared = packaged["project"]["version"]
    assert workspace["workspace"]["package"]["version"] == declared

    from feather import _core

    assert _core.__version__ == declared


def test_help_works_without_the_compiled_extension(tmp_path: Path) -> None:
    """`feather --help` is the one thing that has to work on a broken install.

    A subprocess, because this session imported the extension at collection time
    and an in-process check would be reading a module the session put in
    sys.modules itself. The blocker makes the extension unimportable whether or
    not a build left a binary beside the sources, so the test fails if the import
    ever moves back to the top of cli.py, and stays green because the command
    genuinely does not need it.
    """
    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_EXTENSION],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(README.parent / "python")},
    )
    assert result.returncode == 0, result.stderr
    assert "feather" in result.stdout

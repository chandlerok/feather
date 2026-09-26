"""The ``feather`` command.

Three commands, and the third is a refresh rather than an apply. There is nothing
to apply: the core reads ``feather.toml``, definition modules are imported, and a
refresh writes each view's values into the online store. A view a previous refresh
declared and this one does not is retired by that run, so a rename or a removal
needs no separate step.

``init`` writes the tree the README documents, and only that tree. Sample data is
a separate command, because a project that ships with data in it is a project
whose data has to be deleted before the real thing is loaded.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SETTINGS_NAME = "feather.toml"
"""The file every command reads, at the root of a project."""

DEFINITION = "definitions/user_clicks.py"
"""The module ``init`` writes, and the one the generated ``feather.toml`` lists."""

DEMO_FILES = ("user_stats.parquet", "training_labels.parquet")
"""The files `demo` writes, checked before it overwrites anything."""

_SETTINGS = """\
# Infrastructure config, validated by the Rust core when it is loaded, so a
# mistake fails before an engine starts. Credentials live here and nowhere else.
project = "{project}"
definitions = ["{definition}"]
"""

_DEFINITION_MODULE = """\
from feather import Entity, FeatureView, Field, FileSource, feature_view
from feather.types import Int64

user_entity = Entity(name="user_id", join_key="user_id")
user_stats_source = FileSource(path="data/user_stats.parquet")


@feature_view(
    name="user_clicks",
    entity=user_entity,
    source=user_stats_source,
    ttl_days=30,
)
class UserClicks(FeatureView):
    click_count = Field(Int64)
    purchase_count = Field(Int64)
"""

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def init(directory: Path, *, force: bool = False) -> None:
    """Write a project: the settings file and one definition module.

    Args:
        directory: The project directory, created if it does not exist. An existing
            directory is fine as long as this command's two files are not already
            there.
        force: Overwrite the two files if they are already there. The directory's
            other contents are left alone either way.

    Raises:
        FileExistsError: If either file is already there and ``force`` is false.
        OSError: If a file or directory cannot be created.
    """
    if not force and any((directory / name).exists() for name in (SETTINGS_NAME, DEFINITION)):
        raise FileExistsError(
            f"{directory} already holds a feather project; pass --force to overwrite "
            f"{SETTINGS_NAME} and {DEFINITION}"
        )
    definition = directory / DEFINITION
    definition.parent.mkdir(parents=True, exist_ok=True)
    definition.write_text(_DEFINITION_MODULE, encoding="utf-8")
    (directory / SETTINGS_NAME).write_text(
        _SETTINGS.format(project=project_name(directory), definition=DEFINITION),
        encoding="utf-8",
    )
    print(f"created {directory / SETTINGS_NAME}")
    print(f"created {definition}")
    print(f"next: cd {directory} && feather demo")


def project_name(directory: Path) -> str:
    """Derive a project name from the directory name.

    The name only has to be a non-empty string the core accepts, and a directory
    called ``my store`` is a legal directory. Anything that is not a letter, a
    digit, a dash, or an underscore becomes one, and the root directory, which has
    no name to work from, falls back rather than writing an empty one.

    Args:
        directory: The project directory.

    Returns:
        A name for the ``project`` key.
    """
    cleaned = _UNSAFE.sub("_", directory.resolve().name)
    return cleaned or "feather_project"


def demo(directory: Path, *, force: bool = False) -> None:
    """Write the two Parquet files the generated view reads.

    Args:
        directory: The project directory. ``data`` is created inside it.
        force: Overwrite the files if they are already there.

    Raises:
        FileExistsError: If either file is already there and ``force`` is false.
        OSError: If a directory or file cannot be created.
        ImportError: If the compiled extension is not built. The Parquet writer is
            in it, so there is no pure-Python fallback.
    """
    if not force and any((directory / "data" / name).exists() for name in DEMO_FILES):
        raise FileExistsError(
            f"{directory / 'data'} already holds the demo data; pass --force to overwrite it"
        )
    from feather import _core

    for path in _core.write_demo_data(str(directory)):
        print(f"created {path}")


def refresh(directory: Path, views: Sequence[str]) -> None:
    """Refresh the project's views into the online store.

    A project with no Valkey in its settings is local mode, and its online store
    is in-process and belongs to the store object that opened it. So the values
    written here are served to this process and to nothing after it; a
    deployment that serves from more than one process configures a Valkey, and
    the same call writes to that.

    Args:
        directory: The project directory, holding ``feather.toml``.
        views: The views to refresh, by name, or empty for every view the project
            declares.

    Raises:
        FileNotFoundError: If there is no ``feather.toml`` in ``directory``.
        ValueError: If a named view is not declared, or a source cannot be read as
            its view declares it.
        ConnectionError: If the settings declare a Valkey that cannot be reached.
    """
    from feather import FeatureStore

    # A source is declared as a path relative to the working directory, and the
    # core hands it to DuckDB as written, so a project is refreshed from its own
    # root. Definitions already resolve against the settings file, so this only
    # moves the source paths.
    previous = Path.cwd()
    try:
        os.chdir(directory)
        report = FeatureStore(SETTINGS_NAME).materialize(views or None)
    finally:
        os.chdir(previous)
    for view in report.views:
        print(f"refreshed {view.name}: {view.rows} rows in {view.elapsed_seconds:.2f}s")
    for name in report.retired:
        print(f"retired {name}")
    print(f"{len(report.views)} views, {report.total_rows} rows, {report.elapsed_seconds:.2f}s")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and run the command.

    Args:
        argv: The arguments, or ``None`` for the process's own.

    Returns:
        The process exit code: 0 on success, 1 on a failure it reports, and 2 for
        arguments argparse rejects.

    Raises:
        Nothing. A failure a command raises is reported and becomes an exit code,
        so a traceback is not the output of a mistyped path.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init(args.directory, force=args.force)
        elif args.command == "demo":
            demo(args.directory, force=args.force)
        elif args.command == "refresh":
            refresh(args.directory, args.views)
    except Exception as error:
        # ConnectionError is an OSError, so a Valkey that cannot be reached is
        # already covered. The catch is broad because a definition module is the
        # user's own Python, imported by exec_module, so a typo in it is a
        # SyntaxError, a NameError or an AttributeError and none of those is an
        # OSError or an ImportError. Every one of them carries a message worth
        # reading, and the docstring above promises a traceback is not what a
        # mistyped path produces.
        print(f"feather {args.command}: {error}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser, with one subcommand per operation.
    """
    parser = argparse.ArgumentParser(
        prog="feather",
        description="An opinionated feature store.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    created = commands.add_parser("init", help="write a project: settings and one view")
    created.add_argument("directory", type=Path, nargs="?", default=Path())
    created.add_argument(
        "--force",
        action="store_true",
        help="overwrite the files if they are already there",
    )

    sample = commands.add_parser("demo", help="write sample data the generated view reads")
    sample.add_argument("directory", type=Path, nargs="?", default=Path())
    sample.add_argument(
        "--force",
        action="store_true",
        help="overwrite the files if they are already there",
    )

    materialized = commands.add_parser(
        "refresh",
        help="refresh feature values from their sources into the online store",
    )
    # The directory is an option and not a positional, because the README's
    # sentence for this command is `feather refresh user_clicks`, and a
    # positional directory in front of the views would bind that word to it. The
    # project being refreshed is the one you are standing in, which is the common
    # case, so it does not need to be typed.
    materialized.add_argument(
        "-C",
        "--directory",
        type=Path,
        default=Path(),
        help="the project directory; the current one if this is not given",
    )
    materialized.add_argument(
        "views",
        nargs="*",
        help="the views to refresh, by name; every view if none is named",
    )
    return parser


__all__ = ["demo", "init", "main", "project_name", "refresh"]

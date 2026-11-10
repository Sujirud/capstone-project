"""Project paths from config.toml.

The project root is the folder that holds config.toml. Set CAPSTONE_ROOT to use
another root; otherwise the root is found by walking up from the current folder.
"""

import os
import tomllib
from pathlib import Path

CONFIG_NAME = "config.toml"
ROOT_ENV = "CAPSTONE_ROOT"


def project_root() -> Path:
    """Return the folder that holds config.toml."""
    override = os.environ.get(ROOT_ENV)
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / CONFIG_NAME).is_file():
            raise FileNotFoundError(f"{ROOT_ENV}={override} has no {CONFIG_NAME}")
        return root
    start = Path.cwd().resolve()
    for folder in (start, *start.parents):
        if (folder / CONFIG_NAME).is_file():
            return folder
    raise FileNotFoundError(
        f"No {CONFIG_NAME} in {start} or its parents; run from the project or set {ROOT_ENV}"
    )


def _paths_table() -> dict[str, str]:
    with open(project_root() / CONFIG_NAME, "rb") as f:
        return tomllib.load(f)["paths"]


def names() -> list[str]:
    """Return every path name in config.toml."""
    return list(_paths_table())


def path(name: str) -> Path:
    """Return the absolute path for a name in the [paths] table of config.toml."""
    table = _paths_table()
    if name not in table:
        raise KeyError(f"Unknown path name {name!r}; config.toml has {', '.join(table)}")
    return project_root() / table[name]


def ensure_dir(name: str, *parts: str) -> Path:
    """Return path(name) joined with parts, creating the folder if it does not exist."""
    folder = path(name).joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

"""facemeasure.paths on a synthetic project in tmp_path."""

from pathlib import Path

import pytest

from facemeasure import paths

CONFIG = """
[paths]
data = "data"
outputs = "data/outputs"
shared = "shared"
"""


def make_project(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / paths.CONFIG_NAME).write_text(CONFIG, encoding="utf-8")
    return folder


def test_root_found_by_walking_up(tmp_path, monkeypatch):
    root = make_project(tmp_path / "project")
    nested = root / "model" / "scripts"
    nested.mkdir(parents=True)
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    monkeypatch.chdir(nested)
    assert paths.project_root() == root.resolve()
    assert paths.path("outputs") == root.resolve() / "data" / "outputs"


def test_env_overrides_current_folder(tmp_path, monkeypatch):
    here = make_project(tmp_path / "here")
    other = make_project(tmp_path / "other")
    monkeypatch.chdir(here)
    monkeypatch.setenv(paths.ROOT_ENV, str(other))
    assert paths.project_root() == other.resolve()
    assert paths.path("shared") == other.resolve() / "shared"


def test_env_without_config_fails(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match=paths.ROOT_ENV):
        paths.project_root()


def test_missing_config_fails(tmp_path, monkeypatch):
    monkeypatch.delenv(paths.ROOT_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    if any((p / paths.CONFIG_NAME).is_file() for p in tmp_path.resolve().parents):
        pytest.skip("a parent of tmp_path has config.toml")
    with pytest.raises(FileNotFoundError):
        paths.project_root()


def test_names_and_unknown_name(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    assert paths.names() == ["data", "outputs", "shared"]
    with pytest.raises(KeyError, match="nope"):
        paths.path("nope")


def test_ensure_dir_creates_nested_folder(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    folder = paths.ensure_dir("outputs", "scut_preview", "run1")
    assert folder == tmp_path.resolve() / "data" / "outputs" / "scut_preview" / "run1"
    assert folder.is_dir()
    assert paths.ensure_dir("outputs", "scut_preview", "run1") == folder


def test_project_config_paths_are_relative_posix():
    import tomllib

    with open(paths.project_root() / paths.CONFIG_NAME, "rb") as f:
        table = tomllib.load(f)["paths"]
    for name, value in table.items():
        assert not Path(value).is_absolute(), name
        assert "\\" not in value, f"{name} must use / so it works on macOS and Windows"

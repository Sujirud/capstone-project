"""What git holds: no photos and nothing from the datasets (model.md section 5.1, api.md section 6),
and the trained bundle.

The only images allowed are the graphs in model/reports/figures/, and a person must still
look at them to see that none shows a face. The test reads `git ls-files`, so it skips
outside a git checkout, such as a zip for the graders.
"""

import json
import subprocess
from pathlib import PurePosixPath

import pytest

from facemeasure import paths

# Photos, and the fake camera video of the end-to-end test (data/outputs/e2e/)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".gif", ".tif", ".tiff", ".avif"}
VIDEO_SUFFIXES = {".mjpeg", ".mjpg", ".y4m", ".mp4", ".mov", ".webm"}
FIGURE_SUFFIXES = {".png"}
# Folders that stay on the team's machines: the datasets and outputs, and the downloaded
# model files. The bundle holds only weights and numbers, so it is committed.
OUTSIDE_GIT = ("data", "assets")


def tracked_files() -> list[PurePosixPath]:
    root = paths.project_root()
    try:
        done = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, or git is missing")
    return [PurePosixPath(p) for p in done.stdout.decode("utf-8").split("\0") if p]


def relative(name: str) -> PurePosixPath:
    return PurePosixPath(paths.path(name).relative_to(paths.project_root()).as_posix())


def is_under(file: PurePosixPath, folder: PurePosixPath) -> bool:
    return folder in file.parents


@pytest.fixture(scope="module")
def files():
    return tracked_files()


def test_git_holds_no_photos_or_videos_outside_the_figures(files):
    figures = relative("reports") / "figures"
    found = [
        str(f)
        for f in files
        if f.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
        and not (is_under(f, figures) and f.suffix.lower() in FIGURE_SUFFIXES)
    ]
    assert found == [], f"images or videos in git: {found}"


def test_the_figures_are_png_graphs_only(files):
    figures = relative("reports") / "figures"
    assert all(f.suffix.lower() in FIGURE_SUFFIXES for f in files if is_under(f, figures))


@pytest.mark.parametrize("name", OUTSIDE_GIT)
def test_git_holds_nothing_from_folders_that_stay_on_the_machine(files, name):
    folder = relative(name)
    found = [str(f) for f in files if is_under(f, folder)]
    assert found == [], f"files from {folder}/ in git: {found}"


def test_git_holds_a_trained_bundle(files):
    """A clone gets results for every region; make_stub_bundle.py writes a stub over it."""
    folder = relative("bundle")
    manifest = folder / "manifest.json"
    assert manifest in files, f"{folder}/ is not in git"
    done = subprocess.run(["git", "show", f":{manifest}"], cwd=paths.project_root(), capture_output=True, check=True)
    data = json.loads(done.stdout.decode("utf-8"))  # the version git holds, not the one on disk
    stubs = [name for name, head in data["heads"].items() if head["stub"]]
    assert not data["stub"] and stubs == [], f"the bundle in git has stub heads: {stubs}"
    assert all(folder / name in files for name in data["files"])


def test_the_checks_see_a_path_with_upper_case_suffixes():
    assert PurePosixPath("web/x/Photo.JPG").suffix.lower() in IMAGE_SUFFIXES
    assert is_under(PurePosixPath("data/raw/fei/1-01.jpg"), PurePosixPath("data"))
    assert not is_under(PurePosixPath("database.md"), PurePosixPath("data"))

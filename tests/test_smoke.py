"""Smoke tests: the environment installs and the project layout is readable."""

import pytest

from facemeasure import paths

EXPECTED_PATH_NAMES = {
    "data", "raw", "scut", "ffhq", "fei", "labels", "grading",
    "features", "outputs", "shared", "assets", "bundle",
}
SCUT_COUNT = 5500


def test_import_facemeasure():
    import facemeasure

    assert facemeasure.__version__


def test_every_config_path_reads():
    root = paths.project_root()
    assert EXPECTED_PATH_NAMES <= set(paths.names())
    for name in paths.names():
        p = paths.path(name)
        assert p.is_absolute()
        assert p.is_relative_to(root), f"{name} must stay inside the project root"


def test_mediapipe_face_landmarker():
    import mediapipe
    from mediapipe.tasks.python import vision

    assert mediapipe.__version__
    assert hasattr(vision, "FaceLandmarker")


def test_torch_and_device():
    import torch

    from facemeasure.device import pick_device

    device = pick_device()
    print(f"torch {torch.__version__} on {device}")
    assert device.type in {"cuda", "mps", "cpu"}
    assert torch.zeros(2, device=device).sum().item() == 0


def test_scut_files_complete():
    scut = paths.path("scut")
    if not scut.is_dir():
        pytest.skip(f"{scut} not found")
    images = [p for p in (scut / "Images").iterdir() if p.is_file() and not p.name.startswith(".")]
    landmarks = [p for p in (scut / "facial landmark").iterdir() if p.is_file() and not p.name.startswith(".")]
    assert len(images) == SCUT_COUNT
    assert len(landmarks) == SCUT_COUNT

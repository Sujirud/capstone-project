"""model/scripts/make_stub_bundle.py with a synthetic label file."""

import warnings

import pandas as pd
import pytest

from facemeasure import bundle, criteria
from facemeasure.bundle import BundleWarning
from make_stub_bundle import STUB_FALLBACK, label_medians, main, stub_bundle


def labels_csv(tmp_path):
    path = tmp_path / "scut_labels.csv"
    pd.DataFrame(
        {"image": ["a", "b", "c", "d"], "jaw_ratio": [0.80, 0.84, 0.86, None], "nose_ratio": [1.0, 1.1, 1.2, 1.3]}
    ).to_csv(path, index=False, encoding="utf-8")
    return path


def test_medians_of_the_labels(tmp_path):
    medians, found = label_medians(labels_csv(tmp_path))
    assert found and medians == pytest.approx({"jaw": 0.84, "nose_base": 1.15})


def test_fallback_without_labels(tmp_path):
    medians, found = label_medians(tmp_path / "none.csv")
    assert not found and medians == {"jaw": 0.85, "nose_base": 1.09} == STUB_FALLBACK


def test_stub_contents():
    b = stub_bundle({"jaw": 0.84, "nose_base": 1.1})
    m = b.manifest
    assert m["bundle_version"] == "0.0.0-stub" and m["stub"] is True
    assert m["train_face_iod_px"] is None and m["degrade"] is None
    assert m["backbone"]["name"] == "dinov2_vitb14" and m["backbone"]["features"] == {"cls": 768}
    for name in ("jaw", "nose_base", "undereye"):
        head = b.heads[name]
        assert head.stub and head.feature == "cls" and not head.weights.any() and head.weights.shape == (768,)
    assert b.heads["jaw"].band == criteria.DEFAULT_BANDS["facial_taperness"]
    assert b.heads["nose_base"].band == criteria.DEFAULT_BANDS["alar_intercanthal_ratio"]
    assert b.heads["jaw"].rmse is None
    u = b.heads["undereye"]
    assert (u.temperature, u.low, u.high) == (1.0, 0.3, 0.7)
    assert u.predict(u.weights) == pytest.approx([0.5, 0.2])
    assert all((x.slope, x.intercept) == (1.0, 0.0) for x in b.baseline.values())


def test_main_writes_a_bundle_that_loads(tmp_path, capsys):
    out = tmp_path / "bundle"
    assert main(["--out", str(out), "--labels", str(labels_csv(tmp_path))]) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        b = bundle.load_bundle(out)
    assert b.version == "0.0.0-stub" and b.heads["nose_base"].bias == pytest.approx(1.15)
    assert "0.0.0-stub" in capsys.readouterr().out

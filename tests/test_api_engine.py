"""api/engine.py without the real models: the head step and the start checks."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import __main__ as api_main
from api import engine as E
from api.main import LOGGER_NAME, create_app
from facemeasure import assets, bundle, quality
from facemeasure.bundle import ordinal, ridge
from facemeasure.i18n import Texts
from make_stub_bundle import stub_parts

DIMS = {"cls": 768, "cls_mean": 1536, "grid4": 12288}


class FakeBackbone:
    """Feature k of crop i is (i + 1) * (k + 1) / dim, so each crop and feature type differs."""

    def __init__(self):
        self.batches = []

    def features(self, images):
        images = np.asarray(images)
        self.batches.append(images.shape)
        n = len(images)
        return {name: np.outer(np.arange(1, n + 1), np.arange(1, d + 1) / d).astype(np.float32) for name, d in DIMS.items()}


def heads_bundle(trained):
    """The stub with weights that read the features: the ratio heads on cls_mean, the under-eye head on grid4."""
    manifest, arrays = stub_parts()
    manifest = trained(manifest)
    manifest["backbone"]["features"] = dict(DIMS)
    rng = np.random.default_rng(0)
    for name, feature in (("jaw", "cls_mean"), ("nose_base", "cls_mean"), ("undereye", "grid4")):
        manifest["heads"][name]["feature"] = feature
        arrays[f"{name}_weights"] = rng.normal(scale=1e-3, size=DIMS[feature])
    manifest["files"] = {bundle.HEADS_NAME: "0" * 64}
    return bundle.parse(manifest, arrays)


def test_predict_runs_the_three_crops_in_one_batch_with_each_heads_feature(trained, frontal_result):
    b = heads_bundle(trained)
    backbone = FakeBackbone()
    engine = E.Engine(E.Parts(quality.load_config(), Texts(), b, ()), models=None, backbone=backbone, device="cpu")
    p = engine.predict(frontal_result())
    assert backbone.batches == [(3, 224, 224, 3)]
    feats = FakeBackbone().features(np.zeros((3, 224, 224, 3), dtype=np.uint8))
    jaw, nose, eye = b.heads["jaw"], b.heads["nose_base"], b.heads["undereye"]
    assert p.jaw == pytest.approx(ridge(feats["cls_mean"][0], jaw.weights, jaw.bias))
    assert p.nose_base == pytest.approx(ridge(feats["cls_mean"][0], nose.weights, nose.bias))
    left = ordinal(feats["grid4"][1], eye.weights, eye.bias, eye.temperature)
    right = ordinal(feats["grid4"][2], eye.weights, eye.bias, eye.temperature)
    assert p.undereye_left == pytest.approx(tuple(left)) and p.undereye_right == pytest.approx(tuple(right))
    assert p.undereye_left != pytest.approx(p.undereye_right)


def test_predict_without_a_face_runs_nothing(ready_bundle, frontal_result):
    backbone = FakeBackbone()
    engine = E.Engine(E.Parts(quality.load_config(), Texts(), ready_bundle, ()), models=None, backbone=backbone, device="cpu")
    p = engine.predict(frontal_result(face=False))
    assert backbone.batches == [] and p == E.Predictions()


def test_no_shrinking_without_a_training_range(ready_bundle, trained):
    engine = E.Engine(E.Parts(quality.load_config(), Texts(), ready_bundle, ()), models=None, backbone=None, device="cpu")
    assert ready_bundle.train_face_iod_px is None and engine.max_iod_px is None
    manifest, arrays = stub_parts()
    manifest = trained(manifest)
    manifest["train_face_iod_px"] = {"min": 75.0, "max": 110.0}
    manifest["files"] = {bundle.HEADS_NAME: "0" * 64}
    b = bundle.parse(manifest, arrays)
    assert E.Engine(E.Parts(quality.load_config(), Texts(), b, ()), None, None, "cpu").max_iod_px == 110.0


def test_check_features_refuses_a_bundle_for_other_features(ready_bundle):
    E.check_features(ready_bundle, DIMS)
    with pytest.raises(E.StartupError, match="cls"):
        E.check_features(ready_bundle, {"cls": 384})


# ---------------------------------------------------------------- start


def test_missing_assets_name_the_download_command(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "assets_dir", lambda: tmp_path / "empty")
    with pytest.raises(E.StartupError, match="uv run python -m facemeasure.assets"):
        E.load_parts()


def test_assets_that_differ_from_the_lock_say_to_copy_the_folder(monkeypatch):
    def mismatch(*a, **kw):
        raise assets.AssetMismatch("face_landmarker.task: sha256 abc, the lock says def")

    monkeypatch.setattr(assets, "verify", mismatch)
    with pytest.raises(E.StartupError, match="copy the whole folder"):
        E.load_parts()


def test_missing_bundle_names_the_stub_command(monkeypatch, tmp_path):
    monkeypatch.setattr(assets, "verify", lambda *a, **kw: [])
    monkeypatch.setattr(bundle, "bundle_dir", lambda: tmp_path / "no_bundle")
    with pytest.raises(E.StartupError, match="make_stub_bundle.py"):
        E.load_parts()


def test_bundle_warnings_are_kept_for_the_log(monkeypatch, tmp_path):
    manifest, arrays = stub_parts()
    manifest["runtime"]["torch"] = "0.0.1"
    bundle.write_bundle(manifest, arrays, tmp_path / "bundle")
    monkeypatch.setattr(assets, "verify", lambda *a, **kw: [])
    monkeypatch.setattr(bundle, "bundle_dir", lambda: tmp_path / "bundle")
    parts = E.load_parts()
    assert parts.bundle.version == "0.0.0-stub"
    assert any("torch 0.0.1" in w for w in parts.warnings)


def test_bad_quality_file_stops_the_start(monkeypatch):
    def broken(*a, **kw):
        raise quality.QualityConfigError("frontal: missing keys ['min_iod_px']")

    monkeypatch.setattr(quality, "load_config", broken)
    with pytest.raises(E.StartupError, match="quality.json"):
        E.load_parts()


def test_a_failed_start_is_logged_and_stops_the_app(tmp_path):
    import logging

    def factory():
        raise E.StartupError("no manifest.json in api/bundle; run: uv run python model/scripts/make_stub_bundle.py")

    seen = []
    handler = logging.Handler()
    handler.emit = lambda record: seen.append(record.getMessage())
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    try:
        with pytest.raises(E.StartupError), TestClient(create_app(factory, web_dist=tmp_path)):
            pass
    finally:
        logger.removeHandler(handler)
    assert any("cannot start the api" in s and "make_stub_bundle.py" in s for s in seen)


def test_python_m_api_checks_the_files_first_and_keeps_the_host(monkeypatch, capsys):
    def missing():
        raise E.StartupError("face_landmarker.task is not in the assets folder; run: uv run python -m facemeasure.assets")

    monkeypatch.setattr(api_main, "load_parts", missing)
    started = []
    monkeypatch.setattr(api_main.uvicorn, "run", lambda *a, **kw: started.append(kw))
    assert api_main.main([]) == 1
    assert "facemeasure.assets" in capsys.readouterr().err and started == []

    monkeypatch.setattr(api_main, "load_parts", lambda: None)
    assert api_main.main(["--port", "8123"]) == 0
    assert started == [{"host": "127.0.0.1", "port": 8123, "access_log": False}]
    with pytest.raises(SystemExit):
        api_main.main(["--host", "0.0.0.0"])

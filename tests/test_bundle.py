"""api/bundle/: writing, loading, checks and the head formulas (bundle.py)."""

import copy
import json
import math
import warnings

import numpy as np
import pytest

from facemeasure import bundle, crops
from facemeasure.bundle import BundleError, BundleWarning
from make_stub_bundle import stub_parts


@pytest.fixture
def parts():
    return stub_parts({"jaw": 0.84, "nose_base": 1.1})


def write(tmp_path, manifest, arrays):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        return bundle.write_bundle(manifest, arrays, tmp_path / "bundle")


def read_manifest(folder):
    with open(folder / bundle.MANIFEST_NAME, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(folder, manifest):
    with open(folder / bundle.MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def test_stub_writes_and_loads_back(tmp_path, parts):
    b = write(tmp_path, *parts)
    folder = tmp_path / "bundle"
    assert sorted(p.name for p in folder.iterdir()) == ["heads.npz", "manifest.json"]
    loaded = bundle.load_bundle(folder)
    assert loaded.version == "0.0.0-stub" and loaded.stub and loaded.stub_heads == ("jaw", "nose_base", "undereye")
    assert loaded.heads["jaw"].bias == pytest.approx(0.84) and loaded.heads["nose_base"].band == 0.04
    assert loaded.train_face_iod_px is None
    assert loaded.manifest["crop_version"] == crops.CROP_VERSION
    assert set(loaded.manifest) == set(bundle.MANIFEST_KEYS)
    assert b.manifest == loaded.manifest
    # nothing is left next to the bundle
    assert [p.name for p in tmp_path.iterdir()] == ["bundle"]


def test_same_arrays_give_the_same_file(tmp_path, parts):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    bundle.write_npz(a, parts[1])
    bundle.write_npz(b, parts[1])
    assert a.read_bytes() == b.read_bytes()
    got = bundle.read_npz(a)
    assert set(got) == set(parts[1]) and got["jaw_bias"].shape == ()


def test_a_changed_file_fails_the_sha256(tmp_path, parts):
    write(tmp_path, *parts)
    folder = tmp_path / "bundle"
    arrays = bundle.read_npz(folder / "heads.npz")
    arrays["jaw_bias"] = np.array(0.9)
    bundle.write_npz(folder / "heads.npz", arrays)
    with pytest.raises(BundleError, match="sha256"):
        bundle.load_bundle(folder)


def test_a_missing_file_is_reported(tmp_path, parts):
    write(tmp_path, *parts)
    (tmp_path / "bundle" / "heads.npz").unlink()
    with pytest.raises(BundleError, match="heads.npz"):
        bundle.load_bundle(tmp_path / "bundle")


def test_other_crop_version_does_not_load(tmp_path, parts):
    write(tmp_path, *parts)
    folder = tmp_path / "bundle"
    m = read_manifest(folder)
    m["crop_version"] = crops.CROP_VERSION + 1
    save_manifest(folder, m)
    with pytest.raises(BundleError, match="crop_version"):
        bundle.load_bundle(folder)


def test_no_bundle_says_how_to_make_one(tmp_path):
    with pytest.raises(BundleError, match="make_stub_bundle.py"):
        bundle.load_bundle(tmp_path / "nothing")


def test_other_runtime_and_assets_warn(tmp_path, parts):
    write(tmp_path, *parts)
    folder = tmp_path / "bundle"
    m = read_manifest(folder)
    m["runtime"]["torch"] = "0.0.1"
    m["assets"] = {"face_landmarker.task": "0" * 64}
    save_manifest(folder, m)
    with pytest.warns(BundleWarning) as caught:
        bundle.load_bundle(folder)
    text = " ".join(str(w.message) for w in caught)
    assert "torch 0.0.1" in text and "model files" in text


def test_same_runtime_does_not_warn(tmp_path, parts):
    write(tmp_path, *parts)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BundleWarning)
        bundle.load_bundle(tmp_path / "bundle")


def test_write_replaces_a_bundle_but_not_other_files(tmp_path, parts):
    manifest, arrays = parts
    write(tmp_path, manifest, arrays)
    newer = copy.deepcopy(manifest)
    newer["bundle_version"] = "0.1.0"
    assert write(tmp_path, newer, arrays).version == "0.1.0"
    assert bundle.load_bundle(tmp_path / "bundle").version == "0.1.0"
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(BundleError, match="not a bundle"):
        bundle.write_bundle(manifest, arrays, other)
    assert (other / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_a_bad_bundle_is_not_written(tmp_path, parts):
    manifest, arrays = parts
    write(tmp_path, manifest, arrays)
    bad = copy.deepcopy(manifest)
    bad["bundle_version"] = "1.0"
    with pytest.raises(BundleError, match="semver"):
        write(tmp_path, bad, arrays)
    assert bundle.load_bundle(tmp_path / "bundle").version == "0.0.0-stub"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]


def test_object_arrays_are_refused(tmp_path, parts):
    arrays = parts[1]
    with pytest.raises(BundleError, match="objects"):
        bundle.write_npz(tmp_path / "x.npz", {**arrays, "jaw_bias": np.array([{"a": 1}], dtype=object)})
    np.savez(tmp_path / "pickled.npz", jaw_bias=np.array([{"a": 1}], dtype=object))
    with pytest.raises(BundleError, match="cannot read"):
        bundle.read_npz(tmp_path / "pickled.npz")


def with_files(manifest):
    return {**manifest, "files": {"heads.npz": "0" * 64}}


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda m, a: m.pop("baseline"), "baseline"),
        (lambda m, a: m.update(extra=1), "unknown keys"),
        (lambda m, a: m.update(stub=False), "stub"),
        (lambda m, a: m.update(created="yesterday"), "created"),
        (lambda m, a: m["backbone"].update(name="dinov3_vitb16"), "dinov2_vitb14"),
        (lambda m, a: m["heads"]["jaw"].update(feature="grid4"), "feature"),
        (lambda m, a: m["heads"]["jaw"].update(band=0), "band"),
        (lambda m, a: m["heads"]["undereye"]["thresholds"].update(low=0.8), "thresholds"),
        (lambda m, a: m["heads"]["undereye"].update(temperature=-1), "temperature"),
        (lambda m, a: a.update(jaw_weights=np.zeros(10)), "shape"),
        (lambda m, a: a.update(undereye_bias=np.zeros(3)), "shape"),
        (lambda m, a: a.update(spare=np.zeros(2)), "spare"),
        (lambda m, a: m.update(train_face_iod_px={"min": 110, "max": 75}), "below min"),
        (lambda m, a: m.update(degrade={"sigma": 1}), "degrade"),
        (lambda m, a: m.update(files={}), "files"),
    ],
)
def test_parse_reports_what_is_wrong(parts, change, match):
    manifest, arrays = copy.deepcopy(parts[0]), dict(parts[1])
    manifest = with_files(manifest)
    change(manifest, arrays)
    with pytest.raises(BundleError, match=match):
        bundle.parse(manifest, arrays)


def test_trained_fields_are_accepted(parts):
    manifest, arrays = copy.deepcopy(parts[0]), dict(parts[1])
    manifest["heads"]["jaw"].update(stub=False, rmse=0.03, alpha=10.0)
    manifest["train_face_iod_px"] = {"min": 75, "max": 110}
    manifest["degrade"] = {"version": 1, "jpeg_quality": 85}
    manifest["backbone"]["features"]["grid4"] = 12288
    b = bundle.parse(with_files(manifest), arrays)
    assert b.stub_heads == ("nose_base", "undereye") and b.heads["jaw"].rmse == 0.03
    assert b.train_face_iod_px == {"min": 75, "max": 110}


# ---------------------------------------------------------------- formulas


def test_ridge_by_hand():
    w = np.array([0.5, -1.0, 2.0])
    x = np.array([2.0, 1.0, 0.25])
    assert bundle.ridge(x, w, 0.1) == pytest.approx(1.0 - 1.0 + 0.5 + 0.1)
    assert bundle.ridge(np.stack([x, 2 * x]), w, 0.1) == pytest.approx([0.6, 1.1])


def test_ordinal_by_hand():
    w = np.array([1.0, 2.0])
    x = np.array([0.5, 0.25])  # x . w = 1
    b = np.array([0.5, -2.0])
    t = 2.0
    expected = [1 / (1 + math.exp(-(1 + 0.5) / t)), 1 / (1 + math.exp(-(1 - 2.0) / t))]
    assert bundle.ordinal(x, w, b, t) == pytest.approx(expected)
    assert bundle.ordinal(np.stack([x, x]), w, b, t) == pytest.approx(np.array([expected, expected]))


def test_stub_heads_give_their_fixed_answers(parts):
    b = bundle.parse(with_files(parts[0]), parts[1])
    x = np.random.default_rng(1).normal(size=(4, 768))
    assert b.heads["jaw"].predict(x) == pytest.approx([0.84] * 4)
    assert b.heads["undereye"].predict(x[0]) == pytest.approx([0.5, 0.2])
    assert b.baseline["jaw"].apply(0.842) == pytest.approx(0.842)

"""Fidelity check of the under-eye head (api.md section 7) on 3 test sides with the real data.

The lowest-IOD copy of each side is made again from its seed and run the api's way
(analyze_for_model, under-eye crop, DINOv2, the under-eye head of api/bundle/); P(grade >= 1)
and P(grade >= 2) must match those of `train_undereye_head.py evaluate` within 0.0001.
Skips without a trained bundle, the evaluate results, the images or the model files.
"""

import warnings

import pytest

import extract_features as ef
import train_undereye_head as tuh
from facemeasure import assets, bundle, paths, quality

SIDES = 3


def _folder():
    return paths.path("outputs") / tuh.OUT_NAME


def _ready() -> str | None:
    if not (_folder() / tuh.PREDICTIONS_NAME).is_file():
        return "no evaluate results in outputs/undereye_head/"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bundle.BundleWarning)
            b = bundle.load_bundle()
    except bundle.BundleError as e:
        return f"no bundle: {e}"
    if b.heads[bundle.UNDEREYE_HEAD].stub:
        return "api/bundle/ still has the stub under-eye head; run train_undereye_head.py bundle"
    try:
        for name in ("face_landmarker.task", "selfie_multiclass_256x256.tflite", "dinov2_vitb14_pretrain.pth"):
            assets.asset_path(name)
        assets.extracted("dinov2_src")
    except (assets.AssetError, FileNotFoundError):
        return "no model files in api/assets/"
    return None


def test_bundle_gives_the_probabilities_of_evaluate_on_copies_made_again():
    why = _ready()
    if why:
        pytest.skip(why)
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bundle.BundleWarning)
        b = bundle.load_bundle()
    rows = tuh.pick_fidelity(tuh.read_predictions(_folder()), SIDES)
    image_paths = {s.image: s.path for s in ef.undereye_sources()}
    missing = [n for n in rows["image"] if n not in image_paths or not image_paths[n].is_file()]
    if missing:
        pytest.skip(f"graded images missing, for example {missing[0]}")
    with FaceModels() as models:
        check = tuh.fidelity_check(b, rows, models, Backbone(), quality.load_config(), image_paths)
    assert check["sides"] == SIDES
    assert check["passed"], [(r["image"], r["side"], r["diff"]) for r in check["rows"]]

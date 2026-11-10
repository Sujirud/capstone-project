"""Fidelity check of the jaw and alar base heads (api.md section 7) on 3 test images with the real data.

The lowest-IOD copy of each image is made again from its seed and run the api's way
(analyze_for_model, crop, DINOv2, the heads of api/bundle/); predictions and
landmark_value must match those of `train_ratio_heads.py evaluate` within 0.0005.
Skips without a trained bundle, the evaluate results, the images or the model files.
"""

import warnings

import pytest

import extract_features as ef
import train_ratio_heads as trh
from facemeasure import assets, bundle, paths, quality

IMAGES = 3


def _folder():
    return paths.path("outputs") / trh.OUT_NAME


def _ready() -> str | None:
    if not (_folder() / trh.PREDICTIONS_NAME).is_file():
        return "no evaluate results in outputs/ratio_heads/"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bundle.BundleWarning)
            b = bundle.load_bundle()
    except bundle.BundleError as e:
        return f"no bundle: {e}"
    if any(b.heads[t].stub for t in bundle.RATIO_HEADS):
        return "api/bundle/ still has stub ratio heads; run train_ratio_heads.py bundle"
    try:
        for name in ("face_landmarker.task", "selfie_multiclass_256x256.tflite", "dinov2_vitb14_pretrain.pth"):
            assets.asset_path(name)
        assets.extracted("dinov2_src")
    except (assets.AssetError, FileNotFoundError):
        return "no model files in api/assets/"
    return None


def test_bundle_gives_the_values_of_evaluate_on_copies_made_again():
    why = _ready()
    if why:
        pytest.skip(why)
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bundle.BundleWarning)
        b = bundle.load_bundle()
    rows = trh.pick_fidelity(trh.read_predictions(_folder()), IMAGES)
    image_paths = {s.image: s.path for s in ef.face_sources()}
    missing = [n for n in rows["image"] if not image_paths.get(n, None) or not image_paths[n].is_file()]
    if missing:
        pytest.skip(f"SCUT-FBP5500 images missing, for example {missing[0]}")
    with FaceModels() as models:
        check = trh.fidelity_check(b, rows, models, Backbone(), quality.load_config(), image_paths)
    assert check["images"] == IMAGES
    assert check["passed"], check["max_by_value"]

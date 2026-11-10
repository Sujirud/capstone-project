"""Stored features against copies made again from their seeds (api.md section 7, first fidelity step).

Each sampled row is made again: the source image, analyze_frontal for its IOD, the copy from
its seed, then the api's way in (model_input.analyze_for_model) and DINOv2. The features
must match the stored ones with cosine similarity of at least 0.9999. Skips without
data/features/, the images or the model files.
"""

import json

import numpy as np
import pytest

import extract_features as ef
from facemeasure import assets, crops, degrade, paths, quality
from facemeasure.backbone import FEATURE_TYPES

MIN_COSINE = 0.9999
CONFIG = quality.load_config()


def _ready(task: str) -> bool:
    if not (ef.task_folder(task) / ef.INDEX_NAME).is_file():
        return False
    try:
        for name in ("face_landmarker.task", "selfie_multiclass_256x256.tflite", "dinov2_vitb14_pretrain.pth"):
            assets.asset_path(name)
        assets.extracted("dinov2_src")
        return True
    except (assets.AssetError, FileNotFoundError):
        return False


@pytest.fixture(scope="module")
def runtime():
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    with FaceModels() as models:
        yield models, Backbone()


def sample(task: str, index) -> list:
    """A few rows with features: train copies and both fixed copies, both sides for the under-eye."""
    have = index[index["row"].notna()]
    out = []
    for kind, copy in (("train", None), ("fixed", 1), ("fixed", 2)):
        part = have[(have["kind"] == kind) & ((have["copy"] == copy) if copy else True)]
        if task == ef.UNDEREYE:
            images = sorted(part["image"].unique())[:: max(1, part["image"].nunique() // 2)][:2]
            part = part[part["image"].isin(images)].groupby(["image", "side"]).head(1)
        else:
            part = part.iloc[:: max(1, len(part) // 2)][:2]
        out += [r for _, r in part.iterrows()]
    return out


@pytest.mark.parametrize("task", ef.TASKS)
def test_copies_from_seeds_give_the_stored_features(task, request):
    if not _ready(task):
        pytest.skip(f"no features in data/features/{task}/ or no model files")
    with open(paths.path("features") / ef.META_NAME, encoding="utf-8") as f:
        version = json.load(f)["degrade"]["version"]
    assert version == degrade.DEGRADE_VERSION, (
        f"data/features/ was made with DEGRADE_VERSION {version}; run extract_features.py {task} --restart")
    models, bb = request.getfixturevalue("runtime")
    from facemeasure.frontal import analyze_frontal
    from facemeasure.model_input import analyze_for_model

    index = ef.read_index(ef.task_folder(task))
    paths_by = {s.image: s.path for s in ef.sources_for(task)}
    workers = ef.Workers(
        analyze=lambda im: analyze_frontal(im, models, CONFIG.frontal),
        model_input=lambda im: analyze_for_model(im, models, degrade.TRAIN_IOD_MAX, CONFIG.frontal),
        embed=bb.features,
        device=str(bb.device),
    )
    rows = sample(task, index)
    assert rows
    got = []
    for r in rows:
        _, result = ef.regenerate(task, r["image"], paths_by[r["image"]], r["kind"], int(r["copy"]), workers, CONFIG)
        assert result.face_count == r["n_faces"]
        assert round(result.quality.iod_px, 3) == pytest.approx(r["iod_px"], abs=1e-3)
        got.append(ef.crop_of(task, result, r.get("side")))
    new = bb.features(np.stack(got))
    for f in FEATURE_TYPES:
        stored = []
        for r in rows:
            with np.load(ef.shard_path(ef.task_folder(task), int(r["shard"])), allow_pickle=False) as z:
                stored.append(z[f][int(r["row"])].astype(np.float32))
        stored = np.stack(stored)
        cos = (new[f] * stored).sum(1) / (np.linalg.norm(new[f], axis=1) * np.linalg.norm(stored, axis=1))
        assert cos.min() >= MIN_COSINE, (f, cos.min())


def test_meta_records_what_the_heads_need():
    path = paths.path("features") / ef.META_NAME
    if not path.is_file():
        pytest.skip("no data/features/meta.json yet")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    low, high = degrade.iod_range(CONFIG)
    assert meta["degrade"]["version"] == degrade.DEGRADE_VERSION, "data/features/ is from another degradation; run with --restart"
    assert meta["crop_version"] == crops.CROP_VERSION and meta["degrade"]["version"] == degrade.DEGRADE_VERSION
    assert meta["train_face_iod_px"] == {"min": low, "max": high}
    assert meta["degrade"]["fixed"]["iod_px"] == list(degrade.fixed_iod(CONFIG))
    assert meta["backbone"]["name"] == "dinov2_vitb14" and meta["backbone"]["features"] == {"cls": 768, "cls_mean": 1536, "grid4": 12288}
    assert set(meta["runtime"]) == {"python", "torch", "mediapipe", "facemeasure"}
    for task, entry in meta["tasks"].items():
        assert task in ef.TASKS and entry["devices"] and entry["crops"] >= entry["with_features"]

"""Stub bundle, so the api runs before the heads are trained.

Run from the project root:

    uv run python model/scripts/make_stub_bundle.py

Writes api/bundle/ with version 0.0.0-stub, over the trained bundle that git holds (git
restore api/bundle brings that back; tests/test_repo_privacy.py fails if a stub is added
to git). Every head is a stub on the
DINOv2 class token (cls, 768 dimensions) with zero weights, so it always gives the same
answer: the jaw and alar base heads the median ratio of data/labels/scut_labels.csv
(STUB_FALLBACK when the file is missing), and the under-eye head P(grade >= 1) = 0.5 and
P(grade >= 2) = 0.2 at temperature 1. The bands and thresholds are the defaults of
criteria.py, the baseline is slope 1 and intercept 0, and train_face_iod_px and degrade
are null, so the api does not downscale. Running it again gives the same heads.npz.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from facemeasure import bundle, codes, criteria, paths

STUB_VERSION = "0.0.0-stub"
STUB_FEATURE = "cls"
STUB_FEATURE_DIM = 768
LABELS_NAME = "scut_labels.csv"
LABEL_COLUMNS = {codes.JAW: "jaw_ratio", codes.NOSE_BASE: "nose_ratio"}
STUB_FALLBACK = {codes.JAW: 0.85, codes.NOSE_BASE: 1.09}
STUB_UNDEREYE_P = (0.5, 0.2)  # P(grade >= 1), P(grade >= 2)
# Normalization of the DINOv2 models (ImageNet mean and std); extract_features.py records the one it uses
STUB_NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
RATIO_MEASURE = {codes.JAW: codes.FACIAL_TAPERNESS, codes.NOSE_BASE: codes.ALAR_INTERCANTHAL_RATIO}


def label_medians(labels: Path | None = None) -> tuple[dict[str, float], bool]:
    """Median jaw and alar base ratio of the labels, and whether the file was there."""
    labels = Path(labels) if labels is not None else paths.path("labels") / LABELS_NAME
    if not labels.is_file():
        return dict(STUB_FALLBACK), False
    table = pd.read_csv(labels, usecols=list(LABEL_COLUMNS.values()), encoding="utf-8")
    out = {}
    for head, column in LABEL_COLUMNS.items():
        values = table[column].dropna()
        if values.empty:
            raise ValueError(f"{labels.name} has no {column} values")
        out[head] = float(values.median())
    return out, True


def stub_parts(medians: dict[str, float] | None = None, lock: Path | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Manifest (without files) and arrays of the stub; medians default to STUB_FALLBACK."""
    medians = dict(STUB_FALLBACK) if medians is None else medians
    p1, p2 = STUB_UNDEREYE_P
    low, high = criteria.UNDEREYE_THRESHOLDS
    arrays: dict[str, np.ndarray] = {}
    heads: dict[str, dict] = {}
    for name in bundle.RATIO_HEADS:
        arrays[f"{name}_weights"] = np.zeros(STUB_FEATURE_DIM)
        arrays[f"{name}_bias"] = np.array(float(medians[name]))
        heads[name] = {
            "feature": STUB_FEATURE,
            "stub": True,
            "weights": f"{name}_weights",
            "bias": f"{name}_bias",
            "rmse": None,
            "band": criteria.DEFAULT_BANDS[RATIO_MEASURE[name]],
        }
    u = bundle.UNDEREYE_HEAD
    arrays[f"{u}_weights"] = np.zeros(STUB_FEATURE_DIM)
    arrays[f"{u}_bias"] = np.array([math.log(p / (1 - p)) for p in (p1, p2)])  # sigmoid(b_k / 1) = p_k
    heads[u] = {
        "feature": STUB_FEATURE,
        "stub": True,
        "weights": f"{u}_weights",
        "bias": f"{u}_bias",
        "temperature": 1.0,
        "thresholds": {"low": low, "high": high},
    }
    manifest = bundle.new_manifest(
        version=STUB_VERSION,
        heads=heads,
        baseline={name: {"slope": 1.0, "intercept": 0.0} for name in bundle.RATIO_HEADS},
        backbone={
            "name": bundle.BACKBONE_NAME,
            "image_size": bundle.BACKBONE_IMAGE_SIZE,
            "normalize": STUB_NORMALIZE,
            "features": {STUB_FEATURE: STUB_FEATURE_DIM},
        },
        lock=lock,
    )
    return manifest, arrays


def stub_bundle(medians: dict[str, float] | None = None, lock: Path | None = None) -> bundle.Bundle:
    """The stub in memory, for tests and fixtures; files is filled with a placeholder sha256."""
    manifest, arrays = stub_parts(medians, lock)
    manifest["files"] = {bundle.HEADS_NAME: "0" * 64}
    return bundle.parse(manifest, arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a stub api/bundle/ so the api runs before training")
    parser.add_argument("--out", type=Path, default=None, help="bundle folder (default: bundle in config.toml)")
    parser.add_argument("--labels", type=Path, default=None, help="SCUT labels CSV (default: labels/scut_labels.csv)")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else bundle.bundle_dir()
    medians, found = label_medians(args.labels)
    if not found:
        print(f"ไม่พบ {LABELS_NAME} จึงใช้ bias ค่าเริ่มต้น {STUB_FALLBACK[codes.JAW]} และ {STUB_FALLBACK[codes.NOSE_BASE]}")
    manifest, arrays = stub_parts(medians)
    b = bundle.write_bundle(manifest, arrays, out)
    print(f"เขียน bundle รุ่น {b.version} ที่ {out} แล้ว และโหลดกลับได้")
    print(f"bias กราม {medians[codes.JAW]:.4f} ฐานจมูก {medians[codes.NOSE_BASE]:.4f} หัวที่เป็น stub: {', '.join(b.stub_heads)}")
    print("api จะรายงานรุ่นนี้ใน versions.model_bundle จนกว่า train_ratio_heads.py และ train_undereye_head.py จะเขียน bundle ที่เทรนแล้ว")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

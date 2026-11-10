"""Webcam-like copies and DINOv2 features of the training crops (model.md sections 3.3, 5.3 and 6).

Run from the project root, one command at a time:

    uv run python model/scripts/extract_features.py face
    uv run python model/scripts/extract_features.py undereye
    uv run python model/scripts/extract_features.py report

- face: the whole-face crop of every SCUT-FBP5500 image with a label in
  labels/scut_labels.csv. Train images get 4 train copies, val and test images 2 fixed
  copies (IOD at the bottom and at the top of the training range).
- undereye: both under-eye crops of the graded images of grading/manifest.csv. Train
  images get 6 train copies, val and test images 2 fixed copies, and SCUT train images
  2 fixed copies more for the cross-dataset check of the under-eye head (model.md section 5.3).
  Grades are not needed yet.
- report: model/reports/features.md and check images; see features_report.py.

Every copy takes the api's steps: the source image goes through analyze_frontal once
for its IOD, then each copy through degrade.degrade, model_input.analyze_for_model with
TRAIN_IOD_MAX (the api's way in, so the crops, ratios and quality codes come from the
degraded image, as the api finds them in a camera image), the crops of crops.py and
backbone.Backbone. All in one process: MediaPipe's landmarker cannot move to another
process, and Windows starts new processes with spawn. DINOv2 runs in batches of 32 on
CUDA or MPS and 8 on the CPU.

Output in features/<task>/ (not in git):
- shard_NNNN.npz: cls and cls_mean (float32) and grid4 (float16) of about SHARD_CROPS
  crops; all copies of an image stay in one shard. Each shard also holds its index rows,
  its source rows and the device as JSON, and is written to a .tmp file, then moved
  into place, so a run that stops goes on from the last finished shard.
- index.csv: one row per crop, also for copies without a face (n_faces 0, no shard and
  no row). shard and row point at the features. Labels are not stored; the training scripts
  join them by image name.
- sources.csv: faces and IOD that analyze_frontal found on each source image.
- plan.json: the settings of the run; if any of them changes, the script stops and asks
  for --restart.
features/meta.json records the backbone, CROP_VERSION, the degradation, the IOD range,
the versions and the devices of both tasks.
"""

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import mediapipe
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import grading_set
from facemeasure import (
    assets,
    codes,
    crops,
    degrade,
    image_io,
    mp_points,
    paths,
    quality,
    scut,
)
from facemeasure.frontal import FrontalResult, analyze_frontal
from facemeasure.landmarks import LANDMARKER_FILE, SEGMENTER_FILE, FaceModels

PLAN_VERSION = 1
SHARD_CROPS = 1000
FACE, UNDEREYE = degrade.FACE, degrade.UNDEREYE
TASKS = degrade.TASKS
TRAIN, FIXED = degrade.TRAIN, degrade.FIXED
FACE_TRAIN_COPIES = 4
UNDEREYE_TRAIN_COPIES = 6
SCUT_CHECK_COPIES = 2  # fixed copies of SCUT train images for the under-eye cross check
SPLITS = ("train", "val", "test")
SIDES = ("left", "right")  # person's side, as person_side of the grading tool
SIDE_REGION = {"left": codes.UNDEREYE_LEFT, "right": codes.UNDEREYE_RIGHT}
CROPS_PER_COPY = {FACE: 1, UNDEREYE: 2}
FEATURE_DTYPES = {"cls": np.float32, "cls_mean": np.float32, "grid4": np.float16}
ASSET_FILES = [LANDMARKER_FILE, SEGMENTER_FILE]

PLAN_NAME = "plan.json"
INDEX_NAME = "index.csv"
SOURCES_NAME = "sources.csv"
META_NAME = "meta.json"
LABELS_NAME = "scut_labels.csv"
MANIFEST_NAME = "manifest.csv"

FACE_COLUMNS = [
    "image", "group", "split", "kind", "copy", "flipped", "seed", "target_iod", "iod_px", "n_faces",
    "reasons_jaw", "reasons_nose_base", "jaw_ratio_mp", "nose_ratio_mp", "shard", "row",
]
UNDEREYE_COLUMNS = [
    "image", "source", "side", "split", "kind", "copy", "seed", "target_iod", "iod_px", "n_faces",
    "reasons", "hair", "shard", "row",
]
COLUMNS = {FACE: FACE_COLUMNS, UNDEREYE: UNDEREYE_COLUMNS}
SOURCE_COLUMNS = ["image", "n_faces", "iod_px"]
TEXT_COLUMNS = ("reasons", "reasons_jaw", "reasons_nose_base", "group", "source", "side", "split", "kind", "image")
INT_COLUMNS = ("copy", "flipped", "seed", "n_faces", "shard", "row")


@dataclass(frozen=True)
class Source:
    """One image and the copies to make of it."""

    image: str
    path: Path
    split: str
    extra: dict = field(default_factory=dict)  # face: group; undereye: source
    copies: tuple[tuple[str, int], ...] = ()


@dataclass
class Workers:
    """What turns images into rows and features; tests pass fakes."""

    analyze: Callable[[np.ndarray], FrontalResult]  # analyze_frontal on a source image
    model_input: Callable[[np.ndarray], FrontalResult]  # analyze_for_model on a copy
    embed: Callable[[np.ndarray], dict[str, np.ndarray]]  # (n, 224, 224, 3) uint8 -> feature arrays
    device: str
    read: Callable[[Path], np.ndarray] = image_io.read_image


# ---------------------------------------------------------------- sources


def fixed_copies() -> list[tuple[str, int]]:
    return [(FIXED, k) for k in range(1, degrade.FIXED_COPIES + 1)]


def face_copies(split: str) -> tuple[tuple[str, int], ...]:
    if split == "train":
        return tuple((TRAIN, k) for k in range(1, FACE_TRAIN_COPIES + 1))
    return tuple(fixed_copies())


def undereye_copies(split: str, source: str) -> tuple[tuple[str, int], ...]:
    if split != "train":
        return tuple(fixed_copies())
    out = [(TRAIN, k) for k in range(1, UNDEREYE_TRAIN_COPIES + 1)]
    if source == "scut":
        out += fixed_copies()[:SCUT_CHECK_COPIES]
    return tuple(out)


def face_sources() -> list[Source]:
    labels = pd.read_csv(paths.path("labels") / LABELS_NAME, dtype=str, keep_default_na=False, encoding="utf-8")
    folder = paths.path("scut") / "Images"
    out = []
    for image, group, split in zip(labels["image"], labels["group"], labels["split"], strict=True):
        if split not in SPLITS:
            raise SystemExit(f"{LABELS_NAME}: {image} has split {split!r}")
        out.append(Source(image, folder / image, split, {"group": group}, face_copies(split)))
    return sorted(out, key=lambda s: (scut.GROUPS.index(scut.parse_name(s.image).group), scut.parse_name(s.image).number))


def undereye_sources() -> list[Source]:
    manifest = grading_set.read_manifest(paths.path("grading") / MANIFEST_NAME)
    graded = manifest[manifest["role"] == "graded"].sort_values(["source", "image"])
    out = []
    for _, r in graded.iterrows():
        if r["split"] not in SPLITS:
            raise SystemExit(f"{MANIFEST_NAME}: graded image {r['image']} has split {r['split']!r}")
        out.append(
            Source(r["image"], grading_set.image_path(r), r["split"], {"source": r["source"]}, undereye_copies(r["split"], r["source"]))
        )
    return out


def sources_for(task: str) -> list[Source]:
    return face_sources() if task == FACE else undereye_sources()


def count_crops(task: str, sources: Sequence[Source]) -> int:
    return sum(len(s.copies) for s in sources) * CROPS_PER_COPY[task]


# ---------------------------------------------------------------- plan


def shard_plan(task: str, sources: Sequence[Source], shard_crops: int = SHARD_CROPS) -> list[list[int]]:
    """Source indices of each shard: about shard_crops crops, never splitting the copies of one image."""
    shards, current, n = [], [], 0
    for k, s in enumerate(sources):
        current.append(k)
        n += len(s.copies) * CROPS_PER_COPY[task]
        if n >= shard_crops:
            shards.append(current)
            current, n = [], 0
    if current:
        shards.append(current)
    return shards


def sources_sha256(sources: Sequence[Source]) -> str:
    lines = [f"{s.image}|{s.split}|{json.dumps(s.extra, sort_keys=True)}|{s.copies}" for s in sources]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def common_settings(config: quality.QualityConfig, backbone_info: dict) -> dict:
    """What both tasks must share, and what meta.json reports."""
    low, high = degrade.iod_range(config)
    return {
        "backbone": backbone_info,
        "crop_version": crops.CROP_VERSION,
        "degrade": degrade.settings(config),
        "train_face_iod_px": {"min": low, "max": high},
        "assets": assets.fingerprint(ASSET_FILES),
        "alar_base": mp_points.pair_key(mp_points.ALAR_BASE) if mp_points.ALAR_BASE else None,
        "mediapipe": mediapipe.__version__,
        "torch": torch.__version__,
    }


def plan_settings(task: str, sources: Sequence[Source], config: quality.QualityConfig, backbone_info: dict,
                  shard_crops: int = SHARD_CROPS) -> dict:
    return {
        "plan_version": PLAN_VERSION,
        "task": task,
        "shard_crops": shard_crops,
        "quality": asdict(config),
        **common_settings(config, backbone_info),
        "sources": len(sources),
        "sources_sha256": sources_sha256(sources),
    }


def shard_path(folder: Path, k: int) -> Path:
    return folder / f"shard_{k:04d}.npz"


def prepare(folder: Path, want: dict, restart: bool) -> None:
    """Check plan.json against the settings, or start a new run."""
    folder.mkdir(parents=True, exist_ok=True)
    for p in folder.glob("*.tmp"):
        p.unlink()
    path = folder / PLAN_NAME
    if restart:
        for p in [*folder.glob("shard_*.npz"), folder / INDEX_NAME, folder / SOURCES_NAME, path]:
            p.unlink(missing_ok=True)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
        if have != want:
            changed = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            raise SystemExit(
                f"The shards in {folder} were made with other settings ({', '.join(changed)}). "
                "Run again with --restart to extract every crop again."
            )
    else:
        _write_json(path, want)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- one shard


def _round(value, digits: int):
    return None if value is None else round(float(value), digits)


def _codes(r: FrontalResult, region: str) -> str:
    return ";".join(r.region_failed.get(region, ()))


def copy_rows(task: str, src: Source, kind: str, copy: int, params: degrade.Params, r: FrontalResult) -> tuple[list[dict], list[np.ndarray]]:
    """Index rows of one copy (shard and row still empty) and the crops that get features."""
    iod = _round(r.quality.iod_px, 3) if r.quality is not None else None
    base = {"image": src.image, **src.extra, "split": src.split, "kind": kind, "copy": copy, "seed": params.seed,
            "target_iod": round(params.target_iod, 3), "iod_px": iod, "n_faces": r.face_count, "shard": None, "row": None}
    found = r.face_count > 0 and r.crops is not None
    if task == FACE:
        row = {
            **base,
            "flipped": int(params.flip),
            "reasons_jaw": _codes(r, codes.JAW),
            "reasons_nose_base": _codes(r, codes.NOSE_BASE),
            "jaw_ratio_mp": _round(r.ratios.jaw_ratio, 6) if r.ratios is not None else None,
            "nose_ratio_mp": _round(r.ratios.nose_ratio, 6) if r.ratios is not None else None,
        }
        return [row], [r.crops.face.image] if found else []
    rows, images = [], []
    for side in SIDES:
        hair = getattr(r.quality, f"hair_undereye_{side}") if r.quality is not None else None
        rows.append({**base, "side": side, "reasons": _codes(r, SIDE_REGION[side]), "hair": _round(hair, 6)})
        if found:
            images.append(getattr(r.crops, f"undereye_{side}").image)
    return rows, images


def process_shard(task: str, sources: Sequence[Source], k: int, workers: Workers, config: quality.QualityConfig, bar=None):
    """Rows, source rows and features of one shard."""
    rows, source_rows, images = [], [], []
    for src in sources:
        image = workers.read(src.path)
        first = workers.analyze(image)
        source_iod = first.quality.iod_px if first.face_count and first.quality is not None else None
        source_rows.append({"image": src.image, "n_faces": first.face_count, "iod_px": _round(source_iod, 3)})
        for kind, copy in src.copies:
            params = degrade.copy_params(task, src.image, kind, copy, config)
            copy_image = degrade.degrade(image, source_iod, params).image
            r = workers.model_input(copy_image)
            new_rows, new_images = copy_rows(task, src, kind, copy, params, r)
            if new_images:
                for row in new_rows:
                    row["shard"], row["row"] = k, len(images) + (SIDES.index(row["side"]) if task == UNDEREYE else 0)
                images.extend(new_images)
            rows.extend(new_rows)
            if bar is not None:
                bar.update(len(new_rows))
    if images:
        feats = workers.embed(np.stack(images))
    else:
        feats = {name: np.zeros((0, dim)) for name, dim in workers_dims().items()}
    return rows, source_rows, {name: np.asarray(feats[name], dtype=dt) for name, dt in FEATURE_DTYPES.items()}


def workers_dims() -> dict[str, int]:
    from facemeasure.backbone import FEATURES

    return FEATURES


def write_shard(path: Path, rows: list[dict], source_rows: list[dict], features: dict[str, np.ndarray], device: str) -> None:
    n = sum(1 for r in rows if r["row"] is not None)
    for name, a in features.items():
        if len(a) != n:
            raise RuntimeError(f"{path.name}: {name} has {len(a)} rows, the index points at {n}")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(
            f,
            **features,
            index=np.array(json.dumps(rows, ensure_ascii=False, allow_nan=False)),
            sources=np.array(json.dumps(source_rows, ensure_ascii=False, allow_nan=False)),
            device=np.array(device),
        )
    tmp.replace(path)


def read_shard_meta(path: Path) -> tuple[list[dict], list[dict], str]:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["index"])), json.loads(str(z["sources"])), str(z["device"])


# ---------------------------------------------------------------- whole task


def run_task(
    task: str,
    sources: Sequence[Source],
    folder: Path,
    workers: Workers,
    settings: dict,
    config: quality.QualityConfig,
    restart: bool = False,
    shard_crops: int = SHARD_CROPS,
) -> dict:
    """Make every missing shard, then index.csv and sources.csv; return counts for meta.json."""
    plan = shard_plan(task, sources, shard_crops)
    want = {**settings, "shards": len(plan)}
    prepare(folder, want, restart)
    sizes = [count_crops(task, [sources[i] for i in shard]) for shard in plan]
    todo = [k for k in range(len(plan)) if not shard_path(folder, k).is_file()]
    done = sum(sizes[k] for k in range(len(plan)) if k not in todo)
    if todo:
        with tqdm(total=sum(sizes), initial=done, unit="crop", desc=task, smoothing=0.05) as bar:
            for k in todo:
                rows, source_rows, feats = process_shard(task, [sources[i] for i in plan[k]], k, workers, config, bar)
                write_shard(shard_path(folder, k), rows, source_rows, feats, workers.device)
    return write_index(task, folder, len(plan))


def write_index(task: str, folder: Path, n_shards: int) -> dict:
    rows, source_rows, devices = [], [], []
    for k in range(n_shards):
        r, s, d = read_shard_meta(shard_path(folder, k))
        rows += r
        source_rows += s
        devices.append(d)
    index = pd.DataFrame(rows, columns=COLUMNS[task])
    for c in INT_COLUMNS:
        if c in index:
            index[c] = index[c].astype("Int64")
    _write_csv(index, folder / INDEX_NAME)
    sources_df = pd.DataFrame(source_rows, columns=SOURCE_COLUMNS)
    sources_df["n_faces"] = sources_df["n_faces"].astype("Int64")
    _write_csv(sources_df, folder / SOURCES_NAME)
    with_features = int(index["row"].notna().sum())
    return {
        "images": len(sources_df),
        "copies": int(len(index) // CROPS_PER_COPY[task]),
        "crops": len(index),
        "with_features": with_features,
        "without_face": int(len(index) - with_features),
        "shards": n_shards,
        "devices": sorted(set(devices)),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    tmp.replace(path)


# ---------------------------------------------------------------- one copy again


def regenerate(task: str, image_name: str, path: Path, kind: str, copy: int, workers: Workers,
               config: quality.QualityConfig) -> tuple[np.ndarray, FrontalResult]:
    """One stored copy made again from its seed: the degraded image and what the api's way in finds on it."""
    image = workers.read(path)
    first = workers.analyze(image)
    source_iod = first.quality.iod_px if first.face_count and first.quality is not None else None
    params = degrade.copy_params(task, image_name, kind, copy, config)
    copy_image = degrade.degrade(image, source_iod, params).image
    return copy_image, workers.model_input(copy_image)


def crop_of(task: str, r: FrontalResult, side: str | None = None) -> np.ndarray:
    """The crop a row's features come from: the whole face, or the under-eye of the person's side."""
    if task == FACE:
        return r.crops.face.image
    return getattr(r.crops, f"undereye_{side}").image


# ---------------------------------------------------------------- reading


def task_folder(task: str) -> Path:
    return paths.path("features") / task


def read_index(folder: Path) -> pd.DataFrame:
    """index.csv with text columns as text ('' for none) and integer columns as Int64."""
    df = pd.read_csv(folder / INDEX_NAME, dtype={c: str for c in TEXT_COLUMNS}, keep_default_na=False, na_values=[""], encoding="utf-8")
    for c in TEXT_COLUMNS:
        if c in df:
            df[c] = df[c].fillna("")
    for c in INT_COLUMNS:
        if c in df:
            df[c] = df[c].astype("Int64")
    return df


def load_features(folder: Path, feature: str, index: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows of index.csv that have features, and their (n, dim) float32 matrix in the same order."""
    if feature not in FEATURE_DTYPES:
        raise ValueError(f"feature must be one of {list(FEATURE_DTYPES)}, got {feature!r}")
    index = read_index(folder) if index is None else index
    rows = index[index["row"].notna()].reset_index(drop=True)
    out = np.empty((len(rows), workers_dims()[feature]), dtype=np.float32)
    for shard, part in rows.groupby("shard", sort=True):
        with np.load(shard_path(folder, int(shard)), allow_pickle=False) as z:
            out[part.index.to_numpy()] = z[feature][part["row"].to_numpy(dtype=np.int64)]
    return rows, out


# ---------------------------------------------------------------- meta.json


def write_meta(task: str, common: dict, counts: dict, copies: dict) -> Path:
    """features/meta.json: the common settings and one entry per task made with them."""
    path = paths.ensure_dir("features") / META_NAME
    common_sha = hashlib.sha256(json.dumps(common, sort_keys=True).encode("utf-8")).hexdigest()
    tasks = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        for name, entry in old.get("tasks", {}).items():
            if name == task:
                continue
            if entry.get("common_sha256") == common_sha:
                tasks[name] = entry
            else:
                print(f"features/{name} was made with other settings and is dropped from {META_NAME}; "
                      f"run: uv run python model/scripts/extract_features.py {name} --restart", file=sys.stderr)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    tasks[task] = {"created": now, "common_sha256": common_sha, "copies_per_image": copies, **counts}
    meta = {
        "updated": now,
        "backbone": common["backbone"],
        "crop_version": common["crop_version"],
        "degrade": common["degrade"],
        "train_face_iod_px": common["train_face_iod_px"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mediapipe": mediapipe.__version__,
            "facemeasure": metadata.version("facemeasure"),
        },
        "assets": common["assets"],
        "alar_base": common["alar_base"],
        "tasks": dict(sorted(tasks.items())),
    }
    _write_json(path, meta)
    return path


COPIES_TEXT = {
    FACE: {"train": f"{FACE_TRAIN_COPIES} train", "val": "2 fixed", "test": "2 fixed"},
    UNDEREYE: {"train": f"{UNDEREYE_TRAIN_COPIES} train (+{SCUT_CHECK_COPIES} fixed for SCUT)", "val": "2 fixed", "test": "2 fixed"},
}


# ---------------------------------------------------------------- main


def real_workers(models: FaceModels, backbone, config: quality.QualityConfig) -> Workers:
    from facemeasure.model_input import analyze_for_model

    def analyze(image):
        return analyze_frontal(image, models, config.frontal)

    def model_input(image):
        return analyze_for_model(image, models, degrade.TRAIN_IOD_MAX, config.frontal)

    return Workers(analyze=analyze, model_input=model_input, embed=backbone.features, device=str(backbone.device))


def extract(task: str, restart: bool) -> None:
    from facemeasure.backbone import Backbone

    config = quality.load_config()
    sources = sources_for(task)
    if not sources:
        raise SystemExit(f"no images for {task}; grading_set.py and make_scut_labels.py make the lists")
    missing = [s.path for s in sources if not s.path.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing, for example {missing[0]}")
    backbone = Backbone()
    common = common_settings(config, backbone.info)
    settings = plan_settings(task, sources, config, backbone.info)
    folder = task_folder(task)
    print(f"{task}: {len(sources)} images, {count_crops(task, sources)} crops, DINOv2 on {backbone.device}")
    with FaceModels() as models:
        counts = run_task(task, sources, folder, real_workers(models, backbone, config), settings, config, restart)
    meta = write_meta(task, common, counts, COPIES_TEXT[task])
    print(
        f"{task}: {counts['crops']} crops, {counts['with_features']} with features, {counts['without_face']} without a face; "
        f"{counts['shards']} shards in {folder}; {meta}"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Webcam-like copies and DINOv2 features of the SCUT and graded crops")
    sub = ap.add_subparsers(dest="command", required=True)
    for task in TASKS:
        p = sub.add_parser(task, help=f"extract the {task} features")
        p.add_argument("--restart", action="store_true", help="drop the finished shards and extract every crop again")
    p = sub.add_parser("report", help="model/reports/features.md and the check images")
    p.add_argument("--no-check-images", action="store_true", help="skip outputs/features_check/")
    args = ap.parse_args(argv)
    if args.command == "report":
        import features_report

        features_report.run(check_images=not args.no_check_images)
        return
    extract(args.command, args.restart)


if __name__ == "__main__":
    main()

"""Run analyze_frontal over SCUT-FBP5500 and the graded FFHQ images to check every step.

Run from the project root:

    uv run python model/scripts/frontal_check.py

Writes
- labels/frontal_mediapipe.csv      one row per image: faces, quality values, failed codes, ratios
- features/landmarks_frontal.npz    478 points, blendshapes and matrix of the measured face
- reports/frontal_mediapipe.md      counts, pose sign checks, Face++ comparison, ratio correlations
  and hair fractions, with figures/frontal_*.png; numbers and charts only, so it can be committed
- outputs/frontal_check/            check images with the points, boxes and crops; faces, never commit

Every image goes through facemeasure.frontal.analyze_frontal, the same call as the api.
Work is saved in parts of PART_SIZE images under features/frontal_parts/, so a run that
stops goes on from the last finished part. The parts record the crop version, the
quality limits, the model files and the image list; if any of them changes the script
stops and asks for --restart. Ratio correlations use the SCUT train split only.
"""

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mediapipe
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from tqdm import tqdm

import grading_set
from facemeasure import (
    assets,
    codes,
    crops,
    draw,
    image_io,
    mp_points,
    paths,
    quality,
    scut,
)
from facemeasure.face_geometry import GeometryError, catmull_rom, measure_jaw_nose
from facemeasure.frontal import FrontalResult, analyze_frontal
from facemeasure.landmarks import LANDMARKER_FILE, SEGMENTER_FILE, FaceModels

PARTS_VERSION = 1
PART_SIZE = 200
PARTS_FOLDER = "frontal_parts"
SETTINGS_NAME = "settings.json"
CSV_NAME = "frontal_mediapipe.csv"
NPZ_NAME = "landmarks_frontal.npz"
REPORT_NAME = "frontal_mediapipe.md"
FIGURES_FOLDER = "figures"
CHECK_FOLDER = "frontal_check"
SEED = "frontal-check-v1"
CHECK_PER_GROUP = 12  # SCUT: 4 groups, FFHQ: 4 age bands -> 48 each
NOT_ONE_FACE_SHEET = 48
FFHQ_ROLES = ("graded", "practice")
SOURCES = ("scut", "ffhq")
LABELS_FILE = "ffhq_aging_labels.csv"

QUALITY_COLUMNS = [
    "iod_px", "yaw_deg", "pitch_deg", "roll_deg", "smile", "jaw_open",
    "blink_left", "blink_right", "hair_jaw", "hair_undereye_left", "hair_undereye_right",
]
PAIR_KEYS = [mp_points.pair_key(p) for p in mp_points.ALAR_BASE_PAIRS]
NOSE_COLUMNS = [f"nose_ratio_{k}" for k in PAIR_KEYS]
COLUMNS = [
    "image", "source", "group", "role", "split", "width", "height", "read_error", "face_count",
    *QUALITY_COLUMNS, "failed", "jaw_ratio", "nose_ratio", *NOSE_COLUMNS, "ratios_error",
]
DECIMALS = {"iod_px": 3, "yaw_deg": 3, "pitch_deg": 3, "roll_deg": 3}
DEFAULT_DECIMALS = 6

CHECK_SIZE = 672  # the drawn image; the three crops stack to the same height
PAIR_COLORS = [(255, 60, 60), (255, 150, 0), (230, 60, 230), (0, 220, 220), (170, 255, 60), (255, 255, 255), (255, 150, 190)]
BOX_COLOR = (0, 200, 255)
FACE_BOX_COLOR = (255, 255, 255)
EDGE_COLOR = (60, 150, 255)


@dataclass(frozen=True)
class Item:
    image: str
    source: str
    group: str  # SCUT group (AF, AM, CF, CM) or FFHQ age band
    role: str  # FFHQ graded or practice; empty for SCUT
    split: str  # SCUT label split, or the FFHQ under-eye split
    path: Path


# ---------------------------------------------------------------- images


def scut_items() -> list[Item]:
    folder = paths.path("scut") / "Images"
    labels_path = paths.path("labels") / "scut_labels.csv"
    splits = {}
    if labels_path.is_file():
        labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False, encoding="utf-8")
        splits = dict(zip(labels["image"], labels["split"], strict=True))
    names = sorted((p.name for p in folder.glob("*.jpg") if scut.NAME.match(p.name)), key=_scut_key)
    return [Item(n, "scut", scut.parse_name(n).group, "", splits.get(n, ""), folder / n) for n in names]


def _scut_key(name: str) -> tuple[int, int]:
    s = scut.parse_name(name)
    return (scut.GROUPS.index(s.group), s.number)


def ffhq_items() -> list[Item]:
    manifest = grading_set.read_manifest(paths.path("grading") / "manifest.csv")
    rows = manifest[(manifest["source"] == "ffhq") & manifest["role"].isin(FFHQ_ROLES)].sort_values("image")
    return [
        Item(r["image"], "ffhq", r["age_band"], r["role"], r["split"], grading_set.image_path(r))
        for _, r in rows.iterrows()
    ]


def all_items() -> list[Item]:
    return scut_items() + ffhq_items()


# ---------------------------------------------------------------- parts


def settings(items: list[Item], config: quality.QualityConfig) -> dict:
    names = "\n".join(f"{i.source}/{i.image}" for i in items)
    return {
        "parts_version": PARTS_VERSION,
        "part_size": PART_SIZE,
        "crop_version": crops.CROP_VERSION,
        "quality": asdict(config),
        "models": assets.fingerprint([LANDMARKER_FILE, SEGMENTER_FILE]),
        "mediapipe": mediapipe.__version__,
        "alar_pairs": PAIR_KEYS,
        "images": len(items),
        "images_sha256": hashlib.sha256(names.encode("utf-8")).hexdigest(),
    }


def part_paths(folder: Path, k: int) -> tuple[Path, Path]:
    return folder / f"part_{k:04d}.csv", folder / f"part_{k:04d}.npz"


def prepare_parts(folder: Path, want: dict, restart: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / SETTINGS_NAME
    if restart:
        for p in folder.glob("part_*"):
            p.unlink()
        path.unlink(missing_ok=True)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
        if have != want:
            changed = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
            raise SystemExit(
                f"ผลที่ค้างไว้ใน {folder} ทำด้วยค่าที่ต่างจากตอนนี้ ({', '.join(changed)}) "
                "ให้รันใหม่ด้วย --restart"
            )
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(want, ensure_ascii=False, indent=2) + "\n")


def nan_array(shape) -> np.ndarray:
    return np.full(shape, np.nan)


def result_row(item: Item, r: FrontalResult | None, read_error: str = "") -> dict:
    row = {c: None for c in COLUMNS}
    row.update(image=item.image, source=item.source, group=item.group, role=item.role, split=item.split, read_error=read_error)
    if r is None:
        return row
    row.update(width=r.width, height=r.height, face_count=r.face_count, failed=";".join(r.failed))
    if r.quality is not None:
        row.update({c: getattr(r.quality, c) for c in QUALITY_COLUMNS})
    if r.ratios is not None:
        row["jaw_ratio"] = r.ratios.jaw_ratio
        row["nose_ratio"] = r.ratios.nose_ratio
        row.update({f"nose_ratio_{k}": r.ratios.nose_ratios[k] for k in PAIR_KEYS})
    row["ratios_error"] = r.ratios_error
    return row


def run_part(items: list[Item], models: FaceModels, criteria: quality.FrontalCriteria, bar) -> tuple[pd.DataFrame, dict]:
    rows, lms, bss, mats = [], [], [], []
    names: list[str] = []
    for item in items:
        try:
            image = image_io.read_image(item.path)
        except (image_io.ImageReadError, OSError) as e:
            rows.append(result_row(item, None, type(e).__name__))
            lms.append(nan_array((mp_points.N_POINTS, 3)))
            bss.append(None)
            mats.append(nan_array((4, 4)))
            bar.update(1)
            continue
        r = analyze_frontal(image, models, criteria)
        rows.append(result_row(item, r))
        if r.face_count:
            if not names:
                names = list(r.blendshapes)
            elif list(r.blendshapes) != names:
                raise RuntimeError(f"{item.image}: blendshape names differ from the first face")
            lms.append(r.landmarks)
            bss.append(np.array([r.blendshapes[n] for n in names]))
            mats.append(r.matrix)
        else:
            lms.append(nan_array((mp_points.N_POINTS, 3)))
            bss.append(None)
            mats.append(nan_array((4, 4)))
        bar.update(1)
    width = len(names)
    blend = np.stack([b if b is not None else nan_array(width) for b in bss]) if width else np.empty((len(items), 0))
    arrays = {
        "image": np.array([i.image for i in items]),
        "source": np.array([i.source for i in items]),
        "landmarks": np.stack(lms),
        "blendshapes": blend,
        "blendshape_names": np.array(names, dtype=str),
        "matrix": np.stack(mats),
    }
    return pd.DataFrame(rows, columns=COLUMNS), arrays


def write_part(folder: Path, k: int, df: pd.DataFrame, arrays: dict) -> None:
    csv_path, npz_path = part_paths(folder, k)
    tmp_csv, tmp_npz = csv_path.with_suffix(".csv.tmp"), npz_path.with_suffix(".npz.tmp")
    df.to_csv(tmp_csv, index=False, encoding="utf-8", lineterminator="\n")
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp_npz.replace(npz_path)
    tmp_csv.replace(csv_path)  # the CSV marks the part as done, so it goes last


def run_all(items: list[Item], folder: Path, config: quality.QualityConfig) -> None:
    chunks = [items[k : k + PART_SIZE] for k in range(0, len(items), PART_SIZE)]
    todo = [k for k in range(len(chunks)) if not all(p.is_file() for p in part_paths(folder, k))]
    done = sum(len(chunks[k]) for k in range(len(chunks)) if k not in todo)
    if not todo:
        return
    with FaceModels() as models, tqdm(total=len(items), initial=done, unit="image", desc="analyze_frontal") as bar:
        for k in todo:
            df, arrays = run_part(chunks[k], models, config.frontal, bar)
            write_part(folder, k, df, arrays)


def merge_parts(folder: Path, n_parts: int) -> tuple[pd.DataFrame, dict]:
    frames, arrays = [], []
    for k in range(n_parts):
        csv_path, npz_path = part_paths(folder, k)
        frames.append(pd.read_csv(csv_path, dtype={"failed": str, "read_error": str, "ratios_error": str, "split": str, "role": str}, keep_default_na=False, na_values=[""], encoding="utf-8"))
        with np.load(npz_path) as z:
            arrays.append({key: z[key] for key in z.files})
    names = next((a["blendshape_names"] for a in arrays if len(a["blendshape_names"])), np.array([], dtype=str))
    for a in arrays:
        if len(a["blendshape_names"]) and not np.array_equal(a["blendshape_names"], names):
            raise RuntimeError("blendshape names differ between parts")
    blend = [a["blendshapes"] if a["blendshapes"].shape[1] else nan_array((len(a["image"]), len(names))) for a in arrays]
    df = pd.concat(frames, ignore_index=True)
    for c in ("failed", "read_error", "ratios_error", "split", "role"):
        df[c] = df[c].fillna("")
    merged = {
        "image": np.concatenate([a["image"] for a in arrays]),
        "source": np.concatenate([a["source"] for a in arrays]),
        "landmarks": np.concatenate([a["landmarks"] for a in arrays]),
        "blendshapes": np.concatenate(blend),
        "blendshape_names": names,
        "matrix": np.concatenate([a["matrix"] for a in arrays]),
    }
    return df, merged


def finish_table(df: pd.DataFrame) -> pd.DataFrame:
    """Fill nose_ratio from the chosen pair and round the numbers for the CSV."""
    out = df.copy()
    if mp_points.ALAR_BASE is not None:
        out["nose_ratio"] = out[f"nose_ratio_{mp_points.pair_key(mp_points.ALAR_BASE)}"]
    else:
        out["nose_ratio"] = np.nan
    for c in [*QUALITY_COLUMNS, "jaw_ratio", "nose_ratio", *NOSE_COLUMNS]:
        out[c] = pd.to_numeric(out[c]).round(DECIMALS.get(c, DEFAULT_DECIMALS))
    out["face_count"] = pd.to_numeric(out["face_count"]).astype("Int64")
    out["width"] = pd.to_numeric(out["width"]).astype("Int64")
    out["height"] = pd.to_numeric(out["height"]).astype("Int64")
    return out


def write_outputs(df: pd.DataFrame, arrays: dict, settings_used: dict) -> tuple[Path, Path]:
    csv_path = paths.ensure_dir("labels") / CSV_NAME
    npz_path = paths.ensure_dir("features") / NPZ_NAME
    tmp = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n")
    tmp.replace(csv_path)
    tmp = npz_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays, settings=np.array(json.dumps(settings_used, ensure_ascii=False)))
    tmp.replace(npz_path)
    return csv_path, npz_path


# ---------------------------------------------------------------- report numbers


def pearson(a, b) -> tuple[int, float]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return int(ok.sum()), math.nan
    return int(ok.sum()), float(np.corrcoef(a[ok], b[ok])[0, 1])


def pose_geometry(landmarks: np.ndarray) -> pd.DataFrame:
    """Image-geometry stand-ins for roll, yaw and pitch, one row per image (NaN without a face).

    eye_line_deg: angle of the line from the image-left to the image-right eye center,
    positive when the image-right eye (the person's left) is higher.
    nose_offset_iod: in the level frame, nose tip x minus the midpoint of the face edge
    at points 234 and 454, over the IOD; the nose moves to the image right when the
    person turns to their own left.
    nose_drop: in the level frame, how far down the nose tip is from eye level over how
    far down the mouth corners are; it gets smaller when the person looks up.
    """
    lm = np.asarray(landmarks, dtype=float)[:, :, :2]
    el = lm[:, list(mp_points.EYE_PERSON_RIGHT)].mean(axis=1)
    er = lm[:, list(mp_points.EYE_PERSON_LEFT)].mean(axis=1)
    d = er - el
    angle = np.arctan2(d[:, 1], d[:, 0])
    iod = np.hypot(d[:, 0], d[:, 1])
    mid = (el + er) / 2
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]

    def level(p):
        q = p - mid
        return np.stack([q[:, 0:1] * cos + q[:, 1:2] * sin, -q[:, 0:1] * sin + q[:, 1:2] * cos], axis=-1)[:, 0]

    nose = level(lm[:, mp_points.NOSE_TIP])
    edge = (level(lm[:, mp_points.CHEEKS[0]]) + level(lm[:, mp_points.CHEEKS[1]])) / 2
    mouth = (level(lm[:, mp_points.MOUTH_CORNERS[0]]) + level(lm[:, mp_points.MOUTH_CORNERS[1]])) / 2
    return pd.DataFrame(
        {
            "eye_line_deg": np.degrees(np.arctan2(-d[:, 1], d[:, 0])),
            "nose_offset_iod": (nose[:, 0] - edge[:, 0]) / iod,
            "nose_drop": nose[:, 1] / mouth[:, 1],
        }
    )


SIGN_CHECKS = [
    ("roll_deg", "eye_line_deg", 1, "roll เทียบมุมของเส้นระหว่างกลางตา"),
    ("yaw_deg", "nose_offset_iod", 1, "yaw เทียบระยะแนวนอนจากกึ่งกลางของจุด 234 กับ 454 ถึงปลายจมูก ÷ IOD"),
    ("pitch_deg", "nose_drop", -1, "pitch เทียบระยะแนวตั้งจากกลางตาถึงปลายจมูก ÷ ระยะจากกลางตาถึงมุมปาก"),
]


def sign_checks(df: pd.DataFrame, geo: pd.DataFrame) -> list[dict]:
    one = (df["face_count"] >= 1).to_numpy()
    out = []
    for angle, indicator, expected, text in SIGN_CHECKS:
        n, r = pearson(df.loc[one, angle], geo.loc[one, indicator])
        out.append({"angle": angle, "indicator": indicator, "text": text, "n": n, "r": r, "expected": expected,
                    "ok": bool(np.isfinite(r) and np.sign(r) == expected)})
    return out


def facepp_comparison(df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    labels_path = paths.path("ffhq") / LABELS_FILE
    ff = df[(df["source"] == "ffhq") & (df["face_count"] >= 1)].copy()
    if not labels_path.is_file() or ff.empty:
        return [], ff
    labels = pd.read_csv(labels_path, encoding="utf-8")
    ff["image_number"] = ff["image"].str.removesuffix(".png").astype(int)
    ff = ff.merge(labels[["image_number", "head_yaw", "head_pitch", "head_roll"]], on="image_number", how="left")
    out = []
    for ours, theirs in (("yaw_deg", "head_yaw"), ("pitch_deg", "head_pitch"), ("roll_deg", "head_roll")):
        n, r = pearson(ff[ours], ff[theirs])
        sign = 1 if not np.isfinite(r) or r >= 0 else -1
        diff = ff[ours] - sign * ff[theirs]
        out.append({"ours": ours, "facepp": theirs, "n": n, "r": r, "sign": sign,
                    "mae": float(diff.abs().mean()), "mean_diff": float(diff.mean())})
    return out, ff


def ratio_correlations(df: pd.DataFrame, config: quality.QualityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """r of each MediaPipe ratio with the SCUT label, on the train split only."""
    labels = pd.read_csv(paths.path("labels") / "scut_labels.csv", encoding="utf-8")
    train = labels[labels["split"] == "train"][["image", "group", "jaw_ratio", "nose_ratio"]]
    mp = df[(df["source"] == "scut") & (df["face_count"] >= 1)]
    m = train.merge(mp, on="image", suffixes=("_label", ""), how="inner")
    passing = m["failed"] == ""
    rows = []
    measures = [("jaw", "jaw_ratio", "jaw_ratio_label")] + [
        (f"alar {k}", f"nose_ratio_{k}", "nose_ratio_label") for k in PAIR_KEYS
    ]
    for name, ours, label in measures:
        n, r = pearson(m[ours], m[label])
        n_pass, r_pass = pearson(m.loc[passing, ours], m.loc[passing, label])
        row = {"measure": name, "column": ours, "n": n, "r": r, "n_pass": n_pass, "r_pass": r_pass,
               "mean_mp": float(m[ours].mean()), "mean_label": float(m[label].mean())}
        for g in scut.GROUPS:
            row[f"r_{g}"] = pearson(m.loc[m["group_label"] == g, ours], m.loc[m["group_label"] == g, label])[1]
        rows.append(row)
    return pd.DataFrame(rows), m


def percentiles(values, qs=(5, 50, 95)) -> list[float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, q)) if len(v) else math.nan for q in qs]


# ---------------------------------------------------------------- figures


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=100, metadata={"Software": None})


def scatter_panels(panels, title: str) -> Figure:
    """panels: list of (x, y, colors, xlabel, ylabel, panel title)."""
    fig = Figure(figsize=(14, 4.6), layout="constrained")
    axes = fig.subplots(1, len(panels))
    for ax, (x, y, colors, xl, yl, t) in zip(np.atleast_1d(axes), panels, strict=True):
        ax.scatter(x, y, s=4, c=colors, alpha=0.5, linewidths=0)
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
    fig.suptitle(title)
    return fig


SOURCE_COLORS = {"scut": "#4a7ab5", "ffhq": "#d35400"}


def write_figures(df, geo, ff, ratio_rows, ratio_data, config, folder: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    face = df["face_count"] >= 1
    colors = df.loc[face, "source"].map(SOURCE_COLORS).to_numpy()

    fig = Figure(figsize=(11, 4), layout="constrained")
    ax = fig.subplots()
    bins = np.arange(0, df["iod_px"].max() + 10, 5)
    for source in SOURCES:
        ax.hist(df.loc[face & (df["source"] == source), "iod_px"], bins=bins, alpha=0.7, color=SOURCE_COLORS[source], label=source)
    ax.axvline(config.frontal.min_iod_px, color="#c0392b", linestyle="--", label="min_iod_px")
    ax.set_xlabel("IOD (px)")
    ax.set_ylabel("images")
    ax.legend()
    fig.suptitle("Distance between eye centers (MediaPipe)")
    out["iod"] = "frontal_iod_hist.png"
    save_figure(fig, folder / out["iod"])

    panels = []
    for angle, indicator, _, _ in SIGN_CHECKS:
        panels.append((geo.loc[face, indicator], df.loc[face, angle], colors, indicator, angle, f"{angle} vs {indicator}"))
    out["pose_geometry"] = "frontal_pose_geometry.png"
    save_figure(scatter_panels(panels, "Pose angles against image geometry (blue SCUT, orange FFHQ)"), folder / out["pose_geometry"])

    if not ff.empty:
        panels = [(ff[t], ff[o], SOURCE_COLORS["ffhq"], f"Face++ {t}", o, f"{o} vs Face++ {t}")
                  for o, t in (("yaw_deg", "head_yaw"), ("pitch_deg", "head_pitch"), ("roll_deg", "head_roll"))]
        out["pose_facepp"] = "frontal_pose_facepp.png"
        save_figure(scatter_panels(panels, "MediaPipe pose against Face++ labels of FFHQ-Aging"), folder / out["pose_facepp"])

    best = ratio_rows[ratio_rows["measure"].str.startswith("alar")].sort_values("r", ascending=False).iloc[0]
    group_colors = ratio_data["group_label"].map({"AF": "#c0392b", "AM": "#2471a3", "CF": "#e59866", "CM": "#7fb3d5"})
    fig = Figure(figsize=(11, 4.8), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax, (ours, label, t) in zip(axes, (("jaw_ratio", "jaw_ratio_label", "jaw"), (best["column"], "nose_ratio_label", best["measure"])), strict=True):
        ax.scatter(ratio_data[label], ratio_data[ours], s=4, c=group_colors, alpha=0.5, linewidths=0)
        ax.set_xlabel(f"SCUT label {label.removesuffix('_label')}")
        ax.set_ylabel(f"MediaPipe {ours}")
        ax.set_title(f"{t}: r = {pearson(ratio_data[ours], ratio_data[label])[1]:.3f}", fontsize=10)
    fig.suptitle("MediaPipe ratios against SCUT-FBP5500 labels, train split (AF red, AM blue, CF and CM light)")
    out["ratios"] = "frontal_ratio_scatter.png"
    save_figure(fig, folder / out["ratios"])

    fig = Figure(figsize=(14, 4.2), layout="constrained")
    axes = fig.subplots(1, 3)
    bins = np.linspace(0, 1, 41)
    for ax, column in zip(axes, ("hair_jaw", "hair_undereye_left", "hair_undereye_right"), strict=True):
        for source in SOURCES:
            ax.hist(df.loc[face & (df["source"] == source), column], bins=bins, alpha=0.7, color=SOURCE_COLORS[source], label=source)
        ax.axvline(config.frontal.max_hair_fraction, color="#c0392b", linestyle="--", label="max_hair_fraction")
        ax.set_yscale("log")
        ax.set_xlabel(column)
        ax.set_ylabel("images (log)")
        ax.legend()
    fig.suptitle("Hair fractions")
    out["hair"] = "frontal_hair_hist.png"
    save_figure(fig, folder / out["hair"])
    return out


# ---------------------------------------------------------------- report


def fmt(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    return f"{v:.{digits}f}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def code_counts(df: pd.DataFrame, source: str) -> dict[str, int]:
    failed = df.loc[df["source"] == source, "failed"]
    return {c: int(failed.str.split(";").apply(lambda xs, c=c: c in xs).sum()) for c in codes.FRONTAL_CODES}


def report_text(df, checks, facepp, ratio_rows, figures, config) -> str:
    parts = [
        "# ตรวจโค้ดวัดค่าภาพหน้าตรงกับ SCUT-FBP5500 และ FFHQ",
        "",
        (
            "สร้างด้วย `uv run python model/scripts/frontal_check.py` ทุกภาพผ่าน `facemeasure.frontal.analyze_frontal` "
            f"ซึ่งเป็นฟังก์ชันเดียวกับที่ api เรียก (ครอปรุ่น {crops.CROP_VERSION}, mediapipe {mediapipe.__version__}) "
            "เกณฑ์คุณภาพมาจาก `shared/quality.json` รายงานนี้มีเฉพาะตัวเลขและกราฟ ภาพตรวจที่มีใบหน้าอยู่ใน "
            "`frontal_check/` ของโฟลเดอร์ outputs ซึ่งห้าม commit"
        ),
        "",
        "- SCUT-FBP5500 ทุกภาพ และ FFHQ เฉพาะบทบาท graded กับ practice ใน manifest ของชุดเกรด",
        "- ค่าคุณภาพและอัตราส่วนวัดจากใบหน้าที่ระยะระหว่างตามากที่สุด ภาพที่พบสองใบหน้าจึงมีค่าด้วย",
        "- อัตราส่วนจากจุดของ MediaPipe ยังไม่ปรับด้วยสมการเส้นตรง (train_ratio_heads.py)",
        "",
        "## จำนวนใบหน้าที่พบ",
        "",
    ]
    rows = []
    for source in SOURCES:
        s = df[df["source"] == source]
        n = len(s)
        fc = s["face_count"]
        one = int((fc == 1).sum())
        rows.append([source, n, int((s["read_error"] != "").sum()), int((fc == 0).sum()), one, int((fc == 2).sum()), f"{100 * one / n:.2f}%" if n else "–"])
    parts += [table(["แหล่ง", "ภาพ", "อ่านไม่ได้", "0 ใบหน้า", "1 ใบหน้า", "2 ใบหน้า", "1 ใบหน้า (%)"], rows), ""]
    for source in SOURCES:
        s = df[(df["source"] == source) & (df["face_count"] != 1)]
        if len(s):
            names = ", ".join(f"{r.image} ({r.face_count if pd.notna(r.face_count) else r.read_error})" for r in s.head(30).itertuples())
            parts += [f"ภาพของ {source} ที่ไม่ได้พบหนึ่งใบหน้า: {names}" + (" ..." if len(s) > 30 else ""), ""]

    parts += ["## รหัสที่ไม่ผ่าน", "", "นับภาพ ภาพหนึ่งมีได้หลายรหัส", ""]
    counts = {s: code_counts(df, s) for s in SOURCES}
    totals = {s: int((df["source"] == s).sum()) for s in SOURCES}
    rows = [[c] + [f"{counts[s][c]} ({100 * counts[s][c] / totals[s]:.1f}%)" if totals[s] else "–" for s in SOURCES] for c in codes.FRONTAL_CODES]
    passed = [int(((df["source"] == s) & (df["face_count"] == 1) & (df["failed"] == "")).sum()) for s in SOURCES]
    rows.append(["ผ่านทุกข้อ"] + [f"{p} ({100 * p / totals[s]:.1f}%)" if totals[s] else "–" for p, s in zip(passed, SOURCES, strict=True)])
    parts += [table(["รหัส", *SOURCES], rows), ""]

    parts += ["## ระยะระหว่างตา (IOD)", "", "เฉพาะภาพที่พบใบหน้า `extract_features.py` ใช้เลือกขนาดใบหน้าตอนเทรน", ""]
    rows = []
    for source in SOURCES:
        s = df[(df["source"] == source) & (df["face_count"] >= 1)]
        p5, p50, p95 = percentiles(s["iod_px"])
        below = int((s["iod_px"] < config.frontal.min_iod_px).sum())
        rows.append([source, len(s), fmt(p5, 1), fmt(p50, 1), fmt(p95, 1), f"{below} ({100 * below / max(len(s), 1):.1f}%)"])
    parts += [table(["แหล่ง", "n", "p5", "p50", "p95", f"ต่ำกว่า {config.frontal.min_iod_px} px"], rows), "",
              f"![IOD]({FIGURES_FOLDER}/{figures['iod']})", ""]

    parts += ["## การกระจายของค่าคุณภาพ", "", "p5 / p50 / p95 ของภาพที่พบใบหน้า เทียบกับเกณฑ์ใน `shared/quality.json`", ""]
    limits = {"yaw_deg": f"±{config.frontal.max_abs_yaw_deg}", "pitch_deg": f"±{config.frontal.max_abs_pitch_deg}",
              "roll_deg": f"±{config.frontal.max_abs_roll_deg}", "smile": f"≤ {config.frontal.max_smile}",
              "jaw_open": f"≤ {config.frontal.max_jaw_open}", "blink_max": f"≤ {config.frontal.max_eye_blink}"}
    rows = []
    face = df["face_count"] >= 1
    blink_max = df[["blink_left", "blink_right"]].max(axis=1)
    for column, limit in limits.items():
        row = [column, limit]
        for source in SOURCES:
            values = blink_max[face & (df["source"] == source)] if column == "blink_max" else df.loc[face & (df["source"] == source), column]
            row.append(" / ".join(fmt(v, 2) for v in percentiles(values)))
        rows.append(row)
    parts += [table(["ค่า", "เกณฑ์", *SOURCES], rows), ""]

    parts += ["## เครื่องหมายของท่าเทียบเรขาคณิตบนภาพ", "",
              ("Pearson r ระหว่างมุมจาก transformation matrix กับค่าที่วัดจากตำแหน่งจุดบนภาพ ทุกภาพที่พบใบหน้า "
               "ถ้าสูตรใน `facemeasure.pose` ให้ทิศตามนิยาม r ของ roll และ yaw ต้องเป็นบวก และของ pitch ต้องเป็นลบ "
               "เพราะเมื่อเงย ปลายจมูกเข้าใกล้ระดับตา"), ""]
    rows = [[c["text"], c["n"], fmt(c["r"]), "บวก" if c["expected"] > 0 else "ลบ", "ตรง" if c["ok"] else "**ไม่ตรง**"] for c in checks]
    parts += [table(["การตรวจ", "n", "r", "ทิศที่ต้องได้", "ผล"], rows), "",
              f"![pose geometry]({FIGURES_FOLDER}/{figures['pose_geometry']})", ""]

    if facepp:
        parts += ["## เทียบท่ากับป้ายของ Face++ (FFHQ-Aging)", "",
                  ("ภาพ FFHQ ที่พบใบหน้า ป้ายนี้มาจากเครื่องมืออีกตัว ไม่ใช่คำตอบที่ถูกแน่นอน (api.md หัวข้อ 7) "
                  "ภาพชุดเกรดคัดมาให้ Face++ yaw และ pitch อยู่ใน ±10° ช่วงค่าจึงแคบ "
                  "เครื่องหมายที่พบคือทิศของ r ค่าต่างคำนวณหลังกลับเครื่องหมายของ Face++ ให้ตรงกับของเราแล้ว"), ""]
        rows = [[f["ours"], f["facepp"], f["n"], fmt(f["r"]), "เดียวกัน" if f["sign"] > 0 else "ตรงข้าม", fmt(f["mae"], 2), fmt(f["mean_diff"], 2)] for f in facepp]
        parts += [table(["ของเรา", "Face++", "n", "Pearson r", "เครื่องหมาย", "ค่าต่างเฉลี่ยสัมบูรณ์ (°)", "ค่าต่างเฉลี่ย ของเรา − Face++ (°)"], rows), "",
                  f"![pose Face++]({FIGURES_FOLDER}/{figures['pose_facepp']})", ""]

    parts += ["## อัตราส่วนจากจุดของ MediaPipe เทียบ label ของ SCUT-FBP5500", "",
              ("ชุด train ของ `scut_labels.csv` เท่านั้น (ไม่รวม val และ test) ทุกภาพที่พบใบหน้า และเฉพาะภาพที่ผ่านเกณฑ์คุณภาพทุกข้อ "
              "อัตราส่วนฐานจมูกของทุกคู่ใช้หัวตา 133 กับ 362"), ""]
    rows = []
    best_r = ratio_rows.loc[ratio_rows["measure"].str.startswith("alar"), "r"].max()
    for r in ratio_rows.itertuples():
        name = f"**{r.measure}**" if r.measure.startswith("alar") and r.r == best_r else r.measure
        rows.append([name, r.n, fmt(r.r), *(fmt(getattr(r, f"r_{g}")) for g in scut.GROUPS), r.n_pass, fmt(r.r_pass), fmt(r.mean_mp), fmt(r.mean_label)])
    parts += [table(["ค่าวัด", "n", "r", *(f"r {g}" for g in scut.GROUPS), "n ผ่านเกณฑ์", "r ผ่านเกณฑ์", "เฉลี่ย MediaPipe", "เฉลี่ย label"], rows), "",
              f"![ratios]({FIGURES_FOLDER}/{figures['ratios']})", ""]
    chosen = mp_points.pair_key(mp_points.ALAR_BASE) if mp_points.ALAR_BASE else None
    parts += [f"คู่จุดฐานจมูกที่ตั้งใน `mp_points.ALAR_BASE`: {chosen or 'ยังไม่ได้เลือก'}", ""]
    if chosen:
        parts += [
            (
                "ทุกคู่มี r ต่างกันไม่เกินความคลาดเคลื่อนจากการสุ่มตัวอย่าง จึงเลือกคู่ที่อยู่ตรงนิยามของ model.md หัวข้อ 3.4 "
                "คือจุดนอกสุดของฐานปีกจมูกตรงที่ต่อกับริมฝีปาก ส่วนคู่ที่ r สูงสุดอยู่บนสุดของร่องปีกจมูก"
            ),
            "",
        ]

    parts += [
        "## สัดส่วนผม",
        "",
        (
            f"ภาพที่พบใบหน้า เกณฑ์คือไม่เกิน {config.frontal.max_hair_fraction} แถบกรามอยู่ด้านในเส้นขอบหน้า "
            f"กว้าง {config.frontal.hair_band_width_iod} × IOD จากกึ่งกลางระหว่างระดับตากับใต้จมูก "
            "ลงไปถึงกึ่งกลางระหว่างระดับมุมปากกับปลายคาง (`facemeasure.quality`)"
        ),
        "",
    ]
    rows = []
    for column in ("hair_jaw", "hair_undereye_left", "hair_undereye_right"):
        for source in SOURCES:
            v = df.loc[face & (df["source"] == source), column]
            p50, p90, p95, p99 = percentiles(v, (50, 90, 95, 99))
            over = int((v > config.frontal.max_hair_fraction).sum())
            rows.append([column, source, len(v), fmt(p50), fmt(p90), fmt(p95), fmt(p99), f"{over} ({100 * over / max(len(v), 1):.1f}%)"])
    parts += [table(["ค่า", "แหล่ง", "n", "p50", "p90", "p95", "p99", "เกินเกณฑ์"], rows), "",
              f"![hair]({FIGURES_FOLDER}/{figures['hair']})", ""]

    errors = df[df["ratios_error"] != ""]
    if len(errors):
        parts += ["## ภาพที่วัดอัตราส่วนไม่ได้", "", table(["ภาพ", "เหตุ"], [[r.image, r.ratios_error] for r in errors.itertuples()]), ""]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------- check images


def rank(name: str) -> str:
    return hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()


def check_selection(items: list[Item]) -> list[Item]:
    picked = []
    for source in SOURCES:
        groups = sorted({i.group for i in items if i.source == source})
        for g in groups:
            members = sorted((i for i in items if i.source == source and i.group == g), key=lambda i: rank(i.image))
            picked += members[:CHECK_PER_GROUP]
    return picked


def scaled_jaw(lm: np.ndarray, pair: tuple[int, int], s: float):
    fp = mp_points.frontal_points(lm, pair).map_points(lambda p: p * s)
    return fp, measure_jaw_nose(fp)


def check_image(image: np.ndarray, r: FrontalResult, title: str) -> Image.Image:
    """The image with the face edges, Go, Zy, canthi, alar points and boxes, next to the three crops."""
    h, w = image.shape[:2]
    s = CHECK_SIZE / max(w, h)
    small = Image.fromarray(image).resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CHECK_SIZE + crops.CROP_SIZE, CHECK_SIZE + 64), draw.SHEET_BACKGROUND)
    font = draw.load_font(13)
    if r.landmarks is None:
        canvas.paste(small, (0, 0))
        d = draw.ImageDraw.Draw(canvas)
        d.multiline_text((8, CHECK_SIZE + 6), f"{title}\nno face", fill=draw.WHITE, font=font, spacing=4)
        return canvas
    lm = r.landmarks
    chosen = mp_points.ALAR_BASE
    pairs = [chosen] if chosen else list(mp_points.ALAR_BASE_PAIRS)
    colors = [draw.RED] if chosen else PAIR_COLORS
    try:
        fp, jaw = scaled_jaw(lm, chosen or mp_points.ALAR_BASE_PAIRS[0], s)
        alar_pts = [mp_points.xy(lm, list(p)) * s for p in pairs]
        points = np.concatenate([fp.canthus_l[None], fp.canthus_r[None], *alar_pts])
        point_colors = [draw.RED, draw.RED] + [c for c in colors for _ in range(2)]
        drawn = draw.draw_jaw_nose(small, fp, jaw, points=points, point_colors=point_colors, alar_line=chosen is not None)
    except GeometryError:
        drawn = small.copy()
    frame = r.crops.frame
    edges = [frame.to_image(catmull_rom(frame.to_level(mp_points.xy(lm, chain)))) for chain in (mp_points.CONTOUR_IMAGE_LEFT, mp_points.CONTOUR_IMAGE_RIGHT)]
    draw.draw_polylines(drawn, edges, [EDGE_COLOR, EDGE_COLOR], width=1, scale=s, closed=False)
    draw.draw_polylines(
        drawn,
        [r.crops.undereye_right.corners, r.crops.undereye_left.corners, r.crops.face.corners],
        [BOX_COLOR, BOX_COLOR, FACE_BOX_COLOR],
        width=2,
        scale=s,
    )
    canvas.paste(drawn, (0, 0))
    d = draw.ImageDraw.Draw(canvas)
    for k, (crop, label) in enumerate(((r.crops.face, "face"), (r.crops.undereye_right, "under-eye R (person)"), (r.crops.undereye_left, "under-eye L (person, mirrored)"))):
        canvas.paste(Image.fromarray(crop.image), (CHECK_SIZE, k * crops.CROP_SIZE))
        # Label at the bottom: the top of an under-eye crop is the eye
        d.text((CHECK_SIZE + 4, (k + 1) * crops.CROP_SIZE - 18), label, fill=draw.WHITE, font=font, stroke_width=2, stroke_fill=draw.BLACK)
    q = r.quality
    jaw_text = f"jaw {r.ratios.jaw_ratio:.3f}" if r.ratios else f"jaw – ({r.ratios_error})"
    nose_text = f"nose {r.ratios.nose_ratio:.3f}" if r.ratios and r.ratios.nose_ratio is not None else ""
    lines = [
        (
            f"{title}  faces {r.face_count}  IOD {q.iod_px:.0f}px  yaw {q.yaw_deg:+.1f}  pitch {q.pitch_deg:+.1f}  "
            f"roll {q.roll_deg:+.1f}  smile {q.smile:.2f}  jaw_open {q.jaw_open:.2f}  hair jaw {q.hair_jaw:.2f} "
            f"eyeL {q.hair_undereye_left:.2f} eyeR {q.hair_undereye_right:.2f}"
        ),
        f"{jaw_text}  {nose_text}  failed: {', '.join(r.failed) or 'none'}",
    ]
    if not chosen:
        lines.append("alar pairs: " + "  ".join(f"{mp_points.pair_key(p)}" for p in mp_points.ALAR_BASE_PAIRS))
    d.multiline_text((8, CHECK_SIZE + 6), "\n".join(lines), fill=draw.WHITE, font=font, spacing=4)
    if not chosen:  # color key for the alar pairs on the last line
        x = 8 + d.textlength("alar pairs: ", font=font)
        y = CHECK_SIZE + 6 + 2 * (13 + 4) + 7
        for p, c in zip(mp_points.ALAR_BASE_PAIRS, PAIR_COLORS, strict=True):
            label = mp_points.pair_key(p)
            d.ellipse((x - 6, y - 3, x, y + 3), fill=c)
            x += d.textlength(label + "  ", font=font)
    return canvas


def write_check_images(items: list[Item], df: pd.DataFrame, folder: Path, config: quality.QualityConfig) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        old.unlink()
    picked = check_selection(items)
    with FaceModels() as models:
        for item in tqdm(picked, unit="image", desc="check images"):
            image = image_io.read_image(item.path)
            r = analyze_frontal(image, models, config.frontal)
            title = f"{item.source} {item.image} ({item.group})"
            check_image(image, r, title).save(folder / f"{item.source}_{Path(item.image).stem}.png")
    by_name = {(i.source, i.image): i for i in items}
    odd = df[df["face_count"] != 1]
    if len(odd):
        tiles, captions = [], []
        for row in odd.head(NOT_ONE_FACE_SHEET).itertuples():
            item = by_name[(row.source, row.image)]
            im = Image.open(item.path).convert("RGB")
            im.thumbnail((240, 240))
            tiles.append(im)
            captions.append(f"{row.source} {row.image}\nfaces {row.face_count} {row.read_error}")
        draw.contact_sheet(tiles, captions, columns=8).save(folder / "not_one_face.png")
    return len(picked)


# ---------------------------------------------------------------- main


def print_summary(df, checks, ratio_rows, csv_path, npz_path, report_path) -> None:
    for source in SOURCES:
        s = df[df["source"] == source]
        one = int((s["face_count"] == 1).sum())
        print(f"{source}: {len(s)} ภาพ พบหนึ่งใบหน้า {one} ({100 * one / max(len(s), 1):.2f}%)")
    for c in checks:
        print(f"{c['angle']} เทียบ {c['indicator']}: r = {c['r']:.3f} ({'ตรง' if c['ok'] else 'ไม่ตรง'}นิยาม)")
    alar = ratio_rows[ratio_rows["measure"].str.startswith("alar")].sort_values("r", ascending=False)
    jaw = ratio_rows[ratio_rows["measure"] == "jaw"].iloc[0]
    print(f"กราม r = {jaw['r']:.3f} (n = {jaw['n']})")
    print("ฐานจมูก r เรียงจากมากไปน้อย: " + ", ".join(f"{r.measure.removeprefix('alar ')} {r.r:.3f}" for r in alar.itertuples()))
    print(f"เขียน {csv_path}, {npz_path} และ {report_path}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Check analyze_frontal on SCUT-FBP5500 and the graded FFHQ images")
    ap.add_argument("--restart", action="store_true", help="drop the saved parts and analyze every image again")
    ap.add_argument("--no-check-images", action="store_true", help="skip the check images in outputs/frontal_check/")
    args = ap.parse_args(argv)

    config = quality.load_config()
    items = all_items()
    if not items:
        raise SystemExit("ไม่พบภาพของ SCUT-FBP5500 หรือ FFHQ ใน data/raw/")
    folder = paths.path("features") / PARTS_FOLDER
    want = settings(items, config)
    prepare_parts(folder, want, args.restart)
    run_all(items, folder, config)

    n_parts = math.ceil(len(items) / PART_SIZE)
    raw, arrays = merge_parts(folder, n_parts)
    df = finish_table(raw)
    csv_path, npz_path = write_outputs(df, arrays, want)

    geo = pose_geometry(arrays["landmarks"])
    checks = sign_checks(df, geo)
    facepp, ff = facepp_comparison(df)
    ratio_rows, ratio_data = ratio_correlations(df, config)
    report_dir = paths.ensure_dir("reports")
    figures = write_figures(df, geo, ff, ratio_rows, ratio_data, config, report_dir / FIGURES_FOLDER)
    report_path = report_dir / REPORT_NAME
    with open(report_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report_text(df, checks, facepp, ratio_rows, figures, config))
    print_summary(df, checks, ratio_rows, csv_path, npz_path, report_path)

    if not args.no_check_images:
        check_dir = paths.ensure_dir("outputs", CHECK_FOLDER)
        n = write_check_images(items, df, check_dir, config)
        print(f"ภาพตรวจ {n} ภาพอยู่ที่ {check_dir} (มีใบหน้า ห้าม commit)")


if __name__ == "__main__":
    main()


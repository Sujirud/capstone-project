"""Jaw and alar base heads: ridge on frozen DINOv2 features of the whole-face crop.

model.md sections 2.3, 4.1, 6 and 7.1 item 1. Run from the project root, in this order:

    uv run python model/scripts/train_ratio_heads.py select
    uv run python model/scripts/train_ratio_heads.py evaluate
    uv run python model/scripts/train_ratio_heads.py wu2021
    uv run python model/scripts/train_ratio_heads.py bundle

Every step rewrites model/reports/jaw_nose_heads.md from the results so far; `report`
rewrites it alone.

- select: train and val only. For the jaw and the alar base, every feature type (cls,
  cls_mean, grid4) with every alpha of ridge.ALPHAS, trained on every train copy and
  scored by RMSE on the val copies that pass that region's quality checks; the grid
  grows past an end when the best alpha sits there, then ridge.choose picks. Results go
  to select.json one feature type at a time, so a run that stops goes on from there.
- evaluate: train and test. The chosen settings are trained on train and predict the
  test copies once. The baseline is the MediaPipe ratio of the same copy through a line
  fitted on train. MAE, RMSE, bias, slope, kappa, bootstrap of the RMSE difference, the
  borderline band (0.5244 x test RMSE; women for the jaw) and how it does. Stop
  conditions become flags that bundle will not pass without --accept-flags.
- wu2021: the 8 faces of figure 2 of [3] in data/raw/wu2021_fig2/, made bigger or smaller
  so the IOD is the middle of the training range, through the api's way in and the jaw
  head. They are for this check only: never trained on, never in a report as images.
- bundle: api/bundle/ version 0.1.0 with the trained jaw and alar base heads (folded
  weights) and the stub under-eye head, then the fidelity check of api.md section 7:
  the lowest-IOD copy of FIDELITY_IMAGES test images made again from its seed and run
  the api's way with the bundle read back must give the predictions and landmark_value
  of evaluate within FIDELITY_TOLERANCE.

Results in outputs/ratio_heads/ (not in git): select.json, evaluate.json, heads.npz
(weights and bias on raw features), test_predictions.csv, wu2021.json, fidelity.json
and run_log.txt, which says which splits each command read.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

import extract_features as ef
import make_stub_bundle
from facemeasure import assets, bundle, codes, criteria, image_io, paths, quality, scut
from facemeasure.training import metrics, ratio_data, ridge
from facemeasure.training.ratio_data import (
    LABEL_COLUMN,
    MP_COLUMN,
    REASONS_COLUMN,
    TARGETS,
)

FEATURE_TYPES = tuple(ef.FEATURE_DTYPES)
OUT_NAME = "ratio_heads"
SELECT_NAME = "select.json"
EVALUATE_NAME = "evaluate.json"
HEADS_NAME = "heads.npz"
PREDICTIONS_NAME = "test_predictions.csv"
WU_NAME = "wu2021.json"
FIDELITY_NAME = "fidelity.json"
LOG_NAME = "run_log.txt"
REPORT_NAME = "jaw_nose_heads.md"
LABELS_NAME = "scut_labels.csv"
BUNDLE_VERSION = "0.1.0"

GROUPS = scut.GROUPS
SLOPE_MIN = 0.8
JAW_GROUP_BIAS_MAX = 0.01  # 1 percentage point
WU_FOLDER = "wu2021_fig2"
WU_RATIOS = {"A": 0.90, "B": 0.87, "C": 0.85, "D": 0.83, "E": 0.80, "F": 0.75, "G": 0.70, "H": 0.65}
WU_RHO_MIN = 0.7
KAPPA_WEAK = 0.2  # below this a head hardly tells the grades apart
SMALL_GROUP = 50  # rows of a group in the evaluation set
FIDELITY_IMAGES = 20
FIDELITY_TOLERANCE = 0.0005
FIDELITY_SALT = "ratio-heads-fidelity-v1"
LOW_COPY, HIGH_COPY = 1, 2  # fixed copies at the lowest and the highest IOD
RESIZE = Image.Resampling.BICUBIC
COLORS = {"model": "#2a78d6", "baseline": "#eb6834"}
INK, MUTED = "#52514e", "#9a9893"
TARGET_TEXT = {codes.JAW: "กราม", codes.NOSE_BASE: "ฐานจมูก"}


# ---------------------------------------------------------------- files


def out_dir() -> Path:
    return paths.ensure_dir("outputs", OUT_NAME)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def log(command: str, text: str, folder: Path | None = None) -> None:
    """One line in run_log.txt and on screen: what a command read."""
    folder = out_dir() if folder is None else folder
    line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {command}: {text}"
    with open(folder / LOG_NAME, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    print(line)


def read_meta() -> dict:
    meta = read_json(paths.path("features") / ef.META_NAME)
    if meta is None or ef.FACE not in meta.get("tasks", {}):
        raise SystemExit("no face features yet; run: uv run python model/scripts/extract_features.py face")
    return meta


def labels_path() -> Path:
    return paths.path("labels") / LABELS_NAME


def fingerprint() -> dict:
    """What the results depend on; a change asks for select again."""
    meta = read_meta()
    face = meta["tasks"][ef.FACE]
    return {
        "features": face["common_sha256"],
        "features_created": face["created"],
        "index_sha256": assets.file_sha256(ef.task_folder(ef.FACE) / ef.INDEX_NAME),
        "labels_sha256": assets.file_sha256(labels_path()),
        "alphas": list(ridge.ALPHAS),
        "expand_decades": ridge.EXPAND_DECADES,
        "tolerance": ridge.TOLERANCE,
        "std_floor": ridge.STD_FLOOR,
    }


def load_joined(splits: tuple[str, ...]) -> pd.DataFrame:
    """index.csv rows with features in these splits only, joined with the labels."""
    index = ef.read_index(ef.task_folder(ef.FACE))
    labels = pd.read_csv(labels_path(), dtype={"image": str, "sex": str, "split": str}, encoding="utf-8")
    return ratio_data.join_labels(index[index["split"].isin(splits)], labels)


def load_matrix(joined: pd.DataFrame, feature: str) -> np.ndarray:
    rows, x = ef.load_features(ef.task_folder(ef.FACE), feature, joined)
    if len(rows) != len(joined) or not (rows["image"].to_numpy() == joined["image"].to_numpy()).all():
        raise RuntimeError("the feature rows do not line up with the joined rows")
    return x


def print_counts(parts: dict) -> pd.DataFrame:
    table = ratio_data.counts(parts)
    wide = table.pivot_table(index=["target", "split"], columns="group", values=["rows", "images"], aggfunc="sum")
    print(wide.to_string())
    return table


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far; None where the resource module is missing (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def copy_iod(meta: dict) -> dict[int, float]:
    low, high = meta["degrade"]["fixed"]["iod_px"]
    return {LOW_COPY: float(low), HIGH_COPY: float(high)}


# ---------------------------------------------------------------- select


def grid_dict(g: ridge.Grid) -> dict:
    return {"alphas": list(g.alphas), "rmse": list(g.rmse), "expanded": list(g.expanded), "at_edge": g.at_edge}


def grid_of(feature: str, d: dict) -> ridge.Grid:
    return ridge.Grid(feature, tuple(d["alphas"]), tuple(d["rmse"]), tuple(d["expanded"]), d["at_edge"])


def select_feature(feature: str, x: np.ndarray, parts: dict) -> tuple[dict, dict]:
    """Grids of both targets for one feature matrix (rows of the joined train and val)."""
    train = {t: parts[(t, "train")] for t in TARGETS}
    first = train[TARGETS[0]].positions
    if any(not np.array_equal(first, s.positions) for s in train.values()):
        raise RuntimeError("the train rows of the jaw and the alar base differ; every train copy should have both labels")
    path = ridge.RidgePath(x[first], np.column_stack([train[t].y for t in TARGETS]))
    grids = {}
    for k, t in enumerate(TARGETS):
        val = parts[(t, "val")]
        xv = x[val.positions]

        def score(alphas, k=k, xv=xv, yv=val.y):
            return [ridge.rmse(path.model(a).column(k).predict(xv), yv) for a in alphas]

        grids[t] = ridge.search(feature, score)
    info = {
        "dim": int(x.shape[1]),
        "train_rows": len(first),
        "solver": "dual (X X^T)" if path.dual else "primal (X^T X)",
        "eigh_seconds": round(path.seconds, 2),
        "matrix_mb": round(path.peak_matrix_bytes / 2**20, 1),
        "kept_mb": round(path.memory_bytes / 2**20, 1),
    }
    return info, grids


def choose_all(state: dict, dims: dict[str, int]) -> dict:
    out = {}
    for t in TARGETS:
        grids = [grid_of(f, state["features"][f]["targets"][t]) for f in FEATURE_TYPES if f in state["features"]]
        c = ridge.choose(grids, dims)
        best = min((g.best for g in grids), key=lambda tr: tr.rmse)
        out[t] = {"feature": c.feature, "alpha": c.alpha, "val_rmse": c.rmse, "lowest_feature": best.feature,
                  "lowest_alpha": best.alpha, "lowest_rmse": best.rmse}
    return out


def cmd_select(restart: bool) -> None:
    folder = out_dir()
    log("select", "reads the train and val splits of features/face (test is not read)", folder)
    fp = fingerprint()
    state = read_json(folder / SELECT_NAME)
    if restart or state is None or state.get("fingerprint") != fp:
        if state is not None and not restart:
            print("features, labels or settings changed since the last select; starting again")
        state = {"fingerprint": fp, "features": {}}
    joined = load_joined(("train", "val"))
    parts = ratio_data.subsets(joined, ("train", "val"))
    state["counts"] = print_counts(parts).to_dict(orient="records")
    dims = ef.workers_dims()
    for feature in tqdm(FEATURE_TYPES, desc="select", unit="feature"):
        if feature in state["features"]:
            print(f"{feature}: done in an earlier run")
            continue
        x = load_matrix(joined, feature)
        info, grids = select_feature(feature, x, parts)
        del x
        info["peak_rss_mb"] = peak_rss_mb()
        state["features"][feature] = {**info, "targets": {t: grid_dict(g) for t, g in grids.items()}}
        write_json(folder / SELECT_NAME, state)
        for t, g in grids.items():
            b = g.best
            edge = " (still at an end of the grid)" if g.at_edge else ""
            print(f"{feature} {t}: best alpha {b.alpha:g} val RMSE {b.rmse:.5f}{edge}")
    state["choice"] = choose_all(state, dims)
    write_json(folder / SELECT_NAME, state)
    for t, c in state["choice"].items():
        print(f"choice {t}: {c['feature']} alpha {c['alpha']:g} val RMSE {c['val_rmse']:.5f} "
              f"(lowest {c['lowest_feature']} alpha {c['lowest_alpha']:g} {c['lowest_rmse']:.5f})")


# ---------------------------------------------------------------- evaluate


def _both(pred, base, y, mask) -> dict:
    return {"model": metrics.summary(pred[mask], y[mask]), "baseline": metrics.summary(base[mask], y[mask])}


def nose_side(value: float, sex: str) -> str:
    c = criteria.NOSE_CUTOFFS[sex]
    return "narrow" if value < c.low else ("wide" if value > c.high else "middle")


def copy_difference(rows: pd.DataFrame, pred: np.ndarray, base: np.ndarray) -> dict:
    """Images with both fixed copies in the set: prediction at the highest IOD minus at the lowest."""
    df = pd.DataFrame({"image": rows["image"].to_numpy(), "copy": rows["copy"].to_numpy(dtype=int), "model": pred, "baseline": base})
    wide = df.pivot_table(index="image", columns="copy", values=["model", "baseline"])
    out = {"images": 0}
    for method in ("model", "baseline"):
        if (method, LOW_COPY) not in wide or (method, HIGH_COPY) not in wide:
            return out
        d = (wide[(method, HIGH_COPY)] - wide[(method, LOW_COPY)]).dropna()
        out["images"] = len(d)
        out[method] = {"mean_abs": float(d.abs().mean()) if len(d) else None, "mean": float(d.mean()) if len(d) else None,
                       "p95_abs": float(d.abs().quantile(0.95)) if len(d) else None}
    return out


def evaluate_target(target: str, rows: pd.DataFrame, pred, base) -> dict:
    """Every measure of one head on the rows of the evaluation set (test copies that pass that region)."""
    pred, base = np.asarray(pred, dtype=np.float64), np.asarray(base, dtype=np.float64)
    y = rows[LABEL_COLUMN[target]].to_numpy(dtype=np.float64)
    sex = rows["sex"].to_numpy()
    masks = {"all": np.ones(len(rows), dtype=bool)}
    masks.update({g: (rows["group"] == g).to_numpy() for g in GROUPS})
    masks.update({s: sex == s for s in codes.SEXES})
    masks.update({f"copy{c}": (rows["copy"] == c).to_numpy() for c in (LOW_COPY, HIGH_COPY)})
    table = {k: _both(pred, base, y, m) for k, m in masks.items()}
    slopes = {k: {"model": metrics.slope(pred[m], y[m]), "baseline": metrics.slope(base[m], y[m])}
              for k, m in masks.items() if m.sum() > 2}

    g = metrics.graded(target, sex)
    truth = metrics.grades(target, y[g], sex[g])
    by = {"model": metrics.grades(target, pred[g], sex[g]), "baseline": metrics.grades(target, base[g], sex[g])}
    grading = {
        "rows": int(g.sum()),
        "kappa": {k: metrics.weighted_kappa(truth, v) for k, v in by.items()},
        "confusion": {k: metrics.confusion(truth, v).tolist() for k, v in by.items()},
        "agreement": {k: float(np.mean([a == b for a, b in zip(truth, v, strict=True)])) if len(v) else None for k, v in by.items()},
    }
    if target == codes.NOSE_BASE:
        opposite = 0
        for lv, pv, s in zip(y[g], pred[g], sex[g], strict=True):
            a, b = nose_side(lv, s), nose_side(pv, s)
            opposite += {a, b} == {"narrow", "wide"} and metrics.grade(target, lv, s) == metrics.grade(target, pv, s) == "1"
        grading["grade1_opposite_sides"] = int(opposite)

    images = rows["image"].to_numpy()
    boot = {"all": metrics.bootstrap_rmse_diff(images, pred - y, base - y)}
    if target == codes.JAW:
        f = masks[codes.FEMALE]
        boot[codes.FEMALE] = metrics.bootstrap_rmse_diff(images[f], (pred - y)[f], (base - y)[f])

    band_rows = codes.FEMALE if target == codes.JAW else "all"
    band_rmse = table[band_rows]["model"]["rmse"]
    band = metrics.band_from_rmse(band_rmse)
    caps = metrics.band_caps(target, band)
    return {
        "table": table,
        "slope": slopes,
        "grading": grading,
        "bootstrap": boot,
        "band": {
            "rows": band_rows,
            "rmse": band_rmse,
            "factor": criteria.RMSE_BAND_FACTOR,
            "band": band,
            "caps": caps,
            "at_cap": any(any(v["capped"]) for v in caps.values()),
            "outcomes": metrics.band_outcomes(target, pred[g], y[g], sex[g], band),
        },
        "copies": copy_difference(rows, pred, base),
    }


def flags_of(results: dict) -> list[str]:
    """The stop conditions: slope below SLOPE_MIN, a jaw group bias beyond 1 point."""
    out = []
    for t in TARGETS:
        s = results[t]["slope"]["all"]["model"]
        if s < SLOPE_MIN:
            out.append(f"{t}: slope of the predictions on the labels is {s:.3f}, below {SLOPE_MIN}")
    for g in GROUPS:
        b = results[codes.JAW]["table"][g]["model"]["bias"]
        if b is not None and abs(b) > JAW_GROUP_BIAS_MAX:
            out.append(f"jaw: bias of group {g} is {b * 100:+.2f} percentage points, beyond {JAW_GROUP_BIAS_MAX * 100:g}")
    return out


PREDICTION_COLUMNS = ["image", "group", "sex", "copy", "target_iod", "iod_px", "reasons_jaw", "reasons_nose_base",
                      "jaw_ratio", "nose_ratio", "jaw_ratio_mp", "nose_ratio_mp"]


def cmd_evaluate() -> None:
    folder = out_dir()
    sel = read_json(folder / SELECT_NAME)
    if sel is None or "choice" not in sel:
        raise SystemExit("run select first")
    if sel["fingerprint"] != fingerprint():
        raise SystemExit("features, labels or settings changed since select; run select again")
    log("evaluate", "reads the train split (fit) and the test split (one prediction with the chosen settings)", folder)
    joined = load_joined(("train", "test"))
    parts = ratio_data.subsets(joined, ("train", "test"))
    counts = print_counts(parts)
    test_pos = np.flatnonzero((joined["split"] == "test").to_numpy())
    table = joined.iloc[test_pos][PREDICTION_COLUMNS].reset_index(drop=True)
    matrices = {f: load_matrix(joined, f) for f in sorted({c["feature"] for c in sel["choice"].values()})}
    arrays, heads, results, baseline = {}, {}, {}, {}
    for t in TARGETS:
        choice = sel["choice"][t]
        x = matrices[choice["feature"]]
        train = parts[(t, "train")]
        model = ridge.fit(x[train.positions], train.y, choice["alpha"])
        w, b = model.folded()
        pred = bundle.ridge(x[test_pos], w, b)
        gap = float(np.max(np.abs(model.predict(x[test_pos]) - pred)))
        if gap > 1e-6:
            raise RuntimeError(f"{t}: folded weights predict {gap:.2e} away from the standardized model")
        slope, intercept = metrics.line_fit(train.mp, train.y)
        base = slope * table[MP_COLUMN[t]].to_numpy(dtype=np.float64) + intercept
        used = (table[REASONS_COLUMN[t]].fillna("") == "").to_numpy()
        table[f"pred_{t}"], table[f"landmark_{t}"], table[f"used_{t}"] = pred, base, used
        rows = table[used].reset_index(drop=True)
        results[t] = evaluate_target(t, rows, pred[used], base[used])
        arrays[f"{t}_weights"], arrays[f"{t}_bias"] = w, np.array(b)
        baseline[t] = {"slope": slope, "intercept": intercept}
        band = results[t]["band"]
        heads[t] = {
            "feature": choice["feature"],
            "alpha": choice["alpha"],
            "rmse": band["rmse"],
            "rmse_rows": band["rows"],
            "rmse_all": results[t]["table"]["all"]["model"]["rmse"],
            "band": band["band"],
            "train_rows": len(train.rows),
            "train_images": train.images,
            "standardize_gap": gap,
        }
        m = results[t]["table"]["all"]
        print(f"{t}: {choice['feature']} alpha {choice['alpha']:g}; test RMSE model {m['model']['rmse']:.5f} "
              f"baseline {m['baseline']['rmse']:.5f}; band {band['band']:.5f} from RMSE of {band['rows']}")
    flags = flags_of(results)
    bundle.write_npz(folder / HEADS_NAME, arrays)
    table.to_csv(folder / PREDICTIONS_NAME, index=False, encoding="utf-8", lineterminator="\n", float_format="%.8g")
    write_json(folder / EVALUATE_NAME, {
        "fingerprint": sel["fingerprint"],
        "choice": sel["choice"],
        "counts": counts.to_dict(orient="records"),
        "copy_iod": {str(k): v for k, v in copy_iod(read_meta()).items()},
        "heads": heads,
        "baseline": baseline,
        "results": results,
        "flags": flags,
    })
    for f in flags:
        print(f"FLAG {f}")
    if flags:
        print("bundle will not run until these are discussed; then: train_ratio_heads.py bundle --accept-flags")


# ---------------------------------------------------------------- the api's way, wu2021 and fidelity


def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    """The whole image scale times bigger or smaller with Pillow's bicubic resize."""
    h, w = image.shape[:2]
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return np.array(Image.fromarray(image).resize(size, RESIZE), dtype=np.uint8)


def head_values(heads: dict, base: dict, result, feats: dict[str, np.ndarray]) -> dict[str, float | None]:
    """api.md section 3 step 6 for the ratio heads: prediction and landmark_value of each head.

    heads maps a head name to (feature, weights, bias); base to (slope, intercept).
    """
    out = {}
    for t in TARGETS:
        feature, w, b = heads[t]
        out[f"pred_{t}"] = bundle.ridge(feats[feature], w, b)
        mp = None if result.ratios is None else (result.ratios.jaw_ratio if t == codes.JAW else result.ratios.nose_ratio)
        out[f"landmark_{t}"] = None if mp is None else base[t][0] * float(mp) + base[t][1]
    return out


def bundle_heads(b: bundle.Bundle) -> tuple[dict, dict]:
    heads = {t: (b.heads[t].feature, b.heads[t].weights, b.heads[t].bias) for t in TARGETS}
    return heads, {t: (b.baseline[t].slope, b.baseline[t].intercept) for t in TARGETS}


def evaluate_heads(ev: dict, arrays: dict) -> tuple[dict, dict]:
    heads = {t: (ev["heads"][t]["feature"], arrays[f"{t}_weights"], float(arrays[f"{t}_bias"])) for t in TARGETS}
    return heads, {t: (ev["baseline"][t]["slope"], ev["baseline"][t]["intercept"]) for t in TARGETS}


def read_evaluate(folder: Path) -> tuple[dict, dict]:
    ev = read_json(folder / EVALUATE_NAME)
    if ev is None or not (folder / HEADS_NAME).is_file():
        raise SystemExit("run evaluate first")
    if ev["fingerprint"] != fingerprint():
        raise SystemExit("features, labels or settings changed since evaluate; run select and evaluate again")
    return ev, bundle.read_npz(folder / HEADS_NAME)


def wu_images() -> dict[str, Path]:
    folder = paths.path("raw") / WU_FOLDER
    found = {k: folder / f"{k}.png" for k in WU_RATIOS}
    missing = [p.name for p in found.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in {folder}; cut faces A to H from figure 2 of [3] and save them there")
    return found


def wu_rows(images: dict[str, Path], heads: dict, base: dict, models, backbone, config: quality.QualityConfig,
            target_iod: float, max_iod: float) -> list[dict]:
    from facemeasure.frontal import analyze_frontal
    from facemeasure.model_input import analyze_for_model

    out = []
    for letter, path in images.items():
        image = image_io.read_image(path)
        row = {"image": letter, "paper": WU_RATIOS[letter], "width": image.shape[1], "height": image.shape[0]}
        first = analyze_frontal(image, models, config.frontal)
        row["faces"] = first.face_count
        if first.face_count == 0:
            out.append(row)
            continue
        row["iod_source"] = first.quality.iod_px
        row["scale"] = target_iod / first.quality.iod_px
        r = analyze_for_model(rescale(image, row["scale"]), models, max_iod, config.frontal)
        row["faces_scaled"] = r.face_count
        if r.face_count == 0:
            out.append(row)
            continue
        row["iod_px"] = r.quality.iod_px
        row["reasons_jaw"] = ";".join(r.region_failed.get(codes.JAW, ()))
        feats = backbone.features(r.crops.face.image[None])
        values = head_values(heads, base, r, {f: v[0] for f, v in feats.items()})
        row["model"], row["baseline"] = values[f"pred_{codes.JAW}"], values[f"landmark_{codes.JAW}"]
        row["jaw_ratio_mp"] = None if r.ratios is None else r.ratios.jaw_ratio
        out.append(row)
    return out


def wu_summary(rows: list[dict]) -> dict:
    out = {}
    for method in ("model", "baseline"):
        have = [r for r in rows if r.get(method) is not None]
        if len(have) < 3:
            out[method] = {"n": len(have), "spearman": None, "bias": None, "mae": None, "slope": None}
            continue
        p, v = np.array([r["paper"] for r in have]), np.array([r[method] for r in have])
        out[method] = {"n": len(have), "spearman": float(spearmanr(p, v).statistic), "bias": float(np.mean(v - p)),
                       "mae": float(np.mean(np.abs(v - p))), "slope": metrics.slope(v, p)}
    return out


def cmd_wu2021() -> None:
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    folder = out_dir()
    ev, arrays = read_evaluate(folder)
    images = wu_images()
    log("wu2021", f"reads data/raw/{WU_FOLDER}/ (8 images of [3], check only) and the heads of evaluate; no SCUT split", folder)
    meta = read_meta()
    low, high = meta["train_face_iod_px"]["min"], meta["train_face_iod_px"]["max"]
    target = (low + high) / 2
    heads, base = evaluate_heads(ev, arrays)
    config = quality.load_config()
    with FaceModels() as models:
        rows = wu_rows(images, heads, base, models, Backbone(), config, target, high)
    summary = wu_summary(rows)
    write_json(folder / WU_NAME, {"fingerprint": ev["fingerprint"], "target_iod": target, "rows": rows, "summary": summary})
    for r in rows:
        print(f"{r['image']}: paper {r['paper']:.2f} model {fmt(r.get('model'), 4)} baseline {fmt(r.get('baseline'), 4)} "
              f"IOD {fmt(r.get('iod_source'), 1)} -> {fmt(r.get('iod_px'), 1)} codes {r.get('reasons_jaw', '-') or '-'}")
    for method, s in summary.items():
        print(f"{method}: Spearman {fmt(s['spearman'], 3)} bias {fmt(s['bias'], 4)} MAE {fmt(s['mae'], 4)}")
    if summary["model"]["spearman"] is None or summary["model"]["spearman"] < WU_RHO_MIN:
        print(f"Spearman of the model is below {WU_RHO_MIN}: look for the cause (faces found, image size, codes)")


def pick_fidelity(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    """n test images whose lowest-IOD copy is in the evaluation of both heads, ranked by sha256 of a salt and the name."""
    ok = pred[(pred["copy"] == LOW_COPY) & pred[f"used_{codes.JAW}"] & pred[f"used_{codes.NOSE_BASE}"]]
    key = ok["image"].map(lambda s: hashlib.sha256(f"{FIDELITY_SALT}:{s}".encode()).hexdigest())
    return ok.assign(_key=key).sort_values("_key").head(n).drop(columns="_key").reset_index(drop=True)


def read_predictions(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(folder / PREDICTIONS_NAME, dtype={"image": str, "reasons_jaw": str, "reasons_nose_base": str},
                     keep_default_na=False, na_values=[""], encoding="utf-8")
    for c in ("reasons_jaw", "reasons_nose_base"):
        df[c] = df[c].fillna("")
    for t in TARGETS:
        df[f"used_{t}"] = df[f"used_{t}"].map({True: True, False: False, "True": True, "False": False}).astype(bool)
    return df


def fidelity_check(b: bundle.Bundle, rows: pd.DataFrame, models, backbone, config: quality.QualityConfig,
                   image_paths: dict[str, Path]) -> dict:
    """api.md section 7: copies made again from their seeds, run the api's way with the bundle, against evaluate."""
    from facemeasure.frontal import analyze_frontal
    from facemeasure.model_input import analyze_for_model

    max_iod = None if b.train_face_iod_px is None else b.train_face_iod_px["max"]
    workers = ef.Workers(
        analyze=lambda im: analyze_frontal(im, models, config.frontal),
        model_input=lambda im: analyze_for_model(im, models, max_iod, config.frontal),
        embed=backbone.features,
        device=str(backbone.device),
    )
    results = [ef.regenerate(ef.FACE, r["image"], image_paths[r["image"]], ef.FIXED, LOW_COPY, workers, config)[1]
               for _, r in tqdm(rows.iterrows(), total=len(rows), desc="fidelity", unit="image")]
    feats = backbone.features(np.stack([r.crops.face.image for r in results]))
    heads, base = bundle_heads(b)
    out = []
    for i, (res, (_, row)) in enumerate(zip(results, rows.iterrows(), strict=True)):
        values = head_values(heads, base, res, {f: v[i] for f, v in feats.items()})
        entry = {"image": row["image"], "iod_px": res.quality.iod_px}
        for k, v in values.items():
            entry[k] = v
            entry[f"{k}_stored"] = float(row[k])
            entry[f"{k}_diff"] = None if v is None else abs(v - float(row[k]))
        out.append(entry)
    diffs = [e[f"{k}_diff"] for e in out for k in values]
    worst = None if any(d is None for d in diffs) else max(diffs)
    by = {k: max(e[f"{k}_diff"] for e in out) for k in values} if worst is not None else {}
    return {"images": len(out), "tolerance": FIDELITY_TOLERANCE, "max_diff": worst, "max_by_value": by,
            "passed": worst is not None and worst <= FIDELITY_TOLERANCE, "rows": out}


def bundle_parts(ev: dict, arrays: dict, meta: dict, lock: Path | None = None) -> tuple[dict, dict]:
    """Manifest (without files) and arrays of the bundle: the trained ratio heads and the stub under-eye head."""
    heads, out = {}, {}
    for t in TARGETS:
        h = ev["heads"][t]
        out[f"{t}_weights"] = np.asarray(arrays[f"{t}_weights"], dtype=np.float64)
        out[f"{t}_bias"] = np.array(float(arrays[f"{t}_bias"]))
        heads[t] = {
            "feature": h["feature"],
            "stub": False,
            "weights": f"{t}_weights",
            "bias": f"{t}_bias",
            "rmse": h["rmse"],
            "band": h["band"],
            "alpha": h["alpha"],
            "rmse_rows": h["rmse_rows"],
            "rmse_all": h["rmse_all"],
            "train_rows": h["train_rows"],
            "train_images": h["train_images"],
            "selected_on": "val",
        }
    stub_manifest, stub_arrays = make_stub_bundle.stub_parts(lock=lock)
    u = bundle.UNDEREYE_HEAD
    heads[u] = stub_manifest["heads"][u]
    for k in (heads[u]["weights"], heads[u]["bias"]):
        out[k] = stub_arrays[k]
    manifest = bundle.new_manifest(
        version=BUNDLE_VERSION,
        heads=heads,
        baseline={t: dict(ev["baseline"][t]) for t in TARGETS},
        backbone=meta["backbone"],
        train_face_iod_px=meta["train_face_iod_px"],
        degrade=meta["degrade"],
        lock=lock,
    )
    return manifest, out


def cmd_bundle(accept_flags: bool) -> int:
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    folder = out_dir()
    ev, arrays = read_evaluate(folder)
    if ev["flags"] and not accept_flags:
        for f in ev["flags"]:
            print(f"FLAG {f}")
        raise SystemExit("evaluate raised the flags above; after they are discussed, run again with --accept-flags")
    log("bundle", f"writes api/bundle/ {BUNDLE_VERSION} from evaluate; reads {FIDELITY_IMAGES} test images again for the "
                  "fidelity check only (no new measures)" + ("; flags accepted" if ev["flags"] else ""), folder)
    manifest, out = bundle_parts(ev, arrays, read_meta())
    b = bundle.write_bundle(manifest, out)
    print(f"wrote bundle {b.version} to {bundle.bundle_dir()}; stub heads: {', '.join(b.stub_heads)}")
    rows = pick_fidelity(read_predictions(folder), FIDELITY_IMAGES)
    image_paths = {s.image: s.path for s in ef.face_sources()}
    with FaceModels() as models:
        check = fidelity_check(b, rows, models, Backbone(), quality.load_config(), image_paths)
    check["bundle_version"] = b.version
    check["flags_accepted"] = list(ev["flags"]) if accept_flags else []
    write_json(folder / FIDELITY_NAME, check)
    print(f"fidelity: {check['images']} images, largest difference {fmt(check['max_diff'], 6)} "
          f"(at most {FIDELITY_TOLERANCE}): {'passed' if check['passed'] else 'FAILED'}")
    return 0 if check["passed"] else 1


# ---------------------------------------------------------------- report


def fmt(x, digits: int = 4) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{digits}f}"


def pp(x) -> str:
    """A ratio difference in percentage points."""
    return "–" if x is None else f"{x * 100:+.2f}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def counts_section(sel: dict | None, ev: dict | None) -> str:
    records = (sel or {}).get("counts", []) + [r for r in (ev or {}).get("counts", []) if r["split"] == "test"]
    if not records:
        return "ยังไม่ได้รัน `select`"
    df = pd.DataFrame(records).drop_duplicates(["target", "split", "group"])
    rows = []
    for t in TARGETS:
        for s in ratio_data.SPLITS:
            part = df[(df["target"] == t) & (df["split"] == s)]
            if part.empty:
                continue
            cell = {r["group"]: f"{r['rows']} / {r['images']}" for _, r in part.iterrows()}
            rows.append([TARGET_TEXT[t], s, *[cell.get(g, "–") for g in (*GROUPS, "all")]])
    return ("แต่ละช่องคือจำนวนแถว (สำเนา) / จำนวนภาพ ชุด train ใช้ทุกสำเนา ชุด val และ test ใช้เฉพาะสำเนาที่ผ่านเกณฑ์คุณภาพของบริเวณนั้น "
            "(`reasons_jaw` หรือ `reasons_nose_base` ว่าง)\n\n" + md_table(["หัว", "ชุด", *GROUPS, "รวม"], rows))


def select_section(sel: dict | None) -> str:
    if sel is None or "choice" not in sel:
        return "ยังไม่ได้รัน `select`"
    parts = []
    feats = [f for f in FEATURE_TYPES if f in sel["features"]]
    for t in TARGETS:
        c = sel["choice"][t]
        alphas = sorted({a for f in feats for a in sel["features"][f]["targets"][t]["alphas"]})
        rows = []
        for a in alphas:
            row = [f"{a:g}"]
            for f in feats:
                g = sel["features"][f]["targets"][t]
                if a in g["alphas"]:
                    v = fmt(g["rmse"][g["alphas"].index(a)], 5)
                    row.append(f"**{v}**" if (f == c["feature"] and a == c["alpha"]) else v)
                else:
                    row.append("")
            rows.append(row)
        notes = []
        for f in feats:
            g = sel["features"][f]["targets"][t]
            if g["expanded"]:
                notes.append(f"`{f}` ขยายช่วงไปที่ alpha {', '.join(f'{a:g}' for a in g['expanded'])}"
                             + (" แล้วยังติดขอบ" if g["at_edge"] else ""))
        parts.append(
            f"### {TARGET_TEXT[t]}\n\nRMSE บนชุด val (ตัวหนาคือค่าที่เลือก)\n\n"
            + md_table(["alpha", *[f"`{f}` ({sel['features'][f]['dim']})" for f in feats]], rows)
            + f"\n\nเลือก `{c['feature']}` alpha {c['alpha']:g} RMSE {fmt(c['val_rmse'], 5)} "
            f"ส่วนค่าต่ำสุดคือ `{c['lowest_feature']}` alpha {c['lowest_alpha']:g} RMSE {fmt(c['lowest_rmse'], 5)}"
            + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "")
        )
    mem = [[f"`{f}`", sel["features"][f]["dim"], sel["features"][f]["train_rows"], sel["features"][f]["solver"],
            fmt(sel["features"][f]["eigh_seconds"], 1), fmt(sel["features"][f]["matrix_mb"], 0),
            fmt(sel["features"][f].get("peak_rss_mb"), 0)] for f in feats]
    rule = (f"ลองทุกชนิด feature กับ alpha {', '.join(f'{a:g}' for a in sel['fingerprint']['alphas'])} เทรนด้วยทุกสำเนาของชุด train "
            f"แล้ววัด RMSE บนชุด val ที่รวมสองสำเนา เลือกค่าที่ RMSE ต่ำสุด แต่ถ้าต่างกันไม่ถึง {ridge.TOLERANCE:.0%} "
            f"เลือกชนิดที่มิติน้อยกว่าแล้ว alpha ที่มากกว่า ถ้า alpha ที่ดีที่สุดอยู่ที่ขอบ ขยายช่วงอีก {ridge.EXPAND_DECADES} หลักหนึ่งครั้ง")
    memory = ("### หน่วยความจำและเวลา\n\nแก้สมการด้วย eigendecomposition ครั้งเดียวต่อชนิด feature แล้วใช้กับทุก alpha และทั้งสองหัว "
              "ชนิดที่มิติมากกว่าจำนวนแถวใช้รูปแบบ dual (X Xᵀ) \"เมทริกซ์\" คือขนาดของแถวที่ standardize แล้ว เมทริกซ์ Gram "
              "และ eigenvector แบบ float64 รวมกัน ส่วน peak RSS คือหน่วยความจำสูงสุดของ process จนจบชนิดนั้น (สะสม วัดไม่ได้บน Windows)\n\n"
              + md_table(["feature", "มิติ", "แถว train", "รูปแบบ", "eigh (s)", "เมทริกซ์ (MB)", "peak RSS (MB)"], mem))
    return rule + "\n\n" + "\n\n".join(parts) + "\n\n" + memory


SUBSET_TEXT = {"all": "ทั้งหมด", "female": "หญิง", "male": "ชาย"}


def result_rows(r: dict, copy_names: dict[str, str]) -> list[list]:
    rows = []
    for key in ("all", *GROUPS, *codes.SEXES, f"copy{LOW_COPY}", f"copy{HIGH_COPY}"):
        cell = r["table"][key]
        m, b = cell["model"], cell["baseline"]
        name = SUBSET_TEXT.get(key) or (copy_names.get(key.removeprefix("copy")) if key.startswith("copy") else key)
        s = r["slope"].get(key, {})
        rows.append([name, m["n"], fmt(m["mae"], 4), fmt(m["rmse"], 4), pp(m["bias"]), fmt(s.get("model"), 3),
                     fmt(b["mae"], 4), fmt(b["rmse"], 4), pp(b["bias"]), fmt(s.get("baseline"), 3)])
    return rows


def confusion_table(conf: list[list[int]]) -> str:
    return md_table(["label (แถว) / ทาย (คอลัมน์)", *metrics.GRADES], [[g, *row] for g, row in zip(metrics.GRADES, conf, strict=True)])


def test_section(ev: dict | None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    copy_names = {k: f"สำเนา IOD {v:g} px" for k, v in ev["copy_iod"].items()}
    parts = []
    for t in TARGETS:
        r, h = ev["results"][t], ev["heads"][t]
        gr = r["grading"]
        who = "เฉพาะผู้หญิง" if t == codes.JAW else "ทุกภาพ เส้นแบ่งตามเพศ"
        extra = ""
        if t == codes.NOSE_BASE:
            extra = (f"\n\n`criteria.nose_grade` ให้เกรด 1 ทั้งฝั่งแคบและฝั่งกว้าง แถวที่ label กับค่าทายได้เกรด 1 คนละฝั่งนับเป็นตรงกัน "
                     f"พบ {gr['grade1_opposite_sides']} แถว")
        parts.append(
            f"### {TARGET_TEXT[t]}\n\n`{h['feature']}` alpha {h['alpha']:g} เทรนด้วย {h['train_rows']} แถวจาก {h['train_images']} ภาพ "
            "ค่าคลาดเฉลี่ย (bias) เป็นจุดเปอร์เซ็นต์ของอัตราส่วน slope คือความชันของค่าทายเทียบกับ label\n\n"
            + md_table(["ชุดย่อย", "n", "MAE", "RMSE", "bias (pp)", "slope", "MAE base", "RMSE base", "bias base (pp)", "slope base"],
                       result_rows(r, copy_names))
            + f"\n\nเกรดแบบไม่มีช่วงก้ำกึ่ง {who} {gr['rows']} แถว: weighted kappa (linear) โมเดล {fmt(gr['kappa']['model'], 3)} "
            f"baseline {fmt(gr['kappa']['baseline'], 3)} เกรดตรงกัน โมเดล {fmt(gr['agreement']['model'], 3)} "
            f"baseline {fmt(gr['agreement']['baseline'], 3)}\n\nโมเดล\n\n" + confusion_table(gr["confusion"]["model"])
            + "\n\nbaseline\n\n" + confusion_table(gr["confusion"]["baseline"]) + extra
        )
    return ("ทายชุด test ครั้งเดียวด้วยค่าที่เลือก baseline คืออัตราส่วนจากจุดของ MediaPipe ของสำเนาเดียวกัน ปรับด้วยสมการเส้นตรงที่ fit บนชุด train "
            "(คอลัมน์ base)\n\n" + "\n\n".join(parts))


def baseline_section(ev: dict | None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    rows = []
    for t in TARGETS:
        for key, boot in ev["results"][t]["bootstrap"].items():
            m = ev["results"][t]["table"][key]
            rows.append([TARGET_TEXT[t], SUBSET_TEXT.get(key, key), boot["images"], fmt(m["model"]["rmse"], 4),
                         fmt(m["baseline"]["rmse"], 4), f"{boot['diff']:+.4f}", f"{boot['low']:+.4f} ถึง {boot['high']:+.4f}"])
    lines = []
    for t in TARGETS:
        b = ev["results"][t]["bootstrap"]["all"]
        if b["low"] > 0:
            lines.append(f"- {TARGET_TEXT[t]}: baseline แม่นกว่าโมเดล (ช่วงความเชื่อมั่นอยู่เหนือ 0 ทั้งช่วง)")
        elif b["high"] < 0:
            lines.append(f"- {TARGET_TEXT[t]}: โมเดลแม่นกว่า baseline (ช่วงความเชื่อมั่นอยู่ใต้ 0 ทั้งช่วง)")
        else:
            lines.append(f"- {TARGET_TEXT[t]}: แยกไม่ได้ว่าวิธีใดแม่นกว่า (ช่วงความเชื่อมั่นคร่อม 0)")
    base = "; ".join(f"{TARGET_TEXT[t]} slope {ev['baseline'][t]['slope']:.4f} intercept {ev['baseline'][t]['intercept']:.4f}" for t in TARGETS)
    boot = next(iter(ev["results"][TARGETS[0]]["bootstrap"].values()))
    return (f"ผลต่าง = RMSE ของโมเดล − RMSE ของ baseline ค่าบวกแปลว่า baseline แม่นกว่า ช่วงความเชื่อมั่น 95% จาก bootstrap ที่สุ่มภาพ "
            f"(พร้อมทุกสำเนาของภาพนั้น) {boot['rounds']} รอบ seed {boot['seed']} สมการ baseline: {base}\n\n"
            + md_table(["หัว", "ชุดย่อย", "ภาพ", "RMSE โมเดล", "RMSE baseline", "ผลต่าง", "95% CI"], rows) + "\n\n" + "\n".join(lines))


def band_section(ev: dict | None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    rows, caps, notes = [], [], []
    for t in TARGETS:
        b = ev["results"][t]["band"]
        o = b["outcomes"]
        rows.append([TARGET_TEXT[t], SUBSET_TEXT.get(b["rows"], b["rows"]), fmt(b["rmse"], 4), fmt(b["band"], 4), o["n"],
                     fmt(o["borderline"], 3), fmt(o["clear_match"], 3), fmt(o["borderline_cover"], 3)])
        for sex, c in b["caps"].items():
            caps.append([TARGET_TEXT[t], SUBSET_TEXT[sex], ", ".join(f"{v:g}" for v in c["cutoffs"]),
                         ", ".join(f"{v:.4f}" + (" (เพดาน)" if k else "") for v, k in zip(c["bands"], c["capped"], strict=True))])
        if b["at_cap"]:
            notes.append(f"- **{TARGET_TEXT[t]}: ช่วงก้ำกึ่ง {b['band']:.4f} ชนเพดาน** ที่เส้นแบ่งที่ระบุว่าเพดานในตาราง "
                         "เพดานคือครึ่งหนึ่งของระยะระหว่างเส้นแบ่งที่ติดกัน ดังนั้นค่ากลางระหว่างสองเส้นแบ่งนั้น (เช่นค่ากลางของเกรด 1) "
                         "จะได้ผลก้ำกึ่งทุกครั้ง และเกรดนั้นได้ผลชัดไม่ได้เลย")
    return (f"ช่วงก้ำกึ่ง = {criteria.RMSE_BAND_FACTOR} × RMSE ของโมเดลบนชุด test (model.md หัวข้อ 4.1) กรามใช้ RMSE ของผู้หญิง ฐานจมูกใช้ทุกภาพ "
            "ค่านี้คือค่าก่อนตัดเพดาน ซึ่ง bundle เก็บไว้ ส่วน `criteria.py` ตัดเพดานตอนให้เกรด ตารางแรกใช้ช่วงนี้กับชุด test เอง "
            "(กรามเฉพาะผู้หญิง): สัดส่วนที่ก้ำกึ่ง สัดส่วนที่เกรดตรงกับ label เมื่อผลชัด และสัดส่วนที่ช่วงครอบเกรดของ label เมื่อผลก้ำกึ่ง\n\n"
            + md_table(["หัว", "RMSE ของ", "RMSE", "ช่วง", "n", "ก้ำกึ่ง", "ชัดและตรง", "ก้ำกึ่งและครอบ"], rows)
            + "\n\nช่วงที่ใช้จริงที่แต่ละเส้นแบ่งหลังตัดเพดาน\n\n" + md_table(["หัว", "เพศ", "เส้นแบ่ง", "ช่วง"], caps)
            + ("\n\n" + "\n".join(notes) if notes else ""))


def copies_section(ev: dict | None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    low, high = ev["copy_iod"][str(LOW_COPY)], ev["copy_iod"][str(HIGH_COPY)]
    rows = []
    for t in TARGETS:
        c = ev["results"][t]["copies"]
        for method in ("model", "baseline"):
            m = c.get(method, {})
            rows.append([TARGET_TEXT[t], method, c["images"], fmt(m.get("mean_abs"), 4), fmt(m.get("p95_abs"), 4), pp(m.get("mean"))])
    return (f"ภาพ test ที่ทั้งสำเนา IOD {low:g} px และ {high:g} px อยู่ในชุดประเมิน ผลต่าง = ค่าที่ IOD {high:g} px − ค่าที่ IOD {low:g} px "
            "ภาพที่ใบหน้าเล็กกว่าเป้าหมายไม่ถูกขยาย สำเนาสองชุดของภาพนั้นจึงต่างกันแค่ noise\n\n"
            + md_table(["หัว", "วิธี", "ภาพ", "ผลต่างสัมบูรณ์ เฉลี่ย", "ผลต่างสัมบูรณ์ p95", "ผลต่างเฉลี่ย (pp)"], rows))


def wu_section(wu: dict | None) -> str:
    if wu is None:
        return "ยังไม่ได้รัน `wu2021`"
    rows = [[r["image"], f"{r['paper']:.2f}", fmt(r.get("model"), 4), fmt(r.get("baseline"), 4), fmt(r.get("iod_source"), 1),
             fmt(r.get("iod_px"), 1), (r.get("reasons_jaw") or "–") if r.get("faces_scaled") else "ไม่พบใบหน้า"] for r in wu["rows"]]
    s = wu["summary"]
    lines = [f"- {m}: Spearman ρ {fmt(s[m]['spearman'], 3)} ค่าคลาดเฉลี่ย {pp(s[m]['bias'])} pp MAE {fmt(s[m]['mae'], 4)} "
             f"slope {fmt(s[m].get('slope'), 3)} (n {s[m]['n']})" for m in ("model", "baseline")]
    rho = s["model"]["spearman"]
    warn = "" if rho is not None and rho >= WU_RHO_MIN else f"\n\n**Spearman ρ ของโมเดลต่ำกว่า {WU_RHO_MIN}** ดูสาเหตุในหัวข้อข้อจำกัด"
    return (f"ภาพ 8 ภาพในรูปที่ 2 ของ \\[3\\] (ภาพไม่อยู่ในรายงานนี้) ปรับขนาดให้ IOD เป็น {wu['target_iod']:g} px ซึ่งอยู่กลางช่วงที่เทรน "
            "แล้วส่งเข้า `analyze_for_model` ครอป DINOv2 และหัวกรามจาก `evaluate` ภาพเป็นภาพดัดแปลงจากใบหน้าเดียวและอาจมีเส้นหรือตัวอักษรกำกับ "
            "ผลจึงใช้ดูแนวโน้มเท่านั้น\n\n"
            + md_table(["ภาพ", "บทความ", "โมเดล", "baseline", "IOD เดิม", "IOD ที่ใช้", "รหัสคุณภาพของกราม"], rows)
            + "\n\n" + "\n".join(lines) + warn)


def fidelity_section(fid: dict | None) -> str:
    if fid is None:
        return "ยังไม่ได้รัน `bundle`"
    by = ", ".join(f"`{k}` {v:.2e}" for k, v in fid["max_by_value"].items())
    return (f"bundle รุ่น {fid['bundle_version']} โหลดกลับมาแล้ว สร้างสำเนา IOD ต่ำสุดของภาพ test {fid['images']} ภาพขึ้นใหม่จาก seed "
            "ส่งเข้า `analyze_for_model` (IOD สูงสุดจาก bundle) ครอป DINOv2 และหัวใน bundle แล้วเทียบค่าทายและ `landmark_value` กับค่าจาก `evaluate` "
            f"ผลต่างมากสุด {fmt(fid['max_diff'], 6)} เกณฑ์ไม่เกิน {fid['tolerance']}: **{'ผ่าน' if fid['passed'] else 'ไม่ผ่าน'}**\n\n"
            f"ผลต่างมากสุดแยกค่า: {by}")


def log_section(folder: Path) -> str:
    path = folder / LOG_NAME
    if not path.is_file():
        return "ยังไม่มี log"
    last = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            when, _, rest = line.strip().partition(" ")
            command, _, text = rest.partition(": ")
            last[command] = (when, text)
    rows = [[f"`{c}`", w, t] for c, (w, t) in last.items()]
    return ("ครั้งล่าสุดของแต่ละคำสั่งจาก `run_log.txt` ใน outputs/ratio_heads/ ชุด test ถูกใช้ประเมินเฉพาะใน `evaluate` "
            "ส่วน `bundle` สร้างภาพ test 20 ภาพขึ้นใหม่เพื่อตรวจความตรงเท่านั้น\n\n" + md_table(["คำสั่ง", "เวลา (UTC)", "สิ่งที่อ่าน"], rows))


def limits_section(ev: dict | None, wu: dict | None, fid: dict | None = None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    out = []
    for t in TARGETS:
        b = ev["results"][t]["bootstrap"]["all"]
        if b["low"] > 0:
            out.append(f"- **{TARGET_TEXT[t]}: baseline จากจุดของ MediaPipe แม่นกว่าโมเดล** RMSE ต่างกัน {b['diff']:+.4f} ระบบยังใช้ค่าจากโมเดล "
                       "เพราะโปรเจกต์ตั้งใจแสดงการเทรนโมเดล (model.md หัวข้อ 7.1) ค่าจาก baseline อยู่ใน `landmark_value` ของผล")
    for f in ev["flags"]:
        out.append(f"- ธงจากเงื่อนไขหยุด: {f}")
    if ev["flags"] and fid is not None and fid.get("flags_accepted") == ev["flags"]:
        out.append("- ธงข้างบนถูกพิจารณาแล้ว และผู้ใช้ตัดสินใจรับผลตามนี้ bundle จึงเขียนด้วย `bundle --accept-flags` โดยไม่ปรับค่าทาย")
    if any("slope" in f for f in ev["flags"]):
        out.append("- slope ต่ำกว่า 1 แปลว่า ridge ดึงค่าทายเข้าหาค่าเฉลี่ย ใบหน้าที่อัตราส่วนสูงหรือต่ำมากจึงได้ค่าทายใกล้กลางกว่าความจริง "
                   "และเกรดที่ปลายทั้งสองข้างถูกทายน้อยกว่าจริง")
    for t in TARGETS:
        k = ev["results"][t]["grading"]["kappa"]["model"]
        if k is not None and k < KAPPA_WEAK:
            out.append(f"- **{TARGET_TEXT[t]}: โมเดลแทบแยกเกรดไม่ได้** weighted kappa {k:.3f} ค่าทายเกือบทั้งหมดได้เกรด 0 (ดู confusion matrix)")
        small = [f"{g} {ev['results'][t]['table'][g]['model']['n']} แถว" for g in GROUPS
                 if ev["results"][t]["table"][g]["model"]["n"] < SMALL_GROUP]
        if small:
            out.append(f"- {TARGET_TEXT[t]}: ชุดประเมินมีแถวน้อยในบางกลุ่ม ({', '.join(small)}) เพราะภาพส่วนใหญ่ของกลุ่มนั้นไม่ผ่านเกณฑ์คุณภาพ "
                       "ตัวเลขของกลุ่มนี้จึงไม่แน่นอน")
    if wu is not None and (wu["summary"]["model"]["spearman"] is None or wu["summary"]["model"]["spearman"] < WU_RHO_MIN):
        out.append(f"- ภาพของ \\[3\\]: Spearman ρ ของโมเดลต่ำกว่า {WU_RHO_MIN}")
    out.append("- RMSE และช่วงก้ำกึ่งวัดบนสำเนาที่ย่อเลียนแบบเว็บแคมของ SCUT-FBP5500 ไม่รวมความคลาดเคลื่อนจากการถ่ายภาพจริง (model.md หัวข้อ 7.2)")
    out.append("- label มาจากจุดของ SCUT-FBP5500 ที่หา Go และ Zy ด้วยกฎ คลาดได้ในระดับที่คนวางจุดต่างกัน (model.md หัวข้อ 5.2)")
    return "\n".join(out)


def scatter_figure(pred: pd.DataFrame, target: str, path: Path) -> None:
    """Predictions against labels, one panel per group; top row the model, bottom row the baseline."""
    rows = pred[pred[f"used_{target}"]]
    y = rows[LABEL_COLUMN[target]]
    lo, hi = np.percentile(np.concatenate([y, rows[f"pred_{target}"], rows[f"landmark_{target}"]]), [0.5, 99.5])
    pad = (hi - lo) * 0.05
    lim = (lo - pad, hi + pad)
    fig = Figure(figsize=(13, 6.6), layout="constrained")
    axes = fig.subplots(2, len(GROUPS), sharex=True, sharey=True)
    for i, (method, column) in enumerate((("model", f"pred_{target}"), ("baseline", f"landmark_{target}"))):
        for j, g in enumerate(GROUPS):
            ax = axes[i, j]
            part = rows[rows["group"] == g]
            sex = scut.SEX[g[1]]
            s = criteria.scale(ratio_data.MEASURE[target], sex)
            for c in (s.cutoffs if s is not None else ()):
                ax.axvline(c, color=MUTED, linestyle="--", linewidth=0.7)
                ax.axhline(c, color=MUTED, linestyle="--", linewidth=0.7)
            ax.plot(lim, lim, color=INK, linewidth=0.9)
            ax.scatter(part[LABEL_COLUMN[target]], part[column], s=8, alpha=0.45, c=COLORS[method], linewidths=0)
            e = part[column] - part[LABEL_COLUMN[target]]
            ax.set_title(f"{g} {method}: n {len(part)}, RMSE {np.sqrt(np.mean(e ** 2)):.4f}", fontsize=9, color=INK)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.tick_params(labelsize=8, colors=INK)
            ax.grid(color="#e6e5e1", linewidth=0.5)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel(f"label {LABEL_COLUMN[target]}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{method}", fontsize=9, color=INK)
    fig.suptitle(f"{target}: prediction against label on the test copies used (line: y = x, dashed: grade cutoffs of that sex)",
                 fontsize=10, color=INK)
    fig.savefig(path, dpi=100, metadata={"Software": None})


def write_report() -> Path:
    folder = out_dir()
    sel = read_json(folder / SELECT_NAME)
    ev = read_json(folder / EVALUATE_NAME)
    wu = read_json(folder / WU_NAME)
    fid = read_json(folder / FIDELITY_NAME)
    meta = read_json(paths.path("features") / ef.META_NAME) or {}
    reports = paths.ensure_dir("reports")
    figures = paths.ensure_dir("reports", "figures")
    figure_text = "ยังไม่ได้รัน `evaluate`"
    if ev is not None and (folder / PREDICTIONS_NAME).is_file():
        pred = read_predictions(folder)
        names = []
        for t in TARGETS:
            name = f"ratio_heads_{t}_scatter.png"
            scatter_figure(pred, t, figures / name)
            names.append(f"![{t}](figures/{name})")
        figure_text = ("จุดคือสำเนาของชุด test ที่อยู่ในชุดประเมิน แถวบนคือโมเดล แถวล่างคือ baseline เส้นทึบคือ y = x "
                       "เส้นประคือเส้นแบ่งเกรดของเพศนั้น (กรามของผู้ชายไม่มีเส้นแบ่ง)\n\n" + "\n\n".join(names))
    bb = meta.get("backbone", {})
    header = [
        "# หัวกรามและหัวฐานจมูก",
        "",
        "สร้างด้วย `uv run python model/scripts/train_ratio_heads.py` (`select`, `evaluate`, `wu2021`, `bundle`) รายงานนี้มีเฉพาะตัวเลขและกราฟ ไม่มีภาพใบหน้า",
        "",
        (f"- ridge regression บน feature ของ `{bb.get('name', '?')}` แบบแช่แข็งจากครอปทั้งใบหน้า standardize ด้วย mean และ std ของชุด train "
        f"(std ต่ำสุด {ridge.STD_FLOOR:g}) แล้วรวมเข้าไปใน w และ b ด้วย float64 ก่อนบันทึก (model.md หัวข้อ 2.3 และ 6)"),
        (f"- ครอปรุ่น {meta.get('crop_version', '?')} การย่อภาพรุ่น {meta.get('degrade', {}).get('version', '?')} "
        f"ช่วง IOD ที่เทรน {meta.get('train_face_iod_px', {}).get('min', '?')}–{meta.get('train_face_iod_px', {}).get('max', '?')} px"),
        "- อัตราส่วนเป็นสัดส่วน ไม่ใช่เปอร์เซ็นต์ ค่าคลาดเฉลี่ยเขียนเป็นจุดเปอร์เซ็นต์ (pp)",
    ]
    sections = [
        ("จำนวนข้อมูล", counts_section(sel, ev)),
        ("การเลือกชนิด feature และ alpha", select_section(sel)),
        ("ผลบนชุด test", test_section(ev)),
        ("เทียบกับ baseline", baseline_section(ev)),
        ("ช่วงก้ำกึ่ง", band_section(ev)),
        ("ค่าทายของภาพเดียวกันที่ IOD ต่ำสุดกับสูงสุด", copies_section(ev)),
        ("ภาพจาก \\[3\\]", wu_section(wu)),
        ("ความตรงกับตอนเตรียมข้อมูล", fidelity_section(fid)),
        ("ข้อจำกัด", limits_section(ev, wu, fid)),
        ("ชุดข้อมูลที่แต่ละคำสั่งอ่าน", log_section(folder)),
        ("กราฟค่าทายเทียบกับ label", figure_text),
    ]
    text = "\n".join(header) + "\n\n" + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections) + "\n"
    path = reports / REPORT_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Jaw and alar base heads: select, evaluate, wu2021, bundle")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="choose the feature type and alpha on val")
    p.add_argument("--restart", action="store_true", help="drop the finished feature types and search again")
    sub.add_parser("evaluate", help="train the chosen settings and predict the test set once")
    sub.add_parser("wu2021", help="the 8 faces of figure 2 of [3] through the jaw head")
    p = sub.add_parser("bundle", help=f"write api/bundle/ {BUNDLE_VERSION} and check it against evaluate")
    p.add_argument("--accept-flags", action="store_true", help="write the bundle although evaluate raised flags")
    sub.add_parser("report", help=f"rewrite model/reports/{REPORT_NAME}")
    args = ap.parse_args(argv)
    status = 0
    if args.command == "select":
        cmd_select(args.restart)
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "wu2021":
        cmd_wu2021()
    elif args.command == "bundle":
        status = cmd_bundle(args.accept_flags)
    write_report()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

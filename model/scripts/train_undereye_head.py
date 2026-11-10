"""Under-eye head: ordinal regression on frozen DINOv2 features of the under-eye crops.

model.md sections 3.3, 4.2, 5.2, 5.3, 6 and 7.1 item 3. Run from the project root, in this order:

    uv run python model/scripts/train_undereye_head.py labels
    uv run python model/scripts/train_undereye_head.py select
    uv run python model/scripts/train_undereye_head.py calibrate
    uv run python model/scripts/train_undereye_head.py evaluate
    uv run python model/scripts/train_undereye_head.py cross
    uv run python model/scripts/train_undereye_head.py bundle

Every step rewrites model/reports/undereye_head.md from the results so far; `report`
rewrites it alone.

- labels: the grader CSVs of grading/results/ (training_set/undereye_labels.py) become
  labels/undereye_labels.csv, one row per side, and labels.json holds the files, the
  grade distribution and, with two raters or more, their agreement.
- select: train and val only. Every feature type with every lam of LAMBDAS, trained on
  the train copies and scored by cross-entropy on the val copies that pass the quality
  checks; the grid grows past an end and ties go to the smaller feature type and the
  larger lam, as for the ratio heads (ridge.search and ridge.choose). Saved one feature type at
  a time, so a run that stops goes on from there.
- calibrate: train and val only. The chosen head trained on train; T on val; ECE before
  and after; the candidate thresholds on val. Supporting result: T on every val fixed copy.
- evaluate: the test split once, with the head and T of calibrate.
- cross: a head on FFHQ only (train to fit, val to choose and calibrate), checked on
  every SCUT fixed copy of every split; it runs after evaluate and never goes into the bundle.
- bundle: api/bundle/ version 0.2.0 with the jaw, alar base and baseline of the bundle
  already there and the under-eye head of calibrate, then the fidelity check of api.md
  section 7 on FIDELITY_SIDES test sides made again from their seeds.

Results in outputs/undereye_head/ (not in git): labels.json, raters.json (the real names
behind R1, R2, ...; never in a report), select.json, calibrate.json, head.npz,
val_predictions.csv, evaluate.json, test_predictions.csv, cross.json, fidelity.json and
run_log.txt, which says which splits each command read.
"""

import argparse
import copy
import hashlib
import math
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm

import extract_features as ef
import grading_set
from facemeasure import assets, bundle, criteria, paths, quality
from facemeasure.training import coral, ridge
from facemeasure.training import undereye_data as ud
from facemeasure.training import undereye_labels as ul
from facemeasure.training import undereye_metrics as um
from train_ratio_heads import fmt, md_table, read_json, write_json

OUT_NAME = "undereye_head"
LABELS_NAME = "undereye_labels.csv"
LABELS_JSON = "labels.json"
RATERS_JSON = "raters.json"
SELECT_NAME = "select.json"
CALIBRATE_NAME = "calibrate.json"
HEAD_NAME = "head.npz"
VAL_PREDICTIONS = "val_predictions.csv"
EVALUATE_NAME = "evaluate.json"
PREDICTIONS_NAME = "test_predictions.csv"
CROSS_NAME = "cross.json"
FIDELITY_NAME = "fidelity.json"
LOG_NAME = "run_log.txt"
REPORT_NAME = "undereye_head.md"
MANIFEST_NAME = "manifest.csv"
RESULTS_FOLDER = "results"
BUNDLE_VERSION = "0.2.0"

FEATURE_TYPES = tuple(ef.FEATURE_DTYPES)
LAMBDAS = tuple(10.0**k for k in range(-5, 2))
LOW_COPY, HIGH_COPY = 1, 2
FIDELITY_SIDES = 20
FIDELITY_TOLERANCE = 1e-4
FIDELITY_SALT = "undereye-head-fidelity-v1"
T_USUAL = (0.5, 2.0)  # outside this the calibration probably shifts with the data
ECE_SHIFT_RATIO, ECE_SHIFT_GAP = 2.0, 0.05  # "much higher": at least twice and at least 0.05 above
FOLD_GAP_MAX = 1e-6
LOSS_TEXT_TH = "ค่าเฉลี่ยทุกแถวของ BCE(k = 1) + BCE(k = 2) เทียบกับ soft label บวก λ‖w‖² บน feature ที่ standardize แล้ว (ไม่ลงโทษ bias)"
LOSS_TEXT = "mean over rows of BCE(k=1) + BCE(k=2) on soft labels, plus lam |w|^2 on standardized features (biases not penalized)"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
STAGE_COLORS = {"before": "#eb6834", "after": "#2a78d6"}  # validated pair (dataviz palette slots 2 and 1)
GRADE_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # ordinal blue, grades 0-3, validated
SUBSET_TEXT = {
    "all": "ทั้งหมด (ไม่ flagged)", "ffhq": "FFHQ", "scut": "SCUT", "female": "หญิง", "male": "ชาย", "flagged": "flagged",
    "all_copies": "ทุกสำเนา fixed (ไม่กรองคุณภาพ)", "all_copies_ffhq": "ทุกสำเนา fixed: FFHQ", "all_copies_scut": "ทุกสำเนา fixed: SCUT",
}


# ---------------------------------------------------------------- files


def out_dir() -> Path:
    return paths.ensure_dir("outputs", OUT_NAME)


def labels_path() -> Path:
    return paths.path("labels") / LABELS_NAME


def log(command: str, text: str, folder: Path | None = None) -> None:
    """One line in run_log.txt and on screen: what a command read."""
    folder = out_dir() if folder is None else folder
    line = f"{datetime.now(UTC).isoformat(timespec='seconds')} {command}: {text}"
    with open(folder / LOG_NAME, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
    print(line)


def read_meta() -> dict:
    meta = read_json(paths.path("features") / ef.META_NAME)
    if meta is None or ef.UNDEREYE not in meta.get("tasks", {}):
        raise SystemExit("no under-eye features yet; run: uv run python model/scripts/extract_features.py undereye")
    return meta


def fingerprint() -> dict:
    """What the results after labels depend on; a change asks for select again."""
    if not labels_path().is_file():
        raise SystemExit("no labels yet; run: uv run python model/scripts/train_undereye_head.py labels")
    task = read_meta()["tasks"][ef.UNDEREYE]
    return {
        "features": task["common_sha256"],
        "features_created": task["created"],
        "index_sha256": assets.file_sha256(ef.task_folder(ef.UNDEREYE) / ef.INDEX_NAME),
        "labels_sha256": assets.file_sha256(labels_path()),
        "lambdas": list(LAMBDAS),
        "expand_decades": ridge.EXPAND_DECADES,
        "tolerance": ridge.TOLERANCE,
        "std_floor": ridge.STD_FLOOR,
        "max_iter": coral.MAX_ITER,
        "history": coral.HISTORY,
        "tolerance_grad": coral.TOLERANCE_GRAD,
        "tolerance_change": coral.TOLERANCE_CHANGE,
        "t_bounds": list(coral.T_BOUNDS),
    }


def require(name: str, step: str, fp: dict | None = None) -> dict:
    """The results of an earlier step, made with the same data and settings."""
    data = read_json(out_dir() / name)
    if data is None:
        raise SystemExit(f"run {step} first")
    if data.get("fingerprint") != (fingerprint() if fp is None else fp):
        raise SystemExit(f"labels, features or settings changed since {step}; run select and the steps after it again")
    return data


def load_joined(splits: tuple[str, ...]) -> pd.DataFrame:
    """index.csv rows with features in these splits only, joined with the labels (excluded sides dropped)."""
    index = ef.read_index(ef.task_folder(ef.UNDEREYE))
    return ud.join_labels(index[index["split"].isin(splits)], ul.read_labels(labels_path()))


def load_matrix(joined: pd.DataFrame, feature: str) -> np.ndarray:
    rows, x = ef.load_features(ef.task_folder(ef.UNDEREYE), feature, joined)
    same = all((rows[c].to_numpy() == joined[c].to_numpy()).all() for c in ("image", "side", "kind", "copy"))
    if len(rows) != len(joined) or not same:
        raise RuntimeError("the feature rows do not line up with the joined rows")
    return x


def rater_codes(df: pd.DataFrame) -> list[str]:
    return [c.removeprefix("grade_") for c in ul.rater_columns(df)]


def copy_iod(meta: dict) -> dict[str, float]:
    low, high = meta["degrade"]["fixed"]["iod_px"]
    return {str(LOW_COPY): float(low), str(HIGH_COPY): float(high)}


# ---------------------------------------------------------------- labels


def file_rows(g: ul.Grades) -> list[dict]:
    """The files as the report shows them: F1, F2, ... with rater codes, no names."""
    codes = {**g.codes, ul.CONSENSUS: ul.CONSENSUS}
    out = []
    for k, c in enumerate(g.files, start=1):
        out.append({"code": f"F{k}", "file": c.file, "used": c.used, "rows": c.rows, "skipped_rows": c.skipped_rows,
                    "outside_rows": c.outside_rows, "raters": [codes.get(r, "?") for r in c.raters], "set_ids": c.set_ids,
                    "reason": c.reason})
    return out


def label_groups(sides: pd.DataFrame) -> dict[str, pd.Series]:
    groups = {"all": pd.Series(True, index=sides.index)}
    for s in ("ffhq", "scut"):
        groups[s] = sides["source"] == s
    for a in sorted(sides.loc[sides["source"] == "ffhq", "age_group"].unique()):
        groups[f"ffhq {a}"] = (sides["source"] == "ffhq") & (sides["age_group"] == a)
    for s in ("female", "male"):
        groups[s] = sides["sex"] == s
    for s in ud.SPLITS:
        groups[s] = sides["split"] == s
    groups["pilot"] = sides["pilot"] == 1
    return groups


def label_summary(g: ul.Grades, sides: pd.DataFrame, graded: pd.DataFrame) -> dict:
    kept = sides[sides["excluded"] == ""]
    pilot = graded.loc[graded["pilot"] == "1", "image"]
    dist = ul.distribution(sides, label_groups(sides))
    overall = dist[0]
    low = [k for k in ("ge2_share", "ge2_share_unflagged") if overall[k] is not None and overall[k] < ul.LOW_GE2_SHARE]
    return {
        "files": file_rows(g),
        "raters": list(g.codes.values()),
        "consensus": g.has_consensus,
        "consensus_sides": int(sides["consensus_grade"].notna().sum()),
        "sides": len(sides),
        "images": int(sides["image"].nunique()),
        "excluded": {k: int(v) for k, v in sides.loc[sides["excluded"] != "", "excluded"].value_counts().items()},
        "flagged": int((kept["flagged"] == 1).sum()),
        "distribution": dist,
        "low_ge2": low,
        "median_seconds": ul.median_seconds(g),
        "agreement": None if len(g.raters) < 2 else {"pilot": ul.agreement(g, pilot), "all": ul.agreement(g, graded["image"])},
        "pilot_images": len(pilot),
    }


def cmd_labels() -> None:
    folder = out_dir()
    log("labels", "reads grading/results/*.csv and grading/manifest.csv: the grades of every split (no features)", folder)
    manifest = grading_set.read_manifest(paths.path("grading") / MANIFEST_NAME)
    graded = manifest[manifest["role"] == "graded"]
    files = sorted((paths.path("grading") / RESULTS_FOLDER).glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV in {paths.path('grading') / RESULTS_FOLDER}; export the full_600 set from the grading tool")
    try:
        g = ul.collect(files, graded["image"])
    except ul.LabelError as e:
        raise SystemExit(str(e)) from None
    for note in g.notes:
        print(note)
    sides = ul.sides_table(g, manifest)
    ul.write_labels(sides, labels_path())
    summary = label_summary(g, sides, graded)
    write_json(folder / LABELS_JSON, {"labels_sha256": assets.file_sha256(labels_path()), **summary})
    write_json(folder / RATERS_JSON, {"note": "real names behind the codes; keep out of reports and git",
                                      "raters": {code: g.names[key] for key, code in g.codes.items()}})
    print(f"{len(g.raters)} rater(s) {', '.join(summary['raters'])}; consensus: {g.has_consensus}; "
          f"{summary['sides']} sides, excluded {sum(summary['excluded'].values())}, flagged {summary['flagged']}; wrote {labels_path()}")
    if summary["agreement"] is None:
        print("agreement skipped: there is a single rater, so agreement between raters cannot be measured")
    else:
        for part in ("pilot", "all"):
            for p in summary["agreement"][part]["pairs"]:
                print(f"{part}: {p['a']}-{p['b']} kappa 0-3 {fmt(p['kappa4'], 3)} kappa 0/1/>=2 {fmt(p['kappa3'], 3)} n {p['n']}")
    for k in summary["low_ge2"]:
        print(f"WARNING {k} is below {ul.LOW_GE2_SHARE:.0%}: the transition at 2 will be estimated unsteadily")


# ---------------------------------------------------------------- select


def grid_dict(g: ridge.Grid) -> dict:
    return {"lambdas": list(g.alphas), "ce": list(g.rmse), "expanded": list(g.expanded), "at_edge": g.at_edge}


def grid_of(feature: str, d: dict) -> ridge.Grid:
    return ridge.Grid(feature, tuple(d["lambdas"]), tuple(d["ce"]), tuple(d["expanded"]), d["at_edge"])


def select_feature(feature: str, x: np.ndarray, train: ud.Subset, val: ud.Subset, lambdas=LAMBDAS) -> dict:
    """Val cross-entropy of every lam (the grid grows past an end), with how each fit ended."""
    problem = coral.Problem(x[train.positions], train.y)
    xv = x[val.positions]
    fits = {}

    def score(lams):
        out = []
        for lam in tqdm(lams, desc=feature, unit="lam", leave=False):
            m = problem.fit(lam)
            ce = coral.cross_entropy(m.logits(xv), val.y)
            fits[float(lam)] = {**m.info(), "val_ce": ce}
            out.append(ce)
        return out

    grid = ridge.search(feature, score, lambdas)
    return {"dim": int(x.shape[1]), "train_rows": len(train.rows), "val_rows": len(val.rows), "grid": grid_dict(grid),
            "fits": [fits[k] for k in sorted(fits)]}


def choose(features: dict, dims: dict[str, int]) -> dict:
    grids = [grid_of(f, features[f]["grid"]) for f in FEATURE_TYPES if f in features]
    c = ridge.choose(grids, dims)
    best = min((g.best for g in grids), key=lambda t: t.rmse)
    return {"feature": c.feature, "lam": c.alpha, "val_ce": c.rmse, "lowest_feature": best.feature, "lowest_lam": best.alpha,
            "lowest_ce": best.rmse}


def run_selection(state: dict, joined: pd.DataFrame, train: ud.Subset, val: ud.Subset, save) -> dict:
    """Fill state["features"] one feature type at a time (save after each), then state["choice"]."""
    state.setdefault("features", {})
    for feature in FEATURE_TYPES:
        if feature in state["features"]:
            print(f"{feature}: done in an earlier run")
            continue
        x = load_matrix(joined, feature)
        state["features"][feature] = select_feature(feature, x, train, val)
        del x
        save()
        g = grid_of(feature, state["features"][feature]["grid"])
        edge = " (still at an end of the grid)" if g.at_edge else ""
        print(f"{feature}: best lam {g.best.alpha:g} val CE {g.best.rmse:.5f}{edge}")
    state["choice"] = choose(state["features"], ef.workers_dims())
    save()
    c = state["choice"]
    print(f"choice: {c['feature']} lam {c['lam']:g} val CE {c['val_ce']:.5f} "
          f"(lowest {c['lowest_feature']} lam {c['lowest_lam']:g} {c['lowest_ce']:.5f})")
    return state


def cmd_select(restart: bool) -> None:
    folder = out_dir()
    log("select", "reads the train and val splits of features/undereye (test is not read)", folder)
    fp = fingerprint()
    state = read_json(folder / SELECT_NAME)
    if restart or state is None or state.get("fingerprint") != fp:
        if state is not None and not restart:
            print("labels, features or settings changed since the last select; starting again")
        state = {"fingerprint": fp, "features": {}}
    joined = load_joined(("train", "val"))
    parts = ud.subsets(joined, ("train", "val"))
    state["counts"] = ud.counts(parts)
    for name, s in parts.items():
        print(f"{name}: {len(s.rows)} rows, {s.sides} sides, {s.images} images")
    run_selection(state, joined, parts["train"], parts["val"], lambda: write_json(folder / SELECT_NAME, state))


# ---------------------------------------------------------------- calibrate


def threshold_table(p: np.ndarray, rows: pd.DataFrame) -> list[dict]:
    """The candidate thresholds on these rows: borderline share and, per rater, clear match and borderline cover."""
    out = []
    for low, high in um.CANDIDATE_THRESHOLDS:
        decided = um.decisions(p, low, high)
        out.append({"low": low, "high": high, "rows": len(rows), "borderline": um.borderline_share(decided),
                    "raters": {r: um.decision_outcomes(decided, um.rater_grades(rows[f"grade_{r}"])) for r in rater_codes(rows)}})
    return out


def train_head(x: np.ndarray, train: ud.Subset, lam: float) -> tuple[coral.Ordinal, np.ndarray, np.ndarray]:
    m = coral.fit(x[train.positions], train.y, lam)
    w, b = m.folded()
    return m, w, b


def fold_gap(m: coral.Ordinal, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    gap = float(np.max(np.abs(coral.folded_logits(x, w, b) - m.logits(x)))) if len(x) else 0.0
    if gap > FOLD_GAP_MAX:
        raise RuntimeError(f"folded weights give logits {gap:.2e} away from the standardized head")
    return gap


def calibration_block(logits: np.ndarray, y: np.ndarray, temperature: float) -> dict:
    return {"before": um.calibration(coral.sigmoid(logits), y), "after": um.calibration(coral.sigmoid(logits / temperature), y)}


def head_dict(m: coral.Ordinal, train: ud.Subset, feature: str) -> dict:
    return {"feature": feature, **m.info(), "train_rows": len(train.rows), "train_sides": train.sides,
            "train_images": train.images, "bias": m.intercept.tolist()}


def prediction_table(rows: pd.DataFrame, logits: np.ndarray, temperature: float) -> pd.DataFrame:
    keep = ["image", "source", "side", "split", "kind", "copy", "target_iod", "iod_px", "reasons", "sex", "age_group", "flagged",
            "p_ge1", "p_ge2", "target_ge1", "target_ge2", *ul.rater_columns(rows)]
    out = rows[keep].copy()
    p = coral.sigmoid(logits / temperature)
    out["logit1"], out["logit2"] = logits[:, 0], logits[:, 1]
    out["p1_raw"], out["p2_raw"] = coral.sigmoid(logits[:, 0]), coral.sigmoid(logits[:, 1])
    out["p1"], out["p2"] = p[:, 0], p[:, 1]
    out["model_grade"] = um.single_grade(p)
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")


def cmd_calibrate() -> None:
    folder = out_dir()
    fp = fingerprint()
    sel = require(SELECT_NAME, "select", fp)
    if "choice" not in sel:
        raise SystemExit("select did not finish; run it again")
    log("calibrate", "reads the train split (fit) and the val split (T, ECE, thresholds); test is not read", folder)
    joined = load_joined(("train", "val"))
    parts = ud.subsets(joined, ("train", "val", "val_all"))
    choice = sel["choice"]
    x = load_matrix(joined, choice["feature"])
    train, val, val_all = parts["train"], parts["val"], parts["val_all"]
    m, w, b = train_head(x, train, choice["lam"])
    gap = fold_gap(m, x[val_all.positions], w, b)
    lv = coral.folded_logits(x[val.positions], w, b)
    temp = coral.fit_temperature(lv, val.y)
    t = temp.temperature
    p = coral.sigmoid(lv / t)
    la = coral.folded_logits(x[val_all.positions], w, b)
    temp_all = coral.fit_temperature(la, val_all.y)
    supporting = {
        "rows": len(val_all.rows), "sides": val_all.sides,
        "temperature": temp_all.as_dict(),
        "ce_with_val_t": coral.cross_entropy(la / t, val_all.y),
        "calibration_with_val_t": calibration_block(la, val_all.y, t),
        "ece_with_own_t": um.calibration(coral.sigmoid(la / temp_all.temperature), val_all.y),
    }
    bundle.write_npz(folder / HEAD_NAME, {"weights": w, "bias": b})
    write_csv(prediction_table(val.rows, lv, t), folder / VAL_PREDICTIONS)
    result = {
        "fingerprint": fp,
        "choice": choice,
        "head": {**head_dict(m, train, choice["feature"]), "standardize_gap": gap, "val_rows": len(val.rows), "val_sides": val.sides},
        "temperature": temp.as_dict(),
        "val": {"rows": len(val.rows), "sides": val.sides, "calibration": calibration_block(lv, val.y, t),
                "model_grades": np.bincount(um.single_grade(p), minlength=3).tolist()},
        "thresholds": threshold_table(p, val.rows),
        "bundle_thresholds": list(criteria.UNDEREYE_THRESHOLDS),
        "supporting": supporting,
    }
    write_json(folder / CALIBRATE_NAME, result)
    cal = result["val"]["calibration"]
    print(f"head {choice['feature']} lam {choice['lam']:g}: {m.iterations} iterations, converged {m.converged}, fold gap {gap:.1e}")
    print(f"T {t:.4f}{' (at a bound)' if temp.at_bound else ''}: val CE {temp.ce_before:.4f} -> {temp.ce_after:.4f}; "
          f"ECE k1 {fmt(cal['before']['k1']['ece'])} -> {fmt(cal['after']['k1']['ece'])}, "
          f"k2 {fmt(cal['before']['k2']['ece'])} -> {fmt(cal['after']['k2']['ece'])}")
    print(f"supporting: T on every val fixed copy ({len(val_all.rows)} rows) {temp_all.temperature:.4f}")


# ---------------------------------------------------------------- evaluate


def evaluate_rows(rows: pd.DataFrame, low: float, high: float) -> dict:
    """Every measure of the head on rows of a prediction table (p1_raw, p2_raw, p1, p2)."""
    raw = rows[["p1_raw", "p2_raw"]].to_numpy(dtype=np.float64)
    p = rows[["p1", "p2"]].to_numpy(dtype=np.float64)
    y = rows[["target_ge1", "target_ge2"]].to_numpy(dtype=np.float64)
    grade = um.single_grade(p)
    raters = rater_codes(rows)
    graded = {r: um.rater_grades(rows[f"grade_{r}"]) for r in raters}
    decided = um.decisions(p, low, high)
    between = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            between.append({"a": raters[i], "b": raters[j], **um.between_raters(graded[raters[i]], graded[raters[j]])})
    return {
        "rows": len(rows),
        "sides": int(rows.groupby(["image", "side"]).ngroups) if len(rows) else 0,
        "images": int(rows["image"].nunique()),
        "label_ge1": float(y[:, 0].sum()) if len(rows) else 0.0,
        "label_ge2": float(y[:, 1].sum()) if len(rows) else 0.0,
        "model_grades": np.bincount(grade, minlength=3).tolist(),
        "versus": {r: um.versus_rater(grade, graded[r]) for r in raters},
        "between": between,
        "calibration": {"before": um.calibration(raw, y), "after": um.calibration(p, y)},
        "borderline": um.borderline_share(decided),
        "decisions": {r: um.decision_outcomes(decided, graded[r]) for r in raters},
    }


def breakdown_masks(table: pd.DataFrame) -> dict[str, np.ndarray]:
    used, clean = table["used"].to_numpy(), (table["flagged"] == 0).to_numpy()
    main = used & clean
    src, sex, age = table["source"].to_numpy(), table["sex"].to_numpy(), table["age_group"].to_numpy()
    masks = {"all": main, "ffhq": main & (src == "ffhq"), "scut": main & (src == "scut")}
    for a in sorted(set(age[src == "ffhq"])):
        masks[f"ffhq {a}"] = main & (src == "ffhq") & (age == a)
    masks["female"], masks["male"] = main & (sex == "female"), main & (sex == "male")
    for c in (LOW_COPY, HIGH_COPY):
        masks[f"copy{c}"] = main & (table["copy"].to_numpy() == c)
    masks["flagged"] = used & ~clean
    every = table["all_copies"].to_numpy() & clean
    masks["all_copies"] = every
    masks["all_copies_ffhq"], masks["all_copies_scut"] = every & (src == "ffhq"), every & (src == "scut")
    return masks


def cmd_evaluate() -> None:
    folder = out_dir()
    fp = fingerprint()
    cal = require(CALIBRATE_NAME, "calibrate", fp)
    head_path = folder / HEAD_NAME
    if not head_path.is_file():
        raise SystemExit("run calibrate first")
    log("evaluate", "reads the test split once, with the head and T of calibrate (train and val are not read)", folder)
    arrays = bundle.read_npz(head_path)
    joined = load_joined(("test",))
    parts = ud.subsets(joined, ("test", "test_all"))
    x = load_matrix(joined, cal["choice"]["feature"])
    t = cal["temperature"]["temperature"]
    logits = coral.folded_logits(x, arrays["weights"], arrays["bias"])
    table = prediction_table(joined, logits, t)
    table["used"] = False
    table.loc[parts["test"].positions, "used"] = True
    table["all_copies"] = False
    table.loc[parts["test_all"].positions, "all_copies"] = True
    table = table[table["all_copies"]].reset_index(drop=True)
    low, high = criteria.UNDEREYE_THRESHOLDS
    results = {k: evaluate_rows(table[m], low, high) for k, m in breakdown_masks(table).items()}
    write_csv(table, folder / PREDICTIONS_NAME)
    write_json(folder / EVALUATE_NAME, {
        "fingerprint": fp,
        "choice": cal["choice"],
        "temperature": t,
        "thresholds": {"low": low, "high": high},
        "copy_iod": copy_iod(read_meta()),
        "counts": ud.counts(parts),
        "results": results,
    })
    r = results["all"]
    for code, v in r["versus"].items():
        print(f"test vs {code}: {v['n']} rows, accuracy {fmt(v['accuracy'], 3)}, kappa 0/1/>=2 {fmt(v['kappa'], 3)}")
    c = r["calibration"]
    print(f"ECE k1 {fmt(c['before']['k1']['ece'])} -> {fmt(c['after']['k1']['ece'])}, "
          f"k2 {fmt(c['before']['k2']['ece'])} -> {fmt(c['after']['k2']['ece'])}; borderline {fmt(r['borderline'], 3)}")


# ---------------------------------------------------------------- cross


def cross_flags(t: float, at_bound: bool, ffhq: dict, scut: dict, scut_ge2_sides: int) -> list[str]:
    out = []
    if not T_USUAL[0] <= t <= T_USUAL[1]:
        out.append(f"T {t:.3f} is outside {T_USUAL[0]}-{T_USUAL[1]}")
    if at_bound:
        out.append(f"T sits at a bound of {coral.T_BOUNDS[0]}-{coral.T_BOUNDS[1]}")
    for k in ("k1", "k2"):
        if k == "k2" and scut_ge2_sides == 0:
            continue
        a, b = ffhq["after"][k]["ece"], scut["after"][k]["ece"]
        if a is not None and b is not None and b >= ECE_SHIFT_RATIO * a and b - a >= ECE_SHIFT_GAP:
            out.append(f"{k}: ECE on SCUT {b:.3f} is much higher than on FFHQ val {a:.3f}")
    return out


def cmd_cross(restart: bool) -> None:
    folder = out_dir()
    fp = fingerprint()
    require(EVALUATE_NAME, "evaluate", fp)
    log("cross", "reads FFHQ train (fit) and val (choice and T) for a head that never goes into the bundle, and the SCUT "
                 "fixed copies of every split (train, val and test) to check it; runs after evaluate", folder)
    state = read_json(folder / CROSS_NAME)
    if restart or state is None or state.get("fingerprint") != fp:
        state = {"fingerprint": fp, "select": {"features": {}}}
    joined = load_joined(ud.SPLITS)
    parts = ud.subsets(joined, ("cross_train", "cross_val", "cross_eval"))
    state["counts"] = ud.counts(parts)
    for name, s in parts.items():
        print(f"{name}: {len(s.rows)} rows, {s.sides} sides, {s.images} images")
    sel = run_selection(state["select"], joined, parts["cross_train"], parts["cross_val"], lambda: write_json(folder / CROSS_NAME, state))
    choice = sel["choice"]
    x = load_matrix(joined, choice["feature"])
    train, val, ev = parts["cross_train"], parts["cross_val"], parts["cross_eval"]
    m, w, b = train_head(x, train, choice["lam"])
    gap = fold_gap(m, x[val.positions], w, b)
    lv = coral.folded_logits(x[val.positions], w, b)
    temp = coral.fit_temperature(lv, val.y)
    t = temp.temperature
    le = coral.folded_logits(x[ev.positions], w, b)
    table = prediction_table(ev.rows, le, t)
    low, high = criteria.UNDEREYE_THRESHOLDS
    clean = table[table["flagged"] == 0]
    scut = evaluate_rows(clean, low, high)
    ffhq_val = calibration_block(lv, val.y, t)
    ge2 = int((clean.groupby(["image", "side"])["target_ge2"].max() >= 0.5).sum()) if len(clean) else 0
    ge1 = int((clean.groupby(["image", "side"])["target_ge1"].max() >= 0.5).sum()) if len(clean) else 0
    state.update({
        "head": {**head_dict(m, train, choice["feature"]), "standardize_gap": gap},
        "temperature": temp.as_dict(),
        "ffhq_val": {"rows": len(val.rows), "sides": val.sides, "calibration": ffhq_val},
        "scut": scut,
        "scut_flagged_rows": int((table["flagged"] == 1).sum()),
        "scut_ge1_sides": ge1,
        "scut_ge2_sides": ge2,
        "flags": cross_flags(t, temp.at_bound, ffhq_val, scut["calibration"], ge2),
    })
    write_json(folder / CROSS_NAME, state)
    c = scut["calibration"]
    print(f"cross head {choice['feature']} lam {choice['lam']:g} T {t:.4f}; SCUT {scut['rows']} rows: "
          f"ECE k1 {fmt(c['before']['k1']['ece'])} -> {fmt(c['after']['k1']['ece'])}, sides at >= 1 {ge1}, at >= 2 {ge2}")
    for f in state["flags"]:
        print(f"NOTE {f}")


# ---------------------------------------------------------------- bundle and fidelity


def bundle_parts(old: bundle.Bundle, cal: dict, head: dict[str, np.ndarray], meta: dict, lock: Path | None = None) -> tuple[dict, dict]:
    """Manifest (without files) and arrays of 0.2.0: the ratio heads and baseline of old, the under-eye head of calibrate."""
    for key in ("backbone", "train_face_iod_px", "degrade"):
        if old.manifest[key] != meta[key]:
            raise SystemExit(f"{key} of the features differs from bundle {old.version}; the ratio heads were trained on other "
                             "features: run train_ratio_heads.py select, evaluate and bundle again first")
    stubs = [t for t in bundle.RATIO_HEADS if old.heads[t].stub]
    if stubs:
        raise SystemExit(f"api/bundle/ {old.version} has stub heads {stubs}; run train_ratio_heads.py bundle first")
    heads, arrays = {}, {}
    for t in bundle.RATIO_HEADS:
        h = copy.deepcopy(old.manifest["heads"][t])
        h.setdefault("from_bundle", old.version)
        new_w, new_b = f"{t}_weights", f"{t}_bias"
        arrays[new_w], arrays[new_b] = np.asarray(old.arrays[h["weights"]], dtype=np.float64), np.asarray(old.arrays[h["bias"]], dtype=np.float64)
        h["weights"], h["bias"] = new_w, new_b
        heads[t] = h
    u = bundle.UNDEREYE_HEAD
    low, high = criteria.UNDEREYE_THRESHOLDS
    arrays[f"{u}_weights"] = np.asarray(head["weights"], dtype=np.float64)
    arrays[f"{u}_bias"] = np.asarray(head["bias"], dtype=np.float64)
    h = cal["head"]
    heads[u] = {
        "feature": cal["choice"]["feature"],
        "stub": False,
        "weights": f"{u}_weights",
        "bias": f"{u}_bias",
        "temperature": cal["temperature"]["temperature"],
        "thresholds": {"low": low, "high": high},
        "lam": cal["choice"]["lam"],
        "loss": LOSS_TEXT,
        "train_rows": h["train_rows"],
        "train_sides": h["train_sides"],
        "train_images": h["train_images"],
        "val_rows": h["val_rows"],
        "converged": h["converged"],
        "selected_on": "val",
        "calibrated_on": "val",
    }
    manifest = bundle.new_manifest(
        version=BUNDLE_VERSION,
        heads=heads,
        baseline=copy.deepcopy(old.manifest["baseline"]),
        backbone=meta["backbone"],
        train_face_iod_px=meta["train_face_iod_px"],
        degrade=meta["degrade"],
        lock=lock,
    )
    return manifest, arrays


def read_predictions(folder: Path) -> pd.DataFrame:
    text = {c: str for c in ("image", "source", "side", "split", "kind", "reasons", "sex", "age_group")}
    df = pd.read_csv(folder / PREDICTIONS_NAME, dtype=text, keep_default_na=False, na_values=[""], encoding="utf-8")
    for c in text:
        df[c] = df[c].fillna("")
    for c in ("used", "all_copies"):
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False}).astype(bool)
    return df


def pick_fidelity(pred: pd.DataFrame, n: int) -> pd.DataFrame:
    """n test sides of the main evaluation at the lowest-IOD copy, ranked by sha256 of a salt, the image and the side."""
    ok = pred[(pred["copy"] == LOW_COPY) & pred["used"] & (pred["flagged"] == 0)]
    key = [hashlib.sha256(f"{FIDELITY_SALT}:{i}:{s}".encode()).hexdigest() for i, s in zip(ok["image"], ok["side"], strict=True)]
    return ok.assign(_key=key).sort_values("_key").head(n).drop(columns="_key").reset_index(drop=True)


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
    results = {}
    for image, copy_no in tqdm(sorted(set(zip(rows["image"], rows["copy"], strict=True))), desc="fidelity", unit="image"):
        results[(image, int(copy_no))] = ef.regenerate(ef.UNDEREYE, image, image_paths[image], ef.FIXED, int(copy_no), workers, config)[1]
    crops = [ef.crop_of(ef.UNDEREYE, results[(r["image"], int(r["copy"]))], r["side"]) for _, r in rows.iterrows()]
    feats = backbone.features(np.stack(crops))
    head = b.heads[bundle.UNDEREYE_HEAD]
    p = head.predict(feats[head.feature])
    out = []
    for i, (_, r) in enumerate(rows.iterrows()):
        res = results[(r["image"], int(r["copy"]))]
        region = ef.SIDE_REGION[r["side"]]
        decision = um.decisions(p[i : i + 1], head.low, head.high)[0]
        stored = um.decisions(np.array([[r["p1"], r["p2"]]]), head.low, head.high)[0]
        out.append({
            "image": r["image"], "side": r["side"], "copy": int(r["copy"]), "iod_px": res.quality.iod_px,
            "reasons": ";".join(res.region_failed.get(region, ())), "reasons_stored": r["reasons"],
            "p1": float(p[i, 0]), "p2": float(p[i, 1]), "p1_stored": float(r["p1"]), "p2_stored": float(r["p2"]),
            "diff": float(max(abs(p[i, 0] - r["p1"]), abs(p[i, 1] - r["p2"]))),
            "same_decision": decision.as_dict() == stored.as_dict(),
        })
    worst = max((e["diff"] for e in out), default=None)
    return {"sides": len(out), "images": len({e["image"] for e in out}), "tolerance": FIDELITY_TOLERANCE, "max_diff": worst,
            "same_reasons": all(e["reasons"] == e["reasons_stored"] for e in out),
            "same_decisions": all(e["same_decision"] for e in out),
            "passed": worst is not None and worst <= FIDELITY_TOLERANCE, "rows": out}


def cmd_bundle() -> int:
    from facemeasure.backbone import Backbone
    from facemeasure.landmarks import FaceModels

    folder = out_dir()
    fp = fingerprint()
    cal = require(CALIBRATE_NAME, "calibrate", fp)
    ev = require(EVALUATE_NAME, "evaluate", fp)
    if ev["choice"] != cal["choice"]:
        raise SystemExit("evaluate used another head than calibrate; run evaluate again")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", bundle.BundleWarning)
        try:
            old = bundle.load_bundle()
        except bundle.BundleError as e:
            raise SystemExit(f"cannot read api/bundle/: {e}") from None
    log("bundle", f"writes api/bundle/ {BUNDLE_VERSION}: jaw, nose_base and baseline from bundle {old.version}, the under-eye head "
                  f"of calibrate; reads {FIDELITY_SIDES} test sides again for the fidelity check only (no new measures)", folder)
    manifest, arrays = bundle_parts(old, cal, bundle.read_npz(folder / HEAD_NAME), read_meta())
    b = bundle.write_bundle(manifest, arrays)
    print(f"wrote bundle {b.version} to {bundle.bundle_dir()}; stub: {b.stub}")
    rows = pick_fidelity(read_predictions(folder), FIDELITY_SIDES)
    image_paths = {s.image: s.path for s in ef.undereye_sources()}
    with FaceModels() as models:
        check = fidelity_check(b, rows, models, Backbone(), quality.load_config(), image_paths)
    check["bundle_version"] = b.version
    check["ratio_heads_from"] = old.version
    write_json(folder / FIDELITY_NAME, check)
    print(f"fidelity: {check['sides']} sides of {check['images']} images, largest difference {sci(check['max_diff'])} "
          f"(at most {FIDELITY_TOLERANCE}): {'passed' if check['passed'] else 'FAILED'}")
    return 0 if check["passed"] else 1


# ---------------------------------------------------------------- figures


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def reliability_figure(cal: dict, title: str, path: Path) -> None:
    """Mean soft label against mean probability per bin, before and after T, with the rows per bin below."""
    fig = Figure(figsize=(10, 6), layout="constrained", facecolor=SURFACE)
    axes = fig.subplots(2, 2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for j, k in enumerate(("k1", "k2")):
        ax = axes[0, j]
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
        for stage, style, marker in (("before", "--", "s"), ("after", "-", "o")):
            bins = [b for b in cal[stage][k]["bins"] if b["n"]]
            e = cal[stage][k]["ece"]
            ax.plot([b["mean_p"] for b in bins], [b["mean_y"] for b in bins], color=STAGE_COLORS[stage], linestyle=style,
                    linewidth=2, marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                    label=f"{'before T' if stage == 'before' else 'after T'} (ECE {fmt(e, 3)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"P(grade >= {k[1]})", fontsize=10, color=INK)
        ax.set_ylabel("mean label in bin", fontsize=9, color=MUTED)
        ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor=INK)
        bar = axes[1, j]
        _style(bar)
        for offset, stage in ((-0.022, "before"), (0.022, "after")):
            bins = cal[stage][k]["bins"]
            bar.bar([(b["low"] + b["high"]) / 2 + offset for b in bins], [b["n"] for b in bins], width=0.04,
                    color=STAGE_COLORS[stage], edgecolor=SURFACE, linewidth=0)
        bar.set_xlabel("predicted probability (10 equal bins)", fontsize=9, color=MUTED)
        bar.set_ylabel("rows", fontsize=9, color=MUTED)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(path, dpi=100, metadata={"Software": None}, facecolor=SURFACE)


def grades_figure(dist: list[dict], path: Path) -> None:
    """Share of each grade per group, all raters pooled, as stacked bars."""
    order = [d for d in dist if d["group"].startswith("ffhq ")] + [d for d in dist if d["group"] == "scut"]
    labels, shares, totals = [], [], []
    for d in order:
        counts = np.array([sum(r[str(g)] for r in d["raters"].values()) for g in ul.GRADES4], dtype=np.float64)
        total = counts.sum()
        labels.append(d["group"].replace("ffhq ", "FFHQ ").replace("scut", "SCUT"))
        shares.append(counts / total if total else counts)
        totals.append(int(total))
    fig = Figure(figsize=(9, 0.55 * len(order) + 1.4), layout="constrained", facecolor=SURFACE)
    ax = fig.subplots()
    _style(ax)
    ax.grid(axis="y", visible=False)
    left = np.zeros(len(order))
    y = np.arange(len(order))[::-1]
    for g in ul.GRADES4:
        width = np.array([s[g] for s in shares])
        ax.barh(y, width, left=left, color=GRADE_COLORS[g], edgecolor=SURFACE, linewidth=2, height=0.62, label=f"grade {g}")
        left += width
    ax.set_yticks(y, [f"{lab} (n {n})" for lab, n in zip(labels, totals, strict=True)], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of graded sides", fontsize=9, color=MUTED)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=INK)
    fig.suptitle("Under-eye grades by group (all raters pooled; unassessable sides left out)", fontsize=10, color=INK)
    fig.savefig(path, dpi=100, metadata={"Software": None}, facecolor=SURFACE)


# ---------------------------------------------------------------- report


def sci(x) -> str:
    return "–" if x is None else f"{x:.1e}"


def share(x) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def kappa_level(k) -> str:
    if k is None or (isinstance(k, float) and not math.isfinite(k)):
        return ""
    return ("poor" if k < 0 else "slight" if k <= 0.2 else "fair" if k <= 0.4 else "moderate" if k <= 0.6
            else "substantial" if k <= 0.8 else "almost perfect")


def labels_section(lab: dict | None) -> str:
    if lab is None:
        return "ยังไม่ได้รัน `labels`"
    files = [[f["code"], "ใช้" if f["used"] else "ข้าม", f["rows"], ", ".join(f["raters"]) or "–", ", ".join(f["set_ids"]) or "–",
              f["reason"] or "–"] for f in lab["files"]]
    excluded = ", ".join(f"{k} {v}" for k, v in lab["excluded"].items()) or "0"
    n = len(lab["raters"])
    who = f"ผู้ให้เกรด {n} คน ({', '.join(lab['raters'])})" + (f" และเกรดร่วม (`consensus`) {lab['consensus_sides']} ข้าง" if lab["consensus"] else "")
    return (f"{who} {lab['images']} ภาพ {lab['sides']} ข้าง ข้างที่ตัดออก: {excluded} ข้างที่ `flagged` (ถุงใต้ตาหรือรอยคล้ำ "
            f"ตามผู้ให้เกรดตั้งแต่ครึ่งหนึ่ง) {lab['flagged']} ข้าง ซึ่งไม่ใช้เทรนแต่รายงานผลบนชุด test แยกไว้\n\n"
            "ไฟล์ใน `data/grading/results/` (ชื่อไฟล์และชื่อผู้ให้เกรดไม่อยู่ในรายงาน ใช้รหัสแทน)\n\n"
            + md_table(["ไฟล์", "ผล", "แถว", "ผู้ให้เกรด", "set_id", "หมายเหตุ"], files)
            + "\n\nเมื่อมีผู้ให้เกรดคนเดียว label ของแต่ละข้างคือเกรดของเขา P(เกรด ≥ 1) และ P(เกรด ≥ 2) จึงเป็น 0 หรือ 1 (model.md หัวข้อ 5.2 และ 6)")


GROUP_TEXT = {"all": "ทั้งหมด", "ffhq": "FFHQ", "scut": "SCUT", "female": "หญิง", "male": "ชาย", "train": "ชุด train",
              "val": "ชุด val", "test": "ชุด test", "pilot": "200 ภาพแรก (pilot)"}


def group_name(g: str) -> str:
    return GROUP_TEXT.get(g) or g.replace("ffhq ", "FFHQ อายุ ")


def distribution_section(lab: dict | None, figure: str | None) -> str:
    if lab is None:
        return "ยังไม่ได้รัน `labels`"
    parts = []
    for code in lab["raters"]:
        rows = []
        for d in lab["distribution"]:
            c = d["raters"][code]
            graded = sum(c[str(g)] for g in ul.GRADES4)
            rows.append([group_name(d["group"]), d["sides"], *[f"{c[str(g)]} ({share(c[str(g)] / graded if graded else None)})" for g in ul.GRADES4],
                         c["na"], d["flagged"], share(d["ge2_share"]), share(d["ge2_share_unflagged"])])
        parts.append(f"**{code}**\n\n" + md_table(["กลุ่ม", "ข้าง", "เกรด 0", "เกรด 1", "เกรด 2", "เกรด 3", "ประเมินไม่ได้", "flagged",
                                                   "≥2 (ทุกข้าง)", "≥2 (ไม่ flagged)"], rows))
    sec = ", ".join(f"{k} {fmt(v, 1)} วินาที" for k, v in lab["median_seconds"].items())
    overall = lab["distribution"][0]
    if lab["low_ge2"]:
        warn = (f"**คำเตือน: ข้างที่ผู้ให้เกรดตั้งแต่ครึ่งหนึ่งให้ตั้งแต่ 2 มีไม่ถึง {ul.LOW_GE2_SHARE:.0%}** "
                "จุดเปลี่ยนที่ 2 จะประมาณได้ไม่นิ่ง")
    else:
        warn = (f"ข้างที่ผู้ให้เกรดตั้งแต่ครึ่งหนึ่งให้ตั้งแต่ 2 มี {share(overall['ge2_share'])} ของข้างที่ไม่ถูกตัด "
                f"({share(overall['ge2_share_unflagged'])} เมื่อไม่นับ flagged) ไม่ต่ำกว่าเกณฑ์เตือน {ul.LOW_GE2_SHARE:.0%} "
                "แต่ดูสัดส่วนของชุด val และ test ในหัวข้อจำนวนแถวด้วย")
    return ("เปอร์เซ็นต์ในวงเล็บคิดจากข้างที่ให้เกรด ไม่นับข้างที่ประเมินไม่ได้ คอลัมน์ ≥2 คือสัดส่วนของข้างที่ไม่ถูกตัดซึ่งผู้ให้เกรดตั้งแต่ครึ่งหนึ่งให้ตั้งแต่ 2 "
            "SCUT-FBP5500 ไม่มีป้ายอายุรายภาพ\n\n" + "\n\n".join(parts) + f"\n\n{warn}\n\nเวลาเกรดมัธยฐานต่อภาพ (ตามแท็บสรุปผลของเครื่องมือ): {sec}"
            + (f"\n\n![grades](figures/{figure})" if figure else ""))


def agreement_table(part: dict) -> str:
    rows = [[f"{p['a']} กับ {p['b']}", p["n"], share(p["exact"]), share(p["within1"]), fmt(p["kappa4"], 3), kappa_level(p["kappa4"]),
             "ถึง" if p["kappa4"] is not None and math.isfinite(p["kappa4"]) and p["kappa4"] >= ul.KAPPA_TARGET else "ไม่ถึง",
             fmt(p["kappa3"], 3)] for p in part["pairs"]]
    return (md_table(["คู่", "ข้าง", "ตรงกัน", "ต่างไม่เกิน 1", "κ เกรด 0–3", "Landis–Koch", f"เป้า {ul.KAPPA_TARGET}", "κ เกรด 0/1/≥2"], rows)
            + f"\n\nICC(2,1) เกรด 0–3 บน {part['icc_sides']} ข้างที่ทุกคนให้เกรด: {fmt(part['icc21'], 3)}")


def agreement_section(lab: dict | None) -> str:
    if lab is None:
        return "ยังไม่ได้รัน `labels`"
    if lab["agreement"] is None:
        return ("**วัดความสอดคล้องระหว่างผู้ให้เกรดไม่ได้ เพราะมีผู้ให้เกรดคนเดียว** (model.md หัวข้อ 5.2 และ 7.2) label จึงเป็นการตัดสินของคนเดียวที่ไม่ใช่แพทย์ "
                "และความน่าจะเป็นหลังปรับเทียบบอกเพียงว่าตรงกับผู้ให้เกรดคนนี้แค่ไหน ถ้าภายหลังมีผู้ให้เกรดเพิ่ม ให้ทุกคนเกรดชุด `full_600` แล้วรัน `labels` ใหม่ "
                "หัวข้อนี้จะแสดง weighted kappa และ ICC(2,1) ที่ต้องตรงกับแท็บ \"สรุปผลหลายคน\" ของเครื่องมือ")
    a = lab["agreement"]
    return ("นับเฉพาะข้างที่ทั้งคู่ให้เกรด κ คือ weighted kappa แบบ linear (Cicchetti–Allison) ใช้นิยามเดียวกับแท็บ \"สรุปผลหลายคน\" ของเครื่องมือ "
            "ตัวเลขจึงต้องตรงกันเมื่อเปิดเฉพาะ CSV ของผู้ให้เกรด (ไม่รวมไฟล์ `consensus`) "
            f"เป้าคือ {ul.KAPPA_TARGET} ส่วนแพทย์ที่ประเมินตัวจริงใน \\[7\\] ได้ค่ามัธยฐาน {ul.PAPER_KAPPA['non_caucasian']} ในกลุ่มที่รวมเชื้อชาติอื่น "
            f"และ {ul.PAPER_KAPPA['caucasian']} ในกลุ่มผิวขาว\n\n"
            f"### {lab['pilot_images']} ภาพแรก (pilot)\n\n" + agreement_table(a["pilot"]) + "\n\n### ทั้ง 600 ภาพ\n\n" + agreement_table(a["all"]))


COUNT_TEXT = {"train": "train", "val": "val", "test": "test", "val_all": "val ทุกสำเนา (ผลประกอบ)", "test_all": "test ทุกสำเนา (ผลประกอบ)",
              "cross_train": "cross: FFHQ train", "cross_val": "cross: FFHQ val", "cross_eval": "cross: SCUT ทุกชุด"}


def counts_section(sel: dict | None, ev: dict | None, cal: dict | None, cross: dict | None) -> str:
    records = [*(sel or {}).get("counts", []), *(ev or {}).get("counts", []), *(cross or {}).get("counts", [])]
    if not records:
        return "ยังไม่ได้รัน `select`"
    rows = [[COUNT_TEXT.get(r["subset"], r["subset"]), r["source"], r["rows"], r["sides"], r["images"], r["flagged_rows"],
             fmt(r["ge1_rows"], 0), fmt(r["ge2_rows"], 0), share(r["ge2_rows"] / r["rows"] if r["rows"] else None)]
            for r in records]
    sup = ""
    if cal is not None:
        s = cal["supporting"]
        sup = f"\n\nผลประกอบ `val_all` มี {s['rows']} แถว {s['sides']} ข้าง"
    return ("แต่ละข้างมีหลายสำเนา (train 6 สำเนา val และ test 2 สำเนา fixed ที่ IOD ต่ำสุดและสูงสุด) จึงนับทั้งแถวและข้าง\n\n"
            "- train: สำเนาแบบ train ของข้างที่พบใบหน้า ไม่ถูกตัด และไม่ flagged ไม่กรองคุณภาพ\n"
            "- val: สำเนาแบบ fixed ของข้างที่ไม่ถูกตัด ไม่ flagged และผ่านเกณฑ์คุณภาพของข้างนั้น (`reasons` ว่าง)\n"
            "- test: สำเนาแบบ fixed ที่ผ่านเกณฑ์คุณภาพ ข้างที่ flagged อยู่ในตารางนี้แต่ผลหลักไม่นับ\n"
            "- ผลประกอบ (`val_all`, `test_all`): ทุกสำเนา fixed ไม่กรองคุณภาพ ไม่ใช้เลือกค่าและไม่ลง bundle ใช้ดูว่าตัวเลขเปลี่ยนแค่ไหนเมื่อมีแถวมากขึ้น "
            "เพราะเกณฑ์คุณภาพ (ส่วนใหญ่คือ `smiling`) ตัดสำเนา fixed ของ FFHQ ไปราวสามในสี่\n\n"
            "แถว ≥1 และ ≥2 คือผลรวมของ soft label ซึ่งเท่ากับจำนวนแถวเมื่อมีผู้ให้เกรดคนเดียว\n\n"
            + md_table(["ชุดย่อย", "แหล่ง", "แถว", "ข้าง", "ภาพ", "แถว flagged", "แถว ≥1", "แถว ≥2", "สัดส่วน ≥2"], rows) + sup)


def select_table(state: dict) -> str:
    feats = [f for f in FEATURE_TYPES if f in state["features"]]
    c = state.get("choice", {})
    lams = sorted({lam for f in feats for lam in state["features"][f]["grid"]["lambdas"]})
    rows = []
    for lam in lams:
        row = [f"{lam:g}"]
        for f in feats:
            g = state["features"][f]["grid"]
            if lam in g["lambdas"]:
                v = fmt(g["ce"][g["lambdas"].index(lam)], 5)
                row.append(f"**{v}**" if (f == c.get("feature") and lam == c.get("lam")) else v)
            else:
                row.append("")
        rows.append(row)
    notes = []
    for f in feats:
        g = state["features"][f]["grid"]
        if g["expanded"]:
            notes.append(f"`{f}` ขยายช่วงไปที่ λ {', '.join(f'{a:g}' for a in g['expanded'])}" + (" แล้วยังติดขอบ" if g["at_edge"] else ""))
        bad = [fit for fit in state["features"][f]["fits"] if not fit["converged"]]
        if bad:
            lams = ", ".join(f"{b['lam']:g}" for b in bad)
            notes.append(f"`{f}` L-BFGS ไม่ลู่เข้าภายใน {coral.MAX_ITER} รอบที่ λ {lams}")
    slow = max((fit["iterations"] for f in feats for fit in state["features"][f]["fits"]), default=0)
    notes.append(f"L-BFGS ใช้มากสุด {slow} รอบ (เพดาน {coral.MAX_ITER})")
    header = ["λ", *[f"`{f}` ({state['features'][f]['dim']})" for f in feats]]
    text = md_table(header, rows)
    if c:
        text += (f"\n\nเลือก `{c['feature']}` λ {c['lam']:g} CE {fmt(c['val_ce'], 5)} ส่วนค่าต่ำสุดคือ `{c['lowest_feature']}` "
                 f"λ {c['lowest_lam']:g} CE {fmt(c['lowest_ce'], 5)}")
    return text + "\n\n" + "\n".join(f"- {n}" for n in notes)


def select_section(sel: dict | None) -> str:
    if sel is None or "choice" not in sel:
        return "ยังไม่ได้รัน `select`"
    train = next((r for r in sel["counts"] if r["subset"] == "train" and r["source"] == "all"), {})
    val = next((r for r in sel["counts"] if r["subset"] == "val" and r["source"] == "all"), {})
    return (f"cross-entropy บนชุด val ({val.get('rows', '?')} แถว) ของหัวที่เทรนด้วยชุด train ({train.get('rows', '?')} แถว) "
            f"ทุกชนิด feature คู่กับ λ {', '.join(f'{a:g}' for a in LAMBDAS)} loss คือ {LOSS_TEXT_TH} เลือกค่าที่ต่ำสุด "
            f"แต่ถ้าต่างกันไม่ถึง {ridge.TOLERANCE:.0%} เลือกชนิดที่มิติน้อยกว่าแล้ว λ ที่มากกว่า ถ้า λ ที่ดีที่สุดอยู่ที่ขอบ ขยายช่วงอีก "
            f"{ridge.EXPAND_DECADES} หลักหนึ่งครั้ง (กติกาเดียวกับหัวกรามและฐานจมูก) ตัวหนาคือค่าที่เลือก\n\n" + select_table(sel))


def ece_rows(cal: dict, name: str) -> list[list]:
    return [[name, f"P(เกรด ≥ {k[1]})", fmt(cal["before"][k]["ece"], 4), fmt(cal["after"][k]["ece"], 4)] for k in ("k1", "k2")]


def t_note(t: float, at_bound: bool) -> str:
    out = []
    if not T_USUAL[0] <= t <= T_USUAL[1]:
        out.append(f"**T อยู่นอกช่วง {T_USUAL[0]}–{T_USUAL[1]}** ซึ่งเป็นสัญญาณว่าการปรับเทียบอาจเลื่อนเมื่อข้อมูลเปลี่ยน (model.md หัวข้อ 4.2)")
    if at_bound:
        out.append(f"**T ชนขอบของช่วงที่ค้น ({coral.T_BOUNDS[0]}–{coral.T_BOUNDS[1]})**")
    return " ".join(out)


def outcomes_cells(o: dict) -> list:
    return [o["n"], share(o["borderline"]), share(o["clear_match"]), share(o["borderline_cover"])]


def calibrate_section(cal: dict | None) -> str:
    if cal is None:
        return "ยังไม่ได้รัน `calibrate`"
    t = cal["temperature"]
    h = cal["head"]
    v = cal["val"]
    s = cal["supporting"]
    raters = list(cal["thresholds"][0]["raters"]) if cal["thresholds"] else []
    th_rows = []
    for row in cal["thresholds"]:
        cur = (row["low"], row["high"]) == tuple(cal["bundle_thresholds"])
        name = f"{row['low']:g} / {row['high']:g}" + (" (ใช้อยู่)" if cur else "")
        for r in raters:
            o = row["raters"][r]
            th_rows.append([name, r, *outcomes_cells(o)])
    note = t_note(t["temperature"], t["at_bound"])
    return (f"เทรนหัว `{cal['choice']['feature']}` λ {cal['choice']['lam']:g} ด้วยชุด train {h['train_rows']} แถว ({h['train_sides']} ข้าง "
            f"{h['train_images']} ภาพ) L-BFGS {h['iterations']} รอบ ลู่เข้า: {'ใช่' if h['converged'] else 'ไม่'} "
            f"bias b₁ = {h['bias'][0]:.3f} และ b₂ = {h['bias'][1]:.3f} การรวม standardization เข้าไปใน w และ b ทำให้ logit ต่างไปไม่เกิน {h['standardize_gap']:.1e}\n\n"
            f"T = **{t['temperature']:.4f}** หาบนชุด val {v['rows']} แถว ({v['sides']} ข้าง) ในช่วง {t['bounds'][0]}–{t['bounds'][1]} "
            f"cross-entropy {t['ce_before']:.4f} → {t['ce_after']:.4f} {note}\n\n"
            "ECE แบบ 10 ช่วงเท่ากันของชุด val เทียบความน่าจะเป็นกับ soft label\n\n"
            + md_table(["ชุด", "จุดเปลี่ยน", "ECE ก่อนปรับ", "ECE หลังปรับ"], ece_rows(v["calibration"], "val"))
            + "\n\n### เกณฑ์ทางเลือกบนชุด val\n\n"
            "ใช้ความน่าจะเป็นหลังปรับ ตัดสินด้วย `undereye.decide` ก้ำกึ่งคือสัดส่วนที่ผลเป็นช่วงเกรด ชัดและตรงคือสัดส่วนของผลชัดที่ตรงกับเกรด 0/1/≥2 "
            "ของผู้ให้เกรด ก้ำกึ่งและครอบคือสัดส่วนของผลก้ำกึ่งที่ช่วงครอบเกรดของผู้ให้เกรด ระบบยังใช้ 0.3/0.7 จนกว่าผู้ใช้จะตกลงเปลี่ยน\n\n"
            + md_table(["เกณฑ์ (low / high)", "ผู้ให้เกรด", "n", "ก้ำกึ่ง", "ชัดและตรง", "ก้ำกึ่งและครอบ"], th_rows)
            + "\n\n### ผลประกอบ: ทุกสำเนา fixed ของชุด val\n\n"
            f"ถ้าหา T บนทุกสำเนา fixed ของชุด val ({s['rows']} แถว {s['sides']} ข้าง ไม่กรองคุณภาพ) จะได้ T = {s['temperature']['temperature']:.4f} "
            f"เทียบกับ {t['temperature']:.4f} จากสำเนาที่ผ่านเกณฑ์ ค่านี้ไม่ได้ใช้ใน bundle\n\n"
            + md_table(["ชุด", "จุดเปลี่ยน", "ECE ก่อนปรับ", "ECE หลังปรับด้วย T ของ val"], ece_rows(s["calibration_with_val_t"], "val ทุกสำเนา")))


def subset_name(k: str, copy_iod_px: dict) -> str:
    if k.startswith("copy"):
        return f"สำเนา IOD {copy_iod_px[k.removeprefix('copy')]:g} px"
    return SUBSET_TEXT.get(k) or k.replace("ffhq ", "FFHQ อายุ ")


def confusion_table(conf: list[list[int]]) -> str:
    return md_table(["ผู้ให้เกรด (แถว) / โมเดล (คอลัมน์)", "0", "1", "≥2"], [[g, *row] for g, row in zip(("0", "1", "≥2"), conf, strict=True)])


def test_section(ev: dict | None, figure: str | None) -> str:
    if ev is None:
        return "ยังไม่ได้รัน `evaluate`"
    r = ev["results"]["all"]
    raters = list(r["versus"])
    parts = []
    main_rows = [[code, v["n"], share(v["accuracy"]), fmt(v["kappa"], 3), kappa_level(v["kappa"])] for code, v in r["versus"].items()]
    parts.append(f"ชุด test ที่ผ่านเกณฑ์คุณภาพและไม่ flagged {r['rows']} แถว ({r['sides']} ข้าง {r['images']} ภาพ) label ≥1 {fmt(r['label_ge1'], 0)} แถว "
                 f"≥2 {fmt(r['label_ge2'], 0)} แถว เกรดเดี่ยวของโมเดลคือจำนวนจุดเปลี่ยนที่ P(เกรด ≥ k) ตั้งแต่ 0.5 "
                 f"(โมเดลให้เกรด 0/1/≥2 = {'/'.join(str(n) for n in r['model_grades'])} แถว)\n\n"
                 + md_table(["เทียบกับ", "n", "accuracy", "κ เกรด 0/1/≥2 (linear)", "Landis–Koch"], main_rows))
    if r["between"]:
        b = [[f"{x['a']} กับ {x['b']}", x["n"], share(x["accuracy"]), fmt(x["kappa"], 3)] for x in r["between"]]
        parts.append("ระหว่างผู้ให้เกรดบนแถวเดียวกัน\n\n" + md_table(["คู่", "n", "ตรงกัน", "κ เกรด 0/1/≥2"], b))
    else:
        parts.append(f"มีผู้ให้เกรดคนเดียว จึงไม่มี kappa ระหว่างผู้ให้เกรดให้วางคู่ ค่าที่ใช้เป็นหลักอ้างอิงคร่าว ๆ คือ "
                     f"{ul.PAPER_KAPPA['non_caucasian']} และ {ul.PAPER_KAPPA['caucasian']} ของแพทย์ใน \\[7\\] **ซึ่งเป็นการเทียบคนละแบบ** "
                     "คือแพทย์ 3 คนประเมินผู้เข้าร่วมตัวจริงด้วยเกรด 0–3 ส่วนนี่คือโมเดลเทียบกับผู้ให้เกรดคนเดียวที่ไม่ใช่แพทย์บนภาพถ่าย ด้วยเกรด 0/1/≥2 (model.md หัวข้อ 7.1 ข้อ 3)")
    for code in raters:
        parts.append(f"confusion matrix เทียบกับ {code}\n\n" + confusion_table(r["versus"][code]["confusion"]))
    parts.append("### การปรับเทียบ\n\nECE แบบ 10 ช่วงเท่ากัน เทียบความน่าจะเป็นกับ soft label ก่อนปรับคือ T = 1\n\n"
                 + md_table(["ชุด", "จุดเปลี่ยน", "ECE ก่อนปรับ", "ECE หลังปรับ"], ece_rows(r["calibration"], "test"))
                 + (f"\n\n![reliability test](figures/{figure})\n\nกราฟบนคือค่าเฉลี่ยของ label เทียบกับค่าเฉลี่ยของความน่าจะเป็นในแต่ละช่วง เส้นจุดคือการปรับเทียบที่สมบูรณ์ "
                    "กราฟล่างคือจำนวนแถวในแต่ละช่วง ตัวเลขเดียวกันอยู่ใน `evaluate.json`" if figure else ""))
    th = ev["thresholds"]
    dec = [[code, *outcomes_cells(o)] for code, o in r["decisions"].items()]
    parts.append(f"### การตัดสินด้วยเกณฑ์ {th['low']:g}/{th['high']:g} ของ bundle\n\n`undereye.decide` บนความน่าจะเป็นหลังปรับ\n\n"
                 + md_table(["ผู้ให้เกรด", "n", "ก้ำกึ่ง", "ชัดและตรง", "ก้ำกึ่งและครอบ"], dec))
    rows = []
    for k, v in ev["results"].items():
        if k.startswith("all_copies"):
            continue
        for code in raters:
            vs, d = v["versus"][code], v["decisions"][code]
            rows.append([subset_name(k, ev["copy_iod"]), code, v["rows"], v["sides"], share(vs["accuracy"]), fmt(vs["kappa"], 3),
                         fmt(v["calibration"]["after"]["k1"]["ece"], 3), fmt(v["calibration"]["after"]["k2"]["ece"], 3),
                         share(d["borderline"]), share(d["clear_match"]), share(d["borderline_cover"])])
    parts.append("### แยกกลุ่ม\n\nทุกกลุ่มยกเว้น flagged คือแถวที่ไม่ flagged กลุ่มที่มีแถวน้อยให้ตัวเลขที่ไม่แน่นอนมาก ECE เป็นค่าหลังปรับ\n\n"
                 + md_table(["กลุ่ม", "ผู้ให้เกรด", "แถว", "ข้าง", "accuracy", "κ", "ECE ≥1", "ECE ≥2", "ก้ำกึ่ง", "ชัดและตรง", "ก้ำกึ่งและครอบ"], rows))
    sup = []
    for k in ("all_copies", "all_copies_ffhq", "all_copies_scut"):
        v = ev["results"][k]
        for code in raters:
            vs, d = v["versus"][code], v["decisions"][code]
            sup.append([subset_name(k, ev["copy_iod"]), code, v["rows"], v["sides"], share(vs["accuracy"]), fmt(vs["kappa"], 3),
                        fmt(v["calibration"]["before"]["k1"]["ece"], 3), fmt(v["calibration"]["after"]["k1"]["ece"], 3),
                        fmt(v["calibration"]["before"]["k2"]["ece"], 3), fmt(v["calibration"]["after"]["k2"]["ece"], 3),
                        share(d["borderline"]), share(d["clear_match"])])
    parts.append("### ผลประกอบ: ทุกสำเนา fixed ของชุด test\n\nรวมสำเนาที่ไม่ผ่านเกณฑ์คุณภาพ (ส่วนใหญ่ยิ้ม) ซึ่ง api จะไม่ให้เกรด ไม่นับข้างที่ flagged "
                 "ใช้ดูความไม่แน่นอนของผลหลักที่มีแถวน้อย ไม่ใช่ผลที่ระบบจะได้จริง\n\n"
                 + md_table(["กลุ่ม", "ผู้ให้เกรด", "แถว", "ข้าง", "accuracy", "κ", "ECE ≥1 ก่อน", "ECE ≥1 หลัง", "ECE ≥2 ก่อน", "ECE ≥2 หลัง",
                             "ก้ำกึ่ง", "ชัดและตรง"], sup))
    return "\n\n".join(parts)


def cross_section(cross: dict | None, figure: str | None, ev: dict | None = None) -> str:
    if cross is None or "scut" not in cross:
        return "ยังไม่ได้รัน `cross`"
    c, t, h, s = cross["select"]["choice"], cross["temperature"], cross["head"], cross["scut"]
    rows = [*ece_rows(cross["ffhq_val"]["calibration"], "FFHQ val (ใช้หา T)"), *ece_rows(s["calibration"], "SCUT")]
    vs = [[code, v["n"], share(v["accuracy"]), fmt(v["kappa"], 3)] for code, v in s["versus"].items()]
    lines = []
    if cross["scut_ge2_sides"] == 0:
        lines.append("- **ใต้ตาของภาพ SCUT ไม่มีข้างที่ได้เกรดตั้งแต่ 2 จึงตรวจจุดเปลี่ยนที่ 2 ข้ามชุดข้อมูลไม่ได้** ผลของจุดเปลี่ยนที่ 1 เป็นหลัก "
                     "ECE ของ P(เกรด ≥ 2) ข้างบนบอกเพียงว่าโมเดลให้ความน่าจะเป็นต่ำแค่ไหนกับข้างที่ label เป็น 0 ทั้งหมด")
    lines.append(f"- SCUT มีข้างที่ได้เกรดตั้งแต่ 1 อยู่ {cross['scut_ge1_sides']} ข้าง จากที่ใช้ {s['sides']} ข้าง kappa จึงขึ้นกับข้างไม่กี่ข้าง")
    g = s["model_grades"]
    bins = [b for b in s["calibration"]["after"]["k1"]["bins"] if b["n"]]
    mean_p = sum(b["n"] * b["mean_p"] for b in bins) / sum(b["n"] for b in bins) if bins else None
    train_share = next((r["ge1_rows"] / r["rows"] for r in cross.get("counts", []) if r["subset"] == "cross_train" and r["source"] == "all"
                        and r["rows"]), None)
    text = (f"- หัว FFHQ ให้ P(เกรด ≥ 1) หลังปรับกับแถว SCUT เฉลี่ย {fmt(mean_p, 2)} ทุกแถวอยู่ในช่วง "
            f"{bins[0]['low']:.1f}–{bins[-1]['high']:.1f} ใกล้กับสัดส่วนเกรด ≥1 ของภาพ FFHQ ที่ใช้เทรน ({share(train_share)}) "
            f"จึงได้เกรด 0/1/≥2 = {g[0]}/{g[1]}/{g[2]} แถว ขณะที่ label ≥1 มี {fmt(s['label_ge1'], 0)} แถว "
            "ECE ที่สูงจึงมาจากสัดส่วนของเกรดที่ต่างกันมากระหว่างสองชุดเป็นหลัก") if bins else ""
    main = (ev or {}).get("results", {}).get("scut")
    if text and main is not None and main["rows"]:
        m = main["model_grades"]
        text += (f" ส่วนหัวหลักซึ่งเห็นภาพ SCUT ของชุด train ตอนเทรน ให้เกรด 0/1/≥2 กับแถว SCUT ของชุด test = {m[0]}/{m[1]}/{m[2]} แถว "
                 "ข้อมูลชุดนี้แยกไม่ได้ว่าหัวหลักแยกตามร่องใต้ตา ตามลักษณะของชุดข้อมูล (แสง กล้อง การแต่งภาพ) หรือผู้ให้เกรดเห็นใต้ตาของ SCUT ต่างจาก FFHQ")
    if text:
        lines.append(text)
    for f in cross["flags"]:
        lines.append(f"- **ข้อสังเกต:** {flag_text(f)}")
    if not cross["flags"]:
        lines.append(f"- T อยู่ในช่วง {T_USUAL[0]}–{T_USUAL[1]} และ ECE หลังปรับของ SCUT ไม่สูงกว่าของ FFHQ val ตั้งแต่ {ECE_SHIFT_RATIO:g} เท่าและ {ECE_SHIFT_GAP} ขึ้นไป")
    return (f"หัวที่เทรนด้วยภาพ FFHQ ของชุด train อย่างเดียว ({h['train_rows']} แถว) เลือกค่าและหา T ด้วยภาพ FFHQ ของชุด val ({cross['ffhq_val']['rows']} แถว) "
            f"ด้วยกติกาเดียวกับหัวหลัก ได้ `{c['feature']}` λ {c['lam']:g} T = {t['temperature']:.4f} แล้วตรวจบนสำเนา fixed ที่ผ่านเกณฑ์คุณภาพของใต้ตา SCUT "
            f"ที่เกรดแล้วทุกชุด {s['rows']} แถว ({s['sides']} ข้าง ไม่นับ flagged {cross['scut_flagged_rows']} แถว) หัวนี้ใช้ตรวจเท่านั้น ไม่อยู่ใน bundle "
            "(model.md หัวข้อ 4.2 และ 5.3) \"สูงกว่ามาก\" ในรายงานนี้คือ ECE หลังปรับของ SCUT ตั้งแต่ "
            f"{ECE_SHIFT_RATIO:g} เท่าของ FFHQ val และห่างกันตั้งแต่ {ECE_SHIFT_GAP}\n\n"
            + md_table(["ชุด", "จุดเปลี่ยน", "ECE ก่อนปรับ", "ECE หลังปรับ"], rows)
            + "\n\nเกรดเดี่ยวของโมเดลบน SCUT เทียบกับผู้ให้เกรด\n\n" + md_table(["เทียบกับ", "n", "accuracy", "κ เกรด 0/1/≥2"], vs)
            + "\n\n" + "\n".join(lines) + (f"\n\n![reliability cross](figures/{figure})" if figure else ""))


def flag_text(f: str) -> str:
    if f.startswith("T ") and "outside" in f:
        return f"T อยู่นอกช่วง {T_USUAL[0]}–{T_USUAL[1]} ({f.split()[1]}) สัญญาณว่าการปรับเทียบเลื่อนเมื่อข้อมูลเปลี่ยน"
    if "bound" in f:
        return f"T ชนขอบของช่วงที่ค้น {coral.T_BOUNDS[0]}–{coral.T_BOUNDS[1]}"
    if "much higher" in f:
        k, rest = f.split(": ", 1)
        return f"P(เกรด ≥ {k[1]}): ECE ของ SCUT สูงกว่าของ FFHQ มาก ({rest.replace('ECE on SCUT ', '').replace(' is much higher than on FFHQ val ', ' เทียบกับ ')}) T จาก FFHQ อาจใช้กับคนเอเชียได้ไม่ดี"
    return f


def bundle_section(fid: dict | None) -> str:
    if fid is None:
        return "ยังไม่ได้รัน `bundle`"
    return (f"`api/bundle/` รุ่น {fid['bundle_version']} มีหัว `undereye` ที่เทรนแล้ว (`stub` เป็น false) ส่วนหัว `jaw`, `nose_base` และ baseline คัดลอกจากรุ่น "
            f"{fid['ratio_heads_from']} ทุกหัวจึงไม่เป็น stub\n\n"
            f"ความตรงกับตอนเตรียมข้อมูล (api.md หัวข้อ 7): สร้างสำเนา IOD ต่ำสุดของ {fid['sides']} ข้างจาก {fid['images']} ภาพในชุด test ขึ้นใหม่จาก seed "
            "ส่งเข้า `analyze_for_model` (IOD สูงสุดจาก bundle) ครอปใต้ตาของข้างนั้น DINOv2 และหัวใน bundle ที่โหลดกลับมา แล้วเทียบ P(เกรด ≥ 1) และ P(เกรด ≥ 2) "
            f"กับค่าจาก `evaluate` ผลต่างมากสุด {sci(fid['max_diff'])} เกณฑ์ไม่เกิน {fid['tolerance']}: **{'ผ่าน' if fid['passed'] else 'ไม่ผ่าน'}** "
            f"รหัสคุณภาพตรงกันทุกข้าง: {'ใช่' if fid['same_reasons'] else 'ไม่'} การตัดสินตรงกันทุกข้าง: {'ใช่' if fid['same_decisions'] else 'ไม่'}")


def findings_section(lab, cal, ev, cross) -> str:
    out = []
    if lab is not None and lab["agreement"] is None:
        out.append("- label มาจากผู้ให้เกรดคนเดียวที่ไม่ใช่แพทย์ วัดความสอดคล้องระหว่างผู้ให้เกรดไม่ได้ (model.md หัวข้อ 7.2)")
    if cal is not None:
        note = t_note(cal["temperature"]["temperature"], cal["temperature"]["at_bound"])
        if note:
            out.append(f"- หัวหลัก: {note}")
    if ev is not None:
        r = ev["results"]["all"]
        out.append(f"- ผลหลักบนชุด test มี {r['rows']} แถว ({r['sides']} ข้าง) และ label ≥2 {fmt(r['label_ge2'], 0)} แถว ตัวเลขของจุดเปลี่ยนที่ 2 "
                   "จึงไม่แน่นอนมาก ผลประกอบบนทุกสำเนา fixed ช่วยดูทิศทาง แต่รวมภาพที่ api ไม่ให้เกรด")
        res = ev["results"]
        for code, v in r["versus"].items():
            if v["kappa"] is not None and v["kappa"] < ul.PAPER_KAPPA["non_caucasian"]:
                out.append(f"- κ ของโมเดลเทียบกับ {code} ({v['kappa']:.3f}) ต่ำกว่า {ul.PAPER_KAPPA['non_caucasian']} ของแพทย์ใน \\[7\\] "
                           "(เทียบคนละแบบ) รายงานตามจริงโดยไม่แก้ label")
            ffhq = res.get("ffhq", {}).get("versus", {}).get(code, {}).get("kappa")
            every = res.get("all_copies_ffhq", {}).get("versus", {}).get(code, {}).get("kappa")
            scut_rows = res.get("scut", {}).get("rows", 0)
            if v["kappa"] is not None and ffhq is not None and ffhq < v["kappa"] and scut_rows:
                out.append(f"- κ รวมเทียบกับ {code} ({v['kappa']:.3f}) สูงกว่า κ ของ FFHQ อย่างเดียว ({ffhq:.3f}) เพราะแถว SCUT {scut_rows} แถว "
                           "เกือบทั้งหมดเป็นเกรด 0 ที่ทายถูกง่าย κ ของ FFHQ บนทุกสำเนา fixed (ผลประกอบ) คือ "
                           f"{fmt(every, 3)} ค่าที่ใกล้กับการใช้งานกับใบหน้าที่มีร่องใต้ตาจึงต่ำกว่าตัวเลขรวม")
        empty = [subset_name(k, ev["copy_iod"]) for k, v in res.items() if k.startswith("ffhq ") and v["rows"] == 0]
        if empty:
            out.append(f"- กลุ่ม {', '.join(empty)} ไม่มีแถวในผลหลักของชุด test เพราะทุกสำเนาไม่ผ่านเกณฑ์คุณภาพหรือ flagged "
                       "จึงไม่มีผลของกลุ่มอายุนี้บนภาพที่ api จะให้เกรด ดูได้เฉพาะในผลประกอบ")
    if cross is not None and "flags" in cross:
        for f in cross["flags"]:
            out.append(f"- ตรวจข้ามชุดข้อมูล: {flag_text(f)}")
        if cross.get("scut_ge2_sides") == 0:
            out.append("- ตรวจจุดเปลี่ยนที่ 2 ข้ามชุดข้อมูลไม่ได้ เพราะใต้ตาของภาพ SCUT ไม่มีข้างที่ได้เกรดตั้งแต่ 2")
    out.append("- ทุกผลวัดบนภาพจากอินเทอร์เน็ตที่ย่อเลียนแบบเว็บแคม ยังไม่ได้ยืนยันกับกล้องจริง web จึงแสดงผลเป็นระดับ \"ชัด\" หรือ \"ก้ำกึ่ง\" ไม่แสดงตัวเลข (model.md หัวข้อ 4.2)")
    out.append("- ระบบประเมินเฉพาะร่องใต้ตา ข้างที่มีถุงใต้ตาหรือรอยคล้ำเด่นไม่ใช้เทรน ผลของกลุ่ม flagged ในตารางแยกกลุ่มจึงเป็นกรณีที่หัวไม่ได้เรียน")
    return "\n".join(out)


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
    return ("ครั้งล่าสุดของแต่ละคำสั่งจาก `run_log.txt` ใน outputs/undereye_head/ ชุด test ใช้ประเมินหัวที่ลง bundle เฉพาะใน `evaluate` "
            "`cross` อ่านภาพ SCUT ของชุด test ด้วยโดยตั้งใจ (ทุกภาพ SCUT จากทุกชุด) แต่ใช้กับหัวที่เทรนด้วย FFHQ อย่างเดียวซึ่งไม่ลง bundle "
            "และรันหลัง `evaluate` ส่วน `bundle` สร้างภาพ test 20 ข้างขึ้นใหม่เพื่อตรวจความตรงเท่านั้น\n\n" + md_table(["คำสั่ง", "เวลา (UTC)", "สิ่งที่อ่าน"], rows))


def write_report() -> Path:
    folder = out_dir()
    lab = read_json(folder / LABELS_JSON)
    sel = read_json(folder / SELECT_NAME)
    cal = read_json(folder / CALIBRATE_NAME)
    ev = read_json(folder / EVALUATE_NAME)
    cross = read_json(folder / CROSS_NAME)
    fid = read_json(folder / FIDELITY_NAME)
    meta = read_json(paths.path("features") / ef.META_NAME) or {}
    reports = paths.ensure_dir("reports")
    figures = paths.ensure_dir("reports", "figures")
    names = {"grades": None, "test": None, "cross": None}
    if lab is not None:
        names["grades"] = "undereye_grades.png"
        grades_figure(lab["distribution"], figures / names["grades"])
    if ev is not None:
        names["test"] = "undereye_reliability_test.png"
        reliability_figure(ev["results"]["all"]["calibration"], "Test (passes quality, not flagged): reliability before and after T",
                           figures / names["test"])
    if cross is not None and "scut" in cross:
        names["cross"] = "undereye_reliability_cross.png"
        reliability_figure(cross["scut"]["calibration"], "Cross-dataset: FFHQ-only head on SCUT, before and after the FFHQ T",
                           figures / names["cross"])
    bb = meta.get("backbone", {})
    deg = meta.get("degrade", {})
    iod = meta.get("train_face_iod_px", {})
    header = [
        "# หัวใต้ตา",
        "",
        ("สร้างด้วย `uv run python model/scripts/train_undereye_head.py` (`labels`, `select`, `calibrate`, `evaluate`, `cross`, `bundle`) "
         "รายงานนี้มีเฉพาะตัวเลขและกราฟ ไม่มีครอปใต้ตา ภาพจากชุดข้อมูล หรือภาพอ้างอิง GIHS และใช้รหัส R1, R2, … แทนชื่อผู้ให้เกรด"),
        "",
        (f"- หัว ordinal ในรูปเดียวกับ CORAL \\[28\\] บน feature ของ `{bb.get('name', '?')}` แบบแช่แข็งจากครอปใต้ตาทีละข้าง: คะแนน s = x·w ตัวเดียว "
         "และ P(เกรด ≥ k) = sigmoid(s + b_k) สำหรับ k = 1 และ 2 (model.md หัวข้อ 4.2 และ 6)"),
        (f"- loss: {LOSS_TEXT_TH} เทรนด้วย PyTorch แบบ float64 บน CPU ใช้ L-BFGS ทั้งชุด เริ่มจากศูนย์ standardize ด้วยค่าของชุด train แล้วรวมเข้าไปใน w และ b "
         "ก่อนบันทึก api จึงใช้ sigmoid((x·w + b_k) / T) ได้ตรง ๆ"),
        f"- temperature T ตัวเดียวของทั้งสองจุดเปลี่ยน หาบนชุด val ในช่วง {coral.T_BOUNDS[0]}–{coral.T_BOUNDS[1]} (\\[10\\])",
        (f"- ครอปรุ่น {meta.get('crop_version', '?')} การย่อภาพรุ่น {deg.get('version', '?')} ช่วง IOD ที่เทรน {iod.get('min', '?')}–{iod.get('max', '?')} px "
         "ชุดข้อมูลแบ่งตามภาพ 70/15/15 จาก `grading_set.py` ใต้ตาสองข้างของภาพเดียวกันอยู่ชุดเดียวกัน"),
    ]
    sections = [
        ("ผู้ให้เกรดและ label", labels_section(lab)),
        ("การกระจายของเกรด", distribution_section(lab, names["grades"])),
        ("ความสอดคล้องระหว่างผู้ให้เกรด", agreement_section(lab)),
        ("จำนวนแถว", counts_section(sel, ev, cal, cross)),
        ("การเลือกชนิด feature และ λ", select_section(sel)),
        ("การปรับเทียบบนชุด val", calibrate_section(cal)),
        ("ผลบนชุด test", test_section(ev, names["test"])),
        ("ตรวจข้ามชุดข้อมูล", cross_section(cross, names["cross"], ev)),
        ("bundle และความตรงกับตอนเตรียมข้อมูล", bundle_section(fid)),
        ("ข้อสังเกตและข้อจำกัด", findings_section(lab, cal, ev, cross)),
        ("ชุดข้อมูลที่แต่ละคำสั่งอ่าน", log_section(folder)),
    ]
    text = "\n".join(header) + "\n\n" + "\n\n".join(f"## {title}\n\n{body}" for title, body in sections) + "\n"
    path = reports / REPORT_NAME
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Under-eye head: labels, select, calibrate, evaluate, cross, bundle")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("labels", help="grader CSVs -> labels/undereye_labels.csv, distribution and agreement")
    p = sub.add_parser("select", help="choose the feature type and lam on val")
    p.add_argument("--restart", action="store_true", help="drop the finished feature types and search again")
    sub.add_parser("calibrate", help="train the chosen head, find T and try the thresholds on val")
    sub.add_parser("evaluate", help="the test split once with the head and T of calibrate")
    p = sub.add_parser("cross", help="FFHQ-only head checked on every graded SCUT image")
    p.add_argument("--restart", action="store_true", help="drop the finished feature types of the FFHQ-only search")
    sub.add_parser("bundle", help=f"write api/bundle/ {BUNDLE_VERSION} and check it against evaluate")
    sub.add_parser("report", help=f"rewrite model/reports/{REPORT_NAME}")
    args = ap.parse_args(argv)
    status = 0
    if args.command == "labels":
        cmd_labels()
    elif args.command == "select":
        cmd_select(args.restart)
    elif args.command == "calibrate":
        cmd_calibrate()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "cross":
        cmd_cross(args.restart)
    elif args.command == "bundle":
        status = cmd_bundle()
    write_report()
    return status


if __name__ == "__main__":
    raise SystemExit(main())

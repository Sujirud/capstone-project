"""Under-eye grades of the raters as labels (model.md sections 5.2, 6 and 7.2).

The grader CSVs of tools/undereye_grader.html have one row per under-eye (person_side
right or left): grade 0-3, empty when unassessable is 1, and the eye_bags and
dark_circles marks. They are read as the tool's summary tab reads them:

- UTF-8 with a BOM and CRLF line ends; a row the tool would reject is skipped.
- Rater names compare after trimming, collapsing runs of white space and lowering the
  case; a missing name falls back to the file name. A leading ' that the tool puts
  before =, +, - or @ is taken off.
- When a rater has a side in several rows, the row with the latest graded_at wins
  (text order of the ISO time; on a tie the later file in name order).
- A file is used only when each rater in it covers both sides of every graded image of
  grading/manifest.csv; other files (the practice set, the pilot 200) are skipped with a
  note. Rows of images outside the graded set are dropped. The rater name "consensus"
  holds grades agreed for sides where the raters differed: it is not a rater and its
  file need not be complete.

Per side (sides_table):

- excluded "unassessable" when at least half the raters marked the side unassessable.
- p_ge1 and p_ge2: share of the raters who graded the side that gave at least 1 and at
  least 2 (p_ge1 and p_ge2 of the tool's merged table).
- flagged when at least half the raters marked eye bags or dark circles; such sides do
  not train the head (model.md section 5.2).
- target_ge1 and target_ge2: the training targets, the consensus grade when there is one
  (as 0 or 1 at each transition), otherwise p_ge1 and p_ge2. A consensus row marked
  unassessable gives no consensus grade.

Agreement uses each rater's own grades only, on the same definitions as the tool: pairs
of raters on sides both graded (linear weighted kappa of grades 0-3 and 0/1/>=2, exact
and within-one agreement) and ICC(2,1) on sides every rater graded.
"""

import csv
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CSV_COLUMNS = (
    "rater", "set_id", "image", "person_side", "image_side", "grade", "unassessable", "eye_bags", "dark_circles",
    "note", "seconds", "seq", "graded_at",
)
REQUIRED_COLUMNS = ("image", "person_side", "grade", "unassessable")  # as the tool's csvObjects
SIDES = ("right", "left")  # person's side; the tool lists right first
IMAGE_SIDE = {"right": "left", "left": "right"}  # in an image that is not mirrored
CONSENSUS = "consensus"
UNASSESSABLE = "unassessable"  # reason a side is excluded
GRADES4 = (0, 1, 2, 3)
GRADES3 = (0, 1, 2)  # 0, 1 and >= 2
KAPPA_TARGET = 0.61  # substantial (Landis and Koch [17]), model.md section 5.2
PAPER_KAPPA = {"non_caucasian": 0.55, "caucasian": 0.65}  # median kappa of the infraorbital scale in [7]
LOW_GE2_SHARE = 0.10  # below this the transition at 2 is estimated unsteadily
LABEL_COLUMNS_FIRST = ("image", "person_side", "image_side", "source", "sex", "age_group", "split", "pilot",
                       "n_raters", "n_unassessable")
LABEL_COLUMNS_LAST = ("n_marked", "p_ge1", "p_ge2", "flagged", "consensus_grade", "target_ge1", "target_ge2", "excluded")


class LabelError(RuntimeError):
    """No usable grader file, or files that cannot make labels."""


# ---------------------------------------------------------------- names


def norm_name(s) -> str:
    """normName of the tool: trimmed, runs of white space as one space."""
    return re.sub(r"\s+", " ", str(s or "").strip())


def name_key(s) -> str:
    """nameKey of the tool: norm_name in lower case."""
    return norm_name(s).lower()


def unsafe(s: str) -> str:
    """Take off the ' the tool puts before a leading =, +, - or @ so a spreadsheet does not run it."""
    return s[1:] if re.match(r"^'[=+\-@]", s) else s


def rater_codes(keys: Iterable[str]) -> dict[str, str]:
    """R1, R2, ... in order of the name key, so reports never show real names."""
    return {k: f"R{i}" for i, k in enumerate(sorted(set(keys)), start=1)}


# ---------------------------------------------------------------- reading


def _grade(o: dict) -> tuple[int | None, bool] | None:
    """(grade, unassessable) of a row as the tool's readRow takes it, or None for a row it rejects."""
    side = o.get("person_side", "")
    if not o.get("image") or side not in SIDES:
        return None
    na = o.get("unassessable", "") == "1"
    text = o.get("grade", "")
    g = None
    if not na and text != "":
        try:
            value = float(text)
        except ValueError:
            return None
        if not (value.is_integer() and 0 <= value <= 3):
            return None
        g = int(value)
    if not na and g is None:
        return None
    return g, na


def read_grader_csv(path: Path) -> tuple[pd.DataFrame, int]:
    """Rows of one grader CSV and the number of rows skipped; raises LabelError if it is not one."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            head = [h.strip() for h in next(reader)]
        except StopIteration:
            raise LabelError(f"{path.name} is empty") from None
        if not all(h in head for h in REQUIRED_COLUMNS):
            raise LabelError(f"{path.name} is not a CSV of the grading tool (needs {', '.join(REQUIRED_COLUMNS)})")
        rows, skipped = [], 0
        for k, cells in enumerate(reader):
            if not any(c.strip() for c in cells):
                continue
            o = {h: (cells[i].strip() if i < len(cells) else "") for i, h in enumerate(head)}
            parsed = _grade(o)
            if parsed is None:
                skipped += 1
                continue
            g, na = parsed
            name = norm_name(unsafe(o.get("rater", ""))) or path.stem
            try:
                seconds = float(o.get("seconds", "") or 0.0)
            except ValueError:
                seconds = 0.0
            rows.append({
                "file": path.name, "line": k, "rater": name, "rater_key": name_key(name), "set_id": o.get("set_id", ""),
                "image": o["image"], "side": o["person_side"], "grade": g, "na": na,
                "eye_bags": o.get("eye_bags", "") == "1", "dark_circles": o.get("dark_circles", "") == "1",
                "seconds": seconds if math.isfinite(seconds) else 0.0, "graded_at": o.get("graded_at", ""),
            })
    columns = ["file", "line", "rater", "rater_key", "set_id", "image", "side", "grade", "na", "eye_bags", "dark_circles",
               "seconds", "graded_at"]
    df = pd.DataFrame(rows, columns=columns)
    df["grade"] = df["grade"].astype("Int64")
    return df, skipped


@dataclass
class FileCheck:
    file: str
    rows: int
    skipped_rows: int
    raters: list[str]  # rater keys, consensus included
    set_ids: list[str]
    used: bool
    reason: str = ""
    outside_rows: int = 0  # rows of images outside the graded set, dropped


@dataclass
class Grades:
    """Rows of the usable files, latest per rater and side, with what was skipped."""

    rows: pd.DataFrame  # one row per rater key and side, consensus included
    files: list[FileCheck]
    names: dict[str, str]  # rater key -> name as first seen (outputs only, never in reports)
    notes: list[str] = field(default_factory=list)

    @property
    def raters(self) -> list[str]:
        """Rater keys without the consensus, sorted."""
        return sorted(k for k in self.rows["rater_key"].unique() if k != CONSENSUS)

    @property
    def codes(self) -> dict[str, str]:
        return rater_codes(self.raters)

    @property
    def has_consensus(self) -> bool:
        return bool((self.rows["rater_key"] == CONSENSUS).any())


def _missing_sides(rows: pd.DataFrame, graded: set[str]) -> dict[str, int]:
    """rater key -> sides of graded images that the rows do not cover."""
    out = {}
    for key, part in rows.groupby("rater_key", sort=True):
        have = set(zip(part["image"], part["side"], strict=True))
        out[key] = sum((image, side) not in have for image in graded for side in SIDES)
    return out


def check_file(df: pd.DataFrame, name: str, skipped: int, graded: set[str]) -> tuple[FileCheck, pd.DataFrame]:
    """Whether a file is used, and its rows of graded images that count."""
    inside = df[df["image"].isin(graded)]
    outside = len(df) - len(inside)
    raters = sorted(df["rater_key"].unique())
    check = FileCheck(name, len(df), skipped, raters, sorted(df["set_id"].unique()), True, "", outside)
    rater_rows = inside[inside["rater_key"] != CONSENSUS]
    consensus_rows = inside[inside["rater_key"] == CONSENSUS]
    if df.empty:
        check.used, check.reason = False, "no rows"
        return check, inside.iloc[0:0]
    missing = _missing_sides(rater_rows, graded)
    gaps = {k: v for k, v in missing.items() if v}
    if gaps or (rater_rows.empty and consensus_rows.empty):
        check.used = not consensus_rows.empty
        if rater_rows.empty and consensus_rows.empty:
            check.reason = "no rows of graded images (another set)"
        else:
            worst = max(gaps.values())
            check.reason = (f"does not cover every graded image: {worst} of {2 * len(graded)} sides missing "
                            "(practice set, pilot or another set)")
            if check.used:
                check.reason += "; only its consensus rows are used"
        return check, consensus_rows
    return check, inside


def latest(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per rater key, image and side: the latest graded_at, then the later file and line."""
    order = rows.sort_values(["graded_at", "file", "line"], kind="stable")
    return order.drop_duplicates(["rater_key", "image", "side"], keep="last").sort_values(
        ["rater_key", "image", "side"], kind="stable").reset_index(drop=True)


def collect(paths: Sequence[Path], graded: Iterable[str]) -> Grades:
    """Read every grader CSV, keep the usable files and the latest row of each rater and side."""
    graded = set(graded)
    if not graded:
        raise LabelError("the manifest has no graded images")
    checks, parts, notes = [], [], []
    for path in sorted(Path(p) for p in paths):
        try:
            df, skipped = read_grader_csv(path)
        except LabelError as e:
            checks.append(FileCheck(path.name, 0, 0, [], [], False, str(e)))
            continue
        check, used = check_file(df, path.name, skipped, graded)
        checks.append(check)
        if check.used:
            parts.append(used)
    for c in checks:
        if not c.used:
            notes.append(f"skipped {c.file}: {c.reason}")
        elif c.reason:
            notes.append(f"{c.file}: {c.reason}")
        if c.skipped_rows:
            notes.append(f"{c.file}: {c.skipped_rows} rows the grading tool would reject were skipped")
        if c.outside_rows:
            notes.append(f"{c.file}: {c.outside_rows} rows of images outside the graded set were dropped")
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if rows.empty or not (rows["rater_key"] != CONSENSUS).any():
        found = ", ".join(c.file for c in checks) or "none"
        raise LabelError(f"no grader file covers every graded image (files: {found}); export the full_600 set "
                         "from the grading tool into grading/results/")
    names = {}
    for key, name in zip(rows["rater_key"], rows["rater"], strict=True):
        names.setdefault(key, name)
    out = Grades(latest(rows), checks, names, notes)
    if out.has_consensus and len(out.raters) < 2:
        out.notes.append("a consensus file came with a single rater; its grades still replace the rater's")
    return out


# ---------------------------------------------------------------- per side


def by_rater(rows: pd.DataFrame) -> dict[str, dict[tuple[str, str], dict]]:
    """rater key -> {(image, side): row as a dict}."""
    out: dict[str, dict] = {}
    for r in rows.to_dict(orient="records"):
        out.setdefault(r["rater_key"], {})[(r["image"], r["side"])] = r
    return out


def side_label(entries: Sequence[dict | None], consensus: dict | None) -> dict:
    """Counts, soft labels, flag, consensus and target of one side from the raters' rows (None: no row)."""
    have = [e for e in entries if e is not None]
    n = len(have)
    na = sum(bool(e["na"]) for e in have)
    grades = [int(e["grade"]) for e in have if not e["na"]]
    marked = sum(bool(e["eye_bags"] or e["dark_circles"]) for e in have)
    p1 = sum(g >= 1 for g in grades) / len(grades) if grades else math.nan
    p2 = sum(g >= 2 for g in grades) / len(grades) if grades else math.nan
    excluded = ""
    if n == 0:
        excluded = "no_grade"
    elif 2 * na >= n:
        excluded = UNASSESSABLE
    cg = None if consensus is None or consensus["na"] else int(consensus["grade"])
    t1, t2 = (math.nan, math.nan) if excluded else ((float(cg >= 1), float(cg >= 2)) if cg is not None else (p1, p2))
    return {
        "n_raters": n, "n_unassessable": na, "n_marked": marked, "p_ge1": p1, "p_ge2": p2,
        "flagged": int(n > 0 and 2 * marked >= n), "consensus_grade": cg, "target_ge1": t1, "target_ge2": t2,
        "excluded": excluded,
    }


def sides_table(grades: Grades, manifest: pd.DataFrame) -> pd.DataFrame:
    """One row per side of every graded image: manifest columns, each rater's grade and the label."""
    graded = manifest[manifest["role"] == "graded"].sort_values("image", kind="stable")
    raters = grades.raters
    codes = grades.codes
    table = by_rater(grades.rows)
    cons = table.get(CONSENSUS, {})
    out = []
    for m in graded.to_dict(orient="records"):
        for side in SIDES:
            key = (m["image"], side)
            entries = [table.get(r, {}).get(key) for r in raters]
            row = {
                "image": m["image"], "person_side": side, "image_side": IMAGE_SIDE[side], "source": m["source"],
                "sex": m["sex"], "age_group": m["age_group"], "split": m["split"], "pilot": int(m["pilot"] == "1"),
            }
            label = side_label(entries, cons.get(key))
            row.update({k: label[k] for k in ("n_raters", "n_unassessable")})
            for r, e in zip(raters, entries, strict=True):
                row[f"grade_{codes[r]}"] = None if e is None or e["na"] else int(e["grade"])
            row.update({k: label[k] for k in LABEL_COLUMNS_LAST})
            out.append(row)
    columns = [*LABEL_COLUMNS_FIRST, *[f"grade_{codes[r]}" for r in raters], *LABEL_COLUMNS_LAST]
    df = pd.DataFrame(out, columns=columns)
    for c in [f"grade_{codes[r]}" for r in raters] + ["consensus_grade"]:
        df[c] = df[c].astype("Int64")
    return df


def write_labels(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6g")
    tmp.replace(path)


def read_labels(path: Path) -> pd.DataFrame:
    text = ("image", "person_side", "image_side", "source", "sex", "age_group", "split", "excluded")
    df = pd.read_csv(path, dtype={c: str for c in text}, keep_default_na=False, na_values=[""], encoding="utf-8")
    for c in text:
        df[c] = df[c].fillna("")
    for c in [c for c in df.columns if c.startswith("grade_")] + ["consensus_grade"]:
        df[c] = df[c].astype("Int64")
    for c in ("pilot", "flagged", "n_raters", "n_unassessable", "n_marked"):
        df[c] = df[c].astype(int)
    return df


def rater_columns(df: pd.DataFrame) -> list[str]:
    """grade_R1, grade_R2, ... of a sides table."""
    return [c for c in df.columns if c.startswith("grade_")]


# ---------------------------------------------------------------- agreement


def linear_kappa(a: Sequence[int], b: Sequence[int], cats: Sequence[int]) -> float:
    """weightedKappa of the tool: linear (Cicchetti-Allison) weights from the grade values; nan when undefined."""
    n = len(a)
    if n == 0:
        return math.nan
    ix = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b, strict=True):
        o[ix[x], ix[y]] += 1
    w = np.abs(np.subtract.outer(np.asarray(cats, dtype=np.float64), np.asarray(cats, dtype=np.float64)))
    observed = float((w * o).sum())
    expected = float((w * np.outer(o.sum(axis=1), o.sum(axis=0))).sum() / n)
    return math.nan if expected == 0 else 1.0 - observed / expected


def collapse(g: int) -> int:
    """Grade 0-3 as 0, 1 or 2 (>= 2)."""
    return min(int(g), 2)


def icc21(y) -> float:
    """ICC(2,1) of Shrout and Fleiss: two-way random, absolute agreement, single rater; rows are targets."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2:
        return math.nan
    n, k = y.shape
    grand = y.mean()
    ssr = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ssc = n * ((y.mean(axis=0) - grand) ** 2).sum()
    sst = ((y - grand) ** 2).sum()
    msr, msc = ssr / (n - 1), ssc / (k - 1)
    mse = (sst - ssr - ssc) / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return math.nan if den == 0 else float((msr - mse) / den)


def pair_stats(a: dict, b: dict, keys: Iterable | None = None) -> dict:
    """pairStats of the tool on sides both raters graded (keys limits the sides)."""
    keys = a.keys() if keys is None else keys
    xa, xb = [], []
    for key in keys:
        x, y = a.get(key), b.get(key)
        if x is not None and y is not None and not x["na"] and not y["na"]:
            xa.append(int(x["grade"]))
            xb.append(int(y["grade"]))
    n = len(xa)
    same = sum(p == q for p, q in zip(xa, xb, strict=True))
    near = sum(abs(p - q) <= 1 for p, q in zip(xa, xb, strict=True))
    return {
        "n": n,
        "exact": same / n if n else math.nan,
        "within1": near / n if n else math.nan,
        "kappa4": linear_kappa(xa, xb, GRADES4),
        "kappa3": linear_kappa([collapse(g) for g in xa], [collapse(g) for g in xb], GRADES3),
    }


def agreement(grades: Grades, images: Iterable[str]) -> dict:
    """Pairwise statistics and ICC(2,1) of the raters on the sides of these images; empty with one rater."""
    raters = grades.raters
    if len(raters) < 2:
        return {}
    images = set(images)
    table = by_rater(grades.rows)
    keys = sorted((i, s) for i in images for s in SIDES)
    codes = grades.codes
    pairs = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            s = pair_stats(table[raters[i]], table[raters[j]], keys)
            pairs.append({"a": codes[raters[i]], "b": codes[raters[j]], **s})
    common = []
    for key in keys:
        xs = [table[r].get(key) for r in raters]
        if all(x is not None and not x["na"] for x in xs):
            common.append([int(x["grade"]) for x in xs])
    return {"images": len(images), "pairs": pairs, "icc_sides": len(common), "icc21": icc21(common)}


# ---------------------------------------------------------------- distribution


def grade_counts(values: pd.Series) -> dict:
    """Counts of grades 0-3 and of unassessable (NA) in a rater's grade column."""
    v = values.astype("Int64")
    out = {str(g): int((v == g).sum()) for g in GRADES4}
    out["na"] = int(v.isna().sum())
    return out


def distribution(sides: pd.DataFrame, groups: dict[str, pd.Series]) -> list[dict]:
    """Per group: sides, excluded, flagged, each rater's grade counts, and the share at >= 2 by half the raters."""
    out = []
    for name, mask in groups.items():
        part = sides[mask.to_numpy()]
        kept = part[part["excluded"] == ""]
        clean = kept[kept["flagged"] == 0]
        out.append({
            "group": name,
            "sides": len(part),
            "excluded": int((part["excluded"] != "").sum()),
            "flagged": int((kept["flagged"] == 1).sum()),
            "raters": {c.removeprefix("grade_"): grade_counts(part[c]) for c in rater_columns(part)},
            "ge2_share": float((kept["p_ge2"] >= 0.5).mean()) if len(kept) else None,
            "ge2_share_unflagged": float((clean["p_ge2"] >= 0.5).mean()) if len(clean) else None,
        })
    return out


def median_seconds(grades: Grades) -> dict[str, float | None]:
    """Median seconds per image of each rater, as the tool's summary (images with time > 0)."""
    out = {}
    codes = grades.codes
    for key in grades.raters:
        part = grades.rows[grades.rows["rater_key"] == key]
        per_image = part.groupby("image")["seconds"].max()
        per_image = per_image[per_image > 0]
        out[codes[key]] = float(per_image.median()) if len(per_image) else None
    return out

"""Rows of the under-eye head (model.md sections 5.3 and 6).

The rows come from features/undereye/index.csv (one row per crop: image, person's side,
copy) joined with labels/undereye_labels.csv by image and side. Only crops with
features (a face was found) take part, and sides the labels exclude never do.

- train: the train copies of train images, not flagged.
- val: the fixed copies of val images that pass the quality checks of that side
  (reasons empty), not flagged.
- test: the fixed copies of test images that pass the quality checks; flagged sides
  stay, marked, so evaluate can report them apart.
- cross (model.md sections 4.2 and 5.3): the same train and val rules on FFHQ only, and
  every SCUT fixed copy of every split that passes the quality checks.
- val_all and test_all: every fixed copy whether it passes the quality checks or not,
  for the supporting results of the report only; nothing is chosen on them.

Rows keep the order of the index they come from, so positions index the feature matrix
of the joined rows.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAIN, FIXED = "train", "fixed"
SPLITS = ("train", "val", "test")
LABEL_KEEP = ("image", "person_side", "source", "sex", "age_group", "split", "pilot", "flagged", "excluded",
              "p_ge1", "p_ge2", "target_ge1", "target_ge2", "consensus_grade")
SUBSETS = ("train", "val", "test", "val_all", "test_all", "cross_train", "cross_val", "cross_eval")


@dataclass(frozen=True)
class Subset:
    """Rows of one subset; positions index the feature matrix of the joined rows."""

    name: str
    rows: pd.DataFrame
    positions: np.ndarray

    @property
    def y(self) -> np.ndarray:
        """(n, 2) targets P(grade >= 1) and P(grade >= 2)."""
        return self.rows[["target_ge1", "target_ge2"]].to_numpy(dtype=np.float64)

    @property
    def sides(self) -> int:
        return int(self.rows.groupby(["image", "side"]).ngroups) if len(self.rows) else 0

    @property
    def images(self) -> int:
        return int(self.rows["image"].nunique())


def join_labels(index: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Rows of index.csv that have features, with the labels and rater grades of their side.

    Every row must find its side in the labels with the same split and source. Sides the
    labels exclude are dropped. The result keeps the order of index and gets a fresh index.
    """
    rows = index[index["row"].notna() & (index["n_faces"] > 0)].reset_index(drop=True)
    keep = [*LABEL_KEEP, *[c for c in labels.columns if c.startswith("grade_")]]
    lab = labels[keep].rename(columns={"person_side": "side", "split": "label_split", "source": "label_source"})
    if lab.duplicated(["image", "side"]).any():
        raise ValueError("the labels list a side more than once")
    joined = rows.merge(lab, on=["image", "side"], how="left", validate="many_to_one")
    missing = joined["label_split"].isna()
    if missing.any():
        r = joined[missing].iloc[0]
        raise ValueError(f"{int(missing.sum())} index rows have no label, for example {r['image']} {r['side']}")
    other = (joined["split"] != joined["label_split"]) | (joined["source"] != joined["label_source"])
    if other.any():
        r = joined[other].iloc[0]
        raise ValueError(f"{r['image']} is {r['source']} {r['split']} in the index and {r['label_source']} {r['label_split']} in the labels")
    joined = joined.drop(columns=["label_split", "label_source"])
    return joined[joined["excluded"] == ""].reset_index(drop=True)


def _mask(joined: pd.DataFrame, name: str) -> np.ndarray:
    split, kind = joined["split"], joined["kind"]
    passing = joined["reasons"].fillna("") == ""
    clean = joined["flagged"] == 0
    ffhq, scut = joined["source"] == "ffhq", joined["source"] == "scut"
    masks = {
        "train": (split == "train") & (kind == TRAIN) & clean,
        "val": (split == "val") & (kind == FIXED) & passing & clean,
        "test": (split == "test") & (kind == FIXED) & passing,
        "val_all": (split == "val") & (kind == FIXED) & clean,
        "test_all": (split == "test") & (kind == FIXED),
        "cross_train": (split == "train") & (kind == TRAIN) & clean & ffhq,
        "cross_val": (split == "val") & (kind == FIXED) & passing & clean & ffhq,
        "cross_eval": (kind == FIXED) & passing & scut,
    }
    if name not in masks:
        raise ValueError(f"unknown subset {name!r}; one of {SUBSETS}")
    return masks[name].to_numpy()


def subset(joined: pd.DataFrame, name: str) -> Subset:
    keep = _mask(joined, name)
    part = joined[keep]
    return Subset(name, part.reset_index(drop=True), np.flatnonzero(keep))


def subsets(joined: pd.DataFrame, names=SUBSETS) -> dict[str, Subset]:
    return {n: subset(joined, n) for n in names}


def counts(parts: dict[str, Subset]) -> list[dict]:
    """Rows, sides and images of every subset by source, with the rows at each grade of the targets."""
    out = []
    for name, s in parts.items():
        for source, g in [*s.rows.groupby("source", sort=True), ("all", s.rows)]:
            t = g[["target_ge1", "target_ge2"]].to_numpy(dtype=np.float64)
            out.append({
                "subset": name, "source": source, "rows": len(g),
                "sides": int(g.groupby(["image", "side"]).ngroups) if len(g) else 0,
                "images": int(g["image"].nunique()),
                "flagged_rows": int((g["flagged"] == 1).sum()),
                "ge1_rows": float(t[:, 0].sum()) if len(g) else 0.0,
                "ge2_rows": float(t[:, 1].sum()) if len(g) else 0.0,
            })
    return out

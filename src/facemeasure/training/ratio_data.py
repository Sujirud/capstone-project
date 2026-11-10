"""Rows of the jaw and alar base heads (model.md sections 5.3 and 6).

The rows come from features/face/index.csv (one row per copy of a SCUT-FBP5500 image)
joined with labels/scut_labels.csv by image name. Only copies with features (a face
was found) take part.

- train: every copy, for both targets, so both heads train on the same rows.
- val and test: only copies that pass the quality checks of that region, that is
  reasons_jaw empty for the jaw and reasons_nose_base empty for the alar base, because
  in use an image that fails gets no grade.

Rows keep the order of the index they come from, so a row's position is its row in
the feature matrix of that index.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from facemeasure import codes

TARGETS = (codes.JAW, codes.NOSE_BASE)
SPLITS = ("train", "val", "test")
TRAIN = "train"
LABEL_COLUMN = {codes.JAW: "jaw_ratio", codes.NOSE_BASE: "nose_ratio"}
MP_COLUMN = {codes.JAW: "jaw_ratio_mp", codes.NOSE_BASE: "nose_ratio_mp"}
REASONS_COLUMN = {codes.JAW: "reasons_jaw", codes.NOSE_BASE: "reasons_nose_base"}
MEASURE = {codes.JAW: codes.FACIAL_TAPERNESS, codes.NOSE_BASE: codes.ALAR_INTERCANTHAL_RATIO}
LABEL_KEEP = ("image", "sex", "split", "jaw_ratio", "nose_ratio")


@dataclass(frozen=True)
class Subset:
    """The rows of one target in one split; positions index the feature matrix of the joined rows."""

    target: str
    split: str
    rows: pd.DataFrame
    positions: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return self.rows[LABEL_COLUMN[self.target]].to_numpy(dtype=np.float64)

    @property
    def mp(self) -> np.ndarray:
        return self.rows[MP_COLUMN[self.target]].to_numpy(dtype=np.float64)

    @property
    def images(self) -> int:
        return int(self.rows["image"].nunique())


def join_labels(index: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Rows of index.csv that have features, with sex, labels and split from scut_labels.csv.

    Every row must find its image in the labels with the same split. The result keeps
    the order of index (rows without features dropped) and gets a fresh 0..n-1 index.
    """
    rows = index[index["row"].notna() & (index["n_faces"] > 0)].reset_index(drop=True)
    lab = labels[list(LABEL_KEEP)].rename(columns={"split": "label_split"})
    if lab["image"].duplicated().any():
        raise ValueError("the labels list an image more than once")
    joined = rows.merge(lab, on="image", how="left", validate="many_to_one")
    missing = joined["label_split"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} index rows have no label, for example {joined.loc[missing, 'image'].iloc[0]}")
    other = joined["split"] != joined["label_split"]
    if other.any():
        r = joined[other].iloc[0]
        raise ValueError(f"{r['image']} is in {r['split']} in the index and in {r['label_split']} in the labels")
    return joined.drop(columns="label_split")


def subset(joined: pd.DataFrame, target: str, split: str) -> Subset:
    """The rows of one target in one split: every copy for train, passing copies for val and test."""
    if target not in TARGETS or split not in SPLITS:
        raise ValueError(f"unknown target {target!r} or split {split!r}")
    keep = (joined["split"] == split) & joined[LABEL_COLUMN[target]].notna()
    if split != TRAIN:
        keep &= joined[REASONS_COLUMN[target]].fillna("") == ""
    part = joined[keep]
    return Subset(target, split, part.reset_index(drop=True), part.index.to_numpy())


def subsets(joined: pd.DataFrame, splits=SPLITS) -> dict[tuple[str, str], Subset]:
    return {(t, s): subset(joined, t, s) for t in TARGETS for s in splits}


def counts(parts: dict[tuple[str, str], Subset]) -> pd.DataFrame:
    """Rows and images of every target, split and group, with a total per target and split."""
    out = []
    for (target, split), s in parts.items():
        for group, g in s.rows.groupby("group", sort=True):
            out.append({"target": target, "split": split, "group": group, "rows": len(g), "images": int(g["image"].nunique())})
        out.append({"target": target, "split": split, "group": "all", "rows": len(s.rows), "images": s.images})
    return pd.DataFrame(out, columns=["target", "split", "group", "rows", "images"])

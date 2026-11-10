"""Rows of the under-eye head from the feature index and the labels."""

import numpy as np
import pandas as pd
import pytest

from facemeasure.training import undereye_data as ud


def index_and_labels():
    """Two images per split and source, both sides; train images have train copies, SCUT train also fixed."""
    index, labels = [], []
    row = 0
    for split in ("train", "val", "test"):
        for source in ("ffhq", "scut"):
            for k in range(2):
                image = f"{source}_{split}_{k}.png"
                copies = [("train", c) for c in (1, 2)] if split == "train" else [("fixed", 1), ("fixed", 2)]
                if split == "train" and source == "scut":
                    copies += [("fixed", 1), ("fixed", 2)]
                for kind, c in copies:
                    for side in ("left", "right"):
                        reasons = "smiling" if (k == 1 and c == 2) else ""
                        index.append({"image": image, "source": source, "side": side, "split": split, "kind": kind, "copy": c,
                                      "n_faces": 1, "reasons": reasons, "row": row, "shard": 0})
                        row += 1
                for side in ("left", "right"):
                    labels.append({"image": image, "person_side": side, "source": source, "sex": "female", "age_group": "",
                                   "split": split, "pilot": 0, "flagged": int(k == 0 and side == "left"),
                                   "excluded": "unassessable" if (k == 1 and side == "right") else "",
                                   "p_ge1": 1.0, "p_ge2": 0.0, "target_ge1": 1.0, "target_ge2": 0.0, "consensus_grade": pd.NA,
                                   "grade_R1": 1})
    idx = pd.DataFrame(index)
    idx.loc[0, ["row", "n_faces"]] = [np.nan, 0]  # a copy without a face
    return idx, pd.DataFrame(labels)


def test_rules_of_each_subset():
    idx, labels = index_and_labels()
    joined = ud.join_labels(idx, labels)
    assert (joined["excluded"] == "").all() and joined["row"].notna().all()
    parts = ud.subsets(joined)
    tr = parts["train"].rows
    assert set(tr["kind"]) == {"train"} and set(tr["split"]) == {"train"} and (tr["flagged"] == 0).all()
    for name in ("val", "test"):
        r = parts[name].rows
        assert set(r["kind"]) == {"fixed"} and set(r["split"]) == {name} and (r["reasons"] == "").all()
    assert (parts["val"].rows["flagged"] == 0).all() and (parts["test"].rows["flagged"] == 1).any()
    assert len(parts["val_all"].rows) > len(parts["val"].rows) and (parts["test_all"].rows["reasons"] != "").any()
    assert set(parts["cross_train"].rows["source"]) == {"ffhq"} and set(parts["cross_val"].rows["source"]) == {"ffhq"}
    ce = parts["cross_eval"].rows
    assert set(ce["source"]) == {"scut"} and set(ce["split"]) == {"train", "val", "test"} and set(ce["kind"]) == {"fixed"}
    # positions point at the joined rows
    for s in parts.values():
        assert (joined.iloc[s.positions]["image"].to_numpy() == s.rows["image"].to_numpy()).all()
    assert parts["train"].y.shape == (len(tr), 2)


def test_join_stops_on_rows_without_a_label_or_another_split():
    idx, labels = index_and_labels()
    with pytest.raises(ValueError, match="no label"):
        ud.join_labels(idx, labels.iloc[1:])
    bad = labels.copy()
    bad.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="in the index"):
        ud.join_labels(idx, bad)


def test_counts_by_subset_and_source():
    idx, labels = index_and_labels()
    parts = ud.subsets(ud.join_labels(idx, labels), ("val",))
    table = {(c["subset"], c["source"]): c for c in ud.counts(parts)}
    assert table[("val", "all")]["rows"] == table[("val", "ffhq")]["rows"] + table[("val", "scut")]["rows"]
    assert table[("val", "all")]["ge1_rows"] == table[("val", "all")]["rows"]

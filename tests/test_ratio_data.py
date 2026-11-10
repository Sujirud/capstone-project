"""facemeasure.training.ratio_data on a synthetic index and label file."""

import pandas as pd
import pytest

from facemeasure.training import ratio_data


def index_rows():
    rows = []
    for image, group, split, copies in (("AF1.jpg", "AF", "train", 4), ("AM1.jpg", "AM", "train", 4),
                                        ("AF2.jpg", "AF", "val", 2), ("CM1.jpg", "CM", "test", 2), ("CF1.jpg", "CF", "test", 2)):
        for c in range(1, copies + 1):
            rows.append({"image": image, "group": group, "split": split, "kind": "train" if split == "train" else "fixed",
                         "copy": c, "n_faces": 1, "reasons_jaw": "", "reasons_nose_base": "", "jaw_ratio_mp": 0.84,
                         "nose_ratio_mp": 1.0, "shard": 0, "row": len(rows)})
    df = pd.DataFrame(rows)
    df.loc[1, "reasons_jaw"] = "smiling"  # a train copy that fails still trains
    df.loc[(df["image"] == "AF2.jpg") & (df["copy"] == 1), "reasons_jaw"] = "hair_occlusion"  # jaw only
    df.loc[(df["image"] == "CM1.jpg") & (df["copy"] == 2), ["reasons_jaw", "reasons_nose_base"]] = "face_too_small"
    df.loc[(df["image"] == "CF1.jpg") & (df["copy"] == 1), ["n_faces", "shard", "row"]] = [0, None, None]  # no face
    return df


def labels():
    return pd.DataFrame({
        "image": ["AF1.jpg", "AM1.jpg", "AF2.jpg", "CM1.jpg", "CF1.jpg"],
        "sex": ["female", "male", "female", "male", "female"],
        "split": ["train", "train", "val", "test", "test"],
        "jaw_ratio": [0.80, 0.85, 0.82, 0.88, 0.86],
        "nose_ratio": [1.05, 1.10, 1.00, 1.12, 1.08],
        "beauty_score": [3.0] * 5,
    })


def test_train_takes_every_copy_val_and_test_only_passing_ones():
    joined = ratio_data.join_labels(index_rows(), labels())
    assert len(joined) == 13  # the copy without a face is dropped
    parts = ratio_data.subsets(joined)
    for t in ratio_data.TARGETS:
        assert len(parts[(t, "train")].rows) == 8
    assert len(parts[("jaw", "val")].rows) == 1 and len(parts[("nose_base", "val")].rows) == 2
    assert parts[("jaw", "test")].rows["image"].tolist() == ["CM1.jpg", "CF1.jpg"]
    assert parts[("nose_base", "test")].rows["copy"].tolist() == [1, 2]


def test_positions_point_at_the_joined_rows_and_labels_come_along():
    joined = ratio_data.join_labels(index_rows(), labels())
    s = ratio_data.subset(joined, "jaw", "test")
    assert joined.loc[s.positions, "image"].tolist() == s.rows["image"].tolist()
    assert s.y.tolist() == [0.88, 0.86] and s.mp.tolist() == [0.84, 0.84] and s.images == 2
    assert s.rows["sex"].tolist() == ["male", "female"]


def test_counts_by_split_and_group():
    parts = ratio_data.subsets(ratio_data.join_labels(index_rows(), labels()))
    c = ratio_data.counts(parts)
    train = c[(c["target"] == "jaw") & (c["split"] == "train")].set_index("group")
    assert train.loc["AF", "rows"] == 4 and train.loc["all", "rows"] == 8 and train.loc["all", "images"] == 2
    test = c[(c["target"] == "nose_base") & (c["split"] == "test")].set_index("group")
    assert test.loc["CM", "rows"] == 1 and test.loc["CF", "rows"] == 1


def test_a_row_without_a_label_or_with_another_split_is_refused():
    with pytest.raises(ValueError, match="no label"):
        ratio_data.join_labels(index_rows(), labels().iloc[1:])
    other = labels()
    other.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="AF1.jpg is in train"):
        ratio_data.join_labels(index_rows(), other)

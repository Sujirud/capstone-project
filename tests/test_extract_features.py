"""extract_features.py with fake models: copies, shards, stopping and going on."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import extract_features as ef
from facemeasure import quality

CONFIG = quality.load_config()
SETTINGS = {"plan_version": ef.PLAN_VERSION, "test": True}


class Stop(Exception):
    pass


def picture(name: str) -> np.ndarray:
    if "noface" in name:
        return np.zeros((90, 80, 3), dtype=np.uint8)
    rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    return rng.integers(30, 220, (120, 100, 3), dtype=np.uint8)


def mark(image: np.ndarray) -> tuple[int, int]:
    d = hashlib.sha256(image.tobytes()).digest()
    return d[0], d[1]


class Fakes:
    """analyze_frontal and analyze_for_model stand-ins; the crops carry a mark of the copy they come from."""

    def __init__(self, frontal_result, stop_at: int | None = None):
        self.make, self.stop_at, self.embed_calls = frontal_result, stop_at, 0

    def analyze(self, image):
        if image.mean() < 10:
            return self.make(face=False)
        return self.make(values=dict(iod_px=96.0, **{k: 0.0 for k in ("yaw_deg", "pitch_deg", "roll_deg", "smile", "jaw_open",
                                                                         "blink_left", "blink_right", "hair_jaw", "hair_undereye_left",
                                                                         "hair_undereye_right")}))

    def model_input(self, image):
        r = self.analyze(image)
        if r.face_count == 0:
            return r
        a, b = mark(image)
        c = r.crops

        def filled(crop, v):
            return replace(crop, image=np.full(crop.image.shape, (a, b, v), dtype=np.uint8))

        crops = replace(c, face=filled(c.face, 0), undereye_left=filled(c.undereye_left, 1), undereye_right=filled(c.undereye_right, 2))
        return replace(r, crops=crops, quality=replace(r.quality, iod_px=float(image.shape[1])))

    def embed(self, batch):
        self.embed_calls += 1
        if self.stop_at is not None and self.embed_calls == self.stop_at:
            raise Stop
        n = len(batch)
        first = batch[:, 0, 0, :].astype(np.float32)
        cls = np.zeros((n, 768), np.float32)
        cls[:, :3] = first
        grid = np.zeros((n, 12288), np.float32)
        grid[:, -3:] = first
        return {"cls": cls, "cls_mean": np.concatenate([cls, -cls], axis=1), "grid4": grid}

    def workers(self) -> ef.Workers:
        return ef.Workers(analyze=self.analyze, model_input=self.model_input, embed=self.embed, device="fake",
                          read=lambda p: picture(Path(p).name))


def undereye_sources() -> list[ef.Source]:
    rows = [("1.png", "ffhq", "train"), ("2.png", "ffhq", "val"), ("AF1.jpg", "scut", "train"), ("AF2.jpg", "scut", "test"),
            ("noface.png", "ffhq", "train"), ("3.png", "ffhq", "test"), ("AM3.jpg", "scut", "train")]
    return [ef.Source(i, Path(i), split, {"source": s}, ef.undereye_copies(split, s)) for i, s, split in rows]


def face_sources() -> list[ef.Source]:
    rows = [("AF1.jpg", "AF", "train"), ("AF2.jpg", "AF", "val"), ("AM1.jpg", "AM", "test"), ("CF1_noface.jpg", "CF", "train"),
            ("CM1.jpg", "CM", "train")]
    return [ef.Source(i, Path(i), split, {"group": g}, ef.face_copies(split)) for i, g, split in rows]


SOURCES = {ef.FACE: face_sources, ef.UNDEREYE: undereye_sources}
SHARD = {ef.FACE: 4, ef.UNDEREYE: 10}


def test_copies_per_split():
    assert ef.face_copies("train") == tuple(("train", k) for k in (1, 2, 3, 4))
    assert ef.face_copies("val") == ef.face_copies("test") == (("fixed", 1), ("fixed", 2))
    assert ef.undereye_copies("train", "ffhq") == tuple(("train", k) for k in range(1, 7))
    assert ef.undereye_copies("train", "scut") == (*(("train", k) for k in range(1, 7)), ("fixed", 1), ("fixed", 2))
    assert ef.undereye_copies("test", "scut") == ef.undereye_copies("val", "ffhq") == (("fixed", 1), ("fixed", 2))


def test_shards_keep_the_copies_of_an_image_together():
    sources = undereye_sources()
    plan = ef.shard_plan(ef.UNDEREYE, sources, shard_crops=20)
    assert [i for shard in plan for i in shard] == list(range(len(sources)))
    sizes = [ef.count_crops(ef.UNDEREYE, [sources[i] for i in shard]) for shard in plan]
    assert all(s >= 20 for s in sizes[:-1]) and sum(sizes) == ef.count_crops(ef.UNDEREYE, sources)


def expected_keys(task, sources):
    sides = ef.SIDES if task == ef.UNDEREYE else (None,)
    return sorted((s.image, kind, copy, side or "") for s in sources for kind, copy in s.copies for side in sides)


def keys_of(task, index):
    side = index["side"] if task == ef.UNDEREYE else [""] * len(index)
    return sorted(zip(index["image"], index["kind"], index["copy"].astype(int), side, strict=True))


@pytest.mark.parametrize("task", ef.TASKS)
def test_stop_and_go_on_gives_every_crop_once(task, tmp_path, frontal_result):
    sources, size = SOURCES[task](), SHARD[task]
    stopped = Fakes(frontal_result, stop_at=2)
    with pytest.raises(Stop):
        ef.run_task(task, sources, tmp_path / "a", stopped.workers(), SETTINGS, CONFIG, shard_crops=size)
    assert sorted(p.name for p in (tmp_path / "a").glob("shard_*")) == ["shard_0000.npz"]
    assert not (tmp_path / "a" / ef.INDEX_NAME).exists()
    (tmp_path / "a" / "shard_0001.npz.tmp").write_bytes(b"half a shard from a crash")

    first_shard = (tmp_path / "a" / "shard_0000.npz").read_bytes()
    going_on = Fakes(frontal_result)
    counts = ef.run_task(task, sources, tmp_path / "a", going_on.workers(), SETTINGS, CONFIG, shard_crops=size)
    plan = ef.shard_plan(task, sources, size)
    n_shards = len(plan)
    with_faces = [shard for shard in plan[1:] if any("noface" not in sources[i].image for i in shard)]
    assert going_on.embed_calls == len(with_faces) and counts["shards"] == n_shards > 2  # a shard without faces runs no DINOv2
    assert (tmp_path / "a" / "shard_0000.npz").read_bytes() == first_shard
    assert not list((tmp_path / "a").glob("*.tmp"))

    index = ef.read_index(tmp_path / "a")
    assert list(index.columns) == ef.COLUMNS[task]
    assert keys_of(task, index) == expected_keys(task, sources)  # nothing missing, nothing twice
    pointers = index.loc[index["row"].notna(), ["shard", "row"]]
    assert not pointers.duplicated().any()
    no_face = index["image"].str.contains("noface")
    assert (index.loc[no_face, "n_faces"] == 0).all() and index.loc[no_face, "row"].isna().all()
    assert index.loc[~no_face, "row"].notna().all()
    assert counts["crops"] == len(index) == ef.count_crops(task, sources) and counts["without_face"] == int(no_face.sum())

    # One run without stopping gives the same files
    whole = Fakes(frontal_result)
    ef.run_task(task, sources, tmp_path / "b", whole.workers(), SETTINGS, CONFIG, shard_crops=size)
    assert (tmp_path / "a" / ef.INDEX_NAME).read_bytes() == (tmp_path / "b" / ef.INDEX_NAME).read_bytes()
    for k in range(n_shards):
        assert ef.shard_path(tmp_path / "a", k).read_bytes() == ef.shard_path(tmp_path / "b", k).read_bytes()


@pytest.mark.parametrize("task", ef.TASKS)
def test_each_row_points_at_the_features_of_its_own_copy(task, tmp_path, frontal_result):
    sources = SOURCES[task]()
    fakes = Fakes(frontal_result)
    ef.run_task(task, sources, tmp_path, fakes.workers(), SETTINGS, CONFIG, shard_crops=10)
    by_name = {s.image: s for s in sources}
    for feature in ("cls", "cls_mean", "grid4"):
        rows, x = ef.load_features(tmp_path, feature)
        assert len(rows) == len(x) == int(ef.read_index(tmp_path)["row"].notna().sum())
        assert x.dtype == np.float32
        for r, v in zip(rows.itertuples(), x, strict=True):
            _, result = ef.regenerate(task, r.image, by_name[r.image].path, r.kind, int(r.copy), fakes.workers(), CONFIG)
            want = ef.crop_of(task, result, getattr(r, "side", None))[0, 0].astype(np.float32)
            got = v[:3] if feature != "grid4" else v[-3:]
            assert np.array_equal(got, want), (r.image, r.kind, r.copy)


def test_rows_record_the_copy(tmp_path, frontal_result):
    ef.run_task(ef.FACE, face_sources(), tmp_path, Fakes(frontal_result).workers(), SETTINGS, CONFIG, shard_crops=10)
    index = ef.read_index(tmp_path)
    train = index[index["kind"] == "train"]
    assert (train["flipped"] == train["copy"].isin([2, 4]).astype(int)).all()
    assert (index.loc[index["kind"] == "fixed", "flipped"] == 0).all()
    fixed = index[index["kind"] == "fixed"]
    assert set(fixed.loc[fixed["copy"] == 1, "target_iod"]) == {77.0} and set(fixed.loc[fixed["copy"] == 2, "target_iod"]) == {110.0}
    ok = index[index["n_faces"] > 0]
    assert (ok["reasons_jaw"] == "").all() and ok["jaw_ratio_mp"].notna().all()
    assert (index.loc[index["n_faces"] == 0, "reasons_jaw"] == "no_face").all()
    sources = (tmp_path / ef.SOURCES_NAME).read_text(encoding="utf-8").splitlines()
    assert sources[0] == "image,n_faces,iod_px" and len(sources) == 1 + len(face_sources())


def test_other_settings_ask_for_restart(tmp_path, frontal_result):
    sources = face_sources()
    ef.run_task(ef.FACE, sources, tmp_path, Fakes(frontal_result).workers(), SETTINGS, CONFIG, shard_crops=10)
    with pytest.raises(SystemExit, match="--restart"):
        ef.run_task(ef.FACE, sources, tmp_path, Fakes(frontal_result).workers(), {**SETTINGS, "test": False}, CONFIG, shard_crops=10)
    fakes = Fakes(frontal_result)
    ef.run_task(ef.FACE, sources, tmp_path, fakes.workers(), {**SETTINGS, "test": False}, CONFIG, restart=True, shard_crops=10)
    assert fakes.embed_calls == len(ef.shard_plan(ef.FACE, sources, 10))

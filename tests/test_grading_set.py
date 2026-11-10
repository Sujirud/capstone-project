"""grading_set selection, replacement, download loop and packaging on synthetic data."""

import hashlib
import re
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import grading_set as gs

PER_SUBGROUP = 80  # eligible FFHQ images per age band and sex, more than any quota needs


def make_labels() -> pd.DataFrame:
    """Eligible adults in every subgroup plus rows that each fail exactly one criterion."""
    rows = []

    def add(age_group, gender, **override):
        row = {
            "image_number": len(rows), "age_group": age_group, "age_group_confidence": 0.9,
            "gender": gender, "gender_confidence": 1.0, "head_pitch": 2.0, "head_roll": 0.5,
            "head_yaw": -3.0, "left_eye_occluded": 0.0, "right_eye_occluded": 0.2, "glasses": "None",
        }
        row.update(override)
        rows.append(row)

    for gender in ("female", "male"):
        for age_group in ("20-29", "30-39", "40-49"):
            for _ in range(PER_SUBGROUP):
                add(age_group, gender)
        for age_group in ("50-69", "70-120"):
            for _ in range(PER_SUBGROUP // 2):
                add(age_group, gender)
    for gender in ("female", "male"):
        for _ in range(20):
            add("15-19", gender)
            add("30-39", gender, glasses="Normal")
            add("30-39", gender, glasses="Dark")
            add("30-39", gender, age_group_confidence=0.59)
            add("30-39", gender, left_eye_occluded=1.0)
            add("30-39", gender, right_eye_occluded=40.0)
            add("30-39", gender, head_yaw=10.5)
            add("30-39", gender, head_pitch=-11.0)
    return pd.DataFrame(rows)


def ineligible_ids(labels: pd.DataFrame) -> set[str]:
    eligible = set(gs.eligible_ffhq(labels)["image_number"])
    return {str(n) for n in labels["image_number"] if n not in eligible}


def scut_names() -> list[str]:
    names = [f"{p}{i}.jpg" for p in ("AF", "AM", "CF", "CM") for i in range(1, 2001)]
    return [*names, "AF2001.jpg", "AF0.jpg", "AF5.png", "readme.txt"]


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    return gs.select_images(make_labels(), scut_names())


def role(manifest, name):
    return manifest[manifest["role"] == name]


def test_counts_in_every_subgroup(manifest):
    for (source, band, sex), quota in gs.QUOTAS.items():
        part = manifest[(manifest["source"] == source) & (manifest["age_band"] == band) & (manifest["sex"] == sex)]
        graded = part[part["role"] == "graded"]
        assert len(graded) == quota.graded
        assert (graded["pilot"] == "1").sum() == quota.pilot
        assert (part["role"] == "practice").sum() == quota.practice
        assert (part["role"] == "reserve").sum() == quota.reserve

    graded = role(manifest, "graded")
    ffhq = graded[graded["source"] == "ffhq"]
    assert len(ffhq) == 450
    assert ffhq["age_band"].value_counts().to_dict() == {"20-29": 113, "30-39": 113, "40-49": 112, "50+": 112}
    assert ffhq["sex"].value_counts().to_dict() == {"female": 225, "male": 225}
    scut = graded[graded["source"] == "scut"]
    assert scut["source_id"].str[:2].value_counts().to_dict() == {"AF": 75, "AM": 75}

    practice = role(manifest, "practice")
    assert len(practice) == 24
    assert (practice["source"] == "ffhq").sum() == 18
    assert set(practice.loc[practice["source"] == "ffhq", "age_band"]) == {"20-29", "30-39", "40-49", "50+"}
    assert practice.loc[practice["source"] == "scut", "sex"].value_counts().to_dict() == {"female": 3, "male": 3}

    pilot = graded[graded["pilot"] == "1"]
    assert (pilot["source"] == "ffhq").sum() == 150
    assert set(pilot.loc[pilot["source"] == "ffhq", "age_band"]) == {"20-29", "30-39", "40-49", "50+"}
    assert pilot.loc[pilot["source"] == "scut", "sex"].value_counts().to_dict() == {"female": 25, "male": 25}


def test_only_eligible_images(manifest):
    labels = make_labels()
    ffhq = manifest[manifest["source"] == "ffhq"]
    assert not set(ffhq["source_id"]) & ineligible_ids(labels)
    assert set(ffhq["age_group"]) <= set(gs.AGE_BANDS)
    scut = manifest[manifest["source"] == "scut"]
    assert scut["image"].str.fullmatch(r"A[FM]([1-9]\d{0,2}|1\d{3}|2000)\.jpg").all()
    assert (ffhq["image"] == ffhq["source_id"].astype(int).map("{:05d}.png".format)).all()


def test_no_image_in_two_roles(manifest):
    assert manifest["image"].is_unique
    sets = {name: set(role(manifest, name)["image"]) for name in ("graded", "practice", "reserve")}
    assert not sets["graded"] & sets["practice"]
    assert not sets["graded"] & sets["reserve"]
    assert not sets["practice"] & sets["reserve"]
    assert manifest["role"].isin(["graded", "practice", "reserve"]).all()


def test_split_exact_and_stratified(manifest):
    graded = role(manifest, "graded")
    assert graded["split"].value_counts().to_dict() == {"train": 420, "val": 90, "test": 90}
    assert (manifest.loc[manifest["role"] != "graded", "split"] == "").all()
    assert (manifest.loc[manifest["role"] != "graded", "pilot"] == "0").all()
    for key, quota in gs.QUOTAS.items():
        part = graded[[gs.subgroup(r) == key for _, r in graded.iterrows()]]
        for split, percent in gs.SPLIT_PERCENT.items():
            assert abs((part["split"] == split).sum() - quota.graded * percent / 100) < 1.5


def test_allocate_splits_totals():
    counts = gs.allocate_splits({"a": 57, "b": 56, "c": 56, "d": 75, "e": 56})
    total = sum(sum(c.values()) for c in counts.values())
    assert total == 300
    assert sum(c["test"] for c in counts.values()) == 45
    assert sum(c["val"] for c in counts.values()) == 45
    assert all(c["train"] > 0 for c in counts.values())


def test_repeatable(manifest):
    labels = make_labels()
    again = gs.select_images(labels, scut_names())
    pd.testing.assert_frame_equal(manifest, again)
    shuffled = labels.sample(frac=1, random_state=7).reset_index(drop=True)
    reordered = gs.select_images(shuffled, list(reversed(scut_names())))
    pd.testing.assert_frame_equal(manifest, reordered)


def test_replace_keeps_split_and_pilot(manifest):
    graded = role(manifest, "graded")
    target = graded[(graded["pilot"] == "1") & (graded["split"] == "test") & (graded["source"] == "ffhq")].iloc[0]
    old_reserves = role(manifest, "reserve")
    out, swaps = gs.replace_images(manifest, [target["image"]], "hand covers eye")
    new_image = swaps[target["image"]]
    new = out[out["image"] == new_image].iloc[0]
    old = out[out["image"] == target["image"]].iloc[0]
    assert new_image in set(old_reserves["image"])
    assert gs.subgroup(new) == gs.subgroup(target)
    assert (new["role"], new["split"], new["pilot"]) == ("graded", "test", "1")
    assert target["image"] in new["reason"]
    assert (old["role"], old["split"], old["pilot"]) == ("dropped", "", "0")
    assert "hand covers eye" in old["reason"] and new_image in old["reason"]
    graded_after = role(out, "graded")
    assert graded_after["split"].value_counts().to_dict() == {"train": 420, "val": 90, "test": 90}
    assert (graded_after["pilot"] == "1").sum() == 200
    assert len(role(out, "reserve")) == len(old_reserves) - 1
    # the input manifest is untouched
    assert manifest.loc[manifest["image"] == target["image"], "role"].item() == "graded"


def test_replace_uses_reserves_in_order_until_none_left(manifest):
    key = ("scut", "", "female")
    quota = gs.QUOTAS[key]
    reserves = role(manifest, "reserve")
    reserves = reserves[[gs.subgroup(r) == key for _, r in reserves.iterrows()]]
    expected = sorted(reserves["image"], key=lambda img: gs.rank("pick", img))
    graded = role(manifest, "graded")
    graded = list(graded.loc[[gs.subgroup(r) == key for _, r in graded.iterrows()], "image"])
    out, swaps = gs.replace_images(manifest, graded[:quota.reserve], "test")
    assert list(swaps.values()) == expected
    with pytest.raises(ValueError, match="no reserve left"):
        gs.replace_images(out, [graded[quota.reserve]], "test")


def test_replace_chain_and_practice(manifest):
    practice = role(manifest, "practice").iloc[0]
    out, swaps = gs.replace_images(manifest, [practice["image"]], "blurred")
    first = swaps[practice["image"]]
    assert out.loc[out["image"] == first, "role"].item() == "practice"
    out, swaps = gs.replace_images(out, [first], "heavy filter")
    row = out[out["image"] == first].iloc[0]
    assert row["role"] == "dropped"
    assert practice["image"] in row["reason"] and "heavy filter" in row["reason"]
    with pytest.raises(ValueError, match="dropped"):
        gs.replace_images(out, [first], "again")
    with pytest.raises(ValueError, match="not in the manifest"):
        gs.replace_images(out, ["99999.png"], "x")


def test_glasses_none_survives_read(tmp_path):
    path = tmp_path / "labels.csv"
    make_labels().to_csv(path, index=False)
    labels = gs.read_labels(path)
    assert labels["glasses"].isna().sum() == 0
    assert set(labels["glasses"]) == {"None", "Normal", "Dark"}
    assert len(gs.eligible_ffhq(labels)) == 6 * PER_SUBGROUP + 4 * (PER_SUBGROUP // 2)
    naive = pd.read_csv(path)
    assert naive["glasses"].isna().any(), "pandas defaults turn None into NaN; read_labels must not"


def test_manifest_round_trip(manifest, tmp_path):
    filled = manifest.copy()
    filled["file_md5"] = [hashlib.md5(img.encode()).hexdigest() for img in filled["image"]]
    filled.loc[filled["source"] == "ffhq", "author"] = "Ünïcode Author, Jr."
    path = tmp_path / "manifest.csv"
    gs.write_manifest(filled, path)
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.decode("utf-8").splitlines()[0] == ",".join(gs.MANIFEST_COLUMNS)
    back = gs.read_manifest(path)
    assert gs.manifest_text(back) == gs.manifest_text(filled)


def test_center_saturation():
    gray = Image.new("L", (200, 200), 128).convert("RGB")
    assert gs.center_saturation(gray) < 0.01
    skin = Image.new("RGB", (200, 200), (205, 160, 130))
    assert gs.center_saturation(skin) > gs.MIN_SATURATION
    # a grayscale face on a colourful background still counts as grayscale
    framed = Image.new("RGB", (200, 200), (30, 200, 60))
    framed.paste(gray.crop((0, 0, 100, 100)), (50, 50))
    assert gs.center_saturation(framed) < gs.MIN_SATURATION


def test_check_image(tmp_path):
    path = tmp_path / "00001.png"
    Image.new("RGB", (64, 64), (205, 160, 130)).save(path)
    md5 = gs.file_md5(path)
    assert gs.check_image("00001.png", path, md5, (64, 64)).problem == ""
    assert gs.check_image("00001.png", path, md5, (1024, 1024)).problem == "size 64x64"
    assert gs.check_image("00001.png", path, "0" * 32, None).problem == "md5 mismatch"
    assert gs.check_image("x.png", tmp_path / "x.png", md5, None).problem == "missing"
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert gs.check_image("broken.png", broken, gs.file_md5(broken), None).problem.startswith("unreadable")


def md5_of(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def job(tmp_path: Path, name: str, data: bytes) -> gs.DownloadJob:
    return gs.DownloadJob(name, f"https://example.invalid/{name}", tmp_path / "raw" / name, md5_of(data))


def test_download_all_skips_retries_and_never_overwrites(tmp_path):
    good, flaky, broken, present, conflict = (f"{n}.png" for n in ("good", "flaky", "broken", "present", "conflict"))
    data = {name: name.encode() * 10 for name in (good, flaky, broken, present, conflict)}
    jobs = [job(tmp_path, name, data[name]) for name in data]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / present).write_bytes(data[present])
    (tmp_path / "raw" / conflict).write_bytes(b"something else")
    calls: dict[str, int] = {}
    lock = threading.Lock()

    def fetch(url, out, stop):
        name = url.rsplit("/", 1)[1]
        with lock:
            calls[name] = calls.get(name, 0) + 1
        if name == flaky and calls[name] == 1:
            raise ConnectionError("reset")
        out.write_bytes(b"wrong bytes" if name == broken else data[name])

    staging = tmp_path / "staging"
    report = gs.download_all(jobs, fetch, staging, workers=2, attempts=3, base_delay=0)
    assert report.done == [flaky, good]
    assert report.skipped == [present]
    assert report.conflicts == [conflict]
    assert list(report.failed) == [broken] and report.failed[broken] == "md5 mismatch"
    assert calls == {good: 1, flaky: 2, broken: 3}
    assert (tmp_path / "raw" / good).read_bytes() == data[good]
    assert (tmp_path / "raw" / conflict).read_bytes() == b"something else"
    assert not (tmp_path / "raw" / broken).exists()
    assert list(staging.iterdir()) == []


def test_download_all_stops_on_quota(tmp_path):
    names = [f"{i:05d}.png" for i in range(12)]
    jobs = [job(tmp_path, name, name.encode()) for name in names]

    def fetch(url, out, stop):
        name = url.rsplit("/", 1)[1]
        if name == names[3]:
            raise gs.QuotaExceeded("Too many users have viewed or downloaded this file recently.")
        if stop.is_set():
            raise gs.Cancelled
        out.write_bytes(name.encode())

    report = gs.download_all(jobs, fetch, tmp_path / "staging", workers=1, attempts=3, base_delay=0)
    assert report.quota_message.startswith("Too many users")
    assert report.done == names[:3]
    assert report.not_started == names[3:]
    assert not report.failed
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == names[:3]


def test_share_readmes_fill_every_folder():
    readmes = gs.share_readmes(gs.SHARE_SETS)
    assert set(readmes) == set(gs.SHARE_SETS)
    for name, count in gs.SHARE_SETS.items():
        text = readmes[name]
        assert f"พบ {count} ภาพ" in text
        assert "Chrome" in text and "ส่งออก CSV" in text and "ลบ" in text
        assert "{" not in text
        assert not re.search(r"\b(train|val|test|split|manifest)\b", text, re.IGNORECASE)
    assert "pilot" not in readmes["full_600"].lower()


def test_build_share(tmp_path, manifest):
    sets = gs.share_sets(manifest)
    assert {name: len(rows) for name, rows in sets.items()} == gs.SHARE_SETS
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    files = {}
    for name, rows in sets.items():
        files[name] = []
        for image in rows["image"]:
            path = src / image
            if not path.exists():
                Image.fromarray(rng.integers(0, 255, (4, 4, 3), dtype=np.uint8)).save(path)
            files[name].append(path)
    grader = tmp_path / gs.GRADER_NAME
    grader.write_text("<html></html>", encoding="utf-8")
    share = tmp_path / "share"
    zips = gs.build_share(share, files, grader, gs.share_readmes(gs.SHARE_SETS))
    first = {name: path.read_bytes() for name, path in zips.items()}
    for name, count in gs.SHARE_SETS.items():
        folder = share / name
        images = sorted(p.name for p in folder.iterdir() if p.suffix in (".png", ".jpg"))
        assert images == sorted(sets[name]["image"])
        assert len(images) == count
        assert (folder / gs.GRADER_NAME).is_file()
        assert "Chrome" in (folder / "README.txt").read_text(encoding="utf-8-sig")
        assert not any(p.suffix == ".csv" for p in folder.iterdir())
        with zipfile.ZipFile(zips[name]) as zf:
            members = zf.namelist()
        assert len(members) == count + 2
        assert all(m.startswith(f"{name}/") for m in members)
        assert not any(p.is_symlink() for p in folder.iterdir())
    pilot = set(sets["pilot_200"]["image"])
    assert pilot <= set(sets["full_600"]["image"])
    # rebuilding gives byte-identical zips
    zips = gs.build_share(share, files, grader, gs.share_readmes(gs.SHARE_SETS))
    assert {name: path.read_bytes() for name, path in zips.items()} == first


def test_contact_sheets_mark_problem_images(tmp_path, monkeypatch, manifest):
    rows = role(manifest, "practice").head(gs.CONTACT_COLUMNS * gs.CONTACT_ROWS + 3)
    rows = pd.concat([rows, role(manifest, "graded").head(60)]).head(51)
    for image in rows["image"]:
        Image.new("RGB", (32, 40), (205, 160, 130)).save(tmp_path / image)
    missing = rows["image"].iloc[5]
    (tmp_path / missing).unlink()
    monkeypatch.setattr(gs, "image_path", lambda row: tmp_path / row["image"])
    out = tmp_path / "sheets"
    out.mkdir()
    pages = gs.contact_sheets(rows, out, "graded", {missing: "missing"})
    assert [p.name for p in pages] == ["graded_01.jpg", "graded_02.jpg"]
    with Image.open(pages[0]) as page:
        assert page.size == (gs.CONTACT_COLUMNS * gs.CONTACT_THUMB, gs.CONTACT_ROWS * (gs.CONTACT_THUMB + gs.CONTACT_LABEL))

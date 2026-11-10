"""FEI download, unpack and pose numbering helpers of profile_check.py, on small synthetic zips."""

import io
import zipfile

import pytest
import requests

import profile_check as pc
from facemeasure import fei


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


GOOD = zip_bytes({"1-01.jpg": b"\xff\xd8\xff photo", "1-02.jpg": b"\xff\xd8\xff other"})


def writer(data: bytes):
    calls = []

    def fetch(url, part):
        calls.append(url)
        part.write_bytes(data)

    return fetch, calls


def test_ensure_zip_downloads_then_keeps(tmp_path):
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    raw.mkdir()
    staging.mkdir()
    fetch, calls = writer(GOOD)
    assert pc.ensure_zip(1, raw, staging, fetch) is True
    assert (raw / fei.zip_name(1)).read_bytes() == GOOD
    assert calls == [fei.zip_url(1)] and not any(staging.iterdir())
    assert pc.ensure_zip(1, raw, staging, fetch) is False  # a readable zip is not downloaded again
    assert len(calls) == 1


def test_ensure_zip_resumes_after_a_network_error(tmp_path):
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    raw.mkdir()
    staging.mkdir()
    half = len(GOOD) // 2
    sizes = []

    def fetch(url, part):
        have = part.stat().st_size if part.is_file() else 0
        sizes.append(have)
        with open(part, "ab") as f:
            if have == 0:
                f.write(GOOD[:half])
                raise requests.ConnectionError("dropped")
            f.write(GOOD[have:])

    assert pc.ensure_zip(1, raw, staging, fetch, attempts=3) is True
    assert sizes == [0, half]  # the second attempt went on from the first one's bytes
    assert (raw / fei.zip_name(1)).read_bytes() == GOOD


def test_ensure_zip_starts_over_after_a_bad_file(tmp_path):
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    raw.mkdir()
    staging.mkdir()
    results = [b"not a zip at all", GOOD]

    def fetch(url, part):
        with open(part, "ab") as f:
            f.write(results.pop(0))

    assert pc.ensure_zip(1, raw, staging, fetch, attempts=3) is True
    assert (raw / fei.zip_name(1)).read_bytes() == GOOD


def test_ensure_zip_gives_up(tmp_path):
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    raw.mkdir()
    staging.mkdir()

    def fetch(url, part):
        raise requests.Timeout("slow")

    with pytest.raises(pc.FeiError):
        pc.ensure_zip(1, raw, staging, fetch, attempts=2)
    assert not (raw / fei.zip_name(1)).exists()


def test_bad_zip_in_raw_is_left_alone(tmp_path):
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    raw.mkdir()
    staging.mkdir()
    (raw / fei.zip_name(2)).write_bytes(b"broken")
    with pytest.raises(pc.FeiError):
        pc.ensure_zip(2, raw, staging, writer(GOOD)[0])
    assert (raw / fei.zip_name(2)).read_bytes() == b"broken"


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status_code, self.body = status, body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, size):
        yield self.body


@pytest.mark.parametrize("honours_range", [True, False])
def test_fetch_resumable(tmp_path, monkeypatch, honours_range):
    part = tmp_path / "a.zip.part"
    part.write_bytes(GOOD[:10])
    seen = {}

    def get(url, headers=None, stream=None, timeout=None):
        seen.update(headers or {})
        return FakeResponse(206, GOOD[10:]) if honours_range else FakeResponse(200, GOOD)

    monkeypatch.setattr(pc.requests, "get", get)
    pc.fetch_resumable("https://example.test/a.zip", part)
    assert seen == {"Range": "bytes=10-"}
    assert part.read_bytes() == GOOD


def test_fetch_resumable_already_whole(tmp_path, monkeypatch):
    part = tmp_path / "a.zip.part"
    part.write_bytes(GOOD)
    monkeypatch.setattr(pc.requests, "get", lambda *a, **k: FakeResponse(416, b""))
    pc.fetch_resumable("https://example.test/a.zip", part)
    assert part.read_bytes() == GOOD


def test_unpack(tmp_path):
    staging, images = tmp_path / "staging", tmp_path / "images"
    staging.mkdir()
    first = tmp_path / "one.zip"
    first.write_bytes(zip_bytes({
        "originalimages/1-01.jpg": b"a",
        "../1-02.jpg": b"b",  # only the last part of the name is used, so it lands in images/
        "readme.txt": b"x",
        "201-01.jpg": b"x",  # no person 201
        "1-15.jpg": b"x",  # no pose 15
    }))
    second = tmp_path / "two.zip"
    second.write_bytes(zip_bytes({"1-01.jpg": b"dup", "1-03.jpg": b"c"}))
    out, seen = pc.Unpacked(), {}
    pc.unpack(first, images, staging, seen, out)
    pc.unpack(second, images, staging, seen, out)
    assert sorted(p.name for p in images.iterdir()) == ["1-01.jpg", "1-02.jpg", "1-03.jpg"]
    assert (images / "1-02.jpg").read_bytes() == b"b" and not (tmp_path / "1-02.jpg").exists()
    assert sorted(out.written) == ["1-01.jpg", "1-02.jpg", "1-03.jpg"]
    assert len(out.ignored) == 3 and len(out.conflicts) == 1  # 1-01.jpg in both zips
    # A second run keeps what is there and reports a file that differs
    (images / "1-03.jpg").write_bytes(b"changed")
    again = pc.Unpacked()
    pc.unpack(first, images, staging, {}, again)
    pc.unpack(second, images, staging, {"1-01.jpg": first.name}, again)
    assert sorted(again.kept) == ["1-01.jpg", "1-02.jpg"] and "1-03.jpg" in again.conflicts
    assert (images / "1-03.jpg").read_bytes() == b"changed"


def test_checksums(tmp_path):
    for part in fei.ZIP_PARTS:
        (tmp_path / fei.zip_name(part)).write_bytes(zip_bytes({f"{part}-01.jpg": bytes([part])}))
    path = pc.ensure_checksums(tmp_path)
    recorded = pc.read_checksums(path)
    assert set(recorded) == {fei.zip_name(p) for p in fei.ZIP_PARTS}
    assert path.read_bytes().count(b"\r") == 0
    assert pc.ensure_checksums(tmp_path) == path
    (tmp_path / fei.zip_name(3)).write_bytes(GOOD)
    with pytest.raises(pc.FeiError):
        pc.ensure_checksums(tmp_path)


def test_missing_images(tmp_path):
    (tmp_path / "1-01.jpg").write_bytes(b"x")
    missing = pc.missing_images(tmp_path)
    assert len(missing) == fei.PEOPLE * fei.POSES - 1 and "1-01.jpg" not in missing


def stat(number, yaws, smile=0.03, found=5, brightness=132.0):
    return pc.PoseStat(number, 5, found, tuple(yaws), tuple([smile] * len(yaws)), brightness)


def test_guess_numbers_like_fei():
    # Medians from the first five FEI people: 1 and 10 turn the most, 13 and 14 are darker
    stats = [
        stat(1, [-50, -49, -52, -48], found=4), stat(2, [-38] * 5), stat(3, [-29] * 5), stat(4, [-23] * 5),
        stat(5, [-7.5] * 5), stat(6, [6.7] * 5), stat(7, [22] * 5), stat(8, [33] * 5), stat(9, [38] * 5),
        stat(10, [49, 50], found=2), stat(11, [0.5] * 5), stat(12, [0.2] * 5, smile=0.73),
        stat(13, [-0.2] * 5, smile=0.2, brightness=115.0), stat(14, [-0.4] * 5, smile=0.1, brightness=57.0),
    ]
    assert pc.guess_numbers(stats) == pc.Guess(full=(1, 10), second=(2, 9), neutral=11, smile=12)


def test_guess_numbers_needs_both_sides():
    stats = [stat(n, [-40 + n]) for n in range(1, 15)]
    with pytest.raises(pc.FeiError):
        pc.guess_numbers(stats)


# ---------------------------------------------------------------- report helpers


def test_every_reason_has_a_text():
    assert set(pc.REASON_TEXT) == set(pc.P.REASONS)


def test_poses_follow_fei_numbers():
    poses = pc.poses()
    assert poses[fei.PROFILE_FULL[0]] == ("full", "image_left")
    assert poses[fei.PROFILE_FULL[1]] == ("full", "image_right")
    assert {v[0] for v in poses.values()} == {"full", "second", "frontal_neutral", "frontal_smile"}


def test_bands():
    import pandas as pd

    legan = pd.Series([10.0, 16.0, 16.5, 20.0, 23.0, 30.0])
    assert [n for _, n in pc.legan_bands(legan)] == [2, 2, 1, 1]
    c = pc.criteria.DORSUM_CUTOFFS["female"]
    naso = pd.Series([c.small - 1, c.small, c.low, c.high, c.high + 1])
    assert [n for _, n in pc.dorsum_bands(naso, "female")] == [2, 1, 1, 1]
    assert len(pc.DORSUM_GRADES) == len(pc.dorsum_bands(naso, "male"))


def test_passes_yaw_and_pairs():
    import pandas as pd

    df = pd.DataFrame({
        "person": [1, 1, 2, 2],
        "pose": ["full"] * 4,
        "facing": ["image_left", "image_right"] * 2,
        "face_found": [True, False, True, True],
        "face_yaw_deg": [-45.0, None, 30.0, 50.0],
        "legan_angle": [10.0, 12.0, None, 8.0],
        "nasofacial_angle": [30.0, 33.0, None, 35.0],
    })
    assert pc.passes_yaw(df, 40).tolist() == [True, True, False, True]
    lr = pc.left_right(df, "full")
    assert lr.index.tolist() == [1] and lr["legan_angle_image_right"].iloc[0] == 12.0


def test_table_escapes_pipes():
    out = pc.table(["|yaw|", "n"], [["|G-Pog|", 3]])
    assert out.splitlines()[0] == "| \\|yaw\\| | n |"
    assert out.splitlines()[2] == "| \\|G-Pog\\| | 3 |"

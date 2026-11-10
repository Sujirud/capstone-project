"""facemeasure.assets with a fake download in place of requests."""

import hashlib
import json

import pytest

from facemeasure import assets

CONTENT = b"model bytes " * 100
OTHER = b"a newer latest file " * 50
URL = "https://example.com/latest/model.task"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_lock(path, files):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"version": 1, "files": files}, f)


def entry(name="model.task", size=None, sha256=None, url=URL):
    return {"name": name, "url": url, "size": size, "sha256": sha256}


class FakeFetch:
    def __init__(self, data: bytes):
        self.data = data
        self.calls = []

    def __call__(self, url, dest):
        self.calls.append((url, dest.name))
        dest.write_bytes(self.data)


@pytest.fixture
def setup(tmp_path):
    lock, folder = tmp_path / "assets.lock.json", tmp_path / "assets"
    return lock, folder


def test_first_download_records_size_and_sha256(setup):
    lock, folder = setup
    write_lock(lock, [entry()])
    fetch = FakeFetch(CONTENT)
    downloaded, locked = assets.ensure(lock, folder, fetch)
    assert downloaded == ["model.task"] and locked == ["model.task"]
    assert fetch.calls == [(URL, "model.task.part")]
    assert (folder / "model.task").read_bytes() == CONTENT
    assert not (folder / "model.task.part").exists()
    (e,) = assets.read_lock(lock)
    assert e.size == len(CONTENT) and e.sha256 == sha(CONTENT)
    text = lock.read_text(encoding="utf-8")
    assert text.endswith("}\n") and "\r" not in text
    assert assets.read_verified("model.task", lock, folder) == CONTENT


def test_matching_file_is_not_downloaded_again(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT))])
    folder.mkdir()
    (folder / "model.task").write_bytes(CONTENT)
    before = lock.read_text(encoding="utf-8")
    fetch = FakeFetch(OTHER)
    assert assets.ensure(lock, folder, fetch) == ([], [])
    assert fetch.calls == [] and lock.read_text(encoding="utf-8") == before


def test_missing_locked_file_is_downloaded_and_checked(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT))])
    assert assets.ensure(lock, folder, FakeFetch(CONTENT)) == (["model.task"], [])
    assert (folder / "model.task").read_bytes() == CONTENT


def test_changed_latest_file_is_refused_and_lock_kept(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT))])
    before = lock.read_text(encoding="utf-8")
    with pytest.raises(assets.AssetMismatch):
        assets.ensure(lock, folder, FakeFetch(OTHER))
    assert not (folder / "model.task").exists() and not (folder / "model.task.part").exists()
    assert lock.read_text(encoding="utf-8") == before


def test_changed_file_on_disk_is_refused_and_left_alone(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT))])
    folder.mkdir()
    (folder / "model.task").write_bytes(OTHER)
    with pytest.raises(assets.AssetMismatch):
        assets.ensure(lock, folder, FakeFetch(CONTENT))
    assert (folder / "model.task").read_bytes() == OTHER
    with pytest.raises(assets.AssetMismatch):
        assets.read_verified("model.task", lock, folder)


def test_new_entry_needs_no_code_change(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT)), entry(name="weights.pth", url="https://example.com/w.pth")])
    folder.mkdir()
    (folder / "model.task").write_bytes(CONTENT)
    downloaded, locked = assets.ensure(lock, folder, FakeFetch(OTHER))
    assert downloaded == ["weights.pth"] and locked == ["weights.pth"]
    assert [e.name for e in assets.read_lock(lock)] == ["model.task", "weights.pth"]
    assert assets.read_verified("weights.pth", lock, folder) == OTHER


def test_read_before_locking_or_downloading(setup):
    lock, folder = setup
    write_lock(lock, [entry()])
    with pytest.raises(assets.AssetMissing):
        assets.read_verified("model.task", lock, folder)
    with pytest.raises(assets.AssetMissing):
        assets.read_verified("unknown.task", lock, folder)


def test_download_error_leaves_no_part_file(setup):
    lock, folder = setup
    write_lock(lock, [entry()])

    def broken(url, dest):
        dest.write_bytes(b"half")
        raise OSError("connection reset")

    with pytest.raises(OSError):
        assets.ensure(lock, folder, broken)
    assert list(folder.iterdir()) == []
    assert assets.read_lock(lock)[0].sha256 is None


@pytest.mark.parametrize(
    "files",
    [
        [{"name": "a.task", "url": URL}],
        [entry(name="../a.task")],
        [entry(url="http://example.com/a")],
        [entry(size=10)],
        [entry(size=10, sha256="ABC")],
        [entry(), entry()],
    ],
)
def test_bad_lock(setup, files):
    lock, _ = setup
    write_lock(lock, files)
    with pytest.raises(assets.AssetError):
        assets.read_lock(lock)


def test_project_lock_lists_the_mediapipe_files():
    names = [e.name for e in assets.read_lock()]
    assert names[:2] == ["face_landmarker.task", "selfie_multiclass_256x256.tflite"]


# ---------------------------------------------------------------- zip archives

COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def zip_bytes(files: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
    return buf.getvalue()


SOURCE_ZIP = zip_bytes({"pkg-abc/pkg/__init__.py": b"VERSION = 1\n", "pkg-abc/README.md": b"code\n"})


def zip_entry(data: bytes | None = None, **kw):
    e = entry(name="code.zip", url="https://example.com/code.zip", **kw)
    if data is not None:
        e.update(size=len(data), sha256=sha(data))
    return {**e, "commit": COMMIT, "extract": "code_src"}


def test_zip_is_unpacked_once(setup):
    lock, folder = setup
    write_lock(lock, [zip_entry()])
    assets.ensure(lock, folder, FakeFetch(SOURCE_ZIP))
    assert assets.extract_all(lock, folder) == ["code.zip"]
    target, e = assets.extracted("code_src", lock, folder)
    assert (target / "pkg-abc" / "pkg" / "__init__.py").read_bytes() == b"VERSION = 1\n"
    assert e.commit == COMMIT and e.sha256 == sha(SOURCE_ZIP)
    (target / "touched.txt").write_text("still here", encoding="utf-8")
    assert assets.extract_all(lock, folder) == []  # same zip: not unpacked again
    assert (target / "touched.txt").is_file()
    assert not (folder / "code_src.tmp").exists()
    # optional keys stay in the lock; entries without them keep four keys
    raw = json.loads(lock.read_text(encoding="utf-8"))["files"][0]
    assert raw["commit"] == COMMIT and raw["extract"] == "code_src"


def test_new_zip_sha256_unpacks_again(setup):
    lock, folder = setup
    write_lock(lock, [zip_entry(SOURCE_ZIP)])
    folder.mkdir()
    (folder / "code.zip").write_bytes(SOURCE_ZIP)
    assets.extract_all(lock, folder)
    (folder / "code_src" / "touched.txt").write_text("old", encoding="utf-8")
    newer = zip_bytes({"pkg-def/pkg/__init__.py": b"VERSION = 2\n"})
    (folder / "code.zip").write_bytes(newer)
    write_lock(lock, [zip_entry(newer)])
    assert assets.extract_all(lock, folder) == ["code.zip"]
    target, _ = assets.extracted("code_src", lock, folder)
    assert not (target / "touched.txt").exists() and not (target / "pkg-abc").exists()
    assert (target / "pkg-def" / "pkg" / "__init__.py").read_bytes() == b"VERSION = 2\n"


def test_zip_that_does_not_match_is_refused_and_not_unpacked(setup):
    lock, folder = setup
    write_lock(lock, [zip_entry(SOURCE_ZIP)])
    folder.mkdir()
    (folder / "code.zip").write_bytes(zip_bytes({"evil/pkg/__init__.py": b"x = 1\n"}))
    with pytest.raises(assets.AssetMismatch):
        assets.extract_all(lock, folder)
    assert not (folder / "code_src").exists() and not (folder / "code_src.tmp").exists()
    with pytest.raises(assets.AssetMissing):
        assets.extracted("code_src", lock, folder)


def test_zip_member_outside_the_folder_is_refused(setup):
    lock, folder = setup
    bad = zip_bytes({"../outside.py": b"x = 1\n"})
    write_lock(lock, [zip_entry(bad)])
    folder.mkdir()
    (folder / "code.zip").write_bytes(bad)
    with pytest.raises(assets.AssetError, match="outside"):
        assets.extract_all(lock, folder)
    assert not (folder.parent / "outside.py").exists() and not (folder / "code_src").exists()


def test_unpacked_folder_is_needed_before_use(setup):
    lock, folder = setup
    write_lock(lock, [zip_entry(SOURCE_ZIP)])
    with pytest.raises(assets.AssetMissing):
        assets.extracted("code_src", lock, folder)
    with pytest.raises(assets.AssetMissing):
        assets.extracted("other_src", lock, folder)


@pytest.mark.parametrize(
    "extra",
    [{"commit": "abc"}, {"extract": "../up"}, {"extract": "a.tmp"}, {"extract": "model.task"}, {"other": 1}],
)
def test_bad_optional_keys(setup, extra):
    lock, _ = setup
    files = [entry(), {**zip_entry(), **extra}] if extra.get("extract") == "model.task" else [{**zip_entry(), **extra}]
    write_lock(lock, files)
    with pytest.raises(assets.AssetError):
        assets.read_lock(lock)


def test_extract_needs_a_zip_entry(setup):
    lock, _ = setup
    write_lock(lock, [{**entry(), "extract": "x_src"}])
    with pytest.raises(assets.AssetError, match="zip"):
        assets.read_lock(lock)


def test_project_lock_lists_dinov2_with_its_commit():
    entries = {e.name: e for e in assets.read_lock()}
    weights = entries["dinov2_vitb14_pretrain.pth"]
    code = next(e for e in entries.values() if e.extract == "dinov2_src")
    assert weights.locked and code.locked and code.commit == COMMIT and COMMIT in code.url
    assert entries["face_landmarker.task"].as_dict().keys() == {"name", "url", "size", "sha256"}


def test_verify_checks_every_file_and_unpacked_zip_without_downloading(setup):
    lock, folder = setup
    write_lock(lock, [entry(size=len(CONTENT), sha256=sha(CONTENT)), zip_entry(SOURCE_ZIP)])
    with pytest.raises(assets.AssetMissing, match="facemeasure.assets"):
        assets.verify(lock, folder)  # nothing there yet, and nothing is fetched
    assert not folder.exists() or not any(folder.iterdir())
    folder.mkdir(exist_ok=True)
    (folder / "model.task").write_bytes(CONTENT)
    (folder / "code.zip").write_bytes(SOURCE_ZIP)
    with pytest.raises(assets.AssetMissing, match="code_src"):
        assets.verify(lock, folder)  # the zip is there but not unpacked
    assets.extract_all(lock, folder)
    assert [e.name for e in assets.verify(lock, folder)] == ["model.task", "code.zip"]
    (folder / "model.task").write_bytes(OTHER)
    with pytest.raises(assets.AssetMismatch):
        assets.verify(lock, folder)


def test_verify_needs_every_entry_locked(setup):
    lock, folder = setup
    write_lock(lock, [entry()])
    with pytest.raises(assets.AssetMissing, match="no sha256"):
        assets.verify(lock, folder)

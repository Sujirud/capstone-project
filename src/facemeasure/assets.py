"""Model files the project downloads but does not train, locked by sha256 (api.md section 5).

Download the missing files from the project root:

    uv run python -m facemeasure.assets

The list is api/assets.lock.json (committed); the files go to api/assets/ (not in git).
Each entry has name, url, size and sha256. Add an entry with size and sha256 set to null
and the next run downloads it and records both, so new files need no code change.

Two optional keys: commit, the source commit a code archive was made from, and extract,
a folder name under api/assets/ that a .zip entry is unpacked into. A zip is unpacked
only after its sha256 matches the lock, into a temporary folder that then takes the
place of the old one, with EXTRACTED_MARKER holding the sha256 of the zip it came from.
It is unpacked once, and again only when the zip's sha256 changes.

Google publishes the MediaPipe models only under "latest" links, so a later download can
differ. A file whose sha256 does not match the lock is never used and the lock is never
changed to fit it: copy api/assets/ from the machine that made the lock instead. The api
reads files only through read_verified(), checks them all at start with verify(), and
never downloads anything.
"""

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests
from tqdm import tqdm

from facemeasure import paths

LOCK_VERSION = 1
ENTRY_KEYS = ("name", "url", "size", "sha256")
OPTIONAL_KEYS = ("commit", "extract")
CHUNK_BYTES = 1 << 20
DOWNLOAD_TIMEOUT = 60  # seconds without data before requests gives up
PART_SUFFIX = ".part"
EXTRACT_SUFFIX = ".tmp"
EXTRACTED_MARKER = "extracted.json"


class AssetError(RuntimeError):
    """A model file cannot be used."""


class AssetMismatch(AssetError):
    """A file's size or sha256 differs from the lock."""


class AssetMissing(AssetError):
    """A file is not in the assets folder, or the lock has no sha256 for it yet."""


@dataclass(frozen=True)
class Entry:
    name: str
    url: str
    size: int | None
    sha256: str | None
    commit: str | None = None
    extract: str | None = None

    @property
    def locked(self) -> bool:
        return self.sha256 is not None

    def as_dict(self) -> dict:
        """The lock entry; optional keys only when set, so entries without them keep their form."""
        out = {k: getattr(self, k) for k in ENTRY_KEYS}
        out.update({k: getattr(self, k) for k in OPTIONAL_KEYS if getattr(self, k) is not None})
        return out


Fetch = Callable[[str, Path], None]


def lock_path() -> Path:
    return paths.path("assets_lock")


def assets_dir() -> Path:
    return paths.path("assets")


def _hex(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _plain_name(value) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).name == value and value not in (".", "..")


def _entry(raw: dict, k: int) -> Entry:
    if not isinstance(raw, dict) or not set(ENTRY_KEYS) <= set(raw) or not set(raw) <= {*ENTRY_KEYS, *OPTIONAL_KEYS}:
        keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
        raise AssetError(
            f"lock entry {k} must have the keys {', '.join(ENTRY_KEYS)} and may have {', '.join(OPTIONAL_KEYS)}, got {keys}"
        )
    name, url, size, sha = raw["name"], raw["url"], raw["size"], raw["sha256"]
    commit, extract = raw.get("commit"), raw.get("extract")
    if not isinstance(name, str) or not name or Path(name).name != name or name.endswith(PART_SUFFIX):
        raise AssetError(f"lock entry {k}: name must be a plain file name, got {name!r}")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise AssetError(f"lock entry {k} ({name}): url must start with https://")
    if (size is None) != (sha is None):
        raise AssetError(f"lock entry {k} ({name}): size and sha256 must both be set or both be null")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        raise AssetError(f"lock entry {k} ({name}): size must be a positive integer")
    if sha is not None and not _hex(sha, 64):
        raise AssetError(f"lock entry {k} ({name}): sha256 must be 64 lowercase hex digits")
    if "commit" in raw and not _hex(commit, 40):
        raise AssetError(f"lock entry {k} ({name}): commit must be 40 lowercase hex digits")
    if "extract" in raw:
        if not name.endswith(".zip"):
            raise AssetError(f"lock entry {k} ({name}): only a .zip entry can have extract")
        if not _plain_name(extract) or extract.endswith((EXTRACT_SUFFIX, PART_SUFFIX)):
            raise AssetError(f"lock entry {k} ({name}): extract must be a plain folder name, got {extract!r}")
    return Entry(name=name, url=url, size=size, sha256=sha, commit=commit, extract=extract)


def read_lock(path: Path | None = None) -> list[Entry]:
    path = Path(path) if path is not None else lock_path()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or set(data) != {"version", "files"}:
        raise AssetError(f"{path.name} must have exactly the keys version and files")
    if data["version"] != LOCK_VERSION:
        raise AssetError(f"{path.name}: version {data['version']!r} is not {LOCK_VERSION}")
    if not isinstance(data["files"], list):
        raise AssetError(f"{path.name}: files must be a list")
    entries = [_entry(raw, k) for k, raw in enumerate(data["files"])]
    names = [e.name for e in entries] + [e.extract for e in entries if e.extract is not None]
    if len(set(names)) != len(names):
        raise AssetError(f"{path.name}: a file or folder name appears twice")
    return entries


def write_lock(entries: list[Entry], path: Path | None = None) -> None:
    """Write the lock as UTF-8 JSON with \\n line ends, through a temporary file."""
    path = Path(path) if path is not None else lock_path()
    text = json.dumps({"version": LOCK_VERSION, "files": [e.as_dict() for e in entries]}, ensure_ascii=False, indent=2)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    tmp.replace(path)


def file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream url into dest with a progress bar."""
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as res:
        res.raise_for_status()
        total = int(res.headers.get("Content-Length", 0)) or None
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name.removesuffix(PART_SUFFIX)) as bar:
            for chunk in res.iter_content(CHUNK_BYTES):
                f.write(chunk)
                bar.update(len(chunk))


def _check(entry: Entry, path: Path) -> None:
    """Raise AssetMismatch when the file at path differs from a locked entry."""
    size = path.stat().st_size
    if size != entry.size:
        raise AssetMismatch(f"{entry.name}: {size} bytes, the lock says {entry.size}")
    sha = file_sha256(path)
    if sha != entry.sha256:
        raise AssetMismatch(f"{entry.name}: sha256 {sha}, the lock says {entry.sha256}")


def _fetch_checked(entry: Entry, folder: Path, fetch: Fetch) -> Entry:
    """Download entry to a temporary file in folder, check it, then move it into place.

    For an entry without sha256 the downloaded size and sha256 are returned as the new lock entry.
    """
    part = folder / (entry.name + PART_SUFFIX)
    try:
        fetch(entry.url, part)
        locked = entry
        if not entry.locked:
            locked = dataclasses.replace(entry, size=part.stat().st_size, sha256=file_sha256(part))
        _check(locked, part)
        part.replace(folder / entry.name)
        return locked
    finally:
        part.unlink(missing_ok=True)


def ensure(
    lock: Path | None = None, folder: Path | None = None, fetch: Fetch = download
) -> tuple[list[str], list[str]]:
    """Make every file in the lock present and matching. Return (downloaded, newly locked) names.

    A file already in folder is checked, never downloaded again. A locked file that does
    not match raises AssetMismatch and is left as it is. An entry without sha256 takes
    the size and sha256 of the file in folder, or of the download, and the lock is
    rewritten. Entries are handled in order; one that fails stops the run, and the lock
    keeps the entries recorded before it.
    """
    lock = Path(lock) if lock is not None else lock_path()
    folder = Path(folder) if folder is not None else assets_dir()
    folder.mkdir(parents=True, exist_ok=True)
    entries = read_lock(lock)
    downloaded: list[str] = []
    newly_locked: list[str] = []
    for k, entry in enumerate(entries):
        target = folder / entry.name
        if target.is_file():
            if entry.locked:
                _check(entry, target)
                continue
            entries[k] = dataclasses.replace(entry, size=target.stat().st_size, sha256=file_sha256(target))
        else:
            entries[k] = _fetch_checked(entry, folder, fetch)
            downloaded.append(entry.name)
        if not entry.locked:
            newly_locked.append(entry.name)
            write_lock(entries, lock)
    return downloaded, newly_locked


def asset_path(name: str, lock: Path | None = None, folder: Path | None = None) -> Path:
    """Path of a locked file in the assets folder, without reading it."""
    entries = {e.name: e for e in read_lock(lock)}
    if name not in entries:
        raise AssetMissing(f"{name} is not in the asset lock")
    if not entries[name].locked:
        raise AssetMissing(f"{name} has no sha256 in the lock yet; run: uv run python -m facemeasure.assets")
    path = (Path(folder) if folder is not None else assets_dir()) / name
    if not path.is_file():
        raise AssetMissing(f"{name} is not in the assets folder; run: uv run python -m facemeasure.assets")
    return path


def read_verified(name: str, lock: Path | None = None, folder: Path | None = None) -> bytes:
    """Bytes of a locked file after checking its size and sha256 against the lock."""
    entry = {e.name: e for e in read_lock(lock)}.get(name)
    path = asset_path(name, lock, folder)
    data = path.read_bytes()
    if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.sha256:
        raise AssetMismatch(f"{name} in the assets folder differs from the lock")
    return data


def verify(lock: Path | None = None, folder: Path | None = None) -> list[Entry]:
    """Check every file of the lock against its size and sha256, and every unpacked zip; never download.

    Raises AssetMissing (with the command that fetches the files) or AssetMismatch.
    """
    folder = Path(folder) if folder is not None else assets_dir()
    entries = read_lock(lock)
    for e in entries:
        path = asset_path(e.name, lock, folder)
        _check(e, path)
    for e in entries:
        if e.extract is not None:
            extracted(e.extract, lock, folder)
    return entries


# ---------------------------------------------------------------- zip archives


def _marker(entry: Entry) -> dict:
    return {"zip": entry.name, "sha256": entry.sha256, "commit": entry.commit}


def _marker_matches(target: Path, entry: Entry) -> bool:
    path = target / EXTRACTED_MARKER
    if not path.is_file():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) == _marker(entry)
    except (OSError, ValueError):
        return False


def _check_member(name: str, zip_name: str) -> None:
    """Refuse a member that would land outside the folder (zip slip)."""
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or (p.parts and ":" in p.parts[0]):
        raise AssetError(f"{zip_name}: member {name!r} points outside the folder")


def extract(entry: Entry, folder: Path | None = None) -> bool:
    """Unpack a locked zip into folder/entry.extract unless that folder holds it already.

    Returns True when it unpacked. The zip's size and sha256 are checked first, so a
    zip that differs from the lock raises AssetMismatch and nothing is unpacked.
    """
    folder = Path(folder) if folder is not None else assets_dir()
    if entry.extract is None:
        raise AssetError(f"{entry.name} has no extract folder in the lock")
    if not entry.locked:
        raise AssetMissing(f"{entry.name} has no sha256 in the lock yet; run: uv run python -m facemeasure.assets")
    target = folder / entry.extract
    if _marker_matches(target, entry):
        return False
    zip_path = folder / entry.name
    if not zip_path.is_file():
        raise AssetMissing(f"{entry.name} is not in the assets folder; run: uv run python -m facemeasure.assets")
    _check(entry, zip_path)
    tmp = folder / (entry.extract + EXTRACT_SUFFIX)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                _check_member(info.filename, entry.name)
            zf.extractall(tmp)
        with open(tmp / EXTRACTED_MARKER, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(_marker(entry), ensure_ascii=False, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return True


def extract_all(lock: Path | None = None, folder: Path | None = None) -> list[str]:
    """Unpack every locked zip that has an extract folder; return the names unpacked this time."""
    return [e.name for e in read_lock(lock) if e.extract is not None and extract(e, folder)]


def extracted(folder_name: str, lock: Path | None = None, folder: Path | None = None) -> tuple[Path, Entry]:
    """The unpacked folder of the zip whose extract is folder_name, and its lock entry.

    Raises AssetMissing when the folder is not there or was unpacked from another zip.
    Nothing is downloaded or unpacked here.
    """
    entry = next((e for e in read_lock(lock) if e.extract == folder_name), None)
    if entry is None:
        raise AssetMissing(f"no zip in the asset lock unpacks into {folder_name}")
    if not entry.locked:
        raise AssetMissing(f"{entry.name} has no sha256 in the lock yet; run: uv run python -m facemeasure.assets")
    target = (Path(folder) if folder is not None else assets_dir()) / folder_name
    if not _marker_matches(target, entry):
        raise AssetMissing(f"{folder_name} is missing or was unpacked from another zip; run: uv run python -m facemeasure.assets")
    return target, entry


def entry(name: str, lock: Path | None = None) -> Entry:
    """The lock entry of a file name."""
    found = {e.name: e for e in read_lock(lock)}.get(name)
    if found is None:
        raise AssetMissing(f"{name} is not in the asset lock")
    return found


def fingerprint(names: list[str], lock: Path | None = None) -> dict[str, str]:
    """name -> sha256 from the lock, to record which model files made a result."""
    entries = {e.name: e for e in read_lock(lock)}
    return {n: entries[n].sha256 or "" for n in names}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the model files listed in api/assets.lock.json")
    parser.parse_args(argv)
    lock, folder = lock_path(), assets_dir()
    try:
        downloaded, newly_locked = ensure(lock, folder)
        unpacked = extract_all(lock, folder)
    except AssetMismatch as e:
        print(f"File does not match the lock: {e}", file=sys.stderr)
        print(
            "The link may now serve another file (Google's MediaPipe links are 'latest' links), so this file "
            "is not used and the lock is not changed. "
            f"Copy the whole folder {folder} from the machine that made the lock instead.",
            file=sys.stderr,
        )
        return 1
    except (zipfile.BadZipFile, AssetError) as e:
        print(f"Cannot unpack: {e}", file=sys.stderr)
        return 1
    except (requests.RequestException, OSError) as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Check the connection and run the same command again; files already complete are not downloaded again.", file=sys.stderr)
        return 1
    for name in downloaded:
        print(f"Downloaded {name}")
    for name in unpacked:
        print(f"Unpacked {name}")
    if newly_locked:
        print(f"Recorded size and sha256 of {', '.join(newly_locked)} in {lock.name}; commit {lock.name}")
    entries = read_lock(lock)
    print(f"All {len(entries)} files in {folder} match the lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

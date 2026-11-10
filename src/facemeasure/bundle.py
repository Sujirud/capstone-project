"""What the model part hands to the api: api/bundle/ (model.md section 1.4, api.md section 5).

The folder holds manifest.json and heads.npz. The three heads are linear layers, so
their weights are plain numpy arrays; heads.npz is read with allow_pickle=False and
written without pickle, with fixed zip timestamps so the same arrays give the same file.

manifest.json fields:

- bundle_version: semver, reported by the api as versions.model_bundle.
- stub: true while any head is a stub; each head has its own stub too.
- created: ISO 8601 time with a time zone.
- crop_version: must equal crops.CROP_VERSION, or the bundle does not load.
- runtime: Python, mediapipe and torch versions of the machine that made it; a
  difference from this machine gives a BundleWarning.
- assets: sha256 of the model files of api/assets.lock.json used to prepare the data;
  a difference from the lock gives a BundleWarning.
- backbone: name (dinov2_vitb14), image_size (224), normalize (mean and std) and
  features (feature type -> number of dimensions), set by extract_features.py. Other keys are kept.
- train_face_iod_px: {"min", "max"} IOD range of training, or null (no downscaling).
- degrade: {"version", ...} of the webcam-like degradation, or null.
- heads: jaw and nose_base have feature, stub, weights, bias (array names), rmse
  (test set, or null) and band (before the cap of criteria.capped_bands); undereye
  has feature, stub, weights, bias (two values, k = 1 and 2), temperature and
  thresholds {"low", "high"}. Other keys, such as training settings, are kept.
- baseline: jaw and nose_base {"slope", "intercept"}, the line that turns a MediaPipe
  ratio into landmark_value.
- files: sha256 of every other file of the bundle.

Formulas, the same for evaluation (the training scripts) and the api:
ratio head x . w + b; under-eye P(grade >= k) = sigmoid((x . w + b_k) / T).
"""

import copy
import json
import math
import numbers
import platform
import re
import shutil
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy.special import expit

from facemeasure import assets, codes, crops, paths

MANIFEST_NAME = "manifest.json"
HEADS_NAME = "heads.npz"
BACKBONE_NAME = "dinov2_vitb14"
BACKBONE_IMAGE_SIZE = 224
UNDEREYE_HEAD = "undereye"
RATIO_HEADS = (codes.JAW, codes.NOSE_BASE)
HEADS = (*RATIO_HEADS, UNDEREYE_HEAD)
RUNTIME_PACKAGES = ("mediapipe", "torch")
MANIFEST_KEYS = (
    "bundle_version", "stub", "created", "crop_version", "runtime", "assets", "backbone",
    "train_face_iod_px", "degrade", "heads", "baseline", "files",
)
RATIO_HEAD_KEYS = ("feature", "stub", "weights", "bias", "rmse", "band")
UNDEREYE_HEAD_KEYS = ("feature", "stub", "weights", "bias", "temperature", "thresholds")
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
# Files the operating system may leave in a folder someone opened
OS_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}
STUB_COMMAND = "uv run python model/scripts/make_stub_bundle.py"


class BundleError(RuntimeError):
    """The bundle is missing, incomplete, changed, or made for other code."""


class BundleWarning(UserWarning):
    """The bundle was made with other library versions or model files than this machine has."""


# ---------------------------------------------------------------- head formulas


def ridge(x, weights: np.ndarray, bias: float):
    """Ratio head: x . w + b. x is (d,) for one crop (a float) or (n, d) for several ((n,))."""
    out = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(bias)
    return float(out) if np.ndim(out) == 0 else out


def ordinal(x, weights: np.ndarray, biases: np.ndarray, temperature: float) -> np.ndarray:
    """Under-eye head: P(grade >= k) = sigmoid((x . w + b_k) / T), (2,) for one crop or (n, 2)."""
    score = np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
    return expit((np.asarray(score)[..., None] + np.asarray(biases, dtype=np.float64)) / float(temperature))


@dataclass(frozen=True)
class RatioHead:
    feature: str
    stub: bool
    weights: np.ndarray
    bias: float
    rmse: float | None
    band: float

    def predict(self, x):
        return ridge(x, self.weights, self.bias)


@dataclass(frozen=True)
class UndereyeHead:
    feature: str
    stub: bool
    weights: np.ndarray
    bias: np.ndarray  # (2,): b_1 and b_2
    temperature: float
    low: float
    high: float

    def predict(self, x) -> np.ndarray:
        return ordinal(x, self.weights, self.bias, self.temperature)


@dataclass(frozen=True)
class Baseline:
    slope: float
    intercept: float

    def apply(self, value: float) -> float:
        return self.slope * float(value) + self.intercept


@dataclass(frozen=True)
class Bundle:
    manifest: dict
    arrays: dict[str, np.ndarray]
    heads: dict[str, RatioHead | UndereyeHead]
    baseline: dict[str, Baseline]

    @property
    def version(self) -> str:
        return self.manifest["bundle_version"]

    @property
    def stub(self) -> bool:
        return self.manifest["stub"]

    @property
    def stub_heads(self) -> tuple[str, ...]:
        return tuple(name for name in HEADS if self.heads[name].stub)

    @property
    def train_face_iod_px(self) -> dict | None:
        return self.manifest["train_face_iod_px"]


# ---------------------------------------------------------------- checks


def _fail(where: str, message: str):
    raise BundleError(f"{MANIFEST_NAME}: {where} {message}")


def _number(value, where: str, minimum: float | None = None, above: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        _fail(where, f"must be a finite number, got {value!r}")
    if minimum is not None and (value <= minimum if above else value < minimum):
        _fail(where, f"must be {'>' if above else '>='} {minimum}, got {value!r}")
    return float(value)


def _object(value, where: str, keys: tuple[str, ...], exact: bool = False) -> dict:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    missing = [k for k in keys if k not in value]
    extra = sorted(set(value) - set(keys)) if exact else []
    if missing or extra:
        _fail(where, f"is missing {missing} or has unknown keys {extra}")
    return value


def _array(arrays: dict, name, where: str, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(name, str) or name not in arrays:
        _fail(where, f"names array {name!r}, which is not in {HEADS_NAME}")
    a = np.asarray(arrays[name])
    if a.dtype.kind != "f" or a.shape != shape or not np.isfinite(a).all():
        _fail(where, f"array {name!r} must be finite floats of shape {shape}, got {a.dtype} {a.shape}")
    return a.astype(np.float64)


def _ratio_head(raw, where: str, dims: dict, arrays: dict) -> RatioHead:
    raw = _object(raw, where, RATIO_HEAD_KEYS)
    feature, stub = raw["feature"], raw["stub"]
    if feature not in dims:
        _fail(f"{where}.feature", f"{feature!r} is not a feature of the backbone {sorted(dims)}")
    if not isinstance(stub, bool):
        _fail(f"{where}.stub", "must be true or false")
    rmse = None if raw["rmse"] is None else _number(raw["rmse"], f"{where}.rmse", 0.0, above=True)
    return RatioHead(
        feature=feature,
        stub=stub,
        weights=_array(arrays, raw["weights"], f"{where}.weights", (dims[feature],)),
        bias=float(_array(arrays, raw["bias"], f"{where}.bias", ())),
        rmse=rmse,
        band=_number(raw["band"], f"{where}.band", 0.0, above=True),
    )


def _undereye_head(raw, where: str, dims: dict, arrays: dict) -> UndereyeHead:
    raw = _object(raw, where, UNDEREYE_HEAD_KEYS)
    feature, stub = raw["feature"], raw["stub"]
    if feature not in dims:
        _fail(f"{where}.feature", f"{feature!r} is not a feature of the backbone {sorted(dims)}")
    if not isinstance(stub, bool):
        _fail(f"{where}.stub", "must be true or false")
    t = _object(raw["thresholds"], f"{where}.thresholds", ("low", "high"), exact=True)
    low, high = _number(t["low"], f"{where}.thresholds.low", 0.0), _number(t["high"], f"{where}.thresholds.high", 0.0)
    if not low < high <= 1.0:
        _fail(f"{where}.thresholds", f"must have 0 <= low < high <= 1, got {low} and {high}")
    return UndereyeHead(
        feature=feature,
        stub=stub,
        weights=_array(arrays, raw["weights"], f"{where}.weights", (dims[feature],)),
        bias=_array(arrays, raw["bias"], f"{where}.bias", (2,)),
        temperature=_number(raw["temperature"], f"{where}.temperature", 0.0, above=True),
        low=low,
        high=high,
    )


def _backbone(raw) -> dict[str, int]:
    raw = _object(raw, "backbone", ("name", "image_size", "normalize", "features"))
    if raw["name"] != BACKBONE_NAME:
        _fail("backbone.name", f"must be {BACKBONE_NAME} (model.md section 6), got {raw['name']!r}")
    if raw["image_size"] != BACKBONE_IMAGE_SIZE or isinstance(raw["image_size"], bool):
        _fail("backbone.image_size", f"must be {BACKBONE_IMAGE_SIZE}, got {raw['image_size']!r}")
    norm = _object(raw["normalize"], "backbone.normalize", ("mean", "std"))
    for key in ("mean", "std"):
        values = norm[key]
        if not isinstance(values, list) or len(values) != 3:
            _fail(f"backbone.normalize.{key}", "must be a list of 3 numbers")
        for v in values:
            _number(v, f"backbone.normalize.{key}", 0.0, above=key == "std")
    features = raw["features"]
    if not isinstance(features, dict) or not features:
        _fail("backbone.features", "must map each feature type to its number of dimensions")
    for name, dim in features.items():
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            _fail(f"backbone.features.{name}", f"must be a whole number > 0, got {dim!r}")
    return features


def _created(value) -> None:
    try:
        when = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        when = None
    if when is None or when.tzinfo is None:
        _fail("created", f"must be an ISO 8601 time with a time zone, got {value!r}")


def _iod_range(value) -> None:
    if value is None:
        return
    r = _object(value, "train_face_iod_px", ("min", "max"), exact=True)
    low, high = _number(r["min"], "train_face_iod_px.min", 0.0, above=True), _number(r["max"], "train_face_iod_px.max")
    if high < low:
        _fail("train_face_iod_px", f"max {high} is below min {low}")


def _strings(value, where: str, pattern: re.Pattern | None = None) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail(where, "must be an object")
    for k, v in value.items():
        if not isinstance(v, str) or (pattern is not None and not pattern.match(v)):
            _fail(f"{where}.{k}", f"must be {'a sha256' if pattern else 'a text'}, got {v!r}")
    return value


def parse(manifest: dict, arrays: dict[str, np.ndarray]) -> Bundle:
    """Check a manifest and its arrays and build the Bundle (raises BundleError)."""
    m = _object(manifest, "the top level", MANIFEST_KEYS, exact=True)
    if not isinstance(m["bundle_version"], str) or not SEMVER.match(m["bundle_version"]):
        _fail("bundle_version", f"must be a semver such as 1.0.0, got {m['bundle_version']!r}")
    _created(m["created"])
    crop_version = m["crop_version"]
    if isinstance(crop_version, bool) or not isinstance(crop_version, int):
        _fail("crop_version", f"must be a whole number, got {crop_version!r}")
    if crop_version != crops.CROP_VERSION:
        raise BundleError(
            f"the bundle was made with crop_version {crop_version}, the code crops with version "
            f"{crops.CROP_VERSION}; features from other crops do not fit. Extract the features and train the heads "
            f"again, or for development make a stub: {STUB_COMMAND}"
        )
    _object(_strings(m["runtime"], "runtime"), "runtime", ("python", *RUNTIME_PACKAGES), exact=True)
    _strings(m["assets"], "assets", SHA256)
    dims = _backbone(m["backbone"])
    _iod_range(m["train_face_iod_px"])
    if m["degrade"] is not None:
        d = _object(m["degrade"], "degrade", ("version",))
        if isinstance(d["version"], bool) or not isinstance(d["version"], int):
            _fail("degrade.version", f"must be a whole number, got {d['version']!r}")
    files = _strings(m["files"], "files", SHA256)
    if HEADS_NAME not in files or MANIFEST_NAME in files:
        _fail("files", f"must list {HEADS_NAME} and not {MANIFEST_NAME}")

    raw_heads = _object(m["heads"], "heads", HEADS, exact=True)
    heads: dict[str, RatioHead | UndereyeHead] = {
        name: _ratio_head(raw_heads[name], f"heads.{name}", dims, arrays) for name in RATIO_HEADS
    }
    heads[UNDEREYE_HEAD] = _undereye_head(raw_heads[UNDEREYE_HEAD], f"heads.{UNDEREYE_HEAD}", dims, arrays)
    if not isinstance(m["stub"], bool) or m["stub"] != any(h.stub for h in heads.values()):
        _fail("stub", "must be true exactly when some head is a stub")
    used = {raw_heads[name][k] for name in HEADS for k in ("weights", "bias")}
    unused = sorted(set(arrays) - used)
    if unused:
        _fail("heads", f"use no array named {unused}, which {HEADS_NAME} holds")

    raw_base = _object(m["baseline"], "baseline", RATIO_HEADS, exact=True)
    baseline = {}
    for name in RATIO_HEADS:
        b = _object(raw_base[name], f"baseline.{name}", ("slope", "intercept"), exact=True)
        baseline[name] = Baseline(_number(b["slope"], f"baseline.{name}.slope"), _number(b["intercept"], f"baseline.{name}.intercept"))
    return Bundle(manifest=copy.deepcopy(m), arrays={k: np.array(v) for k, v in arrays.items()}, heads=heads, baseline=baseline)


# ---------------------------------------------------------------- this machine


def runtime_versions() -> dict[str, str]:
    """Python, mediapipe and torch versions of this machine, as the manifest records them."""
    return {"python": platform.python_version(), **{name: metadata.version(name) for name in RUNTIME_PACKAGES}}


def asset_hashes(lock: Path | None = None) -> dict[str, str]:
    """name -> sha256 of every locked model file in api/assets.lock.json."""
    out = {}
    for entry in assets.read_lock(lock):
        if not entry.locked:
            raise BundleError(f"{entry.name} has no sha256 in the asset lock yet; run: uv run python -m facemeasure.assets")
        out[entry.name] = entry.sha256
    return out


def compatibility_warnings(b: Bundle, runtime: dict[str, str] | None = None, hashes: dict[str, str] | None = None) -> list[str]:
    """Differences between the bundle's runtime and assets and this machine's."""
    runtime = runtime_versions() if runtime is None else runtime
    out = []
    for name, made in b.manifest["runtime"].items():
        here = runtime.get(name)
        if here != made:
            out.append(f"the bundle was made with {name} {made}, this machine has {here}; points and features may differ")
    if hashes is None:
        try:
            hashes = asset_hashes()
        except (OSError, ValueError, assets.AssetError, BundleError) as e:
            return [*out, f"cannot read the asset lock to compare with the bundle: {e}"]
    if b.manifest["assets"] != hashes:
        out.append(
            "the bundle was made with other model files than api/assets.lock.json lists; "
            "copy api/assets/ and api/assets.lock.json from the machine that made the bundle"
        )
    return out


def new_manifest(
    version: str,
    heads: dict,
    baseline: dict,
    backbone: dict,
    train_face_iod_px: dict | None = None,
    degrade: dict | None = None,
    created: datetime | None = None,
    lock: Path | None = None,
) -> dict:
    """A manifest without files, with stub, created, crop_version, runtime and assets from this machine."""
    created = datetime.now(UTC) if created is None else created
    return {
        "bundle_version": version,
        "stub": any(bool(h.get("stub")) for h in heads.values()),
        "created": created.isoformat(timespec="seconds"),
        "crop_version": crops.CROP_VERSION,
        "runtime": runtime_versions(),
        "assets": asset_hashes(lock),
        "backbone": copy.deepcopy(backbone),
        "train_face_iod_px": copy.deepcopy(train_face_iod_px),
        "degrade": copy.deepcopy(degrade),
        "heads": copy.deepcopy(heads),
        "baseline": copy.deepcopy(baseline),
    }


# ---------------------------------------------------------------- files


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """An .npz that np.load reads, with no pickle and the same bytes for the same arrays."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name in sorted(arrays):
            a = np.array(arrays[name], order="C")  # ascontiguousarray would turn a 0-d array into 1-d
            if a.dtype.hasobject:
                raise BundleError(f"array {name!r} holds Python objects; heads.npz takes numbers only")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_TIME)
            info.create_system = 3
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, a, allow_pickle=False)


def read_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {name: np.array(data[name]) for name in data.files}
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        raise BundleError(f"cannot read {path.name}: {e}") from e


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def bundle_dir() -> Path:
    return paths.path("bundle")


def load_bundle(folder: Path | None = None) -> Bundle:
    """Read and check api/bundle/ (or folder): fields, sha256 of every file and crop_version.

    Differences in runtime and model files only warn (BundleWarning).
    """
    folder = Path(folder) if folder is not None else bundle_dir()
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleError(f"no {MANIFEST_NAME} in {folder}; copy the bundle there or make a stub: {STUB_COMMAND}")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise BundleError(f"cannot read {manifest_path}: {e}") from e
    if not isinstance(manifest, dict):
        raise BundleError(f"{MANIFEST_NAME} must hold one JSON object")
    files = _strings(manifest.get("files"), "files", SHA256)
    for name, sha in files.items():
        path = folder / name
        if Path(name).name != name or not path.is_file():
            raise BundleError(f"{name} of {MANIFEST_NAME} is not in {folder}; copy the whole bundle folder again")
        if assets.file_sha256(path) != sha:
            raise BundleError(
                f"{name} in {folder} does not match the sha256 in {MANIFEST_NAME}: the file was changed or copied "
                "only in part; copy the whole bundle folder again or write the bundle again"
            )
    arrays = read_npz(folder / HEADS_NAME) if HEADS_NAME in files else {}
    b = parse(manifest, arrays)
    for message in compatibility_warnings(b):
        warnings.warn(message, BundleWarning, stacklevel=2)
    return b


def _replaceable(folder: Path) -> bool:
    """True when folder is absent, holds nothing but OS files, or holds a bundle."""
    if not folder.exists():
        return True
    if not folder.is_dir():
        return False
    names = {p.name for p in folder.iterdir()}
    return MANIFEST_NAME in names or names <= OS_FILES


def write_bundle(manifest: dict, arrays: dict[str, np.ndarray], folder: Path | None = None) -> Bundle:
    """Write a bundle to a temporary folder next to folder, check it, then move it into place.

    manifest is everything but files (see new_manifest); write_bundle adds files. An
    existing bundle in folder is replaced; a folder that holds anything else is left
    alone and raises BundleError.
    """
    folder = Path(folder) if folder is not None else bundle_dir()
    if not _replaceable(folder):
        raise BundleError(f"{folder} holds files that are not a bundle; move them away first")
    manifest = {k: copy.deepcopy(v) for k, v in manifest.items() if k != "files"}
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{folder.name}-new-", dir=folder.parent))
    tmp.chmod(0o755)  # mkdtemp makes it private to the user
    old = None
    try:
        write_npz(tmp / HEADS_NAME, arrays)
        manifest["files"] = {HEADS_NAME: assets.file_sha256(tmp / HEADS_NAME)}
        parse(manifest, arrays)
        _write_json(tmp / MANIFEST_NAME, manifest)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BundleWarning)
            load_bundle(tmp)
        if folder.exists():
            old = Path(tempfile.mkdtemp(prefix=f".{folder.name}-old-", dir=folder.parent))
            old.rmdir()
            folder.rename(old)
        tmp.rename(folder)
        tmp = None
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
            if old is not None and not folder.exists():
                old.rename(folder)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BundleWarning)
        return load_bundle(folder)

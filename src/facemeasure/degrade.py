"""Webcam-like copies of a dataset image (model.md sections 3.3 and 6, web.md section 3.1).

Order: mirror if asked -> shrink the whole image so the IOD is the target, with Pillow's
resize, which filters before it shrinks -> Gaussian blur -> Gaussian noise -> JPEG
encode and decode. An image is never enlarged: a face smaller than the target keeps
its size, and Degraded.iod_px says the IOD it then has.

| step | train copies (drawn per copy) | val and test copies (fixed) |
| --- | --- | --- |
| face size | IOD uniform in TRAIN_IOD_MIN..TRAIN_IOD_MAX | copy 1 at TRAIN_IOD_MIN + FIXED_IOD_MARGIN_PX, copy 2 at TRAIN_IOD_MAX |
| blur | sigma 0-0.8 px | 0.4 px |
| noise | sigma 0-4 on 0-255 | 2 |
| JPEG | quality 75-95 | 85 |
| mirror | train copies 2 and 4 of the whole-face crop; never the under-eye | never |

TRAIN_IOD_MIN is min_iod_px of shared/quality.json, read when asked for; TRAIN_IOD_MAX
is 110 px, a 1080p camera at about 0.8 m. Fixed copy 1 sits FIXED_IOD_MARGIN_PX above
min_iod_px: the IOD that analyze_frontal finds on a copy is off by up to about 1.5%, so a
target right on the threshold gave face_too_small to about half of those copies
(DEGRADE_VERSION 1), and val and test keep only copies that pass. The blur is an exact Gaussian (scipy), because
Pillow's GaussianBlur is a box approximation that is poor below 1 px. Noise is drawn
for each channel of each pixel. JPEG uses 4:2:0 chroma subsampling, as browsers do.

The under-eye copies are never mirrored: the two under-eye crops are already turned to
one direction, and mirroring the image would swap the sides of the grades.

Each copy's seed comes from sha256 of "task/image/kind/copy", never from Python's
hash(), which changes every time Python starts; copies are numbered from 1 within
each kind. The api downscales with shrink() too (model_input.py).

The numbers are the team's starting point. Change them only in this file and raise
DEGRADE_VERSION, because features made with other copies do not match.
"""

import hashlib
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from facemeasure import quality
from facemeasure.landmarks import check_rgb

DEGRADE_VERSION = 2
TRAIN_IOD_MAX = 110.0  # px: a 1080p camera at about 0.8 m (web.md section 3.1)
FIXED_IOD_MARGIN_PX = 2.0  # fixed copy 1 above min_iod_px, past the error of the measured IOD
RESAMPLE = Image.Resampling.BICUBIC
TRAIN_BLUR_SIGMA = (0.0, 0.8)  # px
TRAIN_NOISE_SIGMA = (0.0, 4.0)  # on 0-255
TRAIN_JPEG_QUALITY = (75, 95)  # both ends included
FIXED_BLUR_SIGMA = 0.4
FIXED_NOISE_SIGMA = 2.0
FIXED_JPEG_QUALITY = 85
FACE_FLIPPED_COPIES = (2, 4)
JPEG_SUBSAMPLING = 2  # Pillow's code for 4:2:0
BLUR_TRUNCATE = 4.0  # sigmas
SEED_BITS = 63  # fits a signed 64-bit integer, so CSV readers keep it exact

FACE = "face"
UNDEREYE = "undereye"
TASKS = (FACE, UNDEREYE)
TRAIN = "train"
FIXED = "fixed"
KINDS = (TRAIN, FIXED)
FIXED_COPIES = 2  # copy 1 at TRAIN_IOD_MIN + FIXED_IOD_MARGIN_PX, copy 2 at TRAIN_IOD_MAX

_PARAMS_STREAM = 0
_NOISE_STREAM = 1


def __getattr__(name: str):
    if name == "TRAIN_IOD_MIN":
        return float(quality.load_config().frontal.min_iod_px)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def iod_range(config: quality.QualityConfig | None = None) -> tuple[float, float]:
    """(TRAIN_IOD_MIN, TRAIN_IOD_MAX): the face sizes of training, also the bundle's train_face_iod_px."""
    config = config if config is not None else quality.load_config()
    low = float(config.frontal.min_iod_px)
    if not low < TRAIN_IOD_MAX:
        raise ValueError(f"min_iod_px {low} of quality.json is not below TRAIN_IOD_MAX {TRAIN_IOD_MAX}")
    return low, TRAIN_IOD_MAX


def fixed_iod(config: quality.QualityConfig | None = None) -> tuple[float, float]:
    """Target IOD of fixed copies 1 and 2: TRAIN_IOD_MIN + FIXED_IOD_MARGIN_PX and TRAIN_IOD_MAX."""
    low, high = iod_range(config)
    first = low + FIXED_IOD_MARGIN_PX
    if not first < high:
        raise ValueError(f"fixed copy 1 at {first} px is not below TRAIN_IOD_MAX {high}")
    return first, high


@dataclass(frozen=True)
class Params:
    seed: int
    target_iod: float
    blur_sigma: float
    noise_sigma: float
    jpeg_quality: int
    flip: bool


@dataclass(frozen=True)
class Degraded:
    image: np.ndarray  # (h, w, 3) uint8
    scale: float  # new width over old width; 1.0 when the image kept its size
    iod_px: float | None  # the source IOD times scale; None when the source has no IOD


def copy_seed(task: str, image: str, kind: str, copy: int) -> int:
    """Seed of one copy from sha256 of "task/image/kind/copy"; the same on every run and machine."""
    if task not in TASKS or kind not in KINDS or isinstance(copy, bool) or not isinstance(copy, int) or copy < 1:
        raise ValueError(f"bad copy {task!r} {image!r} {kind!r} {copy!r}")
    digest = hashlib.sha256(f"{task}/{image}/{kind}/{copy}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - SEED_BITS)


def train_params(seed: int, flip: bool = False, config: quality.QualityConfig | None = None) -> Params:
    """A train copy: target IOD, blur, noise and JPEG quality drawn from the seed."""
    rng = np.random.default_rng([seed, _PARAMS_STREAM])
    low, high = iod_range(config)
    return Params(
        seed=seed,
        target_iod=float(rng.uniform(low, high)),
        blur_sigma=float(rng.uniform(*TRAIN_BLUR_SIGMA)),
        noise_sigma=float(rng.uniform(*TRAIN_NOISE_SIGMA)),
        jpeg_quality=int(rng.integers(TRAIN_JPEG_QUALITY[0], TRAIN_JPEG_QUALITY[1] + 1)),
        flip=bool(flip),
    )


def fixed_params(seed: int, copy: int, config: quality.QualityConfig | None = None) -> Params:
    """A val or test copy: copy 1 just above TRAIN_IOD_MIN, copy 2 at TRAIN_IOD_MAX, fixed blur, noise and JPEG."""
    if copy not in range(1, FIXED_COPIES + 1):
        raise ValueError(f"fixed copies are numbered 1..{FIXED_COPIES}, got {copy}")
    return Params(
        seed=seed,
        target_iod=fixed_iod(config)[copy - 1],
        blur_sigma=FIXED_BLUR_SIGMA,
        noise_sigma=FIXED_NOISE_SIGMA,
        jpeg_quality=FIXED_JPEG_QUALITY,
        flip=False,
    )


def copy_params(task: str, image: str, kind: str, copy: int, config: quality.QualityConfig | None = None) -> Params:
    """The parameters of one copy, the same every time; what extract_features and later checks use."""
    seed = copy_seed(task, image, kind, copy)
    if kind == FIXED:
        return fixed_params(seed, copy, config)
    return train_params(seed, flip=task == FACE and copy in FACE_FLIPPED_COPIES, config=config)


def shrunk_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(width * scale)), max(1, round(height * scale))


def shrink(image: np.ndarray, scale: float) -> np.ndarray:
    """The whole image made scale times smaller (scale < 1), filtered before it shrinks."""
    if not 0 < scale < 1:
        raise ValueError(f"shrink takes a scale between 0 and 1, got {scale}")
    a = check_rgb(image)
    h, w = a.shape[:2]
    return np.array(Image.fromarray(a).resize(shrunk_size(w, h, scale), RESAMPLE), dtype=np.uint8)


def blur_and_noise(image: np.ndarray, blur_sigma: float, noise_sigma: float, seed: int) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if blur_sigma > 0:
        x = gaussian_filter(x, sigma=(blur_sigma, blur_sigma, 0), mode="nearest", truncate=BLUR_TRUNCATE)
    if noise_sigma > 0:
        x = x + np.random.default_rng([seed, _NOISE_STREAM]).normal(0.0, noise_sigma, x.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def jpeg_round_trip(image: np.ndarray, quality_value: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(check_rgb(image)).save(buf, "JPEG", quality=int(quality_value), subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.array(im.convert("RGB"), dtype=np.uint8)


def degrade(image: np.ndarray, source_iod: float | None, params: Params) -> Degraded:
    """One webcam-like copy of an RGB uint8 image whose face has source_iod px between the eyes.

    source_iod comes from analyze_frontal on the source image; mirroring does not change
    it. Without a face (None) the image keeps its size and gets the other steps.
    """
    a = check_rgb(image)
    if params.flip:
        a = np.ascontiguousarray(a[:, ::-1])
    scale, iod = 1.0, source_iod
    if source_iod is not None and source_iod > params.target_iod:
        width = a.shape[1]
        a = shrink(a, params.target_iod / source_iod)
        scale = a.shape[1] / width
        iod = source_iod * scale
    a = blur_and_noise(a, params.blur_sigma, params.noise_sigma, params.seed)
    a = jpeg_round_trip(a, params.jpeg_quality)
    return Degraded(image=a, scale=scale, iod_px=None if iod is None else float(iod))


def settings(config: quality.QualityConfig | None = None) -> dict:
    """The degradation as data, for data/features/meta.json and the bundle's degrade field."""
    low, high = iod_range(config)
    return {
        "version": DEGRADE_VERSION,
        "order": ["mirror", "shrink", "blur", "noise", "jpeg"],
        "shrink": {"resample": RESAMPLE.name.lower(), "enlarge": False},
        "blur": {"kind": "gaussian", "truncate_sigmas": BLUR_TRUNCATE},
        "noise": {"kind": "gaussian per channel", "scale": "0-255"},
        "jpeg": {"subsampling": "4:2:0"},
        "train": {
            "iod_px": [low, high],
            "blur_sigma_px": list(TRAIN_BLUR_SIGMA),
            "noise_sigma": list(TRAIN_NOISE_SIGMA),
            "jpeg_quality": list(TRAIN_JPEG_QUALITY),
            "flipped_face_copies": list(FACE_FLIPPED_COPIES),
            "flipped_undereye_copies": [],
        },
        "fixed": {
            "iod_px": list(fixed_iod(config)),
            "iod_margin_px": FIXED_IOD_MARGIN_PX,
            "blur_sigma_px": FIXED_BLUR_SIGMA,
            "noise_sigma": FIXED_NOISE_SIGMA,
            "jpeg_quality": FIXED_JPEG_QUALITY,
            "flipped": False,
        },
        "seed": "first 63 bits of sha256('task/image/kind/copy'); copies numbered from 1 within each kind",
    }

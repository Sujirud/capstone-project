"""Frozen DINOv2 ViT-B/14 on the 224 x 224 crops (model.md sections 2.3 and 6, api.md section 3).

The code comes from the zip of api/assets.lock.json unpacked in api/assets/dinov2_src/
and the weights from api/assets/dinov2_vitb14_pretrain.pth; nothing is downloaded. The
model is built with dinov2.hub.backbones.dinov2_vitb14(pretrained=False), never through
torch.hub (it downloads) or hubconf.py (at this commit it imports modules that need
more dependencies). xFormers is turned off so every machine runs the same attention.

Input: RGB uint8 crops from crops.py, divided by 255 and normalized with the ImageNet
mean and std. forward_features gives the class token (768) and 16 x 16 patch tokens
(256 x 768); one run gives every feature type:

- cls: the class token, 768 dimensions.
- cls_mean: the class token, then the mean of the patch tokens, 1,536 dimensions.
- grid4: the mean patch token of each cell of a 4 x 4 grid (4 x 4 patches per cell),
  ordered by row, then column, then channel, 12,288 dimensions.

A batch of crops runs together, so the api runs the face crop and both under-eye crops
at once (api.md section 3 step 6). Backbone.info is what the bundle records.
"""

import io
import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from facemeasure import assets, crops
from facemeasure.bundle import BACKBONE_IMAGE_SIZE, BACKBONE_NAME
from facemeasure.device import pick_device

WEIGHTS_FILE = "dinov2_vitb14_pretrain.pth"
SOURCE_FOLDER = "dinov2_src"
PACKAGE = "dinov2"
IMAGE_SIZE = BACKBONE_IMAGE_SIZE
PATCH_SIZE = 14
GRID = IMAGE_SIZE // PATCH_SIZE  # 16 patches per side
EMBED_DIM = 768
GRID4 = 4  # cells per side of grid4
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FEATURES = {"cls": EMBED_DIM, "cls_mean": 2 * EMBED_DIM, "grid4": GRID4 * GRID4 * EMBED_DIM}
FEATURE_TYPES = tuple(FEATURES)
BATCH_ACCELERATOR = 32  # CUDA or MPS
BATCH_CPU = 8

if crops.CROP_SIZE != IMAGE_SIZE:
    raise RuntimeError(f"crops.CROP_SIZE {crops.CROP_SIZE} is not the backbone image size {IMAGE_SIZE}")


class BackboneError(RuntimeError):
    """The DINOv2 code or weights cannot be used."""


def batch_size(device: torch.device) -> int:
    return BATCH_CPU if device.type == "cpu" else BATCH_ACCELERATOR


def source_root(lock: Path | None = None, folder: Path | None = None) -> Path:
    """The folder under api/assets/dinov2_src/ that holds the dinov2 package."""
    target, _ = assets.extracted(SOURCE_FOLDER, lock, folder)
    found = sorted(p.parent.parent for p in target.glob(f"*/{PACKAGE}/__init__.py"))
    if len(found) != 1:
        raise BackboneError(f"expected one {PACKAGE} package in {target}, found {len(found)}")
    return found[0]


def _hub_backbones(root: Path):
    """dinov2.hub.backbones from root, after putting root on sys.path."""
    loaded = sys.modules.get(PACKAGE)
    if loaded is not None:
        where = Path(getattr(loaded, "__file__", "") or "").resolve().parent.parent
        if where != root.resolve():
            raise BackboneError(f"a {PACKAGE} package from {where} is already imported, not the one in {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ["XFORMERS_DISABLED"] = "1"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="xFormers is")
        from dinov2.hub import backbones
        from dinov2.models import vision_transformer  # noqa: F401  backbones imports it lazily; load it here, warnings off
    return backbones


def normalize(images) -> torch.Tensor:
    """(n, 224, 224, 3) uint8 RGB -> (n, 3, 224, 224) float32, / 255 then ImageNet mean and std."""
    a = np.asarray(images)
    if a.ndim == 3:
        a = a[None]
    if a.dtype != np.uint8 or a.ndim != 4 or a.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"crops must be ({IMAGE_SIZE}, {IMAGE_SIZE}, 3) uint8 RGB, got {a.dtype} {a.shape}")
    x = a.astype(np.float32) / np.float32(255)
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def pool(cls: torch.Tensor, patches: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every feature type from the class token (n, 768) and the patch tokens (n, 256, 768).

    The patch tokens are in DINOv2's order, row by row of the 16 x 16 grid.
    """
    n, count, dim = patches.shape
    if count != GRID * GRID or cls.shape != (n, dim):
        raise ValueError(f"expected ({n}, {dim}) and ({n}, {GRID * GRID}, {dim}), got {tuple(cls.shape)} and {tuple(patches.shape)}")
    side = GRID // GRID4
    cells = patches.reshape(n, GRID4, side, GRID4, side, dim).mean(dim=(2, 4))  # (n, row, col, dim)
    return {
        "cls": cls,
        "cls_mean": torch.cat([cls, patches.mean(dim=1)], dim=1),
        "grid4": cells.reshape(n, GRID4 * GRID4 * dim),
    }


class Backbone:
    """DINOv2 ViT-B/14 in eval mode, without gradients, on pick_device() unless told otherwise."""

    def __init__(self, device: torch.device | str | None = None, lock: Path | None = None, folder: Path | None = None):
        self.device = torch.device(device) if device is not None else pick_device()
        root = source_root(lock, folder)
        backbones = _hub_backbones(root)
        source = next(e for e in assets.read_lock(lock) if e.extract == SOURCE_FOLDER)
        weights = assets.entry(WEIGHTS_FILE, lock)
        state = torch.load(
            io.BytesIO(assets.read_verified(WEIGHTS_FILE, lock, folder)), map_location="cpu", weights_only=True
        )
        with torch.random.fork_rng(devices=[]):  # building the model draws random weights; keep the global seed
            model = backbones.dinov2_vitb14(pretrained=False)
        model.load_state_dict(state, strict=True)
        del state
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)
        self.info = {
            "name": BACKBONE_NAME,
            "image_size": IMAGE_SIZE,
            "normalize": {"mean": list(MEAN), "std": list(STD)},
            "features": dict(FEATURES),
            "commit": source.commit,
            "weights_sha256": weights.sha256,
        }

    @property
    def batch_size(self) -> int:
        return batch_size(self.device)

    def tokens(self, images) -> tuple[torch.Tensor, torch.Tensor]:
        """Class token and patch tokens of one batch, on the device."""
        x = normalize(images).to(self.device)
        with torch.inference_mode():
            out = self.model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    def features(self, images: Sequence[np.ndarray] | np.ndarray, batch: int | None = None) -> dict[str, np.ndarray]:
        """Every feature type as float32 arrays (n, dim), run in batches of batch (default by device)."""
        a = np.asarray(images)
        if a.ndim == 3:
            a = a[None]
        size = batch or self.batch_size
        parts: dict[str, list[np.ndarray]] = {k: [] for k in FEATURE_TYPES}
        for start in range(0, len(a), size):
            cls, patches = self.tokens(a[start : start + size])
            with torch.inference_mode():
                pooled = pool(cls.float(), patches.float())
            for k in FEATURE_TYPES:
                parts[k].append(pooled[k].cpu().numpy().astype(np.float32, copy=False))
        if not len(a):
            return {k: np.zeros((0, d), dtype=np.float32) for k, d in FEATURES.items()}
        return {k: np.concatenate(v) for k, v in parts.items()}

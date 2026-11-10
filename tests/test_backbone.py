"""facemeasure.backbone: pooling and normalization on synthetic tokens; the real DINOv2 when its files are there."""

import numpy as np
import pytest
import torch

from facemeasure import assets, backbone, bundle


def test_grid4_means_each_cell_in_row_column_channel_order():
    n, dim = 2, 5
    k = torch.arange(backbone.GRID * backbone.GRID, dtype=torch.float32)  # patch number, row by row
    patches = k[None, :, None].expand(n, -1, dim).clone()
    patches[1] *= -1
    cls = torch.full((n, dim), 7.0)
    out = backbone.pool(cls, patches)
    grid = out["grid4"]
    assert grid.shape == (n, 16 * dim)
    for row in range(4):
        for col in range(4):
            # mean of rows 4*row..4*row+3 and columns 4*col..4*col+3 of row*16 + col
            want = (4 * row + 1.5) * 16 + (4 * col + 1.5)
            cell = grid[0, (row * 4 + col) * dim : (row * 4 + col + 1) * dim]
            assert torch.allclose(cell, torch.full((dim,), want))
            assert torch.allclose(grid[1, (row * 4 + col) * dim], torch.tensor(-want))
    assert torch.allclose(out["cls"], cls)
    assert torch.allclose(out["cls_mean"][0], torch.cat([torch.full((dim,), 7.0), torch.full((dim,), 127.5)]))


def test_channels_stay_apart_in_grid4():
    dim = 3
    patches = torch.zeros(1, 256, dim)
    patches[0, :, 1] = 1.0  # only channel 1 is set
    grid = backbone.pool(torch.zeros(1, dim), patches)["grid4"].reshape(16, dim)
    assert torch.equal(grid[:, 1], torch.ones(16)) and torch.equal(grid[:, [0, 2]], torch.zeros(16, 2))


def test_feature_sizes():
    assert backbone.FEATURES == {"cls": 768, "cls_mean": 1536, "grid4": 12288}
    assert backbone.IMAGE_SIZE == bundle.BACKBONE_IMAGE_SIZE == 224 and backbone.GRID == 16


def test_normalize():
    crops = np.zeros((2, 224, 224, 3), dtype=np.uint8)
    crops[1] = 255
    x = backbone.normalize(crops)
    assert x.shape == (2, 3, 224, 224) and x.dtype == torch.float32
    for c in range(3):
        assert x[0, c, 0, 0].item() == pytest.approx(-backbone.MEAN[c] / backbone.STD[c], abs=1e-6)
        assert x[1, c, 5, 9].item() == pytest.approx((1 - backbone.MEAN[c]) / backbone.STD[c], abs=1e-6)
    assert backbone.normalize(crops[0]).shape == (1, 3, 224, 224)
    for bad in (crops.astype(np.float32), crops[:, :100], crops[..., :1]):
        with pytest.raises(ValueError):
            backbone.normalize(bad)


def test_batch_sizes():
    assert backbone.batch_size(torch.device("cpu")) == 8
    assert backbone.batch_size(torch.device("mps")) == backbone.batch_size(torch.device("cuda")) == 32


# ---------------------------------------------------------------- the real model


def _have_files() -> bool:
    try:
        backbone.source_root()
        assets.asset_path(backbone.WEIGHTS_FILE)
        return True
    except assets.AssetError:
        return False


needs_files = pytest.mark.skipif(not _have_files(), reason="DINOv2 files are not in api/assets/; run: uv run python -m facemeasure.assets")


def smooth_crops(n: int) -> np.ndarray:
    y, x = np.mgrid[0:224, 0:224] / 224.0
    out = []
    for k in range(n):
        img = np.stack([np.sin((k + 1) * 3 * x + y), np.cos(5 * y - k * x), x * y * (k % 3)], axis=-1)
        out.append(((img - img.min()) / (np.ptp(img) + 1e-9) * 255).astype(np.uint8))
    return np.stack(out)


@pytest.fixture(scope="module")
def models():
    device = backbone.Backbone()  # loads with load_state_dict(strict=True)
    cpu = device if device.device.type == "cpu" else backbone.Backbone("cpu")
    return device, cpu


@needs_files
def test_real_model_shapes_and_info(models):
    device, _ = models
    out = device.features(smooth_crops(3))
    assert {k: v.shape for k, v in out.items()} == {"cls": (3, 768), "cls_mean": (3, 1536), "grid4": (3, 12288)}
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in out.values())
    assert np.allclose(out["cls_mean"][:, :768], out["cls"])
    info = device.info
    lock = {e.name: e for e in assets.read_lock()}
    assert info["name"] == "dinov2_vitb14" and info["features"] == backbone.FEATURES
    assert info["weights_sha256"] == lock[backbone.WEIGHTS_FILE].sha256 and len(info["commit"]) == 40
    assert not any(p.requires_grad for p in device.model.parameters()) and not device.model.training


@needs_files
def test_real_model_same_crop_same_feature_and_cpu_matches_device(models):
    device, cpu = models
    crops = smooth_crops(9)
    a, again, b = device.features(crops), device.features(crops), device.features(crops[::-1])
    for k in a:
        assert np.array_equal(a[k], again[k])
        # Another batch order and size: the same up to float rounding of the kernels
        assert np.allclose(a[k], b[k][::-1], rtol=1e-4, atol=1e-4)
    c = cpu.features(crops)
    for k in a:
        cos = (a[k] * c[k]).sum(1) / (np.linalg.norm(a[k], axis=1) * np.linalg.norm(c[k], axis=1))
        assert cos.min() >= 0.999, k

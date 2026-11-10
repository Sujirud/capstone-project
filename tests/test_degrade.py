"""facemeasure.degrade on synthetic images."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import ndimage

from facemeasure import degrade, quality

CONFIG = quality.load_config()
LOW, HIGH = degrade.iod_range(CONFIG)


def two_eyes(width=640, height=520, iod=240.0, radius=9.0) -> np.ndarray:
    """A light gray image with two dark discs whose centers are iod apart, and some texture."""
    y, x = np.mgrid[0:height, 0:width] + 0.5
    img = np.full((height, width, 3), 200.0)
    img += 20 * np.sin(x / 7.0)[..., None]  # texture that changes with the blur and the JPEG
    cx, cy = width / 2, height / 2
    for ex in (cx - iod / 2, cx + iod / 2):
        img[(x - ex) ** 2 + (y - cy) ** 2 <= radius**2] = 20
    return np.clip(img, 0, 255).astype(np.uint8)


def measured_iod(image: np.ndarray) -> float:
    """Distance between the centroids of the two dark discs."""
    dark = image.mean(axis=2) < 110
    labels, n = ndimage.label(dark)
    assert n == 2
    weights = 255 - image.mean(axis=2)
    (y1, x1), (y2, x2) = ndimage.center_of_mass(weights * dark, labels, [1, 2])
    return float(np.hypot(x2 - x1, y2 - y1))


def params(target, blur=degrade.FIXED_BLUR_SIGMA, noise=degrade.FIXED_NOISE_SIGMA, jpeg=degrade.FIXED_JPEG_QUALITY, seed=7, flip=False):
    return degrade.Params(seed=seed, target_iod=target, blur_sigma=blur, noise_sigma=noise, jpeg_quality=jpeg, flip=flip)


def test_seed_is_fixed_and_matches_the_written_value():
    assert degrade.copy_seed("face", "AF1.jpg", "train", 1) == 8685850577127609211
    assert degrade.copy_seed("undereye", "48264.png", "fixed", 2) == 6292632747423789760
    assert degrade.copy_seed("face", "AF1.jpg", "train", 1) == degrade.copy_seed("face", "AF1.jpg", "train", 1)
    seeds = {degrade.copy_seed(t, i, k, c) for t in degrade.TASKS for i in ("AF1.jpg", "AF2.jpg") for k in degrade.KINDS for c in (1, 2)}
    assert len(seeds) == 16 and all(0 <= s < 2**63 for s in seeds)


@pytest.mark.parametrize("bad", [("face", "a.jpg", "train", 0), ("faces", "a.jpg", "train", 1), ("face", "a.jpg", "random", 1)])
def test_bad_copy_names_are_refused(bad):
    with pytest.raises(ValueError):
        degrade.copy_seed(*bad)


def test_same_seed_same_bytes_other_seed_other_image():
    img = two_eyes()
    a = degrade.degrade(img, 240.0, degrade.copy_params("face", "AF1.jpg", "train", 1, CONFIG))
    b = degrade.degrade(img, 240.0, degrade.copy_params("face", "AF1.jpg", "train", 1, CONFIG))
    c = degrade.degrade(img, 240.0, degrade.copy_params("face", "AF1.jpg", "train", 3, CONFIG))
    assert a.image.tobytes() == b.image.tobytes() and a.scale == b.scale
    assert a.image.shape != c.image.shape or a.image.tobytes() != c.image.tobytes()
    # Only the seed differs: the noise must differ too
    p = params(90.0)
    d1 = degrade.degrade(img, 240.0, p).image
    d2 = degrade.degrade(img, 240.0, params(90.0, seed=8)).image
    assert d1.shape == d2.shape and not np.array_equal(d1, d2)


@pytest.mark.parametrize("target", [LOW, 92.5, HIGH])
def test_iod_reaches_the_target_within_one_percent(target):
    out = degrade.degrade(two_eyes(iod=240.0), 240.0, params(target))
    assert out.scale < 1 and out.iod_px == pytest.approx(target, rel=0.005)
    assert measured_iod(out.image) == pytest.approx(target, rel=0.01)
    assert out.image.shape[1] == round(640 * target / 240.0)


def test_never_enlarges():
    img = two_eyes(iod=60.0, width=300, height=200)
    out = degrade.degrade(img, 60.0, params(HIGH))
    assert out.image.shape == img.shape and out.scale == 1.0 and out.iod_px == 60.0
    assert measured_iod(out.image) == pytest.approx(60.0, rel=0.01)
    no_face = degrade.degrade(img, None, params(HIGH))
    assert no_face.image.shape == img.shape and no_face.iod_px is None


def test_flip_mirrors_before_the_other_steps():
    img = two_eyes()
    img[10:40, 10:60] = (255, 0, 0)  # something on the image left
    flipped = degrade.degrade(img, 240.0, params(100.0, flip=True)).image
    mirrored = degrade.degrade(np.ascontiguousarray(img[:, ::-1]), 240.0, params(100.0)).image
    assert np.array_equal(flipped, mirrored)


def test_shrink_filters_and_refuses_to_enlarge():
    stripes = np.zeros((64, 64, 3), dtype=np.uint8)
    stripes[:, ::2] = 255  # one-pixel stripes alias without a filter
    small = degrade.shrink(stripes, 0.5)
    assert small.shape == (32, 32, 3) and np.abs(small.astype(int) - 128).max() < 20
    for bad in (1.0, 1.5, 0.0):
        with pytest.raises(ValueError):
            degrade.shrink(stripes, bad)


def test_copy_params_follow_the_table():
    assert degrade.TRAIN_IOD_MIN == CONFIG.frontal.min_iod_px == LOW and degrade.TRAIN_IOD_MAX == HIGH == 110
    for copy in range(1, 7):
        p = degrade.copy_params("face", "AM5.jpg", "train", copy, CONFIG)
        assert p.flip == (copy in (2, 4))
        assert LOW <= p.target_iod <= HIGH and 0 <= p.blur_sigma <= 0.8 and 0 <= p.noise_sigma <= 4 and 75 <= p.jpeg_quality <= 95
        assert not degrade.copy_params("undereye", "AM5.jpg", "train", copy, CONFIG).flip
    for task in degrade.TASKS:
        lo, hi = (degrade.copy_params(task, "AM5.jpg", "fixed", c, CONFIG) for c in (1, 2))
        # Copy 1 sits 2 px above min_iod_px, past the error of the IOD measured on the copy
        assert (lo.target_iod, hi.target_iod) == (LOW + 2, HIGH) == degrade.fixed_iod(CONFIG)
        assert (lo.blur_sigma, lo.noise_sigma, lo.jpeg_quality, lo.flip) == (0.4, 2.0, 85, False)
    with pytest.raises(ValueError):
        degrade.copy_params("face", "AM5.jpg", "fixed", 3, CONFIG)


def test_train_draws_cover_the_ranges():
    ps = [degrade.train_params(degrade.copy_seed("face", f"AF{k}.jpg", "train", 1), config=CONFIG) for k in range(400)]
    iods = np.array([p.target_iod for p in ps])
    assert iods.min() < LOW + 3 and iods.max() > HIGH - 3
    assert {p.jpeg_quality for p in ps} == set(range(75, 96))


def test_settings_record_the_table():
    s = degrade.settings(CONFIG)
    assert s["version"] == degrade.DEGRADE_VERSION
    assert s["train"]["iod_px"] == [LOW, HIGH] and s["fixed"]["jpeg_quality"] == 85
    assert s["fixed"]["iod_px"] == [LOW + 2, HIGH] and s["fixed"]["iod_margin_px"] == degrade.FIXED_IOD_MARGIN_PX == 2
    assert s["train"]["flipped_face_copies"] == [2, 4] and s["train"]["flipped_undereye_copies"] == []


def test_fixed_copy_1_must_stay_below_the_maximum():
    config = replace(CONFIG, frontal=replace(CONFIG.frontal, min_iod_px=108.5))  # 108.5 + 2 is past 110
    with pytest.raises(ValueError, match="fixed copy 1"):
        degrade.fixed_iod(config)

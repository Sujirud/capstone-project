"""facemeasure.image_io on images made in the tests."""

import io

import numpy as np
import pytest
from PIL import Image

from facemeasure import image_io

W, H = 64, 40
BLOCK = (8, 6)  # top-left corner of a red 8x8 block in the stored image
ORIENTATION_TAG = 0x0112


def stored_image() -> Image.Image:
    a = np.full((H, W, 3), 40, dtype=np.uint8)
    x, y = BLOCK
    a[y : y + 8, x : x + 8] = (230, 20, 20)
    return Image.fromarray(a)


def jpeg_bytes(im: Image.Image, orientation: int | None = None, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION_TAG] = orientation
    im.save(buf, "JPEG", quality=quality, exif=exif.tobytes())
    return buf.getvalue()


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def block_center_after(orientation: int) -> tuple[tuple[int, int], tuple[float, float]]:
    """(width, height) and the (x, y) center of the red block once the image is upright."""
    cx, cy = BLOCK[0] + 4, BLOCK[1] + 4  # continuous center of the block
    if orientation == 1:
        return (W, H), (cx, cy)
    if orientation == 3:  # stored upside down
        return (W, H), (W - cx, H - cy)
    if orientation == 6:  # shown after turning 90 degrees clockwise
        return (H, W), (H - cy, cx)
    if orientation == 8:  # shown after turning 90 degrees counter-clockwise
        return (H, W), (cy, W - cx)
    raise ValueError(orientation)


def red_center(a: np.ndarray) -> tuple[float, float]:
    mask = (a[:, :, 0] > 150) & (a[:, :, 1] < 100)
    ys, xs = np.nonzero(mask)
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_jpeg_turned_upright_by_exif(orientation):
    a = image_io.read_image(jpeg_bytes(stored_image(), orientation))
    (w, h), (x, y) = block_center_after(orientation)
    assert a.dtype == np.uint8 and a.shape == (h, w, 3)
    rx, ry = red_center(a)
    assert abs(rx - x) <= 1 and abs(ry - y) <= 1


def test_reads_path_and_ignores_the_file_name(tmp_path):
    path = tmp_path / "photo.jpg"  # PNG bytes under a .jpg name
    path.write_bytes(png_bytes(stored_image()))
    a = image_io.read_image(path)
    assert a.shape == (H, W, 3)
    assert tuple(a[BLOCK[1] + 2, BLOCK[0] + 2]) == (230, 20, 20)


def test_png_alpha_goes_on_white():
    a = np.zeros((10, 10, 4), dtype=np.uint8)
    a[:, :5] = (200, 0, 0, 0)  # fully transparent
    a[:, 5:] = (255, 0, 0, 128)  # half red
    out = image_io.read_image(png_bytes(Image.fromarray(a, "RGBA")))
    assert out.shape == (10, 10, 3)
    assert tuple(out[0, 0]) == (255, 255, 255)
    r, g, b = (int(v) for v in out[0, 9])
    assert r == 255 and abs(g - 127) <= 2 and abs(b - 127) <= 2


def test_palette_png_with_transparency_goes_on_white():
    im = Image.new("P", (4, 4), 0)
    im.putpalette([10, 20, 30] + [0, 0, 0] * 255)
    buf = io.BytesIO()
    im.save(buf, "PNG", transparency=0)
    assert tuple(image_io.read_image(buf.getvalue())[0, 0]) == (255, 255, 255)


def test_gray_becomes_rgb():
    gray = Image.fromarray(np.arange(100, dtype=np.uint8).reshape(10, 10), "L")
    for data in (png_bytes(gray), jpeg_bytes(gray)):
        out = image_io.read_image(data)
        assert out.shape == (10, 10, 3)
        assert (out[:, :, 0] == out[:, :, 1]).all() and (out[:, :, 1] == out[:, :, 2]).all()


def test_16_bit_gray_png_scales_to_8_bit():
    a = np.full((4, 4), 65535, dtype=np.uint16)
    a[0, 0] = 0
    out = image_io.read_image(png_bytes(Image.fromarray(a)))
    assert out.shape == (4, 4, 3) and out[1, 1, 0] == 255 and out[0, 0, 0] == 0


def test_bytes_that_are_not_an_image():
    with pytest.raises(image_io.UnsupportedImageType):
        image_io.read_image(b"hello, this is not a picture at all")
    with pytest.raises(image_io.UnsupportedImageType):
        image_io.read_image(b"")


def test_gif_is_not_accepted():
    buf = io.BytesIO()
    stored_image().save(buf, "GIF")
    with pytest.raises(image_io.UnsupportedImageType):
        image_io.read_image(buf.getvalue())


def test_jpeg_cut_short_does_not_decode():
    data = jpeg_bytes(Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)))
    with pytest.raises(image_io.ImageDecodeError):
        image_io.read_image(data[: len(data) // 2])


def test_png_with_a_broken_header_does_not_decode():
    with pytest.raises(image_io.ImageDecodeError):
        image_io.read_image(image_io.PNG_MAGIC + b"\x00" * 40)


def test_pixel_limit():
    data = png_bytes(stored_image())
    assert image_io.read_image(data, max_pixels=W * H).shape == (H, W, 3)
    with pytest.raises(image_io.ImageTooLarge):
        image_io.read_image(data, max_pixels=W * H - 1)


def test_errors_share_a_base_class():
    for cls in (image_io.UnsupportedImageType, image_io.ImageDecodeError, image_io.ImageTooLarge):
        assert issubclass(cls, image_io.ImageReadError)
    assert image_io.DEFAULT_MAX_PIXELS == 50_000_000

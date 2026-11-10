"""Read an uploaded or dataset image into an RGB uint8 array (api.md sections 2.1, 3 and 6).

Only JPEG and PNG are accepted, judged from the first bytes, never from the file name.
The image is turned upright by its EXIF orientation and every other EXIF field is
dropped, because it may hold GPS data. Alpha is put on white and gray becomes RGB.

The three errors map to the HTTP codes of api.md section 2.3 in the api:
UnsupportedImageType -> 415, ImageDecodeError -> 400, ImageTooLarge -> 413.
"""

import io
import struct
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_MAX_PIXELS = 50_000_000
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
WHITE = (255, 255, 255, 255)
_PIL_FORMAT = {"jpeg": "JPEG", "png": "PNG"}
_DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, EOFError, struct.error)


class ImageReadError(ValueError):
    """The input cannot be used as an image."""


class UnsupportedImageType(ImageReadError):
    """Neither JPEG nor PNG."""


class ImageDecodeError(ImageReadError):
    """JPEG or PNG bytes that do not decode, for example a file cut short."""


class ImageTooLarge(ImageReadError):
    """More pixels than the limit."""


def sniff(data: bytes) -> str | None:
    """'jpeg' or 'png' from the first bytes, else None."""
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(PNG_MAGIC):
        return "png"
    return None


def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode.startswith("I"):  # 16-bit gray PNG
        values = np.clip(np.asarray(im, dtype=np.int64), 0, 65535) >> 8
        im = Image.fromarray(values.astype(np.uint8), "L")
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        background = Image.new("RGBA", rgba.size, WHITE)
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return im.convert("RGB")


def read_image(source: bytes | bytearray | memoryview | str | Path, max_pixels: int = DEFAULT_MAX_PIXELS) -> np.ndarray:
    """Upright (h, w, 3) RGB uint8 array from JPEG or PNG bytes or a file path."""
    data = bytes(source) if isinstance(source, bytes | bytearray | memoryview) else Path(source).read_bytes()
    kind = sniff(data)
    if kind is None:
        raise UnsupportedImageType("only JPEG and PNG images are accepted")
    try:
        with warnings.catch_warnings():
            # The pixel limit below replaces Pillow's own decompression bomb warning
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im = Image.open(io.BytesIO(data), formats=[_PIL_FORMAT[kind]])
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(f"more than {max_pixels} pixels") from e
    except _DECODE_ERRORS as e:
        raise ImageDecodeError(f"the {kind.upper()} data does not decode") from e
    with im:
        width, height = im.size
        if width * height > max_pixels:
            raise ImageTooLarge(f"{width}x{height} is more than {max_pixels} pixels")
        try:
            im.load()
            upright = ImageOps.exif_transpose(im)
            rgb = _to_rgb(upright)
        except _DECODE_ERRORS as e:
            raise ImageDecodeError(f"the {kind.upper()} data does not decode") from e
    return np.array(rgb, dtype=np.uint8)

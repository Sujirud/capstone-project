"""Draw measurement points over face images for visual checks.

Colors follow tools/scut_landmark_preview.py: red = inner canthi and alar base,
blue = face edge, yellow = Zy, green = Go. On profiles, blue = front edge, yellow =
the Legan angle lines and green = the nasofacial angle lines. Images with faces stay
in data/outputs/, and drawings over FEI photos must never leave it.
"""

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from facemeasure.face_geometry import FrontalPoints, JawNose
from facemeasure.profile import ProfileResult

RED = (255, 60, 60)
BLUE = (60, 150, 255)
YELLOW = (255, 215, 0)
GREEN = (60, 220, 90)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (170, 170, 170)
SHEET_BACKGROUND = (30, 30, 30)

Color = tuple[int, int, int]


def load_font(size: int = 14) -> ImageFont.ImageFont:
    """Pillow's built-in font at size, or its fixed bitmap font when FreeType is missing."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_jaw_nose(
    image: Image.Image,
    frontal: FrontalPoints,
    result: JawNose,
    *,
    points: np.ndarray | None = None,
    point_colors: Sequence[Color] | None = None,
    point_labels: Sequence[str] | None = None,
    label: str | None = None,
    scale: int = 1,
    font_size: int = 14,
    alar_line: bool = True,
) -> Image.Image:
    """Return a copy of image, enlarged scale times, with the jaw and nose measurement drawn on it.

    Lines join the inner canthi and the alar base points (red), Zy (yellow) and Go
    (green); circles mark Zy and Go. points are dots on top, colored by point_colors
    and numbered by point_labels when given. Without points, the face edges are blue
    dots and the canthi and alar base points red dots. label goes at the top left.
    alar_line=False leaves out the alar base line, for when the alar points are not chosen yet.
    """
    im = image.convert("RGB")
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    font = load_font(font_size)

    def s(p) -> tuple[float, float]:
        return (float(p[0]) * scale, float(p[1]) * scale)

    d.line((*s(frontal.canthus_l), *s(frontal.canthus_r)), fill=RED, width=2)
    if alar_line:
        d.line((*s(frontal.alar_l), *s(frontal.alar_r)), fill=RED, width=2)
    d.line((*s(result.zy_l), *s(result.zy_r)), fill=YELLOW, width=2)
    d.line((*s(result.go_l), *s(result.go_r)), fill=GREEN, width=2)
    for p, color in ((result.zy_l, YELLOW), (result.zy_r, YELLOW), (result.go_l, GREEN), (result.go_r, GREEN)):
        x, y = s(p)
        d.ellipse((x - 7, y - 7, x + 7, y + 7), outline=color, width=3)

    if points is None:
        reds = [frontal.canthus_l, frontal.canthus_r, frontal.alar_l, frontal.alar_r]
        points = np.concatenate([frontal.contour_l, frontal.contour_r, np.array(reds)])
        point_colors = [BLUE] * (len(frontal.contour_l) + len(frontal.contour_r)) + [RED] * len(reds)
    points = np.asarray(points, dtype=np.float64)
    colors = list(point_colors) if point_colors is not None else [WHITE] * len(points)
    if len(colors) != len(points) or (point_labels is not None and len(point_labels) != len(points)):
        raise ValueError("points, point_colors and point_labels must have the same length")
    for i, p in enumerate(points):
        x, y = s(p)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[i])
        if point_labels is not None:
            d.text((x + 4, y - 8), point_labels[i], fill=colors[i], font=font, stroke_width=2, stroke_fill=BLACK)
    if label:
        d.text((8, 8), label, fill=WHITE, font=font, stroke_width=2, stroke_fill=BLACK)
    return im


def draw_polylines(
    image: Image.Image,
    lines: Sequence[np.ndarray],
    colors: Sequence[Color],
    *,
    width: int = 2,
    scale: float = 1.0,
    closed: bool = True,
) -> Image.Image:
    """Draw polylines, such as crop boxes (closed) or face edges (open), on image in place and return it.

    Each line is an (n, 2) array in image coordinates, multiplied by scale before drawing.
    """
    if len(lines) != len(colors):
        raise ValueError("need one color per line")
    d = ImageDraw.Draw(image)
    for line, color in zip(lines, colors, strict=True):
        p = [(float(x) * scale, float(y) * scale) for x, y in np.asarray(line, dtype=np.float64)]
        if len(p) < 2:
            continue
        d.line([*p, p[0]] if closed else p, fill=color, width=width, joint="curve")
    return image


def contact_sheet(
    tiles: Sequence[Image.Image], captions: Sequence[str], columns: int = 8, font_size: int = 13
) -> Image.Image:
    """Tiles in a grid, each with its caption (one or more lines) underneath."""
    if not tiles or len(tiles) != len(captions):
        raise ValueError("need at least one tile and one caption per tile")
    width = max(t.width for t in tiles)
    height = max(t.height for t in tiles)
    font = load_font(font_size)
    lines = max(len(c.splitlines()) for c in captions)
    caption_height = lines * (font_size + 4) + 8
    columns = min(columns, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + caption_height)), SHEET_BACKGROUND)
    d = ImageDraw.Draw(sheet)
    for k, (tile, caption) in enumerate(zip(tiles, captions, strict=True)):
        x, y = (k % columns) * width, (k // columns) * (height + caption_height)
        sheet.paste(tile.convert("RGB"), (x, y))
        d.multiline_text((x + 4, y + height + 4), caption, fill=WHITE, font=font, spacing=4)
    return sheet


PROFILE_LABELS = {"g": "G", "n": "N", "prn": "Prn", "sn": "Sn", "pog": "Pog", "ls": "Ls"}


def draw_profile(
    image: Image.Image,
    result: ProfileResult,
    *,
    label: str | None = None,
    scale: float = 1.0,
    font_size: int = 14,
) -> Image.Image:
    """Return a copy of image, scaled, with the profile edge, points, angle lines and values on it.

    The front edge is blue, G-Sn and Sn-Pog (Legan) yellow, G-Pog and N-Prn
    (nasofacial) green. Points found before a check failed are drawn gray with the
    reason, so failures can be read off the image. label goes at the top left.
    """
    im = image.convert("RGB")
    if scale != 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    font = load_font(font_size)

    def s(p) -> tuple[float, float]:
        return (float(p[0]) * scale, float(p[1]) * scale)

    if result.edge is not None and len(result.edge) >= 2:
        d.line([s(p) for p in result.edge], fill=BLUE, width=2, joint="curve")
    pts = result.points
    for (a, b), color in ((("g", "sn"), YELLOW), (("sn", "pog"), YELLOW), (("g", "pog"), GREEN), (("n", "prn"), GREEN)):
        if a in pts and b in pts:
            d.line((*s(pts[a]), *s(pts[b])), fill=color if result.measured else GRAY, width=2)
    for name, p in pts.items():
        x, y = s(p)
        color = (RED if name != "ls" else GRAY) if result.measured else GRAY
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=BLACK)
        d.text((x + 7, y - 8), PROFILE_LABELS.get(name, name), fill=WHITE, font=font, stroke_width=2, stroke_fill=BLACK)

    lines = [label] if label else []
    if result.measured:
        lines.append(f"Legan {result.legan_angle:.1f} deg  nasofacial {result.nasofacial_angle:.1f} deg")
    if result.quality is not None:
        q = result.quality
        yaw = f"yaw {q.face_yaw_deg:+.1f}" if q.face_yaw_deg is not None else "no face"
        lines.append(f"{yaw}  flipped {q.flipped}")
    if result.reasons:
        lines.append("invalid: " + ", ".join(result.reasons))
    if lines:
        d.multiline_text((8, 8), "\n".join(lines), fill=WHITE, font=font, spacing=4, stroke_width=2, stroke_fill=BLACK)
    return im

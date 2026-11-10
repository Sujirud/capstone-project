"""Shared synthetic data for the tests."""

import math
from dataclasses import dataclass

import numpy as np
import pytest


def ellipse_scut_points(cx=175.0, cy=190.0, a=110.0, b=140.0, eye=-0.3, sub=0.15, mouth=0.5) -> np.ndarray:
    """86 SCUT-numbered points of a face whose edge is the ellipse centered at (cx, cy).

    a and b are the half width and half height. eye, sub and mouth are the levels of
    the eye centers, subnasale and mouth corners as fractions of b below the center.
    Points the rules do not use sit at the ellipse center.
    """
    pts = np.tile([cx, cy], (86, 1)).astype(np.float64)

    def put(number, x, y):
        pts[number - 1] = (x, y)

    phi = np.linspace(0.15 * np.pi, 0.92 * np.pi, 10)  # 0 = top of the ellipse, pi = bottom
    for number, f in zip(range(2, 12), phi, strict=True):  # image left, top to bottom
        put(number, cx - a * np.sin(f), cy - b * np.cos(f))
    for number, f in zip(range(22, 12, -1), phi, strict=True):  # image right, top to bottom
        put(number, cx + a * np.sin(f), cy - b * np.cos(f))
    put(1, cx - 0.2 * a, cy - b)
    put(12, cx, cy + b)
    put(59, cx - 0.4 * a, cy + eye * b)
    put(60, cx + 0.4 * a, cy + eye * b)
    put(43, cx - 0.15 * a, cy + eye * b)
    put(55, cx + 0.15 * a, cy + eye * b)
    put(67, cx, cy + sub * b)
    put(65, cx - 0.18 * a, cy + 0.1 * b)
    put(69, cx + 0.18 * a, cy + 0.1 * b)
    put(80, cx - 0.3 * a, cy + mouth * b)
    put(74, cx + 0.3 * a, cy + mouth * b)
    return pts


@pytest.fixture
def scut_face():
    """Factory for ellipse_scut_points."""
    return ellipse_scut_points


def synthetic_mp_landmarks(
    cx=200.0, iod=100.0, eye_y=200.0, sub_y=260.0, mouth_y=300.0, chin_y=360.0, top_y=80.0, half_width=110.0
) -> np.ndarray:
    """(478, 3) MediaPipe-numbered points of an upright face, eyes level, not mirrored.

    The 17 inner points of each face edge lie on the vertical lines x = cx -/+ half_width
    from top_y + 20 to chin_y - 10, so the smoothed edge is straight between eye level
    and halfway from mouth level to the chin. Points the code does not use sit at the
    face center. z is 0.
    """
    from facemeasure import mp_points

    lm = np.zeros((mp_points.N_POINTS, 3))
    lm[:, 0], lm[:, 1] = cx, (top_y + chin_y) / 2

    def put(n, x, y):
        lm[n, :2] = (x, y)

    for chain, sign in ((mp_points.CONTOUR_IMAGE_LEFT, -1), (mp_points.CONTOUR_IMAGE_RIGHT, 1)):
        ys = np.linspace(top_y + 20, chin_y - 10, len(chain) - 2)
        for n, y in zip(chain[1:-1], ys, strict=True):
            put(n, cx + sign * half_width, y)
    put(mp_points.FOREHEAD, cx, top_y)
    put(mp_points.CHIN, cx, chin_y)
    left, right = cx - iod / 2, cx + iod / 2
    corner = 0.15 * iod
    put(33, left - corner, eye_y)
    put(133, left + corner, eye_y)
    put(362, right - corner, eye_y)
    put(263, right + corner, eye_y)
    put(mp_points.SUBNASALE, cx, sub_y)
    put(mp_points.NOSE_TIP, cx, sub_y - 12)
    put(mp_points.MOUTH_CORNERS[0], cx - 0.25 * iod, mouth_y)
    put(mp_points.MOUTH_CORNERS[1], cx + 0.25 * iod, mouth_y)
    for k, (a, b) in enumerate(mp_points.ALAR_BASE_PAIRS):
        w = (0.16 + 0.01 * k) * iod
        put(a, cx - w, sub_y - 3)
        put(b, cx + w, sub_y - 3)
    return lm


@pytest.fixture
def mp_face():
    """Factory for synthetic_mp_landmarks."""
    return synthetic_mp_landmarks


# ---------------------------------------------------------------- synthetic profile

# Design of a profile facing +x with y down, in pixels at scale 1. G is near the origin
# and the facial line G-Pog near vertical: G and Pog sit near the rightmost points of
# the forehead circle and of the chin circle. The front outline is a polyline whose
# corners are rounded with a Gaussian of CORNER_SIGMA pixels, as faces have no sharp
# corners; the expected points come from the definitions applied to that exact outline.
FOREHEAD_R = 150.0
NASION_Y = 60.0
TIP_CENTER = (66.0, 180.0)
TIP_R = 12.0
CHIN_R = 40.0
CHIN_Y = 400.0
HAIRLINE_Y = -110.0
MENTON_ANGLE = math.radians(80)  # where the chin circle meets the under-chin, from +x towards +y
CORNER_SIGMA = 4.0


def _arc(center, r, a0, a1, step=1.0):
    n = max(2, math.ceil(abs(a1 - a0) * r / step))
    t = np.linspace(a0, a1, n)
    return np.column_stack([center[0] + r * np.cos(t), center[1] + r * np.sin(t)])


def _line(a, b, step=1.0):
    n = max(2, math.ceil(math.dist(a, b) / step))
    return np.linspace(a, b, n)


def _tangent_angle(center, r, p, pick):
    """Angle on the circle (y down, so angles turn clockwise) of a tangent point seen from p.

    pick chooses between the two tangent points: "top" (smaller y), "bottom" or "front" (larger x).
    """
    d = np.subtract(p, center)
    base, off = math.atan2(d[1], d[0]), math.acos(r / math.hypot(*d))
    angles = (base - off, base + off)
    points = [(center[0] + r * math.cos(t), center[1] + r * math.sin(t)) for t in angles]
    key = {"top": lambda k: points[k][1], "bottom": lambda k: -points[k][1], "front": lambda k: -points[k][0]}[pick]
    return angles[min((0, 1), key=key)]


def _round_corners(poly, sigma):
    """Resample a polyline at 1 px and smooth it with a Gaussian, keeping both ends."""
    from scipy.ndimage import gaussian_filter1d

    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], math.ceil(s[-1]) + 1)
    pts = np.column_stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])])
    return gaussian_filter1d(pts, sigma, axis=0, mode="nearest")


def profile_design(
    sn_x=23.0, ls=(33.0, 240.0), st=(18.0, 262.0), li=(28.0, 284.0), sm=(-12.0, 330.0), hair_below_chin=False
):
    """The front outline (top to bottom), the polygons (head, face, neck) and the parts, in design pixels.

    parts maps forehead, nose, upper_lip and chin to slices of the outline. With
    hair_below_chin, long hair hangs in front of the neck and reaches forward of the chin.
    """
    fc = (-FOREHEAD_R, 0.0)
    n = (-FOREHEAD_R + math.sqrt(FOREHEAD_R**2 - NASION_Y**2), NASION_Y)
    forehead = _arc(fc, FOREHEAD_R, math.asin(HAIRLINE_Y / FOREHEAD_R), math.asin(NASION_Y / FOREHEAD_R))
    sn = (sn_x, 204.0)
    t_top = _tangent_angle(TIP_CENTER, TIP_R, n, "top")
    t_low = _tangent_angle(TIP_CENTER, TIP_R, sn, "bottom")
    t_low += 2 * math.pi * math.ceil((t_top - t_low) / (2 * math.pi))  # run clockwise from the top tangent
    tip = _arc(TIP_CENTER, TIP_R, t_top, t_low)
    cc = (-CHIN_R, CHIN_Y)
    chin = _arc(cc, CHIN_R, _tangent_angle(cc, CHIN_R, sm, "front"), MENTON_ANGLE)
    me = tuple(chin[-1])
    neck_front = (me[0] - 60.0, me[1] + 14.0)
    pieces = [
        forehead, _line(n, tip[0]), tip, _line(tip[-1], sn), _line(sn, ls), _line(ls, st), _line(st, li),
        _line(li, sm), _line(sm, chin[0]), chin, _line(me, neck_front), _line(neck_front, (neck_front[0] - 5, 700.0)),
    ]
    front = _round_corners(np.vstack(pieces), CORNER_SIGMA)
    # Parts by length along the outline, which the 1 px resampling keeps as the index
    ends = np.cumsum([np.hypot(*np.diff(q, axis=0).T).sum() for q in pieces])
    starts = np.concatenate([[0.0], ends[:-1]])

    def span(first, last, trim=0.0):
        return np.arange(math.ceil(starts[first] + trim), math.floor(ends[last] - trim) + 1)

    trim = 3 * CORNER_SIGMA  # keep clear of the rounded corners between parts
    parts = {"forehead": span(0, 0), "nose": span(1, 3), "upper_lip": span(4, 5, trim), "chin": span(8, 9)}
    k = math.floor(ends[9])  # end of the face outline, at the menton
    top = front[0]
    skull = _arc((-170.0, -40.0), math.dist(top, (-170.0, -40.0)), math.atan2(top[1] + 40, top[0] + 170), -math.pi)
    back = np.array([(-340.0, 120.0), (-290.0, 330.0), (-260.0, 430.0), (-265.0, 700.0)])
    head = np.vstack([front, back[::-1], skull[::-1]])
    jaw = (-200.0, 420.0)
    face = np.vstack([front[: k + 1], [jaw, (-200.0, -100.0)]])
    neck = np.vstack([front[k:], [(-265.0, 700.0), (-260.0, 430.0), jaw]])
    if hair_below_chin:
        # A lock from behind the jaw that falls past the under-chin and sticks out in front of Pog
        lock = np.array([(-150.0, 440.0), (me[0] - 10, me[1] + 6), (20.0, 520.0), (10.0, 640.0), (-90.0, 650.0)])
        head = np.vstack([front[: k + 1], lock[1:4], [(neck_front[0] - 5, 700.0)], back[::-1], skull[::-1]])
    return front, parts, head, face, neck


def expected_points(front, parts):
    """G, N, Prn, Sn and Pog by their definitions on the exact outline, without the code under test."""
    from scipy.spatial import ConvexHull

    def ahead(a, b, p):  # distance in front of the line a -> b (a above b), towards +x
        d = b - a
        return (d[1] * (p[:, 0] - a[0]) - d[0] * (p[:, 1] - a[1])) / math.hypot(*d)

    fc = np.concatenate([parts["forehead"], parts["chin"]])
    hull = ConvexHull(front[fc]).vertices  # counterclockwise
    edges = [(fc[a], fc[b]) for a, b in zip(hull, np.roll(hull, -1), strict=True)]
    bridges = [
        (a, b) if a in parts["forehead"] else (b, a)
        for a, b in edges
        if (a in parts["forehead"]) != (b in parts["forehead"])
    ]
    ig, ipog = max(bridges, key=lambda e: front[e[0], 0] + front[e[1], 0])  # the front one of the two
    g, pog = front[ig], front[ipog]
    nose = parts["nose"]
    prn = front[nose[np.argmax(ahead(g, pog, front[nose]))]]
    iprn = nose[np.argmax(ahead(g, pog, front[nose]))]
    mid = np.arange(ig + 1, iprn)
    n = front[mid[np.argmax(-ahead(g, prn, front[mid]))]]
    lip = parts["upper_lip"]
    ils = lip[np.argmax(ahead(g, pog, front[lip]))]
    ls = front[ils]
    mid = np.arange(iprn + 1, ils)
    sn = front[mid[np.argmax(-ahead(prn, ls, front[mid]))]]
    return {"g": g, "n": n, "prn": prn, "sn": sn, "pog": pog}


@dataclass
class SyntheticProfile:
    head: np.ndarray  # (h, w) head probability, the face points to +x
    face: np.ndarray  # face skin probability
    hair: np.ndarray
    body: np.ndarray
    image: np.ndarray  # (h, w, 3) uint8: hair red, face skin green, body skin blue, background black
    points: dict[str, np.ndarray]  # expected points in pixels
    legan: float
    nasofacial: float


def synthetic_profile(scale=1.0, angle_deg=0.0, size=(900, 1000), origin=(560.0, 300.0), supersample=4, **design):
    """Render the design turned by angle_deg about (0, 200) and scaled, with soft edges.

    size is (width, height) at scale 1; origin is where the design origin lands at
    scale 1. The expected angles do not change when the design turns or scales.
    """
    from PIL import Image, ImageDraw

    from facemeasure.profile import legan_angle, nasofacial_angle

    front, parts, head, face, neck = profile_design(**design)
    pts = expected_points(front, parts)
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pivot = np.array([0.0, 200.0])

    def place(p):
        return ((np.asarray(p) - pivot) @ rot.T + pivot + origin) * scale

    w, h = round(size[0] * scale), round(size[1] * scale)
    labels = Image.new("L", (w * supersample, h * supersample), 0)
    d = ImageDraw.Draw(labels)
    for value, poly in ((1, head), (3, face), (2, neck)):  # hair, then face skin, then body skin
        d.polygon([tuple(q) for q in place(poly) * supersample], fill=value)
    lab = np.asarray(labels).reshape(h, supersample, w, supersample)
    cover = {v: (lab == v).mean(axis=(1, 3)) for v in (1, 2, 3)}
    rgb = np.stack([cover[1], cover[3], cover[2]], axis=-1)
    return SyntheticProfile(
        head=cover[1] + cover[2] + cover[3],
        face=cover[3],
        hair=cover[1],
        body=cover[2],
        image=np.round(rgb * 255).astype(np.uint8),
        points={k: place(v) for k, v in pts.items()},
        legan=legan_angle(pts["g"], pts["sn"], pts["pog"]),
        nasofacial=nasofacial_angle(pts["g"], pts["pog"], pts["n"], pts["prn"]),
    )


@pytest.fixture
def profile_face():
    """Factory for synthetic_profile."""
    return synthetic_profile


# ---------------------------------------------------------------- the example of api.md


def load_api_example() -> dict:
    """The one JSON block of docs/api.md section 2.2."""
    import json
    import re

    from facemeasure import paths

    text = (paths.project_root() / "docs" / "api.md").read_text(encoding="utf-8")
    section = text.split("\n### 2.2", 1)[1].split("\n### ", 1)[0]
    blocks = re.findall(r"```json\n(.*?)\n```", section, flags=re.DOTALL)
    assert len(blocks) == 1, "section 2.2 of api.md should hold one JSON example"
    return json.loads(blocks[0])


@pytest.fixture(scope="session")
def api_example():
    """load_api_example(); a fresh copy is not needed, so do not change it."""
    return load_api_example()


# ---------------------------------------------------------------- bundles for response.py and the api


def mark_trained(manifest: dict) -> dict:
    """The manifest with every head marked as trained (stub false), so results do not get model_not_ready."""
    import copy

    out = copy.deepcopy(manifest)
    for head in out["heads"].values():
        head["stub"] = False
    out["stub"] = False
    return out


def ready_stub_bundle():
    """The stub bundle of make_stub_bundle with its heads marked as trained: fixed answers, assessed regions."""
    from facemeasure import bundle
    from make_stub_bundle import stub_parts

    manifest, arrays = stub_parts()
    manifest = mark_trained(manifest)
    manifest["files"] = {bundle.HEADS_NAME: "0" * 64}
    return bundle.parse(manifest, arrays)


@pytest.fixture
def trained():
    """Factory for mark_trained."""
    return mark_trained


@pytest.fixture(scope="session")
def ready_bundle():
    """ready_stub_bundle(), for tests of results that need assessed regions."""
    return ready_stub_bundle()


# ---------------------------------------------------------------- synthetic results for response.py

# Values of the example in api.md section 2.2
EXAMPLE_FRONTAL_QUALITY = {
    "iod_px": 96.0, "yaw_deg": 0.9, "pitch_deg": -0.8, "roll_deg": 0.5, "smile": 0.05, "jaw_open": 0.02,
    "blink_left": 0.03, "blink_right": 0.04, "hair_jaw": 0.02, "hair_undereye_left": 0.0, "hair_undereye_right": 0.0,
}
EXAMPLE_FRONTAL_POINTS = {
    "go_l": (405.0, 980.0), "go_r": (877.0, 978.0), "zy_l": (356.0, 640.0), "zy_r": (920.0, 642.0),
    "en_l": (590.0, 560.0), "en_r": (690.0, 561.0), "al_l": (585.0, 800.0), "al_r": (695.0, 801.0),
}
EXAMPLE_PROFILE_POINTS = {"g": (640.0, 410.0), "n": (655.0, 520.0), "prn": (774.0, 760.0), "sn": (700.0, 800.0), "pog": (610.0, 1210.0)}


def box(x0: float, y0: float, side: float) -> np.ndarray:
    """Corners top-left, top-right, bottom-right, bottom-left of an axis-aligned square."""
    return np.array([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)], dtype=np.float64)


def synthetic_frontal_result(
    failed=(), hair_over=(), face=True, ratios=True, jaw_ratio=0.842, nose_ratio=1.08, values=None, points=None
):
    """A FrontalResult as analyze_frontal returns it, without running any model.

    failed and hair_over are the codes and the hair values over the limit; face=False
    gives the result of an image without a face (no_face only).
    """
    from facemeasure import codes, crops, quality
    from facemeasure.face_geometry import LevelFrame
    from facemeasure.frontal import FrontalRatios, FrontalResult

    if not face:
        failed = (codes.NO_FACE,)
        return FrontalResult(1280, 1600, 0, None, None, None, None, failed, codes.by_region(failed), None, None, None)
    values = dict(EXAMPLE_FRONTAL_QUALITY if values is None else values)
    blank = np.zeros((crops.CROP_SIZE, crops.CROP_SIZE, 3), dtype=np.uint8)
    face_crops = crops.Crops(
        face=crops.Crop(blank, box(300.0, 300.0, 700.0), False),
        undereye_left=crops.Crop(blank, box(683.25, 560.5, 57.6), True),
        undereye_right=crops.Crop(blank, box(561.0, 560.0, 57.6), False),
        frame=LevelFrame(center=np.array([640.0, 560.0]), cos=1.0, sin=0.0),
        iod_px=values["iod_px"],
    )
    r = None
    if ratios:
        r = FrontalRatios(
            jaw_ratio=jaw_ratio,
            nose_ratio=nose_ratio,
            nose_ratios={"98_327": nose_ratio},
            points=dict(EXAMPLE_FRONTAL_POINTS if points is None else points),
        )
    failed = tuple(failed)
    return FrontalResult(
        width=1280,
        height=1600,
        face_count=2 if codes.MULTIPLE_FACES in failed else 1,
        landmarks=np.zeros((478, 3)),
        blendshapes={},
        matrix=np.eye(4),
        quality=quality.FrontalQuality(**values, failed=failed),
        failed=failed,
        region_failed=codes.by_region(failed, tuple(hair_over)),
        crops=face_crops,
        ratios=r,
        ratios_error=None if ratios else "left face edge does not reach mouth level",
    )


def synthetic_profile_result(legan=14.0, nasofacial=34.0, failed=(), measured=True, face_found=False, yaw=None, points=None):
    """A ProfileResult as analyze_profile returns it; measured=False has no angles (the edge failed)."""
    from facemeasure import codes
    from facemeasure import profile as P

    failed = tuple(failed)
    pts = dict(EXAMPLE_PROFILE_POINTS if points is None else points)
    q = P.ProfileQuality(face_found=face_found, face_yaw_deg=yaw, silhouette_found=True, flipped=False, failed=failed)
    kw = {}
    if measured:
        overlays = {
            region: P.Overlay("profile", {k: pts[k] for k in names}, P.OVERLAY_LINES[region])
            for region, names in P.OVERLAY_POINTS.items()
        }
        kw = {"legan_angle": legan, "nasofacial_angle": nasofacial, "nose_depth": 0.2, "face_height": 900.0, "overlays": overlays}
    reasons = (P.FRONTAL_YAW,) if failed and measured else ((P.CHIN_CUT,) if failed else ())
    return P.ProfileResult(1280, 1600, q, failed, codes.profile_by_region(failed), reasons, points=pts, **kw)


@pytest.fixture
def frontal_result():
    """Factory for synthetic_frontal_result."""
    return synthetic_frontal_result


@pytest.fixture
def profile_result():
    """Factory for synthetic_profile_result."""
    return synthetic_profile_result

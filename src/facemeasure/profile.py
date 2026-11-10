"""The one entry point for a profile image (model.md sections 3.2 and 3.4, api.md section 3).

analyze_profile() checks that the photo is turned to the side, finds the front edge of
the head silhouette, the points G, N, Prn, Sn and Pog on it, and the Legan and
nasofacial angles. The api and model/scripts/profile_check.py call only
this function.

Steps (api.md section 3 step 3):

1. Face Landmarker. A face with |yaw| under profile.min_abs_yaw_deg of
   shared/quality.json is not turned far enough: profile_invalid. Measuring still
   goes on, as analyze_frontal does for a face that fails a check, so lab mode
   (api.md section 2.4) and the FEI check can see the angles.
2. Selfie multiclass on the whole image. The preliminary head is the largest
   connected blob of the hair and face skin classes; without one, silhouette_found
   is false.
3. A square crop of CROP_SCALE times the longer side of the blob's box, segmented
   again because the segmenter works at 256 x 256. The head probability is the sum
   of the hair, body skin and face skin confidences, kept to its largest blob.
4. Direction: the sign of yaw when a face was found (positive when the face points to
   the image right), else the face skin centroid against the hair centroid. When the
   face points to the image left, the whole image is mirrored and steps 2 and 3 run
   again on the mirrored image, so the work frame always has the face to the right
   (+x) with y down. The segmenter is not mirror-symmetric, so mirroring its masks
   instead would give a photo and its mirror image different edges; this way both
   reach it as the same pixels (PROFILE_VERSION 2).
5. measure_silhouette() on the head probability: front edge, points, angles, checks.
6. Points and edge go back to image coordinates: add the crop corner, then mirror
   back. crop is in coordinates of the received image too.

Coordinates are pixels with pixel (row i, column j) covering [j, j + 1) x [i, i + 1),
so its center is (j + 0.5, i + 0.5), as in quality.HairCrop.

Points, all on the smoothed front edge:

- G and Pog: where the common front tangent of the forehead and chin parts touches
  them, so no part of either lies in front of the line G-Pog.
- Prn: the point of the nose part furthest in front of G-Pog.
- N: the point between G and Prn furthest behind G-Prn.
- Ls: the point of the upper lip furthest in front of G-Pog; Sn: the point between
  Prn and Ls furthest behind Prn-Ls.

The parts are split first with guesses. "Behind the front hull" below means the
distance of an edge point behind the convex hull of the edge on the face side.

- Prn0: the most forward point among the rows whose edge is face skin.
- The forehead is the run of face skin edge rows above Prn0, up to where hair (a
  fringe, say) takes over the edge. N0 is its deepest point behind the front hull.
- The face edge ends at the menton guess Me0: the last point below the nose with face
  skin within FACE_BAND_FRACTION of the face height behind the edge, and no lower
  than CHIN_MARGIN_FRACTION of the face height below the face skin. The neck, and
  long hair that hangs in front of it, have no face skin behind them.
- Sn0 is the first clear peak of the depth behind the front hull below Prn0.
- The labiomental fold Sm0 is the deepest point behind the front hull of Sn0..Me0
  within FOLD_WINDOW of the way from Sn0 to Me0, which covers the fold in usual face
  proportions and leaves out the lips above it. The chin part runs from Sm0 to Me0.

Lines and distances from lines do not change when the image turns, so the points stay
put when the head tips up or down (model.md section 3.2). The edge is still read row
by row, so a nose tip that hangs below subnasale hides Sn.

Checks that make the photo profile_invalid, each with its reason: the face rows or the
front edge reach the image border, or the image ends less than BELOW_POG_FRACTION
below Pog; a part cannot be found or the points are out of order (G, N, Prn, Sn, Pog
from top to bottom); the forehead run is shorter than MIN_FOREHEAD or G is not
MIN_GLABELLA above N0, which is what glasses and hair over the forehead do; the nose
is less than MIN_NOSE_DEPTH in front of G-Pog; G-Pog is shorter than MIN_FACIAL_LINE;
or an angle is outside LEGAN_RANGE or NASOFACIAL_RANGE.
"""

import itertools
import math
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter

from facemeasure import codes, pose, quality
from facemeasure.landmarks import HAIR, SEGMENT_CLASSES, FaceModels, check_rgb
from facemeasure.quality import HairCrop

Point = tuple[float, float]

FACE_SKIN = SEGMENT_CLASSES.index("face_skin")
BODY_SKIN = SEGMENT_CLASSES.index("body_skin")
HEAD_CLASSES = (HAIR, BODY_SKIN, FACE_SKIN)  # their confidences add up to the head probability
SEED_CLASSES = (HAIR, FACE_SKIN)  # the preliminary head on the whole image

# 2: a face that points to the image left is segmented on the mirrored image (step 4)
PROFILE_VERSION = 2
CROP_SCALE = 1.4
EDGE_LEVEL = 0.5
KEEP_MARGIN_PX = 2  # pixels kept around the head blob, so the edge can be interpolated
SMOOTH_FRACTION = 0.02  # Savitzky-Golay window, of the face skin height
SMOOTH_POLYORDER = 2
MIN_SMOOTH_WINDOW = 5
MIN_DIRECTION_SHARE = 0.02  # hair (else body skin) mass, of the head mass, to tell the direction
EDGE_INSET_FRACTION = 0.015  # the class of an edge row is read this far inside the edge
HAIR_GAP_FRACTION = 0.02  # rows of other classes allowed inside the forehead run, such as a stray lock
CHIN_MARGIN_FRACTION = 0.10  # rows searched below the face skin, for a beard the segmenter calls hair
FACE_BAND_FRACTION = 0.08  # face skin within this far behind the edge keeps the edge part of the face
DIP_FRACTION = 0.01  # depth behind the front hull that counts as a dip, of the face height
SN_PROMINENCE = 0.005  # how much the depth must fall again after Sn0, of the face height
FOLD_WINDOW = (0.45, 0.85)  # where the fold is looked for, as a share of the way from Sn0 to Me0
BELOW_POG_FRACTION = 0.05  # the edge and the image go on this far below Pog, else the chin may be cut

# Plausibility checks, set from the 400 FEI full profiles (the spread of each
# value is in model/reports/fei_profiles.md). Check images of the photos each one
# rejects showed misplaced points: G on glasses, on a fringe or at the nose when hair
# covers the forehead, Sn or Pog on hair under the chin, and photos not turned enough.
# A glabella that does not reach in front of the nasion, or a forehead run shorter
# than about 1 cm, means the forehead edge is not skin.
MIN_FOREHEAD = 0.05  # face skin edge from the top of the forehead run to N0, of the face height
MIN_GLABELLA = 0.02  # G above N0, of the face height (on a face the glabella is about 1 cm above)
# Prn in front of G-Pog, of |G-Pog|: 0.05 is about 6 mm, which no real nose falls short of;
# p2 of the FEI full profiles, and every one below it had misplaced points
MIN_NOSE_DEPTH = 0.05
MIN_FACIAL_LINE = 0.40  # |G-Pog|, of the face height; the FEI full profiles start at 0.48
# Every FEI full profile below -15 degrees had Sn or Pog misplaced; a chin set back by more
# than 40 degrees did not occur. The nasofacial range is wide, a check on broken lines only.
LEGAN_RANGE = (-15.0, 40.0)
NASOFACIAL_RANGE = (15.0, 60.0)

# Why a profile is invalid; for reports and check images only, not part of the api result
FRONTAL_YAW = "frontal_yaw"  # a face was found with |yaw| under the limit
NO_SILHOUETTE = "no_silhouette"
NO_FACE_SKIN = "no_face_skin"
NO_DIRECTION = "no_direction"
HEAD_CUT = "head_cut"  # the face rows or the front edge reach the image border
CHIN_CUT = "chin_cut"
EDGE_SHORT = "edge_short"
NO_FOREHEAD = "no_forehead"
NO_SUBNASALE = "no_subnasale"
ORDER = "order"
FACIAL_LINE_SHORT = "facial_line_short"
NOSE_SHALLOW = "nose_shallow"
ANGLE_RANGE = "angle_range"
REASONS = (
    FRONTAL_YAW, NO_SILHOUETTE, NO_FACE_SKIN, NO_DIRECTION, HEAD_CUT, CHIN_CUT, EDGE_SHORT,
    NO_FOREHEAD, NO_SUBNASALE, ORDER, FACIAL_LINE_SHORT, NOSE_SHALLOW, ANGLE_RANGE,
)

POINT_NAMES = ("g", "n", "prn", "sn", "pog")
# Overlays of api.md section 2.2
OVERLAY_POINTS = {codes.CHIN: ("g", "sn", "pog"), codes.NOSE_DORSUM: ("g", "n", "prn", "pog")}
OVERLAY_LINES = {codes.CHIN: (("g", "sn"), ("sn", "pog")), codes.NOSE_DORSUM: (("g", "pog"), ("n", "prn"))}


# ---------------------------------------------------------------- geometry in the work frame


def front_of(a, b, p) -> np.ndarray:
    """Signed distance of p in front of the line from a down to b; positive towards +x when b is below a."""
    a, b, p = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(p, dtype=np.float64)
    d = b - a
    return (d[1] * (p[..., 0] - a[0]) - d[0] * (p[..., 1] - a[1])) / math.hypot(d[0], d[1])


def front_chain(points: np.ndarray) -> list[int]:
    """Indices of the front side of the convex hull of points that run strictly down in y."""
    pts = np.asarray(points, dtype=np.float64)
    chain: list[int] = []
    for k in range(len(pts)):
        while len(chain) >= 2 and front_of(pts[chain[-2]], pts[k], pts[chain[-1]]) <= 0:
            chain.pop()
        chain.append(k)
    return chain


def hull_depth(points: np.ndarray) -> np.ndarray:
    """Distance of each point behind the front hull of points (0 on the hull); points run down in y."""
    pts = np.asarray(points, dtype=np.float64)
    depth = np.zeros(len(pts))
    for a, b in itertools.pairwise(front_chain(pts)):
        if b - a >= 2:
            depth[a + 1 : b] = -front_of(pts[a], pts[b], pts[a + 1 : b])
    return depth


def hull_dips(points: np.ndarray) -> list[tuple[int, float]]:
    """(index, depth) of the deepest point between each pair of front hull vertices, top to bottom."""
    depth = hull_depth(points)
    out = []
    for a, b in itertools.pairwise(front_chain(points)):
        if b - a >= 2:
            k = a + 1 + int(np.argmax(depth[a + 1 : b]))
            out.append((k, float(depth[k])))
    return out


def legan_angle(g, sn, pog) -> float:
    """Angle from the direction G->Sn to Sn->Pog in degrees, in the work frame.

    Positive when Pog lies behind the extension of G-Sn, that is a chin set back.
    """
    u = np.asarray(sn, dtype=np.float64) - np.asarray(g, dtype=np.float64)
    v = np.asarray(pog, dtype=np.float64) - np.asarray(sn, dtype=np.float64)
    return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]))


def nasofacial_angle(g, pog, n, prn) -> float:
    """Angle between G->Pog and N->Prn in degrees, in the work frame; positive when the nose points forward."""
    d1 = np.asarray(pog, dtype=np.float64) - np.asarray(g, dtype=np.float64)
    d2 = np.asarray(prn, dtype=np.float64) - np.asarray(n, dtype=np.float64)
    return math.degrees(math.atan2(-(d1[0] * d2[1] - d1[1] * d2[0]), d1[0] * d2[0] + d1[1] * d2[1]))


@dataclass(frozen=True)
class Bounds:
    """Rows top..bottom-1 and columns left..right-1 of an array that are pixels of the image."""

    top: int
    bottom: int
    left: int
    right: int

    @classmethod
    def of(cls, shape: tuple[int, ...]) -> "Bounds":
        return cls(0, shape[0], 0, shape[1])


def largest_blob(mask: np.ndarray) -> np.ndarray | None:
    """The largest 8-connected blob of mask, or None when mask is empty."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def face_rows(face_prob: np.ndarray) -> tuple[int, int] | None:
    """First and last row of the largest face skin blob (confidence >= EDGE_LEVEL), or None."""
    blob = largest_blob(np.asarray(face_prob) >= EDGE_LEVEL)
    if blob is None:
        return None
    rows = np.flatnonzero(blob.any(axis=1))
    return int(rows[0]), int(rows[-1])


def front_edge(head: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, x, at_border): per row with head pixels, where the probability last falls below EDGE_LEVEL.

    x is interpolated between the centers of the last pixel at or above the level and
    the next one. A row whose last such pixel is the last column has x at the array
    edge and at_border set.
    """
    p = np.asarray(head, dtype=np.float64)
    width = p.shape[1]
    above = p >= EDGE_LEVEL
    rows = np.flatnonzero(above.any(axis=1))
    last = width - 1 - np.argmax(above[rows, ::-1], axis=1)
    inside = last < width - 1
    x = np.full(len(rows), float(width))
    a = p[rows[inside], last[inside]]
    b = p[rows[inside], last[inside] + 1]
    x[inside] = last[inside] + 0.5 + (a - EDGE_LEVEL) / (a - b)
    return rows, x, ~inside


def smooth_window(face_height: float) -> int:
    return max(MIN_SMOOTH_WINDOW, round(SMOOTH_FRACTION * face_height) | 1)


def smooth_edge(rows: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """(k, 2) edge resampled every pixel of its length and smoothed with Savitzky-Golay.

    Smoothing along the length instead of down the rows treats every part of the
    edge alike: under the nose the edge runs nearly flat, so it spans few rows but
    many pixels of length, and a filter down the rows would round it off. y is kept
    running down.
    """
    pts = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(rows, dtype=np.float64) + 0.5])
    length = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    t = np.linspace(0.0, length[-1], max(2, math.ceil(length[-1]) + 1))
    res = np.column_stack([np.interp(t, length, pts[:, 0]), np.interp(t, length, pts[:, 1])])
    if len(res) < window:
        return res
    out = savgol_filter(res, window, SMOOTH_POLYORDER, axis=0, mode="interp")
    out[:, 1] = np.maximum.accumulate(out[:, 1])
    return out


def edge_near(prob: np.ndarray | None, edge: np.ndarray, inset: float, band: float, samples: int = 8) -> np.ndarray:
    """Per edge point: the class confidence reaches EDGE_LEVEL somewhere inset to inset + band inside the edge."""
    if prob is None:
        return np.ones(len(edge), dtype=bool)
    out = np.zeros(len(edge), dtype=bool)
    for t in np.linspace(inset, inset + band, samples):
        out |= edge_class(prob, edge, t, True)
    return out


def edge_class(prob: np.ndarray | None, edge: np.ndarray, inset: float, default: bool) -> np.ndarray:
    """Per edge point: the class confidence inset pixels inside the edge is at least EDGE_LEVEL."""
    if prob is None:
        return np.full(len(edge), default)
    p = np.asarray(prob)
    cols = np.clip(np.floor(edge[:, 0] - inset).astype(int), 0, p.shape[1] - 1)
    rows = np.clip(np.floor(edge[:, 1]).astype(int), 0, p.shape[0] - 1)
    return p[rows, cols] >= EDGE_LEVEL


@dataclass(frozen=True)
class Silhouette:
    """What measure_silhouette found, in the work frame (face to +x, y down)."""

    edge: np.ndarray  # (k, 2) smoothed front edge (x, y), top to bottom
    points: dict[str, np.ndarray]  # g n prn sn pog and ls; the points found before a check failed
    legan_angle: float | None
    nasofacial_angle: float | None
    nose_depth: float | None  # Prn in front of G-Pog, of |G-Pog|
    reason: str | None  # None when every check passed and the angles are set
    guesses: dict[str, np.ndarray] = field(default_factory=dict)  # prn0 n0 sn0 sm0 me0 that split the edge


def _argmax(values: np.ndarray, index: np.ndarray) -> int:
    return int(index[int(np.argmax(values))])


def measure_silhouette(
    head: np.ndarray, rows_of_face: tuple[int, int], bounds: Bounds | None = None, face: np.ndarray | None = None
) -> Silhouette:
    """Points and angles from a head probability array whose face points to +x.

    rows_of_face are the first and last row of the face skin (face_rows), which set
    the face height, the rows searched for the nose and the chin, and the smoothing.
    bounds are the array rows and columns that are image pixels (default: all), so a
    head cut by the image border is caught. face is the face skin confidence on the
    same grid: edge points that are not face skin, such as a fringe over the
    forehead, are not forehead or nose, and below the nose the face edge ends where
    no face skin lies close behind the edge, as under the chin. Without it every
    point counts as face skin.
    """
    head = np.asarray(head, dtype=np.float64)
    bounds = bounds or Bounds.of(head.shape)
    top, bottom = rows_of_face
    height = bottom - top + 1
    rows, x_raw, at_border = front_edge(head)
    window = smooth_window(height)
    if len(rows) < window or height < MIN_SMOOTH_WINDOW:
        return Silhouette(np.empty((0, 2)), {}, None, None, None, EDGE_SHORT)
    edge = smooth_edge(rows, x_raw, window)
    xs, ys = edge[:, 0], edge[:, 1]
    found: dict[str, np.ndarray] = {}
    guess: dict[str, np.ndarray] = {}

    def fail(reason: str, depth: float | None = None) -> Silhouette:
        return Silhouette(edge, dict(found), None, None, depth, reason, dict(guess))

    if top <= bounds.top or bottom >= bounds.bottom - 1:
        return fail(HEAD_CUT)
    last_row = bottom + CHIN_MARGIN_FRACTION * height
    near = (rows >= top) & (rows <= last_row)
    if np.any(at_border[near]) or np.any(x_raw[near] >= bounds.right - 0.5):
        return fail(HEAD_CUT)
    i0 = int(np.searchsorted(ys, top + 0.5))
    i1 = int(np.searchsorted(ys, bottom + 0.5, side="right")) - 1
    i2 = int(np.searchsorted(ys, last_row + 0.5, side="right")) - 1
    if i1 - i0 < window:
        return fail(EDGE_SHORT)
    inset = max(2.0, EDGE_INSET_FRACTION * height)
    skin = edge_class(face, edge, inset, True)
    near_face = edge_near(face, edge, inset, FACE_BAND_FRACTION * height)
    gap = round(HAIR_GAP_FRACTION * height)

    # Prn0: the most forward face skin edge among the face rows
    face_index = np.arange(i0, i1 + 1)
    skin_index = face_index[skin[face_index]]
    if len(skin_index) == 0:
        return fail(NO_FOREHEAD)
    iprn0 = _argmax(xs[skin_index], skin_index)
    guess["prn0"] = edge[iprn0]

    # Forehead: the face skin edge above Prn0 up to where another class takes over.
    # The rows the smoothing blends with that class are left out.
    itop, run = iprn0, 0
    for k in range(iprn0 - 1, i0 - 1, -1):
        run = 0 if skin[k] else run + 1
        if run > gap:
            itop = min(iprn0, itop + window // 2)
            break
        if skin[k]:
            itop = k
    if iprn0 - itop < 3:
        return fail(NO_FOREHEAD)
    depth = hull_depth(edge[itop : iprn0 + 1])
    if depth.max() <= 0:
        return fail(NO_FOREHEAD)
    in0 = itop + int(np.argmax(depth))
    guess["n0"] = edge[in0]
    if ys[in0] - ys[itop] < MIN_FOREHEAD * height:
        return fail(NO_FOREHEAD)  # hair covers the forehead, or the edge above the nose is not skin

    # Me0: the last point below the nose with face skin close behind the edge, within the
    # face rows and margin. The neck and hair that hangs in front of it have none; a beard
    # is thin, so the chin is still close behind it.
    ime0, run = i2, 0
    for k in range(iprn0 + 1, i2 + 1):
        run = 0 if near_face[k] else run + 1
        if run > gap:
            ime0 = k - run
            break
    guess["me0"] = edge[ime0]
    if ime0 - iprn0 < 3 * window:
        return fail(NO_SUBNASALE)
    # Sn0: the first clear peak of the depth behind the front hull below Prn0. Not the
    # deepest point: when the lips do not reach the hull, Sn and the labiomental fold lie in
    # one dip and the fold can be the deeper of the two.
    dip = DIP_FRACTION * height
    peaks = find_peaks(hull_depth(edge[iprn0 : ime0 + 1]), height=dip, prominence=SN_PROMINENCE * height)[0]
    if len(peaks) == 0:
        return fail(NO_SUBNASALE)
    isn0 = iprn0 + int(peaks[0])
    guess["sn0"] = edge[isn0]
    # Sm0: the deepest point behind the front hull of Sn0..Me0 inside FOLD_WINDOW
    length = edge[ime0, 1] - edge[isn0, 1]
    depth = hull_depth(edge[isn0 : ime0 + 1])
    window_rows = isn0 + np.flatnonzero(
        (edge[isn0 : ime0 + 1, 1] >= edge[isn0, 1] + FOLD_WINDOW[0] * length)
        & (edge[isn0 : ime0 + 1, 1] <= edge[isn0, 1] + FOLD_WINDOW[1] * length)
    )
    if len(window_rows) == 0:
        return fail(ORDER)
    ism0 = _argmax(depth[window_rows - isn0], window_rows)
    guess["sm0"] = edge[ism0]
    # The upper lip ends at the deepest point behind the hull between Sn0 and the fold
    if ism0 - isn0 < 3:
        return fail(ORDER)
    dips = [isn0 + k for k, d in hull_dips(edge[isn0 : ism0 + 1]) if d >= DIP_FRACTION * height]
    ilip_end = dips[0] if dips else ism0

    # G and Pog: the bridge of the front hull of the forehead and chin parts
    index = np.concatenate([np.arange(itop, in0 + 1), np.arange(ism0, ime0 + 1)])
    chain = front_chain(edge[index])
    k = next(k for k in range(len(chain) - 1) if index[chain[k]] <= in0 < index[chain[k + 1]])
    ig, ipog = int(index[chain[k]]), int(index[chain[k + 1]])
    g, pog = edge[ig], edge[ipog]
    found.update(g=g, pog=pog)
    if ys[in0] - g[1] < MIN_GLABELLA * height:
        return fail(NO_FOREHEAD)  # no forehead in front of the line to the chin: not turned enough, or covered

    nose = np.arange(in0 + 1, isn0)
    if len(nose) == 0:
        return fail(ORDER)
    iprn = _argmax(front_of(g, pog, edge[nose]), nose)
    found["prn"] = edge[iprn]
    between = np.arange(ig + 1, iprn)
    if len(between) == 0:
        return fail(ORDER)
    i_n = _argmax(-front_of(g, edge[iprn], edge[between]), between)
    found["n"] = edge[i_n]
    lip = np.arange(isn0, ilip_end + 1)
    ils = _argmax(front_of(g, pog, edge[lip]), lip)
    found["ls"] = edge[ils]
    between = np.arange(iprn + 1, ils)
    if len(between) == 0:
        return fail(ORDER)
    isn = _argmax(-front_of(edge[iprn], edge[ils], edge[between]), between)
    found["sn"] = edge[isn]

    if not ig < i_n < iprn < isn < ipog:
        return fail(ORDER)
    margin = BELOW_POG_FRACTION * height
    if pog[1] + margin > bounds.bottom or edge[-1, 1] < pog[1] + margin:
        return fail(CHIN_CUT)
    facial = math.hypot(*(pog - g))
    if facial < MIN_FACIAL_LINE * height:
        return fail(FACIAL_LINE_SHORT)
    nose_depth = float(front_of(g, pog, edge[iprn])) / facial
    if nose_depth < MIN_NOSE_DEPTH:
        return fail(NOSE_SHALLOW, nose_depth)
    legan = legan_angle(g, found["sn"], pog)
    naso = nasofacial_angle(g, pog, found["n"], found["prn"])
    if not (LEGAN_RANGE[0] <= legan <= LEGAN_RANGE[1] and NASOFACIAL_RANGE[0] <= naso <= NASOFACIAL_RANGE[1]):
        return fail(ANGLE_RANGE, nose_depth)
    return Silhouette(edge, dict(found), legan, naso, nose_depth, None, dict(guess))


# ---------------------------------------------------------------- the whole image


@dataclass(frozen=True)
class ProfileQuality:
    """quality.profile of api.md section 2.2."""

    face_found: bool
    face_yaw_deg: float | None
    silhouette_found: bool
    flipped: bool
    failed: tuple[str, ...]

    def as_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["failed"] = list(self.failed)
        return out


@dataclass(frozen=True)
class Overlay:
    """overlay of one region in api.md section 2.2: points in pixels of the received image."""

    image: str
    points: dict[str, Point]
    lines: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict:
        return {
            "image": self.image,
            "points": {k: [x, y] for k, (x, y) in self.points.items()},
            "lines": [list(line) for line in self.lines],
        }


@dataclass(frozen=True)
class ProfileResult:
    """Everything the api and the FEI check need from one profile image.

    Without a profile image, quality is None and both regions get missing_input.
    points holds the points found, even when a check failed, for check images;
    the angles and overlays are set only when every check of the edge passed, so
    a photo that is only short of the yaw limit still has them (lab mode).
    """

    width: int | None
    height: int | None
    quality: ProfileQuality | None
    failed: tuple[str, ...]
    region_failed: dict[str, tuple[str, ...]]
    reasons: tuple[str, ...]
    legan_angle: float | None = None
    nasofacial_angle: float | None = None
    nose_depth: float | None = None
    face_height: float | None = None  # rows of the face skin, in pixels
    points: dict[str, Point] = field(default_factory=dict)
    overlays: dict[str, Overlay] = field(default_factory=dict)
    edge: np.ndarray | None = None  # (k, 2) front edge in pixels of the received image
    crop: HairCrop | None = None  # the head crop that was segmented the second time

    @property
    def measured(self) -> bool:
        return self.legan_angle is not None


def missing_profile() -> ProfileResult:
    """The result when no profile image was sent."""
    failed = (codes.MISSING_INPUT,)
    return ProfileResult(None, None, None, failed, codes.profile_by_region(failed), ())


def head_crop(blob: np.ndarray) -> HairCrop:
    """Square crop CROP_SCALE times the longer side of the blob's box, on the box center."""
    rows = np.flatnonzero(blob.any(axis=1))
    cols = np.flatnonzero(blob.any(axis=0))
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    side = max(1, math.ceil(CROP_SCALE * max(x1 - x0, y1 - y0)))
    return HairCrop(x0=round((x0 + x1) / 2 - side / 2), y0=round((y0 + y1) / 2 - side / 2), side=side)


def faces_right(confidence: np.ndarray, keep: np.ndarray) -> bool | None:
    """True when the face skin centroid is right of the hair centroid (else body skin); None if unknown."""
    xs = np.arange(confidence.shape[2]) + 0.5
    head_mass = float(sum(confidence[c][keep].sum() for c in HEAD_CLASSES))

    def centroid(c: int) -> tuple[float, float]:
        column_mass = (confidence[c] * keep).sum(axis=0)
        mass = float(column_mass.sum())
        return (float(column_mass @ xs) / mass if mass > 0 else math.nan), mass

    face_x, face_mass = centroid(FACE_SKIN)
    if face_mass <= 0:
        return None
    for c in (HAIR, BODY_SKIN):
        x, mass = centroid(c)
        if mass >= MIN_DIRECTION_SHARE * head_mass:
            return face_x > x
    return None


def _result(width, height, face_found, yaw, silhouette, flipped, reasons, **kw) -> ProfileResult:
    failed = (codes.PROFILE_INVALID,) if reasons else ()
    q = ProfileQuality(face_found=face_found, face_yaw_deg=yaw, silhouette_found=silhouette, flipped=flipped, failed=failed)
    return ProfileResult(width, height, q, failed, codes.profile_by_region(failed), tuple(reasons), **kw)


@dataclass(frozen=True)
class _Head:
    """Steps 2 and 3 on one image: the head crop, its confidences and the head kept to its blob."""

    crop: HairCrop
    confidence: np.ndarray | None  # (6, side, side); None when the crop has no head blob
    head: np.ndarray | None  # head probability kept to the blob and its margin
    face: np.ndarray | None  # face skin confidence inside the blob
    keep: np.ndarray | None


def segment_head(image: np.ndarray, models: FaceModels) -> _Head | None:
    """The head of one image, or None when the whole image has no hair or face skin blob."""
    height, width = image.shape[:2]
    seed = largest_blob(np.isin(models.segment(image).category, SEED_CLASSES))
    if seed is None:
        return None
    crop = head_crop(seed)
    confidence = models.segment(crop.cut(image)).confidence
    inside = crop.inside_image(width, height)
    head = np.clip(confidence[list(HEAD_CLASSES)].sum(axis=0), 0.0, 1.0) * inside
    blob = largest_blob(head >= EDGE_LEVEL)
    if blob is None:
        return _Head(crop, None, None, None, None)
    keep = ndimage.binary_dilation(blob, iterations=KEEP_MARGIN_PX) & inside
    return _Head(crop, confidence, head * keep, confidence[FACE_SKIN] * blob, keep)


def mirror_crop(crop: HairCrop, width: int) -> HairCrop:
    """The same square in the image mirrored left to right."""
    return HairCrop(x0=width - crop.x0 - crop.side, y0=crop.y0, side=crop.side)


def analyze_profile(
    image: np.ndarray | None, models: FaceModels, criteria: quality.ProfileCriteria | None = None
) -> ProfileResult:
    """Everything the api and the FEI check need from one upright RGB uint8 profile image.

    image None means no profile image was sent. criteria defaults to the profile
    section of shared/quality.json.
    """
    if image is None:
        return missing_profile()
    image = check_rgb(image)
    if criteria is None:
        criteria = quality.load_config().profile
    height, width = image.shape[:2]
    reasons: list[str] = []

    faces = models.detect(image)
    yaw = pose.angles(faces[0].matrix).yaw_deg if faces else None
    if yaw is not None and abs(yaw) < criteria.min_abs_yaw_deg:
        reasons.append(FRONTAL_YAW)

    h = segment_head(image, models)
    if h is None:
        return _result(width, height, bool(faces), yaw, False, False, [*reasons, NO_SILHOUETTE])
    if h.head is None:
        return _result(width, height, bool(faces), yaw, False, False, [*reasons, NO_SILHOUETTE], crop=h.crop)

    right = (yaw > 0) if yaw is not None else faces_right(h.confidence, h.keep)
    if right is None:
        return _result(width, height, bool(faces), yaw, True, False, [*reasons, NO_DIRECTION], crop=h.crop)
    flipped = not right
    if flipped:
        h = segment_head(np.ascontiguousarray(image[:, ::-1]), models)
        if h is None or h.head is None:
            crop = None if h is None else mirror_crop(h.crop, width)
            return _result(width, height, bool(faces), yaw, False, True, [*reasons, NO_SILHOUETTE], crop=crop)
    crop = h.crop
    side = crop.side
    bounds = Bounds(max(0, -crop.y0), min(side, height - crop.y0), max(0, -crop.x0), min(side, width - crop.x0))

    def to_image(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).copy()
        xy[..., 0] += crop.x0
        xy[..., 1] += crop.y0
        if flipped:
            xy[..., 0] = width - xy[..., 0]
        return xy

    image_crop = mirror_crop(crop, width) if flipped else crop
    rows = face_rows(h.face)
    if rows is None:
        return _result(width, height, bool(faces), yaw, True, flipped, [*reasons, NO_FACE_SKIN], crop=image_crop)
    s = measure_silhouette(h.head, rows, bounds, h.face)
    if s.reason is not None:
        reasons.append(s.reason)

    points = {k: (float(p[0]), float(p[1])) for k, p in ((k, to_image(v)) for k, v in s.points.items())}
    overlays = {}
    if s.reason is None:
        overlays = {
            region: Overlay("profile", {k: points[k] for k in names}, OVERLAY_LINES[region])
            for region, names in OVERLAY_POINTS.items()
        }
    return _result(
        width, height, bool(faces), yaw, True, flipped, reasons,
        legan_angle=s.legan_angle, nasofacial_angle=s.nasofacial_angle, nose_depth=s.nose_depth,
        face_height=float(rows[1] - rows[0] + 1), points=points, overlays=overlays, edge=to_image(s.edge), crop=image_crop,
    )

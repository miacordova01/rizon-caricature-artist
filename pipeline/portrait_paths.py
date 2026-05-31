"""
pipeline/portrait_paths.py
Generates labeled portrait strokes and serialises them to portrait_paths.json.

Locked coordinate schema (matches Bubble's robot pipeline):
  Origin     = bottom-left
  u-axis     = left → right,  u ∈ [0, 1]
  v-axis     = bottom → top,  v ∈ [0, 1]   ← NOTE: flipped from image-y

  MediaPipe gives (x, y) with top-left origin, y increasing downward.
  Conversion:  u = x,   v = 1.0 - y

CV pipeline contract (Bubble's spec)
-------------------------------------
  1. Points normalised to [0,1] × [0,1]           ← coordinate flip applied
  2. Points ordered in drawing direction           ← landmark order preserved
  3. Each stroke is continuous                     ← one polyline per feature
  4. Stroke boundaries = pen lifts                 ← one stroke per feature group
  5. Strokes ordered to reduce travel              ← nearest-neighbour ordering
  6. Points simplified (not thousands of noisy pts)← RDP via cv2.approxPolyDP

JSON schema v1
--------------
{
  "version": 1,
  "name": "<portrait name>",
  "source": {
    "type": "cv_lineart",
    "image_width_px": <int>,
    "image_height_px": <int>
  },
  "coordinate_system": {
    "units": "normalized_page",
    "origin": "bottom_left",
    "u_axis": "left_to_right",
    "v_axis": "bottom_to_top",
    "bounds": {"u_min": 0.0, "u_max": 1.0, "v_min": 0.0, "v_max": 1.0}
  },
  "strokes": [
    {
      "id":     "stroke_NNNN",
      "label":  "<feature name>",     # extra field for robot feature routing
      "closed": true | false,
      "points": [[u1, v1], [u2, v2], ...]
    },
    ...
  ]
}
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.artistic import (smooth_curve, extend_face_oval_to_head,
                             generate_eyelashes, generate_iris,
                             generate_hair_lines)
from vision.caricature import exaggerate
from vision.face_detector import FaceLandmarks

PortraitStroke = Dict[str, Any]


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _to_uv(pts: List[np.ndarray]) -> List[List[float]]:
    """
    Convert MediaPipe (x, y) top-left-origin points to (u, v) bottom-left-origin.
    u = x,  v = 1 - y.  Rounded to 3 dp (sub-pixel precision is meaningless).
    """
    return [[round(float(p[0]), 3), round(1.0 - float(p[1]), 3)] for p in pts]


def _closed_pts(pts: List[np.ndarray]) -> List[np.ndarray]:
    """Append first point so the contour forms a closed loop."""
    return pts + [pts[0]] if pts else pts


def _simplify(pts: List[np.ndarray], epsilon_frac: float = 0.004,
              closed: bool = False) -> List[np.ndarray]:
    """
    Ramer–Douglas–Peucker simplification via cv2.approxPolyDP.
    *epsilon_frac* is the tolerance as a fraction of the polyline perimeter.
    """
    if len(pts) < 3:
        return pts
    arr = (np.array([[p[0], p[1]] for p in pts], dtype=np.float32) * 10_000
           ).astype(np.int32).reshape(-1, 1, 2)
    eps    = cv2.arcLength(arr, closed) * epsilon_frac
    approx = cv2.approxPolyDP(arr, eps, closed)
    return [np.array([p[0][0] / 10_000, p[0][1] / 10_000]) for p in approx]


# ── Head-zone crop ────────────────────────────────────────────────────────────

def _crop_to_head_zone(
        head_outline:  List[np.ndarray],
        face_lm:       "FaceLandmarks",
        hair_above:    float = 0.80,
        side_margin:   float = 0.55,
        chin_margin:   float = 0.10) -> Optional[List[np.ndarray]]:
    """
    Remove shoulder/body points from the segmentation contour.

    Keeps only the points that fall within a generous bounding zone around
    the detected face — hair above the forehead, a little below the chin,
    and some margin on each side.  Points outside this zone are almost
    certainly neck/shoulders, which produce the ugly A-frame triangle.

    Coordinates are in MediaPipe image space: (x, y) top-left origin ∈ [0,1].

    Args:
        head_outline: Raw contour from HeadSegmenter (image-space x,y).
        face_lm:      Detected FaceLandmarks (provides bbox_min/max).
        hair_above:   How far above the face box top to include (fraction of face height).
        side_margin:  How much wider than the face box to keep on each side (fraction of fw).
        chin_margin:  How far below the face box bottom to keep (fraction of face height).

    Returns:
        Filtered contour (≥6 pts) or None if too few points survive the crop.
    """
    bmin = face_lm.bbox_min      # (x1, y1) — top-left of face box
    bmax = face_lm.bbox_max      # (x2, y2) — bottom-right
    fw   = bmax[0] - bmin[0]     # face width
    fh   = bmax[1] - bmin[1]     # face height

    x1 = max(0.0, bmin[0] - fw * side_margin)
    x2 = min(1.0, bmax[0] + fw * side_margin)
    y1 = max(0.0, bmin[1] - fh * hair_above)   # up above forehead for hair
    y2 = min(1.0, bmax[1] + fh * chin_margin)  # just below chin

    cropped = [p for p in head_outline if x1 <= float(p[0]) <= x2 and y1 <= float(p[1]) <= y2]
    return cropped if len(cropped) >= 6 else None


# ── Canvas auto-scale ──────────────────────────────────────────────────────────

def _auto_scale_strokes(
        strokes: List[PortraitStroke],
        margin:  float = 0.05) -> List[PortraitStroke]:
    """
    Uniformly scale and centre all strokes so the drawing fills the canvas.

    After exaggeration and coordinate conversion, the face often occupies
    only the centre 50-60% of the page.  This function stretches the whole
    composition to fill [margin, 1-margin] on both axes while preserving
    aspect ratio, so the portrait always looks bold and properly sized.

    Operates on (u, v) bottom-left-origin coordinates (already converted).

    Args:
        strokes: List of stroke dicts with "points" in UV space.
        margin:  Fraction of canvas to leave as border on each side.

    Returns:
        New list of stroke dicts with rescaled points.
    """
    all_pts = [pt for s in strokes for pt in s["points"]]
    if not all_pts:
        return strokes

    us    = [p[0] for p in all_pts]
    vs    = [p[1] for p in all_pts]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    u_span = u_max - u_min or 1.0
    v_span = v_max - v_min or 1.0

    fit   = 1.0 - 2 * margin
    scale = min(fit / u_span, fit / v_span)   # uniform — preserves aspect ratio

    u_mid = (u_min + u_max) / 2
    v_mid = (v_min + v_max) / 2

    result = []
    for s in strokes:
        new_pts = [
            [round((p[0] - u_mid) * scale + 0.5, 3),
             round((p[1] - v_mid) * scale + 0.5, 3)]
            for p in s["points"]
        ]
        result.append({**s, "points": new_pts})
    return result


# ── Stroke-ordering: greedy nearest-neighbour ─────────────────────────────────

def _dist2(a: Tuple[float, float], b: List[float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _order_strokes(strokes: List[PortraitStroke]) -> List[PortraitStroke]:
    """
    Reorder strokes with a greedy nearest-neighbour algorithm to minimise
    total pen-up travel distance (contract requirement 5).

    For each stroke the algorithm considers both drawing it forward and in
    reverse; it picks whichever start endpoint is closest to the current
    pen position.  Reversing a closed contour or an open line is visually
    equivalent — only direction changes, not the drawn shape.

    Strokes are re-numbered stroke_0001 … stroke_NNNN after reordering.
    """
    if len(strokes) <= 1:
        return strokes

    remaining = list(strokes)
    ordered: List[PortraitStroke] = []
    pen: Tuple[float, float] = (0.5, 0.5)   # pen starts near canvas centre

    while remaining:
        best_i     = 0
        best_dist  = math.inf
        best_flip  = False

        for i, s in enumerate(remaining):
            pts = s["points"]
            d_fwd = _dist2(pen, pts[0])
            d_rev = _dist2(pen, pts[-1])
            if d_fwd < best_dist:
                best_dist = d_fwd; best_i = i; best_flip = False
            if d_rev < best_dist:
                best_dist = d_rev; best_i = i; best_flip = True

        s = remaining.pop(best_i)
        if best_flip:
            s = {**s, "points": list(reversed(s["points"]))}
        ordered.append(s)
        pen = tuple(s["points"][-1])

    # Re-number IDs in final draw order
    for n, s in enumerate(ordered, start=1):
        s["id"] = f"stroke_{n:04d}"

    return ordered


# ── Main API ──────────────────────────────────────────────────────────────────

def generate_portrait_paths(
        landmarks:    FaceLandmarks,
        drawing_cfg:  dict,
        head_outline: Optional[List[np.ndarray]] = None,
        simplify:     bool = True) -> List[PortraitStroke]:
    """
    Build portrait strokes from *landmarks*, ready to hand to the robot.

    Processing:
      1. Caricature exaggeration applied.
      2. Each facial feature → one continuous polyline (stroke).
      3. RDP simplification reduces point count.
      4. Coordinates converted to (u, v) bottom-left origin.
      5. Strokes reordered by nearest-neighbour to minimise travel.

    Args:
        landmarks:    Detected FaceLandmarks (normalised MediaPipe coords).
        drawing_cfg:  'drawing' sub-dict from config.yaml.
        head_outline: Optional head+hair contour from HeadSegmenter (normalised).
        simplify:     Apply RDP simplification (recommended: True).

    Returns:
        Ordered list of stroke dicts in the locked portrait_paths schema.
    """
    exag   = exaggerate(landmarks, drawing_cfg)
    raw:   List[PortraitStroke] = []
    _idx   = [0]

    def add(label: str, pts: List[np.ndarray],
            closed: bool = False, smooth_n: int = 60) -> None:
        """
        Smooth → simplify → convert → append one stroke.

        smooth_n controls Catmull-Rom resolution before RDP simplification.
        Higher values preserve more curve detail; lower values draw faster.
        """
        if not pts:
            return
        # 1. Catmull-Rom spline: turns the landmark polygon into a smooth curve
        p = smooth_curve(pts, n_total=smooth_n, closed=closed) if smooth_n > 0 else list(pts)
        # 2. RDP simplification: prunes redundant points while keeping shape
        p = _simplify(p, closed=closed) if simplify else p
        if len(p) < 2:
            return
        if closed:
            p = _closed_pts(p)
        _idx[0] += 1
        raw.append({
            "id":     f"stroke_{_idx[0]:04d}",
            "label":  label,
            "closed": closed,
            "points": _to_uv(p),
        })

    def add_raw_pts(label: str, uv_pts: List[List[float]],
                    closed: bool = False) -> None:
        """Append a stroke whose points are already in UV space."""
        if len(uv_pts) < 2:
            return
        _idx[0] += 1
        raw.append({
            "id":     f"stroke_{_idx[0]:04d}",
            "label":  label,
            "closed": closed,
            "points": uv_pts,
        })

    # ── Head outline (always built from face landmarks — segmenter-independent) ──
    #
    # The image segmenter captures shoulders / full body and is unreliable for
    # the head silhouette.  Instead we extend the MediaPipe face oval upward and
    # outward to create a skull + hair dome, which is always proportional to the
    # detected face.  Segmenter output (head_outline arg) is no longer used for
    # the outline shape but can still influence hair-line anchors below.
    #
    # Build head outline from the EXAGGERATED face oval so the jaw/cheek
    # caricature carries through to the outer boundary.
    # hair_lift / side_expand tuned for a natural rounded head shape.
    head_pts    = extend_face_oval_to_head(
        exag.face_oval, landmarks.bbox_min, landmarks.bbox_max,
        hair_lift=0.38, side_expand=0.42)
    smoothed_hd = smooth_curve(head_pts, n_total=200, closed=True)
    outline     = _simplify(smoothed_hd, closed=True) if simplify else smoothed_hd
    _idx[0] += 1
    raw.append({
        "id":     f"stroke_{_idx[0]:04d}",
        "label":  "head_outline",
        "closed": True,
        "points": _to_uv(_closed_pts(outline)),
    })
    # NOTE: face_oval is intentionally NOT drawn as a separate inner stroke —
    # drawing both head_outline and face_oval creates an alien-mask look.

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    add("left_eyebrow",  exag.left_eyebrow,  smooth_n=40)
    add("right_eyebrow", exag.right_eyebrow, smooth_n=40)

    # ── Eyes (outline) ────────────────────────────────────────────────────────
    add("left_eye",  exag.left_eye,  closed=True, smooth_n=60)
    add("right_eye", exag.right_eye, closed=True, smooth_n=60)

    # ── Iris circles ─────────────────────────────────────────────────────────
    if exag.left_pupil is not None and exag.left_eye:
        iris_l = generate_iris(exag.left_pupil, exag.left_eye)
        if iris_l:
            add_raw_pts("left_iris", _to_uv(iris_l), closed=True)

    if exag.right_pupil is not None and exag.right_eye:
        iris_r = generate_iris(exag.right_pupil, exag.right_eye)
        if iris_r:
            add_raw_pts("right_iris", _to_uv(iris_r), closed=True)

    # ── Eyelashes ─────────────────────────────────────────────────────────────
    for lash in generate_eyelashes(exag.left_eye,  n_lashes=11,
                                   base_len=0.020, peak_scale=2.2):
        add_raw_pts("left_eyelash",  _to_uv(lash))
    for lash in generate_eyelashes(exag.right_eye, n_lashes=11,
                                   base_len=0.020, peak_scale=2.2):
        add_raw_pts("right_eyelash", _to_uv(lash))

    # ── Nose ─────────────────────────────────────────────────────────────────
    add("nose_bridge", exag.nose_bridge, smooth_n=20)
    add("nose_tip",    exag.nose_tip,    smooth_n=30)

    # ── Lips ─────────────────────────────────────────────────────────────────
    add("lips_outer", exag.lips_outer, closed=True, smooth_n=60)
    add("lips_inner", exag.lips_inner, closed=True, smooth_n=60)

    # ── Hair flow lines (anchored at scalp, flow down naturally) ─────────────
    # Anchors are now restricted to the scalp zone only, so max_len can be
    # long (full shoulder-length hair) without going off the canvas.
    for strand in generate_hair_lines(
            head_pts, landmarks.bbox_min, landmarks.bbox_max,
            n_per_side=8, max_len=0.40):
        add_raw_pts("hair_flow", _to_uv(strand))

    # ── Reorder strokes to minimise pen travel (contract req. 5) ──────────────
    ordered = _order_strokes(raw)

    # ── Auto-scale to fill the canvas (fixes tiny-face issue) ─────────────────
    return _auto_scale_strokes(ordered, margin=0.05)


def save_portrait_json(strokes:     List[PortraitStroke],
                       path:        str,
                       name:        str = "portrait",
                       source_type: str = "cv_lineart",
                       image_wh:    Optional[Tuple[int, int]] = None) -> None:
    """
    Write strokes to *path* in the locked JSON schema.

    Args:
        strokes:     Output of generate_portrait_paths().
        path:        Destination file path.
        name:        Portrait identifier, e.g. 'face_lineart_001'.
        source_type: Typically 'cv_lineart'.
        image_wh:    (width, height) of source frame in pixels.
    """
    w, h = image_wh if image_wh else (1024, 1024)
    payload: Dict[str, Any] = {
        "version": 1,
        "name":    name,
        "source": {
            "type":             source_type,
            "image_width_px":  int(w),
            "image_height_px": int(h),
        },
        "coordinate_system": {
            "units":   "normalized_page",
            "origin":  "bottom_left",
            "u_axis":  "left_to_right",
            "v_axis":  "bottom_to_top",
            "bounds": {"u_min": 0.0, "u_max": 1.0,
                       "v_min": 0.0, "v_max": 1.0},
        },
        "strokes": strokes,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_portrait_json(path: str) -> List[PortraitStroke]:
    """Load strokes from a saved portrait_paths JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data["strokes"]

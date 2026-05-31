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

    def add(label: str, pts: List[np.ndarray], closed: bool = False) -> None:
        if not pts:
            return
        p = _simplify(pts, closed=closed) if simplify else list(pts)
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

    # ── Head shape (outermost) ────────────────────────────────────────────────
    if head_outline:
        outline = _simplify(head_outline, closed=True) if simplify else head_outline
        _idx[0] += 1
        raw.append({
            "id":     f"stroke_{_idx[0]:04d}",
            "label":  "head_outline",
            "closed": True,
            "points": _to_uv(_closed_pts(outline)),
        })
        add("face_oval",     exag.face_oval,     closed=True)
    else:
        add("face_oval",     exag.face_oval,     closed=True)

    # ── Facial features ───────────────────────────────────────────────────────
    add("left_eyebrow",  exag.left_eyebrow)
    add("right_eyebrow", exag.right_eyebrow)
    add("left_eye",      exag.left_eye,      closed=True)
    add("right_eye",     exag.right_eye,     closed=True)
    add("nose_bridge",   exag.nose_bridge)
    add("nose_tip",      exag.nose_tip)
    add("lips_outer",    exag.lips_outer,    closed=True)
    add("lips_inner",    exag.lips_inner,    closed=True)

    # ── Reorder strokes to minimise pen travel (contract req. 5) ──────────────
    return _order_strokes(raw)


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

"""
pipeline/portrait_paths.py
Generates labeled portrait strokes and serialises them to portrait_paths.json.

Locked coordinate schema (matches Bubble's robot pipeline):
  Origin     = bottom-left
  u-axis     = left → right,  u ∈ [0, 1]
  v-axis     = bottom → top,  v ∈ [0, 1]   ← NOTE: flipped from image-y

  MediaPipe gives (x, y) with top-left origin, y increasing downward.
  Conversion:  u = x,   v = 1.0 - y

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
      "label":  "<feature name>",   # extra field for robot feature routing
      "closed": true | false,
      "points": [[u1, v1], [u2, v2], ...]
    },
    ...
  ]
}
"""

import datetime
import json
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from vision.caricature import exaggerate
from vision.face_detector import FaceLandmarks

PortraitStroke = Dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_list(pts: List[np.ndarray]) -> List[List[float]]:
    """Convert to [u, v] with bottom-left origin (v = 1 - image_y)."""
    return [[round(float(p[0]), 3), round(1.0 - float(p[1]), 3)] for p in pts]


def _closed(pts: List[np.ndarray]) -> List[np.ndarray]:
    return pts + [pts[0]] if pts else pts


def _simplify(pts: List[np.ndarray], epsilon_frac: float = 0.004,
              closed: bool = False) -> List[np.ndarray]:
    """
    Ramer–Douglas–Peucker simplification via OpenCV approxPolyDP.
    *epsilon_frac* is expressed as a fraction of the perimeter.
    """
    if len(pts) < 3:
        return pts
    arr = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
    arr = (arr * 10_000).astype(np.int32).reshape(-1, 1, 2)
    peri = cv2.arcLength(arr, closed)
    eps  = peri * epsilon_frac
    approx = cv2.approxPolyDP(arr, eps, closed)
    return [np.array([p[0][0] / 10_000, p[0][1] / 10_000]) for p in approx]


# ── Main API ──────────────────────────────────────────────────────────────────

def generate_portrait_paths(
        landmarks:    FaceLandmarks,
        drawing_cfg:  dict,
        head_outline: Optional[List[np.ndarray]] = None,
        simplify:     bool = True) -> List[PortraitStroke]:
    """
    Build labeled portrait strokes from *landmarks*.

    All coordinates are normalised [0, 1] image-space (origin top-left).
    Caricature exaggeration is applied before output.

    Args:
        landmarks:    Detected FaceLandmarks.
        drawing_cfg:  'drawing' sub-dict from config.yaml.
        head_outline: Optional head+hair contour from HeadSegmenter.
        simplify:     Apply RDP simplification to reduce waypoint count.

    Returns:
        List of stroke dicts in the locked portrait_paths JSON schema.
    """
    exag = exaggerate(landmarks, drawing_cfg)
    strokes: List[PortraitStroke] = []

    def add(label: str, pts: List[np.ndarray], closed: bool = False) -> None:
        if not pts:
            return
        p = _simplify(pts, closed=closed) if simplify else pts
        if len(p) < 2:
            return
        if closed:
            p = _closed(p)
        strokes.append({"label": label, "closed": closed, "points": _to_list(p)})

    stroke_idx = [0]   # mutable counter for stroke IDs

    def add(label: str, pts: List[np.ndarray], closed: bool = False) -> None:
        if not pts:
            return
        p = _simplify(pts, closed=closed) if simplify else pts
        if len(p) < 2:
            return
        if closed:
            p = _closed(p)
        stroke_idx[0] += 1
        strokes.append({
            "id":     f"stroke_{stroke_idx[0]:04d}",
            "label":  label,
            "closed": closed,
            "points": _to_list(p),
        })

    # ── Outer head shape ──────────────────────────────────────────────────────
    if head_outline:
        outline = _simplify(head_outline, closed=True) if simplify else head_outline
        stroke_idx[0] += 1
        strokes.append({
            "id":     f"stroke_{stroke_idx[0]:04d}",
            "label":  "head_outline",
            "closed": True,
            "points": _to_list(_closed(outline)),
        })
        add("face_oval",      exag.face_oval,      closed=True)
    else:
        add("face_oval",      exag.face_oval,      closed=True)

    # ── Facial features ───────────────────────────────────────────────────────
    add("left_eyebrow",   exag.left_eyebrow)
    add("right_eyebrow",  exag.right_eyebrow)
    add("left_eye",       exag.left_eye,       closed=True)
    add("right_eye",      exag.right_eye,      closed=True)
    add("nose_bridge",    exag.nose_bridge)
    add("nose_tip",       exag.nose_tip)
    add("lips_outer",     exag.lips_outer,     closed=True)
    add("lips_inner",     exag.lips_inner,     closed=True)

    return strokes


def save_portrait_json(strokes:     List[PortraitStroke],
                       path:        str,
                       name:        str = "portrait",
                       source_type: str = "cv_lineart",
                       image_wh:    Optional[tuple] = None) -> None:
    """
    Serialise strokes to *path* in the locked schema (matches Bubble's robot pipeline).

    Args:
        strokes:     Output of generate_portrait_paths().
        path:        Output file path.
        name:        Portrait name, e.g. 'face_lineart_001'.
        source_type: Typically 'cv_lineart'.
        image_wh:    (width, height) of source image in pixels, or None.
    """
    w, h = (image_wh[0], image_wh[1]) if image_wh else (1024, 1024)
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
            "bounds": {
                "u_min": 0.0, "u_max": 1.0,
                "v_min": 0.0, "v_max": 1.0,
            },
        },
        "strokes": strokes,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_portrait_json(path: str) -> List[PortraitStroke]:
    """Load strokes from a saved portrait_paths JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Restore points as plain lists — downstream can convert to np.ndarray if needed
    return data["strokes"]

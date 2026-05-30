"""
pipeline/portrait_paths.py
Generates labeled portrait strokes in normalised [0, 1] image coordinates
and serialises them to portrait_paths.json.

Coordinate convention (locked schema):
  Origin = top-left of the captured frame
  x ∈ [0, 1]  increases rightward
  y ∈ [0, 1]  increases downward
  (These are the raw MediaPipe normalised coords — no robot transform applied.)

JSON schema v1.0
----------------
{
  "version": "1.0",
  "timestamp": "<ISO-8601>",
  "source": "webcam" | "static_image",
  "image_shape": [height, width],
  "strokes": [
    {
      "label":  "<feature name>",
      "closed": true | false,
      "points": [[x1, y1], [x2, y2], ...]   # normalised [0, 1]
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
    return [[round(float(p[0]), 5), round(float(p[1]), 5)] for p in pts]


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

    # ── Outer head shape ──────────────────────────────────────────────────────
    if head_outline:
        outline = _simplify(head_outline, closed=True) if simplify else head_outline
        strokes.append({
            "label":  "head_outline",
            "closed": True,
            "points": _to_list(_closed(outline)),
        })
        # Inner face/jaw boundary (visible inside the hair)
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


def save_portrait_json(strokes:    List[PortraitStroke],
                       path:       str,
                       source:     str = "webcam",
                       image_shape: Optional[tuple] = None) -> None:
    """Serialise strokes to *path* in the locked JSON schema."""
    payload: Dict[str, Any] = {
        "version":     "1.0",
        "timestamp":   datetime.datetime.now().isoformat(),
        "source":      source,
        "strokes":     strokes,
    }
    if image_shape:
        payload["image_shape"] = list(image_shape[:2])   # [height, width]

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_portrait_json(path: str) -> List[PortraitStroke]:
    """Load strokes from a saved portrait_paths JSON file."""
    with open(path) as f:
        data = json.load(f)
    # Restore points as plain lists — downstream can convert to np.ndarray if needed
    return data["strokes"]

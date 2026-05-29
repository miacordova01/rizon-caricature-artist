"""
vision/stroke_generator.py
Converts a FaceLandmarks object into a list of pen strokes for the robot.

A stroke is a list of (u, v) canvas-coordinate tuples, where u, v ∈ [-1, 1].
Closed contours (face oval, eyes, lips) repeat their first point at the end
so the robot draws a continuous loop.

Draw order (roughly front-to-back on the paper):
  1. Face oval
  2. Eyebrows
  3. Eyes (closed)
  4. Nose bridge
  5. Nose tip
  6. Outer lips (closed)
  7. Inner lips (closed)
"""

from typing import List, Tuple

import numpy as np

from utils.geometry import CanvasTransform
from vision.face_detector import FaceLandmarks

Stroke = List[Tuple[float, float]]


def _to_canvas(pts: List[np.ndarray], transform: CanvasTransform,
               scale: float) -> Stroke:
    """
    Convert a list of normalised landmark points (x, y ∈ [0, 1]) to
    canvas (u, v) coordinates.

    Landmarks are already fractional image coords so we pass img_w=img_h=1.
    """
    return [transform.pixel_to_canvas(float(p[0]), float(p[1]), 1, 1, scale)
            for p in pts]


def _closed(stroke: Stroke) -> Stroke:
    """Append the first point so the contour closes cleanly."""
    return stroke + [stroke[0]] if stroke else stroke


def generate(landmarks: FaceLandmarks, transform: CanvasTransform,
             drawing_cfg: dict) -> List[Stroke]:
    """
    Build a list of pen strokes from *landmarks*.

    Args:
        landmarks:   FaceLandmarks (may already be caricature-exaggerated).
        transform:   CanvasTransform for this session's canvas geometry.
        drawing_cfg: Sub-dict from config.yaml under key 'drawing'.

    Returns:
        List of strokes.  Each stroke is a list of (u, v) tuples.
    """
    scale = drawing_cfg.get('scale', 0.85)
    strokes: List[Stroke] = []

    def add(pts, close=False):
        if not pts:
            return
        s = _to_canvas(pts, transform, scale)
        if close:
            s = _closed(s)
        strokes.append(s)

    add(landmarks.face_oval,      close=True)
    add(landmarks.left_eyebrow)
    add(landmarks.right_eyebrow)
    add(landmarks.left_eye,       close=True)
    add(landmarks.right_eye,      close=True)
    add(landmarks.nose_bridge)
    add(landmarks.nose_tip)
    add(landmarks.lips_outer,     close=True)
    add(landmarks.lips_inner,     close=True)

    return strokes

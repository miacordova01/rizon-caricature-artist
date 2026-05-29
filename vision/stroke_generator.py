"""
vision/stroke_generator.py
Converts a FaceLandmarks object (+ optional head silhouette) into pen strokes.

A stroke is a list of (u, v) canvas-coordinate tuples where u, v ∈ [-1, 1].
Closed contours repeat their first point at the end.

Draw order
----------
1. Head silhouette  — from HeadSegmenter (includes hair).  Falls back to face
                      oval if segmentation was not run or failed.
2. Face oval        — inner face / jaw outline (drawn when head outline exists,
                      so the face boundary is visible inside the hair shape).
3. Left eyebrow
4. Right eyebrow
5. Left eye         (closed)
6. Right eye        (closed)
7. Nose bridge
8. Nose tip
9. Outer lips       (closed)
10. Inner lips      (closed)
"""

from typing import List, Optional, Tuple

import numpy as np

from utils.geometry import CanvasTransform
from vision.face_detector import FaceLandmarks

Stroke = List[Tuple[float, float]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_canvas(pts: List[np.ndarray], transform: CanvasTransform,
               scale: float) -> Stroke:
    """
    Convert normalised landmark / contour points (x, y ∈ [0, 1]) to
    canvas (u, v) coordinates.  We pass img_w = img_h = 1 because the
    points are already fractional image coordinates.
    """
    return [transform.pixel_to_canvas(float(p[0]), float(p[1]), 1, 1, scale)
            for p in pts]


def _closed(stroke: Stroke) -> Stroke:
    return stroke + [stroke[0]] if stroke else stroke


# ── Public API ────────────────────────────────────────────────────────────────

def generate(landmarks: FaceLandmarks,
             transform: CanvasTransform,
             drawing_cfg: dict,
             head_outline: Optional[List[np.ndarray]] = None) -> List[Stroke]:
    """
    Build the full list of pen strokes.

    Args:
        landmarks:    FaceLandmarks (may already be caricature-exaggerated).
        transform:    CanvasTransform for this session's canvas geometry.
        drawing_cfg:  Sub-dict from config.yaml under key 'drawing'.
        head_outline: Optional list of normalised (x, y) points from
                      HeadSegmenter.get_outline() — includes hair.
                      Pass None to fall back to the face oval only.

    Returns:
        Ordered list of strokes (each a list of (u, v) tuples).
    """
    scale   = drawing_cfg.get('scale', 0.85)
    strokes: List[Stroke] = []

    def add(pts, close=False):
        if not pts:
            return
        s = _to_canvas(pts, transform, scale)
        strokes.append(_closed(s) if close else s)

    # ── Head shape (outermost stroke) ─────────────────────────────────────────
    if head_outline:
        # Full head + hair silhouette from selfie segmentation
        strokes.append(_closed(_to_canvas(head_outline, transform, scale)))
        # Draw the face oval too so the chin / jaw line reads clearly
        # inside the hair shape
        add(landmarks.face_oval, close=True)
    else:
        # No segmentation available — face oval is the outermost line
        add(landmarks.face_oval, close=True)

    # ── Facial features ───────────────────────────────────────────────────────
    add(landmarks.left_eyebrow)
    add(landmarks.right_eyebrow)
    add(landmarks.left_eye,   close=True)
    add(landmarks.right_eye,  close=True)
    add(landmarks.nose_bridge)
    add(landmarks.nose_tip)
    add(landmarks.lips_outer, close=True)
    add(landmarks.lips_inner, close=True)

    return strokes

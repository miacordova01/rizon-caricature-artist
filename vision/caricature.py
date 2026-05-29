"""
vision/caricature.py
Multi-feature caricature exaggeration applied to a FaceLandmarks object.

Five features are exaggerated simultaneously — the same combination a human
caricature artist would reach for:

  Eyes       — enlarged radially from each pupil centre
  Eyebrows   — pushed outward / upward from each pupil (raises and arches)
  Nose       — nose tip scaled outward from the nasal root
  Mouth      — lips stretched horizontally from their shared centroid
  Jaw / chin — lower face elongated downward from the nose level
               (square jaw → squarer, pointed chin → pointier)

All scale factors are configurable in config.yaml under the 'drawing' key.
"""

import copy
from typing import List

import numpy as np

from vision.face_detector import FaceLandmarks


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _scale_pts(pts: List[np.ndarray], center: np.ndarray,
               scale: float) -> List[np.ndarray]:
    """Radial scale: every point moves away from *center* by *scale*."""
    return [center + (p - center) * scale for p in pts]


def _scale_pts_x(pts: List[np.ndarray], center: np.ndarray,
                 scale: float) -> List[np.ndarray]:
    """Horizontal-only scale: stretches along the X axis from *center*."""
    result = []
    for p in pts:
        d = p.copy() - center
        d[0] *= scale
        result.append(center + d)
    return result


def _exaggerate_jaw(face_oval: List[np.ndarray],
                    nose_y_ref: float,
                    jaw_scale: float) -> List[np.ndarray]:
    """
    Elongate the lower face by scaling every oval point that sits below
    *nose_y_ref* (in image-Y, so higher numerical value = lower on screen).
    Points above the nose are untouched so forehead/cheek structure is preserved.
    """
    result = []
    for p in face_oval:
        if p[1] > nose_y_ref:
            excess = p[1] - nose_y_ref
            q = p.copy()
            q[1] = nose_y_ref + excess * jaw_scale
            result.append(q)
        else:
            result.append(p.copy())
    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def exaggerate(landmarks: FaceLandmarks, drawing_cfg: dict) -> FaceLandmarks:
    """
    Return a deep copy of *landmarks* with all caricature exaggerations applied.

    Args:
        landmarks:   Detected face landmarks (normalised coords, x/y ∈ [0, 1]).
        drawing_cfg: Sub-dict from config.yaml under key 'drawing'.
    """
    eye_scale   = drawing_cfg.get('caricature_eye_scale',   1.60)
    brow_scale  = drawing_cfg.get('caricature_brow_lift',   1.25)
    nose_scale  = drawing_cfg.get('caricature_nose_scale',  1.35)
    mouth_scale = drawing_cfg.get('caricature_mouth_scale', 1.50)
    jaw_scale   = drawing_cfg.get('caricature_jaw_scale',   1.15)

    result = copy.deepcopy(landmarks)

    # ── Eyes: radial expansion from each pupil ────────────────────────────────
    if landmarks.left_pupil is not None and landmarks.left_eye:
        result.left_eye = _scale_pts(landmarks.left_eye,
                                     landmarks.left_pupil, eye_scale)
    if landmarks.right_pupil is not None and landmarks.right_eye:
        result.right_eye = _scale_pts(landmarks.right_eye,
                                      landmarks.right_pupil, eye_scale)

    # ── Eyebrows: push away from pupils (raises + arches them) ───────────────
    # Scaling eyebrow points from the pupil moves the entire brow upward and
    # outward, creating the "raised eyebrow" look typical of caricatures.
    if landmarks.left_pupil is not None and landmarks.left_eyebrow:
        result.left_eyebrow = _scale_pts(landmarks.left_eyebrow,
                                         landmarks.left_pupil, brow_scale)
    if landmarks.right_pupil is not None and landmarks.right_eyebrow:
        result.right_eyebrow = _scale_pts(landmarks.right_eyebrow,
                                          landmarks.right_pupil, brow_scale)

    # ── Nose: expand tip from nasal root ─────────────────────────────────────
    if landmarks.nose_bridge and landmarks.nose_tip:
        nasal_root = landmarks.nose_bridge[0]   # landmark 168 — between the eyes
        result.nose_tip = _scale_pts(landmarks.nose_tip, nasal_root, nose_scale)

    # ── Mouth: horizontal stretch from lip centroid ───────────────────────────
    all_lip_pts = landmarks.lips_outer + landmarks.lips_inner
    if all_lip_pts:
        centroid = np.mean(all_lip_pts, axis=0)
        result.lips_outer = _scale_pts_x(landmarks.lips_outer, centroid, mouth_scale)
        result.lips_inner = _scale_pts_x(landmarks.lips_inner, centroid, mouth_scale)

    # ── Jaw / chin: elongate lower face ──────────────────────────────────────
    if landmarks.face_oval and landmarks.nose_bridge:
        # Use the bottom of the nose bridge as the y pivot so cheekbones stay fixed.
        nose_y = float(landmarks.nose_bridge[-1][1])
        result.face_oval = _exaggerate_jaw(landmarks.face_oval, nose_y, jaw_scale)

    return result

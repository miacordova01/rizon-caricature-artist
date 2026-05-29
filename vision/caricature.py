"""
vision/caricature.py
Applies caricature exaggeration to a FaceLandmarks object.

Three features are exaggerated:
  - Eyes   : scaled outward from each pupil centre
  - Nose   : scaled outward from the nasal root (top of nose bridge)
  - Mouth  : stretched horizontally from the lip centroid
"""

import copy
import numpy as np
from typing import List

from vision.face_detector import FaceLandmarks


def _scale_pts(pts: List[np.ndarray], center: np.ndarray,
               scale: float) -> List[np.ndarray]:
    return [center + (p - center) * scale for p in pts]


def _scale_pts_x(pts: List[np.ndarray], center: np.ndarray,
                 scale: float) -> List[np.ndarray]:
    result = []
    for p in pts:
        d = p - center
        d = d.copy()
        d[0] *= scale          # X only (horizontal stretch)
        result.append(center + d)
    return result


def exaggerate(landmarks: FaceLandmarks, drawing_cfg: dict) -> FaceLandmarks:
    """
    Return a deep copy of *landmarks* with exaggerated facial features.

    Args:
        landmarks:   Detected face landmarks (normalised coords).
        drawing_cfg: Sub-dict from config.yaml under key 'drawing'.
    """
    eye_scale   = drawing_cfg.get('caricature_eye_scale',   1.45)
    nose_scale  = drawing_cfg.get('caricature_nose_scale',  1.20)
    mouth_scale = drawing_cfg.get('caricature_mouth_scale', 1.25)

    result = copy.deepcopy(landmarks)

    # ── Eyes: expand every contour point radially from pupil centre ──────────
    if landmarks.left_pupil is not None and landmarks.left_eye:
        result.left_eye = _scale_pts(landmarks.left_eye,
                                     landmarks.left_pupil, eye_scale)
    if landmarks.right_pupil is not None and landmarks.right_eye:
        result.right_eye = _scale_pts(landmarks.right_eye,
                                      landmarks.right_pupil, eye_scale)

    # ── Nose: expand tip region from nasal-root anchor (top of bridge) ───────
    if landmarks.nose_bridge and landmarks.nose_tip:
        nasal_root = landmarks.nose_bridge[0]
        result.nose_tip = _scale_pts(landmarks.nose_tip, nasal_root, nose_scale)

    # ── Mouth: stretch lips horizontally from their shared centroid ───────────
    all_lip_pts = landmarks.lips_outer + landmarks.lips_inner
    if all_lip_pts:
        centroid = np.mean(all_lip_pts, axis=0)
        result.lips_outer = _scale_pts_x(landmarks.lips_outer, centroid, mouth_scale)
        result.lips_inner = _scale_pts_x(landmarks.lips_inner, centroid, mouth_scale)

    return result

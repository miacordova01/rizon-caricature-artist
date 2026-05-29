"""
vision/face_detector.py
Face landmark detection using MediaPipe Face Mesh.

Returns normalised landmark coordinates (x, y ∈ [0,1]) relative to the
image frame.  All downstream code should use these normalised values and
convert to canvas coordinates via CanvasTransform.pixel_to_canvas().
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List

log = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    log.warning("MediaPipe not installed — face detection unavailable. "
                "Install with: pip install mediapipe")


# ── MediaPipe landmark index groups ──────────────────────────────────────────
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109]

LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]

LEFT_EYEBROW  = [70,63,105,66,107,55,65,52,53,46]
RIGHT_EYEBROW = [300,293,334,296,336,285,295,282,283,276]

NOSE_BRIDGE  = [168,6,197,195,5]
NOSE_TIP     = [1,2,98,327,326,2]
NOSE_BOTTOM  = [94,2,370,94]

LIPS_OUTER = [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
LIPS_INNER = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

LEFT_PUPIL  = 468   # available with refine_landmarks=True
RIGHT_PUPIL = 473


@dataclass
class FaceLandmarks:
    """Normalised (x, y) landmark positions for one detected face."""
    face_oval:      List[np.ndarray] = field(default_factory=list)
    left_eye:       List[np.ndarray] = field(default_factory=list)
    right_eye:      List[np.ndarray] = field(default_factory=list)
    left_eyebrow:   List[np.ndarray] = field(default_factory=list)
    right_eyebrow:  List[np.ndarray] = field(default_factory=list)
    nose_bridge:    List[np.ndarray] = field(default_factory=list)
    nose_tip:       List[np.ndarray] = field(default_factory=list)
    lips_outer:     List[np.ndarray] = field(default_factory=list)
    lips_inner:     List[np.ndarray] = field(default_factory=list)
    left_pupil:     Optional[np.ndarray] = None
    right_pupil:    Optional[np.ndarray] = None
    # Bounding box of the face in normalised coords
    bbox_min:       np.ndarray = field(default_factory=lambda: np.zeros(2))
    bbox_max:       np.ndarray = field(default_factory=lambda: np.ones(2))


class FaceDetector:
    """Detects face landmarks in a BGR image using MediaPipe Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        if not _MP_AVAILABLE:
            raise ImportError("MediaPipe is required. pip install mediapipe")

        mp_fm = mp.solutions.face_mesh
        self._mesh = mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # enables iris / pupil landmarks
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        log.info("FaceDetector initialised")

    def detect(self, bgr_frame: np.ndarray) -> Optional[FaceLandmarks]:
        """
        Run face mesh on a BGR frame.

        Returns a FaceLandmarks dataclass with normalised (x, y) coords
        for each facial feature, or None if no face was found.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        raw = results.multi_face_landmarks[0].landmark
        h, w = bgr_frame.shape[:2]

        def pts(indices):
            return [np.array([raw[i].x, raw[i].y]) for i in indices]

        lm = FaceLandmarks(
            face_oval     = pts(FACE_OVAL),
            left_eye      = pts(LEFT_EYE),
            right_eye     = pts(RIGHT_EYE),
            left_eyebrow  = pts(LEFT_EYEBROW),
            right_eyebrow = pts(RIGHT_EYEBROW),
            nose_bridge   = pts(NOSE_BRIDGE),
            nose_tip      = pts(NOSE_TIP),
            lips_outer    = pts(LIPS_OUTER),
            lips_inner    = pts(LIPS_INNER),
        )

        # Iris centres if refined landmarks are present
        if len(raw) > LEFT_PUPIL:
            lm.left_pupil  = np.array([raw[LEFT_PUPIL].x,  raw[LEFT_PUPIL].y])
            lm.right_pupil = np.array([raw[RIGHT_PUPIL].x, raw[RIGHT_PUPIL].y])

        # Bounding box from face oval
        all_pts = np.array(lm.face_oval)
        lm.bbox_min = all_pts.min(axis=0)
        lm.bbox_max = all_pts.max(axis=0)

        return lm

    def close(self) -> None:
        self._mesh.close()

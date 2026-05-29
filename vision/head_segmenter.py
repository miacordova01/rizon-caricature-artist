"""
vision/head_segmenter.py
Extracts the full head and hair silhouette from a BGR frame using
MediaPipe Selfie Segmentation.  Returns a simplified closed contour
as normalised (x, y ∈ [0,1]) points, ready to be passed into
stroke_generator.generate() as the head_outline argument.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    log.warning("MediaPipe not installed — head segmentation unavailable")


class HeadSegmenter:
    """Produces a head-outline stroke that includes hair."""

    def __init__(self, confidence: float = 0.55, max_pts: int = 80):
        """
        Args:
            confidence: Mask threshold (0–1). Higher = tighter crop around person.
            max_pts:    Target contour points after simplification (~60–100 works well).
        """
        if not _MP_AVAILABLE:
            raise ImportError("MediaPipe required. pip install mediapipe")
        seg = mp.solutions.selfie_segmentation
        # model_selection=1 is the landscape/general model — better for full-body frames
        self._seg = seg.SelfieSegmentation(model_selection=1)
        self.confidence = confidence
        self.max_pts    = max_pts

    def get_outline(self, bgr_frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Run segmentation on *bgr_frame* and return the head/hair outline.

        Returns a list of normalised (x, y) np.ndarray points (closed contour),
        or None if segmentation fails.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._seg.process(rgb)

        if result.segmentation_mask is None:
            log.warning("Selfie segmentation returned no mask")
            return None

        # Binary mask — person vs background
        mask = (result.segmentation_mask > self.confidence).astype(np.uint8) * 255

        # Morphological close fills small gaps at hair edges (wispy hair, etc.)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log.warning("No contours found in segmentation mask")
            return None

        # The largest contour is the person (head + body)
        main = max(contours, key=cv2.contourArea)

        # Ramer–Douglas–Peucker simplification to target point count
        perim   = cv2.arcLength(main, True)
        epsilon = perim / self.max_pts
        approx  = cv2.approxPolyDP(main, epsilon, True)

        pts = [np.array([p[0][0] / w, p[0][1] / h], dtype=float) for p in approx]
        log.debug("Head outline: %d points simplified from %d raw", len(pts), len(main))
        return pts

    def close(self) -> None:
        self._seg.close()

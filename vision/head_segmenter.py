"""
vision/head_segmenter.py
Extracts the full head and hair silhouette from a BGR frame using
MediaPipe ImageSegmenter (Tasks API).

Compatible with mediapipe >= 0.10.30 (Apple Silicon / Python 3.11+).
Requires models/selfie_segmenter.tflite — download with:
    python3 models/download_models.py

Returns a simplified closed contour as normalised (x, y ∈ [0,1]) points,
ready to be passed into stroke_generator.generate() as head_outline.
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__),
                               '..', 'models', 'selfie_segmenter.tflite')

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    log.warning("MediaPipe not installed — head segmentation unavailable")


class HeadSegmenter:
    """Produces a head-outline stroke that includes hair."""

    def __init__(self, model_path: Optional[str] = None,
                 confidence: float = 0.55, max_pts: int = 80):
        """
        Args:
            model_path:  Path to selfie_segmenter.tflite (default: models/).
            confidence:  Mask threshold (0–1). Higher = tighter crop.
            max_pts:     Target contour points after RDP simplification.
        """
        if not _MP_AVAILABLE:
            raise ImportError("MediaPipe required. pip install mediapipe")

        model = os.path.abspath(model_path or _DEFAULT_MODEL)
        if not os.path.exists(model):
            raise FileNotFoundError(
                f"Model file not found: {model}\n"
                "Run:  python3 models/download_models.py")

        ImageSegmenter        = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        BaseOptions           = mp.tasks.BaseOptions
        RunningMode           = mp.tasks.vision.RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self._segmenter = ImageSegmenter.create_from_options(options)
        self.confidence = confidence
        self.max_pts    = max_pts
        log.info("HeadSegmenter initialised (Tasks API)")

    def get_outline(self, bgr_frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Return the head/hair outline as normalised (x, y) points, or None.

        The contour is a closed polygon suitable for passing directly to
        stroke_generator.generate() as the head_outline argument.
        """
        h, w = bgr_frame.shape[:2]
        rgb      = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self._segmenter.segment(mp_image)

        masks = result.confidence_masks
        if not masks:
            log.warning("Segmenter returned no confidence masks")
            return None

        # selfie_segmenter.tflite: mask[0] = background confidence.
        # Person confidence = 1 - background.
        # If model returns 2 masks, mask[1] is person directly.
        if len(masks) >= 2:
            person_arr = masks[1].numpy_view()
        else:
            person_arr = 1.0 - masks[0].numpy_view()

        binary = (person_arr > self.confidence).astype(np.uint8) * 255

        # Morphological close — fills gaps at hair edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log.warning("No contours in segmentation mask")
            return None

        main  = max(contours, key=cv2.contourArea)
        perim = cv2.arcLength(main, True)
        eps   = perim / self.max_pts
        approx = cv2.approxPolyDP(main, eps, True)

        pts = [np.array([p[0][0] / w, p[0][1] / h], dtype=float) for p in approx]
        log.debug("Head outline: %d points", len(pts))
        return pts

    def close(self) -> None:
        self._segmenter.close()

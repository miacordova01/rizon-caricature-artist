"""
vision/camera.py
Camera interface — supports standard OpenCV cameras and Intel RealSense D405.
"""

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


class Camera:
    """
    Wraps an OpenCV VideoCapture.  Falls back gracefully if RealSense
    is not installed or not connected.
    """

    def __init__(self, device_id: int = 0, width: int = 1280,
                 height: int = 720, fps: int = 30):
        self.device_id = device_id
        self.width     = width
        self.height    = height
        self.fps       = fps
        self._cap      = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)
        log.info("Camera %d opened (%dx%d @ %d fps)",
                 self.device_id, self.width, self.height, self.fps)

    def read(self) -> np.ndarray:
        """Return the latest BGR frame, or raise RuntimeError on failure."""
        if self._cap is None:
            raise RuntimeError("Camera not opened — call open() first")
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera closed")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

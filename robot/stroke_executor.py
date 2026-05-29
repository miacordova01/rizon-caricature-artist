"""
robot/stroke_executor.py
Iterates over a list of pen strokes and drives PenController for each one.

Stroke format: List[List[Tuple[float, float]]]
  Outer list  = strokes (one per facial feature contour or line)
  Inner list  = (u, v) canvas-coordinate waypoints within a stroke

Draw sequence per stroke:
  1. Travel to stroke start   (pen up, at travel speed)
  2. Pen down                 (approach speed, move to canvas surface)
  3. Draw through waypoints   (drawing speed, pen on canvas)
  4. Pen up                   (approach speed, lift off canvas)
"""

import logging
from typing import Callable, List, Optional, Tuple

from robot.pen_controller import PenController
from utils.geometry import CanvasTransform

log = logging.getLogger(__name__)

Stroke = List[Tuple[float, float]]


class StrokeExecutor:

    def __init__(self, pen_ctrl: PenController,
                 transform: CanvasTransform, cfg: dict):
        self.ctrl      = pen_ctrl
        self.transform = transform
        self.margin    = cfg['drawing']['canvas_margin']

    def execute_strokes(
            self,
            strokes: List[Stroke],
            progress_cb: Optional[Callable[[int, int], None]] = None,
            stop_flag: Optional[Callable[[], bool]] = None) -> None:
        """
        Execute all strokes in order.

        Args:
            strokes:     List of strokes from stroke_generator.generate().
            progress_cb: Called after each stroke with (completed, total).
            stop_flag:   If provided and returns True, drawing halts early.
        """
        total = len(strokes)
        log.info("Executing %d strokes", total)

        for idx, stroke in enumerate(strokes):
            if stop_flag and stop_flag():
                log.info("Stop requested — aborting drawing at stroke %d", idx)
                break

            if len(stroke) < 2:
                log.debug("Stroke %d has fewer than 2 points — skipped", idx)
                continue

            # Clamp every waypoint so pen stays inside canvas bounds.
            clamped = [
                self.transform.clamp_canvas(u, v, self.margin)
                for u, v in stroke
            ]

            start = clamped[0]
            rest  = clamped[1:]

            log.debug("Stroke %d: %d pts, start=(%.3f, %.3f)",
                      idx, len(clamped), *start)

            self.ctrl.travel_to(*start)
            self.ctrl.pen_down()
            for pt in rest:
                self.ctrl.draw_to(*pt)
            self.ctrl.pen_up()

            if progress_cb:
                progress_cb(idx + 1, total)

        log.info("All strokes complete")

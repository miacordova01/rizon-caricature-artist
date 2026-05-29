"""
utils/geometry.py
Coordinate transforms between canvas space and robot world frame.

Canvas frame:
  Origin  = canvas centre   (robot world: CANVAS_POS)
  +U axis = robot world +Y  (left when facing canvas)
  +V axis = robot world +Z  (up)

Robot world frame is the Flexiv base frame (X forward, Y left, Z up).
"""

import numpy as np
from typing import List, Tuple


class CanvasTransform:
    """Maps 2D canvas coordinates (u, v) ↔ 3D robot world coordinates (x, y, z)."""

    def __init__(self, canvas_pos: List[float], canvas_width: float,
                 canvas_height: float, pen_tip_length: float):
        """
        Args:
            canvas_pos:     [x, y, z] of canvas centre in robot world frame (m).
            canvas_width:   Canvas width  in Y direction (m).
            canvas_height:  Canvas height in Z direction (m).
            pen_tip_length: Distance from flange origin to pen tip along flange Z (m).
        """
        self.pos    = np.array(canvas_pos, dtype=float)
        self.width  = canvas_width
        self.height = canvas_height

        # When pen points in world +X, the TCP (flange origin) must sit
        # pen_tip_length behind the canvas surface so the tip touches it.
        self.tcp_x_on_canvas  = self.pos[0] - pen_tip_length
        self.tcp_x_pen_lifted = None   # set by set_lift_distance()

    def set_lift_distance(self, lift_m: float) -> None:
        """Retract this far from canvas in X direction when pen is up."""
        self.tcp_x_pen_lifted = self.tcp_x_on_canvas - lift_m

    def canvas_to_robot(self, u: float, v: float,
                        pen_down: bool = True) -> np.ndarray:
        """
        Convert canvas (u, v) in normalised coordinates [-1, 1] to robot TCP pose.

        Returns:
            np.ndarray [x, y, z] of the TCP (flange) position in world frame.
        """
        y_world = self.pos[1] + u * (self.width  / 2)
        z_world = self.pos[2] + v * (self.height / 2)
        x_world = self.tcp_x_on_canvas if pen_down else self.tcp_x_pen_lifted
        return np.array([x_world, y_world, z_world])

    def pixel_to_canvas(self, px: float, py: float,
                        img_w: int, img_h: int,
                        scale: float = 0.85) -> Tuple[float, float]:
        """
        Convert image pixel (px, py) to normalised canvas (u, v) ∈ [-1, 1].

        Image coordinates: origin top-left, +x right, +y down.
        Canvas coordinates: origin centre, +u right (robot +Y), +v up (robot +Z).

        Args:
            scale: Draw face at this fraction of canvas extent (avoids edge clipping).
        """
        u =  ((px / img_w) - 0.5) * 2.0 * scale
        v = -((py / img_h) - 0.5) * 2.0 * scale   # flip Y so image-down → canvas-down
        return u, v

    def clamp_canvas(self, u: float, v: float,
                     margin_m: float = 0.025) -> Tuple[float, float]:
        """Clamp (u, v) so the pen tip stays margin_m inside canvas edges."""
        u_lim = 1.0 - 2 * margin_m / self.width
        v_lim = 1.0 - 2 * margin_m / self.height
        return float(np.clip(u, -u_lim, u_lim)), float(np.clip(v, -v_lim, v_lim))


def pose_from_position(xyz: np.ndarray,
                       orientation_deg: List[float]) -> List[float]:
    """
    Build a Flexiv 6-DOF pose list [x, y, z, Rx, Ry, Rz] from a position
    and a fixed orientation expressed as intrinsic Euler angles in degrees.

    For pen-on-canvas, orientation_deg = [0, 90, 0] points flange-Z → world-X.
    """
    return [xyz[0], xyz[1], xyz[2],
            orientation_deg[0], orientation_deg[1], orientation_deg[2]]


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))

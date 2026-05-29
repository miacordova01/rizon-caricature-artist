"""
robot/pen_controller.py
Low-level pen motion primitives for the Flexiv Rizon 4s.

All robot moves use NRT_PRIMITIVE_EXECUTION / MoveL.  The pen is considered
"down" when its TCP sits at tcp_x_on_canvas (touching the paper surface) and
"up" when retracted by lift_distance in the –X direction.

Coordinate convention (matches CanvasTransform):
  Robot base frame  X forward, Y left, Z up.
  Canvas normal     –X (canvas faces the robot).
  Pen orientation   Euler [Rx=0, Ry=90, Rz=0] deg — flange-Z aligns with +X.
"""

import logging
import time

from utils.geometry import CanvasTransform, pose_from_position

log = logging.getLogger(__name__)


# Safe home: pen centred on canvas in Y/Z, retracted 200 mm from canvas in X.
_HOME_XYZ = None   # computed lazily from transform in go_home()


class PenController:
    """Drives the robot pen through approach, draw, and lift motions."""

    def __init__(self, robot, transform: CanvasTransform, cfg: dict):
        self.robot     = robot
        self.transform = transform
        self.orient    = cfg['pen']['drawing_orientation_deg']   # [Rx, Ry, Rz] deg
        self.v_approach = cfg['motion']['approach_speed']
        self.v_travel   = cfg['motion']['travel_speed']
        self.v_draw     = cfg['motion']['drawing_speed']
        self.accel      = cfg['motion']['max_accel']

        # Home position: canvas centre in Y/Z, 200 mm in front of canvas in X.
        cpos = cfg['canvas']['position']
        self._home_xyz = [cpos[0] - 0.200, cpos[1], cpos[2] + 0.050]

        self._current_uv = (0.0, 0.0)   # last known canvas UV

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _move_l(self, xyz, speed: float) -> None:
        """Block until a Cartesian MoveL to *xyz* completes."""
        pose = pose_from_position(xyz, self.orient)
        pose_str = " ".join(f"{v:.5f}" for v in pose)
        cmd = (f"MoveL(target={pose_str} WORLD WORLD, "
               f"vel={speed:.3f}, acc={self.accel:.3f})")
        log.debug("Primitive: %s", cmd)
        self.robot.execute_primitive(cmd)
        while not self.robot.is_primitive_done():
            time.sleep(0.002)

    # ── Public API ────────────────────────────────────────────────────────────

    def go_home(self) -> None:
        """Move to the safe home position (pen lifted, away from canvas)."""
        log.info("PenController: going home")
        self._move_l(self._home_xyz, self.v_travel)

    def travel_to(self, u: float, v: float) -> None:
        """Move to canvas position (u, v) with pen in the lifted position."""
        xyz = self.transform.canvas_to_robot(u, v, pen_down=False)
        log.debug("Travel to (u=%.3f, v=%.3f) → %s", u, v, xyz)
        self._move_l(xyz, self.v_travel)
        self._current_uv = (u, v)

    def pen_down(self) -> None:
        """Lower pen onto the canvas at the current UV position."""
        xyz = self.transform.canvas_to_robot(*self._current_uv, pen_down=True)
        log.debug("Pen down at (u=%.3f, v=%.3f) → x=%.4f", *self._current_uv, xyz[0])
        self._move_l(xyz, self.v_approach)

    def pen_up(self) -> None:
        """Lift pen off the canvas at the current UV position."""
        xyz = self.transform.canvas_to_robot(*self._current_uv, pen_down=False)
        log.debug("Pen up at (u=%.3f, v=%.3f) → x=%.4f", *self._current_uv, xyz[0])
        self._move_l(xyz, self.v_approach)

    def draw_to(self, u: float, v: float) -> None:
        """Draw a line to canvas position (u, v) — pen must already be down."""
        xyz = self.transform.canvas_to_robot(u, v, pen_down=True)
        log.debug("Draw to (u=%.3f, v=%.3f)", u, v)
        self._move_l(xyz, self.v_draw)
        self._current_uv = (u, v)

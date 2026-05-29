"""
robot/state_machine.py
Flexiv Rizon 4s caricature-artist state machine.

States
------
IDLE             Initial state; robot is not yet enabled or homed.
HOMING           Robot moves to the safe home position away from the easel.
WAIT_FOR_FACE    Camera polls for a face; counts consecutive detection frames.
CAPTURE_FACE     Snaps the reference frame and clears the face-hold counter.
GENERATE_STROKES Runs caricature exaggeration + stroke planning (CPU only).
APPROACH_CANVAS  Robot pre-positions the pen above the canvas centre (pen up).
DRAWING          StrokeExecutor draws every stroke; robot presses pen on canvas.
RETURN_HOME      Robot lifts pen and retreats to the safe home position.
DONE             Pauses briefly, then loops back to WAIT_FOR_FACE.
ERROR            Unrecoverable fault; loop exits and operator must intervene.

Transitions
-----------
IDLE         --[start()]-->       HOMING
HOMING       --[done]-->          WAIT_FOR_FACE
WAIT_FOR_FACE--[face N frames]--> CAPTURE_FACE
CAPTURE_FACE --[snap taken]-->    GENERATE_STROKES
GENERATE_STROKES--[strokes ready]-->APPROACH_CANVAS
APPROACH_CANVAS--[positioned]-->  DRAWING
DRAWING      --[all strokes]-->   RETURN_HOME
RETURN_HOME  --[home reached]-->  DONE
DONE         --[after pause]-->   WAIT_FOR_FACE
Any          --[exception]-->     ERROR
"""

import enum
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

import flexivrdk

from vision.camera import Camera
from vision.face_detector import FaceDetector, FaceLandmarks
from vision.caricature import exaggerate
from vision.stroke_generator import generate as gen_strokes, Stroke
from robot.pen_controller import PenController
from robot.stroke_executor import StrokeExecutor
from utils.geometry import CanvasTransform

log = logging.getLogger(__name__)


# ── State enumeration ─────────────────────────────────────────────────────────

class DrawingState(enum.Enum):
    IDLE             = "IDLE"
    HOMING           = "HOMING"
    WAIT_FOR_FACE    = "WAIT_FOR_FACE"
    CAPTURE_FACE     = "CAPTURE_FACE"
    GENERATE_STROKES = "GENERATE_STROKES"
    APPROACH_CANVAS  = "APPROACH_CANVAS"
    DRAWING          = "DRAWING"
    RETURN_HOME      = "RETURN_HOME"
    DONE             = "DONE"
    ERROR            = "ERROR"


# ── State machine ─────────────────────────────────────────────────────────────

class CaricatureStateMachine:
    """
    Top-level controller.  Call start() then run() from your main entry point.

    Args:
        robot:          Connected and enabled flexivrdk.Robot instance.
        camera:         Opened vision.Camera instance.
        face_detector:  Initialised vision.FaceDetector.
        pen_ctrl:       robot.PenController wired to *robot* and *transform*.
        stroke_exec:    robot.StrokeExecutor wired to *pen_ctrl*.
        transform:      utils.CanvasTransform for this session's canvas.
        cfg:            Full config dict loaded from config/config.yaml.
    """

    # Seconds to wait between successive caricatures before re-entering WAIT_FOR_FACE.
    _DONE_PAUSE_S = 4.0

    def __init__(self,
                 robot:        flexivrdk.Robot,
                 camera:       Camera,
                 face_detector: FaceDetector,
                 pen_ctrl:     PenController,
                 stroke_exec:  StrokeExecutor,
                 transform:    CanvasTransform,
                 cfg:          dict):
        self.robot        = robot
        self.camera       = camera
        self.detector     = face_detector
        self.pen_ctrl     = pen_ctrl
        self.stroke_exec  = stroke_exec
        self.transform    = transform
        self.cfg          = cfg

        self._state: DrawingState = DrawingState.IDLE
        self._running:  bool = False

        # Face detection accumulator
        self._face_hold: int = 0
        self._face_hold_target: int = cfg['camera']['face_hold_frames']

        # Data passed between states
        self._captured_frame: Optional[np.ndarray] = None
        self._captured_lm:    Optional[FaceLandmarks] = None
        self._strokes:        List[Stroke] = []

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> DrawingState:
        return self._state

    def start(self) -> None:
        """Enable the robot and kick off the state machine loop."""
        log.info("CaricatureStateMachine starting")
        self._running = True
        self._switch_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self._transition(DrawingState.HOMING)

    def stop(self) -> None:
        """Request a clean shutdown after the current state finishes."""
        log.info("Stop requested")
        self._running = False

    def run(self) -> None:
        """
        Blocking main loop.  Returns when stop() is called or ERROR is reached.
        """
        while self._running and self._state != DrawingState.ERROR:
            try:
                self._tick()
            except KeyboardInterrupt:
                log.info("KeyboardInterrupt — stopping")
                self._running = False
            except Exception:
                log.exception("Unhandled exception in state %s", self._state.value)
                self._transition(DrawingState.ERROR)
                break
            time.sleep(0.010)

        if self._state == DrawingState.ERROR:
            log.error("State machine exited with ERROR — operator intervention required")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transition(self, new_state: DrawingState) -> None:
        log.info("[%s] → [%s]", self._state.value, new_state.value)
        self._state = new_state

    def _switch_mode(self, mode) -> None:
        """Switch robot operational mode and wait for confirmation."""
        self.robot.switch_mode(mode)
        deadline = time.time() + 5.0
        while self.robot.mode() != mode:
            if time.time() > deadline:
                raise RuntimeError(f"Timed out switching robot to mode {mode}")
            time.sleep(0.05)

    # ── Per-state handlers ────────────────────────────────────────────────────

    def _tick(self) -> None:
        s = self._state

        # ── HOMING ────────────────────────────────────────────────────────────
        if s == DrawingState.HOMING:
            log.info("Homing robot")
            self.pen_ctrl.go_home()
            self._face_hold = 0
            self._transition(DrawingState.WAIT_FOR_FACE)

        # ── WAIT_FOR_FACE ─────────────────────────────────────────────────────
        elif s == DrawingState.WAIT_FOR_FACE:
            frame = self.camera.read()
            lm = self.detector.detect(frame)
            if lm is not None:
                self._face_hold += 1
                log.debug("Face hold count: %d / %d",
                          self._face_hold, self._face_hold_target)
                if self._face_hold >= self._face_hold_target:
                    self._captured_frame = frame.copy()
                    self._captured_lm    = lm
                    self._transition(DrawingState.CAPTURE_FACE)
            else:
                if self._face_hold > 0:
                    log.debug("Face lost — resetting hold counter")
                self._face_hold = 0

        # ── CAPTURE_FACE ──────────────────────────────────────────────────────
        elif s == DrawingState.CAPTURE_FACE:
            log.info("Face captured — proceeding to stroke generation")
            self._face_hold = 0
            self._transition(DrawingState.GENERATE_STROKES)

        # ── GENERATE_STROKES ──────────────────────────────────────────────────
        elif s == DrawingState.GENERATE_STROKES:
            drawing_cfg = self.cfg['drawing']

            # Apply caricature exaggeration then plan strokes.
            exag_lm = exaggerate(self._captured_lm, drawing_cfg)
            self._strokes = gen_strokes(exag_lm, self.transform, drawing_cfg)

            log.info("Generated %d strokes", len(self._strokes))
            self._transition(DrawingState.APPROACH_CANVAS)

        # ── APPROACH_CANVAS ───────────────────────────────────────────────────
        elif s == DrawingState.APPROACH_CANVAS:
            log.info("Approaching canvas centre (pen up)")
            # Move pen to canvas-centre with pen lifted; acts as a pre-position
            # so the first stroke travel is short.
            self.pen_ctrl.travel_to(0.0, 0.0)
            self._transition(DrawingState.DRAWING)

        # ── DRAWING ───────────────────────────────────────────────────────────
        elif s == DrawingState.DRAWING:
            log.info("Drawing caricature (%d strokes)", len(self._strokes))
            self.stroke_exec.execute_strokes(
                self._strokes,
                progress_cb=lambda done, total:
                    log.info("Stroke %d / %d", done, total),
                stop_flag=lambda: not self._running,
            )
            self._transition(DrawingState.RETURN_HOME)

        # ── RETURN_HOME ───────────────────────────────────────────────────────
        elif s == DrawingState.RETURN_HOME:
            log.info("Returning to home position")
            self.pen_ctrl.go_home()
            self._transition(DrawingState.DONE)

        # ── DONE ──────────────────────────────────────────────────────────────
        elif s == DrawingState.DONE:
            log.info("Caricature complete — pausing %.1f s before next person",
                     self._DONE_PAUSE_S)
            time.sleep(self._DONE_PAUSE_S)
            if self._running:
                self._transition(DrawingState.WAIT_FOR_FACE)

        # ── ERROR (should not tick again after transition, but guard anyway) ──
        elif s == DrawingState.ERROR:
            self._running = False

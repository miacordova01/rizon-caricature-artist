"""
robot/state_machine.py
Flexiv Rizon 4s caricature-artist state machine.

States
------
IDLE             Initial state; robot is not yet enabled or homed.
HOMING           Robot moves to the safe home position away from the easel.
WAIT_FOR_FACE    Camera polls for a face; counts consecutive detection frames.
COUNTDOWN        Face is stable — speaks "3… 2… 1…" then grabs the snapshot.
GENERATE_STROKES Runs caricature exaggeration + stroke planning (CPU only).
APPROACH_CANVAS  Robot pre-positions the pen above the canvas centre (pen up).
DRAWING          StrokeExecutor draws every stroke; robot presses pen on canvas.
RETURN_HOME      Robot lifts pen and retreats to the safe home position.
DONE             Pauses briefly, then loops back to WAIT_FOR_FACE.
ERROR            Unrecoverable fault; loop exits and operator must intervene.

Transitions
-----------
IDLE         --[start()]-->            HOMING
HOMING       --[done]-->               WAIT_FOR_FACE
WAIT_FOR_FACE--[face N frames]-->      COUNTDOWN
COUNTDOWN    --[3-2-1 + snapshot]-->   GENERATE_STROKES
GENERATE_STROKES--[strokes ready]-->   APPROACH_CANVAS
APPROACH_CANVAS--[positioned]-->       DRAWING
DRAWING      --[all strokes]-->        RETURN_HOME
RETURN_HOME  --[home reached]-->       DONE
DONE         --[after pause]-->        WAIT_FOR_FACE
Any          --[exception]-->          ERROR
"""

import enum
import logging
import time
from typing import List, Optional

import numpy as np

import flexivrdk

from utils.audio import say, say_async, countdown
from utils.geometry import CanvasTransform
from vision.camera import Camera
from vision.caricature import exaggerate
from vision.face_detector import FaceDetector, FaceLandmarks
from vision.head_segmenter import HeadSegmenter
from vision.stroke_generator import Stroke, generate as gen_strokes
from robot.pen_controller import PenController
from robot.stroke_executor import StrokeExecutor

log = logging.getLogger(__name__)


# ── State enumeration ─────────────────────────────────────────────────────────

class DrawingState(enum.Enum):
    IDLE             = "IDLE"
    HOMING           = "HOMING"
    WAIT_FOR_FACE    = "WAIT_FOR_FACE"
    COUNTDOWN        = "COUNTDOWN"
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

    _DONE_PAUSE_S = 4.0

    def __init__(self,
                 robot:          flexivrdk.Robot,
                 camera:         Camera,
                 face_detector:  FaceDetector,
                 head_segmenter: HeadSegmenter,
                 pen_ctrl:       PenController,
                 stroke_exec:    StrokeExecutor,
                 transform:      CanvasTransform,
                 cfg:            dict):
        self.robot         = robot
        self.camera        = camera
        self.detector      = face_detector
        self.head_seg      = head_segmenter
        self.pen_ctrl      = pen_ctrl
        self.stroke_exec   = stroke_exec
        self.transform     = transform
        self.cfg           = cfg

        self._state: DrawingState = DrawingState.IDLE
        self._running: bool = False

        # Face detection accumulator
        self._face_hold: int = 0
        self._face_hold_target: int = cfg['camera']['face_hold_frames']
        self._said_hold_still: bool = False   # avoid repeating the prompt

        # Data passed between states
        self._captured_frame: Optional[np.ndarray] = None
        self._captured_lm:    Optional[FaceLandmarks] = None
        self._strokes:        List[Stroke] = []

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> DrawingState:
        return self._state

    def start(self) -> None:
        log.info("CaricatureStateMachine starting")
        self._running = True
        self._switch_mode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)
        self._transition(DrawingState.HOMING)

    def stop(self) -> None:
        log.info("Stop requested")
        self._running = False

    def run(self) -> None:
        """Blocking main loop. Returns when stop() is called or ERROR is reached."""
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
            self._said_hold_still = False
            self._transition(DrawingState.WAIT_FOR_FACE)
            say_async("Step in front of the camera when you are ready.")

        # ── WAIT_FOR_FACE ─────────────────────────────────────────────────────
        elif s == DrawingState.WAIT_FOR_FACE:
            frame = self.camera.read()
            lm = self.detector.detect(frame)

            if lm is not None:
                self._face_hold += 1

                # First detection — prompt the subject to stay still.
                if self._face_hold == 1 and not self._said_hold_still:
                    say_async("Hold still.")
                    self._said_hold_still = True

                if self._face_hold >= self._face_hold_target:
                    # Store best-effort snapshot; COUNTDOWN will grab a fresh one.
                    self._captured_frame = frame.copy()
                    self._captured_lm    = lm
                    self._transition(DrawingState.COUNTDOWN)
            else:
                if self._face_hold > 0:
                    log.debug("Face lost — resetting hold counter")
                    self._said_hold_still = False
                    say_async("I lost you. Step back into frame.")
                self._face_hold = 0

        # ── COUNTDOWN ─────────────────────────────────────────────────────────
        elif s == DrawingState.COUNTDOWN:
            say("Get ready.")
            countdown(3, gap=0.85)

            # Grab the freshest possible frame right after "one".
            fresh_frame = self.camera.read()
            fresh_lm    = self.detector.detect(fresh_frame)
            if fresh_lm is not None:
                self._captured_frame = fresh_frame
                self._captured_lm    = fresh_lm
                log.info("Snapshot captured after countdown")
            else:
                log.warning("Face not detected on countdown snapshot — using buffered frame")
                # _captured_frame / _captured_lm already set in WAIT_FOR_FACE

            self._face_hold        = 0
            self._said_hold_still  = False
            self._transition(DrawingState.GENERATE_STROKES)

        # ── GENERATE_STROKES ──────────────────────────────────────────────────
        elif s == DrawingState.GENERATE_STROKES:
            say_async("Got it. Planning your caricature now.")
            drawing_cfg = self.cfg['drawing']

            # Head + hair outline (best-effort — None falls back to face oval)
            head_outline = self.head_seg.get_outline(self._captured_frame)
            if head_outline:
                log.info("Head outline: %d points", len(head_outline))
            else:
                log.warning("Head segmentation failed — using face oval fallback")

            exag_lm = exaggerate(self._captured_lm, drawing_cfg)
            self._strokes = gen_strokes(exag_lm, self.transform, drawing_cfg,
                                        head_outline=head_outline)
            log.info("Generated %d strokes", len(self._strokes))
            self._transition(DrawingState.APPROACH_CANVAS)

        # ── APPROACH_CANVAS ───────────────────────────────────────────────────
        elif s == DrawingState.APPROACH_CANVAS:
            log.info("Approaching canvas centre")
            say_async("Starting to draw. Please stand by.")
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
            say("Your caricature is complete! Hope you like it.")
            log.info("Caricature complete — pausing %.1f s", self._DONE_PAUSE_S)
            time.sleep(self._DONE_PAUSE_S)
            if self._running:
                say_async("Next person, step up when you are ready.")
                self._transition(DrawingState.WAIT_FOR_FACE)

        # ── ERROR ─────────────────────────────────────────────────────────────
        elif s == DrawingState.ERROR:
            self._running = False

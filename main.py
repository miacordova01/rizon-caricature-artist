#!/usr/bin/env python3
"""
main.py — Rizon 4s Caricature Artist entry point.

Usage:
    python main.py [--config config/config.yaml]
"""

import argparse
import logging
import sys
import time

import yaml

import flexivrdk

from vision.camera import Camera
from vision.face_detector import FaceDetector
from robot.pen_controller import PenController
from robot.stroke_executor import StrokeExecutor
from robot.state_machine import CaricatureStateMachine
from utils.geometry import CanvasTransform


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flexiv Rizon 4s Caricature Artist")
    p.add_argument("--config", default="config/config.yaml",
                   help="Path to config YAML (default: config/config.yaml)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(cfg: dict) -> CanvasTransform:
    c = cfg["canvas"]
    p = cfg["pen"]
    t = CanvasTransform(
        canvas_pos    = c["position"],
        canvas_width  = c["width"],
        canvas_height = c["height"],
        pen_tip_length= p["tip_offset"][2],   # z component = 0.150 m
    )
    t.set_lift_distance(p["lift_distance"])
    return t


def connect_robot(cfg: dict) -> flexivrdk.Robot:
    robot_cfg = cfg["robot"]
    log = logging.getLogger(__name__)
    log.info("Connecting to robot at %s (local %s)",
             robot_cfg["ip"], robot_cfg["local_ip"])
    robot = flexivrdk.Robot(robot_cfg["ip"], robot_cfg["local_ip"])
    robot.enable()

    log.info("Waiting for robot to become operational...")
    deadline = time.time() + 15.0
    while not robot.is_operational():
        if time.time() > deadline:
            raise RuntimeError("Robot did not become operational within 15 s")
        time.sleep(0.1)
    log.info("Robot operational")
    return robot


def main() -> int:
    args  = parse_args()
    logging.basicConfig(
        level   = args.log_level,
        format  = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    cfg = load_cfg(args.config)

    # ── Robot ─────────────────────────────────────────────────────────────────
    robot = connect_robot(cfg)

    # ── Canvas geometry ───────────────────────────────────────────────────────
    transform = build_transform(cfg)

    # ── Vision ────────────────────────────────────────────────────────────────
    cam_cfg = cfg["camera"]
    camera  = Camera(cam_cfg["device_id"],
                     cam_cfg["width"],
                     cam_cfg["height"],
                     cam_cfg["fps"])
    camera.open()
    face_detector = FaceDetector()

    # ── Robot controllers ─────────────────────────────────────────────────────
    pen_ctrl    = PenController(robot, transform, cfg)
    stroke_exec = StrokeExecutor(pen_ctrl, transform, cfg)

    # ── State machine ─────────────────────────────────────────────────────────
    sm = CaricatureStateMachine(
        robot        = robot,
        camera       = camera,
        face_detector= face_detector,
        pen_ctrl     = pen_ctrl,
        stroke_exec  = stroke_exec,
        transform    = transform,
        cfg          = cfg,
    )

    try:
        sm.start()
        sm.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        sm.stop()
    finally:
        log.info("Shutting down")
        camera.close()
        face_detector.close()

    return 0 if sm.state.value != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
webcam_pipeline.py — Flexiv Picasso Saturday deliverable

Webcam → face detection → landmark extraction → caricature → line art → JSON

Usage:
    python webcam_pipeline.py
    python webcam_pipeline.py --config config/config.yaml --output output/

Controls (camera window):
    SPACE   Manually trigger capture when face is ready
    R       Reset / prepare for next person
    Q       Quit

Auto-capture:
    Face must be stable for FACE_HOLD_FRAMES consecutive frames.
    A 3-2-1 countdown is spoken + displayed, then the snapshot is taken
    automatically.  SPACE short-circuits the countdown immediately.

Outputs (written to --output directory):
    portrait_NNN.json   Stroke paths in locked schema (coords ∈ [0, 1])
    portrait_NNN.png    Rendered line-art image (black on white)
"""

import argparse
import logging
import os
import subprocess
import time

import cv2
import yaml

from pipeline.portrait_paths import generate_portrait_paths, save_portrait_json
from pipeline.renderer import render_portrait, add_overlay_text
from vision.camera import Camera
from vision.face_detector import FaceDetector
from vision.head_segmenter import HeadSegmenter

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

WIN_CAM      = "Flexiv Picasso — Camera"
WIN_PORTRAIT = "Flexiv Picasso — Portrait"
GREEN  = (30,  200,  30)
ORANGE = (0,  165,  255)
RED    = (0,   50,  220)
GREY   = (120, 120, 120)

COUNTDOWN_S  = 3          # seconds in "3-2-1" before auto-capture
FACE_HOLD_FRAMES = 20     # consecutive frames with face before countdown starts


# ── Audio ─────────────────────────────────────────────────────────────────────

def _say(text: str) -> None:
    """Non-blocking macOS TTS."""
    subprocess.Popen(["say", "-v", "Samantha", text])


def _say_sync(text: str) -> None:
    subprocess.run(["say", "-v", "Samantha", text], check=False)


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _draw_face_box(img, lm, color, label=""):
    h, w = img.shape[:2]
    x1 = int(lm.bbox_min[0] * w);  y1 = int(lm.bbox_min[1] * h)
    x2 = int(lm.bbox_max[0] * w);  y2 = int(lm.bbox_max[1] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return x1, y1, x2, y2


def _draw_progress_bar(img, x1, x2, y2, fraction, color):
    bar_len = int(fraction * (x2 - x1))
    cv2.rectangle(img, (x1, y2 + 6), (x1 + bar_len, y2 + 16), color, -1)
    cv2.rectangle(img, (x1, y2 + 6), (x2, y2 + 16), GREY, 1)


def _put_center(img, text, y, scale=1.2, color=(0, 0, 0), thickness=2):
    w = img.shape[1]
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (w - size[0]) // 2
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_capture(frame, lm, head_segmenter, drawing_cfg, output_dir, session):
    """Run head segmentation → portrait paths → render → save."""
    log.info("Running head segmentation…")
    head_outline = head_segmenter.get_outline(frame)
    if head_outline:
        log.info("  Head outline: %d pts", len(head_outline))
    else:
        log.warning("  Head segmentation failed — using face oval only")

    log.info("Generating portrait paths…")
    strokes = generate_portrait_paths(lm, drawing_cfg,
                                      head_outline=head_outline, simplify=True)
    log.info("  %d strokes, %d total waypoints",
             len(strokes), sum(len(s["points"]) for s in strokes))

    portrait_img = render_portrait(strokes)
    portrait_img = add_overlay_text(portrait_img,
                                    f"Flexiv Picasso — portrait {session:03d}",
                                    pos=(16, 30))

    json_path = os.path.join(output_dir, f"portrait_{session:03d}.json")
    img_path  = os.path.join(output_dir, f"portrait_{session:03d}.png")
    save_portrait_json(strokes, json_path,
                       source="webcam", image_shape=frame.shape[:2])
    cv2.imwrite(img_path, portrait_img)

    log.info("Saved → %s", json_path)
    log.info("Saved → %s", img_path)
    return portrait_img


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="config/config.yaml")
    ap.add_argument("--output",    default="output")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    drawing_cfg = cfg["drawing"]
    cam_cfg     = cfg["camera"]

    camera        = Camera(cam_cfg["device_id"], cam_cfg["width"],
                           cam_cfg["height"], cam_cfg["fps"])
    face_detector = FaceDetector()
    head_segmenter = HeadSegmenter(
        confidence = cfg.get("head_segmentation", {}).get("confidence", 0.55),
        max_pts    = cfg.get("head_segmentation", {}).get("contour_pts", 80),
    )

    camera.open()
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM, 800, 600)
    cv2.namedWindow(WIN_PORTRAIT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PORTRAIT, 500, 660)

    _say("Flexiv Picasso ready. Step in front of the camera.")

    # ── State variables ───────────────────────────────────────────────────────
    face_hold      = 0
    countdown_end  = None     # time.time() value when auto-capture fires
    in_countdown   = False
    last_lm        = None
    last_frame     = None
    session        = 0
    portrait_shown = False

    log.info("Pipeline running.  SPACE=capture  R=reset  Q=quit")

    try:
        while True:
            frame = camera.read()
            lm    = face_detector.detect(frame)
            disp  = frame.copy()

            # ── Key handling ──────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                face_hold = 0; in_countdown = False; countdown_end = None
                log.info("Reset — ready for next person")
                _say("Ready for the next person.")
                continue
            force_capture = (key == ord(' ') and lm is not None)

            # ── Face detection overlay ────────────────────────────────────────
            if lm is not None:
                last_lm    = lm
                last_frame = frame.copy()
                face_hold += 1
                frac = min(face_hold / FACE_HOLD_FRAMES, 1.0)

                x1, y1, x2, y2 = _draw_face_box(disp, lm, GREEN,
                                                 label="Hold still")
                _draw_progress_bar(disp, x1, x2, y2, frac, GREEN)

                if face_hold == 1:
                    _say("Hold still.")
            else:
                if face_hold > 0:
                    _say("Face lost — step back in frame.")
                face_hold = 0; in_countdown = False; countdown_end = None

                _put_center(disp, "Looking for face...", disp.shape[0] // 2,
                            scale=1.0, color=RED)

            # ── Start countdown once face is stable ───────────────────────────
            if face_hold >= FACE_HOLD_FRAMES and not in_countdown:
                in_countdown  = True
                countdown_end = time.time() + COUNTDOWN_S
                _say(f"Get ready. Three. Two. One.")
                log.info("Countdown started (%.1f s)", COUNTDOWN_S)

            # ── Countdown display ─────────────────────────────────────────────
            if in_countdown and lm is not None:
                remaining = countdown_end - time.time()
                if remaining > 0:
                    n = int(remaining) + 1
                    _put_center(disp, str(n), disp.shape[0] // 2 - 60,
                                scale=4.0, color=ORANGE, thickness=5)
                    _put_center(disp, "CAPTURING SOON", disp.shape[0] // 2 + 30,
                                scale=0.9, color=ORANGE)
                else:
                    force_capture = True   # countdown expired → auto-fire

            # ── Capture ───────────────────────────────────────────────────────
            if force_capture and last_lm is not None:
                in_countdown = False; face_hold = 0

                _put_center(disp, "Capturing!", disp.shape[0] // 2,
                            scale=1.5, color=GREEN, thickness=3)
                cv2.imshow(WIN_CAM, disp)
                cv2.waitKey(200)

                session += 1
                _say("Got it. Generating your portrait.")

                portrait_img = run_capture(last_frame, last_lm, head_segmenter,
                                           drawing_cfg, args.output, session)
                cv2.imshow(WIN_PORTRAIT, portrait_img)
                portrait_shown = True

                _say("Your portrait is ready!")
                log.info("Session %d complete.  Press R for next person.", session)

            # ── Status footer ─────────────────────────────────────────────────
            footer = (f"Session {session}  |  "
                      f"{'face OK' if lm else 'no face'}  |  "
                      f"SPACE=capture  R=reset  Q=quit")
            cv2.putText(disp, footer, (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREY, 1, cv2.LINE_AA)

            cv2.imshow(WIN_CAM, disp)

    finally:
        camera.close()
        face_detector.close()
        head_segmenter.close()
        cv2.destroyAllWindows()
        log.info("Pipeline shut down.  %d portrait(s) saved to %s/",
                 session, args.output)


if __name__ == "__main__":
    main()

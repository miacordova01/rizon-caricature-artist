#!/usr/bin/env python3
"""
webcam_pipeline.py — Flexiv Picasso webcam pipeline

Webcam → face detection → SPACE to start countdown → capture → line art + JSON

Usage:
    python3 webcam_pipeline.py
    python3 webcam_pipeline.py --config config/config.yaml --output output/

Controls:
    SPACE   Start the 3-2-1 countdown (only works when a face is visible)
    R       Reset / next person
    Q       Quit

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

WIN_CAM      = "Flexiv Picasso — Camera"
WIN_PORTRAIT = "Flexiv Picasso — Portrait"
GREEN  = (30,  200,  30)
ORANGE = (0,  165,  255)
RED    = (0,   50,  220)
GREY   = (120, 120, 120)
WHITE  = (255, 255, 255)

COUNTDOWN_S = 3     # seconds between SPACE press and snapshot


# ── Audio ─────────────────────────────────────────────────────────────────────

def _say(text: str) -> None:
    subprocess.Popen(["say", "-v", "Samantha", text])


# ── Overlay helpers ───────────────────────────────────────────────────────────

def _draw_face_box(img, lm, color, label=""):
    h, w = img.shape[:2]
    x1 = int(lm.bbox_min[0] * w);  y1 = int(lm.bbox_min[1] * h)
    x2 = int(lm.bbox_max[0] * w);  y2 = int(lm.bbox_max[1] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return x1, y1, x2, y2


def _put_center(img, text, y, scale=1.2, color=(0, 0, 0), thickness=2):
    w = img.shape[1]
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x  = (w - sz[0]) // 2
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _banner(img, text, color, alpha=0.55):
    """Semi-transparent banner strip across the bottom third."""
    overlay = img.copy()
    h, w = img.shape[:2]
    cv2.rectangle(overlay, (0, h // 2), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    _put_center(img, text, int(h * 0.72), scale=1.4, color=WHITE, thickness=3)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_capture(frame, lm, head_segmenter, drawing_cfg, output_dir, session):
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
    h_fr, w_fr = frame.shape[:2]
    save_portrait_json(strokes, json_path,
                       name=f"portrait_{session:03d}",
                       image_wh=(w_fr, h_fr))
    cv2.imwrite(img_path, portrait_img)
    log.info("Saved → %s  |  %s", json_path, img_path)
    return portrait_img


# ── Main ──────────────────────────────────────────────────────────────────────

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

    camera         = Camera(cam_cfg["device_id"], cam_cfg["width"],
                            cam_cfg["height"], cam_cfg["fps"])
    face_detector  = FaceDetector()
    head_segmenter = HeadSegmenter(
        confidence = cfg.get("head_segmentation", {}).get("confidence", 0.55),
        max_pts    = cfg.get("head_segmentation", {}).get("contour_pts", 80),
    )

    camera.open()
    cv2.namedWindow(WIN_CAM,      cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CAM,     800, 600)
    cv2.namedWindow(WIN_PORTRAIT, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PORTRAIT, 500, 660)

    _say("Flexiv Picasso ready. Step into frame and press Space when ready.")

    # ── State ─────────────────────────────────────────────────────────────────
    #  WAITING   → watching for a face, SPACE starts countdown
    #  COUNTDOWN → 3-2-1 display, then snapshot
    #  DONE      → showing portrait, SPACE or R resets

    STATE_WAITING   = "WAITING"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_DONE      = "DONE"

    state          = STATE_WAITING
    countdown_end  = 0.0
    last_lm        = None
    last_frame     = None
    session        = 0

    log.info("Controls: SPACE = start countdown | R = reset | Q = quit")

    try:
        while True:
            frame = camera.read()
            lm    = face_detector.detect(frame)
            disp  = frame.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # ── WAITING ───────────────────────────────────────────────────────
            if state == STATE_WAITING:
                if lm is not None:
                    last_lm    = lm
                    last_frame = frame.copy()
                    _draw_face_box(disp, lm, GREEN, label="Face detected")
                    _banner(disp, "Press SPACE to capture", GREEN)

                    if key == ord(' '):
                        # User pressed SPACE → start countdown
                        countdown_end = time.time() + COUNTDOWN_S
                        state = STATE_COUNTDOWN
                        _say("Get ready. Three. Two. One.")
                        log.info("Countdown started")
                else:
                    _banner(disp, "Step into frame", RED)

            # ── COUNTDOWN ─────────────────────────────────────────────────────
            elif state == STATE_COUNTDOWN:
                remaining = countdown_end - time.time()

                if lm is not None:
                    last_lm    = lm
                    last_frame = frame.copy()
                    _draw_face_box(disp, lm, ORANGE)

                if remaining > 0:
                    n = int(remaining) + 1
                    _put_center(disp, str(n),
                                disp.shape[0] // 2 - 80,
                                scale=6.0, color=ORANGE, thickness=8)
                    _put_center(disp, "Hold still...",
                                disp.shape[0] // 2 + 60,
                                scale=1.1, color=WHITE, thickness=2)
                else:
                    # Fire!
                    if last_lm is None:
                        # Face disappeared during countdown — restart
                        _say("I lost you. Let's try again.")
                        state = STATE_WAITING
                    else:
                        _put_center(disp, "Got it!", disp.shape[0] // 2,
                                    scale=2.0, color=GREEN, thickness=4)
                        cv2.imshow(WIN_CAM, disp)
                        cv2.waitKey(300)

                        session += 1
                        _say("Generating your portrait.")
                        portrait_img = run_capture(last_frame, last_lm,
                                                   head_segmenter, drawing_cfg,
                                                   args.output, session)
                        cv2.imshow(WIN_PORTRAIT, portrait_img)
                        _say("Your portrait is ready!")
                        log.info("Session %d complete.  SPACE or R for next person.", session)
                        state = STATE_DONE

            # ── DONE ──────────────────────────────────────────────────────────
            elif state == STATE_DONE:
                if lm is not None:
                    _draw_face_box(disp, lm, GREY)
                _banner(disp, "Press SPACE for next person", GREY)

                if key in (ord(' '), ord('r')):
                    state  = STATE_WAITING
                    last_lm = last_frame = None
                    _say("Ready for the next person. Step into frame and press Space.")
                    log.info("Reset")
                    continue

            # ── R always resets from any state ────────────────────────────────
            if key == ord('r') and state != STATE_DONE:
                state = STATE_WAITING
                last_lm = last_frame = None
                log.info("Manual reset")

            # ── Footer ────────────────────────────────────────────────────────
            footer = (f"Session {session}  |  {state}  |  "
                      f"SPACE=capture  R=reset  Q=quit")
            cv2.putText(disp, footer, (10, disp.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)
            cv2.imshow(WIN_CAM, disp)

    finally:
        camera.close()
        face_detector.close()
        head_segmenter.close()
        cv2.destroyAllWindows()
        log.info("Shut down. %d portrait(s) saved to %s/", session, args.output)


if __name__ == "__main__":
    main()

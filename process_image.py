#!/usr/bin/env python3
"""
process_image.py — Run a static image through the full CV pipeline.

Usage:
    python3 process_image.py path/to/face.jpg
    python3 process_image.py path/to/face.heic        # auto-converted
    python3 process_image.py path/to/face.jpg --output output/my_portrait

Outputs (in --output dir):
    portrait_paths.json   Strokes in Bubble's locked schema (bottom-left coords)
    portrait_preview.png  Black-on-white line art preview
"""

import argparse
import logging
import os
import subprocess
import sys

import cv2
import yaml

from pipeline.portrait_paths import generate_portrait_paths, save_portrait_json
from pipeline.renderer import render_portrait, add_overlay_text
from vision.face_detector import FaceDetector
from vision.head_segmenter import HeadSegmenter

log = logging.getLogger(__name__)


def load_image(path: str) -> "cv2 BGR ndarray":
    """Load image; auto-converts HEIC via sips on macOS."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        tmp = "/tmp/_picasso_input.jpg"
        result = subprocess.run(
            ["sips", "-s", "format", "jpeg", path, "--out", tmp],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sips conversion failed: {result.stderr}")
        path = tmp
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image",         help="Path to face image (jpg/png/heic)")
    ap.add_argument("--config",      default="config/config.yaml")
    ap.add_argument("--output",      default="output")
    ap.add_argument("--name",        default=None,
                    help="Portrait name for JSON (default: filename stem)")
    ap.add_argument("--no-preview",  action="store_true",
                    help="Skip OpenCV display window")
    ap.add_argument("--log-level",   default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    drawing_cfg = cfg["drawing"]

    os.makedirs(args.output, exist_ok=True)
    stem = args.name or os.path.splitext(os.path.basename(args.image))[0]

    # ── Load image ────────────────────────────────────────────────────────────
    log.info("Loading image: %s", args.image)
    bgr = load_image(args.image)
    h, w = bgr.shape[:2]
    log.info("Image size: %d × %d", w, h)

    # ── Face detection ────────────────────────────────────────────────────────
    detector = FaceDetector(min_detection_confidence=0.4,
                            min_tracking_confidence=0.3)
    lm = detector.detect(bgr)
    detector.close()

    if lm is None:
        log.error("No face detected in image.  Try a clearer frontal photo.")
        sys.exit(1)
    log.info("Face detected")

    # ── Head segmentation ─────────────────────────────────────────────────────
    seg_cfg = cfg.get("head_segmentation", {})
    segmenter = HeadSegmenter(
        confidence = seg_cfg.get("confidence", 0.55),
        max_pts    = seg_cfg.get("contour_pts", 80),
    )
    head_outline = segmenter.get_outline(bgr)
    segmenter.close()
    if head_outline:
        log.info("Head outline: %d points", len(head_outline))
    else:
        log.warning("Head segmentation failed — using face oval only")

    # ── Generate strokes ──────────────────────────────────────────────────────
    strokes = generate_portrait_paths(lm, drawing_cfg,
                                      head_outline=head_outline,
                                      simplify=True)
    n_pts = sum(len(s["points"]) for s in strokes)
    log.info("Generated %d strokes, %d total waypoints", len(strokes), n_pts)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(args.output, f"{stem}_paths.json")
    save_portrait_json(strokes, json_path,
                       name=stem,
                       source_type="cv_lineart",
                       image_wh=(w, h))
    log.info("Saved → %s", json_path)

    # ── Render and save preview ───────────────────────────────────────────────
    portrait = render_portrait(strokes)
    portrait = add_overlay_text(portrait, stem, pos=(16, 30))
    img_path = os.path.join(args.output, f"{stem}_preview.png")
    cv2.imwrite(img_path, portrait)
    log.info("Saved → %s", img_path)

    # ── Optional display ──────────────────────────────────────────────────────
    if not args.no_preview:
        cv2.imshow("Portrait preview", portrait)
        log.info("Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
pipeline/renderer.py
Renders portrait stroke dicts to a clean black-line-on-white image.

The renderer is intentionally coordinate-system agnostic: it accepts any
list of strokes whose points are normalised floats in [0, 1] and scales
them to the requested canvas size.
"""

from typing import Dict, List, Any, Tuple

import cv2
import numpy as np

PortraitStroke = Dict[str, Any]

# Stroke style per feature label (thickness in pixels at 800-wide output)
_STYLE: Dict[str, int] = {
    "head_outline":  2,
    "face_oval":     1,
    "left_eyebrow":  2,
    "right_eyebrow": 2,
    "left_eye":      2,
    "right_eye":     2,
    "nose_bridge":   1,
    "nose_tip":      2,
    "lips_outer":    2,
    "lips_inner":    1,
}
_DEFAULT_THICKNESS = 2


def render_portrait(
        strokes:       List[PortraitStroke],
        canvas_wh:     Tuple[int, int] = (800, 1050),
        margin:        float = 0.06,
        bg_color:      Tuple[int, int, int] = (255, 255, 255),
        line_color:    Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Render portrait strokes to a BGR image.

    Args:
        strokes:    List of stroke dicts (portrait_paths schema).
        canvas_wh:  Output image size (width, height) in pixels.
        margin:     Fraction of canvas reserved as white border on each side.
        bg_color:   Background colour (BGR).
        line_color: Stroke colour (BGR).

    Returns:
        np.ndarray  BGR image, dtype=uint8.
    """
    W, H = canvas_wh
    img = np.full((H, W, 3), bg_color, dtype=np.uint8)

    draw_w = W * (1.0 - 2 * margin)
    draw_h = H * (1.0 - 2 * margin)
    ox     = W * margin
    oy     = H * margin

    for stroke in strokes:
        pts = stroke.get("points", [])
        if len(pts) < 2:
            continue

        label     = stroke.get("label", "")
        thickness = _STYLE.get(label, _DEFAULT_THICKNESS)

        # Scale normalised [0,1] → pixel coordinates.
        # Points are in bottom-left-origin schema (v=1 = top of portrait).
        # Screen coords have y=0 at top, so flip v: screen_y = (1 - v) * draw_h
        pixel_pts = [
            (int(u * draw_w + ox), int((1.0 - v) * draw_h + oy))
            for u, v in pts
        ]

        # Draw as polyline
        arr = np.array(pixel_pts, dtype=np.int32).reshape(-1, 1, 2)
        closed = stroke.get("closed", False)
        cv2.polylines(img, [arr], isClosed=closed,
                      color=line_color, thickness=thickness,
                      lineType=cv2.LINE_AA)

    return img


def add_overlay_text(img: np.ndarray, text: str,
                     pos: Tuple[int, int] = (20, 40),
                     color: Tuple[int, int, int] = (100, 100, 100)) -> np.ndarray:
    """Stamp a small info label onto the image (non-destructive copy)."""
    out = img.copy()
    cv2.putText(out, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out

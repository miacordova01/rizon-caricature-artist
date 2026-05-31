"""
vision/artistic.py
Artistic stroke generation utilities.

Provides five tools that turn raw MediaPipe landmark polygons into the
kind of clean, expressive lines you'd expect from a skilled illustrator:

  smooth_curve()              — Catmull-Rom spline through any point list
  extend_face_oval_to_head()  — generates a full head outline from face landmarks
  generate_eyelashes()        — radiating lash strokes on the upper eyelid
  generate_iris()             — circle for the iris / pupil
  generate_hair_lines()       — flowing strands down the sides of the head

All coordinates are MediaPipe image space: (x, y) top-left origin, [0, 1].
The caller converts to Bubble's UV schema via _to_uv() in portrait_paths.py.
"""

import math
from typing import List, Optional

import numpy as np


# ── Catmull-Rom spline ─────────────────────────────────────────────────────────

def smooth_curve(pts: List[np.ndarray],
                 n_total: int = 60,
                 closed: bool = False) -> List[np.ndarray]:
    """
    Catmull-Rom spline interpolation through *pts*.

    Replaces the jagged polygon that raw MediaPipe landmarks form with a smooth,
    flowing curve.  The number of output points is ~n_total (distributed evenly
    across segments).

    Args:
        pts:     Control points (list of np.ndarray or [x, y] pairs).
        n_total: Approximate number of output points.
        closed:  True for closed contours (face oval, head outline, eyes, lips).

    Returns:
        List of np.ndarray(2,) smooth curve points.
    """
    pts = [np.asarray(p, dtype=float) for p in pts]
    n = len(pts)
    if n < 2:
        return pts
    if n == 2:
        # Linear interpolation
        return [pts[0] + t * (pts[1] - pts[0])
                for t in np.linspace(0, 1, max(4, n_total))]

    # Add phantom endpoints so the spline passes through first/last real points
    if closed:
        ctrl = [pts[-1]] + pts + [pts[0], pts[1]]
        n_segs = n
    else:
        ctrl = [pts[0]] + pts + [pts[-1]]
        n_segs = n - 1

    n_per = max(3, n_total // n_segs)
    result: List[np.ndarray] = []

    for i in range(1, 1 + n_segs):
        p0, p1, p2, p3 = ctrl[i-1], ctrl[i], ctrl[i+1], ctrl[i+2]
        # Open last segment includes its endpoint; closed doesn't (avoids duplicate)
        include_end = (not closed) and (i == n_segs)
        t_vals = np.linspace(0.0, 1.0, n_per + int(include_end),
                             endpoint=include_end)
        for t in t_vals:
            t2, t3 = t * t, t * t * t
            result.append(0.5 * (
                2 * p1 +
                (-p0 + p2) * t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            ))

    return result


# ── Synthetic head outline ────────────────────────────────────────────────────

def extend_face_oval_to_head(
        face_oval:   List[np.ndarray],
        bbox_min:    np.ndarray,
        bbox_max:    np.ndarray,
        hair_lift:   float = 0.60,
        side_expand: float = 0.20) -> List[np.ndarray]:
    """
    Build a full head outline (skull + hair zone) by extending the face oval.

    MediaPipe's face oval follows the face closely but stops at the hairline —
    it gives no skull or hair above the forehead.  This function pushes every
    upper-face point upward (for the hair dome) and slightly outward (for head
    width), turning the face oval into a complete head silhouette without needing
    the image segmenter.

    Transformation rules (applied continuously, not as a binary threshold):
      • Points above the face centre → pushed up by up to hair_lift × face_height
        and outward by up to side_expand × face_width
      • Points below the face centre (jaw, chin) → unchanged

    Args:
        face_oval:    Face oval landmark points  (image-space x, y).
        bbox_min:     Top-left corner of face bounding box [x1, y1].
        bbox_max:     Bottom-right corner of face bounding box [x2, y2].
        hair_lift:    Vertical lift at the topmost point as a fraction of face height.
        side_expand:  Horizontal expansion at the topmost point as a fraction of face width.

    Returns:
        Extended point list — same length and ordering as face_oval.
    """
    pts    = [np.asarray(p, dtype=float) for p in face_oval]
    cx     = float((bbox_min[0] + bbox_max[0]) / 2.0)
    cy     = float((bbox_min[1] + bbox_max[1]) / 2.0)
    face_h = float(bbox_max[1] - bbox_min[1])
    half_h = face_h / 2.0 + 1e-9

    result = []
    for p in pts:
        # t_up: 0 at the face centre or below it, 1 at the very top of the face
        dy   = cy - float(p[1])          # positive → point is above centre
        t_up = max(0.0, min(1.0, dy / half_h))

        new_y = float(p[1]) - face_h * hair_lift * t_up
        dx    = float(p[0]) - cx
        new_x = cx + dx * (1.0 + side_expand * t_up)

        result.append(np.array([
            max(0.0, min(1.0, new_x)),
            max(0.0, min(1.0, new_y)),
        ]))

    return result


# ── Eyelashes ──────────────────────────────────────────────────────────────────

def generate_eyelashes(eye_pts: List[np.ndarray],
                       n_lashes: int = 11,
                       base_len: float = 0.016,
                       peak_scale: float = 2.0) -> List[List[np.ndarray]]:
    """
    Generate upper-eyelid lash strokes for one eye.

    Each lash is a 2-point stroke [start, end] that radiates outward (away from
    the eye centre) from a point on the upper lid.  Lashes are longest near the
    centre and taper toward the inner and outer corners, matching how real lashes
    look.

    Args:
        eye_pts:    Eye outline points (image space, x/y top-left origin).
        n_lashes:   Number of lash strokes.
        base_len:   Length at corners (normalised, ~1.6% of frame width).
        peak_scale: Centre lashes are this many × longer than corner lashes.

    Returns:
        List of [start, end] pairs (each a 2-element list of np.ndarray).
    """
    if len(eye_pts) < 4:
        return []

    pts = [np.asarray(p, dtype=float) for p in eye_pts]
    ctr = np.mean(pts, axis=0)

    # Upper lid = points above the eye centroid (smaller y in image coords)
    upper = sorted([p for p in pts if p[1] < ctr[1]], key=lambda p: p[0])
    if len(upper) < 2:
        return []

    idx     = np.linspace(0, len(upper) - 1, n_lashes).astype(int)
    sampled = [upper[i] for i in idx]

    lashes = []
    for k, pt in enumerate(sampled):
        d    = pt - ctr
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue
        d = d / norm

        # Bell curve: 0 at corners → 1 at centre → 0 at corners
        frac   = k / max(n_lashes - 1, 1)
        bell   = 1.0 - abs(2 * frac - 1)
        length = base_len * (1.0 + bell * (peak_scale - 1.0))

        # Slight lateral curl so lashes fan outward naturally
        perp    = np.array([-d[1], d[0]])
        d_curl  = d + perp * 0.30 * (frac - 0.5)
        curl_n  = np.linalg.norm(d_curl)
        if curl_n > 1e-9:
            d_curl /= curl_n

        lashes.append([pt, pt + d_curl * length])

    return lashes


# ── Iris circle ────────────────────────────────────────────────────────────────

def generate_iris(pupil:       np.ndarray,
                  eye_pts:     List[np.ndarray],
                  radius_frac: float = 0.40,
                  n_pts:       int = 20) -> List[np.ndarray]:
    """
    Draw the iris as a smooth circle centred on the pupil landmark.

    Radius is estimated as a fraction of the eye's horizontal span so it scales
    correctly with the caricature exaggeration applied to the eye outline.

    Args:
        pupil:       Pupil/iris-centre landmark (np.ndarray [x, y]).
        eye_pts:     Eye outline points — used to gauge the eye's width.
        radius_frac: Iris radius as a fraction of eye horizontal width.
        n_pts:       Circle resolution (20 is visually smooth).

    Returns:
        List of np.ndarray points forming a closed circle.
    """
    if not eye_pts:
        return []

    xs      = [float(p[0]) for p in eye_pts]
    eye_w   = max(xs) - min(xs)
    radius  = eye_w * radius_frac
    if radius < 1e-6:
        return []

    ctr    = np.asarray(pupil, dtype=float)
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    return [ctr + np.array([math.cos(a), math.sin(a)]) * radius for a in angles]


# ── Hair flow lines ────────────────────────────────────────────────────────────

def generate_hair_lines(head_outline:  List[np.ndarray],
                        face_bbox_min: np.ndarray,
                        face_bbox_max: np.ndarray,
                        n_per_side:    int   = 7,
                        max_len:       float = 0.28,
                        tilt:          float = 0.035) -> List[List[np.ndarray]]:
    """
    Generate flowing hair strands on the left and right sides of the head.

    Hair lines are smooth 3-point curves (spline-interpolated) that start at
    anchor points on the head silhouette and flow downward with a slight outward
    tilt, mimicking the look of long straight/wavy hair.

    Args:
        head_outline:   Head+hair silhouette (cropped, image-space x,y).
        face_bbox_min:  Top-left corner of face bounding box [x1, y1].
        face_bbox_max:  Bottom-right corner of face bounding box [x2, y2].
        n_per_side:     Hair strands per side.
        max_len:        Longest strand length as fraction of canvas height.
        tilt:           Horizontal outward drift per unit of drop.

    Returns:
        List of smooth stroke point lists (one list of np.ndarray per strand).
    """
    if not head_outline or len(head_outline) < 6:
        return []

    pts     = [np.asarray(p, dtype=float) for p in head_outline]
    face_cx = float((face_bbox_min[0] + face_bbox_max[0]) / 2)
    face_cy = float((face_bbox_min[1] + face_bbox_max[1]) / 2)
    face_h  = float(face_bbox_max[1] - face_bbox_min[1])

    # Anchor zone: from above forehead to slightly below chin
    y_top    = max(0.0, float(face_bbox_min[1]) - face_h * 0.70)
    y_bottom = min(1.0, float(face_bbox_max[1]) + face_h * 0.10)

    left_pts  = sorted(
        [p for p in pts if p[0] < face_cx and y_top <= p[1] <= y_bottom],
        key=lambda p: p[1]
    )
    right_pts = sorted(
        [p for p in pts if p[0] > face_cx and y_top <= p[1] <= y_bottom],
        key=lambda p: p[1]
    )

    lines = []

    for side_pts, sign in [(left_pts, -1.0), (right_pts, +1.0)]:
        if len(side_pts) < 2:
            continue

        idx     = np.linspace(0, len(side_pts) - 1, n_per_side).astype(int)
        anchors = [side_pts[i] for i in idx]

        for anchor in anchors:
            # Strands near the chin level are shorter; near ear/cheek = longer
            vert_t = 1.0 - abs(anchor[1] - face_cy) / max(face_h, 1e-6)
            vert_t = max(0.35, min(1.0, vert_t + 0.35))
            length = max_len * vert_t

            # 3-control-point curve: anchor → mid → end
            dx_mid = sign * tilt * length * 0.4
            dx_end = sign * tilt * length
            mid    = anchor + np.array([dx_mid, length * 0.45])
            end    = anchor + np.array([dx_end, length])

            # Clamp to [0, 1]
            mid = np.clip(mid, 0.0, 1.0)
            end = np.clip(end, 0.0, 1.0)

            strand = smooth_curve([anchor, mid, end], n_total=10)
            if len(strand) >= 2:
                lines.append(strand)

    return lines

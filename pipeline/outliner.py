"""
pipeline/outliner.py
Portrait line-art via semantic face segmentation — Portrait-Outliner approach.

Credit: https://github.com/horatioh13/Portrait-Outliner  (horatioh13)

Instead of connecting MediaPipe landmark dots, this module does pixel-level
face segmentation: it finds the *actual contour* of your hair, skin, eyes,
brows, nose, and lips, then thins each region to a 1-pixel skeleton and
converts to our normalised stroke JSON.  The result is a drawing that follows
your real face rather than a generic landmark template.

Pipeline
--------
  1. Save webcam frame → temp file
  2. face_crop_plus Cropper(strategy="largest") → tightly-cropped, aligned face
  3. face_crop_plus Cropper(mask_groups=…)      → per-region binary masks
  4. Edge extraction per mask (outer boundary ring)
  5. Dilate-and-subtract to clean overlapping edges between adjacent regions
  6. skimage skeletonize → 1-pixel-wide lines
  7. cv2.findContours + RDP simplification → polylines
  8. Normalise to (u, v) bottom-left origin → stroke dicts

Install
-------
  pip install face-crop-plus scikit-image

The face-crop-plus BiSeNet parsing model (~50 MB) downloads automatically
on first use.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from vision.artistic import smooth_curve   # Catmull-Rom spline for smooth lines

log = logging.getLogger(__name__)

PortraitStroke = Dict[str, Any]

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    from face_crop_plus import Cropper
    _FCP_AVAILABLE = True
except ImportError:
    _FCP_AVAILABLE = False
    log.warning("face-crop-plus not installed — outliner unavailable. "
                "Run:  pip install face-crop-plus scikit-image")

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False
    log.warning("scikit-image not installed — skeletonization skipped. "
                "Run:  pip install scikit-image")


# ── Region config ─────────────────────────────────────────────────────────────

# face_crop_plus BiSeNet label IDs → our feature group names.
# Labels: 1=skin,2=l-brow,3=r-brow,4=l-eye,5=r-eye,10=nose,
#         11=mouth,12=upper-lip,13=lower-lip,17=hair
_MASK_GROUPS: Dict[str, List[int]] = {
    "hair":     [17],
    "skin":     [1],
    "eyebrows": [2, 3],
    "eyes":     [4, 5],
    "nose":     [10],
    "lips":     [11, 12, 13],
}

# Order in which regions are drawn (back to front)
_DRAW_ORDER = ["hair", "skin", "eyebrows", "eyes", "nose", "lips"]


# ── Edge / outline helpers ────────────────────────────────────────────────────

def _extract_outline(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    1-pixel outer boundary of a binary mask.
    Computed as: mask − erode(mask).
    Much faster than Portrait-Outliner's pixel-loop version.
    """
    if mask is None or mask.max() == 0:
        return None
    binary = (mask > 127).astype(np.uint8) * 255
    kernel  = np.ones((3, 3), np.uint8)
    eroded  = cv2.erode(binary, kernel, iterations=1)
    return cv2.subtract(binary, eroded)


def _dilate_subtract(target: Optional[np.ndarray],
                     source: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Remove 1-pixel dilated *source* from *target*.
    Cleans the shared edge between two adjacent face regions so no line
    is drawn twice at a boundary.
    """
    if target is None or source is None:
        return target
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(source, kernel, iterations=1)
    return cv2.subtract(target, dilated)


def _process_outlines(masks: Dict[str, Optional[np.ndarray]]
                      ) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract and clean outlines for every region.
    Mirrors Portrait-Outliner's process_NUMPY_outlines().
    """
    out: Dict[str, Optional[np.ndarray]] = {
        k: _extract_outline(v) for k, v in masks.items()
    }

    # Skin outline: remove every feature that sits on top of skin
    for feat in ["hair", "eyes", "eyebrows", "nose", "lips"]:
        out["skin"] = _dilate_subtract(out.get("skin"), masks.get(feat))

    # Eyes: remove eyebrow overlap
    out["eyes"] = _dilate_subtract(out.get("eyes"), masks.get("eyebrows"))

    return out


# ── Skeletonize & vectorise ───────────────────────────────────────────────────

def _outline_to_strokes(outline: Optional[np.ndarray],
                        label: str,
                        h: int, w: int,
                        min_pts: int = 4) -> List[PortraitStroke]:
    """
    Binary outline image → list of normalised stroke dicts.

    Steps:
      1. skimage skeletonize → single-pixel lines
      2. cv2.findContours   → polylines
      3. RDP simplify       → fewer redundant points
      4. Normalise to (u,v)  bottom-left origin

    Args:
        outline:  Binary edge image (255 = line pixel).
        label:    Feature name for the stroke dict.
        h, w:     Height and width of the image (used for normalisation).
        min_pts:  Minimum contour length to keep.
    """
    if outline is None or outline.max() == 0:
        return []

    if _SKIMAGE_AVAILABLE:
        skel = (_sk_skeletonize(outline // 255) * 255).astype(np.uint8)
    else:
        skel = outline   # skip thinning — lines may be 2-3 px wide

    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_TC89_KCOS)

    strokes: List[PortraitStroke] = []
    for contour in contours:
        if len(contour) < min_pts:
            continue

        is_closed = cv2.contourArea(contour) > 20

        # 1. Light RDP — just removes blatant duplicate pixels, keeps most points
        #    so the spline has enough control points to be accurate.
        eps    = 0.0008 * cv2.arcLength(contour, is_closed)
        approx = cv2.approxPolyDP(contour, max(eps, 0.5), is_closed)
        if len(approx) < 2:
            continue

        # 2. Convert pixel coords → normalised [0,1] float points
        norm_pts = [
            np.array([float(p[0][0]) / w, float(p[0][1]) / h])
            for p in approx
        ]

        # 3. Catmull-Rom spline — turns staircase pixel edges into smooth curves.
        #    n_total scales with contour size so tiny details aren't over-smoothed.
        n_smooth = max(12, len(norm_pts) * 3)
        smooth   = smooth_curve(norm_pts, n_total=n_smooth, closed=is_closed)

        # 4. Second RDP pass after smoothing — reduces redundant collinear points.
        arr2   = (np.array([[p[0], p[1]] for p in smooth], dtype=np.float32)
                  * 10_000).astype(np.int32).reshape(-1, 1, 2)
        eps2   = cv2.arcLength(arr2, is_closed) * 0.002
        approx2 = cv2.approxPolyDP(arr2, max(eps2, 1.0), is_closed)
        if len(approx2) < 2:
            continue

        # 5. Back to [0,1] and flip y → bottom-left origin
        uv_pts = [
            [round(float(p[0][0]) / 10_000, 3),
             round(1.0 - float(p[0][1]) / 10_000, 3)]
            for p in approx2
        ]
        strokes.append({
            "label":  label,
            "closed": is_closed,
            "points": uv_pts,
        })

    return strokes


# ── Auto-scale (same logic as portrait_paths) ─────────────────────────────────

def _auto_scale(strokes: List[PortraitStroke],
                margin: float = 0.05) -> List[PortraitStroke]:
    """Uniformly scale and centre all strokes to fill [margin, 1-margin]."""
    all_pts = [pt for s in strokes for pt in s["points"]]
    if not all_pts:
        return strokes
    us = [p[0] for p in all_pts];  vs = [p[1] for p in all_pts]
    u_span = max(us) - min(us) or 1.0
    v_span = max(vs) - min(vs) or 1.0
    fit    = 1.0 - 2 * margin
    scale  = min(fit / u_span, fit / v_span)
    u_mid  = (min(us) + max(us)) / 2
    v_mid  = (min(vs) + max(vs)) / 2
    result = []
    for s in strokes:
        new_pts = [[round((p[0] - u_mid) * scale + 0.5, 3),
                    round((p[1] - v_mid) * scale + 0.5, 3)]
                   for p in s["points"]]
        result.append({**s, "points": new_pts})
    return result


# ── Nearest-neighbour stroke ordering (same logic as portrait_paths) ──────────

import math as _math

def _order_strokes(strokes: List[PortraitStroke]) -> List[PortraitStroke]:
    """Greedy nearest-neighbour ordering to minimise pen travel."""
    if len(strokes) <= 1:
        return strokes
    remaining = list(strokes)
    ordered: List[PortraitStroke] = []
    pen: Tuple[float, float] = (0.5, 0.5)

    while remaining:
        best_i, best_d, best_flip = 0, _math.inf, False
        for i, s in enumerate(remaining):
            pts = s["points"]
            d0 = (pen[0]-pts[0][0])**2 + (pen[1]-pts[0][1])**2
            d1 = (pen[0]-pts[-1][0])**2 + (pen[1]-pts[-1][1])**2
            if d0 < best_d: best_i, best_d, best_flip = i, d0, False
            if d1 < best_d: best_i, best_d, best_flip = i, d1, True
        s = remaining.pop(best_i)
        if best_flip:
            s = {**s, "points": list(reversed(s["points"]))}
        ordered.append(s)
        pen = tuple(s["points"][-1])

    for n, s in enumerate(ordered, start=1):
        s["id"] = f"stroke_{n:04d}"
    return ordered


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_portrait_from_frame(bgr_frame: np.ndarray) -> List[PortraitStroke]:
    """
    Full Portrait-Outliner pipeline for one webcam frame.

    Returns a list of stroke dicts in the locked JSON schema,
    ready to be saved with save_portrait_json().

    Raises ImportError if face-crop-plus is not installed.

    Args:
        bgr_frame:  Raw webcam frame (BGR, any resolution).

    Returns:
        Ordered, auto-scaled list of stroke dicts with (u, v) coords.
        Empty list if no face is detected.
    """
    if not _FCP_AVAILABLE:
        raise ImportError(
            "face-crop-plus is required for the high-quality outliner pipeline.\n"
            "Install it with:  pip install face-crop-plus scikit-image\n"
            "(The BiSeNet model, ~50 MB, downloads automatically on first use.)")

    tmpdir = tempfile.mkdtemp(prefix="flexiv_portrait_")
    try:
        orig_dir = os.path.join(tmpdir, "original_image")
        crop_dir = os.path.join(tmpdir, "cropped_image")
        mask_dir = os.path.join(tmpdir, "masks")
        for d in (orig_dir, crop_dir, mask_dir):
            os.makedirs(d)

        # ── 1. Save frame ─────────────────────────────────────────────────────
        cv2.imwrite(os.path.join(orig_dir, "image1.png"), bgr_frame)

        # ── 2. Crop + align face ──────────────────────────────────────────────
        log.info("Cropping and aligning face…")
        Cropper(strategy="largest", face_factor=0.75).process_dir(
            input_dir=orig_dir, output_dir=crop_dir)

        cropped_path = os.path.join(crop_dir, "image1.png")
        if not os.path.exists(cropped_path):
            log.warning("Face crop produced no output — face not detected")
            return []

        cropped = cv2.imread(cropped_path)
        h, w    = cropped.shape[:2]
        log.info("Cropped face: %d × %d px", w, h)

        # ── 3. Semantic segmentation ──────────────────────────────────────────
        log.info("Segmenting face regions (BiSeNet)…")
        Cropper(mask_groups=_MASK_GROUPS).process_dir(
            input_dir=crop_dir, output_dir=mask_dir)

        # ── 4. Load per-region masks ──────────────────────────────────────────
        masks: Dict[str, Optional[np.ndarray]] = {}
        for feature in _MASK_GROUPS:
            path = os.path.join(mask_dir, f"{feature}_mask", "image1.png")
            if os.path.exists(path):
                masks[feature] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                log.debug("  Loaded: %s", feature)
            else:
                masks[feature] = None
                log.debug("  Missing mask: %s (feature not in image)", feature)

        # ── 5. Extract and clean outlines ─────────────────────────────────────
        log.info("Extracting outlines…")
        outlines = _process_outlines(masks)

        # ── 6. Vectorise each region → stroke dicts ───────────────────────────
        all_strokes: List[PortraitStroke] = []
        for feature in _DRAW_ORDER:
            region = _outline_to_strokes(outlines.get(feature), feature, h, w)
            log.info("  %-12s %d stroke(s)", feature, len(region))
            all_strokes.extend(region)

        if not all_strokes:
            log.warning("No strokes generated — segmentation returned empty masks")
            return []

        # ── 7. Order → auto-scale ─────────────────────────────────────────────
        ordered = _order_strokes(all_strokes)
        scaled  = _auto_scale(ordered, margin=0.05)
        log.info("Total: %d strokes", len(scaled))
        return scaled

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

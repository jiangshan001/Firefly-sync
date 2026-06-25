"""Automatic flashing-region localisation via temporal variance.

Analyses a short sequence of frames to find the image region that
changes most over time — typically the flashing leader target on an
otherwise static screen.  Returns a bounding-box ROI suitable for
feed into the flash detector.

This module uses **numpy** and **OpenCV** (cv2) but requires no
Raspberry Pi hardware — it is fully testable on Windows / macOS.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# cv2 is imported lazily so the module can at least be imported when
# opencv is not installed (the functions will raise at call time).
_CV2: Any = None


def _cv2() -> Any:
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as _cv2_mod
        except ImportError:
            _CV2 = False
        else:
            _CV2 = _cv2_mod
    return _CV2


def _resize_area_fallback(g: np.ndarray, downsample: int) -> np.ndarray:
    if downsample <= 1:
        return g
    h, w = g.shape[:2]
    h2 = max(1, h // downsample)
    w2 = max(1, w // downsample)
    cropped = g[:h2 * downsample, :w2 * downsample]
    return cropped.reshape(h2, downsample, w2, downsample).mean(axis=(1, 3)).astype(np.uint8)


def _otsu_threshold_u8(img: np.ndarray) -> float:
    hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = float(img.size)
    if total <= 0:
        return 0.0
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold = 0.0
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > max_var:
            max_var = between
            threshold = float(i)
    return threshold


def _components_with_stats_fallback(binary: np.ndarray) -> list[dict[str, Any]]:
    mask = binary > 0
    h, w = mask.shape[:2]
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack = [(x0, y0)]
            seen[y0, x0] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                for ny in range(max(0, y - 1), min(h, y + 2)):
                    for nx in range(max(0, x - 1), min(w, x + 2)):
                        if not seen[ny, nx] and mask[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((nx, ny))
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
            components.append({
                "area": len(xs),
                "left": xmin,
                "top": ymin,
                "width": xmax - xmin + 1,
                "height": ymax - ymin + 1,
                "mask": (ys, xs),
            })
    return components


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def locate_flashing_region(
    frames: list[Any],
    method: str = "temporal_variance",
    min_area_px: int = 50,
    padding_px: int = 20,
    downsample: int = 1,
    change_threshold: float | None = None,
) -> dict[str, Any] | None:
    """Find the bounding box of the most temporally-active image region.

    The function converts each frame to greyscale, computes a per-pixel
    change map over time, thresholds it, and returns the largest (or
    highest-scoring) connected component as a candidate ROI.

    Parameters
    ----------
    frames:
        List of >= 2 frames as numpy arrays.  Can be BGR (H×W×3),
        greyscale (H×W), or mixed.
    method:
        Change metric — ``"temporal_variance"`` (std over time),
        ``"max_min_range"``, or ``"mean_abs_diff"``.
    min_area_px:
        Minimum area (pixels) of a candidate region.  Smaller blobs
        are rejected as noise.
    padding_px:
        Extra pixels added to each side of the detected bounding box,
        clipped to frame boundaries.
    downsample:
        Integer downsample factor applied before change computation
        to reduce noise and computation cost (1 = full resolution).
    change_threshold:
        Absolute threshold applied to the change map.  If *None*, an
        automatic threshold is estimated via Otsu's method.

    Returns
    -------
    dict or None
        ``{"x", "y", "width", "height", "confidence", "method",
        "score", "area_px"}`` on success, or *None* if no reliable
        region was found.
    """
    if len(frames) < 2:
        return None

    cv2 = _cv2()

    # --- Convert to greyscale and optionally downsample ---
    grey_frames: list[Any] = []
    for f in frames:
        g = _to_grey(f, cv2)
        if downsample > 1:
            if cv2:
                h, w = g.shape[:2]
                g = cv2.resize(g, (w // downsample, h // downsample),
                               interpolation=cv2.INTER_AREA)
            else:
                g = _resize_area_fallback(g, downsample)
        grey_frames.append(g)

    # --- Stack into (H, W, N) array ---
    stack = np.stack(grey_frames, axis=-1).astype(np.float32)

    # --- Compute change map ---
    if method == "max_min_range":
        change_map = np.max(stack, axis=-1) - np.min(stack, axis=-1)
    elif method == "mean_abs_diff":
        # Mean absolute frame-to-frame difference
        diffs = np.abs(np.diff(stack, axis=-1))
        change_map = np.mean(diffs, axis=-1)
    else:  # temporal_variance (default)
        change_map = np.std(stack, axis=-1)

    # Normalise to 0–255 for thresholding
    cmin, cmax = change_map.min(), change_map.max()
    if cmax - cmin < 1e-6:
        return None  # no variation at all
    change_map_u8 = ((change_map - cmin) / (cmax - cmin) * 255).astype(np.uint8)

    # --- Threshold ---
    if change_threshold is not None:
        thresh_val = change_threshold
        if cv2:
            _, binary = cv2.threshold(change_map_u8, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            binary = (change_map_u8 > thresh_val).astype(np.uint8) * 255
    else:
        if cv2:
            thresh_val, binary = cv2.threshold(
                change_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            thresh_val = _otsu_threshold_u8(change_map_u8)
            binary = (change_map_u8 > thresh_val).astype(np.uint8) * 255

    # --- Find connected components ---
    components: list[dict[str, Any]] = []
    if cv2:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        for label_idx in range(1, num_labels):
            mask = labels == label_idx
            components.append({
                "area": int(stats[label_idx, cv2.CC_STAT_AREA]),
                "left": int(stats[label_idx, cv2.CC_STAT_LEFT]),
                "top": int(stats[label_idx, cv2.CC_STAT_TOP]),
                "width": int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
                "mask": mask,
            })
    else:
        components = _components_with_stats_fallback(binary)

    if not components:
        return None

    # Evaluate each component
    best_score = -1.0
    best_component = None
    best_stats = None

    for component in components:
        area = int(component["area"])
        if area < min_area_px:
            continue

        # Mean change within this component
        mask = component["mask"]
        mean_change = float(np.mean(change_map[mask]))
        max_change = float(np.max(change_map[mask]))

        # Score: area-weighted mean change (prefer bigger, brighter regions)
        score = mean_change * np.log1p(area)

        if score > best_score:
            best_score = score
            best_component = component
            best_stats = {
                "area": area,
                "mean_change": mean_change,
                "max_change": max_change,
            }

    if best_component is None or best_stats is None:
        return None

    # --- Extract bounding box (in downsampled coords) ---
    left = int(best_component["left"])
    top = int(best_component["top"])
    w = int(best_component["width"])
    h = int(best_component["height"])

    # Scale back to original resolution
    left *= downsample
    top *= downsample
    w *= downsample
    h *= downsample

    # Get original frame dimensions for clipping
    orig_h, orig_w = grey_frames[0].shape[:2]
    orig_h *= downsample
    orig_w *= downsample

    # --- Apply padding, clipped to frame ---
    left = max(0, left - padding_px)
    top = max(0, top - padding_px)
    w = min(orig_w - left, w + 2 * padding_px)
    h = min(orig_h - top, h + 2 * padding_px)

    if w < min_area_px or h < min_area_px:
        return None

    # --- Confidence heuristic ---
    # High confidence = large change relative to frame + large area fraction
    frame_area = orig_w * orig_h
    area_fraction = (w * h) / frame_area if frame_area > 0 else 0
    change_relative = best_stats["max_change"] / 255.0 if cmax > 0 else 0
    confidence = min(1.0, 0.4 * (best_stats["max_change"] / (cmin + 1.0)) +
                          0.3 * min(1.0, area_fraction * 10) +
                          0.3 * min(1.0, best_stats["area"] / 500))

    return {
        "x": left,
        "y": top,
        "width": w,
        "height": h,
        "confidence": round(confidence, 4),
        "method": method,
        "score": round(best_score, 2),
        "area_px": best_stats["area"],
        "mean_change": round(best_stats["mean_change"], 2),
        "max_change": round(best_stats["max_change"], 2),
        "auto_threshold": round(float(thresh_val), 1) if change_threshold is None else change_threshold,
    }


def _bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    min_area = max(1, min(aw * ah, bw * bh))
    return inter / min_area


def locate_flashing_regions(
    frames: list[Any],
    method: str = "temporal_variance",
    max_regions: int = 2,
    min_area_px: int = 50,
    padding_px: int = 20,
    downsample: int = 1,
    change_threshold: float | None = None,
    max_overlap_ratio: float = 0.1,
    min_width_px: int = 5,
    min_height_px: int = 5,
    max_area_fraction: float = 0.35,
) -> dict[str, Any]:
    """Find multiple non-overlapping flashing-region candidates.

    This is the multi-target counterpart to ``locate_flashing_region``.  It
    returns all candidate components plus the top non-overlapping selections.
    """
    if len(frames) < 2:
        return {
            "regions": [],
            "candidates": [],
            "image_size": None,
            "auto_threshold": None,
            "failure_reason": "insufficient_frames",
        }

    cv2 = _cv2()
    grey_frames: list[np.ndarray] = []
    orig_h, orig_w = _to_grey(np.asarray(frames[0]), cv2).shape[:2]
    for f in frames:
        g = _to_grey(np.asarray(f), cv2)
        if downsample > 1:
            if cv2:
                h, w = g.shape[:2]
                g = cv2.resize(g, (w // downsample, h // downsample),
                               interpolation=cv2.INTER_AREA)
            else:
                g = _resize_area_fallback(g, downsample)
        grey_frames.append(g)

    stack = np.stack(grey_frames, axis=-1).astype(np.float32)
    if method == "max_min_range":
        change_map = np.max(stack, axis=-1) - np.min(stack, axis=-1)
    elif method == "mean_abs_diff":
        diffs = np.abs(np.diff(stack, axis=-1))
        change_map = np.mean(diffs, axis=-1)
    else:
        change_map = np.std(stack, axis=-1)

    cmin, cmax = change_map.min(), change_map.max()
    if cmax - cmin < 1e-6:
        return {
            "regions": [],
            "candidates": [],
            "image_size": [orig_w, orig_h],
            "auto_threshold": None,
            "failure_reason": "no_temporal_variation",
        }
    change_map_u8 = ((change_map - cmin) / (cmax - cmin) * 255).astype(np.uint8)

    if change_threshold is not None:
        thresh_val = float(change_threshold)
        if cv2:
            _, binary = cv2.threshold(change_map_u8, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            binary = (change_map_u8 > thresh_val).astype(np.uint8) * 255
    else:
        if cv2:
            thresh_val, binary = cv2.threshold(
                change_map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            thresh_val = _otsu_threshold_u8(change_map_u8)
            binary = (change_map_u8 > thresh_val).astype(np.uint8) * 255

    components: list[dict[str, Any]] = []
    if cv2:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        for label_idx in range(1, num_labels):
            components.append({
                "area": int(stats[label_idx, cv2.CC_STAT_AREA]),
                "left": int(stats[label_idx, cv2.CC_STAT_LEFT]),
                "top": int(stats[label_idx, cv2.CC_STAT_TOP]),
                "width": int(stats[label_idx, cv2.CC_STAT_WIDTH]),
                "height": int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
                "mask": labels == label_idx,
            })
    else:
        components = _components_with_stats_fallback(binary)

    frame_area = max(1, orig_w * orig_h)
    candidates: list[dict[str, Any]] = []
    for idx, component in enumerate(components):
        area = int(component["area"])
        if area < min_area_px:
            continue
        left = int(component["left"]) * downsample
        top = int(component["top"]) * downsample
        width = int(component["width"]) * downsample
        height = int(component["height"]) * downsample
        x = max(0, left - padding_px)
        y = max(0, top - padding_px)
        w = min(orig_w - x, width + 2 * padding_px)
        h = min(orig_h - y, height + 2 * padding_px)
        padded_area = w * h
        if w < min_width_px or h < min_height_px:
            continue
        if padded_area / frame_area > max_area_fraction:
            continue

        mask = component["mask"]
        mean_change = float(np.mean(change_map[mask]))
        max_change = float(np.max(change_map[mask]))
        area_score = np.log1p(area)
        compactness_penalty = max(1.0, padded_area / max(area, 1))
        score = (mean_change * area_score) / np.sqrt(compactness_penalty)
        candidates.append({
            "candidate_id": idx,
            "roi": [int(x), int(y), int(w), int(h)],
            "component_bbox": [int(left), int(top), int(width), int(height)],
            "area_px": area,
            "padded_area_px": int(padded_area),
            "mean_change": round(mean_change, 4),
            "max_change": round(max_change, 4),
            "score": round(float(score), 4),
            "method": method,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        max_overlap = max(
            (_bbox_overlap_ratio(candidate["roi"], existing["roi"]) for existing in selected),
            default=0.0,
        )
        if max_overlap <= max_overlap_ratio and len(selected) < max_regions:
            item = dict(candidate)
            item["selection_overlap_ratio"] = round(max_overlap, 6)
            selected.append(item)
        else:
            item = dict(candidate)
            item["rejection_reason"] = (
                "overlaps_selected_region" if max_overlap > max_overlap_ratio
                else "more_than_requested_regions"
            )
            item["selection_overlap_ratio"] = round(max_overlap, 6)
            rejected.append(item)

    return {
        "regions": selected,
        "candidates": candidates,
        "rejected_candidates": rejected,
        "image_size": [orig_w, orig_h],
        "auto_threshold": round(float(thresh_val), 1),
        "method": method,
        "failure_reason": None if selected else "no_valid_components",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_grey(frame: np.ndarray, cv2: Any) -> np.ndarray:
    """Convert a BGR or greyscale frame to uint8 greyscale."""
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        if cv2 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(frame[:, :, :3], axis=2).astype(np.uint8)
    # Single-channel but 3D (H, W, 1)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0].astype(np.uint8)
    raise ValueError(f"Unexpected frame shape: {frame.shape}")

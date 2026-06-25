"""Pure-Python flash detection state machine.

This module contains zero hardware dependencies (no gpiozero, picamera2,
numpy, or flask).  All state is passed in/out explicitly via a dataclass,
making the hysteresis logic fully unit-testable on any platform.

The ``PicameraFlashDetector`` in ``picamera_flash_detector.py`` delegates
its per-frame detection decisions to this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Default detector parameters
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD_ON = 128       # brightness level that triggers ON state
DEFAULT_THRESHOLD_OFF = 64       # brightness level that triggers OFF state
DEFAULT_MIN_INTERVAL_S = 0.1     # minimum seconds between rising edges
DEFAULT_FREQ_WINDOW_S = 5.0      # sliding window for frequency estimation


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class FlashDetectorState:
    """Snapshot of the hysteresis state machine at one point in time.

    Attributes:
        state: Current detection state — ``"ON"`` or ``"OFF"``.
        brightness_mean: Mean brightness of the most recent frame.
        rising_edge_count: Total number of valid rising edges detected.
        last_rising_edge_time_s: ``perf_counter`` value of the last
            rising edge, or *None* if none yet.
        estimated_frequency_hz: Estimated leader frequency based on
            recent inter-edge intervals.  0.0 if insufficient data.
        event_type: ``"leader_rising_edge"`` for frames where a valid
            rising edge was detected, *None* otherwise.
        edge_times: Sliding buffer of ``perf_counter`` timestamps for
            recent rising edges (used for frequency estimation).
    """

    state: str = "OFF"
    brightness_mean: float = 0.0
    rising_edge_count: int = 0
    last_rising_edge_time_s: float | None = None
    estimated_frequency_hz: float = 0.0
    event_type: str | None = None
    edge_times: list[float] = field(default_factory=list)


@dataclass
class FlashEpisodeLatchState:
    """State for one-event-per-visual-flash episode detection."""

    currently_on: bool = False
    armed: bool = True
    last_event_time_s: float | None = None
    below_off_since_s: float | None = None
    off_frame_count: int = 0
    duplicate_suppressed_count: int = 0
    raw_on_threshold_crossing_count: int = 0
    accepted_event_count: int = 0
    previous_signal_on: bool = False
    event_type: str | None = None
    raw_on_crossing: bool = False
    rearmed_event: bool = False
    below_off_duration_s: float = 0.0
    time_since_last_event_s: float | None = None
    reject_reason: str = ""


# ---------------------------------------------------------------------------
# Core update function
# ---------------------------------------------------------------------------

def update_flash_detector(
    brightness: float,
    prev_state: FlashDetectorState,
    now_s: float,
    threshold_on: float = DEFAULT_THRESHOLD_ON,
    threshold_off: float = DEFAULT_THRESHOLD_OFF,
    min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
    freq_window_s: float = DEFAULT_FREQ_WINDOW_S,
) -> FlashDetectorState:
    """Advance the hysteresis state machine by one frame.

    Hysteresis rules
    ----------------
    * **OFF → ON**:  when ``brightness >= threshold_on``.
      A rising edge is emitted if at least ``min_interval_s`` has
      elapsed since the previous rising edge.
    * **ON → OFF**:  when ``brightness <= threshold_off``.
      This is a falling edge — no event is emitted.
    * **Otherwise**: state and event_type remain unchanged.

    Frequency is estimated as the reciprocal of the mean inter-edge
    interval over the most recent ``freq_window_s`` seconds.

    Parameters
    ----------
    brightness:
        Mean brightness of the current frame (e.g. 0–255).
    prev_state:
        State from the previous frame.
    now_s:
        Current timestamp from ``time.perf_counter()``.
    threshold_on:
        Brightness above which the state transitions to ON.
    threshold_off:
        Brightness below which the state transitions to OFF.
    min_interval_s:
        Minimum interval (seconds) between successive rising edges.
        Edges arriving sooner are ignored.
    freq_window_s:
        Sliding time window (seconds) for frequency estimation.

    Returns
    -------
    FlashDetectorState
        New state reflecting the current frame.
    """
    # Start from previous state, reset per-frame fields
    new_state = FlashDetectorState(
        state=prev_state.state,
        brightness_mean=brightness,
        rising_edge_count=prev_state.rising_edge_count,
        last_rising_edge_time_s=prev_state.last_rising_edge_time_s,
        estimated_frequency_hz=prev_state.estimated_frequency_hz,
        event_type=None,
        edge_times=list(prev_state.edge_times),
    )

    # --- Hysteresis ---
    if prev_state.state == "OFF" and brightness >= threshold_on:
        # Dark → Bright transition (potential rising edge)
        new_state.state = "ON"
        if (
            prev_state.last_rising_edge_time_s is None
            or (now_s - prev_state.last_rising_edge_time_s) >= min_interval_s
        ):
            # Valid rising edge
            new_state.event_type = "leader_rising_edge"
            new_state.rising_edge_count += 1
            new_state.last_rising_edge_time_s = now_s
            new_state.edge_times.append(now_s)

    elif prev_state.state == "ON" and brightness <= threshold_off:
        # Bright → Dark transition (falling edge, no event)
        new_state.state = "OFF"

    # else: state unchanged, event_type stays None

    # --- Frequency estimation ---
    new_state.estimated_frequency_hz = estimate_frequency(
        new_state.edge_times,
        now_s=now_s,
        window_s=freq_window_s,
    )

    return new_state


def update_flash_episode_latch(
    signal_value: float,
    prev_state: FlashEpisodeLatchState,
    now_s: float,
    threshold_on: float,
    threshold_off: float,
    min_interval_s: float,
    rearm_off_duration_s: float = 0.05,
    off_frames_to_rearm: int = 2,
    rearm_requires_both: bool = False,
) -> FlashEpisodeLatchState:
    """Accept at most one rising-edge event per flash episode.

    A new event is emitted only when the signal crosses the ON threshold while
    the latch is armed.  After an event, the latch re-arms only once the signal
    has stayed below the OFF threshold for the requested duration or frame
    count.  If ``rearm_requires_both`` is true, both the duration and frame
    count conditions must be satisfied; otherwise either condition re-arms.
    ``min_interval_s`` remains an additional safety guard.
    """
    signal_on = signal_value >= threshold_on
    signal_off = signal_value <= threshold_off
    raw_on_crossing = signal_on and not prev_state.previous_signal_on

    below_off_since_s = prev_state.below_off_since_s
    off_frame_count = prev_state.off_frame_count
    currently_on = prev_state.currently_on
    armed = prev_state.armed
    last_event_time_s = prev_state.last_event_time_s
    duplicate_suppressed_count = prev_state.duplicate_suppressed_count
    raw_on_threshold_crossing_count = prev_state.raw_on_threshold_crossing_count
    accepted_event_count = prev_state.accepted_event_count
    event_type = None
    rearmed_event = False
    reject_reason = ""

    if raw_on_crossing:
        raw_on_threshold_crossing_count += 1

    if signal_off:
        if below_off_since_s is None:
            below_off_since_s = now_s
        off_frame_count += 1
    else:
        below_off_since_s = None
        off_frame_count = 0

    below_long_enough = (
        below_off_since_s is not None
        and (now_s - below_off_since_s) >= rearm_off_duration_s
    )
    below_off_duration_s = (
        now_s - below_off_since_s
        if below_off_since_s is not None else 0.0
    )
    enough_off_frames = off_frame_count >= max(1, off_frames_to_rearm)
    rearm_condition_met = (
        (below_long_enough and enough_off_frames)
        if rearm_requires_both
        else (below_long_enough or enough_off_frames)
    )
    if currently_on and signal_off and rearm_condition_met:
        currently_on = False
        armed = True
        rearmed_event = True

    min_interval_ok = (
        last_event_time_s is None
        or (now_s - last_event_time_s) >= min_interval_s
    )
    time_since_last_event_s = (
        now_s - last_event_time_s
        if last_event_time_s is not None else None
    )
    if signal_on and armed and not currently_on and min_interval_ok:
        event_type = "leader_rising_edge"
        currently_on = True
        armed = False
        last_event_time_s = now_s
        accepted_event_count += 1
        below_off_since_s = None
        off_frame_count = 0
    elif raw_on_crossing and signal_on:
        duplicate_suppressed_count += 1
        if currently_on:
            reject_reason = "same_episode_latch_on"
        elif not armed:
            reject_reason = "latch_disarmed"
        elif not min_interval_ok:
            reject_reason = "min_interval"
        else:
            reject_reason = "not_currently_off"

    return FlashEpisodeLatchState(
        currently_on=currently_on,
        armed=armed,
        last_event_time_s=last_event_time_s,
        below_off_since_s=below_off_since_s,
        off_frame_count=off_frame_count,
        duplicate_suppressed_count=duplicate_suppressed_count,
        raw_on_threshold_crossing_count=raw_on_threshold_crossing_count,
        accepted_event_count=accepted_event_count,
        previous_signal_on=signal_on,
        event_type=event_type,
        raw_on_crossing=raw_on_crossing,
        rearmed_event=rearmed_event,
        below_off_duration_s=below_off_duration_s,
        time_since_last_event_s=time_since_last_event_s,
        reject_reason=reject_reason,
    )


# ---------------------------------------------------------------------------
# Frequency estimation
# ---------------------------------------------------------------------------

def estimate_frequency(
    edge_times: list[float],
    now_s: float | None = None,
    window_s: float = DEFAULT_FREQ_WINDOW_S,
) -> float:
    """Estimate frequency from a list of rising-edge timestamps.

    The estimate is the reciprocal of the mean inter-edge interval
    computed over edges that fall within the trailing time window.

    Parameters
    ----------
    edge_times:
        Monotonically increasing list of ``perf_counter`` values for
        each rising edge.
    now_s:
        Current timestamp.  Edges older than ``now_s - window_s`` are
        excluded.  If *None*, all edges are used.
    window_s:
        Sliding window width in seconds.

    Returns
    -------
    float
        Estimated frequency in Hz.  Returns 0.0 when fewer than two
        edges are available within the window.
    """
    if len(edge_times) < 2:
        return 0.0

    # Restrict to recent window
    if now_s is not None:
        cutoff = now_s - window_s
        recent = [t for t in edge_times if t >= cutoff]
    else:
        recent = edge_times

    if len(recent) < 2:
        return 0.0

    intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    mean_interval = sum(intervals) / len(intervals)

    if mean_interval <= 0.0:
        return 0.0

    return 1.0 / mean_interval


# ---------------------------------------------------------------------------
# Brightness extraction helpers
#
# These functions accept a numpy array (frame) and an optional ROI.
# ROI format: ``[x, y, width, height]`` or *None* for full frame.
# They require numpy but no Pi hardware — testable on any platform.
# ---------------------------------------------------------------------------

def _crop_roi(frame: np.ndarray, roi: list[int] | None) -> np.ndarray:
    """Crop *frame* to *roi* ``[x, y, w, h]``.  Returns full frame if
    *roi* is *None*."""
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


def normalize_frame_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Return *frame* as a single-channel uint8 intensity image.

    Picamera2 may return greyscale ``HxW``, colour ``HxWx3``, or
    ``XBGR8888``/RGBA/BGRA-style ``HxWx4`` arrays.  Detection metrics operate
    on intensity, so 3/4-channel inputs are averaged across the first three
    colour channels and any alpha/X channel is ignored.
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        grey = arr
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        grey = np.mean(arr[:, :, :3], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        grey = arr[:, :, 0]
    else:
        raise ValueError(f"Unsupported frame shape for flash detection: {arr.shape}")

    if grey.dtype == np.uint8:
        return grey

    grey_float = grey.astype(np.float32)
    if np.issubdtype(grey.dtype, np.floating) and grey_float.size and float(np.nanmax(grey_float)) <= 1.0:
        grey_float = grey_float * 255.0
    return np.clip(grey_float, 0, 255).astype(np.uint8)


def compute_roi_mean_brightness(
    frame: np.ndarray,
    roi: list[int] | None = None,
) -> float:
    """Mean pixel brightness over the (optionally cropped) frame.

    Parameters
    ----------
    frame:
        Greyscale or BGR numpy array.  If BGR it is converted to grey.
    roi:
        Optional ``[x, y, width, height]`` region of interest.

    Returns
    -------
    float
        Mean brightness (0–255).
    """
    crop = _crop_roi(normalize_frame_to_grayscale(frame), roi)
    return float(np.mean(crop))


def compute_top_percentile_brightness(
    frame: np.ndarray,
    percentile: float = 99.0,
    roi: list[int] | None = None,
) -> float:
    """Brightness at a high percentile — robust when the target is small.

    Instead of the frame mean (which is diluted by dark background),
    this returns the *p*-th percentile of pixel brightness.  For a small
    flashing target, ``p=99`` captures its peak brightness while
    ignoring the majority-dark background.

    Parameters
    ----------
    frame:
        Greyscale or BGR numpy array.
    percentile:
        Percentile (0–100).  Default 99.0.
    roi:
        Optional ``[x, y, width, height]``.

    Returns
    -------
    float
        Brightness value at the given percentile.
    """
    crop = _crop_roi(normalize_frame_to_grayscale(frame), roi).astype(np.float32)
    return float(np.percentile(crop, percentile))


def compute_bright_blob_metrics(
    frame: np.ndarray,
    threshold: float = 180.0,
    roi: list[int] | None = None,
) -> dict[str, Any]:
    """Find the largest bright connected region above *threshold*.

    Useful when the flashing target is a compact bright shape on a
    dark background.

    Parameters
    ----------
    frame:
        Greyscale or BGR numpy array.
    threshold:
        Brightness threshold (0–255) for binarisation.
    roi:
        Optional ``[x, y, width, height]``.

    Returns
    -------
    dict
        ``{"blob_found": bool, "blob_area_px": int,
        "blob_mean_brightness": float, "blob_max_brightness": float,
        "blob_bbox": [x, y, w, h] | None}``
    """
    grey = _crop_roi(normalize_frame_to_grayscale(frame), roi)

    # Binary mask of bright pixels
    binary = (grey >= threshold).astype(np.uint8)
    if binary.ndim == 3:
        binary = normalize_frame_to_grayscale(binary)
    binary = np.ascontiguousarray(binary.astype(np.uint8))

    # OpenCV for connected components
    try:
        import cv2
    except ImportError:
        # Fallback: no cv2 — just count bright pixels
        count = int(np.sum(binary))
        if count == 0:
            return {
                "blob_found": False, "blob_area_px": 0,
                "blob_count": 0,
                "blob_mean_brightness": 0.0, "blob_max_brightness": 0.0,
                "blob_bbox": None,
            }
        mean_b = float(np.mean(grey[binary > 0])) if count > 0 else 0.0
        max_b = float(np.max(grey[binary > 0])) if count > 0 else 0.0
        ys, xs = np.where(binary)
        bbox = [int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
        return {
            "blob_found": True, "blob_area_px": count,
            "blob_count": 1,
            "blob_mean_brightness": round(mean_b, 2),
            "blob_max_brightness": round(max_b, 2),
            "blob_bbox": bbox,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    if num_labels <= 1:
        return {
            "blob_found": False, "blob_area_px": 0,
            "blob_count": 0,
            "blob_mean_brightness": 0.0, "blob_max_brightness": 0.0,
            "blob_bbox": None,
        }

    # Find largest blob (skip label 0 = background)
    best_label = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for l in range(2, num_labels):
        area = int(stats[l, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = l

    mask = (labels == best_label)
    mean_b = float(np.mean(grey[mask]))
    max_b = float(np.max(grey[mask]))

    left = int(stats[best_label, cv2.CC_STAT_LEFT])
    top = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

    return {
        "blob_found": True,
        "blob_area_px": best_area,
        "blob_count": int(num_labels - 1),
        "blob_mean_brightness": round(mean_b, 2),
        "blob_max_brightness": round(max_b, 2),
        "blob_bbox": [left, top, w, h],
    }


def compute_roi_median_brightness(
    frame: np.ndarray,
    roi: list[int] | None = None,
) -> float:
    """Median pixel brightness over the (optionally cropped) frame.

    More robust to outliers than mean for skewed brightness distributions.
    """
    crop = _crop_roi(normalize_frame_to_grayscale(frame), roi)
    return float(np.median(crop))


def compute_local_contrast(
    frame: np.ndarray,
    roi: list[int] | None = None,
    percentile: float = 99.0,
    epsilon: float = 1.0,
) -> dict[str, Any]:
    """Compute local contrast inside the ROI.

    ``local_contrast = top_percentile - median`` captures the brightness
    difference between the flashing target and the dark background,
    independent of absolute illumination levels.

    Returns
    -------
    dict
        ``roi_median_brightness``, ``roi_top_percentile_brightness``,
        ``local_contrast``, ``local_contrast_ratio``.
    """
    roi_median = compute_roi_median_brightness(frame, roi=roi)
    roi_top_pct = compute_top_percentile_brightness(frame, percentile=percentile, roi=roi)
    contrast = roi_top_pct - roi_median
    ratio = contrast / max(roi_median + epsilon, epsilon)
    return {
        "roi_median_brightness": round(roi_median, 2),
        "roi_top_percentile_brightness": round(roi_top_pct, 2),
        "local_contrast": round(contrast, 2),
        "local_contrast_ratio": round(ratio, 4),
    }

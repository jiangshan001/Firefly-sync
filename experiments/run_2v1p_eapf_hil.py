#!/usr/bin/env python3
"""2-virtual + 1-Pi EAPF mutual HIL validation modes.

Safe modes:

* ``frontend_multi_agent_smoke``: laptop/server only, no Pi camera required.
* ``roi_calibration``: save ROI overlay frames and signal diagnostics only.
* ``auto_roi_calibration``: auto-detect V0/V1 camera ROIs from simultaneous flashing.
* ``multi_roi_detection_test``: Pi camera detects V0/V1 ROIs, no closed-loop Pi adaptation.
* ``2v1p_eapf_smoke``: short closed-loop 2V+1P EAPF smoke trial.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from firefly_sync.core.event_based_consensus_pll import (
    ConsensusPLLConfig,
    EventBasedConsensusPLLOscillator,
)
from firefly_sync.multi_agent.hil_topology import build_mixed_reality_topology


_PiGPIOLED: Any = None
_MultiROIFlashDetector: Any = None
_PicameraFlashDetector: Any = None


def _ensure_hw(dry_run: bool = False) -> None:
    global _PiGPIOLED, _MultiROIFlashDetector, _PicameraFlashDetector
    if dry_run:
        return
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED as _PL
        from firefly_sync.hardware.multi_roi_flash_detector import MultiROIFlashDetector as _MRFD
        from firefly_sync.hardware.picamera_flash_detector import PicameraFlashDetector as _PFD
    except ImportError as exc:
        print(f"ERROR: hardware import failed: {exc}")
        sys.exit(1)
    _PiGPIOLED = _PL
    _MultiROIFlashDetector = _MRFD
    _PicameraFlashDetector = _PFD


def _api(api_base: str, path: str, method: str = "GET", data: dict | None = None,
         timeout: float = 5.0, events: list[dict] | None = None) -> dict:
    url = api_base.rstrip("/") + path
    started = time.monotonic()
    ok = 0
    error = ""
    try:
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"} if body else {},
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read())
        ok = 1
        return payload
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if events is not None:
            events.append({
                "monotonic_time_s": round(started, 6),
                "endpoint": f"{method} {path}",
                "ok": ok,
                "error": error,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            })


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    if fields is None:
        fields = sorted({k for row in rows for k in row.keys()})
    else:
        fields = list(fields)
        for key in sorted({k for row in rows for k in row.keys()}):
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _parse_roi(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    values = [int(p) for p in parts]
    if values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return values


def _make_out_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        out = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.log_dir) / f"{stamp}_{args.mode}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _apply_detection_preset(args: argparse.Namespace) -> None:
    args.v0_enabled = True
    args.v1_enabled = True
    if args.detection_preset == "none":
        return
    if args.detection_preset == "v0_only":
        args.v0_enabled = True
        args.v1_enabled = False
    elif args.detection_preset == "v1_only":
        args.v0_enabled = False
        args.v1_enabled = True
    elif args.detection_preset == "two_freq_1hz_2hz":
        args.v0_freq = 1.0
        args.v1_freq = 2.0
    elif args.detection_preset == "same_freq_phase_offset":
        args.v0_freq = 2.0
        args.v1_freq = 2.0
        args.v0_phase_rad = 0.0
        args.v1_phase_rad = math.pi
    elif args.detection_preset == "same_freq_near_simultaneous":
        args.v0_freq = 2.0
        args.v1_freq = 2.0
        args.v0_phase_rad = 0.0
        args.v1_phase_rad = 0.08


def _frame_for_display(frame: Any, camera_format: str) -> Any:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        if camera_format.upper().startswith("BGR"):
            rgb = rgb[:, :, ::-1]
        return rgb
    return arr


def _save_roi_debug_frame(
    frame: Any,
    args: argparse.Namespace,
    out_dir: Path,
    filename: str,
    rois: dict[str, list[int] | None] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    img = _frame_for_display(frame, args.camera_format)
    if np.asarray(img).ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    if rois is None:
        rois = {"V0": args.roi_v0, "V1": args.roi_v1}
    for label, roi, color in (
        ("ROI 0 / V0", rois.get("V0"), "lime"),
        ("ROI 1 / V1", rois.get("V1"), "cyan"),
    ):
        if roi is None:
            continue
        x, y, w, h = roi
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor=color, linewidth=2.0))
        ax.text(
            x,
            max(0, y - 4),
            f"{label} [{x},{y},{w},{h}]",
            color="black",
            fontsize=9,
            va="bottom",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )
    ax.set_axis_off()
    ax.set_title("Multi-ROI camera debug frame")
    path = out_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _save_raw_frame_debug(frame: Any, args: argparse.Namespace, out_dir: Path,
                          filename: str, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    img = _frame_for_display(frame, args.camera_format)
    if np.asarray(img).ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title)
    path = out_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _to_gray_float(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        gray = np.mean(arr[:, :, :3], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        gray = arr[:, :, 0]
    else:
        gray = arr
    return gray.astype(np.float32)


def _roi_mean(gray: np.ndarray, roi: list[int] | None) -> float | None:
    if roi is None:
        return None
    x, y, w, h = [int(v) for v in roi]
    height, width = gray.shape[:2]
    x0 = max(0, min(width, x))
    y0 = max(0, min(height, y))
    x1 = max(0, min(width, x + w))
    y1 = max(0, min(height, y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.mean(gray[y0:y1, x0:x1]))


def _background_mean(gray: np.ndarray, rois: dict[str, list[int] | None]) -> float:
    mask = np.ones(gray.shape[:2], dtype=bool)
    height, width = gray.shape[:2]
    for roi in rois.values():
        if roi is None:
            continue
        x, y, w, h = [int(v) for v in roi]
        x0 = max(0, min(width, x))
        y0 = max(0, min(height, y))
        x1 = max(0, min(width, x + w))
        y1 = max(0, min(height, y + h))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = False
    vals = gray[mask]
    if vals.size == 0:
        return float(np.mean(gray))
    return float(np.mean(vals))


def _camera_display_evidence(
    samples: list[dict[str, Any]],
    rois: dict[str, list[int] | None],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    if not samples:
        return {"available": False, "warnings": ["no_camera_samples"]}
    grays = [_to_gray_float(sample["frame"]) for sample in samples]
    roi_sum_scores: list[float] = []
    for gray in grays:
        score = 0.0
        for roi in rois.values():
            mean = _roi_mean(gray, roi)
            if mean is not None:
                score += mean
        roi_sum_scores.append(score)
    background_idx = int(np.argmin(roi_sum_scores)) if roi_sum_scores else 0
    background_frame = samples[background_idx]["frame"]
    background_gray = grays[background_idx]
    background_mean = _background_mean(background_gray, rois)

    evidence: dict[str, Any] = {
        "available": True,
        "background_frame_index": background_idx,
        "background_frame_t_s": round(float(samples[background_idx]["t_s"]), 6),
        "background_mean": round(background_mean, 6),
        "debug_files": {
            "camera_background_raw.jpg": str(_save_raw_frame_debug(
                background_frame,
                args,
                out_dir,
                "camera_background_raw.jpg",
                "Camera evidence: background / flash-off frame",
            )),
        },
        "agents": {},
        "warnings": [],
    }

    contrasts: list[float] = []
    for agent_id, roi in rois.items():
        if roi is None:
            evidence["agents"][agent_id] = {"roi": None, "warnings": ["missing_roi"]}
            continue
        roi_means = [_roi_mean(gray, roi) for gray in grays]
        valid = [(idx, mean) for idx, mean in enumerate(roi_means) if mean is not None]
        if not valid:
            evidence["agents"][agent_id] = {"roi": roi, "warnings": ["invalid_roi"]}
            continue
        flash_idx, flash_mean = max(valid, key=lambda item: item[1])
        contrast = float(flash_mean) - background_mean
        contrasts.append(contrast)
        filename = f"camera_{agent_id.lower()}_flash_on_raw.jpg"
        evidence["debug_files"][filename] = str(_save_raw_frame_debug(
            samples[flash_idx]["frame"],
            args,
            out_dir,
            filename,
            f"Camera evidence: {agent_id} flash-on frame",
        ))
        evidence["agents"][agent_id] = {
            "roi": roi,
            "flash_on_frame_index": int(flash_idx),
            "flash_on_frame_t_s": round(float(samples[flash_idx]["t_s"]), 6),
            "flash_on_roi_mean": round(float(flash_mean), 6),
            "background_mean": round(background_mean, 6),
            "measured_contrast": round(contrast, 6),
            "roi_mean_min": round(float(min(mean for _, mean in valid)), 6),
            "roi_mean_max": round(float(max(mean for _, mean in valid)), 6),
        }
    evidence["mean_measured_contrast"] = (
        round(float(np.mean(contrasts)), 6) if contrasts else None
    )
    return evidence


def _save_change_map_debug(frames: list[Any], out_dir: Path, filename: str) -> Path | None:
    if len(frames) < 2:
        return None
    stack = np.stack([_to_gray_float(frame) for frame in frames], axis=-1)
    change_map = np.std(stack, axis=-1)
    path = out_dir / filename
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(change_map, cmap="inferno")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_axis_off()
    ax.set_title("Temporal variance image")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _change_map_u8(frames: list[Any], method: str) -> np.ndarray | None:
    if len(frames) < 2:
        return None
    stack = np.stack([_to_gray_float(frame) for frame in frames], axis=-1)
    if method == "max_min_range":
        change_map = np.max(stack, axis=-1) - np.min(stack, axis=-1)
    elif method == "mean_abs_diff":
        change_map = np.mean(np.abs(np.diff(stack, axis=-1)), axis=-1)
    else:
        change_map = np.std(stack, axis=-1)
    cmin = float(np.min(change_map))
    cmax = float(np.max(change_map))
    if cmax - cmin < 1e-6:
        return np.zeros(change_map.shape, dtype=np.uint8)
    return ((change_map - cmin) / (cmax - cmin) * 255.0).astype(np.uint8)


def _save_threshold_mask_debug(frames: list[Any], out_dir: Path, filename: str,
                               method: str, threshold: float | None) -> Path | None:
    change = _change_map_u8(frames, method)
    if change is None:
        return None
    if threshold is None:
        nonzero = change[change > 0]
        thresh = float(np.percentile(nonzero, 80.0)) if nonzero.size else 0.0
    else:
        thresh = float(threshold)
    mask = (change > thresh).astype(np.uint8) * 255
    path = out_dir / filename
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f"Threshold mask (debug threshold={thresh:.1f})")
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _save_candidate_debug_frame(
    frame: Any,
    out_dir: Path,
    filename: str,
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    assigned_rois: dict[str, list[int] | None],
    camera_format: str,
) -> Path:
    selected_ids = {item.get("candidate_id") for item in selected}
    fig, ax = plt.subplots(figsize=(9, 6))
    img = _frame_for_display(frame, camera_format)
    if np.asarray(img).ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img)
    for item in candidates:
        roi = item.get("roi")
        if not roi:
            continue
        x, y, w, h = roi
        cid = item.get("candidate_id")
        selected_item = cid in selected_ids
        color = "yellow" if selected_item else "gray"
        linewidth = 2.0 if selected_item else 1.0
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor=color, linewidth=linewidth,
                                       linestyle="-" if selected_item else "--"))
        ax.text(
            x,
            y + h + 10,
            f"cand {cid} score={item.get('score')}",
            color="black",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.75, "pad": 2, "edgecolor": "none"},
        )
    for agent_id, roi in assigned_rois.items():
        if not roi:
            continue
        color = "lime" if agent_id == "V0" else "cyan"
        x, y, w, h = roi
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor=color, linewidth=3.0))
        ax.text(
            x,
            max(0, y - 4),
            f"{agent_id} selected [{x},{y},{w},{h}]",
            color="black",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2, "edgecolor": "none"},
        )
    ax.set_axis_off()
    ax.set_title("Auto ROI candidates and selected assignments")
    path = out_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def _save_trace_plot(traces: dict[str, dict[str, Any]], out_dir: Path,
                     filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, trace in traces.items():
        times = trace.get("times", [])
        values = trace.get("values", [])
        freq = trace.get("roi_estimated_frequency_hz", trace.get("estimated_frequency_hz"))
        suffix = f" ({freq:.2f} Hz)" if isinstance(freq, (int, float)) and math.isfinite(freq) else ""
        ax.plot(times, values, label=f"{label}{suffix}", linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Local contrast")
    ax.set_title("Selected ROI brightness traces")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    path = out_dir / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _capture_frames(detector: Any, duration_s: float, target_fps: float) -> list[Any]:
    frames: list[Any] = []
    period_s = 1.0 / max(1.0, target_fps)
    t0 = time.monotonic()
    next_t = t0
    while time.monotonic() - t0 < duration_s:
        now = time.monotonic()
        if now < next_t:
            time.sleep(min(0.01, next_t - now))
            continue
        frames.append(detector.capture_raw_frame())
        next_t += period_s
    return frames


def _capture_frame_sequence(detector: Any, duration_s: float,
                            target_fps: float) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    period_s = 1.0 / max(1.0, target_fps)
    t0 = time.monotonic()
    next_t = t0
    while time.monotonic() - t0 < duration_s:
        now = time.monotonic()
        if now < next_t:
            time.sleep(min(0.01, next_t - now))
            continue
        samples.append({
            "t_s": now - t0,
            "frame": detector.capture_raw_frame(),
        })
        next_t += period_s
    return samples


def _roi_from_locator_result(result: dict[str, Any] | None) -> list[int] | None:
    if not result:
        return None
    return [
        int(result["x"]),
        int(result["y"]),
        int(result["width"]),
        int(result["height"]),
    ]


def _roi_warnings(roi: list[int] | None, image_size: list[int],
                  signal_range: float, args: argparse.Namespace) -> list[str]:
    warnings: list[str] = []
    if roi is None:
        return ["no_flashing_blob_found"]
    x, y, w, h = roi
    width, height = image_size
    if x <= args.auto_roi_boundary_margin_px or y <= args.auto_roi_boundary_margin_px:
        warnings.append("roi_close_to_image_boundary")
    if x + w >= width - args.auto_roi_boundary_margin_px:
        warnings.append("roi_close_to_image_boundary")
    if y + h >= height - args.auto_roi_boundary_margin_px:
        warnings.append("roi_close_to_image_boundary")
    if w <= 0 or h <= 0:
        warnings.append("invalid_roi_size")
    if signal_range < args.min_amplitude:
        warnings.append("signal_amplitude_too_low")
    return sorted(set(warnings))


def _roi_overlap_ratio(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    min_area = max(1, min(aw * ah, bw * bh))
    return inter / min_area


def _roi_signal_stats_from_frames(frames: list[Any], roi: list[int] | None) -> dict[str, Any]:
    if roi is None:
        return {"raw_signal": _stats([]), "raw_signal_range": 0.0}
    from firefly_sync.hardware.flash_detector import compute_local_contrast

    values = [
        compute_local_contrast(frame, roi=roi, percentile=99.0)["local_contrast"]
        for frame in frames
    ]
    stats = _stats(values)
    signal_range = (
        float(stats["max"]) - float(stats["min"])
        if stats["max"] is not None and stats["min"] is not None else 0.0
    )
    return {"raw_signal": stats, "raw_signal_range": round(signal_range, 6)}


def _estimate_trace_rising_edges(
    times: list[float],
    values: list[float],
    *,
    max_freq_hz: float = 3.5,
) -> dict[str, Any]:
    if len(times) < 3 or len(values) != len(times):
        return {
            "frequency_hz": None,
            "edge_count": 0,
            "edge_times_s": [],
            "on_threshold": None,
            "off_threshold": None,
            "amplitude": 0.0,
        }
    vals = np.asarray(values, dtype=float)
    low = float(np.percentile(vals, 20.0))
    high = float(np.percentile(vals, 90.0))
    amplitude = high - low
    if amplitude <= 1e-9:
        return {
            "frequency_hz": None,
            "edge_count": 0,
            "edge_times_s": [],
            "on_threshold": high,
            "off_threshold": low,
            "amplitude": 0.0,
        }

    on_threshold = low + 0.55 * amplitude
    off_threshold = low + 0.30 * amplitude
    min_edge_interval_s = 0.45 / max(0.1, max_freq_hz)
    edge_times: list[float] = []
    state_on = False
    last_edge = -math.inf
    for t, value in zip(times, values):
        if not state_on and value >= on_threshold and t - last_edge >= min_edge_interval_s:
            edge_times.append(float(t))
            last_edge = float(t)
            state_on = True
        elif state_on and value <= off_threshold:
            state_on = False

    frequency = None
    if len(edge_times) >= 2:
        span = edge_times[-1] - edge_times[0]
        if span > 0:
            frequency = (len(edge_times) - 1) / span
    return {
        "frequency_hz": round(float(frequency), 6) if frequency else None,
        "edge_count": len(edge_times),
        "edge_times_s": [round(float(t), 6) for t in edge_times],
        "on_threshold": round(on_threshold, 6),
        "off_threshold": round(off_threshold, 6),
        "amplitude": round(amplitude, 6),
    }


def _trace_for_roi(samples: list[dict[str, Any]], roi: list[int]) -> dict[str, Any]:
    from firefly_sync.hardware.flash_detector import compute_local_contrast
    from firefly_sync.hardware.signal_detector import estimate_frequency_autocorrelation

    times = [float(sample["t_s"]) for sample in samples]
    values = [
        float(compute_local_contrast(sample["frame"], roi=roi, percentile=99.0)["local_contrast"])
        for sample in samples
    ]
    freq, confidence = estimate_frequency_autocorrelation(
        times,
        values,
        min_freq_hz=0.4,
        max_freq_hz=3.5,
    )
    edge_estimate = _estimate_trace_rising_edges(times, values, max_freq_hz=3.5)
    edge_freq = edge_estimate.get("frequency_hz")
    primary_freq = edge_freq if edge_freq is not None else (round(float(freq), 6) if freq else None)
    frequency_method = "rising_edge" if edge_freq is not None else "autocorrelation"
    stats = _stats(values)
    signal_range = (
        float(stats["max"]) - float(stats["min"])
        if stats["max"] is not None and stats["min"] is not None else 0.0
    )
    return {
        "times": times,
        "values": values,
        "estimated_frequency_hz": primary_freq,
        "roi_estimated_frequency_hz": primary_freq,
        "trace_frequency_method": frequency_method,
        "autocorrelation_frequency_hz": round(float(freq), 6) if freq else None,
        "frequency_confidence": round(float(confidence), 6),
        "trace_rising_edge_count": edge_estimate["edge_count"],
        "trace_rising_edge_times_s": edge_estimate["edge_times_s"],
        "trace_on_threshold": edge_estimate["on_threshold"],
        "trace_off_threshold": edge_estimate["off_threshold"],
        "trace_threshold_amplitude": edge_estimate["amplitude"],
        "raw_signal": stats,
        "raw_signal_range": round(signal_range, 6),
    }


def _assign_regions_to_agents(
    regions: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, list[str]]:
    traces: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if len(regions) < 2:
        return {}, traces, "failed", ["only_one_valid_flashing_component_found"]

    for i, region in enumerate(regions[:2]):
        label = f"candidate_{i}"
        trace = _trace_for_roi(samples, region["roi"])
        trace["roi"] = region["roi"]
        trace["candidate_id"] = region.get("candidate_id")
        trace["score"] = region.get("score")
        traces[label] = trace

    labels = list(traces.keys())
    freqs = {
        label: traces[label].get("roi_estimated_frequency_hz", traces[label].get("estimated_frequency_hz"))
        for label in labels
    }
    valid_freqs = {
        label: float(freq)
        for label, freq in freqs.items()
        if isinstance(freq, (int, float)) and math.isfinite(float(freq)) and float(freq) > 0
    }
    assignment_method = "frequency_based"
    assignments: dict[str, dict[str, Any]]
    if len(valid_freqs) == 2:
        first, second = labels[0], labels[1]
        v0_target = float(args.auto_roi_v0_frequency)
        v1_target = float(args.auto_roi_v1_frequency)
        cost_normal = abs(valid_freqs[first] - v0_target) + abs(valid_freqs[second] - v1_target)
        cost_swapped = abs(valid_freqs[first] - v1_target) + abs(valid_freqs[second] - v0_target)
        if abs(cost_normal - cost_swapped) < args.auto_roi_frequency_ambiguity_hz:
            assignment_method = "left_right_fallback"
            warnings.append("frequency_assignment_ambiguous")
        elif cost_normal <= cost_swapped:
            assignments = {"V0": traces[first], "V1": traces[second]}
            return assignments, traces, assignment_method, warnings
        else:
            assignments = {"V0": traces[second], "V1": traces[first]}
            return assignments, traces, assignment_method, warnings
    else:
        assignment_method = "left_right_fallback"
        warnings.append("frequency_estimation_ambiguous")

    left_to_right = sorted(
        traces.values(),
        key=lambda trace: trace["roi"][0] + trace["roi"][2] / 2.0,
    )
    return {"V0": left_to_right[0], "V1": left_to_right[1]}, traces, assignment_method, warnings


def _finite_floats(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            result.append(f)
    return result


def _stats(values: list[Any]) -> dict[str, float | int | None]:
    vals = _finite_floats(values)
    if not vals:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
    }


def _frequency_from_event_times(times: list[float]) -> float | None:
    if len(times) < 2:
        return None
    intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    intervals = [v for v in intervals if v > 0]
    if not intervals:
        return None
    return round(1.0 / float(np.median(intervals)), 6)


def _roi_signal_summary(
    roi_rows: list[dict],
    events: list[dict],
    actual_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for agent_id in ("V0", "V1"):
        rows = [row for row in roi_rows if row.get("agent_id") == agent_id]
        evts = [evt for evt in events if evt.get("agent_id") == agent_id]
        raw_stats = _stats([row.get("raw_brightness", row.get("brightness_used")) for row in rows])
        norm_stats = _stats([row.get("normalized_signal", row.get("signal_norm")) for row in rows])
        detected_count = len(evts)
        actual_count = int(actual_counts.get(agent_id, 0))
        raw_range = (
            float(raw_stats["max"]) - float(raw_stats["min"])
            if raw_stats["max"] is not None and raw_stats["min"] is not None else 0.0
        )
        norm_max = float(norm_stats["max"]) if norm_stats["max"] is not None else 0.0
        warnings: list[str] = []
        if detected_count == 0:
            warnings.append("no_rising_edges_detected")
        if raw_range < args.min_amplitude:
            warnings.append("raw_signal_amplitude_too_low")
        if norm_stats["count"] and norm_max < args.norm_on_threshold:
            warnings.append("normalised_signal_never_reached_on_threshold")
        summary[agent_id] = {
            "roi_id": 0 if agent_id == "V0" else 1,
            "roi": args.roi_v0 if agent_id == "V0" else args.roi_v1,
            "detected_rising_edge_count": detected_count,
            "actual_virtual_flash_count": actual_count,
            "detection_fcr": round(detected_count / actual_count, 6) if actual_count > 0 else None,
            "raw_signal": raw_stats,
            "normalized_signal": norm_stats,
            "raw_signal_range": round(raw_range, 6),
            "estimated_detected_frequency_hz": _frequency_from_event_times(
                [float(evt["t_s"]) for evt in evts if evt.get("t_s") is not None]
            ),
            "last_detector_frequency_hz": rows[-1].get("estimated_frequency_hz") if rows else None,
            "warnings": warnings,
        }
    return summary


def _fetch_virtual_flash_counts(args: argparse.Namespace,
                                api_events: list[dict]) -> dict[str, int]:
    try:
        payload = _api(args.leader_api, "/api/agents", "GET",
                       timeout=args.api_timeout, events=api_events)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for agent in payload.get("agents", []):
        aid = agent.get("agent_id")
        if aid in ("V0", "V1"):
            counts[aid] = int(agent.get("fire_count") or 0)
    return counts


def _fetch_virtual_agent_snapshot(args: argparse.Namespace,
                                  api_events: list[dict]) -> dict[str, dict[str, Any]]:
    try:
        payload = _api(args.leader_api, "/api/agents", "GET",
                       timeout=args.api_timeout, events=api_events)
    except Exception:
        return {}
    snapshot: dict[str, dict[str, Any]] = {}
    for agent in payload.get("agents", []):
        aid = agent.get("agent_id")
        if aid not in ("V0", "V1"):
            continue
        snapshot[aid] = {
            "agent_id": aid,
            "enabled": bool(agent.get("enabled", True)),
            "initial_frequency_hz": agent.get("initial_frequency_hz"),
            "frequency_hz": agent.get("frequency_hz"),
            "initial_phase_rad": agent.get("initial_phase_rad"),
            "phase_rad": agent.get("phase_rad"),
            "fire_count": int(agent.get("fire_count") or 0),
            "flash_on": bool(agent.get("flash_on", False)),
        }
    return snapshot


def _server_flash_diagnostics(
    start_snapshot: dict[str, dict[str, Any]],
    end_snapshot: dict[str, dict[str, Any]],
    duration_s: float,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    duration = max(0.0, float(duration_s))
    for agent_id in ("V0", "V1"):
        start_count = int(start_snapshot.get(agent_id, {}).get("fire_count") or 0)
        end_count = int(end_snapshot.get(agent_id, {}).get("fire_count") or 0)
        count = max(0, end_count - start_count)
        diagnostics[agent_id] = {
            "start_fire_count": start_count,
            "end_fire_count": end_count,
            "flash_count": count,
            "frequency_estimate_hz": round(count / duration, 6) if duration > 0 else None,
        }
    return diagnostics


def _load_roi_config(path: str | None) -> tuple[list[int] | None, list[int] | None]:
    if not path:
        return None, None
    with open(path, "r") as f:
        payload = json.load(f)
    v0 = payload.get("V0", {}).get("roi")
    v1 = payload.get("V1", {}).get("roi")
    return (list(v0) if v0 else None, list(v1) if v1 else None)


def _apply_roi_config(args: argparse.Namespace) -> None:
    cfg_v0, cfg_v1 = _load_roi_config(args.roi_config)
    if args.roi_v0 is None and cfg_v0 is not None:
        args.roi_v0 = cfg_v0
    if args.roi_v1 is None and cfg_v1 is not None:
        args.roi_v1 = cfg_v1


def _locked_eapf_config(freq: float) -> ConsensusPLLConfig:
    return ConsensusPLLConfig(
        natural_frequency_hz=freq,
        phase_gain=0.02,
        frequency_gain=0.02,
        phase_error_filter_alpha=0.2,
        frequency_error_filter_alpha=0.2,
        max_phase_step_rad=0.2,
        max_frequency_step_hz=0.05,
        frequency_min_hz=0.8,
        frequency_max_hz=3.2,
    )


def _configure_server(args: argparse.Namespace, api_events: list[dict]) -> None:
    _api(args.leader_api, "/api/mode", "POST", {
        "mode": "mutual_hil_multi",
        "topology": args.topology,
    }, timeout=args.api_timeout, events=api_events)
    _api(args.leader_api, "/api/leader/config", "POST", {
        "background_brightness": args.background_brightness,
        "brightness_on": args.flash_brightness,
        "brightness_off": args.off_brightness,
    }, timeout=args.api_timeout, events=api_events)
    _api(args.leader_api, "/api/mutual/config", "POST", {
        "mutual_agent_mode": "multi_2v1p",
        "topology": args.topology,
    }, timeout=args.api_timeout, events=api_events)
    v0_size = args.v0_size if args.v0_size is not None else args.dot_size
    v1_size = args.v1_size if args.v1_size is not None else args.dot_size
    positions = [
        (
            0,
            "V0",
            args.v0_freq,
            args.v0_x,
            args.v0_y,
            v0_size,
            args.v0_phase_rad,
            args.v0_enabled,
            getattr(args, "v0_flash_brightness", args.flash_brightness),
            getattr(args, "v0_off_brightness", args.off_brightness),
            getattr(args, "v0_background_brightness", args.background_brightness),
        ),
        (
            1,
            "V1",
            args.v1_freq,
            args.v1_x,
            args.v1_y,
            v1_size,
            args.v1_phase_rad,
            args.v1_enabled,
            getattr(args, "v1_flash_brightness", args.flash_brightness),
            getattr(args, "v1_off_brightness", args.off_brightness),
            getattr(args, "v1_background_brightness", args.background_brightness),
        ),
    ]
    for idx, agent_id, freq, x, y, size, phase_rad, enabled, on_bright, off_bright, bg_bright in positions:
        _api(args.leader_api, f"/api/agents/{idx}", "POST", {
            "agent_id": agent_id,
            "role": "virtual",
            "model": "eapf_consensus",
            "initial_frequency_hz": freq,
            "frequency_hz": freq,
            "initial_phase_rad": phase_rad,
            "x": x,
            "y": y,
            "size": size,
            "brightness_on": on_bright,
            "brightness_off": off_bright,
            "background_brightness": bg_bright,
            "enabled": enabled,
        }, timeout=args.api_timeout, events=api_events)
    _api(args.leader_api, "/api/agents/2", "POST", {
        "agent_id": "P0",
        "role": "pi",
        "model": "eapf_consensus",
        "initial_frequency_hz": args.pi_freq,
        "frequency_hz": args.pi_freq,
        "initial_phase_rad": getattr(args, "pi_phase_rad", 0.0),
        "enabled": True,
    }, timeout=args.api_timeout, events=api_events)


def _start_server_trial(args: argparse.Namespace, api_events: list[dict], feedback: bool) -> None:
    _api(args.leader_api, "/api/reset", "POST", {}, timeout=args.api_timeout, events=api_events)
    _configure_server(args, api_events)
    _api(args.leader_api, "/api/start", "POST", {}, timeout=args.api_timeout, events=api_events)
    _api(args.leader_api, "/api/feedback", "POST", {"enabled": feedback},
         timeout=args.api_timeout, events=api_events)


def _poll_agents(args: argparse.Namespace, out_dir: Path, api_events: list[dict],
                 duration_s: float) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    events: list[dict] = []
    last_counts: dict[str, int] = {}
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration_s:
        payload = _api(args.leader_api, "/api/agents", "GET",
                       timeout=args.api_timeout, events=api_events)
        t = time.monotonic() - t0
        for agent in payload.get("agents", []):
            aid = agent.get("agent_id", str(agent.get("id")))
            row = {
                "t_s": round(t, 6),
                "agent_id": aid,
                "role": agent.get("role"),
                "frequency_hz": agent.get("frequency_hz"),
                "phase_rad": agent.get("phase_rad"),
                "flash_on": int(bool(agent.get("flash_on"))),
                "fire_count": agent.get("fire_count"),
                "received_neighbour_events": agent.get("received_neighbour_events"),
                "pi_flash_events_consumed": agent.get("pi_flash_events_consumed"),
                "topology": agent.get("topology"),
            }
            rows.append(row)
            count = int(agent.get("fire_count") or 0)
            if count > last_counts.get(aid, 0):
                events.append({
                    "t_s": round(t, 6),
                    "event_type": "virtual_flash" if agent.get("role") == "virtual" else "pi_flash",
                    "agent_id": aid,
                    "fire_count": count,
                })
            last_counts[aid] = count
        time.sleep(max(0.05, args.poll_interval))
    _write_csv(out_dir / "virtual_agents.csv", rows)
    return rows, events


def run_frontend_smoke(args: argparse.Namespace, out_dir: Path) -> dict:
    api_events: list[dict] = []
    _start_server_trial(args, api_events, feedback=True)
    rows, events = _poll_agents(args, out_dir, api_events, args.duration)
    _api(args.leader_api, "/api/pause", "POST", {}, timeout=args.api_timeout, events=api_events)
    _write_csv(out_dir / "api_events.csv", api_events)
    _write_csv(out_dir / "events_all.csv", events)
    counts: dict[str, int] = {}
    final_freq: dict[str, float] = {}
    for row in rows:
        counts[row["agent_id"]] = int(row.get("fire_count") or 0)
        if row.get("frequency_hz") not in (None, ""):
            final_freq[row["agent_id"]] = float(row["frequency_hz"])
    return {
        "mode": args.mode,
        "topology": args.topology,
        "duration_s": args.duration,
        "flash_counts": counts,
        "final_frequency_hz": final_freq,
        "api_post_success_count": sum(1 for e in api_events if e.get("ok") == 1),
    }


def _make_multi_roi_detector(args: argparse.Namespace):
    assert _MultiROIFlashDetector is not None
    return _MultiROIFlashDetector(
        rois=[
            {"roi_id": 0, "agent_id": "V0", "roi": args.roi_v0},
            {"roi_id": 1, "agent_id": "V1", "roi": args.roi_v1},
        ],
        resolution=[args.width, args.height],
        detection_mode="local_contrast",
        min_interval_s=args.min_interval,
        window_s=args.window_s,
        norm_on_threshold=args.norm_on_threshold,
        norm_off_threshold=args.norm_off_threshold,
        min_amplitude=args.min_amplitude,
        target_fps=args.camera_fps,
        frame_rate=args.camera_fps,
        camera_format=args.camera_format,
        episode_latch_enabled=args.episode_latch,
    )


def run_detection_test(args: argparse.Namespace, out_dir: Path) -> dict:
    api_events: list[dict] = []
    _start_server_trial(args, api_events, feedback=False)
    if args.dry_run:
        _write_csv(out_dir / "api_events.csv", api_events)
        return {"mode": args.mode, "dry_run": True, "requires_hardware": True}

    _ensure_hw(False)
    detector = _make_multi_roi_detector(args)
    roi_rows: list[dict] = []
    events: list[dict] = []
    debug_frames: list[str] = []
    actual_counts: dict[str, int] = {}
    try:
        detector.start()
        t0 = time.monotonic()
        mid_frame_saved = False
        while time.monotonic() - t0 < args.duration:
            raw_frame = detector.capture_raw_frame()
            t = time.monotonic() - t0
            if not debug_frames:
                debug_frames.append(str(_save_roi_debug_frame(
                    raw_frame, args, out_dir, "roi_debug_frame_start.jpg"
                )))
            if (
                args.save_mid_roi_debug_frame
                and not mid_frame_saved
                and t >= max(0.0, args.duration / 2.0)
            ):
                debug_frames.append(str(_save_roi_debug_frame(
                    raw_frame, args, out_dir, "roi_debug_frame_mid.jpg"
                )))
                mid_frame_saved = True
            res = detector.process_frame(raw_frame)
            for row in res["roi_results"]:
                out = dict(row)
                out["t_s"] = round(t, 6)
                roi_rows.append(out)
            for evt in res["events"]:
                events.append({
                    "t_s": round(t, 6),
                    "event_type": "pi_detected_virtual_flash",
                    "agent_id": evt["agent_id"],
                    "roi_id": evt["roi_id"],
                    "raw_brightness": evt["raw_brightness"],
                    "normalized_signal": evt["normalized_signal"],
                })
        actual_counts = _fetch_virtual_flash_counts(args, api_events)
    finally:
        detector.stop()
        _api(args.leader_api, "/api/pause", "POST", {}, timeout=args.api_timeout, events=api_events)

    _write_csv(out_dir / "pi_detection_roi.csv", roi_rows, [
        "t_s", "timestamp_s", "frame_index", "roi_id", "agent_id",
        "raw_brightness", "brightness_used", "normalized_signal", "signal_norm",
        "state", "rising_edge", "accepted_flash_event", "event_type",
        "threshold_on", "threshold_off", "norm_on_threshold", "norm_off_threshold",
        "adaptive_low", "adaptive_high", "adaptive_amplitude",
        "estimated_frequency_hz", "signal_frequency_hz",
    ])
    _write_csv(out_dir / "events_all.csv", events)
    _write_csv(out_dir / "api_events.csv", api_events)
    counts = {aid: sum(1 for e in events if e["agent_id"] == aid) for aid in ("V0", "V1")}
    roi_summary = _roi_signal_summary(roi_rows, events, actual_counts, args)
    return {
        "mode": args.mode,
        "topology": args.topology,
        "detection_preset": args.detection_preset,
        "detected_counts": counts,
        "actual_virtual_flash_counts": actual_counts,
        "roi_signal_summary": roi_summary,
        "roi_debug_frames": debug_frames,
        "closed_loop_adaptation_enabled": False,
    }


def _set_auto_sequence_state(
    args: argparse.Namespace,
    api_events: list[dict],
    *,
    v0_enabled: bool,
    v1_enabled: bool,
) -> None:
    old_v0 = args.v0_enabled
    old_v1 = args.v1_enabled
    args.v0_enabled = v0_enabled
    args.v1_enabled = v1_enabled
    try:
        _start_server_trial(args, api_events, feedback=False)
    finally:
        args.v0_enabled = old_v0
        args.v1_enabled = old_v1
    time.sleep(max(0.0, args.auto_roi_warmup_s))


def run_auto_roi_calibration(args: argparse.Namespace, out_dir: Path) -> dict:
    api_events: list[dict] = []
    if args.dry_run:
        _write_csv(out_dir / "api_events.csv", api_events)
        return {"mode": args.mode, "dry_run": True, "requires_hardware": True}

    _ensure_hw(False)
    assert _PicameraFlashDetector is not None
    from firefly_sync.hardware.roi_locator import locate_flashing_region, locate_flashing_regions

    detector = _PicameraFlashDetector(
        resolution=[args.width, args.height],
        detection_mode="local_contrast",
        roi=None,
        target_fps=int(args.camera_fps),
        frame_rate=args.camera_fps,
        camera_format=args.camera_format,
    )
    sequential_results: dict[str, dict[str, Any]] = {}
    debug_files: dict[str, str | None] = {}
    image_size = [args.width, args.height]
    combined_locator: dict[str, Any] = {}
    assignments: dict[str, dict[str, Any]] = {}
    assignment_method = "failed"
    assignment_warnings: list[str] = []
    combined_server_start: dict[str, dict[str, Any]] = {}
    combined_server_end: dict[str, dict[str, Any]] = {}
    combined_server_diagnostics: dict[str, dict[str, Any]] = {}
    combined_mutual_config_snapshot: dict[str, Any] = {}
    frontend_display_state_snapshot: dict[str, Any] = {}
    camera_display_evidence: dict[str, Any] = {}
    combined_capture_duration_s = 0.0
    calibration_valid = False
    failure_reason = ""

    try:
        detector.start()
        if args.auto_roi_sequential_diagnostics:
            # Optional diagnostic only. Final ROIs come from the combined
            # multi-component stage below.
            steps = [
                ("V0", True, False, "auto_roi_v0_debug.jpg", "auto_roi_v0_temporal_variance.jpg"),
                ("V1", False, True, "auto_roi_v1_debug.jpg", "auto_roi_v1_temporal_variance.jpg"),
            ]
            for agent_id, v0_on, v1_on, debug_name, change_name in steps:
                _set_auto_sequence_state(args, api_events, v0_enabled=v0_on, v1_enabled=v1_on)
                frames = _capture_frames(detector, args.auto_roi_duration, args.auto_roi_capture_fps)
                if frames:
                    frame_arr = np.asarray(frames[0])
                    image_size = [int(frame_arr.shape[1]), int(frame_arr.shape[0])]
                result = locate_flashing_region(
                    frames,
                    method=args.auto_roi_method,
                    min_area_px=args.auto_roi_min_area,
                    padding_px=args.auto_roi_padding,
                    downsample=args.auto_roi_downsample,
                    change_threshold=args.auto_roi_change_threshold,
                )
                roi = _roi_from_locator_result(result)
                signal = _roi_signal_stats_from_frames(frames, roi)
                warnings = _roi_warnings(roi, image_size, float(signal["raw_signal_range"]), args)
                info = {
                    "roi": roi,
                    "method": args.auto_roi_method,
                    "score": result.get("score") if result else None,
                    "confidence": result.get("confidence") if result else 0.0,
                    "area_px": result.get("area_px") if result else 0,
                    "mean_change": result.get("mean_change") if result else None,
                    "max_change": result.get("max_change") if result else None,
                    "raw_signal": signal["raw_signal"],
                    "raw_signal_range": signal["raw_signal_range"],
                    "warnings": warnings,
                }
                sequential_results[agent_id] = info
                if frames:
                    overlay_rois = {"V0": roi if agent_id == "V0" else None,
                                    "V1": roi if agent_id == "V1" else None}
                    debug_files[debug_name] = str(_save_roi_debug_frame(
                        frames[-1], args, out_dir, debug_name, rois=overlay_rois,
                    ))
                    change_path = _save_change_map_debug(frames, out_dir, change_name)
                    debug_files[change_name] = str(change_path) if change_path is not None else None

        old_v0_freq, old_v1_freq = args.v0_freq, args.v1_freq
        old_v0_phase, old_v1_phase = args.v0_phase_rad, args.v1_phase_rad
        old_v0_enabled, old_v1_enabled = args.v0_enabled, args.v1_enabled
        args.v0_freq = args.auto_roi_v0_frequency
        args.v1_freq = args.auto_roi_v1_frequency
        args.v0_phase_rad = 0.0
        args.v1_phase_rad = args.auto_roi_v1_phase_rad
        args.v0_enabled = True
        args.v1_enabled = True
        try:
            _start_server_trial(args, api_events, feedback=False)
        finally:
            args.v0_freq, args.v1_freq = old_v0_freq, old_v1_freq
            args.v0_phase_rad, args.v1_phase_rad = old_v0_phase, old_v1_phase
            args.v0_enabled, args.v1_enabled = old_v0_enabled, old_v1_enabled
        time.sleep(max(0.0, args.auto_roi_warmup_s))
        combined_server_start = _fetch_virtual_agent_snapshot(args, api_events)
        try:
            combined_mutual_config_snapshot = _api(
                args.leader_api,
                "/api/mutual/config",
                "GET",
                timeout=args.api_timeout,
                events=api_events,
            )
        except Exception:
            combined_mutual_config_snapshot = {}

        combined_capture_started = time.monotonic()
        combined_samples = _capture_frame_sequence(
            detector,
            args.auto_roi_combined_duration,
            args.auto_roi_capture_fps,
        )
        combined_capture_duration_s = time.monotonic() - combined_capture_started
        combined_server_end = _fetch_virtual_agent_snapshot(args, api_events)
        combined_server_diagnostics = _server_flash_diagnostics(
            combined_server_start,
            combined_server_end,
            combined_capture_duration_s,
        )
        combined_frames = [sample["frame"] for sample in combined_samples]
        if combined_frames:
            frame_arr = np.asarray(combined_frames[0])
            image_size = [int(frame_arr.shape[1]), int(frame_arr.shape[0])]
        combined_locator = locate_flashing_regions(
            combined_frames,
            method=args.auto_roi_method,
            max_regions=2,
            min_area_px=args.auto_roi_min_area,
            padding_px=args.auto_roi_padding,
            downsample=args.auto_roi_downsample,
            change_threshold=args.auto_roi_change_threshold,
            max_overlap_ratio=args.auto_roi_overlap_warning_ratio,
            max_area_fraction=args.auto_roi_max_area_fraction,
        )
        selected_regions = combined_locator.get("regions", [])
        assignments, traces, assignment_method, assignment_warnings = _assign_regions_to_agents(
            selected_regions,
            combined_samples,
            args,
        )
        v0_roi = assignments.get("V0", {}).get("roi")
        v1_roi = assignments.get("V1", {}).get("roi")
        overlap = _roi_overlap_ratio(v0_roi, v1_roi)
        final_warnings: dict[str, list[str]] = {"V0": [], "V1": []}
        if len(selected_regions) < 2:
            failure_reason = "only_one_valid_flashing_component_found"
        elif overlap > args.auto_roi_overlap_warning_ratio:
            failure_reason = "selected_rois_overlap"
            for agent_id in ("V0", "V1"):
                final_warnings[agent_id].append("v0_v1_rois_overlap")
        elif assignment_method == "failed":
            failure_reason = "assignment_failed"
        else:
            calibration_valid = True

        for agent_id, trace in assignments.items():
            signal_range = float(trace.get("raw_signal_range") or 0.0)
            final_warnings[agent_id].extend(_roi_warnings(
                trace.get("roi"), image_size, signal_range, args
            ))
            final_warnings[agent_id].extend(assignment_warnings)

        if combined_frames:
            debug_files["auto_roi_combined_debug.jpg"] = str(_save_roi_debug_frame(
                combined_frames[-1], args, out_dir, "auto_roi_combined_debug.jpg",
                rois={"V0": v0_roi, "V1": v1_roi},
            ))
            camera_display_evidence = _camera_display_evidence(
                combined_samples,
                {"V0": v0_roi, "V1": v1_roi},
                args,
                out_dir,
            )
            for name, path in camera_display_evidence.get("debug_files", {}).items():
                debug_files[name] = path
            debug_files["auto_roi_candidates_debug.jpg"] = str(_save_candidate_debug_frame(
                combined_frames[-1],
                out_dir,
                "auto_roi_candidates_debug.jpg",
                combined_locator.get("candidates", []),
                selected_regions,
                {"V0": v0_roi, "V1": v1_roi},
                args.camera_format,
            ))
            change_path = _save_change_map_debug(
                combined_frames,
                out_dir,
                "auto_roi_combined_temporal_variance.jpg",
            )
            debug_files["auto_roi_combined_temporal_variance.jpg"] = (
                str(change_path) if change_path is not None else None
            )
            mask_path = _save_threshold_mask_debug(
                combined_frames,
                out_dir,
                "auto_roi_combined_threshold_mask.jpg",
                args.auto_roi_method,
                combined_locator.get("auto_threshold"),
            )
            debug_files["auto_roi_combined_threshold_mask.jpg"] = (
                str(mask_path) if mask_path is not None else None
            )
        if assignments:
            trace_plot_data = {
                agent_id: {
                    "times": trace.get("times", []),
                    "values": trace.get("values", []),
                    "roi_estimated_frequency_hz": trace.get("roi_estimated_frequency_hz"),
                }
                for agent_id, trace in assignments.items()
            }
            debug_files["auto_roi_selected_traces.png"] = str(_save_trace_plot(
                trace_plot_data,
                out_dir,
                "auto_roi_selected_traces.png",
            ))
        try:
            frontend_display_state_snapshot = _api(
                args.leader_api,
                "/api/frontend/display_state",
                "GET",
                timeout=args.api_timeout,
                events=api_events,
            )
        except Exception:
            frontend_display_state_snapshot = {}
    finally:
        detector.stop()
        _api(args.leader_api, "/api/pause", "POST", {}, timeout=args.api_timeout, events=api_events)

    def _agent_payload(agent_id: str) -> dict[str, Any]:
        trace = assignments.get(agent_id, {})
        roi = trace.get("roi") if calibration_valid else None
        server_diag = combined_server_diagnostics.get(agent_id, {})
        return {
            "roi": roi,
            "method": "combined_multi_component",
            "requested_frequency_hz": (
                args.auto_roi_v0_frequency if agent_id == "V0" else args.auto_roi_v1_frequency
            ),
            "score": trace.get("score"),
            "candidate_id": trace.get("candidate_id"),
            "estimated_frequency_hz": trace.get("estimated_frequency_hz"),
            "roi_estimated_frequency_hz": trace.get("roi_estimated_frequency_hz"),
            "trace_frequency_method": trace.get("trace_frequency_method"),
            "trace_rising_edge_count": trace.get("trace_rising_edge_count"),
            "trace_rising_edge_times_s": trace.get("trace_rising_edge_times_s"),
            "trace_on_threshold": trace.get("trace_on_threshold"),
            "trace_off_threshold": trace.get("trace_off_threshold"),
            "autocorrelation_frequency_hz": trace.get("autocorrelation_frequency_hz"),
            "frequency_confidence": trace.get("frequency_confidence"),
            "actual_virtual_flash_count": server_diag.get("flash_count"),
            "actual_virtual_frequency_estimate": server_diag.get("frequency_estimate_hz"),
            "raw_signal": trace.get("raw_signal"),
            "raw_signal_range": trace.get("raw_signal_range"),
            "warnings": final_warnings.get(agent_id, []),
        }

    if not calibration_valid and not failure_reason:
        failure_reason = "calibration_invalid"
    if not calibration_valid:
        print(f"Auto ROI calibration failed: {failure_reason.replace('_', ' ')}.")

    payload = {
        "V0": _agent_payload("V0"),
        "V1": _agent_payload("V1"),
        "candidate_components": combined_locator.get("candidates", []),
        "selected_components": combined_locator.get("regions", []),
        "rejected_candidate_components": combined_locator.get("rejected_candidates", []),
        "assignment_method": assignment_method,
        "image_size": image_size,
        "timestamp": datetime.now().isoformat(),
        "calibration_sequence": (
            ["v0_only_diagnostic", "v1_only_diagnostic", "combined_simultaneous"]
            if args.auto_roi_sequential_diagnostics else ["combined_simultaneous"]
        ),
        "sequential_diagnostics": sequential_results,
        "requested_v0_frequency_hz": args.auto_roi_v0_frequency,
        "requested_v1_frequency_hz": args.auto_roi_v1_frequency,
        "requested_v1_phase_rad": args.auto_roi_v1_phase_rad,
        "combined_api_config_requested": {
            "V0": {
                "frequency_hz": args.auto_roi_v0_frequency,
                "phase_rad": 0.0,
                "enabled": True,
            },
            "V1": {
                "frequency_hz": args.auto_roi_v1_frequency,
                "phase_rad": args.auto_roi_v1_phase_rad,
                "enabled": True,
            },
        },
        "combined_server_start_snapshot": combined_server_start,
        "combined_server_end_snapshot": combined_server_end,
        "combined_mutual_config_snapshot": combined_mutual_config_snapshot,
        "frontend_display_state_snapshot": frontend_display_state_snapshot,
        "camera_display_evidence": camera_display_evidence,
        "combined_capture_duration_s": round(float(combined_capture_duration_s), 6),
        "actual_v0_flash_count": combined_server_diagnostics.get("V0", {}).get("flash_count"),
        "actual_v1_flash_count": combined_server_diagnostics.get("V1", {}).get("flash_count"),
        "actual_v0_frequency_estimate": (
            combined_server_diagnostics.get("V0", {}).get("frequency_estimate_hz")
        ),
        "actual_v1_frequency_estimate": (
            combined_server_diagnostics.get("V1", {}).get("frequency_estimate_hz")
        ),
        "combined_frequencies_hz": {
            "V0": args.auto_roi_v0_frequency,
            "V1": args.auto_roi_v1_frequency,
        },
        "debug_files": debug_files,
        "overlap_ratio": round(overlap, 6) if "overlap" in locals() else None,
        "calibration_valid": calibration_valid,
        "failure_reason": None if calibration_valid else failure_reason,
    }
    _write_json(out_dir / "auto_rois.json", payload)
    _write_csv(out_dir / "api_events.csv", api_events)
    return {
        "mode": args.mode,
        "closed_loop_adaptation_enabled": False,
        "auto_roi_config": str(out_dir / "auto_rois.json"),
        "auto_rois": payload,
        "api_post_success_count": sum(1 for e in api_events if e.get("ok") == 1),
    }


def run_closed_loop_smoke(args: argparse.Namespace, out_dir: Path) -> dict:
    api_events: list[dict] = []
    _start_server_trial(args, api_events, feedback=True)
    if args.dry_run:
        _write_csv(out_dir / "api_events.csv", api_events)
        return {"mode": args.mode, "dry_run": True, "requires_hardware": True}

    _ensure_hw(False)
    assert _PiGPIOLED is not None
    detector = _make_multi_roi_detector(args)
    led = _PiGPIOLED(pin=args.led_pin, flash_duration_s=args.led_pulse_duration)
    pi_osc = EventBasedConsensusPLLOscillator(_locked_eapf_config(args.pi_freq))
    topology = build_mixed_reality_topology(args.topology)
    roi_rows: list[dict] = []
    pi_rows: list[dict] = []
    events: list[dict] = []
    try:
        detector.start()
        t0 = time.monotonic()
        last = t0
        while time.monotonic() - t0 < args.duration:
            now_abs = time.monotonic()
            t = now_abs - t0
            dt = max(0.001, now_abs - last)
            last = now_abs
            res = detector.capture_frame()
            neighbour_sources: list[str] = []
            for row in res["roi_results"]:
                out = dict(row)
                out["t_s"] = round(t, 6)
                roi_rows.append(out)
            for evt in res["events"]:
                src = evt["agent_id"]
                events.append({
                    "t_s": round(t, 6),
                    "event_type": "pi_detected_virtual_flash",
                    "agent_id": src,
                    "roi_id": evt["roi_id"],
                    "raw_brightness": evt["raw_brightness"],
                    "normalized_signal": evt["normalized_signal"],
                })
                if topology.can_observe("P0", src):
                    neighbour_sources.append(src)
            neighbour_ids = topology.numeric_neighbour_ids("P0", neighbour_sources)
            state = pi_osc.step(dt_s=dt, t_s=t, neighbour_flash_ids=neighbour_ids)
            pi_rows.append({
                "t_s": round(t, 6),
                "agent_id": "P0",
                "frequency_hz": state["frequency_hz"],
                "phase_rad": state["phase_rad"],
                "fire_count": state["fire_count"],
                "neighbour_events": len(neighbour_ids),
            })
            if state["follower_flash_event"]:
                led.flash(args.led_pulse_duration)
                _api(args.leader_api, "/api/pi_flash", "POST",
                     {"timestamp": t, "agent_id": "P0"},
                     timeout=args.api_timeout, events=api_events)
                events.append({"t_s": round(t, 6), "event_type": "pi_flash", "agent_id": "P0"})
    finally:
        try:
            led.close()
        finally:
            detector.stop()
            _api(args.leader_api, "/api/pause", "POST", {}, timeout=args.api_timeout, events=api_events)

    agent_rows, server_events = _poll_agents(args, out_dir, api_events, 0.1)
    events.extend(server_events)
    _write_csv(out_dir / "pi_detection_roi.csv", roi_rows)
    _write_csv(out_dir / "pi_oscillator.csv", pi_rows)
    _write_csv(out_dir / "events_all.csv", events)
    _write_csv(out_dir / "api_events.csv", api_events)
    counts: dict[str, int] = {}
    for event in events:
        aid = event.get("agent_id")
        counts[aid] = counts.get(aid, 0) + 1
    final_pi_freq = pi_rows[-1]["frequency_hz"] if pi_rows else None
    return {
        "mode": args.mode,
        "topology": args.topology,
        "flash_counts": counts,
        "pi_final_frequency_hz": final_pi_freq,
        "api_post_success_count": sum(1 for e in api_events if e.get("ok") == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="2V+1P EAPF HIL validation runner.")
    parser.add_argument("--mode", choices=[
        "frontend_multi_agent_smoke",
        "roi_calibration",
        "auto_roi_calibration",
        "multi_roi_detection_test",
        "2v1p_eapf_smoke",
    ], default="frontend_multi_agent_smoke")
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--topology", choices=[
        "all_to_all", "chain_pi_middle", "chain_pi_downstream",
    ], default="all_to_all")
    parser.add_argument("--v0-freq", "--v0-frequency", dest="v0_freq", type=float, default=1.9)
    parser.add_argument("--pi-freq", type=float, default=2.0)
    parser.add_argument("--v1-freq", "--v1-frequency", dest="v1_freq", type=float, default=2.1)
    parser.add_argument("--v0-phase-rad", type=float, default=0.0)
    parser.add_argument("--v1-phase-rad", type=float, default=0.0)
    parser.add_argument("--detection-preset", choices=[
        "none",
        "v0_only",
        "v1_only",
        "two_freq_1hz_2hz",
        "same_freq_phase_offset",
        "same_freq_near_simultaneous",
    ], default="none")
    parser.add_argument("--v0-x", type=int, default=520)
    parser.add_argument("--v0-y", type=int, default=420)
    parser.add_argument("--v1-x", type=int, default=1180)
    parser.add_argument("--v1-y", type=int, default=420)
    parser.add_argument("--dot-size", type=int, default=280)
    parser.add_argument("--v0-size", type=int, default=None)
    parser.add_argument("--v1-size", type=int, default=None)
    parser.add_argument("--background-brightness", type=int, default=0)
    parser.add_argument("--flash-brightness", type=int, default=255)
    parser.add_argument("--off-brightness", type=int, default=15)
    parser.add_argument("--log-dir", default="experiments/logs/2v1p_eapf_hil")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--api-timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--roi-v0", type=_parse_roi, default=None,
                        help="Manual V0 camera ROI as x,y,w,h")
    parser.add_argument("--roi-v1", type=_parse_roi, default=None,
                        help="Manual V1 camera ROI as x,y,w,h")
    parser.add_argument("--roi-config", default=None,
                        help="Path to auto_rois.json from auto_roi_calibration")
    parser.add_argument("--auto-roi", action="store_true",
                        help="Run auto ROI calibration before this mode and reuse detected ROIs")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-format", default="BGR888")
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--norm-on-threshold", type=float, default=0.65)
    parser.add_argument("--norm-off-threshold", type=float, default=0.35)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    parser.add_argument("--episode-latch", action="store_true")
    parser.add_argument("--no-mid-roi-debug-frame", dest="save_mid_roi_debug_frame",
                        action="store_false", default=True)
    parser.add_argument("--auto-roi-duration", type=float, default=3.0)
    parser.add_argument("--auto-roi-combined-duration", type=float, default=5.0)
    parser.add_argument("--auto-roi-verify-duration", type=float, default=1.0)
    parser.add_argument("--auto-roi-warmup-s", type=float, default=0.5)
    parser.add_argument("--auto-roi-capture-fps", type=float, default=15.0)
    parser.add_argument("--auto-roi-v0-frequency", type=float, default=1.0)
    parser.add_argument("--auto-roi-v1-frequency", type=float, default=2.0)
    parser.add_argument("--auto-roi-v1-phase-rad", type=float, default=1.57079632679)
    parser.add_argument("--auto-roi-sequential-diagnostics", action="store_true",
                        help="Also save optional V0-only/V1-only diagnostic captures")
    parser.add_argument("--auto-roi-frequency-ambiguity-hz", type=float, default=0.25)
    parser.add_argument("--auto-roi-method", choices=[
        "temporal_variance", "max_min_range", "mean_abs_diff",
    ], default="temporal_variance")
    parser.add_argument("--auto-roi-padding", type=int, default=35)
    parser.add_argument("--auto-roi-min-area", type=int, default=50)
    parser.add_argument("--auto-roi-downsample", type=int, default=1)
    parser.add_argument("--auto-roi-change-threshold", type=float, default=None)
    parser.add_argument("--auto-roi-boundary-margin-px", type=int, default=5)
    parser.add_argument("--auto-roi-overlap-warning-ratio", type=float, default=0.05)
    parser.add_argument("--auto-roi-max-area-fraction", type=float, default=0.35)
    parser.add_argument("--led-pin", type=int, default=17)
    parser.add_argument("--led-pulse-duration", type=float, default=0.06)
    args = parser.parse_args()
    _apply_detection_preset(args)
    _apply_roi_config(args)

    out_dir = _make_out_dir(args)

    auto_metrics: dict[str, Any] | None = None
    if args.mode == "auto_roi_calibration":
        metrics = run_auto_roi_calibration(args, out_dir)
        _write_json(out_dir / "trial_config.json", {
            "mode": args.mode,
            "leader_api": args.leader_api,
            "duration_s": args.duration,
            "topology": args.topology,
            "initial_frequencies_hz": {"V0": args.v0_freq, "P0": args.pi_freq, "V1": args.v1_freq},
            "auto_roi_parameters": {
                "duration_s": args.auto_roi_duration,
                "combined_duration_s": args.auto_roi_combined_duration,
                "verify_duration_s": args.auto_roi_verify_duration,
                "method": args.auto_roi_method,
                "padding_px": args.auto_roi_padding,
                "min_area_px": args.auto_roi_min_area,
                "downsample": args.auto_roi_downsample,
                "v0_frequency_hz": args.auto_roi_v0_frequency,
                "v1_frequency_hz": args.auto_roi_v1_frequency,
                "sequential_diagnostics": args.auto_roi_sequential_diagnostics,
            },
            "created_at": datetime.now().isoformat(),
        })
        _write_json(out_dir / "metrics_summary.json", metrics)
        print(f"Output: {out_dir}")
        print(json.dumps(metrics, indent=2))
        return

    if args.auto_roi:
        auto_metrics = run_auto_roi_calibration(args, out_dir)
        auto_rois = auto_metrics.get("auto_rois", {})
        if not auto_rois.get("calibration_valid", False):
            raise SystemExit(
                "Auto ROI calibration failed; inspect auto_rois.json and debug images "
                "before running detection or closed-loop modes."
            )
        args.roi_v0 = auto_rois.get("V0", {}).get("roi")
        args.roi_v1 = auto_rois.get("V1", {}).get("roi")

    if args.mode in ("roi_calibration", "multi_roi_detection_test", "2v1p_eapf_smoke"):
        if args.roi_v0 is None or args.roi_v1 is None:
            parser.error(
                "Camera modes require --roi-v0/--roi-v1, --roi-config, "
                "or --auto-roi. Run --mode auto_roi_calibration first if unsure."
            )

    _write_json(out_dir / "trial_config.json", {
        "mode": args.mode,
        "leader_api": args.leader_api,
        "duration_s": args.duration,
        "topology": args.topology,
        "initial_frequencies_hz": {"V0": args.v0_freq, "P0": args.pi_freq, "V1": args.v1_freq},
        "initial_phases_rad": {"V0": args.v0_phase_rad, "V1": args.v1_phase_rad},
        "virtual_enabled": {"V0": args.v0_enabled, "V1": args.v1_enabled},
        "virtual_display": {
            "positions": {"V0": [args.v0_x, args.v0_y], "V1": [args.v1_x, args.v1_y]},
            "sizes_px": {
                "V0": args.v0_size if args.v0_size is not None else args.dot_size,
                "V1": args.v1_size if args.v1_size is not None else args.dot_size,
            },
            "background_brightness": args.background_brightness,
            "flash_brightness": args.flash_brightness,
            "off_brightness": args.off_brightness,
        },
        "detection_preset": args.detection_preset,
        "model": "eapf_consensus",
        "model_parameters": {
            "g_p": 0.02,
            "g_f": 0.02,
            "alpha_p": 0.2,
            "alpha_f": 0.2,
            "delta_theta_max_rad": 0.2,
            "delta_f_max_hz": 0.05,
        },
        "roi_v0": args.roi_v0,
        "roi_v1": args.roi_v1,
        "roi_config": args.roi_config,
        "auto_roi_inline": args.auto_roi,
        "created_at": datetime.now().isoformat(),
    })

    if args.mode == "frontend_multi_agent_smoke":
        metrics = run_frontend_smoke(args, out_dir)
    elif args.mode in ("roi_calibration", "multi_roi_detection_test"):
        metrics = run_detection_test(args, out_dir)
    else:
        metrics = run_closed_loop_smoke(args, out_dir)
    if auto_metrics is not None:
        metrics["auto_roi_preflight"] = auto_metrics
    _write_json(out_dir / "metrics_summary.json", metrics)
    print(f"Output: {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

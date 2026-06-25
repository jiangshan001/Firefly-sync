#!/usr/bin/env python3
"""Formal Step5c1 two-flash multi-ROI detection robustness batch.

This runner is detection-only. It reuses the 2V+1P auto-ROI and
multi-ROI detection primitives, but never enables closed-loop EAPF
adaptation or Pi LED feedback.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.run_2v1p_eapf_hil import (
    _api,
    _write_csv,
    _write_json,
    run_auto_roi_calibration,
    run_detection_test,
)


DEFAULT_LOG_DIR = "experiments/logs/step5c1_multi_roi_detection"


def _condition_definitions() -> list[dict[str, Any]]:
    base = {
        "v0_x": 520,
        "v0_y": 420,
        "v1_x": 1180,
        "v1_y": 420,
        "v0_size": 280,
        "v1_size": 280,
        "background_brightness": 0,
        "flash_brightness": 255,
        "off_brightness": 0,
        "v0_freq": 1.0,
        "v1_freq": 2.0,
        "v0_phase_rad": 0.0,
        "v1_phase_rad": math.pi / 2.0,
    }

    def c(
        name: str,
        category: str,
        description: str,
        *,
        required_for_ready: bool = True,
        intended_difficulty: str = "realistic",
        failure_interpretation: str = "Failure suggests the detection setup needs adjustment before 2V+1P smoke.",
        **updates: Any,
    ) -> dict[str, Any]:
        item = deepcopy(base)
        item.update({
            "name": name,
            "group": category,
            "category": category,
            "description": description,
            "required_for_ready": required_for_ready,
            "intended_difficulty": intended_difficulty,
            "failure_interpretation": failure_interpretation,
        })
        item.update(updates)
        return item

    return [
        c("baseline", "baseline",
          "Default positions, large dots, high contrast, V0=1 Hz, V1=2 Hz.",
          failure_interpretation="Baseline failure means the camera/display alignment is not ready."),
        c("position_shifted", "position",
          "Moderately shifted target positions.",
          failure_interpretation="Realistic shifted-position failure suggests ROI calibration is position-sensitive.",
          v0_x=440, v0_y=360, v1_x=1260, v1_y=470),
        c("position_too_close_stress", "position",
          "Stress limit: targets close enough that flash halos or padded ROIs may overlap.",
          required_for_ready=False,
          intended_difficulty="stress_limit",
          failure_interpretation=(
              "Failure is expected if halos or padded ROIs overlap; this is not representative "
              "of the intended 2V+1P layout."
          ),
          v0_x=680, v0_y=420, v1_x=1020, v1_y=420, v0_size=240, v1_size=240),
        c("size_large_small", "size",
          "V0 large and V1 small.",
          failure_interpretation="Failure suggests asymmetric target size affects one ROI detector.",
          v0_size=280, v1_size=100),
        c("size_small_small", "size",
          "Both targets small.",
          failure_interpretation="Failure suggests the chosen small formal target size is below reliable detectability.",
          v0_size=100, v1_size=100),
        c("tiny_small_small_stress", "size",
          "Stress limit: both targets are tiny.",
          required_for_ready=False,
          intended_difficulty="stress_limit",
          failure_interpretation="Failure indicates the lower size limit, not formal readiness.",
          v0_size=70, v1_size=70),
        c("contrast_medium", "contrast",
          "Medium contrast flashes.",
          failure_interpretation="Failure suggests moderate contrast is not robust enough under current lighting.",
          background_brightness=40, off_brightness=40, flash_brightness=180),
        c("contrast_low", "contrast",
          "Low contrast flashes.",
          intended_difficulty="challenging",
          failure_interpretation="Low-contrast failure means this lighting/display contrast should be avoided formally.",
          background_brightness=80, off_brightness=80, flash_brightness=120),
        c("small_low_contrast_stress", "stress",
          "Stress limit: both targets small and low contrast.",
          required_for_ready=False,
          intended_difficulty="stress_limit",
          failure_interpretation="Difficult small-target low-contrast condition; failure does not block readiness.",
          v0_size=100, v1_size=100,
          background_brightness=80, off_brightness=80, flash_brightness=120,
          v0_freq=1.0, v1_freq=2.0),
        c("frequency_1p5_2p5", "frequency",
          "Separated frequencies V0=1.5 Hz, V1=2.5 Hz.",
          failure_interpretation="Failure suggests the detector struggles outside the 1/2 Hz calibration pair.",
          v0_freq=1.5, v1_freq=2.5),
        c("frequency_2hz_phase_offset", "frequency",
          "Both targets 2 Hz with half-cycle phase offset.",
          failure_interpretation="Failure suggests same-frequency separated flashes are not independently counted.",
          v0_freq=2.0, v1_freq=2.0, v0_phase_rad=0.0, v1_phase_rad=math.pi),
        c("frequency_2hz_near_simultaneous", "frequency",
          "Both targets 2 Hz with near-simultaneous flashes.",
          intended_difficulty="challenging",
          failure_interpretation="Failure suggests near-simultaneous independent detections need caution.",
          v0_freq=2.0, v1_freq=2.0, v0_phase_rad=0.0, v1_phase_rad=0.08),
    ]


def _make_runner_args(args: argparse.Namespace, condition: dict[str, Any],
                      run_dir: Path, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        mode=mode,
        leader_api=args.leader_api,
        duration=args.duration,
        topology="all_to_all",
        v0_freq=condition["v0_freq"],
        pi_freq=2.0,
        v1_freq=condition["v1_freq"],
        v0_phase_rad=condition["v0_phase_rad"],
        v1_phase_rad=condition["v1_phase_rad"],
        detection_preset="none",
        v0_x=condition["v0_x"],
        v0_y=condition["v0_y"],
        v1_x=condition["v1_x"],
        v1_y=condition["v1_y"],
        dot_size=condition["v0_size"],
        v0_size=condition["v0_size"],
        v1_size=condition["v1_size"],
        background_brightness=condition["background_brightness"],
        flash_brightness=condition["flash_brightness"],
        off_brightness=condition["off_brightness"],
        log_dir=args.log_dir,
        run_dir=str(run_dir),
        api_timeout=args.api_timeout,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        roi_v0=None,
        roi_v1=None,
        roi_config=None,
        auto_roi=False,
        width=args.width,
        height=args.height,
        camera_fps=args.camera_fps,
        camera_format=args.camera_format,
        min_interval=args.min_interval,
        window_s=args.window_s,
        norm_on_threshold=args.norm_on_threshold,
        norm_off_threshold=args.norm_off_threshold,
        min_amplitude=args.min_amplitude,
        episode_latch=args.episode_latch,
        save_mid_roi_debug_frame=True,
        auto_roi_duration=args.auto_roi_duration,
        auto_roi_combined_duration=args.auto_roi_combined_duration,
        auto_roi_verify_duration=1.0,
        auto_roi_warmup_s=args.auto_roi_warmup_s,
        auto_roi_capture_fps=args.auto_roi_capture_fps,
        auto_roi_v0_frequency=condition["v0_freq"],
        auto_roi_v1_frequency=condition["v1_freq"],
        auto_roi_v1_phase_rad=condition["v1_phase_rad"],
        auto_roi_sequential_diagnostics=args.auto_roi_sequential_diagnostics,
        auto_roi_frequency_ambiguity_hz=args.auto_roi_frequency_ambiguity_hz,
        auto_roi_method=args.auto_roi_method,
        auto_roi_padding=args.auto_roi_padding,
        auto_roi_min_area=args.auto_roi_min_area,
        auto_roi_downsample=args.auto_roi_downsample,
        auto_roi_change_threshold=args.auto_roi_change_threshold,
        auto_roi_boundary_margin_px=args.auto_roi_boundary_margin_px,
        auto_roi_overlap_warning_ratio=args.auto_roi_overlap_warning_ratio,
        auto_roi_max_area_fraction=args.auto_roi_max_area_fraction,
        led_pin=17,
        led_pulse_duration=0.06,
        v0_enabled=True,
        v1_enabled=True,
    )


def _gray_rgb(value: int) -> list[int]:
    v = max(0, min(255, int(value)))
    return [v, v, v]


def _condition_display_config(condition: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": condition["name"],
        "category": condition["category"],
        "description": condition["description"],
        "required_for_ready": condition["required_for_ready"],
        "intended_difficulty": condition["intended_difficulty"],
        "failure_interpretation": condition["failure_interpretation"],
        "display_model": "browser_canvas_grayscale_radial_targets",
        "background": {
            "brightness": condition["background_brightness"],
            "rgb": _gray_rgb(condition["background_brightness"]),
            "css_rgb": f"rgb({condition['background_brightness']},"
                       f"{condition['background_brightness']},"
                       f"{condition['background_brightness']})",
        },
        "flash_on": {
            "brightness": condition["flash_brightness"],
            "rgb": _gray_rgb(condition["flash_brightness"]),
            "css_rgb": f"rgb({condition['flash_brightness']},"
                       f"{condition['flash_brightness']},"
                       f"{condition['flash_brightness']})",
        },
        "flash_off": {
            "brightness": condition["off_brightness"],
            "rgb": _gray_rgb(condition["off_brightness"]),
            "css_rgb": f"rgb({condition['off_brightness']},"
                       f"{condition['off_brightness']},"
                       f"{condition['off_brightness']})",
        },
        "contrast": {
            "flash_minus_background": (
                condition["flash_brightness"] - condition["background_brightness"]
            ),
            "flash_minus_off": condition["flash_brightness"] - condition["off_brightness"],
            "off_minus_background": condition["off_brightness"] - condition["background_brightness"],
        },
        "agents": {
            "V0": {
                "position_xy": [condition["v0_x"], condition["v0_y"]],
                "dot_size_px": condition["v0_size"],
                "frequency_hz": condition["v0_freq"],
                "phase_rad": condition["v0_phase_rad"],
            },
            "V1": {
                "position_xy": [condition["v1_x"], condition["v1_y"]],
                "dot_size_px": condition["v1_size"],
                "frequency_hz": condition["v1_freq"],
                "phase_rad": condition["v1_phase_rad"],
            },
        },
    }


def _fetch_display_verification(args: argparse.Namespace,
                                condition: dict[str, Any]) -> dict[str, Any]:
    verification = {
        "requested_display_config": _condition_display_config(condition),
        "server_applied_mutual_config": None,
        "frontend_applied_display_state": None,
        "warnings": [],
    }
    try:
        verification["server_applied_mutual_config"] = _api(
            args.leader_api,
            "/api/mutual/config",
            "GET",
            timeout=args.api_timeout,
        )
    except Exception as exc:
        verification["warnings"].append(f"server_display_config_unavailable:{type(exc).__name__}")
    try:
        frontend_state = _api(
            args.leader_api,
            "/api/frontend/display_state",
            "GET",
            timeout=args.api_timeout,
        )
        verification["frontend_applied_display_state"] = frontend_state
        if not frontend_state:
            verification["warnings"].append("frontend_display_state_empty")
    except Exception as exc:
        verification["warnings"].append(f"frontend_display_state_unavailable:{type(exc).__name__}")
    return verification


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _api_summary(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    latencies = []
    ok_count = 0
    for row in rows:
        if str(row.get("ok", "")) == "1":
            ok_count += 1
        try:
            latencies.append(float(row.get("elapsed_ms", "")))
        except (TypeError, ValueError):
            pass
    return {
        "api_success_count": ok_count,
        "api_request_count": len(rows),
        "api_latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "api_latency_max_ms": round(max(latencies), 3) if latencies else None,
    }


def _capture_fps(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    frame_indices = set()
    times = []
    for row in rows:
        try:
            frame_indices.add(int(float(row.get("frame_index", ""))))
        except (TypeError, ValueError):
            pass
        try:
            times.append(float(row.get("t_s", "")))
        except (TypeError, ValueError):
            pass
    span = max(times) - min(times) if len(times) >= 2 else 0.0
    frame_count = len(frame_indices)
    return {
        "capture_frame_count": frame_count,
        "processing_fps": round(frame_count / span, 3) if span > 0 else None,
    }


def _agent_repeat_row(condition: dict[str, Any], repeat_index: int,
                      repeat_dir: Path, metrics: dict[str, Any],
                      auto_rois: dict[str, Any], agent_id: str,
                      thresholds: argparse.Namespace) -> dict[str, Any]:
    summary = metrics.get("roi_signal_summary", {}).get(agent_id, {})
    requested_freq = condition["v0_freq"] if agent_id == "V0" else condition["v1_freq"]
    detected_count = int(summary.get("detected_rising_edge_count") or 0)
    actual_count = int(summary.get("actual_virtual_flash_count") or 0)
    recall = summary.get("detection_fcr")
    freq_est = summary.get("estimated_detected_frequency_hz")
    freq_error = (
        abs(float(freq_est) - float(requested_freq))
        if isinstance(freq_est, (int, float)) else None
    )
    warnings = list(summary.get("warnings") or [])
    row = {
        "row_type": "repeat",
        "condition": condition["name"],
        "group": condition["group"],
        "category": condition["category"],
        "required_for_ready": condition["required_for_ready"],
        "intended_difficulty": condition["intended_difficulty"],
        "failure_interpretation": condition["failure_interpretation"],
        "background_brightness": condition["background_brightness"],
        "flash_brightness": condition["flash_brightness"],
        "off_brightness": condition["off_brightness"],
        "v0_position_xy": f"{condition['v0_x']},{condition['v0_y']}",
        "v1_position_xy": f"{condition['v1_x']},{condition['v1_y']}",
        "v0_size_px": condition["v0_size"],
        "v1_size_px": condition["v1_size"],
        "repeat": repeat_index,
        "agent_id": agent_id,
        "repeat_dir": str(repeat_dir),
        "calibration_valid": bool(auto_rois.get("calibration_valid", False)),
        "calibration_assignment_method": auto_rois.get("assignment_method"),
        "roi_overlap_ratio": auto_rois.get("overlap_ratio"),
        "requested_frequency_hz": requested_freq,
        "actual_virtual_flash_count": actual_count,
        "detected_rising_edge_count": detected_count,
        "count_recall": recall,
        "count_absolute_error": abs(detected_count - actual_count),
        "detected_frequency_estimate_hz": freq_est,
        "frequency_absolute_error_hz": round(freq_error, 6) if freq_error is not None else None,
        "raw_signal_min": (summary.get("raw_signal") or {}).get("min"),
        "raw_signal_max": (summary.get("raw_signal") or {}).get("max"),
        "raw_signal_mean": (summary.get("raw_signal") or {}).get("mean"),
        "raw_signal_std": (summary.get("raw_signal") or {}).get("std"),
        "raw_signal_range": summary.get("raw_signal_range"),
        "normalized_signal_min": (summary.get("normalized_signal") or {}).get("min"),
        "normalized_signal_max": (summary.get("normalized_signal") or {}).get("max"),
        "normalized_signal_mean": (summary.get("normalized_signal") or {}).get("mean"),
        "normalized_signal_std": (summary.get("normalized_signal") or {}).get("std"),
        "warnings": ";".join(warnings),
        "pass_repeat_agent": (
            bool(auto_rois.get("calibration_valid", False))
            and float(auto_rois.get("overlap_ratio") or 0.0) < thresholds.pass_overlap_ratio
            and recall is not None
            and float(recall) >= thresholds.pass_count_recall
            and freq_error is not None
            and freq_error <= thresholds.pass_frequency_error_hz
            and "no_rising_edges_detected" not in warnings
        ),
    }
    row.update(_api_summary(repeat_dir / "api_events.csv"))
    row.update(_capture_fps(repeat_dir / "pi_detection_roi.csv"))
    return row


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None, None
    if len(vals) == 1:
        return round(vals[0], 6), 0.0
    return round(statistics.mean(vals), 6), round(statistics.stdev(vals), 6)


def _condition_summary(condition: dict[str, Any], rows: list[dict[str, Any]],
                       auto_rois: dict[str, Any], thresholds: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "condition": condition["name"],
        "group": condition["group"],
        "category": condition["category"],
        "description": condition["description"],
        "required_for_ready": condition["required_for_ready"],
        "intended_difficulty": condition["intended_difficulty"],
        "failure_interpretation": condition["failure_interpretation"],
        "display_config": _condition_display_config(condition),
        "display_verification": auto_rois.get("display_verification"),
        "camera_display_evidence": auto_rois.get("camera_display_evidence"),
        "camera_mean_measured_contrast": (
            (auto_rois.get("camera_display_evidence") or {}).get("mean_measured_contrast")
        ),
        "background_brightness": condition["background_brightness"],
        "flash_brightness": condition["flash_brightness"],
        "off_brightness": condition["off_brightness"],
        "v0_size_px": condition["v0_size"],
        "v1_size_px": condition["v1_size"],
        "auto_roi_success": bool(auto_rois.get("calibration_valid", False)),
        "assignment_method": auto_rois.get("assignment_method"),
        "overlap_ratio": auto_rois.get("overlap_ratio"),
        "repeats_completed": len({row["repeat"] for row in rows}),
        "warnings_count": sum(1 for row in rows if row.get("warnings")),
        "no_rising_edge_repeats": sum(
            1 for row in rows if "no_rising_edges_detected" in str(row.get("warnings", ""))
        ),
    }
    for agent_id in ("V0", "V1"):
        agent_rows = [row for row in rows if row.get("agent_id") == agent_id]
        recall_mean, recall_std = _mean_std([row.get("count_recall") for row in agent_rows])
        freq_mean, freq_std = _mean_std([row.get("frequency_absolute_error_hz") for row in agent_rows])
        summary[f"{agent_id}_count_recall_mean"] = recall_mean
        summary[f"{agent_id}_count_recall_std"] = recall_std
        summary[f"{agent_id}_frequency_error_mean_hz"] = freq_mean
        summary[f"{agent_id}_frequency_error_std_hz"] = freq_std
    summary["pass_condition"] = (
        summary["auto_roi_success"]
        and float(summary.get("overlap_ratio") or 0.0) < thresholds.pass_overlap_ratio
        and all(row.get("pass_repeat_agent") for row in rows)
        and len(rows) > 0
    )
    return summary


def _contrast_order_warnings(condition_summaries: list[dict[str, Any]]) -> list[str]:
    by_name = {item.get("condition"): item for item in condition_summaries}
    needed = ["baseline", "contrast_medium", "contrast_low"]
    if not all(name in by_name for name in needed):
        return []
    contrasts: dict[str, float] = {}
    for name in needed:
        value = by_name[name].get("camera_mean_measured_contrast")
        if value is None:
            return ["contrast_order_not_checked_missing_camera_evidence"]
        try:
            contrasts[name] = float(value)
        except (TypeError, ValueError):
            return ["contrast_order_not_checked_invalid_camera_evidence"]
    if not (contrasts["baseline"] > contrasts["contrast_medium"] > contrasts["contrast_low"]):
        return [
            "measured_contrast_order_unexpected:"
            f"baseline={contrasts['baseline']:.3f},"
            f"medium={contrasts['contrast_medium']:.3f},"
            f"low={contrasts['contrast_low']:.3f}"
        ]
    return []


def _write_trial_config(path: Path, args: argparse.Namespace,
                        condition: dict[str, Any], mode: str,
                        roi_config: str | None = None) -> None:
    _write_json(path / "trial_config.json", {
        "mode": mode,
        "step": "step5c1_multi_roi_detection",
        "leader_api": args.leader_api,
        "duration_s": args.duration,
        "condition": condition,
        "display_config": _condition_display_config(condition),
        "roi_config": roi_config,
        "closed_loop_adaptation_enabled": False,
        "created_at": datetime.now().isoformat(),
    })


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    conditions = _condition_definitions()
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [condition for condition in conditions if condition["name"] in wanted]
        missing = sorted(wanted - {condition["name"] for condition in conditions})
        if missing:
            raise SystemExit(f"Unknown condition(s): {', '.join(missing)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = args.batch_name or "formal_detection_batch"
    batch_dir = Path(args.log_dir) / f"{stamp}_{batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "conditions").mkdir(exist_ok=True)

    batch_config = {
        "step": "step5c1_multi_roi_detection",
        "created_at": datetime.now().isoformat(),
        "leader_api": args.leader_api,
        "repeats": args.repeats,
        "duration_s": args.duration,
        "log_dir": args.log_dir,
        "batch_name": batch_name,
        "closed_loop_adaptation_enabled": False,
        "pass_thresholds": {
            "overlap_ratio": args.pass_overlap_ratio,
            "count_recall": args.pass_count_recall,
            "frequency_error_hz": args.pass_frequency_error_hz,
        },
        "conditions": conditions,
    }
    _write_json(batch_dir / "batch_config.json", batch_config)

    repeat_rows: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    auto_valid_count = 0
    auto_failed_count = 0

    for condition in conditions:
        condition_dir = batch_dir / "conditions" / condition["name"]
        condition_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            condition_dir / "condition_display_config.json",
            _condition_display_config(condition),
        )
        auto_dir = condition_dir / "auto_roi"
        auto_dir.mkdir(parents=True, exist_ok=False)
        auto_args = _make_runner_args(args, condition, auto_dir, "auto_roi_calibration")
        _write_trial_config(auto_dir, args, condition, "auto_roi_calibration")
        auto_metrics = run_auto_roi_calibration(auto_args, auto_dir)
        auto_rois = auto_metrics.get("auto_rois", {})
        display_verification = _fetch_display_verification(args, condition)
        auto_rois["display_verification"] = display_verification
        display_config_with_evidence = _condition_display_config(condition)
        display_config_with_evidence["display_verification"] = display_verification
        display_config_with_evidence["camera_display_evidence"] = auto_rois.get(
            "camera_display_evidence"
        )
        _write_json(condition_dir / "condition_display_config.json", display_config_with_evidence)
        _write_json(auto_dir / "auto_rois.json", auto_rois)
        _write_json(auto_dir / "metrics_summary.json", auto_metrics)

        if auto_rois.get("calibration_valid", False):
            auto_valid_count += 1
        else:
            auto_failed_count += 1
            fail_row = {
                "row_type": "calibration_failed",
                "condition": condition["name"],
                "group": condition["group"],
                "category": condition["category"],
                "required_for_ready": condition["required_for_ready"],
                "intended_difficulty": condition["intended_difficulty"],
                "failure_interpretation": condition["failure_interpretation"],
                "repeat": "",
                "agent_id": "",
                "repeat_dir": "",
                "calibration_valid": False,
                "calibration_assignment_method": auto_rois.get("assignment_method"),
                "roi_overlap_ratio": auto_rois.get("overlap_ratio"),
                "warnings": auto_rois.get("failure_reason", "auto_roi_calibration_failed"),
                "pass_repeat_agent": False,
            }
            repeat_rows.append(fail_row)
            condition_summaries.append(_condition_summary(condition, [], auto_rois, args))
            continue

        roi_config = str(auto_dir / "auto_rois.json")
        condition_repeat_rows: list[dict[str, Any]] = []
        for repeat in range(1, args.repeats + 1):
            repeat_dir = condition_dir / f"repeat_{repeat:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=False)
            det_args = _make_runner_args(args, condition, repeat_dir, "multi_roi_detection_test")
            det_args.roi_v0 = auto_rois.get("V0", {}).get("roi")
            det_args.roi_v1 = auto_rois.get("V1", {}).get("roi")
            det_args.roi_config = roi_config
            _write_trial_config(repeat_dir, args, condition, "multi_roi_detection_test", roi_config)
            metrics = run_detection_test(det_args, repeat_dir)
            _write_json(repeat_dir / "metrics_summary.json", metrics)
            for agent_id in ("V0", "V1"):
                row = _agent_repeat_row(
                    condition, repeat, repeat_dir, metrics, auto_rois, agent_id, args
                )
                repeat_rows.append(row)
                condition_repeat_rows.append(row)

        condition_summaries.append(
            _condition_summary(condition, condition_repeat_rows, auto_rois, args)
        )

    _write_csv(batch_dir / "batch_summary.csv", repeat_rows)
    required_summaries = [
        item for item in condition_summaries if item.get("required_for_ready", True)
    ]
    optional_summaries = [
        item for item in condition_summaries if not item.get("required_for_ready", True)
    ]
    pass_conditions = sum(1 for item in condition_summaries if item.get("pass_condition"))
    required_pass_conditions = sum(1 for item in required_summaries if item.get("pass_condition"))
    optional_pass_conditions = sum(1 for item in optional_summaries if item.get("pass_condition"))
    worst = _worst_condition(condition_summaries)
    worst_required = _worst_condition(required_summaries)
    required_failed = [
        item["condition"] for item in required_summaries if not item.get("pass_condition")
    ]
    optional_failed = [
        item["condition"] for item in optional_summaries if not item.get("pass_condition")
    ]
    contrast_order_warnings = _contrast_order_warnings(condition_summaries)
    batch_summary = {
        "batch_dir": str(batch_dir),
        "conditions_requested": len(conditions),
        "conditions_completed": len(condition_summaries),
        "required_conditions_count": len(required_summaries),
        "optional_stress_conditions_count": len(optional_summaries),
        "auto_roi_valid_count": auto_valid_count,
        "auto_roi_failed_count": auto_failed_count,
        "pass_conditions": pass_conditions,
        "failed_conditions": len(condition_summaries) - pass_conditions,
        "required_pass_conditions": required_pass_conditions,
        "required_failed_conditions": required_failed,
        "optional_stress_pass_conditions": optional_pass_conditions,
        "optional_stress_failed_conditions": optional_failed,
        "contrast_order_warnings": contrast_order_warnings,
        "condition_summaries": condition_summaries,
        "worst_condition": worst,
        "worst_required_condition": worst_required,
        "ready_for_2v1p_eapf_smoke": (
            len(required_summaries) > 0
            and required_pass_conditions == len(required_summaries)
        ),
    }
    _write_json(batch_dir / "batch_summary.json", batch_summary)
    return batch_summary


def _worst_condition(condition_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not condition_summaries:
        return None

    def score(item: dict[str, Any]) -> tuple[float, float]:
        recalls = [
            item.get("V0_count_recall_mean"),
            item.get("V1_count_recall_mean"),
        ]
        freq_errors = [
            item.get("V0_frequency_error_mean_hz"),
            item.get("V1_frequency_error_mean_hz"),
        ]
        finite_recalls = [float(v) for v in recalls if v is not None]
        finite_freq = [float(v) for v in freq_errors if v is not None]
        min_recall = min(finite_recalls) if finite_recalls else -1.0
        max_freq = max(finite_freq) if finite_freq else math.inf
        return (min_recall, -max_freq)

    return min(condition_summaries, key=score)


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Batch output: {summary['batch_dir']}")
    print(f"Conditions completed: {summary['conditions_completed']}")
    print(
        "Auto ROI calibrations: "
        f"{summary['auto_roi_valid_count']} valid, {summary['auto_roi_failed_count']} failed"
    )
    print(
        "Readiness gate: "
        f"{summary['required_pass_conditions']}/{summary['required_conditions_count']} "
        "required conditions passed"
    )
    print("Mean count recall / frequency error by condition:")
    for item in summary["condition_summaries"]:
        tag = "required" if item.get("required_for_ready", True) else "stress"
        print(
            f"  {item['condition']} [{tag}]: "
            f"V0 recall={item.get('V0_count_recall_mean')} "
            f"V1 recall={item.get('V1_count_recall_mean')} "
            f"V0 ferr={item.get('V0_frequency_error_mean_hz')} "
            f"V1 ferr={item.get('V1_frequency_error_mean_hz')} "
            f"pass={item.get('pass_condition')}"
        )
    worst = summary.get("worst_condition")
    if worst:
        print(f"Worst-performing condition: {worst.get('condition')}")
    worst_required = summary.get("worst_required_condition")
    if worst_required:
        print(f"Worst required condition: {worst_required.get('condition')}")
    if summary.get("contrast_order_warnings"):
        print("Contrast order warnings:")
        for warning in summary["contrast_order_warnings"]:
            print(f"  {warning}")
    ready = "YES" if summary.get("ready_for_2v1p_eapf_smoke") else "NO"
    print(f"Ready for 2V+1P EAPF smoke: {ready}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--batch-name", default="formal_detection_batch")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Optional subset of condition names to run")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-format", default="BGR888")
    parser.add_argument("--api-timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)

    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--norm-on-threshold", type=float, default=0.65)
    parser.add_argument("--norm-off-threshold", type=float, default=0.35)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    parser.add_argument("--episode-latch", action="store_true")

    parser.add_argument("--auto-roi-combined-duration", type=float, default=6.0)
    parser.add_argument("--auto-roi-duration", type=float, default=3.0)
    parser.add_argument("--auto-roi-warmup-s", type=float, default=0.5)
    parser.add_argument("--auto-roi-capture-fps", type=float, default=15.0)
    parser.add_argument("--auto-roi-sequential-diagnostics", action="store_true")
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

    parser.add_argument("--pass-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--pass-count-recall", type=float, default=0.85)
    parser.add_argument("--pass-frequency-error-hz", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_batch(args)
    _print_summary(summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate the Step5c1 two-flash multi-ROI detection report.

The script reads existing experiment logs only. It writes derived CSV/JSON
tables, figures, curated appendix images, and LaTeX source into this report
folder without modifying raw experiment data.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
FIG_DIR = REPORT_DIR / "figures"
APPENDIX_DIR = FIG_DIR / "appendix"
TABLE_DIR = REPORT_DIR / "tables"
DERIVED_DIR = REPORT_DIR / "derived"

MAIN_BATCH = ROOT / "experiments/logs/step5c1_multi_roi_detection/20260624_163557_formal_detection_batch"
CONTRAST_RERUN = ROOT / "experiments/logs/step5c1_multi_roi_detection/20260625_152611_contrast_rerun_corrected"
CONTRAST_DEBUG = ROOT / "experiments/logs/step5c1_multi_roi_detection/20260624_173430_contrast_visual_debug"
SMALL_LOW_STRESS = ROOT / "experiments/logs/step5c1_multi_roi_detection/20260624_175949_small_low_contrast_stress"

REQUIRED_CONDITIONS = [
    "baseline",
    "position_shifted",
    "size_large_small",
    "size_small_small",
    "frequency_1p5_2p5",
    "frequency_2hz_phase_offset",
    "frequency_2hz_near_simultaneous",
    "contrast_medium",
    "contrast_low",
]

NON_CONTRAST_FROM_MAIN = [
    "baseline",
    "position_shifted",
    "size_large_small",
    "size_small_small",
    "frequency_1p5_2p5",
    "frequency_2hz_phase_offset",
    "frequency_2hz_near_simultaneous",
]

CONTRAST_FROM_RERUN = [
    "contrast_medium",
    "contrast_low",
]

REPEAT_REQUIRED_CONDITIONS = NON_CONTRAST_FROM_MAIN + CONTRAST_FROM_RERUN

STRESS_CONDITIONS = [
    "position_too_close_stress",
    "tiny_small_small_stress",
    "small_low_contrast_stress",
]

APPENDIX_CONDITIONS = [
    "baseline",
    "position_shifted",
    "size_small_small",
    "contrast_medium",
    "contrast_low",
    "frequency_2hz_near_simultaneous",
    "position_too_close_stress",
    "tiny_small_small_stress",
    "small_low_contrast_stress",
]

LABELS = {
    "baseline": "Baseline",
    "position_shifted": "Shifted position",
    "size_large_small": "Large-small size",
    "size_small_small": "Small-small size",
    "frequency_1p5_2p5": "1.5/2.5 Hz",
    "frequency_2hz_phase_offset": "2 Hz phase offset",
    "frequency_2hz_near_simultaneous": "2 Hz near-simultaneous",
    "contrast_medium": "Medium contrast",
    "contrast_low": "Low contrast",
    "position_too_close_stress": "Too-close stress",
    "tiny_small_small_stress": "Tiny-small stress",
    "small_low_contrast_stress": "Small low-contrast stress",
}

IMAGE_TITLES = {
    "auto_roi_combined_debug.jpg": "Selected ROIs",
    "auto_roi_combined_temporal_variance.jpg": "Temporal variance",
    "auto_roi_combined_threshold_mask.jpg": "Threshold mask",
    "auto_roi_selected_traces.png": "ROI brightness traces",
    "camera_background_raw.jpg": "Background frame",
    "camera_v0_flash_on_raw.jpg": "V0 flash-on frame",
    "camera_v1_flash_on_raw.jpg": "V1 flash-on frame",
}

DIAGNOSTIC_IMAGES = [
    "auto_roi_combined_debug.jpg",
    "auto_roi_combined_temporal_variance.jpg",
    "auto_roi_combined_threshold_mask.jpg",
    "auto_roi_selected_traces.png",
]

CAMERA_IMAGES = [
    "camera_background_raw.jpg",
    "camera_v0_flash_on_raw.jpg",
    "camera_v1_flash_on_raw.jpg",
]

GENERATION_WARNINGS: list[str] = []


def label(condition: str) -> str:
    return LABELS.get(condition, condition.replace("_", " ").title())


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def fnum(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def fmt(value: Any, digits: int = 3, missing: str = "--") -> str:
    num = fnum(value)
    if num is None:
        return missing
    return f"{num:.{digits}f}"


def tex_escape(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def main_summary_by_condition(path: Path) -> dict[str, dict[str, Any]]:
    batch = read_json(path / "batch_summary.json")
    return {item["condition"]: item for item in batch.get("condition_summaries", [])}


def repeat_rows_by_condition(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_csv(path / "batch_summary.csv")
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["condition"], []).append(row)
    return out


def condition_auto_dir(condition: str) -> Path:
    if condition in ("contrast_medium", "contrast_low"):
        return CONTRAST_RERUN / "conditions" / condition / "auto_roi"
    if condition == "baseline":
        # Baseline main metrics come from the formal batch; contrast evidence
        # uses the corrected debug folder only in the contrast figure.
        return MAIN_BATCH / "conditions" / condition / "auto_roi"
    if condition == "small_low_contrast_stress":
        return SMALL_LOW_STRESS / "conditions" / condition / "auto_roi"
    return MAIN_BATCH / "conditions" / condition / "auto_roi"


def contrast_auto_dir(condition: str) -> Path:
    return CONTRAST_RERUN / "conditions" / condition / "auto_roi"


def load_auto_rois(condition: str) -> dict[str, Any]:
    path = condition_auto_dir(condition) / "auto_rois.json"
    if path.exists():
        return read_json(path)
    return {}


def load_contrast_debug_auto_rois(condition: str) -> dict[str, Any]:
    path = contrast_auto_dir(condition) / "auto_rois.json"
    if path.exists():
        return read_json(path)
    return {}


def build_dataset_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path, purpose, notes in [
        (
            "20260624_163557_formal_detection_batch",
            MAIN_BATCH,
            "Main formal detection batch",
            "Used for baseline, position, size, frequency, and original stress conditions. Original contrast rows excluded from final contrast evidence.",
        ),
        (
            "20260625_152611_contrast_rerun_corrected",
            CONTRAST_RERUN,
            "Corrected contrast repeat batch",
            "Used for final medium- and low-contrast recall and frequency-error metrics.",
        ),
        (
            "20260624_173430_contrast_visual_debug",
            CONTRAST_DEBUG,
            "Previous contrast/display debug run",
            "Optional historical display evidence only; not used for final contrast recall or frequency metrics.",
        ),
        (
            "20260624_175949_small_low_contrast_stress",
            SMALL_LOW_STRESS,
            "Small + low contrast stress batch",
            "Optional stress condition; not part of readiness gate.",
        ),
    ]:
        conditions: list[str] = []
        repeats = "--"
        duration = "--"
        if (path / "batch_summary.json").exists():
            batch = read_json(path / "batch_summary.json")
            conditions = [c["condition"] for c in batch.get("condition_summaries", [])]
            cfg = read_json(path / "batch_config.json") if (path / "batch_config.json").exists() else {}
            repeats = str(cfg.get("repeats", "--"))
            duration = str(cfg.get("duration_s", "--"))
        else:
            conditions = [p.name.replace("_auto_roi", "") for p in sorted(path.glob("*_auto_roi"))]
            duration = "approx. 6 s auto-ROI capture"
            repeats = "0 detection repeats"
        rows.append({
            "folder": name,
            "purpose": purpose,
            "conditions": ", ".join(conditions),
            "repeats": repeats,
            "duration_s": duration,
            "notes": notes,
        })
    return rows


def inspect_corrected_contrast_repeats() -> dict[str, Any]:
    conditions = ["baseline", "contrast_medium", "contrast_low"]
    cfg = read_json(CONTRAST_RERUN / "batch_config.json") if (CONTRAST_RERUN / "batch_config.json").exists() else {}
    csv_rows = read_csv(CONTRAST_RERUN / "batch_summary.csv") if (CONTRAST_RERUN / "batch_summary.csv").exists() else []
    inspection: dict[str, Any] = {
        "folder": str(CONTRAST_RERUN.relative_to(ROOT)),
        "batch_summary_json_exists": (CONTRAST_RERUN / "batch_summary.json").exists(),
        "batch_summary_csv_exists": (CONTRAST_RERUN / "batch_summary.csv").exists(),
        "conditions_dir_exists": (CONTRAST_RERUN / "conditions").exists(),
        "configured_repeats": cfg.get("repeats"),
        "configured_duration_s": cfg.get("duration_s"),
        "conditions": {},
        "corrected_contrast_detection_repeats_exist": False,
        "recall_frequency_error_computable": False,
    }
    for condition in conditions:
        existing = CONTRAST_RERUN / "conditions" / condition
        exists = existing.exists()
        repeat_dirs = sorted(existing.glob("repeat_*")) if exists else []
        repeat_metrics = [p / "metrics_summary.json" for p in repeat_dirs]
        repeat_roi_logs = [p / "pi_detection_roi.csv" for p in repeat_dirs]
        condition_rows = [
            row for row in csv_rows
            if row.get("condition") == condition and row.get("row_type") == "repeat"
        ]
        per_agent: dict[str, dict[str, Any]] = {}
        for agent in ("V0", "V1"):
            agent_rows = [row for row in condition_rows if row.get("agent_id") == agent]
            recalls = [fnum(row.get("count_recall")) for row in agent_rows]
            freq_errors = [fnum(row.get("frequency_absolute_error_hz")) for row in agent_rows]
            per_agent[agent] = {
                "repeat_count": len(agent_rows),
                "count_recall_mean": mean(recalls),
                "frequency_absolute_error_mean_hz": mean(freq_errors),
                "pass_flags": [row.get("pass_repeat_agent") for row in agent_rows],
            }
        inspection["conditions"][condition] = {
            "path": str(existing.relative_to(ROOT)) if exists else None,
            "condition_folder_exists": exists,
            "auto_rois_json_exists": bool(exists and (existing / "auto_roi" / "auto_rois.json").exists()),
            "repeat_folder_count": len(repeat_dirs),
            "repeat_folders": [p.name for p in repeat_dirs],
            "per_repeat_metrics_summary_count": sum(p.exists() for p in repeat_metrics),
            "per_repeat_pi_detection_roi_count": sum(p.exists() for p in repeat_roi_logs),
            "per_agent": per_agent,
        }
    repeat_counts = [
        item["repeat_folder_count"]
        for item in inspection["conditions"].values()
    ]
    inspection["corrected_contrast_detection_repeats_exist"] = any(count > 0 for count in repeat_counts)
    inspection["recall_frequency_error_computable"] = all(
        item["repeat_folder_count"] == cfg.get("repeats", 3)
        and item["per_repeat_metrics_summary_count"] == item["repeat_folder_count"]
        and item["per_repeat_pi_detection_roi_count"] == item["repeat_folder_count"]
        for item in inspection["conditions"].values()
    )
    inspection["duration_s"] = cfg.get("duration_s")
    inspection["diagnosis"] = (
        "Corrected contrast repeat detection data are complete and usable."
        if inspection["recall_frequency_error_computable"]
        else "Corrected contrast repeat detection data are incomplete; do not regenerate the report."
    )
    return inspection


def ensure_corrected_contrast_complete(inspection: dict[str, Any]) -> None:
    if not inspection.get("batch_summary_json_exists"):
        raise SystemExit("Corrected contrast audit failed: missing batch_summary.json.")
    if not inspection.get("batch_summary_csv_exists"):
        raise SystemExit("Corrected contrast audit failed: missing batch_summary.csv.")
    if not inspection.get("conditions_dir_exists"):
        raise SystemExit("Corrected contrast audit failed: missing conditions/ directory.")
    missing: list[str] = []
    for condition in ("baseline", "contrast_medium", "contrast_low"):
        item = inspection["conditions"].get(condition, {})
        expected = int(inspection.get("configured_repeats") or 3)
        if not item.get("condition_folder_exists"):
            missing.append(f"{condition}: condition folder")
        if item.get("repeat_folder_count") != expected:
            missing.append(f"{condition}: expected {expected} repeat folders, found {item.get('repeat_folder_count')}")
        if item.get("per_repeat_metrics_summary_count") != expected:
            missing.append(f"{condition}: expected {expected} metrics_summary.json files, found {item.get('per_repeat_metrics_summary_count')}")
        if item.get("per_repeat_pi_detection_roi_count") != expected:
            missing.append(f"{condition}: expected {expected} pi_detection_roi.csv files, found {item.get('per_repeat_pi_detection_roi_count')}")
    if missing:
        raise SystemExit("Corrected contrast audit failed:\n" + "\n".join(f"- {m}" for m in missing))


def display_setting(condition: str, auto: dict[str, Any] | None = None) -> str:
    if auto:
        cfg = (
            auto.get("display_verification", {})
            .get("requested_display_config", {})
        )
        if not cfg:
            cfg = auto.get("combined_mutual_config_snapshot", {}).get("display_config", {})
        bg = cfg.get("background", {}).get("brightness")
        on = cfg.get("flash_on", {}).get("brightness")
        off = cfg.get("flash_off", {}).get("brightness")
        if bg is not None and on is not None:
            return f"bg {bg}, off {off}, on {on}"
    if condition.startswith("contrast"):
        return "corrected contrast debug"
    return "high contrast"


def condition_metadata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in (MAIN_BATCH, CONTRAST_RERUN, SMALL_LOW_STRESS):
        if (source / "batch_summary.json").exists():
            for item in read_json(source / "batch_summary.json").get("condition_summaries", []):
                out[item["condition"]] = item
    return out


def build_merged_condition_table() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_summary = main_summary_by_condition(MAIN_BATCH)
    contrast_summary = main_summary_by_condition(CONTRAST_RERUN)
    stress_summary = main_summary_by_condition(SMALL_LOW_STRESS)
    main_rows = repeat_rows_by_condition(MAIN_BATCH)
    contrast_rows = repeat_rows_by_condition(CONTRAST_RERUN)
    stress_rows = repeat_rows_by_condition(SMALL_LOW_STRESS)

    rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []

    def add_repeat_rows(condition: str, source_label: str, condition_rows: list[dict[str, str]]) -> None:
        for row in condition_rows:
            copied = dict(row)
            copied["report_source"] = source_label
            repeat_rows.append(copied)

    for condition in NON_CONTRAST_FROM_MAIN:
        item = main_summary[condition]
        add_repeat_rows(condition, "main_formal_batch", main_rows.get(condition, []))
        rows.append({
            "condition": condition,
            "category": item.get("category"),
            "required_for_ready": item.get("required_for_ready"),
            "evidence_source": "main formal batch",
            "evidence_type": "auto ROI + 3 detection repeats",
            "repeats_completed": item.get("repeats_completed"),
            "duration_s": 30,
            "auto_roi_success": item.get("auto_roi_success"),
            "overlap_ratio": item.get("overlap_ratio"),
            "V0_count_recall_mean": item.get("V0_count_recall_mean"),
            "V1_count_recall_mean": item.get("V1_count_recall_mean"),
            "V0_frequency_error_mean_hz": item.get("V0_frequency_error_mean_hz"),
            "V1_frequency_error_mean_hz": item.get("V1_frequency_error_mean_hz"),
            "V0_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in main_rows.get(condition, []) if r.get("agent_id") == "V0"]),
            "V1_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in main_rows.get(condition, []) if r.get("agent_id") == "V1"]),
            "pass_condition": item.get("pass_condition"),
            "notes": "",
        })

    for condition in CONTRAST_FROM_RERUN:
        item = contrast_summary[condition]
        add_repeat_rows(condition, "corrected_contrast_rerun", contrast_rows.get(condition, []))
        rows.append({
            "condition": condition,
            "category": item.get("category"),
            "required_for_ready": item.get("required_for_ready"),
            "evidence_source": "corrected contrast repeat batch",
            "evidence_type": "auto ROI + 3 detection repeats",
            "repeats_completed": item.get("repeats_completed"),
            "duration_s": 30,
            "auto_roi_success": item.get("auto_roi_success"),
            "overlap_ratio": item.get("overlap_ratio"),
            "V0_count_recall_mean": item.get("V0_count_recall_mean"),
            "V1_count_recall_mean": item.get("V1_count_recall_mean"),
            "V0_frequency_error_mean_hz": item.get("V0_frequency_error_mean_hz"),
            "V1_frequency_error_mean_hz": item.get("V1_frequency_error_mean_hz"),
            "V0_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in contrast_rows.get(condition, []) if r.get("agent_id") == "V0"]),
            "V1_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in contrast_rows.get(condition, []) if r.get("agent_id") == "V1"]),
            "pass_condition": item.get("pass_condition"),
            "notes": (
                "Corrected contrast repeat condition passed."
                if str(item.get("pass_condition")).lower() == "true"
                else "Corrected contrast repeat condition failed; inspect per-agent warnings and counts."
            ),
        })

    for condition in ("position_too_close_stress", "tiny_small_small_stress"):
        item = main_summary[condition]
        add_repeat_rows(condition, "main_formal_batch", main_rows.get(condition, []))
        if condition == "position_too_close_stress":
            notes = (
                "Intentionally difficult; failure indicates a boundary case when large halos "
                "nearly touch or padded ROIs overlap. Not representative of the intended 2V+1P layout."
            )
        elif str(item.get("pass_condition")).lower() == "true":
            notes = "Passed; this optional very-small-target condition shows additional size robustness."
        else:
            notes = "Failed; this optional very-small-target condition identifies a lower size boundary."
        rows.append({
            "condition": condition,
            "category": item.get("category"),
            "required_for_ready": False,
            "evidence_source": "main formal batch",
            "evidence_type": "optional stress",
            "repeats_completed": item.get("repeats_completed"),
            "duration_s": 30 if item.get("repeats_completed") else 0,
            "auto_roi_success": item.get("auto_roi_success"),
            "overlap_ratio": item.get("overlap_ratio"),
            "V0_count_recall_mean": item.get("V0_count_recall_mean"),
            "V1_count_recall_mean": item.get("V1_count_recall_mean"),
            "V0_frequency_error_mean_hz": item.get("V0_frequency_error_mean_hz"),
            "V1_frequency_error_mean_hz": item.get("V1_frequency_error_mean_hz"),
            "V0_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in main_rows.get(condition, []) if r.get("agent_id") == "V0"]),
            "V1_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in main_rows.get(condition, []) if r.get("agent_id") == "V1"]),
            "pass_condition": item.get("pass_condition"),
            "notes": notes,
        })

    condition = "small_low_contrast_stress"
    item = stress_summary[condition]
    add_repeat_rows(condition, "small_low_contrast_stress_batch", stress_rows.get(condition, []))
    rows.append({
        "condition": condition,
        "category": item.get("category"),
        "required_for_ready": False,
        "evidence_source": "small+low stress batch",
        "evidence_type": "optional stress",
        "repeats_completed": item.get("repeats_completed"),
        "duration_s": 30,
        "auto_roi_success": item.get("auto_roi_success"),
        "overlap_ratio": item.get("overlap_ratio"),
        "V0_count_recall_mean": item.get("V0_count_recall_mean"),
        "V1_count_recall_mean": item.get("V1_count_recall_mean"),
        "V0_frequency_error_mean_hz": item.get("V0_frequency_error_mean_hz"),
        "V1_frequency_error_mean_hz": item.get("V1_frequency_error_mean_hz"),
        "V0_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in stress_rows.get(condition, []) if r.get("agent_id") == "V0"]),
        "V1_signal_range_mean": mean([fnum(r.get("raw_signal_range")) for r in stress_rows.get(condition, []) if r.get("agent_id") == "V1"]),
        "pass_condition": item.get("pass_condition"),
        "notes": (
            "Optional combined small-target low-contrast stress. Failure does not block readiness, "
            "but marks a difficult visual boundary case."
            if str(item.get("pass_condition")).lower() != "true"
            else "Passed optional combined small-target low-contrast stress."
        ),
    })
    return rows, repeat_rows


def readiness_summary(merged: list[dict[str, Any]]) -> dict[str, Any]:
    required = [r for r in merged if r.get("required_for_ready") is True]
    repeat_required = required
    repeat_pass = [
        r for r in repeat_required
        if str(r.get("pass_condition")).lower() == "true"
    ]
    contrast_repeat = [
        r for r in required
        if r["condition"] in ("contrast_medium", "contrast_low")
    ]
    contrast_pass = [
        r for r in required
        if r["condition"] in ("contrast_medium", "contrast_low")
        and str(r.get("pass_condition")).lower() == "true"
    ]
    return {
        "required_conditions": [r["condition"] for r in required],
        "repeat_based_required_conditions": [r["condition"] for r in repeat_required],
        "repeat_based_required_pass_count": len(repeat_pass),
        "repeat_based_required_count": len(repeat_required),
        "corrected_contrast_repeat_pass_count": len(contrast_pass),
        "corrected_contrast_repeat_count": len(contrast_repeat),
        "failed_required_conditions": [
            r["condition"] for r in repeat_required
            if str(r.get("pass_condition")).lower() != "true"
        ],
        "ready_for_2v1p_eapf_smoke": len(repeat_pass) == len(repeat_required),
        "caveat": (
            "Readiness is evaluated on required conditions only. Optional stress "
            "conditions remain boundary probes and do not block the gate."
        ),
    }


def plot_recall(merged: list[dict[str, Any]]) -> None:
    rows = [
        r for r in merged
        if r["condition"] in REPEAT_REQUIRED_CONDITIONS
        and fnum(r.get("V0_count_recall_mean")) is not None
    ]
    labels = [label(r["condition"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    v0 = [fnum(r.get("V0_count_recall_mean")) for r in rows]
    v1 = [fnum(r.get("V1_count_recall_mean")) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, [np.nan if v is None else v for v in v0], width, label="V0")
    ax.bar(x + width / 2, [np.nan if v is None else v for v in v1], width, label="V1")
    ax.axhline(0.85, color="firebrick", linestyle="--", linewidth=1.0, label="pass threshold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Count recall")
    ax.set_title("Detection count recall by required condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.legend(ncol=3, loc="lower left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_detection_recall_by_condition.pdf")
    fig.savefig(FIG_DIR / "fig1_detection_recall_by_condition.png", dpi=200)
    plt.close(fig)


def plot_frequency_error(merged: list[dict[str, Any]]) -> None:
    rows = [
        r for r in merged
        if r["condition"] in REPEAT_REQUIRED_CONDITIONS
    ]
    labels = [label(r["condition"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    v0 = [fnum(r.get("V0_frequency_error_mean_hz")) for r in rows]
    v1 = [fnum(r.get("V1_frequency_error_mean_hz")) for r in rows]
    max_err = max([v for v in v0 + v1 if v is not None] + [0.002])
    ylim_top = max(0.003, max_err * 1.35)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - width / 2, [np.nan if v is None else v for v in v0], width, label="V0")
    ax.bar(x + width / 2, [np.nan if v is None else v for v in v1], width, label="V1")
    for idx, (a, b) in enumerate(zip(v0, v1)):
        if a is None:
            ax.text(idx - width / 2, ylim_top * 0.68, "no\nestimate", ha="center",
                    va="center", fontsize=7, color="firebrick")
        if b is None:
            ax.text(idx + width / 2, ylim_top * 0.68, "no\nestimate", ha="center",
                    va="center", fontsize=7, color="firebrick")
    ax.text(
        0.02,
        0.95,
        "Pass threshold = 0.10 Hz (far above plotted range)",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="firebrick",
    )
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Frequency absolute error (Hz)")
    ax.set_title("Detected frequency error by required condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_frequency_error_by_condition.pdf")
    fig.savefig(FIG_DIR / "fig2_frequency_error_by_condition.png", dpi=200)
    plt.close(fig)


def plot_contrast_validation(merged: list[dict[str, Any]]) -> None:
    rows = [
        r for r in merged
        if r["condition"] in ("contrast_medium", "contrast_low")
    ]
    labels = [label(r["condition"]) for r in rows]
    x = np.arange(len(rows))
    width = 0.36
    v0_recall = [fnum(r.get("V0_count_recall_mean")) for r in rows]
    v1_recall = [fnum(r.get("V1_count_recall_mean")) for r in rows]
    v0_range = [fnum(r.get("V0_signal_range_mean")) for r in rows]
    v1_range = [fnum(r.get("V1_signal_range_mean")) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(x - width / 2, v0_recall, width, label="V0")
    axes[0].bar(x + width / 2, v1_recall, width, label="V1")
    axes[0].axhline(0.85, color="firebrick", linestyle="--", linewidth=1.0)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Repeat count recall")
    axes[0].set_ylabel("Count recall")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x - width / 2, v0_range, width, label="V0")
    axes[1].bar(x + width / 2, v1_range, width, label="V1")
    axes[1].set_title("ROI-local signal range")
    axes[1].set_ylabel("Camera units")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    fig.suptitle("Corrected contrast repeat validation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_contrast_validation.pdf")
    fig.savefig(FIG_DIR / "fig3_contrast_validation.png", dpi=200)
    plt.close(fig)


def plot_auto_roi_status(merged: list[dict[str, Any]]) -> None:
    rows = merged
    labels = [label(r["condition"]) for r in rows]
    success = [1 if r.get("auto_roi_success") is True else 0 for r in rows]
    colors = ["#4c78a8" if r.get("required_for_ready") else "#f58518" for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(np.arange(len(rows)), success, color=colors)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Auto ROI valid")
    ax.set_title("Auto ROI calibration validity")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.01, 0.98, "blue: readiness condition; orange: stress condition",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_auto_roi_validity.pdf")
    fig.savefig(FIG_DIR / "fig4_auto_roi_validity.png", dpi=200)
    plt.close(fig)


def plot_signal_range(merged: list[dict[str, Any]]) -> None:
    labels = [label(r["condition"]) for r in merged]
    x = np.arange(len(merged))
    width = 0.36
    v0 = [fnum(r.get("V0_signal_range_mean")) for r in merged]
    v1 = [fnum(r.get("V1_signal_range_mean")) for r in merged]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width / 2, [np.nan if v is None else v for v in v0], width, label="V0")
    ax.bar(x + width / 2, [np.nan if v is None else v for v in v1], width, label="V1")
    ax.set_ylabel("ROI local signal range (camera units)")
    ax.set_title("ROI-local signal range by condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_signal_range_by_condition.pdf")
    fig.savefig(FIG_DIR / "fig5_signal_range_by_condition.png", dpi=200)
    plt.close(fig)


def copy_image(src: Path, dest_name: str) -> Path | None:
    if not src.exists():
        return None
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    dest = APPENDIX_DIR / dest_name
    shutil.copy2(src, dest)
    return dest


def image_path_for(condition: str, image_name: str) -> Path:
    return condition_auto_dir(condition) / image_name


def make_image_grid(condition: str, image_names: list[str], title: str, out_name: str) -> Path | None:
    panels: list[tuple[str, Any | None, str | None]] = []
    found = 0
    for name in image_names:
        src = image_path_for(condition, name)
        panel_title = IMAGE_TITLES.get(name, name)
        if src.exists():
            panels.append((panel_title, mpimg.imread(src), None))
            found += 1
        else:
            msg = f"Missing appendix image for {condition}: {name}"
            GENERATION_WARNINGS.append(msg)
            panels.append((panel_title, None, "not available"))
    if found == 0:
        return None
    cols = 3 if len(panels) > 4 else (2 if len(panels) > 1 else 1)
    rows = math.ceil(len(panels) / cols)
    fig_height = 2.55 * rows if cols == 3 else 3.25 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, fig_height))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, (panel_title, img, missing_text) in zip(axes_arr, panels):
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, missing_text or "not available", ha="center", va="center",
                    fontsize=12, color="firebrick")
            ax.set_facecolor("#f5f5f5")
        ax.set_title(panel_title, fontsize=9)
        ax.set_axis_off()
    for ax in axes_arr[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = APPENDIX_DIR / out_name
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_all_image_grids() -> dict[str, str]:
    grids: dict[str, str] = {}
    for condition in APPENDIX_CONDITIONS:
        names = list(DIAGNOSTIC_IMAGES)
        if condition in ("contrast_medium", "contrast_low", "small_low_contrast_stress"):
            names += CAMERA_IMAGES
        grid = make_image_grid(
            condition,
            names,
            label(condition),
            f"{condition}_diagnostic_grid.png",
        )
        if grid:
            grids[condition] = f"figures/appendix/{grid.name}"

    baseline_grid = make_image_grid(
        "baseline",
        DIAGNOSTIC_IMAGES[:3],
        "Baseline successful auto-ROI calibration",
        "fig6_baseline_auto_roi_example.png",
    )
    if baseline_grid:
        shutil.copy2(baseline_grid, FIG_DIR / "fig6_baseline_auto_roi_example.png")
    contrast_grid = make_image_grid(
        "contrast_low",
        CAMERA_IMAGES,
        "Corrected low-contrast camera evidence",
        "fig7_low_contrast_camera_evidence.png",
    )
    if contrast_grid:
        shutil.copy2(contrast_grid, FIG_DIR / "fig7_low_contrast_camera_evidence.png")
    return grids


def table_to_latex(tabular: str, caption: str, label: str) -> str:
    return "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        tabular,
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])


def make_dataset_table_tex(rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{tabular}{p{0.20\textwidth}p{0.22\textwidth}p{0.12\textwidth}p{0.13\textwidth}p{0.23\textwidth}}",
        r"\toprule",
        r"Dataset & Purpose & Repeats & Duration & Notes \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(
            f"\\scriptsize\\path{{{row['folder']}}} & {tex_escape(row['purpose'])} & "
            f"{tex_escape(row['repeats'])} & {tex_escape(row['duration_s'])} & "
            f"{tex_escape(row['notes'])} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    return table_to_latex("\n".join(body), "Datasets used in this report.", "tab:datasets")


def make_condition_table_tex(rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{tabular}{p{0.25\textwidth}p{0.12\textwidth}p{0.12\textwidth}p{0.18\textwidth}p{0.23\textwidth}}",
        r"\toprule",
        r"Condition & Category & Required & Visual setting & Frequency setting \\",
        r"\midrule",
    ]
    for row in rows:
        cond = row["condition"]
        auto = load_auto_rois(cond)
        freqs = "V0/V1 "
        if cond in ("frequency_1p5_2p5",):
            freqs += "1.5/2.5 Hz"
        elif cond in ("frequency_2hz_phase_offset", "frequency_2hz_near_simultaneous"):
            freqs += "2/2 Hz"
        else:
            freqs += "1/2 Hz"
        body.append(
            f"{tex_escape(label(cond))} & {tex_escape(row.get('category'))} & "
            f"{'yes' if row.get('required_for_ready') else 'no'} & "
            f"{tex_escape(display_setting(cond, auto))} & {tex_escape(freqs)} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    return table_to_latex("\n".join(body), "Step5c1 condition set and display settings.", "tab:conditions")


def make_results_table_tex(rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{tabular}{p{0.26\textwidth}rrrrcc}",
        r"\toprule",
        r"Condition & V0 recall & V1 recall & V0 $|e_f|$ & V1 $|e_f|$ & Auto ROI & Pass \\",
        r"\midrule",
    ]
    for row in rows:
        if row["condition"] not in REPEAT_REQUIRED_CONDITIONS:
            continue
        pass_text = "yes" if str(row.get("pass_condition")).lower() == "true" else "no"
        body.append(
            f"{tex_escape(label(row['condition']))} & {fmt(row.get('V0_count_recall_mean'))} & "
            f"{fmt(row.get('V1_count_recall_mean'))} & "
            f"{fmt(row.get('V0_frequency_error_mean_hz'), 4)} & "
            f"{fmt(row.get('V1_frequency_error_mean_hz'), 4)} & "
            f"{'yes' if row.get('auto_roi_success') else 'no'} & {pass_text} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    return table_to_latex(
        "\n".join(body),
        "Repeat-based required-condition results. Medium and low contrast use the corrected contrast rerun.",
        "tab:results",
    )


def make_contrast_table_tex(rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{tabular}{p{0.24\textwidth}rrrrcc}",
        r"\toprule",
        r"Condition & V0 recall & V1 recall & V0 $|e_f|$ & V1 $|e_f|$ & Auto ROI & Pass \\",
        r"\midrule",
    ]
    for row in rows:
        if row["condition"] not in ("contrast_medium", "contrast_low"):
            continue
        pass_text = "yes" if str(row.get("pass_condition")).lower() == "true" else "no"
        body.append(
            f"{tex_escape(label(row['condition']))} & "
            f"{fmt(row.get('V0_count_recall_mean'), 3)} & "
            f"{fmt(row.get('V1_count_recall_mean'), 3)} & "
            f"{fmt(row.get('V0_frequency_error_mean_hz'), 4)} & "
            f"{fmt(row.get('V1_frequency_error_mean_hz'), 4)} & "
            f"{'valid' if row.get('auto_roi_success') else 'invalid'} & {pass_text} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    return table_to_latex(
        "\n".join(body),
        "Corrected contrast repeat validation from the corrected rerun.",
        "tab:contrast",
    )


def make_stress_table_tex(rows: list[dict[str, Any]]) -> str:
    body = [
        r"\begin{tabular}{p{0.26\textwidth}p{0.17\textwidth}p{0.17\textwidth}p{0.30\textwidth}}",
        r"\toprule",
        r"Stress condition & Auto ROI & Detection result & Interpretation \\",
        r"\midrule",
    ]
    for row in rows:
        if row["condition"] not in STRESS_CONDITIONS:
            continue
        result = "pass" if str(row.get("pass_condition")).lower() == "true" else "boundary/fail"
        if row.get("repeats_completed") in (0, "0"):
            result = "auto-ROI failed; repeats skipped"
        body.append(
            f"{tex_escape(label(row['condition']))} & {'valid' if row.get('auto_roi_success') else 'invalid'} & "
            f"{tex_escape(result)} & {tex_escape(row.get('notes', ''))} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    return table_to_latex("\n".join(body), "Optional stress outcomes.", "tab:stress")


def write_tables(dataset_rows: list[dict[str, Any]], merged: list[dict[str, Any]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "table_datasets.tex").write_text(make_dataset_table_tex(dataset_rows))
    (TABLE_DIR / "table_conditions.tex").write_text(make_condition_table_tex(merged))
    (TABLE_DIR / "table_results.tex").write_text(make_results_table_tex(merged))
    (TABLE_DIR / "table_contrast.tex").write_text(make_contrast_table_tex(merged))
    (TABLE_DIR / "table_stress.tex").write_text(make_stress_table_tex(merged))


def write_report_tex(readiness: dict[str, Any], image_grids: dict[str, str]) -> None:
    ready_text = "ready" if readiness["ready_for_2v1p_eapf_smoke"] else "not yet fully ready"
    contrast_diag = readiness.get("contrast_repeat_inspection", {}).get("diagnosis", "")
    failed_required = ", ".join(label(c) for c in readiness.get("failed_required_conditions", [])) or "none"
    pass_count = readiness.get("repeat_based_required_pass_count", 0)
    required_count = readiness.get("repeat_based_required_count", 0)
    tex = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{placeins}}
\usepackage{{array}}
\usepackage{{url}}
\hypersetup{{hidelinks}}
\emergencystretch=1em

\title{{Two-Flash Multi-ROI Visual Detection Robustness\\for Mixed-Reality Multi-Agent HIL}}
\date{{\today}}

\newcommand{{\figdir}}{{figures}}
\newcommand{{\tabledir}}{{tables}}

\begin{{document}}
\maketitle

\FloatBarrier

\section{{Introduction}}

The broader project investigates decentralised visual flashing as a mechanism
for coordinating multi-drone behaviour. Previous stages completed fixed-leader
visual HIL, a 1-virtual + 1-Pi mutual visual HIL comparison, and simulation
model selection in which EAPF Consensus was selected as the primary candidate
for the next 2-virtual + 1-Pi mixed-reality HIL stage. Before closing that
three-agent loop, the visual sensing layer must first demonstrate that the Pi
camera can observe two simultaneous virtual flashes as two independent event
streams.

This report therefore evaluates two-flash multi-ROI detection robustness. It
does not test synchronisation performance and does not alter the locked EAPF
Consensus parameters. It asks a narrower engineering question: can the Pi
automatically locate two browser-rendered flash targets, assign independent
ROIs to V0 and V1, detect each flash stream independently, and estimate
frequency accurately across visual conditions? The experiment is detection-only:
no EAPF closed-loop adaptation or Pi oscillator feedback is enabled. On the
available required evidence the detection pipeline is {ready_text} for a
cautious 2V+1P EAPF smoke test under the full readiness gate. The corrected
contrast rerun supplies repeat-based evidence: medium contrast passed, whereas
low contrast failed because V0 produced no rising-edge detections.

\section{{Experimental Objective}}

The objective was to validate the visual detection layer for mixed-reality
multi-agent HIL. The readiness conditions covered baseline geometry,
moderately shifted positions, asymmetric and small target sizes, corrected
medium and low contrast settings, and several frequency relationships. Optional
stress conditions probed boundary cases and were not allowed to block the
readiness decision.

\section{{Method}}

\subsection{{Detection-only setup}}

The browser rendered two virtual flash targets, V0 and V1, at fixed screen
positions. The Pi camera observed both targets. For each condition the runner
first performed automatic simultaneous ROI calibration and then, where the
condition came from a formal batch, ran detection-only repeats. The Pi did not
run a closed-loop EAPF oscillator, did not flash a GPIO LED, and did not post
Pi flash events into a mutual synchronisation loop.

\subsection{{Auto ROI calibration}}

Auto ROI calibration used simultaneous two-target flashing. V0 and V1 were
configured with distinct frequencies when possible, typically 1\,Hz and
2\,Hz. The Pi captured a short frame sequence, computed a temporal-variation
image, found connected components, selected two non-overlapping candidate
regions, and assigned the components to V0 and V1 using the brightness-trace
frequency estimate. The resulting ROIs were then used by the multi-ROI detector.
Each ROI maintained independent signal history, adaptive normalisation,
hysteresis state, rising-edge detection, and duplicate suppression.

\subsection{{Data selection}}

\input{{\tabledir/table_datasets.tex}}

The main formal batch is used for baseline, position, size, frequency, and
general stress results. The original contrast rows from that batch are not used
as final contrast evidence because the display pathway was later debugged.
Final medium- and low-contrast recall and frequency-error metrics are taken
from the corrected contrast repeat batch. The older contrast visual debug
folder is retained only as optional historical display evidence and is not used
for final contrast performance metrics.

\input{{\tabledir/table_conditions.tex}}

\section{{Results}}

\subsection{{Required-condition detection performance}}

The repeat-based required conditions are shown together in
Figures~\ref{{fig:recall}} and~\ref{{fig:freqerr}}. Baseline, position, size,
and frequency rows come from the main formal batch; medium and low contrast
come from the corrected contrast repeat batch. {pass_count} of {required_count}
required conditions passed. The failed required condition was {tex_escape(failed_required)}.

\input{{\tabledir/table_results.tex}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig1_detection_recall_by_condition.pdf}}
\caption{{Detection count recall for required conditions. Baseline, position,
size, and frequency rows use the main formal batch; medium and low contrast use
the corrected contrast repeat batch.}}
\label{{fig:recall}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig2_frequency_error_by_condition.pdf}}
\caption{{Detected frequency absolute error for repeat-based required
conditions. The y-axis is scaled to show the observed errors; the 0.10\,Hz pass
threshold is far above this plotted range.}}
\label{{fig:freqerr}}
\end{{figure}}
\FloatBarrier

\subsection{{Corrected contrast repeat validation}}

The frontend/server/browser contrast pathway was debugged after the original
formal batch. The corrected contrast repeat folder was inspected for
\texttt{{batch\_summary.json}}, \texttt{{batch\_summary.csv}}, condition
folders, repeat folders, per-repeat \texttt{{metrics\_summary.json}}, and
\texttt{{pi\_detection\_roi.csv}} logs. {tex_escape(contrast_diag)} Medium
contrast passed the repeat-based threshold. Low contrast failed because V0 had
zero detected rising edges across the three repeats; V1 remained detectable.

\input{{\tabledir/table_contrast.tex}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig3_contrast_validation.pdf}}
\caption{{Corrected contrast repeat validation. Count recall and ROI-local
signal range show that medium contrast remained detectable, while low contrast
suppressed V0 detection under the current camera/display setup.}}
\label{{fig:contrast_validation}}
\end{{figure}}
\FloatBarrier

\subsection{{Auto ROI and ROI-local signal evidence}}

Auto ROI calibration was valid for all required conditions, including the
corrected contrast repeats. The required-condition failure in low contrast was
therefore not an ROI-localisation failure; it was a detection-threshold/signal
amplitude failure for V0 during the repeat trials. ROI-local signal range is
reported because it is more informative than a full-frame background mean when
the camera frame includes bright non-target regions.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig4_auto_roi_validity.pdf}}
\caption{{Auto ROI calibration validity across required and optional stress
conditions. The too-close stress case failed auto-ROI validation and was not
part of the readiness gate.}}
\label{{fig:auto_roi}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig5_signal_range_by_condition.pdf}}
\caption{{ROI-local signal range by condition. ROI-local metrics are preferred
over global background contrast because camera frames can include bright
non-target regions such as browser chrome or window borders.}}
\label{{fig:signal_range}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig6_baseline_auto_roi_example.png}}
\caption{{Representative successful baseline auto-ROI calibration: selected
ROIs, temporal variation image, and threshold mask.}}
\label{{fig:baseline_auto_roi}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{\figdir/fig7_low_contrast_camera_evidence.png}}
\caption{{Corrected low-contrast camera evidence from the contrast rerun. The
global frame background mean is not used as the primary contrast metric because
non-target regions can dominate it; ROI-local signal ranges and trace
frequencies provide the more relevant evidence.}}
\label{{fig:contrast_evidence}}
\end{{figure}}
\FloatBarrier

\subsection{{Optional stress conditions}}

Optional stress conditions were analysed separately from the readiness gate.
They are intended to reveal visual boundary cases rather than to define the
geometry for the first 2V+1P smoke test. The too-close stress condition failed
before repeats because the calibration could not produce a valid two-ROI
configuration. The tiny-small stress condition passed and therefore provides
additional evidence of size robustness. The small low-contrast stress condition
failed on V0 detection while V1 remained detectable, marking a combined
small-target/low-contrast boundary case.

\input{{\tabledir/table_stress.tex}}

\subsection{{Readiness for 2V+1P EAPF smoke}}

The readiness decision is based on required conditions only. {pass_count} of
{required_count} repeat-based required conditions passed. Medium contrast
passed, but low contrast failed on V0 detection. Under the current readiness
gate the detection system is therefore not yet fully ready for a formal 2V+1P
EAPF smoke test that must include the low-contrast setting. A cautious smoke
test remains technically defensible only if the display/camera setup uses the
validated high- or medium-contrast operating range.

\section{{Discussion}}

The main formal batch supports the core two-flash detection pipeline. All
required non-contrast conditions passed the count-recall and frequency-error
thresholds, with auto ROI calibration valid and overlap ratios at zero. The
moderately shifted position condition passed, indicating that the connected
component based auto-ROI procedure is not tied to only one screen placement.
The size conditions also passed, including the required small-small case.

The corrected contrast evidence changes the readiness interpretation. The
frontend/server/browser pathway can apply separate baseline, medium, and low
brightness settings, and auto ROI calibration remained valid. However, the
repeat detector failed for V0 in the low-contrast condition. This makes low
contrast a practical lower bound for the current detector/camera exposure
configuration rather than a passed robustness condition.

Optional stress conditions behaved as useful boundary probes. The
position-too-close stress condition was intentionally difficult: large flash
halos were nearly touching, so padded ROIs could become ambiguous. Its failure
does not represent the intended 2V+1P layout and does not block readiness. The
tiny-small and small-low-contrast stress conditions are likewise reported as
limits rather than required success criteria.

\subsection{{Limitations}}

First, low contrast failed for V0 under the current camera/display setup, so
the low-contrast setting should be avoided or retuned before formal closed-loop
testing. Second, camera auto-exposure and bright non-target regions can make a full-frame
background mean misleading; ROI-local signal range is the preferred contrast
metric here. Third, the experiments use browser-rendered targets rather than
two physical drone LEDs. Fourth, passing detection-only tests does not prove
closed-loop 2V+1P synchronisation stability; it only validates that the visual
event streams are sufficiently separable to attempt a smoke test.

\section{{Conclusion}}

The Step5c1 evidence supports the core two-flash detection pipeline in the
validated high- and medium-contrast operating range, but it does not pass the
full required robustness gate because low contrast failed for V0. The next
2V+1P EAPF smoke test should therefore either use the validated contrast range
or first improve/retune low-contrast detection. Stress condition failures
should be interpreted as boundary cases, not blockers for the intended
smoke-test geometry.

\appendix

\section{{Validation Tables}}

The merged machine-readable analysis table is saved as
\path{{derived/merged_condition_summary.csv}}, and per-repeat rows used from
formal detection batches are saved as
\path{{derived/merged_repeat_rows.csv}}.
The corrected contrast repeat-data inspection is saved as
\path{{derived/contrast_repeat_inspection.json}}.
Any missing appendix image panels or generation warnings are saved as
\path{{derived/generation_warnings.json}}.

\section{{Diagnostic Images}}

The following appendix grids copy representative diagnostics into this report
folder. They are curated from the raw logs; the raw experiment folders are not
modified.

"""
    for condition in APPENDIX_CONDITIONS:
        key = condition
        if key not in image_grids:
            continue
        tex += rf"""
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{{{image_grids[key]}}}
\caption{{Auto-ROI diagnostics for {tex_escape(label(condition))}.}}
\end{{figure}}
"""
    tex += r"""
\section{Reproducibility}

This report was generated by running
\path{python generate_step5c1_report.py} from
\path{docs/step5c1_multi_roi_detection_report/}. The LaTeX source is
\path{step5c1_multi_roi_detection_report.tex}. The PDF can be rebuilt with
\path{pdflatex step5c1_multi_roi_detection_report.tex}.

\end{document}
"""
    (REPORT_DIR / "step5c1_multi_roi_detection_report.tex").write_text(tex)


def generate() -> dict[str, Any]:
    GENERATION_WARNINGS.clear()
    for path in (FIG_DIR, APPENDIX_DIR, TABLE_DIR, DERIVED_DIR):
        path.mkdir(parents=True, exist_ok=True)
    contrast_inspection = inspect_corrected_contrast_repeats()
    ensure_corrected_contrast_complete(contrast_inspection)
    dataset_rows = build_dataset_table()
    merged, repeat_rows = build_merged_condition_table()
    ready = readiness_summary(merged)
    ready["contrast_repeat_inspection"] = contrast_inspection
    write_csv(DERIVED_DIR / "dataset_sources.csv", dataset_rows)
    write_csv(DERIVED_DIR / "merged_condition_summary.csv", merged)
    write_csv(DERIVED_DIR / "merged_repeat_rows.csv", repeat_rows)
    write_json(DERIVED_DIR / "contrast_repeat_inspection.json", contrast_inspection)
    write_json(DERIVED_DIR / "readiness_summary.json", ready)
    write_tables(dataset_rows, merged)
    plot_recall(merged)
    plot_frequency_error(merged)
    plot_contrast_validation(merged)
    plot_auto_roi_status(merged)
    plot_signal_range(merged)
    grids = make_all_image_grids()
    write_json(DERIVED_DIR / "appendix_image_index.json", grids)
    write_json(DERIVED_DIR / "generation_warnings.json", {"warnings": GENERATION_WARNINGS})
    write_report_tex(ready, grids)
    return {
        "readiness": ready,
        "conditions": merged,
        "image_grids": grids,
        "warnings": GENERATION_WARNINGS,
    }


if __name__ == "__main__":
    summary = generate()
    print(json.dumps(summary["readiness"], indent=2))

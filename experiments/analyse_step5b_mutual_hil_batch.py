#!/usr/bin/env python3
"""Analyse Step 5B mutual HIL model-comparison batches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _condition(row: dict) -> str:
    return f"V{float(row['virtual_initial_freq']):g}/P{float(row['pi_initial_freq']):g}"


def _group_values(rows: list[dict], metric: str) -> tuple[list[str], list[str], dict[tuple[str, str], list[float]]]:
    models = sorted({r["model"] for r in rows})
    conditions = sorted({_condition(r) for r in rows})
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = _as_float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault((row["model"], _condition(row)), []).append(value)
    return models, conditions, grouped


def _bar_metric(rows: list[dict], metric: str, ylabel: str, title: str,
                fig_dir: Path, filename: str) -> None:
    models, conditions, grouped = _group_values(rows, metric)
    if not models or not conditions:
        return
    x = np.arange(len(conditions))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.4), 4))
    for i, model in enumerate(models):
        means = []
        stds = []
        for condition in conditions:
            vals = grouped.get((model, condition), [])
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else np.nan)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, means, width,
               yerr=stds, capsize=4, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _successful_rows(rows: list[dict], model: str) -> list[dict]:
    candidates = [
        r for r in rows
        if r.get("model") == model and str(r.get("timeout_or_failure", "")).lower() not in ("true", "1")
    ]
    candidates.sort(key=lambda r: _as_float(r.get("final_frequency_error_hz")) or 999.0)
    return candidates


def _plot_representative_raster(row: dict, fig_dir: Path, suffix: str) -> None:
    trial_dir = Path(row["trial_dir"])
    flash_path = trial_dir / "flash_events.csv"
    if not flash_path.exists():
        return
    events = _read_csv(flash_path)
    virtual = [_as_float(e.get("t_s")) for e in events if e.get("event") == "virtual_flash"]
    pi = [_as_float(e.get("t_s")) for e in events if e.get("event") == "pi_flash"]
    virtual = [v for v in virtual if v is not None]
    pi = [v for v in pi if v is not None]
    if not virtual and not pi:
        return
    fig, ax = plt.subplots(figsize=(10, 2.5))
    if virtual:
        ax.eventplot(virtual, lineoffsets=1, colors="steelblue", linewidths=0.8)
    if pi:
        ax.eventplot(pi, lineoffsets=0, colors="darkorange", linewidths=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Pi", "Virtual"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Representative Flash Raster: {row['model']} {suffix}")
    fig.savefig(fig_dir / f"representative_flash_raster_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_representative_frequency(row: dict, fig_dir: Path, suffix: str) -> None:
    trial_dir = Path(row["trial_dir"])
    osc_path = trial_dir / "oscillator_log.csv"
    if not osc_path.exists():
        return
    rows = _read_csv(osc_path)
    t = [_as_float(r.get("t_s")) for r in rows]
    pi = [_as_float(r.get("frequency_hz")) for r in rows]
    virt = [_as_float(r.get("virtual_frequency_hz")) for r in rows]
    data = [(a, b, c) for a, b, c in zip(t, pi, virt) if a is not None and b is not None]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot([d[0] for d in data], [d[1] for d in data], label="Pi")
    virt_data = [(a, c) for a, _b, c in data if c is not None]
    if virt_data:
        ax.plot([d[0] for d in virt_data], [d[1] for d in virt_data], label="Virtual")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Representative Frequency Trajectory: {row['model']} {suffix}")
    ax.legend()
    fig.savefig(fig_dir / f"representative_frequency_trajectory_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a Step 5B mutual HIL batch directory.")
    parser.add_argument("batch_dir", type=Path)
    args = parser.parse_args()

    batch_dir = args.batch_dir
    aggregate = batch_dir / "aggregate_metrics.csv"
    if not aggregate.exists():
        raise SystemExit(f"aggregate_metrics.csv not found: {aggregate}")

    rows = _read_csv(aggregate)
    fig_dir = batch_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _bar_metric(rows, "actual_detection_fcr", "Actual detection FCR",
                "Actual Detection FCR by Model and Frequency Pair",
                fig_dir, "actual_detection_fcr_by_model_condition.png")
    _bar_metric(rows, "final_frequency_error_hz", "Final frequency error (Hz)",
                "Final Frequency Error by Model and Frequency Pair",
                fig_dir, "final_frequency_error_by_model_condition.png")
    _bar_metric(rows, "mean_timing_error_final_10s", "Final-10s timing error (s)",
                "Final-10s Timing Error by Model and Frequency Pair",
                fig_dir, "final10_timing_error_by_model_condition.png")
    _bar_metric(rows, "time_to_frequency_lock_s", "Time to frequency lock (s)",
                "Time to Frequency Lock by Model and Frequency Pair",
                fig_dir, "time_to_frequency_lock_by_model_condition.png")
    _bar_metric(rows, "time_to_timing_lock_s", "Time to timing lock (s)",
                "Time to Timing Lock by Model and Frequency Pair",
                fig_dir, "time_to_timing_lock_by_model_condition.png")
    _bar_metric(rows, "pi_final_frequency_hz", "Final common/Pi frequency (Hz)",
                "Final Common Frequency by Model and Frequency Pair",
                fig_dir, "final_common_frequency_by_model_condition.png")

    for model in ("eapf_consensus", "kuramoto"):
        reps = _successful_rows(rows, model)
        if not reps:
            continue
        suffix = model
        _plot_representative_raster(reps[0], fig_dir, suffix)
        _plot_representative_frequency(reps[0], fig_dir, suffix)

    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()

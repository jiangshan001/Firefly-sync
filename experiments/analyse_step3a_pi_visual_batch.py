#!/usr/bin/env python3
r"""Step 3A-3 Batch Analysis — Corrected Summary + Thesis Figures.

Reads a Pi visual batch output directory and generates:
  - ``corrected_summary_by_condition.csv``
  - ``figures/fig1_*.png/pdf`` through ``figures/fig7_*.png/pdf``

Usage::

    PYTHONPATH=. python3 experiments/analyse_step3a_pi_visual_batch.py \
        --batch-dir experiments/logs/step3a_pi_visual_batch/<timestamp>_kuramoto_pi_visual_batch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# Column alias resolution
# ======================================================================

_COLUMN_ALIASES: dict[str, list[str]] = {
    "steady_state_mae_s": [
        "steady_state_mean_abs_timing_error_s",
        "steady_state_mae_s",
        "mean_abs_timing_error_s",
    ],
    "steady_state_rmse_s": [
        "steady_state_rmse_timing_error_s",
        "steady_state_rmse_s",
        "rmse_timing_error_s",
    ],
    "steady_state_jitter_s": [
        "steady_state_jitter_s",
        "jitter_s",
    ],
    "final_frequency_error_hz": [
        "final_frequency_error_hz",
        "frequency_error_hz",
    ],
    "convergence_quality": [
        "convergence_quality",
    ],
    "time_to_sync_s": [
        "time_to_synchronization_s",
        "time_to_sync_s",
    ],
    "leader_flash_count_ratio": [
        "leader_flash_count_ratio",
        "detected_to_expected_leader_flash_ratio",
    ],
    "detected_leader_flash_count": [
        "detected_leader_flash_count",
    ],
    "expected_leader_flash_count": [
        "expected_leader_flash_count",
    ],
    "actual_wall_duration_s": [
        "actual_trial_wall_duration_s",
        "actual_wall_duration_s",
    ],
    "requested_duration_s": [
        "requested_duration_s",
    ],
    "effective_loop_rate_hz": [
        "effective_loop_rate_hz",
        "loop_rate_hz",
    ],
    "mean_loop_dt_ms": [
        "mean_loop_dt_ms",
        "mean_loop_dt_s",
    ],
    "mean_loop_dt_s": [
        "mean_loop_dt_s",
        "mean_loop_dt_ms",
    ],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the first matching column name in *df*, or None."""
    for alias in _COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def _safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# ======================================================================
# Corrected summary
# ======================================================================

def build_corrected_summary(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Compute per-condition statistics from aggregate_metrics rows."""

    metric_cols = [
        "time_to_sync_s", "steady_state_mae_s", "steady_state_rmse_s",
        "steady_state_jitter_s", "final_frequency_error_hz",
        "convergence_quality",
        "leader_flash_count_ratio", "detected_leader_flash_count",
        "expected_leader_flash_count",
        "actual_wall_duration_s", "requested_duration_s",
        "effective_loop_rate_hz", "mean_loop_dt_ms",
    ]

    # Resolve columns
    resolved: dict[str, str | None] = {}
    print("\nColumn mapping:")
    for canon in metric_cols:
        r = _resolve_column(df, canon)
        resolved[canon] = r
        status = f"  {canon} -> {r}" if r else f"  {canon} -> [NOT FOUND]"
        print(status)
    print()

    # Convert to numeric
    for _, col in resolved.items():
        if col:
            df[col] = _safe_float(df[col])

    # Group
    grouped = df.groupby(group_cols, dropna=False)
    records: list[dict] = []

    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec: dict[str, Any] = {}
        for i, gcol in enumerate(group_cols):
            rec[gcol] = keys[i]

        n = len(grp)
        rec["n_trials"] = n

        # Sync success
        sync_col = "synchronization_success"
        if sync_col in grp.columns:
            sync_vals = _safe_float(grp[sync_col]).fillna(0).astype(int)
            rec["success_rate"] = round(sync_vals.mean(), 4)

        # Means / stds for each metric
        for canon in metric_cols:
            col = resolved[canon]
            if col is None:
                rec[f"mean_{canon}"] = ""
                rec[f"std_{canon}"] = ""
                continue
            series = _safe_float(grp[col]).dropna()
            if len(series) == 0:
                rec[f"mean_{canon}"] = ""
                rec[f"std_{canon}"] = ""
            else:
                rec[f"mean_{canon}"] = round(float(series.mean()), 6)
                rec[f"std_{canon}"] = round(float(series.std()), 6) if len(series) >= 2 else ""

        records.append(rec)

    return pd.DataFrame(records)


# ======================================================================
# Thesis figure style
# ======================================================================

def _setup_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)


def _save_both(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")


# ======================================================================
# Figure generators
# ======================================================================

def fig1_success_rate(sdf: pd.DataFrame, out_dir: Path, subtitle: str) -> None:
    freqs = sdf["follower_initial_freq_hz"].astype(str).tolist()
    rates = [float(sdf.iloc[i]["success_rate"]) * 100 for i in range(len(sdf))]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(freqs))
    bars = ax.bar(x, rates, color="steelblue", edgecolor="black", width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Synchronisation Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Synchronisation success rate")
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig1_success_rate_by_condition")
    plt.close(fig)


def fig2_time_to_sync(sdf: pd.DataFrame, df: pd.DataFrame,
                      out_dir: Path, subtitle: str) -> None:
    freqs = sdf["follower_initial_freq_hz"].astype(str).tolist()
    means = [float(sdf.iloc[i].get("mean_time_to_sync_s", 0) or 0) for i in range(len(sdf))]
    stds = [float(sdf.iloc[i].get("std_time_to_sync_s", 0) or 0) for i in range(len(sdf))]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(freqs))
    ax.bar(x, means, yerr=stds, capsize=6, color="darkorange",
           edgecolor="black", width=0.5)

    # Overlay individual trial points
    tts_col = _resolve_column(df, "time_to_sync_s")
    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        vals = _safe_float(df.loc[mask, tts_col]).dropna() if tts_col else pd.Series(dtype=float)
        if len(vals) > 0:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(x[i] + jitter, vals, color="black", s=20, alpha=0.5, zorder=3)

    n_failed = 0
    if tts_col:
        sync_mask = df["synchronization_success"].astype(str).str.lower().isin(["false", "0", "0.0"])
        n_failed = sync_mask.sum()

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Time to Synchronisation (s)")
    ax.set_title("Time to synchronisation")
    if subtitle:
        sub = subtitle
        if n_failed > 0:
            sub += f"  [{n_failed} trial(s) did not sync]"
        ax.text(0.5, -0.14, sub, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig2_time_to_sync_by_condition")
    plt.close(fig)


def fig3_steady_state_mae(sdf: pd.DataFrame, df: pd.DataFrame,
                          out_dir: Path, subtitle: str) -> None:
    freqs = sdf["follower_initial_freq_hz"].astype(str).tolist()
    # Convert to ms
    means = [float(sdf.iloc[i].get("mean_steady_state_mae_s", 0) or 0) * 1000
             for i in range(len(sdf))]
    stds = [float(sdf.iloc[i].get("std_steady_state_mae_s", 0) or 0) * 1000
            for i in range(len(sdf))]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(freqs))
    ax.bar(x, means, yerr=stds, capsize=6, color="darkorange",
           edgecolor="black", width=0.5)

    mae_col = _resolve_column(df, "steady_state_mae_s")
    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        if mae_col:
            vals = _safe_float(df.loc[mask, mae_col]).dropna() * 1000
            if len(vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(x[i] + jitter, vals, color="black", s=20, alpha=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Steady-State MAE (ms)")
    ax.set_title("Steady-state timing error after synchronisation")
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig3_steady_state_mae_by_condition")
    plt.close(fig)


def fig4_detection_reliability(sdf: pd.DataFrame, df: pd.DataFrame,
                               out_dir: Path, subtitle: str) -> None:
    freqs = sdf["follower_initial_freq_hz"].astype(str).tolist()

    ratio_col = _resolve_column(df, "leader_flash_count_ratio")
    means: list[float] = []
    stds: list[float] = []
    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        if ratio_col:
            vals = _safe_float(df.loc[mask, ratio_col]).dropna() * 100
            means.append(float(vals.mean()) if len(vals) > 0 else 0)
            stds.append(float(vals.std()) if len(vals) >= 2 else 0)
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(freqs))
    ax.bar(x, means, yerr=stds, capsize=6, color="seagreen",
           edgecolor="black", width=0.5)

    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        if ratio_col:
            vals = _safe_float(df.loc[mask, ratio_col]).dropna() * 100
            if len(vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(x[i] + jitter, vals, color="black", s=20, alpha=0.5, zorder=3)

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Detected / Expected Leader Flashes (%)")
    ax.set_title("Leader flash detection reliability")
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig4_leader_detection_reliability")
    plt.close(fig)


# ======================================================================
# Supplementary figures
# ======================================================================

def _compute_nearest_cycle_error(
    leader_times: list[float],
    follower_times: list[float],
) -> list[tuple[float, float]]:
    """Return [(follower_t, error_s)] using nearest-leader matching."""
    lt = np.array(leader_times)
    result: list[tuple[float, float]] = []
    for ft in follower_times:
        if len(lt) == 0:
            continue
        idx = int(np.argmin(np.abs(lt - ft)))
        error = ft - lt[idx]
        result.append((ft, float(error)))
    return result


def _pick_representative(trial_dirs: list[Path], mae_col: str) -> Path | None:
    """Pick the trial closest to the median MAE."""
    maes: list[tuple[float, Path]] = []
    for td in trial_dirs:
        ms = td / "metrics_summary.json"
        if not ms.exists():
            continue
        try:
            m = json.loads(ms.read_text())
            maes.append((float(m.get(mae_col, float("inf"))), td))
        except (ValueError, KeyError):
            continue
    if not maes:
        return None
    maes.sort(key=lambda x: x[0])
    median_idx = len(maes) // 2
    return maes[median_idx][1]


def fig5_timing_error_timeseries(
    batch_dir: Path, df: pd.DataFrame, out_dir: Path, subtitle: str,
) -> None:
    """One representative trial per follower frequency."""
    freq_col = "follower_initial_freq_hz"
    freqs = sorted(df[freq_col].dropna().unique())
    trials_dir = batch_dir / "trials"

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(freqs)))

    for i, freq in enumerate(freqs):
        mask = (df[freq_col].astype(float) - float(freq)).abs() < 0.01
        trial_ids = df.loc[mask, "trial_id"].tolist()
        sub_dirs = [trials_dir / tid for tid in trial_ids if (trials_dir / tid).is_dir()]
        if not sub_dirs:
            continue
        rep = _pick_representative(sub_dirs, "steady_state_mean_abs_timing_error_s")
        if rep is None:
            continue
        flash_csv = rep / "flash_events.csv"
        if not flash_csv.exists():
            continue

        # Parse leader/follower flash times
        leader_t: list[float] = []
        follower_t: list[float] = []
        with open(flash_csv, newline="") as f:
            for row in csv.DictReader(f):
                etype = row.get("event_type", "")
                t_val = float(row.get("t", 0))
                if etype == "leader_flash":
                    leader_t.append(t_val)
                elif etype == "follower_flash":
                    follower_t.append(t_val)

        errors = _compute_nearest_cycle_error(leader_t, follower_t)
        if errors:
            times = [e[0] for e in errors]
            errs_ms = [e[1] * 1000 for e in errors]
            ax.plot(times, errs_ms, linewidth=1.0, alpha=0.8,
                    color=colors[i], label=f"Follower {freq:.1f} Hz")

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=-100, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Trial Time (s)")
    ax.set_ylabel("Nearest-Cycle Timing Error (ms)")
    ax.set_title("Representative timing error convergence")
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    ax.legend(fontsize=8)
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig5_representative_timing_error_timeseries")
    plt.close(fig)


def fig6_flash_raster(
    batch_dir: Path, df: pd.DataFrame, out_dir: Path, subtitle: str,
) -> None:
    """Flash raster for one representative trial (first available)."""
    trials_dir = batch_dir / "trials"

    # Find first trial with flash_events.csv
    chosen_dir: Path | None = None
    for td in sorted(trials_dir.iterdir()):
        if (td / "flash_events.csv").exists():
            chosen_dir = td
            break

    if chosen_dir is None:
        print("  [WARN] No trial with flash_events.csv — skipping fig6.")
        return

    leader_t: list[float] = []
    follower_t: list[float] = []
    with open(chosen_dir / "flash_events.csv", newline="") as f:
        for row in csv.DictReader(f):
            etype = row.get("event_type", "")
            t_val = float(row.get("t", 0))
            if etype == "leader_flash":
                leader_t.append(t_val)
            elif etype == "follower_flash":
                follower_t.append(t_val)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Leader row (y=1), Follower row (y=0)
    if leader_t:
        ax.eventplot(leader_t, lineoffsets=1, colors="steelblue",
                     linewidths=1.5, label="Leader")
    if follower_t:
        ax.eventplot(follower_t, lineoffsets=0, colors="darkorange",
                     linewidths=1.5, label="Follower")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Follower", "Leader"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Representative leader-follower flash raster")
    if subtitle:
        ax.text(0.5, -0.18, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    if leader_t or follower_t:
        ax.legend(fontsize=9, loc="upper right")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig6_representative_flash_raster")
    plt.close(fig)


def fig7_loop_rate(sdf: pd.DataFrame, df: pd.DataFrame,
                   out_dir: Path, subtitle: str) -> None:
    """Effective loop rate by condition, if available."""
    rate_col = _resolve_column(df, "effective_loop_rate_hz")
    if rate_col is None:
        # Try to derive from mean_loop_dt_ms
        dt_col = _resolve_column(df, "mean_loop_dt_ms")
        if dt_col:
            df["_derived_rate"] = 1000.0 / _safe_float(df[dt_col])
            rate_col = "_derived_rate"

    freqs = sdf["follower_initial_freq_hz"].astype(str).tolist()
    means: list[float] = []
    stds: list[float] = []
    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        if rate_col:
            vals = _safe_float(df.loc[mask, rate_col]).dropna()
            means.append(float(vals.mean()) if len(vals) > 0 else 0)
            stds.append(float(vals.std()) if len(vals) >= 2 else 0)
        else:
            means.append(0)
            stds.append(0)

    if all(m == 0 for m in means):
        print("  [WARN] No loop rate data — skipping fig7.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(freqs))
    ax.bar(x, means, yerr=stds, capsize=6, color="mediumpurple",
           edgecolor="black", width=0.5)

    for i, fstr in enumerate(freqs):
        f_val = float(fstr)
        mask = (df["follower_initial_freq_hz"].astype(float) - f_val).abs() < 0.01
        if rate_col:
            vals = _safe_float(df.loc[mask, rate_col]).dropna()
            if len(vals) > 0:
                jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
                ax.scatter(x[i] + jitter, vals, color="black", s=20, alpha=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f} Hz" for f in freqs])
    ax.set_ylabel("Effective Loop Rate (Hz)")
    ax.set_title("Pi visual loop performance")
    if subtitle:
        ax.text(0.5, -0.14, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9, color="gray")
    _clean_axes(ax)
    _save_both(fig, out_dir, "fig7_effective_loop_rate_by_condition")
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a Pi visual Kuramoto batch and generate thesis figures.",
    )
    parser.add_argument("--batch-dir", required=True,
                        help="Path to the batch output directory")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_dir():
        print(f"ERROR: batch directory not found: {batch_dir}")
        sys.exit(1)

    agg_csv = batch_dir / "aggregate_metrics.csv"
    if not agg_csv.exists():
        print(f"ERROR: aggregate_metrics.csv not found in {batch_dir}")
        sys.exit(1)

    # --- Load data ---
    df = pd.read_csv(agg_csv)
    print(f"\nLoaded {len(df)} trials from {agg_csv}")
    print(f"Columns: {list(df.columns)}")

    group_cols = ["leader_freq_hz", "follower_initial_freq_hz", "coupling_gain"]
    group_cols = [c for c in group_cols if c in df.columns]

    conditions = sorted(df["follower_initial_freq_hz"].dropna().unique())
    print(f"Conditions detected: {[f'{c:.1f} Hz' for c in conditions]}")

    # --- Build corrected summary ---
    sdf = build_corrected_summary(df, group_cols)
    summary_path = batch_dir / "corrected_summary_by_condition.csv"
    sdf.to_csv(summary_path, index=False)
    print(f"\nCorrected summary saved to {summary_path}")
    print(sdf.to_string(index=False))

    # --- Setup output ---
    out_dir = batch_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    # Subtitle line
    lf_val = df["leader_freq_hz"].iloc[0] if len(df) > 0 else "?"
    k_val = df["coupling_gain"].iloc[0] if len(df) > 0 else "?"
    subtitle = f"Kuramoto Pi visual closed-loop, leader = {lf_val} Hz, K = {k_val}"

    # --- Generate figures ---
    print("\nGenerating figures...")
    fig1_success_rate(sdf, out_dir, subtitle)
    print("  fig1 — success rate [OK]")
    fig2_time_to_sync(sdf, df, out_dir, subtitle)
    print("  fig2 — time to sync [OK]")
    fig3_steady_state_mae(sdf, df, out_dir, subtitle)
    print("  fig3 — steady-state MAE [OK]")
    fig4_detection_reliability(sdf, df, out_dir, subtitle)
    print("  fig4 — detection reliability [OK]")
    fig5_timing_error_timeseries(batch_dir, df, out_dir, subtitle)
    print("  fig5 — timing error timeseries [OK]")
    fig6_flash_raster(batch_dir, df, out_dir, subtitle)
    print("  fig6 — flash raster [OK]")
    fig7_loop_rate(sdf, df, out_dir, subtitle)
    print("  fig7 — loop rate [OK]")

    # --- List generated files ---
    print(f"\nGenerated figures in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

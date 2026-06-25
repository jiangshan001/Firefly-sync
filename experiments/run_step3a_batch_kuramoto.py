#!/usr/bin/env python3
r"""Step 3A-2 — Batch Kuramoto Leader-Follower Evaluation.

Runs multiple closed-loop synchronisation trials across a range of
follower initial frequencies, collecting aggregate metrics and
generating summary plots.

Each trial reuses the single-trial ``run_trial()`` function from
``experiments/run_step3a_kuramoto_closed_loop``.

Usage (smoke test)::

    PYTHONPATH=. python experiments/run_step3a_batch_kuramoto.py \
        --duration 3 --repeats 1 --follower-freqs 1.5 --no-plots

Usage (full batch)::

    PYTHONPATH=. python experiments/run_step3a_batch_kuramoto.py \
        --duration 30 --repeats 5 --follower-freqs 1.5 1.8 2.3
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ======================================================================
# Helpers
# ======================================================================

def _make_batch_dir(root: str, trial_prefix: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{trial_prefix}_" if trial_prefix else ""
    name = f"{ts}_{prefix}kuramoto_batch"
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ======================================================================
# Batch runner
# ======================================================================

def run_batch(args: argparse.Namespace) -> Path:
    """Execute the full batch and return the output directory."""

    # -- import single-trial runner (deferred so CLI help is fast) --
    from experiments.run_step3a_kuramoto_closed_loop import run_trial

    batch_dir = _make_batch_dir(args.log_dir, args.trial_prefix)
    trials_dir = batch_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = batch_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    follower_freqs: list[float] = args.follower_freqs
    repeats: int = args.repeats

    # ------------------------------------------------------------------
    # 1. Batch metadata
    # ------------------------------------------------------------------
    batch_meta = {
        "model_name": "kuramoto",
        "mode": "mock",
        "leader_freq_hz": args.leader_freq,
        "follower_initial_freqs_hz": follower_freqs,
        "coupling_gain": args.coupling_gain,
        "duration_s": args.duration,
        "dt": args.dt,
        "flash_on_time_s": args.flash_on_time,
        "sync_threshold_s": args.sync_threshold_s,
        "sync_cycles": args.sync_cycles,
        "repeats_per_condition": repeats,
        "timestamp": datetime.now().isoformat(),
        "trial_prefix": args.trial_prefix or "",
    }
    with open(batch_dir / "batch_metadata.json", "w") as f:
        json.dump(batch_meta, f, indent=2)

    # ------------------------------------------------------------------
    # 2. Run trials
    # ------------------------------------------------------------------
    aggregate_rows: list[dict[str, Any]] = []
    trial_file_data: list[dict] = []   # for plotting

    total = len(follower_freqs) * repeats
    count = 0

    for freq in follower_freqs:
        for rep in range(1, repeats + 1):
            count += 1
            trial_id = f"f{freq:.1f}Hz_r{rep:02d}"
            print(f"\n{'='*60}")
            print(f"[batch] Trial {count}/{total}: {trial_id}")
            print(f"{'='*60}")

            # Build per-trial Namespace
            trial_args = argparse.Namespace(
                mode="mock",
                duration=args.duration,
                leader_freq=args.leader_freq,
                follower_freq=freq,
                coupling_gain=args.coupling_gain,
                dt=args.dt,
                flash_on_time=args.flash_on_time,
                sync_threshold_s=args.sync_threshold_s,
                sync_cycles=args.sync_cycles,
                log_dir=str(trials_dir),
                trial_id=trial_id,
                notes=f"batch trial, rep {rep}/{repeats}",
                # unused in mock mode but required
                allow_hardware_fallback=False,
                led_pin=17,
                threshold_on=180.0,
                threshold_off=120.0,
                min_interval=0.2,
                window_s=5.0,
            )

            metrics = run_trial(trial_args)

            # Find the trial output dir (created by run_trial inside trials_dir)
            trial_dirs = sorted(
                Path(str(trials_dir)).glob(f"*{trial_id}"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            trial_log_dir = str(trial_dirs[0]) if trial_dirs else ""

            # --- aggregate row ---
            row = {
                "batch_id": batch_dir.name,
                "trial_id": trial_id,
                "model_name": "kuramoto",
                "leader_freq_hz": args.leader_freq,
                "follower_initial_freq_hz": freq,
                "coupling_gain": args.coupling_gain,
                "duration_s": args.duration,
                "synchronization_success": metrics["synchronization_success"],
                "time_to_synchronization_s": metrics.get("time_to_synchronization_s"),
                "steady_state_mean_abs_timing_error_s": metrics["steady_state_mean_abs_timing_error_s"],
                "steady_state_rmse_timing_error_s": metrics["steady_state_rmse_timing_error_s"],
                "steady_state_jitter_s": metrics["steady_state_jitter_s"],
                "final_frequency_error_hz": metrics["final_frequency_error_hz"],
                "convergence_quality": metrics["convergence_quality"],
                "detection_success_rate": metrics.get("detection_success_rate"),
                "false_positive_rate": metrics.get("false_positive_rate"),
                "trial_log_dir": trial_log_dir,
            }
            aggregate_rows.append(row)

            # Collect file paths for plotting
            if trial_dirs:
                td = trial_dirs[0]
                flash_csv = td / "flash_events.csv"
                osc_csv = td / "oscillator_log.csv"
                trial_file_data.append({
                    "label": f"{freq:.1f} Hz (rep {rep})",
                    "freq": freq,
                    "flash_events_csv": str(flash_csv) if flash_csv.exists() else None,
                    "oscillator_log_csv": str(osc_csv) if osc_csv.exists() else None,
                })

    # ------------------------------------------------------------------
    # 3. Save aggregate_metrics.csv
    # ------------------------------------------------------------------
    agg_fields = [
        "batch_id", "trial_id", "model_name", "leader_freq_hz",
        "follower_initial_freq_hz", "coupling_gain", "duration_s",
        "synchronization_success", "time_to_synchronization_s",
        "steady_state_mean_abs_timing_error_s", "steady_state_rmse_timing_error_s",
        "steady_state_jitter_s", "final_frequency_error_hz",
        "convergence_quality", "detection_success_rate", "false_positive_rate",
        "trial_log_dir",
    ]
    agg_path = batch_dir / "aggregate_metrics.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregate_rows)

    # ------------------------------------------------------------------
    # 4. Compute summary_by_condition.csv
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []
    for freq in follower_freqs:
        cond_rows = [r for r in aggregate_rows
                     if abs(r["follower_initial_freq_hz"] - freq) < 0.001]
        n = len(cond_rows)
        successes = [r for r in cond_rows if r["synchronization_success"]]
        n_success = len(successes)
        success_rate = n_success / n if n > 0 else 0.0

        sync_times = [float(r["time_to_synchronization_s"])
                      for r in successes
                      if r["time_to_synchronization_s"] is not None]
        maes = [float(r["steady_state_mean_abs_timing_error_s"])
                for r in cond_rows
                if r["steady_state_mean_abs_timing_error_s"] is not None
                and not (isinstance(r["steady_state_mean_abs_timing_error_s"], float)
                         and np.isnan(r["steady_state_mean_abs_timing_error_s"]))]
        jitters = [float(r["steady_state_jitter_s"])
                   for r in cond_rows
                   if r["steady_state_jitter_s"] is not None
                   and not (isinstance(r["steady_state_jitter_s"], float)
                            and np.isnan(r["steady_state_jitter_s"]))]
        freq_errs = [float(r["final_frequency_error_hz"])
                     for r in cond_rows
                     if r["final_frequency_error_hz"] is not None
                     and not (isinstance(r["final_frequency_error_hz"], float)
                              and np.isnan(r["final_frequency_error_hz"]))]
        convergences = [float(r["convergence_quality"])
                        for r in cond_rows
                        if r["convergence_quality"] is not None]

        summary_rows.append({
            "follower_initial_freq_hz": freq,
            "n_trials": n,
            "success_rate": round(success_rate, 4),
            "mean_time_to_sync_s": round(float(np.mean(sync_times)), 4) if sync_times else "",
            "std_time_to_sync_s": round(float(np.std(sync_times)), 4) if len(sync_times) >= 2 else "",
            "mean_steady_state_mae_s": round(float(np.mean(maes)), 6) if maes else "",
            "std_steady_state_mae_s": round(float(np.std(maes)), 6) if len(maes) >= 2 else "",
            "mean_jitter_s": round(float(np.mean(jitters)), 6) if jitters else "",
            "mean_final_frequency_error_hz": round(float(np.mean(freq_errs)), 6) if freq_errs else "",
            "mean_convergence_quality": round(float(np.mean(convergences)), 4) if convergences else "",
        })

    sum_fields = [
        "follower_initial_freq_hz", "n_trials", "success_rate",
        "mean_time_to_sync_s", "std_time_to_sync_s",
        "mean_steady_state_mae_s", "std_steady_state_mae_s",
        "mean_jitter_s", "mean_final_frequency_error_hz",
        "mean_convergence_quality",
    ]
    sum_path = batch_dir / "summary_by_condition.csv"
    with open(sum_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    # ------------------------------------------------------------------
    # 5. Generate plots (unless --no-plots)
    # ------------------------------------------------------------------
    if not args.no_plots and trial_file_data:
        print("\nGenerating plots...")
        from firefly_sync.utils.visualization import (
            plot_timing_error_by_trial,
            plot_phase_error_by_trial,
            plot_time_to_sync_by_condition,
            plot_steady_state_error_by_condition,
            plot_success_rate_by_condition,
        )

        # Filter valid entries
        flash_data = [td for td in trial_file_data if td.get("flash_events_csv")]
        osc_data = [td for td in trial_file_data if td.get("oscillator_log_csv")]

        if flash_data:
            plot_timing_error_by_trial(
                flash_data,
                title=f"Timing Error — leader {args.leader_freq} Hz, K={args.coupling_gain}",
                save_path=str(plots_dir / "timing_error_trials.png"),
            )

        if osc_data:
            plot_phase_error_by_trial(
                osc_data,
                title=f"Phase Error — leader {args.leader_freq} Hz, K={args.coupling_gain}",
                save_path=str(plots_dir / "phase_error_trials.png"),
            )

        plot_time_to_sync_by_condition(
            summary_rows,
            title=f"Time to Sync — leader {args.leader_freq} Hz, K={args.coupling_gain}",
            save_path=str(plots_dir / "time_to_sync_by_condition.png"),
        )

        plot_steady_state_error_by_condition(
            summary_rows,
            title=f"Steady-State Error — leader {args.leader_freq} Hz, K={args.coupling_gain}",
            save_path=str(plots_dir / "steady_state_error_by_condition.png"),
        )

        plot_success_rate_by_condition(
            summary_rows,
            title=f"Sync Success Rate — leader {args.leader_freq} Hz, K={args.coupling_gain}",
            save_path=str(plots_dir / "success_rate_by_condition.png"),
        )

        print(f"  Plots saved to {plots_dir}")

    # ------------------------------------------------------------------
    # 6. Print batch summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"  Conditions:  {len(follower_freqs)}")
    print(f"  Trials:      {total}")
    print(f"  Output:      {batch_dir}")
    print()
    print("  Summary by condition:")
    for sr in summary_rows:
        freq = sr["follower_initial_freq_hz"]
        rate = sr["success_rate"]
        mae = sr.get("mean_steady_state_mae_s", "")
        print(f"    {freq:.1f} Hz → success_rate={rate:.2f}  MAE={mae}")
    print("=" * 60)

    return batch_dir


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3A-2 — Batch Kuramoto Leader-Follower Evaluation.",
    )
    parser.add_argument("--leader-freq", type=float, default=2.0)
    parser.add_argument("--follower-freqs", type=float, nargs="+",
                        default=[1.5, 1.8, 2.3])
    parser.add_argument("--coupling-gain", type=float, default=3.5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--flash-on-time", type=float, default=0.06)
    parser.add_argument("--sync-threshold-s", type=float, default=0.10)
    parser.add_argument("--sync-cycles", type=int, default=5)
    parser.add_argument("--log-dir", default="experiments/logs/step3a_batch")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--trial-prefix", default=None,
                        help="Optional prefix for batch directory name")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    run_batch(args)


if __name__ == "__main__":
    main()

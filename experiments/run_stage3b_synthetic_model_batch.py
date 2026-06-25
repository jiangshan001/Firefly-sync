#!/usr/bin/env python3
r"""Stage 3B — Synthetic Model Batch Evaluation (no hardware).

Runs PCO-I&F or EAPF closed-loop synchronisation trials with synthetic
leader flash events.  No camera, GPIO, or leader UI required.

Usage (PCO-I&F)::

    PYTHONPATH=. python experiments/run_stage3b_synthetic_model_batch.py \
        --model pco_if --duration 30 --repeats 5 \
        --leader-freqs 2.0 --follower-freqs 1.5 1.8 2.3 \
        --epsilon 0.25

Usage (EAPF)::

    PYTHONPATH=. python experiments/run_stage3b_synthetic_model_batch.py \
        --model eapf --duration 30 --repeats 5 \
        --leader-freqs 2.0 --follower-freqs 1.5 1.8 2.3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from firefly_sync.logging.metrics import (
    check_flash_synchronisation,
    compute_flash_timing_metrics,
    pair_flash_events,
)


# ======================================================================
# Output helpers
# ======================================================================

def _make_output_dir(root: str, prefix: str | None, model: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pfx = f"{prefix}_" if prefix else ""
    name = f"{ts}_{pfx}{model}_synthetic_batch"
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ======================================================================
# Failure classification
# ======================================================================

def _classify_failure(
    metrics: dict,
    leader_flash_count: int,
    follower_flash_count: int,
    expected_leader: float,
    leader_freq_hz: float,
    follower_initial_freq_hz: float,
) -> str:
    """Return a diagnostic label for a trial."""
    if metrics["synchronization_success"]:
        return "success"
    if follower_flash_count < 3:
        return "too_few_follower_flashes"
    if follower_flash_count > expected_leader * 2.5:
        return "too_many_follower_flashes"
    mae = metrics.get("steady_state_mean_abs_timing_error_s", 999)
    if mae is not None and not (isinstance(mae, float) and np.isnan(mae)) and float(mae) > 0.15:
        return "steady_offset_above_threshold"
    if metrics.get("final_frequency_error_hz", 999) > 0.5:
        return "frequency_not_converged"
    return "no_lock_detected"


# ======================================================================
# Synthetic leader generator
# ======================================================================

def _generate_leader_flash_times(
    duration_s: float, leader_freq_hz: float,
    jitter_std_s: float = 0.0,
    missed_prob: float = 0.0,
    false_positive_rate_hz: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Generate synthetic leader flash timestamps.

    Returns a sorted list of timestamps in seconds.
    """
    if rng is None:
        rng = np.random.default_rng()
    if leader_freq_hz <= 0:
        return []
    period = 1.0 / leader_freq_hz
    n = int(duration_s / period) + 1
    base = [i * period for i in range(n)]
    # Jitter
    if jitter_std_s > 0:
        base = [t + rng.normal(0, jitter_std_s) for t in base]
    # Missed detections
    if missed_prob > 0:
        base = [t for t in base if rng.random() > missed_prob]
    # False positives: insert extra events
    if false_positive_rate_hz > 0:
        extras: list[float] = []
        t = 0.0
        while t < duration_s:
            t += rng.exponential(1.0 / false_positive_rate_hz)
            if t < duration_s:
                extras.append(t)
        base = sorted(base + extras)
    return base


# ======================================================================
# Single trial runner
# ======================================================================

def _run_synthetic_trial(
    model_name: str,
    model_kwargs: dict,
    leader_flash_times: list[float],
    leader_freq_hz: float,
    follower_initial_freq_hz: float,
    duration_s: float,
    dt: float,
    sync_threshold_s: float,
    sync_cycles: int,
    trial_id: str,
    out_dir: Path,
) -> dict:
    """Run one synthetic closed-loop trial.  Returns metrics dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- import model --
    if model_name == "pco_if":
        from firefly_sync.core.pco_integrate_fire import (
            PulseCoupledIFConfig,
            PulseCoupledIntegrateFireOscillator,
        )
        config = PulseCoupledIFConfig(
            natural_frequency_hz=follower_initial_freq_hz,
            epsilon=model_kwargs.get("epsilon", 0.25),
            refractory_period_s=model_kwargs.get("refractory_period_s", 0.05),
            coupling_mode=model_kwargs.get("pco_coupling_mode", "proportional_gap"),
            state_curve_beta=model_kwargs.get("pco_state_curve_beta", 3.0),
        )
        oscillator = PulseCoupledIntegrateFireOscillator(config)
    elif model_name == "eapf":
        from firefly_sync.core.event_based_phase_lock import (
            EventBasedPhaseLockConfig,
            EventBasedPhaseLockOscillator,
        )
        config = EventBasedPhaseLockConfig(
            natural_frequency_hz=follower_initial_freq_hz,
            phase_gain=model_kwargs.get("phase_gain", 0.2),
            frequency_gain=model_kwargs.get("frequency_gain", 0.05),
            frequency_min_hz=model_kwargs.get("frequency_min_hz", 0.5),
            frequency_max_hz=model_kwargs.get("frequency_max_hz", 4.0),
            leader_period_window=model_kwargs.get("leader_period_window", 6),
        )
        oscillator = EventBasedPhaseLockOscillator(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # -- metadata --
    metadata = {
        "trial_id": trial_id, "model_name": model_name,
        "leader_freq_hz": leader_freq_hz,
        "follower_initial_freq_hz": follower_initial_freq_hz,
        "duration_s": duration_s, "dt": dt,
        "sync_threshold_s": sync_threshold_s, "sync_cycles": sync_cycles,
        "timestamp": datetime.now().isoformat(),
        **{f"model_{k}": v for k, v in model_kwargs.items()},
    }
    _save_json(out_dir / "metadata.json", metadata)

    # -- run simulation --
    t = 0.0
    leader_idx = 0
    follower_flash_times: list[float] = []
    detected_leader_times: list[float] = []
    oscillator_log: list[dict] = []
    flash_events: list[dict] = []
    sync_achieved = False

    while t < duration_s:
        # Detect leader flash events at this timestep
        leader_event = False
        while leader_idx < len(leader_flash_times) and leader_flash_times[leader_idx] <= t + dt * 0.5:
            lf = leader_flash_times[leader_idx]
            detected_leader_times.append(lf)
            leader_event = True
            leader_idx += 1
            flash_events.append({
                "t_s": round(lf, 6), "event_type": "leader_flash",
                "source": "synthetic_leader",
            })

        # Step oscillator
        result = oscillator.step(dt_s=dt, leader_flash_event=leader_event, t_s=t)

        # Log oscillator state
        log_entry: dict[str, Any] = {"t_s": round(t, 6), "model_name": model_name}

        if model_name == "pco_if":
            log_entry.update({
                "phase": result["phase"],
                "leader_flash_event": 1 if leader_event else 0,
                "follower_flash_event": 1 if result["follower_flash_event"] else 0,
                "refractory_active": 1 if result["refractory_active"] else 0,
                "leader_flash_event_used": 1 if result["leader_flash_event_used"] else 0,
                "phase_before_coupling": result["phase_before_coupling"],
                "phase_after_coupling": result["phase_after_coupling"],
            })
        else:  # eapf
            log_entry.update({
                "phase_rad": result["phase_rad"],
                "frequency_hz": result["frequency_hz"],
                "omega_rad_s": result["omega_rad_s"],
                "phase_error_rad": result["phase_error_rad"],
                "leader_period_estimate_s": result["leader_period_estimate_s"],
                "leader_flash_event": 1 if leader_event else 0,
                "follower_flash_event": 1 if result["follower_flash_event"] else 0,
                "leader_flash_event_used": 1 if result["leader_flash_event_used"] else 0,
                "follower_frequency_estimate_hz": result["frequency_hz"],
            })

        oscillator_log.append(log_entry)

        # Follower flash
        if result["follower_flash_event"]:
            follower_flash_times.append(t)
            flash_events.append({
                "t_s": round(t, 6), "event_type": "follower_flash",
                "source": "model_follower",
            })

            # Check sync
            sc = check_flash_synchronisation(
                detected_leader_times, follower_flash_times,
                sync_threshold_s, sync_cycles,
            )
            if sc["synchronization_success"] and not sync_achieved:
                sync_achieved = True

        t += dt

    # -- metrics --
    # For PCO-I&F, estimate final follower frequency from last 10 s of flashes
    if model_name == "pco_if":
        recent_f = [ft for ft in follower_flash_times if ft > duration_s - 10.0]
        if len(recent_f) >= 2:
            intervals = [recent_f[i + 1] - recent_f[i] for i in range(len(recent_f) - 1)]
            final_freq = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0.0
        else:
            final_freq = float("nan")
    else:
        final_freq = oscillator.frequency_hz if hasattr(oscillator, 'frequency_hz') else float("nan")

    metrics = compute_flash_timing_metrics(
        leader_times=detected_leader_times,
        follower_times=follower_flash_times,
        sync_threshold_s=sync_threshold_s, sync_cycles=sync_cycles,
        detection_success_rate=1.0, false_positive_rate=0.0,
    )

    # Additional fields
    expected = duration_s * leader_freq_hz
    detected = len(detected_leader_times)
    metrics["final_frequency_error_hz"] = abs(final_freq - leader_freq_hz) if not np.isnan(final_freq) else float("nan")
    metrics["leader_flash_count"] = detected
    metrics["follower_flash_count"] = len(follower_flash_times)
    metrics["expected_leader_flash_count"] = round(expected, 1)
    metrics["leader_flash_count_ratio"] = round(detected / expected, 3) if expected > 0 else 0.0

    # -- save logs --
    if model_name == "pco_if":
        osc_fields = ["t_s", "model_name", "phase", "leader_flash_event",
                       "follower_flash_event", "refractory_active",
                       "leader_flash_event_used", "phase_before_coupling",
                       "phase_after_coupling"]
    else:
        osc_fields = ["t_s", "model_name", "phase_rad", "frequency_hz",
                       "omega_rad_s", "phase_error_rad", "leader_period_estimate_s",
                       "leader_flash_event", "follower_flash_event",
                       "leader_flash_event_used", "follower_frequency_estimate_hz"]

    _save_csv(out_dir / "oscillator_log.csv", oscillator_log, osc_fields)
    _save_csv(out_dir / "flash_events.csv", flash_events,
              ["t_s", "event_type", "source"])
    _save_json(out_dir / "metrics_summary.json", metrics)

    return metrics


# ======================================================================
# Batch runner
# ======================================================================

def run_synthetic_batch(args: argparse.Namespace) -> Path:
    batch_dir = _make_output_dir(args.log_dir, args.trial_prefix, args.model)
    trials_dir = batch_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Model kwargs
    if args.model == "pco_if":
        model_kwargs = {
            "epsilon": args.epsilon,
            "refractory_period_s": args.refractory_period_s,
            "pco_coupling_mode": args.pco_coupling_mode,
            "pco_state_curve_beta": args.pco_state_curve_beta,
        }
    else:
        model_kwargs = {
            "phase_gain": args.phase_gain,
            "frequency_gain": args.frequency_gain,
            "frequency_min_hz": args.frequency_min_hz,
            "frequency_max_hz": args.frequency_max_hz,
            "leader_period_window": args.leader_period_window,
        }

    rng = np.random.default_rng(args.random_seed)

    # Batch metadata
    batch_meta = {
        "model_name": args.model,
        "leader_freqs_hz": args.leader_freqs,
        "follower_initial_freqs_hz": args.follower_freqs,
        "duration_s": args.duration, "dt": args.dt,
        "repeats_per_condition": args.repeats,
        "sync_threshold_s": args.sync_threshold_s,
        "sync_cycles": args.sync_cycles,
        "random_seed": args.random_seed,
        "timestamp": datetime.now().isoformat(),
        **{f"param_{k}": v for k, v in model_kwargs.items()},
    }
    _save_json(batch_dir / "batch_metadata.json", batch_meta)

    follower_freqs: list[float] = args.follower_freqs
    leader_freqs: list[float] = args.leader_freqs
    repeats: int = args.repeats

    aggregate_rows: list[dict] = []
    total = len(leader_freqs) * len(follower_freqs) * repeats
    count = 0

    for lf in leader_freqs:
        for freq in follower_freqs:
            for rep in range(1, repeats + 1):
                count += 1
                trial_id = f"L{lf:.1f}Hz_F{freq:.1f}Hz_r{rep:02d}"
                print(f"[{count}/{total}] {trial_id}")

                out_dir = trials_dir / trial_id
                out_dir.mkdir(parents=True, exist_ok=True)

                leader_times = _generate_leader_flash_times(
                    args.duration, lf,
                    jitter_std_s=args.leader_jitter_std_s,
                    missed_prob=args.missed_detection_prob,
                    false_positive_rate_hz=args.false_positive_rate_hz,
                    rng=rng,
                )

                metrics = _run_synthetic_trial(
                    model_name=args.model,
                    model_kwargs=model_kwargs,
                    leader_flash_times=leader_times,
                    leader_freq_hz=lf,
                    follower_initial_freq_hz=freq,
                    duration_s=args.duration, dt=args.dt,
                    sync_threshold_s=args.sync_threshold_s,
                    sync_cycles=args.sync_cycles,
                    trial_id=trial_id, out_dir=out_dir,
                )

                aggregate_rows.append({
                    "batch_id": batch_dir.name, "trial_id": trial_id,
                    "model_name": args.model,
                    "leader_freq_hz": lf,
                    "follower_initial_freq_hz": freq,
                    "duration_s": args.duration,
                    "synchronization_success": metrics["synchronization_success"],
                    "time_to_synchronization_s": metrics.get("time_to_synchronization_s"),
                    "steady_state_mean_abs_timing_error_s": metrics["steady_state_mean_abs_timing_error_s"],
                    "steady_state_rmse_timing_error_s": metrics["steady_state_rmse_timing_error_s"],
                    "steady_state_jitter_s": metrics["steady_state_jitter_s"],
                    "final_frequency_error_hz": metrics["final_frequency_error_hz"],
                    "convergence_quality": metrics["convergence_quality"],
                    "leader_flash_count": metrics["leader_flash_count"],
                    "follower_flash_count": metrics["follower_flash_count"],
                    "expected_leader_flash_count": metrics["expected_leader_flash_count"],
                    "leader_flash_count_ratio": metrics["leader_flash_count_ratio"],
                    "failure_label": _classify_failure(
                        metrics, metrics["leader_flash_count"],
                        metrics["follower_flash_count"],
                        metrics["expected_leader_flash_count"],
                        lf, freq,
                    ),
                    "pco_coupling_mode": model_kwargs.get("pco_coupling_mode", ""),
                    "trial_log_dir": str(out_dir),
                })

    # Aggregate CSV
    agg_fields = [
        "batch_id", "trial_id", "model_name", "leader_freq_hz",
        "follower_initial_freq_hz", "duration_s",
        "synchronization_success", "time_to_synchronization_s",
        "steady_state_mean_abs_timing_error_s", "steady_state_rmse_timing_error_s",
        "steady_state_jitter_s", "final_frequency_error_hz",
        "convergence_quality", "leader_flash_count", "follower_flash_count",
        "expected_leader_flash_count", "leader_flash_count_ratio",
        "failure_label", "pco_coupling_mode",
        "trial_log_dir",
    ]
    _save_csv(batch_dir / "aggregate_metrics.csv", aggregate_rows, agg_fields)

    # Summary by condition
    summary_rows = []
    for freq in follower_freqs:
        cond = [r for r in aggregate_rows if abs(r["follower_initial_freq_hz"] - freq) < 0.001]
        n = len(cond)
        succ = [r for r in cond if r["synchronization_success"]]
        sync_times = [float(r["time_to_synchronization_s"]) for r in succ
                      if r["time_to_synchronization_s"] is not None]
        maes = [float(r["steady_state_mean_abs_timing_error_s"]) for r in cond
                if r["steady_state_mean_abs_timing_error_s"] is not None
                and not (isinstance(r["steady_state_mean_abs_timing_error_s"], float)
                         and np.isnan(r["steady_state_mean_abs_timing_error_s"]))]
        summary_rows.append({
            "follower_initial_freq_hz": freq,
            "n_trials": n,
            "success_rate": round(len(succ) / n, 4) if n > 0 else 0.0,
            "mean_time_to_sync_s": round(float(np.mean(sync_times)), 4) if sync_times else "",
            "std_time_to_sync_s": round(float(np.std(sync_times)), 4) if len(sync_times) >= 2 else "",
            "mean_steady_state_mae_s": round(float(np.mean(maes)), 6) if maes else "",
            "std_steady_state_mae_s": round(float(np.std(maes)), 6) if len(maes) >= 2 else "",
        })

    _save_csv(batch_dir / "summary_by_condition.csv", summary_rows,
              ["follower_initial_freq_hz", "n_trials", "success_rate",
               "mean_time_to_sync_s", "std_time_to_sync_s",
               "mean_steady_state_mae_s", "std_steady_state_mae_s"])

    # Print
    print(f"\nBatch complete: {batch_dir}")
    for sr in summary_rows:
        print(f"  {sr['follower_initial_freq_hz']:.1f} Hz -> "
              f"success={sr['success_rate']:.2f}  MAE={sr.get('mean_steady_state_mae_s','')}")

    return batch_dir


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3B — Synthetic Model Batch Evaluation.",
    )
    parser.add_argument("--model", choices=["pco_if", "eapf"], required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--leader-freqs", type=float, nargs="+", default=[2.0])
    parser.add_argument("--follower-freqs", type=float, nargs="+", default=[1.5, 1.8, 2.3])
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sync-threshold-s", type=float, default=0.10)
    parser.add_argument("--sync-cycles", type=int, default=5)
    parser.add_argument("--log-dir", default="experiments/logs/stage3b_synthetic_models")
    parser.add_argument("--trial-prefix", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    # Noise
    parser.add_argument("--leader-jitter-std-s", type=float, default=0.0)
    parser.add_argument("--missed-detection-prob", type=float, default=0.0)
    parser.add_argument("--false-positive-rate-hz", type=float, default=0.0)
    # PCO-I&F params
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--fire-threshold", type=float, default=1.0)
    parser.add_argument("--refractory-period-s", type=float, default=0.05)
    parser.add_argument("--pco-coupling-mode", default="proportional_gap",
                        choices=["proportional_gap", "additive_phase", "mirollo_state"])
    parser.add_argument("--pco-state-curve-beta", type=float, default=3.0)
    # EAPF params
    parser.add_argument("--phase-gain", type=float, default=0.20)
    parser.add_argument("--frequency-gain", type=float, default=0.05)
    parser.add_argument("--frequency-min-hz", type=float, default=0.5)
    parser.add_argument("--frequency-max-hz", type=float, default=4.0)
    parser.add_argument("--leader-period-window", type=int, default=6)

    args = parser.parse_args()
    run_synthetic_batch(args)


if __name__ == "__main__":
    main()

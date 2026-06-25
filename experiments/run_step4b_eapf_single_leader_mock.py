#!/usr/bin/env python3
r"""Step 4B.1 — EAPF Consensus Single-Leader Mock Validation.

Tests whether the EventBasedConsensusPLLOscillator (designed for
multi-neighbour consensus) behaves reasonably as a fixed-leader follower.

This is NOT the final HIL comparison — it is a preliminary diagnostic
to decide whether EAPF single-leader Pi visual HIL is worth doing.

Usage::

    PYTHONPATH=. python experiments/run_step4b_eapf_single_leader_mock.py \
        --duration 60 --repeats 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from firefly_sync.core.event_based_consensus_pll import (
    EventBasedConsensusPLLOscillator,
)

# ======================================================================
# Locked EAPF Consensus parameters (Stage 4A)
# ======================================================================

LOCKED = {
    "phase_gain": 0.02,
    "frequency_gain": 0.02,
    "phase_error_filter_alpha": 0.2,
    "frequency_error_filter_alpha": 0.2,
    "max_phase_step_rad": 0.2,
    "max_frequency_step_hz": 0.05,
    "frequency_min_hz": 0.5,
    "frequency_max_hz": 4.0,
}

FOLLOWER_FREQS = [1.5, 1.8, 2.3]


# ======================================================================
# Helpers
# ======================================================================

def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _leader_flash_times(duration: float, freq: float = 2.0) -> list[float]:
    period = 1.0 / freq if freq > 0 else 1.0
    return [i * period for i in range(int(duration / period) + 1)]


# ======================================================================
# Single trial
# ======================================================================

def _run_trial(
    follower_freq_hz: float,
    leader_freq_hz: float,
    duration_s: float,
    dt: float,
    params: dict,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run one single-leader EAPF consensus trial."""
    leader_times = _leader_flash_times(duration_s, leader_freq_hz)
    osc = EventBasedConsensusPLLOscillator()
    # Override config with locked params
    osc.config.phase_gain = params["phase_gain"]
    osc.config.frequency_gain = params["frequency_gain"]
    osc.config.phase_error_filter_alpha = params["phase_error_filter_alpha"]
    osc.config.frequency_error_filter_alpha = params["frequency_error_filter_alpha"]
    osc.config.max_phase_step_rad = params["max_phase_step_rad"]
    osc.config.max_frequency_step_hz = params["max_frequency_step_hz"]
    osc.config.frequency_min_hz = params["frequency_min_hz"]
    osc.config.frequency_max_hz = params["frequency_max_hz"]
    osc._frequency_hz = follower_freq_hz
    osc._omega_rad_s = 2.0 * np.pi * follower_freq_hz

    t = 0.0
    leader_idx = 0
    follower_flash_times: list[float] = []
    timing_errors: list[tuple[float, float]] = []  # (t, error_s)
    osc_log: list[dict] = []
    flash_events: list[dict] = []
    sync_achieved = False

    while t < duration_s:
        # Check for leader flash
        leader_event = False
        while leader_idx < len(leader_times) and leader_times[leader_idx] <= t + dt * 0.5:
            leader_event = True
            lf = leader_times[leader_idx]
            flash_events.append({"t_s": round(lf, 6), "event_type": "leader_flash"})
            leader_idx += 1

        # Step oscillator
        nids = [0] if leader_event else []
        result = osc.step(dt_s=dt, t_s=t, neighbour_flash_ids=nids)

        osc_log.append({
            "t_s": round(t, 6),
            "phase_rad": result["phase_rad"],
            "frequency_hz": result["frequency_hz"],
            "phase_error_rad": result["phase_error_rad"],
            "freq_error_hz": result.get("freq_error_hz", 0),
            "follower_flash": 1 if result["follower_flash_event"] else 0,
        })

        if result["follower_flash_event"]:
            follower_flash_times.append(t)
            flash_events.append({"t_s": round(t, 6), "event_type": "follower_flash"})
            # Nearest-leader timing error
            if leader_times:
                nearest = min(leader_times, key=lambda lt: abs(lt - t))
                err = t - nearest
                timing_errors.append((t, err))

            # Check sync: last 5 errors all < 0.10
            if len(timing_errors) >= 5:
                last5 = [abs(e[1]) for e in timing_errors[-5:]]
                if all(e < 0.10 for e in last5) and not sync_achieved:
                    sync_achieved = True

        t += dt

    # Metrics
    sync_success = sync_achieved
    time_to_sync = None
    if sync_achieved and len(timing_errors) >= 5:
        # Find first sustained 5-cycle block
        for i in range(len(timing_errors) - 4):
            if all(abs(timing_errors[j][1]) < 0.10 for j in range(i, i + 5)):
                time_to_sync = timing_errors[i + 4][0]
                break

    # Steady-state (final 10 s, or last half)
    window = min(10.0, duration_s * 0.3)
    cutoff = max(0, (follower_flash_times[-1] if follower_flash_times else duration_s) - window)
    steady_errs = [abs(e[1]) for e in timing_errors if e[0] >= cutoff]
    steady_mae = float(np.mean(steady_errs)) if steady_errs else 0.0
    steady_jitter = float(np.std(steady_errs)) if len(steady_errs) >= 2 else 0.0

    # Final frequency
    final_freq = osc.frequency_hz

    # Flash counts
    leader_count = len(leader_times)
    follower_count = len(follower_flash_times)
    expected_follower = leader_count
    fcr = follower_count / leader_count if leader_count > 0 else 0.0

    return {
        "sync_success": sync_success,
        "time_to_sync_s": round(time_to_sync, 4) if time_to_sync else None,
        "steady_state_mae_s": round(steady_mae, 6),
        "steady_state_jitter_s": round(steady_jitter, 6),
        "final_frequency_hz": round(final_freq, 6),
        "leader_flash_count": leader_count,
        "follower_flash_count": follower_count,
        "flash_count_ratio": round(fcr, 4),
        "extra_flash_count": follower_count - expected_follower,
        "follower_flash_times": follower_flash_times,
        "timing_errors": timing_errors,
        "osc_log": osc_log,
        "flash_events": flash_events,
        "leader_times": leader_times,
    }


# ======================================================================
# Figures
# ======================================================================

def _generate_figures(agg: list[dict], rep_trials: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    freqs = sorted(set(r["follower_freq_hz"] for r in agg))
    df = pd.DataFrame(agg)

    def _bar(fname, title, ylabel, col, ylim=None):
        fig, ax = plt.subplots(figsize=(7, 4))
        grp = df.groupby("follower_freq_hz")[col].mean()
        vals = [grp.get(f, 0) for f in freqs]
        ax.bar([f"{f} Hz" for f in freqs], vals, color="steelblue", edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(i, v + (0.01 if max(vals) < 0.1 else max(vals)*0.02),
                    f"{v:.4f}" if v < 1 else f"{v:.0f}", ha="center", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        fig.savefig(fig_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _bar("success_rate_by_condition.png", "Sync Success Rate",
         "Success Rate", "sync_success", ylim=(0, 1.1))
    _bar("flash_count_ratio_by_condition.png", "Flash Count Ratio (Follower/Leader)",
         "FCR", "flash_count_ratio")
    _bar("steady_state_mae_by_condition.png", "Steady-State MAE (s)",
         "MAE (s)", "steady_state_mae_s")
    _bar("time_to_sync_by_condition.png", "Time to Sync (s)",
         "Time (s)", "time_to_sync_s")

    # Timing error timeseries (one rep per freq)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(freqs)))
    for i, freq in enumerate(freqs):
        key = f"{freq:.1f}"
        if key in rep_trials and rep_trials[key]["timing_errors"]:
            te = rep_trials[key]["timing_errors"]
            ts = [e[0] for e in te]
            es = [e[1] * 1000 for e in te]  # ms
            ax.plot(ts, es, color=colors[i], linewidth=0.8, label=f"{freq} Hz")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=-100, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Timing Error (ms)")
    ax.set_title("Representative Timing Error — EAPF Consensus, Single Leader")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "representative_timing_error_timeseries.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Flash raster (first available trial)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    first_key = list(rep_trials.keys())[0] if rep_trials else None
    if first_key:
        tr = rep_trials[first_key]
        if tr["leader_times"]:
            ax.eventplot(tr["leader_times"], lineoffsets=1, colors="steelblue",
                         linewidths=0.8, label="Leader")
        if tr["follower_flash_times"]:
            ax.eventplot(tr["follower_flash_times"], lineoffsets=0, colors="darkorange",
                         linewidths=0.8, label="Follower")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Follower", "Leader"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Representative Flash Raster — EAPF Consensus, Follower {first_key or '?'} Hz")
    ax.legend(loc="upper right")
    fig.savefig(fig_dir / "representative_flash_raster.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4B.1 — EAPF Consensus Single-Leader Mock Validation.",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--leader-freq", type=float, default=2.0)
    parser.add_argument("--log-dir", default="experiments/logs/step4b_eapf_single_leader_mock")
    parser.add_argument("--seed", type=int, default=42)
    # Optional diagnostic gain sweep
    parser.add_argument("--sweep-gains", action="store_true",
                        help="Run a small gain sweep instead of locked params")
    args = parser.parse_args()

    out_dir = Path(args.log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eapf_single_leader_mock"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    if args.sweep_gains:
        gain_pairs = [(0.02, 0.02), (0.05, 0.02), (0.05, 0.05), (0.10, 0.02), (0.10, 0.05)]
    else:
        gain_pairs = [(LOCKED["phase_gain"], LOCKED["frequency_gain"])]

    aggregate_rows: list[dict] = []
    rep_trials: dict[str, dict] = {}  # first rep per freq for figures

    for pg, fg in gain_pairs:
        params = {**LOCKED, "phase_gain": pg, "frequency_gain": fg}
        label = f"pg{pg}_fg{fg}"
        for freq in FOLLOWER_FREQS:
            for rep in range(args.repeats):
                trial = _run_trial(freq, args.leader_freq, args.duration, args.dt, params, rng)
                row = {
                    "follower_freq_hz": freq,
                    "rep": rep + 1,
                    "phase_gain": pg,
                    "frequency_gain": fg,
                    "sync_success": trial["sync_success"],
                    "time_to_sync_s": trial["time_to_sync_s"],
                    "steady_state_mae_s": trial["steady_state_mae_s"],
                    "steady_state_jitter_s": trial["steady_state_jitter_s"],
                    "final_frequency_hz": trial["final_frequency_hz"],
                    "flash_count_ratio": trial["flash_count_ratio"],
                    "extra_flash_count": trial["extra_flash_count"],
                }
                aggregate_rows.append(row)
                print(f"  [{label}] f={freq}Hz r{rep+1}: "
                      f"sync={trial['sync_success']} "
                      f"tts={trial['time_to_sync_s']} "
                      f"mae={trial['steady_state_mae_s']:.5f} "
                      f"freq={trial['final_frequency_hz']:.4f} "
                      f"FCR={trial['flash_count_ratio']:.3f}")
                if rep == 0:
                    rep_trials[f"{freq:.1f}"] = trial

    # Save CSVs
    agg_fields = list(aggregate_rows[0].keys())
    _save_csv(out_dir / "aggregate_metrics.csv", aggregate_rows, agg_fields)

    # Summary
    df = pd.DataFrame(aggregate_rows)
    summary_rows = []
    for freq in FOLLOWER_FREQS:
        d = df[df["follower_freq_hz"] == freq]
        summary_rows.append({
            "follower_freq_hz": freq,
            "n_trials": len(d),
            "sync_success_rate": round(d["sync_success"].mean(), 4),
            "mean_time_to_sync_s": round(d["time_to_sync_s"].dropna().astype(float).mean(), 4) if d["sync_success"].sum() > 0 else "",
            "mean_steady_state_mae_s": round(d["steady_state_mae_s"].mean(), 6),
            "mean_final_frequency_hz": round(d["final_frequency_hz"].mean(), 6),
            "mean_flash_count_ratio": round(d["flash_count_ratio"].mean(), 4),
        })
    _save_csv(out_dir / "summary_by_condition.csv", summary_rows,
              list(summary_rows[0].keys()))

    # Metadata
    _save_json(out_dir / "metadata.json", {
        "model": "eapf_consensus",
        "mode": "single_leader_mock",
        "leader_freq_hz": args.leader_freq,
        "duration_s": args.duration,
        "dt": args.dt,
        "repeats": args.repeats,
        "locked_params": LOCKED,
        "sweep_gains": args.sweep_gains,
        "timestamp": datetime.now().isoformat(),
    })

    # Figures
    _generate_figures(aggregate_rows, rep_trials, out_dir)

    # Print summary
    print(f"\nOutput: {out_dir}")
    for sr in summary_rows:
        print(f"  {sr['follower_freq_hz']} Hz: "
              f"sync={sr['sync_success_rate']:.2f}  "
              f"MAE={sr['mean_steady_state_mae_s']:.6f}  "
              f"FCR={sr['mean_flash_count_ratio']:.4f}  "
              f"freq={sr['mean_final_frequency_hz']:.4f}")


if __name__ == "__main__":
    main()

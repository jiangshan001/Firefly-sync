#!/usr/bin/env python3
r"""Stage 3B — Parameter Sweep for PCO-I&F and EAPF Synthetic Models.

Runs a grid of parameter values and recommends baseline parameters.

Usage::

    PYTHONPATH=. python experiments/sweep_stage3b_synthetic_models.py \
        --duration 30 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from experiments.run_stage3b_synthetic_model_batch import (
    _generate_leader_flash_times,
    _run_synthetic_trial,
    _save_json,
    _save_csv,
    _make_output_dir,
)


# ======================================================================
# Sweep definitions
# ======================================================================

PCO_IF_SWEEP: dict[str, list[Any]] = {
    "pco_coupling_mode": ["proportional_gap", "additive_phase", "mirollo_state"],
    "epsilon": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60],
    "refractory_period_s": [0.05, 0.10],
    "pco_state_curve_beta": [3.0],
}

EAPF_SWEEP: dict[str, list[Any]] = {
    "phase_gain": [0.05, 0.10, 0.20, 0.30, 0.40],
    "frequency_gain": [0.01, 0.03, 0.05, 0.10, 0.15],
}


# ======================================================================
# Recommendation logic
# ======================================================================

def _score_params(rows: list[dict]) -> float:
    """Higher score = better parameters.
    Weight: success rate (0.5) + low MAE (0.25) + low time-to-sync (0.25).
    """
    succ = np.mean([float(r["success_rate"]) for r in rows])
    maes = [float(r["mean_steady_state_mae_s"]) for r in rows
            if r.get("mean_steady_state_mae_s") not in ("", None)]
    tts = [float(r["mean_time_to_sync_s"]) for r in rows
           if r.get("mean_time_to_sync_s") not in ("", None)]
    mae_score = 1.0 - min(1.0, (np.mean(maes) / 0.5)) if maes else 0.0
    tts_score = 1.0 - min(1.0, (np.mean(tts) / 30.0)) if tts else 0.0
    return 0.5 * succ + 0.25 * mae_score + 0.25 * tts_score


def _recommend(results: list[dict], model: str) -> dict:
    """Pick the best parameter set from sweep results."""
    # Group by param key
    grouped: dict[tuple, list[dict]] = {}
    for r in results:
        key = tuple(sorted((k, v) for k, v in r.items()
                           if k not in ("success_rate", "mean_time_to_sync_s",
                                        "mean_steady_state_mae_s", "follower_initial_freq_hz",
                                        "n_trials")))
        grouped.setdefault(key, []).append(r)

    best_key = max(grouped, key=lambda k: _score_params(grouped[k]))
    return dict(best_key)


# ======================================================================
# Main sweep
# ======================================================================

def run_sweep(args: argparse.Namespace) -> Path:
    sweep_dir = _make_output_dir(args.log_dir, "sweep", "parameter_sweep")

    leader_freqs = [2.0]
    follower_freqs = [1.5, 1.8, 2.3]
    rng = np.random.default_rng(args.random_seed)

    all_results: dict[str, list[dict]] = {"pco_if": [], "eapf": []}

    # --- PCO-I&F sweep ---
    if not args.skip_pco_if:
        print("=" * 60)
        print("PCO-I&F sweep")
        print("=" * 60)
        modes = args.pco_modes if args.pco_modes else PCO_IF_SWEEP["pco_coupling_mode"]
        epsilons = args.pco_epsilons if args.pco_epsilons else PCO_IF_SWEEP["epsilon"]
        refracs = args.pco_refractories if args.pco_refractories else PCO_IF_SWEEP["refractory_period_s"]
        betas = args.pco_betas if args.pco_betas else PCO_IF_SWEEP["pco_state_curve_beta"]

        for mode in modes:
            for eps in epsilons:
                for refr in refracs:
                    for beta in betas:
                        model_kwargs = {
                            "epsilon": eps, "refractory_period_s": refr,
                            "pco_coupling_mode": mode,
                            "pco_state_curve_beta": beta,
                        }
                        label = f"pco_if_{mode}_eps{eps}_refr{refr}"
                        print(f"\n[{label}]")

                        cond_rows = []
                        for freq in follower_freqs:
                            succs = 0
                            maes: list[float] = []
                            tts_list: list[float] = []
                            for rep in range(args.repeats):
                                lt = _generate_leader_flash_times(args.duration, 2.0)
                                metrics = _run_synthetic_trial(
                                    model_name="pco_if", model_kwargs=model_kwargs,
                                    leader_flash_times=lt, leader_freq_hz=2.0,
                                    follower_initial_freq_hz=freq,
                                    duration_s=args.duration, dt=args.dt,
                                    sync_threshold_s=args.sync_threshold_s,
                                    sync_cycles=args.sync_cycles,
                                    trial_id=f"{label}_f{freq}_{rep}",
                                    out_dir=sweep_dir / "trials",
                                )
                                (sweep_dir / "trials").mkdir(parents=True, exist_ok=True)
                                if metrics["synchronization_success"]:
                                    succs += 1
                                if metrics["time_to_synchronization_s"] is not None:
                                    tts_list.append(float(metrics["time_to_synchronization_s"]))
                                mae = metrics["steady_state_mean_abs_timing_error_s"]
                                if mae is not None and not (isinstance(mae, float) and np.isnan(mae)):
                                    maes.append(float(mae))

                            cond_rows.append({
                                "follower_initial_freq_hz": freq,
                                "n_trials": args.repeats,
                                "success_rate": succs / args.repeats,
                                "mean_time_to_sync_s": round(float(np.mean(tts_list)), 4) if tts_list else "",
                                "mean_steady_state_mae_s": round(float(np.mean(maes)), 6) if maes else "",
                            })

                        for cr in cond_rows:
                            all_results["pco_if"].append({**model_kwargs, **cr})
                        rates_str = ", ".join(f"{cr['success_rate']:.2f}" for cr in cond_rows)
                        print(f"  Success rates: [{rates_str}]")

    # --- EAPF sweep ---
    if not args.skip_eapf:
        print("\n" + "=" * 60)
        print("EAPF sweep")
        print("=" * 60)
        pgs = args.eapf_phase_gains if args.eapf_phase_gains else EAPF_SWEEP["phase_gain"]
        fgs = args.eapf_freq_gains if args.eapf_freq_gains else EAPF_SWEEP["frequency_gain"]

        for pg in pgs:
            for fg in fgs:
                model_kwargs = {"phase_gain": pg, "frequency_gain": fg,
                                "frequency_min_hz": 0.5, "frequency_max_hz": 4.0,
                                "leader_period_window": 6}
                label = f"eapf_pg{pg}_fg{fg}"
                print(f"\n[{label}]")

                cond_rows = []
                for freq in follower_freqs:
                    succs = 0
                    maes: list[float] = []
                    tts_list: list[float] = []
                    for rep in range(args.repeats):
                        lt = _generate_leader_flash_times(args.duration, 2.0)
                        metrics = _run_synthetic_trial(
                            model_name="eapf", model_kwargs=model_kwargs,
                            leader_flash_times=lt, leader_freq_hz=2.0,
                            follower_initial_freq_hz=freq,
                            duration_s=args.duration, dt=args.dt,
                            sync_threshold_s=args.sync_threshold_s,
                            sync_cycles=args.sync_cycles,
                            trial_id=f"{label}_f{freq}_{rep}",
                            out_dir=sweep_dir / "trials",
                        )
                        (sweep_dir / "trials").mkdir(parents=True, exist_ok=True)
                        if metrics["synchronization_success"]:
                            succs += 1
                        if metrics["time_to_synchronization_s"] is not None:
                            tts_list.append(float(metrics["time_to_synchronization_s"]))
                        mae = metrics["steady_state_mean_abs_timing_error_s"]
                        if mae is not None and not (isinstance(mae, float) and np.isnan(mae)):
                            maes.append(float(mae))

                    cond_rows.append({
                        "follower_initial_freq_hz": freq,
                        "n_trials": args.repeats,
                        "success_rate": succs / args.repeats,
                        "mean_time_to_sync_s": round(float(np.mean(tts_list)), 4) if tts_list else "",
                        "mean_steady_state_mae_s": round(float(np.mean(maes)), 6) if maes else "",
                    })

                for cr in cond_rows:
                    all_results["eapf"].append({**model_kwargs, **cr})
                rates_str = ", ".join(f"{cr['success_rate']:.2f}" for cr in cond_rows)
                print(f"  Success rates: [{rates_str}]")

    # --- Save sweep CSVs ---
    for model in ["pco_if", "eapf"]:
        if all_results[model]:
            _save_csv(sweep_dir / f"{model}_sweep_results.csv", all_results[model],
                      list(all_results[model][0].keys()))

    # --- Recommendations ---
    recommendations = {}
    for model in ["pco_if", "eapf"]:
        if all_results[model]:
            rec = _recommend(all_results[model], model)
            recommendations[model] = rec

    _save_json(sweep_dir / "recommended_parameters.json", recommendations)
    print(f"\nRecommendations: {json.dumps(recommendations, indent=2)}")
    print(f"\nSweep complete: {sweep_dir}")
    return sweep_dir


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3B — Parameter Sweep for Synthetic Models.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sync-threshold-s", type=float, default=0.10)
    parser.add_argument("--sync-cycles", type=int, default=5)
    parser.add_argument("--log-dir", default="experiments/logs/stage3b_synthetic_models")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--skip-pco-if", action="store_true")
    parser.add_argument("--skip-eapf", action="store_true")
    # Override sweep ranges
    parser.add_argument("--pco-modes", nargs="*", default=None)
    parser.add_argument("--pco-epsilons", type=float, nargs="*", default=None)
    parser.add_argument("--pco-refractories", type=float, nargs="*", default=None)
    parser.add_argument("--pco-betas", type=float, nargs="*", default=None)
    parser.add_argument("--eapf-phase-gains", type=float, nargs="*", default=None)
    parser.add_argument("--eapf-freq-gains", type=float, nargs="*", default=None)

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

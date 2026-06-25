#!/usr/bin/env python3
r"""Stage 4A — Diagnostic Model Sweep for Multi-Neighbour Synchronisation.

Runs Kuramoto, PCO (simple + adaptive PRC), EAPF (tracker + consensus)
across topologies and frequency sets, producing comparative metrics.

Usage::

    PYTHONPATH=. python experiments/sweep_stage4a_multi_neighbour_models.py \
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

import numpy as np
import pandas as pd

# Reuse trial runner from Stage 4A
from experiments.run_stage4a_multi_neighbour_simulation import _run_trial, _save_csv, _save_json

TOPOLOGIES = ["all_to_all", "chain", "directed_chain"]

FREQ_SETS = {
    "identical": [2.0, 2.0, 2.0],
    "near_identical": [1.9, 2.0, 2.1],
    "moderate_heterogeneity": [1.8, 2.0, 2.2],
    "strong_heterogeneity": [1.5, 2.0, 2.3],
}

# ======================================================================
# Model variant definitions
# ======================================================================

VARIANTS: dict[str, list[dict]] = {
    "kuramoto": [{"variant": "baseline", "kuramoto_k": 3.5}],
    "pco_simple": [{"variant": "simple", "pco_coupling_mode": "mirollo_state",
                     "pco_epsilon": 0.25, "pco_refractory_period_s": 0.05,
                     "pco_state_curve_beta": 3.0}],
    "pco_adaptive_prc": [
        {"variant": f"prc_biphasic_eps{eps}_adapt{adapt}_gain{gain}",
         "pco_coupling_mode": "biphasic_sine", "pco_epsilon": eps,
         "pco_enable_phase_delay": True, "pco_enable_frequency_adaptation": adapt,
         "pco_frequency_adaptation_gain": gain,
         "pco_max_phase_correction": 0.20,
         "pco_min_inter_flash_interval_s": 0.20,
         "pco_post_flash_lockout_s": 0.10}
        for eps in [0.10, 0.20]
        for adapt in [True, False]
        for gain in ([0.01, 0.03] if adapt else [0.0])
    ],
    "eapf_tracker": [{"variant": "tracker", "eapf_phase_gain": 0.3,
                       "eapf_frequency_gain": 0.1}],
    "eapf_consensus": [
        {"variant": f"cons_pg{pg}_fg{fg}",
         "eapf_phase_gain": pg, "eapf_frequency_gain": fg}
        for pg in [0.05, 0.10]
        for fg in [0.02, 0.05]
    ],
}

TOP_FREQ_SETS = ["near_identical", "strong_heterogeneity"]


# ======================================================================
# Main sweep
# ======================================================================

def run_sweep(args: argparse.Namespace) -> Path:
    out_dir = Path(args.log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    # Use reduced frequency sets for speed unless --full is given
    freq_sets_to_run = list(FREQ_SETS.keys()) if args.full else TOP_FREQ_SETS

    aggregate_rows: list[dict] = []
    total = sum(len(FREQ_SETS[f]) if f in freq_sets_to_run else 0
                for f in freq_sets_to_run)  # rough
    count = 0

    for model_name, variant_list in VARIANTS.items():
        for vcfg in variant_list:
            for topo in TOPOLOGIES:
                for fset_name in freq_sets_to_run:
                    freqs = FREQ_SETS[fset_name]
                    for rep in range(1, args.repeats + 1):
                        count += 1
                        vid = vcfg["variant"]
                        tid = f"{model_name}_{vid}_{topo}_{fset_name}_r{rep:02d}"
                        print(f"[{count}] {tid}")

                        # Build trial args
                        trial_args = argparse.Namespace()
                        trial_args.kuramoto_k = vcfg.get("kuramoto_k", 3.5)
                        trial_args.pco_coupling_mode = vcfg.get("pco_coupling_mode", "mirollo_state")
                        trial_args.pco_epsilon = vcfg.get("pco_epsilon", 0.25)
                        trial_args.pco_refractory_period_s = vcfg.get("pco_refractory_period_s", 0.05)
                        trial_args.pco_state_curve_beta = vcfg.get("pco_state_curve_beta", 3.0)
                        trial_args.pco_enable_phase_delay = vcfg.get("pco_enable_phase_delay", False)
                        trial_args.pco_enable_frequency_adaptation = vcfg.get("pco_enable_frequency_adaptation", False)
                        trial_args.pco_frequency_adaptation_gain = vcfg.get("pco_frequency_adaptation_gain", 0.0)
                        trial_args.pco_max_phase_correction = vcfg.get("pco_max_phase_correction", 0.40)
                        trial_args.pco_min_inter_flash_interval_s = vcfg.get("pco_min_inter_flash_interval_s", 0.0)
                        trial_args.pco_post_flash_lockout_s = vcfg.get("pco_post_flash_lockout_s", 0.0)
                        trial_args.eapf_phase_gain = vcfg.get("eapf_phase_gain", 0.3)
                        trial_args.eapf_frequency_gain = vcfg.get("eapf_frequency_gain", 0.1)
                        trial_args.eapf_frequency_min_hz = 0.5
                        trial_args.eapf_frequency_max_hz = 4.0
                        trial_args.event_delay_s = 0.0
                        trial_args.missed_event_prob = 0.0

                        trial = _run_trial(
                            model=_model_key(model_name),
                            topology_type=topo,
                            initial_frequencies=freqs,
                            duration_s=args.duration, dt=args.dt,
                            args=trial_args, rng=rng,
                        )
                        m = trial["metrics"]
                        row = {
                            "model": model_name, "variant": vid,
                            "topology": topo, "freq_set": fset_name,
                            "trial_id": tid,
                            "zero_lag_group_sync_success": m["zero_lag_group_sync_success"],
                            "phase_locked_group_success": m["phase_locked_group_success"],
                            "phase_sync_success": m["phase_sync_success"],
                            "frequency_lock_success": m["frequency_lock_success"],
                            "one_to_one_flash_lock_success": m["one_to_one_flash_lock_success"],
                            "final_frequency_spread_hz": m["final_frequency_spread_hz"],
                            "flash_count_ratio": m["flash_count_ratio"],
                            "extra_flash_rate_hz": m["extra_flash_rate_hz"],
                            "mean_pairwise_timing_error_s": m["mean_pairwise_timing_error_s"],
                            "mean_pairwise_offset_jitter_s": m["mean_pairwise_offset_jitter_s"],
                            "mean_order_parameter_R": m["mean_order_parameter_R"],
                            "offset_phase_lock_success": m["offset_phase_lock_success"],
                            "sync_diagnostic_label": m["sync_diagnostic_label"],
                        }
                        aggregate_rows.append(row)

    # Save aggregate
    df = pd.DataFrame(aggregate_rows)
    _save_csv(out_dir / "aggregate_metrics.csv", aggregate_rows,
              list(aggregate_rows[0].keys()) if aggregate_rows else [])

    # Summary by model+variant+topology+freq_set
    summary_rows = []
    grp = df.groupby(["model", "variant", "topology", "freq_set"])
    for keys, g in grp:
        sr = {"model": keys[0], "variant": keys[1],
              "topology": keys[2], "freq_set": keys[3],
              "n_trials": len(g)}
        for col in ["zero_lag_group_sync_success", "phase_locked_group_success",
                     "phase_sync_success", "frequency_lock_success",
                     "one_to_one_flash_lock_success"]:
            sr[f"{col}_rate"] = round(g[col].astype(float).mean(), 4)
        for col in ["final_frequency_spread_hz", "flash_count_ratio",
                     "extra_flash_rate_hz", "mean_pairwise_timing_error_s",
                     "mean_pairwise_offset_jitter_s", "mean_order_parameter_R"]:
            vals = g[col].astype(str).replace("inf", np.nan).astype(float).dropna()
            sr[f"mean_{col}"] = round(vals.mean(), 6) if len(vals) > 0 else ""
        sr["dominant_diagnostic"] = g["sync_diagnostic_label"].mode().iloc[0] if len(g["sync_diagnostic_label"].mode()) > 0 else ""
        summary_rows.append(sr)

    _save_csv(out_dir / "summary_by_model_variant_topology_frequency_set.csv",
              summary_rows, list(summary_rows[0].keys()) if summary_rows else [])

    # Recommendations
    recs = _compute_recommendations(df)
    _save_json(out_dir / "recommended_variants.json", recs)

    # Diagnosis
    diag = _generate_diagnosis(summary_rows, df)
    (out_dir / "model_failure_diagnosis.md").write_text(diag)

    # Print key summary
    print(f"\nSweep complete: {out_dir}")
    _print_key_table(summary_rows)

    return out_dir


def _model_key(name: str) -> str:
    if name == "eapf_consensus":
        return "eapf_consensus"
    if name == "eapf_tracker":
        return "eapf"
    if name in ("pco_simple", "pco_adaptive_prc"):
        return "pco_if"
    return name


def _compute_recommendations(df: pd.DataFrame) -> dict:
    recs = {}
    # Best zero-lag sync
    zg = df.groupby("model")["zero_lag_group_sync_success"].mean()
    recs["best_zero_lag"] = zg.idxmax() if zg.max() > 0 else "none"
    # Best offset lock
    pl = df.groupby("model")["phase_locked_group_success"].mean()
    recs["best_offset_lock"] = pl.idxmax() if pl.max() > 0 else "none"
    # Best overall (weighted)
    df["_score"] = (df["zero_lag_group_sync_success"].astype(float) * 0.3
                    + df["phase_locked_group_success"].astype(float) * 0.3
                    + df["one_to_one_flash_lock_success"].astype(float) * 0.2
                    + df["phase_sync_success"].astype(float) * 0.2)
    score = df.groupby("model")["_score"].mean()
    recs["best_overall"] = score.idxmax()
    recs["scores"] = {m: round(float(s), 4) for m, s in score.items()}
    return recs


def _generate_diagnosis(summary: list[dict], df: pd.DataFrame) -> str:
    lines = ["# Stage 4A Model Failure Diagnosis", ""]
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Trials: {len(df)}")
    lines.append("")

    # 1. PCO simple only works for near-identical?
    pco_s = [r for r in summary if r["model"] == "pco_simple"]
    near_id = [r for r in pco_s if r["freq_set"] == "near_identical"]
    strong = [r for r in pco_s if r["freq_set"] == "strong_heterogeneity"]
    if near_id:
        zl = np.mean([r["zero_lag_group_sync_success_rate"] for r in near_id])
        pl = np.mean([r["phase_locked_group_success_rate"] for r in near_id])
        lines.append(f"## 1. PCO simple — near_identical: zero_lag={zl:.2f}, offset_lock={pl:.2f}")
    if strong:
        zl = np.mean([r["zero_lag_group_sync_success_rate"] for r in strong])
        pl = np.mean([r["phase_locked_group_success_rate"] for r in strong])
        lines.append(f"   PCO simple — strong_heterogeneity: zero_lag={zl:.2f}, offset_lock={pl:.2f}")

    # 2-8. Compare key metrics
    for model in ["pco_simple", "pco_adaptive_prc", "eapf_tracker", "eapf_consensus"]:
        rows = [r for r in summary if r["model"] == model]
        if not rows:
            continue
        fcr = np.mean([r.get("mean_flash_count_ratio", 0) or 0 for r in rows])
        zl = np.mean([r.get("zero_lag_group_sync_success_rate", 0) or 0 for r in rows])
        pl = np.mean([r.get("phase_locked_group_success_rate", 0) or 0 for r in rows])
        lines.append(f"## {model}: mean FCR={fcr:.3f}, zero_lag={zl:.2f}, offset_lock={pl:.2f}")

    # Kuramoto
    kr = [r for r in summary if r["model"] == "kuramoto"]
    if kr:
        zl = np.mean([r["zero_lag_group_sync_success_rate"] for r in kr])
        lines.append(f"## Kuramoto baseline: zero_lag={zl:.2f}")

    # 9. Best for HIL
    lines.append("")
    lines.append("## Recommended for Stage 4B HIL")
    recs = _compute_recommendations(df)
    lines.append(f"- Best zero-lag: **{recs['best_zero_lag']}**")
    lines.append(f"- Best offset-lock: **{recs['best_offset_lock']}**")
    lines.append(f"- Best overall: **{recs['best_overall']}**")

    return "\n".join(lines)


def _print_key_table(summary: list[dict]) -> None:
    print("\nKey results (strong_heterogeneity, all_to_all):")
    strong_aa = [r for r in summary if r["freq_set"] == "strong_heterogeneity"
                 and r["topology"] == "all_to_all"]
    for r in sorted(strong_aa, key=lambda x: x["model"]):
        print(f"  {r['model']:>20s} | {r['variant']:>25s} | "
              f"ZL={r['zero_lag_group_sync_success_rate']:.2f} "
              f"PL={r['phase_locked_group_success_rate']:.2f} "
              f"FCR={r.get('mean_flash_count_ratio','')} "
              f"Dx={r['dominant_diagnostic']}")


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4A — Diagnostic Model Sweep.",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--log-dir", default="experiments/logs/stage4a_multi_neighbour")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="Run all 4 frequency sets (slower)")
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

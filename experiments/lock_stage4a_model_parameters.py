#!/usr/bin/env python3
r"""Stage 4A — Parameter Locking for Multi-Neighbour Models.

Sweeps theory-informed parameter ranges, computes validation scores,
and selects one locked parameter set per model family.

Usage (quick smoke)::

    PYTHONPATH=. python experiments/lock_stage4a_model_parameters.py \
        --duration 10 --repeats 1 --quick

Usage (full locking)::

    PYTHONPATH=. python experiments/lock_stage4a_model_parameters.py \
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

from experiments.run_stage4a_multi_neighbour_simulation import (
    _run_trial, _save_csv, _save_json,
)

TOPOLOGIES = ["all_to_all", "chain", "directed_chain"]

FREQ_SETS_ALL = {
    "identical": [2.0, 2.0, 2.0],
    "near_identical": [1.9, 2.0, 2.1],
    "moderate_heterogeneity": [1.8, 2.0, 2.2],
    "strong_heterogeneity": [1.5, 2.0, 2.3],
}
FREQ_SETS_QUICK = {"near_identical": [1.9, 2.0, 2.1],
                    "strong_heterogeneity": [1.5, 2.0, 2.3]}

SEEDS = [1000, 1001, 1002, 1003, 1004]


# ======================================================================
# Scoring
# ======================================================================

def _validation_score(row: dict) -> float:
    """Compute a [0,1] validation score for a single trial."""
    zl = float(row.get("zero_lag_group_sync_success", 0) or 0)
    pl = float(row.get("phase_locked_group_success", 0) or 0)
    fl = float(row.get("frequency_lock_success", 0) or 0)
    oto = float(row.get("one_to_one_flash_lock_success", 0) or 0)
    ps = float(row.get("phase_sync_success", 0) or 0)

    fcr_s = row.get("flash_count_ratio", 1.0)
    if isinstance(fcr_s, str) and fcr_s == "inf":
        fcr = 99.0
    else:
        fcr = float(fcr_s) if fcr_s not in ("", None) else 1.0
    fcr_score = max(0.0, 1.0 - abs(fcr - 1.0) / 1.0)

    fs_s = row.get("final_frequency_spread_hz")
    fs = float(fs_s) if fs_s not in ("", None) else 0.5
    fs_score = max(0.0, 1.0 - fs / 0.5)

    te_s = row.get("mean_pairwise_timing_error_s")
    te = float(te_s) if te_s not in ("", None) else 0.2
    te_score = max(0.0, 1.0 - te / 0.2)

    return (0.25 * zl + 0.15 * pl + 0.15 * fl + 0.15 * oto
            + 0.10 * ps + 0.10 * fcr_score + 0.05 * fs_score + 0.05 * te_score)


# ======================================================================
# Parameter sweeps
# ======================================================================

def _sweep_kuramoto(args, freq_sets, rng) -> list[dict]:
    rows = []
    for K in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for topo in TOPOLOGIES:
            for fsn, freqs in freq_sets.items():
                for rep, seed in enumerate(SEEDS[:args.repeats]):
                    rng2 = np.random.default_rng(seed)
                    ta = _make_args(args, kuramoto_k=K)
                    trial = _run_trial("kuramoto", topo, freqs, args.duration, args.dt, ta, rng2)
                    rows.append(_trial_row(trial, "kuramoto", "baseline", topo, fsn,
                                           f"K={K}", {"K": K}))
                    print(f"  K={K} {topo} {fsn} rep{rep+1}")
    return rows


def _sweep_pco_simple(args, freq_sets, rng) -> list[dict]:
    rows = []
    for mode in ["additive_phase", "mirollo_state"]:
        for eps in [0.10, 0.15, 0.20, 0.25, 0.30]:
            for refr in [0.05, 0.10]:
                for topo in TOPOLOGIES:
                    for fsn, freqs in freq_sets.items():
                        for rep, seed in enumerate(SEEDS[:args.repeats]):
                            rng2 = np.random.default_rng(seed)
                            ta = _make_args(args, pco_coupling_mode=mode, pco_epsilon=eps,
                                            pco_refractory_period_s=refr, pco_state_curve_beta=3.0)
                            trial = _run_trial("pco_if", topo, freqs, args.duration, args.dt, ta, rng2)
                            rows.append(_trial_row(trial, "pco_simple", f"{mode}_eps{eps}_r{refr}",
                                                   topo, fsn, f"{mode} eps={eps}", {"coupling_mode": mode, "epsilon": eps, "refractory_period_s": refr}))
        print(f"  PCO simple {mode}")
    return rows


def _sweep_pco_adaptive(args, freq_sets, rng) -> list[dict]:
    rows = []
    for prc in ["biphasic_sine", "piecewise_advance_delay"]:
        for eps in [0.05, 0.10, 0.15, 0.20]:
            for adapt in [True, False]:
                for again in ([0.01, 0.03, 0.05] if adapt else [0.0]):
                    for mpc in [0.10, 0.20]:
                        for topo in TOPOLOGIES:
                            for fsn, freqs in freq_sets.items():
                                for rep, seed in enumerate(SEEDS[:args.repeats]):
                                    rng2 = np.random.default_rng(seed)
                                    ta = _make_args(args,
                                        pco_coupling_mode=prc, pco_epsilon=eps,
                                        pco_enable_phase_delay=True,
                                        pco_enable_frequency_adaptation=adapt,
                                        pco_frequency_adaptation_gain=again,
                                        pco_max_phase_correction=mpc,
                                        pco_min_inter_flash_interval_s=0.20,
                                        pco_post_flash_lockout_s=0.10)
                                    trial = _run_trial("pco_if", topo, freqs, args.duration, args.dt, ta, rng2)
                                    rows.append(_trial_row(trial, "pco_adaptive_prc",
                                        f"{prc}_eps{eps}_adapt{adapt}_g{again}_mpc{mpc}",
                                        topo, fsn, f"{prc} eps={eps}", {"prc_mode": prc, "epsilon": eps}))
        print(f"  PCO adaptive {prc}")
    return rows


def _sweep_eapf_tracker(args, freq_sets, rng) -> list[dict]:
    rows = []
    for pg in [0.1, 0.2, 0.3, 0.4]:
        for fg in [0.03, 0.05, 0.1, 0.15]:
            for topo in TOPOLOGIES:
                for fsn, freqs in freq_sets.items():
                    for rep, seed in enumerate(SEEDS[:args.repeats]):
                        rng2 = np.random.default_rng(seed)
                        ta = _make_args(args, eapf_phase_gain=pg, eapf_frequency_gain=fg)
                        trial = _run_trial("eapf", topo, freqs, args.duration, args.dt, ta, rng2)
                        rows.append(_trial_row(trial, "eapf_tracker", f"pg{pg}_fg{fg}",
                                               topo, fsn, f"pg={pg} fg={fg}", {"phase_gain": pg, "frequency_gain": fg}))
    print("  EAPF tracker done")
    return rows


def _sweep_eapf_consensus(args, freq_sets, rng) -> list[dict]:
    rows = []
    for pg in [0.02, 0.05, 0.10, 0.15]:
        for fg in [0.01, 0.02, 0.05, 0.08]:
            for pfa in [0.2, 0.4]:
                for ffa in [0.2, 0.4]:
                    for mps in [0.1, 0.2, 0.3]:
                        for mfs in [0.03, 0.05, 0.08]:
                            # Limit combinatorial explosion: tie filter alphas and step limits
                            if pfa != 0.2 or ffa != 0.2 or mps != 0.2 or mfs != 0.05:
                                continue  # Only sweep gains in detail for now
                            for topo in TOPOLOGIES:
                                for fsn, freqs in freq_sets.items():
                                    for rep, seed in enumerate(SEEDS[:args.repeats]):
                                        rng2 = np.random.default_rng(seed)
                                        ta = _make_args(args, eapf_phase_gain=pg, eapf_frequency_gain=fg)
                                        trial = _run_trial("eapf_consensus", topo, freqs, args.duration, args.dt, ta, rng2)
                                        rows.append(_trial_row(trial, "eapf_consensus", f"pg{pg}_fg{fg}",
                                                               topo, fsn, f"pg={pg} fg={fg}", {"phase_gain": pg, "frequency_gain": fg}))
    # Additional tuned: sweep best filter/step params for best gain pair
    print("  EAPF consensus done")
    return rows


def _make_args(base, **overrides) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.kuramoto_k = overrides.get("kuramoto_k", 3.5)
    ns.pco_coupling_mode = overrides.get("pco_coupling_mode", "mirollo_state")
    ns.pco_epsilon = overrides.get("pco_epsilon", 0.25)
    ns.pco_refractory_period_s = overrides.get("pco_refractory_period_s", 0.05)
    ns.pco_state_curve_beta = overrides.get("pco_state_curve_beta", 3.0)
    ns.pco_enable_phase_delay = overrides.get("pco_enable_phase_delay", False)
    ns.pco_enable_frequency_adaptation = overrides.get("pco_enable_frequency_adaptation", False)
    ns.pco_frequency_adaptation_gain = overrides.get("pco_frequency_adaptation_gain", 0.0)
    ns.pco_max_phase_correction = overrides.get("pco_max_phase_correction", 0.40)
    ns.pco_min_inter_flash_interval_s = overrides.get("pco_min_inter_flash_interval_s", 0.0)
    ns.pco_post_flash_lockout_s = overrides.get("pco_post_flash_lockout_s", 0.0)
    ns.eapf_phase_gain = overrides.get("eapf_phase_gain", 0.3)
    ns.eapf_frequency_gain = overrides.get("eapf_frequency_gain", 0.1)
    ns.eapf_frequency_min_hz = 0.5
    ns.eapf_frequency_max_hz = 4.0
    ns.event_delay_s = 0.0
    ns.missed_event_prob = 0.0
    return ns


def _trial_row(trial, model, variant, topo, fsn, label, params) -> dict:
    m = trial["metrics"]
    return {
        "model": model, "variant": variant, "topology": topo,
        "freq_set": fsn, "param_label": label, **params,
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
        "sync_diagnostic_label": m["sync_diagnostic_label"],
    }


# ======================================================================
# Selection logic
# ======================================================================

def _select_best(rows: list[dict], group_key: str) -> dict:
    df = pd.DataFrame(rows)
    df["_score"] = df.apply(_validation_score, axis=1)
    grp = df.groupby(group_key)["_score"].mean().sort_values(ascending=False)
    best_label = grp.index[0]
    best_score = round(float(grp.iloc[0]), 4)
    best_rows = df[df[group_key] == best_label]
    # Get params from first row
    params = {k: best_rows.iloc[0][k] for k in best_rows.columns
              if k not in ("model", "variant", "topology", "freq_set",
                           "param_label", "_score",
                           "zero_lag_group_sync_success", "phase_locked_group_success",
                           "phase_sync_success", "frequency_lock_success",
                           "one_to_one_flash_lock_success", "final_frequency_spread_hz",
                           "flash_count_ratio", "extra_flash_rate_hz",
                           "mean_pairwise_timing_error_s", "mean_pairwise_offset_jitter_s",
                           "mean_order_parameter_R", "sync_diagnostic_label")
              and not k.startswith("_")}
    # Summary stats for the best label
    best_df = df[df[group_key] == best_label]
    return {
        "param_label": best_label,
        "validation_score": best_score,
        "parameters": {k: (v.item() if hasattr(v, 'item') else v) for k, v in params.items()},
        "n_trials": len(best_df),
        "zl_rate": round(best_df["zero_lag_group_sync_success"].astype(float).mean(), 4),
        "pl_rate": round(best_df["phase_locked_group_success"].astype(float).mean(), 4),
        "oto_rate": round(best_df["one_to_one_flash_lock_success"].astype(float).mean(), 4),
        "mean_fcr": round(best_df["flash_count_ratio"].astype(str).replace("inf","99").astype(float).mean(), 3),
    }


# ======================================================================
# Figures
# ======================================================================

def _generate_figures(all_rows: list[dict], locked: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)

    # 1. Kuramoto K sensitivity
    kr = df[df["model"] == "kuramoto"]
    if len(kr) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        kr_grp = kr.groupby("param_label")["_score"].agg(["mean", "std"])
        ks = [float(str(k).split("=")[-1]) for k in kr_grp.index]
        ax.errorbar(ks, kr_grp["mean"], yerr=kr_grp["std"], marker="o", capsize=4)
        ax.set_xlabel("K")
        ax.set_ylabel("Validation Score")
        ax.set_title("Kuramoto — Coupling Gain Sensitivity")
        ax.grid(True, alpha=0.3)
        fig.savefig(fig_dir / "kuramoto_K_sensitivity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 2. PCO simple epsilon sensitivity
    ps = df[df["model"] == "pco_simple"]
    if len(ps) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode in ps["param_label"].apply(lambda x: x.split()[0]).unique():
            subset = ps[ps["param_label"].str.startswith(mode)]
            grp = subset.groupby("param_label")["_score"].mean()
            eps_vals = [float(str(k).split("eps=")[-1].split("_")[0].split()[0]) for k in grp.index]
            ax.plot(eps_vals, grp.values, marker="o", label=mode)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("Validation Score")
        ax.set_title("PCO Simple — Epsilon Sensitivity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(fig_dir / "pco_simple_epsilon_sensitivity.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3. Locked model validation scores
    models = list(locked.keys())
    scores = [locked[m]["validation_score"] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, scores, color="steelblue", edgecolor="black")
    ax.set_ylabel("Validation Score")
    ax.set_title("Locked Model Validation Scores")
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.savefig(fig_dir / "locked_model_validation_scores.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ======================================================================
# Report
# ======================================================================

def _generate_report(locked: dict, out_dir: Path) -> None:
    lines = [
        "# Stage 4A Parameter Locking Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Purpose",
        "Lock one representative parameter set per model family for final thesis evaluation.",
        "",
        "## Validation Conditions",
        "- Topologies: all_to_all, chain, directed_chain",
        "- Frequency sets: identical, near_identical, moderate_heterogeneity, strong_heterogeneity (or quick subset)",
        "- Duration: 60 s, Repeats: 5, Seeds: 1000-1004",
        "",
        "## Locked Parameters",
    ]
    for model in ["kuramoto", "pco_simple", "pco_adaptive_prc", "eapf_tracker", "eapf_consensus"]:
        if model not in locked:
            continue
        l = locked[model]
        lines.append(f"### {model}")
        lines.append(f"- **Parameters:** {json.dumps(l['parameters'])}")
        lines.append(f"- **Validation score:** {l['validation_score']}")
        lines.append(f"- **Zero-lag rate:** {l['zl_rate']}, Phase-lock rate: {l['pl_rate']}")
        lines.append(f"- **Mean FCR:** {l['mean_fcr']}")
        lines.append(f"- **Reason:** {l.get('selection_reason', 'Best overall validation score across all conditions.')}")
        lines.append("")

    lines.append("## Warning")
    lines.append("Final evaluation must use different random seeds (2000-2029 recommended) "
                 "and must not change the locked parameters.")
    (out_dir / "parameter_locking_report.md").write_text("\n".join(lines))


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4A — Parameter Locking for Multi-Neighbour Models.",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--log-dir", default="experiments/logs/stage4a_model_selection")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced freq sets (near_identical + strong_heterogeneity only)")
    args = parser.parse_args()

    out_dir = Path(args.log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_parameter_locking"
    out_dir.mkdir(parents=True, exist_ok=True)
    freq_sets = FREQ_SETS_QUICK if args.quick else FREQ_SETS_ALL

    rng = np.random.default_rng(42)
    all_rows: list[dict] = []

    print("=" * 60)
    print("PARAMETER LOCKING SWEEP")
    print(f"Duration: {args.duration}s, Repeats: {args.repeats}, "
          f"Freq sets: {list(freq_sets.keys())}")
    print("=" * 60)

    # Kuramoto
    print("\n--- Kuramoto ---")
    all_rows.extend(_sweep_kuramoto(args, freq_sets, rng))

    # PCO simple
    print("\n--- PCO Simple ---")
    all_rows.extend(_sweep_pco_simple(args, freq_sets, rng))

    # PCO adaptive PRC
    print("\n--- PCO Adaptive PRC ---")
    all_rows.extend(_sweep_pco_adaptive(args, freq_sets, rng))

    # EAPF tracker
    print("\n--- EAPF Tracker ---")
    all_rows.extend(_sweep_eapf_tracker(args, freq_sets, rng))

    # EAPF consensus
    print("\n--- EAPF Consensus ---")
    all_rows.extend(_sweep_eapf_consensus(args, freq_sets, rng))

    # Save aggregate
    _save_csv(out_dir / "parameter_locking_aggregate_metrics.csv", all_rows,
              list(all_rows[0].keys()) if all_rows else [])

    # Select locked parameters
    locked = {}
    for model in ["kuramoto", "pco_simple", "pco_adaptive_prc", "eapf_tracker", "eapf_consensus"]:
        mr = [r for r in all_rows if r["model"] == model]
        if not mr:
            continue
        best = _select_best(mr, "param_label")
        locked[model] = best
        print(f"\n{model}: best={best['param_label']} score={best['validation_score']} "
              f"ZL={best['zl_rate']} PL={best['pl_rate']} FCR={best['mean_fcr']}")

    _save_json(out_dir / "locked_model_parameters.json", locked)

    # Summary CSV
    sum_rows = []
    for model, l in locked.items():
        sum_rows.append({"model": model, **{f"locked_{k}": v for k, v in l.items()}})
    _save_csv(out_dir / "parameter_locking_summary.csv", sum_rows,
              list(sum_rows[0].keys()) if sum_rows else [])

    # Figures and report
    df = pd.DataFrame(all_rows)
    df["_score"] = df.apply(_validation_score, axis=1)
    all_rows_with_score = df.to_dict("records")
    _generate_figures(all_rows_with_score, locked, out_dir)
    _generate_report(locked, out_dir)

    print(f"\nLocked parameters saved to {out_dir}")
    print(json.dumps({m: {"score": l["validation_score"], "params": l["parameters"]}
                      for m, l in locked.items()}, indent=2))


if __name__ == "__main__":
    main()

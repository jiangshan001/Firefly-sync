#!/usr/bin/env python3
r"""Stage 4A — Multi-Neighbour Synchronisation Simulation.

Compares Kuramoto, PCO-I&F, and EAPF under decentralised multi-agent
topologies (all-to-all, chain, directed_chain, ring).

No hardware.  No camera.  No leader API.

Usage::

    PYTHONPATH=. python experiments/run_stage4a_multi_neighbour_simulation.py \
        --model all --topology all --duration 60 --repeats 20

    PYTHONPATH=. python experiments/run_stage4a_multi_neighbour_simulation.py \
        --model kuramoto --topology all_to_all --duration 60 --repeats 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from firefly_sync.multi_agent.agent import AgentConfig
from firefly_sync.multi_agent.topology import build_topology, TOPOLOGY_TYPES
from firefly_sync.multi_agent.simulation import MultiAgentSimulation
from firefly_sync.multi_agent.metrics import (
    check_group_synchronisation,
    compute_group_metrics,
)

MODELS = ["kuramoto", "pco_if", "eapf", "eapf_consensus"]
DEFAULT_TOPOLOGIES = ["all_to_all", "chain", "directed_chain"]


# ======================================================================
# Helpers
# ======================================================================

def _make_output_dir(root: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(root) / f"{ts}_multi_neighbour_simulation"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _safe_mean(df, col: str) -> str:
    """Mean of a column that may contain 'inf' strings."""
    vals = df[col].astype(str).replace("inf", np.nan).astype(float).dropna()
    return str(round(vals.mean(), 4)) if len(vals) > 0 else ""


def _save_json(path: Path, data: dict) -> None:
    # Convert numpy bools to Python bools for JSON serialization
    clean = {}
    for k, v in data.items():
        if isinstance(v, (np.bool_,)):
            clean[k] = bool(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


# ======================================================================
# Single trial
# ======================================================================

def _run_trial(
    model: str,
    topology_type: str,
    initial_frequencies: list[float],
    duration_s: float,
    dt: float,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    """Run one simulation trial. Returns metrics dict."""
    n = len(initial_frequencies)
    configs = []
    for i, freq in enumerate(initial_frequencies):
        phase = rng.uniform(0, 2.0 * np.pi)
        cfg = AgentConfig(
            agent_id=i, model=model,
            initial_frequency_hz=freq,
            initial_phase=phase,
            coupling_strength=args.kuramoto_k,
            pco_epsilon=args.pco_epsilon,
            pco_coupling_mode=args.pco_coupling_mode,
            pco_refractory_period_s=args.pco_refractory_period_s,
            pco_state_curve_beta=args.pco_state_curve_beta,
            pco_enable_phase_delay=getattr(args, 'pco_enable_phase_delay', False),
            pco_enable_frequency_adaptation=getattr(args, 'pco_enable_frequency_adaptation', False),
            pco_frequency_adaptation_gain=getattr(args, 'pco_frequency_adaptation_gain', 0.0),
            pco_max_phase_correction=getattr(args, 'pco_max_phase_correction', 0.40),
            pco_min_inter_flash_interval_s=getattr(args, 'pco_min_inter_flash_interval_s', 0.0),
            pco_post_flash_lockout_s=getattr(args, 'pco_post_flash_lockout_s', 0.0),
            eapf_phase_gain=args.eapf_phase_gain,
            eapf_frequency_gain=args.eapf_frequency_gain,
            eapf_frequency_min_hz=args.eapf_frequency_min_hz,
            eapf_frequency_max_hz=args.eapf_frequency_max_hz,
            eapf_leader_period_window=6,
        )
        configs.append(cfg)

    topo = build_topology(n, topology_type)
    sim = MultiAgentSimulation(
        configs, topo, dt=dt,
        event_delay_s=args.event_delay_s,
        missed_event_prob=args.missed_event_prob,
        rng=rng,
    )
    sim.run(duration_s)
    results = sim.get_results()

    # Collect phase histories for order parameter
    phase_histories = [a._phase_history for a in sim.agents]

    metrics = compute_group_metrics(
        results["agent_flash_times"],
        agents=sim.agents,
        phases_over_time=phase_histories,
    )
    sync = check_group_synchronisation(
        results["agent_flash_times"],
        agents=sim.agents,
        phases_over_time=phase_histories,
    )
    metrics["group_sync_success"] = sync["group_sync_success"]
    metrics["model"] = model
    metrics["topology"] = topology_type
    metrics["n_agents"] = n

    return {
        "metrics": metrics,
        "flash_events": results["flash_events"],
        "agent_logs": results["agent_logs"],
        "agent_final_frequencies": results["agent_final_frequencies"],
        "phase_histories": phase_histories,
    }


# ======================================================================
# Plotting
# ======================================================================

def _generate_plots(agg_rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(agg_rows)

    # Group by model+topology
    grp = df.groupby(["model", "topology"])

    # 1. Success rate (strict group sync)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    rates = []
    for (m, t), g in grp:
        labels.append(f"{m}\n{t}")
        rates.append(g["group_sync_success"].mean() * 100)
    x = np.arange(len(labels))
    ax.bar(x, rates, color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Strict Group Sync Success Rate")
    ax.set_ylim(0, 105)
    fig.savefig(out / "success_rate_by_model_topology.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1b. Phase sync success rate
    fig, ax = plt.subplots(figsize=(8, 5))
    rates2 = [g["phase_sync_success"].mean() * 100 for (_, _), g in grp]
    ax.bar(x, rates2, color="darkseagreen", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Phase Sync Success Rate (R>0.85 & timing<0.1s)")
    ax.set_ylim(0, 105)
    fig.savefig(out / "phase_sync_success_rate_by_model_topology.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1c. Flash count ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    fcr_means = []
    for (_, _), g in grp:
        vals = g["flash_count_ratio"].astype(str).replace("inf", np.nan).astype(float).dropna()
        fcr_means.append(vals.mean() if len(vals) > 0 else 0)
    ax.bar(x, fcr_means, color="indianred", edgecolor="black")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=1.2, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Flash Count Ratio (max/min)")
    ax.set_title("Flash Count Ratio (1:1 lock when ≤ 1.2)")
    fig.savefig(out / "flash_count_ratio_by_model_topology.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2-4. Other metrics similarly
    for metric, ylabel, fname in [
        ("final_frequency_spread_hz", "Freq Spread (Hz)", "final_frequency_spread"),
        ("mean_pairwise_timing_error_s", "Timing Error (s)", "pairwise_timing_error"),
        ("mean_order_parameter_R", "Order Parameter R", "order_parameter"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        means = []
        for (m, t), g in grp:
            vals = g[metric].dropna().astype(float)
            means.append(vals.mean() if len(vals) > 0 else 0)
        ax.bar(x, means, color="darkorange", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        fig.savefig(out / f"{fname}_by_model_topology.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"  Plots saved to {out}")


# ======================================================================
# Batch runner
# ======================================================================

def run_batch(args: argparse.Namespace) -> Path:
    import pandas as pd

    out_dir = _make_output_dir(args.log_dir)
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = MODELS if args.model == "all" else [args.model]
    tops_to_run = (DEFAULT_TOPOLOGIES if args.topology == "all"
                    else (list(TOPOLOGY_TYPES) if args.topology == "all_with_ring"
                          else [args.topology]))

    rng = np.random.default_rng(args.random_seed)

    batch_meta = {
        "models": models_to_run, "topologies": tops_to_run,
        "duration_s": args.duration, "dt": args.dt,
        "repeats": args.repeats,
        "initial_frequencies": args.initial_frequencies,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(out_dir / "batch_metadata.json", batch_meta)

    aggregate_rows: list[dict] = []
    total = len(models_to_run) * len(tops_to_run) * args.repeats
    count = 0

    for model in models_to_run:
        for topo in tops_to_run:
            for rep in range(1, args.repeats + 1):
                count += 1
                tid = f"{model}_{topo}_r{rep:02d}"
                print(f"[{count}/{total}] {tid}")

                trial = _run_trial(
                    model=model, topology_type=topo,
                    initial_frequencies=args.initial_frequencies,
                    duration_s=args.duration, dt=args.dt,
                    args=args, rng=rng,
                )
                m = trial["metrics"]
                row = {
                    "trial_id": tid, "model": model, "topology": topo,
                    "n_agents": len(args.initial_frequencies),
                    "group_sync_success": m["group_sync_success"],
                    "phase_sync_success": m["phase_sync_success"],
                    "offset_phase_lock_success": m["offset_phase_lock_success"],
                    "phase_locked_group_success": m["phase_locked_group_success"],
                    "frequency_lock_success": m["frequency_lock_success"],
                    "one_to_one_flash_lock_success": m["one_to_one_flash_lock_success"],
                    "final_frequency_spread_hz": m["final_frequency_spread_hz"],
                    "mean_pairwise_timing_error_s": m["mean_pairwise_timing_error_s"],
                    "flash_timing_dispersion_s": m["flash_timing_dispersion_s"],
                    "mean_order_parameter_R": m["mean_order_parameter_R"],
                    "final_order_parameter_R": m["final_order_parameter_R"],
                    "flash_count_ratio": m["flash_count_ratio"],
                    "extra_flash_rate_hz": m["extra_flash_rate_hz"],
                    "sync_diagnostic_label": m["sync_diagnostic_label"],
                    "sync_failure_reason": m["sync_failure_reason"],
                }
                aggregate_rows.append(row)

                # Save trial logs
                td = trials_dir / tid
                td.mkdir(parents=True, exist_ok=True)
                _save_json(td / "metrics_summary.json", m)

                # Combine agent logs
                all_logs = []
                for agent_id, logs in enumerate(trial["agent_logs"]):
                    for entry in logs:
                        entry["agent_id"] = agent_id
                        all_logs.append(entry)
                if all_logs:
                    _save_csv(td / "agent_log.csv", all_logs,
                              list(all_logs[0].keys()))

                _save_csv(td / "flash_events.csv", trial["flash_events"],
                          ["t_s", "agent_id", "event_type", "model"])

    # Aggregate CSV
    agg_fields = list(aggregate_rows[0].keys()) if aggregate_rows else []
    _save_csv(out_dir / "aggregate_metrics.csv", aggregate_rows, agg_fields)

    # Summary by model+topology
    df = pd.DataFrame(aggregate_rows)
    summary_rows = []
    for (m, t), grp in df.groupby(["model", "topology"]):
        summary_rows.append({
            "model": m, "topology": t, "n_trials": len(grp),
            "group_sync_success_rate": round(grp["group_sync_success"].mean(), 4),
            "phase_sync_success_rate": round(grp["phase_sync_success"].mean(), 4),
            "frequency_lock_success_rate": round(grp["frequency_lock_success"].mean(), 4),
            "one_to_one_flash_lock_success_rate": round(grp["one_to_one_flash_lock_success"].mean(), 4),
            "mean_flash_count_ratio": _safe_mean(grp, "flash_count_ratio"),
            "mean_extra_flash_rate_hz": _safe_mean(grp, "extra_flash_rate_hz"),
            "mean_final_frequency_spread_hz": _safe_mean(grp, "final_frequency_spread_hz"),
            "mean_pairwise_timing_error_s": _safe_mean(grp, "mean_pairwise_timing_error_s"),
            "mean_order_parameter_R": _safe_mean(grp, "mean_order_parameter_R"),
            "dominant_diagnostic": grp["sync_diagnostic_label"].mode().iloc[0] if len(grp["sync_diagnostic_label"].mode()) > 0 else "",
        })
    _save_csv(out_dir / "summary_by_model_topology.csv", summary_rows,
              list(summary_rows[0].keys()) if summary_rows else [])

    # Plots
    if not args.no_plots:
        _generate_plots(aggregate_rows, out_dir)

    # Print summary
    print(f"\nBatch complete: {out_dir}")
    for sr in summary_rows:
        print(f"  {sr['model']:>10s} {sr['topology']:>16s}  "
              f"GrpSync={sr['group_sync_success_rate']:.2f}  "
              f"PhSync={sr['phase_sync_success_rate']:.2f}  "
              f"1:1={sr['one_to_one_flash_lock_success_rate']:.2f}  "
              f"FCR={sr['mean_flash_count_ratio']}  "
              f"Dx={sr['dominant_diagnostic']}")

    return out_dir


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4A — Multi-Neighbour Synchronisation Simulation.",
    )
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    parser.add_argument("--topology", choices=list(TOPOLOGY_TYPES) + ["all"], default="all")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--initial-frequencies", type=float, nargs="+", default=[1.5, 2.0, 2.3])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-dir", default="experiments/logs/stage4a_multi_neighbour")
    parser.add_argument("--no-plots", action="store_true")
    # Kuramoto
    parser.add_argument("--kuramoto-k", type=float, default=3.5)
    # PCO
    parser.add_argument("--pco-coupling-mode", default="mirollo_state")
    parser.add_argument("--pco-epsilon", type=float, default=0.25)
    parser.add_argument("--pco-refractory-period-s", type=float, default=0.05)
    parser.add_argument("--pco-state-curve-beta", type=float, default=3.0)
    parser.add_argument("--pco-variant", choices=["simple", "adaptive_prc"], default="simple")
    parser.add_argument("--pco-enable-phase-delay", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--pco-enable-frequency-adaptation", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--pco-frequency-adaptation-gain", type=float, default=0.01)
    parser.add_argument("--pco-min-inter-flash-interval-s", type=float, default=0.0)
    parser.add_argument("--pco-post-flash-lockout-s", type=float, default=0.0)
    # EAPF
    parser.add_argument("--eapf-phase-gain", type=float, default=0.3)
    parser.add_argument("--eapf-frequency-gain", type=float, default=0.1)
    parser.add_argument("--eapf-frequency-min-hz", type=float, default=0.5)
    parser.add_argument("--eapf-frequency-max-hz", type=float, default=4.0)
    # Noise
    parser.add_argument("--event-delay-s", type=float, default=0.0)
    parser.add_argument("--missed-event-prob", type=float, default=0.0)

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be > 0")

    run_batch(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Compare two final Stage 4A evaluations for repeatability.

Usage::

    PYTHONPATH=. python experiments/compare_stage4a_final_evaluations.py \
        --run1 experiments/logs/stage4a_model_selection/20260616_100238_final_evaluation \
        --run2 experiments/logs/stage4a_model_selection/<ts>_final_evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_scores(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "final_model_scores.csv")


def _load_agg(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "final_evaluation_aggregate_metrics.csv")


def _safe_mean(series) -> float:
    return float(pd.to_numeric(series, errors="coerce").dropna().mean())


def compare(run1: Path, run2: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures" / "repeatability"
    fig_dir.mkdir(parents=True, exist_ok=True)

    s1 = _load_scores(run1)
    s2 = _load_scores(run2)
    a1 = _load_agg(run1) if (run1 / "final_evaluation_aggregate_metrics.csv").exists() else None
    a2 = _load_agg(run2) if (run2 / "final_evaluation_aggregate_metrics.csv").exists() else None

    models = s1["model_family"].tolist()

    # ---- Total score comparison ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, s1["total_weighted_score"], w, label="Run 1 (seeds 2000)", color="steelblue")
    ax.bar(x + w/2, s2["total_weighted_score"], w, label="Run 2 (seeds 4000)", color="darkorange")
    for i, (v1, v2) in enumerate(zip(s1["total_weighted_score"], s2["total_weighted_score"])):
        ax.text(i - w/2, v1 + 0.01, f"{v1:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, v2 + 0.01, f"{v2:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Total Weighted Score")
    ax.set_title("Repeatability — Total Weighted Score")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.savefig(fig_dir / "repeatability_total_score_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Score delta ----
    deltas = (s2["total_weighted_score"].values - s1["total_weighted_score"].values)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["green" if abs(d) < 0.03 else "orange" if abs(d) < 0.07 else "red" for d in deltas]
    ax.bar(models, deltas, color=colors, edgecolor="black")
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.003 if d >= 0 else -0.01), f"{d:+.4f}", ha="center", fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="-")
    ax.axhline(y=0.03, color="green", linestyle="--", alpha=0.4)
    ax.axhline(y=-0.03, color="green", linestyle="--", alpha=0.4)
    ax.set_ylabel("Score Delta (Run 2 - Run 1)")
    ax.set_title("Repeatability — Score Change")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.savefig(fig_dir / "repeatability_score_delta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- N3 success comparison ----
    if a1 is not None and a2 is not None:
        zl1 = a1.groupby("model")["zero_lag_group_sync_success"].mean()
        zl2 = a2.groupby("model")["zero_lag_group_sync_success"].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w/2, [zl1.get(m, 0) for m in models], w, label="Run 1", color="seagreen")
        ax.bar(x + w/2, [zl2.get(m, 0) for m in models], w, label="Run 2", color="mediumseagreen")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Zero-Lag Success Rate")
        ax.set_title("Repeatability — N=3 Success Rate")
        ax.legend()
        fig.savefig(fig_dir / "repeatability_n3_success_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # FCR
        def _fcr_mean(g):
            return _safe_mean(g["flash_count_ratio"])
        fcr1 = a1.groupby("model").apply(_fcr_mean)
        fcr2 = a2.groupby("model").apply(_fcr_mean)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w/2, [fcr1.get(m, 0) for m in models], w, label="Run 1", color="indianred")
        ax.bar(x + w/2, [fcr2.get(m, 0) for m in models], w, label="Run 2", color="salmon")
        ax.axhline(y=1.0, color="gray", linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Flash Count Ratio")
        ax.set_title("Repeatability — Flash Count Ratio")
        ax.legend()
        fig.savefig(fig_dir / "repeatability_flash_count_ratio_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- N5 comparison ----
    n5_1 = run1 / "final_evaluation_aggregate_n5.csv"
    n5_2 = run2 / "final_evaluation_aggregate_n5.csv"
    if n5_1.exists() and n5_2.exists():
        n5df1 = pd.read_csv(n5_1)
        n5df2 = pd.read_csv(n5_2)
        n5zl1 = n5df1.groupby("model")["zero_lag_group_sync_success"].mean()
        n5zl2 = n5df2.groupby("model")["zero_lag_group_sync_success"].mean()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w/2, [n5zl1.get(m, 0) for m in models], w, label="Run 1", color="mediumpurple")
        ax.bar(x + w/2, [n5zl2.get(m, 0) for m in models], w, label="Run 2", color="plum")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("N5 Zero-Lag Success Rate")
        ax.set_title("Repeatability — N=5 Scalability")
        ax.legend()
        fig.savefig(fig_dir / "repeatability_n5_success_comparison.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---- Rank stability ----
    r1_ranks = {m: i+1 for i, m in enumerate(s1["model_family"])}
    r2_ranks = {m: i+1 for i, m in enumerate(s2["model_family"])}
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        r1, r2 = r1_ranks[m], r2_ranks[m]
        ax.plot([r1, r2], [0, 0.2], marker="o", color=f"C{i}", linewidth=2)
        ax.text(r1, -0.05, f"R1", ha="center", fontsize=7)
        ax.text(r2, 0.25, f"R2", ha="center", fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel("Rank")
    ax.set_title("Rank Stability (Run 1 → Run 2)")
    ax.set_xlim(0.5, 5.5)
    ax.invert_yaxis()
    fig.savefig(fig_dir / "repeatability_rank_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Report ----
    lines = [
        "# Final Evaluation Repeatability Report",
        f"Run 1: {run1.name}",
        f"Run 2: {run2.name}",
        "",
        "## Score Comparison",
        "| Model | Run 1 Score | Run 2 Score | Delta | Stability |",
        "|-------|:-----------:|:-----------:|:-----:|:---------:|",
    ]
    for m in models:
        v1 = float(s1[s1["model_family"] == m]["total_weighted_score"].iloc[0])
        v2 = float(s2[s2["model_family"] == m]["total_weighted_score"].iloc[0])
        d = v2 - v1
        if abs(d) < 0.03:
            stab = "very stable"
        elif abs(d) < 0.07:
            stab = "moderately stable"
        else:
            stab = "investigate"
        lines.append(f"| {m} | {v1:.4f} | {v2:.4f} | {d:+.4f} | {stab} |")

    lines += [
        "",
        "## Ranking Stability",
        "| Model | Run 1 Rank | Run 2 Rank | Changed? |",
        "|-------|:----------:|:----------:|:--------:|",
    ]
    for m in models:
        r1 = r1_ranks[m]
        r2 = r2_ranks[m]
        ch = "YES" if r1 != r2 else "no"
        lines.append(f"| {m} | {r1} | {r2} | {ch} |")

    lines += [
        "",
        "## Interpretation",
    ]
    # Check if EAPF consensus is still #1
    r1_top = s1["model_family"].iloc[0]
    r2_top = s2["model_family"].iloc[0]
    if r1_top == r2_top == "eapf_consensus":
        lines.append("- [OK] EAPF consensus remains primary HIL candidate in both runs.")
    else:
        lines.append(f"- [WARN] Primary candidate changed: R1={r1_top}, R2={r2_top}")

    r1_2nd = s1["model_family"].iloc[1]
    r2_2nd = s2["model_family"].iloc[1]
    if r1_2nd == r2_2nd == "kuramoto":
        lines.append("- [OK] Kuramoto remains secondary comparison model in both runs.")

    max_delta = max(abs(d) for d in deltas)
    lines.append(f"- Maximum score delta: {max_delta:.4f}")
    if max_delta < 0.03:
        lines.append("- All score changes are within the 'very stable' threshold.")
    elif max_delta < 0.07:
        lines.append("- Score changes are within the 'moderately stable' threshold.")
    else:
        lines.append("- Some score changes exceed 0.07 — see per-model details.")

    (out_dir / "final_evaluation_repeatability_report.md").write_text("\n".join(lines))

    # Print summary
    print(f"\nComparison saved to {out_dir}")
    for m in models:
        v1 = float(s1[s1["model_family"] == m]["total_weighted_score"].iloc[0])
        v2 = float(s2[s2["model_family"] == m]["total_weighted_score"].iloc[0])
        print(f"  {m:>20s}: {v1:.4f} → {v2:.4f}  (Δ={v2-v1:+.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two final evaluations.")
    parser.add_argument("--run1", required=True)
    parser.add_argument("--run2", required=True)
    args = parser.parse_args()
    r1 = Path(args.run1)
    r2 = Path(args.run2)
    if not r1.is_dir() or not r2.is_dir():
        print("ERROR: both --run1 and --run2 must be existing directories")
        sys.exit(1)
    compare(r1, r2, r1.parent / f"{r1.name}_vs_{r2.name}_repeatability")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regenerate only Figure A2 (K-sweep frequency std) from existing CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
# Go up from v2/ → step5b_mutual_hil_report/ → docs/ → firefly-sync/
REPO_ROOT = REPORT_DIR.parents[2]

K_SWEEP_DIR = (
    REPO_ROOT
    / "experiments"
    / "logs"
    / "step5b_kuramoto_k_sensitivity"
    / "k_sweep_20260621_v4"
)

FIG_DIR = REPORT_DIR / "figures"

CONDITION_ORDER = [1.2, 1.5, 2.5]
K_VALUES = [2.5, 3.0, 3.5, 4.0, 4.5]
K_COLORS = {
    2.5: "#1b9e77",
    3.0: "#d95f02",
    3.5: "#7570b3",
    4.0: "#e7298a",
    4.5: "#66a61e",
}

# Match main script rcParams
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    # Load data
    k_aggregate = pd.read_csv(K_SWEEP_DIR / "aggregate_metrics.csv")
    print(f"Loaded {len(k_aggregate)} rows from aggregate_metrics.csv")
    print(f"Columns: {list(k_aggregate.columns)}")

    # Check data
    for pi_freq in CONDITION_ORDER:
        sub = k_aggregate[k_aggregate["pi_initial_freq"] == pi_freq]
        print(f"\nPi {pi_freq} Hz: {len(sub)} rows")
        for k in K_VALUES:
            ksub = sub[sub["kuramoto_K"] == k]
            n_virtual = pd.to_numeric(ksub["virtual_freq_final_5s_std"], errors="coerce").dropna()
            n_pi = pd.to_numeric(ksub["pi_freq_final_5s_std"], errors="coerce").dropna()
            print(f"  K={k}: virtual_std={list(n_virtual.values)}, pi_std={list(n_pi.values)}")

    # Appendix Figure A2: Virtual/Pi final-window frequency std vs K
    # Layout: 2 rows (Virtual / Pi) × 3 columns (Pi initial frequency conditions)
    # Each subplot shows individual trial points + mean trend line across K values
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))

    # Jitter offset for individual trial points to avoid overplotting
    _jitter = 0.04

    for col_idx, (pi_freq, label) in enumerate(zip(CONDITION_ORDER, ["Pi 1.2 Hz", "Pi 1.5 Hz", "Pi 2.5 Hz"])):
        sub = k_aggregate[k_aggregate["pi_initial_freq"] == pi_freq]
        for row_idx, (metric, ylabel) in enumerate([
            ("virtual_freq_final_5s_std", "Virtual frequency std (Hz)"),
            ("pi_freq_final_5s_std", "Pi frequency std (Hz)"),
        ]):
            ax = axes[row_idx][col_idx]

            # Collect means for trend line
            k_means = []
            for k in K_VALUES:
                ksub = sub[sub["kuramoto_K"] == k]
                vals = pd.to_numeric(ksub[metric], errors="coerce").dropna().to_numpy()
                if len(vals) == 0:
                    continue

                # Individual trial points: small, semi-transparent, with jitter
                n = len(vals)
                x_jittered = np.array([k] * n) + np.random.default_rng(42).uniform(-_jitter, _jitter, n)
                ax.plot(x_jittered, vals, "o", color=K_COLORS[k],
                        alpha=0.3, markersize=3.5, markeredgewidth=0)

                k_means.append((k, vals.mean()))

            # Connect means with a line to show trend across K
            if k_means:
                ks, means = zip(*k_means)
                ax.plot(ks, means, "D-", color="0.25", markersize=7,
                        markeredgecolor="black", markeredgewidth=0.6,
                        linewidth=1.5, markerfacecolor="white",
                        zorder=10, label="_mean_trend")

            # Subplot labels
            if row_idx == 0:
                ax.set_title(label, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

            # X-axis: show K values explicitly
            ax.set_xlabel("Kuramoto K")
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([f"{k:g}" for k in K_VALUES])
            ax.set_xlim(K_VALUES[0] - 0.3, K_VALUES[-1] + 0.3)
            ax.grid(alpha=0.25)

    # Add legend for individual trial K colours (use the top-right subplot)
    legend_ax = axes[0, -1]
    legend_handles = []
    for k in K_VALUES:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=K_COLORS[k],
                       markersize=7, label=f"K = {k:g}")
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor="black", markeredgewidth=0.6,
                   markersize=7, label="Mean")
    )
    legend_ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                     title="Kuramoto K", title_fontsize=8.5,
                     ncol=2, framealpha=0.8)

    # Save
    fig.tight_layout()
    png_path = FIG_DIR / "figA2_k_sweep_frequency_std.png"
    pdf_path = FIG_DIR / "figA2_k_sweep_frequency_std.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")
    print("Done.")


if __name__ == "__main__":
    main()

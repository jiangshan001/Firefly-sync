#!/usr/bin/env python3
"""Analyse Appendix Step 5B Kuramoto K-sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCORE_FORMULA = (
    "score = mean(final_5s_mean_abs_error) "
    "+ 0.5*mean(virtual_final_5s_std + pi_final_5s_std) "
    "+ 2*mean(max(0, 0.90-FCR)) + 5*mean(max(0, FCR-1.02)) "
    "+ 0.02*mean(api_drops + capture_drops + short_interval_warnings "
    "+ frontend_warnings) + 10*(failed_or_invalid_trials/n_trials)"
)
FAILURE_MARKER = "trial_failed.json"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _truthy(value: Any) -> bool:
    return str(value).lower() in ("1", "true", "yes")


def _is_valid_completed_row(row: dict) -> bool:
    if _truthy(row.get("timeout_or_failure")) or _truthy(row.get("failed_or_invalid_trial")):
        return False
    trial_dir = row.get("trial_dir")
    if trial_dir and (Path(trial_dir) / FAILURE_MARKER).exists():
        return False
    return True


def _condition(row: dict) -> str:
    return f"V{float(row['virtual_initial_freq']):g}/P{float(row['pi_initial_freq']):g}"


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    return [v for v in (_as_float(row.get(key)) for row in rows) if v is not None]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None and math.isfinite(value) else None


def _score_group(rows: list[dict]) -> float | None:
    if not rows:
        return None
    err = _mean(_numeric_values(rows, "frequency_error_final_5s_mean_abs"))
    variability = _mean([
        (_as_float(row.get("virtual_freq_final_5s_std")) or 0.0)
        + (_as_float(row.get("pi_freq_final_5s_std")) or 0.0)
        for row in rows
    ])
    fcr_penalties = []
    for row in rows:
        fcr = _as_float(row.get("actual_detection_fcr"))
        if fcr is None:
            fcr_penalties.append(1.0)
        else:
            fcr_penalties.append(2.0 * max(0.0, 0.90 - fcr) + 5.0 * max(0.0, fcr - 1.02))
    warning_penalties = []
    for row in rows:
        warning_penalties.append(0.02 * sum(
            _as_float(row.get(key)) or 0.0
            for key in (
                "api_drop_count",
                "capture_drop_count",
                "short_interval_warning_count",
                "frontend_warning_count",
            )
        ))
    failure_rate = sum(1 for row in rows if _truthy(row.get("failed_or_invalid_trial"))) / len(rows)
    if err is None and all(_truthy(row.get("timeout_or_failure")) for row in rows):
        err = 10.0
    if err is None:
        return None
    return float(
        err
        + 0.5 * (variability or 0.0)
        + (_mean(fcr_penalties) or 0.0)
        + (_mean(warning_penalties) or 0.0)
        + 10.0 * failure_rate
    )


def _summarise(rows: list[dict], group_keys: list[str],
               include_score: bool = False) -> list[dict]:
    groups: dict[tuple[Any, ...], list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(k) for k in group_keys), []).append(row)
    metrics = [
        "actual_detection_fcr",
        "frequency_error_final_5s_mean_abs",
        "frequency_error_final_5s_abs_of_means",
        "virtual_freq_final_5s_std",
        "pi_freq_final_5s_std",
        "api_drop_count",
        "capture_drop_count",
        "short_interval_warning_count",
        "frontend_warning_count",
        "kuramoto_debug_warning_count",
    ]
    out_rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        out: dict[str, Any] = {name: value for name, value in zip(group_keys, key)}
        out["n_trials"] = len(group)
        out["failed_or_invalid_trials"] = sum(1 for r in group if _truthy(r.get("failed_or_invalid_trial")))
        out["failure_timeout_count"] = sum(1 for r in group if _truthy(r.get("timeout_or_failure")))
        for metric in metrics:
            vals = _numeric_values(group, metric)
            out[f"mean_{metric}"] = _round_or_none(_mean(vals))
            out[f"std_{metric}"] = _round_or_none(_std(vals))
        if include_score:
            out["appendix_ranking_score"] = _round_or_none(_score_group(group))
            out["score_formula"] = SCORE_FORMULA
        out_rows.append(out)
    return out_rows


def _ranking(rows: list[dict]) -> list[dict]:
    ranked = [
        row for row in _summarise(rows, ["kuramoto_K"], include_score=True)
        if _as_float(row.get("appendix_ranking_score")) is not None
    ]
    ranked.sort(key=lambda row: float(row["appendix_ranking_score"]))
    for idx, row in enumerate(ranked, 1):
        row["rank"] = idx
        row["best_overall"] = idx == 1
    return ranked


def _mean_by(rows: list[dict], x_key: str, metric: str,
             condition: str | None = None) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if condition is not None and _condition(row) != condition:
            continue
        x = _as_float(row.get(x_key))
        value = _as_float(row.get(metric))
        if x is not None and value is not None:
            grouped.setdefault(x, []).append(value)
    xs = sorted(grouped)
    means = [float(np.mean(grouped[x])) for x in xs]
    stds = [
        float(np.std(grouped[x], ddof=1)) if len(grouped[x]) > 1 else 0.0
        for x in xs
    ]
    return xs, means, stds


def _plot_metric_vs_k(rows: list[dict], metric: str, ylabel: str,
                      title: str, path: Path) -> None:
    conditions = sorted({_condition(row) for row in rows})
    if not conditions:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for condition in conditions:
        xs, means, stds = _mean_by(rows, "kuramoto_K", metric, condition)
        if xs:
            plotted = True
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=condition)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Kuramoto K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_frequency_std(rows: list[dict], fig_dir: Path) -> None:
    xs, v_mean, v_std = _mean_by(rows, "kuramoto_K", "virtual_freq_final_5s_std")
    xs2, p_mean, p_std = _mean_by(rows, "kuramoto_K", "pi_freq_final_5s_std")
    if not xs and not xs2:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if xs:
        ax.errorbar(xs, v_mean, yerr=v_std, marker="o", capsize=3, label="Virtual final-5s std")
    if xs2:
        ax.errorbar(xs2, p_mean, yerr=p_std, marker="s", capsize=3, label="Pi final-5s std")
    ax.set_xlabel("Kuramoto K")
    ax.set_ylabel("Frequency std (Hz)")
    ax.set_title("Final-5s Frequency Variability vs Kuramoto K")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "final5_frequency_std_vs_k.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_ranking(ranking: list[dict], fig_dir: Path) -> None:
    if not ranking:
        return
    labels = [str(row["kuramoto_K"]) for row in ranking]
    values = [float(row["appendix_ranking_score"]) for row in ranking]
    colors = ["#2c7fb8" if i == 0 else "#9ecae1" for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors)
    ax.set_xlabel("Kuramoto K")
    ax.set_ylabel("Appendix ranking score (lower is better)")
    ax.set_title("Best-K Ranking Summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_k_ranking_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_final_frequency_scatter(rows: list[dict], fig_dir: Path) -> None:
    data = []
    for row in rows:
        vx = _as_float(row.get("virtual_freq_final_5s_mean"))
        py = _as_float(row.get("pi_freq_final_5s_mean"))
        k = _as_float(row.get("kuramoto_K"))
        if vx is not None and py is not None and k is not None:
            data.append((vx, py, k))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5))
    scatter = ax.scatter([d[0] for d in data], [d[1] for d in data],
                         c=[d[2] for d in data], cmap="viridis", edgecolor="black")
    lo = min(min(d[0], d[1]) for d in data) - 0.05
    hi = max(max(d[0], d[1]) for d in data) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Virtual final-5s mean frequency (Hz)")
    ax.set_ylabel("Pi final-5s mean frequency (Hz)")
    ax.set_title("Final Virtual vs Pi Frequency")
    fig.colorbar(scatter, ax=ax, label="Kuramoto K")
    fig.tight_layout()
    fig.savefig(fig_dir / "final_virtual_vs_pi_frequency_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_k_vs_eapf(rows: list[dict], fig_dir: Path) -> None:
    models = sorted({row.get("model") for row in rows})
    if "eapf_consensus" not in models or "kuramoto" not in models:
        return
    metric = "frequency_error_final_5s_mean_abs"
    conditions = sorted({_condition(row) for row in rows})
    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model in enumerate(["eapf_consensus", "kuramoto"]):
        means = []
        stds = []
        for condition in conditions:
            vals = [
                _as_float(row.get(metric))
                for row in rows
                if row.get("model") == model and _condition(row) == condition
            ]
            vals = [v for v in vals if v is not None]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if vals else np.nan)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right")
    ax.set_ylabel("Final-5s mean abs frequency error (Hz)")
    ax.set_title("Best-K Kuramoto vs EAPF Confirmation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "best_k_vs_eapf_final5_error.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a Step 5B Kuramoto K-sweep run directory.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    aggregate = run_dir / "aggregate_metrics.csv"
    if not aggregate.exists():
        raise SystemExit(f"aggregate_metrics.csv not found: {aggregate}")

    all_rows = _read_csv(aggregate)
    rows = [row for row in all_rows if _is_valid_completed_row(row)]
    excluded = len(all_rows) - len(rows)
    if excluded:
        print(f"Excluded {excluded} failed/invalid trial row(s) from analysis.")
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    k_rows = [row for row in rows if row.get("model") == "kuramoto" and row.get("kuramoto_K", "") != ""]
    if k_rows:
        summary_condition = _summarise(
            k_rows,
            ["kuramoto_K", "virtual_initial_freq", "pi_initial_freq"],
            include_score=True,
        )
        summary_overall = _summarise(k_rows, ["kuramoto_K"], include_score=True)
        ranking = _ranking(k_rows)
        _save_csv(run_dir / "summary_by_k_condition.csv", summary_condition)
        _save_csv(run_dir / "summary_by_k_overall.csv", summary_overall)
        _save_csv(run_dir / "k_ranking.csv", ranking)

        _plot_metric_vs_k(
            k_rows,
            "frequency_error_final_5s_mean_abs",
            "Final-5s mean abs frequency error (Hz)",
            "Final-5s Mean Abs Error vs Kuramoto K",
            fig_dir / "final5_mean_abs_error_vs_k.png",
        )
        _plot_frequency_std(k_rows, fig_dir)
        _plot_metric_vs_k(
            k_rows,
            "actual_detection_fcr",
            "Actual detection FCR",
            "Actual Detection FCR vs Kuramoto K",
            fig_dir / "actual_detection_fcr_vs_k.png",
        )
        _plot_ranking(ranking, fig_dir)
        _plot_final_frequency_scatter(k_rows, fig_dir)

    if len({row.get("model") for row in rows}) > 1:
        _save_csv(run_dir / "summary_by_model_condition.csv",
                  _summarise(rows, ["model", "virtual_initial_freq", "pi_initial_freq"]))
        _save_csv(run_dir / "summary_by_model_overall.csv",
                  _summarise(rows, ["model"]))
        _plot_best_k_vs_eapf(rows, fig_dir)

    print(f"Analysis written to {run_dir}")
    print(f"Figures written to {fig_dir}")


if __name__ == "__main__":
    main()

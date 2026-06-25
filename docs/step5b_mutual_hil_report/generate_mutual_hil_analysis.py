#!/usr/bin/env python3
"""Generate mutual visual HIL report figures and tables.

Generates figures/tables for the main EAPF-vs-Kuramoto comparison,
the Kuramoto K-sensitivity appendix, and the best-K vs EAPF comparison.
Read-only with respect to experiment logs.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[1]

# Main mutual HIL formal dataset
MAIN_DATA_DIR = (
    REPO_ROOT
    / "experiments"
    / "logs"
    / "step5b_formal_chunked"
    / "formal_step5b_chunked_20260621"
)

# Appendix K-sweep dataset
K_SWEEP_DIR = (
    REPO_ROOT
    / "experiments"
    / "logs"
    / "step5b_kuramoto_k_sensitivity"
    / "k_sweep_20260621_v4"
)

FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
DERIVED_DIR = REPORT_DIR / "derived"

MODEL_ORDER = ["eapf_consensus", "kuramoto"]
MODEL_LABEL = {
    "eapf_consensus": "EAPF Consensus",
    "kuramoto": "Kuramoto",
}
MODEL_COLOR = {
    "eapf_consensus": "#2f6db3",
    "kuramoto": "#c96b2c",
}
CONDITION_ORDER = [1.2, 1.5, 2.5]
K_VALUES = [2.5, 3.0, 3.5, 4.0, 4.5]
K_COLORS = {
    2.5: "#1b9e77",
    3.0: "#d95f02",
    3.5: "#7570b3",
    4.0: "#e7298a",
    4.5: "#66a61e",
}


def _read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _escape_latex(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _fmt(value: Any, digits: int = 3) -> str:
    v = _safe_float(value)
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def _mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _std(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) else float("nan")


def _model_label(model: str) -> str:
    return MODEL_LABEL.get(model, model)


def _condition_label(freq: float) -> str:
    return f"Pi {freq:g} Hz"


def _load_main_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, list[dict]]:
    if not MAIN_DATA_DIR.exists():
        raise FileNotFoundError(f"Main mutual HIL data directory not found: {MAIN_DATA_DIR}")
    aggregate = pd.read_csv(MAIN_DATA_DIR / "aggregate_metrics.csv")
    summary = pd.read_csv(MAIN_DATA_DIR / "summary_by_model_condition.csv")
    batch_metadata = _read_json(MAIN_DATA_DIR / "batch_metadata.json")
    run_metadata = _read_json(MAIN_DATA_DIR / "run_metadata.json")
    chunk_metadata = _read_json(MAIN_DATA_DIR / "chunk_metadata.json")
    return aggregate, summary, batch_metadata, run_metadata, chunk_metadata


def _load_k_sweep_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    if not K_SWEEP_DIR.exists():
        raise FileNotFoundError(f"K-sweep data directory not found: {K_SWEEP_DIR}")
    aggregate = pd.read_csv(K_SWEEP_DIR / "aggregate_metrics.csv")
    overall = pd.read_csv(K_SWEEP_DIR / "summary_by_k_overall.csv")
    condition = pd.read_csv(K_SWEEP_DIR / "summary_by_k_condition.csv")
    ranking = pd.read_csv(K_SWEEP_DIR / "k_ranking.csv")
    best_k = pd.read_csv(K_SWEEP_DIR / "best_k_by_condition.csv")
    batch_meta = _read_json(K_SWEEP_DIR / "batch_metadata.json")
    chunk_meta = _read_json(K_SWEEP_DIR / "chunk_metadata.json")
    return aggregate, overall, condition, ranking, best_k, batch_meta, chunk_meta


def _trial_dir_main(row: pd.Series) -> Path:
    trial_id = str(row["trial_id"])
    direct = MAIN_DATA_DIR / trial_id
    if direct.exists():
        return direct
    logged = REPO_ROOT / str(row.get("trial_dir", ""))
    if logged.exists():
        return logged
    raise FileNotFoundError(f"Cannot find trial directory for {trial_id}")


def _validate_main_trials(aggregate: pd.DataFrame, chunk_metadata: list[dict]) -> pd.DataFrame:
    expected = []
    for repeat in [1, 2, 3]:
        for pi_freq in CONDITION_ORDER:
            for model in MODEL_ORDER:
                expected.append(f"{model}_V2_P{pi_freq:g}_r{repeat:02d}")

    present = set(aggregate["trial_id"].astype(str))
    completed_from_chunks = set()
    failed_from_chunks = set()
    for chunk in chunk_metadata:
        completed_from_chunks.update(chunk.get("completed_trial_ids", []))
        failed_from_chunks.update(chunk.get("failed_trial_ids", []))

    rows = []
    for trial_id in expected:
        row = aggregate[aggregate["trial_id"] == trial_id]
        exists = len(row) == 1
        trial_path = MAIN_DATA_DIR / trial_id
        metrics = trial_path / "metrics_summary.json"
        metadata = trial_path / "metadata.json"
        status = "included" if exists and metrics.exists() and metadata.exists() else "missing"
        if exists:
            timeout = str(row.iloc[0].get("timeout_or_failure", "False")).lower() == "true"
            if timeout:
                status = "included_failed_flag"
        rows.append({
            "trial_id": trial_id,
            "model": trial_id.split("_V", 1)[0],
            "pi_initial_freq": float(trial_id.split("_P", 1)[1].split("_r", 1)[0]),
            "repeat": int(trial_id.rsplit("_r", 1)[1]),
            "in_aggregate": exists,
            "metrics_summary": metrics.exists(),
            "metadata": metadata.exists(),
            "chunk_completed": trial_id in completed_from_chunks,
            "chunk_failed": trial_id in failed_from_chunks,
            "status": status,
        })
    return pd.DataFrame(rows)


def _diagnostics_main(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in aggregate.iterrows():
        trial_id = row["trial_id"]
        tdir = _trial_dir_main(row)
        duration = float(row["duration"])

        detection = pd.read_csv(tdir / "detection_log.csv") if (tdir / "detection_log.csv").exists() else pd.DataFrame()
        api = pd.read_csv(tdir / "api_events.csv") if (tdir / "api_events.csv").exists() else pd.DataFrame()

        capture_fps = len(detection) / duration if duration > 0 and len(detection) else float("nan")
        proc_ms = pd.to_numeric(detection.get("camera_processing_time_ms", pd.Series(dtype=float)), errors="coerce")
        processing_fps = 1000.0 / proc_ms.mean() if len(proc_ms.dropna()) and proc_ms.mean() > 0 else float("nan")
        detector_events = int(pd.to_numeric(detection.get("virtual_event", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(detection) else 0
        on_samples = int(pd.to_numeric(detection.get("state", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(detection) else 0

        api_ok = 0
        api_total = 0
        post_latency_mean = float("nan")
        post_latency_p95 = float("nan")
        if len(api):
            api_total = len(api)
            ok_values = pd.to_numeric(api.get("ok", pd.Series(dtype=float)), errors="coerce").fillna(0)
            api_ok = int(ok_values.sum())
            posts = api[api["endpoint"].astype(str).str.contains("POST /api/pi_flash", regex=False, na=False)]
            lat = pd.to_numeric(posts.get("elapsed_ms", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(lat):
                post_latency_mean = float(lat.mean())
                post_latency_p95 = float(lat.quantile(0.95))

        rows.append({
            "trial_id": trial_id,
            "model": row["model"],
            "pi_initial_freq": float(row["pi_initial_freq"]),
            "capture_fps": capture_fps,
            "processing_fps": processing_fps,
            "api_post_latency_mean_ms": post_latency_mean,
            "api_post_latency_p95_ms": post_latency_p95,
            "api_ok_ratio": api_ok / api_total if api_total else float("nan"),
            "detector_events": detector_events,
            "on_samples": on_samples,
            "loop_rate_hz": _safe_float(row.get("loop_rate_hz")),
            "duplicate_suppressed_count": float("nan"),
        })
    return pd.DataFrame(rows)


def _overall_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        sub = aggregate[aggregate["model"] == model]
        rows.append({
            "model": model,
            "n_trials": len(sub),
            "actual_fcr_mean": _mean(sub["actual_detection_fcr"]),
            "actual_fcr_std": _std(sub["actual_detection_fcr"]),
            "final5_mae_mean": _mean(sub["frequency_error_final_5s_mean_abs"]),
            "final5_mae_std": _std(sub["frequency_error_final_5s_mean_abs"]),
            "final5_abs_means_mean": _mean(sub["frequency_error_final_5s_abs_of_means"]),
            "virtual_freq_std_mean": _mean(sub["virtual_freq_final_5s_std"]),
            "pi_freq_std_mean": _mean(sub["pi_freq_final_5s_std"]),
            "pi_flash_count_mean": _mean(sub["pi_flash_count"]),
            "actual_virtual_flashes_mean": _mean(sub["actual_virtual_flashes"]),
            "detected_virtual_flashes_mean": _mean(sub["pi_detected_virtual_flashes"]),
        })
    return pd.DataFrame(rows)


def _condition_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    grouped = aggregate.groupby(["model", "pi_initial_freq"], sort=False)
    rows = []
    for (model, pi_freq), sub in grouped:
        rows.append({
            "model": model,
            "pi_initial_freq": pi_freq,
            "n_trials": len(sub),
            "actual_fcr_mean": _mean(sub["actual_detection_fcr"]),
            "actual_fcr_std": _std(sub["actual_detection_fcr"]),
            "final5_mae_mean": _mean(sub["frequency_error_final_5s_mean_abs"]),
            "final5_mae_std": _std(sub["frequency_error_final_5s_mean_abs"]),
            "final5_abs_means_mean": _mean(sub["frequency_error_final_5s_abs_of_means"]),
            "virtual_final_mean": _mean(sub["virtual_freq_final_5s_mean"]),
            "pi_final_mean": _mean(sub["pi_freq_final_5s_mean"]),
            "virtual_freq_std_mean": _mean(sub["virtual_freq_final_5s_std"]),
            "pi_freq_std_mean": _mean(sub["pi_freq_final_5s_std"]),
        })
    out = pd.DataFrame(rows)
    out["model_order"] = out["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["pi_initial_freq", "model_order"]).drop(columns=["model_order"])


def _write_booktabs_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]], align: str | None = None) -> None:
    align = align or ("l" + "r" * (len(headers) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_escape_latex(h)}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_main_tables(aggregate: pd.DataFrame, validation: pd.DataFrame, diagnostics: pd.DataFrame, batch_meta: dict, chunk_meta: list[dict]) -> None:
    overall = _overall_summary(aggregate)
    condition = _condition_summary(aggregate)

    _write_booktabs_table(
        TABLE_DIR / "table_overall_summary.tex",
        "Mutual HIL overall results across all formal trials (1 virtual + 1 Pi, 30 s, 3 repeats).",
        "tab:overall",
        ["Model", "Trials", "Actual FCR", "5 s MAE", "Mean diff.", "Virt. std", "Pi std"],
        [
            [
                _escape_latex(_model_label(r["model"])),
                str(int(r["n_trials"])),
                f"{r['actual_fcr_mean']:.3f} $\\pm$ {r['actual_fcr_std']:.3f}",
                f"{r['final5_mae_mean']:.3f} $\\pm$ {r['final5_mae_std']:.3f}",
                f"{r['final5_abs_means_mean']:.3f}",
                f"{r['virtual_freq_std_mean']:.3f}",
                f"{r['pi_freq_std_mean']:.3f}",
            ]
            for _, r in overall.iterrows()
        ],
        align="lrrrrrr",
    )

    _write_booktabs_table(
        TABLE_DIR / "table_condition_summary.tex",
        "Mean performance by model and Pi initial frequency condition. Frequency metrics are in Hz.",
        "tab:condition_summary",
        ["Pi", "Model", "Trials", "FCR", "5 s MAE", "Mean diff.", "Virt. final", "Pi final"],
        [
            [
                f"{r['pi_initial_freq']:.1f} Hz",
                _escape_latex("EAPF" if r["model"] == "eapf_consensus" else _model_label(r["model"])),
                str(int(r["n_trials"])),
                f"{r['actual_fcr_mean']:.3f} $\\pm$ {r['actual_fcr_std']:.3f}",
                f"{r['final5_mae_mean']:.3f} $\\pm$ {r['final5_mae_std']:.3f}",
                f"{r['final5_abs_means_mean']:.3f}",
                f"{r['virtual_final_mean']:.3f}",
                f"{r['pi_final_mean']:.3f}",
            ]
            for _, r in condition.iterrows()
        ],
        align="llrrrrrr",
    )

    params = batch_meta["locked_model_parameters"]
    _write_booktabs_table(
        TABLE_DIR / "table_parameters.tex",
        "Locked model parameters used in the formal mutual HIL comparison.",
        "tab:parameters",
        ["Model", "Parameters"],
        [
            [
                "EAPF Consensus",
                (
                    f"$g_p={params['eapf_consensus']['g_p']}$, "
                    f"$g_f={params['eapf_consensus']['g_f']}$, "
                    f"$\\alpha_p=\\alpha_f={params['eapf_consensus']['alpha_p']}$, "
                    f"$\\Delta\\theta_{{\\mathrm{{max}}}}={params['eapf_consensus']['delta_theta_max_rad']}$ rad, "
                    f"$\\Delta f_{{\\mathrm{{max}}}}={params['eapf_consensus']['delta_f_max_hz']}$ Hz"
                ),
            ],
            ["Kuramoto", f"$K={params['kuramoto']['K']}$"],
        ],
        align="lp{10.5cm}",
    )

    chunks = []
    for chunk in chunk_meta:
        freqs = ", ".join(f"{p['virtual']:g}:{p['pi']:g}" for p in chunk.get("freq_pairs", []))
        chunks.append([
            _escape_latex(chunk.get("chunk_label", chunk.get("chunk_id", ""))),
            _escape_latex(freqs),
            str(len(chunk.get("completed_trial_ids", []))),
            str(len(chunk.get("failed_trial_ids", []))),
            _escape_latex(chunk.get("started_at", "")),
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_chunk_validation.tex",
        "Formal chunk validation records from chunk metadata.",
        "tab:chunks",
        ["Chunk", "Freq pairs", "Completed", "Failed", "Started"],
        chunks,
        align="llrrl",
    )

    val_rows = []
    for _, r in validation.iterrows():
        val_rows.append([
            _escape_latex(r["trial_id"]).replace(r"\_", r"\_\allowbreak{}"),
            _escape_latex("EAPF" if r["model"] == "eapf_consensus" else _model_label(r["model"])),
            f"{r['pi_initial_freq']:.1f}",
            str(int(r["repeat"])),
            "yes" if r["in_aggregate"] else "no",
            "yes" if r["metrics_summary"] else "no",
            _escape_latex(r["status"]),
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_trial_validation.tex",
        "Per-trial validation for the formal mutual HIL dataset.",
        "tab:trial_validation",
        ["Trial", "Model", "Pi Hz", "Rep.", "Aggregate", "Metrics", "Status"],
        val_rows,
        align="p{4.55cm}p{2.1cm}rrlll",
    )

    diag = diagnostics.groupby("model").agg({
        "capture_fps": ["mean", "std"],
        "processing_fps": ["mean", "std"],
        "api_post_latency_mean_ms": ["mean", "std"],
        "api_ok_ratio": ["mean"],
        "loop_rate_hz": ["mean"],
        "detector_events": ["mean"],
    })
    diag_rows = []
    for model in MODEL_ORDER:
        r = diag.loc[model]
        diag_rows.append([
            _escape_latex(_model_label(model)),
            f"{r[('capture_fps','mean')]:.1f} $\\pm$ {r[('capture_fps','std')]:.1f}",
            f"{r[('processing_fps','mean')]:.1f} $\\pm$ {r[('processing_fps','std')]:.1f}",
            f"{r[('api_post_latency_mean_ms','mean')]:.1f} $\\pm$ {r[('api_post_latency_mean_ms','std')]:.1f}",
            f"{r[('api_ok_ratio','mean')]:.3f}",
            f"{r[('loop_rate_hz','mean')]:.2f}",
            f"{r[('detector_events','mean')]:.1f}",
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_runtime_diagnostics.tex",
        "Runtime and detection diagnostics derived from per-trial logs.",
        "tab:runtime",
        ["Model", "Cap. FPS", "Proc. FPS", "POST ms", "API OK", "Loop Hz", "Events"],
        diag_rows,
        align="lrrrrrr",
    )

    overall.to_csv(DERIVED_DIR / "overall_summary.csv", index=False)
    condition.to_csv(DERIVED_DIR / "condition_summary.csv", index=False)
    validation.to_csv(DERIVED_DIR / "trial_validation.csv", index=False)
    diagnostics.to_csv(DERIVED_DIR / "runtime_diagnostics.csv", index=False)


def _bar_grouped(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str, title: str, yline: float | None = None) -> None:
    x = np.arange(len(CONDITION_ORDER))
    width = 0.34
    for idx, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for cond in CONDITION_ORDER:
            sub = df[(df["model"] == model) & (df["pi_initial_freq"] == cond)]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        offset = (idx - 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=_model_label(model), color=MODEL_COLOR[model], alpha=0.9)
        for cond_idx, cond in enumerate(CONDITION_ORDER):
            sub = df[(df["model"] == model) & (df["pi_initial_freq"] == cond)]
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
            jitter = np.linspace(-0.055, 0.055, len(vals)) if len(vals) else []
            ax.scatter(np.full(len(vals), x[cond_idx] + offset) + jitter, vals, color="black", s=14, alpha=0.65, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([_condition_label(c) for c in CONDITION_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if yline is not None:
        ax.axhline(yline, color="0.3", linestyle="--", linewidth=1.0)


def _savefig(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_main_figures(aggregate: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Figure 1: Combined robustness overview (FCR + final 5 s MAE) — kept as the sole overview
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    _bar_grouped(axes[0], aggregate, "actual_detection_fcr", "Actual virtual FCR", "Detection robustness", yline=0.85)
    _bar_grouped(axes[1], aggregate, "frequency_error_final_5s_mean_abs", "Final 5 s MAE (Hz)", "Frequency agreement")
    axes[0].legend(loc="lower left")
    _savefig(fig, "fig1_robustness_overview")

    # Figure 2: Virtual final frequency (was fig4)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    _bar_grouped(ax, aggregate, "virtual_freq_final_5s_mean", "Virtual final frequency (Hz)", "Virtual final frequency")
    ax.axhline(2.0, color="0.35", linestyle=":", linewidth=1, label="2.0 Hz initial virtual")
    ax.legend()
    _savefig(fig, "fig2_virtual_final_frequency")

    # Figure 3: Pi final frequency (was fig5)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    _bar_grouped(ax, aggregate, "pi_freq_final_5s_mean", "Pi final frequency (Hz)", "Pi final frequency")
    ax.legend()
    _savefig(fig, "fig3_pi_final_frequency")

    # Figure 4: Final frequency agreement scatter (was fig6)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for model in MODEL_ORDER:
        sub = aggregate[aggregate["model"] == model]
        ax.scatter(
            pd.to_numeric(sub["virtual_freq_final_5s_mean"], errors="coerce"),
            pd.to_numeric(sub["pi_freq_final_5s_mean"], errors="coerce"),
            s=56,
            color=MODEL_COLOR[model],
            label=_model_label(model),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
    lo = min(aggregate["virtual_freq_final_5s_mean"].min(), aggregate["pi_freq_final_5s_mean"].min()) - 0.05
    hi = max(aggregate["virtual_freq_final_5s_mean"].max(), aggregate["pi_freq_final_5s_mean"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.35", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Virtual final 5 s mean frequency (Hz)")
    ax.set_ylabel("Pi final 5 s mean frequency (Hz)")
    ax.set_title("Final frequency agreement")
    ax.grid(alpha=0.25)
    ax.legend()
    _savefig(fig, "fig4_frequency_agreement_scatter")

    # Figure 5: Initial phase sensitivity (was fig7)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for model in MODEL_ORDER:
        sub = aggregate[aggregate["model"] == model]
        ax.scatter(
            pd.to_numeric(sub["initial_phase_difference_rad"], errors="coerce"),
            pd.to_numeric(sub["frequency_error_final_5s_mean_abs"], errors="coerce"),
            color=MODEL_COLOR[model],
            label=_model_label(model),
            s=48,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_xlabel("Initial phase difference (rad)")
    ax.set_ylabel("Final 5 s MAE (Hz)")
    ax.set_title("Initial phase difference vs final error")
    ax.grid(alpha=0.25)
    ax.legend()
    _savefig(fig, "fig5_initial_phase_sensitivity")

    # Figure 6: Representative frequency trajectories (was fig8)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)
    reps = ["eapf_consensus_V2_P2.5_r03", "kuramoto_V2_P2.5_r03"]
    for ax, tid in zip(axes, reps):
        row = aggregate[aggregate["trial_id"] == tid].iloc[0]
        osc = pd.read_csv(_trial_dir_main(row) / "oscillator_log.csv")
        ax.plot(osc["t_s"], osc["virtual_frequency_hz"], label="Virtual", color="#2f6db3", linewidth=1.4)
        ax.plot(osc["t_s"], osc["frequency_hz"], label="Pi", color="#c96b2c", linewidth=1.1, alpha=0.85)
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"{_model_label(row['model'])}: Pi start {float(row['pi_initial_freq']):.1f} Hz, repeat {int(row['repeat'])}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    _savefig(fig, "fig6_representative_frequency_timeseries")

    # Figure 7: Runtime diagnostics (was fig9)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    diag_metrics = [
        ("capture_fps", "Capture FPS"),
        ("processing_fps", "Processing FPS"),
        ("api_post_latency_mean_ms", "POST latency (ms)"),
    ]
    for ax, (metric, title) in zip(axes, diag_metrics):
        positions = np.arange(len(MODEL_ORDER))
        data = [pd.to_numeric(diagnostics[diagnostics["model"] == m][metric], errors="coerce").dropna().to_numpy() for m in MODEL_ORDER]
        bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True)
        for patch, model in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLOR[model])
            patch.set_alpha(0.75)
        ax.set_xticks(positions)
        ax.set_xticklabels([_model_label(m).replace(" ", "\n") for m in MODEL_ORDER])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    _savefig(fig, "fig7_runtime_diagnostics")

    # Figure 8: Flash counts (was fig10)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    count_metrics = [
        ("pi_flash_count", "Pi flash count"),
        ("pi_detected_virtual_flashes", "Detected virtual flashes"),
        ("actual_virtual_flashes", "Actual virtual flashes"),
    ]
    for ax, (metric, title) in zip(axes, count_metrics):
        _bar_grouped(ax, aggregate, metric, "Count", title)
        ax.legend().remove()
    axes[0].legend(loc="upper left")
    _savefig(fig, "fig8_flash_counts")


# ---------------------------------------------------------------------------
# Appendix figures: K-sweep
# ---------------------------------------------------------------------------

def _plot_k_sweep_figures(k_aggregate: pd.DataFrame, k_overall: pd.DataFrame, k_condition: pd.DataFrame, k_ranking: pd.DataFrame) -> None:
    """Generate K-sensitivity appendix figures."""

    # Appendix Figure A1: Final 5 s MAE vs K
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax_idx, (pi_freq, label) in enumerate(zip(CONDITION_ORDER, ["Pi 1.2 Hz", "Pi 1.5 Hz", "Pi 2.5 Hz"])):
        ax = axes[ax_idx]
        sub = k_aggregate[k_aggregate["pi_initial_freq"] == pi_freq]
        for k in K_VALUES:
            ksub = sub[sub["kuramoto_K"] == k]
            vals = pd.to_numeric(ksub["frequency_error_final_5s_mean_abs"], errors="coerce").dropna().to_numpy()
            if len(vals):
                ax.plot([k] * len(vals), vals, "o", color=K_COLORS[k], alpha=0.7, markersize=7)
                ax.plot(k, vals.mean(), "D", color=K_COLORS[k], markersize=8, markeredgecolor="black", markeredgewidth=0.5)
        ax.set_title(label)
        ax.set_xlabel("Kuramoto K")
        ax.set_ylabel("Final 5 s MAE (Hz)")
        ax.grid(alpha=0.25)
    _savefig(fig, "figA1_k_sweep_final5_mae")

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

    _savefig(fig, "figA2_k_sweep_frequency_std")

    # Appendix Figure A3: Actual FCR vs K
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax_idx, (pi_freq, label) in enumerate(zip(CONDITION_ORDER, ["Pi 1.2 Hz", "Pi 1.5 Hz", "Pi 2.5 Hz"])):
        ax = axes[ax_idx]
        sub = k_aggregate[k_aggregate["pi_initial_freq"] == pi_freq]
        for k in K_VALUES:
            ksub = sub[sub["kuramoto_K"] == k]
            vals = pd.to_numeric(ksub["actual_detection_fcr"], errors="coerce").dropna().to_numpy()
            if len(vals):
                ax.plot([k] * len(vals), vals, "o", color=K_COLORS[k], alpha=0.7, markersize=7)
                ax.plot(k, vals.mean(), "D", color=K_COLORS[k], markersize=8, markeredgecolor="black", markeredgewidth=0.5)
        ax.axhline(0.85, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Kuramoto K")
        ax.set_ylabel("Actual virtual FCR")
        ax.grid(alpha=0.25)
    _savefig(fig, "figA3_k_sweep_actual_fcr")

    # Appendix Figure A4: Ranking score vs K
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    k_vals_ordered = [float(r["kuramoto_K"]) for _, r in k_ranking.iterrows()]
    scores = [float(r["appendix_ranking_score"]) for _, r in k_ranking.iterrows()]
    colors = [K_COLORS.get(k, "#333333") for k in k_vals_ordered]
    ax.bar(range(len(k_vals_ordered)), scores, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(k_vals_ordered)))
    ax.set_xticklabels([f"K={k:g}" for k in k_vals_ordered])
    ax.set_ylabel("Appendix ranking score (lower is better)")
    ax.set_title("Kuramoto K ranking (appendix sweep)")
    ax.grid(axis="y", alpha=0.25)
    # Add best marker
    best_idx = scores.index(min(scores))
    ax.annotate("Best", (best_idx, scores[best_idx]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold", color="#1b9e77")
    _savefig(fig, "figA4_k_sweep_ranking_score")


def _write_k_sweep_tables(k_overall: pd.DataFrame, k_condition: pd.DataFrame, k_ranking: pd.DataFrame, k_best: pd.DataFrame) -> None:
    """Generate K-sweep appendix LaTeX tables."""

    # Overall K-sweep summary table
    headers = ["K", "Trials", "FCR", "5 s MAE", "Mean diff.", "Virt. std", "Pi std", "Score"]
    rows = []
    for _, r in k_overall.iterrows():
        rows.append([
            f"{r['kuramoto_K']:.1f}",
            str(int(r["n_trials"])),
            f"{r['mean_actual_detection_fcr']:.3f}",
            f"{r['mean_frequency_error_final_5s_mean_abs']:.3f}",
            f"{r['mean_frequency_error_final_5s_abs_of_means']:.3f}",
            f"{r['mean_virtual_freq_final_5s_std']:.3f}",
            f"{r['mean_pi_freq_final_5s_std']:.3f}",
            f"{r['appendix_ranking_score']:.3f}",
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_k_sweep_overall.tex",
        "Kuramoto K-sweep: overall results across all Pi initial frequencies (2 repeats per condition).",
        "tab:k_sweep_overall",
        headers,
        rows,
        align="rrrrrrrr",
    )

    # K-sweep by condition table
    headers = ["Pi", "K", "Trials", "FCR", "5 s MAE", "Virt. std", "Pi std", "Score"]
    rows = []
    for _, r in k_condition.iterrows():
        rows.append([
            f"{r['pi_initial_freq']:.1f} Hz",
            f"{r['kuramoto_K']:.1f}",
            str(int(r["n_trials"])),
            f"{r['mean_actual_detection_fcr']:.3f}",
            f"{r['mean_frequency_error_final_5s_mean_abs']:.3f}",
            f"{r['mean_virtual_freq_final_5s_std']:.3f}",
            f"{r['mean_pi_freq_final_5s_std']:.3f}",
            f"{r['appendix_ranking_score']:.3f}",
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_k_sweep_condition.tex",
        "Kuramoto K-sweep results by Pi initial frequency condition.",
        "tab:k_sweep_condition",
        headers,
        rows,
        align="lrrrrrrr",
    )

    # K ranking table
    headers = ["Rank", "K", "Score", "5 s MAE", "Fan-out warnings"]
    rows = []
    for idx, (_, r) in enumerate(k_ranking.iterrows()):
        rows.append([
            str(idx + 1),
            f"{r['kuramoto_K']:.1f}",
            f"{r['appendix_ranking_score']:.3f}",
            f"{r['mean_frequency_error_final_5s_mean_abs']:.3f}",
            f"{r['mean_frontend_warning_count']:.1f}",
        ])
    _write_booktabs_table(
        TABLE_DIR / "table_k_ranking.tex",
        "Kuramoto K ranking from the appendix sensitivity sweep. Lower score is better.",
        "tab:k_ranking",
        headers,
        rows,
        align="rrrrr",
    )


# ---------------------------------------------------------------------------
# Best-K (2.5) vs EAPF comparison
# ---------------------------------------------------------------------------

def _plot_best_k_vs_eapf(k_aggregate: pd.DataFrame, eapf_aggregate: pd.DataFrame) -> None:
    """Generate comparison figures: EAPF vs best Kuramoto K=2.5."""

    k25 = k_aggregate[k_aggregate["kuramoto_K"] == 2.5].copy()
    k25["model"] = "kuramoto_K2.5"
    eapf = eapf_aggregate[eapf_aggregate["model"] == "eapf_consensus"].copy()

    # Merge for comparison plot
    combined = pd.concat([eapf, k25], ignore_index=True)

    model_order_best = ["eapf_consensus", "kuramoto_K2.5"]
    model_label_best = {
        "eapf_consensus": "EAPF Consensus",
        "kuramoto_K2.5": "Kuramoto K=2.5",
    }
    model_color_best = {
        "eapf_consensus": "#2f6db3",
        "kuramoto_K2.5": "#1b9e77",
    }

    def _bar_grouped_best(ax, df, metric, ylabel, title, yline=None):
        x = np.arange(len(CONDITION_ORDER))
        width = 0.34
        for idx, model in enumerate(model_order_best):
            means = []
            stds = []
            for cond in CONDITION_ORDER:
                sub = df[(df["model"] == model) & (df["pi_initial_freq"] == cond)]
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
            offset = (idx - 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=model_label_best[model], color=model_color_best[model], alpha=0.9)
            for cond_idx, cond in enumerate(CONDITION_ORDER):
                sub = df[(df["model"] == model) & (df["pi_initial_freq"] == cond)]
                vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy()
                jitter = np.linspace(-0.055, 0.055, len(vals)) if len(vals) else []
                ax.scatter(np.full(len(vals), x[cond_idx] + offset) + jitter, vals, color="black", s=14, alpha=0.65, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_condition_label(c) for c in CONDITION_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if yline is not None:
            ax.axhline(yline, color="0.3", linestyle="--", linewidth=1.0)

    # Appendix Figure B1: EAPF vs K=2.5 overview
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    _bar_grouped_best(axes[0], combined, "actual_detection_fcr", "Actual virtual FCR", "Detection robustness", yline=0.85)
    _bar_grouped_best(axes[1], combined, "frequency_error_final_5s_mean_abs", "Final 5 s MAE (Hz)", "Frequency agreement")
    axes[0].legend(loc="lower left")
    _savefig(fig, "figB1_eapf_vs_k25_overview")

    # Appendix Figure B2: Frequency std comparison
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    _bar_grouped_best(axes[0], combined, "virtual_freq_final_5s_std", "Virtual freq std (Hz)", "Virtual final-window std")
    _bar_grouped_best(axes[1], combined, "pi_freq_final_5s_std", "Pi freq std (Hz)", "Pi final-window std")
    axes[0].legend(loc="upper left")
    _savefig(fig, "figB2_eapf_vs_k25_frequency_std")


def _write_best_k_vs_eapf_table(k_aggregate: pd.DataFrame, eapf_aggregate: pd.DataFrame) -> None:
    """Generate EAPF vs K=2.5 comparison LaTeX table."""
    k25 = k_aggregate[k_aggregate["kuramoto_K"] == 2.5]

    rows = []
    for pi_freq in CONDITION_ORDER:
        ksub = k25[k25["pi_initial_freq"] == pi_freq]
        esub = eapf_aggregate[(eapf_aggregate["model"] == "eapf_consensus") & (eapf_aggregate["pi_initial_freq"] == pi_freq)]

        k_mae = _mean(ksub["frequency_error_final_5s_mean_abs"])
        e_mae = _mean(esub["frequency_error_final_5s_mean_abs"])
        k_vstd = _mean(ksub["virtual_freq_final_5s_std"])
        e_vstd = _mean(esub["virtual_freq_final_5s_std"])
        k_pstd = _mean(ksub["pi_freq_final_5s_std"])
        e_pstd = _mean(esub["pi_freq_final_5s_std"])
        k_fcr = _mean(ksub["actual_detection_fcr"])
        e_fcr = _mean(esub["actual_detection_fcr"])

        rows.append([
            f"{pi_freq:.1f} Hz",
            str(int(len(esub))),
            str(int(len(ksub))),
            f"{e_mae:.3f}  /  {k_mae:.3f}",
            f"{e_vstd:.3f}  /  {k_vstd:.3f}",
            f"{e_pstd:.3f}  /  {k_pstd:.3f}",
            f"{e_fcr:.3f}  /  {k_fcr:.3f}",
        ])

    _write_booktabs_table(
        TABLE_DIR / "table_best_k_vs_eapf.tex",
        "EAPF Consensus (main) versus best-appendix Kuramoto K=2.5. "
        "Values shown as ``EAPF / K=2.5''. Sample sizes differ: EAPF 3 repeats, K=2.5 has 2 repeats.",
        "tab:best_k_vs_eapf",
        ["Pi", "N (E/K)", "5 s MAE (Hz)", "Virt. std (Hz)", "Pi std (Hz)", "Actual FCR"],
        rows,
        align="lrrrrr",
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Main mutual HIL dataset ----
    aggregate, summary, batch_meta, run_meta, chunk_meta = _load_main_data()
    validation = _validate_main_trials(aggregate, chunk_meta)
    diagnostics = _diagnostics_main(aggregate)

    if len(aggregate) != 18:
        raise RuntimeError(f"Expected 18 formal trials, found {len(aggregate)}")
    if validation["status"].ne("included").any():
        bad = validation[validation["status"] != "included"]
        raise RuntimeError(f"Incomplete formal trial validation:\n{bad}")

    _write_main_tables(aggregate, validation, diagnostics, batch_meta, chunk_meta)
    _plot_main_figures(aggregate, diagnostics)

    # ---- Appendix K-sweep dataset ----
    k_aggregate, k_overall, k_condition, k_ranking, k_best, k_batch_meta, k_chunk_meta = _load_k_sweep_data()

    _plot_k_sweep_figures(k_aggregate, k_overall, k_condition, k_ranking)
    _write_k_sweep_tables(k_overall, k_condition, k_ranking, k_best)

    # ---- Best-K vs EAPF comparison ----
    _plot_best_k_vs_eapf(k_aggregate, aggregate)
    _write_best_k_vs_eapf_table(k_aggregate, aggregate)

    # ---- Copy CSVs for reproducibility ----
    for name in [
        "aggregate_metrics.csv",
        "summary_by_model_condition.csv",
        "batch_metadata.json",
        "run_metadata.json",
        "chunk_metadata.json",
    ]:
        shutil.copy2(MAIN_DATA_DIR / name, DERIVED_DIR / f"main_{name}")

    for name in [
        "aggregate_metrics.csv",
        "summary_by_k_overall.csv",
        "summary_by_k_condition.csv",
        "k_ranking.csv",
        "best_k_by_condition.csv",
        "batch_metadata.json",
        "run_metadata.json",
        "chunk_metadata.json",
    ]:
        src = K_SWEEP_DIR / name
        if src.exists():
            shutil.copy2(src, DERIVED_DIR / f"k_sweep_{name}")

    # ---- Report summary ----
    overall = _overall_summary(aggregate).to_dict(orient="records")
    validation_counts = (
        aggregate.groupby(["model", "pi_initial_freq"])["trial_id"]
        .count()
        .reset_index(name="n_trials")
        .to_dict(orient="records")
    )
    report_summary = {
        "data_dir": str(MAIN_DATA_DIR.relative_to(REPO_ROOT)),
        "k_sweep_dir": str(K_SWEEP_DIR.relative_to(REPO_ROOT)),
        "n_trials_main": int(len(aggregate)),
        "n_trials_k_sweep": int(len(k_aggregate)),
        "models": MODEL_ORDER,
        "conditions": CONDITION_ORDER,
        "k_values": K_VALUES,
        "best_k": 2.5,
        "validation_counts": validation_counts,
        "overall": overall,
        "k_ranking": k_ranking.to_dict(orient="records"),
        "metadata": {
            "duration": batch_meta.get("duration"),
            "seed": batch_meta.get("seed"),
            "random_phase_enabled": batch_meta.get("random_phase_enabled"),
            "canonical_freq_pairs": batch_meta.get("canonical_freq_pairs"),
            "locked_model_parameters": batch_meta.get("locked_model_parameters"),
            "chunks": [
                {
                    "chunk_label": c.get("chunk_label"),
                    "completed": len(c.get("completed_trial_ids", [])),
                    "failed": len(c.get("failed_trial_ids", [])),
                    "freq_pairs": c.get("freq_pairs", []),
                }
                for c in chunk_meta
            ],
        },
        "diagnostic_note": "duplicate_suppressed_count was not recorded by the batch runner.",
    }
    (DERIVED_DIR / "report_summary.json").write_text(json.dumps(report_summary, indent=2), encoding="utf-8")
    print(json.dumps(report_summary, indent=2))


if __name__ == "__main__":
    main()

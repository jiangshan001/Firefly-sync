#!/usr/bin/env python3
"""Appendix Step 5B Kuramoto K-sensitivity workflow.

This is a diagnostic sensitivity analysis for mutual visual HIL. It does not
replace the locked Step 5B formal comparison, where Kuramoto uses K=5.0 from
the earlier model-selection parameter lock.

Theory note stored in run metadata:
for frequency mismatch df, angular mismatch dw = 2*pi*df. A one-way driven
intuition gives K_c ~= dw, while a symmetric two-oscillator mutual Kuramoto
intuition gives K_c ~= dw/2. Step 5B mutual HIL reconstructs neighbour phase
from flash events rather than observing a continuous ideal oscillator, so the
K values are theory-informed diagnostics rather than exact thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_step5b_mutual_hil_batch import (
    DEFAULT_CANONICAL_FREQ_PAIRS,
    LOCKED_EAPF_PARAMS,
    _as_float,
    _canonical_pairs_with_current_pairs,
    _freq_pair_key,
    _parse_freq_pairs,
    _resolve_initial_phases,
    _safe_delete_trial_dir,
    _save_csv,
    _save_json,
    _trial_fields,
    ensure_runtime_imports,
    run_trial,
    TrialSpec,
)


DEFAULT_K_VALUES = [2.5, 3.0, 3.5, 4.0, 4.5]
DEFAULT_PI_FREQS = [1.2, 1.5, 2.5]
DEFAULT_LOG_DIR = "experiments/logs/step5b_kuramoto_k_sensitivity"
FAILURE_MARKER = "trial_failed.json"
THEORY_NOTE = (
    "For mismatch df, dw=2*pi*df. One-way driven intuition: K_c ~= dw. "
    "Symmetric two-oscillator mutual Kuramoto intuition: K_c ~= dw/2. "
    "Step 5B event-based visual HIL reconstructs neighbour phase from flash "
    "events, so these are diagnostic values rather than exact thresholds."
)
SCORE_FORMULA = (
    "score = mean(final_5s_mean_abs_error) "
    "+ 0.5*mean(virtual_final_5s_std + pi_final_5s_std) "
    "+ 2*mean(max(0, 0.90-FCR)) + 5*mean(max(0, FCR-1.02)) "
    "+ 0.02*mean(api_drops + capture_drops + short_interval_warnings "
    "+ frontend_warnings) + 10*(failed_or_invalid_trials/n_trials)"
)


@dataclass(frozen=True)
class AppendixTrialSpec:
    model: str
    kuramoto_gain: float | None
    virtual_freq: float
    pi_freq: float
    repeat: int
    trial_id: str
    phase_seed: int | None
    phase_seed_index: int
    random_phase_enabled: bool
    virtual_initial_phase_rad: float
    pi_initial_phase_rad: float
    initial_phase_difference_rad: float


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _format_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _make_k_trial_id(k: float, virtual_freq: float, pi_freq: float, repeat: int) -> str:
    return (
        f"kuramoto_K{_format_token(k)}_V{_format_token(virtual_freq)}_"
        f"P{_format_token(pi_freq)}_r{repeat:02d}"
    )


def _make_model_trial_id(model: str, k: float | None, virtual_freq: float,
                         pi_freq: float, repeat: int) -> str:
    if model == "kuramoto":
        assert k is not None
        return _make_k_trial_id(k, virtual_freq, pi_freq, repeat)
    return (
        f"{model}_V{_format_token(virtual_freq)}_"
        f"P{_format_token(pi_freq)}_r{repeat:02d}"
    )


def _resolve_freq_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.freq_pairs is not None:
        return args.freq_pairs
    return [(float(args.virtual_freq), float(pi_freq)) for pi_freq in _parse_pi_freq_values(args.pi_freqs)]


def _parse_pi_freq_values(values: list[Any]) -> list[float]:
    freqs: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                freqs.append(float(item))
    if not freqs:
        raise argparse.ArgumentTypeError("At least one Pi frequency is required")
    return freqs


def _phase_seed_index(virtual_freq: float, pi_freq: float, repeat: int,
                      canonical_freq_pairs: list[tuple[float, float]]) -> int:
    pair = _freq_pair_key((virtual_freq, pi_freq))
    pair_index = {
        _freq_pair_key(freq_pair): idx
        for idx, freq_pair in enumerate(canonical_freq_pairs)
    }
    if pair not in pair_index:
        allowed = ", ".join(f"{v:g}:{p:g}" for v, p in canonical_freq_pairs)
        raise ValueError(
            f"Frequency pair {virtual_freq:g}:{pi_freq:g} is not in "
            f"--canonical-freq-pairs ({allowed})."
        )
    return (repeat - 1) * len(canonical_freq_pairs) + pair_index[pair] + 1


def _build_k_sweep_schedule(
    freq_pairs: list[tuple[float, float]],
    gains: list[float],
    repeats: int,
    base_seed: int | None,
    canonical_freq_pairs: list[tuple[float, float]],
    random_phase: bool,
    virtual_phase_rad: float | None,
    pi_phase_rad: float | None,
) -> list[AppendixTrialSpec]:
    schedule: list[AppendixTrialSpec] = []
    for repeat in range(1, repeats + 1):
        for virtual_freq, pi_freq in freq_pairs:
            index = _phase_seed_index(virtual_freq, pi_freq, repeat, canonical_freq_pairs)
            phase_seed = None if base_seed is None else int(base_seed) + index
            v_phase, p_phase, phase_diff = _resolve_initial_phases(
                phase_seed,
                random_phase,
                virtual_phase_rad,
                pi_phase_rad,
            )
            for k in gains:
                schedule.append(AppendixTrialSpec(
                    model="kuramoto",
                    kuramoto_gain=float(k),
                    virtual_freq=virtual_freq,
                    pi_freq=pi_freq,
                    repeat=repeat,
                    trial_id=_make_k_trial_id(float(k), virtual_freq, pi_freq, repeat),
                    phase_seed=phase_seed,
                    phase_seed_index=index,
                    random_phase_enabled=random_phase,
                    virtual_initial_phase_rad=v_phase,
                    pi_initial_phase_rad=p_phase,
                    initial_phase_difference_rad=phase_diff,
                ))
    return schedule


def _build_best_k_vs_eapf_schedule(
    freq_pairs: list[tuple[float, float]],
    best_k: float,
    repeats: int,
    base_seed: int | None,
    canonical_freq_pairs: list[tuple[float, float]],
    random_phase: bool,
    virtual_phase_rad: float | None,
    pi_phase_rad: float | None,
) -> list[AppendixTrialSpec]:
    schedule: list[AppendixTrialSpec] = []
    for repeat in range(1, repeats + 1):
        for virtual_freq, pi_freq in freq_pairs:
            index = _phase_seed_index(virtual_freq, pi_freq, repeat, canonical_freq_pairs)
            phase_seed = None if base_seed is None else int(base_seed) + index
            v_phase, p_phase, phase_diff = _resolve_initial_phases(
                phase_seed,
                random_phase,
                virtual_phase_rad,
                pi_phase_rad,
            )
            for model in ("eapf_consensus", "kuramoto"):
                k = float(best_k) if model == "kuramoto" else None
                schedule.append(AppendixTrialSpec(
                    model=model,
                    kuramoto_gain=k,
                    virtual_freq=virtual_freq,
                    pi_freq=pi_freq,
                    repeat=repeat,
                    trial_id=_make_model_trial_id(model, k, virtual_freq, pi_freq, repeat),
                    phase_seed=phase_seed,
                    phase_seed_index=index,
                    random_phase_enabled=random_phase,
                    virtual_initial_phase_rad=v_phase,
                    pi_initial_phase_rad=p_phase,
                    initial_phase_difference_rad=phase_diff,
                ))
    return schedule


def _to_step5b_spec(spec: AppendixTrialSpec) -> TrialSpec:
    return TrialSpec(
        model=spec.model,
        virtual_freq=spec.virtual_freq,
        pi_freq=spec.pi_freq,
        repeat=spec.repeat,
        trial_id=spec.trial_id,
        trial_seed=spec.phase_seed,
        random_phase_enabled=spec.random_phase_enabled,
        virtual_initial_phase_rad=spec.virtual_initial_phase_rad,
        pi_initial_phase_rad=spec.pi_initial_phase_rad,
        initial_phase_difference_rad=spec.initial_phase_difference_rad,
    )


def _count_short_intervals(events: list[dict], min_interval: float) -> int:
    count = 0
    by_event: dict[str, list[float]] = {}
    for row in events:
        event = str(row.get("event", ""))
        t = _as_float(row.get("t_s"))
        if event and t is not None:
            by_event.setdefault(event, []).append(t)
    for times in by_event.values():
        times = sorted(times)
        count += sum(
            1 for a, b in zip(times, times[1:])
            if (b - a) < min_interval
        )
    return count


def _count_debug_warnings(rows: list[dict]) -> int:
    return sum(1 for row in rows if str(row.get("warning", "")).strip())


def _failure_marker_path(trial_dir: Path) -> Path:
    return trial_dir / FAILURE_MARKER


def _load_completed_trial_metrics(trial_dir: Path) -> dict | None:
    if _failure_marker_path(trial_dir).exists():
        return None
    metrics_path = trial_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("trial_id"):
        return None
    if _truthy(data.get("timeout_or_failure")) or _truthy(data.get("failed_or_invalid_trial")):
        return None
    return data


def _scan_completed_trial_metrics(trial_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not trial_root.exists():
        return rows
    for child in sorted(trial_root.iterdir()):
        if not child.is_dir():
            continue
        metrics = _load_completed_trial_metrics(child)
        if metrics is not None:
            rows.append(metrics)
    return rows


def _trial_action(trial_dir: Path, resume: bool, overwrite: bool) -> str:
    if not trial_dir.exists():
        return "run"
    if overwrite:
        return "overwrite"
    if resume and _load_completed_trial_metrics(trial_dir) is not None:
        return "skip"
    if _failure_marker_path(trial_dir).exists():
        return "fail_failed_existing"
    if resume:
        return "fail_incomplete_existing"
    return "fail_existing"


def _write_trial_failure(trial_dir: Path, row: dict, spec: AppendixTrialSpec,
                         args: argparse.Namespace) -> None:
    failure = {
        "trial_id": spec.trial_id,
        "trial_dir": str(trial_dir),
        "model": spec.model,
        "kuramoto_K": spec.kuramoto_gain,
        "virtual_initial_freq": spec.virtual_freq,
        "pi_initial_freq": spec.pi_freq,
        "repeat": spec.repeat,
        "seed": args.seed,
        "phase_seed": spec.phase_seed,
        "phase_seed_index": spec.phase_seed_index,
        "failed_at": datetime.now().isoformat(),
        "error_message": row.get("error_message", ""),
        "timeout_or_failure": row.get("timeout_or_failure", True),
        "note": (
            "This trial did not complete valid real HIL execution. It is "
            "excluded from aggregate/ranking outputs and is not resumable "
            "without --overwrite."
        ),
        "raw_metrics": row,
    }
    _save_json(_failure_marker_path(trial_dir), failure)
    metrics_path = trial_dir / "metrics_summary.json"
    if metrics_path.exists():
        metrics_path.unlink()


def _augment_metrics(trial_dir: Path, row: dict, spec: AppendixTrialSpec,
                     args: argparse.Namespace) -> dict:
    api_events = _read_csv(trial_dir / "api_events.csv")
    detection_log = _read_csv(trial_dir / "detection_log.csv")
    flash_events = _read_csv(trial_dir / "flash_events.csv")
    kuramoto_debug = _read_csv(trial_dir / "kuramoto_debug.csv")

    api_drop_count = sum(
        1 for event in api_events
        if str(event.get("ok", "1")).lower() in ("0", "false")
    )
    capture_drop_count = sum(
        1 for item in detection_log
        if item.get("state", "") == "" and item.get("virtual_event", "") == ""
    )
    short_interval_warning_count = _count_short_intervals(flash_events, args.min_interval)
    frontend_warning_count = _count_debug_warnings(kuramoto_debug)
    failed_or_invalid = _truthy(row.get("timeout_or_failure")) or (
        spec.model == "kuramoto"
        and _as_float(row.get("frequency_error_final_5s_mean_abs")) is None
        and not args.dry_run
    )
    df = abs(float(spec.virtual_freq) - float(spec.pi_freq))
    angular_mismatch = 2.0 * math.pi * df
    updates = {
        "appendix_workflow": args.mode,
        "appendix_label": "Appendix: Kuramoto gain sensitivity",
        "locked_k_main_step5b": 5.0,
        "sensitivity_analysis": True,
        "phase_seed": spec.phase_seed,
        "phase_seed_index": spec.phase_seed_index,
        "condition_key": f"V{spec.virtual_freq:g}_P{spec.pi_freq:g}",
        "frequency_mismatch_hz": round(df, 6),
        "angular_mismatch_rad_s": round(angular_mismatch, 6),
        "one_way_Kc_approx": round(angular_mismatch, 6),
        "mutual_Kc_approx": round(angular_mismatch / 2.0, 6),
        "theory_note": THEORY_NOTE,
        "api_drop_count": api_drop_count,
        "capture_drop_count": capture_drop_count,
        "short_interval_warning_count": short_interval_warning_count,
        "frontend_warning_count": frontend_warning_count,
        "kuramoto_debug_warning_count": frontend_warning_count,
        "failed_or_invalid_trial": bool(failed_or_invalid),
    }
    if spec.kuramoto_gain is not None:
        updates["kuramoto_K"] = float(spec.kuramoto_gain)
        updates["model_parameters_json"] = json.dumps({"K": float(spec.kuramoto_gain)}, sort_keys=True)
    row.update(updates)
    metrics_path = trial_dir / "metrics_summary.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            if isinstance(metrics, dict):
                metrics.update(updates)
                _save_json(metrics_path, metrics)
        except (OSError, json.JSONDecodeError):
            pass
    return row


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    values = [_as_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None and math.isfinite(value)]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None and math.isfinite(value) else None


def _truthy(value: Any) -> bool:
    return str(value).lower() in ("1", "true", "yes")


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


def _summarise_groups(rows: list[dict], group_keys: list[str],
                      include_score: bool = False) -> list[dict]:
    groups: dict[tuple[Any, ...], list[dict]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        groups.setdefault(key, []).append(row)

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
    overall = _summarise_groups(rows, ["kuramoto_K"], include_score=True)
    ranked = sorted(
        [row for row in overall if _as_float(row.get("appendix_ranking_score")) is not None],
        key=lambda row: float(row["appendix_ranking_score"]),
    )
    for idx, row in enumerate(ranked, 1):
        row["rank"] = idx
        row["best_overall"] = idx == 1
    return ranked


def _best_by_condition(rows: list[dict]) -> list[dict]:
    by_condition: dict[tuple[Any, Any], list[dict]] = {}
    for row in rows:
        by_condition.setdefault((row.get("virtual_initial_freq"), row.get("pi_initial_freq")), []).append(row)
    best_rows: list[dict] = []
    for (vf, pf), group in sorted(by_condition.items(), key=lambda item: tuple(str(v) for v in item[0])):
        by_k: dict[Any, list[dict]] = {}
        for row in group:
            by_k.setdefault(row.get("kuramoto_K"), []).append(row)
        candidates = [
            (k, _score_group(k_rows), k_rows)
            for k, k_rows in by_k.items()
        ]
        candidates = [item for item in candidates if item[1] is not None]
        if not candidates:
            continue
        k, score, k_rows = sorted(candidates, key=lambda item: float(item[1]))[0]
        best_rows.append({
            "virtual_initial_freq": vf,
            "pi_initial_freq": pf,
            "best_k": k,
            "appendix_ranking_score": _round_or_none(score),
            "n_trials": len(k_rows),
        })
    return best_rows


def _aggregate_fields(rows: list[dict]) -> list[str]:
    preferred = _trial_fields() + [
        "appendix_workflow",
        "appendix_label",
        "locked_k_main_step5b",
        "sensitivity_analysis",
        "phase_seed",
        "phase_seed_index",
        "condition_key",
        "frequency_mismatch_hz",
        "angular_mismatch_rad_s",
        "one_way_Kc_approx",
        "mutual_Kc_approx",
        "api_drop_count",
        "capture_drop_count",
        "short_interval_warning_count",
        "frontend_warning_count",
        "kuramoto_debug_warning_count",
        "failed_or_invalid_trial",
        "theory_note",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return preferred + extras


def _write_outputs(batch_dir: Path, rows: list[dict], schedule: list[AppendixTrialSpec],
                   args: argparse.Namespace, best_k: float | None = None) -> None:
    rows = sorted(rows, key=lambda row: str(row.get("trial_id", "")))
    _save_csv(batch_dir / "aggregate_metrics.csv", rows, _aggregate_fields(rows))
    if args.mode == "k-sweep":
        k_condition = _summarise_groups(
            rows,
            ["kuramoto_K", "virtual_initial_freq", "pi_initial_freq"],
            include_score=True,
        )
        k_overall = _summarise_groups(rows, ["kuramoto_K"], include_score=True)
        ranking = _ranking(rows)
        best_condition = _best_by_condition(rows)
        _save_csv(batch_dir / "summary_by_k_condition.csv", k_condition)
        _save_csv(batch_dir / "summary_by_k_overall.csv", k_overall)
        _save_csv(batch_dir / "k_ranking.csv", ranking)
        _save_csv(batch_dir / "best_k_by_condition.csv", best_condition)
    else:
        _save_csv(
            batch_dir / "summary_by_model_condition.csv",
            _summarise_groups(rows, ["model", "virtual_initial_freq", "pi_initial_freq"]),
        )
        _save_csv(
            batch_dir / "summary_by_model_overall.csv",
            _summarise_groups(rows, ["model"]),
        )

    metadata = {
        "created_at": datetime.now().isoformat(),
        "workflow": args.mode,
        "appendix_label": "Appendix: Kuramoto gain sensitivity",
        "main_step5b_locked_k": 5.0,
        "main_result_replacement": False,
        "sensitivity_analysis": True,
        "duration": args.duration,
        "freq_pairs": [{"virtual": v, "pi": p} for v, p in args.freq_pairs],
        "kuramoto_gains": args.kuramoto_gains,
        "best_k": best_k,
        "best_k_selection_source": str(args.selection_source) if args.selection_source else None,
        "models": sorted({spec.model for spec in schedule}),
        "locked_model_parameters": {
            "eapf_consensus": LOCKED_EAPF_PARAMS,
            "kuramoto_main_formal": {"K": 5.0},
            "kuramoto_appendix": (
                {"K_values": args.kuramoto_gains}
                if args.mode == "k-sweep" else {"K": best_k}
            ),
        },
        "theory_note": THEORY_NOTE,
        "score_formula": SCORE_FORMULA,
        "frequency_mismatch_notes": [
            {
                "virtual_hz": v,
                "pi_hz": p,
                "delta_f_hz": abs(v - p),
                "delta_omega_rad_s": 2.0 * math.pi * abs(v - p),
                "one_way_Kc_approx": 2.0 * math.pi * abs(v - p),
                "mutual_Kc_approx": math.pi * abs(v - p),
            }
            for v, p in args.freq_pairs
        ],
        "repeats": args.repeats,
        "seed": args.seed,
        "canonical_freq_pairs": [{"virtual": v, "pi": p} for v, p in args.canonical_freq_pairs],
        "random_phase_enabled": args.random_phase,
        "phase_schedule_note": (
            "phase_seed_index depends only on repeat and frequency pair; K/model "
            "does not enter phase generation, so K/EAPF comparisons are paired."
        ),
        "min_interval": args.min_interval,
        "api_timeout": args.api_timeout,
        "api_retries": args.api_retries,
        "dry_run": args.dry_run,
        "run_dir": str(batch_dir),
        "resume": args.resume,
        "overwrite": args.overwrite,
        "chunk_label": args.chunk_label,
        "schedule": [spec.__dict__ for spec in schedule],
    }
    _save_json(batch_dir / "run_metadata.json", metadata)
    _save_json(batch_dir / "batch_metadata.json", metadata)


def _append_chunk_metadata(batch_dir: Path, chunk_record: dict) -> None:
    chunks_dir = batch_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    _save_json(chunks_dir / f"{chunk_record['chunk_id']}.json", chunk_record)
    combined = batch_dir / "chunk_metadata.json"
    existing: list[dict] = []
    if combined.exists():
        try:
            with open(combined) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []
    existing.append(chunk_record)
    _save_json(combined, existing)


def _load_best_k(selection_source: Path) -> float:
    ranking = selection_source / "k_ranking.csv"
    if not ranking.exists():
        raise SystemExit(f"k_ranking.csv not found in selection source: {selection_source}")
    rows = _read_csv(ranking)
    if not rows:
        raise SystemExit(f"k_ranking.csv is empty: {ranking}")
    best = sorted(
        rows,
        key=lambda row: _as_float(row.get("appendix_ranking_score")) or float("inf"),
    )[0]
    value = _as_float(best.get("kuramoto_K"))
    if value is None:
        raise SystemExit(f"Could not read best K from {ranking}")
    return value


def _make_runner_args(args: argparse.Namespace, kuramoto_gain: float | None) -> argparse.Namespace:
    runner_args = argparse.Namespace(**vars(args))
    runner_args.models = ["eapf_consensus", "kuramoto"] if args.mode == "best-k-vs-eapf" else ["kuramoto"]
    runner_args.kuramoto_gain = float(kuramoto_gain) if kuramoto_gain is not None else 5.0
    return runner_args


def _print_schedule(action_plan: list[tuple[AppendixTrialSpec, Path, str]], args: argparse.Namespace,
                    batch_dir: Path) -> None:
    print(f"Batch dir: {batch_dir}")
    print("Trial layout: direct children of run directory")
    print(f"Workflow: {args.mode}")
    print(f"Phase control: seed={args.seed} random_phase={args.random_phase}")
    print("Stable phase schedule: repeat -> canonical frequency pair; K/model are paired on phases.")
    print("Canonical freq pairs: " + ", ".join(f"{v:g}:{p:g}" for v, p in args.canonical_freq_pairs))
    print("Theory note: " + THEORY_NOTE)
    print("Schedule:")
    for i, (spec, trial_dir, action) in enumerate(action_plan, 1):
        k_label = "n/a" if spec.kuramoto_gain is None else f"{spec.kuramoto_gain:g}"
        print(
            f"  {i:02d}. {spec.trial_id} model={spec.model} K={k_label} "
            f"V={spec.virtual_freq:g}Hz P={spec.pi_freq:g}Hz "
            f"phase_seed={spec.phase_seed} "
            f"phases=({spec.virtual_initial_phase_rad:.4f}, "
            f"{spec.pi_initial_phase_rad:.4f}) "
            f"diff={spec.initial_phase_difference_rad:.4f} "
            f"action={action} path={trial_dir}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Appendix Step 5B Kuramoto K-sensitivity and best-K confirmation workflow."
    )
    parser.add_argument("--mode", choices=["k-sweep", "best-k-vs-eapf"], default="k-sweep")
    parser.add_argument("--leader-api", default="http://192.168.1.111:8000")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--virtual-freq", type=float, default=2.0)
    parser.add_argument("--pi-freqs", nargs="+", default=DEFAULT_PI_FREQS,
                        help="Pi initial frequencies, space-separated or comma-separated")
    parser.add_argument("--freq-pairs", type=_parse_freq_pairs, default=None)
    parser.add_argument("--kuramoto-gains", nargs="+", type=float, default=DEFAULT_K_VALUES)
    parser.add_argument("--best-k", type=float, default=None)
    parser.add_argument("--selection-source", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--random-phase", dest="random_phase", action="store_true", default=True)
    parser.add_argument("--no-random-phase", dest="random_phase", action="store_false")
    parser.add_argument("--virtual-phase-rad", type=float, default=None)
    parser.add_argument("--pi-phase-rad", type=float, default=None)
    parser.add_argument("--canonical-freq-pairs", type=_parse_freq_pairs,
                        default=_parse_freq_pairs(DEFAULT_CANONICAL_FREQ_PAIRS))
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--run-dir", default=None,
                        help="Use this exact shared directory; trial folders are direct children")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chunk-label", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dot-size", type=int, default=450)
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-interval", type=float, default=0.35)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--flash-duration", type=float, default=0.06)
    parser.add_argument("--api-sample-interval", type=float, default=1.0)
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    args.freq_pairs = _resolve_freq_pairs(args)
    args.canonical_freq_pairs = _canonical_pairs_with_current_pairs(
        args.canonical_freq_pairs,
        args.freq_pairs,
    )

    best_k = args.best_k
    if args.mode == "best-k-vs-eapf" and best_k is None:
        if args.selection_source is None:
            parser.error("--best-k or --selection-source is required for --mode best-k-vs-eapf")
        best_k = _load_best_k(args.selection_source)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(args.log_dir) / (
            f"best_k_vs_eapf_{ts}" if args.mode == "best-k-vs-eapf"
            else f"{ts}_step5b_kuramoto_k_sweep"
        )
    )
    trial_root = batch_dir

    schedule = (
        _build_best_k_vs_eapf_schedule(
            args.freq_pairs,
            float(best_k),
            args.repeats,
            args.seed,
            args.canonical_freq_pairs,
            args.random_phase,
            args.virtual_phase_rad,
            args.pi_phase_rad,
        )
        if args.mode == "best-k-vs-eapf"
        else _build_k_sweep_schedule(
            args.freq_pairs,
            args.kuramoto_gains,
            args.repeats,
            args.seed,
            args.canonical_freq_pairs,
            args.random_phase,
            args.virtual_phase_rad,
            args.pi_phase_rad,
        )
    )

    action_plan: list[tuple[AppendixTrialSpec, Path, str]] = []
    unsafe: list[str] = []
    for spec in schedule:
        trial_dir = trial_root / spec.trial_id
        action = _trial_action(trial_dir, args.resume, args.overwrite)
        action_plan.append((spec, trial_dir, action))
        if action == "fail_existing":
            unsafe.append(f"{trial_dir} already exists. Use --resume or --overwrite.")
        elif action == "fail_failed_existing":
            unsafe.append(
                f"{trial_dir} contains {FAILURE_MARKER}; use --overwrite to rerun it."
            )
        elif action == "fail_incomplete_existing":
            unsafe.append(
                f"{trial_dir} exists but has no metrics_summary.json; inspect it or use --overwrite."
            )

    _print_schedule(action_plan, args, batch_dir)

    if args.dry_run:
        print("[DRY-RUN] Schedule printed only; no trial folders, metadata, or metrics files were written.")
        if unsafe:
            print("[DRY-RUN] A real run would stop on:")
            for msg in unsafe:
                print(f"  - {msg}")
        return

    if unsafe and not args.dry_run:
        print("\nRefusing to continue because existing trial folders would be overwritten:")
        for msg in unsafe:
            print(f"  - {msg}")
        sys.exit(2)

    batch_dir.mkdir(parents=True, exist_ok=True)
    chunk_started_at = datetime.now().isoformat()
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    if any(action != "skip" for _spec, _trial_dir, action in action_plan):
        ensure_runtime_imports(dry=False)

    for spec, trial_dir, action in action_plan:
        if action == "skip":
            print(f"[RESUME] Skipping completed trial {spec.trial_id}: {trial_dir}")
            skipped.append(spec.trial_id)
            continue
        if action == "overwrite":
            print(f"[OVERWRITE] Deleting existing trial folder: {trial_dir}")
            _safe_delete_trial_dir(trial_dir, trial_root)
        runner_args = _make_runner_args(args, spec.kuramoto_gain)
        row = run_trial(runner_args, trial_dir, _to_step5b_spec(spec))
        row = _augment_metrics(trial_dir, row, spec, args)
        if _truthy(row.get("timeout_or_failure")) or _truthy(row.get("failed_or_invalid_trial")):
            failed.append(spec.trial_id)
            _write_trial_failure(trial_dir, row, spec, args)
        else:
            completed.append(spec.trial_id)
        rows_now = _scan_completed_trial_metrics(trial_root)
        _write_outputs(batch_dir, rows_now, schedule, args, best_k=best_k)

    rows = _scan_completed_trial_metrics(trial_root)
    _write_outputs(batch_dir, rows, schedule, args, best_k=best_k)

    chunk_record = {
        "chunk_id": ts,
        "chunk_label": args.chunk_label,
        "workflow": args.mode,
        "dry_run": args.dry_run,
        "command_line": sys.argv,
        "freq_pairs": [{"virtual": v, "pi": p} for v, p in args.freq_pairs],
        "kuramoto_gains": args.kuramoto_gains,
        "best_k": best_k,
        "repeats": args.repeats,
        "duration": args.duration,
        "random_phase_enabled": args.random_phase,
        "seed": args.seed,
        "started_at": chunk_started_at,
        "finished_at": datetime.now().isoformat(),
        "completed_trial_ids": completed,
        "skipped_trial_ids": skipped,
        "failed_trial_ids": failed,
        "planned_trial_ids": [spec.trial_id for spec, _path, _action in action_plan],
    }
    _append_chunk_metadata(batch_dir, chunk_record)

    print(f"\nAppendix workflow complete: {batch_dir}")
    print(f"  aggregate: {batch_dir / 'aggregate_metrics.csv'}")
    if args.mode == "k-sweep":
        ranking = _ranking(rows)
        best_conditions = _best_by_condition(rows)
        if ranking:
            print(f"  best K overall: {ranking[0].get('kuramoto_K')} "
                  f"(score={ranking[0].get('appendix_ranking_score')})")
        if best_conditions:
            print("  best K per condition:")
            for row in best_conditions:
                print(
                    f"    V{row['virtual_initial_freq']}/P{row['pi_initial_freq']}: "
                    f"K={row['best_k']} score={row['appendix_ranking_score']}"
                )
        print(f"  ranking:   {batch_dir / 'k_ranking.csv'}")
    else:
        print(f"  summary:   {batch_dir / 'summary_by_model_condition.csv'}")


if __name__ == "__main__":
    main()

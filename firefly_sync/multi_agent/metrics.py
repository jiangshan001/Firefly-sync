"""Group-level synchronisation metrics for multi-agent simulation.

Distinguishes:
  - zero_lag_group_sync (strict, all three conditions)
  - offset_phase_lock (stable non-zero offsets allowed)
  - partial_phase_lock_with_extra_flashes
  - failure
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_group_metrics(
    all_flash_times: list[list[float]],
    final_window_s: float = 10.0,
    agents: list | None = None,
    phases_over_time: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Compute all group-level synchronisation metrics."""
    n = len(all_flash_times)

    # ---- effective frequencies ----
    effective_freqs: list[float] = []
    eff_labels: dict = {}
    for i in range(n):
        ef = _effective_frequency(all_flash_times[i], final_window_s)
        effective_freqs.append(ef)
        eff_labels[f"effective_frequency_agent_{i}_hz"] = (
            round(ef, 6) if not np.isnan(ef) else None)
    valid_ef = [f for f in effective_freqs if not np.isnan(f)]
    eff_labels["valid_frequency_agent_count"] = len(valid_ef)
    f_spread = (round(max(valid_ef) - min(valid_ef), 6)
                if len(valid_ef) >= 2 else None)

    # ---- flash counts ----
    fc = [_flash_count_in_window(ft, final_window_s) for ft in all_flash_times]
    for i in range(n):
        eff_labels[f"flash_count_agent_{i}"] = fc[i]
    min_fc, max_fc = (min(fc), max(fc)) if fc else (0, 0)
    fc_ratio = (max_fc / min_fc) if min_fc > 0 else float("inf")
    extra_fc = max_fc - min_fc

    # ---- pairwise timing ----
    mean_pairwise = _pairwise_error(all_flash_times, final_window_s)
    dispersion = _dispersion(all_flash_times, final_window_s)
    order_param, final_order = _order_param(phases_over_time, final_window_s)

    # ---- offset-aware matching ----
    off = _offset_analysis(all_flash_times, final_window_s)

    # ---- success criteria ----
    phase_sync = (mean_pairwise is not None and mean_pairwise < 0.10
                  and (order_param is None or order_param > 0.85))
    freq_lock = (f_spread is not None and f_spread < 0.05
                 and len(valid_ef) == n)
    one_to_one = (min_fc >= 2 and fc_ratio != float("inf") and fc_ratio <= 1.2)

    zero_lag_sync = phase_sync and freq_lock and one_to_one
    offset_sync = off["offset_phase_lock_success"]
    group_sync = zero_lag_sync  # strict

    label, reason = _diagnose(zero_lag_sync, offset_sync, phase_sync, freq_lock,
                              one_to_one, min_fc, fc_ratio, extra_fc)

    return {
        "flash_count_agent_0": fc[0] if n > 0 else 0,
        "flash_count_agent_1": fc[1] if n > 1 else 0,
        "flash_count_agent_2": fc[2] if n > 2 else 0,
        "min_flash_count": min_fc, "max_flash_count": max_fc,
        "flash_count_ratio": round(fc_ratio, 4) if fc_ratio != float("inf") else "inf",
        "extra_flash_count": extra_fc,
        "extra_flash_rate_hz": round(extra_fc / final_window_s, 4) if final_window_s > 0 else 0,
        **eff_labels,
        "final_frequency_spread_hz": f_spread,
        "mean_pairwise_timing_error_s": round(mean_pairwise, 6) if mean_pairwise is not None else None,
        "flash_timing_dispersion_s": round(dispersion, 6) if dispersion is not None else None,
        "mean_order_parameter_R": round(order_param, 6) if order_param is not None else None,
        "final_order_parameter_R": round(final_order, 6) if final_order is not None else None,
        # offset
        "mean_pairwise_offset_s": round(off["mean_offset"], 6) if off["mean_offset"] is not None else None,
        "mean_pairwise_offset_jitter_s": round(off["mean_jitter"], 6) if off["mean_jitter"] is not None else None,
        "max_pairwise_offset_jitter_s": round(off["max_jitter"], 6) if off["max_jitter"] is not None else None,
        "offset_phase_lock_success": bool(off["offset_phase_lock_success"]),
        # sync
        "phase_sync_success": bool(phase_sync),
        "frequency_lock_success": bool(freq_lock),
        "one_to_one_flash_lock_success": bool(one_to_one),
        "zero_lag_group_sync_success": bool(zero_lag_sync),
        "group_sync_success": bool(group_sync),
        "phase_locked_group_success": bool(offset_sync),
        "sync_diagnostic_label": label,
        "sync_failure_reason": reason,
    }


def check_group_synchronisation(
    all_flash_times: list[list[float]],
    **kwargs,
) -> dict[str, Any]:
    m = compute_group_metrics(all_flash_times, **kwargs)
    return {"group_sync_success": m["group_sync_success"],
            "time_to_group_sync_s": None}


# ======================================================================
# Internal helpers
# ======================================================================

def _effective_frequency(ft: list[float], ws: float) -> float:
    if len(ft) < 2:
        return float("nan")
    cutoff = max(ft[-1] - ws, 0)
    recent = [t for t in ft if t >= cutoff]
    if len(recent) < 2:
        return float("nan")
    intv = np.diff(recent)
    return 1.0 / np.mean(intv) if np.mean(intv) > 0 else float("nan")


def _flash_count_in_window(ft: list[float], ws: float) -> int:
    if not ft:
        return 0
    return sum(1 for t in ft if t >= max(ft[-1] - ws, 0))


def _pairwise_error(all_ft: list[list[float]], ws: float) -> float | None:
    n = len(all_ft)
    w0 = max(min(ft[-1] if ft else 0 for ft in all_ft) - ws, 0) if all_ft else 0
    errs = []
    for i in range(n):
        for j in range(i + 1, n):
            fi = [t for t in all_ft[i] if t >= w0]
            fj = [t for t in all_ft[j] if t >= w0]
            if not fi or not fj:
                continue
            for ti in fi:
                errs.append(abs(ti - min(fj, key=lambda tj: abs(tj - ti))))
    return float(np.mean(errs)) if errs else None


def _dispersion(all_ft: list[list[float]], ws: float) -> float | None:
    w0 = max(min(ft[-1] if ft else 0 for ft in all_ft) - ws, 0) if all_ft else 0
    all_t = sorted(t for ft in all_ft for t in ft if t >= w0)
    groups: list[list[float]] = []
    for t in all_t:
        placed = False
        for g in groups:
            if abs(t - np.mean(g)) < 0.15:
                g.append(t); placed = True; break
        if not placed:
            groups.append([t])
    disp = [max(g) - min(g) for g in groups if len(g) >= 2]
    return float(np.mean(disp)) if disp else None


def _order_param(phases: list[list[float]] | None, ws: float
                 ) -> tuple[float | None, float | None]:
    if not phases or len(phases) == 0:
        return None, None
    n_steps = min(len(p) for p in phases if p)
    if n_steps == 0:
        return None, None
    w_steps = max(1, int(ws / 0.01))
    start = max(0, n_steps - w_steps)
    Rv = []
    for step in range(start, n_steps):
        th = []
        for p in phases:
            if step < len(p):
                v = p[step]
                th.append((v % (2 * np.pi)) if v < 2 * np.pi else ((v * 2 * np.pi) % (2 * np.pi)))
        if th:
            Rv.append(float(np.abs(sum(np.exp(1j * np.array(th)))) / len(th)))
    return (float(np.mean(Rv)), Rv[-1]) if Rv else (None, None)


def _offset_analysis(all_ft: list[list[float]], ws: float) -> dict:
    """Compute stable-offset phase lock metrics."""
    n = len(all_ft)
    w0 = max(min(ft[-1] if ft else 0 for ft in all_ft) - ws, 0) if all_ft else 0
    all_jitters: list[float] = []
    all_offsets: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            fi = [t for t in all_ft[i] if t >= w0]
            fj = [t for t in all_ft[j] if t >= w0]
            if len(fi) < 3 or len(fj) < 3:
                continue
            offsets: list[float] = []
            for ti in fi:
                nearest = min(fj, key=lambda tj: abs(tj - ti))
                offsets.append(ti - nearest)
            if offsets:
                all_offsets.append(float(np.mean(offsets)))
                all_jitters.append(float(np.std(offsets)))
    mean_offset = float(np.mean(all_offsets)) if all_offsets else None
    mean_jitter = float(np.mean(all_jitters)) if all_jitters else None
    max_jitter = float(max(all_jitters)) if all_jitters else None
    success = (max_jitter is not None and max_jitter < 0.05
               and mean_jitter is not None and mean_jitter < 0.03)
    return {"mean_offset": mean_offset, "mean_jitter": mean_jitter,
            "max_jitter": max_jitter, "offset_phase_lock_success": success}


def _diagnose(zero_lag, offset, phase_sync, freq_lock, one_to_one,
              min_fc, fc_ratio, extra_fc):
    if zero_lag:
        return "full_group_sync", "success"
    if offset and one_to_one:
        return "offset_phase_locked", "success"
    if min_fc < 2:
        return "insufficient_flash_events", "insufficient_flash_events"
    if phase_sync and not one_to_one:
        return "partial_phase_lock_with_extra_flashes", "extra_flashes_or_harmonic_locking"
    if phase_sync and not freq_lock:
        return "partial_phase_lock", "frequency_not_locked"
    if not one_to_one and fc_ratio != float("inf") and fc_ratio > 1.2:
        return "extra_flashes_or_harmonic_locking", "extra_flashes_or_harmonic_locking"
    if not freq_lock:
        return "frequency_not_locked", "frequency_not_locked"
    if not phase_sync:
        return "phase_not_synchronised", "phase_not_synchronised"
    return "unknown", "unknown"

#!/usr/bin/env python3
"""Step 5B 1-virtual + 1-real-Pi mutual HIL model comparison batch.

This runner compares EAPF Consensus and Kuramoto under the same feedback-ON
mixed-reality condition. Hardware imports stay lazy so ``--dry-run`` can be
used on non-Pi machines to inspect the schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUPPORTED_MODELS = ("eapf_consensus", "kuramoto")
DEFAULT_FREQ_PAIRS = "2.0:1.5,2.0:1.8,2.0:2.3"
DEFAULT_CANONICAL_FREQ_PAIRS = "2.0:1.2,2.0:1.5,2.0:2.5"
LOCKED_EAPF_PARAMS = {
    "g_p": 0.02,
    "g_f": 0.02,
    "alpha_p": 0.2,
    "alpha_f": 0.2,
    "delta_theta_max_rad": 0.2,
    "delta_f_max_hz": 0.05,
}
LOCKED_KURAMOTO_PARAMS = {
    "K": 5.0,
}

_PiGPIOLED: Any = None
_PicameraFlashDetector: Any = None


@dataclass(frozen=True)
class TrialSpec:
    model: str
    virtual_freq: float
    pi_freq: float
    repeat: int
    trial_id: str
    trial_seed: int | None
    random_phase_enabled: bool
    virtual_initial_phase_rad: float
    pi_initial_phase_rad: float
    initial_phase_difference_rad: float


class ApiError(RuntimeError):
    pass


class FlashPhaseEstimator:
    """Estimate neighbour phase from flash timestamps only."""

    def __init__(self, initial_period_guess_s: float = 0.5,
                 max_stored_flashes: int = 30) -> None:
        self._estimated_period_s = float(initial_period_guess_s)
        self._max_stored = max_stored_flashes
        self._flash_times: list[float] = []

    def record_flash(self, timestamp_s: float) -> None:
        self._flash_times.append(timestamp_s)
        if len(self._flash_times) > self._max_stored:
            self._flash_times.pop(0)
        if len(self._flash_times) >= 2:
            intervals = [
                self._flash_times[i + 1] - self._flash_times[i]
                for i in range(len(self._flash_times) - 1)
            ]
            recent = [v for v in intervals[-10:] if v > 0]
            if recent:
                self._estimated_period_s = float(np.median(recent))

    def estimate_phase(self, now_s: float) -> float:
        if not self._flash_times or self._estimated_period_s <= 0:
            return 0.0
        phase = 2.0 * math.pi * ((now_s - self._flash_times[-1]) / self._estimated_period_s)
        return float(phase % (2.0 * math.pi))


def _ensure_hw(dry: bool = False) -> None:
    global _PiGPIOLED, _PicameraFlashDetector
    if dry:
        return
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED as _PL
        from firefly_sync.hardware.picamera_flash_detector import PicameraFlashDetector as _PFD
        _PiGPIOLED = _PL
        _PicameraFlashDetector = _PFD
    except ImportError as e:
        print(f"ERROR: hardware import failed: {e}")
        sys.exit(1)


def ensure_runtime_imports(dry: bool = False) -> None:
    """Initialise hardware/runtime imports required by real Step 5B trials."""
    _ensure_hw(dry=dry)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({k for row in rows for k in row.keys()})
    else:
        fields = list(fields)
        extras = sorted(
            key for row in rows for key in row.keys()
            if key not in fields
        )
        for key in extras:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _api(api_base: str, path: str, method: str = "GET", data: dict | None = None,
         timeout: float = 10.0, retries: int = 3, label: str = "",
         debug: bool = False, events: list[dict] | None = None) -> dict:
    url = api_base.rstrip("/") + path
    endpoint = f"{method} {path}"
    attempts = max(1, retries)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        print(f"  [API] Calling {endpoint}" + (f" ({label})" if label else ""))
        if attempt > 1:
            print(f"    retry {attempt}/{attempts}")
        try:
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"} if data else {},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                result = json.loads(r.read())
            if debug:
                print(f"    -> {json.dumps(result)[:240]}")
            if events is not None:
                events.append({
                    "t_monotonic": round(started, 6), "endpoint": endpoint,
                    "attempt": attempt, "ok": 1, "status": "ok",
                    "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
                })
            return result
        except (TimeoutError, socket.timeout) as e:
            last_exc = e
            msg = f"TIMEOUT: {endpoint} timed out after {timeout:g}s"
            print(f"    {msg}")
        except urllib.error.URLError as e:
            last_exc = e
            reason = getattr(e, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                msg = f"TIMEOUT: {endpoint} timed out after {timeout:g}s"
            else:
                msg = f"API ERROR: {type(e).__name__}: {e}"
            print(f"    {msg}")
        except Exception as e:
            last_exc = e
            msg = f"API ERROR: {type(e).__name__}: {e}"
            print(f"    {msg}")
        if events is not None:
            events.append({
                "t_monotonic": round(started, 6), "endpoint": endpoint,
                "attempt": attempt, "ok": 0, "status": str(last_exc),
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 3),
            })
        if attempt < attempts:
            time.sleep(1.0)
    raise ApiError(f"{endpoint} failed after {attempts} attempts: {last_exc}") from last_exc


def _fetch_kuramoto_debug(api_base: str, timeout_s: float = 5.0) -> list[dict[str, Any]]:
    try:
        req = urllib.request.Request(api_base.rstrip("/") + "/api/kuramoto_debug", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            payload = json.loads(response.read())
        rows = payload.get("rows", [])
        return rows if isinstance(rows, list) else []
    except Exception as exc:
        print(f"WARNING: could not fetch Kuramoto server debug trace: {exc}")
        return []


def _first_agent(payload: dict) -> dict:
    agents = payload.get("agents", [])
    return agents[0] if agents else {}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_freq_pairs(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v, p = item.split(":", 1)
            pairs.append((float(v), float(p)))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid frequency pair '{item}', expected virtual:pi"
            ) from exc
    if not pairs:
        raise argparse.ArgumentTypeError("At least one frequency pair is required")
    return pairs


def _make_trial_id(model: str, virtual_freq: float, pi_freq: float, repeat: int) -> str:
    return f"{model}_V{virtual_freq:g}_P{pi_freq:g}_r{repeat:02d}"

def _freq_pair_key(pair: tuple[float, float]) -> tuple[str, str]:
    return (f"{pair[0]:.12g}", f"{pair[1]:.12g}")

def _canonical_seed_index(
    model: str,
    virtual_freq: float,
    pi_freq: float,
    repeat: int,
    canonical_freq_pairs: list[tuple[float, float]],
    model_order: tuple[str, ...] = SUPPORTED_MODELS,
) -> int:
    """Return 1-based stable index in the formal Step 5B schedule.

    Canonical order is repeat outermost, then canonical frequency pair order,
    then fixed model order: eapf_consensus, kuramoto. This keeps trial seeds
    invariant whether a run executes all conditions or only one chunk.
    """
    pair = _freq_pair_key((virtual_freq, pi_freq))
    pair_index = {
        _freq_pair_key(freq_pair): idx
        for idx, freq_pair in enumerate(canonical_freq_pairs)
    }
    if pair not in pair_index:
        allowed = ", ".join(f"{v:g}:{p:g}" for v, p in canonical_freq_pairs)
        raise ValueError(
            f"Frequency pair {virtual_freq:g}:{pi_freq:g} is not in "
            f"--canonical-freq-pairs ({allowed}). Add it explicitly to keep "
            "the trial seed schedule well-defined."
        )
    if model not in model_order:
        raise ValueError(f"Model {model!r} is not in canonical model order {model_order}")
    return (
        (repeat - 1) * len(canonical_freq_pairs) * len(model_order)
        + pair_index[pair] * len(model_order)
        + model_order.index(model)
        + 1
    )

def _canonical_pairs_with_current_pairs(
    canonical_freq_pairs: list[tuple[float, float]],
    current_freq_pairs: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Append non-canonical current pairs after the formal canonical schedule."""
    out = list(canonical_freq_pairs)
    seen = {_freq_pair_key(pair) for pair in out}
    for pair in current_freq_pairs:
        key = _freq_pair_key(pair)
        if key not in seen:
            out.append(pair)
            seen.add(key)
    return out

def _wrap_2pi(value: float) -> float:
    return float(value % (2.0 * math.pi))

def _phase_difference_rad(pi_phase_rad: float, virtual_phase_rad: float) -> float:
    return float(((pi_phase_rad - virtual_phase_rad + math.pi) % (2.0 * math.pi)) - math.pi)

def _resolve_initial_phases(
    trial_seed: int | None,
    random_phase: bool,
    virtual_phase_rad: float | None,
    pi_phase_rad: float | None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(trial_seed)
    if random_phase and virtual_phase_rad is None:
        virtual_phase_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    if random_phase and pi_phase_rad is None:
        pi_phase_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    virtual_phase = _wrap_2pi(virtual_phase_rad if virtual_phase_rad is not None else 0.0)
    pi_phase = _wrap_2pi(pi_phase_rad if pi_phase_rad is not None else 0.0)
    return virtual_phase, pi_phase, _phase_difference_rad(pi_phase, virtual_phase)


def _build_schedule(models: list[str], freq_pairs: list[tuple[float, float]],
                    repeats: int, alternate_models: bool,
                    base_seed: int | None = None,
                    canonical_freq_pairs: list[tuple[float, float]] | None = None,
                    random_phase: bool = False,
                    virtual_phase_rad: float | None = None,
                    pi_phase_rad: float | None = None) -> list[TrialSpec]:
    schedule: list[TrialSpec] = []
    canonical_pairs = canonical_freq_pairs or list(freq_pairs)
    for repeat in range(1, repeats + 1):
        for virtual_freq, pi_freq in freq_pairs:
            order = list(models)
            if alternate_models and repeat % 2 == 0:
                order.reverse()
            for model in order:
                stable_index = _canonical_seed_index(
                    model,
                    virtual_freq,
                    pi_freq,
                    repeat,
                    canonical_pairs,
                )
                trial_seed = None if base_seed is None else int(base_seed) + stable_index
                v_phase, p_phase, phase_diff = _resolve_initial_phases(
                    trial_seed,
                    random_phase,
                    virtual_phase_rad,
                    pi_phase_rad,
                )
                schedule.append(TrialSpec(
                    model=model,
                    virtual_freq=virtual_freq,
                    pi_freq=pi_freq,
                    repeat=repeat,
                    trial_id=_make_trial_id(model, virtual_freq, pi_freq, repeat),
                    trial_seed=trial_seed,
                    random_phase_enabled=random_phase,
                    virtual_initial_phase_rad=v_phase,
                    pi_initial_phase_rad=p_phase,
                    initial_phase_difference_rad=phase_diff,
                ))
    return schedule


def _estimate_flash_frequency(times: list[float], start_s: float | None = None) -> float | None:
    selected = [t for t in times if start_s is None or t >= start_s]
    if len(selected) < 2:
        selected = times[-min(len(times), 6):]
    if len(selected) < 2:
        return None
    intervals = [selected[i + 1] - selected[i] for i in range(len(selected) - 1)]
    intervals = [v for v in intervals if v > 0]
    if not intervals:
        return None
    return 1.0 / float(np.median(intervals))


def _nearest_errors(reference_times: list[float], event_times: list[float],
                    start_s: float | None = None) -> list[float]:
    refs = [t for t in reference_times if start_s is None or t >= start_s]
    evs = [t for t in event_times if start_s is None or t >= start_s]
    if not refs or not evs:
        return []
    errors: list[float] = []
    for t in evs:
        nearest = min(refs, key=lambda r: abs(r - t))
        errors.append(t - nearest)
    return errors


def _time_to_frequency_lock(rows: list[dict], threshold_hz: float = 0.05,
                            sustain_s: float = 5.0) -> float | None:
    start: float | None = None
    for row in rows:
        err = _as_float(row.get("frequency_error_hz"))
        t = _as_float(row.get("t_s"))
        if err is None or t is None:
            start = None
            continue
        if abs(err) < threshold_hz:
            if start is None:
                start = t
            if t - start >= sustain_s:
                return start
        else:
            start = None
    return None


def _time_to_timing_lock(virtual_times: list[float], pi_times: list[float],
                         threshold_s: float = 0.10, cycles: int = 5) -> float | None:
    if not virtual_times or not pi_times:
        return None
    streak = 0
    streak_start: float | None = None
    for t in pi_times:
        nearest = min(virtual_times, key=lambda v: abs(v - t))
        if abs(t - nearest) < threshold_s:
            if streak == 0:
                streak_start = t
            streak += 1
            if streak >= cycles:
                return streak_start
        else:
            streak = 0
            streak_start = None
    return None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None


def _format_nullable(value: float | int | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _model_parameters(model: str, kuramoto_gain: float = 5.0) -> dict[str, float]:
    if model == "eapf_consensus":
        return dict(LOCKED_EAPF_PARAMS)
    return {"K": float(kuramoto_gain)}


def _model_parameter_fields(model: str, kuramoto_gain: float = 5.0) -> dict[str, Any]:
    params = _model_parameters(model, kuramoto_gain)
    return {
        "model_parameters_json": json.dumps(params, sort_keys=True),
        "eapf_g_p": params.get("g_p"),
        "eapf_g_f": params.get("g_f"),
        "eapf_alpha_p": params.get("alpha_p"),
        "eapf_alpha_f": params.get("alpha_f"),
        "eapf_delta_theta_max_rad": params.get("delta_theta_max_rad"),
        "eapf_delta_f_max_hz": params.get("delta_f_max_hz"),
        "kuramoto_K": params.get("K"),
    }


def _create_pi_model(model: str, pi_freq: float, virtual_freq: float,
                     pi_initial_phase_rad: float,
                     kuramoto_gain: float) -> tuple[Any, FlashPhaseEstimator | None]:
    if model == "eapf_consensus":
        from firefly_sync.core.event_based_consensus_pll import EventBasedConsensusPLLOscillator
        osc = EventBasedConsensusPLLOscillator()
        for k, v in {
            "phase_gain": 0.02,
            "frequency_gain": 0.02,
            "phase_error_filter_alpha": 0.2,
            "frequency_error_filter_alpha": 0.2,
            "max_phase_step_rad": 0.2,
            "max_frequency_step_hz": 0.05,
            "frequency_min_hz": 0.5,
            "frequency_max_hz": 4.0,
        }.items():
            setattr(osc.config, k, v)
        osc._phase_rad = pi_initial_phase_rad
        osc._frequency_hz = pi_freq
        osc._omega_rad_s = 2.0 * math.pi * pi_freq
        return osc, None

    from firefly_sync.core.kuramoto import KuramotoModel
    osc = KuramotoModel(
        natural_frequency=2.0 * math.pi * pi_freq,
        initial_phase=pi_initial_phase_rad,
        coupling_strength=kuramoto_gain,
        dt=0.01,
    )
    estimator = FlashPhaseEstimator(
        initial_period_guess_s=1.0 / virtual_freq if virtual_freq > 0 else 0.5,
    )
    return osc, estimator


def _step_pi_model(model: str, osc: Any, estimator: FlashPhaseEstimator | None,
                   virtual_event: bool, dt_s: float, t_s: float) -> dict:
    if model == "eapf_consensus":
        return osc.step(dt_s=dt_s, t_s=t_s, neighbour_flash_ids=[0] if virtual_event else [])

    assert estimator is not None
    if virtual_event:
        estimator.record_flash(t_s)
    est_phase = estimator.estimate_phase(t_s)
    phase_before = float(osc.phase)
    phase_error = float(math.atan2(
        math.sin(est_phase - phase_before),
        math.cos(est_phase - phase_before),
    ))
    coupling_input = float(math.sin(est_phase - phase_before))
    saved_dt = osc.dt
    osc.dt = min(dt_s, 0.1)
    state = osc.step(coupling_input)
    osc.dt = saved_dt
    instantaneous_freq = max(
        0.0,
        (osc.natural_frequency + osc.coupling_strength * coupling_input) / (2.0 * math.pi),
    )
    return {
        "phase_rad": round(float(osc.phase), 6),
        "frequency_hz": round(float(instantaneous_freq), 6),
        "phase_error_rad": round(phase_error, 6),
        "freq_error_hz": 0.0,
        "coupling_term": round(coupling_input, 6),
        "follower_flash_event": bool(state.is_firing),
        "fire_count": int(osc.fire_count),
    }


def _get_final_pi_frequency(model: str, osc: Any, pi_flash_times: list[float],
                            duration_s: float) -> float:
    observed = _estimate_flash_frequency(pi_flash_times, start_s=max(0.0, duration_s - 10.0))
    if observed is not None:
        return observed
    if model == "eapf_consensus":
        return float(osc.frequency_hz)
    return float(osc.natural_frequency / (2.0 * math.pi))


def _trial_fields() -> list[str]:
    return [
        "trial_id", "trial_dir", "model", "virtual_initial_freq", "pi_initial_freq",
        "duration", "repeat", "seed", "trial_seed", "random_phase_enabled",
        "virtual_initial_phase_rad", "pi_initial_phase_rad",
        "initial_phase_difference_rad", "feedback_enabled", "model_parameters_json",
        "eapf_g_p", "eapf_g_f", "eapf_alpha_p", "eapf_alpha_f",
        "eapf_delta_theta_max_rad", "eapf_delta_f_max_hz", "kuramoto_K",
        "virtual_fire_count_start",
        "virtual_fire_count_end", "actual_virtual_flashes", "pi_detected_virtual_flashes",
        "actual_detection_fcr", "nominal_expected_flashes", "nominal_fcr",
        "virtual_frequency_start", "virtual_frequency_end", "pi_final_frequency_hz",
        "final_frequency_error_hz", "mean_frequency_error_final_10s",
        "median_frequency_error_final_10s", "mean_timing_error_final_10s",
        "median_timing_error_final_10s", "time_to_frequency_lock_s",
        "time_to_timing_lock_s", "virtual_flash_count", "pi_flash_count",
        "flash_count_ratio", "received_pi_flashes", "pi_flash_posts_received",
        "pi_flash_events_consumed", "virtual_freq_final_5s_mean",
        "pi_freq_final_5s_mean", "frequency_error_final_5s_mean_abs",
        "frequency_error_final_5s_abs_of_means", "virtual_freq_final_5s_std",
        "pi_freq_final_5s_std", "loop_rate_hz",
        "timeout_or_failure", "error_message",
    ]


def run_trial(args: argparse.Namespace, trial_dir: Path, spec: TrialSpec) -> dict:
    trial_dir.mkdir(parents=True, exist_ok=True)
    api_events: list[dict] = []
    terminal_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        terminal_lines.append(msg)

    model_params = _model_parameters(spec.model, args.kuramoto_gain)
    model_param_fields = _model_parameter_fields(spec.model, args.kuramoto_gain)
    metadata = {
        "trial_id": spec.trial_id,
        "model": spec.model,
        "model_parameters": model_params,
        "virtual_initial_freq": spec.virtual_freq,
        "pi_initial_freq": spec.pi_freq,
        "seed": args.seed,
        "trial_seed": spec.trial_seed,
        "random_phase_enabled": spec.random_phase_enabled,
        "virtual_initial_phase_rad": spec.virtual_initial_phase_rad,
        "pi_initial_phase_rad": spec.pi_initial_phase_rad,
        "initial_phase_difference_rad": spec.initial_phase_difference_rad,
        "duration": args.duration,
        "repeat": spec.repeat,
        "feedback_enabled": True,
        "dot_size": args.dot_size,
        "leader_api": args.leader_api,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(trial_dir / "metadata.json", metadata)

    metrics = {field: None for field in _trial_fields()}
    metrics.update({
        "trial_id": spec.trial_id,
        "trial_dir": str(trial_dir),
        "model": spec.model,
        "model_parameters": model_params,
        **model_param_fields,
        "virtual_initial_freq": spec.virtual_freq,
        "pi_initial_freq": spec.pi_freq,
        "seed": args.seed,
        "trial_seed": spec.trial_seed,
        "random_phase_enabled": spec.random_phase_enabled,
        "virtual_initial_phase_rad": round(spec.virtual_initial_phase_rad, 6),
        "pi_initial_phase_rad": round(spec.pi_initial_phase_rad, 6),
        "initial_phase_difference_rad": round(spec.initial_phase_difference_rad, 6),
        "duration": args.duration,
        "repeat": spec.repeat,
        "feedback_enabled": True,
        "timeout_or_failure": False,
        "error_message": "",
    })

    if args.dry_run:
        log(f"[DRY-RUN] {spec.trial_id}: would run {spec.model} "
            f"V={spec.virtual_freq}Hz P={spec.pi_freq}Hz "
            f"seed={spec.trial_seed} phases="
            f"({spec.virtual_initial_phase_rad:.6f}, {spec.pi_initial_phase_rad:.6f})")
        _save_json(trial_dir / "metrics_summary.json", metrics)
        _save_csv(trial_dir / "api_events.csv", api_events)
        (trial_dir / "terminal_summary.txt").write_text("\n".join(terminal_lines))
        return metrics

    detector = None
    led = None
    virtual_fire_count_start = None
    virtual_fire_count_end = None
    virtual_frequency_start = None
    virtual_frequency_end = None
    received_pi_flashes = None
    pi_flash_posts_received_start = None
    pi_flash_posts_received_end = None
    pi_flash_events_consumed_start = None
    pi_flash_events_consumed_end = None
    loop_rate_hz = None

    virtual_detected: list[float] = []
    pi_flash_times: list[float] = []
    detection_log: list[dict] = []
    oscillator_log: list[dict] = []
    flash_events: list[dict] = []

    api = args.leader_api.rstrip("/")
    start_state: dict = {}
    end_state: dict = {}
    trial_start = time.monotonic()

    try:
        log(f"\n[TRIAL] {spec.trial_id}")
        log(f"  initial phases: seed={spec.trial_seed} random_phase={spec.random_phase_enabled} "
            f"virtual={spec.virtual_initial_phase_rad:.6f} rad "
            f"pi={spec.pi_initial_phase_rad:.6f} rad "
            f"diff={spec.initial_phase_difference_rad:.6f} rad")
        _api(api, "/api/mode", "POST", {"mode": "mutual_hil"},
             args.api_timeout, args.api_retries, "POST /api/mode", args.debug, api_events)
        try:
            _api(api, "/api/pause", "POST", timeout=args.api_timeout, retries=1,
                 label="POST /api/pause (best-effort)", debug=args.debug, events=api_events)
        except ApiError:
            pass
        _api(api, "/api/reset", "POST", timeout=args.api_timeout, retries=args.api_retries,
             label="POST /api/reset", debug=args.debug, events=api_events)
        _api(api, "/api/agents/0", "POST", {
            "x": 800,
            "y": 400,
            "size": args.dot_size,
            "initial_frequency_hz": spec.virtual_freq,
            "frequency_hz": spec.virtual_freq,
            "initial_phase_rad": spec.virtual_initial_phase_rad,
            "phase_rad": spec.virtual_initial_phase_rad,
            "model": spec.model,
            "kuramoto_gain": args.kuramoto_gain,
        }, timeout=args.api_timeout, retries=args.api_retries,
            label="POST /api/agents/0", debug=args.debug, events=api_events)
        _api(api, "/api/feedback", "POST", {"enabled": True},
             timeout=args.api_timeout, retries=args.api_retries,
             label="POST /api/feedback enabled=True", debug=args.debug, events=api_events)
        _api(api, "/api/start", "POST", timeout=args.api_timeout, retries=args.api_retries,
             label="POST /api/start", debug=args.debug, events=api_events)

        start_state = _api(api, "/api/agents", "GET", timeout=args.api_timeout,
                           retries=args.api_retries, label="GET /api/agents (start)",
                           debug=args.debug, events=api_events)
        start_a0 = _first_agent(start_state)
        virtual_fire_count_start = _as_int(start_a0.get("fire_count"))
        virtual_frequency_start = _as_float(start_a0.get("frequency_hz"))
        pi_flash_posts_received_start = _as_int(start_a0.get("pi_flash_posts_received"))
        pi_flash_events_consumed_start = _as_int(start_a0.get("pi_flash_events_consumed"))
        log(f"  start: running={start_a0.get('running')} feedback={start_a0.get('feedback_enabled')} "
            f"freq={start_a0.get('frequency_hz')} fire_count={start_a0.get('fire_count')}")

        assert _PicameraFlashDetector is not None
        detector = _PicameraFlashDetector(
            resolution=[args.width, args.height],
            detection_mode="local_contrast",
            min_interval_s=args.min_interval,
            window_s=args.window_s,
        )
        detector.start()
        assert _PiGPIOLED is not None
        led = _PiGPIOLED(pin=17, flash_duration_s=args.flash_duration)

        osc, phase_estimator = _create_pi_model(
            spec.model,
            spec.pi_freq,
            spec.virtual_freq,
            spec.pi_initial_phase_rad,
            args.kuramoto_gain,
        )
        prev_loop = time.monotonic()
        prev_state = False
        last_v_event_t = -999.0
        last_api_sample_t = -999.0
        latest_virtual_frequency = virtual_frequency_start
        loop_dts: list[float] = []

        while (time.monotonic() - trial_start) < args.duration:
            now = time.monotonic()
            dt = now - prev_loop
            prev_loop = now
            loop_dts.append(dt)
            t = now - trial_start

            virtual_event = False
            cam_start = time.monotonic()
            res = detector.capture_frame()
            cam_ms = (time.monotonic() - cam_start) * 1000.0
            detected_state = (res.get("state") == "ON")
            if detected_state and not prev_state and (t - last_v_event_t) >= args.min_interval:
                virtual_event = True
                last_v_event_t = t
            prev_state = detected_state

            if virtual_event:
                virtual_detected.append(t)
                flash_events.append({
                    "t_s": round(t, 6), "event": "virtual_flash",
                    "source": "camera", "model": spec.model,
                })

            if t - last_api_sample_t >= args.api_sample_interval:
                last_api_sample_t = t
                try:
                    sample = _api(api, "/api/agents", "GET", timeout=args.api_timeout,
                                  retries=1, label="GET /api/agents (sample)",
                                  debug=False, events=api_events)
                    a0 = _first_agent(sample)
                    latest_virtual_frequency = _as_float(a0.get("frequency_hz")) or latest_virtual_frequency
                    loop_rate_hz = _as_float(sample.get("loop_rate_hz")) or loop_rate_hz
                except Exception:
                    pass

            step_result = _step_pi_model(spec.model, osc, phase_estimator, virtual_event, dt, t)
            pi_freq = _as_float(step_result.get("frequency_hz"))
            freq_err = (
                abs(pi_freq - latest_virtual_frequency)
                if pi_freq is not None and latest_virtual_frequency is not None else None
            )
            oscillator_log.append({
                "t_s": round(t, 6),
                "model": spec.model,
                "phase_rad": step_result.get("phase_rad"),
                "frequency_hz": pi_freq,
                "virtual_frequency_hz": latest_virtual_frequency,
                "frequency_error_hz": freq_err,
                "phase_error_rad": step_result.get("phase_error_rad"),
                "freq_error_hz": step_result.get("freq_error_hz"),
                "coupling_term": step_result.get("coupling_term", ""),
                "follower_flash_event": int(bool(step_result.get("follower_flash_event"))),
                "loop_dt_ms": round(dt * 1000.0, 3),
            })

            detection_log.append({
                "t_s": round(t, 6),
                "state": int(detected_state),
                "virtual_event": int(virtual_event),
                "brightness": res.get("brightness_used", 0),
                "signal_norm": res.get("signal_norm", 0),
                "camera_processing_time_ms": round(cam_ms, 3),
            })

            if step_result.get("follower_flash_event"):
                pi_flash_times.append(t)
                flash_events.append({
                    "t_s": round(t, 6), "event": "pi_flash",
                    "source": "pi_oscillator", "model": spec.model,
                })
                led.on()
                time.sleep(args.flash_duration)
                led.off()
                try:
                    _api(api, "/api/pi_flash", "POST", {"timestamp": t},
                         timeout=min(args.api_timeout, 3.0), retries=1,
                         label="POST /api/pi_flash", debug=False, events=api_events)
                except Exception:
                    pass

        end_state = _api(api, "/api/agents", "GET", timeout=args.api_timeout,
                         retries=args.api_retries, label="GET /api/agents (end)",
                         debug=args.debug, events=api_events)
        end_a0 = _first_agent(end_state)
        virtual_fire_count_end = _as_int(end_a0.get("fire_count"))
        virtual_frequency_end = _as_float(end_a0.get("frequency_hz"))
        received_pi_flashes = _as_int(end_a0.get("received_pi_flashes"))
        pi_flash_posts_received_end = _as_int(end_a0.get("pi_flash_posts_received"))
        pi_flash_events_consumed_end = _as_int(end_a0.get("pi_flash_events_consumed"))
        loop_rate_hz = _as_float(end_state.get("loop_rate_hz")) or loop_rate_hz

    except Exception as e:
        metrics["timeout_or_failure"] = True
        metrics["error_message"] = f"{type(e).__name__}: {e}"
        log(f"[ERROR] {metrics['error_message']}")
        log(traceback.format_exc())
    finally:
        if not args.dry_run:
            try:
                _api(api, "/api/pause", "POST", timeout=args.api_timeout,
                     retries=args.api_retries, label="POST /api/pause (cleanup)",
                     debug=args.debug, events=api_events)
            except Exception as e:
                log(f"[WARN] cleanup pause failed: {e}")
        if detector is not None:
            detector.stop()
        if led is not None:
            led.close()

    actual_virtual_flashes = (
        virtual_fire_count_end - virtual_fire_count_start
        if virtual_fire_count_start is not None and virtual_fire_count_end is not None
        else None
    )
    nominal_expected = round(args.duration * spec.virtual_freq)
    actual_detection_fcr = (
        len(virtual_detected) / actual_virtual_flashes
        if actual_virtual_flashes and actual_virtual_flashes > 0 else None
    )
    nominal_fcr = len(virtual_detected) / nominal_expected if nominal_expected > 0 else None
    pi_final_freq = _get_final_pi_frequency(spec.model, osc, pi_flash_times, args.duration) if "osc" in locals() else None
    observed_virtual_final_freq = _estimate_flash_frequency(
        virtual_detected, start_s=max(0.0, args.duration - 10.0)
    )
    virtual_final_for_error = observed_virtual_final_freq or virtual_frequency_end
    final_frequency_error = (
        abs(pi_final_freq - virtual_final_for_error)
        if pi_final_freq is not None and virtual_final_for_error is not None else None
    )
    final_5s_start = max(0.0, args.duration - 5.0)
    pi_freq_final_5s_values = [
        float(row["frequency_hz"])
        for row in oscillator_log
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_5s_start
        and _as_float(row.get("frequency_hz")) is not None
    ]
    virtual_freq_final_5s_values = [
        float(row["virtual_frequency_hz"])
        for row in oscillator_log
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_5s_start
        and _as_float(row.get("virtual_frequency_hz")) is not None
    ]
    virtual_freq_final_5s_mean = _mean(virtual_freq_final_5s_values)
    pi_freq_final_5s_mean = _mean(pi_freq_final_5s_values)
    virtual_freq_final_5s_std = _std(virtual_freq_final_5s_values)
    pi_freq_final_5s_std = _std(pi_freq_final_5s_values)
    frequency_error_final_5s_values = [
        abs(float(row["frequency_hz"]) - float(row["virtual_frequency_hz"]))
        for row in oscillator_log
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_5s_start
        and _as_float(row.get("frequency_hz")) is not None
        and _as_float(row.get("virtual_frequency_hz")) is not None
    ]
    frequency_error_final_5s_mean_abs = _mean(frequency_error_final_5s_values)
    frequency_error_final_5s_abs_of_means = (
        abs(pi_freq_final_5s_mean - virtual_freq_final_5s_mean)
        if pi_freq_final_5s_mean is not None and virtual_freq_final_5s_mean is not None else None
    )
    final_window_start = max(0.0, args.duration - 10.0)
    freq_errors_final = [
        abs(float(row["frequency_error_hz"]))
        for row in oscillator_log
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_window_start
        and _as_float(row.get("frequency_error_hz")) is not None
    ]
    timing_errors_final = [abs(v) for v in _nearest_errors(
        virtual_detected, pi_flash_times, start_s=final_window_start
    )]
    time_to_freq_lock = _time_to_frequency_lock(oscillator_log)
    time_to_timing_lock = _time_to_timing_lock(virtual_detected, pi_flash_times)
    flash_count_ratio = (
        len(pi_flash_times) / actual_virtual_flashes
        if actual_virtual_flashes and actual_virtual_flashes > 0 else None
    )

    metrics.update({
        "virtual_fire_count_start": virtual_fire_count_start,
        "virtual_fire_count_end": virtual_fire_count_end,
        "actual_virtual_flashes": actual_virtual_flashes,
        "pi_detected_virtual_flashes": len(virtual_detected),
        "actual_detection_fcr": round(actual_detection_fcr, 4) if actual_detection_fcr is not None else None,
        "nominal_expected_flashes": nominal_expected,
        "nominal_fcr": round(nominal_fcr, 4) if nominal_fcr is not None else None,
        "virtual_frequency_start": round(virtual_frequency_start, 6) if virtual_frequency_start is not None else None,
        "virtual_frequency_end": round(virtual_frequency_end, 6) if virtual_frequency_end is not None else None,
        "pi_final_frequency_hz": round(pi_final_freq, 6) if pi_final_freq is not None else None,
        "final_frequency_error_hz": round(final_frequency_error, 6) if final_frequency_error is not None else None,
        "virtual_freq_final_5s_mean": round(virtual_freq_final_5s_mean, 6) if virtual_freq_final_5s_mean is not None else None,
        "pi_freq_final_5s_mean": round(pi_freq_final_5s_mean, 6) if pi_freq_final_5s_mean is not None else None,
        "frequency_error_final_5s_mean_abs": round(frequency_error_final_5s_mean_abs, 6) if frequency_error_final_5s_mean_abs is not None else None,
        "frequency_error_final_5s_abs_of_means": round(frequency_error_final_5s_abs_of_means, 6) if frequency_error_final_5s_abs_of_means is not None else None,
        "virtual_freq_final_5s_std": round(virtual_freq_final_5s_std, 6) if virtual_freq_final_5s_std is not None else None,
        "pi_freq_final_5s_std": round(pi_freq_final_5s_std, 6) if pi_freq_final_5s_std is not None else None,
        "mean_frequency_error_final_10s": round(_mean(freq_errors_final), 6) if freq_errors_final else None,
        "median_frequency_error_final_10s": round(_median(freq_errors_final), 6) if freq_errors_final else None,
        "mean_timing_error_final_10s": round(_mean(timing_errors_final), 6) if timing_errors_final else None,
        "median_timing_error_final_10s": round(_median(timing_errors_final), 6) if timing_errors_final else None,
        "time_to_frequency_lock_s": round(time_to_freq_lock, 6) if time_to_freq_lock is not None else None,
        "time_to_timing_lock_s": round(time_to_timing_lock, 6) if time_to_timing_lock is not None else None,
        "virtual_flash_count": len(virtual_detected),
        "pi_flash_count": len(pi_flash_times),
        "flash_count_ratio": round(flash_count_ratio, 6) if flash_count_ratio is not None else None,
        "received_pi_flashes": received_pi_flashes,
        "pi_flash_posts_received": (
            pi_flash_posts_received_end
            if pi_flash_posts_received_end is not None else received_pi_flashes
        ),
        "pi_flash_events_consumed": pi_flash_events_consumed_end,
        "pi_flash_posts_received_start": pi_flash_posts_received_start,
        "pi_flash_posts_received_end": pi_flash_posts_received_end,
        "pi_flash_events_consumed_start": pi_flash_events_consumed_start,
        "pi_flash_events_consumed_end": pi_flash_events_consumed_end,
        "loop_rate_hz": round(loop_rate_hz, 3) if loop_rate_hz is not None else None,
    })

    _save_json(trial_dir / "start_state.json", start_state)
    _save_json(trial_dir / "end_state.json", end_state)
    _save_json(trial_dir / "metrics_summary.json", metrics)
    _save_csv(trial_dir / "detection_log.csv", detection_log)
    _save_csv(trial_dir / "oscillator_log.csv", oscillator_log)
    _save_csv(trial_dir / "flash_events.csv", flash_events)
    _save_csv(trial_dir / "api_events.csv", api_events)
    if spec.model == "kuramoto" and not args.dry_run:
        kuramoto_debug_rows = _fetch_kuramoto_debug(api, timeout_s=min(args.api_timeout, 5.0))
        if kuramoto_debug_rows:
            _save_csv(trial_dir / "kuramoto_debug.csv", kuramoto_debug_rows)
    (trial_dir / "terminal_summary.txt").write_text("\n".join(terminal_lines))

    fig_dir = trial_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    if oscillator_log:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot([r["t_s"] for r in oscillator_log], [r["frequency_hz"] for r in oscillator_log],
                label="Pi")
        ax.plot([r["t_s"] for r in oscillator_log], [r["virtual_frequency_hz"] for r in oscillator_log],
                label="Virtual", alpha=0.75)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.legend()
        fig.savefig(fig_dir / "frequency_trajectory.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    if virtual_detected or pi_flash_times:
        fig, ax = plt.subplots(figsize=(9, 2))
        if virtual_detected:
            ax.eventplot(virtual_detected, lineoffsets=1, colors="steelblue", linewidths=0.8)
        if pi_flash_times:
            ax.eventplot(pi_flash_times, lineoffsets=0, colors="darkorange", linewidths=0.8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Pi", "Virtual"])
        ax.set_xlabel("Time (s)")
        fig.savefig(fig_dir / "flash_raster.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log(f"  virtual detected={len(virtual_detected)}/{actual_virtual_flashes} "
        f"actual FCR={_format_nullable(actual_detection_fcr, 3)}")
    log(f"  pi flashes={len(pi_flash_times)} final freq error="
        f"{_format_nullable(final_frequency_error, 4)} Hz")
    log(f"  final 5s error mean_abs={_format_nullable(frequency_error_final_5s_mean_abs, 4)} Hz "
        f"abs_of_means={_format_nullable(frequency_error_final_5s_abs_of_means, 4)} Hz")
    log(f"  pi feedback counters: posts_received={metrics['pi_flash_posts_received']} "
        f"events_consumed={metrics['pi_flash_events_consumed']}")
    (trial_dir / "terminal_summary.txt").write_text("\n".join(terminal_lines))
    return metrics


def _summarise(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, float, float], list[dict]] = {}
    for row in rows:
        key = (row["model"], float(row["virtual_initial_freq"]), float(row["pi_initial_freq"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict] = []
    metrics = [
        "actual_detection_fcr", "final_frequency_error_hz",
        "frequency_error_final_5s_mean_abs",
        "frequency_error_final_5s_abs_of_means",
        "mean_frequency_error_final_10s", "mean_timing_error_final_10s",
        "time_to_frequency_lock_s", "time_to_timing_lock_s",
    ]
    for (model, vf, pf), group in sorted(groups.items()):
        out: dict[str, Any] = {
            "model": model,
            "virtual_initial_freq": vf,
            "pi_initial_freq": pf,
            "n_trials": len(group),
            "success_rate": sum(1 for r in group if not r.get("timeout_or_failure")) / len(group),
            "failure_timeout_count": sum(1 for r in group if r.get("timeout_or_failure")),
        }
        for name in metrics:
            vals = [_as_float(r.get(name)) for r in group]
            vals = [v for v in vals if v is not None]
            out[f"mean_{name}"] = round(_mean(vals), 6) if vals else None
            out[f"std_{name}"] = round(_std(vals), 6) if vals else None
        summary.append(out)
    return summary


def _load_trial_metrics(path: Path) -> dict | None:
    metrics_path = path / "metrics_summary.json"
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("trial_id") else None


def _scan_completed_trial_metrics(trial_root: Path) -> list[dict]:
    rows: list[dict] = []
    if not trial_root.exists():
        return rows
    for child in sorted(trial_root.iterdir()):
        if not child.is_dir():
            continue
        metrics = _load_trial_metrics(child)
        if metrics is not None:
            rows.append(metrics)
    return rows


def _trial_action(trial_dir: Path, resume: bool, overwrite: bool) -> str:
    if not trial_dir.exists():
        return "run"
    if overwrite:
        return "overwrite"
    if resume and _load_trial_metrics(trial_dir) is not None:
        return "skip"
    if resume:
        return "fail_incomplete_existing"
    return "fail_existing"


def _safe_delete_trial_dir(trial_dir: Path, trial_root: Path) -> None:
    resolved_trial = trial_dir.resolve()
    resolved_root = trial_root.resolve()
    if resolved_trial.parent != resolved_root:
        raise RuntimeError(f"Refusing to delete unexpected trial path: {trial_dir}")
    shutil.rmtree(resolved_trial)


def _write_aggregate_outputs(batch_dir: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: str(r.get("trial_id", "")))
    if rows:
        _save_csv(batch_dir / "aggregate_metrics.csv", rows, _trial_fields())
        summary = _summarise(rows)
        _save_csv(batch_dir / "summary_by_model_condition.csv", summary)


def _write_batch_outputs(batch_dir: Path, rows: list[dict], schedule: list[TrialSpec],
                         args: argparse.Namespace) -> None:
    _write_aggregate_outputs(batch_dir, rows)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "duration": args.duration,
        "models": args.models,
        "locked_model_parameters": {
            "eapf_consensus": LOCKED_EAPF_PARAMS,
            "kuramoto": _model_parameters("kuramoto", args.kuramoto_gain),
        },
        "freq_pairs": [{"virtual": v, "pi": p} for v, p in args.freq_pairs],
        "repeats": args.repeats,
        "alternate_models": args.alternate_models,
        "seed": args.seed,
        "canonical_freq_pairs": [
            {"virtual": v, "pi": p}
            for v, p in getattr(args, "canonical_freq_pairs", [])
        ],
        "random_phase_enabled": args.random_phase,
        "virtual_phase_rad": args.virtual_phase_rad,
        "pi_phase_rad": args.pi_phase_rad,
        "dry_run": args.dry_run,
        "run_dir": args.run_dir,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "chunk_label": args.chunk_label,
        "schedule": [spec.__dict__ for spec in schedule],
    }
    _save_json(batch_dir / "batch_metadata.json", metadata)
    _save_json(batch_dir / "run_metadata.json", metadata)


def _append_chunk_metadata(batch_dir: Path, chunk_record: dict) -> None:
    chunks_dir = batch_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = chunk_record["chunk_id"]
    _save_json(chunks_dir / f"{chunk_id}.json", chunk_record)
    combined_path = batch_dir / "chunk_metadata.json"
    existing: list[dict] = []
    if combined_path.exists():
        try:
            with open(combined_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []
    existing.append(chunk_record)
    with open(combined_path, "w") as f:
        json.dump(existing, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5B mutual HIL model comparison batch.")
    parser.add_argument("--leader-api", default="http://192.168.1.111:8000")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS,
                        default=list(SUPPORTED_MODELS))
    parser.add_argument("--freq-pairs", type=_parse_freq_pairs,
                        default=_parse_freq_pairs(DEFAULT_FREQ_PAIRS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dot-size", type=int, default=450)
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--alternate-models", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/step5b_mutual_model_comparison")
    parser.add_argument("--run-dir", default=None,
                        help="Use this exact shared batch directory; no timestamp subfolder is added")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed existing trial folders and rebuild aggregate outputs")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete and rerun scheduled trial folders that already exist")
    parser.add_argument("--chunk-label", default=None,
                        help="Optional label stored in chunk metadata for --run-dir invocations")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--flash-duration", type=float, default=0.06)
    parser.add_argument("--kuramoto-gain", type=float, default=5.0)
    parser.add_argument("--api-sample-interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260620,
                        help="Base seed for deterministic per-trial random phases")
    parser.add_argument("--canonical-freq-pairs", type=_parse_freq_pairs,
                        default=_parse_freq_pairs(DEFAULT_CANONICAL_FREQ_PAIRS),
                        help=(
                            "Frequency-pair order used only for stable seed indexing. "
                            "Formal default: 2.0:1.2,2.0:1.5,2.0:2.5"
                        ))
    parser.add_argument("--random-phase", dest="random_phase", action="store_true", default=True,
                        help="Randomise initial phases per trial (default for formal Step 5B)")
    parser.add_argument("--no-random-phase", dest="random_phase", action="store_false",
                        help="Use explicit phase args or zero phases for compatibility")
    parser.add_argument("--virtual-phase-rad", type=float, default=None,
                        help="Explicit virtual initial phase for all trials")
    parser.add_argument("--pi-phase-rad", type=float, default=None,
                        help="Explicit Pi initial phase for all trials")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    args.canonical_freq_pairs = _canonical_pairs_with_current_pairs(
        args.canonical_freq_pairs,
        args.freq_pairs,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    using_run_dir = args.run_dir is not None
    batch_dir = (
        Path(args.run_dir)
        if using_run_dir
        else Path(args.log_dir) / f"{ts}_step5b_mutual_model_comparison"
    )
    trial_root = batch_dir if using_run_dir else batch_dir / "trials"

    schedule = _build_schedule(
        args.models,
        args.freq_pairs,
        args.repeats,
        args.alternate_models,
        base_seed=args.seed,
        canonical_freq_pairs=args.canonical_freq_pairs,
        random_phase=args.random_phase,
        virtual_phase_rad=args.virtual_phase_rad,
        pi_phase_rad=args.pi_phase_rad,
    )
    run_exists = batch_dir.exists()
    action_plan: list[tuple[TrialSpec, Path, str]] = []
    unsafe: list[str] = []
    for spec in schedule:
        trial_dir = trial_root / spec.trial_id
        action = _trial_action(trial_dir, args.resume, args.overwrite)
        action_plan.append((spec, trial_dir, action))
        if action == "fail_existing":
            unsafe.append(
                f"{trial_dir} already exists. Use --resume to skip completed "
                "trials or --overwrite to rerun."
            )
        elif action == "fail_incomplete_existing":
            unsafe.append(
                f"{trial_dir} already exists but has no metrics_summary.json; "
                "resume only skips completed trials. Use --overwrite or inspect it."
            )

    print(f"Batch dir: {batch_dir}")
    if using_run_dir:
        print(f"Run-dir mode: exact shared directory ({'exists' if run_exists else 'will create'})")
        print("Trial layout: direct children of run directory")
    else:
        print("Legacy mode: timestamped directory under --log-dir")
        print(f"Trial layout: {trial_root}")
    print(f"Phase control: seed={args.seed} random_phase={args.random_phase} "
          f"virtual_phase_arg={args.virtual_phase_rad} pi_phase_arg={args.pi_phase_rad}")
    print("Stable seed schedule: repeat -> canonical freq pair -> model "
          "(model order: eapf_consensus, kuramoto)")
    print("Canonical freq pairs: " + ", ".join(
        f"{v:g}:{p:g}" for v, p in args.canonical_freq_pairs
    ))
    print("Schedule:")
    for i, (spec, trial_dir, action) in enumerate(action_plan, 1):
        print(f"  {i:02d}. {spec.trial_id} model={spec.model} "
              f"virtual={spec.virtual_freq:g}Hz pi={spec.pi_freq:g}Hz "
              f"seed={spec.trial_seed} "
              f"phases=({spec.virtual_initial_phase_rad:.4f}, "
              f"{spec.pi_initial_phase_rad:.4f}) "
              f"diff={spec.initial_phase_difference_rad:.4f} "
              f"action={action} path={trial_dir}")

    if unsafe and not args.dry_run:
        print("\nRefusing to continue because existing trial folders would be overwritten:")
        for msg in unsafe:
            print(f"  - {msg}")
        sys.exit(2)

    if args.dry_run:
        print("[DRY-RUN] No hardware/API trial executed.")
        if unsafe:
            print("[DRY-RUN] Default real run would fail safely on:")
            for msg in unsafe:
                print(f"  - {msg}")
        if using_run_dir:
            batch_dir.mkdir(parents=True, exist_ok=True)
            chunk_record = {
                "chunk_id": ts,
                "chunk_label": args.chunk_label,
                "dry_run": True,
                "command_line": sys.argv,
                "models": args.models,
                "freq_pairs": [{"virtual": v, "pi": p} for v, p in args.freq_pairs],
                "canonical_freq_pairs": [{"virtual": v, "pi": p} for v, p in args.canonical_freq_pairs],
                "repeats": args.repeats,
                "duration": args.duration,
                "random_phase_enabled": args.random_phase,
                "seed": args.seed,
                "started_at": datetime.now().isoformat(),
                "finished_at": datetime.now().isoformat(),
                "completed_trial_ids": [],
                "skipped_trial_ids": [s.trial_id for s, _p, a in action_plan if a == "skip"],
                "failed_trial_ids": [],
                "planned_trial_ids": [s.trial_id for s, _p, _a in action_plan],
            }
            _append_chunk_metadata(batch_dir, chunk_record)
            _write_batch_outputs(batch_dir, _scan_completed_trial_metrics(trial_root), schedule, args)
        return

    batch_dir.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)
    if any(action != "skip" for _spec, _trial_dir, action in action_plan):
        ensure_runtime_imports(dry=args.dry_run)

    chunk_started_at = datetime.now().isoformat()
    completed_trial_ids: list[str] = []
    skipped_trial_ids: list[str] = []
    failed_trial_ids: list[str] = []

    for spec, trial_dir, action in action_plan:
        if action == "skip":
            print(f"[RESUME] Skipping completed trial {spec.trial_id}: {trial_dir}")
            skipped_trial_ids.append(spec.trial_id)
            _write_batch_outputs(batch_dir, _scan_completed_trial_metrics(trial_root), schedule, args)
            continue
        if action == "overwrite":
            print(f"[OVERWRITE] Deleting existing trial folder: {trial_dir}")
            _safe_delete_trial_dir(trial_dir, trial_root)
        row = run_trial(args, trial_dir, spec)
        if row.get("timeout_or_failure"):
            failed_trial_ids.append(spec.trial_id)
        else:
            completed_trial_ids.append(spec.trial_id)
        _write_batch_outputs(batch_dir, _scan_completed_trial_metrics(trial_root), schedule, args)

    if using_run_dir:
        chunk_record = {
            "chunk_id": ts,
            "chunk_label": args.chunk_label,
            "dry_run": False,
            "command_line": sys.argv,
            "models": args.models,
            "freq_pairs": [{"virtual": v, "pi": p} for v, p in args.freq_pairs],
            "canonical_freq_pairs": [{"virtual": v, "pi": p} for v, p in args.canonical_freq_pairs],
            "repeats": args.repeats,
            "duration": args.duration,
            "random_phase_enabled": args.random_phase,
            "seed": args.seed,
            "started_at": chunk_started_at,
            "finished_at": datetime.now().isoformat(),
            "completed_trial_ids": completed_trial_ids,
            "skipped_trial_ids": skipped_trial_ids,
            "failed_trial_ids": failed_trial_ids,
        }
        _append_chunk_metadata(batch_dir, chunk_record)
        _write_batch_outputs(batch_dir, _scan_completed_trial_metrics(trial_root), schedule, args)

    print(f"\nBatch complete: {batch_dir}")
    print(f"  aggregate: {batch_dir / 'aggregate_metrics.csv'}")
    print(f"  summary:   {batch_dir / 'summary_by_model_condition.csv'}")


if __name__ == "__main__":
    main()

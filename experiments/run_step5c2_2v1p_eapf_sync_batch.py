#!/usr/bin/env python3
"""Step5c2 2-virtual + 1-Pi EAPF all-to-all mutual HIL batch runner.

This stage is mixed-reality hardware-in-the-loop: V0 and V1 are browser-rendered
virtual flash targets, while P0 is the Raspberry Pi camera/LED node.  Step5c2 is
EAPF Consensus only; it does not re-run the Kuramoto comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import run_2v1p_eapf_hil as hil
from experiments.run_2v1p_eapf_hil import (
    _api,
    _locked_eapf_config,
    _make_multi_roi_detector,
    _parse_roi,
    _save_roi_debug_frame,
    _start_server_trial,
    _write_csv,
    _write_json,
    run_auto_roi_calibration,
)
from firefly_sync.core.event_based_consensus_pll import EventBasedConsensusPLLOscillator
from firefly_sync.multi_agent.hil_topology import build_mixed_reality_topology


DEFAULT_LOG_DIR = "experiments/logs/step5c2_2v1p_eapf_sync"
AGENTS = ("V0", "V1", "P0")
LOCKED_EAPF_PARAMETERS = {
    "g_p": 0.02,
    "g_f": 0.02,
    "alpha_p": 0.2,
    "alpha_f": 0.2,
    "max_phase_step_rad": 0.2,
    "max_frequency_step_hz": 0.05,
}


@dataclass(frozen=True)
class SyncThresholds:
    final_window_s: float = 10.0
    required_window_s: float = 5.0
    window_pass_ratio: float = 0.8
    mean_pairwise_phase_error_cycles: float = 0.25
    mean_pairwise_phase_error_rad: float = math.pi / 2.0
    frequency_disagreement_hz: float = 0.10
    order_parameter_R: float = 0.90
    frequency_stability_std_hz: float = 0.05
    frequency_stability_slope_hz_per_s: float = 0.005


@dataclass(frozen=True)
class LockHoldConfig:
    stabilizer: str = "none"
    lock_r_threshold: float = 0.95
    lock_phase_error_threshold_cycles: float = 0.08
    lock_frequency_disagreement_threshold_hz: float = 0.05
    lock_window_s: float = 1.0
    lock_window_pass_ratio: float = 0.5
    hold_frequency_anchor: str = "window_median"
    hold_phase_gain_scale: float = 0.1
    hold_frequency_gain_scale: float = 0.0
    hold_anchor_gain: float = 0.08
    unlock_r_threshold: float = 0.85
    unlock_phase_error_threshold_cycles: float = 0.15
    unlock_frequency_disagreement_threshold_hz: float = 0.10
    unlock_window_s: float = 4.0
    unlock_window_fail_ratio: float = 0.8


FREQUENCY_SETS: dict[str, dict[str, Any]] = {
    "random_1p6_2p4": {
        "description": "Legacy default: all agents random from 1.6-2.4 Hz.",
        "agents": {agent_id: (1.6, 2.4) for agent_id in AGENTS},
    },
    "same_2hz_random_phase": {
        "description": "All agents near 2.0 Hz; isolates phase synchronisation.",
        "agents": {agent_id: (1.95, 2.05) for agent_id in AGENTS},
    },
    "close_1p8_2p2": {
        "description": "Easy practical range: all agents random from 1.8-2.2 Hz.",
        "agents": {agent_id: (1.8, 2.2) for agent_id in AGENTS},
    },
    "nominal_1_2": {
        "description": "Consistent with detection validation: V0 around 1 Hz, V1 around 2 Hz.",
        "agents": {"V0": (0.95, 1.05), "V1": (1.95, 2.05), "P0": (1.45, 1.55)},
    },
    "wide_1_3": {
        "description": "Harder initial disagreement: all agents random from 1.0-3.0 Hz.",
        "agents": {agent_id: (1.0, 3.0) for agent_id in AGENTS},
    },
    "mixed_low_mid_high": {
        "description": "Structured disagreement: low, medium, and high initial frequencies.",
        "agents": {"V0": (1.15, 1.25), "V1": (1.75, 1.85), "P0": (2.35, 2.45)},
    },
}


def _condition_definitions() -> dict[str, dict[str, Any]]:
    base = {
        "name": "baseline",
        "stage": "Step5c2-A/B",
        "category": "baseline",
        "description": "Clean all-to-all EAPF Consensus mixed-reality synchronisation.",
        "display": {},
        "fault": {"kind": "none"},
        "required_for_readiness": True,
        "failure_interpretation": (
            "Baseline failure should be investigated before robustness testing."
        ),
    }

    def c(name: str, **updates: Any) -> dict[str, Any]:
        item = deepcopy(base)
        item["name"] = name
        item.update(updates)
        return item

    return {
        "baseline": c("baseline"),
        "baseline_pure_eapf": c(
            "baseline_pure_eapf",
            stage="Step5c2 smoke comparison",
            description=(
                "Clean all-to-all pure EAPF smoke condition; expected to show "
                "relative synchrony but possible common-frequency drift."
            ),
        ),
        "baseline_lock_hold": c(
            "baseline_lock_hold",
            stage="Step5c2 smoke comparison",
            description=(
                "Clean all-to-all EAPF with lock-and-hold stabilisation enabled."
            ),
        ),
        "baseline_random_initial": c(
            "baseline_random_initial",
            stage="Step5c2-B",
            description=(
                "Repeated all-to-all EAPF trials with random initial phase and "
                "frequency for V0, V1, and P0."
            ),
        ),
        "v0_low_contrast": c(
            "v0_low_contrast",
            stage="Step5c2-C",
            category="robustness",
            description=(
                "Boundary visual degradation on V0; use cautiously because Step5c1 "
                "low contrast may be suppressed by the detector amplitude gate."
            ),
            display={"V0": {"flash_brightness": 170, "off_brightness": 60}},
            fault={"kind": "display_degradation", "target": "V0"},
            required_for_readiness=False,
            failure_interpretation=(
                "A failure here is a contrast boundary case unless a later low-contrast "
                "rerun confirms detection readiness."
            ),
        ),
        "v1_low_contrast": c(
            "v1_low_contrast",
            stage="Step5c2-C",
            category="robustness",
            description=(
                "Boundary visual degradation on V1; use cautiously because Step5c1 "
                "low contrast remains a detection boundary."
            ),
            display={"V1": {"flash_brightness": 170, "off_brightness": 60}},
            fault={"kind": "display_degradation", "target": "V1"},
            required_for_readiness=False,
            failure_interpretation=(
                "A failure here is a contrast boundary case unless a later low-contrast "
                "rerun confirms detection readiness."
            ),
        ),
        "p0_event_delay_150ms": c(
            "p0_event_delay_150ms",
            stage="Step5c2-C",
            category="robustness",
            description="Delay P0 flash event routing to the virtual agents by about 150 ms.",
            fault={"kind": "p0_event_delay", "delay_s": 0.150},
            failure_interpretation=(
                "Delay failures should be reported as timing-loop limits, not display bugs."
            ),
        ),
        "v0_event_dropout_20percent": c(
            "v0_event_dropout_20percent",
            stage="Step5c2-C",
            category="robustness",
            description="Drop about 20% of V0 camera-detected events before P0 consumes them.",
            fault={"kind": "virtual_event_dropout", "source": "V0", "drop_probability": 0.20},
            failure_interpretation=(
                "Dropout failures identify visual/event-stream robustness limits."
            ),
        ),
        "temporary_v1_pause_5s": c(
            "temporary_v1_pause_5s",
            stage="Step5c2-C",
            category="robustness",
            description="Pause V1 flashing for 5 s, then restore it.",
            fault={"kind": "temporary_virtual_pause", "target": "V1", "duration_s": 5.0},
            failure_interpretation=(
                "The key question is recovery after the pause, not immunity during the pause."
            ),
        ),
        "p0_event_delay_300ms": c(
            "p0_event_delay_300ms",
            stage="Step5c2-C optional",
            category="stress",
            description="Optional later stress: delay P0 flash routing by about 300 ms.",
            fault={"kind": "p0_event_delay", "delay_s": 0.300},
            required_for_readiness=False,
        ),
        "v0_event_dropout_30percent": c(
            "v0_event_dropout_30percent",
            stage="Step5c2-C optional",
            category="stress",
            description="Optional later stress: drop about 30% of V0 detected events.",
            fault={"kind": "virtual_event_dropout", "source": "V0", "drop_probability": 0.30},
            required_for_readiness=False,
        ),
    }


def _locked_eapf_parameters_from_config() -> dict[str, float]:
    cfg = _locked_eapf_config(2.0)
    return {
        "g_p": cfg.phase_gain,
        "g_f": cfg.frequency_gain,
        "alpha_p": cfg.phase_error_filter_alpha,
        "alpha_f": cfg.frequency_error_filter_alpha,
        "max_phase_step_rad": cfg.max_phase_step_rad,
        "max_frequency_step_hz": cfg.max_frequency_step_hz,
    }


def verify_locked_eapf_config() -> None:
    actual = _locked_eapf_parameters_from_config()
    mismatches = {
        key: (actual[key], expected)
        for key, expected in LOCKED_EAPF_PARAMETERS.items()
        if not math.isclose(float(actual[key]), float(expected), rel_tol=0.0, abs_tol=1e-12)
    }
    if mismatches:
        raise RuntimeError(f"Locked EAPF config mismatch: {mismatches}")


def generate_initial_states(
    seed: int,
    trial_index: int,
    *,
    random_initial: bool,
    frequency_min_hz: float = 1.6,
    frequency_max_hz: float = 2.4,
    freq_set: str = "random_1p6_2p4",
) -> dict[str, dict[str, float]]:
    rng = random.Random(int(seed) + int(trial_index) * 1009)
    defaults = {"V0": 1.9, "V1": 2.1, "P0": 2.0}
    freq_spec = FREQUENCY_SETS.get(freq_set, FREQUENCY_SETS["random_1p6_2p4"])
    states: dict[str, dict[str, float]] = {}
    for agent_id in AGENTS:
        phase = rng.uniform(0.0, 2.0 * math.pi) if random_initial else 0.0
        if random_initial:
            lo, hi = freq_spec.get("agents", {}).get(
                agent_id,
                (float(frequency_min_hz), float(frequency_max_hz)),
            )
            freq = rng.uniform(float(lo), float(hi))
        else:
            freq = defaults[agent_id]
        states[agent_id] = {
            "initial_phase_rad": round(phase, 9),
            "initial_frequency_hz": round(freq, 9),
        }
    return states


class EventFaultInjector:
    """Bookkeeping and event transforms for mild Step5c2 robustness trials."""

    def __init__(self, condition: dict[str, Any], rng: random.Random) -> None:
        self.condition = condition
        self.fault = condition.get("fault", {"kind": "none"})
        self.rng = rng
        self.delayed_p0_events: list[dict[str, Any]] = []
        self.stats = {
            "condition": condition["name"],
            "fault_kind": self.fault.get("kind", "none"),
            "virtual_events_seen": 0,
            "virtual_events_dropped": 0,
            "p0_events_seen": 0,
            "p0_events_delayed": 0,
            "p0_delayed_events_posted": 0,
            "pause_applied": False,
            "pause_restored": False,
        }

    def keep_virtual_detection(self, source_agent_id: str) -> bool:
        self.stats["virtual_events_seen"] += 1
        if self.fault.get("kind") != "virtual_event_dropout":
            return True
        if source_agent_id != self.fault.get("source"):
            return True
        drop_probability = float(self.fault.get("drop_probability", 0.0))
        if self.rng.random() < drop_probability:
            self.stats["virtual_events_dropped"] += 1
            return False
        return True

    def queue_or_post_p0_flash(
        self,
        *,
        t_s: float,
        post_event,
    ) -> None:
        self.stats["p0_events_seen"] += 1
        if self.fault.get("kind") == "p0_event_delay":
            delay_s = float(self.fault.get("delay_s", 0.0))
            self.delayed_p0_events.append({"source_t_s": t_s, "due_t_s": t_s + delay_s})
            self.stats["p0_events_delayed"] += 1
        else:
            post_event(t_s, t_s, 0.0)

    def flush_due_p0_events(self, now_s: float, post_event) -> None:
        ready = [item for item in self.delayed_p0_events if item["due_t_s"] <= now_s]
        self.delayed_p0_events = [
            item for item in self.delayed_p0_events if item["due_t_s"] > now_s
        ]
        for item in ready:
            post_event(item["source_t_s"], now_s, now_s - item["source_t_s"])
            self.stats["p0_delayed_events_posted"] += 1

    def maybe_apply_pause(
        self,
        *,
        t_s: float,
        duration_s: float,
        args: argparse.Namespace,
        api_events: list[dict[str, Any]],
    ) -> dict[str, float | None]:
        if self.fault.get("kind") != "temporary_virtual_pause":
            return {"pause_start_s": None, "pause_end_s": None}
        pause_len = float(self.fault.get("duration_s", 5.0))
        start_s = max(1.0, min(duration_s * 0.35, max(1.0, duration_s - pause_len - 1.0)))
        end_s = min(duration_s, start_s + pause_len)
        target = str(self.fault.get("target", "V1"))
        target_idx = 1 if target == "V1" else 0
        if start_s <= t_s < end_s and not self.stats["pause_applied"]:
            _api(args.leader_api, f"/api/agents/{target_idx}", "POST",
                 {"enabled": False}, timeout=args.api_timeout, events=api_events)
            self.stats["pause_applied"] = True
        if t_s >= end_s and self.stats["pause_applied"] and not self.stats["pause_restored"]:
            _api(args.leader_api, f"/api/agents/{target_idx}", "POST",
                 {"enabled": True}, timeout=args.api_timeout, events=api_events)
            self.stats["pause_restored"] = True
        return {"pause_start_s": start_s, "pause_end_s": end_s}


class LockHoldStabilizer:
    """State machine for optional EAPF acquisition followed by frequency hold."""

    def __init__(self, config: LockHoldConfig) -> None:
        self.config = config
        self.enabled = config.stabilizer == "lock_hold"
        self.state = "acquisition"
        self.f_lock_hz: float | None = None
        self.lock_time_s: float | None = None
        self.unlock_count = 0
        self.relock_count = 0
        self.f_lock_source_window_s: float | None = None
        self.common_frequency_at_lock_mean_hz: float | None = None
        self.common_frequency_at_lock_median_hz: float | None = None
        self.common_frequency_at_lock_std_hz: float | None = None
        self._lock_history: list[dict[str, Any]] = []
        self._unlock_history: list[dict[str, Any]] = []
        self._acquisition_start_t_s: float | None = None
        self._hold_enter_t_s: float | None = None
        self.rows: list[dict[str, Any]] = []

    def _lock_sample_pass(self, sample: dict[str, Any]) -> bool:
        return (
            float(sample["order_parameter_R"]) >= self.config.lock_r_threshold
            and float(sample["mean_pairwise_phase_error_cycles"])
            <= self.config.lock_phase_error_threshold_cycles
            and float(sample["frequency_disagreement_hz"])
            <= self.config.lock_frequency_disagreement_threshold_hz
        )

    def _unlock_sample_fail(self, sample: dict[str, Any]) -> bool:
        return (
            float(sample["order_parameter_R"]) < self.config.unlock_r_threshold
            or float(sample["mean_pairwise_phase_error_cycles"])
            > self.config.unlock_phase_error_threshold_cycles
            or float(sample["frequency_disagreement_hz"])
            > self.config.unlock_frequency_disagreement_threshold_hz
        )

    def _history_span_s(self, history: list[dict[str, Any]]) -> float:
        if len(history) < 2:
            return 0.0
        return float(history[-1]["t_s"]) - float(history[0]["t_s"])

    def _lock_pass_ratio(self) -> float:
        if not self._lock_history:
            return 0.0
        return (
            sum(1 for s in self._lock_history if self._lock_sample_pass(s))
            / len(self._lock_history)
        )

    def _unlock_fail_ratio(self) -> float:
        if not self._unlock_history:
            return 0.0
        return (
            sum(1 for s in self._unlock_history if self._unlock_sample_fail(s))
            / len(self._unlock_history)
        )

    def _set_lock_anchor_from_history(self, sample: dict[str, Any]) -> None:
        freqs = [
            float(s["common_frequency_hz"])
            for s in self._lock_history
            if math.isfinite(float(s.get("common_frequency_hz", float("nan"))))
        ]
        if not freqs:
            freqs = [float(sample["common_frequency_hz"])]
        self.common_frequency_at_lock_mean_hz = float(statistics.mean(freqs))
        self.common_frequency_at_lock_median_hz = float(statistics.median(freqs))
        self.common_frequency_at_lock_std_hz = (
            float(statistics.pstdev(freqs)) if len(freqs) >= 2 else 0.0
        )
        self.f_lock_source_window_s = self._history_span_s(self._lock_history)
        if self.config.hold_frequency_anchor == "current_mean":
            self.f_lock_hz = float(sample["common_frequency_hz"])
        elif self.config.hold_frequency_anchor == "window_mean":
            self.f_lock_hz = self.common_frequency_at_lock_mean_hz
        else:
            self.f_lock_hz = self.common_frequency_at_lock_median_hz

    def update(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        t_s = float(sample["t_s"])
        event: dict[str, Any] | None = None
        if self.enabled:
            if self._acquisition_start_t_s is None and self.state == "acquisition":
                self._acquisition_start_t_s = t_s
            self._lock_history.append(sample)
            self._unlock_history.append(sample)
            eps = 1e-9
            self._lock_history = [
                s for s in self._lock_history
                if t_s - float(s["t_s"]) <= self.config.lock_window_s + eps
            ]
            self._unlock_history = [
                s for s in self._unlock_history
                if t_s - float(s["t_s"]) <= self.config.unlock_window_s + eps
            ]
            if self.state == "acquisition":
                span_ok = (
                    self._acquisition_start_t_s is not None
                    and t_s - self._acquisition_start_t_s + eps >= self.config.lock_window_s
                )
                pass_ratio = self._lock_pass_ratio()
                if span_ok and pass_ratio >= self.config.lock_window_pass_ratio:
                    self._set_lock_anchor_from_history(sample)
                    self.state = "hold"
                    self._hold_enter_t_s = t_s
                    self._unlock_history = [sample]
                    if self.lock_time_s is None:
                        self.lock_time_s = t_s
                    else:
                        self.relock_count += 1
                    event = {
                        "t_s": round(t_s, 6),
                        "event_type": "lock_hold_acquired",
                        "agent_id": "stabilizer",
                        "f_lock_hz": round(self.f_lock_hz, 6),
                        "f_lock_source_window_s": round(self.f_lock_source_window_s, 6),
                        "common_frequency_at_lock_mean_hz": round(
                            self.common_frequency_at_lock_mean_hz, 6
                        ),
                        "common_frequency_at_lock_median_hz": round(
                            self.common_frequency_at_lock_median_hz, 6
                        ),
                        "common_frequency_at_lock_std_hz": round(
                            self.common_frequency_at_lock_std_hz, 6
                        ),
                        "lock_pass_ratio": round(pass_ratio, 6),
                    }
            elif self.state == "hold":
                span_ok = (
                    self._hold_enter_t_s is not None
                    and t_s - self._hold_enter_t_s + eps >= self.config.unlock_window_s
                )
                fail_ratio = self._unlock_fail_ratio()
                if span_ok and fail_ratio >= self.config.unlock_window_fail_ratio:
                    self.state = "acquisition"
                    self.unlock_count += 1
                    self._acquisition_start_t_s = t_s
                    self._hold_enter_t_s = None
                    self._lock_history = [sample]
                    event = {
                        "t_s": round(t_s, 6),
                        "event_type": "lock_hold_unlocked",
                        "agent_id": "stabilizer",
                        "f_lock_hz": round(self.f_lock_hz, 6) if self.f_lock_hz is not None else None,
                        "unlock_fail_ratio": round(fail_ratio, 6),
                    }
        row = {
            "t_s": round(t_s, 6),
            "lock_hold_enabled": int(self.enabled),
            "lock_hold_state": self.state if self.enabled else "disabled",
            "f_lock_hz": round(self.f_lock_hz, 6) if self.f_lock_hz is not None else "",
            "lock_acquired": int(self.lock_time_s is not None),
            "lock_time_s": round(self.lock_time_s, 6) if self.lock_time_s is not None else "",
            "f_lock_source_window_s": (
                round(self.f_lock_source_window_s, 6)
                if self.f_lock_source_window_s is not None else ""
            ),
            "common_frequency_at_lock_mean_hz": (
                round(self.common_frequency_at_lock_mean_hz, 6)
                if self.common_frequency_at_lock_mean_hz is not None else ""
            ),
            "common_frequency_at_lock_median_hz": (
                round(self.common_frequency_at_lock_median_hz, 6)
                if self.common_frequency_at_lock_median_hz is not None else ""
            ),
            "common_frequency_at_lock_std_hz": (
                round(self.common_frequency_at_lock_std_hz, 6)
                if self.common_frequency_at_lock_std_hz is not None else ""
            ),
            "unlock_count": self.unlock_count,
            "relock_count": self.relock_count,
            "order_parameter_R": round(float(sample["order_parameter_R"]), 6),
            "mean_pairwise_phase_error_cycles": round(
                float(sample["mean_pairwise_phase_error_cycles"]), 6
            ),
            "frequency_disagreement_hz": round(float(sample["frequency_disagreement_hz"]), 6),
            "common_frequency_hz": round(float(sample["common_frequency_hz"]), 6),
        }
        self.rows.append(row)
        return event

    def api_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "f_lock_hz": self.f_lock_hz,
            "hold_phase_gain_scale": self.config.hold_phase_gain_scale,
            "hold_frequency_gain_scale": self.config.hold_frequency_gain_scale,
            "hold_anchor_gain": self.config.hold_anchor_gain,
        }


def _phase_distance_rad(a: float, b: float) -> float:
    return abs(((float(a) - float(b) + math.pi) % (2.0 * math.pi)) - math.pi)


def _order_parameter(phases: list[float]) -> float:
    if not phases:
        return float("nan")
    z = sum(complex(math.cos(p), math.sin(p)) for p in phases) / len(phases)
    return abs(z)


def _samples_from_agent_rows(agent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, dict[str, dict[str, float]]] = {}
    for row in agent_rows:
        try:
            t_s = round(float(row["t_s"]), 6)
            phase = float(row["phase_rad"])
            freq = float(row["frequency_hz"])
        except (TypeError, ValueError, KeyError):
            continue
        grouped.setdefault(t_s, {})[str(row.get("agent_id"))] = {
            "phase_rad": phase,
            "frequency_hz": freq,
        }
    samples = []
    for t_s in sorted(grouped):
        agents = grouped[t_s]
        if all(agent_id in agents for agent_id in AGENTS):
            phases = [agents[agent_id]["phase_rad"] for agent_id in AGENTS]
            freqs = [agents[agent_id]["frequency_hz"] for agent_id in AGENTS]
            common_frequency_hz = float(np.mean(freqs))
            pairwise = [
                _phase_distance_rad(phases[i], phases[j])
                for i in range(len(phases))
                for j in range(i + 1, len(phases))
            ]
            samples.append({
                "t_s": t_s,
                "mean_pairwise_phase_error_rad": float(np.mean(pairwise)),
                "max_pairwise_phase_error_rad": float(max(pairwise)),
                "mean_pairwise_phase_error_cycles": float(np.mean(pairwise) / (2.0 * math.pi)),
                "max_pairwise_phase_error_cycles": float(max(pairwise) / (2.0 * math.pi)),
                "frequency_disagreement_hz": float(max(freqs) - min(freqs)),
                "common_frequency_hz": common_frequency_hz,
                "order_parameter_R": _order_parameter(phases),
                "frequencies_hz": dict(zip(AGENTS, freqs)),
                "phases_rad": dict(zip(AGENTS, phases)),
            })
    return samples


def _sample_is_sync(sample: dict[str, Any], thresholds: SyncThresholds) -> bool:
    phase_ok = (
        float(sample["mean_pairwise_phase_error_cycles"])
        < thresholds.mean_pairwise_phase_error_cycles
    )
    r_ok = float(sample["order_parameter_R"]) > thresholds.order_parameter_R
    freq_ok = (
        float(sample["frequency_disagreement_hz"])
        < thresholds.frequency_disagreement_hz
    )
    return bool((phase_ok or r_ok) and freq_ok)


def _window_pass_ratio(
    samples: list[dict[str, Any]],
    *,
    start_t: float,
    end_t: float,
    predicate,
) -> tuple[float, list[dict[str, Any]]]:
    window = [s for s in samples if start_t <= float(s["t_s"]) <= end_t]
    if not window:
        return 0.0, window
    passes = sum(1 for sample in window if predicate(sample))
    return passes / len(window), window


def _time_to_sync(
    samples: list[dict[str, Any]],
    thresholds: SyncThresholds,
    *,
    start_after_s: float = 0.0,
) -> float | None:
    usable = [s for s in samples if float(s["t_s"]) >= start_after_s]
    for idx, sample in enumerate(usable):
        start_t = float(sample["t_s"])
        end_t = start_t + thresholds.required_window_s
        ratio, window = _window_pass_ratio(
            usable,
            start_t=start_t,
            end_t=end_t,
            predicate=lambda s: _sample_is_sync(s, thresholds),
        )
        if not window or float(window[-1]["t_s"]) < end_t:
            continue
        if ratio >= thresholds.window_pass_ratio:
            return round(start_t, 6)
    return None


def _series_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.max(x) - np.min(x) <= 1e-9:
        return None
    slope = float(np.polyfit(x - x[0], y, 1)[0])
    return round(slope, 9)


def _common_frequency_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (float(s["t_s"]), float(s["common_frequency_hz"]))
        for s in samples
        if math.isfinite(float(s.get("common_frequency_hz", float("nan"))))
    ]
    times = [item[0] for item in pairs]
    vals = [item[1] for item in pairs]
    return {
        "mean_common_frequency_hz": round(float(statistics.mean(vals)), 6) if vals else None,
        "common_frequency_std_hz": (
            round(float(statistics.pstdev(vals)), 6) if len(vals) >= 2 else 0.0 if vals else None
        ),
        "common_frequency_slope_hz_per_s": _series_slope(times, vals) if vals else None,
    }


def _compute_lock_hold_metrics(
    samples: list[dict[str, Any]],
    lock_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    enabled = any(int(row.get("lock_hold_enabled") or 0) == 1 for row in lock_rows)
    hold_rows = [row for row in lock_rows if row.get("lock_hold_state") == "hold"]
    lock_times = [
        float(row["lock_time_s"])
        for row in lock_rows
        if row.get("lock_time_s") not in ("", None)
    ]
    hold_times = {round(float(row["t_s"]), 6) for row in hold_rows}
    hold_samples = [
        sample for sample in samples
        if round(float(sample["t_s"]), 6) in hold_times
    ]
    hold_common = _common_frequency_stats(hold_samples)
    final_state = (
        str(lock_rows[-1].get("lock_hold_state"))
        if lock_rows else ("disabled" if not enabled else "acquisition")
    )
    lock_row = next(
        (row for row in lock_rows if row.get("lock_time_s") not in ("", None)),
        None,
    )

    def mean_sample(key: str) -> float | None:
        vals = [
            float(sample[key])
            for sample in hold_samples
            if math.isfinite(float(sample.get(key, float("nan"))))
        ]
        return round(float(statistics.mean(vals)), 6) if vals else None

    return {
        "lock_hold_enabled": bool(enabled),
        "lock_acquired": bool(lock_times),
        "lock_time_s": round(min(lock_times), 6) if lock_times else None,
        "f_lock_hz": (
            float(lock_row["f_lock_hz"])
            if lock_row and lock_row.get("f_lock_hz") not in ("", None) else None
        ),
        "f_lock_source_window_s": (
            float(lock_row["f_lock_source_window_s"])
            if lock_row and lock_row.get("f_lock_source_window_s") not in ("", None) else None
        ),
        "common_frequency_at_lock_mean_hz": (
            float(lock_row["common_frequency_at_lock_mean_hz"])
            if lock_row and lock_row.get("common_frequency_at_lock_mean_hz") not in ("", None)
            else None
        ),
        "common_frequency_at_lock_median_hz": (
            float(lock_row["common_frequency_at_lock_median_hz"])
            if lock_row and lock_row.get("common_frequency_at_lock_median_hz") not in ("", None)
            else None
        ),
        "common_frequency_at_lock_std_hz": (
            float(lock_row["common_frequency_at_lock_std_hz"])
            if lock_row and lock_row.get("common_frequency_at_lock_std_hz") not in ("", None)
            else None
        ),
        "hold_duration_s": (
            round(max(hold_times) - min(hold_times), 6) if len(hold_times) >= 2 else 0.0
        ) if hold_times else None,
        "unlock_count": int(lock_rows[-1].get("unlock_count") or 0) if lock_rows else 0,
        "relock_count": int(lock_rows[-1].get("relock_count") or 0) if lock_rows else 0,
        "final_hold_state": final_state,
        "mean_hold_R": mean_sample("order_parameter_R"),
        "mean_hold_phase_error_cycles": mean_sample("mean_pairwise_phase_error_cycles"),
        "mean_hold_frequency_disagreement_hz": mean_sample("frequency_disagreement_hz"),
        "hold_common_frequency_std_hz": hold_common["common_frequency_std_hz"],
        "hold_common_frequency_slope_hz_per_s": hold_common[
            "common_frequency_slope_hz_per_s"
        ],
    }


def _final_sync_success(final_samples: list[dict[str, Any]], thresholds: SyncThresholds) -> bool:
    if not final_samples:
        return False
    phase_vals = [float(s["mean_pairwise_phase_error_cycles"]) for s in final_samples]
    freq_vals = [float(s["frequency_disagreement_hz"]) for s in final_samples]
    r_vals = [float(s["order_parameter_R"]) for s in final_samples]
    phase_ok = statistics.median(phase_vals) < thresholds.mean_pairwise_phase_error_cycles
    r_ok = statistics.median(r_vals) > thresholds.order_parameter_R
    freq_ok = statistics.median(freq_vals) < thresholds.frequency_disagreement_hz
    return bool((phase_ok or r_ok) and freq_ok)


def _compute_sync_metrics(
    agent_rows: list[dict[str, Any]],
    *,
    thresholds: SyncThresholds,
    disruption_end_s: float | None = None,
    lock_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    samples = _samples_from_agent_rows(agent_rows)
    final_cutoff = (max((float(s["t_s"]) for s in samples), default=0.0)
                    - thresholds.final_window_s)
    final_samples = [s for s in samples if float(s["t_s"]) >= final_cutoff]
    time_to_sync = _time_to_sync(samples, thresholds)
    recovery_time = None
    recovery_time_absolute = None
    if disruption_end_s is not None:
        recovery_time_absolute = _time_to_sync(
            samples,
            thresholds,
            start_after_s=float(disruption_end_s),
        )
        if recovery_time_absolute is not None:
            recovery_time = round(recovery_time_absolute - float(disruption_end_s), 6)

    def mean_of(key: str) -> float | None:
        vals = [float(s[key]) for s in final_samples if math.isfinite(float(s[key]))]
        return round(float(statistics.mean(vals)), 6) if vals else None

    def max_of(key: str) -> float | None:
        vals = [float(s[key]) for s in final_samples if math.isfinite(float(s[key]))]
        return round(float(max(vals)), 6) if vals else None

    final_freqs_by_agent: dict[str, list[float]] = {agent_id: [] for agent_id in AGENTS}
    for sample in final_samples:
        for agent_id, value in sample["frequencies_hz"].items():
            final_freqs_by_agent[agent_id].append(float(value))
    final_frequency_mean_hz = {
        agent_id: round(float(statistics.mean(vals)), 6) if vals else None
        for agent_id, vals in final_freqs_by_agent.items()
    }
    continuous_sync_success = time_to_sync is not None
    final_sync_success = _final_sync_success(final_samples, thresholds)
    final_common = _common_frequency_stats(final_samples)
    slope = final_common.get("common_frequency_slope_hz_per_s")
    std = final_common.get("common_frequency_std_hz")
    frequency_stability_success = (
        std is not None
        and float(std) <= thresholds.frequency_stability_std_hz
        and slope is not None
        and abs(float(slope)) <= thresholds.frequency_stability_slope_hz_per_s
    )
    lock_metrics = _compute_lock_hold_metrics(samples, lock_rows or [])
    return {
        "sync_success": bool(continuous_sync_success),
        "final_sync_success": bool(final_sync_success),
        "continuous_sync_success": bool(continuous_sync_success),
        "time_to_sync_s": time_to_sync,
        "time_to_sync_criterion": {
            "phase_or_order_parameter": True,
            "required_continuous_window_s": thresholds.required_window_s,
            "window_pass_ratio_gte": thresholds.window_pass_ratio,
            "mean_pairwise_phase_error_cycles_lt": thresholds.mean_pairwise_phase_error_cycles,
            "mean_pairwise_phase_error_rad_lt": thresholds.mean_pairwise_phase_error_rad,
            "frequency_disagreement_hz_lt": thresholds.frequency_disagreement_hz,
            "order_parameter_R_gt": thresholds.order_parameter_R,
        },
        "final_window_s": thresholds.final_window_s,
        "final_mean_pairwise_phase_error_rad": mean_of("mean_pairwise_phase_error_rad"),
        "final_max_pairwise_phase_error_rad": max_of("max_pairwise_phase_error_rad"),
        "final_mean_pairwise_phase_error_cycles": mean_of("mean_pairwise_phase_error_cycles"),
        "final_max_pairwise_phase_error_cycles": max_of("max_pairwise_phase_error_cycles"),
        "final_frequency_disagreement_hz": mean_of("frequency_disagreement_hz"),
        "final_max_frequency_disagreement_hz": max_of("frequency_disagreement_hz"),
        "final_mean_order_parameter_R": mean_of("order_parameter_R"),
        "final_min_order_parameter_R": (
            round(float(min(s["order_parameter_R"] for s in final_samples)), 6)
            if final_samples else None
        ),
        "final_frequency_mean_hz": final_frequency_mean_hz,
        "final_mean_common_frequency_hz": final_common["mean_common_frequency_hz"],
        "final_common_frequency_std_hz": final_common["common_frequency_std_hz"],
        "final_common_frequency_slope_hz_per_s": final_common[
            "common_frequency_slope_hz_per_s"
        ],
        "frequency_stability_success": bool(frequency_stability_success),
        "frequency_stability_criterion": {
            "final_common_frequency_std_hz_lte": thresholds.frequency_stability_std_hz,
            "abs_final_common_frequency_slope_hz_per_s_lte": (
                thresholds.frequency_stability_slope_hz_per_s
            ),
        },
        **lock_metrics,
        "recovery_time_s": recovery_time,
        "recovery_time_absolute_s": recovery_time_absolute,
        "sample_count": len(samples),
    }


def _api_health(api_events: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = []
    ok = 0
    for row in api_events:
        if int(row.get("ok") or 0) == 1:
            ok += 1
        try:
            latencies.append(float(row.get("elapsed_ms", "")))
        except (TypeError, ValueError):
            pass
    return {
        "api_request_count": len(api_events),
        "api_success_count": ok,
        "api_failure_count": len(api_events) - ok,
        "api_latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "api_latency_max_ms": round(max(latencies), 3) if latencies else None,
    }


def _detection_metrics(
    roi_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    agent_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    detected = {
        agent_id: sum(
            1 for e in events
            if e.get("event_type") == "pi_detected_virtual_flash"
            and e.get("agent_id") == agent_id
        )
        for agent_id in ("V0", "V1")
    }
    actual = {"V0": 0, "V1": 0, "P0": 0}
    for row in agent_rows:
        aid = str(row.get("agent_id"))
        if aid in actual:
            try:
                actual[aid] = max(actual[aid], int(float(row.get("fire_count") or 0)))
            except (TypeError, ValueError):
                pass
    p0_events = sum(1 for e in events if e.get("event_type") == "pi_flash")
    actual["P0"] = max(actual["P0"], p0_events)
    recall = {
        agent_id: (
            round(detected[agent_id] / actual[agent_id], 6)
            if actual.get(agent_id, 0) > 0 else None
        )
        for agent_id in ("V0", "V1")
    }
    times = []
    frame_indices = set()
    for row in roi_rows:
        try:
            times.append(float(row.get("t_s", "")))
        except (TypeError, ValueError):
            pass
        try:
            frame_indices.add(int(float(row.get("frame_index", ""))))
        except (TypeError, ValueError):
            pass
    span = max(times) - min(times) if len(times) >= 2 else 0.0
    return {
        "p0_detected_events": detected,
        "actual_flash_counts": actual,
        "detection_recall": recall,
        "missed_event_estimate": {
            agent_id: (
                max(0, actual[agent_id] - detected[agent_id])
                if actual.get(agent_id, 0) > 0 else None
            )
            for agent_id in ("V0", "V1")
        },
        "extra_event_estimate": {
            agent_id: max(0, detected[agent_id] - actual.get(agent_id, 0))
            for agent_id in ("V0", "V1")
        },
        "camera_frame_count": len(frame_indices),
        "camera_fps_estimate": round(len(frame_indices) / span, 3) if span > 0 else None,
    }


def _route_detected_virtual_event(
    evt: dict[str, Any],
    *,
    t_s: float,
    injector: EventFaultInjector,
    topology,
) -> tuple[dict[str, Any], str | None]:
    """Apply fault-injection bookkeeping to one detected virtual flash event."""
    source = str(evt["agent_id"])
    kept = True
    delay_s = 0.0
    dropped = False
    drop_reason = None

    kept = injector.keep_virtual_detection(source)
    if not kept:
        dropped = True
        drop_reason = injector.fault.get("kind", "event_dropped")

    row = {
        "t_s": round(t_s, 6),
        "event_type": "pi_detected_virtual_flash",
        "agent_id": source,
        "roi_id": evt["roi_id"],
        "raw_brightness": evt["raw_brightness"],
        "normalized_signal": evt["normalized_signal"],
        "kept_for_p0": int(kept),
        "dropped": int(dropped),
        "drop_reason": drop_reason,
        "injected_delay_s": delay_s,
    }
    routed_source = source if kept and topology.can_observe("P0", source) else None
    return row, routed_source


def _make_trial_args(
    args: argparse.Namespace,
    condition: dict[str, Any],
    initial_states: dict[str, dict[str, float]],
    run_dir: Path,
    *,
    roi_v0: list[int] | None,
    roi_v1: list[int] | None,
    roi_config: str | None,
) -> argparse.Namespace:
    display = condition.get("display", {})
    return argparse.Namespace(
        mode="2v1p_eapf_smoke",
        leader_api=args.leader_api,
        duration=args.duration,
        topology=args.topology,
        v0_freq=initial_states["V0"]["initial_frequency_hz"],
        v1_freq=initial_states["V1"]["initial_frequency_hz"],
        pi_freq=initial_states["P0"]["initial_frequency_hz"],
        v0_phase_rad=initial_states["V0"]["initial_phase_rad"],
        v1_phase_rad=initial_states["V1"]["initial_phase_rad"],
        pi_phase_rad=initial_states["P0"]["initial_phase_rad"],
        detection_preset="none",
        v0_x=args.v0_x,
        v0_y=args.v0_y,
        v1_x=args.v1_x,
        v1_y=args.v1_y,
        dot_size=args.dot_size,
        v0_size=args.v0_size,
        v1_size=args.v1_size,
        background_brightness=args.background_brightness,
        flash_brightness=args.flash_brightness,
        off_brightness=args.off_brightness,
        v0_flash_brightness=display.get("V0", {}).get("flash_brightness", args.flash_brightness),
        v0_off_brightness=display.get("V0", {}).get("off_brightness", args.off_brightness),
        v0_background_brightness=display.get("V0", {}).get(
            "background_brightness", args.background_brightness
        ),
        v1_flash_brightness=display.get("V1", {}).get("flash_brightness", args.flash_brightness),
        v1_off_brightness=display.get("V1", {}).get("off_brightness", args.off_brightness),
        v1_background_brightness=display.get("V1", {}).get(
            "background_brightness", args.background_brightness
        ),
        log_dir=args.log_dir,
        run_dir=str(run_dir),
        api_timeout=args.api_timeout,
        poll_interval=args.poll_interval,
        dry_run=False,
        roi_v0=roi_v0,
        roi_v1=roi_v1,
        roi_config=roi_config,
        auto_roi=False,
        width=args.width,
        height=args.height,
        camera_fps=args.camera_fps,
        camera_format=args.camera_format,
        min_interval=args.min_interval,
        window_s=args.window_s,
        norm_on_threshold=args.norm_on_threshold,
        norm_off_threshold=args.norm_off_threshold,
        min_amplitude=args.min_amplitude,
        episode_latch=args.episode_latch,
        save_mid_roi_debug_frame=True,
        auto_roi_duration=args.auto_roi_duration,
        auto_roi_combined_duration=args.auto_roi_combined_duration,
        auto_roi_verify_duration=args.auto_roi_verify_duration,
        auto_roi_warmup_s=args.auto_roi_warmup_s,
        auto_roi_capture_fps=args.auto_roi_capture_fps,
        auto_roi_v0_frequency=args.auto_roi_v0_frequency,
        auto_roi_v1_frequency=args.auto_roi_v1_frequency,
        auto_roi_v1_phase_rad=args.auto_roi_v1_phase_rad,
        auto_roi_sequential_diagnostics=args.auto_roi_sequential_diagnostics,
        auto_roi_frequency_ambiguity_hz=args.auto_roi_frequency_ambiguity_hz,
        auto_roi_method=args.auto_roi_method,
        auto_roi_padding=args.auto_roi_padding,
        auto_roi_min_area=args.auto_roi_min_area,
        auto_roi_downsample=args.auto_roi_downsample,
        auto_roi_change_threshold=args.auto_roi_change_threshold,
        auto_roi_boundary_margin_px=args.auto_roi_boundary_margin_px,
        auto_roi_overlap_warning_ratio=args.auto_roi_overlap_warning_ratio,
        auto_roi_max_area_fraction=args.auto_roi_max_area_fraction,
        led_pin=args.led_pin,
        led_pulse_duration=args.led_pulse_duration,
        v0_enabled=True,
        v1_enabled=True,
    )


def _write_trial_config(
    trial_dir: Path,
    args: argparse.Namespace,
    condition: dict[str, Any],
    initial_states: dict[str, dict[str, float]],
    thresholds: SyncThresholds,
    lock_config: LockHoldConfig,
    roi_config: str | None,
    freq_set: str,
) -> None:
    _write_json(trial_dir / "trial_config.json", {
        "step": "Step5c2",
        "system": "mixed_reality_2_virtual_1_pi",
        "model": "EAPF Consensus",
        "model_parameters": LOCKED_EAPF_PARAMETERS,
        "leader_api": args.leader_api,
        "duration_s": args.duration,
        "topology": args.topology,
        "condition": condition,
        "frequency_set": freq_set,
        "frequency_set_definition": FREQUENCY_SETS.get(freq_set),
        "initial_states": initial_states,
        "sync_thresholds": thresholds.__dict__,
        "stabilizer": lock_config.__dict__,
        "roi_config": roi_config,
        "created_at": datetime.now().isoformat(),
    })


def _append_agent_poll_rows(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
    *,
    t_s: float,
    p0_state: dict[str, Any],
    stabilizer: LockHoldStabilizer | None = None,
) -> None:
    seen = set()
    stab_state = stabilizer.state if stabilizer is not None and stabilizer.enabled else "disabled"
    f_lock = (
        round(stabilizer.f_lock_hz, 6)
        if stabilizer is not None and stabilizer.f_lock_hz is not None else ""
    )
    for agent in payload.get("agents", []):
        aid = str(agent.get("agent_id", agent.get("id")))
        if aid == "P0":
            continue
        seen.add(aid)
        rows.append({
            "t_s": round(t_s, 6),
            "agent_id": aid,
            "role": agent.get("role"),
            "phase_rad": agent.get("phase_rad"),
            "frequency_hz": agent.get("frequency_hz"),
            "flash_on": int(bool(agent.get("flash_on"))),
            "fire_count": agent.get("fire_count"),
            "received_neighbour_events": agent.get("received_neighbour_events"),
            "pi_flash_events_consumed": agent.get("pi_flash_events_consumed"),
            "topology": agent.get("topology"),
            "lock_hold_state": stab_state,
            "f_lock_hz": f_lock,
        })
    if "P0" not in seen:
        rows.append({
            "t_s": round(t_s, 6),
            "agent_id": "P0",
            "role": "pi",
            "phase_rad": p0_state.get("phase_rad"),
            "frequency_hz": p0_state.get("frequency_hz"),
            "flash_on": int(bool(p0_state.get("flash_on", False))),
            "fire_count": p0_state.get("fire_count"),
            "received_neighbour_events": p0_state.get("received_neighbour_events"),
            "pi_flash_events_consumed": "",
            "topology": p0_state.get("topology"),
            "lock_hold_state": stab_state,
            "f_lock_hz": f_lock,
        })


def _set_oscillator_initial_state(
    osc: EventBasedConsensusPLLOscillator,
    *,
    phase_rad: float,
    frequency_hz: float,
) -> None:
    osc._phase_rad = float(phase_rad) % (2.0 * math.pi)
    osc._frequency_hz = float(frequency_hz)
    osc._omega_rad_s = 2.0 * math.pi * osc._frequency_hz


def _apply_stabilizer_to_pi_oscillator(
    osc: EventBasedConsensusPLLOscillator,
    stabilizer: LockHoldStabilizer,
) -> None:
    base = _locked_eapf_config(osc.config.natural_frequency_hz)
    if stabilizer.enabled and stabilizer.state == "hold":
        osc.config.phase_gain = base.phase_gain * stabilizer.config.hold_phase_gain_scale
        osc.config.frequency_gain = base.frequency_gain * stabilizer.config.hold_frequency_gain_scale
        if stabilizer.f_lock_hz is not None:
            anchored = float(osc.frequency_hz) + stabilizer.config.hold_anchor_gain * (
                float(stabilizer.f_lock_hz) - float(osc.frequency_hz)
            )
            osc._frequency_hz = max(
                osc.config.frequency_min_hz,
                min(osc.config.frequency_max_hz, anchored),
            )
            osc._omega_rad_s = 2.0 * math.pi * osc._frequency_hz
    else:
        osc.config.phase_gain = base.phase_gain
        osc.config.frequency_gain = base.frequency_gain
        osc.config.phase_error_filter_alpha = base.phase_error_filter_alpha
        osc.config.frequency_error_filter_alpha = base.frequency_error_filter_alpha
        osc.config.max_phase_step_rad = base.max_phase_step_rad
        osc.config.max_frequency_step_hz = base.max_frequency_step_hz


def _post_stabilizer_state(
    args: argparse.Namespace,
    stabilizer: LockHoldStabilizer,
    api_events: list[dict[str, Any]],
) -> None:
    _api(
        args.leader_api,
        "/api/stabilizer/config",
        "POST",
        stabilizer.api_payload(),
        timeout=args.api_timeout,
        events=api_events,
    )


def run_trial(
    args: argparse.Namespace,
    condition: dict[str, Any],
    initial_states: dict[str, dict[str, float]],
    trial_dir: Path,
    *,
    roi_v0: list[int],
    roi_v1: list[int],
    roi_config: str | None,
    thresholds: SyncThresholds,
    lock_config: LockHoldConfig,
    trial_seed: int,
    freq_set: str,
) -> dict[str, Any]:
    trial_args = _make_trial_args(
        args,
        condition,
        initial_states,
        trial_dir,
        roi_v0=roi_v0,
        roi_v1=roi_v1,
        roi_config=roi_config,
    )
    _write_trial_config(
        trial_dir,
        args,
        condition,
        initial_states,
        thresholds,
        lock_config,
        roi_config,
        freq_set,
    )

    api_events: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    rng = random.Random(trial_seed)
    injector = EventFaultInjector(condition, rng)
    stabilizer = LockHoldStabilizer(lock_config)
    pause_info: dict[str, float | None] = {"pause_start_s": None, "pause_end_s": None}

    _start_server_trial(trial_args, api_events, feedback=True)
    _post_stabilizer_state(args, stabilizer, api_events)
    hil._ensure_hw(False)
    detector = _make_multi_roi_detector(trial_args)
    led = hil._PiGPIOLED(pin=args.led_pin, flash_duration_s=args.led_pulse_duration)
    pi_osc = EventBasedConsensusPLLOscillator(
        _locked_eapf_config(initial_states["P0"]["initial_frequency_hz"])
    )
    _set_oscillator_initial_state(
        pi_osc,
        phase_rad=initial_states["P0"]["initial_phase_rad"],
        frequency_hz=initial_states["P0"]["initial_frequency_hz"],
    )
    topology = build_mixed_reality_topology(args.topology)

    last_poll = -math.inf
    latest_p0_state: dict[str, Any] = {
        "phase_rad": initial_states["P0"]["initial_phase_rad"],
        "frequency_hz": initial_states["P0"]["initial_frequency_hz"],
        "fire_count": 0,
        "received_neighbour_events": 0,
        "topology": args.topology,
    }

    def post_p0_event(source_t_s: float, post_t_s: float, delay_s: float) -> None:
        _api(
            args.leader_api,
            "/api/pi_flash",
            "POST",
            {"timestamp": source_t_s, "agent_id": "P0", "injected_delay_s": delay_s},
            timeout=args.api_timeout,
            events=api_events,
        )
        events.append({
            "t_s": round(post_t_s, 6),
            "event_type": "pi_flash_routed_to_virtuals",
            "agent_id": "P0",
            "source_t_s": round(source_t_s, 6),
            "injected_delay_s": round(delay_s, 6),
        })

    try:
        detector.start()
        t0 = time.monotonic()
        last = t0
        debug_saved = False
        while time.monotonic() - t0 < args.duration:
            now_abs = time.monotonic()
            t_s = now_abs - t0
            dt = max(0.001, now_abs - last)
            last = now_abs

            pause_info = injector.maybe_apply_pause(
                t_s=t_s,
                duration_s=args.duration,
                args=args,
                api_events=api_events,
            )
            injector.flush_due_p0_events(t_s, post_p0_event)

            raw_frame = detector.capture_raw_frame()
            if not debug_saved:
                _save_roi_debug_frame(raw_frame, trial_args, trial_dir, "roi_debug_frame_start.jpg")
                debug_saved = True
            result = detector.process_frame(raw_frame)
            neighbour_sources: list[str] = []
            for row in result["roi_results"]:
                out = dict(row)
                out["t_s"] = round(t_s, 6)
                roi_rows.append(out)
            for evt in result["events"]:
                event_row, routed_source = _route_detected_virtual_event(
                    evt,
                    t_s=t_s,
                    injector=injector,
                    topology=topology,
                )
                events.append(event_row)
                if routed_source is not None:
                    neighbour_sources.append(routed_source)

            neighbour_ids = topology.numeric_neighbour_ids("P0", neighbour_sources)
            _apply_stabilizer_to_pi_oscillator(pi_osc, stabilizer)
            state = pi_osc.step(dt_s=dt, t_s=t_s, neighbour_flash_ids=neighbour_ids)
            _apply_stabilizer_to_pi_oscillator(pi_osc, stabilizer)
            latest_p0_state = {
                "phase_rad": state["phase_rad"],
                "frequency_hz": round(float(pi_osc.frequency_hz), 6),
                "fire_count": state["fire_count"],
                "received_neighbour_events": len(neighbour_ids),
                "topology": args.topology,
                "flash_on": bool(state.get("follower_flash_event")),
            }
            if state.get("follower_flash_event"):
                led.flash(args.led_pulse_duration)
                events.append({
                    "t_s": round(t_s, 6),
                    "event_type": "pi_flash",
                    "agent_id": "P0",
                    "phase_rad": state["phase_rad"],
                    "frequency_hz": state["frequency_hz"],
                })
                injector.queue_or_post_p0_flash(t_s=t_s, post_event=post_p0_event)

            if t_s - last_poll >= args.poll_interval:
                payload = _api(
                    args.leader_api,
                    "/api/agents",
                    "GET",
                    timeout=args.api_timeout,
                    events=api_events,
                )
                _append_agent_poll_rows(
                    agent_rows,
                    payload,
                    t_s=t_s,
                    p0_state=latest_p0_state,
                    stabilizer=stabilizer,
                )
                recent_sample = _samples_from_agent_rows(agent_rows[-3:])
                if recent_sample:
                    stabilizer_event = stabilizer.update(recent_sample[-1])
                    if stabilizer_event is not None:
                        events.append(stabilizer_event)
                        _post_stabilizer_state(args, stabilizer, api_events)
                last_poll = t_s
    finally:
        try:
            led.close()
        finally:
            try:
                detector.stop()
            finally:
                try:
                    _api(args.leader_api, "/api/stabilizer/config", "POST", {
                        "enabled": False,
                        "state": "acquisition",
                        "f_lock_hz": None,
                    }, timeout=args.api_timeout, events=api_events)
                except Exception:
                    pass
                _api(args.leader_api, "/api/pause", "POST", {},
                     timeout=args.api_timeout, events=api_events)

    _write_csv(trial_dir / "agent_state_timeseries.csv", agent_rows)
    _write_csv(trial_dir / "events.csv", events)
    _write_csv(trial_dir / "api_events.csv", api_events)
    _write_csv(trial_dir / "pi_detection_roi.csv", roi_rows)
    _write_csv(trial_dir / "lock_hold_state_timeseries.csv", stabilizer.rows)

    detection = _detection_metrics(roi_rows, events, agent_rows)
    sync = _compute_sync_metrics(
        agent_rows,
        thresholds=thresholds,
        disruption_end_s=pause_info.get("pause_end_s"),
        lock_rows=stabilizer.rows,
    )
    api_health = _api_health(api_events)
    metrics = {
        "step": "Step5c2",
        "condition": condition["name"],
        "topology": args.topology,
        "model": "EAPF Consensus",
        "stabilizer": lock_config.stabilizer,
        "frequency_set": freq_set,
        "initial_states": initial_states,
        "sync_success": sync["sync_success"],
        "final_sync_success": sync["final_sync_success"],
        "continuous_sync_success": sync["continuous_sync_success"],
        "time_to_sync_s": sync["time_to_sync_s"],
        "final_mean_pairwise_phase_error_cycles": sync["final_mean_pairwise_phase_error_cycles"],
        "final_max_pairwise_phase_error_cycles": sync["final_max_pairwise_phase_error_cycles"],
        "final_frequency_disagreement_hz": sync["final_frequency_disagreement_hz"],
        "final_mean_order_parameter_R": sync["final_mean_order_parameter_R"],
        "final_mean_common_frequency_hz": sync["final_mean_common_frequency_hz"],
        "final_common_frequency_std_hz": sync["final_common_frequency_std_hz"],
        "final_common_frequency_slope_hz_per_s": sync[
            "final_common_frequency_slope_hz_per_s"
        ],
        "frequency_stability_success": sync["frequency_stability_success"],
        "lock_hold_enabled": sync["lock_hold_enabled"],
        "lock_acquired": sync["lock_acquired"],
        "lock_time_s": sync["lock_time_s"],
        "hold_duration_s": sync["hold_duration_s"],
        "unlock_count": sync["unlock_count"],
        "relock_count": sync["relock_count"],
        "final_hold_state": sync["final_hold_state"],
        "recovery_time_s": sync["recovery_time_s"],
        "detection_recall": detection["detection_recall"],
        "api_health": api_health,
        "fault_injection": injector.stats,
        "stabilizer_settings": lock_config.__dict__,
        "pause_window": pause_info,
    }
    _write_json(trial_dir / "detection_metrics.json", detection)
    _write_json(trial_dir / "sync_metrics.json", sync)
    _write_json(trial_dir / "metrics_summary.json", metrics)
    _save_trial_plots(trial_dir, agent_rows, events, stabilizer.rows, lock_config=lock_config)
    return metrics


def _sample_lock_pass(sample: dict[str, Any], config: LockHoldConfig) -> bool:
    return (
        float(sample["order_parameter_R"]) >= config.lock_r_threshold
        and float(sample["mean_pairwise_phase_error_cycles"])
        <= config.lock_phase_error_threshold_cycles
        and float(sample["frequency_disagreement_hz"])
        <= config.lock_frequency_disagreement_threshold_hz
    )


def _boolean_intervals(samples: list[dict[str, Any]], flags: list[bool]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    previous_t: float | None = None
    for sample, flag in zip(samples, flags):
        t_s = float(sample["t_s"])
        if flag and start is None:
            start = t_s
        if not flag and start is not None:
            intervals.append((start, previous_t if previous_t is not None else start))
            start = None
        previous_t = t_s
    if start is not None:
        intervals.append((start, previous_t if previous_t is not None else start))
    return intervals


def _rolling_lock_pass_flags(
    samples: list[dict[str, Any]],
    config: LockHoldConfig,
) -> list[bool]:
    flags = []
    eps = 1e-9
    acquisition_start = float(samples[0]["t_s"]) if samples else 0.0
    for sample in samples:
        t_s = float(sample["t_s"])
        window = [
            s for s in samples
            if 0.0 <= t_s - float(s["t_s"]) <= config.lock_window_s + eps
        ]
        span_ok = bool(window) and t_s - acquisition_start + eps >= config.lock_window_s
        ratio = (
            sum(1 for s in window if _sample_lock_pass(s, config)) / len(window)
            if window else 0.0
        )
        flags.append(bool(span_ok and ratio >= config.lock_window_pass_ratio))
    return flags


def _plot_sync_criterion(
    path: Path,
    samples: list[dict[str, Any]],
    *,
    lock_config: LockHoldConfig,
    lock_rows: list[dict[str, Any]] | None = None,
) -> None:
    if not samples:
        return
    xs = [float(s["t_s"]) for s in samples]
    lock_pass_flags = [_sample_lock_pass(s, lock_config) for s in samples]
    rolling_flags = _rolling_lock_pass_flags(samples, lock_config)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for start, end in _boolean_intervals(samples, lock_pass_flags):
        ax.axvspan(start, end, color="tab:green", alpha=0.10, label="_nolegend_")
    for start, end in _boolean_intervals(samples, rolling_flags):
        ax.axvspan(start, end, color="tab:blue", alpha=0.12, label="_nolegend_")
    ax.plot(xs, [s["order_parameter_R"] for s in samples], label="R", linewidth=1.2)
    ax.plot(
        xs,
        [s["mean_pairwise_phase_error_cycles"] for s in samples],
        label="phase error (cycles)",
        linewidth=1.2,
    )
    ax.plot(
        xs,
        [s["frequency_disagreement_hz"] for s in samples],
        label="frequency disagreement (Hz)",
        linewidth=1.2,
    )
    ax.axhline(lock_config.lock_r_threshold, color="tab:blue", linestyle="--",
               linewidth=0.9, label=f"R lock >= {lock_config.lock_r_threshold:g}")
    ax.axhline(
        lock_config.lock_phase_error_threshold_cycles,
        color="tab:orange",
        linestyle="--",
        linewidth=0.9,
        label=f"phase lock <= {lock_config.lock_phase_error_threshold_cycles:g}",
    )
    ax.axhline(
        lock_config.lock_frequency_disagreement_threshold_hz,
        color="tab:green",
        linestyle="--",
        linewidth=0.9,
        label=f"freq lock <= {lock_config.lock_frequency_disagreement_threshold_hz:g}",
    )
    if lock_rows:
        lock_times = [
            float(row["lock_time_s"])
            for row in lock_rows
            if row.get("lock_time_s") not in ("", None)
        ]
        if lock_times:
            ax.axvline(min(lock_times), color="black", linestyle=":", linewidth=1.1,
                       label=f"lock @ {min(lock_times):.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Criterion values")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_trial_plots(
    trial_dir: Path,
    agent_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    lock_rows: list[dict[str, Any]] | None = None,
    lock_config: LockHoldConfig | None = None,
) -> None:
    if not agent_rows:
        return
    for key, ylabel, filename in [
        ("phase_rad", "Phase (rad)", "phase_vs_time.png"),
        ("frequency_hz", "Frequency (Hz)", "frequency_vs_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for agent_id in AGENTS:
            xs, ys = [], []
            for row in agent_rows:
                if row.get("agent_id") != agent_id:
                    continue
                try:
                    xs.append(float(row["t_s"]))
                    ys.append(float(row[key]))
                except (TypeError, ValueError, KeyError):
                    pass
            if xs:
                ax.plot(xs, ys, label=agent_id, linewidth=1.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.savefig(trial_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    samples = _samples_from_agent_rows(agent_rows)
    if samples:
        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.plot([s["t_s"] for s in samples], [s["order_parameter_R"] for s in samples])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Order parameter R")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
        fig.savefig(trial_dir / "order_parameter_vs_time.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        _plot_sync_criterion(
            trial_dir / "sync_criterion_vs_time.png",
            samples,
            lock_config=lock_config or LockHoldConfig(stabilizer="lock_hold"),
            lock_rows=lock_rows,
        )

        fig, ax = plt.subplots(figsize=(8, 3.0))
        ax.plot([s["t_s"] for s in samples], [s["common_frequency_hz"] for s in samples])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Common frequency (Hz)")
        ax.grid(True, alpha=0.25)
        fig.savefig(trial_dir / "common_frequency_vs_time.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if lock_rows and any(int(row.get("lock_hold_enabled") or 0) == 1 for row in lock_rows):
        state_to_y = {"disabled": 0, "acquisition": 0, "hold": 1}
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.step(
            [float(row["t_s"]) for row in lock_rows],
            [state_to_y.get(str(row.get("lock_hold_state")), 0) for row in lock_rows],
            where="post",
            label="state",
        )
        f_vals = [
            float(row["f_lock_hz"]) if row.get("f_lock_hz") not in ("", None) else np.nan
            for row in lock_rows
        ]
        ax2 = ax.twinx()
        ax2.plot([float(row["t_s"]) for row in lock_rows], f_vals, color="tab:orange",
                 linewidth=1.0, label="f_lock")
        ax.set_yticks([0, 1], ["acquire", "hold"])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Lock state")
        ax2.set_ylabel("f_lock (Hz)")
        ax.grid(True, alpha=0.25)
        fig.savefig(trial_dir / "lock_hold_state_vs_time.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if events:
        y_map = {"V0": 0, "V1": 1, "P0": 2}
        fig, ax = plt.subplots(figsize=(8, 2.8))
        for event in events:
            aid = str(event.get("agent_id"))
            if aid not in y_map:
                continue
            try:
                ax.scatter(float(event["t_s"]), y_map[aid], s=12)
            except (TypeError, ValueError, KeyError):
                pass
        ax.set_yticks([0, 1, 2], ["V0", "V1", "P0"])
        ax.set_xlabel("Time (s)")
        ax.set_title("Event raster")
        ax.grid(True, axis="x", alpha=0.25)
        fig.savefig(trial_dir / "event_raster.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_lock_config_from_batch(batch_dir: Path) -> LockHoldConfig:
    config_path = batch_dir / "batch_config.json"
    if not config_path.exists():
        return LockHoldConfig(stabilizer="lock_hold")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    settings = data.get("stabilizer") or data.get("stabilizer_settings") or {}
    allowed = set(LockHoldConfig.__dataclass_fields__)
    return LockHoldConfig(**{k: v for k, v in settings.items() if k in allowed})


def _longest_interval_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    return round(max(max(0.0, end - start) for start, end in intervals), 6)


def _closest_to_lock(samples: list[dict[str, Any]], config: LockHoldConfig) -> tuple[float | None, str]:
    if not samples:
        return None, "no_samples"

    def score(sample: dict[str, Any]) -> float:
        deficits = [
            max(0.0, config.lock_r_threshold - float(sample["order_parameter_R"])),
            max(
                0.0,
                float(sample["mean_pairwise_phase_error_cycles"])
                - config.lock_phase_error_threshold_cycles,
            ),
            max(
                0.0,
                float(sample["frequency_disagreement_hz"])
                - config.lock_frequency_disagreement_threshold_hz,
            ),
        ]
        return sum(deficits)

    best = min(samples, key=score)
    blockers = []
    if float(best["order_parameter_R"]) < config.lock_r_threshold:
        blockers.append("R")
    if float(best["mean_pairwise_phase_error_cycles"]) > config.lock_phase_error_threshold_cycles:
        blockers.append("phase_error")
    if float(best["frequency_disagreement_hz"]) > config.lock_frequency_disagreement_threshold_hz:
        blockers.append("frequency_disagreement")
    return round(float(best["t_s"]), 6), "|".join(blockers) or "none"


def audit_lock_trigger_batch(batch_dir: Path) -> list[dict[str, Any]]:
    """Write a derived lock-trigger audit for an existing Step5c2 batch."""
    batch_dir = Path(batch_dir)
    lock_config = _load_lock_config_from_batch(batch_dir)
    derived_dir = batch_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(batch_dir.rglob("agent_state_timeseries.csv")):
        trial_dir = csv_path.parent
        agent_rows = _read_csv_rows(csv_path)
        samples = _samples_from_agent_rows(agent_rows)
        if not samples:
            continue
        r_flags = [float(s["order_parameter_R"]) >= lock_config.lock_r_threshold for s in samples]
        phase_flags = [
            float(s["mean_pairwise_phase_error_cycles"])
            <= lock_config.lock_phase_error_threshold_cycles
            for s in samples
        ]
        freq_flags = [
            float(s["frequency_disagreement_hz"])
            <= lock_config.lock_frequency_disagreement_threshold_hz
            for s in samples
        ]
        all_flags = [_sample_lock_pass(s, lock_config) for s in samples]
        rolling_flags = _rolling_lock_pass_flags(samples, lock_config)
        closest_t, blocker = _closest_to_lock(samples, lock_config)
        phase_vals = [float(s["mean_pairwise_phase_error_cycles"]) for s in samples]
        freq_vals = [float(s["frequency_disagreement_hz"]) for s in samples]
        r_vals = [float(s["order_parameter_R"]) for s in samples]
        try:
            rel_trial = trial_dir.relative_to(batch_dir)
        except ValueError:
            rel_trial = trial_dir
        lock_rows = (
            _read_csv_rows(trial_dir / "lock_hold_state_timeseries.csv")
            if (trial_dir / "lock_hold_state_timeseries.csv").exists()
            else []
        )
        _plot_sync_criterion(
            trial_dir / "sync_criterion_vs_time.png",
            samples,
            lock_config=lock_config,
            lock_rows=lock_rows,
        )
        rows.append({
            "trial_dir": str(rel_trial),
            "sample_count": len(samples),
            "lock_r_threshold": lock_config.lock_r_threshold,
            "lock_phase_error_threshold_cycles": lock_config.lock_phase_error_threshold_cycles,
            "lock_frequency_disagreement_threshold_hz": (
                lock_config.lock_frequency_disagreement_threshold_hz
            ),
            "lock_window_s": lock_config.lock_window_s,
            "lock_window_pass_ratio": lock_config.lock_window_pass_ratio,
            "percent_samples_r_ok": round(100.0 * sum(r_flags) / len(r_flags), 6),
            "percent_samples_phase_ok": round(100.0 * sum(phase_flags) / len(phase_flags), 6),
            "percent_samples_frequency_ok": round(100.0 * sum(freq_flags) / len(freq_flags), 6),
            "percent_samples_all_lock_ok": round(100.0 * sum(all_flags) / len(all_flags), 6),
            "longest_continuous_all_lock_s": _longest_interval_duration(
                _boolean_intervals(samples, all_flags)
            ),
            "longest_rolling_window_pass_s": _longest_interval_duration(
                _boolean_intervals(samples, rolling_flags)
            ),
            "closest_to_lock_time_s": closest_t,
            "closest_to_lock_blocker": blocker,
            "min_phase_error_cycles": round(min(phase_vals), 6),
            "median_phase_error_cycles": round(float(statistics.median(phase_vals)), 6),
            "final_phase_error_cycles": round(phase_vals[-1], 6),
            "min_frequency_disagreement_hz": round(min(freq_vals), 6),
            "median_frequency_disagreement_hz": round(float(statistics.median(freq_vals)), 6),
            "final_frequency_disagreement_hz": round(freq_vals[-1], 6),
            "median_R": round(float(statistics.median(r_vals)), 6),
            "final_R": round(r_vals[-1], 6),
        })
    if rows:
        _write_csv(derived_dir / "lock_trigger_audit.csv", rows)
    return rows


def _summary_row(
    condition: dict[str, Any],
    trial_index: int,
    trial_dir: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "condition": condition["name"],
        "category": condition["category"],
        "stabilizer": metrics.get("stabilizer"),
        "frequency_set": metrics.get("frequency_set"),
        "trial": trial_index,
        "trial_dir": str(trial_dir),
        "sync_success": metrics.get("sync_success"),
        "final_sync_success": metrics.get("final_sync_success"),
        "continuous_sync_success": metrics.get("continuous_sync_success"),
        "frequency_stability_success": metrics.get("frequency_stability_success"),
        "time_to_sync_s": metrics.get("time_to_sync_s"),
        "final_mean_pairwise_phase_error_cycles": metrics.get(
            "final_mean_pairwise_phase_error_cycles"
        ),
        "final_max_pairwise_phase_error_cycles": metrics.get(
            "final_max_pairwise_phase_error_cycles"
        ),
        "final_frequency_disagreement_hz": metrics.get("final_frequency_disagreement_hz"),
        "final_mean_order_parameter_R": metrics.get("final_mean_order_parameter_R"),
        "final_mean_common_frequency_hz": metrics.get("final_mean_common_frequency_hz"),
        "final_common_frequency_std_hz": metrics.get("final_common_frequency_std_hz"),
        "final_common_frequency_slope_hz_per_s": metrics.get(
            "final_common_frequency_slope_hz_per_s"
        ),
        "lock_acquired": metrics.get("lock_acquired"),
        "lock_time_s": metrics.get("lock_time_s"),
        "f_lock_hz": metrics.get("f_lock_hz"),
        "f_lock_source_window_s": metrics.get("f_lock_source_window_s"),
        "common_frequency_at_lock_mean_hz": metrics.get("common_frequency_at_lock_mean_hz"),
        "common_frequency_at_lock_median_hz": metrics.get("common_frequency_at_lock_median_hz"),
        "common_frequency_at_lock_std_hz": metrics.get("common_frequency_at_lock_std_hz"),
        "hold_duration_s": metrics.get("hold_duration_s"),
        "unlock_count": metrics.get("unlock_count"),
        "relock_count": metrics.get("relock_count"),
        "final_hold_state": metrics.get("final_hold_state"),
        "recovery_time_s": metrics.get("recovery_time_s"),
        "V0_detection_recall": (metrics.get("detection_recall") or {}).get("V0"),
        "V1_detection_recall": (metrics.get("detection_recall") or {}).get("V1"),
        "api_failure_count": (metrics.get("api_health") or {}).get("api_failure_count"),
        "fault_kind": (metrics.get("fault_injection") or {}).get("fault_kind"),
        "virtual_events_dropped": (metrics.get("fault_injection") or {}).get(
            "virtual_events_dropped"
        ),
        "p0_events_delayed": (metrics.get("fault_injection") or {}).get("p0_events_delayed"),
    }


def _mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            vals.append(f)
    return round(statistics.mean(vals), 6) if vals else None


def _batch_condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    keys = sorted({
        (
            row.get("condition", ""),
            row.get("frequency_set", ""),
            row.get("stabilizer", ""),
        )
        for row in rows
    })
    for condition, freq_set, stabilizer in keys:
        subset = [
            row for row in rows
            if row.get("condition", "") == condition
            and row.get("frequency_set", "") == freq_set
            and row.get("stabilizer", "") == stabilizer
        ]
        continuous_successes = [bool(row.get("continuous_sync_success")) for row in subset]
        final_successes = [bool(row.get("final_sync_success")) for row in subset]
        stable_successes = [bool(row.get("frequency_stability_success")) for row in subset]
        lock_successes = [bool(row.get("lock_acquired")) for row in subset]
        summaries.append({
            "condition": condition,
            "frequency_set": freq_set,
            "stabilizer": stabilizer,
            "trials": len(subset),
            "success_rate": (
                round(sum(continuous_successes) / len(continuous_successes), 6)
                if continuous_successes else None
            ),
            "final_sync_success_rate": (
                round(sum(final_successes) / len(final_successes), 6)
                if final_successes else None
            ),
            "frequency_stability_success_rate": (
                round(sum(stable_successes) / len(stable_successes), 6)
                if stable_successes else None
            ),
            "lock_acquired_rate": (
                round(sum(lock_successes) / len(lock_successes), 6)
                if lock_successes else None
            ),
            "mean_time_to_sync_s": _mean([row.get("time_to_sync_s") for row in subset]),
            "mean_lock_time_s": _mean([row.get("lock_time_s") for row in subset]),
            "mean_final_phase_error_cycles": _mean([
                row.get("final_mean_pairwise_phase_error_cycles") for row in subset
            ]),
            "mean_final_frequency_disagreement_hz": _mean([
                row.get("final_frequency_disagreement_hz") for row in subset
            ]),
            "mean_final_order_parameter_R": _mean([
                row.get("final_mean_order_parameter_R") for row in subset
            ]),
            "mean_final_common_frequency_std_hz": _mean([
                row.get("final_common_frequency_std_hz") for row in subset
            ]),
            "mean_final_common_frequency_slope_hz_per_s": _mean([
                row.get("final_common_frequency_slope_hz_per_s") for row in subset
            ]),
            "mean_hold_duration_s": _mean([row.get("hold_duration_s") for row in subset]),
            "mean_recovery_time_s": _mean([row.get("recovery_time_s") for row in subset]),
        })
    return summaries


def _save_batch_plots(batch_dir: Path, rows: list[dict[str, Any]]) -> None:
    summaries = _batch_condition_summary(rows)
    if not summaries:
        return
    plot_specs = [
        ("success_rate", "Success rate", "success_rate_by_condition.png"),
        ("final_sync_success_rate", "Final sync success rate", "final_sync_success_by_condition.png"),
        ("mean_time_to_sync_s", "Time to sync (s)", "time_to_sync_by_condition.png"),
        ("mean_lock_time_s", "Lock time (s)", "lock_time_by_condition.png"),
        (
            "mean_final_phase_error_cycles",
            "Final phase error (cycles)",
            "final_phase_error_by_condition.png",
        ),
        (
            "mean_final_frequency_disagreement_hz",
            "Final frequency disagreement (Hz)",
            "final_frequency_disagreement_by_condition.png",
        ),
        ("mean_final_order_parameter_R", "Final order parameter R", "final_order_parameter_by_condition.png"),
        (
            "mean_final_common_frequency_std_hz",
            "Common frequency std (Hz)",
            "common_frequency_std_by_condition.png",
        ),
        (
            "mean_final_common_frequency_slope_hz_per_s",
            "Common frequency slope (Hz/s)",
            "common_frequency_slope_by_condition.png",
        ),
        ("mean_hold_duration_s", "Hold duration (s)", "hold_duration_by_condition.png"),
        ("mean_recovery_time_s", "Recovery time (s)", "recovery_time_by_condition.png"),
    ]
    for key, ylabel, filename in plot_specs:
        labels = [
            "\n".join(str(v) for v in (
                item.get("condition"),
                item.get("frequency_set") or None,
                item.get("stabilizer") or None,
            ) if v)
            for item in summaries if item.get(key) is not None
        ]
        vals = [item[key] for item in summaries if item.get(key) is not None]
        if not vals:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 3.6))
        ax.bar(labels, vals)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(batch_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)


def _manual_or_config_rois(args: argparse.Namespace) -> tuple[list[int] | None, list[int] | None, str | None]:
    roi_v0 = args.roi_v0
    roi_v1 = args.roi_v1
    roi_config = args.roi_config
    if roi_config:
        cfg_v0, cfg_v1 = hil._load_roi_config(roi_config)
        roi_v0 = roi_v0 or cfg_v0
        roi_v1 = roi_v1 or cfg_v1
    return roi_v0, roi_v1, roi_config


def _auto_roi_for_condition(
    args: argparse.Namespace,
    condition: dict[str, Any],
    condition_dir: Path,
) -> tuple[list[int], list[int], str]:
    initial = {
        "V0": {"initial_phase_rad": 0.0, "initial_frequency_hz": args.auto_roi_v0_frequency},
        "V1": {
            "initial_phase_rad": args.auto_roi_v1_phase_rad,
            "initial_frequency_hz": args.auto_roi_v1_frequency,
        },
        "P0": {"initial_phase_rad": 0.0, "initial_frequency_hz": 2.0},
    }
    auto_dir = condition_dir / "auto_roi"
    auto_dir.mkdir(parents=True, exist_ok=False)
    auto_args = _make_trial_args(
        args,
        condition,
        initial,
        auto_dir,
        roi_v0=None,
        roi_v1=None,
        roi_config=None,
    )
    auto_args.mode = "auto_roi_calibration"
    metrics = run_auto_roi_calibration(auto_args, auto_dir)
    _write_json(auto_dir / "metrics_summary.json", metrics)
    auto_rois = metrics.get("auto_rois", {})
    _write_json(auto_dir / "auto_rois.json", auto_rois)
    if not auto_rois.get("calibration_valid", False):
        raise RuntimeError(
            f"Auto ROI calibration failed for {condition['name']}: "
            f"{auto_rois.get('failure_reason')}"
        )
    roi_v0 = auto_rois.get("V0", {}).get("roi")
    roi_v1 = auto_rois.get("V1", {}).get("roi")
    if roi_v0 is None or roi_v1 is None:
        raise RuntimeError(f"Auto ROI calibration did not return both ROIs for {condition['name']}")
    return list(roi_v0), list(roi_v1), str(auto_dir / "auto_rois.json")


def _dry_run_plan(args: argparse.Namespace, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dry_run": True,
        "would_create_log_root": args.log_dir,
        "batch_name": args.batch_name,
        "topology": args.topology,
        "trials_per_condition": args.trials,
        "duration_s": args.duration,
        "random_initial": args.random_initial,
        "conditions": [condition["name"] for condition in conditions],
        "frequency_sets": args.freq_sets,
        "stabilizer": args.stabilizer,
        "model": "EAPF Consensus",
        "model_parameters": LOCKED_EAPF_PARAMETERS,
        "stabilizer_settings": _lock_config_from_args(args).__dict__,
        "note": "Dry-run creates no trial folders and does not touch hardware.",
    }


def _lock_config_from_args(args: argparse.Namespace) -> LockHoldConfig:
    return LockHoldConfig(
        stabilizer=args.stabilizer,
        lock_r_threshold=args.lock_r_threshold,
        lock_phase_error_threshold_cycles=args.lock_phase_error_threshold_cycles,
        lock_frequency_disagreement_threshold_hz=args.lock_frequency_disagreement_threshold_hz,
        lock_window_s=args.lock_window_s,
        lock_window_pass_ratio=args.lock_window_pass_ratio,
        hold_frequency_anchor=args.hold_frequency_anchor,
        hold_phase_gain_scale=args.hold_phase_gain_scale,
        hold_frequency_gain_scale=args.hold_frequency_gain_scale,
        hold_anchor_gain=args.hold_anchor_gain,
        unlock_r_threshold=args.unlock_r_threshold,
        unlock_phase_error_threshold_cycles=args.unlock_phase_error_threshold_cycles,
        unlock_frequency_disagreement_threshold_hz=args.unlock_frequency_disagreement_threshold_hz,
        unlock_window_s=args.unlock_window_s,
        unlock_window_fail_ratio=args.unlock_window_fail_ratio,
    )


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    verify_locked_eapf_config()
    if args.topology != "all_to_all":
        raise SystemExit("Step5c2 currently supports --topology all_to_all first.")
    definitions = _condition_definitions()
    missing = [name for name in args.conditions if name not in definitions]
    if missing:
        raise SystemExit(f"Unknown condition(s): {', '.join(missing)}")
    unknown_freq_sets = [name for name in args.freq_sets if name not in FREQUENCY_SETS]
    if unknown_freq_sets:
        raise SystemExit(f"Unknown frequency set(s): {', '.join(unknown_freq_sets)}")
    conditions = [definitions[name] for name in args.conditions]
    thresholds = SyncThresholds(
        final_window_s=args.final_window_s,
        required_window_s=args.sync_window_s,
        window_pass_ratio=args.sync_window_pass_ratio,
        mean_pairwise_phase_error_cycles=args.phase_error_cycles_threshold,
        mean_pairwise_phase_error_rad=args.phase_error_rad_threshold,
        frequency_disagreement_hz=args.frequency_disagreement_threshold_hz,
        order_parameter_R=args.order_parameter_threshold,
        frequency_stability_std_hz=args.frequency_stability_std_threshold_hz,
        frequency_stability_slope_hz_per_s=args.frequency_stability_slope_threshold_hz_per_s,
    )
    lock_config = _lock_config_from_args(args)

    if args.dry_run:
        plan = _dry_run_plan(args, conditions)
        print(json.dumps(plan, indent=2))
        return plan

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.log_dir) / f"{stamp}_{args.batch_name}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    _write_json(batch_dir / "batch_config.json", {
        "step": "Step5c2",
        "system": "mixed_reality_2_virtual_1_pi",
        "model": "EAPF Consensus",
        "model_parameters": LOCKED_EAPF_PARAMETERS,
        "leader_api": args.leader_api,
        "topology": args.topology,
        "trials_per_condition": args.trials,
        "duration_s": args.duration,
        "random_initial": args.random_initial,
        "seed": args.seed,
        "conditions": conditions,
        "frequency_sets": {
            name: FREQUENCY_SETS[name] for name in args.freq_sets
        },
        "sync_thresholds": thresholds.__dict__,
        "stabilizer": lock_config.__dict__,
        "created_at": datetime.now().isoformat(),
    })

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_dir = batch_dir / "conditions" / condition["name"]
        condition_dir.mkdir(parents=True, exist_ok=False)
        _write_json(condition_dir / "condition_config.json", condition)
        roi_v0, roi_v1, roi_config = _manual_or_config_rois(args)
        if roi_v0 is None or roi_v1 is None:
            roi_v0, roi_v1, roi_config = _auto_roi_for_condition(args, condition, condition_dir)
        for freq_set in args.freq_sets:
            freq_dir = condition_dir / freq_set
            freq_dir.mkdir(parents=True, exist_ok=False)
            _write_json(freq_dir / "frequency_set_config.json", FREQUENCY_SETS[freq_set])
            for trial in range(1, args.trials + 1):
                trial_seed = int(args.seed) + len(rows) * 7919 + trial
                initial_states = generate_initial_states(
                    args.seed,
                    len(rows) + 1,
                    random_initial=args.random_initial,
                    frequency_min_hz=args.initial_frequency_min_hz,
                    frequency_max_hz=args.initial_frequency_max_hz,
                    freq_set=freq_set,
                )
                trial_dir = freq_dir / f"trial_{trial:02d}"
                trial_dir.mkdir(parents=True, exist_ok=False)
                try:
                    metrics = run_trial(
                        args,
                        condition,
                        initial_states,
                        trial_dir,
                        roi_v0=roi_v0,
                        roi_v1=roi_v1,
                        roi_config=roi_config,
                        thresholds=thresholds,
                        lock_config=lock_config,
                        trial_seed=trial_seed,
                        freq_set=freq_set,
                    )
                    row = _summary_row(condition, trial, trial_dir, metrics)
                    rows.append(row)
                    print(
                        f"{condition['name']} {freq_set} trial {trial}/{args.trials}: "
                        f"final_sync={row['final_sync_success']} "
                        f"stable={row['frequency_stability_success']} "
                        f"lock={row['lock_acquired']}"
                    )
                except Exception as exc:
                    failure = {
                        "condition": condition["name"],
                        "frequency_set": freq_set,
                        "trial": trial,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_at": datetime.now().isoformat(),
                    }
                    _write_json(trial_dir / "failure_summary.json", failure)
                    if not args.continue_on_failure:
                        raise
                    rows.append({
                        "condition": condition["name"],
                        "category": condition["category"],
                        "frequency_set": freq_set,
                        "stabilizer": args.stabilizer,
                        "trial": trial,
                        "trial_dir": str(trial_dir),
                        "final_sync_success": False,
                        "continuous_sync_success": False,
                        "frequency_stability_success": False,
                        "failure": failure["error"],
                    })

    _write_csv(batch_dir / "batch_summary.csv", rows)
    condition_summaries = _batch_condition_summary(rows)
    batch_summary = {
        "batch_dir": str(batch_dir),
        "conditions": [condition["name"] for condition in conditions],
        "frequency_sets": args.freq_sets,
        "stabilizer": args.stabilizer,
        "trials_completed": len(rows),
        "condition_summaries": condition_summaries,
        "overall_success_rate": (
            round(sum(1 for row in rows if row.get("continuous_sync_success")) / len(rows), 6)
            if rows else None
        ),
        "overall_final_sync_success_rate": (
            round(sum(1 for row in rows if row.get("final_sync_success")) / len(rows), 6)
            if rows else None
        ),
        "overall_frequency_stability_success_rate": (
            round(sum(1 for row in rows if row.get("frequency_stability_success")) / len(rows), 6)
            if rows else None
        ),
    }
    _write_json(batch_dir / "batch_summary.json", batch_summary)
    _save_batch_plots(batch_dir, rows)
    print(f"Batch output: {batch_dir}")
    print(json.dumps(batch_summary, indent=2))
    return batch_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--topology", choices=["all_to_all"], default="all_to_all")
    parser.add_argument("--conditions", nargs="+", default=["baseline"])
    parser.add_argument("--freq-sets", nargs="+", default=["random_1p6_2p4"],
                        choices=sorted(FREQUENCY_SETS.keys()))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--batch-name", default="all_to_all_smoke")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--random-initial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")

    parser.add_argument("--initial-frequency-min-hz", type=float, default=1.6)
    parser.add_argument("--initial-frequency-max-hz", type=float, default=2.4)

    parser.add_argument("--v0-x", type=int, default=520)
    parser.add_argument("--v0-y", type=int, default=420)
    parser.add_argument("--v1-x", type=int, default=1180)
    parser.add_argument("--v1-y", type=int, default=420)
    parser.add_argument("--dot-size", type=int, default=280)
    parser.add_argument("--v0-size", type=int, default=None)
    parser.add_argument("--v1-size", type=int, default=None)
    parser.add_argument("--background-brightness", type=int, default=0)
    parser.add_argument("--flash-brightness", type=int, default=255)
    parser.add_argument("--off-brightness", type=int, default=15)

    parser.add_argument("--roi-v0", type=_parse_roi, default=None)
    parser.add_argument("--roi-v1", type=_parse_roi, default=None)
    parser.add_argument("--roi-config", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--camera-format", default="BGR888")
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--norm-on-threshold", type=float, default=0.65)
    parser.add_argument("--norm-off-threshold", type=float, default=0.35)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    parser.add_argument("--episode-latch", action="store_true")
    parser.add_argument("--led-pin", type=int, default=17)
    parser.add_argument("--led-pulse-duration", type=float, default=0.06)
    parser.add_argument("--api-timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.2)

    parser.add_argument("--auto-roi-combined-duration", type=float, default=6.0)
    parser.add_argument("--auto-roi-duration", type=float, default=3.0)
    parser.add_argument("--auto-roi-verify-duration", type=float, default=1.0)
    parser.add_argument("--auto-roi-warmup-s", type=float, default=0.5)
    parser.add_argument("--auto-roi-capture-fps", type=float, default=15.0)
    parser.add_argument("--auto-roi-v0-frequency", type=float, default=1.0)
    parser.add_argument("--auto-roi-v1-frequency", type=float, default=2.0)
    parser.add_argument("--auto-roi-v1-phase-rad", type=float, default=math.pi / 2.0)
    parser.add_argument("--auto-roi-sequential-diagnostics", action="store_true")
    parser.add_argument("--auto-roi-frequency-ambiguity-hz", type=float, default=0.25)
    parser.add_argument("--auto-roi-method", choices=[
        "temporal_variance", "max_min_range", "mean_abs_diff",
    ], default="temporal_variance")
    parser.add_argument("--auto-roi-padding", type=int, default=35)
    parser.add_argument("--auto-roi-min-area", type=int, default=50)
    parser.add_argument("--auto-roi-downsample", type=int, default=1)
    parser.add_argument("--auto-roi-change-threshold", type=float, default=None)
    parser.add_argument("--auto-roi-boundary-margin-px", type=int, default=5)
    parser.add_argument("--auto-roi-overlap-warning-ratio", type=float, default=0.05)
    parser.add_argument("--auto-roi-max-area-fraction", type=float, default=0.35)

    parser.add_argument("--final-window-s", type=float, default=10.0)
    parser.add_argument("--sync-window-s", type=float, default=5.0)
    parser.add_argument("--sync-window-pass-ratio", type=float, default=0.8)
    parser.add_argument("--phase-error-cycles-threshold", type=float, default=0.25)
    parser.add_argument("--phase-error-rad-threshold", type=float, default=math.pi / 2.0)
    parser.add_argument("--frequency-disagreement-threshold-hz", type=float, default=0.10)
    parser.add_argument("--order-parameter-threshold", type=float, default=0.90)
    parser.add_argument("--frequency-stability-std-threshold-hz", type=float, default=0.05)
    parser.add_argument("--frequency-stability-slope-threshold-hz-per-s", type=float, default=0.005)

    parser.add_argument("--stabilizer", choices=["none", "lock_hold"], default="none")
    parser.add_argument("--lock-r-threshold", type=float, default=0.95)
    parser.add_argument("--lock-phase-error-threshold-cycles", type=float, default=0.08)
    parser.add_argument("--lock-frequency-disagreement-threshold-hz", type=float, default=0.05)
    parser.add_argument("--lock-window-s", type=float, default=1.0)
    parser.add_argument("--lock-window-pass-ratio", type=float, default=0.5)
    parser.add_argument(
        "--hold-frequency-anchor",
        choices=["current_mean", "window_mean", "window_median"],
        default="window_median",
    )
    parser.add_argument("--hold-phase-gain-scale", type=float, default=0.1)
    parser.add_argument("--hold-frequency-gain-scale", type=float, default=0.0)
    parser.add_argument("--hold-anchor-gain", type=float, default=0.08)
    parser.add_argument("--unlock-r-threshold", type=float, default=0.85)
    parser.add_argument("--unlock-phase-error-threshold-cycles", type=float, default=0.15)
    parser.add_argument("--unlock-frequency-disagreement-threshold-hz", type=float, default=0.10)
    parser.add_argument("--unlock-window-s", type=float, default=4.0)
    parser.add_argument("--unlock-window-fail-ratio", type=float, default=0.8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()

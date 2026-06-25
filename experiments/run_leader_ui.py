"""Launch a local HTTP server for the Firefly Leader UI with remote API.

Supports two modes:
  - ``fixed_leader`` (default): single flashing dot at fixed frequency.
  - ``mutual_hil``: virtual agents with their own oscillators that can
    receive Pi flash events via ``POST /api/pi_flash``.

Endpoints (mutual_hil):
  GET  /api/mode              — get current mode
  POST /api/mode              — set mode {"mode":"fixed_leader|mutual_hil"}
  GET  /api/agents            — list all virtual agents
  POST /api/agents            — create/configure agent
  POST /api/agents/{id}       — update agent
  POST /api/start             — start oscillators
  POST /api/pause             — pause oscillators
  POST /api/reset             — reset all agents
  POST /api/pi_flash          — register a Pi flash event {"timestamp": <s>}
  GET  /api/status            — leader config + mode + agents
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in separate threads so polling and API don't block."""
    daemon_threads = True
from pathlib import Path
from urllib.parse import urlparse

from firefly_sync.core.event_based_consensus_pll import (
    ConsensusPLLConfig,
    EventBasedConsensusPLLOscillator,
)
from firefly_sync.multi_agent.hil_topology import (
    MIXED_REALITY_TOPOLOGIES,
    build_mixed_reality_topology,
    build_single_mutual_topology,
)

ROOT = Path(__file__).resolve().parent / "leader_ui"
HOST_DEFAULT = "127.0.0.1"
PORT_DEFAULT = 8000

# ======================================================================
# Shared state
# ======================================================================

_leader_config: dict = {
    "frequency_hz": 1.0, "duty_cycle": 0.5,
    "brightness_on": 255, "brightness_off": 0,
    "background_brightness": 0, "target_size_px": 100,
    "shape": "circle", "offset_x": 0, "offset_y": 0,
    "running": False, "api_controlled": False,
    "feedback_enabled": False, "flash_duration_s": 0.10,
    "feedback_flash_hold_s": 0.25,
    "mutual_agent_mode": "single_1v1p",
    "topology": "all_to_all",
}
_mode: str = "fixed_leader"
_agents: list[dict] = []
_agents_snapshot: list[dict] = []
_agents_lock = threading.RLock()
_snapshot_lock = threading.Lock()  # lightweight, only guards snapshot assignment
_config_lock = threading.Lock()
_pi_flash_times: list[float] = []
_pending_flash_events: list[dict] = []
_frontend_display_state: dict = {}
_frontend_display_state_lock = threading.Lock()
_stabilizer_config: dict = {
    "enabled": False,
    "state": "acquisition",
    "f_lock_hz": None,
    "hold_phase_gain_scale": 0.1,
    "hold_frequency_gain_scale": 0.0,
    "hold_anchor_gain": 0.02,
}
_stabilizer_lock = threading.Lock()
_loop_thread: threading.Thread | None = None
_loop_running: bool = False
_loop_start_time: float = 0.0
_calibration_start_time: float = 0.0
_visual_duty_cycle: float = 0.5

# ======================================================================
# Config helpers
# ======================================================================

def _get_config_copy() -> dict:
    with _config_lock:
        return dict(_leader_config)

def _update_config(updates: dict) -> dict:
    with _config_lock:
        for k, v in updates.items():
            if k in _leader_config:
                _leader_config[k] = v
        _leader_config["api_controlled"] = True
        return dict(_leader_config)

def _wrap_2pi(value: float) -> float:
    return float(value) % (2.0 * math.pi)

def _get_stabilizer_copy() -> dict:
    with _stabilizer_lock:
        return dict(_stabilizer_config)

def _update_stabilizer(updates: dict) -> dict:
    allowed = {
        "enabled", "state", "f_lock_hz", "hold_phase_gain_scale",
        "hold_frequency_gain_scale", "hold_anchor_gain",
    }
    with _stabilizer_lock:
        for key, value in updates.items():
            if key in allowed:
                _stabilizer_config[key] = value
        return dict(_stabilizer_config)

# ======================================================================
# Virtual agent helpers
# ======================================================================

def _init_agent(aid: int, freq: float = 2.0, model: str = "eapf_consensus",
                x: float = 0, y: float = 0, size: int = 200,
                initial_phase_rad: float = 0.0,
                agent_id: str | None = None,
                role: str = "virtual",
                brightness_on: int | None = None,
                brightness_off: int | None = None,
                background_brightness: int | None = None) -> dict:
    phase = _wrap_2pi(initial_phase_rad)
    stable_id = agent_id or f"V{aid}"
    cfg = _get_config_copy()
    return {
        "id": aid, "agent_id": stable_id, "role": role,
        "model": model, "x": x, "y": y, "size": size,
        "brightness_on": int(cfg.get("brightness_on", 255) if brightness_on is None else brightness_on),
        "brightness_off": int(cfg.get("brightness_off", 0) if brightness_off is None else brightness_off),
        "background_brightness": int(
            cfg.get("background_brightness", 0)
            if background_brightness is None else background_brightness
        ),
        "initial_frequency_hz": freq, "frequency_hz": freq,
        "kuramoto_gain": 5.0,
        "initial_phase_rad": phase,
        "phase": phase, "phase_rad": phase,
        "flash_on": False, "last_flash_time": 0.0,
        "fire_count": 0, "enabled": True,
        "received_pi_flashes": 0,
        "pi_flash_posts_received": 0,
        "pi_flash_events_consumed": 0,
        "flash_times": [],
        "received_neighbour_events": 0,
        "visible_neighbours": [],
        "topology": "single_mutual",
        # EAPF Consensus state
        "neighbour_flash_times": [],
        "neighbour_period_est": 1.0 / freq if freq > 0 else 0.5,
        "neighbour_phase_est": 0.0,
        "phase_error_filt": 0.0, "freq_error_filt": 0.0,
        "oscillator": None,
    }

def _init_pi_agent(freq: float = 2.0, initial_phase_rad: float = 0.0) -> dict:
    return _init_agent(
        2, freq=freq, model="eapf_consensus", x=0, y=0, size=0,
        initial_phase_rad=initial_phase_rad, agent_id="P0", role="pi",
    )

def _locked_consensus_config(freq: float) -> ConsensusPLLConfig:
    return ConsensusPLLConfig(
        natural_frequency_hz=float(freq),
        phase_gain=0.02,
        frequency_gain=0.02,
        phase_error_filter_alpha=0.2,
        frequency_error_filter_alpha=0.2,
        max_phase_step_rad=0.2,
        max_frequency_step_hz=0.05,
        frequency_min_hz=0.8,
        frequency_max_hz=3.2,
    )

def _ensure_consensus_oscillator(a: dict) -> EventBasedConsensusPLLOscillator:
    osc = a.get("oscillator")
    if not isinstance(osc, EventBasedConsensusPLLOscillator):
        osc = EventBasedConsensusPLLOscillator(
            _locked_consensus_config(a.get("initial_frequency_hz", 2.0))
        )
        osc._phase_rad = _wrap_2pi(a.get("phase_rad", a.get("initial_phase_rad", 0.0)))
        osc._frequency_hz = float(a.get("frequency_hz", a.get("initial_frequency_hz", 2.0)))
        osc._omega_rad_s = 2.0 * math.pi * osc._frequency_hz
        a["oscillator"] = osc
    return osc

def _apply_stabilizer_to_virtual_oscillator(
    a: dict,
    osc: EventBasedConsensusPLLOscillator,
    stabilizer: dict,
) -> None:
    base = _locked_consensus_config(a.get("initial_frequency_hz", 2.0))
    in_hold = bool(stabilizer.get("enabled")) and stabilizer.get("state") == "hold"
    if in_hold:
        osc.config.phase_gain = base.phase_gain * float(
            stabilizer.get("hold_phase_gain_scale", 0.1)
        )
        osc.config.frequency_gain = base.frequency_gain * float(
            stabilizer.get("hold_frequency_gain_scale", 0.0)
        )
        f_lock = stabilizer.get("f_lock_hz")
        if f_lock is not None:
            anchor_gain = float(stabilizer.get("hold_anchor_gain", 0.02))
            anchored = float(osc.frequency_hz) + anchor_gain * (
                float(f_lock) - float(osc.frequency_hz)
            )
            osc._frequency_hz = max(
                osc.config.frequency_min_hz,
                min(osc.config.frequency_max_hz, anchored),
            )
            osc._omega_rad_s = 2.0 * math.pi * osc._frequency_hz
            a["frequency_hz"] = osc._frequency_hz
    else:
        osc.config.phase_gain = base.phase_gain
        osc.config.frequency_gain = base.frequency_gain
        osc.config.phase_error_filter_alpha = base.phase_error_filter_alpha
        osc.config.frequency_error_filter_alpha = base.frequency_error_filter_alpha
        osc.config.max_phase_step_rad = base.max_phase_step_rad
        osc.config.max_frequency_step_hz = base.max_frequency_step_hz

def _sync_agent_from_oscillator(a: dict, result: dict, now: float,
                                flash_duration_s: float) -> None:
    a["phase_rad"] = float(result.get("phase_rad", a.get("phase_rad", 0.0)))
    a["phase"] = a["phase_rad"]
    a["frequency_hz"] = float(result.get("frequency_hz", a.get("frequency_hz", 0.0)))
    a["phase_error_rad"] = result.get("phase_error_rad", 0.0)
    a["freq_error_hz"] = result.get("freq_error_hz", 0.0)
    if result.get("follower_flash_event"):
        a["flash_on"] = True
        a["fire_count"] = int(result.get("fire_count", a.get("fire_count", 0) + 1))
        a["last_flash_time"] = now
        a.setdefault("flash_times", []).append(now)
    elif a.get("flash_on") and (now - a.get("last_flash_time", 0.0)) >= flash_duration_s:
        a["flash_on"] = False

def _build_default_multi_agents() -> list[dict]:
    return [
        _init_agent(0, freq=1.9, model="eapf_consensus", x=520, y=420, size=280,
                    agent_id="V0", role="virtual"),
        _init_agent(1, freq=2.1, model="eapf_consensus", x=1180, y=420, size=280,
                    agent_id="V1", role="virtual"),
        _init_pi_agent(freq=2.0),
    ]

def _active_hil_topology():
    cfg = _get_config_copy()
    if cfg.get("mutual_agent_mode") == "multi_2v1p":
        return build_mixed_reality_topology(cfg.get("topology", "all_to_all"))
    return build_single_mutual_topology()

def _configure_mutual_agents(agent_mode: str, topology_name: str | None = None) -> dict:
    if agent_mode not in ("single_1v1p", "multi_2v1p"):
        raise ValueError("agent_mode must be single_1v1p or multi_2v1p")
    if topology_name is None:
        topology_name = _leader_config.get("topology", "all_to_all")
    if topology_name not in MIXED_REALITY_TOPOLOGIES:
        raise ValueError(f"topology must be one of {MIXED_REALITY_TOPOLOGIES}")

    cfg = _update_config({"mutual_agent_mode": agent_mode, "topology": topology_name})
    with _agents_lock:
        if agent_mode == "multi_2v1p":
            existing_mode = (
                len(_agents) >= 3
                and {a.get("agent_id") for a in _agents[:3]} >= {"V0", "V1", "P0"}
            )
            if not existing_mode:
                _agents.clear()
                _agents.extend(_build_default_multi_agents())
        else:
            if not _agents or len(_agents) != 1 or _agents[0].get("agent_id") != "V0":
                _agents.clear()
                _agents.append(
                    _init_agent(0, freq=2.0, model="eapf_consensus", x=800, y=400, size=350,
                                agent_id="V0", role="virtual")
                )
        topology = _active_hil_topology()
        for a in _agents:
            aid = a.get("agent_id", "V0")
            a["topology"] = topology.name
            a["visible_neighbours"] = topology.visible_neighbours(aid)
            if a.get("role") == "virtual" and agent_mode == "multi_2v1p":
                a["model"] = "eapf_consensus"
    _refresh_agents_snapshot()
    return cfg

def _compact_snapshot(a: dict) -> dict:
    """Return a lightweight dict for fast HTTP responses (no large lists)."""
    running = _leader_config.get("running", False)
    feedback_enabled = _leader_config.get("feedback_enabled", False)
    if feedback_enabled and _loop_start_time > 0:
        snapshot_time = max(0.0, time.monotonic() - _loop_start_time)
    else:
        snapshot_time = time.monotonic()
    feedback_flash_hold_s = _leader_config.get("feedback_flash_hold_s", 0.25)
    last_flash_time = a.get("last_flash_time", 0.0)
    flash_age_s = snapshot_time - last_flash_time
    if running and feedback_enabled:
        visual_flash_on = (
            last_flash_time > 0.0
            and 0.0 <= flash_age_s < feedback_flash_hold_s
        )
    else:
        visual_flash_on = a.get("flash_on") if running else False
    return {
        "id": a.get("id"), "agent_id": a.get("agent_id", f"V{a.get('id', 0)}"),
        "role": a.get("role", "virtual"), "model": a.get("model"),
        "x": a.get("x"), "y": a.get("y"), "size": a.get("size"),
        "brightness_on": a.get("brightness_on", _leader_config.get("brightness_on", 255)),
        "brightness_off": a.get("brightness_off", _leader_config.get("brightness_off", 0)),
        "background_brightness": a.get(
            "background_brightness",
            _leader_config.get("background_brightness", 0),
        ),
        "initial_frequency_hz": a.get("initial_frequency_hz"),
        "frequency_hz": a.get("frequency_hz"),
        "initial_phase_rad": a.get("initial_phase_rad", 0.0),
        "kuramoto_gain": a.get("kuramoto_gain", 5.0),
        "phase_rad": a.get("phase_rad") if running else a.get("phase_rad", 0.0),
        "flash_on": visual_flash_on,
        "raw_flash_on": a.get("flash_on") if running else False,
        "fire_count": a.get("fire_count"),
        "enabled": a.get("enabled", True),
        "received_pi_flashes": a.get("received_pi_flashes", 0),
        "pi_flash_posts_received": a.get("pi_flash_posts_received", a.get("received_pi_flashes", 0)),
        "pi_flash_events_consumed": a.get("pi_flash_events_consumed", 0),
        "last_flash_time": last_flash_time,
        "flash_age_s": flash_age_s,
        "snapshot_time": snapshot_time,
        "server_time": snapshot_time,
        "feedback_flash_hold_s": feedback_flash_hold_s,
        "flash_times_count": len(a.get("flash_times", [])),
        "visible_neighbours": list(a.get("visible_neighbours", [])),
        "received_neighbour_events": a.get("received_neighbour_events", 0),
        "topology": a.get("topology", _get_config_copy().get("topology", "all_to_all")),
        "mutual_agent_mode": _get_config_copy().get("mutual_agent_mode", "single_1v1p"),
        "kuramoto_last_warning": a.get("kuramoto_last_warning", ""),
        "calibration_elapsed": round(max(0.0, time.monotonic() - _calibration_start_time), 3) if _calibration_start_time > 0 else 0,
        "running": running,
        "feedback_enabled": feedback_enabled,
    }

def _reset_agent(a: dict) -> None:
    freq = a["initial_frequency_hz"]
    phase = _wrap_2pi(a.get("initial_phase_rad", 0.0))
    a["phase"] = phase
    a["phase_rad"] = phase
    a["frequency_hz"] = freq
    a["flash_on"] = False
    a["fire_count"] = 0
    a["last_flash_time"] = 0.0
    a["received_pi_flashes"] = 0
    a["pi_flash_posts_received"] = 0
    a["pi_flash_events_consumed"] = 0
    a["kuramoto_last_warning"] = ""
    a["neighbour_flash_times"] = []
    a["neighbour_period_est"] = 1.0 / freq if freq > 0 else 0.5
    a["phase_error_filt"] = 0.0
    a["freq_error_filt"] = 0.0
    a["received_neighbour_events"] = 0
    a["oscillator"] = None

def _step_calibration_agent(a: dict, _loop_now: float) -> None:
    """Step virtual agent using absolute wall-clock time (monotonic)."""
    now_abs = time.monotonic()
    elapsed = max(0.0, now_abs - _calibration_start_time)
    freq = a["initial_frequency_hz"]
    period = 1.0 / freq if freq > 0 else 0.5
    duty = _visual_duty_cycle
    phase_offset_cycles = _wrap_2pi(a.get("initial_phase_rad", 0.0)) / (2.0 * math.pi)
    cycle_position = (elapsed / period) + phase_offset_cycles
    cycle_index = int(cycle_position)
    cycle_phase = cycle_position % 1.0  # 0..1 within cycle
    a["phase_rad"] = 2.0 * math.pi * cycle_phase
    a["flash_on"] = (cycle_phase < duty)
    a["frequency_hz"] = freq
    # Fire count = cycle_index (incremented naturally as cycles complete)
    prev_fire = a.get("fire_count", 0)
    if cycle_index > prev_fire:
        a["fire_count"] = cycle_index
        a["last_flash_time"] = now_abs
        a.setdefault("flash_times", []).append(now_abs)
    a["received_pi_flashes"] = a.get("received_pi_flashes", 0)
    a["pi_flash_posts_received"] = a.get("pi_flash_posts_received", a.get("received_pi_flashes", 0))
    a["pi_flash_events_consumed"] = a.get("pi_flash_events_consumed", 0)

def _step_eapf_agent(a: dict, dt: float, now: float, pi_flash_times: list,
                     running: bool, feedback_enabled: bool,
                     flash_duration_s: float) -> None:
    """Step one virtual EAPF Consensus agent.  No-op if not running."""
    if not running:
        a["flash_on"] = False
        return

    # Flash hold: keep flash_on true for flash_duration_s after last fire
    if a.get("flash_on") and (now - a.get("last_flash_time", 0)) >= flash_duration_s:
        a["flash_on"] = False

    pg = 0.02; fg = 0.02; ap = 0.2; af = 0.2
    mps = 0.2; mfs = 0.05
    fmin = 0.8; fmax = 3.2  # safety bounds for virtual agent

    omega = 2.0 * math.pi * a["frequency_hz"]
    a["phase_rad"] += omega * dt
    if a["phase_rad"] >= 2.0 * math.pi:
        a["flash_on"] = True
        a["fire_count"] += 1
        a["last_flash_time"] = now
        a.setdefault("flash_times", []).append(now)
        a["phase_rad"] -= 2.0 * math.pi

    # Update neighbour phase estimate (always, even without feedback)
    period = a["neighbour_period_est"]
    if period > 0:
        a["neighbour_phase_est"] = (a["neighbour_phase_est"] + 2.0 * math.pi * (1.0/period) * dt) % (2.0 * math.pi)

    # Process Pi flashes ONLY when feedback is enabled
    if not feedback_enabled:
        return

    for pt in pi_flash_times:
        a["pi_flash_events_consumed"] = a.get("pi_flash_events_consumed", 0) + 1
        a["neighbour_flash_times"].append(pt)
        if len(a["neighbour_flash_times"]) > 12:
            a["neighbour_flash_times"].pop(0)
        if len(a["neighbour_flash_times"]) >= 2:
            intvs = [a["neighbour_flash_times"][i+1] - a["neighbour_flash_times"][i]
                     for i in range(len(a["neighbour_flash_times"]) - 1)]
            intvs.sort()
            a["neighbour_period_est"] = intvs[len(intvs)//2] if intvs else a["neighbour_period_est"]
        a["neighbour_phase_est"] = 0.0
        pe = ((a["neighbour_phase_est"] - a["phase_rad"] + math.pi) % (2.0 * math.pi)) - math.pi
        a["phase_error_filt"] = ap * pe + (1 - ap) * a["phase_error_filt"]
        a["freq_error_filt"] = af * ((1.0 / a["neighbour_period_est"] if a["neighbour_period_est"] > 0 else 0)
                                     - a["frequency_hz"]) + (1 - af) * a["freq_error_filt"]
        ps = max(-mps, min(mps, pg * a["phase_error_filt"]))
        a["phase_rad"] = (a["phase_rad"] + ps) % (2.0 * math.pi)
        fs = max(-mfs, min(mfs, fg * a["freq_error_filt"]))
        a["frequency_hz"] = max(fmin, min(fmax, a["frequency_hz"] + fs))

def _step_kuramoto_agent(a: dict, dt: float, now: float, pi_flash_times: list,
                         running: bool) -> None:
    """Step one virtual Kuramoto agent.  No-op if not running."""
    if not running:
        a["flash_on"] = False
        return
    K = float(a.get("kuramoto_gain", 5.0))
    natural_freq = a["initial_frequency_hz"]
    omega = 2.0 * math.pi * natural_freq
    phase_before = float(a.get("phase_rad", 0.0))
    frequency_before = float(a.get("frequency_hz", natural_freq))
    consumed_this_step = 0

    for pt in pi_flash_times:
        a["pi_flash_events_consumed"] = a.get("pi_flash_events_consumed", 0) + 1
        consumed_this_step += 1
        a["neighbour_flash_times"].append(pt)
        if len(a["neighbour_flash_times"]) > 12:
            a["neighbour_flash_times"].pop(0)
        if len(a["neighbour_flash_times"]) >= 2:
            intvs = [a["neighbour_flash_times"][i+1] - a["neighbour_flash_times"][i]
                     for i in range(len(a["neighbour_flash_times"]) - 1)]
            intvs = [v for v in intvs if v > 0]
            if intvs:
                intvs.sort()
                a["neighbour_period_est"] = intvs[len(intvs)//2]
        a["neighbour_phase_est"] = 0.0

    period = a.get("neighbour_period_est", 1.0 / natural_freq if natural_freq > 0 else 0.5)
    if period > 0:
        a["neighbour_phase_est"] = (
            a.get("neighbour_phase_est", 0.0) + 2.0 * math.pi * (1.0 / period) * dt
        ) % (2.0 * math.pi)

    neighbour_phase = float(a.get("neighbour_phase_est", 0.0))
    wrapped_phase_error = _wrap_pi(neighbour_phase - phase_before)
    coupling_input = math.sin(wrapped_phase_error)
    phase_velocity = omega + K * coupling_input
    applied_correction = K * coupling_input * dt
    phase_delta_unwrapped = phase_velocity * dt
    new_phase = phase_before + phase_delta_unwrapped
    flash_count_delta = 0
    while new_phase >= 2.0 * math.pi:
        flash_count_delta += 1
        new_phase -= 2.0 * math.pi
    while new_phase < 0.0:
        new_phase += 2.0 * math.pi
    a["phase_rad"] = new_phase
    a["phase"] = new_phase

    frequency_after = max(0.0, phase_velocity / (2.0 * math.pi))
    a["frequency_hz"] = frequency_after
    if flash_count_delta > 0:
        a["flash_on"] = True
        a["fire_count"] += flash_count_delta
        a["last_flash_time"] = now
        for _ in range(flash_count_delta):
            a.setdefault("flash_times", []).append(now)
    else:
        a["flash_on"] = False

    warnings: list[str] = []
    if not math.isfinite(frequency_after):
        warnings.append("frequency_nonfinite")
    if frequency_after < 0.5 or frequency_after > 4.0:
        warnings.append("frequency_out_of_safe_range")
    if math.isfinite(frequency_before) and abs(frequency_after - frequency_before) > 0.5:
        warnings.append("frequency_step_gt_0.5hz")
    if abs(phase_delta_unwrapped) > 0.5:
        warnings.append("phase_jump_gt_0.5rad")
    if flash_count_delta > 1:
        warnings.append("multiple_phase_wraps_in_step")
    if warnings:
        a["kuramoto_last_warning"] = ";".join(warnings)
    else:
        a["kuramoto_last_warning"] = ""

    phase_after = float(a["phase_rad"])
    phase_delta = _wrap_pi(phase_after - phase_before)
    effective_period = (1.0 / frequency_after) if frequency_after > 0 else float("inf")
    _append_kuramoto_debug({
        "time": round(now, 6),
        "dt": round(dt, 9),
        "agent_id": a.get("id", 0),
        "phase_before": round(phase_before, 9),
        "phase_after": round(phase_after, 9),
        "phase_delta": round(phase_delta, 9),
        "phase_delta_unwrapped": round(phase_delta_unwrapped, 9),
        "wrapped_phase_error": round(wrapped_phase_error, 9),
        "applied_kuramoto_correction": round(applied_correction, 9),
        "coupling_input": round(coupling_input, 9),
        "K": round(K, 9),
        "pi_flash_events_consumed_this_step": consumed_this_step,
        "frequency_hz_before": round(frequency_before, 9),
        "frequency_hz_after": round(frequency_after, 9),
        "effective_period_s": round(effective_period, 9) if math.isfinite(effective_period) else "inf",
        "fire_count": a.get("fire_count", 0),
        "flash_on": int(bool(a.get("flash_on"))),
        "flash_count_delta": flash_count_delta,
        "neighbour_period_est": round(float(a.get("neighbour_period_est", 0.0)), 9),
        "neighbour_phase_est": round(neighbour_phase, 9),
        "warning": ";".join(warnings),
    })

def _step_multi_eapf_agents(agents: list[dict], dt: float, now: float,
                            pending_events: list[dict], running: bool,
                            feedback_enabled: bool,
                            flash_duration_s: float) -> list[dict]:
    """Step the 2-virtual + 1-Pi EAPF branch and return new virtual events."""
    cfg = _get_config_copy()
    topology = _active_hil_topology()
    stabilizer = _get_stabilizer_copy()
    new_events: list[dict] = []
    if not running:
        for a in agents:
            a["flash_on"] = False
        return new_events

    for a in agents:
        aid = a.get("agent_id", "")
        a["topology"] = topology.name
        a["visible_neighbours"] = topology.visible_neighbours(aid)
        if a.get("role") != "virtual":
            continue
        if not feedback_enabled:
            _step_calibration_agent(a, now)
            continue

        source_ids = [
            str(evt.get("source_agent_id", evt.get("agent_id", "")))
            for evt in pending_events
            if topology.can_observe(aid, str(evt.get("source_agent_id", evt.get("agent_id", ""))))
        ]
        neighbour_ids = topology.numeric_neighbour_ids(aid, source_ids)
        a["received_neighbour_events"] = a.get("received_neighbour_events", 0) + len(neighbour_ids)
        a["pi_flash_events_consumed"] = a.get("pi_flash_events_consumed", 0) + source_ids.count("P0")
        osc = _ensure_consensus_oscillator(a)
        _apply_stabilizer_to_virtual_oscillator(a, osc, stabilizer)
        result = osc.step(dt_s=dt, t_s=now, neighbour_flash_ids=neighbour_ids)
        _sync_agent_from_oscillator(a, result, now, flash_duration_s)
        _apply_stabilizer_to_virtual_oscillator(a, osc, stabilizer)
        if result.get("follower_flash_event"):
            new_events.append({
                "timestamp": now,
                "source_agent_id": aid,
                "agent_id": aid,
                "event_type": "virtual_flash",
                "topology": cfg.get("topology", "all_to_all"),
                "server_time": now,
            })
    return new_events

# ======================================================================
# Background loop
# ======================================================================

_loop_alive: bool = False
_loop_last_step: float = 0.0
_loop_error: str | None = None
_loop_rate: float = 0.0
_loop_step_count: int = 0
_kuramoto_debug_rows: list[dict] = []
_kuramoto_debug_lock = threading.Lock()
_KURAMOTO_DEBUG_MAX_ROWS = 50000

def _wrap_pi(value: float) -> float:
    return ((float(value) + math.pi) % (2.0 * math.pi)) - math.pi

def _append_kuramoto_debug(row: dict) -> None:
    with _kuramoto_debug_lock:
        _kuramoto_debug_rows.append(row)
        if len(_kuramoto_debug_rows) > _KURAMOTO_DEBUG_MAX_ROWS:
            del _kuramoto_debug_rows[:len(_kuramoto_debug_rows) - _KURAMOTO_DEBUG_MAX_ROWS]

def _clear_kuramoto_debug() -> None:
    with _kuramoto_debug_lock:
        _kuramoto_debug_rows.clear()

def _refresh_agents_snapshot() -> None:
    global _agents_snapshot
    with _agents_lock:
        snap = [_compact_snapshot(a) for a in _agents]
    with _snapshot_lock:
        _agents_snapshot = snap

def _agent_loop() -> None:
    global _loop_running, _loop_start_time, _loop_alive, _agents_snapshot
    global _loop_last_step, _loop_error, _loop_rate, _loop_step_count
    dt = 0.033  # ~30 fps
    _loop_start_time = time.monotonic()
    _loop_alive = True
    _loop_error = None
    _loop_step_count = 0
    while _loop_running:
        try:
            now = time.monotonic() - _loop_start_time
            # Acquire lock briefly to copy pi_flash_times and agent refs
            with _agents_lock:
                pi_copy = list(_pi_flash_times)
                _pi_flash_times.clear()
                event_copy = list(_pending_flash_events)
                _pending_flash_events.clear()
                agents_snapshot = list(_agents)  # shallow copy of agent dicts

            # Step agents (only if running)
            is_running = _leader_config.get("running", False)
            feedback_on = _leader_config.get("feedback_enabled", False)
            flash_dur = _leader_config.get("flash_duration_s", 0.10)
            if _leader_config.get("mutual_agent_mode") == "multi_2v1p":
                new_events = _step_multi_eapf_agents(
                    [a for a in agents_snapshot if a.get("enabled", True)],
                    dt, now, event_copy, is_running, feedback_on, flash_dur,
                )
                if new_events:
                    with _agents_lock:
                        _pending_flash_events.extend(new_events)
            else:
                for a in agents_snapshot:
                    if not a.get("enabled", True):
                        continue
                    # Calibration mode: absolute wall-clock timing, no Pi influence
                    if not feedback_on:
                        _step_calibration_agent(a, now)
                    elif a["model"] == "eapf_consensus":
                        _step_eapf_agent(a, dt, now, pi_copy, is_running,
                                         feedback_on, flash_dur)
                    elif a["model"] == "kuramoto":
                        _step_kuramoto_agent(a, dt, now, pi_copy, is_running)

            _loop_step_count += 1
            _loop_last_step = time.monotonic()
            if _loop_step_count > 0 and _loop_start_time > 0:
                elapsed = _loop_last_step - _loop_start_time
                _loop_rate = _loop_step_count / elapsed if elapsed > 0 else 0.0

            # Update compact snapshot for fast HTTP responses (no large lists)
            with _snapshot_lock:
                _agents_snapshot = [_compact_snapshot(a) for a in agents_snapshot]
        except Exception as e:
            import traceback
            _loop_error = f"{e}\n{traceback.format_exc()}"
            print(f"[LOOP ERROR] {_loop_error}")
        time.sleep(dt)
    _loop_alive = False

def _start_agent_loop() -> None:
    global _loop_running, _loop_thread
    if _loop_running:
        return
    _loop_running = True
    _loop_thread = threading.Thread(target=_agent_loop, daemon=True)
    _loop_thread.start()

def _stop_agent_loop() -> None:
    global _loop_running
    _loop_running = False

# ======================================================================
# HTTP handler
# ======================================================================

class LeaderUIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _json(self, code: int, data) -> None:
        self.send_response(code); self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def end_headers(self) -> None:
        if self.path.endswith((".html", ".js", ".css")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        super().end_headers()

    def _read_body(self) -> dict:
        cl = int(self.headers.get("Content-Length", 0))
        if cl == 0:
            return {}
        try:
            return json.loads(self.rfile.read(cl))
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204); self._cors(); self.end_headers()

    # ---- GET ----
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        # Route /mutual to mutual.html
        if path == "/mutual" or path == "/mutual/":
            self.path = "/mutual.html"
            super().do_GET()
            return
        if path == "/api/status":
            cfg = _get_config_copy()
            cfg["mode"] = _mode
            with _snapshot_lock:
                cfg["agents"] = list(_agents_snapshot)
            if not cfg["agents"]:
                with _agents_lock:
                    cfg["agents"] = [_compact_snapshot(a) for a in _agents]
            cfg["agents_count"] = len(_agents_snapshot)
            cfg["loop_alive"] = _loop_alive
            cfg["loop_error"] = _loop_error
            cfg["loop_rate_hz"] = round(_loop_rate, 2)
            cfg["loop_step_count"] = _loop_step_count
            cfg["last_loop_step_time"] = round(_loop_last_step, 3) if _loop_last_step > 0 else 0
            self._json(200, cfg)
        elif path == "/api/leader/config":
            self._json(200, _get_config_copy())
        elif path == "/api/mode":
            self._json(200, {"mode": _mode})
        elif path == "/api/mutual/config":
            cfg = _get_config_copy()
            topology = _active_hil_topology()
            with _snapshot_lock:
                snap = list(_agents_snapshot)
            self._json(200, {
                "mode": _mode,
                "mutual_agent_mode": cfg.get("mutual_agent_mode", "single_1v1p"),
                "topology": cfg.get("topology", "all_to_all"),
                "topology_adjacency": topology.adjacency,
                "available_topologies": list(MIXED_REALITY_TOPOLOGIES),
                "display_config": {
                    "background_brightness": cfg.get("background_brightness", 0),
                    "brightness_on": cfg.get("brightness_on", 255),
                    "brightness_off": cfg.get("brightness_off", 0),
                    "agents": [
                        {
                            "agent_id": a.get("agent_id"),
                            "role": a.get("role"),
                            "x": a.get("x"),
                            "y": a.get("y"),
                            "size": a.get("size"),
                            "brightness_on": a.get("brightness_on"),
                            "brightness_off": a.get("brightness_off"),
                            "background_brightness": a.get("background_brightness"),
                            "initial_frequency_hz": a.get("initial_frequency_hz"),
                            "initial_phase_rad": a.get("initial_phase_rad"),
                            "enabled": a.get("enabled"),
                        }
                        for a in snap
                    ],
                },
            })
        elif path == "/api/frontend/display_state":
            with _frontend_display_state_lock:
                state = dict(_frontend_display_state)
            self._json(200, state)
        elif path == "/api/stabilizer/config":
            self._json(200, _get_stabilizer_copy())
        elif path == "/api/agents":
            # Return compact snapshot if available; build on-demand otherwise
            with _snapshot_lock:
                snap = list(_agents_snapshot)
            if not snap:
                with _agents_lock:
                    snap = [_compact_snapshot(a) for a in _agents]
            self._json(200, {"agents": snap, "mode": _mode,
                             "loop_alive": _loop_alive, "loop_rate_hz": round(_loop_rate, 2)})
        elif path == "/api/kuramoto_debug":
            with _kuramoto_debug_lock:
                rows = list(_kuramoto_debug_rows)
            self._json(200, {
                "rows": rows,
                "row_count": len(rows),
                "max_rows": _KURAMOTO_DEBUG_MAX_ROWS,
            })
        elif path.startswith("/api/agents/"):
            # GET /api/agents/{id} — return specific agent
            aid_str = path.split("/")[-1]
            try:
                aid = int(aid_str)
                with _agents_lock:
                    if aid < len(_agents):
                        self._json(200, _compact_snapshot(_agents[aid]))
                    else:
                        self._json(404, {"error": "agent not found"})
            except ValueError:
                self._json(400, {"error": "invalid id"})
        else:
            super().do_GET()

    # ---- POST ----
    def do_POST(self) -> None:
        global _mode, _pi_flash_times
        path = urlparse(self.path).path
        body = self._read_body()

        if path == "/api/leader/config":
            cfg = _update_config(body)
            _refresh_agents_snapshot()
            print(f"[API] Config: {json.dumps(body)}")
            self._json(200, cfg)

        elif path == "/api/mode":
            global _mode
            new_mode = body.get("mode", "fixed_leader")
            if new_mode not in ("fixed_leader", "mutual_hil", "mutual_hil_multi"):
                self._json(400, {"error": "invalid mode"})
                return
            _mode = "mutual_hil" if new_mode == "mutual_hil_multi" else new_mode
            print(f"[API] Mode → {_mode}")
            # Init agents if switching to mutual
            if _mode == "mutual_hil":
                agent_mode = (
                    "multi_2v1p"
                    if new_mode == "mutual_hil_multi"
                    else body.get("agent_mode", "single_1v1p")
                )
                try:
                    _configure_mutual_agents(
                        agent_mode,
                        body.get("topology", _leader_config.get("topology", "all_to_all")),
                    )
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                    return
                vc = _get_config_copy()
                vc["api_controlled"] = True
                _update_config(vc)
            _refresh_agents_snapshot()
            self._json(200, {"mode": _mode})

        elif path == "/api/mutual/config":
            agent_mode = body.get(
                "mutual_agent_mode",
                body.get("agent_mode", _leader_config.get("mutual_agent_mode", "single_1v1p")),
            )
            topology_name = body.get("topology", _leader_config.get("topology", "all_to_all"))
            try:
                cfg = _configure_mutual_agents(agent_mode, topology_name)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            topology = _active_hil_topology()
            self._json(200, {
                "mutual_agent_mode": cfg.get("mutual_agent_mode"),
                "topology": cfg.get("topology"),
                "topology_adjacency": topology.adjacency,
                "display_config": _get_config_copy(),
            })

        elif path == "/api/frontend/display_state":
            with _frontend_display_state_lock:
                _frontend_display_state.clear()
                _frontend_display_state.update(body)
                _frontend_display_state["server_received_wall_time_s"] = time.time()
            self._json(200, {"ok": True})

        elif path == "/api/stabilizer/config":
            cfg = _update_stabilizer(body)
            print(f"[API] Stabilizer: {json.dumps(body)}")
            self._json(200, cfg)

        elif path == "/api/agents":
            aid = body.get("id", len(_agents))
            initial_phase = body.get("initial_phase_rad", body.get("phase_rad", 0.0))
            ag = _init_agent(aid, freq=body.get("initial_frequency_hz", 2.0),
                             model=body.get("model", "eapf_consensus"),
                             x=body.get("x", 0), y=body.get("y", 0),
                             size=body.get("size", 200),
                             initial_phase_rad=initial_phase,
                             agent_id=body.get("agent_id"),
                             role=body.get("role", "virtual"),
                             brightness_on=body.get("brightness_on"),
                             brightness_off=body.get("brightness_off"),
                             background_brightness=body.get("background_brightness"))
            with _agents_lock:
                _agents.append(ag)
            _refresh_agents_snapshot()
            print(f"[API] Agent {aid} added: {body}")
            self._json(200, _compact_snapshot(ag))

        elif path.startswith("/api/agents/"):
            aid_str = path.split("/")[-1]
            try:
                aid = int(aid_str)
            except ValueError:
                self._json(400, {"error": "invalid agent id"})
                return
            with _agents_lock:
                if aid >= len(_agents):
                    self._json(404, {"error": "agent not found"})
                    return
                for k in (
                    "x", "y", "size", "initial_frequency_hz", "frequency_hz",
                    "model", "enabled", "kuramoto_gain", "agent_id", "role",
                    "brightness_on", "brightness_off", "background_brightness",
                ):
                    if k in body:
                        _agents[aid][k] = body[k]
                if "initial_phase_rad" in body:
                    phase = _wrap_2pi(body["initial_phase_rad"])
                    _agents[aid]["initial_phase_rad"] = phase
                    _agents[aid]["phase"] = phase
                    _agents[aid]["phase_rad"] = phase
                if "phase_rad" in body:
                    phase = _wrap_2pi(body["phase_rad"])
                    _agents[aid]["phase"] = phase
                    _agents[aid]["phase_rad"] = phase
                    if "initial_phase_rad" not in body:
                        _agents[aid]["initial_phase_rad"] = phase
                if "initial_frequency_hz" in body:
                    freq = _agents[aid].get("initial_frequency_hz", 2.0)
                    _agents[aid]["neighbour_period_est"] = 1.0 / freq if freq > 0 else 0.5
                    _agents[aid]["oscillator"] = None
            _refresh_agents_snapshot()
            self._json(200, _compact_snapshot(_agents[aid]))

        elif path == "/api/start":
            global _calibration_start_time
            _clear_kuramoto_debug()
            _start_agent_loop()
            vc = _get_config_copy(); vc["running"] = True; vc["api_controlled"] = True
            _update_config(vc)
            _calibration_start_time = time.monotonic()
            with _agents_lock:
                _pi_flash_times.clear()
                _pending_flash_events.clear()
                # Reset fire tracking for calibration
                for a in _agents:
                    a["fire_count"] = 0
                    a["flash_times"] = []
                    a["flash_on"] = False
                    a["last_flash_time"] = 0.0
                    a["received_neighbour_events"] = 0
                    a["oscillator"] = None
            print("[API] Agents started — stale Pi events cleared, calibration start recorded")
            _refresh_agents_snapshot()
            self._json(200, {"running": True})

        elif path == "/api/pause":
            _stop_agent_loop()
            vc = _get_config_copy(); vc["running"] = False
            _update_config(vc)
            print("[API] Agents paused")
            _refresh_agents_snapshot()
            self._json(200, {"running": False})

        elif path == "/api/reset":
            _stop_agent_loop()
            _clear_kuramoto_debug()
            vc = _get_config_copy(); vc["running"] = False; vc["feedback_enabled"] = False
            _update_config(vc)
            _update_stabilizer({"enabled": False, "state": "acquisition", "f_lock_hz": None})
            with _agents_lock:
                _pi_flash_times.clear()
                _pending_flash_events.clear()
                for a in _agents:
                    _reset_agent(a)
            print("[API] Agents reset — Pi events cleared, feedback disabled")
            _refresh_agents_snapshot()
            self._json(200, {"reset": True})

        elif path == "/api/pi_flash":
            ts = body.get("timestamp", time.monotonic())
            source_agent_id = body.get("agent_id", body.get("source_agent_id", "P0"))
            # Always log Pi flashes (for counting), but only queue if feedback enabled
            with _agents_lock:
                if _leader_config.get("feedback_enabled", False):
                    if _leader_config.get("mutual_agent_mode") == "multi_2v1p":
                        _pending_flash_events.append({
                            "timestamp": ts,
                            "source_agent_id": source_agent_id,
                            "agent_id": source_agent_id,
                            "event_type": "pi_flash",
                            "server_received_time": time.monotonic() - _loop_start_time if _loop_start_time > 0 else 0.0,
                        })
                    else:
                        _pi_flash_times.append(ts)
                # Count POST receipt separately from later agent-loop consumption.
                if _agents:
                    for agent in _agents:
                        if agent.get("role") == "virtual":
                            agent["pi_flash_posts_received"] = agent.get("pi_flash_posts_received", 0) + 1
                            agent["received_pi_flashes"] = agent["pi_flash_posts_received"]
                    for agent in _agents:
                        if agent.get("agent_id") == source_agent_id:
                            agent["fire_count"] = agent.get("fire_count", 0) + 1
                            agent["last_flash_time"] = ts
                            agent.setdefault("flash_times", []).append(ts)
            print(f"[API] Pi flash at t={ts:.3f} (feedback={_leader_config.get('feedback_enabled', False)})")
            self._json(200, {"received": True, "timestamp": ts,
                             "agent_id": source_agent_id,
                             "feedback_enabled": _leader_config.get("feedback_enabled", False)})

        elif path == "/api/feedback":
            enabled = body.get("enabled", False)
            vc = _get_config_copy(); vc["feedback_enabled"] = enabled
            _update_config(vc)
            print(f"[API] Feedback → {enabled}")
            _refresh_agents_snapshot()
            self._json(200, {"feedback_enabled": enabled})

        else:
            self._json(404, {"error": "not found"})

# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Firefly Leader UI — fixed_leader + mutual_hil modes.")
    parser.add_argument("--host", type=str, default=HOST_DEFAULT)
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not ROOT.is_dir():
        print(f"ERROR: leader_ui directory not found at {ROOT}")
        sys.exit(1)

    try:
        httpd = ThreadingHTTPServer((args.host, args.port), LeaderUIHandler)
    except socket.error as exc:
        print(f"ERROR: cannot bind to {args.host}:{args.port} — {exc}")
        sys.exit(1)

    url = f"http://{args.host}:{args.port}"
    print("=" * 60)
    print("  Firefly Leader UI — fixed_leader + mutual_hil")
    print("=" * 60)
    print(f"  URL:     {url}")
    print(f"  Mode:    fixed_leader (POST /api/mode to switch)")
    print(f"  API:     {url}/api/status")
    print(f"           {url}/api/pi_flash  [POST]")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()

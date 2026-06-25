#!/usr/bin/env python3
r"""Step 3A-3 — Automated Pi Visual Kuramoto Batch Testing.

Laptop screen displays the leader dot; the Raspberry Pi 5 camera observes
it, runs a Kuramoto follower oscillator, and drives GPIO17 as the
follower LED.

Key fixes (June 2026):
- Trial duration uses **wall-clock** ``time.monotonic()`` gating, not
  fixed-step iteration count.  Camera capture is blocking, so the old
  ``t += dt`` loop could run 3-7× longer than requested.
- GPIO17 LED is now actually driven during the trial on follower flashes.

Usage (dry-run)::

    PYTHONPATH=. python experiments/run_step3a_pi_visual_batch.py \
        --duration 5 --repeats 1 --follower-freqs 1.5 --dry-run

Usage (real)::

    PYTHONPATH=. python3 experiments/run_step3a_pi_visual_batch.py \
        --leader-api http://<laptop-ip>:8000 \
        --duration 10 --repeats 1 --leader-freqs 2.0 --follower-freqs 1.5 \
        --coupling-gain 3.5 --leader-shape circle --leader-dot-size 350
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from firefly_sync.core.kuramoto import KuramotoModel
from firefly_sync.logging.metrics import (
    check_flash_synchronisation,
    compute_flash_timing_metrics,
    pair_flash_events,
)

# ======================================================================
# Pi hardware — lazy imports
# ======================================================================

_PiGPIOLED: Any = None
_PicameraFlashDetector: Any = None


def _ensure_pi_hardware(dry_run: bool = False) -> None:
    global _PiGPIOLED, _PicameraFlashDetector
    if dry_run:
        return
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED as _PL
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector as _PFD,
        )
        _PiGPIOLED = _PL
        _PicameraFlashDetector = _PFD
    except ImportError as exc:
        print(f"ERROR: {exc}")
        print("Install: sudo apt install -y python3-gpiozero python3-picamera2")
        print("Or use --dry-run for API scheduling test without hardware.")
        sys.exit(1)


def pinattr(led: Any) -> int:
    """Safely get LED pin number for logging."""
    return getattr(led, '_pin', -1)


# ======================================================================
# Leader Phase Estimator
# ======================================================================

class LeaderPhaseEstimator:
    """Estimate leader phase from detected flash timestamps."""

    def __init__(self, initial_period_guess_s: float = 0.5,
                 max_stored_flashes: int = 30) -> None:
        self._initial_period_guess_s = float(initial_period_guess_s)
        self._max_stored = max_stored_flashes
        self._flash_times: list[float] = []
        self._estimated_period_s: float = initial_period_guess_s
        self._bootstrap: bool = True

    @property
    def estimated_period_s(self) -> float:
        return self._estimated_period_s

    @property
    def flash_count(self) -> int:
        return len(self._flash_times)

    @property
    def is_bootstrapping(self) -> bool:
        return self._bootstrap

    @property
    def last_flash_time(self) -> float | None:
        return self._flash_times[-1] if self._flash_times else None

    def record_flash(self, timestamp_s: float) -> None:
        self._flash_times.append(timestamp_s)
        if len(self._flash_times) > self._max_stored:
            self._flash_times.pop(0)
        self._update_period_estimate()

    def estimate_phase(self, now_s: float) -> float:
        last = self.last_flash_time
        if last is None:
            return 0.0
        if self._estimated_period_s <= 0:
            return 0.0
        phase = 2.0 * np.pi * ((now_s - last) / self._estimated_period_s)
        return float(phase % (2.0 * np.pi))

    def reset(self) -> None:
        self._flash_times.clear()
        self._estimated_period_s = self._initial_period_guess_s
        self._bootstrap = True

    def _update_period_estimate(self) -> None:
        if len(self._flash_times) < 2:
            return
        intervals = [self._flash_times[i + 1] - self._flash_times[i]
                     for i in range(len(self._flash_times) - 1)]
        self._estimated_period_s = float(np.median(intervals[-10:]))
        self._bootstrap = False


# ======================================================================
# Leader UI HTTP client
# ======================================================================

def _api_get(api_base: str, path: str) -> dict:
    import urllib.request
    url = api_base.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _api_post(api_base: str, path: str, data: dict) -> dict:
    import urllib.request
    url = api_base.rstrip("/") + path
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _set_leader_config(api_base: str, **kwargs) -> dict:
    return _api_post(api_base, "/api/leader/config", kwargs)


def _get_leader_status(api_base: str) -> dict:
    return _api_get(api_base, "/api/status")


# ======================================================================
# LED status blink helpers
# ======================================================================

def _led_blinks(led: Any, n: int, on_s: float, off_s: float) -> None:
    for _ in range(n):
        led.on()
        time.sleep(on_s)
        led.off()
        time.sleep(off_s)


# ======================================================================
# CPU / memory helpers
# ======================================================================

def _get_cpu_temp() -> float | None:
    for p in ["/sys/class/thermal/thermal_zone0/temp",
              "/sys/class/hwmon/hwmon0/temp1_input"]:
        try:
            with open(p) as f:
                return float(f.read().strip()) / 1000.0
        except (FileNotFoundError, OSError, ValueError):
            continue
    return None


def _get_memory_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, AttributeError):
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# ======================================================================
# Output helpers
# ======================================================================

def _make_output_dir(root: str, trial_prefix: str | None) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{trial_prefix}_" if trial_prefix else ""
    name = f"{ts}_{prefix}kuramoto_pi_visual_batch"
    out = Path(root) / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================
# Computational cost metrics
# ======================================================================

def _compute_cost_metrics(
    loop_dts_s: list[float],
    model_times_ms: list[float],
    camera_times_ms: list[float],
    cpu_temps: list[float],
    mem_mbs: list[float],
    process_start: float,
    sync_achieved: bool,
    final_t: float,
) -> dict:
    arr_loop = np.array(loop_dts_s) * 1000.0
    arr_model = np.array(model_times_ms)
    arr_cam = np.array(camera_times_ms)
    arr_mem = np.array(mem_mbs)
    arr_temp = np.array(cpu_temps)

    return {
        "mean_loop_dt_ms": round(float(np.mean(arr_loop)), 3) if len(arr_loop) > 0 else 0,
        "p95_loop_dt_ms": round(float(np.percentile(arr_loop, 95)), 3) if len(arr_loop) > 0 else 0,
        "effective_loop_rate_hz": round(1000.0 / float(np.mean(arr_loop)), 2) if len(arr_loop) > 0 and np.mean(arr_loop) > 0 else 0,
        "mean_model_update_time_ms": round(float(np.mean(arr_model)), 3) if len(arr_model) > 0 else 0,
        "p95_model_update_time_ms": round(float(np.percentile(arr_model, 95)), 3) if len(arr_model) > 0 else 0,
        "max_model_update_time_ms": round(float(np.max(arr_model)), 3) if len(arr_model) > 0 else 0,
        "mean_camera_processing_time_ms": round(float(np.mean(arr_cam)), 2) if len(arr_cam) > 0 else 0,
        "p95_camera_processing_time_ms": round(float(np.percentile(arr_cam, 95)), 2) if len(arr_cam) > 0 else 0,
        "process_cpu_time_total_s": round(time.process_time() - process_start, 4),
        "mean_cpu_percent": 0.0,
        "peak_memory_rss_mb": round(float(np.max(arr_mem)), 2) if len(arr_mem) > 0 else 0,
        "mean_cpu_temperature_c": round(float(np.mean(arr_temp)), 1) if len(arr_temp) > 0 else None,
        "max_cpu_temperature_c": round(float(np.max(arr_temp)), 1) if len(arr_temp) > 0 else None,
        "dropped_frame_count": 0,
        "model_update_time_to_sync_ms": round(float(np.sum(arr_model)), 3) if arr_model.size > 0 else 0,
    }


# ======================================================================
# Single Pi visual trial (wall-clock gated)
# ======================================================================

def _run_pi_trial(
    api_base: str,
    leader_freq: float,
    follower_freq: float,
    coupling_gain: float,
    duration_s: float,
    dt: float,
    flash_on_time_s: float,
    sync_threshold_s: float,
    sync_cycles: int,
    random_delay: float,
    trial_id: str,
    out_dir: Path,
    dry_run: bool,
    detector_kwargs: dict,
    leader_min_flash_interval_s: float = 0.20,
    led: Any = None,
    target_loop_rate_hz: float = 30.0,
) -> dict:
    """Run one Pi visual closed-loop trial.  Returns metrics dict."""

    led_enabled = (led is not None)
    print(f"\n  [trial:{trial_id}] delay={random_delay:.1f}s  "
          f"f_leader={leader_freq}  f_follower={follower_freq}  K={coupling_gain}")
    if led_enabled:
        print(f"  Follower LED enabled on GPIO{pinattr(led)}: True")

    # -- metadata --
    k_critical = 2.0 * np.pi * abs(leader_freq - follower_freq)
    k_ratio = coupling_gain / k_critical if k_critical > 0 else float("inf")

    metadata = {
        "trial_id": trial_id, "mode": "pi_visual", "model_name": "kuramoto",
        "leader_api_url": api_base, "leader_freq_hz": leader_freq,
        "follower_initial_freq_hz": follower_freq, "coupling_gain": coupling_gain,
        "k_critical_rad_s": round(k_critical, 4),
        "k_ratio": round(k_ratio, 4) if k_ratio != float("inf") else "inf",
        "random_start_delay_s": random_delay,
        "requested_duration_s": duration_s, "dt": dt,
        "sync_threshold_s": sync_threshold_s, "sync_cycles": sync_cycles,
        "follower_led_enabled": led_enabled,
        "follower_led_pin": pinattr(led) if led_enabled else None,
        "flash_on_time_s": flash_on_time_s,
        "target_loop_rate_hz": target_loop_rate_hz,
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
    }

    # -- phase estimator --
    initial_guess = 1.0 / leader_freq if leader_freq > 0 else 0.5
    estimator = LeaderPhaseEstimator(initial_period_guess_s=initial_guess)

    # -- follower oscillator --
    follower = KuramotoModel(
        natural_frequency=2.0 * np.pi * follower_freq,
        initial_phase=0.0, coupling_strength=coupling_gain, dt=dt,
    )

    # -- hardware (or dry-run stubs) --
    detector = None
    if not dry_run:
        assert _PicameraFlashDetector is not None
        detector = _PicameraFlashDetector(**detector_kwargs)
        detector.start()

    # -- random start delay --
    if random_delay > 0:
        print(f"    Waiting {random_delay:.1f}s before trial start...")
        time.sleep(random_delay)

    # -- wall-clock gated loop --
    step = 0
    follower_flash_times: list[float] = []
    detected_leader_times: list[float] = []
    oscillator_log: list[dict] = []
    detection_log: list[dict] = []
    flash_events: list[dict] = []
    sync_achieved = False
    loop_dts_s: list[float] = []
    model_times_ms: list[float] = []
    camera_times_ms: list[float] = []
    cpu_temps: list[float] = []
    mem_mbs: list[float] = []
    process_start = time.process_time()
    trial_start_wall = time.monotonic()
    prev_loop = time.monotonic()
    # Rising-edge detection state
    prev_detected_state: bool = False
    last_leader_event_time: float = -999.0
    # LED state tracking
    led_on_until: float = 0.0  # monotonic timestamp when LED should turn off

    target_loop_dt = 1.0 / target_loop_rate_hz if target_loop_rate_hz > 0 else 0.0

    try:
        while (time.monotonic() - trial_start_wall) < duration_s:
            now = time.monotonic()
            measured_dt = now - prev_loop
            prev_loop = now
            loop_dts_s.append(measured_dt)
            t = now - trial_start_wall  # simulation time ≈ wall time

            # --- leader detection (wall-clock gated) ---
            cam_ms = 0.0
            if not dry_run and detector is not None:
                cam_start = time.monotonic()
                result = detector.capture_frame()
                cam_ms = (time.monotonic() - cam_start) * 1000.0

                # Rising-edge detection on detected state (ON/OFF), not raw event
                detected_state = (result.get("state") == "ON")
                leader_flash_event = False
                if detected_state and not prev_detected_state:
                    if (t - last_leader_event_time) >= leader_min_flash_interval_s:
                        leader_flash_event = True
                        last_leader_event_time = t
                prev_detected_state = detected_state

                roi = result.get("roi") or {}

                detection_log.append({
                    "t": round(t, 6), "frame_id": result.get("frame_index", step),
                    "roi_intensity": result.get("brightness_used", 0),
                    "normalized_signal": result.get("signal_norm", 0),
                    "detected_flash_state": 1 if detected_state else 0,
                    "leader_flash_event": 1 if leader_flash_event else 0,
                    "detected_flash_time": round(t, 6) if leader_flash_event else "",
                    "detection_confidence": result.get("periodicity_confidence", 0),
                    "roi_x": roi.get("x", ""), "roi_y": roi.get("y", ""),
                    "roi_w": roi.get("width", ""), "roi_h": roi.get("height", ""),
                    "camera_processing_time_ms": round(cam_ms, 2),
                })

                if leader_flash_event:
                    estimator.record_flash(t)
                    detected_leader_times.append(t)
                    flash_events.append({
                        "t": round(t, 6), "event_type": "leader_flash",
                        "source": "camera",
                        "detected_leader_flash_time": round(t, 6),
                        "follower_flash_time": None, "timing_error_s": None,
                        "wrapped_timing_error_s": None,
                    })
            else:
                detection_log.append({
                    "t": round(t, 6), "frame_id": step,
                    "roi_intensity": 0, "normalized_signal": 0,
                    "detected_flash_state": 0, "leader_flash_event": 0,
                    "detected_flash_time": "", "detection_confidence": 0,
                    "roi_x": "", "roi_y": "", "roi_w": "", "roi_h": "",
                    "camera_processing_time_ms": 0.0,
                })

            camera_times_ms.append(cam_ms)

            # --- Kuramoto coupling (use measured dt for oscillator update) ---
            est_phase = estimator.estimate_phase(t)
            f_phase = follower.phase
            phase_error = float(np.arctan2(
                np.sin(est_phase - f_phase),
                np.cos(est_phase - f_phase),
            ))
            coupling_input = np.sin(est_phase - f_phase)

            # Use measured_dt for the oscillator step (capped to avoid huge jumps)
            effective_dt = min(measured_dt, 0.1)  # cap at 100ms
            # Temporarily set follower.dt to measured dt for this step
            saved_dt = follower.dt
            follower.dt = effective_dt

            model_start = time.monotonic()
            state = follower.step(coupling_input)
            model_ms = (time.monotonic() - model_start) * 1000.0
            model_times_ms.append(model_ms)

            follower.dt = saved_dt  # restore original dt

            # --- oscillator log ---
            temp_c = _get_cpu_temp()
            mem_mb = _get_memory_rss_mb()
            if temp_c is not None:
                cpu_temps.append(temp_c)
            mem_mbs.append(mem_mb)

            # LED off if flash duration expired
            if led_enabled and t >= led_on_until:
                led.off()

            oscillator_log.append({
                "t": round(t, 6),
                "follower_phase_rad": round(f_phase, 6),
                "follower_frequency_hz": round(follower.natural_frequency / (2.0 * np.pi), 6),
                "estimated_leader_phase_rad": round(est_phase, 6),
                "phase_error_rad": round(phase_error, 6),
                "coupling_term": round(coupling_input, 6),
                "follower_led_state": 1 if state.is_firing else 0,
                "sync_state": 1 if sync_achieved else 0,
                "loop_dt_ms": round(measured_dt * 1000, 3),
                "model_update_time_ms": round(model_ms, 3),
                "cpu_percent": 0.0,
                "process_cpu_time_s": round(time.process_time() - process_start, 4),
                "memory_rss_mb": round(mem_mb, 2),
                "cpu_temperature_c": round(temp_c, 1) if temp_c is not None else "",
            })

            # --- follower flash → drive LED ---
            if state.is_firing:
                follower_flash_times.append(t)

                # Drive real GPIO LED
                if led_enabled:
                    led.on()
                    led_on_until = t + flash_on_time_s

                pair_info = pair_flash_events(detected_leader_times, [t])
                timing_err = pair_info[0]["timing_error_s"] if pair_info else None
                flash_events.append({
                    "t": round(t, 6), "event_type": "follower_flash",
                    "source": "follower_oscillator",
                    "detected_leader_flash_time": pair_info[0].get("leader_t") if pair_info else None,
                    "follower_flash_time": round(t, 6),
                    "timing_error_s": timing_err,
                    "wrapped_timing_error_s": None,
                })

                # Check sync criterion
                sync_check = check_flash_synchronisation(
                    detected_leader_times, follower_flash_times,
                    sync_threshold_s, sync_cycles,
                )
                if sync_check["synchronization_success"] and not sync_achieved:
                    sync_achieved = True
                    flash_events.append({
                        "t": round(t, 6), "event_type": "sync_achieved",
                        "source": "metrics",
                        "detected_leader_flash_time": None,
                        "follower_flash_time": None,
                        "timing_error_s": None, "wrapped_timing_error_s": None,
                    })
                    print(f"    [OK] Sync at t={t:.3f}s")

            # --- rate limiting (sleep spare time) ---
            loop_elapsed = time.monotonic() - now
            if target_loop_dt > 0 and loop_elapsed < target_loop_dt:
                remaining = target_loop_dt - loop_elapsed
                if remaining > 0:
                    time.sleep(remaining)

            step += 1

    except KeyboardInterrupt:
        print("\n    Trial interrupted.")

    # -- stop camera; turn LED off --
    if not dry_run and detector is not None:
        detector.stop()
    if led_enabled:
        led.off()

    # -- wall-duration accounting --
    actual_wall = time.monotonic() - trial_start_wall
    duration_error = actual_wall - duration_s
    wall_ratio = actual_wall / duration_s if duration_s > 0 else 1.0

    # -- sanity --
    expected_flash_requested = duration_s * leader_freq
    expected_flash_wall = actual_wall * leader_freq
    detected_count = len(detected_leader_times)
    ratio_requested = detected_count / expected_flash_requested if expected_flash_requested > 0 else 0.0

    if actual_wall > duration_s * 1.2:
        print(f"    [WARN] Trial exceeded requested wall duration "
              f"({actual_wall:.1f}s vs {duration_s:.1f}s); camera loop may be blocking.")

    if ratio_requested > 2.0:
        print(f"    [WARN] Leader flash count ratio {ratio_requested:.1f} > 2.0 "
              f"(detected={detected_count}, expected~{expected_flash_requested:.0f})")
    elif ratio_requested < 0.5:
        print(f"    [WARN] Leader flash count ratio {ratio_requested:.1f} < 0.5 "
              f"(detected={detected_count}, expected~{expected_flash_requested:.0f})")

    # -- metrics --
    metrics = compute_flash_timing_metrics(
        leader_times=detected_leader_times,
        follower_times=follower_flash_times,
        sync_threshold_s=sync_threshold_s, sync_cycles=sync_cycles,
        detection_success_rate=None, false_positive_rate=None,
    )
    comp_cost = _compute_cost_metrics(
        loop_dts_s, model_times_ms, camera_times_ms, cpu_temps, mem_mbs,
        process_start, sync_achieved, actual_wall,
    )
    metrics.update(comp_cost)
    metrics["expected_leader_flash_count"] = round(expected_flash_requested, 1)
    metrics["detected_leader_flash_count"] = detected_count
    metrics["leader_flash_count_ratio"] = round(ratio_requested, 3)
    metrics["requested_duration_s"] = duration_s
    metrics["actual_trial_wall_duration_s"] = round(actual_wall, 3)
    metrics["trial_duration_error_s"] = round(duration_error, 3)
    metrics["wall_time_ratio"] = round(wall_ratio, 4)

    # -- update metadata with wall-time info --
    metadata["actual_trial_wall_duration_s"] = round(actual_wall, 3)
    metadata["trial_duration_error_s"] = round(duration_error, 3)
    metadata["wall_time_ratio"] = round(wall_ratio, 4)
    metadata["expected_leader_flash_count_requested"] = round(expected_flash_requested, 1)
    metadata["expected_leader_flash_count_wall"] = round(expected_flash_wall, 1)
    metadata["detected_leader_flash_count"] = detected_count
    metadata["leader_flash_count_ratio"] = round(ratio_requested, 3)
    _save_json(out_dir / "metadata.json", metadata)

    # -- save logs --
    _save_csv(out_dir / "oscillator_log.csv", oscillator_log,
              ["t", "follower_phase_rad", "follower_frequency_hz",
               "estimated_leader_phase_rad", "phase_error_rad",
               "coupling_term", "follower_led_state", "sync_state",
               "loop_dt_ms", "model_update_time_ms", "cpu_percent",
               "process_cpu_time_s", "memory_rss_mb", "cpu_temperature_c"])

    _save_csv(out_dir / "detection_log.csv", detection_log,
              ["t", "frame_id", "roi_intensity", "normalized_signal",
               "detected_flash_state", "leader_flash_event", "detected_flash_time",
               "detection_confidence",
               "roi_x", "roi_y", "roi_w", "roi_h", "camera_processing_time_ms"])

    _save_csv(out_dir / "flash_events.csv", flash_events,
              ["t", "event_type", "source", "detected_leader_flash_time",
               "follower_flash_time", "timing_error_s", "wrapped_timing_error_s"])

    _save_json(out_dir / "metrics_summary.json", metrics)

    # -- print trial summary --
    mean_loop = (np.mean(loop_dts_s) * 1000) if loop_dts_s else 0
    effective_hz = 1000.0 / mean_loop if mean_loop > 0 else 0
    print(f"    requested_dur={duration_s:.1f}s  "
          f"actual_wall={actual_wall:.1f}s  "
          f"wall_ratio={wall_ratio:.2f}x")
    print(f"    leader_flashes(detected)={detected_count}  "
          f"expected(req)~{expected_flash_requested:.0f}  "
          f"expected(wall)~{expected_flash_wall:.0f}")
    print(f"    follower_flashes={len(follower_flash_times)}  "
          f"sync={metrics['synchronization_success']}")
    print(f"    mean_loop_dt={mean_loop:.1f}ms  "
          f"effective_rate={effective_hz:.1f}Hz")

    return metrics


# ======================================================================
# Batch runner
# ======================================================================

def run_visual_batch(args: argparse.Namespace) -> Path:
    _ensure_pi_hardware(dry_run=args.dry_run)

    batch_dir = _make_output_dir(args.log_dir, args.trial_prefix)
    trials_dir = batch_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    api_base = args.leader_api.rstrip("/")
    follower_freqs: list[float] = args.follower_freqs
    repeats: int = args.repeats
    leader_freqs: list[float] = args.leader_freqs

    # -- batch metadata --
    batch_meta = {
        "model_name": "kuramoto", "mode": "pi_visual",
        "leader_api_url": api_base,
        "leader_freqs_hz": leader_freqs,
        "follower_initial_freqs_hz": follower_freqs,
        "coupling_gain": args.coupling_gain,
        "duration_s": args.duration, "dt": args.dt,
        "repeats_per_condition": repeats,
        "random_delay_min": args.random_delay_min,
        "random_delay_max": args.random_delay_max,
        "sync_threshold_s": args.sync_threshold_s, "sync_cycles": args.sync_cycles,
        "dry_run": args.dry_run,
        "timestamp": datetime.now().isoformat(),
        "trial_prefix": args.trial_prefix or "",
        "leader_shape": args.leader_shape,
        "leader_dot_size_px": args.leader_dot_size,
        "target_loop_rate_hz": args.target_loop_rate_hz,
        "flash_on_time_s": args.flash_on_time,
    }
    _save_json(batch_dir / "batch_metadata.json", batch_meta)

    # -- initialise LED (shared across trials) --
    if not args.dry_run:
        led = _PiGPIOLED(pin=args.led_pin, flash_duration_s=args.flash_on_time)
        print(f"\n[LED] GPIO{args.led_pin} initialised for follower output. "
              f"Flash duration: {args.flash_on_time}s")
    else:
        led = None

    # -- batch start: 3 fast blinks --
    if led is not None:
        print("\n[LED] Batch start — 3 fast blinks")
        _led_blinks(led, 3, 0.15, 0.15)

    # -- Leader UI init: stop → config → start --
    if not args.dry_run:
        print(f"\n[API] Connecting to leader at {api_base}")
        _set_leader_config(api_base, running=False)
        time.sleep(0.5)
        _set_leader_config(api_base,
                           frequency_hz=args.leader_freqs[0],
                           duty_cycle=0.5,
                           brightness_on=255, brightness_off=0,
                           background_brightness=0,
                           shape=args.leader_shape,
                           target_size_px=args.leader_dot_size,
                           running=False)
        try:
            status = _get_leader_status(api_base)
            print(f"  Leader status: freq={status.get('frequency_hz')}Hz  "
                  f"running={status.get('running')}  shape={status.get('shape')}  "
                  f"size={status.get('target_size_px')}px  "
                  f"api_controlled={status.get('api_controlled')}")
        except Exception:
            print("  [WARN] Could not read leader status")
    else:
        print(f"\n[DRY-RUN] Would connect to leader API at {api_base}")

    # -- build detector kwargs --
    detector_kwargs = {
        "resolution": [args.width, args.height],
        "detection_mode": args.detection_mode,
        "threshold_on": args.threshold_on,
        "threshold_off": args.threshold_off,
        "min_interval_s": args.min_interval,
        "window_s": args.window_s,
    }

    # -- run trials --
    aggregate_rows: list[dict] = []
    total = len(leader_freqs) * len(follower_freqs) * repeats
    count = 0

    for lf in leader_freqs:
        print(f"\n{'='*60}")
        if not args.dry_run:
            print(f"[API] Setting leader to {lf} Hz")
            _set_leader_config(api_base,
                               frequency_hz=lf,
                               duty_cycle=0.5,
                               brightness_on=255, brightness_off=0,
                               background_brightness=0,
                               shape=args.leader_shape,
                               target_size_px=args.leader_dot_size,
                               running=True)
            time.sleep(2.0)
            try:
                st = _get_leader_status(api_base)
                print(f"  Status: freq={st.get('frequency_hz')}Hz  "
                      f"running={st.get('running')}  shape={st.get('shape')}  "
                      f"size={st.get('target_size_px')}px")
            except Exception:
                pass
        else:
            print(f"[DRY-RUN] Would set leader freq to {lf} Hz")
        print(f"{'='*60}")

        for freq in follower_freqs:
            for rep in range(1, repeats + 1):
                count += 1
                trial_id = f"L{lf:.1f}Hz_F{freq:.1f}Hz_r{rep:02d}"
                print(f"\n{'='*60}")
                print(f"[batch] Trial {count}/{total}: {trial_id}")
                print(f"{'='*60}")

                out_dir = trials_dir / trial_id
                out_dir.mkdir(parents=True, exist_ok=True)

                random_delay = random.uniform(args.random_delay_min,
                                              args.random_delay_max)

                metrics = _run_pi_trial(
                    api_base=api_base,
                    leader_freq=lf, follower_freq=freq,
                    coupling_gain=args.coupling_gain,
                    duration_s=args.duration, dt=args.dt,
                    flash_on_time_s=args.flash_on_time,
                    sync_threshold_s=args.sync_threshold_s,
                    sync_cycles=args.sync_cycles,
                    random_delay=random_delay,
                    trial_id=trial_id, out_dir=out_dir,
                    dry_run=args.dry_run,
                    detector_kwargs=detector_kwargs,
                    leader_min_flash_interval_s=args.leader_min_flash_interval_s,
                    led=led,
                    target_loop_rate_hz=args.target_loop_rate_hz,
                )

                row = {
                    "batch_id": batch_dir.name, "trial_id": trial_id,
                    "model_name": "kuramoto", "leader_freq_hz": lf,
                    "follower_initial_freq_hz": freq,
                    "coupling_gain": args.coupling_gain,
                    "duration_s": args.duration,
                    "synchronization_success": metrics["synchronization_success"],
                    "time_to_synchronization_s": metrics.get("time_to_synchronization_s"),
                    "steady_state_mean_abs_timing_error_s": metrics["steady_state_mean_abs_timing_error_s"],
                    "steady_state_rmse_timing_error_s": metrics["steady_state_rmse_timing_error_s"],
                    "steady_state_jitter_s": metrics["steady_state_jitter_s"],
                    "final_frequency_error_hz": metrics["final_frequency_error_hz"],
                    "convergence_quality": metrics["convergence_quality"],
                    "detection_success_rate": metrics.get("detection_success_rate"),
                    "false_positive_rate": metrics.get("false_positive_rate"),
                    "mean_loop_dt_ms": metrics.get("mean_loop_dt_ms"),
                    "peak_memory_rss_mb": metrics.get("peak_memory_rss_mb"),
                    "mean_cpu_temperature_c": metrics.get("mean_cpu_temperature_c"),
                    "expected_leader_flash_count": metrics.get("expected_leader_flash_count"),
                    "detected_leader_flash_count": metrics.get("detected_leader_flash_count"),
                    "leader_flash_count_ratio": metrics.get("leader_flash_count_ratio"),
                    "requested_duration_s": metrics.get("requested_duration_s"),
                    "actual_trial_wall_duration_s": metrics.get("actual_trial_wall_duration_s"),
                    "trial_log_dir": str(out_dir),
                }
                aggregate_rows.append(row)

        # end of condition: 2 fast blinks
        if led is not None:
            led.off()  # ensure LED off first
            _led_blinks(led, 2, 0.15, 0.15)

    # -- batch end: 3 long blinks --
    if led is not None:
        led.off()
        print("\n[LED] Batch complete — 3 long blinks")
        _led_blinks(led, 3, 0.4, 0.3)
        led.close()

    # -- stop leader --
    if not args.dry_run and not args.keep_leader_running:
        try:
            _set_leader_config(api_base, running=False)
            print("[API] Leader stopped.")
        except Exception:
            pass
    elif args.keep_leader_running:
        print("[API] Leader left running (--keep-leader-running).")

    # -- save aggregate CSV --
    agg_fields = [
        "batch_id", "trial_id", "model_name", "leader_freq_hz",
        "follower_initial_freq_hz", "coupling_gain", "duration_s",
        "synchronization_success", "time_to_synchronization_s",
        "steady_state_mean_abs_timing_error_s", "steady_state_rmse_timing_error_s",
        "steady_state_jitter_s", "final_frequency_error_hz",
        "convergence_quality", "detection_success_rate", "false_positive_rate",
        "mean_loop_dt_ms", "peak_memory_rss_mb", "mean_cpu_temperature_c",
        "expected_leader_flash_count", "detected_leader_flash_count",
        "leader_flash_count_ratio",
        "requested_duration_s", "actual_trial_wall_duration_s",
        "trial_log_dir",
    ]
    _save_csv(batch_dir / "aggregate_metrics.csv", aggregate_rows, agg_fields)

    # -- summary by condition --
    summary_rows = []
    for freq in follower_freqs:
        cond_rows = [r for r in aggregate_rows
                     if abs(r["follower_initial_freq_hz"] - freq) < 0.001]
        n = len(cond_rows)
        successes = [r for r in cond_rows if r["synchronization_success"]]
        n_success = len(successes)
        sync_times = [float(r["time_to_synchronization_s"])
                      for r in successes if r["time_to_synchronization_s"] is not None]
        maes = [float(r["steady_state_mean_abs_timing_error_s"])
                for r in cond_rows
                if r["steady_state_mean_abs_timing_error_s"] is not None
                and not (isinstance(r["steady_state_mean_abs_timing_error_s"], float)
                         and np.isnan(r["steady_state_mean_abs_timing_error_s"]))]
        summary_rows.append({
            "follower_initial_freq_hz": freq,
            "n_trials": n,
            "success_rate": round(n_success / n, 4) if n > 0 else 0.0,
            "mean_time_to_sync_s": round(float(np.mean(sync_times)), 4) if sync_times else "",
            "std_time_to_sync_s": round(float(np.std(sync_times)), 4) if len(sync_times) >= 2 else "",
            "mean_steady_state_mae_s": round(float(np.mean(maes)), 6) if maes else "",
            "std_steady_state_mae_s": round(float(np.std(maes)), 6) if len(maes) >= 2 else "",
            "mean_jitter_s": "",
            "mean_final_frequency_error_hz": "",
            "mean_convergence_quality": "",
        })

    _save_csv(batch_dir / "summary_by_condition.csv", summary_rows,
              ["follower_initial_freq_hz", "n_trials", "success_rate",
               "mean_time_to_sync_s", "std_time_to_sync_s",
               "mean_steady_state_mae_s", "std_steady_state_mae_s",
               "mean_jitter_s", "mean_final_frequency_error_hz",
               "mean_convergence_quality"])

    # -- print batch summary --
    print()
    print("=" * 60)
    print("PI VISUAL BATCH COMPLETE")
    print("=" * 60)
    print(f"  Conditions:  {len(follower_freqs)}")
    print(f"  Trials:      {total}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Output:      {batch_dir}")
    for sr in summary_rows:
        print(f"    {sr['follower_initial_freq_hz']:.1f} Hz -> "
              f"success_rate={sr['success_rate']:.2f}  MAE={sr.get('mean_steady_state_mae_s','')}")
    print("=" * 60)

    return batch_dir


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3A-3 — Automated Pi Visual Kuramoto Batch Testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Short detection test
  PYTHONPATH=. python3 experiments/run_step3a_pi_visual_batch.py \
      --leader-api http://<laptop-ip>:8000 \
      --duration 5 --repeats 1 --leader-freqs 2.0 --follower-freqs 1.5 \
      --coupling-gain 3.5 --leader-shape circle --leader-dot-size 120

  # Visible sync demo (1 Hz follower -> 2 Hz leader)
  PYTHONPATH=. python3 experiments/run_step3a_pi_visual_batch.py \
      --leader-api http://<laptop-ip>:8000 \
      --duration 15 --repeats 1 --leader-freqs 2.0 --follower-freqs 1.0 \
      --coupling-gain 9.4 --leader-shape circle --leader-dot-size 120

  # Dry-run (no hardware)
  PYTHONPATH=. python experiments/run_step3a_pi_visual_batch.py \
      --duration 5 --repeats 1 --follower-freqs 1.5 --dry-run
        """,
    )
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--leader-freqs", type=float, nargs="+", default=[2.0])
    parser.add_argument("--follower-freqs", type=float, nargs="+",
                        default=[1.5, 1.8, 2.3])
    parser.add_argument("--coupling-gain", type=float, default=3.5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Oscillator base dt (fallback; real dt is measured via monotonic)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--flash-on-time", type=float, default=0.06,
                        help="Follower LED flash duration (seconds)")
    parser.add_argument("--random-delay-min", type=float, default=0.0)
    parser.add_argument("--random-delay-max", type=float, default=3.0)
    parser.add_argument("--sync-threshold-s", type=float, default=0.10)
    parser.add_argument("--sync-cycles", type=int, default=5)
    parser.add_argument("--log-dir", default="experiments/logs/step3a_pi_visual_batch")
    parser.add_argument("--trial-prefix", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target-loop-rate-hz", type=float, default=30.0,
                        help="Target loop rate; sleep spare time after each iteration")
    # Camera
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--detection-mode", default="local_contrast")
    parser.add_argument("--threshold-on", type=float, default=180)
    parser.add_argument("--threshold-off", type=float, default=120)
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    # LED
    parser.add_argument("--led-pin", type=int, default=17)
    # Leader visual
    parser.add_argument("--leader-shape", default="circle",
                        choices=["circle", "square"])
    parser.add_argument("--leader-dot-size", type=int, default=120)
    parser.add_argument("--leader-min-flash-interval-s", type=float, default=0.20)
    parser.add_argument("--keep-leader-running", action="store_true")

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    run_visual_batch(args)


if __name__ == "__main__":
    main()

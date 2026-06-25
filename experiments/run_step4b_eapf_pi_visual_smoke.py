#!/usr/bin/env python3
r"""Step 4B.2/4B.3 — EAPF Consensus Single-Leader Pi Visual Smoke Test.

Adapts the existing Pi visual pipeline to use ``EventBasedConsensusPLLOscillator``
instead of Kuramoto.  The follower receives only camera-detected leader flash
events; it never accesses the true leader phase or frequency.

Usage (dry-run — no hardware)::

    PYTHONPATH=. python experiments/run_step4b_eapf_pi_visual_smoke.py --dry-run

Usage (real Pi hardware)::

    PYTHONPATH=. python3 experiments/run_step4b_eapf_pi_visual_smoke.py \
        --leader-api http://<laptop-ip>:8000 \
        --duration 30 --follower-freq 1.5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firefly_sync.logging.metrics import (
    check_flash_synchronisation,
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
        print("Or use --dry-run for logic test without hardware.")
        sys.exit(1)


# ======================================================================
# Leader UI HTTP client (minimal)
# ======================================================================

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


# ======================================================================
# Output helpers
# ======================================================================

def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ======================================================================
# Locked EAPF Consensus parameters (Stage 4A)
# ======================================================================

EAPF_LOCKED = {
    "phase_gain": 0.02,
    "frequency_gain": 0.02,
    "phase_error_filter_alpha": 0.2,
    "frequency_error_filter_alpha": 0.2,
    "max_phase_step_rad": 0.2,
    "max_frequency_step_hz": 0.05,
    "frequency_min_hz": 0.5,
    "frequency_max_hz": 4.0,
}


# ======================================================================
# Single smoke trial
# ======================================================================

def _run_smoke_trial(args: argparse.Namespace) -> dict:
    """Run one EAPF single-leader Pi visual smoke test. Returns metrics dict."""
    from firefly_sync.core.event_based_consensus_pll import (
        EventBasedConsensusPLLOscillator,
    )

    _ensure_pi_hardware(dry_run=args.dry_run)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Leader API ---
    if not args.dry_run:
        api = args.leader_api.rstrip("/")
        print(f"[API] Setting leader to {args.leader_freq} Hz")
        _set_leader_config(api, frequency_hz=args.leader_freq, running=True,
                           shape="circle", target_size_px=350,
                           brightness_on=255, brightness_off=0,
                           background_brightness=0)
        time.sleep(2.0)

    # --- Initialise EAPF oscillator ---
    osc = EventBasedConsensusPLLOscillator()
    for k, v in EAPF_LOCKED.items():
        setattr(osc.config, k, v)
    osc._frequency_hz = args.follower_freq
    osc._omega_rad_s = 2.0 * np.pi * args.follower_freq

    # --- Hardware ---
    detector = None
    led = None
    if not args.dry_run:
        assert _PicameraFlashDetector is not None
        detector = _PicameraFlashDetector(
            resolution=[args.width, args.height],
            detection_mode="local_contrast",
            threshold_on=args.threshold_on,
            threshold_off=args.threshold_off,
            min_interval_s=args.min_interval,
            window_s=args.window_s,
        )
        detector.start()
        led = _PiGPIOLED(pin=args.led_pin, flash_duration_s=args.flash_on_time)
        print(f"[HW] Camera started. GPIO{args.led_pin} LED ready.")

    # --- Metadata ---
    metadata = {
        "model": "eapf_consensus",
        "mode": "pi_visual_smoke",
        "leader_freq_hz": args.leader_freq,
        "follower_initial_freq_hz": args.follower_freq,
        "duration_s": args.duration,
        "sync_threshold_s": args.sync_threshold_s,
        "sync_cycles": args.sync_cycles,
        "locked_params": EAPF_LOCKED,
        "dry_run": args.dry_run,
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(out_dir / "metadata.json", metadata)

    # --- Run loop ---
    follower_flash_times: list[float] = []
    detected_leader_times: list[float] = []
    flash_events: list[dict] = []
    osc_log: list[dict] = []
    det_log: list[dict] = []
    sync_achieved = False
    loop_dts: list[float] = []
    trial_start = time.monotonic()
    prev_loop = trial_start
    prev_detected_state = False
    last_leader_event_t = -999.0
    led_on_until = 0.0

    print(f"\n[SMOKE] Starting {args.duration}s trial. "
          f"Leader={args.leader_freq}Hz, Follower={args.follower_freq}Hz")

    try:
        while (time.monotonic() - trial_start) < args.duration:
            now = time.monotonic()
            loop_dt = now - prev_loop
            prev_loop = now
            loop_dts.append(loop_dt)
            t = now - trial_start

            # --- Camera detection ---
            cam_ms = 0.0
            leader_event = False
            if not args.dry_run and detector is not None:
                cam_start = time.monotonic()
                result = detector.capture_frame()
                cam_ms = (time.monotonic() - cam_start) * 1000.0

                # Rising-edge detection on camera state
                detected_state = (result.get("state") == "ON")
                if detected_state and not prev_detected_state:
                    if (t - last_leader_event_t) >= args.leader_min_flash_interval_s:
                        leader_event = True
                        last_leader_event_t = t
                prev_detected_state = detected_state

                det_log.append({
                    "t_s": round(t, 6),
                    "detected_state": 1 if detected_state else 0,
                    "leader_flash_event": 1 if leader_event else 0,
                    "brightness_used": result.get("brightness_used", 0),
                    "signal_norm": result.get("signal_norm", 0),
                    "camera_ms": round(cam_ms, 2),
                })
            else:
                det_log.append({
                    "t_s": round(t, 6),
                    "detected_state": 0, "leader_flash_event": 0,
                    "brightness_used": 0, "signal_norm": 0, "camera_ms": 0,
                })

            # --- EAPF step ---
            nids = [0] if leader_event else []  # leader is always neighbour 0
            if leader_event:
                detected_leader_times.append(t)
                flash_events.append({"t_s": round(t, 6), "event_type": "leader_flash"})

            r = osc.step(dt_s=loop_dt, t_s=t, neighbour_flash_ids=nids)

            osc_log.append({
                "t_s": round(t, 6),
                "phase_rad": r["phase_rad"],
                "frequency_hz": r["frequency_hz"],
                "phase_error_rad": r["phase_error_rad"],
                "follower_flash": 1 if r["follower_flash_event"] else 0,
                "loop_dt_ms": round(loop_dt * 1000, 3),
            })

            # --- Follower flash + LED ---
            if r["follower_flash_event"]:
                follower_flash_times.append(t)
                flash_events.append({"t_s": round(t, 6), "event_type": "follower_flash"})
                if led is not None:
                    led.on()
                    led_on_until = t + args.flash_on_time

            # LED off after flash duration
            if led is not None and t >= led_on_until:
                led.off()

            # Check sync
            if len(follower_flash_times) >= 5:
                sc = check_flash_synchronisation(
                    detected_leader_times, follower_flash_times,
                    args.sync_threshold_s, args.sync_cycles,
                )
                if sc["synchronization_success"] and not sync_achieved:
                    sync_achieved = True
                    print(f"  [SYNC] Achieved at t={t:.2f}s")

    except KeyboardInterrupt:
        print("\n  Trial interrupted.")

    # --- Cleanup ---
    if led is not None:
        led.off()
        led.close()
    if detector is not None:
        detector.stop()

    actual_wall = time.monotonic() - trial_start

    # --- Metrics ---
    sync_success = sync_achieved
    time_to_sync = None
    if sync_success and len(follower_flash_times) >= 5:
        pairs = pair_flash_events(detected_leader_times, follower_flash_times)
        run_len = 0
        for i, p in enumerate(pairs):
            if p["abs_error_s"] is not None and p["abs_error_s"] < args.sync_threshold_s:
                run_len += 1
                if run_len >= args.sync_cycles:
                    time_to_sync = follower_flash_times[i]
                    break
            else:
                run_len = 0

    # Steady-state
    window = min(10.0, args.duration * 0.3)
    cutoff = max(0, (follower_flash_times[-1] if follower_flash_times else args.duration) - window)
    ft_recent = [t for t in follower_flash_times if t >= cutoff]
    lt_recent = [t for t in detected_leader_times if t >= cutoff]
    timing_errs: list[float] = []
    for ft in ft_recent:
        if lt_recent:
            nearest = min(lt_recent, key=lambda lt: abs(lt - ft))
            timing_errs.append(abs(ft - nearest))
    mae = float(np.mean(timing_errs)) if timing_errs else 0.0
    jitter = float(np.std(timing_errs)) if len(timing_errs) >= 2 else 0.0

    # Detection reliability
    expected_leader = args.duration * args.leader_freq
    detected_leader = len(detected_leader_times)
    leader_fcr = detected_leader / expected_leader if expected_leader > 0 else 0

    # Follower flash ratio
    expected_follower = expected_leader
    follower_fcr = len(follower_flash_times) / expected_follower if expected_follower > 0 else 0

    # Loop rate
    mean_dt = float(np.mean(loop_dts)) if loop_dts else 0
    effective_rate = 1.0 / mean_dt if mean_dt > 0 else 0

    metrics = {
        "sync_success": sync_success,
        "time_to_sync_s": round(time_to_sync, 4) if time_to_sync else None,
        "steady_state_mae_s": round(mae, 6),
        "steady_state_jitter_s": round(jitter, 6),
        "final_frequency_hz": round(osc.frequency_hz, 6),
        "leader_flash_count": detected_leader,
        "expected_leader_count": round(expected_leader, 1),
        "leader_fcr": round(leader_fcr, 4),
        "follower_flash_count": len(follower_flash_times),
        "follower_fcr": round(follower_fcr, 4),
        "effective_loop_rate_hz": round(effective_rate, 2),
        "actual_wall_duration_s": round(actual_wall, 3),
        "requested_duration_s": args.duration,
    }
    _save_json(out_dir / "metrics_summary.json", metrics)

    # --- Save logs ---
    _save_csv(out_dir / "oscillator_log.csv", osc_log,
              ["t_s", "phase_rad", "frequency_hz", "phase_error_rad",
               "follower_flash", "loop_dt_ms"])
    _save_csv(out_dir / "detection_log.csv", det_log,
              ["t_s", "detected_state", "leader_flash_event",
               "brightness_used", "signal_norm", "camera_ms"])
    _save_csv(out_dir / "flash_events.csv", flash_events,
              ["t_s", "event_type"])

    # --- Quick-look figures ---
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Frequency convergence
    fig, ax = plt.subplots(figsize=(8, 3))
    ts = [e["t_s"] for e in osc_log]
    fs = [e["frequency_hz"] for e in osc_log]
    ax.plot(ts, fs, linewidth=0.8)
    ax.axhline(y=args.leader_freq, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("EAPF Consensus — Frequency Convergence")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "frequency_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Flash raster
    fig, ax = plt.subplots(figsize=(10, 2))
    if detected_leader_times:
        ax.eventplot(detected_leader_times, lineoffsets=1, colors="steelblue",
                     linewidths=0.8, label="Leader (detected)")
    if follower_flash_times:
        ax.eventplot(follower_flash_times, lineoffsets=0, colors="darkorange",
                     linewidths=0.8, label="Follower")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Follower", "Leader"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"EAPF Consensus Pi Visual Smoke — Flash Raster\n"
                 f"sync={sync_success}, MAE={mae:.4f}s")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(fig_dir / "flash_raster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print(f"\n  === SMOKE RESULT ===")
    print(f"  Sync:       {sync_success}")
    print(f"  Time to sync: {time_to_sync}")
    print(f"  MAE:        {mae:.6f} s")
    print(f"  Final freq: {osc.frequency_hz:.4f} Hz")
    print(f"  Leader FCR: {leader_fcr:.3f} ({detected_leader}/{expected_leader:.0f})")
    print(f"  Follower FCR: {follower_fcr:.3f} ({len(follower_flash_times)}/{expected_follower:.0f})")
    print(f"  Loop rate:  {effective_rate:.1f} Hz")
    print(f"  Wall time:  {actual_wall:.1f} s")

    return metrics


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4B.2/4B.3 — EAPF Consensus Pi Visual Smoke Test.",
    )
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--leader-freq", type=float, default=2.0)
    parser.add_argument("--follower-freq", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--flash-on-time", type=float, default=0.06)
    parser.add_argument("--sync-threshold-s", type=float, default=0.10)
    parser.add_argument("--sync-cycles", type=int, default=5)
    parser.add_argument("--log-dir", default="experiments/logs/step4b_eapf_pi_visual_smoke")
    parser.add_argument("--dry-run", action="store_true")
    # Camera
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--threshold-on", type=float, default=180)
    parser.add_argument("--threshold-off", type=float, default=120)
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--leader-min-flash-interval-s", type=float, default=0.20)
    # LED
    parser.add_argument("--led-pin", type=int, default=17)

    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = str(Path(args.log_dir) / f"{ts}_eapf_pi_visual_smoke")

    _run_smoke_trial(args)


if __name__ == "__main__":
    main()

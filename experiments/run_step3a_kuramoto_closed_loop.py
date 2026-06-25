#!/usr/bin/env python3
r"""Step 3A — Kuramoto Leader-Follower Closed-Loop Synchronisation.

Single visual leader → single Kuramoto follower.
The follower estimates the leader phase from **detected flash timestamps**
and applies Kuramoto phase coupling.  It never directly knows the true
leader frequency.

Modes
-----
``--mode mock``
    Runs entirely on a laptop without Pi hardware.  Synthetic leader flash
    events are generated at a configurable frequency.  The follower
    oscillator synchronises via Kuramoto coupling.

``--mode pi``
    Runs on a Raspberry Pi 5 with the existing camera flash detection
    pipeline and GPIO17 LED.  Requires ``picamera2`` and ``gpiozero``.

Usage (mock)::

    python experiments/run_step3a_kuramoto_closed_loop.py \
        --mode mock --duration 30 --leader-freq 2.0 --follower-freq 1.5 \
        --coupling-gain 3.5

Usage (pi)::

    python experiments/run_step3a_kuramoto_closed_loop.py \
        --mode pi --duration 30 --leader-freq 2.0 --follower-freq 1.5 \
        --coupling-gain 3.5 --allow-hardware-fallback
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
# Leader Phase Estimator
# ======================================================================

class LeaderPhaseEstimator:
    """Estimate leader phase from **detected** flash timestamps only.

    This class never accesses the true leader frequency.  It builds an
    internal estimate of the leader period from recent inter‑flash
    intervals and uses that to project the leader phase forward in time.

    Bootstrap behaviour
    -------------------
    Until at least two leader flashes have been detected the estimator
    uses *initial_period_guess_s*.  This is logged clearly so that it
    can be distinguished from a genuine estimate.
    """

    def __init__(
        self,
        initial_period_guess_s: float = 0.5,
        max_stored_flashes: int = 30,
    ) -> None:
        self._initial_period_guess_s = float(initial_period_guess_s)
        self._max_stored = max_stored_flashes
        self._flash_times: list[float] = []        # detected flash timestamps
        self._estimated_period_s: float = initial_period_guess_s
        self._bootstrap: bool = True                # True until ≥ 2 flashes

    # -- properties --

    @property
    def estimated_period_s(self) -> float:
        return self._estimated_period_s

    @property
    def estimated_frequency_hz(self) -> float:
        if self._estimated_period_s <= 0:
            return 0.0
        return 1.0 / self._estimated_period_s

    @property
    def flash_count(self) -> int:
        return len(self._flash_times)

    @property
    def is_bootstrapping(self) -> bool:
        return self._bootstrap

    @property
    def last_flash_time(self) -> float | None:
        return self._flash_times[-1] if self._flash_times else None

    # -- public API --

    def record_flash(self, timestamp_s: float) -> None:
        """Register a detected leader flash at *timestamp_s*."""
        self._flash_times.append(timestamp_s)
        if len(self._flash_times) > self._max_stored:
            self._flash_times.pop(0)
        self._update_period_estimate()

    def estimate_phase(self, now_s: float) -> float:
        """Estimate leader phase (radians, [0, 2π)) at time *now_s*.

        θ_leader = 2π · ((t − t_last) / T_est)

        If no flashes have been detected yet, returns 0.0.
        """
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

    # -- internal --

    def _update_period_estimate(self) -> None:
        """Recalculate estimated period from recent inter‑flash intervals.

        Uses the **median** of the most recent intervals to be robust
        against occasional missed detections or jitter.
        """
        if len(self._flash_times) < 2:
            return

        intervals = [
            self._flash_times[i + 1] - self._flash_times[i]
            for i in range(len(self._flash_times) - 1)
        ]
        # Use last N intervals (capped at 10 for responsiveness)
        recent = intervals[-10:]
        self._estimated_period_s = float(np.median(recent))
        self._bootstrap = False


# ======================================================================
# Mock-mode helpers
# ======================================================================

def _generate_leader_flash_times(
    duration_s: float,
    leader_freq_hz: float,
) -> list[float]:
    """Generate synthetic leader flash timestamps at exact intervals.

    These represent **ground-truth** leader flashes.  The follower only
    sees them through the ``LeaderPhaseEstimator``, which does not have
    access to ``leader_freq_hz``.

    Returns a sorted list of flash times (seconds).
    """
    if leader_freq_hz <= 0:
        return []
    period = 1.0 / leader_freq_hz
    n_flashes = int(duration_s / period) + 1
    return [i * period for i in range(n_flashes)]


# ======================================================================
# Logging helpers
# ======================================================================

def _ensure_output_dir(base_dir: str, trial_id: str | None) -> Path:
    """Create a timestamped output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{ts}_{trial_id}" if trial_id else ts
    out = Path(base_dir) / folder
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ======================================================================
# Main runner
# ======================================================================

def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    """Execute a single closed‑loop synchronisation trial.

    Returns the metrics summary dict.
    """

    # ------------------------------------------------------------------
    # 1. Setup output directory
    # ------------------------------------------------------------------
    out_dir = _ensure_output_dir(args.log_dir, args.trial_id)

    # ------------------------------------------------------------------
    # 2. Metadata
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {
        "trial_id": args.trial_id or "auto",
        "mode": args.mode,
        "model_name": "kuramoto",
        "duration_s": args.duration,
        "leader_freq_hz": args.leader_freq,
        "follower_initial_freq_hz": args.follower_freq,
        "coupling_gain": args.coupling_gain,
        "dt": args.dt,
        "flash_on_time_s": args.flash_on_time,
        "sync_threshold_s": args.sync_threshold_s,
        "sync_cycles": args.sync_cycles,
        "timestamp": datetime.now().isoformat(),
        "notes": args.notes or "",
    }
    _save_json(out_dir / "metadata.json", metadata)

    # ------------------------------------------------------------------
    # 3. Initialise mode-specific components
    # ------------------------------------------------------------------
    if args.mode == "mock":
        # Generate ground-truth leader flash schedule
        leader_truth = _generate_leader_flash_times(args.duration, args.leader_freq)
        print(f"[mock] Generated {len(leader_truth)} synthetic leader flashes "
              f"at {args.leader_freq} Hz")

        # Bootstrap period guess for the phase estimator.
        # We use 1/follower_freq as a reasonable initial guess, but the
        # estimator is *not* told the true leader frequency.
        initial_period_guess = 1.0 / args.follower_freq if args.follower_freq > 0 else 0.5
        phase_estimator = LeaderPhaseEstimator(
            initial_period_guess_s=initial_period_guess,
        )

        detection_success_rate: float | None = 1.0
        false_positive_rate: float | None = 0.0
        use_hardware = False

    elif args.mode == "pi":
        detection_success_rate = None   # no ground truth in real Pi mode
        false_positive_rate = None

        # Attempt hardware imports
        try:
            from firefly_sync.hardware.pi_led import PiGPIOLED           # type: ignore
            from firefly_sync.hardware.picamera_flash_detector import (  # type: ignore
                PicameraFlashDetector,
            )
            use_hardware = True
        except ImportError as exc:
            if args.allow_hardware_fallback:
                print(f"[pi] WARNING: Hardware import failed ({exc}). "
                      "Falling back to mock LED/detector.")
                use_hardware = False
            else:
                print(f"[pi] ERROR: Hardware import failed: {exc}")
                print("[pi] Use --allow-hardware-fallback to run without "
                      "real hardware, or install Pi dependencies.")
                sys.exit(1)

        if use_hardware:
            # Initialise camera detector
            detector = PicameraFlashDetector(
                resolution=[640, 480],
                detection_mode="local_contrast",
                threshold_on=args.threshold_on,
                threshold_off=args.threshold_off,
                min_interval_s=args.min_interval,
                window_s=args.window_s,
            )
            detector.start()
            print("[pi] Camera detector started.")

            # Initialise GPIO LED
            led = PiGPIOLED(pin=args.led_pin, flash_duration_s=args.flash_on_time)
            print(f"[pi] GPIO{args.led_pin} LED initialised.")

        # Leader phase estimator with bootstrap guess
        initial_period_guess = 1.0 / args.follower_freq if args.follower_freq > 0 else 0.5
        phase_estimator = LeaderPhaseEstimator(
            initial_period_guess_s=initial_period_guess,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # ------------------------------------------------------------------
    # 4. Initialise follower oscillator
    # ------------------------------------------------------------------
    follower = KuramotoModel(
        natural_frequency=2.0 * np.pi * args.follower_freq,  # rad/s
        initial_phase=0.0,
        coupling_strength=args.coupling_gain,
        dt=args.dt,
    )

    # ------------------------------------------------------------------
    # 5. Run closed-loop simulation
    # ------------------------------------------------------------------
    t = 0.0
    step_idx = 0
    leader_flash_idx = 0          # index into leader_truth (mock) or counter (pi)
    follower_flash_times: list[float] = []
    detected_leader_times: list[float] = []   # times the estimator recorded a flash
    oscillator_log: list[dict] = []
    flash_events: list[dict] = []
    sync_achieved = False
    sync_achieved_time: float | None = None

    start_wall = time.perf_counter()

    print(f"\nTrial starting — {args.duration}s, dt={args.dt}s")
    print(f"  Leader freq: {args.leader_freq} Hz  "
          f"Follower initial: {args.follower_freq} Hz  "
          f"K = {args.coupling_gain}")
    print()

    try:
        while t < args.duration:
            # ----------------------------------------------------------
            # 5a. Detect leader flashes (mock or pi)
            # ----------------------------------------------------------
            if args.mode == "mock":
                # Check if a new synthetic leader flash has occurred
                while (leader_flash_idx < len(leader_truth) and
                       leader_truth[leader_flash_idx] <= t + args.dt * 0.5):
                    lf_time = leader_truth[leader_flash_idx]
                    phase_estimator.record_flash(lf_time)
                    detected_leader_times.append(lf_time)
                    flash_events.append({
                        "t": round(lf_time, 6),
                        "event_type": "leader_flash",
                        "source": "synthetic",
                        "leader_flash_time": round(lf_time, 6),
                        "follower_flash_time": None,
                        "timing_error_s": None,
                    })
                    leader_flash_idx += 1
            else:  # pi
                # TODO: poll camera detector for rising edges
                # For now, in pi mode with hardware, we read the detector.
                # When allow_hardware_fallback is used without hardware,
                # we skip detection (no leader flashes → no coupling).
                if use_hardware:
                    result = detector.capture_frame()
                    if result.get("event_type") == "leader_rising_edge":
                        lf_time = t
                        phase_estimator.record_flash(lf_time)
                        detected_leader_times.append(lf_time)
                        flash_events.append({
                            "t": round(lf_time, 6),
                            "event_type": "leader_flash",
                            "source": "camera",
                            "leader_flash_time": round(lf_time, 6),
                            "follower_flash_time": None,
                            "timing_error_s": None,
                        })
                # (no leader flashes if hardware unavailable and fallback active)

            # ----------------------------------------------------------
            # 5b. Estimate leader phase & compute coupling
            # ----------------------------------------------------------
            est_leader_phase = phase_estimator.estimate_phase(t)
            follower_phase = follower.phase

            # Phase error (wrapped difference in [-π, π])
            phase_error = float(
                np.arctan2(
                    np.sin(est_leader_phase - follower_phase),
                    np.cos(est_leader_phase - follower_phase),
                )
            )

            # Kuramoto coupling term (single leader: N=1)
            coupling_input = np.sin(est_leader_phase - follower_phase)

            # Bootstrap indicator for logging
            bootstrap_flag = 1 if phase_estimator.is_bootstrapping else 0

            # ----------------------------------------------------------
            # 5c. Step follower oscillator
            # ----------------------------------------------------------
            state = follower.step(coupling_input)

            oscillator_log.append({
                "t": round(t, 6),
                "follower_phase_rad": round(follower_phase, 6),
                "follower_frequency_hz": round(
                    follower.natural_frequency / (2.0 * np.pi), 6,
                ),
                "estimated_leader_phase_rad": round(est_leader_phase, 6),
                "phase_error_rad": round(phase_error, 6),
                "coupling_term": round(coupling_input, 6),
                "follower_led_state": 1 if state.is_firing else 0,
                "sync_state": 1 if sync_achieved else 0,
                "bootstrap_estimate": bootstrap_flag,
            })

            # ----------------------------------------------------------
            # 5d. Handle follower flash
            # ----------------------------------------------------------
            if state.is_firing:
                follower_flash_times.append(t)

                # Pair with nearest leader flash for timing error
                pair_info = pair_flash_events(
                    detected_leader_times, [t],
                )
                timing_err = pair_info[0]["timing_error_s"] if pair_info else None

                flash_events.append({
                    "t": round(t, 6),
                    "event_type": "follower_flash",
                    "source": "follower_oscillator",
                    "leader_flash_time": pair_info[0].get("leader_t"),
                    "follower_flash_time": round(t, 6),
                    "timing_error_s": timing_err,
                })

                # Check sync criterion
                sync_check = check_flash_synchronisation(
                    detected_leader_times,
                    follower_flash_times,
                    sync_threshold_s=args.sync_threshold_s,
                    sync_cycles=args.sync_cycles,
                )
                if sync_check["synchronization_success"] and not sync_achieved:
                    sync_achieved = True
                    sync_achieved_time = t
                    flash_events.append({
                        "t": round(t, 6),
                        "event_type": "sync_achieved",
                        "source": "metrics",
                        "leader_flash_time": None,
                        "follower_flash_time": None,
                        "timing_error_s": None,
                    })
                    print(f"  [OK] Sync achieved at t = {t:.3f}s")

                # Drive hardware LED in pi mode
                if args.mode == "pi" and use_hardware:
                    led.flash(args.flash_on_time)

            t += args.dt
            step_idx += 1

    except KeyboardInterrupt:
        print("\nTrial interrupted by user.")

    wall_elapsed = time.perf_counter() - start_wall

    # ------------------------------------------------------------------
    # 6. Cleanup hardware
    # ------------------------------------------------------------------
    if args.mode == "pi" and use_hardware:
        try:
            detector.stop()
            led.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7. Compute metrics
    # ------------------------------------------------------------------
    metrics = compute_flash_timing_metrics(
        leader_times=detected_leader_times,
        follower_times=follower_flash_times,
        sync_threshold_s=args.sync_threshold_s,
        sync_cycles=args.sync_cycles,
        detection_success_rate=detection_success_rate,
        false_positive_rate=false_positive_rate,
    )

    # ------------------------------------------------------------------
    # 8. Save logs
    # ------------------------------------------------------------------
    # oscillator_log.csv
    osc_path = out_dir / "oscillator_log.csv"
    osc_fields = [
        "t", "follower_phase_rad", "follower_frequency_hz",
        "estimated_leader_phase_rad", "phase_error_rad",
        "coupling_term", "follower_led_state", "sync_state",
        "bootstrap_estimate",
    ]
    with open(osc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=osc_fields)
        writer.writeheader()
        writer.writerows(oscillator_log)

    # flash_events.csv
    flash_path = out_dir / "flash_events.csv"
    flash_fields = [
        "t", "event_type", "source",
        "leader_flash_time", "follower_flash_time", "timing_error_s",
    ]
    with open(flash_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flash_fields)
        writer.writeheader()
        writer.writerows(flash_events)

    # metrics_summary.json
    _save_json(out_dir / "metrics_summary.json", metrics)

    # ------------------------------------------------------------------
    # 9. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("TRIAL COMPLETE")
    print("=" * 60)
    print(f"  Mode:             {args.mode}")
    print(f"  Duration:         {args.duration:.1f}s (wall: {wall_elapsed:.1f}s)")
    print(f"  Steps:            {step_idx}")
    print(f"  Leader flashes:   {len(detected_leader_times)}")
    print(f"  Follower flashes: {len(follower_flash_times)}")
    print(f"  Sync achieved:    {metrics['synchronization_success']}")
    if metrics["time_to_synchronization_s"] is not None:
        print(f"  Time to sync:     {metrics['time_to_synchronization_s']:.3f}s")
    print(f"  Steady-state MAE: {metrics['steady_state_mean_abs_timing_error_s']:.6f}s")
    print(f"  Steady-state RMSE:{metrics['steady_state_rmse_timing_error_s']:.6f}s")
    print(f"  Freq error:       {metrics['final_frequency_error_hz']:.6f} Hz")
    print(f"  Convergence qual: {metrics['convergence_quality']:.4f}")
    print(f"  Output:           {out_dir}")
    print("=" * 60)

    return metrics


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3A — Kuramoto Leader-Follower Closed-Loop Sync.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["mock", "pi"], default="mock")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--leader-freq", type=float, default=2.0,
                        help="Leader flash frequency in Hz (mock: ground truth; pi: not used by follower)")
    parser.add_argument("--follower-freq", type=float, default=1.5,
                        help="Follower initial natural frequency in Hz")
    parser.add_argument("--coupling-gain", type=float, default=3.5,
                        help="Kuramoto coupling gain K")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Integration timestep in seconds")
    parser.add_argument("--flash-on-time", type=float, default=0.06,
                        help="Follower LED flash duration in seconds")
    parser.add_argument("--sync-threshold-s", type=float, default=0.10,
                        help="Max absolute timing error for sync (seconds)")
    parser.add_argument("--sync-cycles", type=int, default=5,
                        help="Consecutive qualifying cycles required for sync")
    parser.add_argument("--log-dir", default="experiments/logs/step3a")
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--notes", default=None)
    # Pi-specific
    parser.add_argument("--allow-hardware-fallback", action="store_true",
                        help="In pi mode, fall back if hardware is unavailable")
    parser.add_argument("--led-pin", type=int, default=17)
    parser.add_argument("--threshold-on", type=float, default=180.0)
    parser.add_argument("--threshold-off", type=float, default=120.0)
    parser.add_argument("--min-interval", type=float, default=0.2)
    parser.add_argument("--window-s", type=float, default=5.0)

    args = parser.parse_args()

    # Validate
    if args.duration <= 0:
        parser.error("--duration must be > 0")
    if args.leader_freq <= 0:
        parser.error("--leader-freq must be > 0")
    if args.follower_freq <= 0:
        parser.error("--follower-freq must be > 0")

    run_trial(args)


if __name__ == "__main__":
    main()

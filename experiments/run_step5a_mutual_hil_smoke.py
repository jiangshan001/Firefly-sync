#!/usr/bin/env python3
r"""Step 5A — Mutual Mixed-Reality HIL Smoke Test.

Usage::

    PYTHONPATH=. python3 experiments/run_step5a_mutual_hil_smoke.py \
        --leader-api http://<laptop-ip>:8000 --duration 60 --feedback-off

    PYTHONPATH=. python experiments/run_step5a_mutual_hil_smoke.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ======================================================================
# Pi hardware — lazy imports
# ======================================================================

_PiGPIOLED: Any = None; _PicameraFlashDetector: Any = None

def _ensure_hw(dry: bool = False) -> None:
    global _PiGPIOLED, _PicameraFlashDetector
    if dry: return
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED as _PL
        from firefly_sync.hardware.picamera_flash_detector import PicameraFlashDetector as _PFD
        _PiGPIOLED = _PL; _PicameraFlashDetector = _PFD
    except ImportError as e:
        print(f"ERROR: {e}"); sys.exit(1)

# ======================================================================
# Robust API wrapper with retry + debug logging
# ======================================================================

class ApiError(RuntimeError):
    pass

def _first_agent(payload: dict) -> dict:
    agents = payload.get("agents", [])
    return agents[0] if agents else {}

def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)

def _wrap_2pi(value: float) -> float:
    return float(value % (2.0 * np.pi))

def _phase_difference_rad(pi_phase_rad: float, virtual_phase_rad: float) -> float:
    return float(((pi_phase_rad - virtual_phase_rad + np.pi) % (2.0 * np.pi)) - np.pi)

def _resolve_initial_phases(
    seed: int | None,
    random_phase: bool,
    virtual_phase_rad: float | None,
    pi_phase_rad: float | None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if random_phase and virtual_phase_rad is None:
        virtual_phase_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    if random_phase and pi_phase_rad is None:
        pi_phase_rad = float(rng.uniform(0.0, 2.0 * np.pi))
    virtual_phase = _wrap_2pi(virtual_phase_rad if virtual_phase_rad is not None else 0.0)
    pi_phase = _wrap_2pi(pi_phase_rad if pi_phase_rad is not None else 0.0)
    return virtual_phase, pi_phase, _phase_difference_rad(pi_phase, virtual_phase)

CSV_COMMON_COLUMNS = [
    "monotonic_time_s",
    "wall_time_s",
    "event",
    "endpoint",
    "method",
    "ok",
    "elapsed_ms",
    "error",
]

def _csv_fieldnames_for_rows(
    rows: list[dict[str, Any]],
    preferred: list[str] | None = None,
) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row.keys())
    if preferred is None:
        preferred = CSV_COMMON_COLUMNS
    first = [key for key in preferred if key in keys]
    rest = sorted(key for key in keys if key not in first)
    return first + rest

def _write_dict_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    if not rows:
        return
    if fields is None:
        fieldnames = _csv_fieldnames_for_rows(rows)
    else:
        fieldnames = list(fields)
        extras = sorted(
            str(key)
            for row in rows
            for key in row.keys()
            if str(key) not in fieldnames
        )
        for key in extras:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)

def _safe_write_dict_rows_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> bool:
    try:
        _write_dict_rows_csv(path, rows, fields)
        return True
    except Exception as exc:
        print(f"WARNING: failed to write optional CSV {path}: {exc}")
        return False

def _frame_to_gray_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        grey = arr
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        grey = np.mean(arr[:, :, :3], axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        grey = arr[:, :, 0]
    else:
        grey = np.asarray(arr)
    if grey.dtype == np.uint8:
        return grey
    return np.clip(grey.astype(np.float32), 0, 255).astype(np.uint8)

def _save_debug_frame(path: Path, frame: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, _frame_to_gray_uint8(frame), cmap="gray", vmin=0, vmax=255)

def _blob_inside_expected_gate(res: dict, expected_x: float | None,
                               expected_y: float | None,
                               radius_px: float | None) -> tuple[bool, str]:
    if expected_x is None or expected_y is None or radius_px is None:
        return True, ""
    bx = _as_float(res.get("selected_blob_x"))
    by = _as_float(res.get("selected_blob_y"))
    if bx is None or by is None:
        return False, "no_selected_blob_for_spatial_gate"
    dist = float(np.hypot(bx - expected_x, by - expected_y))
    if dist > radius_px:
        return False, f"blob_outside_expected_radius:{dist:.1f}px"
    return True, ""

def _estimate_flash_frequency(times: list[float], window_s: float = 10.0) -> float | None:
    if len(times) < 2:
        return None
    end_t = times[-1]
    recent = [t for t in times if t >= end_t - window_s]
    if len(recent) < 2:
        recent = times[-min(len(times), 6):]
    intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    intervals = [v for v in intervals if v > 0]
    if not intervals:
        return None
    return 1.0 / float(np.median(intervals))

def _model_parameters(model: str, kuramoto_gain: float = 5.0) -> dict[str, float]:
    if model == "eapf_consensus":
        return {
            "g_p": 0.02,
            "g_f": 0.02,
            "alpha_p": 0.2,
            "alpha_f": 0.2,
            "delta_theta_max_rad": 0.2,
            "delta_f_max_hz": 0.05,
        }
    return {"K": float(kuramoto_gain)}

class FlashPhaseEstimator:
    """Estimate a neighbour phase from observed flash timestamps."""

    def __init__(self, initial_period_guess_s: float = 0.5,
                 max_stored_flashes: int = 30) -> None:
        self._initial_period_guess_s = float(initial_period_guess_s)
        self._estimated_period_s = float(initial_period_guess_s)
        self._max_stored = max_stored_flashes
        self._flash_times: list[float] = []

    @property
    def estimated_frequency_hz(self) -> float:
        return 1.0 / self._estimated_period_s if self._estimated_period_s > 0 else 0.0

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
        phase = 2.0 * np.pi * ((now_s - self._flash_times[-1]) / self._estimated_period_s)
        return float(phase % (2.0 * np.pi))

def _parse_roi(value: str | None) -> list[int] | None:
    if not value:
        return None
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h")
    try:
        roi = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--roi values must be integers") from exc
    if roi[2] <= 0 or roi[3] <= 0:
        raise argparse.ArgumentTypeError("--roi width and height must be positive")
    return roi

class CameraCaptureThread:
    """Continuously captures detector frames and keeps only the newest result."""

    def __init__(self, detector: Any, queue_size: int = 2) -> None:
        self.detector = detector
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="step5a-camera-capture", daemon=True)
        self.capture_count = 0
        self.capture_queue_drops = 0
        self.capture_intervals_s: list[float] = []
        self.capture_durations_ms: list[float] = []
        self.errors: list[str] = []
        self._last_capture_t: float | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._stopped_at = time.monotonic()

    def _drop_oldest_if_full(self) -> None:
        if not self._queue.full():
            return
        try:
            self._queue.get_nowait()
            self.capture_queue_drops += 1
        except queue.Empty:
            pass

    def _put_latest(self, payload: dict[str, Any]) -> None:
        self._drop_oldest_if_full()
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            self.capture_queue_drops += 1

    def _run(self) -> None:
        seq = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                frame = self.detector.capture_raw_frame()
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                time.sleep(0.01)
                continue
            captured_at = time.monotonic()
            detector_timestamp = time.perf_counter()
            if self._last_capture_t is not None:
                self.capture_intervals_s.append(captured_at - self._last_capture_t)
            self._last_capture_t = captured_at
            seq += 1
            self.capture_count += 1
            self.capture_durations_ms.append((captured_at - started) * 1000.0)
            self._put_latest({
                "sequence": seq,
                "timestamp_s": captured_at,
                "detector_timestamp_s": detector_timestamp,
                "capture_started_s": started,
                "capture_duration_ms": (captured_at - started) * 1000.0,
                "frame": frame,
            })

    def get_latest(self, timeout_s: float = 0.1) -> dict[str, Any] | None:
        try:
            latest = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest

    @property
    def fps(self) -> float | None:
        if self._started_at is None:
            return None
        end = self._stopped_at or time.monotonic()
        elapsed = end - self._started_at
        return self.capture_count / elapsed if elapsed > 0 else None

    @property
    def max_capture_gap_s(self) -> float | None:
        return float(np.max(self.capture_intervals_s)) if self.capture_intervals_s else None

class APIPosterThread:
    """Posts Pi flash events outside the camera/oscillator loop."""

    def __init__(self, api_base: str, timeout_s: float, debug: bool,
                 queue_size: int = 32) -> None:
        self.api_base = api_base
        self.timeout_s = timeout_s
        self.debug = debug
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="step5a-api-poster", daemon=True)
        self.post_count = 0
        self.errors: list[str] = []
        self.latencies_ms: list[float] = []
        self.events: list[dict[str, Any]] = []
        self.queue_drops = 0
        self.queue_size_max = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    def enqueue(self, timestamp_s: float) -> None:
        payload = {"timestamp": timestamp_s}
        enqueued_at = time.monotonic()
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.queue_drops += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(payload)
            self.queue_size_max = max(self.queue_size_max, self._queue.qsize())
            self.events.append({
                "monotonic_time_s": round(enqueued_at, 6),
                "event": "api_pi_flash_post_enqueued",
                "timestamp_s": timestamp_s,
                "queue_size": self._queue.qsize(),
                "queue_drops": self.queue_drops,
            })
        except queue.Full:
            self.queue_drops += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                payload = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            try:
                _api(self.api_base, "/api/pi_flash", "POST", payload,
                     timeout=self.timeout_s, retries=1, label="Pi flash async",
                     debug=self.debug)
                self.post_count += 1
                ok = 1
                error = ""
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
                ok = 0
                error = f"{type(exc).__name__}: {exc}"
            finally:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                self.latencies_ms.append(elapsed_ms)
                self.events.append({
                    "monotonic_time_s": round(started, 6),
                    "event": "api_pi_flash_post_sent",
                    "timestamp_s": payload.get("timestamp"),
                    "ok": ok,
                    "error": error,
                    "elapsed_ms": round(elapsed_ms, 3),
                })

class AgentPollThread:
    """Quietly samples /api/agents for metrics without blocking camera work."""

    def __init__(self, api_base: str, timeout_s: float, poll_hz: float = 5.0) -> None:
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s
        self.poll_period_s = 1.0 / max(0.1, poll_hz)
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="step5a-agent-poller", daemon=True)
        self._t0: float | None = None

    def start(self, t0: float) -> None:
        self._t0 = t0
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        assert self._t0 is not None
        next_poll = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_poll:
                time.sleep(min(0.02, next_poll - now))
                continue
            try:
                req = urllib.request.Request(self.api_base + "/api/agents", method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read())
                a0 = _first_agent(payload)
                self.rows.append({
                    "t_s": round(time.monotonic() - self._t0, 6),
                    "virtual_frequency_hz": a0.get("frequency_hz"),
                    "phase_rad": a0.get("phase_rad"),
                    "fire_count": a0.get("fire_count"),
                    "received_pi_flashes": a0.get("received_pi_flashes"),
                    "pi_flash_posts_received": a0.get("pi_flash_posts_received"),
                    "pi_flash_events_consumed": a0.get("pi_flash_events_consumed"),
                    "feedback_enabled": a0.get("feedback_enabled"),
                    "flash_on": a0.get("flash_on"),
                })
            except Exception as exc:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            next_poll += self.poll_period_s

def _api(api_base: str, path: str, method: str = "GET", data: dict | None = None,
         timeout: float = 10.0, retries: int = 3, label: str = "",
         debug: bool = False) -> dict:
    """Call a REST endpoint with retry and labelled debug logging."""
    url = api_base.rstrip("/") + path
    endpoint = f"{method} {path}"
    extra = f"[API] {label or endpoint}"
    last_exc = None
    attempts = max(1, retries)
    for attempt in range(1, attempts + 1):
        try:
            print(f"  [API] Calling {endpoint}" + (f" ({label})" if label else ""))
            if attempt > 1:
                print(f"    retry {attempt}/{attempts}")
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(url, data=body,
                                          headers={"Content-Type":"application/json"} if data else {},
                                          method=method)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                result = json.loads(r.read())
            if debug:
                print(f"    → {json.dumps(result)[:200]}")
            return result
        except (TimeoutError, socket.timeout) as e:
            last_exc = e
            print(f"    TIMEOUT: {endpoint} timed out after {timeout:g}s")
            if attempt < attempts:
                time.sleep(1.0)
        except urllib.error.URLError as e:
            last_exc = e
            reason = getattr(e, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                print(f"    TIMEOUT: {endpoint} timed out after {timeout:g}s")
            else:
                print(f"    API ERROR: {type(e).__name__}: {e}")
            if attempt < attempts:
                time.sleep(1.0)
        except Exception as e:
            last_exc = e
            print(f"    ✗ {type(e).__name__}: {e}")
            if attempt < attempts:
                time.sleep(1.0)
    raise ApiError(f"{extra} FAILED after {attempts} attempts: {last_exc}") from last_exc

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

# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5A — Mutual HIL Smoke Test.")
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--model", choices=["eapf_consensus", "kuramoto"],
                        default="eapf_consensus",
                        help="Model family to run on both virtual and Pi sides")
    parser.add_argument("--virtual-freq", type=float, default=2.0)
    parser.add_argument("--pi-freq", type=float, default=1.5)
    parser.add_argument("--kuramoto-gain", type=float, default=5.0,
                        help="Kuramoto coupling gain for Pi-side mutual HIL")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for deterministic random initial phases")
    parser.add_argument("--random-phase", action="store_true",
                        help="Randomise missing initial phases from Uniform(0, 2*pi)")
    parser.add_argument("--virtual-phase-rad", type=float, default=None,
                        help="Explicit virtual initial phase in radians")
    parser.add_argument("--pi-phase-rad", type=float, default=None,
                        help="Explicit Pi oscillator initial phase in radians")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="experiments/logs/step5a_mutual_hil_smoke")
    parser.add_argument("--feedback-off", action="store_true", default=True,
                        help="Disable Pi→virtual feedback (calibration mode, default)")
    parser.add_argument("--feedback-on", action="store_true",
                        help="Enable Pi→virtual feedback (mutual mode)")
    parser.add_argument("--api-timeout", type=float, default=10.0,
                        help="API request timeout in seconds")
    parser.add_argument("--api-retries", type=int, default=3,
                        help="Number of retries on API timeout")
    parser.add_argument("--debug", action="store_true",
                        help="Print verbose API responses")
    parser.add_argument("--detect-only", action="store_true",
                        help="Run HIL camera detection only: no Pi oscillator, GPIO, or Pi feedback POSTs")
    parser.add_argument("--disable-gpio-led", action="store_true",
                        help="Run Pi oscillator but do not drive GPIO LED output")
    parser.add_argument("--no-pi-feedback-post", action="store_true",
                        help="Run Pi oscillator but do not POST Pi flash events to /api/pi_flash")
    # Camera
    parser.add_argument("--width", type=int, default=640,
                        help="Legacy alias for camera width if --camera-width is not set")
    parser.add_argument("--height", type=int, default=480,
                        help="Legacy alias for camera height if --camera-height is not set")
    parser.add_argument("--camera-width", type=int, default=None,
                        help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=None,
                        help="Camera capture height")
    parser.add_argument("--camera-fps", type=float, default=30.0,
                        help="Requested camera FPS / fixed FrameDurationLimits")
    parser.add_argument("--camera-exposure-us", type=int, default=None,
                        help="Optional fixed Picamera2 exposure time in microseconds")
    parser.add_argument("--camera-gain", type=float, default=None,
                        help="Optional fixed Picamera2 analogue gain")
    parser.add_argument("--roi", type=_parse_roi, default=None,
                        help="Optional detector ROI as x,y,w,h")
    parser.add_argument("--expected-blob-x", type=float, default=None,
                        help="Expected virtual-dot blob centre x in full-frame pixels")
    parser.add_argument("--expected-blob-y", type=float, default=None,
                        help="Expected virtual-dot blob centre y in full-frame pixels")
    parser.add_argument("--expected-blob-radius-px", type=float, default=None,
                        help="Reject accepted detector events outside this radius")
    parser.add_argument("--capture-queue-size", type=int, default=2,
                        help="Latest-frame capture queue size; full queues drop oldest")
    parser.add_argument("--api-queue-size", type=int, default=32,
                        help="Async /api/pi_flash queue size; full queues drop oldest")
    parser.add_argument("--agent-poll-hz", type=float, default=5.0,
                        help="Background /api/agents polling rate for final-window metrics")
    parser.add_argument("--led-pulse-duration", type=float, default=0.06,
                        help="GPIO LED pulse duration in seconds, scheduled non-blocking")
    parser.add_argument("--self-flash-blanking-s", type=float, default=0.0,
                        help="Ignore accepted detector events shortly after Pi GPIO LED flashes")
    parser.add_argument("--debug-trace", action="store_true",
                        help="Write event_trace.csv and debug frames for accepted/suspicious events")
    parser.add_argument("--episode-latch", dest="episode_latch", action="store_true", default=True,
                        help="Enable one-event-per-flash episode latch (default for Step 5 HIL)")
    parser.add_argument("--no-episode-latch", dest="episode_latch", action="store_false",
                        help="Disable episode latch and use legacy detector events")
    parser.add_argument("--rearm-off-duration", type=float, default=0.05,
                        help="Seconds below OFF threshold required to re-arm the episode latch")
    parser.add_argument("--off-frames-to-rearm", type=int, default=2,
                        help="Consecutive OFF frames required to re-arm the episode latch")
    parser.add_argument("--rearm-requires-both", dest="rearm_requires_both", action="store_true", default=True,
                        help="Require both below-OFF duration and OFF-frame count before re-arming")
    parser.add_argument("--rearm-any", dest="rearm_requires_both", action="store_false",
                        help="Legacy re-arm rule: duration OR OFF-frame count")
    parser.add_argument("--min-interval", type=float, default=0.35)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--threshold-on", type=float, default=180.0)
    parser.add_argument("--threshold-off", type=float, default=120.0)
    parser.add_argument("--norm-on-threshold", type=float, default=0.65)
    parser.add_argument("--norm-off-threshold", type=float, default=0.35)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    args = parser.parse_args()

    camera_width = args.camera_width if args.camera_width is not None else args.width
    camera_height = args.camera_height if args.camera_height is not None else args.height
    frame_duration_us = int(round(1_000_000.0 / args.camera_fps)) if args.camera_fps > 0 else None
    camera_frame_duration_limits = (
        (frame_duration_us, frame_duration_us)
        if frame_duration_us is not None else None
    )

    _ensure_hw(dry=args.dry_run)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.log_dir) / f"{ts}_step5a_mutual_hil_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = args.leader_api.rstrip("/")
    to = args.api_timeout; rt = args.api_retries; dbg = args.debug
    feedback_on = False if args.detect_only else args.feedback_on
    gpio_enabled = not args.detect_only and not args.disable_gpio_led
    pi_feedback_post_enabled = not args.detect_only and not args.no_pi_feedback_post
    virtual_initial_phase_rad, pi_initial_phase_rad, initial_phase_difference_rad = _resolve_initial_phases(
        args.seed,
        args.random_phase,
        args.virtual_phase_rad,
        args.pi_phase_rad,
    )
    print("[CONFIG] Detector comparison notes:")
    print("  Alignment check: PicameraFlashDetector resolution=640x480 default, "
          "detection_mode=local_contrast, min_interval=0.2, window_s=5.0, "
          "threshold_on/off=180/120, norm_on/off=0.65/0.35, min_amplitude=10, ROI=None")
    print(f"  HIL smoke:       PicameraFlashDetector resolution={camera_width}x{camera_height}, "
          f"detection_mode=local_contrast, min_interval={args.min_interval}, "
          f"window_s={args.window_s}, threshold_on/off={args.threshold_on}/{args.threshold_off}, "
          f"norm_on/off={args.norm_on_threshold}/{args.norm_off_threshold}, "
          f"min_amplitude={args.min_amplitude}, ROI={args.roi}")
    print(f"  Episode latch: enabled={args.episode_latch} "
          f"min_interval={args.min_interval} "
          f"rearm_off_duration={args.rearm_off_duration} "
          f"off_frames_to_rearm={args.off_frames_to_rearm} "
          f"rearm_requires_both={args.rearm_requires_both}")
    print(f"  Spatial/blanking diagnostics: roi={args.roi} "
          f"expected_blob=({args.expected_blob_x},{args.expected_blob_y}) "
          f"radius={args.expected_blob_radius_px} "
          f"self_flash_blanking_s={args.self_flash_blanking_s}")
    print(f"  Camera timing: requested_fps={args.camera_fps} "
          f"FrameDurationLimits={camera_frame_duration_limits} "
          f"exposure_us={args.camera_exposure_us} gain={args.camera_gain}")
    print(f"  Isolation: detect_only={args.detect_only} "
          f"gpio_enabled={gpio_enabled} pi_feedback_post_enabled={pi_feedback_post_enabled}")
    print(f"  Initial phases: seed={args.seed} random_phase={args.random_phase} "
          f"virtual={virtual_initial_phase_rad:.6f} rad "
          f"pi={pi_initial_phase_rad:.6f} rad "
          f"diff(pi-virtual)={initial_phase_difference_rad:.6f} rad")
    virtual_fire_count_start: int | None = None
    virtual_fire_count_end: int | None = None
    virtual_frequency_start: float | None = None
    virtual_frequency_end: float | None = None
    received_pi_flashes: int | None = None
    pi_flash_posts_received_start: int | None = None
    pi_flash_posts_received_end: int | None = None
    pi_flash_events_consumed_start: int | None = None
    pi_flash_events_consumed_end: int | None = None
    running_start: bool | None = None
    running_end: bool | None = None
    feedback_enabled_start: bool | None = None
    feedback_enabled_end: bool | None = None

    # ── Resilient setup sequence ──────────────────────────────────
    if not args.dry_run:
        print(f"\n[SETUP] Configuring mutual HIL at {api}")

        # 1. Switch to mutual_hil mode
        _api(api, "/api/mode", "POST", {"mode": "mutual_hil"},
             timeout=to, retries=rt, label="POST /api/mode", debug=dbg)

        # 2. Pause (best-effort — clean any previous running state)
        try:
            _api(api, "/api/pause", "POST", timeout=to, retries=1, label="POST /api/pause (best-effort)")
        except ApiError:
            pass

        # 3. Reset to clean state
        _api(api, "/api/reset", "POST",
             timeout=to, retries=rt, label="POST /api/reset", debug=dbg)

        # 4. Configure agent 0
        _api(api, "/api/agents/0", "POST", {
            "initial_frequency_hz": args.virtual_freq,
            "frequency_hz": args.virtual_freq,
            "initial_phase_rad": virtual_initial_phase_rad,
            "phase_rad": virtual_initial_phase_rad,
            "x": 800, "y": 400, "size": 450,
            "model": args.model,
            "kuramoto_gain": args.kuramoto_gain,
        }, timeout=to, retries=rt, label="POST /api/agents/0", debug=dbg)

        # 5. Set feedback mode
        _api(api, "/api/feedback", "POST", {"enabled": feedback_on},
             timeout=to, retries=rt,
             label=f"POST /api/feedback enabled={feedback_on}", debug=dbg)

        # 6. Start virtual agent
        _api(api, "/api/start", "POST",
             timeout=to, retries=rt, label="POST /api/start", debug=dbg)

        # 7. Verify
        st = _api(api, "/api/agents", "GET",
                  timeout=to, retries=rt, label="GET /api/agents (verify)", debug=dbg)
        a0 = _first_agent(st)
        virtual_fire_count_start = _as_int(a0.get("fire_count"))
        virtual_frequency_start = _as_float(a0.get("frequency_hz"))
        pi_flash_posts_received_start = _as_int(a0.get("pi_flash_posts_received"))
        pi_flash_events_consumed_start = _as_int(a0.get("pi_flash_events_consumed"))
        running_start = a0.get("running")
        feedback_enabled_start = a0.get("feedback_enabled")
        print(f"  Verify: running={a0.get('running')} "
              f"feedback_enabled={a0.get('feedback_enabled')} "
              f"frequency_hz={a0.get('frequency_hz')} "
              f"fire_count={a0.get('fire_count')}")
        print(f"  Setup complete. Starting {args.duration}s trial.\n")
    else:
        print("[DRY-RUN] Would configure mutual_hil mode")

    # ── Pi oscillator ─────────────────────────────────────────────
    if args.detect_only:
        osc = None
        phase_estimator = None
    elif args.model == "eapf_consensus":
        from firefly_sync.core.event_based_consensus_pll import EventBasedConsensusPLLOscillator
        osc = EventBasedConsensusPLLOscillator()
        for k, v in {"phase_gain": 0.02, "frequency_gain": 0.02, "phase_error_filter_alpha": 0.2,
                      "frequency_error_filter_alpha": 0.2, "max_phase_step_rad": 0.2,
                      "max_frequency_step_hz": 0.05, "frequency_min_hz": 0.5,
                      "frequency_max_hz": 4.0}.items():
            setattr(osc.config, k, v)
        osc._phase_rad = pi_initial_phase_rad
        osc._frequency_hz = args.pi_freq
        osc._omega_rad_s = 2.0 * np.pi * args.pi_freq
        phase_estimator = None
    else:
        from firefly_sync.core.kuramoto import KuramotoModel
        osc = KuramotoModel(
            natural_frequency=2.0 * np.pi * args.pi_freq,
            initial_phase=pi_initial_phase_rad,
            coupling_strength=args.kuramoto_gain,
            dt=0.01,
        )
        phase_estimator = FlashPhaseEstimator(
            initial_period_guess_s=1.0 / args.virtual_freq if args.virtual_freq > 0 else 0.5,
        )

    # ── Hardware ───────────────────────────────────────────────────
    detector = None; led = None; capture_thread = None; api_poster = None; agent_poller = None
    camera_config_actual = None
    camera_controls_requested = {}
    if not args.dry_run:
        assert _PicameraFlashDetector is not None
        detector = _PicameraFlashDetector(resolution=[camera_width, camera_height],
                                           detection_mode="local_contrast",
                                           roi=args.roi,
                                           threshold_on=args.threshold_on,
                                           threshold_off=args.threshold_off,
                                           min_interval_s=args.min_interval,
                                           window_s=args.window_s,
                                           norm_on_threshold=args.norm_on_threshold,
                                           norm_off_threshold=args.norm_off_threshold,
                                           min_amplitude=args.min_amplitude,
                                           target_fps=int(round(args.camera_fps)),
                                           use_video_config=True,
                                           frame_duration_limits=camera_frame_duration_limits,
                                           frame_rate=args.camera_fps,
                                           exposure_time_us=args.camera_exposure_us,
                                           analogue_gain=args.camera_gain,
                                           camera_format="BGR888",
                                           episode_latch_enabled=args.episode_latch,
                                           rearm_off_duration_s=args.rearm_off_duration,
                                           off_frames_to_rearm=args.off_frames_to_rearm,
                                           rearm_requires_both=args.rearm_requires_both)
        detector.start()
        camera_config_actual = _jsonable(getattr(detector, "camera_config_actual", None))
        camera_controls_requested = _jsonable(getattr(detector, "camera_controls_requested", {}))
        print(f"[CAMERA] Requested controls: {camera_controls_requested}")
        print(f"[CAMERA] Actual config: {camera_config_actual}")
        if gpio_enabled:
            led = _PiGPIOLED(pin=17, flash_duration_s=args.led_pulse_duration)
        if pi_feedback_post_enabled:
            api_poster = APIPosterThread(
                api_base=api,
                timeout_s=min(args.api_timeout, 3.0),
                debug=dbg,
                queue_size=args.api_queue_size,
            )
            api_poster.start()
        agent_poller = AgentPollThread(
            api_base=api,
            timeout_s=min(args.api_timeout, 3.0),
            poll_hz=args.agent_poll_hz,
        )

    # ── Trial loop ─────────────────────────────────────────────────
    pi_flash_times: list[float] = []
    virtual_detected: list[float] = []
    osc_log: list[dict] = []
    det_log: list[dict] = []
    flash_events: list[dict] = []
    event_trace: list[dict] = []
    detector_events: list[dict] = []
    oscillator_events: list[dict] = []
    suspicious_rows: list[dict] = []
    frame_history: list[dict[str, Any]] = []
    pending_debug_frame_saves: list[dict[str, Any]] = []
    loop_intervals_s: list[float] = []
    camera_times_ms: list[float] = []
    gpio_flash_count = 0
    led_off_time_s: float | None = None
    trial_start = time.monotonic()
    prev_process = trial_start
    prev_model_t = trial_start
    last_v_event_t = -999.0
    last_processed_sequence: int | None = None
    trial_end_time: float | None = None
    printed_frame_format_debug = False
    last_frame_format_debug: dict[str, Any] = {}
    duplicate_suppressed_count = 0
    raw_on_threshold_crossing_count = 0
    accepted_flash_event_count = 0
    loop_index = 0
    last_pi_flash_time = -999.0
    suspicious_warning_count = 0
    short_interval_warning_count = 0
    self_flash_blank_reject_count = 0
    spatial_gate_reject_count = 0
    debug_dir = out_dir / "debug_frames"

    print(f"[TRIAL] {args.duration}s, virtual={args.virtual_freq}Hz, pi={args.pi_freq}Hz, feedback={feedback_on}")

    try:
        if detector is not None:
            capture_thread = CameraCaptureThread(detector, queue_size=args.capture_queue_size)
            capture_thread.start()
        trial_start = time.monotonic()
        if agent_poller is not None:
            agent_poller.start(trial_start)
        prev_process = trial_start
        prev_model_t = trial_start
        while (time.monotonic() - trial_start) < args.duration:
            loop_index += 1
            gpio_led_off_event = False
            gpio_led_on_event = False
            api_pi_flash_post_enqueued = False
            loop_now = time.monotonic()
            if led is not None and led_off_time_s is not None and loop_now >= led_off_time_s:
                try:
                    led.off()
                    gpio_led_off_event = True
                except Exception:
                    pass
                led_off_time_s = None

            frame_payload: dict[str, Any] | None = None
            if not args.dry_run and capture_thread is not None:
                frame_payload = capture_thread.get_latest(timeout_s=0.1)
                if frame_payload is None:
                    continue
                if frame_payload["sequence"] == last_processed_sequence:
                    continue
                last_processed_sequence = frame_payload["sequence"]
                now = float(frame_payload["timestamp_s"])
                detector_started = time.monotonic()
                res = detector.process_frame(
                    frame_payload["frame"],
                    now_s=float(frame_payload["detector_timestamp_s"]),
                )
                last_frame_format_debug = {
                    "raw_frame_shape": res.get("raw_frame_shape"),
                    "raw_frame_dtype": res.get("raw_frame_dtype"),
                    "normalized_frame_shape": res.get("normalized_frame_shape"),
                    "normalized_frame_dtype": res.get("normalized_frame_dtype"),
                    "camera_format_requested": res.get("camera_format_requested"),
                }
                if not printed_frame_format_debug:
                    print("[CAMERA] First frame format: "
                          f"raw_shape={last_frame_format_debug['raw_frame_shape']} "
                          f"raw_dtype={last_frame_format_debug['raw_frame_dtype']} "
                          f"normalized_shape={last_frame_format_debug['normalized_frame_shape']} "
                          f"normalized_dtype={last_frame_format_debug['normalized_frame_dtype']} "
                          f"requested_format={last_frame_format_debug['camera_format_requested']}")
                    printed_frame_format_debug = True
                cam_ms = (time.monotonic() - detector_started) * 1000.0
                capture_duration_ms = float(frame_payload["capture_duration_ms"])
                frame_history.append({
                    "t_s": None,
                    "sequence": frame_payload.get("sequence"),
                    "frame": frame_payload.get("frame"),
                })
                if len(frame_history) > 4:
                    frame_history.pop(0)
            else:
                time.sleep(0.01)
                now = time.monotonic()
                res = {"state": "OFF", "brightness_used": 0, "signal_norm": 0}
                cam_ms = 0.0
                capture_duration_ms = 0.0

            process_now = time.monotonic()
            process_dt = process_now - prev_process
            prev_process = process_now
            dt = max(1e-6, now - prev_model_t)
            prev_model_t = now
            t = max(0.0, now - trial_start)
            if frame_history:
                frame_history[-1]["t_s"] = t
            if args.debug_trace and frame_payload is not None and pending_debug_frame_saves:
                remaining_pending: list[dict[str, Any]] = []
                for pending in pending_debug_frame_saves:
                    idx = pending["next_index"]
                    _save_debug_frame(
                        debug_dir / f"{pending['event_t']:09.4f}_{pending['event_label']}_{pending['safe_reason']}_{pending['suffix']}_after{idx}.png",
                        frame_payload["frame"],
                    )
                    if idx < 2:
                        pending["next_index"] = idx + 1
                        remaining_pending.append(pending)
                pending_debug_frame_saves = remaining_pending
            loop_intervals_s.append(process_dt)
            camera_times_ms.append(cam_ms)
            duplicate_suppressed_count = _as_int(res.get("duplicate_suppressed_count")) or duplicate_suppressed_count
            raw_on_threshold_crossing_count = _as_int(res.get("raw_on_threshold_crossing_count")) or raw_on_threshold_crossing_count
            accepted_flash_event_count = _as_int(res.get("accepted_flash_event_count")) or accepted_flash_event_count
            detector_accepted_event = bool(res.get("accepted_flash_event"))
            virtual_event = detector_accepted_event
            detector_reject_reason = ""
            accepted_interval = (
                t - last_v_event_t
                if detector_accepted_event and last_v_event_t > -900 else None
            )
            if detector_accepted_event and args.self_flash_blanking_s > 0:
                since_pi_flash = t - last_pi_flash_time
                if 0.0 <= since_pi_flash < args.self_flash_blanking_s:
                    virtual_event = False
                    detector_reject_reason = f"self_flash_blanking:{since_pi_flash:.4f}s"
                    self_flash_blank_reject_count += 1
            spatial_ok, spatial_reason = _blob_inside_expected_gate(
                res,
                args.expected_blob_x,
                args.expected_blob_y,
                args.expected_blob_radius_px,
            )
            if detector_accepted_event and virtual_event and not spatial_ok:
                virtual_event = False
                detector_reject_reason = spatial_reason
                spatial_gate_reject_count += 1
            if detector_accepted_event and virtual_event:
                detector_reject_reason = "accepted"
            elif detector_accepted_event and not detector_reject_reason:
                detector_reject_reason = "rejected_by_external_gate"
            elif res.get("detector_raw_on_crossing") and res.get("detector_reject_reason"):
                detector_reject_reason = str(res.get("detector_reject_reason"))
            impossible_interval_warning = (
                accepted_interval is not None and accepted_interval < 0.40
            )
            if impossible_interval_warning:
                short_interval_warning_count += 1
                suspicious_warning_count += 1
            if args.debug_trace and (detector_accepted_event or impossible_interval_warning or detector_reject_reason):
                bx = _as_float(res.get("selected_blob_x"))
                by = _as_float(res.get("selected_blob_y"))
                area = _as_int(res.get("selected_blob_area"))
                suffix = (
                    f"blob_x{bx:.0f}_y{by:.0f}_area{area}"
                    if bx is not None and by is not None and area is not None
                    else "blob_unknown"
                )
                event_label = "accepted" if virtual_event else ("rejected" if detector_accepted_event else "raw")
                safe_reason = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in detector_reject_reason)[:60]
                current_frame = frame_payload.get("frame") if frame_payload else None
                if current_frame is not None:
                    _save_debug_frame(
                        debug_dir / f"{t:09.4f}_{event_label}_{safe_reason}_{suffix}_current.png",
                        current_frame,
                    )
                if len(frame_history) >= 2 and frame_history[-2].get("frame") is not None:
                    _save_debug_frame(
                        debug_dir / f"{t:09.4f}_{event_label}_{safe_reason}_{suffix}_prev.png",
                        frame_history[-2]["frame"],
                    )
                if frame_payload is not None:
                    pending_debug_frame_saves.append({
                        "event_t": t,
                        "event_label": event_label,
                        "safe_reason": safe_reason,
                        "suffix": suffix,
                        "next_index": 1,
                    })
            if not args.dry_run and detector is not None:
                ds = (res.get("state") == "ON")
                if virtual_event:
                    last_v_event_t = t
                det_log.append({"t_s": round(t,6), "state": int(ds),
                                "virtual_event": int(virtual_event),
                                "event_type": res.get("event_type"),
                                "accepted_flash_event": int(bool(res.get("accepted_flash_event"))),
                                "detector_accepted_after_gates": int(virtual_event),
                                "reason_for_accept_reject": detector_reject_reason,
                                "accepted_inter_event_interval_s": accepted_interval,
                                "impossible_interval_warning": int(impossible_interval_warning),
                                "duplicate_suppressed_count": res.get("duplicate_suppressed_count"),
                                "raw_on_threshold_crossing_count": res.get("raw_on_threshold_crossing_count"),
                                "accepted_flash_event_count": res.get("accepted_flash_event_count"),
                                "episode_armed": res.get("episode_armed"),
                                "episode_currently_on": res.get("episode_currently_on"),
                                "detector_rearmed_event": res.get("detector_rearmed_event"),
                                "detector_time_since_last_accepted": res.get("detector_time_since_last_accepted"),
                                "detector_below_off_duration": res.get("detector_below_off_duration"),
                                "detector_off_frame_count": res.get("detector_off_frame_count"),
                                "detector_raw_on_crossing": res.get("detector_raw_on_crossing"),
                                "latch_signal": res.get("latch_signal"),
                                "selected_blob_x": res.get("selected_blob_x"),
                                "selected_blob_y": res.get("selected_blob_y"),
                                "selected_blob_area": res.get("selected_blob_area"),
                                "selected_blob_mean": res.get("selected_blob_mean"),
                                "selected_blob_max": res.get("selected_blob_max"),
                                "selected_blob_bbox": res.get("selected_blob_bbox_full"),
                                "detected_blob_count": res.get("blob_count"),
                                "brightness": res.get("brightness_used",0),
                                "signal_norm": res.get("signal_norm",0),
                                "camera_processing_time_ms": round(cam_ms, 3),
                                "capture_duration_ms": round(capture_duration_ms, 3),
                                "raw_frame_shape": res.get("raw_frame_shape"),
                                "raw_frame_dtype": res.get("raw_frame_dtype"),
                                "normalized_frame_shape": res.get("normalized_frame_shape"),
                                "normalized_frame_dtype": res.get("normalized_frame_dtype"),
                                "loop_dt_ms": round(process_dt * 1000.0, 3),
                                "model_dt_ms": round(dt * 1000.0, 3),
                                "capture_sequence": frame_payload.get("sequence") if frame_payload else None})
            else:
                det_log.append({"t_s": round(t,6), "state": 0, "virtual_event": 0,
                                "brightness": 0, "signal_norm": 0,
                                "camera_processing_time_ms": 0.0,
                                "capture_duration_ms": round(capture_duration_ms, 3),
                                "raw_frame_shape": None,
                                "raw_frame_dtype": None,
                                "normalized_frame_shape": None,
                                "normalized_frame_dtype": None,
                                "loop_dt_ms": round(process_dt * 1000.0, 3),
                                "model_dt_ms": round(dt * 1000.0, 3),
                                "capture_sequence": None})
            if (
                detector_accepted_event
                or res.get("detector_raw_on_crossing")
                or res.get("detector_rearmed_event")
                or res.get("detector_reject_reason")
            ):
                detector_events.append({
                    "t_s": round(t, 6),
                    "monotonic_time_s": round(now, 6),
                    "event": (
                        "accepted"
                        if virtual_event else "detector_event_rejected"
                        if detector_accepted_event else "raw_or_rearm"
                    ),
                    "reason_for_accept_reject": detector_reject_reason,
                    "detector_signal_value": res.get("latch_signal"),
                    "detector_norm_value": res.get("signal_norm"),
                    "detector_is_on_raw": int(res.get("state") == "ON"),
                    "detector_raw_on_crossing": int(bool(res.get("detector_raw_on_crossing"))),
                    "detector_latch_armed": res.get("episode_armed"),
                    "detector_rearmed_event": int(bool(res.get("detector_rearmed_event"))),
                    "detector_accepted_flash_event": int(detector_accepted_event),
                    "detector_duplicate_suppressed": res.get("duplicate_suppressed_count"),
                    "detector_time_since_last_accepted": res.get("detector_time_since_last_accepted"),
                    "detector_below_off_duration": res.get("detector_below_off_duration"),
                    "detector_off_frame_count": res.get("detector_off_frame_count"),
                    "detected_blob_count": res.get("blob_count"),
                    "selected_blob_x": res.get("selected_blob_x"),
                    "selected_blob_y": res.get("selected_blob_y"),
                    "selected_blob_area": res.get("selected_blob_area"),
                    "selected_blob_mean": res.get("selected_blob_mean"),
                    "selected_blob_max": res.get("selected_blob_max"),
                    "selected_blob_bbox": res.get("selected_blob_bbox_full"),
                    "roi_used": args.roi is not None,
                    "roi": args.roi,
                })
            if virtual_event:
                virtual_detected.append(t)
                flash_events.append({"t_s": round(t,6), "event": "virtual_flash"})

            if args.detect_only:
                event_trace.append({
                    "monotonic_time_s": round(now, 6),
                    "wall_time_s": round(time.time(), 6),
                    "t_s": round(t, 6),
                    "loop_index": loop_index,
                    "camera_frame_time_s": round(now, 6),
                    "detector_signal_value": res.get("latch_signal"),
                    "detector_norm_value": res.get("signal_norm"),
                    "detector_is_on_raw": int(res.get("state") == "ON"),
                    "detector_raw_on_crossing": int(bool(res.get("detector_raw_on_crossing"))),
                    "detector_latch_armed": res.get("episode_armed"),
                    "detector_rearmed_event": int(bool(res.get("detector_rearmed_event"))),
                    "detector_accepted_flash_event": int(detector_accepted_event),
                    "detector_accepted_after_gates": int(virtual_event),
                    "reason_for_accept_reject": detector_reject_reason,
                    "detected_blob_count": res.get("blob_count"),
                    "selected_blob_x": res.get("selected_blob_x"),
                    "selected_blob_y": res.get("selected_blob_y"),
                    "selected_blob_area": res.get("selected_blob_area"),
                    "selected_blob_mean": res.get("selected_blob_mean"),
                    "selected_blob_max": res.get("selected_blob_max"),
                    "selected_blob_bbox": res.get("selected_blob_bbox_full"),
                })
                continue

            if args.model == "eapf_consensus":
                nids = [0] if virtual_event else []
                r = osc.step(dt_s=dt, t_s=t, neighbour_flash_ids=nids)
                phase_rad = r["phase_rad"]
                freq_hz = r["frequency_hz"]
                phase_err = r["phase_error_rad"]
                follower_flash = r["follower_flash_event"]
                coupling_term = ""
            else:
                assert phase_estimator is not None
                if virtual_event:
                    phase_estimator.record_flash(t)
                est_phase = phase_estimator.estimate_phase(t)
                phase_err = float(np.arctan2(
                    np.sin(est_phase - osc.phase),
                    np.cos(est_phase - osc.phase),
                ))
                coupling_term = float(np.sin(est_phase - osc.phase))
                saved_dt = osc.dt
                osc.dt = min(dt, 0.1)
                state = osc.step(coupling_term)
                osc.dt = saved_dt
                phase_rad = float(osc.phase)
                freq_hz = float(max(0.0, (osc.natural_frequency + osc.coupling_strength * coupling_term) / (2.0 * np.pi)))
                follower_flash = bool(state.is_firing)

            osc_log.append({"t_s": round(t,6), "phase_rad": round(phase_rad, 6),
                            "freq_hz": round(freq_hz, 6),
                            "phase_err": round(phase_err, 6),
                            "coupling_term": coupling_term,
                            "flash": int(follower_flash)})
            latest_agent_row = agent_poller.rows[-1] if agent_poller is not None and agent_poller.rows else {}
            trace_row = {
                "monotonic_time_s": round(now, 6),
                "wall_time_s": round(time.time(), 6),
                "t_s": round(t, 6),
                "loop_index": loop_index,
                "virtual_frequency_hz": latest_agent_row.get("virtual_frequency_hz"),
                "virtual_fire_count": latest_agent_row.get("fire_count"),
                "virtual_flash_on": latest_agent_row.get("flash_on"),
                "pi_oscillator_phase_rad": round(phase_rad, 6),
                "pi_frequency_hz": round(freq_hz, 6),
                "pi_flash_event_emitted": int(bool(follower_flash)),
                "gpio_led_on_event": 0,
                "gpio_led_off_event": int(gpio_led_off_event),
                "api_pi_flash_post_enqueued": 0,
                "api_pi_flash_post_sent": "",
                "camera_frame_time_s": round(now, 6),
                "detector_signal_value": res.get("latch_signal"),
                "detector_norm_value": res.get("signal_norm"),
                "detector_is_on_raw": int(res.get("state") == "ON"),
                "detector_raw_on_crossing": int(bool(res.get("detector_raw_on_crossing"))),
                "detector_latch_armed": res.get("episode_armed"),
                "detector_rearmed_event": int(bool(res.get("detector_rearmed_event"))),
                "detector_accepted_flash_event": int(detector_accepted_event),
                "detector_accepted_after_gates": int(virtual_event),
                "detector_duplicate_suppressed": res.get("duplicate_suppressed_count"),
                "detector_time_since_last_accepted": res.get("detector_time_since_last_accepted"),
                "detector_below_off_duration": res.get("detector_below_off_duration"),
                "detector_off_frame_count": res.get("detector_off_frame_count"),
                "detected_blob_count": res.get("blob_count"),
                "selected_blob_x": res.get("selected_blob_x"),
                "selected_blob_y": res.get("selected_blob_y"),
                "selected_blob_area": res.get("selected_blob_area"),
                "selected_blob_mean": res.get("selected_blob_mean"),
                "selected_blob_max": res.get("selected_blob_max"),
                "selected_blob_bbox": res.get("selected_blob_bbox_full"),
                "selected_blob_source": "",
                "roi_used": args.roi is not None,
                "roi": args.roi,
                "reason_for_accept_reject": detector_reject_reason,
                "accepted_inter_event_interval_s": accepted_interval,
                "impossible_interval_warning": int(impossible_interval_warning),
            }
            if follower_flash:
                pi_flash_times.append(t)
                last_pi_flash_time = t
                flash_events.append({"t_s": round(t,6), "event": "pi_flash"})
                if led is not None:
                    gpio_flash_count += 1
                    try:
                        led.on()
                        led_off_time_s = time.monotonic() + args.led_pulse_duration
                        gpio_led_on_event = True
                    except Exception:
                        pass
                if not args.dry_run and pi_feedback_post_enabled and api_poster is not None:
                    api_poster.enqueue(t)
                    api_pi_flash_post_enqueued = True
                oscillator_events.append({
                    "t_s": round(t, 6),
                    "event": "pi_flash",
                    "pi_frequency_hz": round(freq_hz, 6),
                    "phase_rad": round(phase_rad, 6),
                    "gpio_led_on_event": int(gpio_led_on_event),
                    "api_pi_flash_post_enqueued": int(api_pi_flash_post_enqueued),
                })
            trace_row["gpio_led_on_event"] = int(gpio_led_on_event)
            trace_row["api_pi_flash_post_enqueued"] = int(api_pi_flash_post_enqueued)
            event_trace.append(trace_row)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        trial_end_time = time.monotonic()
        if led is not None:
            try:
                led.off()
            except Exception:
                pass
        if capture_thread is not None:
            capture_thread.stop()
        if agent_poller is not None:
            agent_poller.stop()
        if api_poster is not None:
            api_poster.stop()
        if not args.dry_run:
            try:
                end_state = _api(api, "/api/agents", "GET", timeout=to,
                                 retries=rt, label="GET /api/agents (end-state)", debug=dbg)
                end_a0 = _first_agent(end_state)
                virtual_fire_count_end = _as_int(end_a0.get("fire_count"))
                virtual_frequency_end = _as_float(end_a0.get("frequency_hz"))
                received_pi_flashes = _as_int(end_a0.get("received_pi_flashes"))
                pi_flash_posts_received_end = _as_int(end_a0.get("pi_flash_posts_received"))
                pi_flash_events_consumed_end = _as_int(end_a0.get("pi_flash_events_consumed"))
                running_end = end_a0.get("running")
                feedback_enabled_end = end_a0.get("feedback_enabled")
            except Exception as e:
                print(f"[WARN] Could not read end-state before cleanup pause: {e}")
        # Cleanup — pause virtual agent
        if not args.dry_run:
            try:
                cleanup = _api(api, "/api/pause", "POST", timeout=to,
                               retries=rt, label="POST /api/pause (cleanup)", debug=dbg)
                print(f"[Cleanup] POST /api/pause result: {cleanup}")
            except Exception as e:
                print(f"[Cleanup] POST /api/pause failed: {e}")
        if detector: detector.stop()
        if led: led.close()

    # ── Metrics ────────────────────────────────────────────────────
    nominal_expected = round(args.duration * args.virtual_freq)
    virtual_det = len(virtual_detected)
    wall = (trial_end_time if trial_end_time is not None else time.monotonic()) - trial_start
    if args.detect_only:
        pi_final_frequency_hz = None
    elif args.model == "eapf_consensus":
        pi_final_frequency_hz = float(osc.frequency_hz)
    else:
        pi_final_frequency_hz = (
            _estimate_flash_frequency(pi_flash_times)
            or float(osc.natural_frequency / (2.0 * np.pi))
        )
    if virtual_fire_count_start is not None and virtual_fire_count_end is not None:
        actual_virtual_flashes = virtual_fire_count_end - virtual_fire_count_start
    else:
        actual_virtual_flashes = None
    nominal_fcr = virtual_det / nominal_expected if nominal_expected > 0 else None
    if actual_virtual_flashes is not None and actual_virtual_flashes > 0:
        actual_fcr = virtual_det / actual_virtual_flashes
    else:
        actual_fcr = None
        print("[WARN] Actual virtual flash count unavailable or zero; actual FCR set to null.")
    frequency_error_hz = (
        abs(pi_final_frequency_hz - virtual_frequency_end)
        if pi_final_frequency_hz is not None and virtual_frequency_end is not None else None
    )
    agent_poll_rows = agent_poller.rows if agent_poller is not None else []
    final_5s_start = max(0.0, args.duration - 5.0)
    pi_freq_final_5s_values = [
        float(row["freq_hz"])
        for row in osc_log
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_5s_start
        and _as_float(row.get("freq_hz")) is not None
    ]
    virtual_freq_final_5s_rows = [
        row for row in agent_poll_rows
        if _as_float(row.get("t_s")) is not None
        and float(row["t_s"]) >= final_5s_start
        and _as_float(row.get("virtual_frequency_hz")) is not None
    ]
    virtual_freq_final_5s_values = [
        float(row["virtual_frequency_hz"]) for row in virtual_freq_final_5s_rows
    ]
    virtual_freq_final_5s_mean = (
        float(np.mean(virtual_freq_final_5s_values)) if virtual_freq_final_5s_values else None
    )
    pi_freq_final_5s_mean = (
        float(np.mean(pi_freq_final_5s_values)) if pi_freq_final_5s_values else None
    )
    virtual_freq_final_5s_std = (
        float(np.std(virtual_freq_final_5s_values, ddof=1))
        if len(virtual_freq_final_5s_values) > 1 else 0.0 if virtual_freq_final_5s_values else None
    )
    pi_freq_final_5s_std = (
        float(np.std(pi_freq_final_5s_values, ddof=1))
        if len(pi_freq_final_5s_values) > 1 else 0.0 if pi_freq_final_5s_values else None
    )
    paired_final_5s_errors: list[float] = []
    if virtual_freq_final_5s_rows and pi_freq_final_5s_values:
        virtual_pairs = [
            (float(row["t_s"]), float(row["virtual_frequency_hz"]))
            for row in virtual_freq_final_5s_rows
        ]
        for row in osc_log:
            t_val = _as_float(row.get("t_s"))
            pi_val = _as_float(row.get("freq_hz"))
            if t_val is None or pi_val is None or t_val < final_5s_start:
                continue
            _, nearest_virtual_freq = min(virtual_pairs, key=lambda item: abs(item[0] - t_val))
            paired_final_5s_errors.append(abs(pi_val - nearest_virtual_freq))
    frequency_error_final_5s_mean_abs = (
        float(np.mean(paired_final_5s_errors)) if paired_final_5s_errors else None
    )
    frequency_error_final_5s_abs_of_means = (
        abs(pi_freq_final_5s_mean - virtual_freq_final_5s_mean)
        if pi_freq_final_5s_mean is not None and virtual_freq_final_5s_mean is not None else None
    )
    api_post_latencies_ms = api_poster.latencies_ms if api_poster is not None else []
    api_post_count = api_poster.post_count if api_poster is not None else 0
    api_post_errors = api_poster.errors if api_poster is not None else []
    api_queue_drops = api_poster.queue_drops if api_poster is not None else 0
    api_queue_size_max = api_poster.queue_size_max if api_poster is not None else 0
    capture_thread_fps = capture_thread.fps if capture_thread is not None else None
    capture_queue_drops = capture_thread.capture_queue_drops if capture_thread is not None else 0
    max_capture_gap_s = capture_thread.max_capture_gap_s if capture_thread is not None else None
    capture_errors = capture_thread.errors if capture_thread is not None else []
    mean_capture_interval_s = (
        float(np.mean(capture_thread.capture_intervals_s))
        if capture_thread is not None and capture_thread.capture_intervals_s else None
    )
    processing_loop_rate_hz = (len(loop_intervals_s) / wall) if wall > 0 and loop_intervals_s else None
    detector_processing_rate_hz = (
        1000.0 / float(np.mean(camera_times_ms))
        if camera_times_ms and float(np.mean(camera_times_ms)) > 0 else None
    )
    mean_frame_interval_s = mean_capture_interval_s
    max_frame_interval_s = max_capture_gap_s
    camera_loop_rate_hz = processing_loop_rate_hz
    detector_loop_rate_hz = (
        1000.0 / float(np.mean(camera_times_ms))
        if camera_times_ms and float(np.mean(camera_times_ms)) > 0 else None
    )
    max_processing_gap_s = float(np.max(loop_intervals_s)) if loop_intervals_s else None
    api_post_mean_latency_ms = float(np.mean(api_post_latencies_ms)) if api_post_latencies_ms else None
    api_post_max_latency_ms = float(np.max(api_post_latencies_ms)) if api_post_latencies_ms else None
    nominal_period_s = 1.0 / args.virtual_freq if args.virtual_freq > 0 else None
    expected_virtual_flash_timestamps = (
        [round(i * nominal_period_s, 6) for i in range(1, nominal_expected + 1)]
        if nominal_period_s is not None else []
    )
    detected_virtual_flash_timestamps = [round(t, 6) for t in virtual_detected]
    missed_flash_intervals_s = []
    if nominal_period_s is not None and len(virtual_detected) >= 2:
        missed_flash_intervals_s = [
            round(virtual_detected[i + 1] - virtual_detected[i], 6)
            for i in range(len(virtual_detected) - 1)
            if (virtual_detected[i + 1] - virtual_detected[i]) > 1.5 * nominal_period_s
        ]
    fcr_warning = actual_fcr is not None and actual_fcr > 1.02
    pi_runaway_warning = (
        pi_final_frequency_hz is not None
        and args.pi_freq <= 1.21
        and pi_final_frequency_hz > 2.3
    )
    if event_trace and (fcr_warning or pi_runaway_warning or short_interval_warning_count > 0):
        end_t = event_trace[-1].get("t_s", args.duration)
        suspicious_rows = [
            row for row in event_trace
            if _as_float(row.get("t_s")) is not None
            and float(row["t_s"]) >= float(end_t) - 3.0
        ]
    metrics = {
        "feedback_enabled": feedback_enabled_end if feedback_enabled_end is not None else feedback_on,
        "model": args.model,
        "model_parameters": _model_parameters(args.model, args.kuramoto_gain),
        "seed": args.seed,
        "random_phase_enabled": args.random_phase,
        "virtual_initial_phase_rad": round(virtual_initial_phase_rad, 6),
        "pi_initial_phase_rad": round(pi_initial_phase_rad, 6),
        "initial_phase_difference_rad": round(initial_phase_difference_rad, 6),
        "virtual_initial_freq": args.virtual_freq,
        "pi_initial_freq": args.pi_freq,
        "detect_only": args.detect_only,
        "gpio_enabled": gpio_enabled,
        "pi_feedback_post_enabled": pi_feedback_post_enabled,
        "detector_config": {
            "class": "PicameraFlashDetector",
            "resolution": [camera_width, camera_height],
            "detection_mode": "local_contrast",
            "threshold_on": args.threshold_on,
            "threshold_off": args.threshold_off,
            "min_interval_s": args.min_interval,
            "window_s": args.window_s,
            "norm_on_threshold": args.norm_on_threshold,
            "norm_off_threshold": args.norm_off_threshold,
            "min_amplitude": args.min_amplitude,
            "roi": args.roi,
            "episode_latch_enabled": args.episode_latch,
            "rearm_off_duration_s": args.rearm_off_duration,
            "off_frames_to_rearm": args.off_frames_to_rearm,
            "rearm_requires_both": args.rearm_requires_both,
            "self_flash_blanking_s": args.self_flash_blanking_s,
            "expected_blob_x": args.expected_blob_x,
            "expected_blob_y": args.expected_blob_y,
            "expected_blob_radius_px": args.expected_blob_radius_px,
        },
        "virtual_detected_count": virtual_det,
        "nominal_expected_flashes": nominal_expected,
        "nominal_fcr": round(nominal_fcr, 4) if nominal_fcr is not None else None,
        "virtual_fire_count_start": virtual_fire_count_start,
        "virtual_fire_count_end": virtual_fire_count_end,
        "actual_virtual_flashes": actual_virtual_flashes,
        "actual_fcr": round(actual_fcr, 4) if actual_fcr is not None else None,
        "virtual_frequency_start": round(virtual_frequency_start, 6) if virtual_frequency_start is not None else None,
        "virtual_frequency_end": round(virtual_frequency_end, 6) if virtual_frequency_end is not None else None,
        "pi_flash_count": len(pi_flash_times),
        "pi_final_frequency_hz": round(pi_final_frequency_hz, 6) if pi_final_frequency_hz is not None else None,
        "frequency_error_hz": round(frequency_error_hz, 6) if frequency_error_hz is not None else None,
        "virtual_freq_final_5s_mean": round(virtual_freq_final_5s_mean, 6) if virtual_freq_final_5s_mean is not None else None,
        "pi_freq_final_5s_mean": round(pi_freq_final_5s_mean, 6) if pi_freq_final_5s_mean is not None else None,
        "frequency_error_final_5s_mean_abs": round(frequency_error_final_5s_mean_abs, 6) if frequency_error_final_5s_mean_abs is not None else None,
        "frequency_error_final_5s_abs_of_means": round(frequency_error_final_5s_abs_of_means, 6) if frequency_error_final_5s_abs_of_means is not None else None,
        "virtual_freq_final_5s_std": round(virtual_freq_final_5s_std, 6) if virtual_freq_final_5s_std is not None else None,
        "pi_freq_final_5s_std": round(pi_freq_final_5s_std, 6) if pi_freq_final_5s_std is not None else None,
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
        "camera_loop_rate_hz": round(camera_loop_rate_hz, 3) if camera_loop_rate_hz is not None else None,
        "detector_loop_rate_hz": round(detector_loop_rate_hz, 3) if detector_loop_rate_hz is not None else None,
        "capture_thread_fps": round(capture_thread_fps, 3) if capture_thread_fps is not None else None,
        "processing_loop_rate_hz": round(processing_loop_rate_hz, 3) if processing_loop_rate_hz is not None else None,
        "detector_processing_rate_hz": round(detector_processing_rate_hz, 3) if detector_processing_rate_hz is not None else None,
        "capture_queue_drops": capture_queue_drops,
        "max_capture_gap_s": round(max_capture_gap_s, 6) if max_capture_gap_s is not None else None,
        "max_processing_gap_s": round(max_processing_gap_s, 6) if max_processing_gap_s is not None else None,
        "mean_frame_interval_s": round(mean_frame_interval_s, 6) if mean_frame_interval_s is not None else None,
        "max_frame_interval_s": round(max_frame_interval_s, 6) if max_frame_interval_s is not None else None,
        "api_post_count": api_post_count,
        "api_post_mean_latency_ms": round(api_post_mean_latency_ms, 3) if api_post_mean_latency_ms is not None else None,
        "api_post_max_latency_ms": round(api_post_max_latency_ms, 3) if api_post_max_latency_ms is not None else None,
        "api_post_errors": api_post_errors,
        "api_queue_size_max": api_queue_size_max,
        "api_queue_drops": api_queue_drops,
        "gpio_flash_count": gpio_flash_count,
        "gpio_nonblocking": True,
        "led_pulse_duration_s": args.led_pulse_duration,
        "detector_episode_latch_enabled": args.episode_latch,
        "rearm_off_duration_s": args.rearm_off_duration,
        "off_frames_to_rearm": args.off_frames_to_rearm,
        "rearm_requires_both": args.rearm_requires_both,
        "min_interval": args.min_interval,
        "self_flash_blanking_s": args.self_flash_blanking_s,
        "self_flash_blank_reject_count": self_flash_blank_reject_count,
        "spatial_gate_reject_count": spatial_gate_reject_count,
        "short_interval_warning_count": short_interval_warning_count,
        "suspicious_warning_count": suspicious_warning_count,
        "fcr_gt_1p02_warning": fcr_warning,
        "pi_frequency_runaway_warning": pi_runaway_warning,
        "duplicate_suppressed_count": duplicate_suppressed_count,
        "raw_on_threshold_crossing_count": raw_on_threshold_crossing_count,
        "accepted_flash_event_count": accepted_flash_event_count,
        "camera_fps_requested": args.camera_fps,
        "camera_frame_duration_limits_us": camera_frame_duration_limits,
        "camera_config_actual": camera_config_actual,
        "camera_controls_requested": camera_controls_requested,
        "camera_format_requested": "BGR888",
        "frame_format_debug": last_frame_format_debug,
        "roi": args.roi,
        "capture_errors": capture_errors,
        "detected_virtual_flash_timestamps": detected_virtual_flash_timestamps,
        "expected_virtual_flash_timestamps": expected_virtual_flash_timestamps,
        "missed_flash_intervals_s": missed_flash_intervals_s,
        "running_start": running_start,
        "running_end": running_end,
        "feedback_enabled_start": feedback_enabled_start,
        "feedback_enabled_end": feedback_enabled_end,
        "duration_s": args.duration,
        "actual_wall_s": round(wall, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"model": args.model,
                   "model_parameters": _model_parameters(args.model, args.kuramoto_gain),
                   "virtual_freq": args.virtual_freq, "pi_freq": args.pi_freq,
                   "seed": args.seed,
                   "random_phase_enabled": args.random_phase,
                   "virtual_initial_phase_rad": virtual_initial_phase_rad,
                   "pi_initial_phase_rad": pi_initial_phase_rad,
                   "initial_phase_difference_rad": initial_phase_difference_rad,
                   "feedback": feedback_on, "dry_run": args.dry_run,
                   "detect_only": args.detect_only,
                   "gpio_enabled": gpio_enabled,
                   "pi_feedback_post_enabled": pi_feedback_post_enabled,
                   "detector_config": {
                       "class": "PicameraFlashDetector",
                       "resolution": [camera_width, camera_height],
                       "detection_mode": "local_contrast",
                       "threshold_on": args.threshold_on,
                       "threshold_off": args.threshold_off,
                       "min_interval_s": args.min_interval,
                       "window_s": args.window_s,
                       "norm_on_threshold": args.norm_on_threshold,
                       "norm_off_threshold": args.norm_off_threshold,
                       "min_amplitude": args.min_amplitude,
                       "roi": args.roi,
                       "episode_latch_enabled": args.episode_latch,
                       "rearm_off_duration_s": args.rearm_off_duration,
                       "off_frames_to_rearm": args.off_frames_to_rearm,
                       "rearm_requires_both": args.rearm_requires_both,
                       "self_flash_blanking_s": args.self_flash_blanking_s,
                       "expected_blob_x": args.expected_blob_x,
                       "expected_blob_y": args.expected_blob_y,
                       "expected_blob_radius_px": args.expected_blob_radius_px,
                   },
                   "camera_fps_requested": args.camera_fps,
                   "camera_frame_duration_limits_us": camera_frame_duration_limits,
                   "camera_config_actual": camera_config_actual,
                   "camera_controls_requested": camera_controls_requested,
                   "camera_format_requested": "BGR888",
                   "timestamp": datetime.now().isoformat()}, f, indent=2)
    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _safe_write_dict_rows_csv(out_dir/"metrics_summary.csv", [metrics], list(metrics.keys()))
    if osc_log:
        _safe_write_dict_rows_csv(out_dir/"pi_oscillator_log.csv", osc_log)
    if det_log:
        _safe_write_dict_rows_csv(out_dir/"pi_detection_log.csv", det_log)
        _safe_write_dict_rows_csv(out_dir/"detection_log.csv", det_log)
    if flash_events:
        _safe_write_dict_rows_csv(out_dir/"combined_flash_events.csv", flash_events, ["t_s","event"])
    if agent_poll_rows:
        _safe_write_dict_rows_csv(out_dir/"agent_poll_log.csv", agent_poll_rows)
    if event_trace:
        _safe_write_dict_rows_csv(out_dir/"event_trace.csv", event_trace)
    if args.model == "kuramoto" and not args.dry_run:
        kuramoto_debug_rows = _fetch_kuramoto_debug(api, timeout_s=min(to, 5.0))
        if kuramoto_debug_rows:
            _safe_write_dict_rows_csv(out_dir/"kuramoto_debug.csv", kuramoto_debug_rows)
    if detector_events:
        _safe_write_dict_rows_csv(out_dir/"detector_events.csv", detector_events)
    if oscillator_events:
        _safe_write_dict_rows_csv(out_dir/"oscillator_events.csv", oscillator_events)
    api_events = api_poster.events if api_poster is not None else []
    if api_events:
        _safe_write_dict_rows_csv(out_dir/"api_events.csv", api_events)
    if suspicious_rows:
        _safe_write_dict_rows_csv(out_dir/"suspicious_window.csv", suspicious_rows)

    # Quick figures
    fd = out_dir / "figures"; fd.mkdir(parents=True, exist_ok=True)
    if osc_log:
        fig, ax = plt.subplots(figsize=(7,3))
        ax.plot([e["t_s"] for e in osc_log], [e["freq_hz"] for e in osc_log])
        ax.axhline(args.virtual_freq, color="gray", linestyle="--")
        ax.set_ylabel("Pi Freq (Hz)"); ax.set_xlabel("Time (s)")
        ax.set_title("Pi Frequency — Mutual HIL"); fig.savefig(fd/"pi_freq.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    if pi_flash_times or virtual_detected:
        fig, ax = plt.subplots(figsize=(10,2))
        if virtual_detected: ax.eventplot(virtual_detected, lineoffsets=1, colors="steelblue", linewidths=0.8)
        if pi_flash_times: ax.eventplot(pi_flash_times, lineoffsets=0, colors="darkorange", linewidths=0.8)
        ax.set_yticks([0,1]); ax.set_yticklabels(["Pi","Virtual"]); ax.set_xlabel("Time (s)")
        ax.set_title("Mutual HIL — Flash Raster"); fig.savefig(fd/"flash_raster.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    def _fmt_metric(value: float | int | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    mode_label = "feedback ON" if metrics["feedback_enabled"] else "feedback OFF"
    print(f"\nOutput: {out_dir}")
    print(f"  Mode: {mode_label}")
    print(f"  Virtual detected: {virtual_det}")
    print(f"  Virtual nominal: {virtual_det}/{nominal_expected} "
          f"(nominal FCR={_fmt_metric(nominal_fcr)})")
    actual_den = actual_virtual_flashes if actual_virtual_flashes is not None else "n/a"
    print(f"  Virtual actual: {virtual_det}/{actual_den} "
          f"(actual FCR={_fmt_metric(actual_fcr)})")
    print(f"  Virtual freq: start={_fmt_metric(virtual_frequency_start, 4)} Hz "
          f"end={_fmt_metric(virtual_frequency_end, 4)} Hz")
    print(f"  Initial phases: seed={args.seed} random_phase={args.random_phase} "
          f"virtual={virtual_initial_phase_rad:.6f} rad "
          f"pi={pi_initial_phase_rad:.6f} rad "
          f"diff={initial_phase_difference_rad:.6f} rad")
    print(f"  Pi flashes: {len(pi_flash_times)}")
    print(f"  Pi final freq: {_fmt_metric(pi_final_frequency_hz, 4)} Hz")
    print(f"  Frequency error: instant |Pi - virtual| = {_fmt_metric(frequency_error_hz, 4)} Hz")
    print(f"  Final 5s mean freq: virtual={_fmt_metric(virtual_freq_final_5s_mean, 4)} Hz "
          f"pi={_fmt_metric(pi_freq_final_5s_mean, 4)} Hz")
    print(f"  Final 5s error: mean_abs={_fmt_metric(frequency_error_final_5s_mean_abs, 4)} Hz "
          f"abs_of_means={_fmt_metric(frequency_error_final_5s_abs_of_means, 4)} Hz")
    print(f"  Capture FPS: {_fmt_metric(capture_thread_fps, 2)} Hz "
          f"(requested={_fmt_metric(args.camera_fps, 1)} Hz)")
    print(f"  Processing FPS: {_fmt_metric(processing_loop_rate_hz, 2)} Hz "
          f"(detector processing rate={_fmt_metric(detector_processing_rate_hz, 2)} Hz)")
    print(f"  Capture gap: mean={_fmt_metric(mean_frame_interval_s, 4)}s "
          f"max={_fmt_metric(max_capture_gap_s, 4)}s "
          f"drops={capture_queue_drops}")
    print(f"  Processing gap: max={_fmt_metric(max_processing_gap_s, 4)}s")
    print(f"  Episode latch: enabled={args.episode_latch} "
          f"min_interval={args.min_interval:.3f}s "
          f"rearm_off_duration={args.rearm_off_duration:.3f}s "
          f"off_frames_to_rearm={args.off_frames_to_rearm}")
    print(f"  Detector events: accepted={accepted_flash_event_count} "
          f"raw_on_crossings={raw_on_threshold_crossing_count} "
          f"duplicates_suppressed={duplicate_suppressed_count}")
    print(f"  Diagnostic gates: self_blank_rejects={self_flash_blank_reject_count} "
          f"spatial_rejects={spatial_gate_reject_count} "
          f"short_interval_warnings={short_interval_warning_count}")
    print(f"  Pi feedback counters: posts_received={metrics['pi_flash_posts_received']} "
          f"events_consumed={metrics['pi_flash_events_consumed']}")
    print(f"  API POSTs: count={api_post_count} "
          f"mean={_fmt_metric(api_post_mean_latency_ms, 2)} ms "
          f"max={_fmt_metric(api_post_max_latency_ms, 2)} ms "
          f"queue_max={api_queue_size_max} drops={api_queue_drops}")
    print(f"  GPIO: enabled={gpio_enabled} nonblocking=True "
          f"pulse={args.led_pulse_duration:.3f}s flash_count={gpio_flash_count}")
    if missed_flash_intervals_s:
        print(f"  Missed-flash gap candidates: {len(missed_flash_intervals_s)} "
              f"max_gap={max(missed_flash_intervals_s):.3f}s")
    if fcr_warning or pi_runaway_warning or short_interval_warning_count:
        print("  WARNING: suspicious event timing detected; inspect event_trace.csv "
              "and suspicious_window.csv if present.")

if __name__ == "__main__":
    main()

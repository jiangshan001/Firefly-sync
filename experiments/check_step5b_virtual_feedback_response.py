#!/usr/bin/env python3
r"""Step 5B server-only virtual feedback response diagnostic.

This sends synthetic Pi flash events directly to ``/api/pi_flash`` and polls
``/api/agents`` to verify whether the virtual mutual-HIL oscillator responds.
No camera, GPIO, detector, frontend rendering, or model parameters are touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


class ApiError(RuntimeError):
    pass


def _api(api_base: str, path: str, method: str = "GET",
         data: dict | None = None, timeout: float = 10.0,
         retries: int = 3, label: str = "") -> dict:
    url = api_base.rstrip("/") + path
    endpoint = f"{method} {path}"
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            print(f"  [API] Calling {endpoint}" + (f" ({label})" if label else ""))
            if attempt > 1:
                print(f"    retry {attempt}/{retries}")
            body = json.dumps(data).encode() if data is not None else None
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"} if data is not None else {},
                method=method,
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read())
        except (TimeoutError, socket.timeout) as exc:
            last_exc = exc
            print(f"    TIMEOUT: {endpoint} timed out after {timeout:g}s")
        except urllib.error.URLError as exc:
            last_exc = exc
            reason = getattr(exc, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                print(f"    TIMEOUT: {endpoint} timed out after {timeout:g}s")
            else:
                print(f"    API ERROR: {type(exc).__name__}: {exc}")
        except Exception as exc:
            last_exc = exc
            print(f"    ERROR: {type(exc).__name__}: {exc}")
        if attempt < max(1, retries):
            time.sleep(1.0)
    raise ApiError(f"{endpoint} failed after {retries} attempts: {last_exc}") from last_exc


def _first_agent(payload: dict) -> dict:
    agents = payload.get("agents", [])
    return agents[0] if agents else {}


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _setup_virtual_agent(args: argparse.Namespace, feedback_enabled: bool) -> dict:
    api = args.leader_api.rstrip("/")
    timeout = args.api_timeout
    retries = args.api_retries
    print(f"\n[SETUP] mutual_hil model={args.model} feedback_enabled={feedback_enabled}")
    _api(api, "/api/mode", "POST", {"mode": "mutual_hil"}, timeout, retries, "mode")
    try:
        _api(api, "/api/pause", "POST", {}, timeout, 1, "pause best-effort")
    except ApiError:
        pass
    _api(api, "/api/reset", "POST", {}, timeout, retries, "reset")
    _api(api, "/api/agents/0", "POST", {
        "x": 800,
        "y": 400,
        "size": 450,
        "initial_frequency_hz": args.virtual_freq,
        "frequency_hz": args.virtual_freq,
        "model": args.model,
    }, timeout, retries, "configure agent 0")
    _api(api, "/api/feedback", "POST", {"enabled": feedback_enabled},
         timeout, retries, f"feedback={feedback_enabled}")
    _api(api, "/api/start", "POST", {}, timeout, retries, "start")
    start_state = _api(api, "/api/agents", "GET", timeout=timeout,
                       retries=retries, label="start state")
    return _first_agent(start_state)


def run_trial(args: argparse.Namespace, feedback_enabled: bool, out_dir: Path) -> dict:
    api = args.leader_api.rstrip("/")
    poll_period_s = 1.0 / max(0.1, args.poll_hz)
    flash_period_s = 1.0 / args.synthetic_pi_freq if args.synthetic_pi_freq > 0 else args.duration + 1.0
    rows: list[dict] = []
    synthetic_pi_flash_count = 0
    start_agent: dict = {}
    end_agent: dict = {}

    try:
        start_agent = _setup_virtual_agent(args, feedback_enabled)
        start_t = time.monotonic()
        next_flash_t = start_t
        next_poll_t = start_t

        while (time.monotonic() - start_t) < args.duration:
            now = time.monotonic()
            if now >= next_flash_t:
                _api(api, "/api/pi_flash", "POST", {}, timeout=args.api_timeout,
                     retries=args.api_retries, label="synthetic Pi flash")
                synthetic_pi_flash_count += 1
                next_flash_t += flash_period_s

            if now >= next_poll_t:
                payload = _api(api, "/api/agents", "GET", timeout=args.api_timeout,
                               retries=args.api_retries, label="poll")
                a0 = _first_agent(payload)
                t_s = time.monotonic() - start_t
                rows.append({
                    "t_s": round(t_s, 6),
                    "frequency_hz": a0.get("frequency_hz"),
                    "phase_rad": a0.get("phase_rad"),
                    "fire_count": a0.get("fire_count"),
                    "received_pi_flashes": a0.get("received_pi_flashes"),
                    "pi_flash_posts_received": a0.get("pi_flash_posts_received"),
                    "pi_flash_events_consumed": a0.get("pi_flash_events_consumed"),
                    "feedback_enabled": a0.get("feedback_enabled"),
                    "flash_on": a0.get("flash_on"),
                    "raw_flash_on": a0.get("raw_flash_on"),
                })
                next_poll_t += poll_period_s

            sleep_until = min(next_flash_t, next_poll_t, start_t + args.duration)
            time.sleep(max(0.0, min(0.02, sleep_until - time.monotonic())))

        end_state = _api(api, "/api/agents", "GET", timeout=args.api_timeout,
                         retries=args.api_retries, label="end state before pause")
        end_agent = _first_agent(end_state)
    finally:
        try:
            cleanup = _api(api, "/api/pause", "POST", {}, timeout=args.api_timeout,
                           retries=args.api_retries, label="cleanup pause")
            print(f"[Cleanup] POST /api/pause result: {cleanup}")
        except Exception as exc:
            print(f"[Cleanup] POST /api/pause failed: {exc}")

    trial_name = f"{args.model}_V{args.virtual_freq:g}_Psynthetic{args.synthetic_pi_freq:g}_{'feedback_on' if feedback_enabled else 'feedback_off'}"
    _write_csv(out_dir / f"{trial_name}_poll_log.csv", rows)

    start_freq = _as_float(start_agent.get("frequency_hz"))
    end_freq = _as_float(end_agent.get("frequency_hz"))
    start_fire = _as_int(start_agent.get("fire_count"))
    end_fire = _as_int(end_agent.get("fire_count"))
    start_received = _as_int(start_agent.get("received_pi_flashes"))
    end_received = _as_int(end_agent.get("received_pi_flashes"))
    start_posts = _as_int(start_agent.get("pi_flash_posts_received"))
    end_posts = _as_int(end_agent.get("pi_flash_posts_received"))
    start_consumed = _as_int(start_agent.get("pi_flash_events_consumed"))
    end_consumed = _as_int(end_agent.get("pi_flash_events_consumed"))
    final_window_start = max(0.0, args.duration - 5.0)
    final_freqs = [
        float(r["frequency_hz"])
        for r in rows
        if r.get("frequency_hz") is not None and float(r["t_s"]) >= final_window_start
    ]
    virtual_frequency_change = (
        end_freq - start_freq
        if end_freq is not None and start_freq is not None else None
    )
    expected_direction = 0
    if args.synthetic_pi_freq < args.virtual_freq:
        expected_direction = -1
    elif args.synthetic_pi_freq > args.virtual_freq:
        expected_direction = 1
    moved_toward = None
    if virtual_frequency_change is not None:
        if expected_direction == 0:
            moved_toward = abs(virtual_frequency_change) < 1e-6
        else:
            moved_toward = (virtual_frequency_change * expected_direction) > 0

    metrics = {
        "model": args.model,
        "feedback_enabled": feedback_enabled,
        "virtual_initial_freq": args.virtual_freq,
        "synthetic_pi_freq": args.synthetic_pi_freq,
        "duration_s": args.duration,
        "synthetic_pi_flash_count": synthetic_pi_flash_count,
        "virtual_frequency_start": start_freq,
        "virtual_frequency_end": end_freq,
        "virtual_frequency_change": virtual_frequency_change,
        "mean_frequency_final_5s": _mean(final_freqs),
        "virtual_fire_count_start": start_fire,
        "virtual_fire_count_end": end_fire,
        "virtual_fire_count_delta": (
            end_fire - start_fire
            if end_fire is not None and start_fire is not None else None
        ),
        "received_pi_flashes_start": start_received,
        "received_pi_flashes_end": end_received,
        "received_pi_flashes_delta": (
            end_received - start_received
            if end_received is not None and start_received is not None else None
        ),
        "pi_flash_posts_received_start": start_posts,
        "pi_flash_posts_received_end": end_posts,
        "pi_flash_posts_received_delta": (
            end_posts - start_posts
            if end_posts is not None and start_posts is not None else (
                end_received - start_received
                if end_received is not None and start_received is not None else None
            )
        ),
        "pi_flash_events_consumed_start": start_consumed,
        "pi_flash_events_consumed_end": end_consumed,
        "pi_flash_events_consumed_delta": (
            end_consumed - start_consumed
            if end_consumed is not None and start_consumed is not None else None
        ),
        "frequency_moved_toward_synthetic_pi_freq": moved_toward,
        "poll_samples": len(rows),
    }
    with (out_dir / f"{trial_name}_metrics_summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Server-only synthetic Pi feedback response diagnostic for Step 5B."
    )
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--model", choices=["eapf_consensus", "kuramoto"],
                        default="eapf_consensus")
    parser.add_argument("--virtual-freq", type=float, default=2.0)
    parser.add_argument("--synthetic-pi-freq", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--api-timeout", type=float, default=10.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--poll-hz", type=float, default=8.0)
    parser.add_argument("--feedback-off", action="store_true",
                        help="Run feedback-OFF control instead of feedback-ON.")
    parser.add_argument("--log-dir", default="experiments/logs/step5b_virtual_feedback_response")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.log_dir) / f"{ts}_step5b_virtual_feedback_response"
    out_dir.mkdir(parents=True, exist_ok=True)

    feedback_enabled = not args.feedback_off
    print("=" * 72)
    print("Step 5B synthetic Pi feedback response diagnostic")
    print("=" * 72)
    print(f"API: {args.leader_api.rstrip('/')}")
    print(f"Model: {args.model}")
    print(f"Virtual frequency: {args.virtual_freq:.4f} Hz")
    print(f"Synthetic Pi frequency: {args.synthetic_pi_freq:.4f} Hz")
    print(f"Feedback enabled: {feedback_enabled}")
    print(f"Output: {out_dir}")

    try:
        metrics = run_trial(args, feedback_enabled=feedback_enabled, out_dir=out_dir)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\nSummary:")
    print(f"  virtual_frequency_start: {metrics['virtual_frequency_start']}")
    print(f"  virtual_frequency_end: {metrics['virtual_frequency_end']}")
    print(f"  virtual_frequency_change: {metrics['virtual_frequency_change']}")
    print(f"  mean_frequency_final_5s: {metrics['mean_frequency_final_5s']}")
    print(f"  virtual_fire_count_delta: {metrics['virtual_fire_count_delta']}")
    print(f"  pi_flash_posts_received_delta: {metrics['pi_flash_posts_received_delta']}")
    print(f"  pi_flash_events_consumed_delta: {metrics['pi_flash_events_consumed_delta']}")
    print(f"  received_pi_flashes_delta: {metrics['received_pi_flashes_delta']} (backward-compatible alias)")
    print(f"  synthetic_pi_flash_count: {metrics['synthetic_pi_flash_count']}")
    print(f"  moved_toward_synthetic_pi_freq: {metrics['frequency_moved_toward_synthetic_pi_freq']}")


if __name__ == "__main__":
    main()

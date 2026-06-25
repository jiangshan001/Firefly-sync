#!/usr/bin/env python3
r"""Step 5B virtual source stability diagnostic.

Verifies that the /mutual virtual dot flashes at a stable frequency
before involving Pi camera detection.

Usage::

    PYTHONPATH=. python experiments/check_step5b_virtual_source_stability.py \
        --api http://127.0.0.1:8000 --duration 15
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

def _api(url: str, method: str = "GET", data: dict | None = None) -> dict:
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"} if data else {},
                                  method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main() -> None:
    p = argparse.ArgumentParser(description="Virtual source stability check.")
    p.add_argument("--api", default="http://127.0.0.1:8000")
    p.add_argument("--duration", type=float, default=15.0)
    p.add_argument("--feedback-test", action="store_true",
                   help="Run synthetic feedback test after calibration")
    args = p.parse_args()
    api = args.api.rstrip("/")

    print("="*50)
    print("VIRTUAL SOURCE STABILITY CHECK")
    print(f"API: {api}, Duration: {args.duration}s")
    print("="*50)

    # Setup: reset + configure
    print("\n[1] Reset + configure calibration mode...")
    _api(api + "/api/mode", "POST", {"mode": "mutual_hil"})
    _api(api + "/api/reset", "POST")
    _api(api + "/api/feedback", "POST", {"enabled": False})  # feedback OFF
    _api(api + "/api/agents/0", "POST",
         {"x":800, "y":400, "size":450, "initial_frequency_hz":2.0, "model":"eapf_consensus"})

    # Start
    print("\n[2] Starting virtual agent (feedback OFF, calibration mode)...")
    _api(api + "/api/start", "POST")
    time.sleep(1.0)

    # Poll for duration
    print(f"\n[3] Polling for {args.duration}s...")
    fire_counts = []
    freqs = []
    flash_ons = []
    t0 = time.monotonic()
    while (time.monotonic() - t0) < args.duration:
        try:
            r = _api(api + "/api/agents", "GET")
            a = r["agents"][0] if r.get("agents") else {}
            fc = a.get("fire_count", 0)
            fq = a.get("frequency_hz", 0)
            fo = a.get("flash_on", False)
            fire_counts.append(fc)
            freqs.append(fq)
            flash_ons.append(fo)
        except Exception:
            pass
        time.sleep(0.25)

    # Results
    if not fire_counts:
        print("ERROR: No agent data received")
        sys.exit(1)

    fc_start = fire_counts[0]
    fc_end = fire_counts[-1]
    fc_delta = fc_end - fc_start
    expected = args.duration * 2.0  # 2 Hz
    ratio = fc_delta / expected if expected > 0 else 0

    n_flash_on = sum(1 for f in flash_ons if f)
    max_gap = 0
    gap = 0
    for f in flash_ons:
        if not f:
            gap += 1
        else:
            max_gap = max(max_gap, gap)
            gap = 0
    max_gap_s = max_gap * 0.25  # ~250ms polling interval

    mean_freq = sum(freqs) / len(freqs) if freqs else 0

    print(f"\n[4] Results:")
    print(f"  fire_count delta: {fc_delta} (expected ~{expected:.0f}, ratio={ratio:.2f})")
    print(f"  flash_on samples: {n_flash_on}/{len(flash_ons)}")
    print(f"  max no-flash gap: {max_gap_s:.1f}s")
    print(f"  mean frequency_hz: {mean_freq:.3f} Hz")

    passed = (0.80 <= ratio <= 1.20 and max_gap_s < 1.5 and 1.7 <= mean_freq <= 2.3)
    print(f"\n  STABILITY CHECK: {'PASSED' if passed else 'FAILED'}")

    # Cleanup
    _api(api + "/api/pause", "POST")
    print("[5] Paused.")

    if args.feedback_test and passed:
        print("\n[6] Synthetic feedback test...")
        _api(api + "/api/reset", "POST")
        _api(api + "/api/feedback", "POST", {"enabled": True})
        _api(api + "/api/start", "POST")
        time.sleep(1.0)
        fc_start2 = _api(api + "/api/agents")["agents"][0].get("fire_count", 0)
        # Send synthetic Pi flashes at 1.5 Hz
        for i in range(int(args.duration * 1.5)):
            _api(api + "/api/pi_flash", "POST", {"timestamp": time.monotonic()})
            time.sleep(1.0 / 1.5)
        fc_end2 = _api(api + "/api/agents")["agents"][0].get("fire_count", 0)
        delta2 = fc_end2 - fc_start2
        print(f"  fire_count delta with feedback: {delta2} (should be near {args.duration*2:.0f})")
        _api(api + "/api/pause", "POST")
        print("  Feedback test complete.")

    print("\nDone.")


if __name__ == "__main__":
    main()

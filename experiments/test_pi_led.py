#!/usr/bin/env python3
"""Quick LED blink test for Raspberry Pi GPIO.

Verifies that the physical LED on GPIO17 is wired correctly and
responding to commands.  Run this **before** the full camera stream
server to confirm the LED circuit works.

Usage::

    python experiments/test_pi_led.py
    python experiments/test_pi_led.py --pin 17 --cycles 5 --on-time 0.1 --off-time 0.9
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Blink a physical LED on a Raspberry Pi GPIO pin.",
    )
    parser.add_argument(
        "--pin", type=int, default=17,
        help="BCM GPIO pin number (default: 17).",
    )
    parser.add_argument(
        "--cycles", type=int, default=10,
        help="Number of on/off cycles (default: 10).",
    )
    parser.add_argument(
        "--on-time", type=float, default=0.2,
        help="LED on duration per cycle in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--off-time", type=float, default=0.8,
        help="LED off duration per cycle in seconds (default: 0.8).",
    )
    args = parser.parse_args()

    # --- Lazy import so the script can at least print help on non-Pi ---
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED  # noqa: F811
    except ImportError as exc:
        print(f"ERROR: {exc}")
        print("\nThis script must run on a Raspberry Pi with gpiozero installed.")
        print("  sudo apt install -y python3-gpiozero")
        sys.exit(1)

    led = PiGPIOLED(pin=args.pin, flash_duration_s=args.on_time)

    print(f"LED blink test — GPIO{args.pin}, {args.cycles} cycles")
    print(f"  ON  = {args.on_time:.2f} s")
    print(f"  OFF = {args.off_time:.2f} s")
    print("Press Ctrl+C to stop early.\n")

    try:
        for i in range(1, args.cycles + 1):
            led.on()
            t_on = time.perf_counter()
            print(f"  [{i:02d}] ON  @ {t_on:.2f}")
            time.sleep(args.on_time)

            led.off()
            t_off = time.perf_counter()
            print(f"  [{i:02d}] OFF @ {t_off:.2f}")
            time.sleep(args.off_time)

        print("\nDone — all cycles completed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        led.turn_off()
        led.close()
        print("LED off. GPIO released.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
r"""Pi Camera Alignment Check — no synchronisation, visual debug only.

This script helps confirm:
  - the Pi camera sees the screen leader dot clearly
  - the ROI / detection pipeline is working
  - the Pi's own GPIO17 LED is NOT visible in the camera frame
  - if the follower LED is visible, reposition before running real tests

It starts the MJPEG stream server AND blinks GPIO17 so you can
visually check the camera feed for self-interference.

Usage::

    PYTHONPATH=. python3 experiments/run_pi_visual_alignment_check.py \
        --leader-api http://192.168.1.111:8000 \
        --leader-freq 2.0 --leader-shape circle --leader-dot-size 120 \
        --led-test-freq 1.5
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Any

# ======================================================================
# Hardware imports (lazy)
# ======================================================================

_Flask: Any = None
_Response: Any = None
_CV2: Any = None
_NP: Any = None
_PicameraFlashDetector: Any = None
_PiGPIOLED: Any = None


def _ensure_hardware() -> None:
    global _Flask, _Response, _CV2, _NP, _PicameraFlashDetector, _PiGPIOLED
    try:
        from flask import Flask as _F, Response as _R
        _Flask = _F; _Response = _R
        import cv2 as _c; _CV2 = _c
        import numpy as _n; _NP = _n
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector as _PFD,
        )
        _PicameraFlashDetector = _PFD
    except ImportError as e:
        print(f"ERROR: Missing Pi dependencies — {e}")
        print("Install: sudo apt install -y python3-flask python3-opencv python3-picamera2")
        sys.exit(1)
    try:
        from firefly_sync.hardware.pi_led import PiGPIOLED as _PL
        _PiGPIOLED = _PL
    except ImportError:
        print("WARNING: gpiozero not available — LED test disabled.")
        _PiGPIOLED = None


# ======================================================================
# Leader API client (minimal — no urllib needed at module level)
# ======================================================================

def _api_post(url: str, data: dict) -> dict:
    import json, urllib.request
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _api_get(url: str) -> dict:
    import json, urllib.request
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())


# ======================================================================
# MJPEG overlay (minimal copy)
# ======================================================================

def _draw_overlay(frame: Any, result: dict, roi: Any) -> Any:
    h, w = frame.shape[:2]
    if roi is not None:
        rx, ry, rw, rh = roi
        _CV2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
    sc = (0, 255, 0) if result.get("state") == "ON" else (0, 0, 255)
    _CV2.circle(frame, (w - 40, 40), 15, sc, -1)
    lines = [
        f"Bright: {result.get('brightness_used', 0):.1f}",
        f"State: {result.get('state', 'OFF')}",
        f"Edges: {result.get('rising_edge_count', 0)}",
        f"Freq: {result.get('estimated_frequency_hz', 0):.2f} Hz",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 22
        _CV2.putText(frame, line, (10, y), _CV2.FONT_HERSHEY_SIMPLEX,
                     0.55, (255, 255, 255), 1)
    return frame


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    _ensure_hardware()

    parser = argparse.ArgumentParser(
        description="Pi Camera Alignment Check — visual debug, no sync.")
    parser.add_argument("--leader-api", default="http://127.0.0.1:8000")
    parser.add_argument("--leader-freq", type=float, default=2.0)
    parser.add_argument("--leader-shape", default="circle")
    parser.add_argument("--leader-dot-size", type=int, default=120)
    parser.add_argument("--leader-duty-cycle", type=float, default=0.5)
    parser.add_argument("--keep-leader-running", action="store_true")
    # Camera
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    # LED
    parser.add_argument("--led-test-freq", type=float, default=1.5)
    parser.add_argument("--led-duty-cycle", type=float, default=0.5)
    parser.add_argument("--disable-led-test", action="store_true")
    parser.add_argument("--led-pin", type=int, default=17)

    args = parser.parse_args()

    # --- Leader API: start leader ---
    api = args.leader_api.rstrip("/")
    print(f"[API] Starting leader at {api}")
    try:
        _api_post(f"{api}/api/leader/config", {
            "frequency_hz": args.leader_freq,
            "duty_cycle": args.leader_duty_cycle,
            "brightness_on": 255, "brightness_off": 0,
            "background_brightness": 0,
            "shape": args.leader_shape,
            "target_size_px": args.leader_dot_size,
            "running": True,
        })
        time.sleep(1.0)
        st = _api_get(f"{api}/api/status")
        print(f"  Status: freq={st.get('frequency_hz')}Hz  running={st.get('running')}  "
              f"shape={st.get('shape')}  size={st.get('target_size_px')}px")
    except Exception as e:
        print(f"  [WARN] Could not control leader API: {e}")

    # --- LED test ---
    led = None
    if not args.disable_led_test and _PiGPIOLED is not None:
        led = _PiGPIOLED(pin=args.led_pin)
        led_period = 1.0 / args.led_test_freq if args.led_test_freq > 0 else 1.0
        led_on_time = led_period * args.led_duty_cycle
        led_off_time = led_period * (1.0 - args.led_duty_cycle)
    else:
        led_on_time = led_off_time = 0.0

    # --- Camera ---
    detector = _PicameraFlashDetector(
        resolution=[args.width, args.height],
        detection_mode="local_contrast",
    )
    detector.start()

    # Setup Flask
    app = _Flask(__name__)
    import threading
    lock = threading.Lock()
    latest_jpeg = b""
    latest_result: dict = {}

    @app.after_request
    def _cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.route("/status")
    def status():
        with lock:
            r = dict(latest_result)
        r["led_test_active"] = (led is not None)
        return r

    @app.route("/video_feed")
    def video_feed():
        def gen():
            while True:
                with lock:
                    j = bytes(latest_jpeg)
                if j:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                time.sleep(0.033)
        return _Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # --- Print info ---
    print()
    print("=" * 60)
    print("  CAMERA ALIGNMENT CHECK")
    print("=" * 60)
    print(f"  Video feed:    http://<pi-ip>:{args.port}/video_feed")
    print(f"  Status:        http://<pi-ip>:{args.port}/status")
    print(f"  Leader API:    {api}")
    print(f"  Leader freq:   {args.leader_freq} Hz")
    print(f"  LED test freq: {args.led_test_freq if led else 'disabled'} Hz")
    print()
    print("  WARNING: Camera should see the screen leader only.")
    print("  If GPIO17 LED is visible in the video feed, reposition")
    print("  the LED or camera before running batch tests.")
    print("=" * 60)
    print("  Press Ctrl+C to stop.")
    print()

    # --- Capture thread ---
    def _capture_loop() -> None:
        nonlocal latest_jpeg, latest_result
        led_state = False
        last_led_toggle = time.perf_counter()
        while True:
            result = detector.capture_frame()
            raw = detector.capture_raw_frame()
            annotated = _draw_overlay(raw, result, detector.roi)
            ok, buf = _CV2.imencode(".jpg", annotated)
            with lock:
                latest_result = result
                latest_jpeg = buf.tobytes() if ok else b""
            # LED blink
            if led is not None:
                now = time.perf_counter()
                interval = (led_on_time if led_state else led_off_time)
                if now - last_led_toggle >= interval:
                    led_state = not led_state
                    last_led_toggle = now
                    if led_state:
                        led.on()
                    else:
                        led.off()

    import threading as _th
    cap_thread = _th.Thread(target=_capture_loop, daemon=True)
    cap_thread.start()

    # --- Run ---
    def _shutdown(sig, frame):
        print("\nAlignment check stopped.")
        if led is not None:
            led.off(); led.close()
        detector.stop()
        if not args.keep_leader_running:
            try:
                _api_post(f"{api}/api/leader/config", {"running": False})
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()

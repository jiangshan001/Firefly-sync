#!/usr/bin/env python3
r"""Real-time Pi Camera Flash Detection Server (Stage 2D).

Raspberry Pi 5 server providing:
  - ``GET /video_feed``   — MJPEG stream with detection overlay
  - ``GET /status``       — JSON detection status
  - ``GET /health``       — simple health check
  - ``GET /auto_roi``     — trigger auto-ROI localisation
  - ``GET /clear_roi``    — clear active ROI

Usage (recommended — local_contrast + adaptive)::

    PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
        --host 0.0.0.0 --port 5000 \
        --detection-mode local_contrast --auto-roi --auto-roi-duration 3 \
        --window-s 5 --norm-on-threshold 0.65 --norm-off-threshold 0.35 \
        --min-interval 0.2

Manual camera (fixed exposure)::

    PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
        --host 0.0.0.0 --port 5000 \
        --detection-mode local_contrast --auto-roi --auto-roi-duration 3 \
        --manual-camera --exposure-us 8000 --analogue-gain 1.0 \
        --awb-enable false --target-fps 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_NP: Any = None
_CV2: Any = None
_Flask: Any = None
_Response: Any = None
_PicameraFlashDetector: Any = None
_locate_flashing_region: Any = None


def _ensure_deps() -> None:
    global _NP, _CV2, _Flask, _Response
    global _PicameraFlashDetector, _locate_flashing_region

    errors: list[str] = []

    try:
        import numpy as _numpy
        _NP = _numpy
    except ImportError:
        errors.append("numpy")

    try:
        import cv2 as _cv2
        _CV2 = _cv2
    except ImportError:
        errors.append("opencv (sudo apt install -y python3-opencv)")

    try:
        from flask import Flask as _F, Response as _R
        _Flask = _F
        _Response = _R
    except ImportError:
        errors.append("flask (sudo apt install -y python3-flask)")

    try:
        from firefly_sync.hardware.picamera_flash_detector import (
            PicameraFlashDetector as _PFD,
        )
        _PicameraFlashDetector = _PFD
    except ImportError:
        errors.append("picamera2 — ensure firefly_sync is on PYTHONPATH")

    try:
        from firefly_sync.hardware.roi_locator import locate_flashing_region as _lfr
        _locate_flashing_region = _lfr
    except ImportError:
        errors.append("roi_locator")

    if errors:
        print("ERROR: Missing dependencies:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Camera control helpers
# ---------------------------------------------------------------------------

def _apply_camera_controls(
    detector: Any,
    manual: bool = False,
    exposure_us: int | None = None,
    analogue_gain: float | None = None,
    awb_enable: bool = True,
) -> None:
    """Apply manual camera settings via Picamera2 controls if available."""
    if not manual:
        return

    picam2 = getattr(detector, "_picam2", None)
    if picam2 is None:
        print("[camera] Warning: Picamera2 not initialised — cannot set controls.")
        return

    try:
        controls = {}
        if exposure_us is not None:
            controls["ExposureTime"] = exposure_us
        if analogue_gain is not None:
            controls["AnalogueGain"] = analogue_gain
        if not awb_enable:
            controls["AwbEnable"] = 0

        if controls:
            picam2.set_controls(controls)
            applied = ", ".join(f"{k}={v}" for k, v in controls.items())
            print(f"[camera] Manual controls applied: {applied}")
    except Exception as exc:
        print(f"[camera] Warning: Could not apply camera controls: {exc}")


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: Any,
    result: dict,
    roi: list[int] | None,
    roi_source: str = "none",
    roi_confidence: float = 0.0,
    extra: dict | None = None,
    manual_camera: bool = False,
) -> Any:
    """Draw detection information on a camera frame."""
    assert _CV2 is not None

    h, w = frame.shape[:2]
    extra = extra or {}

    # ROI
    if roi is not None:
        rx, ry, rw, rh = roi
        color = (0, 255, 255) if roi_source == "auto" else (0, 255, 0)
        _CV2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

    auto_candidate = extra.get("auto_candidate_roi")
    if auto_candidate is not None:
        ax, ay, aw, ah = auto_candidate
        _CV2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 255, 255), 1)

    # State indicator
    state_color = (0, 255, 0) if result.get("state") == "ON" else (0, 0, 255)
    _CV2.circle(frame, (w - 40, 40), 15, state_color, -1)

    # Text
    mode = result.get("detection_mode", "?")
    lc = result.get("local_contrast", 0)
    sn = result.get("signal_norm", 0)
    sf = result.get("signal_frequency_hz", 0)
    pc = result.get("periodicity_confidence", 0)
    cam_tag = " [MANUAL]" if manual_camera else ""

    lines = [
        f"Mode: {mode}{cam_tag}",
        f"LC: {lc:.1f}  SN: {sn:.3f}",
        f"Adapt: lo={result.get('adaptive_low',0):.0f} hi={result.get('adaptive_high',0):.0f} amp={result.get('adaptive_amplitude',0):.0f}",
        f"State: {result.get('state','OFF')}  Edges: {result.get('rising_edge_count',0)}",
        f"Freq(ev): {result.get('estimated_frequency_hz',0):.2f}  Freq(ac): {sf:.2f} Hz",
        f"PeriodConf: {pc:.2f}  Qual: {result.get('signal_quality',0):.2f}",
        f"FPS: {result.get('fps_estimate',0):.1f}",
    ]
    last_edge = result.get("last_rising_edge_time_s")
    if last_edge is not None:
        lines.append(f"Last edge: {result.get('timestamp_s', time.perf_counter()) - last_edge:.3f}s ago")

    if roi_source != "none":
        lines.append(f"ROI: {roi_source}  conf={roi_confidence:.2f}")

    y0 = 30
    for i, line in enumerate(lines):
        y = y0 + i * 22
        _CV2.putText(frame, line, (12, y + 1), _CV2.FONT_HERSHEY_SIMPLEX,
                     0.50, (0, 0, 0), 2)
        _CV2.putText(frame, line, (10, y), _CV2.FONT_HERSHEY_SIMPLEX,
                     0.50, (255, 255, 255), 1)

    return frame


# ---------------------------------------------------------------------------
# Auto-ROI helper
# ---------------------------------------------------------------------------

def _run_auto_roi(
    detector: Any,
    duration_s: float = 3.0,
    padding_px: int = 20,
    min_area_px: int = 50,
    downsample: int = 2,
) -> dict:
    assert _locate_flashing_region is not None
    print(f"[auto-roi] Calibrating for {duration_s:.1f}s ...")

    frames: list[Any] = []
    start = time.perf_counter()
    while (time.perf_counter() - start) < duration_s:
        f = detector.capture_raw_frame()
        frames.append(f)
        time.sleep(0.05)

    print(f"[auto-roi] Collected {len(frames)} frames.  Locating...")
    result = _locate_flashing_region(
        frames, method="temporal_variance",
        min_area_px=min_area_px, padding_px=padding_px, downsample=downsample,
    )

    if result is None:
        return {"success": False, "message": "No flashing region found",
                "roi": None, "confidence": 0.0, "score": 0.0}

    print(f"[auto-roi] Found ROI: x={result['x']} y={result['y']} "
          f"w={result['width']} h={result['height']} conf={result['confidence']:.3f}")
    return {
        "success": True,
        "roi": {"x": result["x"], "y": result["y"],
                "width": result["width"], "height": result["height"]},
        "confidence": result["confidence"], "score": result["score"],
        "method": result["method"],
        "message": f"ROI found at ({result['x']}, {result['y']}) "
                   f"{result['width']}x{result['height']}",
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "timestamp_s", "elapsed_time_s", "frame_index",
    "detection_mode",
    "brightness_used", "full_frame_mean", "top_percentile_brightness",
    "brightness_mean",
    "roi_median_brightness", "roi_top_percentile_brightness",
    "local_contrast", "local_contrast_ratio",
    "signal_norm", "adaptive_low", "adaptive_high", "adaptive_amplitude",
    "signal_quality", "norm_on_threshold", "norm_off_threshold",
    "signal_frequency_hz", "periodicity_confidence",
    "state", "event_type", "rising_edge_count", "estimated_frequency_hz",
    "roi", "roi_source", "roi_confidence",
    "percentile", "blob_found", "blob_area_px", "blob_bbox",
    "threshold_on", "threshold_off",
    "manual_camera", "exposure_us", "analogue_gain",
]


def _serialise_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        import json
        return json.dumps(v)
    return str(v)


def create_app() -> Any:
    _ensure_deps()
    assert _Flask is not None

    app = _Flask(__name__)
    app.config.setdefault("DETECTOR", None)
    app.config.setdefault("LATEST_RESULT", {})
    app.config.setdefault("LATEST_JPEG", b"")
    app.config.setdefault("START_TIME", 0.0)
    app.config.setdefault("LOCK", threading.Lock())
    app.config.setdefault("CSV_PATH", None)
    app.config.setdefault("AUTO_ROI_INFO", {})

    @app.after_request
    def _add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS, POST"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    def _write_csv_row(result: dict) -> None:
        csv_path = app.config["CSV_PATH"]
        if csv_path is None:
            return
        row = {}
        for k in _CSV_FIELDS:
            v = result.get(k, "")
            if k == "event_type" and v is None:
                v = ""
            row[k] = _serialise_value(v)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writerow(row)

    @app.route("/health")
    def health():
        uptime = time.perf_counter() - app.config["START_TIME"]
        return {"status": "ok", "uptime_s": round(uptime, 2),
                "auto_roi_available": True}

    @app.route("/status")
    def status():
        with app.config["LOCK"]:
            result = dict(app.config["LATEST_RESULT"])

        detector = app.config["DETECTOR"]
        result["connected"] = (detector is not None and detector.started)
        result["server_time_s"] = time.perf_counter()
        result["auto_roi_available"] = True

        if detector is not None:
            result["roi"] = detector.roi
            result["roi_source"] = detector.roi_source
            result["roi_confidence"] = detector.roi_confidence
            result["threshold_on"] = detector.threshold_on
            result["threshold_off"] = detector.threshold_off
            result["detection_mode"] = detector.detection_mode
            result["percentile"] = detector.percentile
            result["use_adaptive"] = detector.use_adaptive

        return result

    @app.route("/auto_roi")
    def auto_roi():
        detector = app.config["DETECTOR"]
        if detector is None or not detector.started:
            return {"success": False, "message": "Detector not running"}, 503

        auto_dur = app.config.get("AUTO_ROI_DURATION", 3.0)
        auto_pad = app.config.get("AUTO_ROI_PADDING", 20)
        auto_min = app.config.get("AUTO_ROI_MIN_AREA", 50)

        result = _run_auto_roi(detector, duration_s=auto_dur,
                               padding_px=auto_pad, min_area_px=auto_min)
        app.config["AUTO_ROI_INFO"] = result

        if result["success"] and result["roi"] is not None:
            r = result["roi"]
            detector.roi = [r["x"], r["y"], r["width"], r["height"]]
            detector.roi_source = "auto"
            detector.roi_confidence = result["confidence"]
            result["message"] += " — ROI activated."
        return result

    @app.route("/clear_roi")
    def clear_roi():
        detector = app.config["DETECTOR"]
        if detector is None:
            return {"success": False, "message": "No detector"}, 503
        detector.roi = None
        detector.roi_source = "none"
        detector.roi_confidence = 0.0
        app.config["AUTO_ROI_INFO"] = {}
        return {"success": True, "message": "ROI cleared."}

    def generate_mjpeg():
        while True:
            with app.config["LOCK"]:
                jpeg = bytes(app.config["LATEST_JPEG"])
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(0.033)

    @app.route("/video_feed")
    def video_feed():
        assert _Response is not None
        return _Response(generate_mjpeg(),
                         mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def _capture_loop(
    detector: Any, app: Any, duration_s: float,
    csv_path: str | None, manual_camera: bool,
) -> None:
    assert _CV2 is not None
    print(f"[capture] Started.  CSV: {csv_path or 'disabled'}")
    start = time.perf_counter()

    if csv_path is not None:
        app.config["CSV_PATH"] = csv_path
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()

    try:
        while True:
            if duration_s > 0 and (time.perf_counter() - start) >= duration_s:
                print("[capture] Duration reached — stopping.")
                break

            result = detector.capture_frame()

            if csv_path is not None:
                row = {}
                for k in _CSV_FIELDS:
                    v = result.get(k, "")
                    if k == "event_type" and v is None:
                        v = ""
                    row[k] = _serialise_value(v)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                    writer.writerow(row)

            raw_frame = detector.capture_raw_frame()
            roi = detector.roi
            extra: dict = {}
            auto_info = app.config.get("AUTO_ROI_INFO", {})
            if auto_info.get("roi"):
                r = auto_info["roi"]
                extra["auto_candidate_roi"] = [r["x"], r["y"], r["width"], r["height"]]

            annotated = _draw_overlay(
                raw_frame, result, roi,
                roi_source=detector.roi_source,
                roi_confidence=detector.roi_confidence,
                extra=extra, manual_camera=manual_camera,
            )

            ok, jpeg_buf = _CV2.imencode(".jpg", annotated)
            jpeg_bytes = jpeg_buf.tobytes() if ok else b""

            with app.config["LOCK"]:
                app.config["LATEST_RESULT"] = result
                app.config["LATEST_JPEG"] = jpeg_bytes

    except KeyboardInterrupt:
        print("\n[capture] Interrupted.")
    except Exception as exc:
        print(f"[capture] Error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        print("[capture] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_deps()
    assert _Flask is not None and _PicameraFlashDetector is not None

    parser = argparse.ArgumentParser(
        description="Pi Camera Flash Detection MJPEG Server (Stage 2D).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--detection-mode", default="local_contrast",
                        choices=["mean", "top_percentile", "bright_blob", "local_contrast"])
    # Adaptive
    parser.add_argument("--adaptive", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--low-percentile", type=float, default=10)
    parser.add_argument("--high-percentile", type=float, default=90)
    parser.add_argument("--norm-on-threshold", type=float, default=0.65)
    parser.add_argument("--norm-off-threshold", type=float, default=0.35)
    parser.add_argument("--min-amplitude", type=float, default=10.0)
    parser.add_argument("--freq-min", type=float, default=0.2)
    parser.add_argument("--freq-max", type=float, default=5.0)
    # Fixed thresholds
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--blob-threshold", type=float, default=180)
    parser.add_argument("--threshold-on", type=float, default=180)
    parser.add_argument("--threshold-off", type=float, default=120)
    parser.add_argument("--min-interval", type=float, default=0.2)
    # ROI
    parser.add_argument("--roi", nargs=4, type=int, default=None,
                        metavar=("X", "Y", "W", "H"))
    parser.add_argument("--auto-roi", action="store_true")
    parser.add_argument("--auto-roi-duration", type=float, default=3.0)
    parser.add_argument("--auto-roi-padding", type=int, default=20)
    parser.add_argument("--auto-roi-min-area", type=int, default=50)
    # Camera
    parser.add_argument("--manual-camera", action="store_true")
    parser.add_argument("--exposure-us", type=int, default=None)
    parser.add_argument("--analogue-gain", type=float, default=None)
    parser.add_argument("--awb-enable", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--target-fps", type=int, default=30)
    # Output
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # CSV path
    if args.output:
        csv_path = args.output
    else:
        log_dir = Path("experiments") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = str(log_dir / f"pi_camera_stream_{ts}.csv")

    # ROI
    manual_roi = list(args.roi) if args.roi is not None else None
    roi_source = "manual" if manual_roi is not None else "none"

    # Detector
    detector = _PicameraFlashDetector(
        resolution=[args.width, args.height],
        detection_mode=args.detection_mode,
        roi=manual_roi,
        threshold_on=args.threshold_on,
        threshold_off=args.threshold_off,
        min_interval_s=args.min_interval,
        percentile=args.percentile,
        blob_threshold=args.blob_threshold,
        use_adaptive=args.adaptive and args.detection_mode == "local_contrast",
        window_s=args.window_s,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        norm_on_threshold=args.norm_on_threshold,
        norm_off_threshold=args.norm_off_threshold,
        min_amplitude=args.min_amplitude,
        roi_source=roi_source,
        target_fps=args.target_fps,
    )

    app = create_app()
    app.config["DETECTOR"] = detector
    app.config["START_TIME"] = time.perf_counter()
    app.config["AUTO_ROI_DURATION"] = args.auto_roi_duration
    app.config["AUTO_ROI_PADDING"] = args.auto_roi_padding
    app.config["AUTO_ROI_MIN_AREA"] = args.auto_roi_min_area

    print("Starting camera...")
    detector.start()

    # Apply manual camera controls
    _apply_camera_controls(
        detector,
        manual=args.manual_camera,
        exposure_us=args.exposure_us,
        analogue_gain=args.analogue_gain,
        awb_enable=args.awb_enable,
    )

    print(f"  Resolution:     {args.width}x{args.height}")
    print(f"  Detection mode: {args.detection_mode}")
    print(f"  Adaptive:       {detector.use_adaptive}")
    print(f"  ROI:            {manual_roi or 'full frame'}")
    print(f"  Thresholds:     ON={args.threshold_on} OFF={args.threshold_off}")
    print(f"  Min interval:   {args.min_interval}s")
    print(f"  Manual camera:  {args.manual_camera}")
    print(f"  CSV log:        {csv_path}")
    print(f"  Server:         http://{args.host}:{args.port}")

    if args.auto_roi:
        print(f"  Auto-ROI:       enabled (duration={args.auto_roi_duration}s)")
        auto_result = _run_auto_roi(
            detector, duration_s=args.auto_roi_duration,
            padding_px=args.auto_roi_padding, min_area_px=args.auto_roi_min_area,
        )
        app.config["AUTO_ROI_INFO"] = auto_result
        if auto_result["success"] and auto_result["roi"] is not None:
            r = auto_result["roi"]
            detector.roi = [r["x"], r["y"], r["width"], r["height"]]
            detector.roi_source = "auto"
            detector.roi_confidence = auto_result["confidence"]
            print(f"  Auto-ROI result: {r} (conf={auto_result['confidence']:.3f})")
        else:
            print(f"  Auto-ROI result: FAILED — {auto_result['message']}")
    print()

    capture_thread = threading.Thread(
        target=_capture_loop,
        args=(detector, app, args.duration, csv_path, args.manual_camera),
        daemon=True,
    )
    capture_thread.start()

    try:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nServer interrupted.")
    finally:
        print("Shutting down camera...")
        detector.stop()
        print("Done.")


if __name__ == "__main__":
    main()

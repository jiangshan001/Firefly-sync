"""Picamera2-based flash detection with hysteresis + adaptive normalisation.

Wraps the pure-Python flash detector, brightness helpers, and the
``RollingAdaptiveSignalDetector``.  All hardware imports are lazy.

Detection modes
---------------
* ``"local_contrast"`` — top-percentile minus median inside ROI,
  fed through rolling adaptive normalisation.  **Default.**
* ``"top_percentile"`` — high-percentile brightness.
* ``"mean"`` — full-frame or ROI mean brightness.
* ``"bright_blob"`` — largest bright connected-component brightness.
"""

from __future__ import annotations

import time
from typing import Any

from firefly_sync.hardware.flash_detector import (
    FlashDetectorState,
    FlashEpisodeLatchState,
    compute_bright_blob_metrics,
    compute_local_contrast,
    compute_roi_mean_brightness,
    compute_top_percentile_brightness,
    normalize_frame_to_grayscale,
    update_flash_episode_latch,
    update_flash_detector,
)
from firefly_sync.hardware.signal_detector import (
    AdaptiveDetectorState,
    RollingAdaptiveSignalDetector,
    estimate_frequency_autocorrelation,
)


# Delayed imports
_PICAM2 = None
_NP = None

_PICAMERA2_MSG = (
    "picamera2 and numpy are required for PicameraFlashDetector.\n"
    "On Raspberry Pi OS:\n"
    "  sudo apt install -y python3-picamera2 python3-numpy\n"
    "Or via pip:\n"
    "  pip install picamera2 numpy"
)


def _ensure_picamera2():
    global _PICAM2, _NP
    if _PICAM2 is not None and _NP is not None:
        return
    try:
        from picamera2 import Picamera2 as _P2  # type: ignore[import-untyped]
        import numpy as _numpy  # type: ignore[import-untyped]
        _PICAM2 = _P2
        _NP = _numpy
    except ImportError:
        raise ImportError(_PICAMERA2_MSG)


# ---------------------------------------------------------------------------
# PicameraFlashDetector
# ---------------------------------------------------------------------------

class PicameraFlashDetector:
    """Real-time leader-flash detector using a Pi camera.

    Parameters
    ----------
    resolution:
        Camera resolution as ``[width, height]``.  Default [640, 480].
    detection_mode:
        ``"local_contrast"`` (default), ``"top_percentile"``,
        ``"mean"``, or ``"bright_blob"``.
    roi:
        Optional ROI ``[x, y, w, h]``.  None = full frame.
    threshold_on / threshold_off:
        Fixed brightness thresholds (used by legacy modes).
    min_interval_s:
        Minimum interval between successive rising edges.
    percentile:
        Percentile for top_percentile and local_contrast modes.
    blob_threshold:
        Brightness threshold for bright_blob mode.
    use_adaptive:
        If True, use RollingAdaptiveSignalDetector for hysteresis.
    window_s / low_percentile / high_percentile /
    norm_on_threshold / norm_off_threshold / min_amplitude:
        Parameters forwarded to RollingAdaptiveSignalDetector.
    roi_source / roi_confidence:
        Metadata about ROI origin.
    target_fps / camera_id:
        Picamera2 config.
    """

    _VALID_MODES = ("mean", "top_percentile", "bright_blob", "local_contrast")

    def __init__(
        self,
        resolution: list[int] | None = None,
        detection_mode: str = "local_contrast",
        roi: list[int] | None = None,
        threshold_on: float = 180.0,
        threshold_off: float = 120.0,
        min_interval_s: float = 0.2,
        percentile: float = 99.0,
        blob_threshold: float = 180.0,
        use_adaptive: bool = True,
        window_s: float = 5.0,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        norm_on_threshold: float = 0.65,
        norm_off_threshold: float = 0.35,
        min_amplitude: float = 10.0,
        roi_source: str = "none",
        roi_confidence: float = 0.0,
        target_fps: int = 30,
        camera_id: int = 0,
        use_video_config: bool = False,
        frame_duration_limits: tuple[int, int] | None = None,
        frame_rate: float | None = None,
        exposure_time_us: int | None = None,
        analogue_gain: float | None = None,
        camera_format: str | None = None,
        episode_latch_enabled: bool = False,
        rearm_off_duration_s: float = 0.05,
        off_frames_to_rearm: int = 2,
        rearm_requires_both: bool = False,
    ) -> None:
        _ensure_picamera2()
        assert _PICAM2 is not None and _NP is not None

        if resolution is None:
            resolution = [640, 480]
        if detection_mode not in self._VALID_MODES:
            raise ValueError(
                f"detection_mode must be one of {self._VALID_MODES}, "
                f"got '{detection_mode}'"
            )

        self._resolution = resolution
        self._detection_mode = detection_mode
        self._roi = roi
        self._roi_source = roi_source
        self._roi_confidence = roi_confidence
        self._threshold_on = threshold_on
        self._threshold_off = threshold_off
        self._min_interval_s = min_interval_s
        self._percentile = percentile
        self._blob_threshold = blob_threshold
        self._target_fps = target_fps
        self._camera_id = camera_id
        self._use_video_config = use_video_config
        self._camera_format_requested = camera_format
        self._episode_latch_enabled = episode_latch_enabled
        self._rearm_off_duration_s = rearm_off_duration_s
        self._off_frames_to_rearm = off_frames_to_rearm
        self._rearm_requires_both = rearm_requires_both
        self._camera_controls_requested: dict[str, Any] = {}
        if frame_duration_limits is not None:
            self._camera_controls_requested["FrameDurationLimits"] = tuple(frame_duration_limits)
        if frame_rate is not None:
            self._camera_controls_requested["FrameRate"] = float(frame_rate)
        if exposure_time_us is not None:
            self._camera_controls_requested["ExposureTime"] = int(exposure_time_us)
        if analogue_gain is not None:
            self._camera_controls_requested["AnalogueGain"] = float(analogue_gain)
        self._camera_config_actual: dict[str, Any] | None = None
        self._camera_controls_actual: dict[str, Any] = {}

        # Adaptive detector (used by local_contrast mode)
        self._use_adaptive = use_adaptive and (detection_mode == "local_contrast")
        self._adaptive = RollingAdaptiveSignalDetector(
            window_s=window_s,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            norm_on_threshold=norm_on_threshold,
            norm_off_threshold=norm_off_threshold,
            min_interval_s=min_interval_s,
            min_amplitude=min_amplitude,
        )

        # Internal state
        self._picam2: Any = None
        self._frame_index: int = 0
        self._start_time_s: float = 0.0
        self._detector_state: FlashDetectorState = FlashDetectorState()
        self._adaptive_state: AdaptiveDetectorState = AdaptiveDetectorState()
        self._episode_latch_state: FlashEpisodeLatchState = FlashEpisodeLatchState()
        self._started: bool = False

        # Signal histories for autocorrelation (sliding windows)
        self._signal_times: list[float] = []
        self._signal_values: list[float] = []
        self._signal_max_len: int = 300  # ~10 s at 30 FPS

        # Latest metrics
        self._last_brightness_used: float = 0.0
        self._last_full_frame_mean: float = 0.0
        self._last_top_percentile_brightness: float = 0.0
        self._last_blob_metrics: dict[str, Any] = {}
        self._last_local_contrast: dict[str, Any] = {}
        self._last_signal_norm: float = 0.0
        self._last_signal_freq_hz: float = 0.0
        self._last_periodicity_conf: float = 0.0

    # ---- Properties ----

    @property
    def threshold_on(self) -> float:
        return self._threshold_on

    @threshold_on.setter
    def threshold_on(self, v: float) -> None:
        self._threshold_on = v

    @property
    def threshold_off(self) -> float:
        return self._threshold_off

    @threshold_off.setter
    def threshold_off(self, v: float) -> None:
        self._threshold_off = v

    @property
    def roi(self) -> list[int] | None:
        return self._roi

    @roi.setter
    def roi(self, v: list[int] | None) -> None:
        self._roi = v

    @property
    def roi_source(self) -> str:
        return self._roi_source

    @roi_source.setter
    def roi_source(self, v: str) -> None:
        self._roi_source = v

    @property
    def roi_confidence(self) -> float:
        return self._roi_confidence

    @roi_confidence.setter
    def roi_confidence(self, v: float) -> None:
        self._roi_confidence = v

    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    @property
    def percentile(self) -> float:
        return self._percentile

    @property
    def blob_threshold(self) -> float:
        return self._blob_threshold

    @property
    def use_adaptive(self) -> bool:
        return self._use_adaptive

    @property
    def resolution(self) -> list[int]:
        return self._resolution

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def started(self) -> bool:
        return self._started

    @property
    def camera_controls_requested(self) -> dict[str, Any]:
        return dict(self._camera_controls_requested)

    @property
    def camera_config_actual(self) -> dict[str, Any] | None:
        return self._camera_config_actual

    @property
    def camera_controls_actual(self) -> dict[str, Any]:
        return dict(self._camera_controls_actual)

    # ---- Lifecycle ----

    def start(self) -> None:
        if self._started:
            return
        _ensure_picamera2()
        assert _PICAM2 is not None and _NP is not None

        self._picam2 = _PICAM2(self._camera_id)
        create_config = (
            self._picam2.create_video_configuration
            if self._use_video_config
            else self._picam2.create_still_configuration
        )
        config_kwargs: dict[str, Any] = {
            "main": {"size": (self._resolution[0], self._resolution[1])},
        }
        if self._camera_format_requested is not None:
            config_kwargs["main"]["format"] = self._camera_format_requested
        if self._camera_controls_requested:
            config_kwargs["controls"] = self._camera_controls_requested
        try:
            config = create_config(**config_kwargs)
        except Exception:
            config_kwargs["main"].pop("format", None)
            config = create_config(main=config_kwargs["main"])
        self._picam2.configure(config)
        if self._camera_controls_requested:
            try:
                self._picam2.set_controls(self._camera_controls_requested)
            except Exception:
                pass
        self._picam2.start()
        try:
            self._camera_config_actual = self._picam2.camera_configuration()
        except Exception:
            self._camera_config_actual = config
        self._camera_controls_actual = dict(self._camera_controls_requested)

        self._start_time_s = time.perf_counter()
        self._frame_index = 0
        self._detector_state = FlashDetectorState()
        self._adaptive.reset()
        self._adaptive_state = AdaptiveDetectorState()
        self._episode_latch_state = FlashEpisodeLatchState()
        self._signal_times.clear()
        self._signal_values.clear()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._picam2 is not None:
                self._picam2.stop()
                self._picam2.close()
        except Exception:
            pass
        self._picam2 = None
        self._started = False

    def close(self) -> None:
        self.stop()

    def capture_raw_frame(self) -> Any:
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")
        return self._picam2.capture_array()

    # ---- Frame capture ----

    def capture_frame(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")

        assert self._picam2 is not None and _NP is not None

        frame: Any = self._picam2.capture_array()
        return self.process_frame(frame, now_s=time.perf_counter())

    def process_frame(self, frame: Any, now_s: float | None = None) -> dict[str, Any]:
        """Process a raw camera frame through the detector state machine."""
        if not self._started:
            raise RuntimeError("Not started. Call start() first.")

        assert _NP is not None

        if now_s is None:
            now_s = time.perf_counter()
        elapsed = now_s - self._start_time_s
        raw_shape = list(getattr(frame, "shape", ()))
        raw_dtype = str(getattr(frame, "dtype", "unknown"))
        normalized_frame = normalize_frame_to_grayscale(frame)
        normalized_shape = list(normalized_frame.shape)
        normalized_dtype = str(normalized_frame.dtype)

        # --- Brightness metrics ---
        full_frame_mean = compute_roi_mean_brightness(normalized_frame, roi=None)
        roi_mean = compute_roi_mean_brightness(normalized_frame, roi=self._roi)
        top_pct = compute_top_percentile_brightness(
            normalized_frame, percentile=self._percentile, roi=self._roi,
        )
        blob = compute_bright_blob_metrics(
            normalized_frame, threshold=self._blob_threshold, roi=self._roi,
        )
        local_ctrst = compute_local_contrast(
            normalized_frame, roi=self._roi, percentile=self._percentile,
        )

        # --- Detection-mode-specific brightness_used ---
        if self._detection_mode == "local_contrast":
            brightness_used = local_ctrst["local_contrast"]
        elif self._detection_mode == "top_percentile":
            brightness_used = top_pct
        elif self._detection_mode == "bright_blob":
            brightness_used = blob.get("blob_mean_brightness", 0.0) if blob.get("blob_found") else 0.0
        else:  # mean
            brightness_used = roi_mean if self._roi is not None else full_frame_mean

        # Store latest metrics
        self._last_brightness_used = brightness_used
        self._last_full_frame_mean = full_frame_mean
        self._last_top_percentile_brightness = top_pct
        self._last_blob_metrics = blob
        self._last_local_contrast = local_ctrst

        # --- Hysteresis (adaptive or fixed) ---
        if self._use_adaptive and self._detection_mode == "local_contrast":
            self._adaptive_state = self._adaptive.update(brightness_used, now_s)
            state_str = self._adaptive_state.state
            event_type = self._adaptive_state.event_type
            rising_edge_count = self._adaptive_state.rising_edge_count
            last_rising_edge_time_s = self._adaptive_state.last_rising_edge_time_s
            estimated_frequency_hz = self._adaptive_state.signal_frequency_hz
            signal_norm = self._adaptive_state.signal_norm
            adaptive_low = self._adaptive_state.adaptive_low
            adaptive_high = self._adaptive_state.adaptive_high
            adaptive_amplitude = self._adaptive_state.adaptive_amplitude
            signal_quality = self._adaptive_state.signal_quality
            norm_on = self._adaptive_state.norm_on_threshold
            norm_off = self._adaptive_state.norm_off_threshold
            sig_freq_hz = self._adaptive_state.signal_frequency_hz
            period_conf = self._adaptive_state.periodicity_confidence
        else:
            self._detector_state = update_flash_detector(
                brightness=brightness_used,
                prev_state=self._detector_state,
                now_s=now_s,
                threshold_on=self._threshold_on,
                threshold_off=self._threshold_off,
                min_interval_s=self._min_interval_s,
            )
            state_str = self._detector_state.state
            event_type = self._detector_state.event_type
            rising_edge_count = self._detector_state.rising_edge_count
            last_rising_edge_time_s = self._detector_state.last_rising_edge_time_s
            estimated_frequency_hz = self._detector_state.estimated_frequency_hz
            signal_norm = 0.0
            adaptive_low = 0.0
            adaptive_high = 0.0
            adaptive_amplitude = 0.0
            signal_quality = 0.0
            norm_on = self._threshold_on
            norm_off = self._threshold_off
            sig_freq_hz = 0.0
            period_conf = 0.0

            # Still maintain signal history for autocorrelation
            self._signal_times.append(now_s)
            self._signal_values.append(brightness_used)
            if len(self._signal_times) > self._signal_max_len:
                self._signal_times.pop(0)
                self._signal_values.pop(0)

        # --- Autocorrelation on raw signal (all modes) ---
        if self._use_adaptive:
            # Already done inside adaptive detector
            self._last_signal_freq_hz = sig_freq_hz
            self._last_periodicity_conf = period_conf
            self._last_signal_norm = signal_norm
        else:
            # Compute autocorrelation from signal history
            self._signal_times.append(now_s)
            self._signal_values.append(brightness_used)
            if len(self._signal_times) > self._signal_max_len:
                self._signal_times.pop(0)
                self._signal_values.pop(0)
            sig_freq_hz, period_conf = estimate_frequency_autocorrelation(
                self._signal_times, self._signal_values,
            )
            self._last_signal_freq_hz = sig_freq_hz
            self._last_periodicity_conf = period_conf
            self._last_signal_norm = 0.0

        if self._use_adaptive and self._detection_mode == "local_contrast":
            latch_signal = signal_norm
            latch_on_threshold = norm_on
            latch_off_threshold = norm_off
        else:
            latch_signal = brightness_used
            latch_on_threshold = self._threshold_on
            latch_off_threshold = self._threshold_off

        if self._episode_latch_enabled:
            self._episode_latch_state = update_flash_episode_latch(
                signal_value=float(latch_signal),
                prev_state=self._episode_latch_state,
                now_s=now_s,
                threshold_on=float(latch_on_threshold),
                threshold_off=float(latch_off_threshold),
                min_interval_s=self._min_interval_s,
                rearm_off_duration_s=self._rearm_off_duration_s,
                off_frames_to_rearm=self._off_frames_to_rearm,
                rearm_requires_both=self._rearm_requires_both,
            )
            event_type = self._episode_latch_state.event_type
            rising_edge_count = self._episode_latch_state.accepted_event_count
            last_rising_edge_time_s = self._episode_latch_state.last_event_time_s

        self._frame_index += 1
        fps_estimate = self._frame_index / elapsed if elapsed > 0 else 0.0
        accepted_flash_event = event_type == "leader_rising_edge"

        # ROI dict for serialisation
        roi_dict = None
        if self._roi is not None:
            roi_dict = {"x": self._roi[0], "y": self._roi[1],
                        "width": self._roi[2], "height": self._roi[3]}
        selected_blob_bbox = blob.get("blob_bbox")
        selected_blob_x = None
        selected_blob_y = None
        selected_blob_bbox_full = None
        selected_blob_inside_roi = self._roi is not None
        if selected_blob_bbox is not None:
            bx, by, bw, bh = selected_blob_bbox
            ox = self._roi[0] if self._roi is not None else 0
            oy = self._roi[1] if self._roi is not None else 0
            selected_blob_x = ox + bx + bw / 2.0
            selected_blob_y = oy + by + bh / 2.0
            selected_blob_bbox_full = [ox + bx, oy + by, bw, bh]

        return {
            "timestamp_s": now_s,
            "elapsed_time_s": elapsed,
            "frame_index": self._frame_index,
            # Brightness
            "brightness_mean": brightness_used,
            "brightness_used": brightness_used,
            "full_frame_mean": round(full_frame_mean, 2),
            "roi_mean": round(roi_mean, 2) if self._roi is not None else None,
            "top_percentile_brightness": round(top_pct, 2),
            "percentile": self._percentile,
            # Blob
            "blob_found": blob.get("blob_found", False),
            "blob_area_px": blob.get("blob_area_px", 0),
            "blob_count": blob.get("blob_count", 0),
            "blob_mean_brightness": blob.get("blob_mean_brightness", 0.0),
            "blob_max_brightness": blob.get("blob_max_brightness", 0.0),
            "blob_bbox": blob.get("blob_bbox"),
            "selected_blob_x": selected_blob_x,
            "selected_blob_y": selected_blob_y,
            "selected_blob_area": blob.get("blob_area_px", 0),
            "selected_blob_mean": blob.get("blob_mean_brightness", 0.0),
            "selected_blob_max": blob.get("blob_max_brightness", 0.0),
            "selected_blob_bbox_full": selected_blob_bbox_full,
            "selected_blob_inside_roi": selected_blob_inside_roi,
            # Local contrast
            "roi_median_brightness": local_ctrst["roi_median_brightness"],
            "roi_top_percentile_brightness": local_ctrst["roi_top_percentile_brightness"],
            "local_contrast": local_ctrst["local_contrast"],
            "local_contrast_ratio": local_ctrst["local_contrast_ratio"],
            # Adaptive
            "signal_norm": signal_norm,
            "adaptive_low": round(adaptive_low, 2),
            "adaptive_high": round(adaptive_high, 2),
            "adaptive_amplitude": round(adaptive_amplitude, 2),
            "signal_quality": signal_quality,
            "norm_on_threshold": norm_on,
            "norm_off_threshold": norm_off,
            # Frequency
            "signal_frequency_hz": round(sig_freq_hz, 3),
            "periodicity_confidence": round(period_conf, 4),
            # Detection
            "detection_mode": self._detection_mode,
            "state": state_str,
            "event_type": event_type,
            "accepted_flash_event": accepted_flash_event,
            "rising_edge_count": rising_edge_count,
            "estimated_frequency_hz": estimated_frequency_hz,
            "last_rising_edge_time_s": last_rising_edge_time_s,
            "detector_episode_latch_enabled": self._episode_latch_enabled,
            "episode_currently_on": self._episode_latch_state.currently_on,
            "episode_armed": self._episode_latch_state.armed,
            "duplicate_suppressed_count": self._episode_latch_state.duplicate_suppressed_count,
            "raw_on_threshold_crossing_count": self._episode_latch_state.raw_on_threshold_crossing_count,
            "accepted_flash_event_count": self._episode_latch_state.accepted_event_count,
            "rearm_off_duration_s": self._rearm_off_duration_s,
            "off_frames_to_rearm": self._off_frames_to_rearm,
            "rearm_requires_both": self._rearm_requires_both,
            "detector_raw_on_crossing": self._episode_latch_state.raw_on_crossing,
            "detector_rearmed_event": self._episode_latch_state.rearmed_event,
            "detector_time_since_last_accepted": self._episode_latch_state.time_since_last_event_s,
            "detector_below_off_duration": self._episode_latch_state.below_off_duration_s,
            "detector_off_frame_count": self._episode_latch_state.off_frame_count,
            "detector_reject_reason": self._episode_latch_state.reject_reason,
            "latch_signal": round(float(latch_signal), 4),
            "latch_on_threshold": latch_on_threshold,
            "latch_off_threshold": latch_off_threshold,
            # Params
            "threshold_on": self._threshold_on,
            "threshold_off": self._threshold_off,
            "roi": roi_dict,
            "roi_source": self._roi_source,
            "roi_confidence": self._roi_confidence,
            "fps_estimate": round(fps_estimate, 2),
            "use_adaptive": self._use_adaptive,
            "camera_config_actual": self._camera_config_actual,
            "camera_controls_requested": self._camera_controls_requested,
            "camera_controls_actual": self._camera_controls_actual,
            "camera_format_requested": self._camera_format_requested,
            "raw_frame_shape": raw_shape,
            "raw_frame_dtype": raw_dtype,
            "normalized_frame_shape": normalized_shape,
            "normalized_frame_dtype": normalized_dtype,
        }

    def __enter__(self) -> PicameraFlashDetector:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

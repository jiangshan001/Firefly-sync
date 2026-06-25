"""Multi-ROI flash detection for two virtual targets.

This wrapper reuses the same brightness extraction, adaptive normalisation,
hysteresis, and optional episode latch logic as the single-ROI Pi detector,
but keeps independent state per ROI.  It is importable and unit-testable on
non-Pi machines; Picamera2 is imported only when ``start()`` is called.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from firefly_sync.hardware.flash_detector import (
    FlashDetectorState,
    FlashEpisodeLatchState,
    compute_bright_blob_metrics,
    compute_local_contrast,
    compute_roi_mean_brightness,
    compute_top_percentile_brightness,
    normalize_frame_to_grayscale,
    update_flash_detector,
    update_flash_episode_latch,
)
from firefly_sync.hardware.signal_detector import (
    AdaptiveDetectorState,
    RollingAdaptiveSignalDetector,
    estimate_frequency_autocorrelation,
)


_PICAM2 = None


def _ensure_picamera2():
    global _PICAM2
    if _PICAM2 is not None:
        return
    try:
        from picamera2 import Picamera2 as _P2  # type: ignore[import-untyped]
        _PICAM2 = _P2
    except ImportError as exc:
        raise ImportError(
            "picamera2 is required to capture frames with MultiROIFlashDetector. "
            "Synthetic process_frame() tests do not require it."
        ) from exc


@dataclass(frozen=True)
class ROIFlashConfig:
    roi_id: int
    agent_id: str
    roi: list[int]


class _ROIDetectorState:
    def __init__(
        self,
        config: ROIFlashConfig,
        *,
        window_s: float,
        low_percentile: float,
        high_percentile: float,
        norm_on_threshold: float,
        norm_off_threshold: float,
        min_interval_s: float,
        min_amplitude: float,
    ) -> None:
        self.config = config
        self.fixed_state = FlashDetectorState()
        self.adaptive = RollingAdaptiveSignalDetector(
            window_s=window_s,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            norm_on_threshold=norm_on_threshold,
            norm_off_threshold=norm_off_threshold,
            min_interval_s=min_interval_s,
            min_amplitude=min_amplitude,
        )
        self.adaptive_state = AdaptiveDetectorState()
        self.latch_state = FlashEpisodeLatchState()
        self.signal_times: list[float] = []
        self.signal_values: list[float] = []
        self.accepted_events = 0

    def reset(self) -> None:
        self.fixed_state = FlashDetectorState()
        self.adaptive.reset()
        self.adaptive_state = AdaptiveDetectorState()
        self.latch_state = FlashEpisodeLatchState()
        self.signal_times.clear()
        self.signal_values.clear()
        self.accepted_events = 0


class MultiROIFlashDetector:
    """Process one camera frame through independent per-ROI detectors."""

    _VALID_MODES = ("mean", "top_percentile", "bright_blob", "local_contrast")

    def __init__(
        self,
        rois: list[ROIFlashConfig] | list[dict[str, Any]],
        resolution: list[int] | None = None,
        detection_mode: str = "local_contrast",
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
        target_fps: int = 30,
        camera_id: int = 0,
        use_video_config: bool = False,
        camera_format: str | None = None,
        frame_duration_limits: tuple[int, int] | None = None,
        frame_rate: float | None = None,
        exposure_time_us: int | None = None,
        analogue_gain: float | None = None,
        episode_latch_enabled: bool = False,
        rearm_off_duration_s: float = 0.05,
        off_frames_to_rearm: int = 2,
        rearm_requires_both: bool = False,
    ) -> None:
        if detection_mode not in self._VALID_MODES:
            raise ValueError(f"detection_mode must be one of {self._VALID_MODES}")
        if not rois:
            raise ValueError("At least one ROI is required")

        self._resolution = resolution or [640, 480]
        self._detection_mode = detection_mode
        self._threshold_on = threshold_on
        self._threshold_off = threshold_off
        self._min_interval_s = min_interval_s
        self._percentile = percentile
        self._blob_threshold = blob_threshold
        self._use_adaptive = use_adaptive and detection_mode == "local_contrast"
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

        self._roi_states: list[_ROIDetectorState] = []
        for item in rois:
            cfg = item if isinstance(item, ROIFlashConfig) else ROIFlashConfig(
                roi_id=int(item["roi_id"]),
                agent_id=str(item["agent_id"]),
                roi=list(item["roi"]),
            )
            self._roi_states.append(
                _ROIDetectorState(
                    cfg,
                    window_s=window_s,
                    low_percentile=low_percentile,
                    high_percentile=high_percentile,
                    norm_on_threshold=norm_on_threshold,
                    norm_off_threshold=norm_off_threshold,
                    min_interval_s=min_interval_s,
                    min_amplitude=min_amplitude,
                )
            )

        self._picam2: Any = None
        self._started = False
        self._start_time_s = 0.0
        self._frame_index = 0
        self._camera_config_actual: dict[str, Any] | None = None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def rois(self) -> list[dict[str, Any]]:
        return [
            {"roi_id": st.config.roi_id, "agent_id": st.config.agent_id, "roi": list(st.config.roi)}
            for st in self._roi_states
        ]

    @property
    def camera_config_actual(self) -> dict[str, Any] | None:
        return self._camera_config_actual

    @property
    def camera_controls_requested(self) -> dict[str, Any]:
        return dict(self._camera_controls_requested)

    def reset(self) -> None:
        for state in self._roi_states:
            state.reset()
        self._frame_index = 0
        self._start_time_s = time.perf_counter()

    def start(self) -> None:
        if self._started:
            return
        _ensure_picamera2()
        assert _PICAM2 is not None
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
        self._started = True
        self.reset()

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

    def capture_frame(self) -> dict[str, Any]:
        frame = self.capture_raw_frame()
        return self.process_frame(frame, now_s=time.perf_counter())

    def process_frame(self, frame: Any, now_s: float | None = None) -> dict[str, Any]:
        """Return per-ROI detection results for one frame."""
        if now_s is None:
            now_s = time.perf_counter()
        if self._start_time_s <= 0:
            self._start_time_s = now_s
        elapsed = now_s - self._start_time_s
        normalized_frame = normalize_frame_to_grayscale(frame)
        full_frame_mean = compute_roi_mean_brightness(normalized_frame, roi=None)
        self._frame_index += 1

        results = [
            self._process_roi(normalized_frame, state, now_s, elapsed, full_frame_mean)
            for state in self._roi_states
        ]
        events = [row for row in results if row["accepted_flash_event"]]
        return {
            "timestamp_s": now_s,
            "elapsed_time_s": elapsed,
            "frame_index": self._frame_index,
            "roi_results": results,
            "events": events,
            "event_count": len(events),
            "raw_frame_shape": list(getattr(frame, "shape", ())),
            "normalized_frame_shape": list(getattr(normalized_frame, "shape", ())),
            "detection_mode": self._detection_mode,
        }

    def _process_roi(
        self,
        frame: Any,
        state: _ROIDetectorState,
        now_s: float,
        elapsed: float,
        full_frame_mean: float,
    ) -> dict[str, Any]:
        roi = state.config.roi
        roi_mean = compute_roi_mean_brightness(frame, roi=roi)
        top_pct = compute_top_percentile_brightness(frame, percentile=self._percentile, roi=roi)
        blob = compute_bright_blob_metrics(frame, threshold=self._blob_threshold, roi=roi)
        local_ctrst = compute_local_contrast(frame, roi=roi, percentile=self._percentile)

        if self._detection_mode == "local_contrast":
            brightness_used = local_ctrst["local_contrast"]
        elif self._detection_mode == "top_percentile":
            brightness_used = top_pct
        elif self._detection_mode == "bright_blob":
            brightness_used = blob.get("blob_mean_brightness", 0.0) if blob.get("blob_found") else 0.0
        else:
            brightness_used = roi_mean

        if self._use_adaptive:
            state.adaptive_state = state.adaptive.update(brightness_used, now_s)
            state_str = state.adaptive_state.state
            event_type = state.adaptive_state.event_type
            rising_edge_count = state.adaptive_state.rising_edge_count
            last_rising_edge_time_s = state.adaptive_state.last_rising_edge_time_s
            estimated_frequency_hz = state.adaptive_state.signal_frequency_hz
            signal_norm = state.adaptive_state.signal_norm
            adaptive_low = state.adaptive_state.adaptive_low
            adaptive_high = state.adaptive_state.adaptive_high
            adaptive_amplitude = state.adaptive_state.adaptive_amplitude
            signal_quality = state.adaptive_state.signal_quality
            norm_on = state.adaptive_state.norm_on_threshold
            norm_off = state.adaptive_state.norm_off_threshold
            sig_freq_hz = state.adaptive_state.signal_frequency_hz
            period_conf = state.adaptive_state.periodicity_confidence
        else:
            state.fixed_state = update_flash_detector(
                brightness=brightness_used,
                prev_state=state.fixed_state,
                now_s=now_s,
                threshold_on=self._threshold_on,
                threshold_off=self._threshold_off,
                min_interval_s=self._min_interval_s,
            )
            state_str = state.fixed_state.state
            event_type = state.fixed_state.event_type
            rising_edge_count = state.fixed_state.rising_edge_count
            last_rising_edge_time_s = state.fixed_state.last_rising_edge_time_s
            estimated_frequency_hz = state.fixed_state.estimated_frequency_hz
            signal_norm = 0.0
            adaptive_low = 0.0
            adaptive_high = 0.0
            adaptive_amplitude = 0.0
            signal_quality = 0.0
            norm_on = self._threshold_on
            norm_off = self._threshold_off
            state.signal_times.append(now_s)
            state.signal_values.append(brightness_used)
            if len(state.signal_times) > 300:
                state.signal_times.pop(0)
                state.signal_values.pop(0)
            sig_freq_hz, period_conf = estimate_frequency_autocorrelation(
                state.signal_times, state.signal_values,
            )

        latch_signal = signal_norm if self._use_adaptive else brightness_used
        if self._episode_latch_enabled:
            state.latch_state = update_flash_episode_latch(
                signal_value=float(latch_signal),
                prev_state=state.latch_state,
                now_s=now_s,
                threshold_on=float(norm_on),
                threshold_off=float(norm_off),
                min_interval_s=self._min_interval_s,
                rearm_off_duration_s=self._rearm_off_duration_s,
                off_frames_to_rearm=self._off_frames_to_rearm,
                rearm_requires_both=self._rearm_requires_both,
            )
            event_type = state.latch_state.event_type
            rising_edge_count = state.latch_state.accepted_event_count
            last_rising_edge_time_s = state.latch_state.last_event_time_s

        accepted = event_type == "leader_rising_edge"
        if accepted:
            state.accepted_events += 1

        roi_dict = {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        return {
            "timestamp_s": now_s,
            "elapsed_time_s": elapsed,
            "frame_index": self._frame_index,
            "roi_id": state.config.roi_id,
            "agent_id": state.config.agent_id,
            "source_agent_id": state.config.agent_id,
            "roi": roi_dict,
            "brightness_used": round(float(brightness_used), 4),
            "brightness_mean": round(float(brightness_used), 4),
            "raw_brightness": round(float(brightness_used), 4),
            "full_frame_mean": round(float(full_frame_mean), 2),
            "roi_mean": round(float(roi_mean), 2),
            "top_percentile_brightness": round(float(top_pct), 2),
            "roi_median_brightness": local_ctrst["roi_median_brightness"],
            "roi_top_percentile_brightness": local_ctrst["roi_top_percentile_brightness"],
            "local_contrast": local_ctrst["local_contrast"],
            "local_contrast_ratio": local_ctrst["local_contrast_ratio"],
            "signal_norm": signal_norm,
            "normalized_signal": signal_norm,
            "norm_on_threshold": norm_on,
            "norm_off_threshold": norm_off,
            "threshold_on": self._threshold_on,
            "threshold_off": self._threshold_off,
            "latch_on_threshold": norm_on,
            "latch_off_threshold": norm_off,
            "adaptive_low": round(float(adaptive_low), 2),
            "adaptive_high": round(float(adaptive_high), 2),
            "adaptive_amplitude": round(float(adaptive_amplitude), 2),
            "signal_quality": signal_quality,
            "state": state_str,
            "event_type": event_type,
            "accepted_flash_event": accepted,
            "rising_edge": accepted,
            "rising_edge_count": rising_edge_count,
            "accepted_flash_event_count": state.accepted_events,
            "last_rising_edge_time_s": last_rising_edge_time_s,
            "estimated_frequency_hz": estimated_frequency_hz,
            "signal_frequency_hz": round(float(sig_freq_hz), 3),
            "periodicity_confidence": round(float(period_conf), 4),
            "detection_mode": self._detection_mode,
            "use_adaptive": self._use_adaptive,
            "detector_episode_latch_enabled": self._episode_latch_enabled,
            "duplicate_suppressed_count": state.latch_state.duplicate_suppressed_count,
            "detector_reject_reason": state.latch_state.reject_reason,
        }

    def __enter__(self) -> "MultiROIFlashDetector":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()

"""Rolling adaptive signal normaliser and autocorrelation frequency estimator.

Pure Python + NumPy — zero Pi hardware dependencies.  Fully testable
on any platform.

The ``RollingAdaptiveSignalDetector`` replaces fixed absolute-brightness
thresholding with adaptive normalisation based on recent signal history.
It is designed for the ``local_contrast`` detection mode where the raw
signal value varies with lighting conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# RollingAdaptiveSignalDetector
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDetectorState:
    """Snapshot of the adaptive detector after processing one frame."""

    # Raw input
    signal_raw: float = 0.0

    # Adaptive normalisation
    adaptive_low: float = 0.0
    adaptive_high: float = 1.0
    adaptive_amplitude: float = 0.0
    signal_norm: float = 0.0
    signal_quality: float = 0.0   # 0–1, how reliable the signal looks

    # Hysteresis state
    state: str = "OFF"
    norm_on_threshold: float = 0.65
    norm_off_threshold: float = 0.35

    # Edge detection
    event_type: str | None = None
    rising_edge_count: int = 0
    last_rising_edge_time_s: float | None = None
    edge_times: list[float] = field(default_factory=list)

    # Autocorrelation frequency
    signal_frequency_hz: float = 0.0
    periodicity_confidence: float = 0.0


class RollingAdaptiveSignalDetector:
    """Adaptive flash detector using rolling normalisation.

    Instead of fixed absolute brightness thresholds, this detector
    maintains a rolling window of recent signal values and normalises
    each new sample against the local low/high percentiles.  This makes
    detection robust to ambient light changes and camera auto-exposure.

    Parameters
    ----------
    window_s:
        Rolling window duration in seconds.
    low_percentile:
        Percentile (0–100) used as the adaptive floor.
    high_percentile:
        Percentile (0–100) used as the adaptive ceiling.
    norm_on_threshold:
        Normalised signal level above which state transitions to ON.
    norm_off_threshold:
        Normalised signal level below which state transitions to OFF.
    min_interval_s:
        Minimum seconds between successive rising edges.
    min_amplitude:
        Minimum adaptive_amplitude required for detection.  If the
        signal swing is below this, edges are suppressed.
    """

    def __init__(
        self,
        window_s: float = 5.0,
        low_percentile: float = 10.0,
        high_percentile: float = 90.0,
        norm_on_threshold: float = 0.65,
        norm_off_threshold: float = 0.35,
        min_interval_s: float = 0.2,
        min_amplitude: float = 10.0,
    ) -> None:
        self.window_s = window_s
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.norm_on_threshold = norm_on_threshold
        self.norm_off_threshold = norm_off_threshold
        self.min_interval_s = min_interval_s
        self.min_amplitude = min_amplitude

        # Rolling history
        self._history: list[float] = []
        self._history_times: list[float] = []

        # State
        self._state = AdaptiveDetectorState()

    # ---- Public API ----

    @property
    def state(self) -> AdaptiveDetectorState:
        return self._state

    @property
    def signal_history(self) -> list[float]:
        return list(self._history)

    def update(self, signal_raw: float, now_s: float) -> AdaptiveDetectorState:
        """Process one signal sample and return the updated state.

        Parameters
        ----------
        signal_raw:
            Raw signal value for this frame (e.g. local_contrast).
        now_s:
            Current ``time.perf_counter()`` value.

        Returns
        -------
        AdaptiveDetectorState
        """
        # --- Maintain rolling window ---
        self._history.append(signal_raw)
        self._history_times.append(now_s)

        cutoff = now_s - self.window_s
        while self._history_times and self._history_times[0] < cutoff:
            self._history_times.pop(0)
            self._history.pop(0)

        # Need a minimum number of samples for reliable percentile
        MIN_SAMPLES = 10
        if len(self._history) < MIN_SAMPLES:
            # Not enough history yet — return idle state
            self._state = AdaptiveDetectorState(
                signal_raw=signal_raw,
                signal_norm=0.0,
                signal_quality=0.0,
                state="OFF",
                norm_on_threshold=self.norm_on_threshold,
                norm_off_threshold=self.norm_off_threshold,
            )
            return self._state

        # --- Adaptive normalisation ---
        arr = np.array(self._history)
        adaptive_low = float(np.percentile(arr, self.low_percentile))
        adaptive_high = float(np.percentile(arr, self.high_percentile))
        adaptive_amplitude = adaptive_high - adaptive_low

        if adaptive_amplitude < self.min_amplitude:
            # Signal too flat — suppress detection
            self._state = AdaptiveDetectorState(
                signal_raw=signal_raw,
                adaptive_low=adaptive_low,
                adaptive_high=adaptive_high,
                adaptive_amplitude=adaptive_amplitude,
                signal_norm=0.0,
                signal_quality=0.0,
                state="OFF",
                norm_on_threshold=self.norm_on_threshold,
                norm_off_threshold=self.norm_off_threshold,
                rising_edge_count=self._state.rising_edge_count,
                last_rising_edge_time_s=self._state.last_rising_edge_time_s,
                edge_times=list(self._state.edge_times),
            )
            return self._state

        # Normalise to [0, 1]
        signal_norm = (signal_raw - adaptive_low) / adaptive_amplitude
        signal_norm = max(0.0, min(1.0, signal_norm))

        # Signal quality: 0–1 based on amplitude relative to raw range
        raw_range = float(np.max(arr) - np.min(arr))
        signal_quality = min(1.0, adaptive_amplitude / max(raw_range, 1.0))

        # --- Adaptive hysteresis ---
        prev_state = self._state.state
        new_state_str = prev_state
        event_type = None

        if prev_state == "OFF" and signal_norm >= self.norm_on_threshold:
            new_state_str = "ON"
            if (
                self._state.last_rising_edge_time_s is None
                or (now_s - self._state.last_rising_edge_time_s) >= self.min_interval_s
            ):
                event_type = "leader_rising_edge"
                self._state.rising_edge_count += 1
                self._state.last_rising_edge_time_s = now_s
                self._state.edge_times.append(now_s)

        elif prev_state == "ON" and signal_norm <= self.norm_off_threshold:
            new_state_str = "OFF"

        # --- Autocorrelation frequency ---
        sig_freq, period_conf = estimate_frequency_autocorrelation(
            self._history_times, self._history,
            min_freq_hz=0.2, max_freq_hz=5.0,
        )

        # --- Build state ---
        self._state = AdaptiveDetectorState(
            signal_raw=signal_raw,
            adaptive_low=adaptive_low,
            adaptive_high=adaptive_high,
            adaptive_amplitude=adaptive_amplitude,
            signal_norm=round(signal_norm, 4),
            signal_quality=round(signal_quality, 4),
            state=new_state_str,
            norm_on_threshold=self.norm_on_threshold,
            norm_off_threshold=self.norm_off_threshold,
            event_type=event_type,
            rising_edge_count=self._state.rising_edge_count,
            last_rising_edge_time_s=self._state.last_rising_edge_time_s,
            edge_times=list(self._state.edge_times),
            signal_frequency_hz=round(sig_freq, 3),
            periodicity_confidence=round(period_conf, 4),
        )

        return self._state

    def reset(self) -> None:
        """Clear rolling history and state."""
        self._history.clear()
        self._history_times.clear()
        self._state = AdaptiveDetectorState()


# ---------------------------------------------------------------------------
# Autocorrelation-based frequency estimation
# ---------------------------------------------------------------------------

def estimate_frequency_autocorrelation(
    times: list[float],
    values: list[float],
    min_freq_hz: float = 0.2,
    max_freq_hz: float = 5.0,
    sample_rate_hz: float | None = None,
) -> tuple[float, float]:
    """Estimate the dominant frequency of *values* via autocorrelation.

    This works directly on the raw signal (e.g. ``local_contrast`` or
    ``signal_norm``) and does **not** depend on rising-edge detection.
    Even if ON/OFF state is imperfect, a periodic signal still produces
    a meaningful autocorrelation peak.

    Parameters
    ----------
    times:
        Timestamps in seconds (monotonically increasing).
    values:
        Signal values corresponding to *times*.
    min_freq_hz:
        Minimum plausible frequency.  Lags corresponding to frequencies
        below this are excluded.
    max_freq_hz:
        Maximum plausible frequency.
    sample_rate_hz:
        If *None*, estimated from median spacing of *times*.

    Returns
    -------
    (frequency_hz, periodicity_confidence)
        Frequency is 0.0 if no reliable peak is found.  Confidence is
        0–1 where higher means stronger periodicity.
    """
    if len(values) < 20:
        return 0.0, 0.0

    arr = np.array(values, dtype=np.float64)

    # Remove mean
    arr = arr - np.mean(arr)

    # Estimate sample rate
    if sample_rate_hz is None and len(times) >= 2:
        diffs = np.diff(np.array(times))
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            return 0.0, 0.0
        sample_rate_hz = 1.0 / median_diff
    elif sample_rate_hz is None:
        return 0.0, 0.0

    # Autocorrelation via numpy.correlate (unbiased normalisation)
    n = len(arr)
    autocorr = np.correlate(arr, arr, mode="full")
    autocorr = autocorr[n - 1:]  # keep only non-negative lags
    if autocorr[0] == 0:
        return 0.0, 0.0
    autocorr = autocorr / autocorr[0]  # normalise so r[0] = 1

    # Lag range corresponding to [min_freq, max_freq]
    min_lag = max(1, int(sample_rate_hz / max_freq_hz))
    max_lag = min(n - 1, int(sample_rate_hz / min_freq_hz))

    if max_lag <= min_lag:
        return 0.0, 0.0

    # Search for highest peak in the allowed lag range
    search = autocorr[min_lag:max_lag + 1]
    peak_idx = int(np.argmax(search))
    peak_val = search[peak_idx]
    peak_lag = min_lag + peak_idx

    # Confidence heuristic: how much higher is the peak vs surrounding?
    # Simple metric: peak height relative to 1.0, penalised by nearby trough
    if peak_lag + 5 < len(autocorr):
        local_min = float(np.min(autocorr[peak_lag:peak_lag + 5]))
    else:
        local_min = 0.0
    periodicity_confidence = max(0.0, min(1.0, peak_val - max(local_min, 0.0)))

    if peak_val < 0.15:
        return 0.0, 0.0

    frequency_hz = sample_rate_hz / peak_lag

    # Sanity check
    if frequency_hz < min_freq_hz or frequency_hz > max_freq_hz:
        return 0.0, 0.0

    return frequency_hz, periodicity_confidence

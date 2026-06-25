"""Tests for RollingAdaptiveSignalDetector and autocorrelation estimator.

Pure Python + numpy — no Pi hardware required.
"""

import math

import numpy as np
import pytest

from firefly_sync.hardware.signal_detector import (
    AdaptiveDetectorState,
    RollingAdaptiveSignalDetector,
    estimate_frequency_autocorrelation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_sine(
    duration_s: float, sample_rate_hz: float, freq_hz: float,
    amplitude: float = 0.5, offset: float = 0.5, noise: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Generate a synthetic sinusoidal signal."""
    n = int(duration_s * sample_rate_hz)
    times = [i / sample_rate_hz for i in range(n)]
    rng = np.random.default_rng(42)
    values = [
        offset + amplitude * math.sin(2 * math.pi * freq_hz * t)
        + rng.normal(0, noise)
        for t in times
    ]
    return times, values


# ---------------------------------------------------------------------------
# RollingAdaptiveSignalDetector
# ---------------------------------------------------------------------------

class TestAdaptiveInitialisation:
    def test_default_construction(self) -> None:
        det = RollingAdaptiveSignalDetector()
        assert det.window_s == 5.0
        assert det.norm_on_threshold == 0.65
        assert det.norm_off_threshold == 0.35

    def test_custom_params(self) -> None:
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, low_percentile=5, high_percentile=95,
            norm_on_threshold=0.7, norm_off_threshold=0.3,
            min_interval_s=0.5, min_amplitude=20.0,
        )
        assert det.window_s == 3.0
        assert det.low_percentile == 5
        assert det.high_percentile == 95

    def test_initial_state_is_off(self) -> None:
        det = RollingAdaptiveSignalDetector()
        assert det.state.state == "OFF"
        assert det.state.rising_edge_count == 0


class TestAdaptiveNormalisation:
    def test_periodic_signal_normalises(self) -> None:
        """A clear sine wave should normalise to ~0–1 range."""
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, min_amplitude=0.05,
        )
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=50, offset=100)
        snorms = []
        for t, v in zip(times, values):
            state = det.update(v, t)
            snorms.append(state.signal_norm)

        # After warmup, normed values should swing between 0 and 1
        warm = 60  # skip first 2 seconds
        assert max(snorms[warm:]) > 0.8, f"max norm={max(snorms[warm:])}"
        assert min(snorms[warm:]) < 0.2, f"min norm={min(snorms[warm:])}"


class TestAdaptiveHysteresis:
    def test_state_switches_on_off(self) -> None:
        """With a strong signal, state should toggle ON/OFF."""
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, norm_on_threshold=0.65, norm_off_threshold=0.35,
            min_interval_s=0.05, min_amplitude=0.05,
        )
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=50, offset=100)
        states = []
        for t, v in zip(times, values):
            state = det.update(v, t)
            states.append(state.state)

        warm = 90  # allow adaptive normalisation to settle
        assert "ON" in states[warm:], "Should detect ON state"
        assert "OFF" in states[warm:], "Should detect OFF state"

    def test_rising_edges_detected(self) -> None:
        """A 1 Hz sine should produce ~5 rising edges over 5 seconds."""
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, norm_on_threshold=0.65, norm_off_threshold=0.35,
            min_interval_s=0.1, min_amplitude=0.05,
        )
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=50, offset=100)
        for t, v in zip(times, values):
            det.update(v, t)

        # Should have roughly 4-6 edges for 5s of 1 Hz
        assert 3 <= det.state.rising_edge_count <= 8, \
            f"got {det.state.rising_edge_count} edges for 1 Hz signal"

    def test_flat_signal_no_edges(self) -> None:
        """A signal with no amplitude should not produce edges."""
        det = RollingAdaptiveSignalDetector(min_amplitude=10.0)
        for i in range(200):
            det.update(128.0, float(i) * 0.033)
        assert det.state.rising_edge_count == 0
        assert det.state.signal_quality == 0.0


class TestMinAmplitude:
    def test_min_amplitude_suppresses_weak_signal(self) -> None:
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, min_amplitude=50.0,
        )
        # Signal with only 5-unit swing — below min_amplitude
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=5, offset=100)
        for t, v in zip(times, values):
            det.update(v, t)
        assert det.state.rising_edge_count == 0

    def test_strong_signal_passes_min_amplitude(self) -> None:
        det = RollingAdaptiveSignalDetector(
            window_s=3.0, min_amplitude=10.0,
        )
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=50, offset=100)
        for t, v in zip(times, values):
            det.update(v, t)
        assert det.state.rising_edge_count > 0


class TestReset:
    def test_reset_clears_history(self) -> None:
        det = RollingAdaptiveSignalDetector()
        for i in range(100):
            det.update(float(i % 50), float(i) * 0.033)
        assert len(det.signal_history) > 0
        det.reset()
        assert len(det.signal_history) == 0
        assert det.state.rising_edge_count == 0


# ---------------------------------------------------------------------------
# Autocorrelation frequency estimation
# ---------------------------------------------------------------------------

class TestAutocorrelation:
    def test_1hz_sine_detected(self) -> None:
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=0.45, offset=0.5)
        freq, conf = estimate_frequency_autocorrelation(times, values)
        assert freq == pytest.approx(1.0, abs=0.15), f"got {freq} Hz"
        assert conf > 0.1, f"confidence too low: {conf}"

    def test_2hz_sine_detected(self) -> None:
        times, values = _synthetic_sine(5.0, 30.0, 2.0, amplitude=0.45, offset=0.5)
        freq, conf = estimate_frequency_autocorrelation(times, values)
        assert freq == pytest.approx(2.0, abs=0.3), f"got {freq} Hz"

    def test_flat_signal_low_confidence(self) -> None:
        times = [i / 30.0 for i in range(150)]
        values = [0.5] * 150
        freq, conf = estimate_frequency_autocorrelation(times, values)
        assert freq == 0.0

    def test_insufficient_samples(self) -> None:
        times = [0.0, 0.1, 0.2]
        values = [0.1, 0.5, 0.1]
        freq, conf = estimate_frequency_autocorrelation(times, values)
        assert freq == 0.0

    def test_noisy_signal_still_detected(self) -> None:
        times, values = _synthetic_sine(5.0, 30.0, 1.0, amplitude=0.45, offset=0.5, noise=0.1)
        freq, conf = estimate_frequency_autocorrelation(times, values)
        assert freq == pytest.approx(1.0, abs=0.3), f"got {freq} Hz"

"""Tests for the Kuramoto oscillator model."""

import numpy as np
import pytest

from firefly_sync.core.kuramoto import KuramotoModel


class TestKuramotoModel:
    """Unit tests for the KuramotoModel class."""

    def test_initialisation_defaults(self) -> None:
        """Oscillator should initialise with correct default values."""
        osc = KuramotoModel(natural_frequency=1.0)
        assert osc.natural_frequency == 1.0
        assert osc.phase == 0.0
        assert osc.coupling_strength == 1.0
        assert osc.fire_count == 0

    def test_initialisation_custom_phase(self) -> None:
        """Oscillator should accept a custom initial phase."""
        osc = KuramotoModel(natural_frequency=1.0, initial_phase=np.pi)
        assert osc.phase == pytest.approx(np.pi)

    def test_negative_frequency_raises(self) -> None:
        """Negative natural frequency should raise ValueError."""
        with pytest.raises(ValueError):
            KuramotoModel(natural_frequency=-1.0)

    def test_negative_coupling_raises(self) -> None:
        """Negative coupling strength should raise ValueError."""
        with pytest.raises(ValueError):
            KuramotoModel(natural_frequency=1.0, coupling_strength=-1.0)

    def test_phase_advances_without_coupling(self) -> None:
        """Phase should advance by ω·dt each step with no coupling."""
        osc = KuramotoModel(natural_frequency=1.0, dt=0.01)
        state = osc.step(coupling_input=0.0)
        assert state.phase == pytest.approx(0.01, rel=0.01)

    def test_phase_wraps_at_threshold(self) -> None:
        """Phase should wrap to 0 after crossing the flash threshold."""
        osc = KuramotoModel(
            natural_frequency=2.0 * np.pi,
            initial_phase=2.0 * np.pi - 0.01,
            dt=0.01,
        )
        state = osc.step(coupling_input=0.0)
        # Phase should have wrapped; oscillators with ω=2π advance by 2π·dt ≈ 0.063
        # Starting at 2π − 0.01, adding ~0.063 crosses 2π → wraps
        assert state.phase < 1.0  # wrapped to near 0

    def test_flash_detected_on_wrap(self) -> None:
        """Oscillator should indicate firing when phase wraps."""
        osc = KuramotoModel(
            natural_frequency=2.0 * np.pi,
            initial_phase=2.0 * np.pi - 0.01,
            dt=0.02,
        )
        state = osc.step(coupling_input=0.0)
        assert state.is_firing
        assert osc.fire_count == 1

    def test_coupling_increases_phase_velocity(self) -> None:
        """Positive coupling input should increase phase advance rate."""
        osc = KuramotoModel(natural_frequency=1.0, dt=0.01)
        state_no_coupling = osc.step(coupling_input=0.0)
        osc.reset(phase=0.0)
        state_with_coupling = osc.step(coupling_input=0.5)
        # With sin(θ_j - θ_i) positive, coupling adds to dθ/dt
        assert state_with_coupling.phase > state_no_coupling.phase

    def test_period_calculation(self) -> None:
        """Period T = 2π/ω should be correct."""
        osc = KuramotoModel(natural_frequency=2.0 * np.pi)
        assert osc.period == pytest.approx(1.0)

    def test_full_cycle_produces_one_fire(self) -> None:
        """One full natural period should produce exactly one fire event."""
        osc = KuramotoModel(
            natural_frequency=2.0 * np.pi,  # T = 1s
            dt=0.001,
            coupling_strength=0.0,
        )
        for _ in range(1002):  # slightly over 1s to handle fp rounding
            osc.step(coupling_input=0.0)
        assert osc.fire_count == 1

    def test_reset_clears_state(self) -> None:
        """Reset should restore initial conditions."""
        osc = KuramotoModel(natural_frequency=1.0, initial_phase=1.5)
        osc.step(coupling_input=0.1)
        osc.reset(phase=0.0)
        assert osc.phase == 0.0
        assert osc.fire_count == 0

    def test_two_identical_oscillators_sync(self) -> None:
        """Two Kuramoto oscillators with coupling should eventually sync.

        This is the fundamental mathematical property of the Kuramoto model:
        identical oscillators with K > 0 approach phase synchrony.
        """
        osc_a = KuramotoModel(
            natural_frequency=1.0, initial_phase=0.0,
            coupling_strength=1.5, dt=0.01,
        )
        osc_b = KuramotoModel(
            natural_frequency=1.0, initial_phase=2.0,  # ~114° offset
            coupling_strength=1.5, dt=0.01,
        )

        for _ in range(3000):  # 30 seconds
            # Kuramoto coupling: (1/2)·[sin(θ_b - θ_a) + sin(θ_a - θ_b)]
            # sin(θ_b - θ_a) + sin(θ_a - θ_b) = 0 for N=2, identical ω
            # So each oscillator only sees the other's contribution
            coupling_a = np.sin(osc_b.phase - osc_a.phase) / 2.0
            coupling_b = np.sin(osc_a.phase - osc_b.phase) / 2.0
            osc_a.step(coupling_a)
            osc_b.step(coupling_b)

        # After 30s with K=1.5, phases should be close
        phase_diff = abs(osc_a.phase - osc_b.phase)
        # Normalise phase difference to [0, π]
        if phase_diff > np.pi:
            phase_diff = 2.0 * np.pi - phase_diff
        assert phase_diff < 0.5, f"Phase diff {phase_diff:.3f} > 0.5 rad"

"""Tests for the Pulse-Coupled / Integrate-and-Fire oscillator model."""

import numpy as np
import pytest

from firefly_sync.core.pulse_coupled import PulseCoupledModel


class TestPulseCoupledModel:
    """Unit tests for the PulseCoupledModel class."""

    def test_initialisation_defaults(self) -> None:
        """Oscillator should initialise with correct defaults."""
        osc = PulseCoupledModel(natural_period=1.0)
        assert osc.natural_period == 1.0
        assert osc.state_variable == 0.0
        assert osc.coupling_increment == 0.1
        assert osc.threshold == 1.0
        assert osc.fire_count == 0

    def test_negative_period_raises(self) -> None:
        """Negative natural period should raise ValueError."""
        with pytest.raises(ValueError):
            PulseCoupledModel(natural_period=-0.5)

    def test_invalid_coupling_increment_raises(self) -> None:
        """Coupling increment ε must be in (0, 1)."""
        with pytest.raises(ValueError):
            PulseCoupledModel(natural_period=1.0, coupling_increment=0.0)
        with pytest.raises(ValueError):
            PulseCoupledModel(natural_period=1.0, coupling_increment=1.0)

    def test_state_variable_charges_without_coupling(self) -> None:
        """State variable x should increase at rate 1/T each step."""
        osc = PulseCoupledModel(natural_period=1.0, dt=0.01)
        osc.step(coupling_input=0.0)
        assert osc.state_variable == pytest.approx(0.01, rel=0.01)

    def test_fires_at_threshold(self) -> None:
        """Oscillator should fire when x reaches 1.0."""
        osc = PulseCoupledModel(
            natural_period=0.1,  # T = 0.1s, charges fast
            dt=0.001,
            coupling_strength=0.0,
        )
        fired = False
        for _ in range(200):  # 0.2s — more than enough
            state = osc.step(coupling_input=0.0)
            if state.is_firing:
                fired = True
                break
        assert fired, "Oscillator did not fire within 2 periods."

    def test_resets_after_fire(self) -> None:
        """State variable should reset to 0 after firing."""
        osc = PulseCoupledModel(
            natural_period=0.1,
            initial_state=0.99,
            dt=0.01,
            coupling_strength=0.0,
            refractory_steps=0,
        )
        state = osc.step(coupling_input=0.0)
        if state.is_firing:
            assert osc.state_variable == 0.0

    def test_refractory_period(self) -> None:
        """Oscillator should hold at 0 during refractory steps."""
        osc = PulseCoupledModel(
            natural_period=0.1,
            initial_state=0.99,
            dt=0.01,
            refractory_steps=5,
            coupling_strength=0.0,
        )
        # Step until fire
        state = None
        for _ in range(20):
            state = osc.step(coupling_input=0.0)
            if state.is_firing:
                break

        assert state is not None and state.is_firing
        assert osc.in_refractory

        # During refractory, state_variable should stay at 0
        for _ in range(4):
            state = osc.step(coupling_input=0.0)
            assert osc.state_variable == 0.0
            assert not state.is_firing

    def test_pulse_coupling_advances_state(self) -> None:
        """A detected neighbour flash should advance the state variable."""
        osc = PulseCoupledModel(
            natural_period=1.0,
            initial_state=0.0,
            coupling_strength=1.0,
            coupling_increment=0.2,
            dt=0.01,
        )
        state_no_input = osc.step(coupling_input=0.0)
        x_alone = osc.state_variable

        osc.reset()
        state_with_input = osc.step(coupling_input=1.0)  # one detected flash
        # Should have advanced by dt/T + coupling_increment ≈ 0.01 + 0.2
        assert osc.state_variable > x_alone + 0.15

    def test_fire_count_increments(self) -> None:
        """Fire counter should increment on each firing event."""
        osc = PulseCoupledModel(
            natural_period=0.1,
            dt=0.02,
            coupling_strength=0.0,
            refractory_steps=0,
        )
        # Run for ~2 seconds = 20 periods
        for _ in range(100):
            osc.step(coupling_input=0.0)
        assert osc.fire_count >= 15, f"Expected ≥15 fires, got {osc.fire_count}"

    def test_reset_clears_state(self) -> None:
        """Reset should restore initial conditions."""
        osc = PulseCoupledModel(natural_period=1.0, initial_state=0.5)
        osc.step(coupling_input=1.0)
        osc.reset()
        assert osc.state_variable == 0.0
        assert osc.fire_count == 0
        assert not osc.in_refractory

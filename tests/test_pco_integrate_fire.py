"""Unit tests for PCO-I&F core oscillator (pure logic, no hardware)."""

import pytest

from firefly_sync.core.pco_integrate_fire import (
    PulseCoupledIFConfig,
    PulseCoupledIntegrateFireOscillator,
)


class TestNaturalEvolution:
    def test_phase_accumulates_and_fires(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=2.0, epsilon=0.1)
        )
        # At 2 Hz, 0.5 s → phase = 1.0 exactly
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result["follower_flash_event"] is True
        assert osc.phase == 0.0

    def test_phase_resets_after_fire(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=2.0, epsilon=0.1)
        )
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result["follower_flash_event"] is True
        assert osc.phase == 0.0

    def test_no_flash_below_threshold(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.1)
        )
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        # 1 Hz * 0.5 s = 0.5 < 1.0
        assert result["follower_flash_event"] is False
        assert osc.phase == pytest.approx(0.5)

    def test_fire_count_increments(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=2.0)
        )
        # Two full cycles
        osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert osc.fire_count == 1
        osc.step(dt_s=0.5, leader_flash_event=False, t_s=1.0)
        assert osc.fire_count == 2


class TestPulseCoupling:
    def test_leader_pulse_advances_phase(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.3)
        )
        # Natural: 0.5 s → phase = 0.5
        # Pulse: phase += 0.3 * (1 - 0.5) = 0.15 → total = 0.65
        result = osc.step(dt_s=0.5, leader_flash_event=True, t_s=0.5)
        assert result["leader_flash_event_used"] is True
        assert osc.phase == pytest.approx(0.65)

    def test_leader_pulse_crosses_threshold(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.6)
        )
        # Natural: 0.5 s → phase = 0.5
        # Pulse: phase += 0.6 * (1 - 0.5) = 0.3 → total = 0.8
        # Wait, 0.8 < 1.0. Let's use a bigger base.
        osc.reset()
        # Natural: 0.8 s → phase = 0.8 (1 Hz * 0.8s)
        # Pulse: phase += 0.6 * (1 - 0.8) = 0.12 → total = 0.92
        # Still < 1.0. Need epsilon bigger or dt bigger.
        osc.reset()
        # Natural: 0.9 s → phase = 0.9
        # Pulse: phase += 0.6 * (1 - 0.9) = 0.06 → total = 0.96 < 1.0
        # Hmm. With epsilon=0.6 and dt=0.9, total=0.9+0.06=0.96. Need epsilon=1.0
        osc2 = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.5)
        )
        # Natural: 0.9 s → phase = 0.9. Pulse: += 0.5*(1-0.9) = 0.05 → 0.95 < 1.0
        # Let me use a bigger dt: 0.8 s → phase=0.8. Pulse += 0.5*0.2 = 0.1 → 0.9
        # Still < 1.0. Let me try epsilon=1.0
        osc3 = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=1.0)
        )
        # Natural: 0.9 s → phase=0.9. Pulse: += 1.0*0.1 = 0.1 → 1.0 ≥ 1.0!
        result = osc3.step(dt_s=0.9, leader_flash_event=True, t_s=0.9)
        assert result["follower_flash_event"] is True
        assert result["leader_flash_event_used"] is True
        assert osc3.phase == 0.0

    def test_coupling_not_applied_when_no_leader(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.5)
        )
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result["leader_flash_event_used"] is False
        # Natural evolution happened: phase went from 0 to 0.5
        assert result["phase_before_coupling"] == 0.0
        assert result["phase_after_coupling"] == pytest.approx(0.5)


class TestRefractoryPeriod:
    def test_refractory_suppresses_pulse(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(
                natural_frequency_hz=2.0, epsilon=0.5,
                refractory_period_s=0.1,
            )
        )
        # Fire naturally at t=0.5
        result1 = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result1["follower_flash_event"] is True
        assert result1["refractory_active"] is True  # just entered refractory

        # Immediately after fire, refractory should be active
        # Small dt → still refractory
        result2 = osc.step(dt_s=0.02, leader_flash_event=True, t_s=0.52)
        assert result2["leader_flash_event_used"] is False
        assert result2["refractory_active"] is True

    def test_refractory_expires(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(
                natural_frequency_hz=2.0, epsilon=0.5,
                refractory_period_s=0.05,
            )
        )
        osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)  # fire + refractory
        # After 0.06 s > refractory_period_s=0.05 → refractory expired
        result = osc.step(dt_s=0.06, leader_flash_event=True, t_s=0.56)
        assert result["leader_flash_event_used"] is True
        assert result["refractory_active"] is False


class TestReturnedDict:
    def test_contains_expected_keys(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator()
        result = osc.step(dt_s=0.1, leader_flash_event=False, t_s=0.1)
        for key in ["phase", "follower_flash_event", "leader_flash_event_used",
                     "refractory_active", "phase_before_coupling",
                     "phase_after_coupling", "fire_count"]:
            assert key in result, f"missing key: {key}"


# ======================================================================
# Coupling modes
# ======================================================================

class TestCouplingModes:
    def test_proportional_gap_preserves_old_behaviour(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.3,
                                  coupling_mode="proportional_gap")
        )
        result = osc.step(dt_s=0.5, leader_flash_event=True, t_s=0.5)
        # Natural: 0.5, coupling: +0.3*(1-0.5) = 0.15 → 0.65
        assert osc.phase == pytest.approx(0.65)

    def test_additive_phase(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.25,
                                  coupling_mode="additive_phase")
        )
        # Natural: 0.5, additive: +0.25 → 0.75
        result = osc.step(dt_s=0.5, leader_flash_event=True, t_s=0.5)
        assert osc.phase == pytest.approx(0.75)

    def test_additive_phase_crosses_threshold(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.60,
                                  coupling_mode="additive_phase")
        )
        # Natural: 0.9 (at 0.9s), additive: +0.60 → 1.50 capped to 1.0 → fire!
        result = osc.step(dt_s=0.9, leader_flash_event=True, t_s=0.9)
        assert result["follower_flash_event"] is True

    def test_mirollo_state(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.3,
                                  coupling_mode="mirollo_state",
                                  state_curve_beta=3.0)
        )
        # Natural: phase=0.3 at dt=0.3
        # U(0.3) ≈ (1-exp(-0.9))/(1-exp(-3)) = 0.593/0.950 = 0.624
        # x = min(1, 0.624+0.3) = 0.924
        # U_inv(0.924) = -ln(1-0.924*(1-exp(-3)))/3 = -ln(1-0.924*0.95)/3
        # = -ln(0.122)/3 = 2.10/3 = 0.702
        result = osc.step(dt_s=0.3, leader_flash_event=True, t_s=0.3)
        assert 0.6 < osc.phase < 0.8

    def test_mirollo_state_crosses_threshold(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(natural_frequency_hz=1.0, epsilon=0.6,
                                  coupling_mode="mirollo_state",
                                  state_curve_beta=3.0)
        )
        # Natural to 0.6: U(0.6)=0.834. x=1.0 (capped). U_inv(1)=1.0 → fire!
        result = osc.step(dt_s=0.6, leader_flash_event=True, t_s=0.6)
        assert result["follower_flash_event"] is True

    def test_mirollo_U_U_inv_consistency(self) -> None:
        from firefly_sync.core.pco_integrate_fire import _U, _U_inv
        for phi in [0.0, 0.1, 0.5, 0.9, 1.0]:
            x = _U(phi, 3.0)
            phi2 = _U_inv(x, 3.0)
            assert phi2 == pytest.approx(phi, rel=1e-6)

    def test_new_return_fields(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator(
            PulseCoupledIFConfig(coupling_mode="additive_phase")
        )
        result = osc.step(dt_s=0.1, leader_flash_event=True, t_s=0.1)
        assert "coupling_mode" in result
        assert "state_variable_before_pulse" in result
        assert "state_variable_after_pulse" in result
        assert result["coupling_mode"] == "additive_phase"

    def test_invalid_coupling_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            PulseCoupledIFConfig(coupling_mode="nonexistent")


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        osc = PulseCoupledIntegrateFireOscillator()
        osc.step(dt_s=0.5, leader_flash_event=True, t_s=0.5)
        osc.reset()
        assert osc.phase == 0.0
        assert osc.fire_count == 0
        assert osc.refractory_active is False
        assert osc.last_flash_time_s is None

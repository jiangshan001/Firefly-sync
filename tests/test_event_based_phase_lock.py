"""Unit tests for EAPF core oscillator (pure logic, no hardware)."""

import math

import pytest

from firefly_sync.core.event_based_phase_lock import (
    EventBasedPhaseLockConfig,
    EventBasedPhaseLockOscillator,
    clamp,
    wrap_to_2pi,
    wrap_to_pi,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestWrapToPi:
    def test_zero(self) -> None:
        assert wrap_to_pi(0.0) == 0.0

    def test_positive_within_range(self) -> None:
        assert wrap_to_pi(1.0) == pytest.approx(1.0)

    def test_negative_within_range(self) -> None:
        assert wrap_to_pi(-1.0) == pytest.approx(-1.0)

    def test_wrap_positive(self) -> None:
        # 3.5π → 3.5π - 4π = -0.5π ≈ -1.57
        assert wrap_to_pi(3.5 * math.pi) == pytest.approx(-0.5 * math.pi)

    def test_wrap_negative(self) -> None:
        # -3.5π → -3.5π + 4π = 0.5π ≈ 1.57
        assert wrap_to_pi(-3.5 * math.pi) == pytest.approx(0.5 * math.pi)

    def test_exact_pi(self) -> None:
        assert wrap_to_pi(math.pi) == pytest.approx(-math.pi)

    def test_exact_minus_pi(self) -> None:
        assert wrap_to_pi(-math.pi) == pytest.approx(-math.pi)


class TestWrapTo2pi:
    def test_zero(self) -> None:
        assert wrap_to_2pi(0.0) == 0.0

    def test_within_range(self) -> None:
        assert wrap_to_2pi(3.0) == pytest.approx(3.0)

    def test_wrap_exact(self) -> None:
        assert wrap_to_2pi(2.0 * math.pi) == pytest.approx(0.0)

    def test_wrap_above(self) -> None:
        # 3π → π
        assert wrap_to_2pi(3.0 * math.pi) == pytest.approx(math.pi)


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below(self) -> None:
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above(self) -> None:
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundary(self) -> None:
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


# ---------------------------------------------------------------------------
# Natural phase evolution
# ---------------------------------------------------------------------------

class TestNaturalEvolution:
    def test_phase_wrap_triggers_flash(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(natural_frequency_hz=2.0)
        )
        # omega = 2π*2 = 4π rad/s. 0.5 s → phase = 2π rad → wrap!
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result["follower_flash_event"] is True
        # phase should be 0 after wrapping
        assert osc.phase_rad == pytest.approx(0.0)

    def test_no_flash_before_wrap(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(natural_frequency_hz=1.0)
        )
        result = osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert result["follower_flash_event"] is False

    def test_fire_count_increments(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(natural_frequency_hz=2.0)
        )
        osc.step(dt_s=0.5, leader_flash_event=False, t_s=0.5)
        assert osc.fire_count == 1
        osc.step(dt_s=0.5, leader_flash_event=False, t_s=1.0)
        assert osc.fire_count == 2


# ---------------------------------------------------------------------------
# Correction sign verification
# ---------------------------------------------------------------------------

class TestCorrectionSign:
    """Verify that correction pushes the follower in the right direction."""

    def test_lagging_follower_catches_up(self) -> None:
        """Follower at phase=5.0 rad when leader fires.
        Desired phase at leader flash = 0.  Follower is nearly at the
        end of its cycle (phase≈5→2π−1.28).  It is lagging behind the
        leader's timing — it should speed up.
        phase_error = wrap_to_pi(0 − 5.0) = wrap_to_pi(−5.0) ≈ +1.283
        So phase_error > 0 when lagging.
        Phase correction: phase += gain * positive → moves forward.
        Frequency correction: freq += gain * positive/(2π) → increases.
        This is correct behaviour.
        """
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(
                natural_frequency_hz=1.0,
                phase_gain=0.2, frequency_gain=0.05,
            )
        )
        # Set follower phase to 5.0 rad (0.796 of cycle, nearly done)
        osc._phase_rad = 5.0
        old_freq = osc.frequency_hz

        result = osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.0)

        # phase_error should be positive (follower lagging)
        assert result["phase_error_rad"] > 0, (
            f"Expected positive phase_error for lagging follower, "
            f"got {result['phase_error_rad']}"
        )
        # Frequency should increase to catch up
        assert osc.frequency_hz > old_freq, (
            f"Frequency should increase for lagging follower"
        )

    def test_leading_follower_slows_down(self) -> None:
        """Follower at phase=1.0 rad when leader fires.
        Follower is early — should slow down.
        phase_error = wrap_to_pi(0 − 1.0) = −1.0 (negative).
        Frequency should decrease.
        """
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(
                natural_frequency_hz=1.5,
                phase_gain=0.2, frequency_gain=0.05,
            )
        )
        osc._phase_rad = 1.0
        old_freq = osc.frequency_hz

        result = osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.0)

        assert result["phase_error_rad"] < 0, (
            f"Expected negative phase_error for leading follower"
        )
        assert osc.frequency_hz < old_freq, (
            f"Frequency should decrease for leading follower"
        )


# ---------------------------------------------------------------------------
# Frequency clamping
# ---------------------------------------------------------------------------

class TestFrequencyClamping:
    def test_freq_clamped_to_min(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(
                natural_frequency_hz=0.6,
                frequency_min_hz=0.5, frequency_max_hz=4.0,
                frequency_gain=0.5,  # large gain to force clamp
            )
        )
        osc._phase_rad = 1.0  # leading → negative correction → freq decreases
        for _ in range(10):
            osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.0)
        assert osc.frequency_hz >= 0.5

    def test_freq_clamped_to_max(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(
                natural_frequency_hz=3.5,
                frequency_min_hz=0.5, frequency_max_hz=4.0,
                frequency_gain=0.5,
            )
        )
        osc._phase_rad = 5.0  # lagging → positive correction → freq increases
        for _ in range(10):
            osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.0)
        assert osc.frequency_hz <= 4.0


# ---------------------------------------------------------------------------
# Leader period estimation
# ---------------------------------------------------------------------------

class TestPeriodEstimation:
    def test_period_updated_after_two_flashes(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(natural_frequency_hz=1.0)
        )
        # Two leader flashes 0.5 s apart → period = 0.5 s → 2 Hz
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.0)
        osc.step(dt_s=0.49, leader_flash_event=False, t_s=0.49)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.5)
        assert osc.leader_period_estimate_s == pytest.approx(0.5)

    def test_median_robust_to_outlier(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(
                natural_frequency_hz=1.0, leader_period_window=4,
            )
        )
        # Regular flashes at 0.5 s intervals + one outlier
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.0)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.5)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.55)  # outlier — only 0.05s!
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.0)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=1.5)
        # Intervals: 0.5, 0.05, 0.45, 0.5 → median ≈ 0.475
        assert 0.4 <= osc.leader_period_estimate_s <= 0.55

    def test_bootstrap_before_two_flashes(self) -> None:
        osc = EventBasedPhaseLockOscillator(
            EventBasedPhaseLockConfig(natural_frequency_hz=1.5)
        )
        # Only 1 flash → period stays at bootstrap (1/1.5 ≈ 0.667)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.0)
        assert osc.leader_period_estimate_s == pytest.approx(1.0 / 1.5)

    def test_leader_flash_count(self) -> None:
        osc = EventBasedPhaseLockOscillator()
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.0)
        osc.step(dt_s=0.01, leader_flash_event=True, t_s=0.5)
        osc.step(dt_s=0.01, leader_flash_event=False, t_s=0.6)
        assert osc.leader_flash_count == 2


# ---------------------------------------------------------------------------
# Returned dict
# ---------------------------------------------------------------------------

class TestReturnedDict:
    def test_contains_expected_keys(self) -> None:
        osc = EventBasedPhaseLockOscillator()
        result = osc.step(dt_s=0.1, leader_flash_event=False, t_s=0.1)
        for key in ["phase_rad", "frequency_hz", "omega_rad_s",
                     "phase_error_rad", "leader_period_estimate_s",
                     "leader_flash_count", "follower_flash_event",
                     "leader_flash_event_used", "fire_count"]:
            assert key in result, f"missing key: {key}"


class TestReset:
    def test_reset_clears_all_state(self) -> None:
        osc = EventBasedPhaseLockOscillator()
        osc.step(dt_s=0.5, leader_flash_event=True, t_s=0.5)
        osc.reset()
        assert osc.phase_rad == 0.0
        assert osc.fire_count == 0
        assert osc.leader_flash_count == 0

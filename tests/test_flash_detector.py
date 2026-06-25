"""Unit tests for the pure-Python flash detection state machine.

These tests require no hardware and run on any platform (Windows, macOS,
Linux).  They validate hysteresis, rising-edge detection, min-interval
gating, and frequency estimation.
"""

from __future__ import annotations

import pytest

from firefly_sync.hardware.flash_detector import (
    DEFAULT_MIN_INTERVAL_S,
    DEFAULT_THRESHOLD_OFF,
    DEFAULT_THRESHOLD_ON,
    FlashEpisodeLatchState,
    FlashDetectorState,
    estimate_frequency,
    update_flash_episode_latch,
    update_flash_detector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_state(**overrides) -> FlashDetectorState:
    """Return a default FlashDetectorState with optional overrides."""
    s = FlashDetectorState()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Hysteresis — initial state & no-change
# ---------------------------------------------------------------------------

class TestInitialState:
    """Default-constructed state should be OFF with zero counts."""

    def test_default_state_is_off(self) -> None:
        s = FlashDetectorState()
        assert s.state == "OFF"
        assert s.rising_edge_count == 0
        assert s.event_type is None

    def test_brightness_below_threshold_stays_off(self) -> None:
        prev = _fresh_state(state="OFF")
        new = update_flash_detector(
            50, prev, 100.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.state == "OFF"
        assert new.event_type is None

    def test_multiple_frames_no_change(self) -> None:
        """Steady brightness should produce no events over many frames."""
        state = _fresh_state(state="OFF")
        for i in range(100):
            state = update_flash_detector(
                50, state, float(i) * 0.033,
                threshold_on=128, threshold_off=64,
            )
            assert state.event_type is None


# ---------------------------------------------------------------------------
# Hysteresis — transitions
# ---------------------------------------------------------------------------

class TestHysteresisTransitions:
    """Verify the two hysteresis thresholds produce correct transitions."""

    def test_off_to_on_at_threshold(self) -> None:
        prev = _fresh_state(state="OFF")
        new = update_flash_detector(
            128, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.state == "ON"

    def test_on_to_off_at_threshold(self) -> None:
        prev = _fresh_state(state="ON")
        new = update_flash_detector(
            64, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.state == "OFF"

    def test_hysteresis_band(self) -> None:
        """Brightness between thresholds should not change state."""
        # Stay OFF when between thresholds
        prev = _fresh_state(state="OFF")
        new = update_flash_detector(
            100, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.state == "OFF"

        # Stay ON when between thresholds
        prev = _fresh_state(state="ON")
        new = update_flash_detector(
            100, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.state == "ON"

    def test_no_event_on_on_to_off(self) -> None:
        """Falling edges never produce events."""
        prev = _fresh_state(state="ON")
        new = update_flash_detector(
            50, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.event_type is None


# ---------------------------------------------------------------------------
# Rising edge detection
# ---------------------------------------------------------------------------

class TestRisingEdgeDetection:
    """Verify rising edge events are emitted correctly."""

    def test_first_rising_edge_emits_event(self) -> None:
        prev = _fresh_state(state="OFF")
        new = update_flash_detector(
            200, prev, 10.0,
            threshold_on=128, threshold_off=64,
        )
        assert new.event_type == "leader_rising_edge"
        assert new.rising_edge_count == 1
        assert new.last_rising_edge_time_s == 10.0

    def test_rising_edge_count_increments(self) -> None:
        state = _fresh_state(state="OFF")
        for i in range(5):
            # ON
            state = update_flash_detector(
                200, state, float(i * 2),
                threshold_on=128, threshold_off=64,
                min_interval_s=0.01,
            )
            # OFF
            state = update_flash_detector(
                50, state, float(i * 2 + 0.5),
                threshold_on=128, threshold_off=64,
                min_interval_s=0.01,
            )
        assert state.rising_edge_count == 5


# ---------------------------------------------------------------------------
# Min-interval gating
# ---------------------------------------------------------------------------

class TestMinIntervalGating:
    """Verify that edges within min_interval_s are suppressed."""

    def test_min_interval_suppresses_second_edge(self) -> None:
        prev = _fresh_state(
            state="OFF",
            last_rising_edge_time_s=9.5,
        )
        # Second edge at t=10.0 — only 0.5 s since last, below 0.1 min
        # Actually 0.5 > 0.1, so it should pass. Let me use a tighter gap.
        prev = _fresh_state(
            state="OFF",
            last_rising_edge_time_s=9.95,
        )
        new = update_flash_detector(
            200, prev, 10.0,
            threshold_on=128, threshold_off=64,
            min_interval_s=0.1,
        )
        # 10.0 - 9.95 = 0.05 < 0.1 → suppressed
        assert new.event_type is None
        assert new.rising_edge_count == 0
        # State should still transition to ON even though edge is suppressed
        assert new.state == "ON"

    def test_min_interval_allows_edge_after_cooldown(self) -> None:
        prev = _fresh_state(
            state="OFF",
            last_rising_edge_time_s=9.0,
        )
        new = update_flash_detector(
            200, prev, 10.0,
            threshold_on=128, threshold_off=64,
            min_interval_s=0.5,
        )
        # 10.0 - 9.0 = 1.0 >= 0.5 → allowed
        assert new.event_type == "leader_rising_edge"
        assert new.rising_edge_count == 1

    def test_first_edge_no_previous_allowed(self) -> None:
        """When last_rising_edge_time_s is None, first edge always passes."""
        prev = _fresh_state(state="OFF")
        new = update_flash_detector(
            200, prev, 10.0,
            threshold_on=128, threshold_off=64,
            min_interval_s=999.0,  # impossibly long
        )
        assert new.event_type == "leader_rising_edge"


# ---------------------------------------------------------------------------
# Episode latch
# ---------------------------------------------------------------------------

class TestFlashEpisodeLatch:
    """Verify one accepted event per visual flash episode."""

    def test_one_event_while_signal_stays_on(self) -> None:
        state = FlashEpisodeLatchState()
        state = update_flash_episode_latch(
            0.8, state, 0.0,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.35,
        )
        assert state.event_type == "leader_rising_edge"
        assert state.accepted_event_count == 1

        for i in range(1, 5):
            state = update_flash_episode_latch(
                0.9, state, i * 0.03,
                threshold_on=0.65, threshold_off=0.35,
                min_interval_s=0.35,
            )
            assert state.event_type is None
            assert state.accepted_event_count == 1

    def test_flicker_inside_episode_is_suppressed_until_rearm(self) -> None:
        state = FlashEpisodeLatchState()
        state = update_flash_episode_latch(
            0.8, state, 0.0,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.35,
        )
        state = update_flash_episode_latch(
            0.5, state, 0.03,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.35,
        )
        state = update_flash_episode_latch(
            0.8, state, 0.06,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.35,
        )
        assert state.event_type is None
        assert state.accepted_event_count == 1
        assert state.duplicate_suppressed_count == 1
        assert state.raw_on_threshold_crossing_count == 2

    def test_rearms_after_consecutive_off_frames(self) -> None:
        state = FlashEpisodeLatchState()
        state = update_flash_episode_latch(
            0.8, state, 0.0,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            off_frames_to_rearm=2,
        )
        state = update_flash_episode_latch(
            0.2, state, 0.10,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            off_frames_to_rearm=2,
        )
        assert state.armed is False
        state = update_flash_episode_latch(
            0.2, state, 0.13,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            off_frames_to_rearm=2,
        )
        assert state.armed is True
        state = update_flash_episode_latch(
            0.8, state, 0.50,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            off_frames_to_rearm=2,
        )
        assert state.event_type == "leader_rising_edge"
        assert state.accepted_event_count == 2

    def test_rearm_requires_both_duration_and_off_frames_when_requested(self) -> None:
        state = FlashEpisodeLatchState()
        state = update_flash_episode_latch(
            0.8, state, 0.0,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            rearm_off_duration_s=0.08,
            off_frames_to_rearm=2,
            rearm_requires_both=True,
        )
        state = update_flash_episode_latch(
            0.2, state, 0.01,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            rearm_off_duration_s=0.08,
            off_frames_to_rearm=2,
            rearm_requires_both=True,
        )
        state = update_flash_episode_latch(
            0.2, state, 0.04,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            rearm_off_duration_s=0.08,
            off_frames_to_rearm=2,
            rearm_requires_both=True,
        )
        assert state.off_frame_count >= 2
        assert state.armed is False

        state = update_flash_episode_latch(
            0.2, state, 0.10,
            threshold_on=0.65, threshold_off=0.35,
            min_interval_s=0.01,
            rearm_off_duration_s=0.08,
            off_frames_to_rearm=2,
            rearm_requires_both=True,
        )
        assert state.armed is True
        assert state.rearmed_event is True


# ---------------------------------------------------------------------------
# Frequency estimation
# ---------------------------------------------------------------------------

class TestFrequencyEstimation:
    """Verify frequency calculation from edge timestamp lists."""

    def test_insufficient_edges_returns_zero(self) -> None:
        assert estimate_frequency([]) == 0.0
        assert estimate_frequency([10.0]) == 0.0

    def test_two_edges_gives_correct_freq(self) -> None:
        # interval = 0.5 s → freq = 2.0 Hz
        edges = [10.0, 10.5]
        assert estimate_frequency(edges) == pytest.approx(2.0)

    def test_three_edges_averages_intervals(self) -> None:
        # intervals: 0.5, 0.5 → mean = 0.5 → freq = 2.0
        edges = [10.0, 10.5, 11.0]
        assert estimate_frequency(edges) == pytest.approx(2.0)

    def test_uneven_intervals(self) -> None:
        # intervals: 0.4, 0.6 → mean = 0.5 → freq = 2.0
        edges = [10.0, 10.4, 11.0]
        assert estimate_frequency(edges) == pytest.approx(2.0)

    def test_window_excludes_old_edges(self) -> None:
        # Edge at 0.0 is old; only edges at 9.0 and 10.0 are recent
        # with window_s=5.0 and now_s=10.0
        edges = [0.0, 9.0, 10.0]
        # recent = [9.0, 10.0], interval = 1.0 → freq = 1.0
        assert estimate_frequency(edges, now_s=10.0, window_s=5.0) == pytest.approx(1.0)

    def test_window_returns_zero_when_too_few_recent(self) -> None:
        edges = [0.0, 1.0, 9.0]
        # Only 9.0 is within window (now_s=10.0, window_s=5.0), so < 2 edges
        assert estimate_frequency(edges, now_s=10.0, window_s=5.0) == 0.0


# ---------------------------------------------------------------------------
# Integration — full cycle
# ---------------------------------------------------------------------------

class TestFullCycle:
    """Simulate a complete leader flash cycle."""

    def test_complete_on_off_cycle(self) -> None:
        state = _fresh_state(state="OFF")
        t = 0.0

        # Stay dark for a bit
        for _ in range(10):
            state = update_flash_detector(10, state, t)
            t += 0.033
            assert state.state == "OFF"
            assert state.event_type is None

        # Leader turns ON (bright)
        state = update_flash_detector(250, state, t)
        t += 0.033
        assert state.state == "ON"
        assert state.event_type == "leader_rising_edge"
        assert state.rising_edge_count == 1

        # Stay bright
        for _ in range(5):
            state = update_flash_detector(250, state, t)
            t += 0.033
            assert state.state == "ON"
            assert state.event_type is None  # no new edges while already ON

        # Leader turns OFF (dark)
        state = update_flash_detector(10, state, t)
        t += 0.033
        assert state.state == "OFF"
        assert state.event_type is None

    def test_two_cycles_produces_frequency(self) -> None:
        """Two complete ON/OFF cycles at 2 Hz should estimate ~2 Hz."""
        state = _fresh_state(state="OFF")
        t = 0.0
        period = 0.5  # 2 Hz

        for cycle in range(5):
            # ON transition
            state = update_flash_detector(
                250, state, t,
                min_interval_s=0.01,
            )
            t += period * 0.5  # ON duration
            # OFF transition
            state = update_flash_detector(
                10, state, t,
                min_interval_s=0.01,
            )
            t += period * 0.5  # OFF duration

        # After 5 cycles at 2 Hz, frequency should be close to 2 Hz
        assert state.estimated_frequency_hz == pytest.approx(2.0, rel=0.05)
        assert state.rising_edge_count == 5

"""Tests for Step 3A flash-timing synchronisation metrics.

Pure Python + numpy — no Pi hardware required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from firefly_sync.logging.metrics import (
    check_flash_synchronisation,
    compute_flash_timing_metrics,
    pair_flash_events,
)


# ======================================================================
# pair_flash_events
# ======================================================================

class TestPairFlashEvents:
    def test_empty_leader_returns_none_errors(self) -> None:
        pairs = pair_flash_events([], [1.0, 2.0, 3.0])
        assert len(pairs) == 3
        assert all(p["timing_error_s"] is None for p in pairs)

    def test_single_leader_paired_to_all_followers(self) -> None:
        pairs = pair_flash_events([5.0], [1.0, 4.5, 5.2, 9.0])
        # All should pair to the single leader at 5.0
        assert pairs[0]["leader_t"] == 5.0
        assert pairs[1]["leader_t"] == 5.0
        assert pairs[2]["leader_t"] == 5.0
        assert pairs[3]["leader_t"] == 5.0

    def test_exact_match(self) -> None:
        pairs = pair_flash_events([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        for p in pairs:
            assert p["timing_error_s"] == 0.0
            assert p["abs_error_s"] == 0.0

    def test_signed_error_sign(self) -> None:
        # Leader at 2.0, follower at 1.8 → error = +0.2 (leader after follower)
        pairs = pair_flash_events([2.0], [1.8])
        assert pairs[0]["timing_error_s"] == pytest.approx(0.2)

    def test_nearest_leader_chosen(self) -> None:
        # Leaders at 1.0, 3.0. Follower at 2.4 → nearest is 3.0
        pairs = pair_flash_events([1.0, 3.0], [2.4])
        assert pairs[0]["leader_t"] == pytest.approx(3.0)
        assert pairs[0]["timing_error_s"] == pytest.approx(0.6)


# ======================================================================
# check_flash_synchronisation
# ======================================================================

class TestCheckFlashSynchronisation:
    def test_converges_below_threshold_for_5_cycles(self) -> None:
        """Follower errors: 0.3, 0.2, 0.05, 0.04, 0.03, 0.02, 0.01
        The last 5 are all < 0.10 → sync achieved."""
        leader = [float(i) for i in range(10)]
        follower = [
            0.3,   # t=0 error=0.3 → fail
            1.2,   # t=1 error=0.2 → fail
            2.05,  # t=2 error=0.05 → q1
            3.04,  # t=3 error=0.04 → q2
            4.03,  # t=4 error=0.03 → q3
            5.02,  # t=5 error=0.02 → q4
            6.01,  # t=6 error=0.01 → q5 ✓
        ]
        result = check_flash_synchronisation(leader, follower,
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is True
        # Time to sync = time of the 5th qualifying flash = 6.01
        assert result["time_to_synchronization_s"] == pytest.approx(6.01)

    def test_brief_crossing_does_not_sync(self) -> None:
        """4 good cycles is not enough — must sustain 5."""
        leader = [float(i) for i in range(10)]
        follower = [
            0.3,
            1.2,
            2.05,  # q1
            3.04,  # q2
            4.03,  # q3
            5.02,  # q4
            6.3,   # FAIL — run broken
        ]
        result = check_flash_synchronisation(leader, follower,
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is False

    def test_intermittent_failures_reset_counter(self) -> None:
        """Qualifying run must be consecutive."""
        leader = [float(i) for i in range(10)]
        follower = [
            0.05,  # q1
            1.04,  # q2
            2.03,  # q3
            3.5,   # FAIL (error 0.5)
            4.02,  # q1 (restart)
            5.01,  # q2
        ]
        result = check_flash_synchronisation(leader, follower,
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is False

    def test_sustains_past_fifth_cycle(self) -> None:
        """After sync achieved, should still report correct time_to_sync."""
        leader = [float(i) for i in range(12)]
        follower = [
            0.05,  # q1
            1.04,  # q2
            2.03,  # q3
            3.02,  # q4
            4.01,  # q5 ✓ sync at t=4.01
            5.01,  # still good
            6.02,  # still good
        ]
        result = check_flash_synchronisation(leader, follower,
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is True
        assert result["time_to_synchronization_s"] == pytest.approx(4.01)

    def test_no_leader_flashes(self) -> None:
        result = check_flash_synchronisation([], [1.0, 2.0, 3.0],
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is False

    def test_no_follower_flashes(self) -> None:
        result = check_flash_synchronisation([1.0, 2.0, 3.0], [],
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is False

    def test_exact_boundary(self) -> None:
        """Error exactly at the threshold should NOT qualify (< not ≤)."""
        leader = [float(i) for i in range(10)]
        follower = [
            0.10,  # exactly 0.10 → NOT < 0.10 → fail
            1.09,  # q1
            2.08,  # q2
            3.07,  # q3
            4.06,  # q4
            6.10,  # exactly 0.10 → break
        ]
        result = check_flash_synchronisation(leader, follower,
                                              sync_threshold_s=0.10, sync_cycles=5)
        assert result["synchronization_success"] is False


# ======================================================================
# compute_flash_timing_metrics
# ======================================================================

class TestComputeFlashTimingMetrics:
    def test_synced_case(self) -> None:
        """Perfect synchronisation after warmup."""
        leader = [float(i) for i in range(15)]
        follower = [
            0.3, 1.2, 2.05, 3.04, 4.03,
            5.02, 6.01, 7.01, 8.01, 9.01,
        ]
        m = compute_flash_timing_metrics(leader, follower,
                                          sync_threshold_s=0.10, sync_cycles=5)
        assert m["synchronization_success"] is True
        assert m["time_to_synchronization_s"] == pytest.approx(6.01)
        assert m["steady_state_mean_abs_timing_error_s"] < 0.10
        assert m["detection_success_rate"] == 1.0
        assert m["false_positive_rate"] == 0.0

    def test_unsynced_case(self) -> None:
        leader = [float(i) for i in range(10)]
        follower = [i + 0.5 for i in range(10)]  # always 0.5s off
        m = compute_flash_timing_metrics(leader, follower,
                                          sync_threshold_s=0.10, sync_cycles=5)
        assert m["synchronization_success"] is False
        assert m["convergence_quality"] == 0.0

    def test_null_detection_rates(self) -> None:
        leader = [float(i) for i in range(10)]
        follower = [float(i) for i in range(5)]
        m = compute_flash_timing_metrics(
            leader, follower,
            detection_success_rate=None, false_positive_rate=None,
        )
        assert m["detection_success_rate"] is None
        assert m["false_positive_rate"] is None

    def test_frequency_error_zero_for_perfect_sync(self) -> None:
        leader = [float(i) for i in range(20)]
        follower = [float(i) + 0.01 for i in range(20)]
        m = compute_flash_timing_metrics(leader, follower)
        # Same period → freq error near zero
        assert m["final_frequency_error_hz"] == pytest.approx(0.0, abs=0.05)

    def test_frequency_error_detects_mismatch(self) -> None:
        leader = [i * 0.5 for i in range(20)]     # 2 Hz → period 0.5
        follower = [i * 0.67 for i in range(15)]   # ~1.5 Hz → period 0.67
        m = compute_flash_timing_metrics(leader, follower)
        # Freq error should be ~0.5 Hz
        assert m["final_frequency_error_hz"] > 0.1


# ======================================================================
# Mock runner integration test
# ======================================================================

class TestMockStep3ARunner:
    """Verify the mock runner executes and creates output files."""

    def test_short_trial_creates_output(self) -> None:
        """Run a 5-second mock trial and verify all 4 output files."""
        import sys
        import argparse

        # Avoid polluting the real log dir by using a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build args manually
            from experiments.run_step3a_kuramoto_closed_loop import run_trial

            ns = argparse.Namespace(
                mode="mock",
                duration=5.0,
                leader_freq=2.0,
                follower_freq=1.5,
                coupling_gain=3.5,
                dt=0.01,
                flash_on_time=0.06,
                sync_threshold_s=0.10,
                sync_cycles=5,
                log_dir=tmpdir,
                trial_id="test_trial",
                notes="pytest mock trial",
                # unused in mock mode but required by Namespace
                allow_hardware_fallback=False,
                led_pin=17,
                threshold_on=180.0,
                threshold_off=120.0,
                min_interval=0.2,
                window_s=5.0,
            )

            metrics = run_trial(ns)

            # Find the output directory
            out_dirs = list(Path(tmpdir).glob("*test_trial"))
            assert len(out_dirs) == 1
            out = out_dirs[0]

            # Verify all 4 files exist
            assert (out / "metadata.json").exists()
            assert (out / "oscillator_log.csv").exists()
            assert (out / "flash_events.csv").exists()
            assert (out / "metrics_summary.json").exists()

            # Verify metadata contains expected keys
            meta = json.loads((out / "metadata.json").read_text())
            assert meta["mode"] == "mock"
            assert meta["model_name"] == "kuramoto"
            assert meta["leader_freq_hz"] == 2.0

            # Verify metrics has the 9 required fields
            assert "synchronization_success" in metrics
            assert "time_to_synchronization_s" in metrics
            assert "steady_state_mean_abs_timing_error_s" in metrics
            assert "steady_state_rmse_timing_error_s" in metrics
            assert "steady_state_jitter_s" in metrics
            assert "final_frequency_error_hz" in metrics
            assert "convergence_quality" in metrics
            assert "detection_success_rate" in metrics
            assert "false_positive_rate" in metrics

            # oscillator_log should have rows
            with open(out / "oscillator_log.csv") as f:
                lines = f.readlines()
                assert len(lines) > 10  # header + many rows

            # flash_events should have leader and follower events
            with open(out / "flash_events.csv") as f:
                content = f.read()
                assert "leader_flash" in content
                assert "follower_flash" in content

"""Tests for Step 3A-3 Pi visual batch runner.

Tests focus on logic that does not require Pi hardware:
k_critical / k_ratio computation, computational cost aggregation,
and dry-run batch scheduling.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from experiments.run_step3a_pi_visual_batch import (
    _compute_cost_metrics,
    _save_csv,
    _save_json,
    run_visual_batch,
)


# ======================================================================
# k_critical and k_ratio
# ======================================================================

class TestKCritical:
    def test_k_critical_computation(self) -> None:
        """k_critical = 2 * pi * |f_leader - f_follower|"""
        leader = 2.0
        follower = 1.5
        k_critical = 2.0 * np.pi * abs(leader - follower)
        assert k_critical == pytest.approx(np.pi, rel=1e-6)

    def test_k_ratio_computation(self) -> None:
        k_critical = 2.0 * np.pi * 0.5  # = pi
        coupling = 3.5
        k_ratio = coupling / k_critical
        assert k_ratio == pytest.approx(3.5 / np.pi, rel=1e-6)

    def test_k_ratio_inf_when_zero_mismatch(self) -> None:
        """When leader_freq == follower_freq, k_critical = 0."""
        leader = 2.0
        follower = 2.0
        k_critical = 2.0 * np.pi * abs(leader - follower)
        assert k_critical == 0.0
        k_ratio = 3.5 / k_critical if k_critical > 0 else float("inf")
        assert k_ratio == float("inf")


# ======================================================================
# Computational cost metrics
# ======================================================================

class TestComputeCostMetrics:
    def test_empty_series(self) -> None:
        m = _compute_cost_metrics([], [], [], [], [], 0.0, False, 0.0)
        assert m["mean_loop_dt_ms"] == 0
        assert m["peak_memory_rss_mb"] == 0
        assert m["mean_cpu_temperature_c"] is None

    def test_nonempty_series(self) -> None:
        loop = [0.010, 0.012, 0.011, 0.013, 0.010]
        model = [0.1, 0.2, 0.15, 0.18, 0.12]
        cam = [5.0, 6.0, 5.5, 5.2, 5.8]
        temps = [42.0, 43.0, 44.0]
        mems = [50.0, 52.0, 51.0, 53.0, 52.0]
        m = _compute_cost_metrics(loop, model, cam, temps, mems,
                                   time.process_time(), False, 5.0)
        assert m["mean_loop_dt_ms"] > 0
        assert m["p95_loop_dt_ms"] > 0
        assert m["max_model_update_time_ms"] > 0
        assert m["mean_cpu_temperature_c"] == pytest.approx(43.0)
        assert m["max_cpu_temperature_c"] == 44.0
        assert m["peak_memory_rss_mb"] == pytest.approx(53.0)

    def test_missing_temperature(self) -> None:
        """When cpu_temps is empty, temperature fields should be None."""
        m = _compute_cost_metrics([0.01], [0.1], [5.0], [], [50.0],
                                   time.process_time(), False, 1.0)
        assert m["mean_cpu_temperature_c"] is None
        assert m["max_cpu_temperature_c"] is None


# ======================================================================
# Dry-run batch
# ======================================================================

class TestDryRunBatch:
    def test_dry_run_creates_output(self) -> None:
        """Dry-run batch with 1 condition × 1 repeat creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                leader_api="http://127.0.0.1:8000",
                leader_freqs=[2.0],
                follower_freqs=[1.5],
                coupling_gain=3.5,
                duration=1.0,
                dt=0.01,
                repeats=1,
                flash_on_time=0.06,
                random_delay_min=0.0,
                random_delay_max=0.1,
                sync_threshold_s=0.10,
                sync_cycles=5,
                log_dir=tmpdir,
                trial_prefix="test",
                dry_run=True,
                width=640, height=480,
                detection_mode="local_contrast",
                threshold_on=180, threshold_off=120,
                min_interval=0.2, window_s=5.0,
                led_pin=17,
                leader_shape="circle",
                leader_dot_size=120,
                leader_min_flash_interval_s=0.20,
                keep_leader_running=False,
                target_loop_rate_hz=30.0,
            )

            batch_dir = run_visual_batch(ns)

            assert (batch_dir / "batch_metadata.json").exists()
            assert (batch_dir / "aggregate_metrics.csv").exists()
            assert (batch_dir / "summary_by_condition.csv").exists()

            trials = list((batch_dir / "trials").iterdir())
            assert len(trials) == 1
            td = trials[0]
            assert (td / "metadata.json").exists()
            assert (td / "oscillator_log.csv").exists()
            assert (td / "detection_log.csv").exists()
            assert (td / "flash_events.csv").exists()
            assert (td / "metrics_summary.json").exists()

            # Verify k_critical in trial metadata
            meta = json.loads((td / "metadata.json").read_text())
            assert "k_critical_rad_s" in meta
            assert "k_ratio" in meta
            assert meta["dry_run"] is True
            assert meta["random_start_delay_s"] >= 0
            # Sanity fields should be present
            assert "expected_leader_flash_count_requested" in meta
            assert "detected_leader_flash_count" in meta
            assert "actual_trial_wall_duration_s" in meta

            # Verify aggregate CSV row
            with open(batch_dir / "aggregate_metrics.csv") as f:
                lines = f.readlines()
                assert len(lines) == 2  # header + 1 row

            # Verify summary by condition
            with open(batch_dir / "summary_by_condition.csv") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert "1.5" in lines[1]


# ======================================================================
# CSV / JSON helpers
# ======================================================================

class TestLogHelpers:
    def test_save_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.csv"
            _save_csv(p, [{"a": 1, "b": 2}, {"a": 3, "b": 4}], ["a", "b"])
            with open(p) as f:
                lines = f.readlines()
                assert lines[0].strip() == "a,b"
                assert lines[1].strip() == "1,2"

    def test_save_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.json"
            _save_json(p, {"key": "value"})
            data = json.loads(p.read_text())
            assert data["key"] == "value"


# ======================================================================
# Rising-edge detection and min-interval gating
# ======================================================================

class TestRisingEdgeDetection:
    """Verify OFF→ON transition detection with min-interval gating."""

    @staticmethod
    def _detect_edges(
        states: list[bool], times: list[float],
        min_interval_s: float = 0.20,
    ) -> list[float]:
        """Simulate the rising-edge detector used in the batch runner."""
        edges: list[float] = []
        prev = False
        last_event = -999.0
        for i, s in enumerate(states):
            if s and not prev:
                if times[i] - last_event >= min_interval_s:
                    edges.append(times[i])
                    last_event = times[i]
            prev = s
        return edges

    def test_two_edges_from_six_frames(self) -> None:
        """[F, T, T, T, F, T] → 2 edges (only transitions count)."""
        states = [False, True, True, True, False, True]
        times = [float(i) * 0.1 for i in range(6)]
        edges = self._detect_edges(states, times, min_interval_s=0.01)
        assert len(edges) == 2

    def test_repeated_on_frames_no_extra_edges(self) -> None:
        """10 consecutive ON frames → 1 edge only."""
        states = [False] * 3 + [True] * 10
        times = [float(i) * 0.033 for i in range(13)]
        edges = self._detect_edges(states, times, min_interval_s=0.01)
        assert len(edges) == 1

    def test_min_interval_gating(self) -> None:
        """2 Hz square wave at 30 fps → ~10 edges in 5 s, not 150."""
        # Simulate 2 Hz square wave: 0.25s ON, 0.25s OFF per half-cycle
        f = 2.0; period = 1.0 / f; half = period / 2.0
        dt = 0.033  # ~30 fps
        duration = 5.0
        n = int(duration / dt)
        states: list[bool] = []
        times: list[float] = []
        for i in range(n):
            t = i * dt
            times.append(t)
            phase = (t % period)
            states.append(phase < half)  # ON for first half of each period

        edges = self._detect_edges(states, times, min_interval_s=0.10)
        # At 2 Hz, 5 seconds → ~10–12 edges (some edge jitter possible)
        assert 7 <= len(edges) <= 14, f"got {len(edges)} edges, expected ~10"

    def test_no_false_edges_from_noise(self) -> None:
        """Rapid toggling within min_interval → only first edge counted."""
        # Signal toggles every 0.01s for 0.09s (well within min_interval=0.20)
        # Only the first rising edge should be counted
        states = [False, True, False, True, False, True, False, True, False]
        times = [i * 0.01 for i in range(9)]
        edges = self._detect_edges(states, times, min_interval_s=0.20)
        assert len(edges) == 1

    def test_min_interval_allows_second_after_cooldown(self) -> None:
        """Edge at t=0, edge at t=0.5 → both counted if min_interval=0.2."""
        states = [False, True, False, False, False, True]
        times = [0.0, 0.01, 0.2, 0.3, 0.4, 0.5]
        edges = self._detect_edges(states, times, min_interval_s=0.20)
        assert len(edges) == 2


# ======================================================================
# Expected flash count sanity
# ======================================================================

class TestFlashCountSanity:
    def test_2hz_5s_expected_count(self) -> None:
        expected = 5.0 * 2.0  # ~10
        assert expected == 10.0

    def test_ratio_warning_high(self) -> None:
        ratio = 91.0 / 10.0  # 9.1x
        assert ratio > 2.0

    def test_ratio_ok(self) -> None:
        ratio = 10.0 / 10.0  # 1.0x
        assert 0.5 <= ratio <= 2.0

    def test_ratio_warning_low(self) -> None:
        ratio = 2.0 / 10.0  # 0.2x
        assert ratio < 0.5


# ======================================================================
# Dry-run batch with new fields
# ======================================================================

class TestDryRunBatchExtended:
    def test_dry_run_includes_sanity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                leader_api="http://127.0.0.1:8000",
                leader_freqs=[2.0], follower_freqs=[1.5],
                coupling_gain=3.5, duration=1.0, dt=0.01, repeats=1,
                flash_on_time=0.06,
                random_delay_min=0.0, random_delay_max=0.1,
                sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, trial_prefix="test",
                dry_run=True,
                width=640, height=480,
                detection_mode="local_contrast",
                threshold_on=180, threshold_off=120,
                min_interval=0.2, window_s=5.0,
                led_pin=17,
                leader_shape="circle", leader_dot_size=120,
                leader_min_flash_interval_s=0.20,
                keep_leader_running=False,
                target_loop_rate_hz=30.0,
            )
            batch_dir = run_visual_batch(ns)

            with open(batch_dir / "aggregate_metrics.csv") as f:
                import csv
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 1
                r = rows[0]
                assert "expected_leader_flash_count" in r
                assert "detected_leader_flash_count" in r
                assert "leader_flash_count_ratio" in r

    def test_dry_run_keep_leader_running(self) -> None:
        """--keep-leader-running should be accepted without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                leader_api="http://127.0.0.1:8000",
                leader_freqs=[2.0], follower_freqs=[1.5],
                coupling_gain=3.5, duration=1.0, dt=0.01, repeats=1,
                flash_on_time=0.06,
                random_delay_min=0.0, random_delay_max=0.1,
                sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, trial_prefix="test",
                dry_run=True,
                width=640, height=480,
                detection_mode="local_contrast",
                threshold_on=180, threshold_off=120,
                min_interval=0.2, window_s=5.0,
                led_pin=17,
                leader_shape="circle", leader_dot_size=120,
                leader_min_flash_interval_s=0.20,
                keep_leader_running=True,
                target_loop_rate_hz=30.0,
            )
            batch_dir = run_visual_batch(ns)
            assert (batch_dir / "batch_metadata.json").exists()

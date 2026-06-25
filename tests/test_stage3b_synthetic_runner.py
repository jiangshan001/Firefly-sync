"""Tests for Stage 3B synthetic model runner and sweep."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.run_stage3b_synthetic_model_batch import (
    _generate_leader_flash_times,
    _run_synthetic_trial,
    run_synthetic_batch,
)
from experiments.sweep_stage3b_synthetic_models import run_sweep


# ======================================================================
# Leader flash generation
# ======================================================================

class TestLeaderFlashGeneration:
    def test_2hz_10s_produces_approx_20_flashes(self) -> None:
        times = _generate_leader_flash_times(10.0, 2.0)
        assert 19 <= len(times) <= 21, f"got {len(times)} flashes"

    def test_zero_freq_produces_empty(self) -> None:
        times = _generate_leader_flash_times(10.0, 0.0)
        assert len(times) == 0

    def test_jitter_introduces_variation(self) -> None:
        times = _generate_leader_flash_times(10.0, 2.0, jitter_std_s=0.02)
        # With jitter, intervals should not all be exactly 0.5
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        assert any(abs(iv - 0.5) > 0.001 for iv in intervals)

    def test_missed_detection_reduces_count(self) -> None:
        times = _generate_leader_flash_times(10.0, 2.0, missed_prob=0.5)
        # With 50% miss rate, should have fewer than 20
        assert len(times) < 21


# ======================================================================
# Single trial runner
# ======================================================================

class TestSyntheticTrialRunner:
    def test_pco_if_creates_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "trial"
            out.mkdir()
            lt = _generate_leader_flash_times(5.0, 2.0)
            metrics = _run_synthetic_trial(
                model_name="pco_if",
                model_kwargs={"epsilon": 0.25, "refractory_period_s": 0.05},
                leader_flash_times=lt, leader_freq_hz=2.0,
                follower_initial_freq_hz=1.5,
                duration_s=5.0, dt=0.01,
                sync_threshold_s=0.10, sync_cycles=5,
                trial_id="test", out_dir=out,
            )
            assert (out / "metadata.json").exists()
            assert (out / "oscillator_log.csv").exists()
            assert (out / "flash_events.csv").exists()
            assert (out / "metrics_summary.json").exists()
            assert "synchronization_success" in metrics

    def test_eapf_creates_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "trial"
            out.mkdir()
            lt = _generate_leader_flash_times(5.0, 2.0)
            metrics = _run_synthetic_trial(
                model_name="eapf",
                model_kwargs={"phase_gain": 0.2, "frequency_gain": 0.05,
                              "frequency_min_hz": 0.5, "frequency_max_hz": 4.0,
                              "leader_period_window": 6},
                leader_flash_times=lt, leader_freq_hz=2.0,
                follower_initial_freq_hz=1.5,
                duration_s=5.0, dt=0.01,
                sync_threshold_s=0.10, sync_cycles=5,
                trial_id="test", out_dir=out,
            )
            assert (out / "metadata.json").exists()
            assert (out / "oscillator_log.csv").exists()
            assert (out / "flash_events.csv").exists()
            assert (out / "metrics_summary.json").exists()

    def test_leader_flash_events_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "trial"
            out.mkdir()
            lt = _generate_leader_flash_times(1.0, 2.0)
            _run_synthetic_trial(
                model_name="pco_if",
                model_kwargs={"epsilon": 0.25, "refractory_period_s": 0.05},
                leader_flash_times=lt, leader_freq_hz=2.0,
                follower_initial_freq_hz=1.5,
                duration_s=1.0, dt=0.01,
                sync_threshold_s=0.10, sync_cycles=5,
                trial_id="test", out_dir=out,
            )
            import csv
            with open(out / "flash_events.csv") as f:
                rows = list(csv.DictReader(f))
            types = [r["event_type"] for r in rows]
            assert "leader_flash" in types


# ======================================================================
# Batch runner
# ======================================================================

class TestSyntheticBatch:
    def test_pco_if_batch_creates_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                model="pco_if", duration=2.0, repeats=1,
                leader_freqs=[2.0], follower_freqs=[1.5],
                dt=0.01, sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, trial_prefix="test",
                random_seed=42,
                epsilon=0.25, fire_threshold=1.0,
                refractory_period_s=0.05,
                pco_coupling_mode="proportional_gap",
                pco_state_curve_beta=3.0,
                phase_gain=0.2, frequency_gain=0.05,
                frequency_min_hz=0.5, frequency_max_hz=4.0,
                leader_period_window=6,
                leader_jitter_std_s=0.0, missed_detection_prob=0.0,
                false_positive_rate_hz=0.0,
            )
            batch_dir = run_synthetic_batch(ns)
            assert (batch_dir / "batch_metadata.json").exists()
            assert (batch_dir / "aggregate_metrics.csv").exists()
            assert (batch_dir / "summary_by_condition.csv").exists()

    def test_eapf_batch_creates_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                model="eapf", duration=2.0, repeats=1,
                leader_freqs=[2.0], follower_freqs=[1.5],
                dt=0.01, sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, trial_prefix="test",
                random_seed=42,
                epsilon=0.25, fire_threshold=1.0,
                refractory_period_s=0.05,
                pco_coupling_mode="proportional_gap",
                pco_state_curve_beta=3.0,
                phase_gain=0.2, frequency_gain=0.05,
                frequency_min_hz=0.5, frequency_max_hz=4.0,
                leader_period_window=6,
                leader_jitter_std_s=0.0, missed_detection_prob=0.0,
                false_positive_rate_hz=0.0,
            )
            batch_dir = run_synthetic_batch(ns)
            assert (batch_dir / "aggregate_metrics.csv").exists()

    def test_aggregate_contains_sync_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                model="pco_if", duration=2.0, repeats=1,
                leader_freqs=[2.0], follower_freqs=[1.5],
                dt=0.01, sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, trial_prefix="test",
                random_seed=42,
                epsilon=0.25, fire_threshold=1.0,
                refractory_period_s=0.05,
                pco_coupling_mode="proportional_gap",
                pco_state_curve_beta=3.0,
                phase_gain=0.2, frequency_gain=0.05,
                frequency_min_hz=0.5, frequency_max_hz=4.0,
                leader_period_window=6,
                leader_jitter_std_s=0.0, missed_detection_prob=0.0,
                false_positive_rate_hz=0.0,
            )
            batch_dir = run_synthetic_batch(ns)
            import csv
            with open(batch_dir / "aggregate_metrics.csv") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert "synchronization_success" in rows[0]
            assert "time_to_synchronization_s" in rows[0]


# ======================================================================
# Sweep script
# ======================================================================

class TestSweepSmoke:
    def test_tiny_sweep_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                duration=2.0, repeats=1, dt=0.01,
                sync_threshold_s=0.10, sync_cycles=5,
                log_dir=tmpdir, random_seed=42,
                skip_pco_if=False, skip_eapf=False,
                pco_modes=["proportional_gap"], pco_epsilons=[0.25],
                pco_refractories=[0.05], pco_betas=[3.0],
                eapf_phase_gains=[0.20], eapf_freq_gains=[0.05],
            )
            sweep_dir = run_sweep(ns)
            assert (sweep_dir / "recommended_parameters.json").exists()
            rec = json.loads((sweep_dir / "recommended_parameters.json").read_text())
            assert "pco_if" in rec
            assert "eapf" in rec

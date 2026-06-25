"""Tests for Stage 4A multi-neighbour simulation."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest

from firefly_sync.multi_agent.topology import build_topology, TOPOLOGY_TYPES
from firefly_sync.multi_agent.agent import Agent, AgentConfig
from firefly_sync.multi_agent.simulation import MultiAgentSimulation
from firefly_sync.multi_agent.metrics import (
    check_group_synchronisation,
    compute_group_metrics,
)
from experiments.run_stage4a_multi_neighbour_simulation import run_batch


# ======================================================================
# Topology
# ======================================================================

class TestTopology:
    def test_all_to_all_3_agents(self) -> None:
        topo = build_topology(3, "all_to_all")
        assert topo.adjacency[0] == [1, 2]
        assert topo.adjacency[1] == [0, 2]
        assert topo.adjacency[2] == [0, 1]

    def test_chain_3_agents(self) -> None:
        topo = build_topology(3, "chain")
        assert topo.adjacency[0] == [1]
        assert topo.adjacency[1] == [0, 2]
        assert topo.adjacency[2] == [1]

    def test_directed_chain_3_agents(self) -> None:
        topo = build_topology(3, "directed_chain")
        assert topo.adjacency[0] == []
        assert topo.adjacency[1] == [0]
        assert topo.adjacency[2] == [1]

    def test_ring_3_agents(self) -> None:
        topo = build_topology(3, "ring")
        assert topo.adjacency[0] == [2, 1]
        assert topo.adjacency[1] == [0, 2]
        assert topo.adjacency[2] == [1, 0]

    def test_invalid_topology_raises(self) -> None:
        with pytest.raises(ValueError):
            build_topology(3, "nonexistent")


# ======================================================================
# Agent
# ======================================================================

class TestAgent:
    def test_creates_kuramoto_agent(self) -> None:
        cfg = AgentConfig(agent_id=0, model="kuramoto", initial_frequency_hz=2.0)
        agent = Agent(cfg)
        assert agent.model == "kuramoto"

    def test_creates_pco_agent(self) -> None:
        cfg = AgentConfig(agent_id=0, model="pco_if", initial_frequency_hz=1.5)
        agent = Agent(cfg)
        assert agent.model == "pco_if"

    def test_creates_eapf_agent(self) -> None:
        cfg = AgentConfig(agent_id=0, model="eapf", initial_frequency_hz=2.3)
        agent = Agent(cfg)
        assert agent.model == "eapf"


# ======================================================================
# Simulation
# ======================================================================

class TestSimulation:
    def test_kuramoto_all_to_all_runs(self) -> None:
        configs = [AgentConfig(agent_id=i, model="kuramoto",
                                initial_frequency_hz=f)
                    for i, f in enumerate([1.5, 2.0, 2.3])]
        topo = build_topology(3, "all_to_all")
        sim = MultiAgentSimulation(configs, topo, dt=0.01)
        sim.run(2.0)
        results = sim.get_results()
        assert len(results["flash_events"]) > 0
        assert len(results["agent_flash_times"]) == 3

    def test_pco_all_to_all_runs(self) -> None:
        configs = [AgentConfig(agent_id=i, model="pco_if",
                                initial_frequency_hz=f)
                    for i, f in enumerate([1.5, 2.0, 2.3])]
        topo = build_topology(3, "all_to_all")
        sim = MultiAgentSimulation(configs, topo, dt=0.01)
        sim.run(2.0)
        results = sim.get_results()
        assert len(results["flash_events"]) > 0

    def test_eapf_all_to_all_runs(self) -> None:
        configs = [AgentConfig(agent_id=i, model="eapf",
                                initial_frequency_hz=f)
                    for i, f in enumerate([1.5, 2.0, 2.3])]
        topo = build_topology(3, "all_to_all")
        sim = MultiAgentSimulation(configs, topo, dt=0.01)
        sim.run(2.0)
        results = sim.get_results()
        assert len(results["flash_events"]) > 0


# ======================================================================
# Metrics
# ======================================================================

class TestGroupMetrics:
    def test_returns_required_fields(self) -> None:
        flash_times = [[0.5, 1.0, 1.5], [0.51, 1.01, 1.51], [0.52, 1.02, 1.52]]
        m = compute_group_metrics(flash_times)
        for key in ["final_frequency_spread_hz", "mean_pairwise_timing_error_s",
                     "flash_timing_dispersion_s", "mean_order_parameter_R",
                     "final_order_parameter_R"]:
            assert key in m, f"missing {key}"

    def test_perfectly_synced(self) -> None:
        flash_times = [[float(i) for i in range(10)],
                        [float(i) for i in range(10)],
                        [float(i) for i in range(10)]]
        m = compute_group_metrics(flash_times)
        assert m["mean_pairwise_timing_error_s"] == pytest.approx(0.0)
        assert m["flash_timing_dispersion_s"] == pytest.approx(0.0)


class TestNewDiagnostics:
    def test_identical_flash_trains_pass_strict_sync(self) -> None:
        """Three agents with identical regular 2 Hz flash trains."""
        ft = [[float(i) * 0.5 for i in range(20)],
              [float(i) * 0.5 for i in range(20)],
              [float(i) * 0.5 for i in range(20)]]
        m = compute_group_metrics(ft)
        assert m["group_sync_success"] is True
        assert m["phase_sync_success"] is True
        assert m["frequency_lock_success"] is True
        assert m["one_to_one_flash_lock_success"] is True
        assert m["flash_count_ratio"] == pytest.approx(1.0)
        assert m["sync_diagnostic_label"] == "full_group_sync"

    def test_extra_flashes_detected(self) -> None:
        """Agent 2 flashes twice as often → 1:1 lock fails."""
        ft = [[float(i) * 0.5 for i in range(20)],       # ~10 flashes in 10s
              [float(i) * 0.5 for i in range(20)],
              [float(i) * 0.25 for i in range(40)]]       # ~20 flashes in 10s
        m = compute_group_metrics(ft)
        assert m["one_to_one_flash_lock_success"] is False
        assert m["group_sync_success"] is False
        # flash_count_ratio should be ~2.0
        assert m["flash_count_ratio"] > 1.2
        assert "extra_flash" in m["sync_diagnostic_label"] or m["sync_diagnostic_label"] == "extra_flashes_or_harmonic_locking"

    def test_effective_frequency_from_flash_events(self) -> None:
        """Effective frequency should be computed from flash timestamps."""
        ft = [[float(i) * 0.5 for i in range(20)],
              [float(i) * 0.5 for i in range(20)]]
        m = compute_group_metrics(ft)
        assert m["effective_frequency_agent_0_hz"] == pytest.approx(2.0, rel=0.05)
        assert m["effective_frequency_agent_1_hz"] == pytest.approx(2.0, rel=0.05)
        assert m["valid_frequency_agent_count"] == 2

    def test_no_flashes_no_crash(self) -> None:
        """Empty flash lists should not crash."""
        m = compute_group_metrics([[], [], []])
        assert m["group_sync_success"] is False
        assert m["sync_diagnostic_label"] == "insufficient_flash_events"
        assert m["flash_count_ratio"] == "inf"

    def test_pco_style_diagnostic(self) -> None:
        """PCO-style: coincident clusters but agent 1 fires ~1.5× more."""
        # Agent 0 and 2 fire together every 0.5s (2 Hz)
        # Agent 1 fires at 3 Hz: every 0.333s, will coincide occasionally
        ft0 = [i * 0.5 for i in range(20)]  # 0.0, 0.5, 1.0, ...
        ft1 = [i * 0.333 for i in range(30)]  # 0.0, 0.333, 0.666, 1.0, ...
        ft2 = [i * 0.5 + 0.02 for i in range(20)]
        m = compute_group_metrics([ft0, ft1, ft2])
        # Flash count ratio should be > 1.2
        assert m["flash_count_ratio"] != "inf"
        assert float(m["flash_count_ratio"]) > 1.2
        assert m["one_to_one_flash_lock_success"] is False
        assert m["group_sync_success"] is False


# ======================================================================
# Runner
# ======================================================================

class TestRunnerSmoke:
    def test_creates_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                model="kuramoto", topology="all_to_all",
                duration=2.0, repeats=1, dt=0.01,
                initial_frequencies=[1.5, 2.0, 2.3],
                random_seed=42,
                log_dir=tmpdir, no_plots=True,
                kuramoto_k=3.5,
                pco_coupling_mode="mirollo_state", pco_epsilon=0.25,
                pco_refractory_period_s=0.05, pco_state_curve_beta=3.0,
                eapf_phase_gain=0.3, eapf_frequency_gain=0.1,
                eapf_frequency_min_hz=0.5, eapf_frequency_max_hz=4.0,
                event_delay_s=0.0, missed_event_prob=0.0,
            )
            batch_dir = run_batch(ns)
            assert (batch_dir / "batch_metadata.json").exists()
            assert (batch_dir / "aggregate_metrics.csv").exists()
            assert (batch_dir / "summary_by_model_topology.csv").exists()

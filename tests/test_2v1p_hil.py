import numpy as np
import pytest
import json
import argparse

from experiments import run_leader_ui
from experiments.run_2v1p_eapf_hil import _apply_roi_config, _roi_overlap_ratio, _trace_for_roi
from firefly_sync.hardware.multi_roi_flash_detector import MultiROIFlashDetector
from firefly_sync.multi_agent.hil_topology import build_mixed_reality_topology


def test_chain_pi_middle_topology_masks_direct_virtual_link():
    topology = build_mixed_reality_topology("chain_pi_middle")

    assert topology.can_observe("V0", "P0")
    assert topology.can_observe("V1", "P0")
    assert topology.can_observe("P0", "V0")
    assert topology.can_observe("P0", "V1")
    assert not topology.can_observe("V0", "V1")
    assert not topology.can_observe("V1", "V0")


def test_chain_pi_downstream_masks_direct_v0_pi_link():
    topology = build_mixed_reality_topology("chain_pi_downstream")

    assert topology.can_observe("V0", "V1")
    assert topology.can_observe("V1", "V0")
    assert topology.can_observe("V1", "P0")
    assert topology.can_observe("P0", "V1")
    assert not topology.can_observe("V0", "P0")
    assert not topology.can_observe("P0", "V0")


def test_multi_roi_keeps_duplicate_suppression_independent():
    detector = MultiROIFlashDetector(
        rois=[
            {"roi_id": 0, "agent_id": "V0", "roi": [0, 0, 20, 20]},
            {"roi_id": 1, "agent_id": "V1", "roi": [30, 0, 20, 20]},
        ],
        detection_mode="mean",
        threshold_on=100.0,
        threshold_off=50.0,
        min_interval_s=0.2,
        use_adaptive=False,
    )

    dark = np.zeros((30, 60), dtype=np.uint8)
    v0_on = dark.copy()
    v0_on[0:20, 0:20] = 255
    v1_on = dark.copy()
    v1_on[0:20, 30:50] = 255

    detector.process_frame(dark, now_s=0.0)
    first = detector.process_frame(v0_on, now_s=1.0)
    detector.process_frame(dark, now_s=1.05)
    second = detector.process_frame(v1_on, now_s=1.08)

    assert [e["agent_id"] for e in first["events"]] == ["V0"]
    assert [e["agent_id"] for e in second["events"]] == ["V1"]


def test_server_multi_mode_routes_pi_events_by_topology():
    run_leader_ui._stop_agent_loop()
    with run_leader_ui._agents_lock:
        run_leader_ui._agents.clear()
        run_leader_ui._pi_flash_times.clear()
        run_leader_ui._pending_flash_events.clear()
    run_leader_ui._update_config({
        "mutual_agent_mode": "multi_2v1p",
        "topology": "chain_pi_downstream",
        "feedback_enabled": True,
    })
    run_leader_ui._configure_mutual_agents("multi_2v1p", "chain_pi_downstream")
    with run_leader_ui._agents_lock:
        agents = [a for a in run_leader_ui._agents if a.get("role") == "virtual"]

    new_events = run_leader_ui._step_multi_eapf_agents(
        agents,
        dt=0.033,
        now=0.5,
        pending_events=[{"source_agent_id": "P0", "timestamp": 0.5}],
        running=True,
        feedback_enabled=True,
        flash_duration_s=0.1,
    )

    assert new_events == []
    counts = {a["agent_id"]: a["received_neighbour_events"] for a in agents}
    assert counts["V0"] == 0
    assert counts["V1"] == 1


def test_single_mode_configuration_remains_one_virtual_agent():
    run_leader_ui._stop_agent_loop()
    with run_leader_ui._agents_lock:
        run_leader_ui._agents.clear()
    run_leader_ui._configure_mutual_agents("single_1v1p", "all_to_all")

    with run_leader_ui._agents_lock:
        snapshot = [run_leader_ui._compact_snapshot(a) for a in run_leader_ui._agents]

    assert len(snapshot) == 1
    assert snapshot[0]["agent_id"] == "V0"
    assert snapshot[0]["role"] == "virtual"


def test_roi_config_fills_missing_manual_rois():
    cfg = {
        "V0": {"roi": [1, 2, 30, 40]},
        "V1": {"roi": [50, 60, 70, 80]},
    }
    from pathlib import Path
    path = Path(".pytest_cache") / "test_auto_rois.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg))
    args = argparse.Namespace(roi_config=str(path), roi_v0=None, roi_v1=None)

    _apply_roi_config(args)

    assert args.roi_v0 == [1, 2, 30, 40]
    assert args.roi_v1 == [50, 60, 70, 80]


def test_roi_overlap_ratio_detects_overlap():
    assert _roi_overlap_ratio([0, 0, 20, 20], [10, 10, 20, 20]) == pytest.approx(0.25)
    assert _roi_overlap_ratio([0, 0, 10, 10], [30, 30, 10, 10]) == 0.0


def test_roi_trace_frequency_estimator_keeps_two_hz_signal():
    roi = [5, 5, 30, 30]
    samples = []
    fps = 30.0
    duration_s = 5.0
    for i in range(int(duration_s * fps)):
        t = i / fps
        frame = np.full((50, 50), 20, dtype=np.uint8)
        if (t % 0.5) < 0.12:
            frame[14:26, 14:26] = 255
        samples.append({"t_s": t, "frame": frame})

    trace = _trace_for_roi(samples, roi)

    assert trace["trace_frequency_method"] == "rising_edge"
    assert trace["trace_rising_edge_count"] >= 8
    assert trace["roi_estimated_frequency_hz"] == pytest.approx(2.0, abs=0.15)

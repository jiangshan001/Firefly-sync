import math
import random
from pathlib import Path

import pytest

from experiments.run_step5c2_2v1p_eapf_sync_batch import (
    EventFaultInjector,
    FREQUENCY_SETS,
    LockHoldConfig,
    LockHoldStabilizer,
    SyncThresholds,
    _batch_condition_summary,
    _common_frequency_stats,
    _compute_sync_metrics,
    _condition_definitions,
    _route_detected_virtual_event,
    build_parser,
    generate_initial_states,
    run_batch,
    verify_locked_eapf_config,
)
from firefly_sync.multi_agent.hil_topology import build_mixed_reality_topology


def _state_rows(sync_after_s: float = 2.0) -> list[dict]:
    rows = []
    for step in range(0, 81):
        t = step * 0.1
        if t < sync_after_s:
            phases = {"V0": 0.0, "V1": math.pi, "P0": math.pi / 2.0}
            freqs = {"V0": 1.6, "V1": 2.3, "P0": 1.9}
        else:
            phases = {"V0": 0.05, "V1": 0.08, "P0": 0.10}
            freqs = {"V0": 2.0, "V1": 2.02, "P0": 2.01}
        for agent_id in ("V0", "V1", "P0"):
            rows.append({
                "t_s": t,
                "agent_id": agent_id,
                "phase_rad": phases[agent_id],
                "frequency_hz": freqs[agent_id],
                "fire_count": step,
            })
    return rows


def test_locked_eapf_config_matches_stage5c2_requirements():
    verify_locked_eapf_config()


def test_random_initial_state_generation_is_deterministic_and_bounded():
    first = generate_initial_states(20260625, 3, random_initial=True)
    second = generate_initial_states(20260625, 3, random_initial=True)

    assert first == second
    assert set(first) == {"V0", "V1", "P0"}
    for state in first.values():
        assert 0.0 <= state["initial_phase_rad"] < 2.0 * math.pi
        assert 1.6 <= state["initial_frequency_hz"] <= 2.4


def test_frequency_set_generation_uses_named_agent_ranges():
    states = generate_initial_states(
        20260625,
        1,
        random_initial=True,
        freq_set="mixed_low_mid_high",
    )

    for agent_id, state in states.items():
        lo, hi = FREQUENCY_SETS["mixed_low_mid_high"]["agents"][agent_id]
        assert lo <= state["initial_frequency_hz"] <= hi


def test_all_to_all_topology_routes_every_pair_except_self():
    topology = build_mixed_reality_topology("all_to_all")

    assert topology.visible_neighbours("V0") == ["V1", "P0"]
    assert topology.visible_neighbours("V1") == ["V0", "P0"]
    assert topology.visible_neighbours("P0") == ["V0", "V1"]
    assert topology.numeric_neighbour_ids("P0", ["V0", "V1", "P0"]) == [0, 1]


def test_sync_metrics_compute_time_to_sync_and_final_window_values():
    metrics = _compute_sync_metrics(
        _state_rows(sync_after_s=2.0),
        thresholds=SyncThresholds(required_window_s=1.0, final_window_s=2.0),
    )

    assert metrics["sync_success"] is True
    assert metrics["continuous_sync_success"] is True
    assert metrics["final_sync_success"] is True
    assert 1.7 <= metrics["time_to_sync_s"] <= 2.0
    assert metrics["final_mean_pairwise_phase_error_cycles"] < 0.02
    assert metrics["final_frequency_disagreement_hz"] < 0.03
    assert metrics["final_mean_order_parameter_R"] > 0.99


def test_final_sync_success_can_pass_when_continuous_window_does_not():
    rows = _state_rows(sync_after_s=7.5)
    metrics = _compute_sync_metrics(
        rows,
        thresholds=SyncThresholds(required_window_s=5.0, final_window_s=1.0),
    )

    assert metrics["continuous_sync_success"] is False
    assert metrics["final_sync_success"] is True


def test_common_frequency_stats_reports_std_and_slope():
    samples = [
        {"t_s": 0.0, "common_frequency_hz": 2.0},
        {"t_s": 1.0, "common_frequency_hz": 1.99},
        {"t_s": 2.0, "common_frequency_hz": 1.98},
    ]

    stats = _common_frequency_stats(samples)

    assert stats["mean_common_frequency_hz"] == pytest.approx(1.99)
    assert stats["common_frequency_std_hz"] > 0
    assert stats["common_frequency_slope_hz_per_s"] == pytest.approx(-0.01)


def _lock_sample(t_s: float, *, good: bool = True, common_frequency_hz: float = 2.0) -> dict:
    if good:
        return {
            "t_s": t_s,
            "order_parameter_R": 0.99,
            "mean_pairwise_phase_error_cycles": 0.02,
            "frequency_disagreement_hz": 0.01,
            "common_frequency_hz": common_frequency_hz,
        }
    return {
        "t_s": t_s,
        "order_parameter_R": 0.70,
        "mean_pairwise_phase_error_cycles": 0.20,
        "frequency_disagreement_hz": 0.20,
        "common_frequency_hz": common_frequency_hz,
    }


def test_lock_hold_fast_lock_acquires_after_short_valid_window():
    stabilizer = LockHoldStabilizer(LockHoldConfig(stabilizer="lock_hold"))

    event = None
    for t_s in [0.0, 0.25, 0.50, 0.75, 1.0]:
        event = stabilizer.update(_lock_sample(t_s, good=True)) or event

    assert stabilizer.state == "hold"
    assert stabilizer.lock_time_s == pytest.approx(1.0)
    assert stabilizer.f_lock_hz == pytest.approx(2.0)
    assert event["event_type"] == "lock_hold_acquired"
    assert event["f_lock_source_window_s"] == pytest.approx(1.0)


def test_lock_hold_window_elapsed_tolerates_irregular_polling():
    stabilizer = LockHoldStabilizer(LockHoldConfig(stabilizer="lock_hold"))

    for t_s in [0.0, 0.41, 0.82, 1.21]:
        stabilizer.update(_lock_sample(t_s, good=True))

    assert stabilizer.state == "hold"
    assert stabilizer.lock_time_s == pytest.approx(1.21)


def test_lock_hold_anchor_uses_rolling_median_over_lock_window():
    stabilizer = LockHoldStabilizer(LockHoldConfig(stabilizer="lock_hold"))

    for t_s, freq in [(0.0, 2.0), (0.25, 2.1), (0.50, 2.2), (0.75, 2.3), (1.0, 3.5)]:
        stabilizer.update(_lock_sample(t_s, good=True, common_frequency_hz=freq))

    assert stabilizer.state == "hold"
    assert stabilizer.f_lock_hz == pytest.approx(2.2)
    assert stabilizer.common_frequency_at_lock_median_hz == pytest.approx(2.2)
    assert stabilizer.common_frequency_at_lock_mean_hz == pytest.approx(2.42)


def test_lock_hold_unlocks_and_relocks():
    stabilizer = LockHoldStabilizer(
        LockHoldConfig(
            stabilizer="lock_hold",
            lock_window_s=0.5,
            unlock_window_s=0.5,
        )
    )
    for t_s in [0.0, 0.25, 0.75]:
        stabilizer.update(_lock_sample(t_s, good=True))
    assert stabilizer.state == "hold"

    for t_s in [1.0, 1.25, 1.75]:
        stabilizer.update(_lock_sample(t_s, good=False))
    assert stabilizer.state == "acquisition"
    assert stabilizer.unlock_count == 1

    for t_s in [2.0, 2.25, 2.75]:
        stabilizer.update(_lock_sample(t_s, good=True))
    assert stabilizer.state == "hold"
    assert stabilizer.relock_count == 1


def test_unlock_hysteresis_ignores_transient_spike():
    stabilizer = LockHoldStabilizer(LockHoldConfig(stabilizer="lock_hold"))
    for t_s in [0.0, 0.25, 0.50, 0.75, 1.0]:
        stabilizer.update(_lock_sample(t_s, good=True))
    assert stabilizer.state == "hold"

    stabilizer.update(_lock_sample(1.25, good=False))

    assert stabilizer.state == "hold"
    assert stabilizer.unlock_count == 0


def test_lock_acquired_metric_becomes_true_when_criteria_met():
    stabilizer = LockHoldStabilizer(LockHoldConfig(stabilizer="lock_hold"))
    agent_rows = []
    for t_s in [0.0, 0.5, 1.0]:
        sample = _lock_sample(t_s, good=True)
        stabilizer.update(sample)
        for agent_id, phase, freq in [
            ("V0", 0.02, 2.0),
            ("V1", 0.04, 2.01),
            ("P0", 0.06, 2.02),
        ]:
            agent_rows.append({
                "t_s": t_s,
                "agent_id": agent_id,
                "phase_rad": phase,
                "frequency_hz": freq,
            })

    metrics = _compute_sync_metrics(agent_rows, thresholds=SyncThresholds(), lock_rows=stabilizer.rows)

    assert metrics["lock_acquired"] is True
    assert metrics["lock_time_s"] == pytest.approx(1.0)


def test_recovery_time_uses_disruption_end_time():
    metrics = _compute_sync_metrics(
        _state_rows(sync_after_s=5.0),
        thresholds=SyncThresholds(required_window_s=1.0, final_window_s=2.0),
        disruption_end_s=4.0,
    )

    assert metrics["sync_success"] is True
    assert 4.8 <= metrics["recovery_time_absolute_s"] <= 5.0
    assert 0.8 <= metrics["recovery_time_s"] <= 1.0


def test_fault_injector_delay_books_and_flushes_p0_events():
    condition = _condition_definitions()["p0_event_delay_150ms"]
    injector = EventFaultInjector(condition, random.Random(1))
    posted = []

    injector.queue_or_post_p0_flash(
        t_s=1.0,
        post_event=lambda source_t, post_t, delay: posted.append((source_t, post_t, delay)),
    )
    injector.flush_due_p0_events(
        1.10,
        lambda source_t, post_t, delay: posted.append((source_t, post_t, delay)),
    )
    assert posted == []
    injector.flush_due_p0_events(
        1.151,
        lambda source_t, post_t, delay: posted.append((source_t, post_t, delay)),
    )

    assert posted[0][0] == pytest.approx(1.0)
    assert posted[0][2] == pytest.approx(0.151)
    assert injector.stats["p0_events_delayed"] == 1
    assert injector.stats["p0_delayed_events_posted"] == 1


def test_fault_injector_dropout_books_dropped_events():
    condition = _condition_definitions()["v0_event_dropout_20percent"]
    condition["fault"]["drop_probability"] = 1.0
    injector = EventFaultInjector(condition, random.Random(1))

    assert injector.keep_virtual_detection("V0") is False
    assert injector.keep_virtual_detection("V1") is True
    assert injector.stats["virtual_events_seen"] == 2
    assert injector.stats["virtual_events_dropped"] == 1


def test_baseline_detected_events_are_kept_by_default():
    topology = build_mixed_reality_topology("all_to_all")
    injector = EventFaultInjector(_condition_definitions()["baseline"], random.Random(1))
    row, routed_source = _route_detected_virtual_event(
        {
            "agent_id": "V0",
            "roi_id": 0,
            "raw_brightness": 123.0,
            "normalized_signal": 0.9,
        },
        t_s=1.25,
        injector=injector,
        topology=topology,
    )

    assert row["kept_for_p0"] == 1
    assert row["dropped"] == 0
    assert row["drop_reason"] is None
    assert routed_source == "V0"


def test_dropout_routing_can_drop_event_and_records_reason():
    topology = build_mixed_reality_topology("all_to_all")
    condition = _condition_definitions()["v0_event_dropout_20percent"]
    condition["fault"]["drop_probability"] = 1.0
    injector = EventFaultInjector(condition, random.Random(1))
    row, routed_source = _route_detected_virtual_event(
        {
            "agent_id": "V0",
            "roi_id": 0,
            "raw_brightness": 123.0,
            "normalized_signal": 0.9,
        },
        t_s=1.25,
        injector=injector,
        topology=topology,
    )

    assert row["kept_for_p0"] == 0
    assert row["dropped"] == 1
    assert row["drop_reason"] == "virtual_event_dropout"
    assert routed_source is None


def test_delay_condition_keeps_detected_virtual_events_unless_dropped():
    topology = build_mixed_reality_topology("all_to_all")
    injector = EventFaultInjector(
        _condition_definitions()["p0_event_delay_150ms"],
        random.Random(1),
    )
    row, routed_source = _route_detected_virtual_event(
        {
            "agent_id": "V1",
            "roi_id": 1,
            "raw_brightness": 150.0,
            "normalized_signal": 0.95,
        },
        t_s=2.0,
        injector=injector,
        topology=topology,
    )

    assert row["kept_for_p0"] == 1
    assert row["dropped"] == 0
    assert row["injected_delay_s"] == 0.0
    assert routed_source == "V1"


def test_baseline_lock_hold_empty_event_frame_has_no_unbound_kept_regression():
    topology = build_mixed_reality_topology("all_to_all")
    injector = EventFaultInjector(_condition_definitions()["baseline"], random.Random(1))
    neighbour_sources = []

    for evt in []:
        _row, routed_source = _route_detected_virtual_event(
            evt,
            t_s=0.0,
            injector=injector,
            topology=topology,
        )
        if routed_source is not None:
            neighbour_sources.append(routed_source)

    assert topology.numeric_neighbour_ids("P0", neighbour_sources) == []


def test_batch_condition_summary_aggregates_success_and_means():
    rows = [
        {
            "condition": "baseline",
            "continuous_sync_success": True,
            "final_sync_success": True,
            "frequency_stability_success": True,
            "lock_acquired": True,
            "stabilizer": "lock_hold",
            "frequency_set": "same_2hz_random_phase",
            "time_to_sync_s": 3.0,
            "lock_time_s": 4.0,
            "final_frequency_disagreement_hz": 0.02,
            "final_mean_pairwise_phase_error_cycles": 0.05,
            "final_mean_order_parameter_R": 0.96,
            "final_common_frequency_std_hz": 0.01,
            "final_common_frequency_slope_hz_per_s": 0.001,
        },
        {
            "condition": "baseline",
            "continuous_sync_success": False,
            "final_sync_success": True,
            "frequency_stability_success": False,
            "lock_acquired": False,
            "stabilizer": "lock_hold",
            "frequency_set": "same_2hz_random_phase",
            "time_to_sync_s": None,
            "final_frequency_disagreement_hz": 0.20,
            "final_mean_pairwise_phase_error_cycles": 0.30,
            "final_mean_order_parameter_R": 0.70,
            "final_common_frequency_std_hz": 0.20,
            "final_common_frequency_slope_hz_per_s": -0.02,
        },
    ]

    summary = _batch_condition_summary(rows)

    assert summary[0]["condition"] == "baseline"
    assert summary[0]["trials"] == 2
    assert summary[0]["success_rate"] == 0.5
    assert summary[0]["final_sync_success_rate"] == 1.0
    assert summary[0]["lock_acquired_rate"] == 0.5
    assert summary[0]["mean_time_to_sync_s"] == 3.0
    assert summary[0]["mean_final_frequency_disagreement_hz"] == 0.11


def test_cli_parses_multiple_frequency_sets_and_lock_hold_options():
    args = build_parser().parse_args([
        "--freq-sets", "same_2hz_random_phase", "wide_1_3",
        "--stabilizer", "lock_hold",
        "--hold-frequency-gain-scale", "0.05",
    ])

    assert args.freq_sets == ["same_2hz_random_phase", "wide_1_3"]
    assert args.stabilizer == "lock_hold"
    assert args.hold_frequency_gain_scale == pytest.approx(0.05)


def test_cli_lock_hold_defaults_are_fast_lock_hil_values():
    args = build_parser().parse_args(["--stabilizer", "lock_hold"])

    assert args.lock_r_threshold == pytest.approx(0.95)
    assert args.lock_phase_error_threshold_cycles == pytest.approx(0.08)
    assert args.lock_frequency_disagreement_threshold_hz == pytest.approx(0.05)
    assert args.lock_window_s == pytest.approx(1.0)
    assert args.lock_window_pass_ratio == pytest.approx(0.5)
    assert args.hold_frequency_anchor == "window_median"
    assert args.hold_anchor_gain == pytest.approx(0.08)
    assert args.unlock_r_threshold == pytest.approx(0.85)
    assert args.unlock_phase_error_threshold_cycles == pytest.approx(0.15)
    assert args.unlock_frequency_disagreement_threshold_hz == pytest.approx(0.10)
    assert args.unlock_window_s == pytest.approx(4.0)
    assert args.unlock_window_fail_ratio == pytest.approx(0.8)


def test_dry_run_creates_no_hardware_log_root():
    log_dir = Path(".pytest_cache") / f"step5c2_dry_run_no_logs_{random.randint(1, 10**9)}"
    args = build_parser().parse_args([
        "--dry-run",
        "--log-dir", str(log_dir),
        "--conditions", "baseline",
        "--freq-sets", "same_2hz_random_phase",
        "--stabilizer", "lock_hold",
    ])

    plan = run_batch(args)

    assert plan["dry_run"] is True
    assert not log_dir.exists()


def test_cli_lock_hold_overrides_still_work():
    args = build_parser().parse_args([
        "--stabilizer", "lock_hold",
        "--lock-r-threshold", "0.93",
        "--lock-window-s", "0.5",
        "--lock-window-pass-ratio", "0.4",
        "--hold-frequency-anchor", "window_mean",
        "--unlock-window-fail-ratio", "0.7",
    ])

    assert args.lock_r_threshold == pytest.approx(0.93)
    assert args.lock_window_s == pytest.approx(0.5)
    assert args.lock_window_pass_ratio == pytest.approx(0.4)
    assert args.hold_frequency_anchor == "window_mean"
    assert args.unlock_window_fail_ratio == pytest.approx(0.7)

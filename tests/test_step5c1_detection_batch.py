import argparse

from experiments.run_step5c1_multi_roi_detection_batch import (
    _condition_definitions,
    _condition_display_config,
    _condition_summary,
)


def test_step5c1_condition_names_are_unique_and_cover_groups():
    conditions = _condition_definitions()
    names = [condition["name"] for condition in conditions]
    groups = {condition["group"] for condition in conditions}
    stress_names = {
        condition["name"] for condition in conditions
        if not condition["required_for_ready"]
    }

    assert len(names) == len(set(names))
    assert "baseline" in names
    assert {"position", "size", "contrast", "frequency"} <= groups
    assert "position_closer" not in names
    assert {
        "position_too_close_stress",
        "tiny_small_small_stress",
        "small_low_contrast_stress",
    } <= stress_names
    for condition in conditions:
        assert condition["category"] == condition["group"]
        assert condition["intended_difficulty"]
        assert condition["failure_interpretation"]


def test_step5c1_condition_summary_passes_when_all_agent_rows_pass():
    thresholds = argparse.Namespace(
        pass_overlap_ratio=0.1,
        pass_count_recall=0.85,
        pass_frequency_error_hz=0.10,
    )
    condition = _condition_definitions()[0]
    rows = [
        {
            "repeat": 1,
            "agent_id": "V0",
            "count_recall": 0.95,
            "frequency_absolute_error_hz": 0.02,
            "warnings": "",
            "pass_repeat_agent": True,
        },
        {
            "repeat": 1,
            "agent_id": "V1",
            "count_recall": 0.93,
            "frequency_absolute_error_hz": 0.03,
            "warnings": "",
            "pass_repeat_agent": True,
        },
    ]
    auto_rois = {
        "calibration_valid": True,
        "assignment_method": "frequency_based",
        "overlap_ratio": 0.0,
    }

    summary = _condition_summary(condition, rows, auto_rois, thresholds)

    assert summary["pass_condition"] is True
    assert summary["required_for_ready"] is True
    assert summary["category"] == "baseline"
    assert summary["V0_count_recall_mean"] == 0.95
    assert summary["V1_frequency_error_mean_hz"] == 0.03


def test_contrast_condition_display_config_is_explicit():
    conditions = {condition["name"]: condition for condition in _condition_definitions()}

    baseline = _condition_display_config(conditions["baseline"])
    medium = _condition_display_config(conditions["contrast_medium"])
    low = _condition_display_config(conditions["contrast_low"])

    assert baseline["background"]["brightness"] == 0
    assert baseline["flash_on"]["brightness"] == 255
    assert medium["background"]["brightness"] == 40
    assert medium["flash_on"]["brightness"] == 180
    assert low["background"]["brightness"] == 80
    assert low["flash_on"]["brightness"] == 120
    assert medium["contrast"]["flash_minus_background"] < baseline["contrast"]["flash_minus_background"]
    assert low["contrast"]["flash_minus_background"] < medium["contrast"]["flash_minus_background"]
    assert low["agents"]["V0"]["position_xy"]
    assert low["agents"]["V1"]["frequency_hz"] == 2.0

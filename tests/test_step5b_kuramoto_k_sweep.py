import csv
import json
import sys
from pathlib import Path

import pytest

from experiments import analyse_step5b_kuramoto_k_sweep as analysis
from experiments import run_step5b_kuramoto_k_sweep as sweep


def _run_main(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["run_step5b_kuramoto_k_sweep.py", *args])
    sweep.main()


def _read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _successful_row(trial_dir: Path, trial_id: str) -> dict:
    return {
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "model": "kuramoto",
        "virtual_initial_freq": 2.0,
        "pi_initial_freq": 1.2,
        "duration": 1.0,
        "repeat": 1,
        "seed": 123,
        "trial_seed": 124,
        "random_phase_enabled": True,
        "virtual_initial_phase_rad": 0.1,
        "pi_initial_phase_rad": 0.2,
        "initial_phase_difference_rad": 0.1,
        "feedback_enabled": True,
        "model_parameters_json": '{"K": 3.5}',
        "kuramoto_K": 3.5,
        "actual_detection_fcr": 1.0,
        "frequency_error_final_5s_mean_abs": 0.1,
        "frequency_error_final_5s_abs_of_means": 0.1,
        "virtual_freq_final_5s_std": 0.01,
        "pi_freq_final_5s_std": 0.02,
        "timeout_or_failure": False,
        "error_message": "",
    }


def test_normal_run_initialises_formal_runtime_before_trial(monkeypatch, tmp_path):
    calls = []

    def fake_ensure_hw(dry=False):
        calls.append(("ensure", dry))

    def fake_run_trial(args, trial_dir, spec):
        calls.append(("trial", spec.trial_id))
        trial_dir.mkdir(parents=True, exist_ok=True)
        row = _successful_row(trial_dir, spec.trial_id)
        sweep._save_json(trial_dir / "metrics_summary.json", row)
        return row

    monkeypatch.setattr(sweep, "ensure_runtime_imports", fake_ensure_hw)
    monkeypatch.setattr(sweep, "run_trial", fake_run_trial)

    _run_main(monkeypatch, [
        "--run-dir", str(tmp_path / "run"),
        "--duration", "1",
        "--freq-pairs", "2.0:1.2",
        "--kuramoto-gains", "3.5",
        "--repeats", "1",
    ])

    assert calls[:2] == [("ensure", False), ("trial", "kuramoto_K3p5_V2_P1p2_r01")]


def test_dry_run_prints_schedule_without_trial_side_effects(monkeypatch, tmp_path):
    run_dir = tmp_path / "dryrun"

    def fail_ensure_hw(dry=False):
        raise AssertionError("dry-run must not initialise hardware")

    def fail_run_trial(args, trial_dir, spec):
        raise AssertionError("dry-run must not execute trials")

    monkeypatch.setattr(sweep, "ensure_runtime_imports", fail_ensure_hw)
    monkeypatch.setattr(sweep, "run_trial", fail_run_trial)

    _run_main(monkeypatch, [
        "--dry-run",
        "--run-dir", str(run_dir),
        "--freq-pairs", "2.0:1.2",
        "--kuramoto-gains", "3.5",
        "--repeats", "1",
    ])

    assert not run_dir.exists()


def test_failed_trial_writes_failure_marker_and_is_excluded(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    trial_id = "kuramoto_K3p5_V2_P1p2_r01"

    def fake_run_trial(args, trial_dir, spec):
        trial_dir.mkdir(parents=True, exist_ok=True)
        row = _successful_row(trial_dir, spec.trial_id)
        row["timeout_or_failure"] = True
        row["error_message"] = "AssertionError: camera runtime missing"
        sweep._save_json(trial_dir / "metrics_summary.json", row)
        return row

    monkeypatch.setattr(sweep, "ensure_runtime_imports", lambda dry=False: None)
    monkeypatch.setattr(sweep, "run_trial", fake_run_trial)

    _run_main(monkeypatch, [
        "--run-dir", str(run_dir),
        "--duration", "1",
        "--freq-pairs", "2.0:1.2",
        "--kuramoto-gains", "3.5",
        "--repeats", "1",
    ])

    trial_dir = run_dir / trial_id
    assert (trial_dir / sweep.FAILURE_MARKER).exists()
    assert not (trial_dir / "metrics_summary.json").exists()
    assert sweep._scan_completed_trial_metrics(run_dir) == []
    assert _read_csv(run_dir / "aggregate_metrics.csv") == []

    monkeypatch.setattr(sys, "argv", [
        "analyse_step5b_kuramoto_k_sweep.py",
        str(run_dir),
    ])
    analysis.main()
    assert _read_csv(run_dir / "k_ranking.csv") == []

    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [
            "--resume",
            "--run-dir", str(run_dir),
            "--duration", "1",
            "--freq-pairs", "2.0:1.2",
            "--kuramoto-gains", "3.5",
            "--repeats", "1",
        ])
    assert excinfo.value.code == 2


def test_resume_skips_valid_completed_trial_but_not_failed_folder(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    trial_id = "kuramoto_K3p5_V2_P1p2_r01"
    trial_dir = run_dir / trial_id
    trial_dir.mkdir(parents=True)
    sweep._save_json(trial_dir / "metrics_summary.json", _successful_row(trial_dir, trial_id))

    def fail_run_trial(args, trial_dir, spec):
        raise AssertionError("valid completed trial should be skipped")

    monkeypatch.setattr(sweep, "ensure_runtime_imports", lambda dry=False: None)
    monkeypatch.setattr(sweep, "run_trial", fail_run_trial)

    _run_main(monkeypatch, [
        "--resume",
        "--run-dir", str(run_dir),
        "--duration", "1",
        "--freq-pairs", "2.0:1.2",
        "--kuramoto-gains", "3.5",
        "--repeats", "1",
    ])

    assert len(_read_csv(run_dir / "aggregate_metrics.csv")) == 1

    (trial_dir / sweep.FAILURE_MARKER).write_text(json.dumps({"trial_id": trial_id}))
    with pytest.raises(SystemExit) as excinfo:
        _run_main(monkeypatch, [
            "--resume",
            "--run-dir", str(run_dir),
            "--duration", "1",
            "--freq-pairs", "2.0:1.2",
            "--kuramoto-gains", "3.5",
            "--repeats", "1",
        ])
    assert excinfo.value.code == 2

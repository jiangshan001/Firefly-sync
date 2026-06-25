import math

import pytest

from experiments import run_leader_ui
from experiments.run_step5b_mutual_hil_batch import _build_schedule


def _phase_pairs(schedule):
    return [
        (
            round(spec.virtual_initial_phase_rad, 12),
            round(spec.pi_initial_phase_rad, 12),
            round(spec.initial_phase_difference_rad, 12),
        )
        for spec in schedule
    ]


def test_batch_random_phase_is_reproducible_with_same_seed():
    kwargs = dict(
        models=["eapf_consensus", "kuramoto"],
        freq_pairs=[(2.0, 1.5)],
        repeats=2,
        alternate_models=True,
        base_seed=1234,
        random_phase=True,
        virtual_phase_rad=None,
        pi_phase_rad=None,
    )

    assert _phase_pairs(_build_schedule(**kwargs)) == _phase_pairs(_build_schedule(**kwargs))


def test_batch_random_phase_changes_with_different_seed():
    common = dict(
        models=["eapf_consensus"],
        freq_pairs=[(2.0, 1.5)],
        repeats=1,
        alternate_models=False,
        random_phase=True,
        virtual_phase_rad=None,
        pi_phase_rad=None,
    )

    first = _phase_pairs(_build_schedule(base_seed=1234, **common))
    second = _phase_pairs(_build_schedule(base_seed=4321, **common))

    assert first != second


def test_batch_fixed_phase_defaults_to_zero_when_random_phase_disabled():
    schedule = _build_schedule(
        models=["eapf_consensus"],
        freq_pairs=[(2.0, 1.5)],
        repeats=1,
        alternate_models=False,
        base_seed=1234,
        random_phase=False,
        virtual_phase_rad=None,
        pi_phase_rad=None,
    )

    spec = schedule[0]
    assert spec.virtual_initial_phase_rad == 0.0
    assert spec.pi_initial_phase_rad == 0.0
    assert spec.initial_phase_difference_rad == 0.0


def test_leader_agent_reset_preserves_initial_phase():
    phase = 7.5
    expected = phase % (2.0 * math.pi)
    agent = run_leader_ui._init_agent(0, freq=2.0, initial_phase_rad=phase)

    agent["phase_rad"] = 1.0
    run_leader_ui._reset_agent(agent)

    assert agent["initial_phase_rad"] == pytest.approx(expected)
    assert agent["phase_rad"] == pytest.approx(expected)
    assert agent["phase"] == pytest.approx(expected)

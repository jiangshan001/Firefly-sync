"""Agent wrapper around core oscillator models.

Each ``Agent`` owns one oscillator (Kuramoto, PCO-I&F, or EAPF) and
exposes a uniform ``step()`` interface.  External coupling inputs or
neighbour-flash events are passed in per call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration for one agent in a multi-agent simulation.

    Attributes:
        agent_id: Integer ID (0-based).
        model: ``"kuramoto"``, ``"pco_if"``, or ``"eapf"``.
        initial_frequency_hz: Natural / initial frequency in Hz.
        initial_phase: Initial phase (radians for Kuramoto/EAPF;
            normalised [0,1) for PCO).
        coupling_strength: Kuramoto coupling gain K.
        pco_epsilon: PCO phase advance per pulse.
        pco_coupling_mode: PCO coupling rule.
        pco_refractory_period_s: PCO refractory window.
        pco_state_curve_beta: β for mirollo_state.
        eapf_phase_gain: EAPF phase correction gain.
        eapf_frequency_gain: EAPF frequency correction gain.
        eapf_frequency_min_hz: EAPF minimum frequency.
        eapf_frequency_max_hz: EAPF maximum frequency.
        eapf_leader_period_window: EAPF period-estimation window size.
    """

    agent_id: int = 0
    model: str = "kuramoto"
    initial_frequency_hz: float = 2.0
    initial_phase: float = 0.0

    # Kuramoto
    coupling_strength: float = 3.5

    # PCO-I&F
    pco_epsilon: float = 0.25
    pco_coupling_mode: str = "mirollo_state"
    pco_refractory_period_s: float = 0.05
    pco_state_curve_beta: float = 3.0

    # EAPF
    eapf_phase_gain: float = 0.3
    eapf_frequency_gain: float = 0.1
    eapf_frequency_min_hz: float = 0.5
    eapf_frequency_max_hz: float = 4.0
    eapf_leader_period_window: int = 6

    # PCO adaptive PRC
    pco_enable_phase_delay: bool = False
    pco_enable_frequency_adaptation: bool = False
    pco_frequency_adaptation_gain: float = 0.0
    pco_max_phase_correction: float = 0.40
    pco_min_inter_flash_interval_s: float = 0.0
    pco_post_flash_lockout_s: float = 0.0
    # Generic
    flash_on_time_s: float = 0.06

    # Tracking
    flash_times: list[float] = field(default_factory=list)


class Agent:
    """Multi-agent wrapper around a core oscillator.

    Provides a uniform interface so the simulation engine can treat
    all models the same way.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.id = config.agent_id
        self.model = config.model
        self._oscillator: Any = None
        self._flash_times: list[float] = []
        self._phase_history: list[float] = []
        self._freq_history: list[float] = []
        self._init_oscillator()

    def _init_oscillator(self) -> None:
        cfg = self.config
        if cfg.model == "kuramoto":
            from firefly_sync.core.kuramoto import KuramotoModel
            self._oscillator = KuramotoModel(
                natural_frequency=2.0 * math.pi * cfg.initial_frequency_hz,
                initial_phase=cfg.initial_phase,
                coupling_strength=cfg.coupling_strength,
                dt=0.01,
            )
        elif cfg.model == "pco_if":
            from firefly_sync.core.pco_integrate_fire import (
                PulseCoupledIFConfig,
                PulseCoupledIntegrateFireOscillator,
            )
            pco_cfg = PulseCoupledIFConfig(
                natural_frequency_hz=cfg.initial_frequency_hz,
                epsilon=cfg.pco_epsilon,
                coupling_mode=cfg.pco_coupling_mode,
                refractory_period_s=cfg.pco_refractory_period_s,
                state_curve_beta=cfg.pco_state_curve_beta,
                enable_phase_delay=cfg.pco_enable_phase_delay,
                enable_frequency_adaptation=cfg.pco_enable_frequency_adaptation,
                frequency_adaptation_gain=cfg.pco_frequency_adaptation_gain,
                max_phase_correction=cfg.pco_max_phase_correction,
                min_inter_flash_interval_s=cfg.pco_min_inter_flash_interval_s,
                post_flash_lockout_s=cfg.pco_post_flash_lockout_s,
            )
            self._oscillator = PulseCoupledIntegrateFireOscillator(pco_cfg)
        elif cfg.model == "eapf_consensus":
            from firefly_sync.core.event_based_consensus_pll import (
                ConsensusPLLConfig,
                EventBasedConsensusPLLOscillator,
            )
            ccfg = ConsensusPLLConfig(
                natural_frequency_hz=cfg.initial_frequency_hz,
                phase_gain=cfg.eapf_phase_gain,
                frequency_gain=cfg.eapf_frequency_gain,
                frequency_min_hz=cfg.eapf_frequency_min_hz,
                frequency_max_hz=cfg.eapf_frequency_max_hz,
            )
            self._oscillator = EventBasedConsensusPLLOscillator(ccfg)
            # Note: ConsensusPLLOscillator uses _phase_rad
        elif cfg.model == "eapf":
            from firefly_sync.core.event_based_phase_lock import (
                EventBasedPhaseLockConfig,
                EventBasedPhaseLockOscillator,
            )
            eapf_cfg = EventBasedPhaseLockConfig(
                natural_frequency_hz=cfg.initial_frequency_hz,
                phase_gain=cfg.eapf_phase_gain,
                frequency_gain=cfg.eapf_frequency_gain,
                frequency_min_hz=cfg.eapf_frequency_min_hz,
                frequency_max_hz=cfg.eapf_frequency_max_hz,
                leader_period_window=cfg.eapf_leader_period_window,
            )
            self._oscillator = EventBasedPhaseLockOscillator(eapf_cfg)
            self._oscillator._phase_rad = cfg.initial_phase
        else:
            raise ValueError(f"Unknown model: {cfg.model}")

    # -- properties --

    @property
    def phase(self) -> float:
        if self.model == "pco_if":
            return self._oscillator.phase
        elif self.model in ("eapf", "eapf_consensus"):
            return self._oscillator.phase_rad
        else:
            return self._oscillator.phase

    @property
    def frequency_hz(self) -> float:
        if self.model in ("eapf", "eapf_consensus"):
            return self._oscillator.frequency_hz
        elif self.model == "pco_if":
            # Estimate from recent flashes
            if len(self._flash_times) >= 2:
                recent = self._flash_times[-10:]
                intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
                return 1.0 / (sum(intervals) / len(intervals)) if intervals else self.config.initial_frequency_hz
            return self.config.initial_frequency_hz
        else:
            return self._oscillator.natural_frequency / (2.0 * math.pi)

    @property
    def flash_times(self) -> list[float]:
        return self._flash_times

    @property
    def fire_count(self) -> int:
        return self._oscillator.fire_count

    # -- step --

    def step(
        self,
        dt_s: float,
        t_s: float,
        coupling_input: float = 0.0,
        neighbour_flash_events: int = 0,
        neighbour_flash_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Advance the agent by one timestep.

        Parameters
        ----------
        dt_s: Measured dt.
        t_s: Current simulation time.
        coupling_input: For Kuramoto — pre-computed Σ sin(θ_j − θ_i).
        neighbour_flash_events: For PCO/EAPF — number of neighbour
            flash events detected this step.

        Returns
        -------
        dict with ``follower_flash_event``, ``phase``, etc.
        """
        if self.model == "kuramoto":
            state = self._oscillator.step(coupling_input)
            is_firing = state.is_firing
            self._phase_history.append(self._oscillator.phase)
            result = {
                "follower_flash_event": is_firing,
                "phase": self._oscillator.phase,
                "frequency_hz": self.frequency_hz,
            }

        elif self.model == "pco_if":
            # Convert neighbour flash count to boolean event
            leader_event = neighbour_flash_events > 0
            r = self._oscillator.step(dt_s=dt_s, leader_flash_event=leader_event, t_s=t_s)
            is_firing = r["follower_flash_event"]
            self._phase_history.append(self._oscillator.phase)
            result = {
                "follower_flash_event": is_firing,
                "phase": self._oscillator.phase,
                "frequency_hz": self.frequency_hz,
                "leader_flash_event_used": r["leader_flash_event_used"],
                "refractory_active": r["refractory_active"],
            }

        elif self.model == "eapf":
            leader_event = neighbour_flash_events > 0
            r = self._oscillator.step(dt_s=dt_s, leader_flash_event=leader_event, t_s=t_s)
            is_firing = r["follower_flash_event"]
            self._phase_history.append(self._oscillator.phase_rad)
            self._freq_history.append(self._oscillator.frequency_hz)
            result = {
                "follower_flash_event": is_firing,
                "phase": self._oscillator.phase_rad,
                "frequency_hz": self._oscillator.frequency_hz,
                "phase_error_rad": r["phase_error_rad"],
                "leader_flash_event_used": r["leader_flash_event_used"],
            }

        elif self.model == "eapf_consensus":
            ids = neighbour_flash_ids or []
            r = self._oscillator.step(dt_s=dt_s, t_s=t_s, neighbour_flash_ids=ids)
            is_firing = r["follower_flash_event"]
            self._phase_history.append(self._oscillator._phase_rad)
            self._freq_history.append(self._oscillator._frequency_hz)
            result = {
                "follower_flash_event": is_firing,
                "phase": self._oscillator._phase_rad,
                "frequency_hz": self._oscillator._frequency_hz,
                "phase_error_rad": r["phase_error_rad"],
            }

        else:
            raise RuntimeError(f"Unknown model: {self.model}")

        if is_firing:
            self._flash_times.append(t_s)

        return result

    def reset(self) -> None:
        self._oscillator.reset()
        self._flash_times.clear()
        self._phase_history.clear()
        self._freq_history.clear()

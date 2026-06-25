"""Pulse-Coupled Integrate-and-Fire (PCO-I&F) Oscillator — Model 2.

Based on Mirollo–Strogatz (1990) pulse-coupled biological oscillator model.
Supports three coupling modes:

* ``proportional_gap``  — phase += ε·(1 − φ) (original gap-proportional rule)
* ``additive_phase``     — phase = min(1.0, phase + ε) (constant additive)
* ``mirollo_state``      — concave state function U(φ), additive in U-space

Pure core model — no camera, no GPIO.  Unit-testable on any platform.

See: ``docs/synchronisation_models/model_2_pco_integrate_fire.md``
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing import Any

# ---------------------------------------------------------------------------
# Mirollo state-curve helpers
# ---------------------------------------------------------------------------

def _U(phi: float, beta: float) -> float:
    """Mirollo concave state function: U(φ) ∈ [0, 1] for φ ∈ [0, 1]."""
    if abs(beta) < 1e-9:
        return phi  # linear limit
    return (1.0 - math.exp(-beta * phi)) / (1.0 - math.exp(-beta))


def _U_inv(x: float, beta: float) -> float:
    """Inverse of U."""
    if abs(beta) < 1e-9:
        return x
    return -math.log(1.0 - x * (1.0 - math.exp(-beta))) / beta


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_COUPLING_MODES = ("proportional_gap", "additive_phase", "mirollo_state",
                    "biphasic_sine", "piecewise_advance_delay")
_PRC_MODES = ("advance_only", "biphasic_sine", "piecewise_advance_delay")


@dataclass
class PulseCoupledIFConfig:
    """Configuration for the PCO-I&F oscillator.

    Attributes:
        natural_frequency_hz: Free-running follower frequency in Hz.
        epsilon: Phase advance per leader pulse (0 < ε < 1 for phase modes;
            additive in U-space for mirollo_state).
        fire_threshold: Phase value at which the oscillator fires.
        refractory_period_s: Post-fire refractory window in seconds.
        coupling_mode: Coupling rule — see module docstring.
        state_curve_beta: Concavity parameter for mirollo_state U(φ).
    """

    natural_frequency_hz: float = 1.5
    epsilon: float = 0.15
    fire_threshold: float = 1.0
    refractory_period_s: float = 0.06
    coupling_mode: str = "proportional_gap"
    state_curve_beta: float = 3.0
    # Adaptive PRC
    prc_mode: str = "advance_only"
    advance_gain: float = 0.25
    delay_gain: float = 0.10
    prc_phase_ref: float = 0.0
    max_phase_correction: float = 0.40
    enable_phase_delay: bool = False
    delay_region_end_phase: float = 0.3
    advance_region_start_phase: float = 0.6
    # Frequency adaptation
    enable_frequency_adaptation: bool = False
    frequency_adaptation_gain: float = 0.01
    frequency_min_hz: float = 0.5
    frequency_max_hz: float = 4.0
    neighbour_period_window: int = 6
    # Safety
    min_inter_flash_interval_s: float = 0.0
    post_flash_lockout_s: float = 0.0

    def __post_init__(self) -> None:
        if self.coupling_mode not in _COUPLING_MODES:
            raise ValueError(
                f"coupling_mode must be one of {_COUPLING_MODES}, "
                f"got '{self.coupling_mode}'"
            )


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------

class PulseCoupledIntegrateFireOscillator:
    """Pulse-coupled integrate-and-fire oscillator (PCO-I&F).

    Loop order (every step):
        1. Compute measured dt.
        2. Advance natural phase: ``phase += f_natural * dt``.
        3. Decrement refractory timer.
        4. If leader flash event and not refractory, apply coupling.
        5. If phase >= fire_threshold → follower flash, reset, start refractory.
        6. Return state dictionary.
    """

    def __init__(self, config: PulseCoupledIFConfig | None = None) -> None:
        self.config = config or PulseCoupledIFConfig()
        self._phase: float = 0.0
        self._refractory_timer_s: float = 0.0
        self._fire_count: int = 0
        self._last_flash_time_s: float | None = None
        self._neighbour_periods: list[float] = []
        self._post_flash_lockout: float = 0.0
        self._last_neighbour_t: float | None = None

    # -- properties --

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def refractory_active(self) -> bool:
        return self._refractory_timer_s > 0.0

    @property
    def fire_count(self) -> int:
        return self._fire_count

    @property
    def last_flash_time_s(self) -> float | None:
        return self._last_flash_time_s

    # -- Mirollo state helpers (convenience) --

    def state_U(self, phi: float | None = None) -> float:
        """Compute U(φ) for the current config beta."""
        return _U(phi if phi is not None else self._phase, self.config.state_curve_beta)

    def state_U_inv(self, x: float) -> float:
        """Compute U⁻¹(x) for the current config beta."""
        return _U_inv(x, self.config.state_curve_beta)

    # -- coupling dispatch --

    def _apply_coupling(self) -> tuple[float, float, float]:
        """Apply the selected coupling mode."""
        mode = self.config.coupling_mode
        eps = self.config.epsilon

        if mode in ("proportional_gap", "additive_phase", "mirollo_state"):
            return self._apply_basic_coupling(mode, eps)

        elif mode == "biphasic_sine":
            return self._apply_prc_biphasic()

        elif mode == "piecewise_advance_delay":
            return self._apply_prc_piecewise()

        raise RuntimeError(f"Unknown coupling_mode: {mode}")

    def _apply_basic_coupling(self, mode: str, eps: float
                              ) -> tuple[float, float, float]:
        svar_before = self._phase
        if mode == "proportional_gap":
            self._phase += eps * (1.0 - self._phase)
        elif mode == "additive_phase":
            self._phase = min(1.0, self._phase + eps)
        elif mode == "mirollo_state":
            beta = self.config.state_curve_beta
            svar_before = _U(self._phase, beta)
            x = min(1.0, svar_before + eps)
            self._phase = _U_inv(x, beta)
            return self._phase, svar_before, x
        return self._phase, svar_before, self._phase

    def _apply_prc_biphasic(self) -> tuple[float, float, float]:
        """Biphasic sine PRC: delta = ε·sin(2π·(φ − φ_ref))."""
        cfg = self.config
        svar_before = self._phase
        delta = cfg.epsilon * math.sin(
            2.0 * math.pi * (self._phase - cfg.prc_phase_ref)
        )
        delta = max(-cfg.max_phase_correction,
                    min(cfg.max_phase_correction, delta))
        if not cfg.enable_phase_delay and delta < 0:
            delta = 0.0
        self._phase = max(0.0, min(1.0, self._phase + delta))
        return self._phase, svar_before, self._phase

    def _apply_prc_piecewise(self) -> tuple[float, float, float]:
        """Piecewise advance/delay PRC based on current phase region."""
        cfg = self.config
        svar_before = self._phase
        if self._phase < cfg.delay_region_end_phase and cfg.enable_phase_delay:
            self._phase -= cfg.delay_gain * self._phase
            self._phase = max(0.0, self._phase)
        elif self._phase > cfg.advance_region_start_phase:
            self._phase += cfg.advance_gain * (1.0 - self._phase)
            self._phase = min(1.0, self._phase)
        return self._phase, svar_before, self._phase

    # -- core step --

    def step(
        self,
        dt_s: float,
        leader_flash_event: bool,
        t_s: float,
    ) -> dict[str, Any]:
        """Advance the oscillator by one timestep.

        Returns
        -------
        dict
            Standard keys plus ``coupling_mode``, ``state_variable_before_pulse``,
            ``state_variable_after_pulse``.
        """
        leader_used = False
        phase_before_coupling = self._phase
        svar_before = self._phase
        svar_after = self._phase

        # Step 2: natural phase evolution
        if dt_s > 0:
            self._phase += self.config.natural_frequency_hz * dt_s

        # Step 3: decay refractory timer
        if self._refractory_timer_s > 0:
            self._refractory_timer_s -= dt_s
            if self._refractory_timer_s < 0:
                self._refractory_timer_s = 0.0

        # Step 4: pulse coupling
        if leader_flash_event and not self.refractory_active:
            _, svar_before, svar_after = self._apply_coupling()
            leader_used = True

        phase_after_coupling = self._phase

        # Step 4b: frequency adaptation (slow)
        self._adapt_frequency()

        # Step 5: threshold check with safety lockout
        follower_flash = False
        lockout = self.config.post_flash_lockout_s
        if self._post_flash_lockout > 0:
            self._post_flash_lockout = max(0.0, self._post_flash_lockout - dt_s)
        if self._phase >= self.config.fire_threshold and self._post_flash_lockout <= 0:
            # min inter-flash interval check
            if (self.config.min_inter_flash_interval_s <= 0
                or self._last_flash_time_s is None
                or (t_s - self._last_flash_time_s) >= self.config.min_inter_flash_interval_s):
                follower_flash = True
                self._fire_count += 1
                self._last_flash_time_s = t_s
                self._phase = 0.0
                self._refractory_timer_s = self.config.refractory_period_s
                if lockout > 0:
                    self._post_flash_lockout = lockout

        return {
            "phase": round(self._phase, 6),
            "follower_flash_event": follower_flash,
            "leader_flash_event_used": leader_used,
            "refractory_active": self.refractory_active,
            "phase_before_coupling": round(phase_before_coupling, 6),
            "phase_after_coupling": round(phase_after_coupling, 6),
            "coupling_mode": self.config.coupling_mode,
            "state_variable_before_pulse": round(svar_before, 6),
            "state_variable_after_pulse": round(svar_after, 6),
            "fire_count": self._fire_count,
        }

    # -- lifecycle --

    def record_neighbour_flash(self, t_s: float) -> None:
        """Track neighbour inter-flash interval for frequency adaptation."""
        if self._last_neighbour_t is not None:
            interval = t_s - self._last_neighbour_t
            if interval > 0:
                self._neighbour_periods.append(interval)
                if len(self._neighbour_periods) > self.config.neighbour_period_window * 2:
                    self._neighbour_periods.pop(0)
        self._last_neighbour_t = t_s

    def _adapt_frequency(self) -> None:
        """Slowly adapt natural frequency toward neighbour median period."""
        if not self.config.enable_frequency_adaptation:
            return
        if len(self._neighbour_periods) < 2:
            return
        recent = self._neighbour_periods[-self.config.neighbour_period_window:]
        if len(recent) < 2:
            return
        target_period = float(np.median(recent)) if hasattr(np, 'median') else sorted(recent)[len(recent)//2]
        if target_period <= 0:
            return
        target_freq = 1.0 / target_period
        current_freq = self.config.natural_frequency_hz
        new_freq = current_freq + self.config.frequency_adaptation_gain * (target_freq - current_freq)
        new_freq = max(self.config.frequency_min_hz, min(self.config.frequency_max_hz, new_freq))
        self.config.natural_frequency_hz = new_freq

    def reset(self) -> None:
        self._phase = 0.0
        self._refractory_timer_s = 0.0
        self._fire_count = 0
        self._last_flash_time_s = None
        self._neighbour_periods.clear()
        self._post_flash_lockout = 0.0
        self._last_neighbour_t = None

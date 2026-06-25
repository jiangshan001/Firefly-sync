"""Kuramoto coupled-oscillator model.

The Kuramoto model describes synchronisation among N coupled phase oscillators.
Each oscillator i evolves according to:

    dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)

where:
  - θᵢ is the phase of oscillator i,
  - ωᵢ is its natural frequency,
  - K is the coupling strength,
  - N is the number of oscillators in the coupling neighbourhood.

References:
  - Kuramoto, Y. (1975). Self-entrainment of a population of coupled
    non-linear oscillators. In Int. Symp. on Mathematical Problems in
    Theoretical Physics.
  - Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset
    of synchronization in populations of coupled oscillators. Physica D.
"""

import numpy as np

from firefly_sync.core.oscillator import Oscillator, OscillatorState


class KuramotoModel(Oscillator):
    """Continuous-phase Kuramoto oscillator.

    Flashes occur when the phase crosses 0 (i.e., wraps from 2π → 0).
    The coupling term is applied as a continuous phase-velocity adjustment.

    Attributes:
        natural_frequency: Intrinsic angular frequency ω (rad/s).
        phase: Current phase θ [0, 2π).
        coupling_strength: Coupling gain K.
        flash_threshold: Phase value at which a flash is emitted (default 2π).
        dt: Integration timestep.
    """

    def __init__(
        self,
        natural_frequency: float,
        initial_phase: float = 0.0,
        coupling_strength: float = 1.0,
        flash_threshold: float = 2.0 * np.pi,
        dt: float = 0.01,
    ) -> None:
        """Initialise the Kuramoto oscillator.

        Args:
            natural_frequency: Intrinsic angular frequency ω in rad/s.
            initial_phase: Starting phase in radians.
            coupling_strength: Coupling gain K.
            flash_threshold: Phase at which the oscillator fires (radians).
                Defaults to 2π (fires once per natural cycle).
            dt: Simulation timestep in seconds.
        """
        super().__init__(
            natural_frequency=natural_frequency,
            initial_phase=initial_phase,
            coupling_strength=coupling_strength,
            dt=dt,
        )
        self.flash_threshold: float = flash_threshold % (2 * np.pi) or (2 * np.pi)
        self._is_firing: bool = False

    def step(self, coupling_input: float = 0.0) -> OscillatorState:
        """Advance the Kuramoto oscillator by one Euler-integration step.

        The phase update follows:
            θ(t + dt) = θ(t) + dt · [ω + K · coupling_input]

        where coupling_input is typically (1/N) · Σⱼ sin(θⱼ − θᵢ) and
        should be computed by the simulation engine.

        Args:
            coupling_input: Pre-computed coupling term from visible neighbours.
                This should already incorporate the 1/N factor.

        Returns:
            OscillatorState after the step.
        """
        # Euler integration of the Kuramoto ODE
        dtheta = self.natural_frequency + self.coupling_strength * coupling_input
        new_phase = self.phase + dtheta * self.dt

        # Detect flash: phase crosses the flash_threshold
        self._is_firing = new_phase >= self.flash_threshold

        if self._is_firing:
            new_phase -= self.flash_threshold  # wrap around
            self._fire_count += 1
            self._time_since_last_fire = 0.0
        else:
            self._time_since_last_fire += self.dt

        self.phase = new_phase % self.flash_threshold
        self._time_elapsed += self.dt

        return self.state

    def reset(self, phase: float | None = None) -> None:
        """Reset the oscillator to a fresh initial state.

        Args:
            phase: New phase in radians. If None, resets to 0.
        """
        self.phase = (phase or 0.0) % self.flash_threshold
        self._fire_count = 0
        self._time_since_last_fire = float("inf")
        self._time_elapsed = 0.0
        self._is_firing = False

    @property
    def period(self) -> float:
        """Natural period of this oscillator in seconds: T = 2π / ω."""
        return (2.0 * np.pi) / self.natural_frequency

    def __repr__(self) -> str:
        return (
            f"KuramotoModel(ω={self.natural_frequency:.2f}, "
            f"θ={self.phase:.2f}, K={self.coupling_strength:.2f})"
        )

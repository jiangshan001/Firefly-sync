"""Pulse-coupled / Integrate-and-Fire oscillator model.

Inspired by the Mirollo–Strogatz model of pulse-coupled oscillators and
the classic integrate-and-fire neuron, each oscillator has a state variable
x(t) that charges toward a threshold. Upon reaching the threshold the
oscillator fires (emits a flash) and resets. Incoming observed flashes
advance the state by a coupling increment, pulling neighbours toward
synchrony.

The state evolves as:
    dx/dt = 1 / T           (linear charging toward threshold)
    if x ≥ 1: fire, then reset x → 0

where T is the natural period. Observed flashes add ε · K to x,
where ε is a coupling increment and K is the coupling strength.

References:
  - Mirollo, R. E. & Strogatz, S. H. (1990). Synchronization of
    pulse-coupled biological oscillators. SIAM J. Appl. Math.
  - Peskin, C. S. (1975). Mathematical Aspects of Heart Physiology.
    Courant Institute.
"""

import numpy as np

from firefly_sync.core.oscillator import Oscillator, OscillatorState


class PulseCoupledModel(Oscillator):
    """Integrate-and-fire oscillator with pulse coupling.

    The state variable x ∈ [0, 1) represents the normalised membrane
    potential. The oscillator charges linearly (or via a configurable
    charging function) until x ≥ 1, at which point it fires and resets.

    Attributes:
        state_variable: Normalised potential x ∈ [0, 1).
        natural_period: Free-running period T in seconds.
        coupling_increment: Fractional advance ε applied when a neighbour
            flash is detected (0 < ε < 1).
        refractory_steps: Number of timesteps to hold after firing before
            resuming charging.
        threshold: Firing threshold (default 1.0).
        dt: Integration timestep.
    """

    def __init__(
        self,
        natural_period: float,
        initial_state: float = 0.0,
        coupling_strength: float = 1.0,
        coupling_increment: float = 0.1,
        refractory_steps: int = 2,
        dt: float = 0.01,
    ) -> None:
        """Initialise the pulse-coupled oscillator.

        Args:
            natural_period: Free-running period T in seconds.
            initial_state: Starting state-variable value x₀ ∈ [0, 1).
            coupling_strength: Overall coupling gain K.
            coupling_increment: Fractional advance ε per detected flash
                (0 < ε < 1). A neighbour flash advances x by ε·K.
            refractory_steps: Number of dt steps to remain at 0 after firing
                before the oscillator resumes charging.
            dt: Simulation timestep in seconds.

        Raises:
            ValueError: If coupling_increment is not in (0, 1) or
                natural_period is not positive.
        """
        # The base Oscillator uses frequency; we convert from period for
        # the interface but store period as our primary parameter.
        frequency = 2.0 * np.pi / natural_period
        super().__init__(
            natural_frequency=frequency,
            initial_phase=initial_state * 2.0 * np.pi,
            coupling_strength=coupling_strength,
            dt=dt,
        )

        if natural_period <= 0:
            raise ValueError("natural_period must be positive.")
        if not 0 < coupling_increment < 1:
            raise ValueError("coupling_increment must be in (0, 1).")

        self.natural_period: float = natural_period
        self.state_variable: float = initial_state
        self.coupling_increment: float = coupling_increment
        self.refractory_steps: int = refractory_steps
        self.threshold: float = 1.0

        # Convert phase-based tracking to state-variable tracking
        self._refractory_counter: int = 0
        self._is_firing: bool = False

        # Override base phase tracking — we use state_variable instead
        self._charge_rate: float = self.threshold / self.natural_period
        # dx/dt in units of threshold per second

    def step(self, coupling_input: float = 0.0) -> OscillatorState:
        """Advance the integrate-and-fire oscillator by one timestep.

        coupling_input represents the number of detected flashes from
        neighbours this timestep. Each detected flash advances x by
        coupling_increment · coupling_strength.

        Args:
            coupling_input: Number of detected neighbour flashes this step.

        Returns:
            OscillatorState after the step (phase is mapped from x).
        """
        self._is_firing = False

        if self._refractory_counter > 0:
            # In refractory: hold at 0, decrement counter
            self._refractory_counter -= 1
            self._time_since_last_fire += self.dt
            self._time_elapsed += self.dt
            self.phase = 0.0
            return self.state

        # Linear charging: x(t+dt) = x(t) + (1/T)·dt
        self.state_variable += self._charge_rate * self.dt

        # Pulse coupling: each detected neighbour flash advances x
        if coupling_input > 0:
            advance = coupling_input * self.coupling_increment * self.coupling_strength
            self.state_variable += advance

        # Check threshold crossing
        if self.state_variable >= self.threshold:
            self._is_firing = True
            self.state_variable = 0.0
            self._refractory_counter = self.refractory_steps
            self._fire_count += 1
            self._time_since_last_fire = 0.0
        else:
            self._time_since_last_fire += self.dt

        # Map state variable back to phase for the OscillatorState interface
        self.phase = self.state_variable * 2.0 * np.pi
        self._time_elapsed += self.dt

        return self.state

    def reset(self, phase: float | None = None) -> None:
        """Reset the oscillator to a fresh initial state.

        Args:
            phase: New phase in radians (mapped to state variable).
                If None, resets to 0.
        """
        if phase is None:
            self.state_variable = 0.0
        else:
            self.state_variable = (phase % (2.0 * np.pi)) / (2.0 * np.pi)
        self.phase = self.state_variable * 2.0 * np.pi
        self._fire_count = 0
        self._time_since_last_fire = float("inf")
        self._time_elapsed = 0.0
        self._refractory_counter = 0
        self._is_firing = False

    @property
    def in_refractory(self) -> bool:
        """Whether the oscillator is currently in its refractory period."""
        return self._refractory_counter > 0

    def __repr__(self) -> str:
        return (
            f"PulseCoupledModel(T={self.natural_period:.3f}s, "
            f"x={self.state_variable:.3f}, ε={self.coupling_increment:.3f})"
        )

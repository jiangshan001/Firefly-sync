"""Abstract base class for all oscillator models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OscillatorState:
    """Snapshot of an oscillator's instantaneous state.

    Attributes:
        phase: Current phase angle in radians [0, 2π).
        frequency: Natural (intrinsic) angular frequency (rad/s).
        is_firing: Whether this oscillator is emitting a flash right now.
        time_since_last_fire: Time elapsed since the last flash (seconds).
    """

    phase: float
    frequency: float
    is_firing: bool = False
    time_since_last_fire: float = 0.0


class Oscillator(ABC):
    """Abstract base class for a synchronisable oscillator.

    Each drone agent owns one Oscillator instance that governs its
    flash timing. Subclasses implement specific mathematical models
    (Kuramoto, integrate-and-fire, etc.).

    Attributes:
        natural_frequency: Intrinsic angular frequency ω (rad/s).
        phase: Current phase angle θ in radians [0, 2π).
        coupling_strength: Global coupling gain K applied to neighbour inputs.
        dt: Integration timestep (seconds).
    """

    def __init__(
        self,
        natural_frequency: float,
        initial_phase: float = 0.0,
        coupling_strength: float = 1.0,
        dt: float = 0.01,
    ) -> None:
        """Initialise the oscillator.

        Args:
            natural_frequency: Intrinsic angular frequency ω in rad/s.
            initial_phase: Starting phase angle in radians.
            coupling_strength: Coupling gain K (≥ 0).
            dt: Simulation timestep in seconds.

        Raises:
            ValueError: If natural_frequency or coupling_strength is negative.
        """
        if natural_frequency < 0:
            raise ValueError("natural_frequency must be non-negative.")
        if coupling_strength < 0:
            raise ValueError("coupling_strength must be non-negative.")

        self.natural_frequency: float = natural_frequency
        self.phase: float = initial_phase % (2 * np.pi)  # wrap to [0, 2π)
        self.coupling_strength: float = coupling_strength
        self.dt: float = dt
        self._fire_count: int = 0
        self._time_since_last_fire: float = float("inf")
        self._time_elapsed: float = 0.0

    @abstractmethod
    def step(self, coupling_input: float = 0.0) -> OscillatorState:
        """Advance the oscillator by one timestep.

        Args:
            coupling_input: Aggregate coupling signal from visible neighbours.
                Interpretation is model-specific (phase influence for Kuramoto,
                charge injection for integrate-and-fire).

        Returns:
            OscillatorState describing the oscillator after the step.
        """
        ...

    @abstractmethod
    def reset(self, phase: float | None = None) -> None:
        """Reset the oscillator to an initial condition.

        Args:
            phase: New phase in radians. If None, resets to 0.
        """
        ...

    @property
    def fire_count(self) -> int:
        """Total number of times this oscillator has fired."""
        return self._fire_count

    @property
    def time_elapsed(self) -> float:
        """Total simulation time elapsed for this oscillator (seconds)."""
        return self._time_elapsed

    @property
    def state(self) -> OscillatorState:
        """Return a snapshot of the current oscillator state."""
        return OscillatorState(
            phase=self.phase,
            frequency=self.natural_frequency,
            is_firing=getattr(self, "_is_firing", False),
            time_since_last_fire=self._time_since_last_fire,
        )


# Deferred import to avoid circular dependency at module level.
import numpy as np

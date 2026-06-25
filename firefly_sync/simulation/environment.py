"""Spatial environment managing inter-drone coupling.

The Environment computes which drones are visible to each other and
calculates distance-weighted coupling signals.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np

from firefly_sync.simulation.drone import Drone


class CouplingMode(Enum):
    """Supported coupling computation strategies."""

    KURAMOTO = "kuramoto"
    PULSE_COUPLED = "pulse_coupled"


class Environment:
    """2D/3D environment managing spatial coupling between drones.

    The environment tracks drone positions and computes the coupling
    signal each drone receives from its visible neighbours. Coupling
    strength decays with distance via a configurable decay function.

    Attributes:
        drones: List of drone agents in the environment.
        dimensions: Number of spatial dimensions (2 or 3).
        decay_function: Distance-decay function g(d) → [0, 1].
        coupling_mode: How coupling inputs are computed for oscillators.
    """

    def __init__(
        self,
        drones: list[Drone] | None = None,
        dimensions: int = 2,
        decay_function: Callable[[float], float] | None = None,
        coupling_mode: CouplingMode = CouplingMode.KURAMOTO,
    ) -> None:
        """Initialise the environment.

        Args:
            drones: Initial list of drone agents.
            dimensions: Spatial dimensions (2 or 3).
            decay_function: Function g(distance) → weight ∈ [0, 1].
                Defaults to exponential decay: g(d) = exp(-d / d₀).
            coupling_mode: Whether to compute Kuramoto-style or
                pulse-coupled-style coupling signals.

        Raises:
            ValueError: If dimensions is not 2 or 3.
        """
        if dimensions not in (2, 3):
            raise ValueError("dimensions must be 2 or 3.")
        self.dimensions: int = dimensions
        self.coupling_mode: CouplingMode = coupling_mode
        self._decay_function: Callable[[float], float] = (
            decay_function or self._default_decay
        )
        self._drones: dict[int, Drone] = {}

        if drones:
            for drone in drones:
                self.add_drone(drone)

    @staticmethod
    def _default_decay(distance: float, length_scale: float = 5.0) -> float:
        """Default exponential distance-decay function.

        g(d) = exp(-d / d₀) where d₀ is the characteristic length scale.

        Args:
            distance: Euclidean distance between drones in metres.
            length_scale: Characteristic decay length d₀.

        Returns:
            Weight ∈ (0, 1] representing relative coupling strength.
        """
        return np.exp(-distance / length_scale)

    def add_drone(self, drone: Drone) -> None:
        """Register a drone in the environment.

        Args:
            drone: The drone agent to add.
        """
        self._drones[drone.drone_id] = drone

    def remove_drone(self, drone_id: int) -> None:
        """Remove a drone from the environment.

        Args:
            drone_id: The identifier of the drone to remove.

        Raises:
            KeyError: If drone_id is not registered.
        """
        del self._drones[drone_id]

    def get_drone(self, drone_id: int) -> Drone:
        """Retrieve a drone by ID.

        Raises:
            KeyError: If drone_id is not registered.
        """
        return self._drones[drone_id]

    @property
    def drones(self) -> list[Drone]:
        """All drones currently in the environment."""
        return list(self._drones.values())

    @property
    def num_drones(self) -> int:
        """Number of drones in the environment."""
        return len(self._drones)

    def distance(self, drone_a: Drone, drone_b: Drone) -> float:
        """Compute Euclidean distance between two drones.

        Args:
            drone_a: First drone.
            drone_b: Second drone.

        Returns:
            Euclidean distance in metres.
        """
        return float(np.linalg.norm(drone_a.position - drone_b.position))

    def _compute_kuramoto_coupling(self, drone: Drone) -> float:
        """Compute the Kuramoto coupling term for a drone.

        coupling = (1/N) · Σⱼ g(d_ij) · sin(θⱼ − θᵢ)

        where N = total number of drones, g(d) is the distance decay.

        Args:
            drone: The drone to compute coupling for.

        Returns:
            Aggregate Kuramoto coupling term.
        """
        if self.num_drones <= 1:
            return 0.0

        coupling_sum = 0.0
        for other in self._drones.values():
            if other.drone_id == drone.drone_id:
                continue
            d = self.distance(drone, other)
            weight = self._decay_function(d)
            phase_diff = other.oscillator.phase - drone.oscillator.phase
            coupling_sum += weight * np.sin(phase_diff)

        return coupling_sum / self.num_drones

    def _compute_pulse_coupling(self, drone: Drone) -> float:
        """Compute the pulse-coupled input for a drone.

        Returns the weighted count of neighbour flashes this timestep.
        Each detected flash from a neighbour at distance d contributes
        g(d) to the coupling input.

        Args:
            drone: The drone to compute coupling for.

        Returns:
            Weighted flash count.
        """
        if self.num_drones <= 1:
            return 0.0

        coupling_sum = 0.0
        for other in self._drones.values():
            if other.drone_id == drone.drone_id:
                continue
            if other.is_firing:
                d = self.distance(drone, other)
                weight = self._decay_function(d)
                coupling_sum += weight

        return coupling_sum

    def compute_coupling(self, drone: Drone) -> float:
        """Compute the coupling input for a single drone.

        Args:
            drone: The drone to compute coupling for.

        Returns:
            Coupling input value (model-specific interpretation).
        """
        if self.coupling_mode == CouplingMode.KURAMOTO:
            return self._compute_kuramoto_coupling(drone)
        elif self.coupling_mode == CouplingMode.PULSE_COUPLED:
            return self._compute_pulse_coupling(drone)
        else:
            raise ValueError(f"Unknown coupling mode: {self.coupling_mode}")

    def compute_all_couplings(self) -> dict[int, float]:
        """Compute coupling inputs for all drones.

        Returns:
            Mapping from drone_id to its coupling input.
        """
        return {
            drone_id: self.compute_coupling(drone)
            for drone_id, drone in self._drones.items()
        }


def inverse_square_decay(distance: float, min_distance: float = 0.1) -> float:
    """Inverse-square distance-decay function for visual coupling.

    g(d) = 1 / (1 + d²) — models light intensity falloff.

    Args:
        distance: Euclidean distance between drones in metres.
        min_distance: Small offset to prevent division by zero.

    Returns:
        Weight ∈ (0, 1] representing coupling strength.
    """
    return 1.0 / (1.0 + max(distance, min_distance) ** 2)


def step_decay(distance: float, max_range: float = 10.0) -> float:
    """Step-function distance decay: 1 if visible, 0 otherwise.

    Args:
        distance: Euclidean distance between drones in metres.
        max_range: Maximum visible range in metres.

    Returns:
        1.0 if distance ≤ max_range, else 0.0.
    """
    return 1.0 if distance <= max_range else 0.0

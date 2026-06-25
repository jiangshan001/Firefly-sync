"""Main simulation engine that steps the system through time."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from firefly_sync.simulation.drone import Drone
from firefly_sync.simulation.environment import Environment, CouplingMode


class SimulationEngine:
    """Orchestrates the multi-drone synchronisation simulation.

    The engine steps all drones forward in time, computes inter-drone
    coupling through the environment, and optionally logs state to an
    experiment logger.

    Attributes:
        environment: The spatial environment managing drones and coupling.
        dt: Simulation timestep in seconds.
        total_steps: Number of steps executed so far.
        running: Whether the simulation is currently active.
    """

    def __init__(
        self,
        environment: Environment,
        dt: float = 0.01,
        logger: Any | None = None,
    ) -> None:
        """Initialise the simulation engine.

        Args:
            environment: Configured Environment with drones registered.
            dt: Simulation timestep in seconds.
            logger: Optional ExperimentLogger for recording state over time.
        """
        self.environment: Environment = environment
        self.dt: float = dt
        self.logger: Any | None = logger
        self.total_steps: int = 0
        self.simulation_time: float = 0.0
        self.running: bool = False

    def step(self) -> dict[int, float]:
        """Advance the simulation by one timestep.

        Order of operations:
        1. Compute coupling inputs for all drones.
        2. Step each drone's oscillator with its coupling input.
        3. Log state if a logger is attached.

        Returns:
            Mapping from drone_id to its new phase.
        """
        # 1. Compute coupling
        couplings = self.environment.compute_all_couplings()

        # 2. Step each drone
        phases = {}
        for drone in self.environment.drones:
            coupling = couplings.get(drone.drone_id, 0.0)
            state = drone.step(coupling)
            phases[drone.drone_id] = state.phase

        # 3. Log
        if self.logger is not None:
            self.logger.log_step(
                step=self.total_steps,
                time=self.simulation_time,
                drones=self.environment.drones,
                couplings=couplings,
            )

        self.total_steps += 1
        self.simulation_time += self.dt

        return phases

    def run(
        self,
        duration: float,
        progress_interval: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run the simulation for a specified duration.

        Args:
            duration: Simulation duration in seconds.
            progress_interval: If set, print a progress line every N seconds
                of simulation time.

        Returns:
            List of per-step state dictionaries (if logger attached, these
            are the logged records; otherwise, an empty list).
        """
        num_steps = int(duration / self.dt)
        self.running = True
        t_start = time.perf_counter()

        for _ in range(num_steps):
            if not self.running:
                break
            self.step()

            if progress_interval and self.simulation_time % progress_interval < self.dt:
                self._print_progress(duration)

        elapsed = time.perf_counter() - t_start
        self.running = False

        print(
            f"Simulation complete: {self.simulation_time:.2f}s simulated "
            f"in {elapsed:.2f}s wall-clock ({num_steps} steps)."
        )

        return self.logger.records if self.logger else []

    def stop(self) -> None:
        """Signal the simulation to stop early."""
        self.running = False

    def reset(self) -> None:
        """Reset the simulation state to time zero."""
        self.total_steps = 0
        self.simulation_time = 0.0
        for drone in self.environment.drones:
            drone.oscillator.reset()

    def _print_progress(self, total_duration: float) -> None:
        """Print a progress indicator."""
        pct = 100 * self.simulation_time / total_duration
        print(
            f"  [{self.simulation_time:6.1f}s / {total_duration:.1f}s] "
            f"{pct:5.1f}% complete"
        )

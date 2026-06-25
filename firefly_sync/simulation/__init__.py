"""Simulation layer for multi-drone firefly synchronisation.

This sub-package provides:
  - Drone: an agent with position, oscillator, LED, and camera.
  - Environment: spatial context managing coupling between drones.
  - Engine: main simulation loop.
"""

from firefly_sync.simulation.drone import Drone
from firefly_sync.simulation.environment import Environment
from firefly_sync.simulation.engine import SimulationEngine

__all__ = ["Drone", "Environment", "SimulationEngine"]

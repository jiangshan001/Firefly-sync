"""Configuration management for simulations and experiments.

Supports YAML config files, dataclass-based defaults, and command-line
argument overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore


@dataclass
class OscillatorConfig:
    """Configuration for a single oscillator model."""

    model: str = "kuramoto"  # "kuramoto" or "pulse_coupled"
    natural_frequency: float = 1.0  # rad/s
    initial_phase_min: float = 0.0  # random initial phase lower bound
    initial_phase_max: float = 6.283185  # random initial phase upper bound (≈2π)
    coupling_strength: float = 1.0
    # Pulse-coupled specific
    coupling_increment: float = 0.1
    refractory_steps: int = 2


@dataclass
class EnvironmentConfig:
    """Configuration for the spatial environment."""

    dimensions: int = 2
    decay_function: str = "exponential"  # "exponential", "inverse_square", "step"
    decay_length_scale: float = 5.0  # metres
    max_detection_range: float = 50.0  # metres
    detection_probability: float = 1.0


@dataclass
class SimulationConfig:
    """Top-level simulation configuration.

    All parameters can be loaded from a YAML file and overridden
    via command-line arguments.

    Attributes:
        num_drones: Number of drone agents (2–3 typical).
        duration: Simulation duration in seconds.
        dt: Integration timestep in seconds.
        seed: Random seed for reproducibility (None = no seeding).
        oscillators: Per-drone oscillator configs (auto-generated if empty).
        environment: Spatial environment configuration.
        log_enabled: Whether to record state to disk.
        log_format: Output format for logs ('csv' or 'json').
        log_dir: Directory for experiment logs.
        drone_positions: Fixed positions for each drone [(x, y), ...].
            If empty, positions are generated randomly.
    """

    num_drones: int = 2
    duration: float = 60.0
    dt: float = 0.01
    seed: int | None = 42
    oscillators: list[OscillatorConfig] = field(default_factory=list)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    log_enabled: bool = False
    log_format: str = "csv"
    log_dir: str = "experiments/logs"
    drone_positions: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Generate default oscillator configs if none were provided."""
        if not self.oscillators:
            for i in range(self.num_drones):
                freq = 1.0 + 0.05 * i  # slight frequency spread
                self.oscillators.append(
                    OscillatorConfig(
                        natural_frequency=freq,
                        initial_phase_min=0.0,
                        initial_phase_max=6.283185,
                    )
                )

        if not self.drone_positions:
            import numpy as np
            rng = np.random.default_rng(self.seed)
            self.drone_positions = [
                tuple(rng.uniform(0, 10, size=2).tolist())
                for _ in range(self.num_drones)
            ]


def load_config(filepath: str | Path) -> SimulationConfig:
    """Load a SimulationConfig from a YAML file.

    The YAML file should mirror the SimulationConfig dataclass structure:

        num_drones: 3
        duration: 120.0
        dt: 0.005
        seed: 42
        environment:
          dimensions: 2
          decay_function: exponential
          decay_length_scale: 5.0
        oscillators:
          - model: kuramoto
            natural_frequency: 1.0
            coupling_strength: 0.8
          - model: kuramoto
            natural_frequency: 1.05
            coupling_strength: 0.8
        log_enabled: true
        log_format: csv

    Args:
        filepath: Path to the YAML configuration file.

    Returns:
        SimulationConfig populated from the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    # Parse environment config
    env_raw = raw.get("environment", {})
    env_config = EnvironmentConfig(
        dimensions=env_raw.get("dimensions", 2),
        decay_function=env_raw.get("decay_function", "exponential"),
        decay_length_scale=env_raw.get("decay_length_scale", 5.0),
        max_detection_range=env_raw.get("max_detection_range", 50.0),
        detection_probability=env_raw.get("detection_probability", 1.0),
    )

    # Parse oscillator configs
    osc_configs = []
    for osc_raw in raw.get("oscillators", []):
        osc_configs.append(
            OscillatorConfig(
                model=osc_raw.get("model", "kuramoto"),
                natural_frequency=osc_raw.get("natural_frequency", 1.0),
                initial_phase_min=osc_raw.get("initial_phase_min", 0.0),
                initial_phase_max=osc_raw.get("initial_phase_max", 6.283185),
                coupling_strength=osc_raw.get("coupling_strength", 1.0),
                coupling_increment=osc_raw.get("coupling_increment", 0.1),
                refractory_steps=osc_raw.get("refractory_steps", 2),
            )
        )

    # Parse positions
    positions = [
        tuple(p) for p in raw.get("drone_positions", [])
    ]

    return SimulationConfig(
        num_drones=raw.get("num_drones", 2),
        duration=raw.get("duration", 60.0),
        dt=raw.get("dt", 0.01),
        seed=raw.get("seed", 42),
        oscillators=osc_configs,
        environment=env_config,
        log_enabled=raw.get("log_enabled", False),
        log_format=raw.get("log_format", "csv"),
        log_dir=raw.get("log_dir", "experiments/logs"),
        drone_positions=positions,
    )

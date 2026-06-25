"""Experiment runner: sets up and executes synchronisation simulations.

Can be invoked as a module:
    python -m firefly_sync.experiments.runner --model kuramoto --drones 2

Or imported and called programmatically:
    from firefly_sync.experiments.runner import run_experiment
    metrics = run_experiment(num_drones=3, model="pulse_coupled", duration=120.0)
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from firefly_sync.core.kuramoto import KuramotoModel
from firefly_sync.core.pulse_coupled import PulseCoupledModel
from firefly_sync.hardware.led import MockLED
from firefly_sync.hardware.camera import MockCamera
from firefly_sync.logging.logger import ExperimentLogger
from firefly_sync.logging.metrics import SynchronizationMetrics
from firefly_sync.simulation.drone import Drone
from firefly_sync.simulation.environment import (
    Environment,
    CouplingMode,
)
from firefly_sync.simulation.engine import SimulationEngine


def run_experiment(
    num_drones: int = 2,
    model: str = "kuramoto",
    duration: float = 60.0,
    dt: float = 0.01,
    coupling_strength: float = 1.0,
    frequency_spread: float = 0.05,
    log_enabled: bool = False,
    seed: int | None = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Set up and run a synchronisation experiment.

    Args:
        num_drones: Number of drone agents (2 or 3).
        model: Oscillator model — 'kuramoto' or 'pulse_coupled'.
        duration: Simulation duration in seconds.
        dt: Integration timestep in seconds.
        coupling_strength: Global coupling gain K.
        frequency_spread: Spread of natural frequencies across drones.
            Drone i gets ω = ω_base · (1 + spread · i).
        log_enabled: Whether to record state to CSV/JSON.
        seed: Random seed for reproducibility.
        **kwargs: Additional model-specific parameters.

    Returns:
        Dictionary with simulation results, metrics, and log file path.

    Raises:
        ValueError: If model is not recognised or num_drones < 2.
    """
    if num_drones < 2:
        raise ValueError("num_drones must be at least 2.")
    if model not in ("kuramoto", "pulse_coupled"):
        raise ValueError(f"Unknown model: {model}. Use 'kuramoto' or 'pulse_coupled'.")

    rng = np.random.default_rng(seed)

    # --- Set up oscillator models ---
    base_freq = 1.0  # rad/s base natural frequency
    drones: list[Drone] = []

    for i in range(num_drones):
        freq = base_freq * (1.0 + frequency_spread * i)
        initial_phase = rng.uniform(0, 2.0 * np.pi)

        if model == "kuramoto":
            oscillator = KuramotoModel(
                natural_frequency=freq,
                initial_phase=initial_phase,
                coupling_strength=coupling_strength,
                dt=dt,
            )
        else:
            # Pulse-coupled model uses period, not frequency
            period = (2.0 * np.pi) / freq
            oscillator = PulseCoupledModel(
                natural_period=period,
                initial_state=initial_phase / (2.0 * np.pi),
                coupling_strength=coupling_strength,
                coupling_increment=kwargs.get("coupling_increment", 0.1),
                refractory_steps=kwargs.get("refractory_steps", 2),
                dt=dt,
            )

        # Create mock hardware for each drone
        led = MockLED(led_id=i)
        camera = MockCamera(
            camera_id=i,
            max_range=kwargs.get("max_range", 50.0),
            detection_probability=kwargs.get("detection_probability", 1.0),
            rng=rng,
        )

        # Generate random 2D position
        position = tuple(rng.uniform(0, 10, size=2).tolist())

        drone = Drone(
            drone_id=i,
            oscillator=oscillator,
            led=led,
            camera=camera,
            position=position,
        )
        drones.append(drone)

    # --- Set up environment ---
    coupling_mode = (
        CouplingMode.KURAMOTO if model == "kuramoto"
        else CouplingMode.PULSE_COUPLED
    )
    environment = Environment(
        drones=drones,
        dimensions=2,
        coupling_mode=coupling_mode,
    )

    # --- Set up logging ---
    logger = None
    if log_enabled:
        logger = ExperimentLogger(
            output_dir=kwargs.get("log_dir", "experiments/logs"),
            format=kwargs.get("log_format", "csv"),
        )

    # --- Run simulation ---
    engine = SimulationEngine(
        environment=environment,
        dt=dt,
        logger=logger,
    )
    engine.run(duration=duration, progress_interval=10.0)

    # --- Compute metrics ---
    if logger and logger.records:
        phase_history = _extract_phase_history(logger.records, num_drones)
    else:
        # TODO: build phase_history from engine state if no logger
        phase_history = np.zeros((engine.total_steps, num_drones))

    metrics = SynchronizationMetrics.summary(phase_history, timestep=dt)

    # --- Save and report ---
    log_path = logger.save() if logger else None

    results = {
        "model": model,
        "num_drones": num_drones,
        "duration": duration,
        "dt": dt,
        "coupling_strength": coupling_strength,
        "frequency_spread": frequency_spread,
        "total_steps": engine.total_steps,
        "metrics": metrics,
        "log_path": str(log_path) if log_path else None,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Model:              {model}")
    print(f"  Drones:             {num_drones}")
    print(f"  Duration:           {duration}s (simulated)")
    print(f"  Steps:              {engine.total_steps}")
    print(f"  Final order param:  {metrics['final_order_parameter']:.4f}")
    if metrics["time_to_sync_seconds"] is not None:
        print(f"  Time to sync:       {metrics['time_to_sync_seconds']:.2f}s")
    else:
        print(f"  Time to sync:       did not sync")
    if log_path:
        print(f"  Log:                {log_path}")
    print("=" * 60 + "\n")

    return results


def _extract_phase_history(
    records: list[dict[str, Any]],
    num_drones: int,
) -> np.ndarray:
    """Extract a (T, N) phase history array from logged records.

    Args:
        records: List of per-step dictionaries from ExperimentLogger.
        num_drones: Number of drone agents.

    Returns:
        Array of shape (len(records), num_drones).
    """
    t = len(records)
    phase_history = np.zeros((t, num_drones))
    for step_idx, record in enumerate(records):
        for drone_id in range(num_drones):
            key = f"drone_{drone_id}_phase"
            phase_history[step_idx, drone_id] = record.get(key, 0.0)
    return phase_history


def main() -> None:
    """Command-line entry point for the experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run a firefly-inspired drone synchronisation experiment.",
    )
    parser.add_argument(
        "--model", type=str, default="kuramoto",
        choices=["kuramoto", "pulse_coupled"],
        help="Oscillator model (default: kuramoto).",
    )
    parser.add_argument(
        "--drones", type=int, default=2,
        help="Number of drone agents (default: 2).",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Simulation duration in seconds (default: 60).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Integration timestep (default: 0.01).",
    )
    parser.add_argument(
        "--coupling", type=float, default=1.0,
        help="Coupling strength K (default: 1.0).",
    )
    parser.add_argument(
        "--spread", type=float, default=0.05,
        help="Natural frequency spread across drones (default: 0.05).",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Enable experiment logging to CSV.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    run_experiment(
        num_drones=args.drones,
        model=args.model,
        duration=args.duration,
        dt=args.dt,
        coupling_strength=args.coupling,
        frequency_spread=args.spread,
        log_enabled=args.log,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

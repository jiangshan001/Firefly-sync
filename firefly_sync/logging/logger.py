"""Experiment logger for recording simulation state over time."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from firefly_sync.simulation.drone import Drone


class ExperimentLogger:
    """Records simulation state at each timestep for later analysis.

    Supports CSV (default) and JSON output formats. Each record contains
    the step number, simulation time, per-drone phase, firing state,
    and coupling inputs.

    Attributes:
        records: List of per-step state dictionaries.
        output_dir: Directory where log files are written.
        format: Output format ('csv' or 'json').
    """

    def __init__(
        self,
        output_dir: str | Path = "experiments/logs",
        format: str = "csv",
    ) -> None:
        """Initialise the experiment logger.

        Args:
            output_dir: Directory to write log files.
            format: Output format — 'csv' or 'json'.

        Raises:
            ValueError: If format is not 'csv' or 'json'.
        """
        if format not in ("csv", "json"):
            raise ValueError(f"Unsupported log format: {format}")

        self.output_dir: Path = Path(output_dir)
        self.format: str = format
        self.records: list[dict[str, Any]] = []
        self._run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs(self.output_dir, exist_ok=True)

    def log_step(
        self,
        step: int,
        time: float,
        drones: list[Drone],
        couplings: dict[int, float],
    ) -> None:
        """Record the state of the simulation at one timestep.

        Args:
            step: Current step number.
            time: Simulation time in seconds.
            drones: List of all drone agents.
            couplings: Mapping from drone_id to coupling input.
        """
        record: dict[str, Any] = {
            "step": step,
            "time": round(time, 6),
        }

        for drone in drones:
            prefix = f"drone_{drone.drone_id}"
            record[f"{prefix}_phase"] = round(drone.phase, 6)
            record[f"{prefix}_is_firing"] = int(drone.is_firing)
            record[f"{prefix}_coupling"] = round(
                couplings.get(drone.drone_id, 0.0), 6
            )
            record[f"{prefix}_pos_x"] = round(float(drone.position[0]), 4)
            record[f"{prefix}_pos_y"] = round(float(drone.position[1]), 4)
            if len(drone.position) > 2:
                record[f"{prefix}_pos_z"] = round(float(drone.position[2]), 4)

        self.records.append(record)

    def save(self, filename: str | None = None) -> Path:
        """Write all logged records to disk.

        Args:
            filename: Output filename (without extension). Defaults to
                'experiment_{run_id}'.

        Returns:
            Path to the saved file.
        """
        if filename is None:
            filename = f"experiment_{self._run_id}"

        if self.format == "csv":
            return self._save_csv(filename)
        else:
            return self._save_json(filename)

    def _save_csv(self, filename: str) -> Path:
        """Write records as a CSV file."""
        filepath = self.output_dir / f"{filename}.csv"
        if not self.records:
            return filepath

        fieldnames = list(self.records[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[ExperimentLogger] Saved {len(self.records)} records → {filepath}")
        return filepath

    def _save_json(self, filename: str) -> Path:
        """Write records as a JSON file."""
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump(
                {
                    "run_id": self._run_id,
                    "num_steps": len(self.records),
                    "records": self.records,
                },
                f,
                indent=2,
            )

        print(f"[ExperimentLogger] Saved {len(self.records)} records → {filepath}")
        return filepath

    def clear(self) -> None:
        """Clear the in-memory record buffer."""
        self.records.clear()

    @property
    def run_id(self) -> str:
        """Unique identifier for this experiment run."""
        return self._run_id

"""Simulated drone agent with oscillator, LED, and camera."""

from __future__ import annotations

import numpy as np

from firefly_sync.core.oscillator import Oscillator, OscillatorState
from firefly_sync.hardware.led import AbstractLED
from firefly_sync.hardware.camera import AbstractCamera


class Drone:
    """A simulated drone agent that participates in visual synchronisation.

    Each drone has:
      - A 2D/3D position in space.
      - An oscillator (Kuramoto, integrate-and-fire, etc.) that determines
        when it flashes.
      - An LED that emits flashes (hardware abstraction).
      - A camera that detects flashes from other drones (hardware abstraction).

    Attributes:
        drone_id: Unique identifier for this drone.
        position: Current (x, y) or (x, y, z) position in metres.
        oscillator: The oscillator model driving this drone's flash timing.
        led: Hardware LED interface (mock or real).
        camera: Hardware camera/flash-detection interface (mock or real).
    """

    def __init__(
        self,
        drone_id: int,
        oscillator: Oscillator,
        led: AbstractLED,
        camera: AbstractCamera,
        position: tuple[float, ...] = (0.0, 0.0),
    ) -> None:
        """Initialise a drone agent.

        Args:
            drone_id: Unique integer identifier.
            oscillator: The synchronisation oscillator model.
            led: LED output interface (e.g., MockLED).
            camera: Flash detection interface (e.g., MockCamera).
            position: Initial position coordinates (x, y) or (x, y, z).
        """
        self.drone_id: int = drone_id
        self.oscillator: Oscillator = oscillator
        self.led: AbstractLED = led
        self.camera: AbstractCamera = camera
        self.position: np.ndarray = np.array(position, dtype=float)
        self._state: OscillatorState = oscillator.state

    def step(self, coupling_input: float = 0.0) -> OscillatorState:
        """Advance the drone by one simulation timestep.

        Steps the oscillator, then triggers the LED if the oscillator fired.

        Args:
            coupling_input: Aggregate coupling signal from visible neighbours.

        Returns:
            OscillatorState after the step.
        """
        self._state = self.oscillator.step(coupling_input)

        if self._state.is_firing:
            self.led.flash()

        return self._state

    def detect_flashes(self, other_drones: list[Drone]) -> int:
        """Use this drone's camera to count flashes from other drones.

        In mock mode, the camera queries each other drone's LED to check
        if it is currently emitting. In real mode, this would process a
        camera frame.

        Args:
            other_drones: List of other drone agents to observe.

        Returns:
            Number of detected flashes from visible neighbours.
        """
        # Build a list of (position, is_flashing) for the camera
        neighbours = [
            (d.position, d.led.is_flashing)
            for d in other_drones
            if d.drone_id != self.drone_id
        ]
        return self.camera.detect_flashes(self.position, neighbours)

    @property
    def state(self) -> OscillatorState:
        """Current oscillator state of this drone."""
        return self._state

    @property
    def is_firing(self) -> bool:
        """Whether this drone's oscillator is currently emitting a flash."""
        return self._state.is_firing

    @property
    def phase(self) -> float:
        """Current oscillator phase in radians."""
        return self._state.phase

    def __repr__(self) -> str:
        pos = tuple(round(p, 2) for p in self.position)
        return (
            f"Drone(id={self.drone_id}, pos={pos}, "
            f"θ={self.phase:.2f}, firing={self.is_firing})"
        )

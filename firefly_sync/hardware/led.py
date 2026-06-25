"""LED hardware abstraction — mock and future GPIO implementations."""

from abc import ABC, abstractmethod
import time


class AbstractLED(ABC):
    """Abstract interface for an LED on a drone.

    Subclasses implement the actual hardware control (e.g., GPIO on a
    Raspberry Pi) or mock behaviour for simulation.
    """

    @abstractmethod
    def flash(self) -> None:
        """Emit a single flash.

        In hardware mode, this turns the LED on briefly; in mock mode
        it records the flash event.
        """
        ...

    @abstractmethod
    def is_flashing(self) -> bool:
        """Check whether the LED is currently emitting light.

        Returns:
            True if the LED is actively flashing.
        """
        ...

    @abstractmethod
    def turn_off(self) -> None:
        """Force the LED off."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset LED state to off with no flash history."""
        ...


class MockLED(AbstractLED):
    """Mock LED that logs flash events to console and an internal buffer.

    Each flash persists for a configurable number of timesteps to simulate
    a realistic flash duration (e.g., 50–100 ms flash).

    Attributes:
        flash_duration: How many timesteps a flash remains visible.
        flash_history: List of (timestamp, duration) tuples of past flashes.
    """

    def __init__(self, led_id: int, flash_duration_steps: int = 3) -> None:
        """Initialise the mock LED.

        Args:
            led_id: Identifier matching the parent drone's drone_id.
            flash_duration_steps: Number of timesteps each flash remains
                active (simulates physical flash duration).
        """
        self.led_id: int = led_id
        self.flash_duration_steps: int = flash_duration_steps
        self._flash_counter: int = 0
        self.flash_history: list[tuple[float, int]] = []

    def flash(self) -> None:
        """Trigger a flash. Prints to console and records to history."""
        self._flash_counter = self.flash_duration_steps
        timestamp = time.time()
        self.flash_history.append((timestamp, self.flash_duration_steps))
        # TODO: replace print with structured logging callback
        print(f"[MockLED {self.led_id}] FLASH!")

    def is_flashing(self) -> bool:
        """Check if the flash is still active.

        Decrements the internal counter each call, so each flash is
        visible for `flash_duration_steps` calls.

        Returns:
            True if the LED is currently in a flash period.
        """
        if self._flash_counter > 0:
            self._flash_counter -= 1
            return True
        return False

    def turn_off(self) -> None:
        """Clear any active flash."""
        self._flash_counter = 0

    def reset(self) -> None:
        """Reset LED state and clear history."""
        self._flash_counter = 0
        self.flash_history.clear()

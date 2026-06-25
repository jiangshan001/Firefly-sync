"""Flight controller abstraction — stub for future MAVLink integration.

This module provides an abstract interface for communicating with a
drone flight controller (e.g., Pixhawk running ArduCopter or PX4).
The mock implementation is a no-op for pure simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Telemetry:
    """Telemetry data from a drone flight controller.

    Attributes:
        latitude: Latitude in degrees.
        longitude: Longitude in degrees.
        altitude: Altitude above takeoff in metres.
        heading: Yaw angle in degrees [0, 360).
        battery_voltage: Battery voltage in volts.
        armed: Whether the motors are armed.
        mode: Current flight mode string (e.g., 'GUIDED', 'LOITER').
    """

    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    heading: float = 0.0
    battery_voltage: float = 0.0
    armed: bool = False
    mode: str = "STABILIZE"


class AbstractFlightController(ABC):
    """Abstract interface to a drone flight controller via MAVLink.

    This will wrap pymavlink in the hardware implementation.
    """

    @abstractmethod
    def connect(self, connection_string: str) -> bool:
        """Establish a connection to the flight controller.

        Args:
            connection_string: e.g., 'tcp:127.0.0.1:5760' for SITL,
                '/dev/ttyAMA0' for serial, 'udp:127.0.0.1:14550' for UDP.

        Returns:
            True if connection succeeded.
        """
        ...

    @abstractmethod
    def get_telemetry(self) -> Telemetry:
        """Fetch the latest telemetry from the flight controller.

        Returns:
            Telemetry dataclass with current flight data.
        """
        ...

    @abstractmethod
    def send_led_command(self, on: bool) -> None:
        """Send a command to toggle an LED connected to the flight controller.

        Args:
            on: True to turn the LED on, False to turn it off.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection to the flight controller."""
        ...


class MockFlightController(AbstractFlightController):
    """No-op mock flight controller for pure software simulation.

    All methods are stubs that return safe defaults.
    """

    def connect(self, connection_string: str) -> bool:
        """Mock connection — always succeeds."""
        print(f"[MockFlightController] Connected (mock): {connection_string}")
        return True

    def get_telemetry(self) -> Telemetry:
        """Return default telemetry."""
        return Telemetry()

    def send_led_command(self, on: bool) -> None:
        """Log the LED command to console."""
        state = "ON" if on else "OFF"
        print(f"[MockFlightController] LED → {state}")

    def disconnect(self) -> None:
        """Mock disconnection."""
        print("[MockFlightController] Disconnected.")

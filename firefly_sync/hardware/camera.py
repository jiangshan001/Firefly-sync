"""Camera / flash-detection hardware abstraction."""

from abc import ABC, abstractmethod

import numpy as np


class AbstractCamera(ABC):
    """Abstract interface for flash detection on a drone.

    In hardware mode, this would process camera frames to detect LED
    flashes from other drones. In mock mode, it queries the MockLED
    state of neighbouring simulated drones directly.
    """

    @abstractmethod
    def detect_flashes(
        self,
        own_position: np.ndarray,
        neighbours: list[tuple[np.ndarray, bool]],
    ) -> int:
        """Detect flashes from neighbouring drones.

        Args:
            own_position: (x, y) or (x, y, z) position of this drone.
            neighbours: List of (position, is_flashing) tuples for each
                other drone within the environment.

        Returns:
            Number of detected flashes from visible neighbours.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal camera state."""
        ...


class MockCamera(AbstractCamera):
    """Mock camera that directly reads neighbour LED states.

    In simulation mode, flash detection is perfect within a configurable
    maximum detection range. No image processing is performed.

    Attributes:
        max_range: Maximum detection range in metres. Neighbours beyond
            this distance are ignored.
        detection_probability: Probability of successfully detecting a
            flash from a neighbour within range (simulates occlusion /
            orientation noise).
    """

    def __init__(
        self,
        camera_id: int,
        max_range: float = 50.0,
        detection_probability: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise the mock camera.

        Args:
            camera_id: Identifier matching the parent drone's drone_id.
            max_range: Maximum detection range in metres.
            detection_probability: P(detect | neighbour within range).
                Default 1.0 = perfect detection. Set < 1.0 to add noise.
            rng: NumPy random generator for detection noise.
        """
        self.camera_id: int = camera_id
        self.max_range: float = max_range
        self.detection_probability: float = detection_probability
        self._rng: np.random.Generator = rng or np.random.default_rng()

    def detect_flashes(
        self,
        own_position: np.ndarray,
        neighbours: list[tuple[np.ndarray, bool]],
    ) -> int:
        """Count flashing neighbours within visible range.

        For each neighbour:
          1. Compute distance to own_position.
          2. If within max_range AND the neighbour is flashing:
             detect with probability detection_probability.

        Args:
            own_position: (x, y) or (x, y, z) of this drone.
            neighbours: List of (position, is_flashing) for other drones.

        Returns:
            Number of detected flashes.
        """
        detected_count = 0
        for pos, is_flashing in neighbours:
            if not is_flashing:
                continue
            distance = float(np.linalg.norm(pos - own_position))
            if distance <= self.max_range:
                if self._rng.random() < self.detection_probability:
                    detected_count += 1
        return detected_count

    def reset(self) -> None:
        """Reset the random generator state (for reproducibility)."""
        # No-op: generator state is managed externally via seed
        pass

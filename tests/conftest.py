"""Shared test fixtures and configuration."""

import numpy as np
import pytest


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    """A seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_positions() -> list[tuple[float, float]]:
    """Sample 2D positions for three drones."""
    return [(0.0, 0.0), (3.0, 4.0), (8.0, 1.0)]

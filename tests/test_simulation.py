"""Integration tests for the full simulation pipeline."""

import numpy as np
import pytest

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


class TestDrone:
    """Tests for the Drone class."""

    def test_drone_creation(self) -> None:
        """Drone should initialise with all components."""
        osc = KuramotoModel(natural_frequency=1.0)
        led = MockLED(led_id=0)
        camera = MockCamera(camera_id=0)
        drone = Drone(
            drone_id=0, oscillator=osc, led=led,
            camera=camera, position=(1.0, 2.0),
        )
        assert drone.drone_id == 0
        assert drone.position.tolist() == [1.0, 2.0]
        assert not drone.is_firing

    def test_drone_step_triggers_led(self) -> None:
        """When oscillator fires, LED should be activated."""
        osc = KuramotoModel(
            natural_frequency=2.0 * np.pi,
            initial_phase=2.0 * np.pi - 0.01,
            dt=0.02,
        )
        led = MockLED(led_id=0)
        camera = MockCamera(camera_id=0)
        drone = Drone(drone_id=0, oscillator=osc, led=led, camera=camera)

        state = drone.step(coupling_input=0.0)
        assert state.is_firing
        assert led.flash_history  # Flash was recorded
        assert led.is_flashing()


class TestEnvironment:
    """Tests for the Environment class."""

    def _make_drones(self, n: int = 2) -> list[Drone]:
        """Create a list of test drones with distinct positions."""
        drones = []
        for i in range(n):
            osc = KuramotoModel(natural_frequency=1.0 + 0.1 * i)
            led = MockLED(led_id=i)
            camera = MockCamera(camera_id=i)
            drone = Drone(
                drone_id=i, oscillator=osc, led=led,
                camera=camera, position=(float(i * 5), 0.0),
            )
            drones.append(drone)
        return drones

    def test_environment_creation(self) -> None:
        """Environment should register drones correctly."""
        drones = self._make_drones(3)
        env = Environment(drones=drones, dimensions=2)
        assert env.num_drones == 3
        assert env.get_drone(0).drone_id == 0

    def test_distance_calculation(self) -> None:
        """Euclidean distance between drones should be correct."""
        drones = self._make_drones(2)
        env = Environment(drones=drones)
        d = env.distance(drones[0], drones[1])
        assert d == pytest.approx(5.0)

    def test_kuramoto_coupling_zero_for_one_drone(self) -> None:
        """Single drone should receive zero coupling."""
        drones = self._make_drones(1)
        env = Environment(drones=drones, coupling_mode=CouplingMode.KURAMOTO)
        coupling = env.compute_coupling(drones[0])
        assert coupling == 0.0

    def test_remove_drone(self) -> None:
        """Removing a drone should work correctly."""
        drones = self._make_drones(2)
        env = Environment(drones=drones)
        env.remove_drone(0)
        assert env.num_drones == 1
        with pytest.raises(KeyError):
            env.get_drone(0)


class TestSimulationEngine:
    """Integration tests for the SimulationEngine."""

    def _make_engine(self, n: int = 2, model: str = "kuramoto") -> SimulationEngine:
        """Create a test engine with n drones."""
        drones = []
        for i in range(n):
            if model == "kuramoto":
                osc = KuramotoModel(natural_frequency=1.0 + 0.05 * i, dt=0.01)
            else:
                osc = PulseCoupledModel(
                    natural_period=(2.0 * np.pi) / (1.0 + 0.05 * i),
                    dt=0.01,
                )
            led = MockLED(led_id=i)
            camera = MockCamera(camera_id=i)
            drone = Drone(
                drone_id=i, oscillator=osc, led=led,
                camera=camera, position=(float(i * 3), 0.0),
            )
            drones.append(drone)

        coupling_mode = (
            CouplingMode.KURAMOTO if model == "kuramoto"
            else CouplingMode.PULSE_COUPLED
        )
        env = Environment(drones=drones, coupling_mode=coupling_mode)
        return SimulationEngine(environment=env, dt=0.01)

    def test_engine_steps_all_drones(self) -> None:
        """Each step should advance all drones."""
        engine = self._make_engine(n=2)
        phases = engine.step()
        assert len(phases) == 2
        assert engine.total_steps == 1

    def test_engine_run_completes(self) -> None:
        """Engine should complete a short run without errors."""
        engine = self._make_engine(n=2)
        records = engine.run(duration=0.5)  # 0.5s simulation
        assert engine.simulation_time == pytest.approx(0.5, abs=0.02)
        assert not engine.running

    def test_engine_reset(self) -> None:
        """Reset should restore initial state."""
        engine = self._make_engine(n=2)
        engine.run(duration=0.2)
        engine.reset()
        assert engine.total_steps == 0
        assert engine.simulation_time == 0.0

    def test_engine_with_logger(self) -> None:
        """Engine with logger should produce records."""
        engine = self._make_engine(n=2)
        engine.logger = ExperimentLogger(
            output_dir="experiments/logs",
            format="csv",
        )
        engine.run(duration=0.5)
        assert len(engine.logger.records) > 0


class TestSynchronizationMetrics:
    """Tests for the SynchronizationMetrics class."""

    def test_order_parameter_synchronised(self) -> None:
        """r should be 1.0 when all phases are identical."""
        r = SynchronizationMetrics.order_parameter([0.0, 0.0, 0.0])
        assert r == pytest.approx(1.0)

    def test_order_parameter_incoherent(self) -> None:
        """r should be near 0 for uniformly distributed phases."""
        # Three phases evenly spaced: 0, 2π/3, 4π/3
        phases = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
        r = SynchronizationMetrics.order_parameter(phases)
        assert r < 0.01

    def test_order_parameter_over_time(self) -> None:
        """Should return per-timestep r values."""
        history = np.array([
            [0.0, 0.0],
            [0.0, np.pi],
            [0.0, 0.1],
        ])
        r_t = SynchronizationMetrics.order_parameter_over_time(history)
        assert len(r_t) == 3
        assert r_t[0] == pytest.approx(1.0)  # identical
        assert r_t[1] == pytest.approx(0.0, abs=0.01)  # opposite

    def test_time_to_sync_detection(self) -> None:
        """Should find the first step where sync is sustained."""
        # Build a history: first 20 steps incoherent, then sync
        rng = np.random.default_rng(42)
        history = np.zeros((100, 3))
        for t in range(20):
            history[t] = rng.uniform(0, 2 * np.pi, size=3)  # random
        for t in range(20, 100):
            history[t] = [0.0, 0.05, 0.02]  # almost in-phase

        t_sync = SynchronizationMetrics.time_to_sync(
            history, threshold=0.9, sustain_steps=10,
        )
        assert t_sync is not None
        # Sync should be detected somewhere in [20, 30] range
        assert 20 <= t_sync <= 30

    def test_frequency_dispersion(self) -> None:
        """Coefficient of variation should match expected value."""
        freqs = [1.0, 1.0, 1.0]
        assert SynchronizationMetrics.frequency_dispersion(freqs) == 0.0

        freqs = [0.5, 1.0, 1.5]
        cv = SynchronizationMetrics.frequency_dispersion(freqs)
        # mean = 1.0, std ≈ 0.408, cv ≈ 0.408
        assert cv == pytest.approx(0.408, abs=0.01)

    def test_phase_coherence_synchronised(self) -> None:
        """Phase coherence should be 1.0 for identical phases."""
        c = SynchronizationMetrics.phase_coherence([0.0, 0.0, 0.0])
        assert c == pytest.approx(1.0)

# Project Requirements

## Project Overview

**Title:** Firefly-Inspired Visual Synchronization for Coordinated Multi-Drone Behaviour

**Degree:** MSc (Neurotechnology / Bioengineering)

This project investigates whether decentralised, visually-mediated synchronisation —
inspired by the flashing behaviour of firefly swarms — can be used to coordinate
the behaviour of multiple drones. Each drone is treated as an autonomous oscillator
that adjusts its flash timing based solely on the flashes it observes from its
neighbours.

## Functional Requirements

### FR1: Synchronisation Models
The system SHALL implement at least two mathematical models of synchronisation:

1. **Kuramoto model** — continuous-phase coupled oscillator dynamics
   governed by `dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)`.
2. **Pulse-coupled / Integrate-and-Fire model** — discrete-event dynamics
   where each oscillator charges toward a threshold, fires a pulse, and resets.

### FR2: Multi-Agent Simulation
The system SHALL simulate 2–3 drone agents, each with:
- An internal oscillator (Kuramoto or integrate-and-fire).
- A 2D/3D position in space.
- An LED that flashes when the oscillator fires.
- A camera/detector that observes flashes from other drones.

### FR3: Visual Coupling
Agents SHALL only be coupled to neighbours they can "see" — coupling strength
decays with distance (e.g., inverse-square or exponential falloff). This models
the real constraint that flash visibility degrades with range.

### FR4: Mock Hardware Layer
To support development without physical hardware, the system SHALL provide:
- **MockLED** — logs flash events to console or memory buffer.
- **MockCamera** — detects mock LED flashes from nearby simulated drones.
- **MockFlightController** — a no-op stub that satisfies the interface.

### FR5: Hardware Interface (Future)
Abstract base classes SHALL define the interface for real hardware:
- **AbstractLED** — GPIO-driven physical LED.
- **AbstractCamera** — OpenCV-based frame capture and flash detection.
- **AbstractFlightController** — MAVLink telemetry and command interface.

These interfaces allow the simulation layer to be swapped with real hardware
without modifying the core synchronisation logic.

### FR6: Experiment Logging
The system SHALL log experiment data to structured files (CSV or JSON):
- Timestep, agent ID, phase, firing state, position.
- Detected neighbour flashes and coupling terms.
- Configurable logging interval.

### FR7: Synchronisation Metrics
The system SHALL compute and report:
- **Kuramoto order parameter** `r(t) = |(1/N) · Σⱼ e^(iθⱼ)|` — measures
  phase coherence (0 = incoherent, 1 = fully synchronised).
- **Time-to-sync** — number of cycles until r(t) exceeds a threshold.
- **Flash synchrony index** — fraction of timesteps where all agents fire
  within a window of each other.

### FR8: Configuration
All simulation parameters SHALL be configurable via YAML files or
command-line arguments:
- Number of drones, initial phases, natural frequencies.
- Coupling model, coupling strength, distance-decay function.
- Simulation duration, integration timestep.

## Non-Functional Requirements

### NFR1: Modularity
The codebase SHALL follow a modular structure with clear separation between
core models, simulation, hardware I/O, logging, and utilities.

### NFR2: Testability
All synchronisation models and metrics SHALL have unit tests verifying
their mathematical correctness.

### NFR3: Reproducibility
Experiments SHALL be reproducible by fixing random seeds and logging
all configuration parameters alongside results.

### NFR4: Extensibility
The hardware abstraction layer SHALL make it straightforward to add
new oscillator models, coupling functions, or hardware backends.

## Deliverables

1. Working simulation of 2–3 drone synchronisation.
2. Comparison of Kuramoto vs pulse-coupled model convergence behaviour.
3. Parameter sweep analysis (coupling strength, distance, frequency spread).
4. (Stretch) Integration with a single physical LED + camera test rig.
5. (Stretch) Gazebo + ROS simulation with virtual camera sensors.

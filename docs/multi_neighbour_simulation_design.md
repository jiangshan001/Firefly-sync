# Multi-Neighbour Synchronisation Simulation — Design

## Purpose

This pure-software simulator compares **Kuramoto**, **PCO-I&F**, and **EAPF**
synchronisation models under **decentralised multi-agent** conditions.  No
camera, no GPIO, no leader UI, no hardware.

## Why Multi-Neighbour?

The single-leader Pi visual experiment (Stage 3A-3) validates real visual
detection and GPIO oscillator output — it is a **hardware validation** platform.

The multi-neighbour simulation is the **main model-comparison** platform because:

1. **Fixed-leader tracking favours bidirectional models.**  PCO-I&F phase-advance
   coupling inherently struggles when the follower starts faster than the leader.
   In a peer-to-peer multi-agent setting, every agent both transmits and receives
   pulses, making the comparison fairer.

2. **Topology matters.**  Real firefly swarms are not all-to-all.  Neighbourhood
   structure, directed information flow, and path length affect synchronisation
   dynamics.  The simulator supports `all_to_all`, `chain`, `directed_chain`,
   and `ring` topologies.

3. **Future hardware-in-the-loop.**  Eventually 2 virtual agents + 1 real Pi
   agent will interact via the real visual pipeline.  The multi-agent simulator
   provides the virtual-agent infrastructure for that mixed-reality setup.

## Model Assumptions

| Aspect | Kuramoto | PCO-I&F | EAPF |
|--------|----------|---------|------|
| Coupling type | Continuous phase | Event-driven pulse | Event-driven correction |
| Coupling input | Σ sin(θ_j − θ_i) per step | Pulse count from neighbours | Neighbour flash events |
| Phase representation | θ ∈ [0, 2π) | φ ∈ [0, 1) | θ ∈ [0, 2π) |
| Frequency adaptation | Fixed ω | Fixed (only phase advances) | Adaptive frequency f |
| Refractory period | No | Yes (configurable) | No |

## Topology Definitions

- **all_to_all:** Every agent sees every other agent.
- **chain:** Agent i sees i−1 and i+1 (undirected).
- **directed_chain:** Agent i sees only i−1 (information flows one way).
- **ring:** Agent i sees (i−1) mod N and (i+1) mod N (undirected ring).

## Metrics Definitions

1. **Group sync success** — all agents remain within 0.10 s timing error,
   freq spread < 0.05 Hz, order parameter > 0.85 for sustained cycles.

2. **Time to group sync** — first time criterion is met.

3. **Final frequency spread** — max − min estimated frequency over final 10 s.

4. **Mean pairwise timing error** — mean nearest-flash error between all
   agent pairs over final 10 s.

5. **Flash timing dispersion** — spread of flash times within each
   group-flash cycle, averaged over final 10 s.

6. **Mean order parameter R** — |mean(exp(i·θ_i))| averaged over final 10 s.

7. **Final order parameter R** — value at end of simulation.

## Limitations

- No spatial geometry or distance-dependent visibility.
- No camera detection pipeline (contrast with real Pi visual batch).
- Event delays and missed detections can be added but are zero by default.
- The simulator does not model CPU/loop timing — it is pure discrete-time.

## Connection to Future Mixed-Reality Setup

The multi-agent simulator will provide 2 virtual agents that interact with
1 real Pi agent via the existing visual pipeline:

```
Virtual Agent 0 (sim) ──flash──→ Virtual Agent 1 (sim) ──flash──→ Pi Agent (real camera + LED)
       ↑                                                                      │
       └──────────────────── observed via simulation ──────────────────────────┘
```

This bridges pure simulation and hardware-in-the-loop testing.

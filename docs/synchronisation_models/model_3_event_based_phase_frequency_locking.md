# Model 3 — Event-Based Adaptive Phase/Frequency Locking (EAPF)

## Model Motivation

The Kuramoto and PCO-I&F models are biologically inspired.  We also want an
**engineering-oriented baseline** that explicitly separates phase and frequency
tracking, similar to a phase-locked loop (PLL).  This model:

1. Provides a clean comparison: *do bio-inspired models outperform a simple
   engineering PFD (phase-frequency detector)?*
2. Extends naturally to multi-neighbour consensus (Olfati-Saber et al. 2007)
   by replacing the single-leader error with a weighted average of neighbour
   timing errors.
3. Is fully discrete-event-driven, matching the camera detection pipeline.

## Key References

- **Gardner 2005** — Standard reference on phase-lock loop engineering.
  Supports using phase error to adjust local oscillator phase and frequency.
- **Olfati-Saber, Fax & Murray 2007** — Consensus algorithms for networked
  multi-agent systems.  Provides the theoretical framework for extending this
  model to multi-neighbour consensus in future work.

## State Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `phase_rad` | float | [0, 2π) | Follower phase angle |
| `frequency_hz` | float | Hz | Adaptive follower frequency |
| `omega` | float | rad/s | `2π · frequency_hz` |
| `leader_period_estimate_s` | float | s | Estimated leader period from recent flashes |
| `leader_flash_times` | list[float] | s | Sliding window of recent leader flash timestamps |
| `phase_gain` | float | — | Gain for phase correction per leader event |
| `frequency_gain` | float | — | Gain for frequency correction per leader event |
| `last_self_flash_time` | float | s | `monotonic` timestamp of last follower flash |

## Input Events

- **Leader flash rising-edge** — detected by `PicameraFlashDetector`.

## Output Events

- **Follower flash** — when `phase_rad` wraps past 2π.  Triggers GPIO17 LED.

## Algorithm Logic

### Natural Phase Evolution (every loop iteration)

```
dt = now - last_loop_time
omega = 2π * frequency_hz
phase_rad += omega * dt
if phase_rad >= 2π:
    trigger_follower_flash()
    phase_rad -= 2π
```

### Leader Flash Event Update

When a valid leader flash rising edge is detected:

```
1. Record leader flash timestamp → leader_flash_times
2. If ≥ 2 timestamps:
     intervals = diff(leader_flash_times[-8:])
     leader_period_estimate_s = median(intervals)
3. Estimate desired follower phase at leader flash = 0 rad
4. phase_error_rad = wrap_to_pi(0 - phase_rad)
5. Apply phase correction:
     phase_rad = wrap_to_2pi(phase_rad + phase_gain * phase_error_rad)
6. Apply frequency correction:
     freq_correction = frequency_gain * phase_error_rad / (2π)
     frequency_hz = clamp(frequency_hz + freq_correction,
                          frequency_min_hz, frequency_max_hz)
     omega = 2π * frequency_hz
7. If phase_rad wraps past 2π after correction:
     trigger_follower_flash()
     phase_rad -= 2π
```

### Safety / Sanity

- `frequency_hz` is clamped to [`frequency_min_hz`, `frequency_max_hz`].
- Phase error is wrapped to [−π, π] to avoid large jumps.
- If fewer than 2 leader flashes, skip step 2 (period remains bootstrap guess).
- Min-interval gating (`leader_min_flash_interval_s`) prevents double-counting.

## Parameters

| Parameter | Candidate Range | Default | Description |
|-----------|----------------|---------|-------------|
| `phase_gain` | 0.05 – 0.40 | 0.20 | Phase correction strength |
| `frequency_gain` | 0.01 – 0.20 | 0.05 | Frequency correction strength |
| `frequency_min_hz` | — | 0.5 | Minimum allowed follower frequency |
| `frequency_max_hz` | — | 4.0 | Maximum allowed follower frequency |
| `leader_period_window` | 4 – 8 | 6 | Number of intervals for median period estimate |
| `min_leader_flash_interval_s` | — | 0.20 | Min interval for leader edge detection |
| `flash_on_time_s` | — | 0.06 | GPIO17 LED pulse duration |

## Expected Behaviour

- If the follower is **lagging** (phase < 0 when leader fires), `phase_error_rad`
  is positive → phase is advanced, frequency is increased → follower catches up.
- If the follower is **leading** (phase > 0 when leader fires), `phase_error_rad`
  is negative → phase is retarded, frequency is decreased → follower slows down.
- Over multiple leader cycles, `frequency_hz` should converge to `leader_freq_hz`.
- The system can lock even with significant initial frequency mismatch,
  limited only by `frequency_min_hz` / `frequency_max_hz`.

## Important Design Question — Sign Verification

Before implementation, verify the correction sign with unit tests:

```python
# Scenario: leader at t=1.0, follower phase_rad=5.0 (nearly done, but leader just fired)
# Expected: follower should delay/slow down because it is ahead.
# phase_error_rad = wrap_to_pi(0 - 5.0) = wrap_to_pi(-5.0) ≈ -1.283 rad
# With positive gain, correction should reduce phase and frequency.
```

If the sign is wrong, the follower will diverge instead of converging.
This must be tested with synthetic events before Pi hardware testing.

## Possible Failure Modes

1. **Divergence** — Wrong sign on correction gains causes frequency to move
   away from leader rather than toward it.
2. **Wind-up** — `frequency_hz` hits `frequency_min_hz` or `frequency_max_hz`
   and stays there; the system behaves like a fixed-frequency oscillator.
3. **Overshoot / ringing** — Gains too high cause the frequency estimate to
   oscillate around the leader frequency.
4. **Slow convergence** — Gains too low cause very slow frequency tracking.
5. **Bootstrap fragility** — With fewer than 2 leader flashes, the period
   estimate is wrong, causing large initial phase errors.

## Planned Evaluation Metrics

Same 9 metrics as Kuramoto, plus computational cost.

## Multi-Neighbour Consensus Extension (Future)

This model is designed to extend naturally to multi-neighbour consensus:

```python
# Instead of:
phase_error = wrap_to_pi(0 - phase_rad)

# Use weighted average of neighbour phase errors:
phase_error = sum(w_i * wrap_to_pi(neighbour_target_phase_i - phase_rad))
              / sum(w_i)
```

Where `neighbour_target_phase_i` is derived from neighbour flash timing
relative to the local oscillator.  This aligns with the Olfati-Saber et al.
(2007) consensus framework for directed information flow and robustness to
topology changes.

## Integration with Pi Visual Batch Runner

Same pattern as PCO-I&F: dispatched via `--model` CLI argument.
Uses `EventBasedPhaseLockOscillator` class with a compatible API.

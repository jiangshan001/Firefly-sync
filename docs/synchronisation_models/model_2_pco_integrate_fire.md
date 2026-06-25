# Model 2 — Pulse-Coupled Integrate-and-Fire (PCO-I&F)

## Model Motivation

The Mirollo–Strogatz (1990) pulse-coupled oscillator model is the canonical
mathematical description of firefly-style synchronisation.  Unlike Kuramoto's
continuous phase coupling, PCO-I&F is **event-driven**: each detected leader
flash is treated as a discrete pulse that advances the follower's internal
phase toward threshold.  This matches our Pi camera detection pipeline more
closely, since the camera produces discrete rising-edge events rather than a
continuous phase signal.

## Key References

- **Peskin 1975** — Origin of the cardiac pacemaker / integrate-and-fire
  framework.  Motivates threshold-reset oscillator representation.
- **Mirollo & Strogatz 1990** — Formal analysis of pulse-coupled synchronisation
  in populations of identical integrate-and-fire oscillators.
- **Tyrrell, Auer & Bettstetter 2006** — Fireflies as role models for
  synchronisation; discusses practical engineering constraints.

## State Variables

| Variable | Type | Unit | Description |
|----------|------|------|-------------|
| `phase` | float | [0, 1) | Normalised phase; 0 = just fired, 1 = threshold |
| `natural_frequency_hz` | float | Hz | Free-running follower frequency |
| `phase_rate` | float | 1/s | `natural_frequency_hz` (phase per second) |
| `last_self_flash_time` | float | s | `monotonic` timestamp of last follower flash |
| `last_leader_flash_time` | float | s | `monotonic` timestamp of last detected leader event |

## Input Events

- **Leader flash rising-edge** — detected by `PicameraFlashDetector` with
  min-interval gating (`leader_min_flash_interval_s`).

## Output Events

- **Follower flash** — triggers GPIO17 LED for `flash_on_time_s` seconds.
- **Oscillator state log** — per-loop-step phase, coupling, LED state.

## Algorithm Logic

### Natural Phase Evolution (every loop iteration)

```
dt = now - last_loop_time          # measured wall-clock dt
phase += natural_frequency_hz * dt
if phase >= fire_threshold:        # default 1.0
    trigger_follower_flash()
    phase = 0.0
    refractory_timer = refractory_period_s
```

### Leader Pulse Update (when a valid leader flash is detected)

```
if refractory_timer <= 0:
    # Mirollo–Strogatz phase advance toward threshold
    phase += epsilon * (1.0 - phase)
    if phase >= fire_threshold:
        trigger_follower_flash()
        phase = 0.0
        refractory_timer = refractory_period_s
else:
    log("refractory_ignore")
```

### Refractory Decay (every loop iteration)

```
if refractory_timer > 0:
    refractory_timer -= dt
```

## Parameters

| Parameter | Candidate Range | Default | Description |
|-----------|----------------|---------|-------------|
| `epsilon` | 0.05 – 0.40 | 0.15 | Phase advance per leader pulse |
| `fire_threshold` | — | 1.0 | Phase value triggering follower flash |
| `refractory_period_s` | 0.05 – 0.10 | 0.06 | Post-fire refractory window |
| `min_leader_flash_interval_s` | — | 0.20 | Min interval for leader edge detection |
| `flash_on_time_s` | — | 0.06 | GPIO17 LED pulse duration |

## Expected Behaviour

- With sufficient coupling (`epsilon` large enough relative to frequency
  difference), the follower should converge to fire in sync with the leader.
- Convergence is **monotonic** in the Mirollo–Strogatz model: phase always
  advances toward threshold, never retreats.
- If `epsilon` is too small, the follower never catches up to the leader
  (natural frequency difference dominates).
- If `epsilon` is too large, the follower fires immediately on every leader
  pulse, producing 1:1 locking regardless of phase alignment.

## Possible Failure Modes

1. **Refractory blocking** — If `refractory_period_s` is too long, leader
   pulses during refractory are ignored, slowing convergence.
2. **Weak coupling** — If `epsilon` is too small relative to the frequency
   difference, synchronisation may never occur.
3. **Over-coupling** — If `epsilon` is too large, the follower fires on every
   leader pulse regardless of phase, appearing "synced" but with poor phase
   alignment.
4. **Missing leader flashes** — If camera detection misses events, the
   follower drifts between detected pulses.

## Planned Evaluation Metrics

Same 9 metrics as Kuramoto:
1. `synchronization_success`
2. `time_to_synchronization_s`
3. `steady_state_mean_abs_timing_error_s`
4. `steady_state_rmse_timing_error_s`
5. `steady_state_jitter_s`
6. `final_frequency_error_hz`
7. `convergence_quality`
8. `detection_success_rate`
9. `false_positive_rate`

Plus computational cost metrics (loop rate, CPU, memory, temperature).

## Integration with Pi Visual Batch Runner

The PCO-I&F model will be selected by a `--model` CLI argument.
The existing `run_step3a_pi_visual_batch.py` will dispatch to a
`PulseCoupledIntegrateFireOscillator` instance instead of `KuramotoModel`.
All logging, metrics, and output formats remain identical.

### Suggested Oscillator API (compatible with existing runner)

```python
class PulseCoupledIntegrateFireOscillator:
    def __init__(self, natural_frequency_hz, epsilon, refractory_period_s, ...)
    def step(self, dt_s: float) -> bool           # returns True if flash
    def on_leader_pulse(self) -> None              # apply phase advance
    @property
    def phase(self) -> float
    @property
    def state(self) -> dict
```

# Kuramoto Frontend Dynamics Audit

Generated: 2026-06-21

This is an audit-only report. No runtime code, model parameters, detector,
camera, GPIO, or frontend logic was changed.

## Executive Summary

The Step 5B Kuramoto visual instability is primarily explained by the current
server-side Kuramoto implementation, not by EAPF, the detector, or a proven
frontend/backend divergence.

In feedback-ON mode the frontend does not run its own Kuramoto clock. It polls
`/api/agents` and displays the server-provided `flash_on` state. Therefore,
visible Kuramoto flashes are driven by the server oscillator's phase crossings.

The virtual Kuramoto agent uses a continuous phase-velocity update:

`phase_velocity = 2*pi*initial_frequency_hz + K*sin(neighbour_phase - phase)`

With the formal setting `initial_frequency_hz=2.0` and `K=5.0`, this means the
instantaneous phase velocity can range from about `1.204 Hz` to `2.796 Hz`.
That large instantaneous swing is not rate-limited or smoothed for Kuramoto.
The debug logs show exactly this range, frequent phase jumps above `0.5 rad`
per 33 ms update, and adjacent displayed `frequency_hz` jumps above `0.5 Hz`.

This is best classified as expected behaviour under the current implementation
and parameter sensitivity, not an event-consumption bug. The implementation
does not show NaN/Inf/negative frequency in the inspected formal logs, and Pi
flash events are consumed once per server update.

## Code Path Diagram

Step 5B mutual HIL feedback path:

```text
Pi oscillator flashes
  -> Step 5 runner POSTs /api/pi_flash
  -> run_leader_ui.py appends timestamp to _pi_flash_times if feedback_enabled
  -> background _agent_loop copies _pi_flash_times and clears queue
  -> _step_kuramoto_agent(..., pi_copy, running=True)
  -> neighbour period/phase estimate updates from Pi flash timestamps
  -> Kuramoto phase velocity is computed from sin(phase error)
  -> phase_rad advances; phase wrap increments fire_count and last_flash_time
  -> _compact_snapshot publishes flash_on based on last_flash_time hold window
  -> browser mutual.js polls /api/agents
  -> feedback-ON rendering uses server flash_on
```

Relevant code:

- `/api/pi_flash` queues timestamps and counts posts in
  [run_leader_ui.py](C:/Users/young/Desktop/BIOE70025/code/firefly-sync/experiments/run_leader_ui.py:660).
- `_agent_loop` copies and clears the Pi flash queue once per loop in
  [run_leader_ui.py](C:/Users/young/Desktop/BIOE70025/code/firefly-sync/experiments/run_leader_ui.py:393).
- `_step_kuramoto_agent` computes phase velocity, phase, frequency estimate,
  `fire_count`, and `flash_on` in
  [run_leader_ui.py](C:/Users/young/Desktop/BIOE70025/code/firefly-sync/experiments/run_leader_ui.py:256).
- `/api/agents` returns compact snapshots in
  [run_leader_ui.py](C:/Users/young/Desktop/BIOE70025/code/firefly-sync/experiments/run_leader_ui.py:512).
- Feedback-ON frontend rendering uses server `flash_on` in
  [mutual.js](C:/Users/young/Desktop/BIOE70025/code/firefly-sync/experiments/leader_ui/mutual.js:204).

## Server-Side Kuramoto Logic

### Pi flash queue and consumption

`POST /api/pi_flash` appends the posted timestamp to `_pi_flash_times` only
when `feedback_enabled` is true. It separately increments
`pi_flash_posts_received` / `received_pi_flashes`.

The background loop then does:

```text
pi_copy = list(_pi_flash_times)
_pi_flash_times.clear()
```

That `pi_copy` is passed to the active agent update. In the Step 5B setup
there is one virtual agent, so each queued Pi flash can affect that agent once.
The inspected formal debug files show `pi_flash_events_consumed_this_step` is
only `0` or `1` in the target trials; there is no evidence of multiple
application of one event to the same agent.

### Phase and frequency update

Kuramoto changes the virtual agent phase directly. It does not permanently
change the natural frequency used for the next step.

Each update uses:

```text
natural_freq = initial_frequency_hz
omega = 2*pi*natural_freq
coupling_input = sin(wrapped_phase_error)
phase_velocity = omega + K*coupling_input
phase_delta = phase_velocity * dt
phase_rad += phase_delta
```

It also writes:

```text
frequency_hz = max(0, phase_velocity / (2*pi))
```

So `frequency_hz` is a diagnostic instantaneous phase-velocity estimate, not a
smoothed physical oscillator frequency state. It is displayed and logged, but
the next Kuramoto update again uses `initial_frequency_hz`, not the previous
`frequency_hz`.

### Flash generation

If `phase_rad` crosses `2*pi`, the server wraps the phase, increments
`fire_count`, records `last_flash_time`, and appends to `flash_times`.

The compact API snapshot does not expose raw one-frame `flash_on` directly in
feedback-ON mode. Instead it computes visual `flash_on` from:

```text
0 <= snapshot_time - last_flash_time < feedback_flash_hold_s
```

The default hold is `0.25 s`. Thus the visible dot timing follows server phase
crossings, with a fixed hold window after each crossing.

### Bounds and safeguards

Kuramoto has warning diagnostics for:

- non-finite frequency
- `frequency_hz < 0.5` or `> 4.0`
- frequency step above `0.5 Hz`
- phase jump above `0.5 rad`
- multiple wraps in one update

But these warnings do not clamp the Kuramoto phase velocity. The only hard
frequency operation is `max(0, phase_velocity / (2*pi))`; no upper bound, rate
limit, or smoothing is applied to Kuramoto `frequency_hz` or phase increment.

For the formal `2 Hz`, `K=5` case:

```text
frequency_hz = 2.0 + (5 / 2*pi) * sin(error)
             = 2.0 +/- 0.7958 Hz
```

So the implementation itself permits approximately `1.204 Hz` to `2.796 Hz`
instantaneous phase velocity before considering noise, polling, or display.

## Frontend Rendering Logic

### Feedback OFF

When feedback is off, `mutual.js` uses a local deterministic browser clock:

```text
periodMs = 1000 / localFrequencyHz
cyclePhase = elapsedMs % period
isOn = cyclePhase < duty
```

This is intentionally smooth and independent of the server oscillator.

### Feedback ON

When feedback is on, the browser does not use `frequency_hz` for display
timing. It uses the latest `/api/agents` snapshot:

```text
return a.flash_on ? 255 : 15
```

The displayed `frequency_hz` is only a UI readout. The actual visual timing is
server `flash_on`, which is derived from server `last_flash_time`.

### Polling and render loop

`requestAnimationFrame(renderFrame)` continues independently of API polling.
The polling loop now reschedules itself in `.finally(scheduleNextPoll)`, so a
single failed `/api/agents` request should not permanently stop polling.

Heartbeat diagnostics are present in `mutual.html` and `mutual.js`:

- last poll age
- render-loop active indicator
- API poll error count
- feedback/debug text with `frequency_hz`, `phase_rad`, `raw_flash_on`, and
  server hold time

If `/api/agents` polling fails repeatedly, the render loop still continues but
uses the last known `agents` snapshot. In that situation stale `flash_on=false`
could make the dot appear stopped while the backend continues. However, the
current formal debug evidence points more strongly to server-side Kuramoto
phase-velocity variability than to a persistent frontend freeze.

## Evidence From Formal Logs

Formal logs were found under:

`experiments/logs/formal_step5b_chunked_20260621/experiments/logs/step5b_mutual_model_comparison/formal_step5b_chunked_20260621/`

The user-specified path
`experiments/logs/step5b_mutual_model_comparison/formal_step5b_chunked_20260621/`
was not present in this workspace.

### Aggregate model comparison

Across 9 formal trials per model:

| Metric | EAPF mean | Kuramoto mean |
|---|---:|---:|
| `frequency_error_final_5s_mean_abs` | 0.0461 Hz | 0.4131 Hz |
| `frequency_error_final_5s_abs_of_means` | 0.0426 Hz | 0.1991 Hz |
| `virtual_freq_final_5s_std` | 0.0006 Hz | 0.2474 Hz |
| `pi_freq_final_5s_std` | 0.0111 Hz | 0.3926 Hz |
| `actual_detection_fcr` | 0.9462 | 0.9027 |

The model-level pattern is consistent with the visual observation: EAPF is
smooth; Kuramoto has much higher frequency variability.

### Kuramoto debug targets

Inspected:

- `kuramoto_V2_P1.2_r01`
- `kuramoto_V2_P1.2_r02`
- `kuramoto_V2_P2.5_r03`

| Trial | Debug Rows | Freq Range | Phase Delta Range | Pi Events Per Step | Flash Count | Warnings | Invalid Freq |
|---|---:|---:|---:|---|---:|---:|---:|
| `kuramoto_V2_P1.2_r01` | 897 | 1.204-2.796 Hz | 0.250-0.580 rad | 0 or 1 | 53 | 89 | 0 |
| `kuramoto_V2_P1.2_r02` | 897 | 1.204-2.795 Hz | 0.250-0.580 rad | 0 or 1 | 52 | 105 | 0 |
| `kuramoto_V2_P2.5_r03` | 897 | 1.289-2.796 Hz | 0.267-0.580 rad | 0 or 1 | 65 | 206 | 0 |

Inter-flash timing from server debug rows:

| Trial | Server Flash Intervals |
|---|---:|
| `kuramoto_V2_P1.2_r01` | min 0.367 s, max 0.702 s, mean 0.570 s, std 0.080 s |
| `kuramoto_V2_P1.2_r02` | min 0.401 s, max 0.803 s, mean 0.572 s, std 0.090 s |
| `kuramoto_V2_P2.5_r03` | min 0.333 s, max 0.568 s, mean 0.459 s, std 0.053 s |

Adjacent instantaneous-frequency jumps:

| Trial | Max Adjacent Jump | Jumps > 0.5 Hz |
|---|---:|---:|
| `kuramoto_V2_P1.2_r01` | 1.568 Hz | 21 |
| `kuramoto_V2_P1.2_r02` | 1.502 Hz | 24 |
| `kuramoto_V2_P2.5_r03` | 1.020 Hz | 30 |

No inspected target showed NaN, Inf, negative frequency, frequency above 4 Hz,
fire-count drops, or multiple wraps in one update.

## Why Kuramoto Appears Visually Unstable

The most likely cause is:

1. The server-side Kuramoto update uses a strong instantaneous phase-velocity
   correction with `K=5.0`.
2. That correction directly changes how quickly `phase_rad` reaches the next
   wrap.
3. Server phase wraps determine `last_flash_time` and therefore the
   `flash_on` sent to the frontend.
4. The frontend faithfully displays server `flash_on` in feedback-ON mode.

So the frontend is not independently drifting. It is showing the server
oscillator's irregular phase crossings.

API/browser polling jitter can make the visual impression rougher because the
browser only receives `flash_on` snapshots at about 10 Hz, while the server
steps at about 30 Hz and holds flashes for 0.25 s. But polling jitter is a
secondary display artifact, not the main cause of the measured Kuramoto
frequency standard deviation.

## Comparison With EAPF

EAPF is more stable because its server update includes explicit filtering and
limits:

- `phase_gain = 0.02`
- `frequency_gain = 0.02`
- phase/frequency error filters with alpha `0.2`
- `max_phase_step_rad = 0.2`
- `max_frequency_step_hz = 0.05`
- `frequency_min_hz = 0.8`
- `frequency_max_hz = 3.2`

EAPF updates `frequency_hz` as a bounded oscillator state and then uses that
state for phase advance. Kuramoto, by contrast, uses a direct instantaneous
phase-velocity term and reports `frequency_hz` as that instantaneous velocity.
This is why EAPF can converge smoothly while Kuramoto visibly floats under the
same mutual-HIL event stream.

## Bug vs Expected Behaviour vs Parameter Sensitivity

Classification:

- Not primarily a frontend rendering bug.
- Not a detector/camera/GPIO bug based on this evidence.
- Not an event double-consumption bug in the inspected formal logs.
- No evidence of NaN/Inf/negative frequency in inspected Kuramoto debug files.
- Best described as expected behaviour under the current Kuramoto virtual-side
  implementation with high parameter sensitivity at `K=5.0`.

There is one frontend caveat: if `/api/agents` polling fails repeatedly, stale
`flash_on=false` can make the dot appear stopped while the backend continues.
The current polling loop is designed to continue after failures, and heartbeat
diagnostics expose poll age and error count. That failure mode should be
monitored, but it does not explain the formal aggregate frequency variability.

## Recommended Next Steps

### A. For Thesis / Reporting

Describe Kuramoto carefully as a continuous phase-coupled baseline whose
current mutual-HIL implementation applies immediate phase-velocity correction
from event-derived neighbour phase estimates. In the Step 5B mutual-HIL setting
this produced higher variability and poorer final-window agreement than EAPF.

Avoid claiming that Kuramoto is intrinsically unsuitable. The fair claim is
that, under the implemented event-driven visual mutual-HIL wrapper and `K=5.0`,
Kuramoto was more sensitive and less visually stable than EAPF Consensus.

Mention that EAPF has explicit event-based filtering and correction limits,
which are structurally well matched to timestamped visual flashes. Kuramoto is
being adapted to sparse visual events through a phase-estimation wrapper, so
its behaviour depends strongly on coupling gain, event timing, and display
semantics.

### B. For Engineering

Do not tune blindly. Suggested next diagnostic/fix options:

1. Run a Kuramoto-only `K` sweep in mock/server-only mode first.
   Check whether lower `K` reduces phase velocity jumps without losing lock.

2. Add a bounded/smoothed display frequency diagnostic.
   Keep model phase dynamics unchanged, but avoid presenting instantaneous
   `frequency_hz` as if it were a stable oscillator frequency.

3. Consider a phase-only visual Kuramoto mode.
   Keep a bounded display oscillator and apply Kuramoto corrections as phase
   nudges rather than unbounded instantaneous phase-velocity changes.

4. Add optional Kuramoto phase-velocity bounds for HIL display safety.
   For example, clamp visual phase velocity to a safe range while logging the
   unclamped theoretical value. This would be an engineering change and should
   be tested separately from the formal evidence.

5. Keep frontend fallback diagnostics enabled.
   Monitor poll age, poll error count, `raw_flash_on`, `flash_on`, and
   `fire_count` during any future Kuramoto HIL run.

6. Preserve EAPF settings.
   EAPF's stability in the formal results comes from the locked Stage 4A
   consensus PLL parameters and should not be changed as part of Kuramoto
   diagnosis.


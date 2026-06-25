# Step 5B Mutual HIL Model Comparison Plan

## Goal

Compare two 1-virtual + 1-real-Pi mutual visual synchronisation conditions:

- `eapf_consensus`
- `kuramoto`

Each condition must use the same model family on both sides of the loop:

- laptop `/mutual` virtual agent model
- Raspberry Pi oscillator model

Mixed-model runs are not part of the formal comparison unless explicitly marked
as diagnostic.

## Locked Model Parameters

Use the Stage 4A model-selection locked parameters:

- `eapf_consensus`: `g_p=0.02`, `g_f=0.02`, `alpha_p=0.2`,
  `alpha_f=0.2`, `delta_theta_max=0.2 rad`, `delta_f_max=0.05 Hz`
- `kuramoto`: `K=5.0`

Do not use the earlier fixed-leader HIL Kuramoto value `K=3.5` for the formal
Step 5B mutual comparison.

## Formal Conditions

Default frequency pairs:

1. virtual 2.0 Hz, Pi 1.5 Hz
2. virtual 2.0 Hz, Pi 1.8 Hz
3. virtual 2.0 Hz, Pi 2.3 Hz

Optional extended frequency pairs:

4. virtual 1.8 Hz, Pi 2.2 Hz
5. virtual 2.3 Hz, Pi 1.5 Hz

Recommended formal run:

```bash
PYTHONPATH=. python experiments/run_step5b_mutual_hil_batch.py \
  --leader-api http://192.168.1.111:8000 \
  --duration 60 \
  --models eapf_consensus kuramoto \
  --freq-pairs 2.0:1.5,2.0:1.8,2.0:2.3 \
  --repeats 3 \
  --dot-size 450 \
  --api-timeout 10 \
  --api-retries 3 \
  --alternate-models
```

## Trial Setup

For each trial, the runner performs:

1. `POST /api/mode {"mode":"mutual_hil"}`
2. `POST /api/pause` best-effort
3. `POST /api/reset`
4. `POST /api/agents/0` with position, size, initial frequency, current
   frequency, and selected model
5. `POST /api/feedback {"enabled": true}`
6. `POST /api/start`
7. `GET /api/agents` for start state
8. Pi camera + selected Pi oscillator loop
9. `GET /api/agents` for end state before pause
10. `POST /api/pause` in cleanup

## Metrics

Per-trial metrics include:

- actual virtual flash count from server `fire_count`
- Pi-detected virtual flashes
- actual and nominal detection FCR
- start/end virtual frequency
- Pi final observed frequency
- final frequency error
- final-10s frequency and timing errors
- time to frequency lock
- time to timing lock
- virtual/Pi flash counts and ratio
- received Pi flashes reported by `/mutual`
- API/server loop rate when available
- timeout/failure state and error message

Frequency lock is defined as `|pi_freq - virtual_freq| < 0.05 Hz` sustained for
5 seconds. Timing lock is defined as nearest flash timing error `< 0.10 s` for
5 consecutive Pi flashes.

## Analysis

After a batch:

```bash
PYTHONPATH=. python experiments/analyse_step5b_mutual_hil_batch.py \
  experiments/logs/step5b_mutual_model_comparison/<batch_dir>
```

Figures are written under `<batch_dir>/figures/`.

## Kuramoto Implementation Note

Kuramoto does not have an EAPF-style adaptive frequency state. Its oscillator
uses fixed natural frequency plus phase-velocity coupling. For final/common
frequency metrics, the Step 5B runner estimates observed frequency from recent
flash intervals when available. The raw server `frequency_hz` remains logged
from `/api/agents`.

## Safety

Do not run hardware trials without confirming camera alignment, dot visibility,
and GPIO readiness. Use `--dry-run` first to inspect schedule and output paths.

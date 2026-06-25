# Step 5A — Mutual Mixed-Reality HIL Design

## Purpose

Step 5A builds the infrastructure for mutual HIL synchronisation where
1 virtual agent (browser) and 1 real Pi agent (camera + GPIO17 LED)
influence each other through flash events, rather than one being a
fixed leader.

## fixed_leader vs mutual_hil

| Aspect | fixed_leader | mutual_hil |
|--------|-------------|------------|
| Leader behaviour | Fixed frequency, prescribed | Virtual agent oscillator, adaptive |
| Leader receives input | No | Yes — via `POST /api/pi_flash` |
| Follower influences leader | No | Yes |
| Asymmetric | Yes (leader ignores follower) | Yes (Pi uses camera; virtual uses API) |
| Pi camera | Detects leader flashes | Detects virtual agent flashes |
| Pi output | GPIO17 LED | GPIO17 LED + POSTs flash events |

## Data Flow

```
Virtual agent flash (canvas)
    → Pi camera detects visual flash
    → Pi EAPF oscillator steps with neighbour_flash_ids=[0]
    → Pi oscillator fires → GPIO17 LED on
    → Pi POSTs /api/pi_flash {"timestamp": t}
    → Server registers Pi flash in _pi_flash_times queue
    → Virtual agent EAPF oscillator processes Pi flash as neighbour event
    → Virtual agent phase/frequency updated
    → Virtual agent flash (canvas) — cycle repeats
```

## API Endpoints (new in Step 5A)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/mode` | Get current mode |
| POST | `/api/mode` | Set mode `{"mode":"fixed_leader"|"mutual_hil"}` |
| GET | `/api/agents` | List all virtual agents |
| POST | `/api/agents` | Create/configure agent |
| POST | `/api/agents/{id}` | Update agent position/size/freq |
| POST | `/api/start` | Start virtual agent oscillators |
| POST | `/api/pause` | Pause oscillators |
| POST | `/api/reset` | Reset all agents |
| POST | `/api/pi_flash` | Register Pi flash `{"timestamp": <s>}` |

## Coordinate System

User-facing coordinates use bottom-left origin (x≥0, y≥0).
Internal canvas conversion: `canvas_y = canvas_height - user_y`.

## Model Handling

Virtual agents support `kuramoto` and `eapf_consensus` models.
The server runs a background thread (~30 fps) that steps each virtual
agent's oscillator and processes queued Pi flash events.

EAPF Consensus parameters (locked Stage 4A):
- `phase_gain = 0.02, frequency_gain = 0.02`
- `phase_error_filter_alpha = 0.2, frequency_error_filter_alpha = 0.2`
- `max_phase_step = 0.2 rad, max_frequency_step = 0.05 Hz`

## Frontend Visualisation

### fixed_leader mode
The existing single-dot leader is preserved unchanged.  ``/api/leader/config``
controls frequency, duty cycle, brightness, size, shape, and offset.
The leader dot flashes according to the browser's ``performance.now()`` timer.

### mutual_hil mode
When the server mode is ``mutual_hil``, the browser polls ``GET /api/agents``
at ~20 Hz and renders each virtual agent as a glowing circle on a black
canvas.  The dot brightness follows the server-side oscillator state
(``flash_on`` field).  Users can click a dot to select it, drag it to
reposition, or edit coordinates/size/frequency via sliders.

### Display-only mode
The **Display Only** button hides all UI panels (controls, camera, mutual HIL),
leaving only the black canvas with virtual dots visible to the Pi camera.

## Limitations

- Virtual agents receive Pi flash timestamps at server receive time,
  not Pi monotonic time. Clock offset is not compensated.
- Asymmetry: Pi observes virtual agent via camera (visual); virtual
  observes Pi via API (network). This reflects real hardware asymmetry.
- Single virtual agent supported in Step 5A; multi-agent extension
  is designed for but not yet implemented.
- Frontend polling at ~20 Hz for agent state; may introduce ~50 ms display latency.

## How to Run

**Laptop (frontend):**
```bash
python experiments/run_leader_ui.py --host 0.0.0.0 --port 8000
```

**Pi (mutual HIL smoke):**
```bash
PYTHONPATH=. python3 experiments/run_step5a_mutual_hil_smoke.py \
    --leader-api http://<laptop-ip>:8000 --duration 60
```

**Dry-run (no hardware):**
```bash
PYTHONPATH=. python experiments/run_step5a_mutual_hil_smoke.py --dry-run --duration 5
```

# Step 5 Mutual HIL — Logic Audit

**Date:** 2026-06-19  
**Files inspected:** `run_leader_ui.py`, `mutual.html`, `mutual.js`, `index.html`, `app.js`, `styles.css`, `run_step5a_mutual_hil_smoke.py`, `step5a_mutual_hil_design.md`, `check_step5b_virtual_source_stability.py`  
**Code modified:** No

---

## 1. High-Level Architecture

```
LAPTOP (server + browser)
├── run_leader_ui.py      — HTTP server, agent loop, REST API
├── leader_ui/index.html  — fixed-leader page (Step 3/4)
├── leader_ui/mutual.html — mutual HIL page (Step 5)
├── leader_ui/mutual.js   — polls /api/agents, renders virtual dots
└── leader_ui/app.js       — fixed-leader JS (unchanged)

PI (hardware)
├── run_step5a_mutual_hil_smoke.py — Pi smoke script
├── PicameraFlashDetector           — detects virtual dot flashes
├── EventBasedConsensusPLLOscillator — Pi-side EAPF oscillator
└── GPIO17 LED                      — Pi follower output
```

**Data flow (intended):**
Virtual dot flash → Pi camera → Pi oscillator step → GPIO17 LED on → POST /api/pi_flash → server registers Pi event → (if feedback ON) virtual oscillator correction

**Calibration vs feedback mode:**
- `feedback_enabled=false` (default): virtual dot flashes at deterministic 2 Hz from `time.monotonic()`. Pi events logged but not applied.
- `feedback_enabled=true`: virtual dot runs EAPF Consensus oscillator, Pi events affect phase/frequency.

---

## 2. Current Route and File Map

| URL | Serves | JS loaded | Cache-bust |
|-----|--------|-----------|:----------:|
| `/` or `/index.html` | Fixed leader page | `app.js` | No |
| `/mutual` | Mutual HIL page | `mutual.js?v=2` | Yes (`?v=2`) |
| `/mutual/` | Same | Same | Yes |

**CSS visibility:**
- `#mutual-hil-panel { display: flex; }` — visible by default
- `body.display-only #mutual-hil-panel { display: none !important; }` — hidden in display-only
- `#btn-exit-display-only` — `display:none` normally; `display:block` when `body.display-only`

**Python endpoints used by mutual page:**
`GET /api/mode`, `POST /api/mode`, `GET /api/agents`, `POST /api/agents/0`, `POST /api/start`, `POST /api/pause`, `POST /api/reset`, `POST /api/feedback`, `POST /api/pi_flash`

---

## 3. Server State Variables

| Variable | Defined | Written by | Read by | Reset |
|----------|---------|-----------|---------|-------|
| `_leader_config` | L44 (global dict) | `_update_config()`, `do_POST` handlers | `_get_config_copy()`, snapshots, step functions | `/api/reset` (running=false, feedback_enabled=false) |
| `_agents` | L53 (global list) | `/api/mode` (auto-create), `/api/agents` (POST), `/api/agents/{id}` (POST) | Agent loop (shallow copy), `/api/agents` (fallback) | `/api/reset` (all agents reset) |
| `_agents_snapshot` | L54 (global list) | Agent loop (after each step) | `/api/agents`, `/api/status` | Empty until first loop cycle |
| `_agents_lock` | L55 (RLock) | All handlers + agent loop | All handlers + agent loop | N/A |
| `_snapshot_lock` | L56 (Lock) | Agent loop (snapshot assignment) | `/api/agents`, `/api/status` | N/A |
| `_pi_flash_times` | L58 (list) | `/api/pi_flash` (append, if feedback ON) | Agent loop (copy+clear) | `/api/reset`, `/api/start` |
| `_calibration_start_time` | L62 (float) | `/api/start` | `_step_calibration_agent()` | On new `/api/start` |
| `_visual_duty_cycle` | L63 (float, 0.5) | Not writable via API | `_step_calibration_agent()` | Never |
| `feedback_enabled` | L50 (`_leader_config["feedback_enabled"]`) | `/api/feedback`, `/api/reset` | Compact snapshot, agent loop dispatch, Pi flash handler | `/api/reset` (→false) |
| `flash_duration_s` | L50 (0.10) | Not actively used | Was in old step function signature | Never |
| `_loop_alive/error/rate/step_count` | L238-241 | Agent loop (each cycle) | `/api/status` | On loop exit |

---

## 4. Mutual Agent State Schema

**Internal agent dict** (`_init_agent`, L84):
```python
id, model, x, y, size, initial_frequency_hz, frequency_hz,
phase, phase_rad, flash_on, last_flash_time, fire_count,
enabled, received_pi_flashes, flash_times,
neighbour_flash_times, neighbour_period_est, neighbour_phase_est,
phase_error_filt, freq_error_filt
```

**Compact snapshot** (`_compact_snapshot`, L98):
```python
id, model, x, y, size, initial_frequency_hz, frequency_hz,
phase_rad, flash_on, fire_count, enabled, received_pi_flashes,
last_flash_time, flash_times_count, calibration_elapsed,
running, feedback_enabled
```

**What `/api/agents` actually returns:**
The compact snapshot (12 primitive fields + running + feedback_enabled + calibration_elapsed). Does NOT include: `flash_times` array itself (only count), `visual_duty_cycle`, `calibration_start_time`.

---

## 5. Lifecycle State Machine

### POST /api/mode {"mode":"mutual_hil"}
- Sets `_mode = "mutual_hil"`
- If `_agents` empty: auto-creates agent 0 at (x=0,y=0,size=200,freq=2.0,model=eapf_consensus)
- Sets `_leader_config["api_controlled"] = True`
- Does NOT start agent loop
- Does NOT set running=true
- Does NOT set feedback_enabled

### POST /api/agents/0
- Updates agent 0 fields from JSON body (x,y,size,initial_frequency_hz,model,enabled,frequency_hz)
- Returns updated agent dict (internal, NOT compact snapshot)

### POST /api/start
- Sets `_calibration_start_time = time.monotonic()`
- Starts background agent loop thread (if not already running)
- Sets `_leader_config["running"] = True, api_controlled = True`
- Clears `_pi_flash_times`
- Resets `fire_count = 0` and `flash_times = []` per agent
- Does NOT affect `feedback_enabled`

### POST /api/pause
- Stops agent loop (`_loop_running = False`)
- Sets `_leader_config["running"] = False`
- Does NOT clear Pi events
- Does NOT affect `feedback_enabled`

### POST /api/reset
- Stops agent loop
- Sets `running = False, feedback_enabled = False`
- Clears `_pi_flash_times`
- Resets all agent state (phase=0, fire_count=0, flash_times=[], frequency=initial, etc.)

### POST /api/feedback {"enabled": true/false}
- Sets `_leader_config["feedback_enabled"] = value`
- Returns `{"feedback_enabled": value}`
- Does NOT start/stop loop
- Does NOT affect running state

### POST /api/pi_flash {"timestamp": 1.5}
- If `feedback_enabled=true`: appends timestamp to `_pi_flash_times` queue
- If `feedback_enabled=false`: does NOT queue; only increments `received_pi_flashes` counter
- Returns `{"received": true, "feedback_enabled": current_value}`

---

## 6. Background Loop Logic

- **Start:** `/api/start` → `_start_agent_loop()` → `threading.Thread(target=_agent_loop, daemon=True).start()`
- **Stop:** `/api/pause` → `_stop_agent_loop()` → sets `_loop_running=False`
- **Frequency:** ~30 fps (`dt = 0.033s` sleep outside lock)
- **Threading:** Uses `Thread`, NOT `ThreadingHTTPServer` (server is `HTTPServer`)
- **Dt computation:** Fixed `dt = 0.033` for EAPF step; calibration mode ignores dt entirely
- **Stepping when running=false:** Calibration/EAPF step functions check `running` and are no-ops when false
- **Snapshot update:** After each step loop completes, `_agents_snapshot` rebuilt from `_compact_snapshot`
- **Sleep outside locks:** Yes — `time.sleep(dt)` is outside `with _agents_lock` (line 256)
- **Exception handling:** `try/except` catches all exceptions, logs to `_loop_error`
- **Keeps running after trial:** Yes — loop continues until `/api/pause` or `/api/reset` called

---

## 7. Calibration / Feedback-OFF Logic

| Question | Answer |
|----------|--------|
| Does `feedback_enabled` exist? | Yes, `_leader_config["feedback_enabled"]`, default `False` |
| Is it default false? | Yes |
| Global or per-agent? | Global (affects ALL agents) |
| Which endpoint changes it? | `POST /api/feedback` and `POST /api/reset` |
| Does `/api/status` expose it? | Yes (in `cfg["agents"][n]` via snapshot) |
| Does `/api/agents` expose it? | Yes (`feedback_enabled` field per agent in snapshot) |
| Does `mutual.js` know about it? | **Yes** — has `btnFeedbackTgl` button and `feedbackOn` variable that toggles via `POST /api/feedback` |
| Does `run_step5a_mutual_hil_smoke.py` know about it? | **NO** — the Pi script has no `--feedback-off` flag and never calls `/api/feedback` |
| When `feedback_enabled=false`, which function computes `flash_on`? | `_step_calibration_agent()` (line 134) |
| Is calibration based on `time.monotonic()` or accumulated dt? | **`time.monotonic()`** — absolute wall clock (line 136) |
| Exact formula | `elapsed = now_abs - _calibration_start_time; cycle_index = floor(elapsed/period); cycle_phase = (elapsed % period) / period; phase_rad = 2π * cycle_phase; flash_on = cycle_phase < visual_duty_cycle` |
| Duty cycle | `_visual_duty_cycle = 0.5` (50%) |
| Does calibration ignore `/api/pi_flash`? | Yes — Pi events not queued when feedback=false |
| Are stale Pi events cleared on start/reset? | Yes — both `/api/start` and `/api/reset` call `_pi_flash_times.clear()` |
| Are Pi events queued while running=false? | Only if `feedback_enabled=true` AND running is irrelevant for queuing |
| Are Pi events consumed once only? | Yes — agent loop copies then clears `_pi_flash_times` atomically |

**Latest timing test evidence (2026-06-19):**
- 6 s test: fire_count = 12, fire deltas = all 1, calibration_elapsed correctly tracks wall time
- Phase progresses smoothly from 5.49→0 rad each cycle
- flash_on toggles at 50% duty boundary
- **Server-side calibration timing appears correct**

---

## 8. True Feedback Logic

When `feedback_enabled=true`:

- `_step_eapf_agent()` runs (line 155)
- Inputs: agent dict, dt=0.033, `pi_flash_times` list, running flag, feedback flag, flash_dur
- Pi events from the queue consumed once (list copied+cleared atomically)
- Pi timestamps: **server receive time** used via `time.monotonic()`
- Frequency clamps: `fmin=0.8, fmax=3.2` (line 172)
- **Can frequency fall to ~0.6–1.0 Hz?** No — clamped to minimum 0.8 Hz. But Pi smoke reported 1.0007 Hz final Pi frequency, which is below the virtual agent's expected 2.0 Hz. **This is the Pi's own oscillator frequency, not the virtual agent's.**
- **Can phase/frequency update cause burst-stop-burst?** The EAPF consensus oscillator was designed for multi-neighbour simulation, not visual flashing. When it receives Pi events at irregular intervals (due to camera detection jitter), it may produce irregular timing. The `flash_on` is set only on the exact frame where phase wraps past 2π, making it sensitive to timing jitter.

---

## 9. Browser Rendering Logic (`mutual.js`)

| Question | Answer |
|----------|--------|
| Polling rate | ~20 Hz (`setTimeout(..., 50)`) |
| Endpoint polled | `GET /api/agents` |
| Brightness decision | `var bright = (isRunning && a.flash_on) ? 255 : 15` (line 115) |
| Uses server `flash_on` directly | Yes |
| Also checks `running` | Yes — `isRunning = a.running !== false` |
| CSS animation | No — no CSS keyframes, no transitions on brightness |
| Browser own timer | No — rendering depends entirely on server snapshot |
| Display-only affects rendering | Only hides panels/overlays; canvas rendering unchanged |
| Panels/buttons interfere with camera | `btn-exit-display-only` is `position:fixed; top-right; z-index:9999` — potentially visible to camera even in display-only mode |

---

## 10. Pi Smoke Script Logic (`run_step5a_mutual_hil_smoke.py`)

**CLI arguments available:**
```
--leader-api, --duration, --virtual-freq, --pi-freq, --dry-run,
--log-dir, --width, --height, --min-interval, --window-s
```
**`--feedback-off` does NOT exist.**

**API call sequence (non-dry-run):**
1. `POST /api/mode {"mode":"mutual_hil"}` — switches mode
2. `POST /api/start` — starts virtual agent
3. (No call to `/api/feedback`, `/api/reset`, or `/api/agents/0` for config)
4. Trial loop: Pi camera detects virtual flashes → steps Pi oscillator → POST `/api/pi_flash`
5. `finally`: `POST /api/pause`

**Expected virtual flashes:** `duration * virtual_freq` (based on CLI `--virtual-freq 2.0` = expects 20 flashes in 10s). This assumes the virtual source IS producing at 2.0 Hz.

**Key observation:** The script never reads the actual `fire_count` or `frequency_hz` from `/api/agents`. It assumes the virtual source is working.

**The 10 s smoke result:** Virtual detected 4/20, FCR=0.200. This means the camera detected only 4 virtual flashes out of 20 expected. The problem is either:
- The virtual dot is not visually flashing clearly enough for the camera, or
- The Pi camera detection pipeline is missing flashes, or
- The virtual dot's position/size/contrast is wrong

---

## 11. Mismatch and Inconsistency List

| # | Finding | Severity |
|---|---------|----------|
| 1 | **Server supports `feedback_enabled` but Pi script has no `--feedback-off` flag** | High — Pi always runs with feedback ON because it never calls `/api/feedback` or `/api/reset` before `/api/start` |
| 2 | **Pi script calls `/api/start` immediately after `/api/mode` without reset** | Medium — stale Pi events from previous runs may persist |
| 3 | **Pi script never reads actual virtual frequency from `/api/agents`** | Medium — FCR calculation assumes perfect 2.0 Hz source |
| 4 | **Server calibration mode verified correct via API tests (6s, 12 fires, uniform)** | Informational — server code is correct but not used by Pi script |
| 5 | **`btn-exit-display-only` is `position:fixed; z-index:9999`** — may be visible to Pi camera even in display-only | Low — user can press Esc instead |
| 6 | **`_visual_duty_cycle` is hardcoded 0.5 and not exposed in any API endpoint** | Low — but prevents diagnostic inspection |
| 7 | **`flash_times` internal array grows unbounded — only count is in snapshot** | Low — memory concern for long trials |
| 8 | **Pi script uses server receive time for `/api/pi_flash` timestamps** — documented, acceptable | Low |

---

## 12. Evidence from Current Logs/Results

| Evidence | Proves | Does NOT prove |
|----------|--------|----------------|
| Dry-run smoke (5s, no hardware): Pi flashes=7, freq=1.5000 Hz | Pi oscillator works in isolation | Virtual dot visual output |
| 10 s smoke: Virtual detected 4/20, FCR=0.200, Pi freq=1.0007 Hz | Camera detection is not capturing virtual flashes | Whether server generates correct flashes |
| Server timing test (6s, feedback OFF): 12 fires, uniform deltas of 1 | Server calibration produces correct phase/fire_count | Browser renders flashes correctly for camera |
| Server /api/agents returns `flash_on: true/false` at 50% duty | Server flash_on toggles at server polling rate (30 fps) | Browser shows it at 20 Hz polling rate without aliasing |
| CLI help shows no `--feedback-off` | Pi script cannot disable feedback | Whether feedback is the cause of the low detection |

**What the smoke result FCR=0.200 means:** Only 4 of 20 expected virtual flashes were detected. The Pi oscillator still fired 12 times at ~1 Hz. This strongly suggests the virtual dot is not visually flashing as a clean source.

---

## 13. What Needs to Be Measured Next

1. **Server virtual fire timestamps:** Poll `/api/agents` rapidly (100ms) and record `fire_count` changes → verify server flash timing
2. **Browser render check:** With a second camera or screen recording, verify the virtual dot visibly flashes at 2 Hz with 50% duty
3. **Pi camera log:** Compare Pi `detection_log.csv` timestamps against server fire timestamps
4. **Pi `/api/pi_flash` log:** Compare Pi POST timestamps against Pi oscillator flash timestamps
5. **Visual contrast:** Verify the virtual dot size (200→450px), brightness (255), and position (800,400) are correct for Pi camera detection
6. **Pi detection settings:** Confirm `threshold_on/off`, `min_interval`, `window_s` match those used in successful Kuramoto fixed-leader batch

---

## 14. Recommended Next Fix Strategy (do not implement yet)

1. **Fix Pi script first:** Add `--feedback-off` flag. Default to feedback OFF for initial smoke.
2. **Add server reset to Pi script:** Call `POST /api/reset` before `POST /api/start` to ensure clean state.
3. **Verify virtual visual output:** Run a 10 s calibration test where the Pi script just records camera detection (doesn't run its own oscillator). This isolates whether the visual problem is in the server→browser→camera chain or the Pi detection pipeline.
4. **If visual detection still low:** Increase virtual dot size to 450px or larger, verify position, use the same camera settings as Kuramoto fixed-leader batch.
5. **Only after visual detection is reliable:** Enable feedback (`--feedback-off` removed or set to false) and test mutual HIL.

---

## 15. Commands for Manual Verification

```bash
# Start server
python experiments/run_leader_ui.py --host 0.0.0.0 --port 8000

# Open mutual page
# http://localhost:8000/mutual

# Check status
curl http://localhost:8000/api/status

# Check agents (with timing)
for i in {1..20}; do curl -s http://localhost:8000/api/agents | python -c "import sys,json; a=json.load(sys.stdin)['agents'][0]; print(a['fire_count'], a['flash_on'], a['phase_rad'])"; sleep 0.25; done

# Start / pause / reset
curl -X POST http://localhost:8000/api/start
curl -X POST http://localhost:8000/api/pause
curl -X POST http://localhost:8000/api/reset

# Toggle feedback
curl -X POST http://localhost:8000/api/feedback -H "Content-Type: application/json" -d '{"enabled":true}'
curl -X POST http://localhost:8000/api/feedback -H "Content-Type: application/json" -d '{"enabled":false}'

# Virtual source diagnostic
PYTHONPATH=. python experiments/check_step5b_virtual_source_stability.py --duration 15

# Pi smoke (dry-run)
PYTHONPATH=. python experiments/run_step5a_mutual_hil_smoke.py --dry-run --duration 5

# Pi smoke (real — copy to Pi first)
scp experiments/run_step5a_mutual_hil_smoke.py pi@dronepi.local:~/firefly-sync/experiments/
```

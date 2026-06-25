# Firefly-Inspired Visual Synchronization for Coordinated Multi-Drone Behaviour

**MSc Project — BIOE70025**

This project implements and simulates biologically-inspired visual synchronization
algorithms for coordinating the flashing behaviour of multiple drones. Drawing
inspiration from firefly swarms, the system models how individual agents can
achieve global synchrony using only local visual cues (flash detection), without
centralised control.

## Features

- **Kuramoto Model** — phase-coupled oscillator model for smooth synchronisation
- **Pulse-Coupled / Integrate-and-Fire Model** — discrete event-based synchronisation
- **Pure Software Simulation** — 2–3 drone agents with configurable parameters
- **Mock Hardware Layer** — simulated LED output and camera-based flash detection
- **Experiment Logging** — structured logging of phase, flash events, and metrics
- **Synchronisation Metrics** — Kuramoto order parameter, phase coherence, time-to-sync
- **Hardware Interface** — abstract base classes for future GPIO LED, camera, and MAVLink

## Project Structure

```
firefly-sync/
├── README.md
├── requirements.txt
├── docs/
│   └── project_requirements.md
├── firefly_sync/
│   ├── core/               # Oscillator and synchronisation models
│   ├── simulation/          # Drone agents, environment, and engine
│   ├── hardware/            # Abstract interfaces + mock implementations
│   ├── logging/             # Experiment logging and metrics
│   ├── utils/               # Configuration, visualisation helpers
│   └── experiments/         # Experiment runner scripts
└── tests/                   # Unit and integration tests
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a basic two-drone Kuramoto simulation
python -m firefly_sync.experiments.runner --model kuramoto --drones 2

# Run a pulse-coupled simulation with logging
python -m firefly_sync.experiments.runner --model pulse-coupled --drones 3 --log

# Run tests
pytest tests/
```

## Raspberry Pi 5 Visual Flash Detection

This section covers the hardware-in-the-loop testbed: a laptop browser
displays a flashing **leader** target, a Raspberry Pi 5 camera observes
it, and the Pi detects flash events in real time.

### Hardware Setup

- Raspberry Pi 5 with Arducam 8MP (imx219) on CAM/DISP 0
- GPIO17 → physical LED through a current-limiting resistor
- Laptop screen displaying the leader UI
- Pi camera positioned to observe the laptop screen at short range

### 1. Copy Project to Pi

```bash
# From the laptop:
scp -r firefly-sync pi@dronepi.local:~/
```

### 2. Install Pi Dependencies (on the Pi)

**Preferred — apt packages (Raspberry Pi OS):**

```bash
sudo apt install -y python3-flask python3-opencv python3-picamera2 python3-gpiozero
```

**Alternative — pip:**

```bash
pip install flask opencv-python picamera2 gpiozero
```

### 3. Test the LED (on the Pi)

```bash
cd ~/firefly-sync
python experiments/test_pi_led.py --pin 17 --cycles 5
```

The physical LED on GPIO17 should blink 5 times.

### 4. Run Leader UI (on the laptop)

```bash
cd firefly-sync
python experiments/run_leader_ui.py
```

Or open `experiments/leader_ui/index.html` directly in a browser.

### 5. Start Pi Camera Detection Server (on the Pi)

**Recommended — top-percentile mode with auto-ROI:**

```bash
cd ~/firefly-sync
PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
    --host 0.0.0.0 --port 5000 \
    --detection-mode top_percentile --percentile 99 \
    --threshold-on 180 --threshold-off 120 --min-interval 0.2 \
    --auto-roi --auto-roi-duration 3
```

**Manual ROI (if you know the target's pixel position):**

```bash
PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
    --host 0.0.0.0 --port 5000 \
    --detection-mode top_percentile \
    --roi 200 120 240 240 --percentile 99 \
    --threshold-on 180 --threshold-off 120 --min-interval 0.2
```

**Detection modes:**
- `top_percentile` — robust for small targets on dark backgrounds (default)
- `mean` — full-frame or ROI mean brightness
- `bright_blob` — largest bright connected component

### 6. Connect Leader UI to Pi Stream

1. In the leader UI, locate the **Pi Camera Monitor** panel (right side).
2. Ensure the URL field shows `http://dronepi.local:5000` (edit as needed,
   e.g. `http://192.168.1.127:5000`).
3. Press **Connect**.
4. The MJPEG video feed should appear with detection overlay.
5. Press **◎ Auto Locate Flash Region** to automatically find the flashing
   target (keep the leader flashing and the camera still for 3 seconds).
6. Status readouts update in real time.

### Expected Signs of Successful Detection

- **Camera stream visible** in the right panel with ROI rectangle
- **ROI source** shows `auto` or `manual` after auto-localisation
- **Bright (used)** changes from dark to bright when the leader flashes ON
- **Top %ile** brightness is high during ON phases even with a small target
- **State** switches between OFF and ON, synchronized with the leader flash
- **Rising edge count** increases by 1 per ON transition
- **Estimated frequency** approaches the leader UI frequency (e.g. ~1 Hz)
- **CSV log** saved under `experiments/logs/pi_camera_stream_YYYYMMDD_HHMMSS.csv`

### CSV Log Columns

```
timestamp_s, elapsed_time_s, frame_index, detection_mode,
brightness_used, full_frame_mean, top_percentile_brightness,
brightness_mean, state, event_type, rising_edge_count,
estimated_frequency_hz, roi, roi_source, roi_confidence,
percentile, blob_found, blob_area_px, blob_bbox,
threshold_on, threshold_off
```

One row is written **per processed frame**.  The `event_type` column is
empty for normal frames and `"leader_rising_edge"` when a valid rising
edge is detected.

### Troubleshooting: Rising Edge Count Does Not Increase

If the camera stream is visible but `rising_edge_count` stays at 0:

1. **Fix the camera physically** — point it directly at the laptop screen.
2. **Set leader UI to high contrast** — ON brightness 255, OFF brightness 0,
   background brightness 0 (the default).
3. **Use a large target size initially** — 150–250 px square, centred.
4. **Start leader flashing at 1 Hz** with 50% duty cycle.
5. **Run the Pi server with local_contrast + adaptive mode** (recommended):
   ```bash
   PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
       --host 0.0.0.0 --port 5000 \
       --detection-mode local_contrast --auto-roi --auto-roi-duration 3 \
       --window-s 5 --norm-on-threshold 0.65 --norm-off-threshold 0.35 \
       --min-interval 0.2
   ```
6. **Optional — fix camera exposure** to prevent auto-exposure from fighting
   the flashing signal:
   ```bash
   PYTHONPATH=. python3 experiments/stream_pi_camera_detection.py \
       --host 0.0.0.0 --port 5000 \
       --detection-mode local_contrast --auto-roi --auto-roi-duration 3 \
       --manual-camera --exposure-us 8000 --analogue-gain 1.0 \
       --awb-enable false --target-fps 30
   ```
7. **Click "Auto Locate Flash Region"** in the leader UI camera panel.
8. **Check the signal_norm plot** in the camera panel:
   - If the trace is a clear square-ish wave swinging 0→1, detection is working.
   - If the trace is flat → ROI/camera positioning is wrong, or the target
     is not flashing with enough contrast, or camera auto-exposure is
     cancelling out the flash.
   - If the trace is noisy → ambient light or screen reflections are
     interfering; increase target size or move camera closer.
9. **Check adaptive thresholds** in the overlay/readouts:
   - `adaptive_low` should be near the dark-phase `local_contrast` value.
   - `adaptive_high` should be near the bright-phase `local_contrast` value.
   - `adaptive_amplitude` should be comfortably above `min_amplitude` (default 10).
10. **If state is stuck ON**, lower `--norm-off-threshold` (e.g. 0.25) or
    check that the signal drops below `adaptive_low` during the OFF phase.
11. **If state is stuck OFF**, raise `--norm-on-threshold` (e.g. 0.55) or
    check that `adaptive_amplitude` exceeds `--min-amplitude`.

### Interpreting the Signal Norm Plot

The signal_norm plot in the leader UI camera panel shows the last ~10 seconds
of normalised detection signal:

- **Green dashed line at ~0.65** — ON threshold. When the trace crosses above
  this line from below, a rising edge is detected.
- **Red dashed line at ~0.35** — OFF threshold. When the trace drops below
  this line from above, the state switches to OFF.
- **Cyan trace** — the normalised signal (0 = dark, 1 = bright).

A working system shows a square-ish waveform swinging between ~0 and ~1,
crossing both thresholds once per flash cycle.

### Tuning Adaptive Thresholds

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| State stuck OFF, trace swings 0→1 | `norm_on_threshold` too high | Lower `--norm-on-threshold` to 0.55 |
| State stuck ON, trace swings 1→0 | `norm_off_threshold` too low | Raise `--norm-off-threshold` to 0.45 |
| Trace flat near 0 | ROI wrong / no flash in frame / camera dark | Reposition camera, enlarge target, check leader UI is flashing |
| Trace flat near 0.5 | Camera auto-exposure cancelling flash | Use `--manual-camera` with fixed exposure |
| Trace periodic but amplitude small | Target too small or dim | Enlarge target, increase ON brightness |
| `signal_quality` stays 0 | `min_amplitude` too high | Lower `--min-amplitude` or check signal_amplitude readout |
| `periodicity_confidence` low | Signal noisy or irregular | Improve lighting, reduce screen reflections |
| `signal_frequency_hz` near 0 but edges fire | Not enough autocorrelation data | Wait 5+ seconds for history to build |

### Known Limitations

- Laptop and Pi clocks are **not synchronised** — compare detection
  timestamps with Pi-side `perf_counter`, not browser `performance.now()`.
- The browser MJPEG stream is for **debugging only**; future synchronisation
  must use Pi-side detection timestamps.
- MJPEG streaming introduces ~100–200 ms latency; detection is real-time on
  the Pi but display on the laptop lags slightly.
- Fullscreen mode hides both the control panel and the camera panel. Use
  windowed mode to monitor detection while the leader runs.

## Authorship

Built as part of the MSc in Neurotechnology / Bioengineering at Imperial College London.

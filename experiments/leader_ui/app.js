/**
 * Firefly Leader UI — Application Logic
 *
 * Drives a flashing visual-leader target on an HTML canvas.  All timing
 * is derived from performance.now() rather than wall-clock time, so the
 * animation remains robust even if the browser tab is throttled.
 *
 * Planned flash events (state transitions ON→OFF, OFF→ON) are recorded
 * in memory and can be exported as CSV for later comparison with Pi
 * camera logs.
 *
 * Default parameters:
 *   1 Hz, 50 % duty, black background, white 100 px square, centred.
 */

(function () {
    "use strict";

    // ── DOM references ───────────────────────────────────────────
    const canvas       = document.getElementById("leader-canvas");
    const ctx          = canvas.getContext("2d");
    const container    = document.getElementById("canvas-container");

    // Overlay readout chips
    const overlayState   = document.getElementById("overlay-state");
    const overlayFreq    = document.getElementById("overlay-freq");
    const overlayDuty    = document.getElementById("overlay-duty");
    const overlayElapsed = document.getElementById("overlay-elapsed");
    const overlayCycle   = document.getElementById("overlay-cycle");

    // Controls
    const btnStart      = document.getElementById("btn-start");
    const btnStop       = document.getElementById("btn-stop");
    const btnReset         = document.getElementById("btn-reset");
    const btnResetDefaults = document.getElementById("btn-reset-defaults");
    const btnFullscreen    = document.getElementById("btn-fullscreen");
    const btnExport     = document.getElementById("btn-export");
    const eventCount    = document.getElementById("event-count");

    // Sliders & value displays
    const sliderFreq        = document.getElementById("freq");
    const spanFreq          = document.getElementById("freq-val");
    const sliderDuty        = document.getElementById("duty");
    const spanDuty          = document.getElementById("duty-val");
    const sliderBrightOn    = document.getElementById("brightness-on");
    const spanBrightOn      = document.getElementById("brightness-on-val");
    const sliderBrightOff   = document.getElementById("brightness-off");
    const spanBrightOff     = document.getElementById("brightness-off-val");
    const sliderBg          = document.getElementById("bg-brightness");
    const spanBg            = document.getElementById("bg-brightness-val");
    const sliderSize        = document.getElementById("target-size");
    const spanSize          = document.getElementById("target-size-val");
    const selectShape       = document.getElementById("target-shape");
    const sliderX           = document.getElementById("offset-x");
    const spanX             = document.getElementById("offset-x-val");
    const sliderY           = document.getElementById("offset-y");
    const spanY             = document.getElementById("offset-y-val");

    // ── State ────────────────────────────────────────────────────
    let running      = false;      // animation loop active?
    let rafId        = null;       // current requestAnimationFrame handle
    let startTime    = 0;          // performance.now() at last Start
    let elapsedPause = 0;          // accumulated elapsed time from prior runs
    let currentState = "OFF";      // "ON" or "OFF"
    let cycleCount   = 0;          // how many ON→OFF transitions completed
    let cycleOnset   = 0;          // performance.now() when current state began
    let events       = [];         // planned-flash event log (array of objects)

    // Cached parameters (updated from sliders every frame)
    let params = {};

    // Default leader parameter values (maximum contrast).
    var DEFAULTS = Object.freeze({
        frequencyHz:        1.0,
        dutyCycle:          0.5,
        brightnessOn:       255,
        brightnessOff:      0,
        bgBrightness:       0,
        targetSizePx:       100,
        shape:              "circle",
        offsetX:            0,
        offsetY:            0,
    });

    // ── Read parameters from DOM ─────────────────────────────────
    function readParams() {
        params.frequencyHz        = parseFloat(sliderFreq.value);
        params.dutyCycle          = parseFloat(sliderDuty.value);
        params.brightnessOn       = parseInt(sliderBrightOn.value, 10);
        params.brightnessOff      = parseInt(sliderBrightOff.value, 10);
        params.bgBrightness       = parseInt(sliderBg.value, 10);
        params.targetSizePx       = parseInt(sliderSize.value, 10);
        params.shape              = selectShape.value;           // "square" | "circle"
        params.offsetX            = parseInt(sliderX.value, 10);
        params.offsetY            = parseInt(sliderY.value, 10);
    }

    // ── Update the readout labels in the DOM ─────────────────────
    function updateReadouts(state, elapsedSec) {
        // Overlay chips
        overlayFreq.textContent    = params.frequencyHz.toFixed(2) + " Hz";
        overlayDuty.textContent    = "Duty " + params.dutyCycle.toFixed(2);
        overlayCycle.textContent   = "Cycle " + cycleCount;

        if (state) {
            overlayState.textContent   = state;
            overlayState.className     = "overlay-chip overlay-" + state.toLowerCase();
        }

        // Elapsed time as MM:SS.s
        const mins = Math.floor(elapsedSec / 60);
        const secs = (elapsedSec % 60).toFixed(1);
        overlayElapsed.textContent = String(mins).padStart(2, "0") + ":" + String(secs).padStart(4, "0");

        // Slider value labels
        spanFreq.textContent        = params.frequencyHz.toFixed(2);
        spanDuty.textContent        = params.dutyCycle.toFixed(2);
        spanBrightOn.textContent    = String(params.brightnessOn);
        spanBrightOff.textContent   = String(params.brightnessOff);
        spanBg.textContent          = String(params.bgBrightness);
        spanSize.textContent        = String(params.targetSizePx);
        spanX.textContent           = String(params.offsetX);
        spanY.textContent           = String(params.offsetY);
        eventCount.textContent      = events.length + " events recorded";
    }

    // ── Record a planned event ───────────────────────────────────
    function recordEvent(state, elapsedSec, perfNow) {
        events.push({
            event_index:            events.length,
            performance_time_s:     perfNow / 1000,       // seconds since page origin
            elapsed_time_s:         elapsedSec,           // seconds since Start
            state:                  state,
            frequency_hz:           params.frequencyHz,
            duty_cycle:             params.dutyCycle,
            brightness_on:          params.brightnessOn,
            brightness_off:         params.brightnessOff,
            background_brightness:  params.bgBrightness,
            target_size_px:         params.targetSizePx,
        });
    }

    // ── Determine ON/OFF from elapsed time ───────────────────────
    /**
     * Given elapsed seconds since Start and current params, return
     * the canonical state ("ON" | "OFF") and the cycle index.
     *
     * A full cycle lasts 1 / frequencyHz seconds.  Within each cycle
     * the leader is ON for the first (dutyCycle * period) seconds
     * and OFF for the remainder.
     */
    function canonicalState(elapsedSec) {
        if (params.frequencyHz <= 0) return { state: "OFF", cycleIndex: 0 };

        const period     = 1.0 / params.frequencyHz;   // seconds per full cycle
        const cycleIndex = Math.floor(elapsedSec / period);
        const phase      = (elapsedSec / period) - cycleIndex;  // 0..1 within cycle

        const state = (phase < params.dutyCycle) ? "ON" : "OFF";
        return { state, cycleIndex };
    }

    // ── Resize canvas to fill container ──────────────────────────
    function resizeCanvas() {
        const rect = container.getBoundingClientRect();
        const dpr  = window.devicePixelRatio || 1;

        // Set the canvas backing-store dimensions to match the CSS
        // size × device-pixel-ratio so the target looks sharp.
        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
            canvas.width  = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        return { w: rect.width, h: rect.height };
    }

    // ── Draw a single frame ──────────────────────────────────────
    function draw(now) {
        readParams();

        const elapsedSec = running
            ? elapsedPause + (now - startTime) / 1000
            : elapsedPause;

        const { state, cycleIndex } = canonicalState(elapsedSec);

        // Detect state transitions (but only when running)
        if (running && state !== currentState) {
            // ON→OFF transition: increment cycle counter
            if (currentState === "ON" && state === "OFF") {
                cycleCount++;
            }
            currentState = state;
            cycleOnset   = now;
            recordEvent(state, elapsedSec, now);
        }

        // If stopped we still draw, but don't record transitions.
        if (!running) {
            currentState = state;   // keep in sync for display
        }

        // --- Render ---
        const { w, h } = resizeCanvas();

        // Background fill
        const bgVal = params.bgBrightness;
        ctx.fillStyle = `rgb(${bgVal}, ${bgVal}, ${bgVal})`;
        ctx.fillRect(0, 0, w, h);

        // Target brightness
        const brightVal = (state === "ON") ? params.brightnessOn : params.brightnessOff;
        ctx.fillStyle = `rgb(${brightVal}, ${brightVal}, ${brightVal})`;

        // Target position: centre + optional offset
        const cx = w / 2 + params.offsetX;
        const cy = h / 2 + params.offsetY;
        const half = params.targetSizePx / 2;

        // Draw shape
        if (params.shape === "circle") {
            // Glow constants (could be made configurable later)
            const glowMultiplier = 1.8;
            const glowAlpha     = 0.5;    // max alpha of outer glow relative to brightVal

            const glowRadius = half * glowMultiplier;
            const brightAlpha = brightVal / 255.0;  // 0–1

            // --- outer glow (radial gradient from transparent → semi-bright) ---
            const grad = ctx.createRadialGradient(cx, cy, half * 0.6, cx, cy, glowRadius);
            grad.addColorStop(0, `rgba(${brightVal}, ${brightVal}, ${brightVal}, ${brightAlpha})`);
            grad.addColorStop(0.5, `rgba(${brightVal}, ${brightVal}, ${brightVal}, ${brightAlpha * glowAlpha})`);
            grad.addColorStop(1, `rgba(${brightVal}, ${brightVal}, ${brightVal}, 0)`);

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(cx, cy, glowRadius, 0, Math.PI * 2);
            ctx.fill();

            // --- bright inner core ---
            ctx.fillStyle = `rgb(${brightVal}, ${brightVal}, ${brightVal})`;
            ctx.beginPath();
            ctx.arc(cx, cy, half, 0, Math.PI * 2);
            ctx.fill();
        } else {
            // square — plain fill (no glow)
            ctx.fillStyle = `rgb(${brightVal}, ${brightVal}, ${brightVal})`;
            ctx.fillRect(cx - half, cy - half, params.targetSizePx, params.targetSizePx);
        }

        // Update readouts
        updateReadouts(state, elapsedSec);
    }

    // ── Animation loop ───────────────────────────────────────────
    function loop(now) {
        if (!running) return;           // stopped from outside
        draw(now);
        rafId = requestAnimationFrame(loop);
    }

    // ── Start ────────────────────────────────────────────────────
    function start() {
        if (running) return;

        // Reset the clock origin so elapsed time continues smoothly.
        startTime = performance.now();
        running   = true;

        // Record the initial state at t=0 if there are no events yet.
        if (events.length === 0) {
            readParams();
            const initial = canonicalState(elapsedPause);
            currentState  = initial.state;
            cycleOnset    = startTime;
            recordEvent(currentState, elapsedPause, startTime);
        }

        rafId = requestAnimationFrame(loop);
    }

    // ── Stop ─────────────────────────────────────────────────────
    function stop() {
        if (!running) return;
        running = false;

        // Freeze elapsed time at the moment we stopped.
        if (rafId !== null) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }

        // Account for the time that just passed.
        const now = performance.now();
        elapsedPause += (now - startTime) / 1000;
        startTime = now;

        // Re-draw one last frame so the readout reflects the stopped
        // state and the target shows its current brightness.
        draw(now);
    }

    // ── Reset ────────────────────────────────────────────────────
    function reset() {
        stop();
        elapsedPause  = 0;
        startTime     = 0;
        currentState  = "OFF";
        cycleCount    = 0;
        cycleOnset    = 0;
        events        = [];
        draw(performance.now());
    }

    // ── Apply defaults to UI sliders / select ────────────────────
    function applyDefaultsToUI(defaults) {
        sliderFreq.value        = defaults.frequencyHz;
        sliderDuty.value        = defaults.dutyCycle;
        sliderBrightOn.value    = defaults.brightnessOn;
        sliderBrightOff.value   = defaults.brightnessOff;
        sliderBg.value          = defaults.bgBrightness;
        sliderSize.value        = defaults.targetSizePx;
        selectShape.value       = defaults.shape;
        sliderX.value           = defaults.offsetX;
        sliderY.value           = defaults.offsetY;
    }

    // ── Reset to Defaults ────────────────────────────────────────
    /**
     * Reset all leader parameters to their default values (maximum
     * contrast: black background, white 100 px square, 1 Hz, 50 %).
     *
     * - If the animation is running it stays running; parameters
     *   take effect on the next frame.
     * - If stopped, the display is repainted with the new defaults
     *   but the animation is NOT automatically started.
     */
    function resetToDefaults() {
        var wasRunning = running;

        applyDefaultsToUI(DEFAULTS);
        readParams();

        if (!wasRunning) {
            // Re-draw to show the new defaults immediately.
            draw(performance.now());
        }
        // If running, no extra action needed — the next animation
        // frame reads the updated slider values automatically.
    }

    // ── Fullscreen toggle ────────────────────────────────────────
    function toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
            document.body.classList.remove("fullscreen");
        } else {
            document.body.requestFullscreen().then(function () {
                document.body.classList.add("fullscreen");
            }).catch(function (err) {
                console.warn("Fullscreen request denied:", err);
            });
        }
    }

    // Also listen for Esc / fullscreenchange to keep the CSS class in sync.
    document.addEventListener("fullscreenchange", function () {
        if (!document.fullscreenElement) {
            document.body.classList.remove("fullscreen");
        }
    });

    // ── Export CSV ───────────────────────────────────────────────
    function exportCSV() {
        if (events.length === 0) {
            alert("No events recorded yet. Start the leader to generate events.");
            return;
        }

        // Build CSV string
        const headers = [
            "event_index",
            "performance_time_s",
            "elapsed_time_s",
            "state",
            "frequency_hz",
            "duty_cycle",
            "brightness_on",
            "brightness_off",
            "background_brightness",
            "target_size_px",
        ];

        const rows = events.map(function (e) {
            return [
                e.event_index,
                e.performance_time_s.toFixed(6),
                e.elapsed_time_s.toFixed(6),
                e.state,
                e.frequency_hz.toFixed(3),
                e.duty_cycle.toFixed(3),
                e.brightness_on,
                e.brightness_off,
                e.background_brightness,
                e.target_size_px,
            ].join(",");
        });

        const csv = headers.join(",") + "\n" + rows.join("\n");

        // Trigger download
        const blob    = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url     = URL.createObjectURL(blob);
        const link    = document.createElement("a");
        const ts      = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        link.href     = url;
        link.download = "leader_events_" + ts + ".csv";
        link.click();
        URL.revokeObjectURL(url);
    }

    // ── Bind UI events ───────────────────────────────────────────
    btnStart.addEventListener("click", start);
    btnStop.addEventListener("click", stop);
    btnReset.addEventListener("click", reset);
    btnResetDefaults.addEventListener("click", resetToDefaults);
    btnFullscreen.addEventListener("click", toggleFullscreen);
    btnExport.addEventListener("click", exportCSV);

    // Keyboard shortcut: Space toggles Start/Stop
    document.addEventListener("keydown", function (e) {
        // Ignore keypresses when focus is on an input/select/button.
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "BUTTON") return;
        if (e.code === "Space") {
            e.preventDefault();
            if (running) { stop(); } else { start(); }
        }
    });

    // ── Remote API polling (for Pi batch control) ──────────────
    var overlayApi       = document.getElementById("overlay-api");
    var lastApiConfig    = null;
    var apiPollTimerId   = null;

    function applyApiConfigToSliders(cfg) {
        if (cfg.frequency_hz != null)           sliderFreq.value      = cfg.frequency_hz;
        if (cfg.duty_cycle != null)             sliderDuty.value      = cfg.duty_cycle;
        if (cfg.brightness_on != null)          sliderBrightOn.value  = cfg.brightness_on;
        if (cfg.brightness_off != null)         sliderBrightOff.value = cfg.brightness_off;
        if (cfg.background_brightness != null)  sliderBg.value        = cfg.background_brightness;
        if (cfg.target_size_px != null)         sliderSize.value      = cfg.target_size_px;
        if (cfg.shape != null)                  selectShape.value     = cfg.shape;
        if (cfg.offset_x != null)               sliderX.value         = cfg.offset_x;
        if (cfg.offset_y != null)               sliderY.value         = cfg.offset_y;
        readParams();

        // Handle run/stop if different from current state
        if (cfg.running === true && !running) {
            start();
        } else if (cfg.running === false && running) {
            stop();
        }
    }

    function pollApiConfig() {
        fetch("/api/leader/config")
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                if (cfg.api_controlled) {
                    overlayApi.style.display = "inline";
                    var cfgStr = JSON.stringify({
                        frequency_hz: cfg.frequency_hz, duty_cycle: cfg.duty_cycle,
                        brightness_on: cfg.brightness_on, target_size_px: cfg.target_size_px,
                        running: cfg.running,
                    });
                    if (cfgStr !== lastApiConfig) {
                        lastApiConfig = cfgStr;
                        applyApiConfigToSliders(cfg);
                        console.log("[API] Config applied:", cfg);
                    }
                } else {
                    overlayApi.style.display = "none";
                }
            })
            .catch(function () {
                // Server not available — not an error
                overlayApi.style.display = "none";
            });
    }

    // Poll every 500 ms
    apiPollTimerId = setInterval(pollApiConfig, 500);
    pollApiConfig();  // immediate first poll

    // ── Initialise ───────────────────────────────────────────────
    readParams();
    resizeCanvas();
    draw(performance.now());   // single draw so the page isn't blank
})();

// ====================================================================
// Camera Module — connects to the Pi MJPEG stream server.
// Handles video feed display, status polling, and readout updates.
// ====================================================================
(function () {
    "use strict";

    // ── DOM references ───────────────────────────────────────────
    var urlInput           = document.getElementById("pi-url");
    var btnConnect         = document.getElementById("btn-connect");
    var btnDisconnect      = document.getElementById("btn-disconnect");
    var connectionStatus   = document.getElementById("cam-connection-status");
    var mjpegImg           = document.getElementById("mjpeg-stream");
    var placeholder        = document.getElementById("stream-placeholder");

    // Button refs
    var btnAutoROI  = document.getElementById("btn-auto-roi");
    var btnClearROI = document.getElementById("btn-clear-roi");

    // Status readout spans
    var camBrightness = document.getElementById("cam-brightness");
    var camState      = document.getElementById("cam-state");
    var camEdges      = document.getElementById("cam-edges");
    var camFreq       = document.getElementById("cam-freq");
    var camLastEdge   = document.getElementById("cam-last-edge");
    var camFps        = document.getElementById("cam-fps");
    var camDetMode    = document.getElementById("cam-det-mode");
    var camBrightUsed = document.getElementById("cam-bright-used");
    var camFullMean   = document.getElementById("cam-full-mean");
    var camTopPct     = document.getElementById("cam-top-pct");
    var camRoiSource  = document.getElementById("cam-roi-source");
    var camRoiConf    = document.getElementById("cam-roi-conf");
    // Stage 2D new readouts
    var camLocalContrast = document.getElementById("cam-local-contrast");
    var camSignalNorm    = document.getElementById("cam-signal-norm");
    var camAdaptiveLoHi  = document.getElementById("cam-adaptive-lo-hi");
    var camAdaptiveAmp   = document.getElementById("cam-adaptive-amp");
    var camSignalQuality = document.getElementById("cam-signal-quality");
    var camSignalFreq    = document.getElementById("cam-signal-freq");
    var camPeriodConf    = document.getElementById("cam-period-conf");
    var camManualCamera  = document.getElementById("cam-manual-camera");
    // Signal plot
    var signalPlotCanvas      = document.getElementById("signal-plot-canvas");
    var signalPlotCtx         = signalPlotCanvas.getContext("2d");
    var signalPlotPlaceholder = document.getElementById("signal-plot-placeholder");
    var signalHistory = [];  // {signal_norm, state}

    // ── State ────────────────────────────────────────────────────
    var pollTimerId = null;
    var connected   = false;

    // ── Get base URL (trim trailing slash) ───────────────────────
    function baseUrl() {
        return urlInput.value.replace(/\/+$/, "");
    }

    // ── Poll /status endpoint ────────────────────────────────────
    function pollStatus() {
        if (!connected) return;

        var url = baseUrl() + "/status";
        fetch(url)
            .then(function (resp) {
                if (!resp.ok) throw new Error("HTTP " + resp.status);
                return resp.json();
            })
            .then(function (data) {
                // Brightness
                if (data.brightness_mean != null) {
                    camBrightness.textContent = data.brightness_mean.toFixed(1);
                }

                // State (ON/OFF)
                if (data.state) {
                    camState.textContent = data.state;
                    camState.style.color = (data.state === "ON") ? "#0f0" : "#f44";
                }

                // Rising edges
                if (data.rising_edge_count != null) {
                    camEdges.textContent = data.rising_edge_count;
                }

                // Estimated frequency
                if (data.estimated_frequency_hz != null) {
                    camFreq.textContent = data.estimated_frequency_hz.toFixed(2) + " Hz";
                }

                // Last rising edge
                if (data.last_rising_edge_time_s != null && data.server_time_s != null) {
                    var elapsed = data.server_time_s - data.last_rising_edge_time_s;
                    camLastEdge.textContent = elapsed.toFixed(3) + " s ago";
                } else if (data.last_rising_edge_time_s == null) {
                    camLastEdge.textContent = "--";
                }

                // FPS estimate
                if (data.fps_estimate != null) {
                    camFps.textContent = data.fps_estimate.toFixed(1);
                }

                // New Stage 2C fields
                if (data.detection_mode != null) {
                    camDetMode.textContent = data.detection_mode;
                }
                if (data.brightness_used != null) {
                    camBrightUsed.textContent = data.brightness_used.toFixed(1);
                }
                if (data.full_frame_mean != null) {
                    camFullMean.textContent = data.full_frame_mean.toFixed(1);
                }
                if (data.top_percentile_brightness != null) {
                    camTopPct.textContent = data.top_percentile_brightness.toFixed(1);
                }
                if (data.roi_source != null) {
                    camRoiSource.textContent = data.roi_source;
                }
                if (data.roi_confidence != null) {
                    camRoiConf.textContent = (data.roi_confidence * 100).toFixed(0) + "%";
                }

                // Stage 2D fields
                if (data.local_contrast != null) {
                    camLocalContrast.textContent = data.local_contrast.toFixed(1);
                }
                if (data.signal_norm != null) {
                    camSignalNorm.textContent = data.signal_norm.toFixed(3);
                }
                if (data.adaptive_low != null && data.adaptive_high != null) {
                    camAdaptiveLoHi.textContent = data.adaptive_low.toFixed(0) + " / " + data.adaptive_high.toFixed(0);
                }
                if (data.adaptive_amplitude != null) {
                    camAdaptiveAmp.textContent = data.adaptive_amplitude.toFixed(1);
                }
                if (data.signal_quality != null) {
                    camSignalQuality.textContent = data.signal_quality.toFixed(2);
                }
                if (data.signal_frequency_hz != null) {
                    camSignalFreq.textContent = data.signal_frequency_hz.toFixed(2) + " Hz";
                }
                if (data.periodicity_confidence != null) {
                    camPeriodConf.textContent = data.periodicity_confidence.toFixed(2);
                }
                if (data.manual_camera !== undefined) {
                    camManualCamera.textContent = data.manual_camera ? "manual" : "auto";
                }

                // Update signal history for plot
                if (data.signal_norm != null) {
                    signalHistory.push({
                        signal_norm: Math.max(0, Math.min(1, data.signal_norm)),
                        state: data.state || "OFF"
                    });
                    // Keep last ~10 seconds at 5 Hz poll rate = 50 points
                    // But we also want enough for a smooth trace. Keep 300 points.
                    if (signalHistory.length > 300) {
                        signalHistory.shift();
                    }
                    drawSignalPlot();
                }

                // Connection status text
                if (data.connected) {
                    connectionStatus.textContent = "Connected";
                    connectionStatus.className = "hint status-connected";
                }

                // Schedule next poll (~5 Hz)
                if (connected) {
                    pollTimerId = setTimeout(pollStatus, 200);
                }
            })
            .catch(function (err) {
                console.warn("Camera status fetch failed:", err);
                connectionStatus.textContent = "Connection lost";
                connectionStatus.className = "hint status-disconnected";
                // Retry after a short delay
                if (connected) {
                    pollTimerId = setTimeout(pollStatus, 1000);
                }
            });
    }

    // ── Connect ──────────────────────────────────────────────────
    function connect() {
        if (connected) return;

        var url = baseUrl();
        connectionStatus.textContent = "Connecting…";
        connectionStatus.className = "hint status-connecting";

        // Quick health check before setting up stream
        fetch(url + "/health")
            .then(function (resp) { return resp.json(); })
            .then(function () {
                // Server is reachable — show MJPEG stream
                mjpegImg.src = url + "/video_feed?" + Date.now();
                mjpegImg.style.display = "block";
                placeholder.style.display = "none";

                // Show signal plot
                signalPlotCanvas.style.display = "block";
                signalPlotPlaceholder.style.display = "none";

                connected = true;
                connectionStatus.textContent = "Connected";
                connectionStatus.className = "hint status-connected";

                // Start polling status
                pollStatus();
            })
            .catch(function (err) {
                console.warn("Cannot reach Pi server:", err);
                connectionStatus.textContent = "Cannot reach " + url;
                connectionStatus.className = "hint status-disconnected";
            });
    }

    // ── Draw signal plot ───────────────────────────────────────
    function drawSignalPlot() {
        var w = signalPlotCanvas.width;
        var h = signalPlotCanvas.height;
        var ctx = signalPlotCtx;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, w, h);

        if (signalHistory.length < 2) {
            ctx.fillStyle = "#555";
            ctx.font = "11px monospace";
            ctx.textAlign = "center";
            ctx.fillText("waiting for data...", w / 2, h / 2);
            return;
        }

        // Grid lines
        ctx.strokeStyle = "#222";
        ctx.lineWidth = 0.5;
        for (var g = 0; g <= 1; g += 0.25) {
            var gy = h - g * h;
            ctx.beginPath();
            ctx.moveTo(0, gy); ctx.lineTo(w, gy);
            ctx.stroke();
        }

        // Threshold lines
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "#0f0";
        ctx.lineWidth = 1;
        var yOn = h - 0.65 * h;
        ctx.beginPath(); ctx.moveTo(0, yOn); ctx.lineTo(w, yOn); ctx.stroke();

        ctx.strokeStyle = "#f44";
        var yOff = h - 0.35 * h;
        ctx.beginPath(); ctx.moveTo(0, yOff); ctx.lineTo(w, yOff); ctx.stroke();
        ctx.setLineDash([]);

        // Signal trace (last N points)
        var maxPoints = 300;
        var start = Math.max(0, signalHistory.length - maxPoints);
        var points = signalHistory.slice(start);

        ctx.strokeStyle = "#0cf";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (var i = 0; i < points.length; i++) {
            var x = (i / maxPoints) * w;
            var y = h - points[i].signal_norm * h;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // ── Auto Locate Flash Region ────────────────────────────────
    function autoLocateROI() {
        if (!connected) {
            alert("Connect to the Pi server first.");
            return;
        }
        var url = baseUrl() + "/auto_roi";
        connectionStatus.textContent = "Calibrating… keep camera still, leader flashing";
        connectionStatus.className = "hint status-connecting";

        fetch(url)
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (data.success) {
                    connectionStatus.textContent = "Connected — ROI: " +
                        (data.roi ? data.roi.x + "," + data.roi.y + " " + data.roi.width + "x" + data.roi.height : "?") +
                        " conf=" + (data.confidence * 100).toFixed(0) + "%";
                    connectionStatus.className = "hint status-connected";
                } else {
                    connectionStatus.textContent = "Auto-ROI failed: " + (data.message || "no region found");
                    connectionStatus.className = "hint status-disconnected";
                }
            })
            .catch(function (err) {
                console.warn("Auto-ROI failed:", err);
                connectionStatus.textContent = "Auto-ROI request failed";
                connectionStatus.className = "hint status-disconnected";
            });
    }

    // ── Clear ROI ────────────────────────────────────────────────
    function clearROI() {
        if (!connected) return;
        var url = baseUrl() + "/clear_roi";
        fetch(url)
            .then(function (resp) { return resp.json(); })
            .then(function (data) {
                if (data.success) {
                    connectionStatus.textContent = "Connected — ROI cleared";
                    connectionStatus.className = "hint status-connected";
                }
            })
            .catch(function (err) {
                console.warn("Clear ROI failed:", err);
            });
    }

    // ── Disconnect ───────────────────────────────────────────────
    function disconnect() {
        connected = false;

        if (pollTimerId !== null) {
            clearTimeout(pollTimerId);
            pollTimerId = null;
        }

        // Clear the MJPEG stream
        mjpegImg.src = "";
        mjpegImg.style.display = "none";
        placeholder.style.display = "flex";

        connectionStatus.textContent = "Disconnected";
        connectionStatus.className = "hint status-disconnected";

        // Reset readouts
        camBrightness.textContent = "--";
        camState.textContent      = "--";
        camState.style.color      = "#0cf";
        camEdges.textContent      = "0";
        camFreq.textContent       = "-- Hz";
        camLastEdge.textContent   = "--";
        camFps.textContent        = "--";
        camDetMode.textContent    = "--";
        camBrightUsed.textContent = "--";
        camFullMean.textContent   = "--";
        camTopPct.textContent     = "--";
        camRoiSource.textContent  = "--";
        camRoiConf.textContent    = "--";
        camLocalContrast.textContent = "--";
        camSignalNorm.textContent    = "--";
        camAdaptiveLoHi.textContent  = "--";
        camAdaptiveAmp.textContent   = "--";
        camSignalQuality.textContent = "--";
        camSignalFreq.textContent    = "-- Hz";
        camPeriodConf.textContent    = "--";
        camManualCamera.textContent  = "auto";

        // Reset plot
        signalHistory = [];
        drawSignalPlot();
        signalPlotCanvas.style.display = "none";
        signalPlotPlaceholder.style.display = "flex";
    }

    // ── Bind UI events ───────────────────────────────────────────
    btnConnect.addEventListener("click", connect);
    btnDisconnect.addEventListener("click", disconnect);
    btnAutoROI.addEventListener("click", autoLocateROI);
    btnClearROI.addEventListener("click", clearROI);

    // Enter key in the URL field triggers connect
    urlInput.addEventListener("keydown", function (e) {
        if (e.code === "Enter") {
            e.preventDefault();
            connect();
        }
    });

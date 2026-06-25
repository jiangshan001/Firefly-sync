/**
 * Firefly Mutual HIL — Virtual Agent Display (v3).
 *
 * Calibration mode (feedback OFF): uses local deterministic
 * requestAnimationFrame + performance.now() for smooth 2 Hz flashing.
 * Feedback ON mode: uses server flash_on from /api/agents snapshot.
 */
(function () {
    "use strict";

    // ── DOM refs ───────────────────────────────────────────────────
    var canvas       = document.getElementById("hil-canvas");
    var ctx          = canvas.getContext("2d");
    var container    = document.getElementById("canvas-container");
    var overlayState = document.getElementById("hil-overlay-state");
    var overlayFreq  = document.getElementById("hil-overlay-freq");
    var overlayCount = document.getElementById("hil-overlay-count");
    var overlayHeartbeat = document.getElementById("hil-overlay-heartbeat");
    var overlayErrors = document.getElementById("hil-overlay-errors");

    var hilModelSelect   = document.getElementById("hil-model-select");
    var hilAgentModeSelect = document.getElementById("hil-agent-mode-select");
    var hilTopologySelect = document.getElementById("hil-topology-select");
    var hilAgentSelect   = document.getElementById("hil-agent-select");
    var hilStatus        = document.getElementById("hil-status");
    var hilAgentX        = document.getElementById("hil-agent-x");
    var hilAgentY        = document.getElementById("hil-agent-y");
    var hilAgentSize     = document.getElementById("hil-agent-size");
    var hilAgentFreq     = document.getElementById("hil-agent-freq");
    var hilAgentXVal     = document.getElementById("hil-agent-x-val");
    var hilAgentYVal     = document.getElementById("hil-agent-y-val");
    var hilAgentSizeVal  = document.getElementById("hil-agent-size-val");
    var hilAgentFreqVal  = document.getElementById("hil-agent-freq-val");
    var hilAgentCurFreq  = document.getElementById("hil-agent-cur-freq");
    var hilAgentPhase    = document.getElementById("hil-agent-phase");
    var hilAgentFireCnt  = document.getElementById("hil-agent-fire-count");
    var hilAgentPiFlash  = document.getElementById("hil-agent-pi-flashes");
    var btnHilStart      = document.getElementById("btn-hil-start");
    var btnHilPause      = document.getElementById("btn-hil-pause");
    var btnHilReset      = document.getElementById("btn-hil-reset");
    var btnDisplayOnly   = document.getElementById("btn-display-only");
    var btnFeedbackTgl   = document.getElementById("btn-feedback-toggle");
    var hilFeedbackLabel = document.getElementById("hil-feedback-label");
    var debugLabel       = document.getElementById("hil-debug-label");

    // ── State ──────────────────────────────────────────────────────
    var agents          = [];
    var selectedAgentId = 0;
    var pollTimerId     = null;
    var rafId           = null;
    var dragging        = false;
    var cw = 0, ch = 0;

    // Local calibration clock
    var visualStartTimeMs = 0;      // performance.now() when Start pressed
    var localRunning       = false;
    var localFeedbackOn    = false;
    var localDutyCycle     = 0.5;
    var localFrequencyHz   = 2.0;
    var localDotX          = 800;
    var localDotY          = 400;
    var localDotSize       = 350;
    var localFireCount     = 0;
    var localLastFireMs    = 0;
    var localFireIntervals = [];     // last 20 inter-fire intervals (ms)
    var fbServerPollSamples = 0;
    var fbServerFlashOnSamples = 0;
    var apiPollErrorCount = 0;
    var lastApiPollMs = 0;
    var lastRenderMs = 0;
    var renderFrameCount = 0;
    var lastAgentSnapshot = null;
    var lastDisplayStatePostMs = 0;

    function virtualAgents() {
        return agents.filter(function(a) { return (a.role || "virtual") === "virtual"; });
    }

    function selectedAgent() {
        var vids = virtualAgents();
        if (vids.length === 0) return null;
        for (var i = 0; i < vids.length; i++) {
            if (parseInt(vids[i].id, 10) === selectedAgentId) return vids[i];
        }
        return vids[0];
    }

    function refreshAgentSelect() {
        if (!hilAgentSelect) return;
        var current = String(selectedAgentId);
        var vids = virtualAgents();
        hilAgentSelect.innerHTML = "";
        vids.forEach(function(a) {
            var opt = document.createElement("option");
            opt.value = String(a.id);
            opt.textContent = a.agent_id || ("V" + a.id);
            hilAgentSelect.appendChild(opt);
        });
        if (vids.length > 0) {
            var hasCurrent = vids.some(function(a) { return String(a.id) === current; });
            selectedAgentId = hasCurrent ? selectedAgentId : parseInt(vids[0].id, 10);
            hilAgentSelect.value = String(selectedAgentId);
        }
    }

    // ── API helpers ───────────────────────────────────────────────
    function apiGet(path, cb, errCb) {
        fetch(path)
            .then(function(r){
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(cb)
            .catch(function(err){
                apiPollErrorCount++;
                console.warn("[MUTUAL] GET " + path + " failed", err);
                if (errCb) errCb(err);
            });
    }
    function apiPost(path, data, cb) {
        fetch(path, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data||{})})
            .then(function(r){
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(cb||function(){})
            .catch(function(err){
                console.warn("[MUTUAL] POST " + path + " failed", err);
            });
    }

    // ── Force visible on load ─────────────────────────────────────
    document.body.classList.remove("display-only");
    var hilPanel = document.getElementById("mutual-hil-panel");
    if (hilPanel) hilPanel.style.display = "flex";
    if (debugLabel) debugLabel.style.display = "block";
    console.log("[MUTUAL v3] Loaded — calibration local clock active");

    // ── Server polling (~10 Hz for config, not flash timing) ──────
    function formatAge(ms) {
        if (!ms) return "--";
        return ((window.performance.now() - ms) / 1000.0).toFixed(1) + "s";
    }

    function updateHeartbeatOverlay() {
        if (overlayHeartbeat) {
            overlayHeartbeat.textContent = "poll: " + formatAge(lastApiPollMs) +
                " raf:" + (renderFrameCount > 0 ? "on" : "--");
        }
        if (overlayErrors) {
            overlayErrors.textContent = "err: " + apiPollErrorCount;
        }
    }

    function scheduleNextPoll() {
        pollTimerId = setTimeout(pollServerConfig, 100); // 10 Hz
    }

    function pollServerConfig() {
        fetch("/api/agents")
            .then(function(r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(function(data) {
            if (data && data.agents && data.agents.length > 0) {
                agents = data.agents;
                refreshAgentSelect();
                var sa = selectedAgent() || data.agents[0];
                lastAgentSnapshot = sa;
                lastApiPollMs = window.performance.now();
                if (hilAgentModeSelect && sa.mutual_agent_mode) {
                    hilAgentModeSelect.value = sa.mutual_agent_mode;
                }
                if (hilTopologySelect && sa.topology && hilTopologySelect.querySelector("option[value='" + sa.topology + "']")) {
                    hilTopologySelect.value = sa.topology;
                }
                // Update local config from server
                var previousRunning = localRunning;
                localRunning      = sa.running !== false;
                localFeedbackOn   = sa.feedback_enabled === true;
                localFrequencyHz  = sa.initial_frequency_hz || 2.0;
                localDotX         = sa.x || 0;
                localDotY         = sa.y || 0;
                localDotSize      = sa.size || 200;
                if (localRunning && !previousRunning) {
                    var elapsedMs = ((sa.calibration_elapsed || 0) * 1000.0);
                    visualStartTimeMs = window.performance.now() - elapsedMs;
                    localFireCount = 0;
                    localLastFireMs = 0;
                    localFireIntervals = [];
                }
                // Update UI readouts
                overlayFreq.textContent  = (sa.frequency_hz || 0).toFixed(2) + " Hz";
                overlayCount.textContent = "fire: " + (sa.fire_count || 0);
                hilAgentCurFreq.textContent = (sa.frequency_hz || 0).toFixed(2) + " Hz";
                hilAgentPhase.textContent   = (sa.phase_rad || 0).toFixed(2) + " rad";
                hilAgentFireCnt.textContent = sa.fire_count || 0;
                hilAgentPiFlash.textContent = sa.received_pi_flashes || 0;
                hilFeedbackLabel.textContent = localFeedbackOn ? "ON" : "OFF";
                hilFeedbackLabel.style.color = localFeedbackOn ? "#0f0" : "#f80";
                btnFeedbackTgl.textContent = localFeedbackOn ? "Feedback: ON (Mutual)" : "Feedback: OFF (Calibration)";
                if (localFeedbackOn) {
                    fbServerPollSamples++;
                    if (sa.flash_on === true) fbServerFlashOnSamples++;
                    if (debugLabel) {
                        debugLabel.textContent = "FB_ON server flash_on " +
                            fbServerFlashOnSamples + "/" + fbServerPollSamples +
                            " running=" + localRunning +
                            " freq=" + ((sa.frequency_hz || 0).toFixed ? sa.frequency_hz.toFixed(3) : "n/a") +
                            " phase=" + ((sa.phase_rad || 0).toFixed ? sa.phase_rad.toFixed(3) : "n/a") +
                            " raw=" + (sa.raw_flash_on === true ? "1" : "0") +
                            " last_flash_time=" + ((sa.last_flash_time || 0).toFixed ? sa.last_flash_time.toFixed(3) : "0.000") +
                            " hold_s=" + ((sa.feedback_flash_hold_s || 0).toFixed ? sa.feedback_flash_hold_s.toFixed(2) : "0.00") +
                            " poll_err=" + apiPollErrorCount;
                    }
                    console.debug("[MUTUAL HB]", {
                        lastApiPollAgeS: formatAge(lastApiPollMs),
                        running: localRunning,
                        feedback_enabled: localFeedbackOn,
                        frequency_hz: sa.frequency_hz,
                        phase_rad: sa.phase_rad,
                        fire_count: sa.fire_count,
                        flash_on: sa.flash_on,
                        raw_flash_on: sa.raw_flash_on,
                        render_loop_active: renderFrameCount > 0,
                        api_poll_error_count: apiPollErrorCount
                    });
                }
            }
            updateHeartbeatOverlay();
        })
        .catch(function(err) {
            apiPollErrorCount++;
            console.warn("[MUTUAL] /api/agents poll failed; keeping render loop alive", err);
            updateHeartbeatOverlay();
        })
        .finally(scheduleNextPoll);
    }

    // ── Canvas rendering ──────────────────────────────────────────
    function resizeCanvas() {
        var rect = container.getBoundingClientRect();
        var dpr = window.devicePixelRatio || 1;
        if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
            canvas.width  = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }
        cw = rect.width;
        ch = rect.height;
    }

    function clampBrightness(value, fallback) {
        var v = Number(value);
        if (!isFinite(v)) v = fallback;
        return Math.max(0, Math.min(255, Math.round(v)));
    }

    function brightnessCss(value) {
        var v = clampBrightness(value, 0);
        return "rgb(" + v + "," + v + "," + v + ")";
    }

    function maybePostDisplayState(bg, vids) {
        var nowMs = window.performance.now();
        if (nowMs - lastDisplayStatePostMs < 500) return;
        lastDisplayStatePostMs = nowMs;
        var renderedAgents = vids.map(function(agent) {
            return {
                id: agent.id,
                agent_id: agent.agent_id || ("V" + agent.id),
                role: agent.role || "virtual",
                x: agent.x || localDotX,
                y: agent.y || localDotY,
                size: agent.size || localDotSize,
                brightness_on: clampBrightness(agent.brightness_on, 255),
                brightness_off: clampBrightness(agent.brightness_off, 0),
                background_brightness: bg,
                initial_frequency_hz: agent.initial_frequency_hz || localFrequencyHz,
                initial_phase_rad: agent.initial_phase_rad || 0,
                enabled: agent.enabled !== false,
                current_draw_brightness: localFlashBrightness(agent)
            };
        });
        apiPost("/api/frontend/display_state", {
            page: "mutual",
            renderer_version: 4,
            render_frame_count: renderFrameCount,
            local_running: localRunning,
            feedback_enabled: localFeedbackOn,
            background_brightness: bg,
            background_css: brightnessCss(bg),
            agents: renderedAgents,
            client_timestamp_ms: Date.now()
        });
    }

    function localFlashBrightness(agent) {
        agent = agent || selectedAgent();
        var brightOn = clampBrightness(agent && agent.brightness_on, 255);
        var brightOff = clampBrightness(agent && agent.brightness_off, 15);
        if (agent && agent.enabled === false) return brightOff;
        // Calibration mode: deterministic local clock
        if (localRunning && !localFeedbackOn) {
            var freq = (agent && agent.initial_frequency_hz) || localFrequencyHz;
            var periodMs = 1000.0 / freq;
            if (periodMs <= 0) return brightOff;
            var nowMs = window.performance.now();
            var elapsedMs = nowMs - visualStartTimeMs;
            var phaseOffset = ((agent && agent.initial_phase_rad) || 0) / (2 * Math.PI);
            var cyclePhase = ((elapsedMs / periodMs) + phaseOffset) % 1.0;
            var isOn = cyclePhase < localDutyCycle;
            // Track local fire count
            if (isOn && localLastFireMs === 0) {
                localLastFireMs = nowMs;
                localFireCount = 1;
            } else if (isOn) {
                var interval = nowMs - localLastFireMs;
                if (interval > periodMs * 0.8) {  // new fire cycle
                    localFireCount++;
                    localFireIntervals.push(interval);
                    if (localFireIntervals.length > 20) localFireIntervals.shift();
                    localLastFireMs = nowMs;
                }
            }
            return isOn ? brightOn : brightOff;
        }
        // Feedback mode: use server flash_on
        if (localRunning && localFeedbackOn && agent) {
            return agent.flash_on ? brightOn : brightOff;
        }
        // Not running: dim
        return brightOff;
    }

    function drawAgent(agent, bright) {
        var cx = agent.x || localDotX;
        var cy = ch - (agent.y || localDotY);
        var r  = ((agent.size || localDotSize || 100) / 2);
        var grad = ctx.createRadialGradient(cx, cy, r * 0.5, cx, cy, r * 1.5);
        grad.addColorStop(0, "rgba(" + bright + "," + bright + "," + bright + ",1)");
        grad.addColorStop(0.6, "rgba(" + bright + "," + bright + "," + bright + ",0.4)");
        grad.addColorStop(1, "rgba(" + bright + "," + bright + "," + bright + ",0)");
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, r * 1.5, 0, Math.PI*2); ctx.fill();

        ctx.fillStyle = "rgb(" + bright + "," + bright + "," + bright + ")";
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI*2); ctx.fill();
        ctx.fillStyle = bright > 200 ? "#111" : "#aaa";
        ctx.font = "14px Consolas, monospace";
        ctx.textAlign = "center";
        ctx.fillText(agent.agent_id || ("V" + agent.id), cx, cy + 5);
    }

    function renderFrame() {
        lastRenderMs = window.performance.now();
        renderFrameCount++;
        resizeCanvas();
        var vids = virtualAgents();
        var bg = vids.length ? clampBrightness(vids[0].background_brightness, 0) : 0;
        var bgCss = brightnessCss(bg);
        document.body.style.backgroundColor = bgCss;
        container.style.backgroundColor = bgCss;
        canvas.style.backgroundColor = bgCss;
        ctx.fillStyle = bgCss;
        ctx.fillRect(0, 0, cw, ch);

        if (vids.length === 0) {
            vids = [{id: 0, agent_id: "V0", x: localDotX, y: localDotY, size: localDotSize, initial_frequency_hz: localFrequencyHz}];
        }
        vids.forEach(function(agent) {
            drawAgent(agent, localFlashBrightness(agent));
        });
        maybePostDisplayState(bg, vids);

        // Update overlay
        var brightVal = localFlashBrightness(selectedAgent());
        overlayState.textContent = (brightVal > 200) ? "ON" : "OFF";
        overlayState.className = "overlay-chip overlay-" + ((brightVal > 200) ? "on" : "off");
        updateHeartbeatOverlay();

        // Local timing debug
        if (localRunning && !localFeedbackOn && localFireIntervals.length > 0) {
            var sum = 0;
            for (var j = 0; j < localFireIntervals.length; j++) sum += localFireIntervals[j];
            var meanI = sum / localFireIntervals.length;
            var maxI = Math.max.apply(null, localFireIntervals);
            hilStatus.textContent = "CAL: fire=" + localFireCount + " meanI=" + meanI.toFixed(0) + "ms maxI=" + maxI.toFixed(0) + "ms";
        }

        rafId = requestAnimationFrame(renderFrame);
    }

    // ── Click-to-select and drag ──────────────────────────────────
    canvas.addEventListener("mousedown", function(e) {
        var rect = canvas.getBoundingClientRect();
        var mx = e.clientX - rect.left;
        var my = e.clientY - rect.top;
        var vids = virtualAgents();
        for (var i = 0; i < vids.length; i++) {
            var a = vids[i];
            var cx = a.x || localDotX, cy = ch - (a.y || localDotY);
            var r = ((a.size || localDotSize || 100) / 2) + 8;
            if ((mx-cx)*(mx-cx) + (my-cy)*(my-cy) < r*r) {
                selectedAgentId = parseInt(a.id, 10);
                if (hilAgentSelect) hilAgentSelect.value = String(selectedAgentId);
                localDotX = a.x || localDotX;
                localDotY = a.y || localDotY;
                localDotSize = a.size || localDotSize;
                localFrequencyHz = a.initial_frequency_hz || localFrequencyHz;
                dragging = true;
                break;
            }
        }
    });
    canvas.addEventListener("mousemove", function(e) {
        if (!dragging) return;
        var rect = canvas.getBoundingClientRect();
        var mx = e.clientX - rect.left;
        var my = ch - (e.clientY - rect.top);
        mx = Math.max(0, Math.min(cw, Math.round(mx)));
        my = Math.max(0, Math.min(ch, Math.round(my)));
        localDotX = mx; localDotY = my;
                hilAgentX.value = mx; hilAgentXVal.textContent = mx;
        hilAgentY.value = my; hilAgentYVal.textContent = my;
        apiPost("/api/agents/" + selectedAgentId, {x: mx, y: my});
    });
    canvas.addEventListener("mouseup", function(){ dragging = false; });
    canvas.addEventListener("mouseleave", function(){ dragging = false; });

    if (hilAgentSelect) {
        hilAgentSelect.addEventListener("change", function() {
            selectedAgentId = parseInt(hilAgentSelect.value, 10);
            var a = selectedAgent();
            if (!a) return;
            localDotX = a.x || localDotX;
            localDotY = a.y || localDotY;
            localDotSize = a.size || localDotSize;
            localFrequencyHz = a.initial_frequency_hz || localFrequencyHz;
            hilAgentX.value = localDotX; hilAgentXVal.textContent = localDotX;
            hilAgentY.value = localDotY; hilAgentYVal.textContent = localDotY;
            hilAgentSize.value = localDotSize; hilAgentSizeVal.textContent = localDotSize;
            hilAgentFreq.value = localFrequencyHz; hilAgentFreqVal.textContent = localFrequencyHz.toFixed(2);
        });
    }

    function postMutualConfig() {
        apiPost("/api/mutual/config", {
            mutual_agent_mode: hilAgentModeSelect ? hilAgentModeSelect.value : "single_1v1p",
            topology: hilTopologySelect ? hilTopologySelect.value : "all_to_all"
        }, function() {
            apiGet("/api/agents", function(data) {
                agents = data.agents || [];
                refreshAgentSelect();
            });
        });
    }
    if (hilAgentModeSelect) hilAgentModeSelect.addEventListener("change", postMutualConfig);
    if (hilTopologySelect) hilTopologySelect.addEventListener("change", postMutualConfig);

    // ── Slider change → POST ──────────────────────────────────────
    [hilAgentX, hilAgentY, hilAgentSize, hilAgentFreq].forEach(function(s) {
        s.addEventListener("input", function() {
            var d = { x: parseInt(hilAgentX.value), y: parseInt(hilAgentY.value),
                      size: parseInt(hilAgentSize.value),
                      initial_frequency_hz: parseFloat(hilAgentFreq.value),
                      model: hilModelSelect.value };
            hilAgentXVal.textContent = d.x; hilAgentYVal.textContent = d.y;
            hilAgentSizeVal.textContent = d.size; hilAgentFreqVal.textContent = d.initial_frequency_hz.toFixed(2);
            localFrequencyHz = d.initial_frequency_hz;
            localDotX = d.x; localDotY = d.y; localDotSize = d.size;
            apiPost("/api/agents/" + selectedAgentId, d);
        });
    });
    hilModelSelect.addEventListener("change", function() {
        apiPost("/api/agents/" + selectedAgentId, {model: hilModelSelect.value});
    });

    // ── Buttons ───────────────────────────────────────────────────
    btnHilStart.addEventListener("click", function() {
        apiPost("/api/start");
        visualStartTimeMs = window.performance.now();
        localRunning = true;
        localFireCount = 0;
        localLastFireMs = 0;
        localFireIntervals = [];
        hilStatus.textContent = "CAL: started — local clock running";
    });
    btnHilPause.addEventListener("click", function() {
        apiPost("/api/pause");
        localRunning = false;
        localFireIntervals = [];
        hilStatus.textContent = "Paused";
    });
    btnHilReset.addEventListener("click", function() {
        apiPost("/api/reset");
        localRunning = false;
        localFeedbackOn = false;
        localFireCount = 0;
        localLastFireMs = 0;
        localFireIntervals = [];
        fbServerPollSamples = 0;
        fbServerFlashOnSamples = 0;
        visualStartTimeMs = window.performance.now();
        hilStatus.textContent = "Reset";
    });

    // Feedback toggle
    btnFeedbackTgl.addEventListener("click", function() {
        localFeedbackOn = !localFeedbackOn;
        // Reset visual clock when switching to calibration
        if (!localFeedbackOn) {
            visualStartTimeMs = window.performance.now();
            localFireCount = 0;
            localLastFireMs = 0;
            localFireIntervals = [];
        } else {
            fbServerPollSamples = 0;
            fbServerFlashOnSamples = 0;
        }
        apiPost("/api/feedback", {enabled: localFeedbackOn});
    });

    // ── Display Only ──────────────────────────────────────────────
    function exitDisplayOnly() {
        document.body.classList.remove("display-only");
        if (hilPanel) hilPanel.style.display = "flex";
        if (debugLabel) debugLabel.style.display = "block";
    }
    function enterDisplayOnly() {
        document.body.classList.add("display-only");
        if (debugLabel) debugLabel.style.display = "none";
    }
    function toggleDisplayOnly() {
        if (document.body.classList.contains("display-only")) exitDisplayOnly();
        else enterDisplayOnly();
    }
    btnDisplayOnly.addEventListener("click", toggleDisplayOnly);
    document.addEventListener("keydown", function(e) {
        if (e.key === "d" || e.key === "D") { e.preventDefault(); toggleDisplayOnly(); }
        if (e.key === "Escape") { exitDisplayOnly(); }
    });
    var btnExit = document.getElementById("btn-exit-display-only");
    if (btnExit) btnExit.addEventListener("click", exitDisplayOnly);

    // ── Init ──────────────────────────────────────────────────────
    resizeCanvas();
    ctx.fillStyle = "rgb(0,0,0)";
    ctx.fillRect(0, 0, cw, ch);

    // Start animation loop
    rafId = requestAnimationFrame(renderFrame);

    // Initialise server mode without clobbering a preconfigured multi-agent setup.
    apiGet("/api/mutual/config", function(cfg) {
        if (hilAgentModeSelect && cfg.mutual_agent_mode) hilAgentModeSelect.value = cfg.mutual_agent_mode;
        if (hilTopologySelect && cfg.topology) hilTopologySelect.value = cfg.topology;
        apiPost("/api/mode", {
            mode: "mutual_hil",
            agent_mode: (cfg && cfg.mutual_agent_mode) || "single_1v1p",
            topology: (cfg && cfg.topology) || "all_to_all"
        }, function() {
        apiGet("/api/agents", function(data) {
            if (!data || !data.agents || data.agents.length === 0) {
                apiPost("/api/agents", {initial_frequency_hz: 2.0, model: "eapf_consensus", x: 800, y: 400, size: 350});
            } else {
                agents = data.agents;
                refreshAgentSelect();
                var a0 = data.agents[0];
                if ((a0.x || 0) === 0 && (a0.y || 0) === 0) {
                    apiPost("/api/agents/0", {x: 800, y: 400, size: 350});
                }
                localDotX = a0.x || 800;
                localDotY = a0.y || 400;
                localDotSize = a0.size || 350;
                localFrequencyHz = a0.initial_frequency_hz || 2.0;
                localFeedbackOn = a0.feedback_enabled === true;
            }
            pollServerConfig();
        });
        });
    });
})();

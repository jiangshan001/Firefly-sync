# Project Status Report - Firefly-Inspired Visual Synchronisation

**MSc Project - BIOE70025**  
**Report last updated: 2026-06-25**

## 1. Project Objective

This project investigates whether decentralised visual flashing can coordinate
multi-drone behaviour. Agents are treated as oscillators that emit light and
adjust their timing from locally observed flashes, without centralised control.

The current hardware-in-the-loop (HIL) system is mixed reality: browser-rendered
virtual agents run on the laptop/frontend, while a Raspberry Pi camera/LED node
provides physical visual sensing and flashing.

## 2. Current Headline Status

| Area | Status |
|------|--------|
| Simulation model selection | Complete; EAPF Consensus ranked first |
| Fixed-leader visual HIL | Complete; EAPF and Kuramoto both track a stable reference |
| 1 virtual + 1 Pi mutual HIL | Complete; EAPF selected over Kuramoto for current event-based HIL |
| Kuramoto K sensitivity appendix | Complete; best tested K=2.5 but still less stable than EAPF |
| Step5c1 two-flash multi-ROI detection | Complete; corrected contrast repeat batch added |
| Step5c2 2 virtual + 1 Pi EAPF all-to-all sync | Pure EAPF smoke run completed; lock-and-hold stabilisation prepared for comparison/formal batches |
| Step5d chain topology | Planned for later, especially after second Pi hardware arrives |

## 3. Implemented Synchronisation Models

### Kuramoto

- Source: `firefly_sync/core/kuramoto.py`
- Continuous phase coupling model.
- Main mutual HIL comparison used locked K=5.0.
- Appendix K sweep tested K in {2.5, 3.0, 3.5, 4.0, 4.5}; best tested K=2.5.
- Performs well in fixed-leader HIL but shows oscillatory behaviour in mutual
  visual HIL due to event-derived phase reconstruction and timing delay.

### EAPF Tracker

- Source: `firefly_sync/core/event_based_phase_lock.py`
- Fixed-leader event-based phase/frequency tracker.
- Useful for one-way tracking but not the selected multi-neighbour HIL model.

### EAPF Consensus

- Source: `firefly_sync/core/event_based_consensus_pll.py`
- Event-based multi-neighbour consensus PLL.
- Selected as the primary HIL model because it consumes flash events directly
  and uses filtered, rate-limited corrections.

Locked EAPF Consensus parameters:

| Parameter | Value |
|-----------|-------|
| `g_p` / `phase_gain` | 0.02 |
| `g_f` / `frequency_gain` | 0.02 |
| `alpha_p`, `alpha_f` | 0.2 |
| `max_phase_step_rad` | 0.2 rad |
| `max_frequency_step_hz` | 0.05 Hz |

## 4. Completed Evidence

### 4.1 Simulation Model Selection

Final model selection selected EAPF Consensus as the primary HIL candidate.

Main dataset:

`experiments/logs/stage4a_model_selection/20260616_113914_final_evaluation/`

Repeatability reference:

`experiments/logs/stage4a_model_selection/20260616_100238_final_evaluation/`

Key result:

| Rank | Model | Total score | Zero-lag rate |
|------|-------|-------------|---------------|
| 1 | EAPF Consensus | 0.938 | 1.000 |
| 2 | Kuramoto | 0.801 | 0.917 |

### 4.2 Fixed-Leader Visual HIL

Fixed-leader visual HIL comparison is complete.

Report:

`docs/step4c_fixed_leader_hil_comparison/step4c_fixed_leader_hil_comparison_v2.pdf`

Interpretation: fixed-leader success does not guarantee mutual-HIL stability,
because the reference is exogenous and does not react to the follower.

### 4.3 Mutual 1-Virtual + 1-Pi HIL Comparison

Formal mutual visual HIL model comparison is complete.

Dataset:

`experiments/logs/step5b_formal_chunked/formal_step5b_chunked_20260621/`

Report:

`docs/mutual_visual_hil_report/mutual_visual_hil_report.pdf`

Configuration:

- 1 browser virtual agent + 1 Raspberry Pi agent
- Bidirectional visual/API feedback
- Virtual initial frequency: 2.0 Hz
- Pi initial frequencies: 1.2, 1.5, 2.5 Hz
- 3 repeats per model-condition pair, 18 formal trials total
- EAPF Consensus with locked parameters
- Kuramoto with locked K=5.0

Overall result:

| Model | 5 s MAE (Hz) | Virtual std (Hz) | Pi std (Hz) |
|-------|--------------|------------------|-------------|
| EAPF Consensus | 0.046 | 0.001 | 0.011 |
| Kuramoto K=5.0 | 0.413 | 0.247 | 0.393 |

Conclusion: EAPF Consensus is better suited to this project's current
binary-flash, camera-mediated, event-based mutual HIL setup.

### 4.4 Kuramoto K Sensitivity Appendix

Dataset:

`experiments/logs/step5b_kuramoto_k_sensitivity/k_sweep_20260621_v4/`

Configuration:

- K values: 2.5, 3.0, 3.5, 4.0, 4.5
- 3 Pi initial frequencies x 2 repeats per K, 30 trials total
- Random initial phases, 30 s trials

Finding: K=2.5 was the best tested value, but Kuramoto still did not achieve
EAPF-level final-window stability. This supports the interpretation that the
limitation is structural, not only gain magnitude.

### 4.5 Step5c1 Two-Flash Multi-ROI Detection

Status: complete.

Purpose: validate whether the Pi camera can automatically locate and
independently detect two browser-rendered flashing targets before attempting
closed-loop 2V+1P synchronisation.

Key result:

- Two-flash multi-ROI detection validated.
- The Pi camera can automatically locate two flashing browser targets.
- Independent V0/V1 detection works sufficiently to proceed to Step5c2.
- Stress conditions should remain separate from readiness conditions.

Important folders:

- Log root: `experiments/logs/step5c1_multi_roi_detection/`
- Corrected contrast repeat rerun:
  `experiments/logs/step5c1_multi_roi_detection/20260625_152611_contrast_rerun_corrected/`
- Final report folder:
  `docs/step5c1_multi_roi_detection_report/`

## 5. Current Code Organisation

| Area | Path |
|------|------|
| EAPF Consensus model | `firefly_sync/core/event_based_consensus_pll.py` |
| Mixed-reality topology masks | `firefly_sync/multi_agent/hil_topology.py` |
| Multi-ROI flash detector | `firefly_sync/hardware/multi_roi_flash_detector.py` |
| Leader UI server/API | `experiments/run_leader_ui.py` |
| Browser frontend | `experiments/leader_ui/` |
| Low-level 2V+1P HIL utilities | `experiments/run_2v1p_eapf_hil.py` |
| Step5c1 detection batch | `experiments/run_step5c1_multi_roi_detection_batch.py` |
| Step5c2 sync batch | `experiments/run_step5c2_2v1p_eapf_sync_batch.py` |
| Step5c2/Step5d plan | `docs/step5c2_step5d_experimental_plan.md` |

The current Step5c2 runner verifies the locked EAPF configuration before
starting a batch. The leader UI now exposes a disabled-by-default stabilizer API
used only when the Step5c2 runner is called with `--stabilizer lock_hold`.

## 6. Step5c2 Planned / Next

Step5c2 tests whether a mixed-reality 3-agent system can synchronise under
all-to-all topology using EAPF Consensus. It is mixed-reality HIL, not a fully
physical three-drone experiment.

System:

- V0: browser-rendered virtual flash target
- V1: browser-rendered virtual flash target
- P0: Raspberry Pi physical flash node

First pure all-to-all smoke batch:

`experiments/logs/step5c2_2v1p_eapf_sync/20260625_180539_all_to_all_smoke`

Current interpretation:

- Pure all-to-all EAPF appears to achieve relative synchrony: small final phase
  error, small final frequency disagreement, and high final R.
- The common frequency can drift because the all-to-all mutual system has no
  fixed leader or absolute frequency anchor.
- Pure EAPF is therefore useful evidence for consensus, but not sufficient by
  itself for stable usable frequency locking.

### Optional Lock-And-Hold Stabilisation

The Step5c2 runner now supports `--stabilizer none` and
`--stabilizer lock_hold`.

Lock-and-hold behaviour:

1. Acquisition mode runs normal locked-parameter EAPF.
2. The runner monitors phase error, frequency disagreement, and R.
3. If fast-lock criteria pass over a short rolling window, hold mode starts and
   records `lock_time_s` and `f_lock`.
4. During hold, phase adaptation is reduced, frequency adaptation is disabled by
   default, and each agent is anchored toward `f_lock`.
5. If synchrony is lost over a slower hysteresis window, the runner exits hold
   mode and can re-lock.

This layer does not change the locked EAPF model parameters; it is an optional
experimental wrapper.

Current fast-lock defaults are R >= 0.95, mean phase error <= 0.08 cycles,
frequency disagreement <= 0.05 Hz, a 1.0 s lock window, and 0.5 pass ratio.
`f_lock` is the median common frequency over the trigger window. Unlock
hysteresis defaults are R < 0.85, phase error > 0.15 cycles, frequency
disagreement > 0.10 Hz, a 4.0 s unlock window, and 0.8 fail ratio.

### Updated Experimental Narrative

1. **Smoke comparison:** pure EAPF versus EAPF + lock-and-hold.
2. **Formal batch:** EAPF + lock-and-hold across multiple initial frequency sets.
3. **Robustness:** EAPF + lock-and-hold under mild delay, dropout, pause, and
   visual degradation.

### Frequency Sets

Formal Step5c2 uses:

- `same_2hz_random_phase`
- `close_1p8_2p2`
- `nominal_1_2`
- `wide_1_3`
- `mixed_low_mid_high`

Each trial saves the actual initial frequencies and phases.

### Metrics

Synchrony metrics:

- `final_sync_success`
- `continuous_sync_success`
- `time_to_sync_s`
- final phase error, frequency disagreement, and order parameter R

Frequency stability metrics:

- `final_mean_common_frequency_hz`
- `final_common_frequency_std_hz`
- `final_common_frequency_slope_hz_per_s`
- `frequency_stability_success`

Lock-and-hold metrics:

- `lock_acquired`
- `lock_time_s`
- `hold_duration_s`
- `unlock_count`
- `relock_count`
- `final_hold_state`
- hold-window R, phase error, frequency disagreement, common-frequency std/slope

Low-contrast caution: Step5c1 low contrast remains unresolved because V0 can be
suppressed by the detector amplitude gate under bright/cropped conditions. Treat
low-contrast robustness as a boundary condition unless a later rerun confirms
low-contrast detection.

## 7. Step5d Planned

Step5d will test chain topology synchronisation.

Intended later physical/mixed setup when hardware arrives:

`frontend/virtual node -- Pi1 -- Pi2`

Purpose:

- Test propagation through a chain without direct endpoint coupling.
- Separate topology effects from visual detection limitations already tested in
  Step5c1 and Step5c2.
- Start with clean chain smoke tests before adding robustness conditions.

Likely sequence:

1. Validate chain topology masks and dry-runs.
2. Use one available Pi for preliminary chain-like mixed tests if useful.
3. Run two-Pi chain smoke after hardware components arrive.
4. Run repeated random-initial chain trials.
5. Add mild robustness conditions after clean convergence is established.

## 8. Current Interpretation and Key Insights

1. Fixed-leader success does not contradict mutual-HIL Kuramoto oscillation.
2. Continuous phase coupling is fragile when neighbour phase must be reconstructed
   from binary flash events.
3. EAPF Consensus is naturally event-based and bounded, making it a better match
   for camera-derived flash timestamps.
4. Lower Kuramoto K improves behaviour but does not fully resolve oscillation.
5. EAPF should not be described as universally better than Kuramoto; it is better
   suited to this project's current event-based HIL architecture.
6. Robustness failures under severe disruption should be reported as boundaries
   or limitations, not hidden as implementation bugs.

## 9. Operational Commands

### Start Leader UI on Windows

```powershell
cd C:\Users\young\Desktop\BIOE70025\code\firefly-sync
$env:PYTHONPATH='.'
python experiments/run_leader_ui.py --host 0.0.0.0 --port 8000
```

### Step5c2 Smoke Comparison

```bash
cd firefly-sync
PYTHONPATH=. python experiments/run_step5c2_2v1p_eapf_sync_batch.py --leader-api http://<laptop-ip>:8000 --topology all_to_all --conditions baseline --trials 3 --duration 60 --random-initial --stabilizer none --batch-name all_to_all_smoke_pure_eapf
```

```bash
cd firefly-sync
PYTHONPATH=. python experiments/run_step5c2_2v1p_eapf_sync_batch.py --leader-api http://<laptop-ip>:8000 --topology all_to_all --conditions baseline --trials 3 --duration 60 --random-initial --stabilizer lock_hold --batch-name all_to_all_smoke_lock_hold
```

### Step5c2 Formal Lock-And-Hold Frequency Sets

```bash
cd firefly-sync
PYTHONPATH=. python experiments/run_step5c2_2v1p_eapf_sync_batch.py --leader-api http://<laptop-ip>:8000 --topology all_to_all --conditions baseline_random_initial --freq-sets same_2hz_random_phase close_1p8_2p2 nominal_1_2 wide_1_3 mixed_low_mid_high --trials 5 --duration 60 --random-initial --stabilizer lock_hold --batch-name all_to_all_lock_hold_freq_sets
```

### Step5c2 Lock-And-Hold Robustness Batch

```bash
cd firefly-sync
PYTHONPATH=. python experiments/run_step5c2_2v1p_eapf_sync_batch.py --leader-api http://<laptop-ip>:8000 --topology all_to_all --conditions v0_low_contrast v1_low_contrast p0_event_delay_150ms v0_event_dropout_20percent temporary_v1_pause_5s --trials 5 --duration 60 --random-initial --stabilizer lock_hold --batch-name all_to_all_lock_hold_robustness
```

### Copy Results from Pi to Windows

From Windows PowerShell:

```powershell
scp -r pi@<pi-ip>:~/firefly-sync/experiments/logs/step5c2_2v1p_eapf_sync/<batch-folder> C:\Users\young\Desktop\BIOE70025\code\firefly-sync\experiments\logs\step5c2_2v1p_eapf_sync\
```

## 10. Risks and Concerns

| Concern | Mitigation |
|---------|------------|
| Real physical LED detectability may differ from screen targets | Keep real LED detectability as a separate future evaluation |
| Multi-agent HIL may fail under severe disruption | Report severe failures as boundaries/limitations |
| Camera alignment may drift between trials | Use auto-ROI calibration or explicit ROI config |
| Raw data could be overwritten accidentally | Always write new timestamped batch directories |
| Stress cases could be mistaken for readiness cases | Keep readiness and stress labels separate in reports |

## 11. Immediate Next Actions

1. Finalise the Step5c1 report with the corrected contrast rerun.
2. Run the Step5c2 smoke comparison: pure EAPF and lock-and-hold.
3. Run the formal lock-and-hold frequency-set batch.
4. Run all-to-all lock-and-hold robustness tests.
5. Analyse pure EAPF consensus versus lock-and-hold common-frequency stability.
6. Later prepare Step5d chain topology when the second Pi hardware arrives.

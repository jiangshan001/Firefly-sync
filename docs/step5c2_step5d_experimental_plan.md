# Step5c2 and Step5d Experimental Plan

Last updated: 2026-06-25

## Current Interpretation

The first Step5c2 all-to-all smoke batch,
`experiments/logs/step5c2_2v1p_eapf_sync/20260625_180539_all_to_all_smoke`,
is preserved as pure EAPF evidence. The result indicates that pure all-to-all
EAPF can achieve relative synchrony: final phase error is small, final frequency
disagreement is small, and order parameter R is high.

However, the common frequency can drift over time. In a symmetric all-to-all
mutual system there is no fixed leader or absolute frequency anchor, so pure
EAPF provides consensus but not necessarily a stable usable common frequency.

Step5c2 is therefore split into:

1. Smoke comparison: pure EAPF vs EAPF + lock-and-hold.
2. Formal batch: EAPF + lock-and-hold across multiple initial frequency sets.
3. Robustness: EAPF + lock-and-hold under mild delay, dropout, pause, and visual
   degradation.

This remains mixed-reality HIL, not a fully physical three-drone experiment:
V0 and V1 are browser-rendered virtual flash targets, while P0 is the Raspberry
Pi camera/LED node.

## Locked EAPF Model

Do not change the underlying EAPF Consensus parameters:

| Parameter | Value |
|-----------|-------|
| `g_p` | 0.02 |
| `g_f` | 0.02 |
| `alpha_p`, `alpha_f` | 0.2 |
| `max_phase_step_rad` | 0.2 rad |
| `max_frequency_step_hz` | 0.05 Hz |

The lock-and-hold layer is an optional stabilising wrapper around the locked
model. Pure EAPF remains available with `--stabilizer none`.

## Lock-And-Hold Stabilisation

Acquisition mode runs normal EAPF. The runner continuously monitors:

- mean pairwise phase error
- frequency disagreement
- order parameter R
- common frequency

When lock criteria are satisfied over a short rolling window, the runner enters
hold mode. It records `lock_time_s`, computes `f_lock` as the median common
frequency over the trigger window, reduces EAPF phase/frequency adaptation, and
anchors each agent's frequency toward `f_lock`. Lock is intentionally fast
because hold mode is meant to prevent further common-frequency drift after
near-synchrony is reached. Unlock uses slower hysteresis so transient HIL jitter
does not immediately return the system to acquisition.

Default lock-and-hold settings:

| Setting | Default |
|---------|---------|
| `--lock-r-threshold` | 0.95 |
| `--lock-phase-error-threshold-cycles` | 0.08 |
| `--lock-frequency-disagreement-threshold-hz` | 0.05 |
| `--lock-window-s` | 1.0 |
| `--lock-window-pass-ratio` | 0.5 |
| `--hold-frequency-anchor` | `window_median` |
| `--hold-phase-gain-scale` | 0.1 |
| `--hold-frequency-gain-scale` | 0.0 |
| `--hold-anchor-gain` | 0.08 |
| `--unlock-r-threshold` | 0.85 |
| `--unlock-phase-error-threshold-cycles` | 0.15 |
| `--unlock-frequency-disagreement-threshold-hz` | 0.10 |
| `--unlock-window-s` | 4.0 |
| `--unlock-window-fail-ratio` | 0.8 |

## Step5c2 Experimental Structure

### 1. Smoke Comparison

Compare:

- `--stabilizer none`: pure EAPF
- `--stabilizer lock_hold`: EAPF acquisition plus hold stabilisation

Purpose:

- show that pure EAPF achieves relative synchrony/high R but may drift in common
  frequency
- show that lock-and-hold preserves synchrony while improving practical common
  frequency stability

### 2. Formal Frequency-Set Batch

Use only `--stabilizer lock_hold`.

Frequency sets:

| Frequency set | Definition | Purpose |
|---------------|------------|---------|
| `same_2hz_random_phase` | all agents near 2.0 Hz | isolate phase synchronisation |
| `close_1p8_2p2` | all agents random from 1.8-2.2 Hz | easy practical range |
| `nominal_1_2` | V0 around 1.0 Hz, V1 around 2.0 Hz, P0 around 1.5 Hz | consistent with detection validation |
| `wide_1_3` | all agents random from 1.0-3.0 Hz | harder initial disagreement |
| `mixed_low_mid_high` | V0 around 1.2 Hz, V1 around 1.8 Hz, P0 around 2.4 Hz | structured disagreement |

Each trial saves actual initial frequencies and phases.

### 3. Robustness Batch

Use only `--stabilizer lock_hold`.

Initial conditions:

| Condition | Intended disruption |
|-----------|---------------------|
| `v0_low_contrast` | medium/boundary V0 visual degradation |
| `v1_low_contrast` | medium/boundary V1 visual degradation |
| `p0_event_delay_150ms` | delay P0 routing to virtual agents by about 150 ms |
| `v0_event_dropout_20percent` | drop about 20% of V0 detected events before P0 consumes them |
| `temporary_v1_pause_5s` | pause V1 flashing for 5 s, then restore it |

Low-contrast caution: Step5c1 low contrast remains unresolved because V0 can be
suppressed by the detector amplitude gate under bright/cropped conditions. Treat
low-contrast robustness as a boundary condition unless a later rerun confirms
low-contrast detection.

## Metrics

Synchrony metrics:

- `final_sync_success`
- `continuous_sync_success`
- `time_to_sync_s`
- `final_mean_pairwise_phase_error_cycles`
- `final_max_pairwise_phase_error_cycles`
- `final_frequency_disagreement_hz`
- `final_mean_order_parameter_R`

Frequency stability metrics:

- `final_mean_common_frequency_hz`
- `final_common_frequency_std_hz`
- `final_common_frequency_slope_hz_per_s`
- `frequency_stability_success`

Lock-and-hold metrics:

- `lock_hold_enabled`
- `lock_acquired`
- `lock_time_s`
- `hold_duration_s`
- `unlock_count`
- `relock_count`
- `final_hold_state`
- `mean_hold_R`
- `mean_hold_phase_error_cycles`
- `mean_hold_frequency_disagreement_hz`
- `hold_common_frequency_std_hz`
- `hold_common_frequency_slope_hz_per_s`

Interpretation:

- `final_sync_success`: are the agents synchronised at the end?
- `continuous_sync_success`: did they satisfy a robust rolling synchrony window?
- `frequency_stability_success`: is the common frequency stable?
- `lock_acquired`: did lock-and-hold enter hold mode?

The rolling synchrony window uses an 80% sample pass ratio by default so a strong
final synchronised state is not hidden by occasional jitter.

## Output Structure

Log root:

`experiments/logs/step5c2_2v1p_eapf_sync/`

Per trial:

- `trial_config.json`
- `agent_state_timeseries.csv`
- `events.csv`
- `api_events.csv`
- `detection_metrics.json`
- `sync_metrics.json`
- `metrics_summary.json`
- `lock_hold_state_timeseries.csv` when lock-and-hold is enabled
- plots: `phase_vs_time.png`, `frequency_vs_time.png`,
  `order_parameter_vs_time.png`, `sync_criterion_vs_time.png`,
  `common_frequency_vs_time.png`, `lock_hold_state_vs_time.png` if enabled,
  `event_raster.png`

Batch level:

- `batch_config.json`
- `batch_summary.csv`
- `batch_summary.json`
- plots for success rate, final sync success, time to sync, lock time, final
  phase error, final frequency disagreement, final R, common-frequency std,
  common-frequency slope, hold duration, and recovery time

## Step5d: Chain Topology Synchronisation

Step5d should test propagation under a chain topology after Step5c2 all-to-all
readiness is established.

Intended later setup when the second Pi node/hardware components arrive:

`frontend/virtual node -- Pi1 -- Pi2`

Recommended sequence:

1. Validate chain topology masks and dry-runs.
2. Use one available Pi for preliminary chain-like mixed tests if useful.
3. Run two-Pi chain smoke after hardware arrives.
4. Run repeated random-initial chain trials.
5. Add mild robustness conditions after clean chain convergence is established.

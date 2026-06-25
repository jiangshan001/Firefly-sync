# Model Comparison Plan

## Fair-Comparison Protocol

All three models (Kuramoto, PCO-I&F, EAPF) will be compared under identical
experimental conditions using the same Pi visual closed-loop pipeline.

| Condition | Value |
|-----------|-------|
| Leader frequency | 2.0 Hz |
| Follower initial frequencies | 1.5, 1.8, 2.3 Hz |
| Trial duration | 30 s |
| Repeats per condition | 5 |
| Camera pipeline | Identical `PicameraFlashDetector` config |
| GPIO output | Identical GPIO17 LED, 0.06 s flash |
| Synchronisation threshold | 0.10 s (100 ms) |
| Consecutive sync cycles | 5 |
| Leader visual | 350 px glowing circle, same API |

## Metrics (Identical Across Models)

### Primary (9 metrics)
1. `synchronization_success` — bool, sync achieved during trial
2. `time_to_synchronization_s` — first time 5-cycle criterion met
3. `steady_state_mean_abs_timing_error_s` — last 50 % of flashes
4. `steady_state_rmse_timing_error_s`
5. `steady_state_jitter_s` — std of steady-state abs error
6. `final_frequency_error_hz` — |mean_leader_freq − mean_follower_freq|
7. `convergence_quality` — ratio of early-half MAE to late-half MAE
8. `detection_success_rate` — null in real Pi mode (no ground truth)
9. `false_positive_rate` — null in real Pi mode

### Secondary
- Leader flash count ratio (detected / expected)
- Actual wall duration vs requested
- Effective loop rate
- Computational cost (CPU, memory, temperature)

## Figures (Identical Per Model)

1. Success rate by condition
2. Time-to-sync by condition
3. Steady-state MAE by condition
4. Detection reliability by condition
5. Representative timing error time series
6. Representative flash raster
7. Loop performance by condition

## Parameter Tuning Strategy

Before running the full 3×5 comparison batch, each model needs parameter tuning:

### Kuramoto (already tuned)
- K = 3.5 (tuned from mock batch sweep)

### PCO-I&F
1. Run a mock parameter sweep: `epsilon` ∈ {0.05, 0.10, 0.15, 0.20, 0.30, 0.40}
   at follower frequencies 1.5, 1.8, 2.3 Hz.
2. Select the smallest `epsilon` that achieves > 80 % success rate across all
   conditions.
3. Verify on one Pi visual trial before committing to the full batch.

### EAPF
1. Run a mock parameter sweep for `phase_gain` × `frequency_gain`:
   `phase_gain` ∈ {0.10, 0.20, 0.30}
   `frequency_gain` ∈ {0.02, 0.05, 0.10}
2. Select the pair with the lowest mean time-to-sync while maintaining > 80 %
   success rate.
3. Verify convergence sign (follower should not diverge).
4. Verify on one Pi visual trial.

### Tuning Constraints
- Tune on **mock** mode only (no Pi hardware needed).
- Use the same leader frequency (2.0 Hz) and follower set (1.5, 1.8, 2.3 Hz).
- Save tuning results as `model_X_tuning_results.csv` in the model's config
  folder.
- Lock parameters before running the formal Pi visual batch.

## Expected Comparison Output

After all three models complete their formal batches, the analysis script
(`analyse_step3a_pi_visual_batch.py`) can be extended with a `--compare-models`
mode that overlays results from all three batch directories on the same axes.

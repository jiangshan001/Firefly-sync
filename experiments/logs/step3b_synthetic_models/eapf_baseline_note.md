# EAPF Synthetic Baseline Candidate

**Date:** 2026-06-15 (synthetic sweep, no hardware)

**Best parameters found:**
- `phase_gain = 0.3`
- `frequency_gain = 0.1`
- `frequency_min_hz = 0.5` (default)
- `frequency_max_hz = 4.0` (default)
- `leader_period_window = 6` (default)

**Synthetic validation results (leader = 2.0 Hz, 30 s, 5 repeats):**

| Follower initial (Hz) | Success rate | Mean time-to-sync (s) | Steady-state MAE (s) |
|-----------------------|-------------|----------------------|---------------------|
| 1.5 | 1.00 | — | ~0.0046 |
| 1.8 | 1.00 | — | ~0.0066 |
| 2.3 | 1.00 | — | ~0.0034 |

**Observations:**
- Bidirectional correction works: EAPF can both speed up slower followers
  and slow down faster followers.
- Frequency converges close to 2.0 Hz in all conditions.
- No hardware testing yet — synthetic only.

**Status:** Ready for Pi visual hardware testing after PCO-I&F sweep is
complete and a decision is made on which models to take to hardware.

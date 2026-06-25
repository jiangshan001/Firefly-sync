# Step 4C — Fixed-Leader Visual HIL Comparison

## Purpose

Compare Kuramoto and EAPF Consensus under the same fixed-leader Raspberry Pi
visual closed-loop setup before moving to multi-neighbour mutual HIL (Step 5).

This report is **not** a final model-selection judgement. It documents both
models' performance under identical single-follower visual HIL conditions.

## Data Sources

| Model | Source Directory | Trials |
|-------|-----------------|:------:|
| Kuramoto | `experiments/logs/step3a_pi_visual_batch/20260611_122117_kuramoto_pi_visual_batch/` | 15 |
| EAPF Consensus | `experiments/logs/step4b_eapf_pi_visual_batch_analysis/20260617_191042_eapf_pi_visual_batch_analysis/` | 15 |

## Trial Inclusion

- **Kuramoto:** All 15 trials from the established 2026-06-11 batch used as-is.
- **EAPF:** 15 formal batch trials selected via strict time-window filter
  (2026-06-17 18:25:17--18:33:50). 24 preliminary smoke/alignment trials excluded.
  Two legitimate outliers retained (2.3 Hz slow sync TTS ≈ 25.5 s and borderline
  MAE ≈ 0.107 s) as real data points.

## Generated Files

```
step4c_fixed_leader_hil_comparison/
├── step4c_fixed_leader_hil_comparison.tex   # LaTeX source
├── step4c_fixed_leader_hil_comparison.pdf   # Compiled 6-page PDF
├── README.md                                # This file
├── figures/
│   ├── comparison_success_rate.png
│   ├── comparison_time_to_sync.png
│   ├── comparison_steady_state_mae.png
│   └── comparison_leader_detection_reliability.png
└── tables/                                  # (CSV tables from analysis)
```

## Exact Compile Command

```bash
cd docs/step4c_fixed_leader_hil_comparison
pdflatex step4c_fixed_leader_hil_comparison.tex
```

## Notes

- Both models achieved 100% synchronisation success under the tested conditions.
- Kuramoto uses K = 3.5 (HIL-tuned); EAPF uses locked Stage 4A parameters
  (pg = 0.02, fg = 0.02).
- No preliminary/dirty EAPF trials are included.
- Kuramoto and EAPF original log directories are unmodified.

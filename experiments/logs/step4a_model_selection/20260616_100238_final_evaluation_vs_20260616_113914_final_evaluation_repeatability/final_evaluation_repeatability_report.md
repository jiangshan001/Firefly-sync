# Final Evaluation Repeatability Report
Run 1: 20260616_100238_final_evaluation
Run 2: 20260616_113914_final_evaluation

## Score Comparison
| Model | Run 1 Score | Run 2 Score | Delta | Stability |
|-------|:-----------:|:-----------:|:-----:|:---------:|
| eapf_consensus | 0.9375 | 0.9375 | +0.0000 | very stable |
| kuramoto | 0.8008 | 0.8020 | +0.0012 | very stable |
| eapf_tracker | 0.6563 | 0.6602 | +0.0039 | very stable |
| pco_adaptive_prc | 0.3625 | 0.3625 | +0.0000 | very stable |
| pco_simple | 0.3579 | 0.3579 | +0.0000 | very stable |

## Ranking Stability
| Model | Run 1 Rank | Run 2 Rank | Changed? |
|-------|:----------:|:----------:|:--------:|
| eapf_consensus | 1 | 1 | no |
| kuramoto | 2 | 2 | no |
| eapf_tracker | 3 | 3 | no |
| pco_adaptive_prc | 4 | 4 | no |
| pco_simple | 5 | 5 | no |

## Interpretation
- [OK] EAPF consensus remains primary HIL candidate in both runs.
- [OK] Kuramoto remains secondary comparison model in both runs.
- Maximum score delta: 0.0039
- All score changes are within the 'very stable' threshold.
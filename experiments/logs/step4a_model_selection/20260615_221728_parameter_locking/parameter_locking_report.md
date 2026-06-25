# Stage 4A Parameter Locking Report
Generated: 2026-06-15T22:17:52.213544

## Purpose
Lock one representative parameter set per model family for final thesis evaluation.

## Validation Conditions
- Topologies: all_to_all, chain, directed_chain
- Frequency sets: identical, near_identical, moderate_heterogeneity, strong_heterogeneity (or quick subset)
- Duration: 60 s, Repeats: 5, Seeds: 1000-1004

## Locked Parameters
### kuramoto
- **Parameters:** {"K": 5.0}
- **Validation score:** 0.9271
- **Zero-lag rate:** 0.8333, Phase-lock rate: 1.0
- **Mean FCR:** 1.0
- **Reason:** Best overall validation score across all conditions.

### pco_simple
- **Parameters:** {"coupling_mode": "additive_phase", "epsilon": 0.1, "refractory_period_s": 0.05}
- **Validation score:** 0.2413
- **Zero-lag rate:** 0.0, Phase-lock rate: 0.0
- **Mean FCR:** 1.459
- **Reason:** Best overall validation score across all conditions.

### pco_adaptive_prc
- **Parameters:** {"prc_mode": "biphasic_sine", "epsilon": 0.2}
- **Validation score:** 0.1814
- **Zero-lag rate:** 0.0, Phase-lock rate: 0.0
- **Mean FCR:** 1.272
- **Reason:** Best overall validation score across all conditions.

### eapf_tracker
- **Parameters:** {"phase_gain": 0.3, "frequency_gain": 0.15}
- **Validation score:** 0.5441
- **Zero-lag rate:** 0.5, Phase-lock rate: 0.1667
- **Mean FCR:** 1.156
- **Reason:** Best overall validation score across all conditions.

### eapf_consensus
- **Parameters:** {"phase_gain": 0.1, "frequency_gain": 0.01}
- **Validation score:** 0.7772
- **Zero-lag rate:** 0.6667, Phase-lock rate: 0.5
- **Mean FCR:** 1.022
- **Reason:** Best overall validation score across all conditions.

## Warning
Final evaluation must use different random seeds (2000-2029 recommended) and must not change the locked parameters.
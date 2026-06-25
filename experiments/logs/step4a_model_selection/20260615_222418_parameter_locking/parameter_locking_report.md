# Stage 4A Parameter Locking Report
Generated: 2026-06-15T22:46:18.889696

## Purpose
Lock one representative parameter set per model family for final thesis evaluation.

## Validation Conditions
- Topologies: all_to_all, chain, directed_chain
- Frequency sets: identical, near_identical, moderate_heterogeneity, strong_heterogeneity (or quick subset)
- Duration: 60 s, Repeats: 5, Seeds: 1000-1004

## Locked Parameters
### kuramoto
- **Parameters:** {"K": 5.0}
- **Validation score:** 0.9427
- **Zero-lag rate:** 0.9167, Phase-lock rate: 0.9
- **Mean FCR:** 1.0
- **Reason:** Best overall validation score across all conditions.

### pco_simple
- **Parameters:** {"coupling_mode": "additive_phase", "epsilon": 0.1, "refractory_period_s": 0.05}
- **Validation score:** 0.4127
- **Zero-lag rate:** 0.25, Phase-lock rate: 0.25
- **Mean FCR:** 1.295
- **Reason:** Best overall validation score across all conditions.

### pco_adaptive_prc
- **Parameters:** {"prc_mode": "biphasic_sine", "epsilon": 0.1}
- **Validation score:** 0.4472
- **Zero-lag rate:** 0.25, Phase-lock rate: 0.25
- **Mean FCR:** 1.175
- **Reason:** Best overall validation score across all conditions.

### eapf_tracker
- **Parameters:** {"phase_gain": 0.4, "frequency_gain": 0.15}
- **Validation score:** 0.7986
- **Zero-lag rate:** 0.7833, Phase-lock rate: 0.7833
- **Mean FCR:** 1.119
- **Reason:** Best overall validation score across all conditions.

### eapf_consensus
- **Parameters:** {"phase_gain": 0.02, "frequency_gain": 0.02}
- **Validation score:** 0.9984
- **Zero-lag rate:** 1.0, Phase-lock rate: 1.0
- **Mean FCR:** 1.0
- **Reason:** Best overall validation score across all conditions.

## Warning
Final evaluation must use different random seeds (2000-2029 recommended) and must not change the locked parameters.
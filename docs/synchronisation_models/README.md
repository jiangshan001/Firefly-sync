# Synchronisation Models — Design & Architecture

## Purpose

After successfully testing **Kuramoto** continuous-phase coupling in the Pi visual
closed-loop pipeline (Stage 3A-3), we now design two additional synchronisation
models for fair comparison under identical hardware conditions.

All three models share the same input/output pipeline:

```
Laptop screen (leader)  →  Pi camera detection  →  Model logic  →  GPIO17 LED (follower)
```

## Models

| # | Name | Type | Phase coupling |
|---|------|------|---------------|
| 1 | **Kuramoto** | Continuous-phase | sin(θ_leader − θ_follower) applied every dt |
| 2 | **PCO-I&F** | Pulse-coupled integrate-and-fire | Phase advance ε per detected leader pulse |
| 3 | **EAPF / Consensus-PLL** | Event-based phase/frequency locking | Phase + frequency correction on each leader event |

## Why Add Two More Models?

1. **PCO-I&F** is the classic biological synchronisation model (Mirollo–Strogatz 1990).
   It is event-driven, matching our camera-based discrete flash detection better than
   continuous coupling, and is the most cited model in firefly-inspired synchronisation.

2. **EAPF / Consensus-PLL** provides an engineering-oriented baseline (Gardner 2005)
   that explicitly tracks both phase and frequency.  It serves as a bridge to future
   multi-neighbour consensus synchronisation (Olfati-Saber et al. 2007).

## Comparison Strategy

All models will be tested under:
- Same leader frequency (2.0 Hz)
- Same follower initial frequencies (1.5, 1.8, 2.3 Hz)
- Same trial duration (30 s) with 5 repeats per condition
- Same synchronisation threshold (100 ms, 5 consecutive cycles)
- Same 9 evaluation metrics + computational cost metrics

## Folder Contents

| File | Purpose |
|------|---------|
| `model_2_pco_integrate_fire.md` | PCO-I&F full design spec |
| `model_3_event_based_phase_frequency_locking.md` | EAPF/PLL full design spec |
| `model_comparison_plan.md` | Fair-comparison protocol |
| `references.bib` | BibTeX entries for all cited works |
| `reference_verification/` | Traceability and verification docs |
| `flowcharts/` | Mermaid algorithm flowcharts |

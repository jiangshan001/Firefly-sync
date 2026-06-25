# Algorithm Claim Traceability

| # | Algorithm design claim | Model(s) | Supporting reference(s) | Evidence type | Confidence | Needs manual verification? |
|---|-----------------------|----------|------------------------|---------------|------------|---------------------------|
| 1 | PCO-I&F should use a threshold-reset oscillator state (normalised phase charging to 1.0, firing, resetting to 0). | PCO-I&F | Peskin 1975; Mirollo & Strogatz 1990 | Prompt-provided reference note | High | Yes — verify exact threshold-reset formulation in Mirollo–Strogatz. |
| 2 | Leader flash should be treated as a pulse event that can advance follower phase via `ε · (1 − φ)`. | PCO-I&F | Mirollo & Strogatz 1990; Tyrrell et al. 2006 | Prompt-provided reference note | High | Yes — verify the exact coupling function. |
| 3 | A refractory period after follower flash should suppress coupling briefly to prevent re-triggering. | PCO-I&F | Mirollo & Strogatz 1990 (inferred) | Prompt-provided reference note | Medium | Yes — confirm whether Mirollo–Strogatz explicitly models refractoriness. |
| 4 | EAPF can use phase error (measured at leader flash event) to adjust local oscillator phase and frequency. | EAPF / PLL | Gardner 2005 | Prompt-provided reference note | Medium-High | Yes — verify PLL discrete-event formulation matches our use case. |
| 5 | Frequency correction should be proportional to phase error, with independent gain for phase and frequency. | EAPF / PLL | Gardner 2005 (inferred); Olfati-Saber et al. 2007 (consensus framework) | Prompt-provided reference note | Medium | Yes — verify gain separation in Gardner. |
| 6 | Multi-neighbour extension can be framed as local consensus over neighbour timing estimates. | Future consensus extension | Olfati-Saber et al. 2007 | Prompt-provided reference note | Medium-High | Yes — verify consensus protocol formulation. |
| 7 | Firefly synchronisation is a valid bio-inspired model for engineered PCO networks. | All | Tyrrell et al. 2006 | Prompt-provided reference note | High | Yes — verify claims about engineering applicability. |
| 8 | Continuous phase coupling (Kuramoto) and pulse coupling (PCO) can be compared under identical visual pipeline. | Kuramoto, PCO-I&F | Kuramoto 1975; Mirollo & Strogatz 1990 | Bibliographic metadata + existing implementation | High | No — comparison methodology is our own contribution. |
| 9 | Median-based leader period estimation is robust to occasional missed or false detections. | EAPF / PLL | Gardner 2005 (PLL lock detection); our own engineering judgment | Prompt-provided reference note + design reasoning | Medium | Yes — Gardner may discuss outlier rejection in PLL. |
| 10 | The `wrap_to_pi` phase-error computation avoids large discontinuities when phase crosses 0/2π. | EAPF / PLL | Gardner 2005; standard signal processing | Prompt-provided reference note + standard DSP knowledge | High | No — standard practice in phase detection. |

**Confidence levels:**
- **High:** Strongly supported by prompt-provided notes; likely verifiable in original paper.
- **Medium-High:** Well-supported but some details inferred.
- **Medium:** Plausible but requires original-text confirmation for thesis-level citation.

**Needs manual verification:** "Yes" means the original paper/book should be consulted before final thesis submission to ensure the citation accurately reflects the source material.

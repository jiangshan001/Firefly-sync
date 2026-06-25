# Open Reference Questions

These questions require manual checking of original papers/books before final
thesis writing.  They cannot be resolved in this coding session because the
full texts were not accessed.

## Mirollo & Strogatz 1990

1. **Exact mathematical form of the phase response curve.**
   Is the coupling rule `φ → φ + ε · (1 − φ)` exactly as stated in the paper,
   or is it a simplification for identical oscillators?  Does the paper provide
   a generalised form for non-identical oscillators?

2. **Refractory period.**
   Does the Mirollo–Strogatz model explicitly include a refractory period after
   firing, or is this an engineering addition for our implementation?

3. **Synchronisation guarantees for non-identical oscillators.**
   The prompt-provided notes indicate that identical oscillators always
   synchronise.  Does the paper provide convergence bounds or conditions for
   oscillators with different natural frequencies?

## Gardner 2005

4. **Discrete-event PLL formulation.**
   Is there a specific section or chapter that discusses discrete-event or
   sampling-based phase correction (as opposed to continuous-time PLL)?
   Should we cite a specific chapter for our event-based model?

5. **Gain separation.**
   Does Gardner recommend independent gains for phase and frequency correction,
   or is this a design choice we are introducing?

6. **Lock detection.**
   Does Gardner discuss how to determine when a PLL has achieved lock?  This
   could inform our `synchronization_success` criterion.

## Olfati-Saber, Fax & Murray 2007

7. **Consensus protocol for timing/phase.**
   The paper covers consensus in networked multi-agent systems generally.
   Is there a specific protocol or section that addresses phase/time consensus
   (as opposed to position/velocity consensus)?

8. **Robustness guarantees.**
   Does the paper provide quantitative bounds on consensus convergence under
   time delays and link failures?  These would be directly relevant to our
   camera-based system where detection latency and missed events are common.

## Tyrrell, Auer & Bettstetter 2006

9. **Engineering constraints.**
   Does the paper provide specific quantitative results on how delays or
   missed detections affect PCO synchronisation?  Or is the discussion
   primarily qualitative?

10. **Firefly-to-PCO mapping.**
    How exactly does the paper map biological firefly flash timing to the
    Mirollo–Strogatz model?  Is there a simplified discrete-time version?

## General

11. **Citation phrasing.**
    For each claim in our `algorithm_claim_traceability.md`, what is the
    correct page, section, or equation number to cite in thesis text?

12. **Avoiding overclaiming.**
    How should we phrase the consensus extension without claiming that it has
    already been implemented?  Suggested wording: *"The EAPF model is designed
    such that its phase/frequency correction loop can be extended to a
    multi-neighbour consensus framework following Olfati-Saber et al. (2007).
    This extension is planned for future work and has not been tested in the
    current Pi visual pipeline."*

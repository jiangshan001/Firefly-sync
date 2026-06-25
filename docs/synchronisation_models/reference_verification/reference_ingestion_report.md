# Reference Ingestion Report

## Reference 1 — Peskin 1975

- **Full citation:** Peskin, C. S. *Mathematical Aspects of Heart Physiology.*
  Courant Institute, New York, 1975.
- **Source type:** Book
- **Full text accessed locally:** No
- **Evidence source used:**
  - Bibliographic metadata
  - Prompt-provided reference note

**Full text was not accessed in this coding session.** The following notes are
based on prompt-provided reference notes and bibliographic metadata only.

- Origin of the integrate-and-fire cardiac pacemaker model.
- Introduced the threshold-reset oscillator representation later adopted by
  Mirollo and Strogatz (1990).
- The model describes a state variable charging toward a threshold, firing,
  and resetting — the basis for PCO-I&F.
- Does not directly address visual synchronisation or camera-based detection.
- Cited primarily for background/motivation in our project.

---

## Reference 2 — Mirollo & Strogatz 1990

- **Full citation:** Mirollo, R. E. & Strogatz, S. H. "Synchronization of
  Pulse-Coupled Biological Oscillators." *SIAM J. Appl. Math.* 50(6):1645–1662,
  1990. doi:10.1137/0150098
- **Source type:** Journal article
- **Full text accessed locally:** No
- **Evidence source used:**
  - Bibliographic metadata
  - Prompt-provided reference note

**Full text was not accessed in this coding session.** The following notes are
based on prompt-provided reference notes and bibliographic metadata only.

- Formal analysis of pulse-coupled synchronisation in populations of identical
  integrate-and-fire oscillators.
- Each oscillator's phase advances toward a firing threshold; detected pulses
  from neighbours advance the phase by a fraction `ε`.
- The model is derived from Peskin's cardiac pacemaker model.
- Provides the mathematical foundation for our PCO-I&F model: the `ε · (1 − φ)`
  coupling rule.
- The paper shows that identical oscillators always synchronise; our project
  tests non-identical oscillators (different natural frequencies).

---

## Reference 3 — Tyrrell, Auer & Bettstetter 2006

- **Full citation:** Tyrrell, A., Auer, G. & Bettstetter, C. "Fireflies as
  Role Models for Synchronization in Ad Hoc Networks." *Proc. 1st Int. Conf.
  Bio Inspired Models of Network, Information and Computing Systems*, Cavalese,
  Italy, 2006. ACM. doi:10.1145/1315843.1315848
- **Source type:** Conference paper
- **Full text accessed locally:** No
- **Evidence source used:**
  - Bibliographic metadata
  - Prompt-provided reference note

**Full text was not accessed in this coding session.** The following notes are
based on prompt-provided reference notes and bibliographic metadata only.

- Discusses fireflies as role models for decentralised synchronisation in
  ad hoc networks.
- Links firefly synchronisation to the Mirollo–Strogatz pulse-coupled model.
- Discusses practical engineering issues: delays, communication constraints,
  and scalability — all relevant to our Pi visual pipeline where detection
  latency and missed events are real concerns.
- Supports the claim that pulse-coupled oscillator models are applicable to
  real engineered systems, not just abstract theory.

---

## Reference 4 — Olfati-Saber, Fax & Murray 2007

- **Full citation:** Olfati-Saber, R., Fax, J. A. & Murray, R. M. "Consensus
  and Cooperation in Networked Multi-Agent Systems." *Proc. IEEE* 95(1):215–233,
  2007. doi:10.1109/JPROC.2006.887293
- **Source type:** Journal article
- **Full text accessed locally:** No
- **Evidence source used:**
  - Bibliographic metadata
  - Prompt-provided reference note

**Full text was not accessed in this coding session.** The following notes are
based on prompt-provided reference notes and bibliographic metadata only.

- Provides the theoretical framework for consensus algorithms in networked
  multi-agent systems.
- Emphasises directed information flow, robustness to topology changes,
  link/node failures, and time delays.
- Supports the future extension of our EAPF model from single leader-follower
  to multi-neighbour consensus synchronisation.
- Not directly about fireflies or visual synchronisation, but provides the
  mathematical machinery for multi-agent timing agreement.

---

## Reference 5 — Gardner 2005

- **Full citation:** Gardner, F. M. *Phaselock Techniques.* 3rd ed., John Wiley
  & Sons, Hoboken, NJ, 2005. ISBN 978-0-471-43063-6.
- **Source type:** Book
- **Full text accessed locally:** No
- **Evidence source used:**
  - Bibliographic metadata
  - Prompt-provided reference note

**Full text was not accessed in this coding session.** The following notes are
based on prompt-provided reference notes and bibliographic metadata only.

- Standard engineering reference on phase-lock loop (PLL) technology.
- Supports using phase error to adjust local oscillator phase/frequency so a
  local oscillator tracks an external periodic reference.
- Provides the engineering foundation for our EAPF model: explicit phase-error
  measurement, gain-based correction, and frequency tracking.
- The book covers both analog and digital PLL implementations; our model
  adapts the discrete-event variant for camera-based flash detection.

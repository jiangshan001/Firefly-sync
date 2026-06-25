# Stage 4A Multi-Agent Simulation — Technical Setup Reference

**Generated from source inspection, 2026-06-19.**

Source files inspected:
- `firefly_sync/multi_agent/topology.py`
- `firefly_sync/multi_agent/simulation.py`
- `firefly_sync/multi_agent/agent.py`
- `firefly_sync/multi_agent/metrics.py`
- `experiments/final_stage4a_model_evaluation.py`
- `experiments/run_stage4a_multi_neighbour_simulation.py`

## 1. Exact Neighbourhood Definitions

All topologies use **graph-based adjacency** (no spatial geometry).  The
adjacency dict maps `agent_id → list[neighbour_id]`, where the list contains
the IDs of agents that `agent_id` **can observe** (i.e. whose flashes it can
detect).  See `topology.py` line 15–20.

### 1.1 N=3 Topologies

| Topology | Adjacency | Description |
|----------|-----------|-------------|
| `all_to_all` | `0:[1,2], 1:[0,2], 2:[0,1]` | Every agent sees every other agent |
| `chain` | `0:[1], 1:[0,2], 2:[1]` | Undirected: endpoint A sees B, middle B sees both A and C, endpoint C sees B |
| `directed_chain` | `0:[], 1:[0], 2:[1]` | Agent i sees only agent i−1. Information flows one way: A→B→C. Agent 0 sees nobody. |
| `ring` | `0:[2,1], 1:[0,2], 2:[1,0]` | Each sees its predecessor and successor modulo N |

### 1.2 N=5 Topologies

| Topology | Adjacency (agent → neighbours) |
|----------|-------------------------------|
| `local_ring_5` | `0:[1,4], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3,0]` |
| `chain_5` | `0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]` |
| `local_degree_2_3` | `0:[1,2], 1:[0,2,3], 2:[0,1,3], 3:[1,2,4], 4:[2,3]` |

## 2. Edge Semantics

The docstring in `topology.py` line 15–20 states: *"The list contains the
IDs of agents that agent\_id can see (i.e., whose flashes it can detect)."*

Therefore **edges mean "i observes j"**, not "i influences j".  Influence
flows in the **opposite** direction: if i observes j, then j's flashes
influence i's oscillator.

## 3. Chain Direction

`chain` is **undirected**: agent 1 sees both 0 and 2, and both 0 and 2 see
agent 1.  It is a symmetric bidirectional propagation chain.

`directed_chain` is **directed one-way**: agent i sees only i−1.  Agent 0
sees nobody (it is a free-running leader).  Information flows from 0 → 1 → 2.

## 4. directed_chain Direction

Agent 0 sees nobody.  Agent 1 sees agent 0.  Agent 2 sees agent 1.
Information flows **0 → 1 → 2** (agent 1 observes agent 0's flashes;
agent 2 observes agent 1's flashes).  Agent 0 is effectively an
uninfluenced oscillator.

## 5. Per-Step Simulation Order

From `simulation.py` lines 60–122, each timestep at `t` executes:

1. **Step all agents** (loop over all agents):
   - Compute coupling input or neighbour-flash count for this agent
     using the topology adjacency
   - Call `agent.step(dt_s, t_s, coupling_input, neighbour_flash_events, neighbour_flash_ids)`
   - Record `is_firing[i]` from the result
2. **Log** (loop over all agents):
   - Record phase, frequency, firing state in `agent_logs`
   - If agent fired, append to `flash_events`
3. **Advance** `t += dt`

Step 1 uses an array `is_firing = [False] * self.n` initialised at the top
of the loop.  For event-based models (PCO/EAPF), neighbour-flash delivery
checks `is_firing[j]` from **this same step**, i.e. flash events generated
in step 1 are delivered to neighbours within the same step iteration.
The loop order is `for i, agent in enumerate(self.agents)`, so agent 0
goes first.  If agent j (where j < i) fired earlier in this loop iteration,
its flash is visible to agent i in the same step.

## 6. Same-Step vs Previous-Step Neighbour Flashes

Event-based models (PCO, EAPF) receive **same-step** neighbour flashes.
The `is_firing[j]` array is populated as agents are stepped in order
0..N−1.  An agent that fires in the current step is immediately visible
to all subsequent agents in the same step iteration.

## 7. Kuramoto Access to True Neighbour Phase

**Yes.** In `simulation.py` lines 72–78, Kuramoto coupling is computed as:
```python
dtheta = self.agents[j].phase - agent.phase
coupling_input += math.sin(dtheta)
```
This uses `self.agents[j].phase` — the **true internal phase** of
neighbour j — not a flash-derived estimate.  This is a simulation
assumption; a hardware implementation would require phase reconstruction
from observed flash timestamps.

## 8. EAPF Consensus Uses Only Flash-Event-Derived Neighbour Estimates

**Yes.**  `EventBasedConsensusPLLOscillator` in
`firefly_sync/core/event_based_consensus_pll.py` updates neighbour state
only via `record_neighbour_flash(neighbour_id, t_s)`, which receives
**only a timestamp**.  Neighbour phase is propagated internally using
locally estimated neighbour frequency (derived from flash intervals).
The oscillator never accesses another agent's true internal phase or
frequency.

## 9. Noise, Delay, Missed Detection, False Positives

All noise parameters exist in the `MultiAgentSimulation` constructor
(`event_delay_s`, `missed_event_prob`) but are set to **zero** in the
final evaluation via `_make_args` defaults:
```python
"event_delay_s": 0.0, "missed_event_prob": 0.0
```
Camera field-of-view, false positives, and detection latency are **not
modelled** in the Stage 4A simulation.  The evaluation uses **ideal
event detection** (every agent flash is perfectly observed by all
neighbours, with no delay or missed events).

## 10. Trial Initialisation

From `run_stage4a_multi_neighbour_simulation.py` and
`final_stage4a_model_evaluation.py`:

| Parameter | Value |
|-----------|-------|
| **Initial phases** | **Random uniform** in [0, 2π) per agent, per trial: `phase = rng.uniform(0, 2.0 * np.pi)` |
| **Random seeds** | Seeds `seed_start + rep`; final evaluation used 2000–2019 (N=3) and 3000–3009 (N=5), repeatability run used 4000–4019 and 5000–5009 |
| **Natural frequencies** | From the frequency set being tested (e.g. [1.5, 2.0, 2.3] Hz for strong heterogeneity); set as `initial_frequency_hz` in `AgentConfig` |
| **Duration** | 60 s per trial |
| **dt** | 0.01 s (6000 steps per trial) |
| **Repeats** | 20 per condition (N=3), 10 per condition (N=5) |

Each trial uses a **fresh random generator** from `np.random.default_rng(seed)`
with a deterministic seed.  Initial phases are the only source of randomness
within a trial (no noise, no missed detections).

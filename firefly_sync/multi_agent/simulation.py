"""Multi-agent discrete-time synchronisation simulation engine.

Advances N agents at fixed dt.  For Kuramoto, coupling is continuous
(computed every step).  For PCO-I&F and EAPF, coupling is event-based
(triggered when a neighbour fires).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from firefly_sync.multi_agent.agent import Agent, AgentConfig
from firefly_sync.multi_agent.topology import Topology


class MultiAgentSimulation:
    """Runs a multi-agent synchronisation trial.

    Parameters
    ----------
    configs: List of AgentConfig, one per agent.
    topology: Topology defining neighbourhood visibility.
    dt: Simulation timestep in seconds.
    event_delay_s: Extra delay before neighbour events are applied.
    missed_event_prob: Probability of missing a neighbour's flash.
    """

    def __init__(
        self,
        configs: list[AgentConfig],
        topology: Topology,
        dt: float = 0.01,
        event_delay_s: float = 0.0,
        missed_event_prob: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(configs) != topology.n_agents:
            raise ValueError(
                f"Got {len(configs)} configs but topology expects {topology.n_agents}"
            )
        self.configs = configs
        self.topology = topology
        self.dt = dt
        self.event_delay_s = event_delay_s
        self.missed_event_prob = missed_event_prob
        self.rng = rng or np.random.default_rng()

        self.agents: list[Agent] = [Agent(cfg) for cfg in configs]
        self.n = len(self.agents)

        # State
        self.t = 0.0
        self.step_idx = 0
        self.flash_events: list[dict] = []
        self.agent_logs: list[list[dict]] = [[] for _ in range(self.n)]

    def run(self, duration_s: float) -> None:
        """Run the simulation for *duration_s* seconds."""
        while self.t < duration_s:
            # 1. Step all agents
            is_firing = [False] * self.n
            results: list[dict] = []

            for i, agent in enumerate(self.agents):
                # Compute coupling input for this agent
                coupling_input = 0.0
                neighbour_events = 0

                if agent.model == "kuramoto":
                    # Continuous coupling: Σ sin(θ_j − θ_i) over visible neighbours
                    neighbours = self.topology.adjacency.get(i, [])
                    for j in neighbours:
                        dtheta = self.agents[j].phase - agent.phase
                        coupling_input += math.sin(dtheta)
                    coupling_input /= max(1, len(neighbours))

                elif agent.model in ("pco_if", "eapf", "eapf_consensus"):
                    # Event-based: count neighbour flashes this step
                    neighbours = self.topology.adjacency.get(i, [])
                    for j in neighbours:
                        if is_firing[j]:
                            if self.rng.random() >= self.missed_event_prob:
                                neighbour_events += 1

                # Pass neighbour flash IDs for consensus PLL
                flash_ids = [j for j in self.topology.adjacency.get(i, [])
                             if is_firing[j]] if agent.model == "eapf_consensus" else None

                r = agent.step(
                    dt_s=self.dt, t_s=self.t,
                    coupling_input=coupling_input,
                    neighbour_flash_events=neighbour_events,
                    neighbour_flash_ids=flash_ids,
                )
                is_firing[i] = r["follower_flash_event"]
                results.append(r)

            # 2. Log
            for i, agent in enumerate(self.agents):
                r = results[i]
                log_entry: dict[str, Any] = {
                    "t_s": round(self.t, 6), "agent_id": i,
                    "model": agent.model,
                    "phase": round(agent.phase, 6),
                    "frequency_hz": round(agent.frequency_hz, 6) if hasattr(agent, 'frequency_hz') else None,
                    "is_firing": 1 if r["follower_flash_event"] else 0,
                }
                self.agent_logs[i].append(log_entry)

                if r["follower_flash_event"]:
                    self.flash_events.append({
                        "t_s": round(self.t, 6),
                        "agent_id": i,
                        "event_type": "agent_flash",
                        "model": agent.model,
                    })

            self.t += self.dt
            self.step_idx += 1

    def get_results(self) -> dict[str, Any]:
        """Return simulation results for logging / metrics."""
        return {
            "flash_events": self.flash_events,
            "agent_flash_times": [a.flash_times for a in self.agents],
            "agent_final_frequencies": [a.frequency_hz for a in self.agents],
            "agent_logs": self.agent_logs,
            "n_steps": self.step_idx,
        }

"""Neighbourhood topology definitions for multi-agent simulation.

Supports graph-based topologies (no geometry needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TOPOLOGY_TYPES = ("all_to_all", "chain", "directed_chain", "ring",
                    "local_ring_5", "chain_5", "local_degree_2_3")


@dataclass
class Topology:
    """Defines which agents can observe which other agents.

    *adjacency* is a dict mapping ``agent_id`` → ``list[neighbour_id]``.
    The list contains the IDs of agents that *agent_id* can see (i.e.,
    whose flashes it can detect).
    """

    n_agents: int
    topology_type: str = "all_to_all"
    adjacency: dict[int, list[int]] = field(default_factory=dict)


def build_topology(n_agents: int, topology_type: str) -> Topology:
    """Build an adjacency graph for *n_agents* IDs 0..n−1.

    Parameters
    ----------
    n_agents:
        Number of agents (≥ 2).
    topology_type:
        One of ``"all_to_all"``, ``"chain"``, ``"directed_chain"``,
        ``"ring"``.

    Returns
    -------
    Topology
    """
    if topology_type not in TOPOLOGY_TYPES:
        raise ValueError(f"Unknown topology type: {topology_type}")
    if n_agents < 2:
        raise ValueError("n_agents must be ≥ 2")

    adj: dict[int, list[int]] = {i: [] for i in range(n_agents)}

    if topology_type == "all_to_all":
        for i in range(n_agents):
            adj[i] = [j for j in range(n_agents) if j != i]

    elif topology_type == "chain":
        for i in range(n_agents):
            if i > 0:
                adj[i].append(i - 1)
            if i < n_agents - 1:
                adj[i].append(i + 1)

    elif topology_type == "directed_chain":
        # Agent i sees agent i−1 (if exists) — information flows one way
        for i in range(1, n_agents):
            adj[i].append(i - 1)

    elif topology_type == "ring":
        for i in range(n_agents):
            adj[i].append((i - 1) % n_agents)
            adj[i].append((i + 1) % n_agents)

    # N=5 specific topologies
    elif topology_type == "local_ring_5":
        # A-B-C-D-E ring, each sees 2 neighbours
        adj[0] = [1, 4]; adj[1] = [0, 2]; adj[2] = [1, 3]
        adj[3] = [2, 4]; adj[4] = [3, 0]

    elif topology_type == "chain_5":
        # A-B-C-D-E chain
        adj[0] = [1]; adj[1] = [0, 2]; adj[2] = [1, 3]
        adj[3] = [2, 4]; adj[4] = [3]

    elif topology_type == "local_degree_2_3":
        # Sparse: A sees B,C; B sees A,C,D; C sees A,B,D; D sees B,C,E; E sees C,D
        adj[0] = [1, 2]; adj[1] = [0, 2, 3]
        adj[2] = [0, 1, 3]; adj[3] = [1, 2, 4]; adj[4] = [2, 3]

    return Topology(n_agents=n_agents, topology_type=topology_type, adjacency=adj)

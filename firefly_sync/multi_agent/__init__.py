"""Multi-neighbour synchronisation simulation package.

Provides agent-based simulation for comparing Kuramoto, PCO-I&F, and
EAPF models under decentralised multi-agent topologies.

Modules:
  - ``agent.py``    — agent wrapper around oscillator models
  - ``topology.py`` — neighbourhood / adjacency graph definitions
  - ``simulation.py`` — discrete-time simulation engine
  - ``metrics.py``  — group-level synchronisation metrics
"""

from firefly_sync.multi_agent.topology import (
    Topology, build_topology, TOPOLOGY_TYPES,
)
from firefly_sync.multi_agent.agent import (
    Agent, AgentConfig,
)
from firefly_sync.multi_agent.simulation import MultiAgentSimulation
from firefly_sync.multi_agent.metrics import (
    compute_group_metrics,
    check_group_synchronisation,
)

__all__ = [
    "Topology", "build_topology", "TOPOLOGY_TYPES",
    "Agent", "AgentConfig",
    "MultiAgentSimulation",
    "compute_group_metrics", "check_group_synchronisation",
]

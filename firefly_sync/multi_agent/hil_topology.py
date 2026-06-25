"""Topology masks for mixed-reality mutual HIL agents.

The labels are stable experiment-facing IDs:

* ``V0`` and ``V1`` are browser-rendered virtual agents.
* ``P0`` is the Raspberry Pi LED/camera agent.

Adjacency maps each receiving agent to the source agents whose flash events it
should consume.
"""

from __future__ import annotations

from dataclasses import dataclass


MIXED_REALITY_AGENT_IDS = ("V0", "V1", "P0")
MIXED_REALITY_TOPOLOGIES = (
    "all_to_all",
    "chain_pi_middle",
    "chain_pi_downstream",
)
AGENT_NUMERIC_IDS = {"V0": 0, "V1": 1, "P0": 2}


@dataclass(frozen=True)
class MixedRealityTopology:
    name: str
    adjacency: dict[str, list[str]]

    def visible_neighbours(self, agent_id: str) -> list[str]:
        return list(self.adjacency.get(agent_id, []))

    def can_observe(self, receiver_id: str, source_id: str) -> bool:
        return source_id in self.adjacency.get(receiver_id, [])

    def numeric_neighbour_ids(self, receiver_id: str, sources: list[str]) -> list[int]:
        return [
            AGENT_NUMERIC_IDS[source]
            for source in sources
            if self.can_observe(receiver_id, source)
        ]


def build_mixed_reality_topology(name: str) -> MixedRealityTopology:
    """Return the requested 2-virtual + 1-Pi topology mask."""
    if name not in MIXED_REALITY_TOPOLOGIES:
        raise ValueError(f"Unknown mixed-reality topology: {name}")

    if name == "all_to_all":
        adjacency = {
            "V0": ["V1", "P0"],
            "V1": ["V0", "P0"],
            "P0": ["V0", "V1"],
        }
    elif name == "chain_pi_middle":
        adjacency = {
            "V0": ["P0"],
            "P0": ["V0", "V1"],
            "V1": ["P0"],
        }
    else:
        adjacency = {
            "V0": ["V1"],
            "V1": ["V0", "P0"],
            "P0": ["V1"],
        }

    return MixedRealityTopology(name=name, adjacency=adjacency)


def build_single_mutual_topology() -> MixedRealityTopology:
    """Backward-compatible 1-virtual + 1-Pi mutual topology."""
    return MixedRealityTopology(
        name="single_mutual",
        adjacency={"V0": ["P0"], "P0": ["V0"]},
    )

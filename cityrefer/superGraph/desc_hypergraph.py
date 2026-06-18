from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PHRASE_TO_RELATION = [
    ("to the left", "left_of"),
    ("left side", "left_of"),
    ("on its left", "left_of"),
    ("left of", "left_of"),
    ("to the right", "right_of"),
    ("right side", "right_of"),
    ("on its right", "right_of"),
    ("right of", "right_of"),
    ("in front of", "front_of"),
    ("front of", "front_of"),
    ("behind", "behind"),
    ("north of", "north_of"),
    ("south of", "south_of"),
    ("above", "above"),
    ("below", "below"),
    ("between", "between"),
    ("in between", "between"),
    ("closest to", "closest_to"),
    ("farthest from", "far_from"),
    ("far from", "far_from"),
    ("next to", "adjacent"),
    ("adjacent to", "adjacent"),
    ("close to", "adjacent"),
    ("nearby", "adjacent"),
    ("near", "adjacent"),
    ("beside", "adjacent"),
    ("by", "adjacent"),
    ("with", "adjacent"),
    ("clustered with", "adjacent"),
    ("connected to", "connected_to"),
    ("surrounded by", "surrounded_by"),
    ("facing", "facing"),
    ("towards", "towards"),
    ("opposite", "opposite"),
    ("across from", "opposite"),
    ("across the street from", "opposite"),
    ("at the corner of", "at_corner"),
    ("on the corner of", "at_corner"),
    ("in the corner of", "at_corner"),
    ("near corner", "near_corner"),
    ("at the end of", "at_end"),
    ("on the edge of", "on_edge"),
    ("along", "along"),
    ("on the side of", "on_side"),
    ("on one side of", "on_side"),
    ("on the other side of", "on_side"),
    ("inside", "inside"),
    ("parked in", "inside"),
    ("in the middle of", "inside"),
    ("in", "inside"),
    ("on", "on_surface"),
    ("of", "belonging"),
    ("off", "outside"),
    ("from", "outside"),
]


@dataclass
class HyperNode:
    id: str
    category: str
    is_main: bool
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class HyperEdge:
    from_node: str
    to_node: str
    relation: str
    raw_relation: str
    score: float
    info: Dict[str, str] = field(default_factory=dict)


@dataclass
class DescriptionHyperGraph:
    scene_id: str
    object_id: str
    ann_id: Optional[int]
    object_name: str
    description: str
    nodes: List[HyperNode] = field(default_factory=list)
    edges: List[HyperEdge] = field(default_factory=list)


def normalize_relation(raw_relation: str, relation_type: str) -> Optional[str]:
    phrase = (raw_relation or "").strip().lower()
    for key, relation in PHRASE_TO_RELATION:
        if key in phrase:
            return relation

    relation_type = (relation_type or "").strip().lower()
    if relation_type == "distance":
        return "adjacent"
    if relation_type == "topological":
        return "inside"
    if relation_type == "directional":
        return "front_of"
    return None


def build_description_hypergraph(data: Dict) -> Optional[DescriptionHyperGraph]:
    construction = data.get("construction") or []
    if not construction:
        return None

    graph = DescriptionHyperGraph(
        scene_id=str(data.get("scene_id", "")),
        object_id=str(data.get("object_id", "")),
        ann_id=data.get("ann_id"),
        object_name=data.get("object_name", ""),
        description=data.get("description", ""),
    )

    for idx, item in enumerate(construction):
        node_id = "main" if item.get("is_main") else f"object{idx}"
        graph.nodes.append(
            HyperNode(
                id=node_id,
                category=item.get("category", ""),
                is_main=bool(item.get("is_main")),
                properties={
                    "spatial_relation": item.get("spatial_relation", ""),
                    "relation_type": item.get("relation_type", ""),
                    "reference_anchor": item.get("reference_anchor", ""),
                },
            )
        )

    main_node = next((node for node in graph.nodes if node.is_main), None)
    if main_node is None:
        return None

    for node in graph.nodes:
        if node.is_main:
            continue
        relation = normalize_relation(
            node.properties.get("spatial_relation", ""),
            node.properties.get("relation_type", ""),
        )
        if not relation:
            continue
        graph.edges.append(
            HyperEdge(
                from_node=main_node.id,
                to_node=node.id,
                relation=relation,
                raw_relation=node.properties.get("spatial_relation", ""),
                score=1.0,
                info={
                    "reference_anchor": node.properties.get("reference_anchor", ""),
                    "target_category": node.category,
                },
            )
        )

    return graph

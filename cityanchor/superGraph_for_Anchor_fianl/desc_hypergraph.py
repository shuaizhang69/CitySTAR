from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


PHRASE_TO_RELATION = [
    ("on the other side of", "on_side"),
    ("on the other side", "on_side"),
    ("on one side of", "on_side"),
    ("to the side of", "on_side"),
    ("on the side of", "on_side"),
    ("across the street from", "opposite"),
    ("around the corner of", "near_corner"),
    ("at the corner of", "at_corner"),
    ("on the corner of", "at_corner"),
    ("in the corner of", "at_corner"),
    ("flanked on both sides by", "on_side"),
    ("flanked by", "on_side"),
    ("clustered with", "adjacent"),
    ("connected to", "connected_to"),
    ("surrounded by", "surrounded_by"),
    ("in the middle of", "inside"),
    ("parked in", "inside"),
    ("in between", "between"),
    ("closest to", "closest_to"),
    ("farthest from", "far_from"),
    ("far from", "far_from"),
    ("next to", "adjacent"),
    ("adjacent to", "adjacent"),
    ("close to", "adjacent"),
    ("alongside", "adjacent"),
    ("in front of", "front_of"),
    ("front of", "front_of"),
    ("left side", "left_of"),
    ("right side", "right_of"),
    ("to the left", "left_of"),
    ("to the right", "right_of"),
    ("left of", "left_of"),
    ("right of", "right_of"),
    ("north of", "north_of"),
    ("south of", "south_of"),
    ("at the end of", "at_end"),
    ("on the edge of", "on_edge"),
    ("underneath", "below"),
    ("parked under", "below"),
    ("beneath", "below"),
    ("around", "adjacent"),
    ("all around", "adjacent"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("under", "below"),
    ("between", "between"),
    ("beside", "adjacent"),
    ("nearby", "adjacent"),
    ("near", "adjacent"),
    ("along", "along"),
    ("inside", "inside"),
    ("facing", "facing"),
    ("towards", "towards"),
    ("opposite", "opposite"),
    ("across from", "opposite"),
    ("off", "outside"),
    ("outside", "outside"),
    ("on", "on_surface"),
    ("in", "inside"),
    ("of", "belonging"),
]


RELATION_CONFIDENCE = {
    "left_of": 1.0,
    "right_of": 1.0,
    "front_of": 1.0,
    "behind": 1.0,
    "north_of": 1.0,
    "south_of": 1.0,
    "above": 0.9,
    "below": 0.85,
    "between": 0.95,
    "closest_to": 0.95,
    "far_from": 0.8,
    "adjacent": 0.6,
    "connected_to": 0.75,
    "surrounded_by": 0.8,
    "facing": 0.8,
    "towards": 0.7,
    "opposite": 0.65,
    "at_corner": 0.8,
    "near_corner": 0.7,
    "at_end": 0.7,
    "on_edge": 0.65,
    "along": 0.6,
    "on_side": 0.7,
    "inside": 0.6,
    "on_surface": 0.5,
    "belonging": 0.35,
    "outside": 0.4,
}


CATEGORY_KEYWORDS = {
    "ground": (
        "ground",
        "grounds",
        "road",
        "roads",
        "street",
        "streets",
        "lane",
        "lanes",
        "intersection",
        "intersections",
        "grass",
        "lawn",
        "clearing",
        "slope",
    ),
    "building": (
        "building",
        "buildings",
        "house",
        "houses",
        "home",
        "homes",
        "suite",
        "suites",
        "station",
        "stations",
        "school",
        "schools",
        "hall",
        "halls",
        "structure",
        "structures",
        "shed",
        "sheds",
    ),
    "parking": (
        "parking",
        "parking lot",
        "parking lots",
        "parking area",
        "parking areas",
        "parking space",
        "parking spaces",
        "car park",
        "car parks",
        "parking stall",
        "parking stalls",
    ),
    "highvegetation": (
        "tree",
        "trees",
        "forest",
        "woods",
        "bush",
        "bushes",
        "hedge",
        "hedges",
        "vegetation",
        "plants",
    ),
    "vehicle": (
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "van",
        "vans",
        "bike",
        "bikes",
        "bicycle",
        "bicycles",
        "motorcycle",
        "motorcycles",
        "bus",
        "buses",
    ),
    "truck": ("truck", "trucks"),
    "fence": ("fence", "fences", "gate", "gates", "wall", "walls"),
    "lightpole": (
        "light pole",
        "light poles",
        "lightpole",
        "lightpoles",
        "street lamp",
        "street lamps",
        "streetlight",
        "streetlights",
        "lamp post",
        "lamp posts",
        "traffic light",
        "traffic lights",
    ),
}


ANCHOR_CATEGORY_PRIORITY = (
    "lightpole",
    "building",
    "parking",
    "truck",
    "vehicle",
    "highvegetation",
    "fence",
    "ground",
)


NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "couple": 2,
    "several": 3,
    "few": 3,
    "many": 4,
}


PRONOUN_ANCHORS = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
GROUND_EXCLUSION_PHRASES = (
    "street lamp",
    "streetlight",
    "traffic light",
    "lamp post",
    "light pole",
    "lightpole",
)


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


def keyword_matches(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def canonicalize_category(category: str) -> str:
    category = (category or "").strip().lower()
    alias = {
        "car": "vehicle",
        "cars": "vehicle",
        "van": "vehicle",
        "vans": "vehicle",
        "bike": "vehicle",
        "bikes": "vehicle",
        "bicycle": "vehicle",
        "bicycles": "vehicle",
        "motorcycle": "vehicle",
        "motorcycles": "vehicle",
        "tree": "highvegetation",
        "trees": "highvegetation",
        "forest": "highvegetation",
        "wall": "fence",
        "walls": "fence",
        "gate": "fence",
        "gates": "fence",
        "streetlight": "lightpole",
        "streetlights": "lightpole",
        "street lamp": "lightpole",
        "street lamps": "lightpole",
        "lamp post": "lightpole",
        "lamp posts": "lightpole",
        "light pole": "lightpole",
        "light poles": "lightpole",
    }
    return alias.get(category, category)


def extract_anchor_categories(anchor_text: str) -> List[str]:
    anchor_text = (anchor_text or "").strip().lower()
    if not anchor_text or anchor_text in PRONOUN_ANCHORS:
        return []

    matches: List[str] = []
    for category in ANCHOR_CATEGORY_PRIORITY:
        if category == "ground" and any(phrase in anchor_text for phrase in GROUND_EXCLUSION_PHRASES):
            continue
        keywords = CATEGORY_KEYWORDS[category]
        if any(keyword_matches(anchor_text, keyword) for keyword in keywords):
            matches.append(category)

    seen = set()
    ordered = []
    for category in matches:
        category = canonicalize_category(category)
        if category not in seen:
            seen.add(category)
            ordered.append(category)
    return ordered


def split_clauses(description: str) -> List[str]:
    text = (description or "").strip().lower()
    if not text:
        return []
    parts = re.split(r"[,;]|\band\b|\bbut\b", text)
    return [part.strip() for part in parts if part.strip()]


def infer_count_hint_from_text(text: str) -> Optional[int]:
    text = (text or "").strip().lower()
    if not text:
        return None
    match = re.search(r"\b(\d+|one|two|three|four|five|couple|several|few|many)\b", text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def extract_count_hint(clause: str, keyword: str) -> Optional[int]:
    pattern = rf"\b(\d+|one|two|three|four|five|couple|several|few|many)\s+{re.escape(keyword)}\b"
    match = re.search(pattern, clause)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def normalize_relation(raw_relation: str, relation_type: str) -> tuple[Optional[str], float]:
    phrase = (raw_relation or "").strip().lower()
    for key, relation in PHRASE_TO_RELATION:
        if key in phrase:
            confidence = RELATION_CONFIDENCE.get(relation, 0.5)
            if key in {"in", "on", "of"}:
                confidence = min(confidence, 0.3)
            elif len(key.split()) == 1:
                confidence = min(confidence, 0.55)
            return relation, confidence

    relation_type = (relation_type or "").strip().lower()
    if relation_type == "distance":
        return "adjacent", 0.4
    if relation_type in {"topological", "directional", "ordinal", "composite"}:
        return None, 0.0
    return None, 0.0


def infer_relation_from_clause(clause: str) -> tuple[Optional[str], float, str]:
    best_key = ""
    best_relation = None
    best_confidence = 0.0
    for key, relation in PHRASE_TO_RELATION:
        if key in clause and len(key) > len(best_key):
            best_key = key
            best_relation = relation
            best_confidence = RELATION_CONFIDENCE.get(relation, 0.5)
    return best_relation, best_confidence, best_key


def infer_fallback_objects(
    description: str,
    main_category: str,
    existing_categories: List[str],
) -> List[Dict[str, object]]:
    inferred: List[Dict[str, object]] = []
    seen = set()
    allowed_categories = {"vehicle", "truck", "highvegetation", "fence", "lightpole"}

    for clause in split_clauses(description):
        relation, confidence, phrase = infer_relation_from_clause(clause)
        if not relation or confidence < 0.35:
            continue

        for category, keywords in CATEGORY_KEYWORDS.items():
            category = canonicalize_category(category)
            if category not in allowed_categories:
                continue
            if category == main_category and relation not in {"between", "on_side"}:
                continue

            matched_keywords = [kw for kw in keywords if keyword_matches(clause, kw)]
            if not matched_keywords:
                continue

            matched_keyword = max(matched_keywords, key=len)
            count_hint = extract_count_hint(clause, matched_keyword) or infer_count_hint_from_text(clause)
            dedup_key = (category, relation, matched_keyword)
            if dedup_key in seen:
                continue

            if category in existing_categories and (count_hint or 1) <= 1:
                continue

            inferred.append(
                {
                    "category": category,
                    "spatial_relation": phrase,
                    "relation_type": "fallback_text",
                    "reference_anchor": matched_keyword,
                    "count_hint": count_hint,
                    "fallback_confidence": round(confidence * 0.9, 3),
                }
            )
            seen.add(dedup_key)
            if len(inferred) >= 1:
                return inferred

    return inferred


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
        category = canonicalize_category(item.get("category", ""))
        if item.get("is_main") and not category:
            category = canonicalize_category(data.get("object_name", ""))
        graph.nodes.append(
            HyperNode(
                id=node_id,
                category=category,
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
    if not main_node.category:
        main_node.category = canonicalize_category(data.get("object_name", ""))

    existing_categories = [node.category for node in graph.nodes if not node.is_main and node.category]
    if not existing_categories:
        for extra_idx, item in enumerate(
            infer_fallback_objects(graph.description, main_node.category, existing_categories)
        ):
            graph.nodes.append(
                HyperNode(
                    id=f"fallback{extra_idx}",
                    category=item["category"],
                    is_main=False,
                    properties={
                        "spatial_relation": str(item.get("spatial_relation") or ""),
                        "relation_type": str(item.get("relation_type") or ""),
                        "reference_anchor": str(item.get("reference_anchor") or ""),
                        "count_hint": str(item.get("count_hint") or ""),
                        "fallback_confidence": str(item.get("fallback_confidence") or ""),
                    },
                )
            )

    for node in graph.nodes:
        if node.is_main or not node.category:
            continue

        relation, relation_confidence = normalize_relation(
            node.properties.get("spatial_relation", ""),
            node.properties.get("relation_type", ""),
        )
        fallback_confidence = float(node.properties.get("fallback_confidence") or 0.0)
        if fallback_confidence:
            relation_confidence = min(relation_confidence, fallback_confidence)
        if not relation:
            continue

        anchor_text = node.properties.get("reference_anchor", "")
        count_hint = node.properties.get("count_hint", "") or infer_count_hint_from_text(anchor_text) or ""
        graph.edges.append(
            HyperEdge(
                from_node=main_node.id,
                to_node=node.id,
                relation=relation,
                raw_relation=node.properties.get("spatial_relation", ""),
                score=relation_confidence,
                info={
                    "reference_anchor": anchor_text,
                    "anchor_categories": extract_anchor_categories(anchor_text),
                    "target_category": node.category,
                    "count_hint": count_hint,
                },
            )
        )

    return graph

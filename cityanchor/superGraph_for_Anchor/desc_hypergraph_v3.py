"""
描述超图构造 v3（CityAnchor superGraph）

在 superGraph_for_Anchor_new/desc_hypergraph 基础上增强 **区分度**：

1. 短语匹配 **按 key 长度降序**，避免短词（如 \"in\"）抢占长短语（\"in front of\"）。
2. 对 `directional`/`topological` 且短语未命中时，用语义关键词推断方位（left/right/…），避免直接丢边。
3. `reference_anchor` 为代词时降低边权重并标记 `weak_anchor`，减轻 \"it\" 类噪声。
4. `object` 节点 id 按非 main 序号递增，避免 construction 顺序导致 object 编号错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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

RELATION_CONFIDENCE = {
    "left_of": 1.0,
    "right_of": 1.0,
    "front_of": 1.0,
    "behind": 1.0,
    "north_of": 1.0,
    "south_of": 1.0,
    "above": 0.9,
    "below": 0.9,
    "between": 1.0,
    "closest_to": 0.95,
    "far_from": 0.9,
    "adjacent": 0.65,
    "connected_to": 0.75,
    "surrounded_by": 0.8,
    "facing": 0.8,
    "towards": 0.7,
    "opposite": 0.6,
    "at_corner": 0.8,
    "near_corner": 0.65,
    "at_end": 0.7,
    "on_edge": 0.65,
    "along": 0.55,
    "on_side": 0.55,
    "inside": 0.55,
    "on_surface": 0.5,
    "belonging": 0.35,
    "outside": 0.35,
}

CATEGORY_KEYWORDS = {
    "ground": ("ground", "grounds", "road", "roads", "street", "streets", "intersection", "intersections"),
    "building": ("building", "buildings", "house", "houses", "suite", "suites", "station", "stations", "mall", "malls", "school", "schools"),
    "parking": ("parking", "parkings", "parking lot", "parking lots", "car park", "car parks"),
    "highvegetation": ("tree", "trees", "bush", "bushes", "hedge", "hedges", "vegetation"),
    "vehicle": ("vehicle", "vehicles", "car", "cars", "van", "vans", "bike", "bikes", "bicycle", "bicycles", "motorcycle", "motorcycles"),
    "truck": ("truck", "trucks"),
    "fence": ("fence", "fences", "gate", "gates", "wall", "walls"),
}

NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

PRONOUN_ANCHORS = {"it", "its", "they", "them", "their", "this", "that", "these", "those"}
GROUND_EXCLUSION_PHRASES = ("street lamp", "streetlight", "traffic light", "lamp post", "light pole")


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


def extract_anchor_categories(anchor_text: str) -> List[str]:
    anchor_text = (anchor_text or "").strip().lower()
    if not anchor_text or anchor_text in PRONOUN_ANCHORS:
        return []
    if any(phrase in anchor_text for phrase in GROUND_EXCLUSION_PHRASES):
        filtered_ground = False
    else:
        filtered_ground = True

    # Prefer the head object category over parts like wall/roof inside a building phrase.
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["building"]):
        return ["building"]
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["parking"]):
        return ["parking"]
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["truck"]):
        return ["truck"]
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["vehicle"]):
        return ["vehicle"]
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["highvegetation"]):
        return ["highvegetation"]
    if any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["fence"]):
        return ["fence"]
    if filtered_ground and any(keyword_matches(anchor_text, keyword) for keyword in CATEGORY_KEYWORDS["ground"]):
        return ["ground"]
    return []


def keyword_matches(text: str, keyword: str) -> bool:
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def canonicalize_category(category: str) -> str:
    category = (category or "").strip().lower()
    alias = {
        "car": "vehicle",
        "cars": "vehicle",
        "tree": "highvegetation",
        "trees": "highvegetation",
        "bush": "highvegetation",
        "bushes": "highvegetation",
        "wall": "fence",
        "walls": "fence",
        "gate": "fence",
        "gates": "fence",
    }
    return alias.get(category, category)


def split_clauses(description: str) -> List[str]:
    text = (description or "").strip().lower()
    if not text:
        return []
    parts = re.split(r"[,;]|\band\b|\bbut\b", text)
    return [part.strip() for part in parts if part.strip()]


def infer_relation_from_clause(clause: str) -> tuple[Optional[str], float, str]:
    clause_l = (clause or "").strip().lower()
    best_key = ""
    best_relation = None
    best_confidence = 0.0
    for key, relation in sorted(PHRASE_TO_RELATION, key=lambda kv: -len(kv[0])):
        if key in clause_l and len(key) > len(best_key):
            best_key = key
            best_relation = relation
            best_confidence = RELATION_CONFIDENCE.get(relation, 0.5)
    return best_relation, best_confidence, best_key


def infer_count_hint_from_text(text: str) -> Optional[int]:
    text = (text or "").strip().lower()
    if not text:
        return None
    match = re.search(r"\b(\d+|one|two|three|four|five)\b", text)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def extract_count_hint(clause: str, keyword: str) -> Optional[int]:
    pattern = rf"\b(\d+|{'|'.join(NUMBER_WORDS.keys())})\s+{re.escape(keyword)}\b"
    match = re.search(pattern, clause)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return NUMBER_WORDS.get(token)


def infer_fallback_objects(
    description: str,
    main_category: str,
    existing_categories: List[str],
) -> List[Dict[str, object]]:
    existing = set(existing_categories)
    inferred = []
    used_categories = set()

    for clause in split_clauses(description):
        relation, confidence, phrase = infer_relation_from_clause(clause)
        if not relation or confidence < 0.45:
            continue

        for category, keywords in CATEGORY_KEYWORDS.items():
            if category == main_category or category in existing or category in used_categories:
                continue
            matched_keyword = next((kw for kw in keywords if keyword_matches(clause, kw)), "")
            if not matched_keyword:
                continue

            count_hint = extract_count_hint(clause, matched_keyword)
            inferred.append(
                {
                    "category": canonicalize_category(category),
                    "spatial_relation": phrase,
                    "relation_type": "fallback_text",
                    "reference_anchor": matched_keyword,
                    "count_hint": count_hint,
                    "fallback_confidence": round(confidence * 0.85, 3),
                }
            )
            used_categories.add(category)
            break

    return inferred


def _infer_directional_from_tokens(phrase: str) -> Optional[str]:
    """短语未进映射表时，用语义提示推断方位（弱置信）。"""
    if not phrase:
        return None
    if re.search(r"\b(left|west)\b", phrase):
        return "left_of"
    if re.search(r"\b(right|east)\b", phrase):
        return "right_of"
    if re.search(r"\b(front|ahead|before|forward)\b", phrase):
        return "front_of"
    if re.search(r"\b(behind|rear|back|after)\b", phrase):
        return "behind"
    if re.search(r"\bnorth\b", phrase):
        return "north_of"
    if re.search(r"\bsouth\b", phrase):
        return "south_of"
    return None


def normalize_relation(raw_relation: str, relation_type: str) -> tuple[Optional[str], float]:
    phrase = (raw_relation or "").strip().lower()
    best_key_len = -1
    picked_relation = None
    picked_confidence = 0.0
    for key, relation in sorted(PHRASE_TO_RELATION, key=lambda kv: -len(kv[0])):
        if key in phrase and len(key) > best_key_len:
            best_key_len = len(key)
            picked_relation = relation
            confidence = RELATION_CONFIDENCE.get(relation, 0.5)
            if key in {"in", "on", "of", "from"}:
                confidence = min(confidence, 0.35)
            elif len(key.split()) == 1:
                confidence = min(confidence, 0.6)
            picked_confidence = confidence

    if picked_relation is not None:
        return picked_relation, picked_confidence

    relation_type = (relation_type or "").strip().lower()
    if relation_type == "distance":
        return "adjacent", 0.45

    if relation_type == "directional":
        guess = _infer_directional_from_tokens(phrase)
        if guess:
            return guess, min(0.74, RELATION_CONFIDENCE.get(guess, 0.5))

    if relation_type == "topological":
        if re.search(r"\b(inside|within|in)\b", phrase):
            return "inside", 0.52
        if re.search(r"\b(on top|surface)\b", phrase):
            return "on_surface", 0.5

    if relation_type in {"topological", "directional"}:
        return None, 0.0
    return None, 0.0


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

    obj_counter = 0
    for idx, item in enumerate(construction):
        if item.get("is_main"):
            node_id = "main"
        else:
            obj_counter += 1
            node_id = f"object{obj_counter}"
        graph.nodes.append(
            HyperNode(
                id=node_id,
                category=canonicalize_category(item.get("category", "")),
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

    existing_categories = [node.category for node in graph.nodes if not node.is_main]
    for extra_idx, item in enumerate(
        infer_fallback_objects(graph.description, main_node.category, existing_categories)
    ):
        node_id = f"fallback{extra_idx}"
        graph.nodes.append(
            HyperNode(
                id=node_id,
                category=item["category"],
                is_main=False,
                properties={
                    "spatial_relation": item["spatial_relation"],
                    "relation_type": item["relation_type"],
                    "reference_anchor": item["reference_anchor"],
                    "count_hint": str(item.get("count_hint") or ""),
                    "fallback_confidence": str(item.get("fallback_confidence") or ""),
                },
            )
        )

    for node in graph.nodes:
        if node.is_main:
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
        anchor_l = anchor_text.strip().lower()
        score_adj = 1.0
        weak = False
        if anchor_l in PRONOUN_ANCHORS:
            score_adj = 0.42
            weak = True
        elif not anchor_l:
            score_adj = 0.82
        final_score = max(0.05, min(1.0, relation_confidence * score_adj))
        edge_info = {
            "reference_anchor": anchor_text,
            "anchor_categories": extract_anchor_categories(anchor_text),
            "target_category": node.category,
            "count_hint": count_hint,
        }
        if weak:
            edge_info["weak_anchor"] = "true"
        graph.edges.append(
            HyperEdge(
                from_node=main_node.id,
                to_node=node.id,
                relation=relation,
                raw_relation=node.properties.get("spatial_relation", ""),
                score=final_score,
                info=edge_info,
            )
        )

    return graph

"""
Build candidate-specific bbox hypergraphs with higher context recall.

Key changes versus the older pipeline:
1. Default to overlap-based neighborhood inclusion instead of center-only cropping.
2. Expand candidate neighborhoods adaptively from query relation/category hints.
3. Guarantee nearest support objects for important categories such as
   building/tree/fence/lightpole/truck.
4. Build edges from a broad relation set instead of only required_relations.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
from tqdm import tqdm

from spatial_relations import SpatialObject, compute_relation


DEFAULT_DESC_INPUTS = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_final/cityanchor_val_ND_desc_hypergraphs.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_final/cityanchor_val_NO_new_desc_hypergraphs.jsonl",
]
DEFAULT_CANDIDATE_INPUTS = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/cityanchor_val_ND_0324_stage1_tight_v1.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/cityanchor_val_NO_0324_stage1_tight_v1.jsonl",
]
DEFAULT_BBOX_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"
DEFAULT_OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_bbox_hypergraphs_final"


FALLBACK_RELATIONS = frozenset(
    {
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "north_of",
        "south_of",
        "adjacent",
        "belonging",
        "inside",
        "on_side",
        "on_surface",
    }
)

SUPPORTED_PAIRWISE_RELATIONS = {
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "north_of",
    "south_of",
    "above",
    "below",
    "inside",
    "on_surface",
    "belonging",
    "adjacent",
    "opposite",
    "at_corner",
    "near_corner",
    "at_end",
    "on_edge",
    "along",
    "outside",
    "surrounded_by",
    "connected_to",
    "on_side",
    "towards",
    "facing",
    "far_from",
}

CORE_CONSTRUCTION_RELATIONS = {
    "left_of",
    "right_of",
    "front_of",
    "behind",
    "above",
    "below",
    "inside",
    "on_surface",
    "belonging",
    "adjacent",
    "along",
    "outside",
    "surrounded_by",
    "connected_to",
    "on_side",
}

RELATION_THRESHOLDS = {
    "adjacent": 0.22,
    "belonging": 0.2,
    "connected_to": 0.2,
    "on_side": 0.2,
    "on_edge": 0.22,
    "along": 0.2,
    "inside": 0.2,
    "on_surface": 0.2,
    "left_of": 0.35,
    "right_of": 0.35,
    "front_of": 0.35,
    "behind": 0.35,
    "north_of": 0.35,
    "south_of": 0.35,
    "below": 0.25,
    "above": 0.25,
    "towards": 0.25,
    "facing": 0.25,
    "near_corner": 0.18,
    "at_corner": 0.22,
    "at_end": 0.2,
    "outside": 0.25,
    "surrounded_by": 0.2,
    "opposite": 0.4,
    "far_from": 0.45,
    "between": 0.3,
    "closest_to": 1.0,
}

IMPORTANT_CATEGORY_MINIMUMS = {
    "building": 3,
    "highvegetation": 2,
    "fence": 2,
    "lightpole": 2,
    "vehicle": 2,
    "truck": 1,
    "parking": 1,
    "ground": 1,
}

PER_CATEGORY_CAP = {
    "building": 5,
    "highvegetation": 3,
    "fence": 3,
    "lightpole": 3,
    "vehicle": 4,
    "truck": 2,
    "parking": 2,
    "ground": 2,
}

MAX_CONTEXT_NODES = 20


def load_jsonl_items(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_bbox_scene(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("bboxes", [])


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
        "bus": "vehicle",
        "buses": "vehicle",
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
        "traffic light": "lightpole",
        "traffic lights": "lightpole",
        "light pole": "lightpole",
        "light poles": "lightpole",
        "lamp post": "lightpole",
        "lamp posts": "lightpole",
    }
    return alias.get(category, category)


def bbox_object_name(bbox_item: dict) -> str:
    return canonicalize_category(str(bbox_item.get("object_name", "")))


def infer_required_relations(desc_item: dict) -> Set[str]:
    relations: Set[str] = set()
    for edge in desc_item.get("hypergraph", {}).get("edges", []):
        relation = (edge.get("relation") or "").strip()
        if relation:
            relations.add(relation)
    if not relations:
        relations = set(FALLBACK_RELATIONS)
    return relations


def relation_scope_for_construction(desc_item: dict) -> Set[str]:
    relations = set(CORE_CONSTRUCTION_RELATIONS)
    required = infer_required_relations(desc_item)
    relations |= required
    if required & {"adjacent", "belonging", "connected_to", "on_side", "along"}:
        relations |= {"adjacent", "belonging", "connected_to", "on_side", "along"}
    if required & {"front_of", "behind", "towards", "facing"}:
        relations |= {"front_of", "behind"}
    if required & {"left_of", "right_of"}:
        relations |= {"left_of", "right_of"}
    if required & {"inside", "on_surface", "below", "above"}:
        relations |= {"inside", "on_surface", "below", "above"}
    if required & {"outside", "surrounded_by"}:
        relations |= {"outside", "surrounded_by", "adjacent"}
    if "between" in required:
        relations.add("between")
    if "closest_to" in required:
        relations.add("closest_to")
    return relations


def infer_relevant_categories(desc_item: dict) -> Set[str]:
    categories: Set[str] = set()
    hypergraph = desc_item.get("hypergraph", {})
    main_cat = canonicalize_category(hypergraph.get("main_category") or "")
    if main_cat:
        categories.add(main_cat)
    for node in hypergraph.get("nodes", []):
        cat = canonicalize_category(node.get("category") or "")
        if cat:
            categories.add(cat)
    for edge in hypergraph.get("edges", []):
        info = edge.get("info") or {}
        target_category = canonicalize_category(info.get("target_category") or "")
        if target_category:
            categories.add(target_category)
        for anchor_cat in info.get("anchor_categories") or []:
            if anchor_cat:
                categories.add(canonicalize_category(anchor_cat))
    return categories


def infer_category_minima(desc_item: dict) -> Dict[str, int]:
    minima: Dict[str, int] = defaultdict(int)
    relevant = infer_relevant_categories(desc_item)
    for edge in desc_item.get("hypergraph", {}).get("edges", []):
        info = edge.get("info") or {}
        target_category = canonicalize_category(info.get("target_category") or "")
        count_hint = str(info.get("count_hint") or "").strip()
        needed = int(count_hint) if count_hint.isdigit() else 1
        if target_category:
            minima[target_category] = max(minima[target_category], needed)
        for anchor_cat in info.get("anchor_categories") or []:
            anchor_cat = canonicalize_category(anchor_cat)
            if anchor_cat:
                minima[anchor_cat] = max(minima[anchor_cat], 1)

    for cat in relevant:
        if cat in IMPORTANT_CATEGORY_MINIMUMS:
            minima[cat] = max(minima[cat], IMPORTANT_CATEGORY_MINIMUMS[cat])
    return dict(minima)


def bbox_to_bounds(
    bbox: List[float],
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z, w, h, d = bbox[:6]
    if expand_z_factor is None:
        expand_z_factor = expand_factor

    w_expanded = w * expand_factor
    h_expanded = h * expand_factor
    d_expanded = d * expand_z_factor

    bbox_min = np.array([x - w_expanded / 2, y - h_expanded / 2, z - d_expanded / 2])
    bbox_max = np.array([x + w_expanded / 2, y + h_expanded / 2, z + d_expanded / 2])
    center = np.array([x, y, z])
    return center, bbox_min, bbox_max


def bbox_item_to_bounds(bbox_item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return bbox_to_bounds(bbox_item["bbox"], 1.0, 1.0)


def bbox_distance_to_center(bbox_item: dict, target_bbox_center: np.ndarray) -> float:
    center, _, _ = bbox_item_to_bounds(bbox_item)
    return float(np.linalg.norm(center - target_bbox_center))


def compute_candidate_geometry_features(bbox_item: dict) -> dict:
    bbox = bbox_item.get("bbox") or [0.0] * 6
    dims = [float(v) for v in bbox[3:6]]
    major = max(dims[:2]) if len(dims) >= 2 else 0.0
    minor = min(dims[:2]) if len(dims) >= 2 else 0.0
    return {
        "length_m": dims[0] if len(dims) > 0 else 0.0,
        "width_m": dims[1] if len(dims) > 1 else 0.0,
        "height_m": dims[2] if len(dims) > 2 else 0.0,
        "major_m": major,
        "minor_m": minor,
        "aspect_ratio": (major / max(minor, 1e-6)) if major > 0 else 0.0,
    }


def _bbox_intersection_volume(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
) -> float:
    inter_min = np.maximum(bbox_min_a, bbox_min_b)
    inter_max = np.minimum(bbox_max_a, bbox_max_b)
    diff = np.maximum(inter_max - inter_min, 0.0)
    return float(np.prod(diff))


def find_bboxes_in_bbox(
    scene_bboxes: List[dict],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    include_mode: str = "overlap",
    overlap_ratio: float = 0.03,
) -> list[dict]:
    result = []
    for bbox_item in scene_bboxes:
        obj_center, obj_bbox_min, obj_bbox_max = bbox_item_to_bounds(bbox_item)
        if include_mode == "center":
            if np.all(obj_center >= bbox_min) and np.all(obj_center <= bbox_max):
                result.append(bbox_item)
        elif include_mode == "overlap":
            obj_vol = _bbox_intersection_volume(
                obj_bbox_min, obj_bbox_max, obj_bbox_min, obj_bbox_max
            )
            if obj_vol <= 1e-9:
                continue
            inter_vol = _bbox_intersection_volume(
                obj_bbox_min, obj_bbox_max, bbox_min, bbox_max
            )
            if inter_vol / obj_vol >= overlap_ratio:
                result.append(bbox_item)
        else:
            raise ValueError(f"Unknown include_mode: {include_mode}")
    return result


def trim_bbox_pool(
    bbox_items: List[dict],
    candidate_id: int,
    category_minima: Dict[str, int],
    target_bbox_center: np.ndarray,
    max_context_nodes: int = MAX_CONTEXT_NODES,
) -> List[dict]:
    candidate = None
    by_category: Dict[str, List[dict]] = defaultdict(list)
    for item in bbox_items:
        object_id = int(item["object_id"])
        if object_id == candidate_id:
            candidate = item
            continue
        by_category[bbox_object_name(item)].append(item)

    trimmed: List[dict] = [candidate] if candidate is not None else []
    for category, items in by_category.items():
        items = sorted(items, key=lambda x: bbox_distance_to_center(x, target_bbox_center))
        cap = PER_CATEGORY_CAP.get(category, 3)
        cap = max(cap, category_minima.get(category, 0))
        trimmed.extend(items[:cap])

    head = trimmed[:1]
    tail = sorted(trimmed[1:], key=lambda x: bbox_distance_to_center(x, target_bbox_center))
    return head + tail[: max_context_nodes - len(head)]


def supplement_relevant_bboxes(
    scene_bboxes: List[dict],
    current_bboxes: List[dict],
    candidate_bbox: dict,
    relevant_categories: Set[str],
    category_minima: Dict[str, int],
    target_bbox_center: np.ndarray,
) -> List[dict]:
    candidate_id = int(candidate_bbox["object_id"])
    chosen = {int(item["object_id"]): item for item in current_bboxes}
    chosen[candidate_id] = candidate_bbox

    scene_by_category: Dict[str, List[dict]] = defaultdict(list)
    for bbox_item in scene_bboxes:
        scene_by_category[bbox_object_name(bbox_item)].append(bbox_item)

    for category in sorted(relevant_categories):
        need = category_minima.get(category, 0)
        if need <= 0:
            continue
        current_count = sum(
            1 for item in chosen.values()
            if bbox_object_name(item) == category and int(item["object_id"]) != candidate_id
        )
        if current_count >= need:
            continue
        candidates = sorted(
            scene_by_category.get(category, []),
            key=lambda x: bbox_distance_to_center(x, target_bbox_center),
        )
        for item in candidates:
            item_id = int(item["object_id"])
            if item_id == candidate_id or item_id in chosen:
                continue
            chosen[item_id] = item
            current_count += 1
            if current_count >= need:
                break

    if len(chosen) <= 1:
        backup_categories = ["building", "highvegetation", "fence", "lightpole", "vehicle", "truck"]
        for category in backup_categories:
            candidates = sorted(
                scene_by_category.get(category, []),
                key=lambda x: bbox_distance_to_center(x, target_bbox_center),
            )
            for item in candidates[:2]:
                item_id = int(item["object_id"])
                if item_id != candidate_id:
                    chosen[item_id] = item

    return trim_bbox_pool(
        list(chosen.values()),
        candidate_id=candidate_id,
        category_minima=category_minima,
        target_bbox_center=target_bbox_center,
    )


def adaptive_expand_factors(
    desc_item: dict,
    candidate_bbox: dict,
    base_expand_factor: float,
    base_expand_z_factor: Optional[float],
) -> tuple[float, float]:
    main_category = bbox_object_name(candidate_bbox)
    relevant = infer_relevant_categories(desc_item)
    required_relations = infer_required_relations(desc_item)

    expand_xy = base_expand_factor
    if main_category in {"vehicle", "truck", "lightpole"}:
        expand_xy = max(expand_xy, 4.5)
    if main_category in {"building", "ground"}:
        expand_xy = max(expand_xy, 4.0)

    if relevant & {"building", "highvegetation", "fence", "lightpole"}:
        expand_xy += 0.8
    if required_relations & {"adjacent", "belonging", "on_side", "along", "connected_to"}:
        expand_xy += 0.8
    if required_relations & {"between", "opposite", "far_from"}:
        expand_xy += 1.0

    expand_z = base_expand_z_factor if base_expand_z_factor is not None else expand_xy
    if required_relations & {"below", "above", "inside", "on_surface"}:
        expand_z = max(expand_z, expand_xy * 0.9)
    return expand_xy, expand_z


def compute_bbox_relations(
    bbox_list: list,
    relation_scope: Set[str],
    target_bbox_center: Optional[np.ndarray] = None,
    max_bboxes: int = MAX_CONTEXT_NODES,
    default_score_threshold: float = 0.35,
) -> List[dict]:
    edges: List[dict] = []
    if len(bbox_list) < 2 or not relation_scope:
        return edges

    if len(bbox_list) > max_bboxes and target_bbox_center is not None:
        bbox_list = sorted(
            bbox_list,
            key=lambda x: bbox_distance_to_center(x, target_bbox_center),
        )[:max_bboxes]

    spatial_objs = []
    for bbox_item in bbox_list:
        center, obj_bbox_min, obj_bbox_max = bbox_item_to_bounds(bbox_item)
        category = bbox_object_name(bbox_item)
        bbox_id = int(bbox_item["object_id"])
        spatial_objs.append(
            (
                bbox_item,
                SpatialObject(
                    id=f"{category}_{bbox_id}",
                    category=category,
                    center=center,
                    bbox_min=obj_bbox_min,
                    bbox_max=obj_bbox_max,
                ),
            )
        )

    pairwise_relations = relation_scope & SUPPORTED_PAIRWISE_RELATIONS
    for i in range(len(spatial_objs)):
        for j in range(i + 1, len(spatial_objs)):
            bbox_a, obj_a = spatial_objs[i]
            bbox_b, obj_b = spatial_objs[j]
            cat_a = bbox_object_name(bbox_a)
            cat_b = bbox_object_name(bbox_b)
            id_a = int(bbox_a["object_id"])
            id_b = int(bbox_b["object_id"])

            for relation in pairwise_relations:
                threshold = min(default_score_threshold, RELATION_THRESHOLDS.get(relation, default_score_threshold))
                is_valid, score = compute_relation(relation, obj_a, obj_b)
                if is_valid and score >= threshold:
                    edges.append(
                        {
                            "from": f"{cat_a}_{id_a}",
                            "to": f"{cat_b}_{id_b}",
                            "relation": relation,
                            "score": round(float(score), 3),
                        }
                    )

                is_valid_rev, score_rev = compute_relation(relation, obj_b, obj_a)
                if is_valid_rev and score_rev >= threshold:
                    edges.append(
                        {
                            "from": f"{cat_b}_{id_b}",
                            "to": f"{cat_a}_{id_a}",
                            "relation": relation,
                            "score": round(float(score_rev), 3),
                        }
                    )

    if "between" in relation_scope:
        threshold = min(default_score_threshold, RELATION_THRESHOLDS["between"])
        for i, (bbox_a, obj_a) in enumerate(spatial_objs):
            for j, (bbox_b, obj_b) in enumerate(spatial_objs):
                if i == j:
                    continue
                for k, (bbox_c, obj_c) in enumerate(spatial_objs):
                    if i == k or j == k:
                        continue
                    is_valid, score = compute_relation("between", obj_a, obj_b, obj_c)
                    if is_valid and score >= threshold:
                        cat_a = bbox_object_name(bbox_a)
                        cat_b = bbox_object_name(bbox_b)
                        cat_c = bbox_object_name(bbox_c)
                        id_a = int(bbox_a["object_id"])
                        id_b = int(bbox_b["object_id"])
                        id_c = int(bbox_c["object_id"])
                        edges.append(
                            {
                                "from": f"{cat_b}_{id_b}",
                                "to": f"{cat_a}_{id_a}",
                                "relation": "between",
                                "score": round(float(score), 3),
                                "anchors": [f"{cat_c}_{id_c}"],
                            }
                        )

    if "closest_to" in relation_scope:
        by_category: Dict[str, list] = defaultdict(list)
        for bbox_item, obj in spatial_objs:
            by_category[bbox_object_name(bbox_item)].append((bbox_item, obj))

        for same_cat in by_category.values():
            if len(same_cat) < 2:
                continue
            for bbox_a, obj_a in same_cat:
                others = [
                    (bbox_b, obj_b)
                    for bbox_b, obj_b in same_cat
                    if int(bbox_b["object_id"]) != int(bbox_a["object_id"])
                ]
                if not others:
                    continue
                closest_bbox, _ = min(
                    others,
                    key=lambda x: np.linalg.norm(obj_a.center - x[1].center),
                )
                edges.append(
                    {
                        "from": f"{bbox_object_name(bbox_a)}_{int(bbox_a['object_id'])}",
                        "to": f"{bbox_object_name(closest_bbox)}_{int(closest_bbox['object_id'])}",
                        "relation": "closest_to",
                        "score": 1.0,
                    }
                )

    unique_edges = {}
    for edge in edges:
        key = (
            edge["from"],
            edge["to"],
            edge["relation"],
            tuple(edge.get("anchors") or []),
        )
        prev = unique_edges.get(key)
        if prev is None or edge["score"] > prev["score"]:
            unique_edges[key] = edge
    return list(unique_edges.values())


def build_candidate_hypergraph(
    desc_item: dict,
    candidate_bbox: dict,
    scene_bboxes: List[dict],
    expand_factor: float = 4.5,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "overlap",
    instance_overlap_ratio: float = 0.03,
    instance_edge_score_threshold: float = 0.35,
) -> dict:
    candidate_id = int(candidate_bbox["object_id"])
    adaptive_xy, adaptive_z = adaptive_expand_factors(
        desc_item,
        candidate_bbox,
        base_expand_factor=expand_factor,
        base_expand_z_factor=expand_z_factor,
    )
    bbox_center, bbox_min, bbox_max = bbox_to_bounds(
        candidate_bbox["bbox"],
        adaptive_xy,
        adaptive_z,
    )
    bbox_list = find_bboxes_in_bbox(
        scene_bboxes,
        bbox_min,
        bbox_max,
        include_mode=instance_inclusion_mode,
        overlap_ratio=instance_overlap_ratio,
    )

    relevant_categories = infer_relevant_categories(desc_item)
    category_minima = infer_category_minima(desc_item)
    bbox_list = supplement_relevant_bboxes(
        scene_bboxes,
        bbox_list,
        candidate_bbox=candidate_bbox,
        relevant_categories=relevant_categories,
        category_minima=category_minima,
        target_bbox_center=bbox_center,
    )

    nodes = []
    node_by_category: Dict[str, List[str]] = defaultdict(list)
    for bbox_item in bbox_list:
        node_center, node_bbox_min, node_bbox_max = bbox_item_to_bounds(bbox_item)
        node_category = bbox_object_name(bbox_item)
        node_object_id = int(bbox_item["object_id"])
        node_id = f"{node_category}_{node_object_id}"
        node = {
            "id": node_id,
            "category": node_category,
            "center": node_center.tolist(),
            "bbox_min": node_bbox_min.tolist(),
            "bbox_max": node_bbox_max.tolist(),
            "object_id": node_object_id,
        }
        if node_object_id == candidate_id:
            node["geometry_features"] = compute_candidate_geometry_features(bbox_item)
        nodes.append(node)
        node_by_category[node_category].append(node_id)

    relation_scope = relation_scope_for_construction(desc_item)
    required_relations = infer_required_relations(desc_item)
    edges = compute_bbox_relations(
        bbox_list,
        relation_scope=relation_scope,
        target_bbox_center=bbox_center,
        default_score_threshold=instance_edge_score_threshold,
    )

    edges_by_relation: Dict[str, List[dict]] = defaultdict(list)
    for edge in edges:
        edges_by_relation[edge["relation"]].append(edge)

    support_category_counts = {
        category: len(ids) for category, ids in node_by_category.items()
    }
    missing_relevant_categories = sorted(
        category
        for category, need in category_minima.items()
        if support_category_counts.get(category, 0) < need
    )

    return {
        "scene_id": desc_item["scene_id"],
        "bbox_id": candidate_id,
        "query_object_id": desc_item.get("object_id"),
        "ann_id": desc_item.get("ann_id"),
        "object_name": candidate_bbox.get("object_name", "").lower(),
        "description": desc_item.get("description", ""),
        "expand_factor": adaptive_xy,
        "expand_z_factor": adaptive_z,
        "bbox_inclusion_mode": instance_inclusion_mode,
        "bbox_overlap_ratio": instance_overlap_ratio,
        "bbox_edge_score_threshold": instance_edge_score_threshold,
        "bbox_center": bbox_center.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "candidate_geometry": compute_candidate_geometry_features(candidate_bbox),
        "description_hypergraph": desc_item.get("hypergraph", {}),
        "hypergraph": {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "required_relations": sorted(required_relations),
            "constructed_relations": sorted(relation_scope),
            "relevant_categories": sorted(relevant_categories),
            "category_minima": category_minima,
            "missing_relevant_categories": missing_relevant_categories,
            "node_by_category": dict(node_by_category),
            "support_category_counts": support_category_counts,
            "edges_by_relation": dict(edges_by_relation),
        },
    }


def process_pair(
    desc_path: str,
    candidates_path: str,
    bbox_root: str,
    output_path: str,
    expand_factor: float = 4.5,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "overlap",
    instance_overlap_ratio: float = 0.03,
    instance_edge_score_threshold: float = 0.35,
) -> tuple[int, int]:
    desc_items = load_jsonl_items(desc_path)
    candidate_items = load_jsonl_items(candidates_path)
    print(f"Processing desc file: {desc_path}")
    print(f"  Loaded {len(desc_items)} description items")
    print(f"  Loaded {len(candidate_items)} candidate items")

    if len(desc_items) != len(candidate_items):
        raise ValueError(
            f"desc/candidates line count mismatch: {desc_path} ({len(desc_items)}) vs "
            f"{candidates_path} ({len(candidate_items)})"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bbox_cache: Dict[str, List[dict]] = {}
    success = 0
    failed = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for desc_item, cand_item in tqdm(
            zip(desc_items, candidate_items),
            total=len(desc_items),
            desc=os.path.basename(desc_path),
        ):
            scene_id = str(desc_item["scene_id"])
            if str(cand_item.get("scene_id")) != scene_id:
                failed += 1
                print(
                    f"  Failed: desc/candidates scene mismatch desc={scene_id} "
                    f"candidates={cand_item.get('scene_id')}"
                )
                continue

            if scene_id not in bbox_cache:
                bbox_path = os.path.join(bbox_root, f"{scene_id}_bbox.json")
                if not os.path.exists(bbox_path):
                    print(f"  Missing bbox file: {bbox_path}")
                    bbox_cache[scene_id] = []
                else:
                    bbox_cache[scene_id] = load_bbox_scene(bbox_path)

            scene_bboxes = bbox_cache.get(scene_id, [])
            if not scene_bboxes:
                failed += 1
                continue

            bbox_map = {str(x["object_id"]): x for x in scene_bboxes}
            candidate_ids = []
            seen = set()
            for candidate_id in cand_item.get("candidates", []) or []:
                cid = str(candidate_id)
                if cid not in seen:
                    seen.add(cid)
                    candidate_ids.append(cid)

            if not candidate_ids:
                failed += 1
                continue

            for cid in candidate_ids:
                candidate_bbox = bbox_map.get(cid)
                if candidate_bbox is None:
                    failed += 1
                    continue
                try:
                    hypergraph = build_candidate_hypergraph(
                        desc_item,
                        candidate_bbox,
                        scene_bboxes,
                        expand_factor=expand_factor,
                        expand_z_factor=expand_z_factor,
                        instance_inclusion_mode=instance_inclusion_mode,
                        instance_overlap_ratio=instance_overlap_ratio,
                        instance_edge_score_threshold=instance_edge_score_threshold,
                    )
                    fout.write(json.dumps(hypergraph, ensure_ascii=False) + "\n")
                    success += 1
                except Exception as e:
                    failed += 1
                    print(
                        f"  Failed: scene={scene_id} query_object_id={desc_item.get('object_id')} "
                        f"candidate_bbox_id={cid} ann_id={desc_item.get('ann_id')} error={e}"
                    )

    return success, failed


def default_output_path(desc_path: str, output_dir: str) -> str:
    name = os.path.splitext(os.path.basename(desc_path))[0]
    return os.path.join(output_dir, f"{name}_bbox_only_hypergraphs.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-root", type=str, default=DEFAULT_BBOX_ROOT)
    parser.add_argument("--desc-inputs", type=str, nargs="+", default=DEFAULT_DESC_INPUTS)
    parser.add_argument(
        "--candidate-inputs",
        type=str,
        nargs="+",
        default=DEFAULT_CANDIDATE_INPUTS,
        help="Stage1 candidate jsonl files aligned 1:1 with --desc-inputs",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expand-factor", type=float, default=4.5)
    parser.add_argument("--expand-z-factor", type=float, default=None)
    parser.add_argument(
        "--instance-inclusion-mode",
        type=str,
        default="overlap",
        choices=["center", "overlap"],
    )
    parser.add_argument("--instance-overlap-ratio", type=float, default=0.03)
    parser.add_argument("--instance-edge-score-threshold", type=float, default=0.35)
    parser.add_argument(
        "--glob-desc",
        type=str,
        default="",
        help="Optional glob for desc inputs.",
    )
    args = parser.parse_args()

    desc_inputs = list(args.desc_inputs)
    if args.glob_desc:
        desc_inputs.extend(sorted(glob.glob(args.glob_desc)))
    candidate_inputs = list(args.candidate_inputs)

    seen = set()
    unique_desc_inputs = []
    for path in desc_inputs:
        if path not in seen:
            unique_desc_inputs.append(path)
            seen.add(path)

    if len(unique_desc_inputs) != len(candidate_inputs):
        raise ValueError(
            f"--desc-inputs count ({len(unique_desc_inputs)}) must match "
            f"--candidate-inputs count ({len(candidate_inputs)})"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print("Building bbox-based candidate hypergraphs from desc + stage1 + bbox")
    print(f"  bbox_root: {args.bbox_root}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  desc files: {len(unique_desc_inputs)}")
    print(f"  candidate files: {len(candidate_inputs)}")
    print(
        f"  expand_factor: {args.expand_factor}  include_mode: {args.instance_inclusion_mode}  "
        f"overlap_ratio: {args.instance_overlap_ratio}  edge_threshold: {args.instance_edge_score_threshold}"
    )

    total_success = 0
    total_failed = 0
    for desc_path, candidates_path in zip(unique_desc_inputs, candidate_inputs):
        output_path = default_output_path(desc_path, args.output_dir)
        success, failed = process_pair(
            desc_path=desc_path,
            candidates_path=candidates_path,
            bbox_root=args.bbox_root,
            output_path=output_path,
            expand_factor=args.expand_factor,
            expand_z_factor=args.expand_z_factor,
            instance_inclusion_mode=args.instance_inclusion_mode,
            instance_overlap_ratio=args.instance_overlap_ratio,
            instance_edge_score_threshold=args.instance_edge_score_threshold,
        )
        total_success += success
        total_failed += failed
        print(f"  Saved to: {output_path}")
        print(f"  Success: {success} | Failed: {failed}")

    print("=" * 60)
    print(f"Total success: {total_success}")
    print(f"Total failed: {total_failed}")


if __name__ == "__main__":
    main()

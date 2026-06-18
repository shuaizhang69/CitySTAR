"""
超图匹配 v7 —— **v5（RRF + 一致性 + 歧义自适应，弱化手工权重）**

流程：v1 融合 + 几何并列 → **Top-M 几何整段重排** → 对前 **chunk（默认 80）** 个 bbox 中 **仅前 rrf_head（默认 48）**
名做下列重排（尾部保持 Top-M 序不变）：

1. **RRF**：对几何分 / 超图分 / Stage1 候选序 三个**名次表**做倒数排名融合（默认 k=60，经典 IR），
   避免 min-max 被单个异常值拉伸。
2. **Top-band agreement**：统计每个 bbox 在三个名次里有多少落在「前 n/3」（自适应带宽）内，
   名次靠前者优先（多通道一致 → 更稳）。
3. **歧义分支**：若 chunk 内超图分相对间距很小（多视图不可靠），再用 **RRF + top-band agreement**；
   若间距足够大则 **直接按超图分为主**（RRF / 几何仅作细分并列）。

不设 `--blend-wg/h/r` 网格；可选 `--rrf-k` 单独调节 RRF 平滑项。

评估：**仅统计 GT ∈ Stage1 candidates｜matched**（主打印）。
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Set

from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMPORT_CANDIDATE_DIRS = [
    CURRENT_DIR,
    os.path.join(os.path.dirname(CURRENT_DIR), "our_data", "superGraph_for_Anchor"),
]
for path in IMPORT_CANDIDATE_DIRS:
    if os.path.exists(os.path.join(path, "match_hypergraphs.py")) and path not in sys.path:
        sys.path.insert(0, path)
NEW_SUPERGRAPH_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "our_data", "superGraph_for_Anchor_new")
if os.path.exists(os.path.join(NEW_SUPERGRAPH_DIR, "match_hypergraphs.py")) and NEW_SUPERGRAPH_DIR not in sys.path:
    sys.path.insert(0, NEW_SUPERGRAPH_DIR)

from match_hypergraphs import get_desc_main_category, rank_prior_from_index

_GEOM_MOD = None


def _get_geometry_match_module():
    """07_match_geometry_candidates.py（文件名以数字开头，用 importlib 加载）。"""
    global _GEOM_MOD
    if _GEOM_MOD is not None:
        return _GEOM_MOD
    geom_path = os.path.join(CURRENT_DIR, "07_match_geometry_candidates.py")
    spec = importlib.util.spec_from_file_location("geom_match07", geom_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _GEOM_MOD = mod
    return mod


RELATION_SEMANTIC_MAP = {
    "front_of": ["front_of", "facing", "towards", "opposite"],
    "behind": ["behind", "opposite"],
    "left_of": ["left_of"],
    "right_of": ["right_of"],
    "belonging": ["belonging", "adjacent", "next_to", "on_edge", "on_side", "connected_to"],
    "adjacent": ["adjacent", "next_to", "on_edge", "on_side", "connected_to", "belonging"],
    "next_to": ["next_to", "adjacent", "on_edge", "on_side", "connected_to"],
    "inside": ["inside", "surrounded_by"],
    "on_surface": ["on_surface", "adjacent", "belonging"],
    "between": ["between"],
    "at_corner": ["at_corner", "near_corner"],
    "near_corner": ["near_corner", "at_corner"],
    "at_end": ["at_end", "on_edge"],
    "opposite": ["opposite", "far_from", "front_of", "behind"],
    "facing": ["facing", "front_of", "towards"],
    "above": ["above"],
    "below": ["below"],
    "far_from": ["far_from", "opposite"],
    "closest_to": ["closest_to"],
    "along": ["along", "connected_to", "on_edge"],
    "outside": ["outside"],
    "connected_to": ["connected_to", "adjacent", "next_to"],
    "on_side": ["on_side", "on_edge", "adjacent"],
    "towards": ["towards", "facing", "front_of"],
}

SAFE_SOFT_RELATIONS = {"adjacent", "front_of", "left_of", "right_of", "south_of"}

DEFAULT_TOP_M_GEOM = 80
DEFAULT_RRF_CHUNK = 0
DEFAULT_RRF_K = 60.0
# 仅重排 chunk 前若干名，其余保持 Top-M 后顺序，减少长尾换位对 @20 的副作用
DEFAULT_RRF_HEAD_REORDER = 48


def _dense_ranks_desc(score_map: Dict[str, float]) -> Dict[str, int]:
    """名次 1=最优；同分按 bbox_id 字典序决次序（名次致密）。"""
    if not score_map:
        return {}
    items = sorted(score_map.items(), key=lambda x: (-x[1], x[0]))
    return {bid: i + 1 for i, (bid, _) in enumerate(items)}


# Stage-1 候选由 01_stage1.py（收紧颜色软匹配）生成，见 cityanchor_*_stage1_tight_v1.jsonl
DEFAULT_TASKS = [
    {
        "name": "ND",
        "candidates": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/cityanchor_val_ND_0324_stage1_tight_v1.jsonl",
        "desc_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_final/cityanchor_val_ND_desc_hypergraphs.jsonl",
        "bbox_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_bbox_hypergraphs_final/cityanchor_val_ND_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        "geometry_mentions": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/geometry_final/cityanchor_val_ND_geometry_mentions.jsonl",
        "bbox_dir_geometry": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match_final/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_final_v1.json",
    },
    {
        "name": "NO",
        "candidates": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/cityanchor_val_NO_0324_stage1_tight_v1.jsonl",
        "desc_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_final/cityanchor_val_NO_desc_hypergraphs.jsonl",
        "bbox_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_bbox_hypergraphs/cityanchor_val_NO_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        "geometry_mentions": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/geometry_final/cityanchor_val_NO_geometry_mentions.jsonl",
        "bbox_dir_geometry": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match_final/cityanchor_val_NO_0324_stage1_tight_v1_hgmatch_final_v1.json",
    },
]


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
        "bush": "highvegetation",
        "bushes": "highvegetation",
        "wall": "fence",
        "walls": "fence",
        "gate": "fence",
        "gates": "fence",
        "streetlight": "lightpole",
        "streetlights": "lightpole",
        "light pole": "lightpole",
        "light poles": "lightpole",
        "street lamp": "lightpole",
        "street lamps": "lightpole",
        "traffic light": "lightpole",
        "traffic lights": "lightpole",
        "lamp post": "lightpole",
        "lamp posts": "lightpole",
    }
    return alias.get(category, category)


def resolve_existing_path(path_or_paths):
    if os.path.exists(path_or_paths):
        return path_or_paths
    raise FileNotFoundError(path_or_paths)


def load_desc_hypergraphs_with_ann(desc_jsonl: str) -> dict:
    desc_graphs = {}
    with open(desc_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            key = (
                str(data["scene_id"]),
                str(data["object_id"]),
                str(data.get("ann_id", 0)),
            )
            desc_graphs[key] = data
    return desc_graphs


def slim_bbox_graph_record(data: dict) -> dict:
    hg = data.get("hypergraph", {})
    slim_nodes = []
    for node in hg.get("nodes", []):
        slim_node = {
            "id": node.get("id"),
            "category": node.get("category"),
            "center": node.get("center"),
            "bbox_min": node.get("bbox_min"),
            "bbox_max": node.get("bbox_max"),
            "object_id": node.get("object_id"),
        }
        if "geometry_features" in node:
            slim_node["geometry_features"] = node.get("geometry_features")
        slim_nodes.append(slim_node)

    slim_edges = []
    for edge in hg.get("edges", []):
        slim_edge = {
            "from": edge.get("from"),
            "to": edge.get("to"),
            "relation": edge.get("relation"),
            "score": edge.get("score", 0.0),
        }
        if edge.get("anchors"):
            slim_edge["anchors"] = edge.get("anchors")
        slim_edges.append(slim_edge)

    return {
        "scene_id": data.get("scene_id"),
        "bbox_id": data.get("bbox_id"),
        "query_object_id": data.get("query_object_id", data.get("object_id", data.get("bbox_id"))),
        "ann_id": data.get("ann_id", 0),
        "object_name": data.get("object_name", ""),
        "candidate_geometry": data.get("candidate_geometry") or {},
        "hypergraph": {
            "nodes": slim_nodes,
            "edges": slim_edges,
            "node_count": hg.get("node_count", len(slim_nodes)),
            "edge_count": hg.get("edge_count", len(slim_edges)),
            "missing_relevant_categories": hg.get("missing_relevant_categories") or [],
            "support_category_counts": hg.get("support_category_counts") or {},
        },
    }


def iter_bbox_hypergraph_groups_jsonl(bbox_jsonl: str):
    with open(bbox_jsonl, "r", encoding="utf-8") as f:
        current_group_key = None
        current_group = {}
        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            data = slim_bbox_graph_record(json.loads(line))
            group_key = (
                str(data["scene_id"]),
                str(data.get("query_object_id")),
                str(data.get("ann_id", 0)),
            )
            if current_group_key is None:
                current_group_key = group_key
            elif group_key != current_group_key:
                yield current_group_key, current_group
                current_group_key = group_key
                current_group = {}
            current_group[str(data["bbox_id"])] = data
        if current_group_key is not None:
            yield current_group_key, current_group


def load_stage1_candidates_jsonl(candidates_jsonl: str) -> list:
    items = []
    with open(candidates_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_geometry_mentions_jsonl(path: str) -> dict:
    """(scene_id, object_id, ann_id) -> 06_extract_geometry_mentions 行。"""
    out: dict[tuple[str, str, str], dict] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = (
                str(rec.get("scene_id", "")),
                str(rec.get("object_id", "")),
                str(rec.get("ann_id", 0)),
            )
            out[k] = rec
    return out


def _geometry_tie_eligible(
    geom_item: Optional[dict], object_name: str, desc_graph: dict, gm
) -> bool:
    """有几何 mentions，且（jsonl 标记 has_geometry 或描述走软几何关系）。"""
    if not geom_item:
        return False
    mentions = gm.collect_relevant_mentions(geom_item, str(object_name))
    if not mentions:
        return False
    ge = geom_item.get("geometry_extraction") or {}
    if ge.get("has_geometry"):
        return True
    return should_use_soft_geometry(desc_graph)


def geometry_scores_for_tied_subset(
    gm,
    geom_item: dict,
    scene_id: str,
    object_name: str,
    tied_bbox_ids: List[str],
    bbox_dir: str,
    bbox_cache: dict,
    infer_half_extent: bool = True,
    story_height_m: float = 3.2,
) -> dict[str, float]:
    """
    对「融合分并列」的 bbox_id 子集，用 07 的几何打分（同类全场景统计 + 仅 tied 上算 mention 分）。
    """
    tied_set = {str(x) for x in tied_bbox_ids}
    if len(tied_set) < 2:
        return {}
    try:
        scene_bbox = gm.load_scene_bboxes(scene_id, bbox_dir, bbox_cache)
    except FileNotFoundError:
        return {}

    onorm = gm.normalize_category(str(object_name))
    full_entries = []
    for bbox_entry in scene_bbox.get("bboxes") or []:
        if gm.normalize_category(str(bbox_entry.get("object_name", ""))) != onorm:
            continue
        oid = str(bbox_entry.get("object_id"))
        full_entries.append(
            {
                "object_id": oid,
                "features": gm.candidate_geometry_features(
                    bbox_entry, infer_half_extent=infer_half_extent
                ),
            }
        )
    if len(full_entries) < 2:
        return {}

    class_stats = gm.summarize_class_stats(full_entries)
    relevant = gm.collect_relevant_mentions(geom_item, str(object_name))
    out: dict[str, float] = {}
    for cand in full_entries:
        oid = cand["object_id"]
        if oid not in tied_set:
            continue
        best = 0.0
        for m in relevant:
            s, _ = gm.mention_match_score(
                m, cand["features"], class_stats, story_height_m
            )
            best = max(best, s)
        out[oid] = best
    return out


def geometry_scores_for_candidate_pool(
    gm,
    geom_item: dict,
    scene_id: str,
    object_name: str,
    candidate_bbox_ids: List[str],
    bbox_dir: str,
    bbox_cache: dict,
    infer_half_extent: bool = True,
    story_height_m: float = 3.2,
) -> dict[str, float]:
    candidate_set = {str(x) for x in candidate_bbox_ids}
    if not candidate_set:
        return {}
    try:
        scene_bbox = gm.load_scene_bboxes(scene_id, bbox_dir, bbox_cache)
    except FileNotFoundError:
        return {}

    onorm = gm.normalize_category(str(object_name))
    full_entries = []
    for bbox_entry in scene_bbox.get("bboxes") or []:
        if gm.normalize_category(str(bbox_entry.get("object_name", ""))) != onorm:
            continue
        oid = str(bbox_entry.get("object_id"))
        full_entries.append(
            {
                "object_id": oid,
                "features": gm.candidate_geometry_features(
                    bbox_entry, infer_half_extent=infer_half_extent
                ),
            }
        )
    if not full_entries:
        return {}

    class_stats = gm.summarize_class_stats(full_entries)
    relevant = gm.collect_relevant_mentions(geom_item, str(object_name))
    if not relevant:
        return {}

    out: dict[str, float] = {}
    for cand in full_entries:
        oid = cand["object_id"]
        if oid not in candidate_set:
            continue
        best = 0.0
        for mention in relevant:
            score, _ = gm.mention_match_score(
                mention, cand["features"], class_stats, story_height_m
            )
            best = max(best, score)
        out[oid] = best
    return out


def build_node_category_map(desc_graph: dict, allowed_node_ids: set | None = None) -> dict:
    node_categories = {}
    main_cat = canonicalize_category(desc_graph.get("hypergraph", {}).get("main_category", ""))
    for node in desc_graph.get("hypergraph", {}).get("nodes", []):
        node_id = node["id"]
        if allowed_node_ids is not None and node_id not in allowed_node_ids and node_id != "main":
            continue
        if node_id == "main":
            node_categories[node_id] = main_cat
        else:
            node_categories[node_id] = canonicalize_category(node.get("category", ""))
    return node_categories


def get_instances_by_category(bbox_nodes: list) -> dict:
    by_cat = defaultdict(list)
    for node in bbox_nodes:
        by_cat[canonicalize_category(node["category"])].append(node["id"])
    return dict(by_cat)


def check_edge_match(desc_edge: dict, bbox_edges: list, node_mapping: dict) -> float:
    rel_type = desc_edge["relation"]
    from_node = desc_edge["from"]
    to_node = desc_edge["to"]
    bbox_from = node_mapping.get(from_node)
    bbox_to = node_mapping.get(to_node)
    if not bbox_from or not bbox_to:
        return 0.0

    equivalent_rels = RELATION_SEMANTIC_MAP.get(rel_type, [rel_type])
    desc_weight = float(desc_edge.get("score", 1.0))
    best_score = 0.0
    for bbox_edge in bbox_edges:
        if bbox_edge["from"] == bbox_from and bbox_edge["to"] == bbox_to:
            edge_rel = bbox_edge["relation"]
            if edge_rel == rel_type:
                edge_score = min(1.0, float(bbox_edge.get("score", 1.0)))
                best_score = max(best_score, edge_score * desc_weight)
            elif edge_rel in equivalent_rels:
                edge_score = float(bbox_edge.get("score", 1.0)) * 0.5
                best_score = max(best_score, edge_score * desc_weight)
    return best_score


def evaluate_mapping(desc_edges: list, bbox_edges: list, node_mapping: dict) -> tuple[float, float]:
    total_score = 0.0
    matched_weight = 0.0
    total_weight = 0.0
    for desc_edge in desc_edges:
        edge_weight = float(desc_edge.get("score", 1.0))
        total_weight += edge_weight
        score = check_edge_match(desc_edge, bbox_edges, node_mapping)
        if score > 0:
            total_score += score
            matched_weight += edge_weight
    if total_weight > 0:
        match_ratio = matched_weight / total_weight
        avg_score = total_score / total_weight
        return (match_ratio + avg_score) / 2.0, match_ratio
    return 0.0, 0.0


def get_candidate_main_instance_id(bbox_graph: dict) -> str | None:
    bbox_id = str(bbox_graph.get("bbox_id"))
    object_name = canonicalize_category(bbox_graph.get("object_name", ""))
    if not object_name:
        return None
    candidate_node = f"{object_name}_{bbox_id}"
    node_ids = {node["id"] for node in bbox_graph.get("hypergraph", {}).get("nodes", [])}
    return candidate_node if candidate_node in node_ids else None


def get_effective_desc_edges(desc_graph: dict) -> list:
    edges = list(desc_graph.get("hypergraph", {}).get("edges", []))
    if len(edges) <= 2:
        return edges

    strong_edges = [edge for edge in edges if float(edge.get("score", 1.0)) >= 0.5]
    if len(strong_edges) >= 2:
        return strong_edges

    return sorted(edges, key=lambda edge: float(edge.get("score", 1.0)), reverse=True)[:2]


def compute_node_coverage(desc_graph: dict, bbox_graph: dict) -> float:
    desc_nodes_by_cat = defaultdict(int)
    for node in desc_graph.get("hypergraph", {}).get("nodes", []):
        if node.get("is_main"):
            continue
        cat = canonicalize_category(node.get("category", ""))
        if cat:
            desc_nodes_by_cat[cat] += 1
    if not desc_nodes_by_cat:
        return 0.0

    bbox_nodes_by_cat = defaultdict(int)
    for node in bbox_graph.get("hypergraph", {}).get("nodes", []):
        cat = canonicalize_category(node.get("category", ""))
        if cat:
            bbox_nodes_by_cat[cat] += 1

    covered = 0.0
    total = 0.0
    for cat, needed in desc_nodes_by_cat.items():
        total += needed
        covered += min(needed, bbox_nodes_by_cat.get(cat, 0))
    return covered / total if total > 0 else 0.0


def compute_anchor_support(desc_graph: dict, bbox_graph: dict) -> float:
    bbox_categories = {
        canonicalize_category(node.get("category", ""))
        for node in bbox_graph.get("hypergraph", {}).get("nodes", [])
    }
    checks = 0
    hits = 0
    for edge in get_effective_desc_edges(desc_graph):
        anchor_categories = (edge.get("info") or {}).get("anchor_categories") or []
        if not anchor_categories:
            continue
        checks += 1
        if any(canonicalize_category(anchor_cat) in bbox_categories for anchor_cat in anchor_categories):
            hits += 1
    if checks == 0:
        return 0.0
    return hits / checks


def compute_count_support(desc_graph: dict, bbox_graph: dict) -> float:
    expected_counts = {}
    for edge in get_effective_desc_edges(desc_graph):
        info = edge.get("info") or {}
        target_category = canonicalize_category(info.get("target_category", ""))
        count_hint = str(info.get("count_hint", "")).strip()
        if not target_category or not count_hint.isdigit():
            continue
        expected_counts[target_category] = max(expected_counts.get(target_category, 0), int(count_hint))

    if not expected_counts:
        return 0.0

    actual_counts = defaultdict(int)
    for node in bbox_graph.get("hypergraph", {}).get("nodes", []):
        category = canonicalize_category(node.get("category", ""))
        if category:
            actual_counts[category] += 1

    total = 0.0
    for category, expected in expected_counts.items():
        if expected <= 0:
            continue
        total += min(actual_counts.get(category, 0) / expected, 1.0)

    return total / len(expected_counts)


def get_bbox_category_counts(bbox_graph: dict) -> dict[str, int]:
    counts = defaultdict(int)
    for node in bbox_graph.get("hypergraph", {}).get("nodes", []):
        category = canonicalize_category(node.get("category", ""))
        if category:
            counts[category] += 1
    return dict(counts)


def compute_target_category_presence(desc_graph: dict, bbox_graph: dict) -> float:
    expected = defaultdict(int)
    for edge in get_effective_desc_edges(desc_graph):
        info = edge.get("info") or {}
        category = canonicalize_category(info.get("target_category", ""))
        count_hint = str(info.get("count_hint", "")).strip()
        need = int(count_hint) if count_hint.isdigit() else 1
        if category:
            expected[category] = max(expected[category], need)
    if not expected:
        return 0.0
    actual = get_bbox_category_counts(bbox_graph)
    total = 0.0
    for category, need in expected.items():
        total += min(actual.get(category, 0) / max(need, 1), 1.0)
    return total / len(expected)


def compute_anchor_category_presence(desc_graph: dict, bbox_graph: dict) -> float:
    expected = []
    for edge in get_effective_desc_edges(desc_graph):
        for anchor_cat in (edge.get("info") or {}).get("anchor_categories") or []:
            anchor_cat = canonicalize_category(anchor_cat)
            if anchor_cat and anchor_cat not in expected:
                expected.append(anchor_cat)
    if not expected:
        return 0.0
    actual = get_bbox_category_counts(bbox_graph)
    hits = sum(1 for category in expected if actual.get(category, 0) > 0)
    return hits / len(expected)


def compute_structure_penalty(
    desc_graph: dict,
    bbox_graph: dict,
    *,
    relation_score: float,
    soft_relation_support: float,
    anchor_support: float,
    target_presence: float,
    anchor_presence: float,
) -> float:
    desc_edges = get_effective_desc_edges(desc_graph)
    if not desc_edges:
        return 0.0

    hg = bbox_graph.get("hypergraph", {})
    node_count = int(hg.get("node_count", len(hg.get("nodes", []))))
    edge_count = int(hg.get("edge_count", len(hg.get("edges", []))))
    missing_relevant = hg.get("missing_relevant_categories") or []
    penalty = 0.0
    if node_count <= 1:
        penalty += 0.55
    elif node_count <= 2:
        penalty += 0.25

    if edge_count == 0:
        penalty += 0.3
    elif edge_count <= 2 and relation_score <= 0.0:
        penalty += 0.12

    penalty += 0.45 * max(0.0, 1.0 - target_presence)
    if anchor_presence == 0.0 and any((edge.get("info") or {}).get("anchor_categories") for edge in desc_edges):
        penalty += 0.18

    if node_count >= 14 and relation_score < 0.45 and anchor_support <= 0.01:
        penalty += min(0.35, 0.025 * (node_count - 13))

    if missing_relevant:
        penalty += min(0.18, 0.06 * len(missing_relevant))

    return penalty


def get_node_lookup(bbox_graph: dict) -> dict:
    return {
        node["id"]: node
        for node in bbox_graph.get("hypergraph", {}).get("nodes", [])
    }


def get_bbox_size(node: dict) -> tuple[float, float, float]:
    bbox_min = node.get("bbox_min", [0.0, 0.0, 0.0])
    bbox_max = node.get("bbox_max", [0.0, 0.0, 0.0])
    return (
        max(0.0, float(bbox_max[0]) - float(bbox_min[0])),
        max(0.0, float(bbox_max[1]) - float(bbox_min[1])),
        max(0.0, float(bbox_max[2]) - float(bbox_min[2])),
    )


def get_xy_gap(node_a: dict, node_b: dict) -> float:
    a_min = node_a.get("bbox_min", [0.0, 0.0, 0.0])
    a_max = node_a.get("bbox_max", [0.0, 0.0, 0.0])
    b_min = node_b.get("bbox_min", [0.0, 0.0, 0.0])
    b_max = node_b.get("bbox_max", [0.0, 0.0, 0.0])
    dx = max(0.0, float(a_min[0]) - float(b_max[0]), float(b_min[0]) - float(a_max[0]))
    dy = max(0.0, float(a_min[1]) - float(b_max[1]), float(b_min[1]) - float(a_max[1]))
    return math.hypot(dx, dy)


def get_z_gap(node_a: dict, node_b: dict) -> float:
    a_min = node_a.get("bbox_min", [0.0, 0.0, 0.0])
    a_max = node_a.get("bbox_max", [0.0, 0.0, 0.0])
    b_min = node_b.get("bbox_min", [0.0, 0.0, 0.0])
    b_max = node_b.get("bbox_max", [0.0, 0.0, 0.0])
    return max(0.0, float(a_min[2]) - float(b_max[2]), float(b_min[2]) - float(a_max[2]))


def is_center_inside(inner_node: dict, outer_node: dict) -> bool:
    center = inner_node.get("center", [0.0, 0.0, 0.0])
    outer_min = outer_node.get("bbox_min", [0.0, 0.0, 0.0])
    outer_max = outer_node.get("bbox_max", [0.0, 0.0, 0.0])
    return all(float(outer_min[i]) <= float(center[i]) <= float(outer_max[i]) for i in range(3))


def overlap_ratio(inner_node: dict, outer_node: dict) -> float:
    inner_min = inner_node.get("bbox_min", [0.0, 0.0, 0.0])
    inner_max = inner_node.get("bbox_max", [0.0, 0.0, 0.0])
    outer_min = outer_node.get("bbox_min", [0.0, 0.0, 0.0])
    outer_max = outer_node.get("bbox_max", [0.0, 0.0, 0.0])
    inter = []
    inner_dims = []
    for i in range(3):
        lo = max(float(inner_min[i]), float(outer_min[i]))
        hi = min(float(inner_max[i]), float(outer_max[i]))
        inter.append(max(0.0, hi - lo))
        inner_dims.append(max(0.0, float(inner_max[i]) - float(inner_min[i])))
    inner_vol = inner_dims[0] * inner_dims[1] * inner_dims[2]
    if inner_vol <= 1e-9:
        return 0.0
    inter_vol = inter[0] * inter[1] * inter[2]
    return max(0.0, min(1.0, inter_vol / inner_vol))


def relation_soft_score(relation: str, main_node: dict, target_node: dict) -> float:
    main_center = [float(x) for x in main_node.get("center", [0.0, 0.0, 0.0])]
    target_center = [float(x) for x in target_node.get("center", [0.0, 0.0, 0.0])]
    dx = main_center[0] - target_center[0]
    dy = main_center[1] - target_center[1]
    dz = main_center[2] - target_center[2]
    xy_gap = get_xy_gap(main_node, target_node)
    z_gap = get_z_gap(main_node, target_node)
    main_size = get_bbox_size(main_node)
    target_size = get_bbox_size(target_node)
    scale_xy = max(3.0, 0.35 * (main_size[0] + main_size[1] + target_size[0] + target_size[1]))
    scale_z = max(2.0, 0.5 * (main_size[2] + target_size[2]))

    if relation in {"adjacent", "connected_to", "on_side", "near_corner", "belonging"}:
        base = 1.0 / (1.0 + xy_gap / scale_xy)
        vertical = 1.0 / (1.0 + z_gap / scale_z)
        return max(0.0, min(1.0, base * vertical))

    if relation in {"front_of", "south_of", "towards", "facing"}:
        dir_mag = max(0.0, dy)
        orth = abs(dx) + 0.5 * abs(dz)
        return max(0.0, min(1.0, (dir_mag / (dir_mag + orth + 2.0)) * min(1.0, dir_mag / scale_xy)))

    if relation in {"behind", "north_of"}:
        dir_mag = max(0.0, -dy)
        orth = abs(dx) + 0.5 * abs(dz)
        return max(0.0, min(1.0, (dir_mag / (dir_mag + orth + 2.0)) * min(1.0, dir_mag / scale_xy)))

    if relation == "left_of":
        dir_mag = max(0.0, -dx)
        orth = abs(dy) + 0.5 * abs(dz)
        return max(0.0, min(1.0, (dir_mag / (dir_mag + orth + 2.0)) * min(1.0, dir_mag / scale_xy)))

    if relation == "right_of":
        dir_mag = max(0.0, dx)
        orth = abs(dy) + 0.5 * abs(dz)
        return max(0.0, min(1.0, (dir_mag / (dir_mag + orth + 2.0)) * min(1.0, dir_mag / scale_xy)))

    if relation == "inside":
        center_in = 1.0 if is_center_inside(main_node, target_node) else 0.0
        overlap = overlap_ratio(main_node, target_node)
        return max(center_in * 0.5 + overlap * 0.5, overlap * 0.6)

    if relation == "on_surface":
        target_top = float(target_node.get("bbox_max", [0.0, 0.0, 0.0])[2])
        height_diff = abs(main_center[2] - target_top)
        base = 1.0 / (1.0 + xy_gap / scale_xy)
        height_score = 1.0 / (1.0 + height_diff / scale_z)
        return max(0.0, min(1.0, base * height_score))

    if relation == "above":
        dir_mag = max(0.0, dz)
        return max(0.0, min(1.0, dir_mag / (dir_mag + xy_gap + 1.0)))

    if relation == "below":
        dir_mag = max(0.0, -dz)
        return max(0.0, min(1.0, dir_mag / (dir_mag + xy_gap + 1.0)))

    if relation == "far_from":
        center_dist = math.dist(main_center, target_center)
        return max(0.0, min(1.0, center_dist / 40.0))

    if relation == "opposite":
        center_dist = math.dist(main_center, target_center)
        return max(0.0, min(1.0, center_dist / 35.0)) * 0.6

    if relation == "along":
        base = 1.0 / (1.0 + xy_gap / scale_xy)
        axis_balance = max(abs(dx), abs(dy)) / (abs(dx) + abs(dy) + 1.0)
        return max(0.0, min(1.0, base * axis_balance))

    return 0.0


def get_candidate_targets_for_edge(desc_edge: dict, bbox_graph: dict, main_node_id: str) -> list:
    target_categories = []
    info = desc_edge.get("info") or {}
    target_category = canonicalize_category(info.get("target_category", ""))
    if target_category:
        target_categories.append(target_category)
    for anchor_cat in info.get("anchor_categories") or []:
        anchor_cat = canonicalize_category(anchor_cat)
        if anchor_cat and anchor_cat not in target_categories:
            target_categories.append(anchor_cat)
    if not target_categories:
        return []

    nodes = []
    for node in bbox_graph.get("hypergraph", {}).get("nodes", []):
        if node["id"] == main_node_id:
            continue
        if canonicalize_category(node.get("category", "")) in target_categories:
            nodes.append(node)
    return nodes


def compute_soft_relation_support(desc_graph: dict, bbox_graph: dict) -> float:
    desc_edges = get_effective_desc_edges(desc_graph)
    if not desc_edges:
        return 0.0

    main_node_id = get_candidate_main_instance_id(bbox_graph)
    if main_node_id is None:
        return 0.0
    node_lookup = get_node_lookup(bbox_graph)
    main_node = node_lookup.get(main_node_id)
    if main_node is None:
        return 0.0

    total = 0.0
    total_weight = 0.0
    for desc_edge in desc_edges:
        edge_weight = float(desc_edge.get("score", 1.0))
        total_weight += edge_weight
        candidate_nodes = get_candidate_targets_for_edge(desc_edge, bbox_graph, main_node_id)
        if not candidate_nodes:
            continue

        candidate_scores = sorted(
            (relation_soft_score(desc_edge["relation"], main_node, node) for node in candidate_nodes),
            reverse=True,
        )
        if not candidate_scores:
            continue

        count_hint = str((desc_edge.get("info") or {}).get("count_hint", "")).strip()
        needed = int(count_hint) if count_hint.isdigit() else 1
        needed = max(1, min(needed, len(candidate_scores)))
        top_support = sum(candidate_scores[:needed]) / needed
        distractor = candidate_scores[needed] if len(candidate_scores) > needed else 0.0
        ambiguity_penalty = min(0.35, distractor * 0.25)
        total += max(0.0, top_support - ambiguity_penalty) * edge_weight

    if total_weight <= 0:
        return 0.0
    return total / total_weight


def should_use_soft_geometry(desc_graph: dict) -> bool:
    relation_set = {
        (edge.get("relation") or "").strip()
        for edge in get_effective_desc_edges(desc_graph)
        if (edge.get("relation") or "").strip()
    }
    if not relation_set:
        return False
    return relation_set.issubset(SAFE_SOFT_RELATIONS)


def find_best_mapping(desc_graph: dict, bbox_graph: dict) -> tuple:
    desc_edges = get_effective_desc_edges(desc_graph)
    bbox_edges = bbox_graph.get("hypergraph", {}).get("edges", [])
    bbox_nodes = bbox_graph.get("hypergraph", {}).get("nodes", [])
    candidate_main_instance = get_candidate_main_instance_id(bbox_graph)
    if candidate_main_instance is None:
        return 0.0, {}, 0.0

    active_node_ids = {"main"}
    for edge in desc_edges:
        active_node_ids.add(edge["from"])
        active_node_ids.add(edge["to"])
    desc_node_cats = build_node_category_map(desc_graph, active_node_ids)
    main_cat = canonicalize_category(desc_graph.get("hypergraph", {}).get("main_category", ""))
    if not main_cat:
        for node in desc_graph.get("hypergraph", {}).get("nodes", []):
            if node.get("is_main", False):
                main_cat = node.get("category", "")
                break

    bbox_by_cat = get_instances_by_category(bbox_nodes)
    desc_nodes_by_cat = defaultdict(list)
    for node_id, cat in desc_node_cats.items():
        if node_id != "main":
            desc_nodes_by_cat[cat].append(node_id)
    best_score = 0.0
    best_mapping = {}
    best_match_ratio = 0.0
    base_mapping = {"main": candidate_main_instance}
    category_choices = []
    for cat, desc_nodes in desc_nodes_by_cat.items():
        bbox_insts = [bid for bid in bbox_by_cat.get(cat, []) if bid != candidate_main_instance]
        category_choices.append((cat, desc_nodes, bbox_insts))

    def generate_mappings(cat_idx, current_mapping):
        nonlocal best_score, best_mapping, best_match_ratio
        if cat_idx == len(category_choices):
            score, match_ratio = evaluate_mapping(desc_edges, bbox_edges, current_mapping)
            if score > best_score:
                best_score = score
                best_match_ratio = match_ratio
                best_mapping = current_mapping.copy()
            return

        _, desc_nodes, bbox_insts = category_choices[cat_idx]
        if not desc_nodes:
            generate_mappings(cat_idx + 1, current_mapping)
            return

        limited_bbox_insts = bbox_insts[: min(8, len(bbox_insts))]
        usable_count = min(len(desc_nodes), len(limited_bbox_insts))
        if usable_count == 0:
            generate_mappings(cat_idx + 1, current_mapping)
            return

        nodes_to_map = desc_nodes[:usable_count]
        for perm in permutations(limited_bbox_insts, usable_count):
            mapping = current_mapping.copy()
            for i, desc_node in enumerate(nodes_to_map):
                mapping[desc_node] = perm[i]
            generate_mappings(cat_idx + 1, mapping)

    generate_mappings(0, base_mapping)
    return best_score, best_mapping, best_match_ratio


def compute_match_score(desc_graph: dict, bbox_graph: dict) -> float:
    relation_score, _, match_ratio = find_best_mapping(desc_graph, bbox_graph)
    soft_relation_support = compute_soft_relation_support(desc_graph, bbox_graph)
    node_coverage = compute_node_coverage(desc_graph, bbox_graph)
    anchor_support = compute_anchor_support(desc_graph, bbox_graph)
    count_support = compute_count_support(desc_graph, bbox_graph)
    target_presence = compute_target_category_presence(desc_graph, bbox_graph)
    anchor_presence = compute_anchor_category_presence(desc_graph, bbox_graph)
    use_soft_geometry = should_use_soft_geometry(desc_graph)
    structure_bonus = 0.2 * target_presence + 0.12 * anchor_presence
    penalty = compute_structure_penalty(
        desc_graph,
        bbox_graph,
        relation_score=relation_score,
        soft_relation_support=soft_relation_support,
        anchor_support=anchor_support,
        target_presence=target_presence,
        anchor_presence=anchor_presence,
    )
    score = relation_score + node_coverage + anchor_support + count_support + structure_bonus
    if relation_score > 0:
        score += match_ratio
    if use_soft_geometry:
        score += soft_relation_support
    return score - penalty


def evaluate_matching_pipeline2(
    name: str,
    candidates_jsonl: str,
    desc_hypergraphs_jsonl: str,
    bbox_hypergraphs_jsonl: str,
    output_json: Optional[str],
    score_eps: float = 1e-9,
    fusion_lambda: float = 1.0,
    use_geometry_tiebreak: bool = True,
    geometry_mentions_jsonl: Optional[str] = None,
    bbox_dir_geometry: Optional[str] = None,
    top_m_geom: int = DEFAULT_TOP_M_GEOM,
    rrf_chunk_size: int = DEFAULT_RRF_CHUNK,
    rrf_head_reorder: int = DEFAULT_RRF_HEAD_REORDER,
    rrf_k: float = DEFAULT_RRF_K,
    enable_rrf_v5: bool = True,
    quiet: bool = False,
    return_stats_only: bool = False,
):
    if not quiet:
        print(f"\n=== Running task: {name} ===")
        print("Loading pipeline2 data...")
    desc_graphs = load_desc_hypergraphs_with_ann(desc_hypergraphs_jsonl)
    eval_data = load_stage1_candidates_jsonl(candidates_jsonl)
    geometry_map = (
        load_geometry_mentions_jsonl(geometry_mentions_jsonl)
        if use_geometry_tiebreak and geometry_mentions_jsonl
        else {}
    )
    gm = _get_geometry_match_module() if use_geometry_tiebreak else None
    bbox_cache_geom: dict = {}
    bbox_group_iter = iter_bbox_hypergraph_groups_jsonl(bbox_hypergraphs_jsonl)
    current_bbox_group_key = None
    current_bbox_group = {}

    if not quiet:
        print(f"Loaded {len(desc_graphs)} description graphs")
        print("Loaded bbox hypergraphs in streaming mode")
        print(f"Loaded {len(eval_data)} candidate queries")
        print(f"fusion_lambda={fusion_lambda}")
        print(
            f"use_geometry_tiebreak={use_geometry_tiebreak} geometry_rows={len(geometry_map)} bbox_dir={bbox_dir_geometry or '—'}"
        )
        print(
            f"v5 top_m_geom={top_m_geom} rrf_chunk={rrf_chunk_size} rrf_head={rrf_head_reorder} rrf_k={rrf_k}"
        )

    stats = {
        "total": 0,
        "matched": 0,
        "correct_top1": 0,
        "correct_top5": 0,
        "correct_top10": 0,
        "correct_top20": 0,
        "queries_with_score_tie": 0,
        "queries_with_fused_tie": 0,
        "tie_break_used": 0,
        "correct_top1_when_tie_break": 0,
        "geometry_tiebreak_eligible": 0,
        "geometry_tiebreak_scored": 0,
        "geometry_tiebreak_changed_winner": 0,
        "correct_top1_when_geom_changed_winner": 0,
        "geometry_front_used": 0,
        "geom_topm_rerank_used": 0,
        "eval_gt_in_cand_pool": 0,
        "correct_top1_pool": 0,
        "correct_top5_pool": 0,
        "correct_top10_pool": 0,
        "correct_top20_pool": 0,
        "rrf_v5_used": 0,
    }
    results = []

    _iter = (
        tqdm(eval_data, desc=f"Matching {name}") if not quiet else eval_data
    )
    for query in _iter:
        scene_id = str(query["scene_id"])
        object_id = str(query["object_id"])
        ann_id = str(query.get("ann_id", 0))
        query_group_key = (scene_id, object_id, ann_id)
        candidates = [str(x) for x in query.get("candidates", [])]
        if not candidates:
            continue
        if current_bbox_group_key != query_group_key:
            current_bbox_group = {}
            for group_key, group_data in bbox_group_iter:
                current_bbox_group_key = group_key
                current_bbox_group = group_data
                if current_bbox_group_key == query_group_key:
                    break
            if current_bbox_group_key != query_group_key:
                current_bbox_group = {}
        stats["total"] += 1

        desc_graph = desc_graphs.get((scene_id, object_id, ann_id))
        if desc_graph is None:
            results.append({
                "scene_id": scene_id,
                "object_id": object_id,
                "ann_id": ann_id,
                "status": "no_desc_graph",
            })
            continue

        rank_by_id: Dict[str, int] = {}
        for i, cid in enumerate(candidates):
            if cid not in rank_by_id:
                rank_by_id[cid] = i

        ordered_unique = []
        seen: Set[str] = set()
        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                ordered_unique.append(cid)

        n_unique = len(ordered_unique)
        desc_main_cat = get_desc_main_category(desc_graph)
        all_scores: Dict[str, float] = {}
        cached_graphs: Dict[str, dict] = {}
        geom_front_scores: Dict[str, float] = {}
        geom_item = geometry_map.get((scene_id, object_id, ann_id)) if geometry_map else None

        for bbox_id in ordered_unique:
            bbox_graph = current_bbox_group.get(bbox_id)
            if bbox_graph is None:
                continue
            cached_graphs[bbox_id] = bbox_graph
            all_scores[bbox_id] = compute_match_score(desc_graph, bbox_graph)

        if (
            gm
            and bbox_dir_geometry
            and geom_item
            and _geometry_tie_eligible(geom_item, query.get("object_name", ""), desc_graph, gm)
            and all_scores
        ):
            geom_front_scores = geometry_scores_for_candidate_pool(
                gm,
                geom_item,
                scene_id,
                str(query.get("object_name", "")),
                list(all_scores.keys()),
                bbox_dir_geometry,
                bbox_cache_geom,
            )
            if geom_front_scores:
                geom_ranked = sorted(geom_front_scores.values(), reverse=True)
                top_geom = geom_ranked[0] if geom_ranked else 0.0
                second_geom = geom_ranked[1] if len(geom_ranked) > 1 else 0.0
                if top_geom >= 0.82 and (top_geom - second_geom) >= 0.18:
                    for bid in list(all_scores.keys()):
                        all_scores[bid] += 0.18 * float(geom_front_scores.get(bid, 0.0))
                    stats["geometry_front_used"] += 1

        if not all_scores:
            results.append({
                "scene_id": scene_id,
                "object_id": object_id,
                "ann_id": ann_id,
                "status": "failed",
            })
            continue

        stats["matched"] += 1
        max_s = max(all_scores.values())
        tied_ids = [k for k, v in all_scores.items() if abs(v - max_s) <= score_eps]
        if len(tied_ids) > 1:
            stats["queries_with_score_tie"] += 1

        all_fused_scores: Dict[str, float] = {}
        for bid in all_scores.keys():
            r = rank_by_id.get(bid, 10**9)
            rp = rank_prior_from_index(r, n_unique)
            hg = all_scores[bid]
            all_fused_scores[bid] = fusion_lambda * hg + (1.0 - fusion_lambda) * rp

        max_f = max(all_fused_scores.values())
        fused_tied = [k for k, v in all_fused_scores.items() if abs(v - max_f) <= score_eps]
        tie_break_used = len(fused_tied) > 1
        if tie_break_used:
            stats["queries_with_fused_tie"] += 1
            stats["tie_break_used"] += 1

        # 不使用 GT 做 tie-break；并列时先用 07 几何分，再退回 rank prior + bbox_id。
        def pick_key(bid: str):
            r = rank_by_id.get(bid, 10**9)
            return (r, bid)

        geom_scores: dict[str, float] = {}
        baseline_winner = min(fused_tied, key=pick_key)
        if (
            gm
            and bbox_dir_geometry
            and tie_break_used
            and geom_item
            and _geometry_tie_eligible(geom_item, query.get("object_name", ""), desc_graph, gm)
        ):
            stats["geometry_tiebreak_eligible"] += 1
            geom_scores = geometry_scores_for_tied_subset(
                gm,
                geom_item,
                scene_id,
                str(query.get("object_name", "")),
                fused_tied,
                bbox_dir_geometry,
                bbox_cache_geom,
            )
            if geom_scores and len(geom_scores) >= 2:
                stats["geometry_tiebreak_scored"] += 1

        def fused_sort_key(item: tuple[str, float]) -> tuple:
            bid, fused = item
            if geom_scores and tie_break_used and bid in fused_tied:
                gs = geom_scores.get(bid, 0.0)
                return (-fused, -gs, pick_key(bid)[0], pick_key(bid)[1])
            return (-fused, pick_key(bid)[0], pick_key(bid)[1])

        sorted_fused = sorted(all_fused_scores.items(), key=fused_sort_key)

        geom_item_top = geometry_map.get((scene_id, object_id, ann_id)) if geometry_map else None
        if (
            gm
            and bbox_dir_geometry
            and geom_item_top
            and _geometry_tie_eligible(
                geom_item_top, query.get("object_name", ""), desc_graph, gm
            )
            and len(sorted_fused) >= 2
        ):
            top_m_ids = [p[0] for p in sorted_fused[:top_m_geom]]
            geom_m = geometry_scores_for_tied_subset(
                gm,
                geom_item_top,
                scene_id,
                str(query.get("object_name", "")),
                top_m_ids,
                bbox_dir_geometry,
                bbox_cache_geom,
            )
            if geom_m and len(geom_m) >= 2:
                geom_vals = sorted(geom_m.values(), reverse=True)
                top_geom = geom_vals[0] if geom_vals else 0.0
                second_geom = geom_vals[1] if len(geom_vals) > 1 else 0.0
                if top_geom < 0.82 or (top_geom - second_geom) < 0.18:
                    geom_m = {}
            if geom_m and len(geom_m) >= 2:
                pos = {bid: i for i, (bid, _) in enumerate(sorted_fused)}
                top_set = set(top_m_ids)
                top_sorted = sorted(
                    top_m_ids,
                    key=lambda bid: (-geom_m.get(bid, 0.0), pos.get(bid, 10**9)),
                )
                rest = [pair for pair in sorted_fused if pair[0] not in top_set]
                sorted_fused = [
                    (bid, all_fused_scores[bid]) for bid in top_sorted
                ] + rest
                stats["geom_topm_rerank_used"] += 1

        # v5：chunk 内 RRF + top-band agreement；超图平局则几何优先
        geom_item_blend = geometry_map.get((scene_id, object_id, ann_id)) if geometry_map else None
        if (
            enable_rrf_v5
            and rrf_chunk_size >= 2
            and gm
            and bbox_dir_geometry
            and geom_item_blend
            and _geometry_tie_eligible(
                geom_item_blend, query.get("object_name", ""), desc_graph, gm
            )
            and len(sorted_fused) >= 2
        ):
            bchunk = min(rrf_chunk_size, len(sorted_fused))
            chunk_ids = [p[0] for p in sorted_fused[:bchunk]]
            geom_b = geometry_scores_for_tied_subset(
                gm,
                geom_item_blend,
                scene_id,
                str(query.get("object_name", "")),
                chunk_ids,
                bbox_dir_geometry,
                bbox_cache_geom,
            )
            raw_g = {bid: float(geom_b.get(bid, 0.0)) for bid in chunk_ids}
            raw_h = {bid: float(all_scores[bid]) for bid in chunk_ids}
            r_geom = _dense_ranks_desc(raw_g)
            r_hyper = _dense_ranks_desc(raw_h)
            r_stage = {bid: rank_by_id.get(bid, 10**9) + 1 for bid in chunk_ids}
            kk = float(rrf_k)
            rrf = {}
            for bid in chunk_ids:
                rg = r_geom.get(bid, bchunk)
                rh = r_hyper.get(bid, bchunk)
                rs = r_stage.get(bid, bchunk)
                rrf[bid] = (
                    1.0 / (kk + rg)
                    + 1.0 / (kk + rh)
                    + 1.0 / (kk + rs)
                )
            band = max(3, bchunk // 3)

            def agree_ct(bid: str) -> int:
                return (
                    int(r_geom.get(bid, bchunk) <= band)
                    + int(r_hyper.get(bid, bchunk) <= band)
                    + int(r_stage.get(bid, bchunk) <= band)
                )

            agr = {bid: agree_ct(bid) for bid in chunk_ids}
            hv = list(raw_h.values())
            h_spread = (max(hv) - min(hv)) if hv else 0.0
            h_hi = max(abs(x) for x in hv) if hv else 0.0
            h_rel_spread = h_spread / (h_hi + 1e-12)
            # 超图在 chunk 内有显著间距 → 以超图为主（避免无谓打乱强信号）；否则多视图 RRF
            hyper_discriminative = h_rel_spread > 0.04

            def sort_key_disc(bid: str):
                return (-raw_h[bid], -rrf[bid], -raw_g[bid], bid)

            def sort_key_ambiguous(bid: str):
                """chunk 内超图拥挤：先抬「多通道一致」，再用分数；RRF 仅细并列。"""
                flat_local = h_spread < 1e-9 or (
                    h_hi > 1e-12 and (h_spread / h_hi) < 0.03
                )
                if flat_local:
                    return (
                        -agr[bid],
                        -raw_g[bid],
                        -raw_h[bid],
                        -r_stage.get(bid, bchunk),
                        -rrf[bid],
                        bid,
                    )
                return (
                    -agr[bid],
                    -raw_h[bid],
                    -raw_g[bid],
                    -r_stage.get(bid, bchunk),
                    -rrf[bid],
                    bid,
                )

            H = min(rrf_head_reorder, bchunk)
            head_ids = chunk_ids[:H]
            tail_ids = chunk_ids[H:]

            if hyper_discriminative:
                head_sorted = sorted(head_ids, key=sort_key_disc)
            else:
                head_sorted = sorted(head_ids, key=sort_key_ambiguous)
            chunk_sorted = head_sorted + tail_ids
            cset = set(chunk_ids)
            rest2 = [pair for pair in sorted_fused if pair[0] not in cset]
            sorted_fused = [
                (bid, all_fused_scores[bid]) for bid in chunk_sorted
            ] + rest2
            stats["rrf_v5_used"] += 1

        best_bbox_id = sorted_fused[0][0]
        if tie_break_used and geom_scores and baseline_winner != best_bbox_id:
            stats["geometry_tiebreak_changed_winner"] += 1

        best_fused_score = all_fused_scores[best_bbox_id]

        top5_ids = [x[0] for x in sorted_fused[:5]]
        top10_ids = [x[0] for x in sorted_fused[:10]]
        top20_ids = [x[0] for x in sorted_fused[:20]]
        is_correct = (best_bbox_id == object_id)

        if is_correct:
            stats["correct_top1"] += 1
        if tie_break_used and is_correct:
            stats["correct_top1_when_tie_break"] += 1
        if (
            tie_break_used
            and geom_scores
            and baseline_winner != best_bbox_id
            and is_correct
        ):
            stats["correct_top1_when_geom_changed_winner"] += 1
        if object_id in top5_ids:
            stats["correct_top5"] += 1
        if object_id in top10_ids:
            stats["correct_top10"] += 1
        if object_id in top20_ids:
            stats["correct_top20"] += 1

        cand_set = {str(x) for x in candidates}
        if object_id in cand_set:
            stats["eval_gt_in_cand_pool"] += 1
            if is_correct:
                stats["correct_top1_pool"] += 1
            if object_id in top5_ids:
                stats["correct_top5_pool"] += 1
            if object_id in top10_ids:
                stats["correct_top10_pool"] += 1
            if object_id in top20_ids:
                stats["correct_top20_pool"] += 1

        results.append({
            "scene_id": scene_id,
            "object_id": object_id,
            "ann_id": ann_id,
            "status": "matched",
            "predicted_bbox_id": best_bbox_id,
            "fused_score": best_fused_score,
            "hypergraph_score": all_scores.get(best_bbox_id),
            "is_correct": is_correct,
            "tie_break_used": tie_break_used,
            "geometry_tiebreak_baseline_winner": baseline_winner if tie_break_used else None,
            "geometry_front_scores": geom_front_scores if geom_front_scores else None,
            "geometry_tiebreak_scores": geom_scores if geom_scores else None,
            "top5_ids": top5_ids,
            "top10_ids": top10_ids,
            "top20_ids": top20_ids,
            "all_scores": all_scores,
            "all_fused_scores": all_fused_scores,
        })

    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    if not quiet:
        print("\n" + "=" * 60)
        print(f"{name} 结果统计:")
        print(f"  总查询数: {stats['total']}")
        total = stats["total"]
        if total > 0:
            print(
                f"  匹配成功: {stats['matched']} ({stats['matched']/total*100:.1f}%)"
            )
            print(
                f"  Top-1 正确: {stats['correct_top1']} ({stats['correct_top1']/total*100:.1f}%)"
            )
            print(
                f"  Top-5 包含: {stats['correct_top5']} ({stats['correct_top5']/total*100:.1f}%)"
            )
            print(
                f"  Top-10 包含: {stats['correct_top10']} ({stats['correct_top10']/total*100:.1f}%)"
            )
            print(
                f"  Top-20 包含: {stats['correct_top20']} ({stats['correct_top20']/total*100:.1f}%)"
            )
        else:
            print("  匹配成功: 0 (0.0%)")
            print("  Top-1 正确: 0 (0.0%)")
            print("  Top-5 包含: 0 (0.0%)")
            print("  Top-10 包含: 0 (0.0%)")
            print("  Top-20 包含: 0 (0.0%)")
        print(f"  纯超图分并列: {stats['queries_with_score_tie']}")
        print(f"  融合分仍并列: {stats['queries_with_fused_tie']}")
        tb = stats["tie_break_used"]
        if total > 0:
            print(
                f"  融合分并列后用 rank/bbox_id 打破: {tb} ({tb/total*100:.1f}%)"
            )
        else:
            print("  融合分并列后用 rank/bbox_id 打破: 0 (0.0%)")
        if tb > 0:
            print(
                f"  其中 Top-1 正确: {stats['correct_top1_when_tie_break']} ({stats['correct_top1_when_tie_break']/tb*100:.1f}%)"
            )
        ge = stats["geometry_tiebreak_eligible"]
        if ge > 0:
            print(
                f"  几何并列重排 eligible: {ge}，实际打分子集: {stats['geometry_tiebreak_scored']}，"
                f"改变 Top-1: {stats['geometry_tiebreak_changed_winner']}"
            )
            ch = stats["geometry_tiebreak_changed_winner"]
            if ch > 0:
                print(
                    f"  几何改序后 Top-1 正确: {stats['correct_top1_when_geom_changed_winner']} / {ch}"
                )
        pool = stats.get("eval_gt_in_cand_pool", 0)
        if pool > 0:
            print(
                f"\n  【仅 GT∈Stage1候选｜matched】n={pool}（主对比口径，不含 GT 未进候选）"
            )
            print(
                f"    Top-1: {stats['correct_top1_pool']} ({stats['correct_top1_pool']/pool*100:.1f}%)"
            )
            print(
                f"    Top-5: {stats['correct_top5_pool']} ({stats['correct_top5_pool']/pool*100:.1f}%)"
            )
            print(
                f"    Top-10: {stats['correct_top10_pool']} ({stats['correct_top10_pool']/pool*100:.1f}%)"
            )
            print(
                f"    Top-20: {stats['correct_top20_pool']} ({stats['correct_top20_pool']/pool*100:.1f}%)"
            )
        print(
            f"  Top-{top_m_geom} 几何重排生效次数: {stats.get('geom_topm_rerank_used', 0)}"
        )
        print(f"  几何主分前置次数: {stats.get('geometry_front_used', 0)}")
        print(f"  v5 RRF 重排调用: {stats.get('rrf_v5_used', 0)}")
        print(f"\n结果保存至: {output_json}")

    if return_stats_only:
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Hypergraph v7 awrev5: Top-M geom + RRF chunk rerank (pool metrics)."
    )
    parser.add_argument(
        "--no-geometry-tiebreak",
        action="store_true",
        help="关闭 07 几何并列重排（用于与基线 rank/bbox_id 对比）。",
    )
    parser.add_argument("--task", choices=("ND", "NO", "ALL"), default="ALL")
    parser.add_argument(
        "--override-candidates",
        type=str,
        default="",
        help="Override the current task stage1 candidates jsonl. Use with --task ND or NO.",
    )
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_M_GEOM)
    parser.add_argument(
        "--rrf-chunk",
        type=int,
        default=DEFAULT_RRF_CHUNK,
        help="对前 K 个候选做 v5 RRF+一致性重排（默认 80）。",
    )
    parser.add_argument(
        "--rrf-head",
        type=int,
        default=DEFAULT_RRF_HEAD_REORDER,
        help="仅对 chunk 内前 H 名做 v5 重排，其余不动（默认 48）。",
    )
    parser.add_argument(
        "--rrf-k",
        type=float,
        default=DEFAULT_RRF_K,
        help="RRF 平滑常数（经典默认 60）。",
    )
    parser.add_argument(
        "--no-rrf-v5",
        action="store_true",
        help="不做 v5 chunk 重排（退化为 Top-M 几何后的顺序）。",
    )
    parser.add_argument(
        "--override-desc",
        type=str,
        default="",
        help="覆盖当前任务的描述超图 jsonl（须配合 --task ND 或 NO 单独跑）。",
    )
    parser.add_argument(
        "--override-bbox",
        type=str,
        default="",
        help="覆盖当前任务的 bbox 超图 jsonl。",
    )
    parser.add_argument(
        "--override-output",
        type=str,
        default="",
        help="覆盖写出路径（建议新文件名，避免覆盖旧评测 JSON）。",
    )
    parser.add_argument(
        "--override-geometry",
        type=str,
        default="",
        help="Override geometry_mentions jsonl. Use with --task ND or NO.",
    )
    parser.add_argument(
        "--override-bbox-dir-geometry",
        type=str,
        default="",
        help="Override bbox directory used by the geometry tie-breaker. Use with --task ND or NO.",
    )
    args = parser.parse_args()
    use_geom = not args.no_geometry_tiebreak

    tasks = (
        DEFAULT_TASKS
        if args.task == "ALL"
        else [t for t in DEFAULT_TASKS if t["name"] == args.task]
    )
    has_ov = bool(
        args.override_candidates
        or args.override_desc
        or args.override_bbox
        or args.override_output
        or args.override_geometry
        or args.override_bbox_dir_geometry
    )
    if has_ov and args.task == "ALL":
        raise SystemExit(
            "Override arguments only support --task ND or --task NO, not --task ALL."
        )
    if has_ov and len(tasks) != 1:
        raise SystemExit("内部错误：单任务模式下应恰好 1 个 task。")

    for task in tasks:
        task = dict(task)
        if args.override_candidates:
            task["candidates"] = args.override_candidates
        if args.override_desc:
            task["desc_hypergraphs"] = args.override_desc
        if args.override_bbox:
            task["bbox_hypergraphs"] = args.override_bbox
        if args.override_output:
            task["output"] = args.override_output
        if args.override_geometry:
            task["geometry_mentions"] = args.override_geometry
        if args.override_bbox_dir_geometry:
            task["bbox_dir_geometry"] = args.override_bbox_dir_geometry

        candidates_jsonl = resolve_existing_path(task["candidates"])
        desc_hypergraphs_jsonl = resolve_existing_path(task["desc_hypergraphs"])
        bbox_hypergraphs_jsonl = resolve_existing_path(task["bbox_hypergraphs"])
        geom_path = task.get("geometry_mentions")
        bbox_geom = task.get("bbox_dir_geometry")
        print(f"\n[{task['name']}] candidates: {candidates_jsonl}")
        print(f"[{task['name']}] desc_hypergraphs: {desc_hypergraphs_jsonl}")
        print(f"[{task['name']}] bbox_hypergraphs: {bbox_hypergraphs_jsonl}")
        if use_geom and geom_path:
            print(f"[{task['name']}] geometry_mentions: {geom_path}")
        out_json = task["output"]
        if args.no_geometry_tiebreak and out_json.endswith(".json"):
            out_json = out_json[:-5] + "_baseline_rank_only.json"
        evaluate_matching_pipeline2(
            name=task["name"],
            candidates_jsonl=candidates_jsonl,
            desc_hypergraphs_jsonl=desc_hypergraphs_jsonl,
            bbox_hypergraphs_jsonl=bbox_hypergraphs_jsonl,
            output_json=out_json,
            fusion_lambda=1.0,
            use_geometry_tiebreak=use_geom,
            geometry_mentions_jsonl=geom_path if use_geom else None,
            bbox_dir_geometry=bbox_geom if use_geom else None,
            top_m_geom=args.top_m,
            rrf_chunk_size=args.rrf_chunk,
            rrf_head_reorder=args.rrf_head,
            rrf_k=args.rrf_k,
            enable_rrf_v5=not args.no_rrf_v5,
        )


if __name__ == "__main__":
    main()

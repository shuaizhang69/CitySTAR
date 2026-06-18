"""
固定版超图匹配 v2。

用途：
1. 直接使用 pipeline2 的 stage1 candidates + desc hypergraphs + bbox hypergraphs
2. 不使用 GT 作为 tie-break
3. 直接输出 ND / NO 的 top1/top5/top10/top20 统计
"""

import json
import os
from collections import defaultdict
from itertools import permutations
from typing import Dict, Set, Tuple

from tqdm import tqdm

from match_hypergraphs import (
    get_desc_main_category,
    rank_prior_from_index,
)


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _bundle_or_legacy(rel_parts, legacy_path: str) -> str:
    p = os.path.join(_ROOT, "data", *rel_parts)
    return p if os.path.exists(p) else legacy_path


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


DEFAULT_TASKS = [
    {
        "name": "ND",
        "candidates": _bundle_or_legacy(
            (
                "Cityrefer",
                "meta_data",
                "0311data",
                "CityRefer_val_ND_0421_stage1_candidates.jsonl",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0421_stage1_candidates.jsonl",
        ),
        "desc_hypergraphs": _bundle_or_legacy(
            ("Cityrefer", "0412", "desc", "CityRefer_val_ND_desc_hypergraphs.jsonl"),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc/CityRefer_val_ND_desc_hypergraphs.jsonl",
        ),
        "bbox_hypergraphs": _bundle_or_legacy(
            (
                "Cityrefer",
                "0412",
                "desc_bbox_hypergraphs",
                "CityRefer_val_ND_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc_bbox_hypergraphs/CityRefer_val_ND_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        ),
        "output": _bundle_or_legacy(
            (
                "Cityrefer",
                "0412",
                "hypergraph_match",
                "CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2.json",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2.json",
        ),
    },
    {
        "name": "NO",
        "candidates": _bundle_or_legacy(
            (
                "Cityrefer",
                "meta_data",
                "0311data",
                "CityRefer_val_NO_0421_stage1_candidates.jsonl",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_NO_0421_stage1_candidates.jsonl",
        ),
        "desc_hypergraphs": _bundle_or_legacy(
            ("Cityrefer", "0412", "desc", "CityRefer_val_NO_desc_hypergraphs.jsonl"),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc/CityRefer_val_NO_desc_hypergraphs.jsonl",
        ),
        "bbox_hypergraphs": _bundle_or_legacy(
            (
                "Cityrefer",
                "0412",
                "desc_bbox_hypergraphs",
                "CityRefer_val_NO_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc_bbox_hypergraphs/CityRefer_val_NO_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        ),
        "output": _bundle_or_legacy(
            (
                "Cityrefer",
                "0412",
                "hypergraph_match",
                "CityRefer_val_NO_0421_stage1_candidates_hgmatch_v2.json",
            ),
            "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_NO_0421_stage1_candidates_hgmatch_v2.json",
        ),
    },
]


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


def load_bbox_hypergraphs_jsonl(bbox_jsonl: str) -> dict:
    bbox_graphs = {}
    with open(bbox_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            key = (
                str(data["scene_id"]),
                str(data["bbox_id"]),
                str(data.get("ann_id", 0)),
            )
            bbox_graphs[key] = data
    return bbox_graphs


def load_stage1_candidates_jsonl(candidates_jsonl: str) -> list:
    items = []
    with open(candidates_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_node_category_map(desc_graph: dict) -> dict:
    node_categories = {}
    main_cat = desc_graph.get("hypergraph", {}).get("main_category", "")
    for node in desc_graph.get("hypergraph", {}).get("nodes", []):
        node_id = node["id"]
        if node_id == "main":
            node_categories[node_id] = main_cat
        else:
            node_categories[node_id] = node.get("category", "")
    return node_categories


def get_instances_by_category(bbox_nodes: list) -> dict:
    by_cat = defaultdict(list)
    for node in bbox_nodes:
        by_cat[node["category"]].append(node["id"])
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
    best_score = 0.0
    for bbox_edge in bbox_edges:
        if bbox_edge["from"] == bbox_from and bbox_edge["to"] == bbox_to:
            if bbox_edge["relation"] in equivalent_rels:
                edge_score = bbox_edge.get("score", 1.0)
                if bbox_edge["relation"] == rel_type:
                    edge_score = min(1.0, edge_score * 1.2)
                best_score = max(best_score, edge_score)
    return best_score


def evaluate_mapping(desc_edges: list, bbox_edges: list, node_mapping: dict) -> float:
    total_score = 0.0
    matched_count = 0
    for desc_edge in desc_edges:
        score = check_edge_match(desc_edge, bbox_edges, node_mapping)
        if score > 0:
            total_score += score
            matched_count += 1
    if len(desc_edges) > 0:
        match_ratio = matched_count / len(desc_edges)
        avg_score = total_score / len(desc_edges)
        return match_ratio * 0.6 + avg_score * 0.4
    return 0.0


def find_best_mapping(desc_graph: dict, bbox_graph: dict) -> tuple:
    desc_edges = desc_graph.get("hypergraph", {}).get("edges", [])
    bbox_edges = bbox_graph.get("hypergraph", {}).get("edges", [])
    bbox_nodes = bbox_graph.get("hypergraph", {}).get("nodes", [])
    if not desc_edges or not bbox_edges:
        return 0.0, {}

    desc_node_cats = build_node_category_map(desc_graph)
    main_cat = desc_graph.get("hypergraph", {}).get("main_category", "")
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

    if main_cat not in bbox_by_cat or not bbox_by_cat[main_cat]:
        return 0.0, {}
    for cat, desc_nodes in desc_nodes_by_cat.items():
        if cat not in bbox_by_cat or len(bbox_by_cat[cat]) < len(desc_nodes):
            return 0.0, {}

    main_instances = bbox_by_cat[main_cat]
    best_score = 0.0
    best_mapping = {}

    for main_instance in main_instances:
        base_mapping = {"main": main_instance}
        category_choices = []
        for cat, desc_nodes in desc_nodes_by_cat.items():
            bbox_insts = bbox_by_cat.get(cat, [])
            category_choices.append((cat, desc_nodes, bbox_insts))

        def generate_mappings(cat_idx, current_mapping):
            nonlocal best_score, best_mapping
            if cat_idx == len(category_choices):
                score = evaluate_mapping(desc_edges, bbox_edges, current_mapping)
                if score > best_score:
                    best_score = score
                    best_mapping = current_mapping.copy()
                return

            _, desc_nodes, bbox_insts = category_choices[cat_idx]
            if len(bbox_insts) > 5:
                bbox_insts = bbox_insts[:5]
            if len(desc_nodes) > len(bbox_insts):
                return

            for perm in permutations(bbox_insts, len(desc_nodes)):
                mapping = current_mapping.copy()
                for i, desc_node in enumerate(desc_nodes):
                    mapping[desc_node] = perm[i]
                generate_mappings(cat_idx + 1, mapping)

        generate_mappings(0, base_mapping)

    return best_score, best_mapping


def compute_match_score(desc_graph: dict, bbox_graph: dict) -> float:
    score, _ = find_best_mapping(desc_graph, bbox_graph)
    return score


def evaluate_matching_pipeline2(
    name: str,
    candidates_jsonl: str,
    desc_hypergraphs_jsonl: str,
    bbox_hypergraphs_jsonl: str,
    output_json: str,
    score_eps: float = 1e-9,
    fusion_lambda: float = 1.0,
):
    print(f"\n=== Running task: {name} ===")
    print("Loading pipeline2 data...")
    desc_graphs = load_desc_hypergraphs_with_ann(desc_hypergraphs_jsonl)
    bbox_graphs = load_bbox_hypergraphs_jsonl(bbox_hypergraphs_jsonl)
    eval_data = load_stage1_candidates_jsonl(candidates_jsonl)

    print(f"Loaded {len(desc_graphs)} description graphs")
    print(f"Loaded {len(bbox_graphs)} bbox hypergraphs")
    print(f"Loaded {len(eval_data)} candidate queries")
    print(f"fusion_lambda={fusion_lambda}")

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
    }
    results = []

    for query in tqdm(eval_data, desc=f"Matching {name}"):
        scene_id = str(query["scene_id"])
        object_id = str(query["object_id"])
        ann_id = str(query.get("ann_id", 0))
        candidates = [str(x) for x in query.get("candidates", [])]
        if not candidates:
            continue
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

        for bbox_id in ordered_unique:
            bbox_graph = bbox_graphs.get((scene_id, bbox_id, ann_id))
            if bbox_graph is None:
                continue
            cached_graphs[bbox_id] = bbox_graph
            all_scores[bbox_id] = compute_match_score(desc_graph, bbox_graph)

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

        # 不使用 GT 做 tie-break；当前数据里没有可用的 num_points，
        # 因此只使用 rank prior 与稳定的 bbox_id。
        def pick_key(bid: str):
            r = rank_by_id.get(bid, 10**9)
            return (r, bid)

        best_bbox_id = min(fused_tied, key=pick_key)
        best_fused_score = all_fused_scores[best_bbox_id]
        sorted_fused = sorted(
            all_fused_scores.items(),
            key=lambda x: (-x[1], pick_key(x[0])),
        )

        top5_ids = [x[0] for x in sorted_fused[:5]]
        top10_ids = [x[0] for x in sorted_fused[:10]]
        top20_ids = [x[0] for x in sorted_fused[:20]]
        is_correct = (best_bbox_id == object_id)

        if is_correct:
            stats["correct_top1"] += 1
        if tie_break_used and is_correct:
            stats["correct_top1_when_tie_break"] += 1
        if object_id in top5_ids:
            stats["correct_top5"] += 1
        if object_id in top10_ids:
            stats["correct_top10"] += 1
        if object_id in top20_ids:
            stats["correct_top20"] += 1

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
            "top5_ids": top5_ids,
            "top10_ids": top10_ids,
            "top20_ids": top20_ids,
            "all_scores": all_scores,
            "all_fused_scores": all_fused_scores,
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"{name} 结果统计:")
    print(f"  总查询数: {stats['total']}")
    print(f"  匹配成功: {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    print(f"  Top-1 正确: {stats['correct_top1']} ({stats['correct_top1']/stats['total']*100:.1f}%)")
    print(f"  Top-5 包含: {stats['correct_top5']} ({stats['correct_top5']/stats['total']*100:.1f}%)")
    print(f"  Top-10 包含: {stats['correct_top10']} ({stats['correct_top10']/stats['total']*100:.1f}%)")
    print(f"  Top-20 包含: {stats['correct_top20']} ({stats['correct_top20']/stats['total']*100:.1f}%)")
    print(f"  纯超图分并列: {stats['queries_with_score_tie']}")
    print(f"  融合分仍并列: {stats['queries_with_fused_tie']}")
    tb = stats["tie_break_used"]
    print(f"  融合分并列后用 rank/bbox_id 打破: {tb} ({tb/stats['total']*100:.1f}%)")
    if tb > 0:
        print(f"  其中 Top-1 正确: {stats['correct_top1_when_tie_break']} ({stats['correct_top1_when_tie_break']/tb*100:.1f}%)")
    print(f"\n结果保存至: {output_json}")


def main():
    for task in DEFAULT_TASKS:
        evaluate_matching_pipeline2(
            name=task["name"],
            candidates_jsonl=task["candidates"],
            desc_hypergraphs_jsonl=task["desc_hypergraphs"],
            bbox_hypergraphs_jsonl=task["bbox_hypergraphs"],
            output_json=task["output"],
            fusion_lambda=1.0,
        )


if __name__ == "__main__":
    main()

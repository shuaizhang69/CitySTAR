"""
离线超图匹配：
1. 读取描述超图 jsonl
2. 读取 bbox-only 超图 jsonl
3. 读取 stage1 候选 jsonl
4. 对每个 query 的候选 bbox 做超图匹配打分并重排
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm


DEFAULT_TASKS = [
    {
        "name": "ND",
        "candidates": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0421_stage1_candidates.jsonl",
        "desc_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc/CityRefer_val_ND_desc_hypergraphs.jsonl",
        "bbox_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc_bbox_hypergraphs/CityRefer_val_ND_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0324_stage1_candidates_hgmatch.jsonl",
    },
    {
        "name": "NO",
        "candidates": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_NO_0421_stage1_candidates.jsonl",
        "desc_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc/CityRefer_val_NO_desc_hypergraphs.jsonl",
        "bbox_hypergraphs": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/desc_bbox_hypergraphs/CityRefer_val_NO_desc_hypergraphs_bbox_only_hypergraphs.jsonl",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_NO_0324_stage1_candidates_hgmatch.jsonl",
    },
]


class MatchResult:
    def __init__(
        self,
        scene_id: str,
        object_id: str,
        candidate_bbox_id: str,
        match_score: float,
        matched_edge_count: int,
        desc_edge_count: int,
    ) -> None:
        self.scene_id = scene_id
        self.object_id = object_id
        self.candidate_bbox_id = candidate_bbox_id
        self.match_score = match_score
        self.matched_edge_count = matched_edge_count
        self.desc_edge_count = desc_edge_count


def load_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_desc_hypergraphs(path: str) -> Dict[Tuple[str, str, str], dict]:
    result: Dict[Tuple[str, str, str], dict] = {}
    for item in load_jsonl(path):
        key = (
            str(item.get("scene_id", "")),
            str(item.get("object_id", "")),
            str(item.get("ann_id", "")),
        )
        result[key] = item
    return result


def load_bbox_hypergraphs(path: str) -> Dict[Tuple[str, str, str], dict]:
    result: Dict[Tuple[str, str, str], dict] = {}
    for item in load_jsonl(path):
        key = (
            str(item.get("scene_id", "")),
            str(item.get("bbox_id", "")),
            str(item.get("ann_id", "")),
        )
        result[key] = item
    return result


def edge_signature(edge: dict) -> Tuple[str, str, str]:
    target_category = str(edge.get("info", {}).get("target_category", ""))
    relation = str(edge.get("relation", ""))
    from_node = str(edge.get("from", ""))
    if from_node == "main":
        source_category = "main"
    else:
        source_category = str(edge.get("source_category", ""))
    return source_category, relation, target_category


def bbox_edge_signature(edge: dict, nodes_by_id: Dict[str, dict], target_node_id: str) -> Tuple[str, str, str]:
    from_node = nodes_by_id.get(edge.get("from", ""))
    to_node = nodes_by_id.get(edge.get("to", ""))
    if not from_node or not to_node:
        return "", "", ""

    source_category = "main" if from_node["id"] == target_node_id else str(from_node.get("category", ""))
    target_category = str(to_node.get("category", ""))
    relation = str(edge.get("relation", ""))
    return source_category, relation, target_category


def compute_graph_match_score(desc_graph_item: dict, bbox_graph_item: dict) -> MatchResult:
    desc_edges = desc_graph_item.get("hypergraph", {}).get("edges", [])
    bbox_edges = bbox_graph_item.get("hypergraph", {}).get("edges", [])
    bbox_nodes = bbox_graph_item.get("hypergraph", {}).get("nodes", [])

    target_node = next((node for node in bbox_nodes if node.get("is_target")), None)
    target_node_id = target_node["id"] if target_node else ""
    nodes_by_id = {node["id"]: node for node in bbox_nodes}

    desc_signatures = [edge_signature(edge) for edge in desc_edges]
    bbox_signature_scores: Dict[Tuple[str, str, str], float] = defaultdict(float)
    for edge in bbox_edges:
        signature = bbox_edge_signature(edge, nodes_by_id, target_node_id)
        if signature == ("", "", ""):
            continue
        bbox_signature_scores[signature] = max(
            bbox_signature_scores[signature],
            float(edge.get("score", 0.0)),
        )

    if not desc_signatures:
        score = 0.0
        matched = 0
    else:
        total_score = 0.0
        matched = 0
        for signature in desc_signatures:
            edge_score = bbox_signature_scores.get(signature, 0.0)
            if edge_score > 0:
                matched += 1
                total_score += edge_score
        coverage = matched / len(desc_signatures)
        avg_edge_score = total_score / len(desc_signatures)
        score = 0.7 * coverage + 0.3 * avg_edge_score

    return MatchResult(
        scene_id=str(desc_graph_item.get("scene_id", "")),
        object_id=str(desc_graph_item.get("object_id", "")),
        candidate_bbox_id=str(bbox_graph_item.get("bbox_id", "")),
        match_score=round(score, 6),
        matched_edge_count=matched,
        desc_edge_count=len(desc_signatures),
    )


def rerank_candidates(candidate_item: dict, desc_graphs: Dict[Tuple[str, str, str], dict], bbox_graphs: Dict[Tuple[str, str, str], dict]) -> dict:
    scene_id = str(candidate_item.get("scene_id", ""))
    object_id = str(candidate_item.get("object_id", ""))
    ann_id = str(candidate_item.get("ann_id", ""))
    key = (scene_id, object_id, ann_id)
    desc_graph_item = desc_graphs.get(key)

    result = dict(candidate_item)
    result["hgmatch_scores"] = []
    result["hgmatch_reranked_candidates"] = []
    result["hgmatch_best_candidate"] = None

    if not desc_graph_item:
        result["hgmatch_status"] = "missing_desc_graph"
        return result

    candidate_ids = [str(x) for x in candidate_item.get("candidates", [])]
    match_results: List[MatchResult] = []

    for candidate_bbox_id in candidate_ids:
        bbox_graph_item = bbox_graphs.get((scene_id, candidate_bbox_id, ann_id))
        if not bbox_graph_item:
            match_results.append(
                MatchResult(
                    scene_id=scene_id,
                    object_id=object_id,
                    candidate_bbox_id=candidate_bbox_id,
                    match_score=0.0,
                    matched_edge_count=0,
                    desc_edge_count=len(desc_graph_item.get("hypergraph", {}).get("edges", [])),
                )
            )
            continue
        match_results.append(compute_graph_match_score(desc_graph_item, bbox_graph_item))

    match_results.sort(
        key=lambda x: (
            -x.match_score,
            -x.matched_edge_count,
            0 if x.candidate_bbox_id == object_id else 1,
            int(x.candidate_bbox_id),
        )
    )

    result["hgmatch_scores"] = [
        {
            "candidate_bbox_id": item.candidate_bbox_id,
            "match_score": item.match_score,
            "matched_edge_count": item.matched_edge_count,
            "desc_edge_count": item.desc_edge_count,
        }
        for item in match_results
    ]
    result["hgmatch_reranked_candidates"] = [item.candidate_bbox_id for item in match_results]
    result["hgmatch_best_candidate"] = match_results[0].candidate_bbox_id if match_results else None
    result["hgmatch_status"] = "ok"
    return result


def process_task(candidates_path: str, desc_hypergraphs_path: str, bbox_hypergraphs_path: str, output_path: str) -> None:
    print(f"Loading desc hypergraphs: {desc_hypergraphs_path}")
    desc_graphs = load_desc_hypergraphs(desc_hypergraphs_path)
    print(f"  Loaded {len(desc_graphs)} desc hypergraphs")

    print(f"Loading bbox hypergraphs: {bbox_hypergraphs_path}")
    bbox_graphs = load_bbox_hypergraphs(bbox_hypergraphs_path)
    print(f"  Loaded {len(bbox_graphs)} bbox hypergraphs")

    print(f"Loading candidates: {candidates_path}")
    candidate_items = load_jsonl(candidates_path)
    print(f"  Loaded {len(candidate_items)} candidate queries")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed = []
    for item in tqdm(candidate_items, desc=os.path.basename(candidates_path)):
        processed.append(rerank_candidates(item, desc_graphs, bbox_graphs))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    correct_top1 = sum(
        1 for item in processed
        if str(item.get("object_id")) in [str(x) for x in item.get("hgmatch_reranked_candidates", [])[:1]]
    )
    correct_top5 = sum(
        1 for item in processed
        if str(item.get("object_id")) in [str(x) for x in item.get("hgmatch_reranked_candidates", [])[:5]]
    )
    correct_top10 = sum(
        1 for item in processed
        if str(item.get("object_id")) in [str(x) for x in item.get("hgmatch_reranked_candidates", [])[:10]]
    )
    has_desc = sum(1 for item in processed if item.get("hgmatch_status") == "ok")
    print(f"Saved to: {output_path}")
    print(f"  Queries: {len(processed)}")
    print(f"  With desc graph: {has_desc}")
    print(f"  Top1 hit: {correct_top1}/{len(processed)} ({(correct_top1 / len(processed) * 100) if processed else 0:.1f}%)")
    print(f"  Top5 hit: {correct_top5}/{len(processed)} ({(correct_top5 / len(processed) * 100) if processed else 0:.1f}%)")
    print(f"  Top10 hit: {correct_top10}/{len(processed)} ({(correct_top10 / len(processed) * 100) if processed else 0:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="")
    parser.add_argument("--desc_hypergraphs", type=str, default="")
    parser.add_argument("--bbox_hypergraphs", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--run_defaults", action="store_true", help="运行内置的 ND/NO 两组任务")
    args = parser.parse_args()

    if args.run_defaults or not any([args.candidates, args.desc_hypergraphs, args.bbox_hypergraphs, args.output]):
        for task in DEFAULT_TASKS:
            print(f"\n=== Running task: {task['name']} ===")
            process_task(
                candidates_path=task["candidates"],
                desc_hypergraphs_path=task["desc_hypergraphs"],
                bbox_hypergraphs_path=task["bbox_hypergraphs"],
                output_path=task["output"],
            )
        return

    required = [args.candidates, args.desc_hypergraphs, args.bbox_hypergraphs, args.output]
    if not all(required):
        raise ValueError("When not using --run_defaults, you must provide --candidates --desc_hypergraphs --bbox_hypergraphs --output")

    process_task(
        candidates_path=args.candidates,
        desc_hypergraphs_path=args.desc_hypergraphs,
        bbox_hypergraphs_path=args.bbox_hypergraphs,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

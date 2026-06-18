"""
基于 bbox 构建 candidate-specific hypergraphs，兼容：
1. 2_generate_desc_hypergraphs.py 的输出 desc jsonl
2. Semantic_landmark_in_stage1.py 的 stage1 candidates jsonl
3. 04_hypergraph_matchv2.py 的 bbox_hypergraphs 输入格式

输出仍沿用 *_desc_hypergraphs_bbox_only_hypergraphs.jsonl 的文件名，方便直接接 04。
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


# 必须与 stage1 候选 jsonl **逐行一一对应**（同序、同条数）。
# ND：`desc/` 下若只有部分行（例如中断跑只生成 20 条）会触发 mismatch，完整版在 desc_new。
DEFAULT_DESC_INPUTS = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_new/cityanchor_val_ND_desc_hypergraphs.jsonl",
    # NO：与 cityanchor_val_NO_new_*.jsonl / stage1 对齐用 desc 下 NO_new 描述超图（与 desc_new 逐行超图内容可能不同，勿与 desc_new 混配 v7）
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc/cityanchor_val_NO_new_desc_hypergraphs.jsonl",
]
DEFAULT_CANDIDATE_INPUTS = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/support_object_proximity_shrink/cityanchor_val_ND_0324_stage1_candidates_support_object_proximity_shrink.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/support_object_proximity_shrink/cityanchor_val_NO_new_0324_new_stage1_candidates_support_object_proximity_shrink.jsonl",
]
DEFAULT_BBOX_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"
# 与 v7 ND 默认一致，输出到 *_new，避免与旧版 desc_bbox_hypergraphs 混淆
DEFAULT_OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc_bbox_hypergraphs_new"


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


def bbox_object_name(bbox_item: dict) -> str:
    return str(bbox_item.get("object_name", "")).strip().lower()


def infer_required_relations(desc_item: dict) -> Set[str]:
    relations: Set[str] = set()
    for edge in desc_item.get("hypergraph", {}).get("edges", []):
        relation = (edge.get("relation") or "").strip()
        if relation:
            relations.add(relation)
    if not relations:
        relations = set(FALLBACK_RELATIONS)
    return relations


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


def bbox_item_to_bounds(
    bbox_item: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, bbox_min, bbox_max = bbox_to_bounds(bbox_item["bbox"], 1.0, 1.0)
    return center, bbox_min, bbox_max


def find_bboxes_in_bbox(
    scene_bboxes: List[dict],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    include_mode: str = "center",
    overlap_ratio: float = 0.1,
) -> list:
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


def compute_bbox_relations_selective(
    bbox_list: list,
    required_relations: Set[str],
    target_bbox_center: Optional[np.ndarray] = None,
    max_bboxes: int = 100,
    score_threshold: float = 0.5,
) -> List[dict]:
    edges: List[dict] = []
    if len(bbox_list) < 2 or not required_relations:
        return edges

    if len(bbox_list) > max_bboxes and target_bbox_center is not None:
        bbox_list = sorted(
            bbox_list,
            key=lambda x: np.linalg.norm(bbox_item_to_bounds(x)[0] - target_bbox_center),
        )[:max_bboxes]
    elif len(bbox_list) > max_bboxes:
        bbox_list = sorted(
            bbox_list,
            key=lambda x: float(np.prod(np.array(x["bbox"][3:6]))),
            reverse=True,
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

    pairwise_relations = required_relations & SUPPORTED_PAIRWISE_RELATIONS
    for i in range(len(spatial_objs)):
        for j in range(i + 1, len(spatial_objs)):
            bbox_a, obj_a = spatial_objs[i]
            bbox_b, obj_b = spatial_objs[j]
            cat_a = bbox_object_name(bbox_a)
            cat_b = bbox_object_name(bbox_b)
            id_a = int(bbox_a["object_id"])
            id_b = int(bbox_b["object_id"])
            for relation in pairwise_relations:
                is_valid, score = compute_relation(relation, obj_a, obj_b)
                if is_valid and score >= score_threshold:
                    edges.append(
                        {
                            "from": f"{cat_a}_{id_a}",
                            "to": f"{cat_b}_{id_b}",
                            "relation": relation,
                            "score": round(score, 3),
                        }
                    )

                is_valid_rev, score_rev = compute_relation(relation, obj_b, obj_a)
                if is_valid_rev and score_rev >= score_threshold:
                    edges.append(
                        {
                            "from": f"{cat_b}_{id_b}",
                            "to": f"{cat_a}_{id_a}",
                            "relation": relation,
                            "score": round(score_rev, 3),
                        }
                    )

    if "between" in required_relations:
        for i, (bbox_a, obj_a) in enumerate(spatial_objs):
            for j, (bbox_b, obj_b) in enumerate(spatial_objs):
                if i == j:
                    continue
                for k, (bbox_c, obj_c) in enumerate(spatial_objs):
                    if i == k or j == k:
                        continue
                    is_valid, score = compute_relation("between", obj_a, obj_b, obj_c)
                    if is_valid and score >= score_threshold:
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
                                "score": round(score, 3),
                                "anchors": [f"{cat_c}_{id_c}"],
                            }
                        )
                        edges.append(
                            {
                                "from": f"{cat_c}_{id_c}",
                                "to": f"{cat_a}_{id_a}",
                                "relation": "between",
                                "score": round(score, 3),
                                "anchors": [f"{cat_b}_{id_b}"],
                            }
                        )

    if "closest_to" in required_relations:
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
                        "from": (
                            f"{bbox_object_name(bbox_a)}_{int(bbox_a['object_id'])}"
                        ),
                        "to": (
                            f"{bbox_object_name(closest_bbox)}_{int(closest_bbox['object_id'])}"
                        ),
                        "relation": "closest_to",
                        "score": 1.0,
                    }
                )

    return edges


def build_candidate_hypergraph(
    desc_item: dict,
    candidate_bbox: dict,
    scene_bboxes: List[dict],
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "center",
    instance_overlap_ratio: float = 0.1,
    instance_edge_score_threshold: float = 0.5,
) -> dict:
    candidate_id = int(candidate_bbox["object_id"])
    bbox_center, bbox_min, bbox_max = bbox_to_bounds(
        candidate_bbox["bbox"], expand_factor, expand_z_factor
    )
    bbox_list = find_bboxes_in_bbox(
        scene_bboxes,
        bbox_min,
        bbox_max,
        include_mode=instance_inclusion_mode,
        overlap_ratio=instance_overlap_ratio,
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
        nodes.append(node)
        node_by_category[node_category].append(node_id)

    required_relations = infer_required_relations(desc_item)
    edges = compute_bbox_relations_selective(
        bbox_list,
        required_relations,
        target_bbox_center=bbox_center,
        score_threshold=instance_edge_score_threshold,
    )

    edges_by_relation: Dict[str, List[dict]] = defaultdict(list)
    for edge in edges:
        edges_by_relation[edge["relation"]].append(edge)

    return {
        "scene_id": desc_item["scene_id"],
        "bbox_id": candidate_id,
        "query_object_id": desc_item.get("object_id"),
        "ann_id": desc_item.get("ann_id"),
        "object_name": candidate_bbox.get("object_name", "").lower(),
        "description": desc_item.get("description", ""),
        "expand_factor": expand_factor,
        "expand_z_factor": expand_z_factor if expand_z_factor is not None else expand_factor,
        "bbox_inclusion_mode": instance_inclusion_mode,
        "bbox_overlap_ratio": instance_overlap_ratio,
        "bbox_edge_score_threshold": instance_edge_score_threshold,
        "bbox_center": bbox_center.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "description_hypergraph": desc_item.get("hypergraph", {}),
        "hypergraph": {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "required_relations": sorted(required_relations),
            "node_by_category": dict(node_by_category),
            "edges_by_relation": dict(edges_by_relation),
        },
    }


def process_pair(
    desc_path: str,
    candidates_path: str,
    bbox_root: str,
    output_path: str,
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "center",
    instance_overlap_ratio: float = 0.1,
    instance_edge_score_threshold: float = 0.5,
) -> tuple[int, int]:
    desc_items = load_jsonl_items(desc_path)
    candidate_items = load_jsonl_items(candidates_path)
    print(f"Processing desc file: {desc_path}")
    print(f"  Loaded {len(desc_items)} description items")
    print(f"  Loaded {len(candidate_items)} candidate items")

    if len(desc_items) != len(candidate_items):
        raise ValueError(
            f"desc/candidates line count mismatch: {desc_path} ({len(desc_items)}) vs "
            f"{candidates_path} ({len(candidate_items)}). "
            "They must be aligned 1:1 in order. Regenerate desc hypergraphs from the full "
            "infer_result (2_generate_desc_hypergraphs.py), or point --desc-inputs to a complete copy "
            "(e.g. desc_new/cityanchor_val_ND_desc_hypergraphs.jsonl for ND)."
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
    parser.add_argument("--expand-factor", type=float, default=3.0)
    parser.add_argument("--expand-z-factor", type=float, default=None)
    parser.add_argument(
        "--instance-inclusion-mode",
        type=str,
        default="center",
        choices=["center", "overlap"],
    )
    parser.add_argument("--instance-overlap-ratio", type=float, default=0.1)
    parser.add_argument("--instance-edge-score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--glob-desc",
        type=str,
        default="",
        help="可选：用 glob 扩展 desc 输入文件，例如 '/path/*.jsonl'",
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

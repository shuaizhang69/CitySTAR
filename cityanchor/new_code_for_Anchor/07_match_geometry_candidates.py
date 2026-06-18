#!/usr/bin/env python3
"""
Rank same-category bbox candidates with geometry mentions and report top-K accuracy.

Input:
1. geometry jsonl from 06_extract_geometry_mentions.py
2. bbox directory, one scene per JSON file

Behavior:
1. Candidate set is all bbox objects with the same category as object_name.
2. Score each candidate against every same-category geometry mention.
3. Keep the best mention score per candidate and rank accordingly.
4. Report top1/top3/top5 hit rate and export per-query candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_CONFIGS = {
    "ND": {
        "geometry_input": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/geometry/cityanchor_val_ND_geometry_mentions.jsonl",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/geometry_match/cityanchor_val_ND_geometry_match.json",
    },
    "NO": {
        "geometry_input": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/geometry/cityanchor_val_NO_geometry_mentions.jsonl",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/geometry_match/cityanchor_val_NO_geometry_match.json",
    },
}


DEFAULT_BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"


CATEGORY_EQUIVALENTS = {
    "building": "Building",
    "vehicle": "Vehicle",
    "truck": "Truck",
    "bike": "Bike",
    "lightpole": "LightPole",
    "fence": "Fence",
    "highvegetation": "HighVegetation",
    "mediumvegetation": "MediumVegetation",
    "lowvegetation": "LowVegetation",
}


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_tasks(task_arg: str) -> Iterable[Tuple[str, Dict[str, str]]]:
    if task_arg == "BOTH":
        for name in ("ND", "NO"):
            yield name, TASK_CONFIGS[name]
        return
    yield task_arg, TASK_CONFIGS[task_arg]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def normalize_category(name: str) -> str:
    key = "".join(ch for ch in (name or "") if ch.isalnum()).lower()
    return CATEGORY_EQUIVALENTS.get(key, name)


def sort_key_candidate(record: Dict[str, Any]) -> Tuple[float, int]:
    try:
        obj_id = int(record["object_id"])
    except Exception:
        obj_id = 10**18
    return (-float(record["score"]), obj_id)


def safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-9:
        return 0.0
    return a / b


def closeness_score(actual: float, target: float, relative_tol: float = 0.35, absolute_tol: float = 2.0) -> float:
    tol = max(abs(target) * relative_tol, absolute_tol)
    diff = abs(actual - target)
    return max(0.0, 1.0 - diff / tol)


def percentile_position(values_sorted: List[float], value: float) -> float:
    if not values_sorted:
        return 0.5
    if len(values_sorted) == 1:
        return 0.5
    idx = 0
    while idx < len(values_sorted) and values_sorted[idx] < value:
        idx += 1
    return idx / (len(values_sorted) - 1)


def candidate_geometry_features(bbox_entry: Dict[str, Any], infer_half_extent: bool) -> Dict[str, float]:
    bbox = bbox_entry.get("bbox") or []
    if len(bbox) < 6:
        raise ValueError(f"Invalid bbox entry: {bbox_entry}")

    length = float(bbox[3])
    width = float(bbox[4])
    height = float(bbox[5])
    scale = 2.0 if infer_half_extent else 1.0

    full_length = length * scale
    full_width = width * scale
    full_height = height * scale
    major = max(full_length, full_width)
    minor = min(full_length, full_width)
    area = major * minor
    volume = major * minor * full_height

    return {
        "length_m": full_length,
        "width_m": full_width,
        "height_m": full_height,
        "major_m": major,
        "minor_m": minor,
        "footprint_area": area,
        "volume": volume,
        "aspect_ratio": safe_div(major, max(minor, 1e-6)),
    }


def collect_relevant_mentions(item: Dict[str, Any], target_category: str) -> List[Dict[str, Any]]:
    geometry_extraction = item.get("geometry_extraction") or {}
    mentions = geometry_extraction.get("mentions") or []
    target_norm = normalize_category(target_category)

    main_same_category = [
        mention
        for mention in mentions
        if str(mention.get("mention_role", "")).strip().lower() == "main"
        and normalize_category(str(mention.get("category_guess", ""))) == target_norm
    ]
    if main_same_category:
        return main_same_category

    main_unspecified = [
        mention
        for mention in mentions
        if str(mention.get("mention_role", "")).strip().lower() == "main"
        and not str(mention.get("category_guess", "")).strip()
    ]
    if main_unspecified:
        return main_unspecified

    main_any = [
        mention
        for mention in mentions
        if str(mention.get("mention_role", "")).strip().lower() == "main"
    ]
    if main_any:
        return main_any

    same_category = [
        mention
        for mention in mentions
        if normalize_category(str(mention.get("category_guess", ""))) == target_norm
    ]
    if same_category:
        return same_category

    unspecified = [
        mention
        for mention in mentions
        if not str(mention.get("category_guess", "")).strip()
    ]
    if unspecified:
        return unspecified

    return list(mentions)


def shape_score(shape_constraints: List[str], candidate: Dict[str, float], class_stats: Dict[str, List[float]]) -> Tuple[float, float]:
    if not shape_constraints:
        return 0.0, 0.0

    score_sum = 0.0
    weight_sum = 0.0
    aspect_ratio = candidate["aspect_ratio"]
    major_percentile = percentile_position(class_stats["major"], candidate["major_m"])
    height_percentile = percentile_position(class_stats["height"], candidate["height_m"])

    for shape in shape_constraints:
        score = None
        if shape == "rectangular":
            score = min(1.0, max(0.0, (aspect_ratio - 1.2) / 1.0))
        elif shape == "square":
            score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 0.35)
        elif shape == "slender":
            score = min(1.0, max(0.0, (aspect_ratio - 2.0) / 2.0))
        elif shape == "round":
            score = max(0.0, 1.0 - abs(aspect_ratio - 1.0) / 0.2)
        elif shape == "irregular":
            score = 0.35
        elif shape in {"u-shaped", "l-shaped", "f-shaped", "t-shaped"}:
            score = 0.25

        if score is None:
            continue
        if shape == "rectangular" and major_percentile >= 0.75:
            score = min(1.0, score + 0.05)
        if shape == "slender" and height_percentile >= 0.6:
            score = min(1.0, score + 0.05)
        score_sum += score
        weight_sum += 1.0

    return score_sum, weight_sum


def size_score(size_constraints: List[str], candidate: Dict[str, float], class_stats: Dict[str, List[float]]) -> Tuple[float, float]:
    if not size_constraints:
        return 0.0, 0.0

    height_percentile = percentile_position(class_stats["height"], candidate["height_m"])
    major_percentile = percentile_position(class_stats["major"], candidate["major_m"])
    area_percentile = percentile_position(class_stats["area"], candidate["footprint_area"])

    score_sum = 0.0
    weight_sum = 0.0
    for size_name in size_constraints:
        score = None
        if size_name in {"tall"}:
            score = height_percentile
        elif size_name in {"short", "low"}:
            score = 1.0 - height_percentile
        elif size_name == "long":
            score = major_percentile
        elif size_name == "large":
            score = 0.6 * area_percentile + 0.4 * height_percentile
        elif size_name == "small":
            score = 1.0 - (0.6 * area_percentile + 0.4 * height_percentile)

        if score is None:
            continue
        score_sum += float(score)
        weight_sum += 1.0

    return score_sum, weight_sum


def numeric_score(numeric_constraints: Dict[str, Any], candidate: Dict[str, float], story_height_m: float) -> Tuple[float, float]:
    score_sum = 0.0
    weight_sum = 0.0

    height_value = numeric_constraints.get("height_m")
    if height_value is not None:
        score_sum += closeness_score(candidate["height_m"], float(height_value), relative_tol=0.35, absolute_tol=2.0)
        weight_sum += 1.0

    length_value = numeric_constraints.get("length_m")
    if length_value is not None:
        score_sum += closeness_score(candidate["major_m"], float(length_value), relative_tol=0.35, absolute_tol=3.0)
        weight_sum += 1.0

    width_value = numeric_constraints.get("width_m")
    if width_value is not None:
        score_sum += closeness_score(candidate["minor_m"], float(width_value), relative_tol=0.35, absolute_tol=2.0)
        weight_sum += 1.0

    stories_value = numeric_constraints.get("stories")
    if stories_value is not None and float(stories_value) > 0:
        expected_height = float(stories_value) * story_height_m
        score_sum += closeness_score(candidate["height_m"], expected_height, relative_tol=0.4, absolute_tol=3.0)
        weight_sum += 0.8

    return score_sum, weight_sum


def mention_match_score(
    mention: Dict[str, Any],
    candidate: Dict[str, float],
    class_stats: Dict[str, List[float]],
    story_height_m: float,
) -> Tuple[float, Dict[str, float]]:
    numeric_sum, numeric_weight = numeric_score(mention.get("numeric_constraints") or {}, candidate, story_height_m)
    shape_sum, shape_weight = shape_score(mention.get("shape_constraints") or [], candidate, class_stats)
    size_sum, size_weight = size_score(mention.get("size_constraints") or [], candidate, class_stats)

    total_weight = numeric_weight + shape_weight + size_weight
    if total_weight <= 0:
        return 0.0, {
            "numeric": 0.0,
            "shape": 0.0,
            "size": 0.0,
            "coverage": 0.0,
        }

    numeric_avg = safe_div(numeric_sum, numeric_weight) if numeric_weight > 0 else 0.0
    shape_avg = safe_div(shape_sum, shape_weight) if shape_weight > 0 else 0.0
    size_avg = safe_div(size_sum, size_weight) if size_weight > 0 else 0.0
    coverage = min(1.0, total_weight / 3.0)

    final_score = safe_div(numeric_sum + shape_sum + size_sum, total_weight)
    final_score = 0.85 * final_score + 0.15 * coverage
    return final_score, {
        "numeric": numeric_avg,
        "shape": shape_avg,
        "size": size_avg,
        "coverage": coverage,
    }


def load_scene_bboxes(scene_id: str, bbox_dir: str, cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if scene_id in cache:
        return cache[scene_id]
    path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cache[scene_id] = data
    return data


def summarize_class_stats(candidates: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    height = sorted(x["features"]["height_m"] for x in candidates)
    major = sorted(x["features"]["major_m"] for x in candidates)
    area = sorted(x["features"]["footprint_area"] for x in candidates)
    return {"height": height, "major": major, "area": area}


def evaluate_item(
    item: Dict[str, Any],
    bbox_dir: str,
    bbox_cache: Dict[str, Dict[str, Any]],
    infer_half_extent: bool,
    topk: int,
    story_height_m: float,
) -> Dict[str, Any]:
    scene_id = str(item.get("scene_id", ""))
    gt_id = str(item.get("object_id", ""))
    object_name = normalize_category(str(item.get("object_name", "")))
    geometry_extraction = item.get("geometry_extraction") or {}
    relevant_mentions = collect_relevant_mentions(item, object_name)

    try:
        scene_bbox = load_scene_bboxes(scene_id, bbox_dir, bbox_cache)
    except FileNotFoundError:
        return {
            "scene_id": scene_id,
            "object_id": gt_id,
            "ann_id": item.get("ann_id"),
            "object_name": object_name,
            "status": "missing_bbox_file",
            "ranked_candidate_ids": [],
        }

    candidate_entries: List[Dict[str, Any]] = []
    for bbox_entry in scene_bbox.get("bboxes") or []:
        candidate_category = normalize_category(str(bbox_entry.get("object_name", "")))
        if candidate_category != object_name:
            continue
        candidate_entries.append(
            {
                "object_id": str(bbox_entry.get("object_id")),
                "features": candidate_geometry_features(bbox_entry, infer_half_extent=infer_half_extent),
            }
        )

    if not candidate_entries:
        return {
            "scene_id": scene_id,
            "object_id": gt_id,
            "ann_id": item.get("ann_id"),
            "object_name": object_name,
            "status": "no_same_category_candidates",
            "ranked_candidate_ids": [],
        }

    class_stats = summarize_class_stats(candidate_entries)
    ranked: List[Dict[str, Any]] = []

    for candidate in candidate_entries:
        best_score = 0.0
        best_mention_id = None
        best_breakdown = {"numeric": 0.0, "shape": 0.0, "size": 0.0, "coverage": 0.0}

        for mention in relevant_mentions:
            score, breakdown = mention_match_score(
                mention=mention,
                candidate=candidate["features"],
                class_stats=class_stats,
                story_height_m=story_height_m,
            )
            if score > best_score:
                best_score = score
                best_mention_id = mention.get("mention_id")
                best_breakdown = breakdown

        ranked.append(
            {
                "object_id": candidate["object_id"],
                "score": round(best_score, 6),
                "best_mention_id": best_mention_id,
                "breakdown": best_breakdown,
                "features": candidate["features"],
            }
        )

    ranked.sort(key=sort_key_candidate)
    ranked_candidate_ids = [x["object_id"] for x in ranked]
    gt_rank = ranked_candidate_ids.index(gt_id) + 1 if gt_id in ranked_candidate_ids else None

    return {
        "scene_id": scene_id,
        "object_id": gt_id,
        "ann_id": item.get("ann_id"),
        "object_name": object_name,
        "status": "ok",
        "has_geometry": bool(geometry_extraction.get("has_geometry")),
        "geometry_types": geometry_extraction.get("geometry_types") or [],
        "num_relevant_mentions": len(relevant_mentions),
        "num_candidates": len(ranked),
        "gt_rank": gt_rank,
        "top1_id": ranked_candidate_ids[0] if ranked_candidate_ids else None,
        "topk_ids": ranked_candidate_ids[:topk],
        "ranked_candidate_ids": ranked_candidate_ids,
        "ranked_candidates": ranked[:topk],
    }


def compute_summary(results: List[Dict[str, Any]], topk: int) -> Dict[str, Any]:
    ok_items = [x for x in results if x.get("status") == "ok"]
    geometry_items = [x for x in ok_items if x.get("has_geometry")]
    rankable_items = [x for x in ok_items if x.get("gt_rank") is not None]
    geometry_rankable_items = [x for x in rankable_items if x.get("has_geometry")]

    def topk_hit(items: List[Dict[str, Any]], k: int) -> int:
        hits = 0
        for item in items:
            gt_rank = item.get("gt_rank")
            if gt_rank is not None and int(gt_rank) <= k:
                hits += 1
        return hits

    summary = {
        "total_queries": len(results),
        "ok_queries": len(ok_items),
        "geometry_queries": len(geometry_items),
        "rankable_queries": len(rankable_items),
        "geometry_rankable_queries": len(geometry_rankable_items),
        "top1_hits": topk_hit(ok_items, 1),
        "top3_hits": topk_hit(ok_items, 3),
        "top5_hits": topk_hit(ok_items, 5),
        "topk_hits": topk_hit(ok_items, topk),
        "geometry_top1_hits": topk_hit(geometry_items, 1),
        "geometry_top3_hits": topk_hit(geometry_items, 3),
        "geometry_top5_hits": topk_hit(geometry_items, 5),
        "geometry_topk_hits": topk_hit(geometry_items, topk),
        "rankable_top1_hits": topk_hit(rankable_items, 1),
        "rankable_top3_hits": topk_hit(rankable_items, 3),
        "rankable_top5_hits": topk_hit(rankable_items, 5),
        "rankable_topk_hits": topk_hit(rankable_items, topk),
        "geometry_rankable_top1_hits": topk_hit(geometry_rankable_items, 1),
        "geometry_rankable_top3_hits": topk_hit(geometry_rankable_items, 3),
        "geometry_rankable_top5_hits": topk_hit(geometry_rankable_items, 5),
        "geometry_rankable_topk_hits": topk_hit(geometry_rankable_items, topk),
    }

    for denom_key, prefix in (("ok_queries", ""), ("geometry_queries", "geometry_")):
        denom = summary[denom_key]
        for k in ("top1", "top3", "top5", "topk"):
            hit_key = f"{prefix}{k}_hits"
            summary[f"{prefix}{k}_acc"] = round(safe_div(summary[hit_key], denom), 6) if denom > 0 else None

    for denom_key, prefix in (
        ("rankable_queries", "rankable_"),
        ("geometry_rankable_queries", "geometry_rankable_"),
    ):
        denom = summary[denom_key]
        for k in ("top1", "top3", "top5", "topk"):
            hit_key = f"{prefix}{k}_hits"
            summary[f"{prefix}{k}_acc"] = round(safe_div(summary[hit_key], denom), 6) if denom > 0 else None

    return summary


def process_file(
    geometry_input: str,
    output_path: str,
    bbox_dir: str,
    infer_half_extent: bool,
    topk: int,
    story_height_m: float,
    limit: int,
) -> Dict[str, Any]:
    items = load_jsonl(geometry_input)
    if limit >= 0:
        items = items[:limit]
    ensure_parent_dir(output_path)

    bbox_cache: Dict[str, Dict[str, Any]] = {}
    results = [
        evaluate_item(
            item=item,
            bbox_dir=bbox_dir,
            bbox_cache=bbox_cache,
            infer_half_extent=infer_half_extent,
            topk=topk,
            story_height_m=story_height_m,
        )
        for item in items
    ]
    summary = compute_summary(results, topk=topk)
    output = {
        "summary": summary,
        "settings": {
            "geometry_input": geometry_input,
            "bbox_dir": bbox_dir,
            "infer_half_extent": infer_half_extent,
            "topk": topk,
            "story_height_m": story_height_m,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Match bbox candidates using geometry mentions.")
    parser.add_argument("--task", choices=["ND", "NO", "BOTH"], default="ND")
    parser.add_argument("--geometry-input", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--bbox-dir", default=DEFAULT_BBOX_DIR)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--story-height-m", type=float, default=3.2)
    parser.add_argument("--raw-bbox", action="store_true", help="Use bbox[3:6] directly instead of doubling half-extents.")
    parser.add_argument("--limit", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: List[Dict[str, Any]] = []

    for task_name, task_cfg in iter_tasks(args.task):
        geometry_input = args.geometry_input or task_cfg["geometry_input"]
        output_path = args.output or task_cfg["output"]
        if args.task == "BOTH" and not args.geometry_input and not args.output:
            geometry_input = task_cfg["geometry_input"]
            output_path = task_cfg["output"]

        output = process_file(
            geometry_input=geometry_input,
            output_path=output_path,
            bbox_dir=args.bbox_dir,
            infer_half_extent=not args.raw_bbox,
            topk=args.topk,
            story_height_m=args.story_height_m,
            limit=args.limit,
        )
        summary = dict(output["summary"])
        summary["task"] = task_name
        summary["output_path"] = output_path
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(summaries) > 1:
        total_queries = sum(x["total_queries"] for x in summaries)
        ok_queries = sum(x["ok_queries"] for x in summaries)
        geometry_queries = sum(x["geometry_queries"] for x in summaries)
        top1_hits = sum(x["top1_hits"] for x in summaries)
        top3_hits = sum(x["top3_hits"] for x in summaries)
        top5_hits = sum(x["top5_hits"] for x in summaries)
        topk_hits = sum(x["topk_hits"] for x in summaries)
        geometry_top1_hits = sum(x["geometry_top1_hits"] for x in summaries)
        geometry_top3_hits = sum(x["geometry_top3_hits"] for x in summaries)
        geometry_top5_hits = sum(x["geometry_top5_hits"] for x in summaries)
        geometry_topk_hits = sum(x["geometry_topk_hits"] for x in summaries)
        print(
            json.dumps(
                {
                    "task": "BOTH",
                    "total_queries": total_queries,
                    "ok_queries": ok_queries,
                    "geometry_queries": geometry_queries,
                    "top1_acc": round(safe_div(top1_hits, ok_queries), 6) if ok_queries else None,
                    "top3_acc": round(safe_div(top3_hits, ok_queries), 6) if ok_queries else None,
                    "top5_acc": round(safe_div(top5_hits, ok_queries), 6) if ok_queries else None,
                    "topk_acc": round(safe_div(topk_hits, ok_queries), 6) if ok_queries else None,
                    "geometry_top1_acc": round(safe_div(geometry_top1_hits, geometry_queries), 6) if geometry_queries else None,
                    "geometry_top3_acc": round(safe_div(geometry_top3_hits, geometry_queries), 6) if geometry_queries else None,
                    "geometry_top5_acc": round(safe_div(geometry_top5_hits, geometry_queries), 6) if geometry_queries else None,
                    "geometry_topk_acc": round(safe_div(geometry_topk_hits, geometry_queries), 6) if geometry_queries else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

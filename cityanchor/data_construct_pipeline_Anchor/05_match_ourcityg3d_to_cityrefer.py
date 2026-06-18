"""
为 checked_templates_merged_annotated.json 中的每个条目，在同场景下匹配 Cityrefer bbox。

规则:
1. 在 bbox1 中找到 template 的 center_instance_id 对应物体。
2. 在同场景 bbox2 中计算所有物体中心点距离。
3. 默认优先选择同类别，最多保留 3 个。
4. 若最近的异类别物体比当前已选中最远的同类别更近，则允许加入 1 个异类别，
   总数仍不超过 3。
5. 最终候选按距离升序排序。
"""

import argparse
import glob
import json
import math
import os
from collections import Counter


TEMPLATE_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/Our_cityG3D/checked_templates_merged_annotated.json"
BBOX1_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Our_cityG3D/0413/bbox/sensaturban"
BBOX2_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d"
CITYREFER_IMAGE_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/single_image"
OUTPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/Our_cityG3D/checked_templates_merged_annotated_cityrefer_matches.json"


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def center_xyz(bbox_item: dict) -> list[float]:
    bbox = bbox_item.get("bbox") or []
    return [float(bbox[0]), float(bbox[1]), float(bbox[2])]


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.dist(a, b)


def find_image_path(scene_id: str, object_id: int) -> str | None:
    pattern = os.path.join(CITYREFER_IMAGE_DIR, scene_id, f"{scene_id}_obj{object_id}.*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def select_candidates(source_bbox: dict, target_bboxes: list[dict], max_count: int = 3) -> list[dict]:
    source_name = str(source_bbox.get("object_name") or "")
    source_center = center_xyz(source_bbox)

    same = []
    other = []
    for item in target_bboxes:
        object_id = to_int(item.get("object_id"))
        if object_id is None:
            continue
        cand = {
            "object_id": object_id,
            "object_name": item.get("object_name"),
            "landmark": item.get("landmark", ""),
            "bbox": item.get("bbox"),
            "distance": euclidean_distance(source_center, center_xyz(item)),
            "image_path": find_image_path(str(item.get("scene_id") or ""), object_id),
        }
        if cand["object_name"] == source_name:
            same.append(cand)
        else:
            other.append(cand)

    same.sort(key=lambda x: (x["distance"], x["object_id"]))
    other.sort(key=lambda x: (x["distance"], x["object_id"]))

    selected = same[:max_count]
    if other:
        best_other = other[0]
        if len(selected) < max_count:
            selected.append(best_other)
        elif selected and best_other["distance"] < selected[-1]["distance"]:
            selected = selected[:-1] + [best_other]

    selected.sort(key=lambda x: (x["distance"], x["object_id"]))
    if len(selected) > max_count:
        selected = selected[:max_count]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-json", default=TEMPLATE_JSON)
    parser.add_argument("--bbox1-dir", default=BBOX1_DIR)
    parser.add_argument("--bbox2-dir", default=BBOX2_DIR)
    parser.add_argument("--output-json", default=OUTPUT_JSON)
    args = parser.parse_args()

    templates = load_json(args.template_json)
    bbox1_cache = {}
    bbox2_cache = {}

    results = []
    stats = Counter()

    for item in templates:
        scene_id = str(item["scene_id"])
        center_id = to_int(item.get("center_instance_id"))
        bbox1_path = os.path.join(args.bbox1_dir, f"{scene_id}_bbox.json")
        bbox2_path = os.path.join(args.bbox2_dir, f"{scene_id}_bbox.json")

        if not os.path.isfile(bbox1_path):
            stats["missing_bbox1"] += 1
            continue
        if not os.path.isfile(bbox2_path):
            stats["missing_bbox2"] += 1
            continue

        if scene_id not in bbox1_cache:
            bbox1_cache[scene_id] = load_json(bbox1_path)
        if scene_id not in bbox2_cache:
            bbox2_cache[scene_id] = load_json(bbox2_path)

        bbox1_items = bbox1_cache[scene_id].get("bboxes") or []
        bbox2_items = bbox2_cache[scene_id].get("bboxes") or []

        source_bbox = None
        for bbox_item in bbox1_items:
            if to_int(bbox_item.get("object_id")) == center_id:
                source_bbox = bbox_item
                break

        if source_bbox is None:
            stats["missing_center_in_bbox1"] += 1
            continue

        for bbox_item in bbox2_items:
            bbox_item["scene_id"] = scene_id

        matches = select_candidates(source_bbox, bbox2_items, max_count=3)
        diff_count = sum(1 for x in matches if x.get("object_name") != source_bbox.get("object_name"))

        results.append(
            {
                "scene_id": scene_id,
                "center_instance_id": center_id,
                "center_object_name": source_bbox.get("object_name"),
                "bbox1_path": bbox1_path,
                "bbox2_path": bbox2_path,
                "source_image_path": item.get("image_path"),
                "source_bbox": source_bbox,
                "matched_count": len(matches),
                "different_category_count": diff_count,
                "matches": matches,
            }
        )
        stats["processed"] += 1

    output = {
        "metadata": {
            "template_json": args.template_json,
            "bbox1_dir": args.bbox1_dir,
            "bbox2_dir": args.bbox2_dir,
            "rule": "same-category first; allow at most one closer different-category candidate; keep at most 3 matches sorted by distance",
            "processed_count": stats["processed"],
            "missing_bbox1_count": stats["missing_bbox1"],
            "missing_bbox2_count": stats["missing_bbox2"],
            "missing_center_in_bbox1_count": stats["missing_center_in_bbox1"],
        },
        "matches": results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(
        "完成: "
        f"processed={stats['processed']}, "
        f"missing_bbox1={stats['missing_bbox1']}, "
        f"missing_bbox2={stats['missing_bbox2']}, "
        f"missing_center_in_bbox1={stats['missing_center_in_bbox1']} -> "
        f"{args.output_json}"
    )


if __name__ == "__main__":
    main()

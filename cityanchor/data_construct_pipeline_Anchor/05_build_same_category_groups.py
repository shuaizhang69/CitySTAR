"""
将 cityanchor_val_ND/NO 中提到的物体，按“同场景 + 同类别”汇总成一个 JSON。

输出按 scene_id + category 分组，避免同一场景中多个被提到的同类物体重复展开。
每个分组会记录：
  - referenced_by: 原始 ND/NO 标注中哪些条目引用了这个分组
  - objects: 该场景中该类别的全部物体，以及对应 crop_all 图片路径
"""

import argparse
import glob
import json
import os
from collections import defaultdict


_DATASET = "/hpc2hdd/home/yxiao224/Henry/dataset"
CITY_ANCHOR_ROOT = os.path.join(_DATASET, "city_Anchor")
DEFAULT_INPUTS = [
    os.path.join(CITY_ANCHOR_ROOT, "cityanchor_val_ND.json"),
    os.path.join(CITY_ANCHOR_ROOT, "cityanchor_val_NO.json"),
]
DEFAULT_BBOX_DIR = os.path.join(CITY_ANCHOR_ROOT, "bbox")
DEFAULT_CROP_ROOT = os.path.join(CITY_ANCHOR_ROOT, "crop_all")
DEFAULT_OUTPUT_JSON = os.path.join(
    CITY_ANCHOR_ROOT,
    "cityanchor_val_same_scene_same_category_objects.json",
)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def object_name_str(item: dict) -> str:
    return str(item.get("object_name") or item.get("label") or "")


def find_crop_image(crop_root: str, scene_id: str, object_id: int) -> str | None:
    pattern = os.path.join(crop_root, scene_id, f"{scene_id}_obj{object_id}.*")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def build_grouped_json(input_jsons: list[str], bbox_dir: str, crop_root: str) -> dict:
    raw_entries = []
    for input_path in input_jsons:
        for item in load_json(input_path):
            item_copy = dict(item)
            item_copy["source_file"] = os.path.basename(input_path)
            raw_entries.append(item_copy)

    bbox_cache: dict[str, dict] = {}
    groups: dict[tuple[str, str], dict] = {}
    stats = defaultdict(int)

    for item in raw_entries:
        scene_id = str(item["scene_id"])
        source_object_id = to_int(item.get("object_id"))
        if source_object_id is None:
            stats["invalid_source_object_id"] += 1
            continue

        bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
        if scene_id not in bbox_cache:
            if not os.path.isfile(bbox_path):
                bbox_cache[scene_id] = {"scene_id": scene_id, "bboxes": []}
            else:
                bbox_cache[scene_id] = load_json(bbox_path)

        scene_bbox = bbox_cache[scene_id]
        all_bboxes = scene_bbox.get("bboxes") or []
        source_bbox = None
        for bbox_item in all_bboxes:
            if to_int(bbox_item.get("object_id")) == source_object_id:
                source_bbox = bbox_item
                break

        if source_bbox is None:
            stats["missing_source_bbox"] += 1
            continue

        category = object_name_str(source_bbox)
        group_key = (scene_id, category)
        if group_key not in groups:
            group_objects = []
            for bbox_item in all_bboxes:
                object_id = to_int(bbox_item.get("object_id"))
                if object_id is None:
                    continue
                if object_name_str(bbox_item) != category:
                    continue
                image_path = find_crop_image(crop_root, scene_id, object_id)
                group_objects.append(
                    {
                        "object_id": object_id,
                        "object_name": category,
                        "landmark": bbox_item.get("landmark", ""),
                        "bbox": bbox_item.get("bbox"),
                        "image_path": image_path,
                        "has_image": image_path is not None,
                    }
                )

            group_objects.sort(key=lambda x: x["object_id"])
            groups[group_key] = {
                "scene_id": scene_id,
                "category": category,
                "bbox_json": bbox_path,
                "crop_dir": os.path.join(crop_root, scene_id),
                "referenced_by": [],
                "objects": group_objects,
            }
            stats["groups"] += 1
            stats["objects_in_groups"] += len(group_objects)

        groups[group_key]["referenced_by"].append(
            {
                "source_file": item["source_file"],
                "scene_id": scene_id,
                "object_id": source_object_id,
                "object_name": item.get("object_name"),
                "resolved_category": category,
                "ann_id": item.get("ann_id"),
                "description": item.get("description"),
            }
        )
        stats["references"] += 1

    output_groups = []
    for _, group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        referenced_ids = {
            to_int(ref["object_id"])
            for ref in group["referenced_by"]
            if to_int(ref["object_id"]) is not None
        }
        for obj in group["objects"]:
            obj["is_referenced_object"] = obj["object_id"] in referenced_ids

        group["reference_count"] = len(group["referenced_by"])
        group["object_count"] = len(group["objects"])
        group["image_count"] = sum(1 for obj in group["objects"] if obj["has_image"])
        output_groups.append(group)

    return {
        "metadata": {
            "input_jsons": input_jsons,
            "bbox_dir": bbox_dir,
            "crop_root": crop_root,
            "reference_count": stats["references"],
            "group_count": stats["groups"],
            "group_object_count": stats["objects_in_groups"],
            "missing_source_bbox_count": stats["missing_source_bbox"],
            "invalid_source_object_id_count": stats["invalid_source_object_id"],
        },
        "groups": output_groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--bbox-dir", default=DEFAULT_BBOX_DIR)
    parser.add_argument("--crop-root", default=DEFAULT_CROP_ROOT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    output = build_grouped_json(args.input_json, args.bbox_dir, args.crop_root)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    meta = output["metadata"]
    print(
        "完成: "
        f"references={meta['reference_count']}, "
        f"groups={meta['group_count']}, "
        f"group_objects={meta['group_object_count']} -> {args.output_json}"
    )


if __name__ == "__main__":
    main()

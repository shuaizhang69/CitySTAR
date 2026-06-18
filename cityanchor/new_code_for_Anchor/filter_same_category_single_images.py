import json
import os
import shutil
from typing import Dict, Iterable, List, Set, Tuple


DEFAULT_ND_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_ND_0324.jsonl"
DEFAULT_NO_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new.json"
DEFAULT_BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"
DEFAULT_SINGLE_IMAGE_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image"
DEFAULT_OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_new"
DEFAULT_MANIFEST_PATH = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_new_manifest.json"


def _normalize_category(name: str) -> str:
    if not name:
        return ""
    return str(name).strip().lower()


def _category_match(a: str, b: str) -> bool:
    norm_a = _normalize_category(a)
    norm_b = _normalize_category(b)
    if norm_a == norm_b:
        return True
    if {norm_a, norm_b} == {"vehicle", "truck"}:
        return True
    return False


def _load_nd_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


def _load_no_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    return data


def _load_requests(nd_path: str, no_path: str) -> List[dict]:
    return _load_nd_jsonl(nd_path) + _load_no_json(no_path)


def _load_scene_bboxes(bbox_dir: str, scene_id: str) -> List[dict]:
    bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
    if not os.path.exists(bbox_path):
        return []
    with open(bbox_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("bboxes") or []


def _candidate_image_paths(single_image_dir: str, scene_id: str, object_id: str) -> Iterable[str]:
    scene_dir = os.path.join(single_image_dir, scene_id)
    stem = f"{scene_id}_obj{object_id}"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        yield os.path.join(scene_dir, stem + ext)


def _find_existing_image(single_image_dir: str, scene_id: str, object_id: str) -> str:
    for path in _candidate_image_paths(single_image_dir, scene_id, object_id):
        if os.path.exists(path):
            return path
    return ""


def collect_targets(
    requests: List[dict],
    bbox_dir: str,
) -> Tuple[Set[Tuple[str, str]], List[dict]]:
    selected: Set[Tuple[str, str]] = set()
    manifest_rows: List[dict] = []
    bbox_cache: Dict[str, List[dict]] = {}

    for item in requests:
        scene_id = item.get("scene_id")
        query_object_id = str(item.get("object_id"))
        query_category = item.get("object_name") or ""
        if not scene_id or not query_category:
            continue

        if scene_id not in bbox_cache:
            bbox_cache[scene_id] = _load_scene_bboxes(bbox_dir, scene_id)
        bboxes = bbox_cache[scene_id]

        matched_object_ids: List[str] = []
        for obj in bboxes:
            object_id = str(obj.get("object_id"))
            object_name = obj.get("object_name") or ""
            if _category_match(object_name, query_category):
                selected.add((scene_id, object_id))
                matched_object_ids.append(object_id)

        manifest_rows.append(
            {
                "scene_id": scene_id,
                "query_object_id": query_object_id,
                "query_category": query_category,
                "matched_object_ids": matched_object_ids,
                "num_matched_objects": len(matched_object_ids),
            }
        )

    return selected, manifest_rows


def copy_selected_images(
    selected: Set[Tuple[str, str]],
    single_image_dir: str,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    copied = 0
    missing = []
    for scene_id, object_id in sorted(selected):
        src_path = _find_existing_image(single_image_dir, scene_id, object_id)
        if not src_path:
            missing.append({"scene_id": scene_id, "object_id": object_id})
            continue

        dst_dir = os.path.join(output_dir, scene_id)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        if os.path.exists(dst_path):
            copied += 1
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1

    return {
        "num_selected_objects": len(selected),
        "num_copied_images": copied,
        "missing_images": missing,
    }


def main():
    requests = _load_requests(DEFAULT_ND_JSONL, DEFAULT_NO_JSON)
    selected, manifest_rows = collect_targets(
        requests=requests,
        bbox_dir=DEFAULT_BBOX_DIR,
    )
    copy_stats = copy_selected_images(
        selected=selected,
        single_image_dir=DEFAULT_SINGLE_IMAGE_DIR,
        output_dir=DEFAULT_OUTPUT_DIR,
    )

    manifest = {
        "num_requests": len(requests),
        "num_request_rows": len(manifest_rows),
        **copy_stats,
        "rows": manifest_rows,
    }
    with open(DEFAULT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"num_requests: {len(requests)}")
    print(f"num_selected_objects: {copy_stats['num_selected_objects']}")
    print(f"num_copied_images: {copy_stats['num_copied_images']}")
    print(f"num_missing_images: {len(copy_stats['missing_images'])}")
    print(f"output_dir: {DEFAULT_OUTPUT_DIR}")
    print(f"manifest: {DEFAULT_MANIFEST_PATH}")


if __name__ == "__main__":
    main()

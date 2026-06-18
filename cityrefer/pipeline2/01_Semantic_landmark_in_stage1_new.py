import argparse
import json
import os
import sys


_BUNDLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _bundle_or_legacy(rel_parts, legacy_path: str) -> str:
    p = os.path.join(_BUNDLE_ROOT, "data", *rel_parts)
    return p if os.path.exists(p) else legacy_path


COLOR_KEYWORD_TO_TIER = {
    "off white": 1,
    "off-white": 1,
    "white": 1,
    "cream": 1,
    "silver": 2,
    "beige": 2,
    "tan": 2,
    "light gray": 2,
    "light grey": 2,
    "light yellow": 2,
    "gold": 2,
    "gray": 3,
    "grey": 3,
    "stone": 3,
    "red": 4,
    "maroon": 4,
    "burgundy": 4,
    "dark red": 4,
    "pink": 4,
    "orange": 5,
    "yellow": 5,
    "rust": 5,
    "green": 6,
    "light green": 6,
    "teal": 6,
    "turquoise": 6,
    "blue": 7,
    "light blue": 7,
    "sky blue": 7,
    "bright blue": 7,
    "dark blue": 8,
    "navy blue": 8,
    "deep blue": 8,
    "purple": 9,
    "violet": 9,
    "magenta": 9,
    "brown": 10,
    "dark brown": 10,
    "light brown": 10,
    "dirt": 10,
    "olive": 10,
    "dark gray": 11,
    "dark grey": 11,
    "charcoal": 11,
    "dim": 11,
    "black": 11,
    "dark green": 11,
}

DEFAULT_OUTPUT_JSONL_ND = _bundle_or_legacy(
    (
        "Cityrefer",
        "meta_data",
        "0311data",
        "CityRefer_val_ND_0421_stage1_candidates.jsonl",
    ),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0421_stage1_candidates.jsonl",
)
DEFAULT_OUTPUT_JSONL_NO = _bundle_or_legacy(
    (
        "Cityrefer",
        "meta_data",
        "0311data",
        "CityRefer_val_NO_0421_stage1_candidates.jsonl",
    ),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_NO_0421_stage1_candidates.jsonl",
)


def _norm_tier(val):
    """Normalize final_tier for comparison (handles int/float/str from JSON)."""
    if val is None or val == "" or val == "-1":
        return None
    return str(val).strip()


def _normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = text.replace("the ", "")
    text = text.replace(" building", "")
    text = text.replace(" street", "")
    text = text.replace(" road", "")
    return text.strip()


def _tier_to_int(val):
    norm = _norm_tier(val)
    if norm is None:
        return None
    try:
        return int(float(norm))
    except (TypeError, ValueError):
        return None


def _normalize_color_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = text.replace("/", ",")
    text = text.replace("&", ",")
    text = text.replace("-", " ")
    return " ".join(text.split())


def _extract_target_tiers_from_color_text(color_text):
    normalized = _normalize_color_text(color_text)
    if not normalized:
        return set()

    tiers = set()
    for color_key, tier in sorted(
        COLOR_KEYWORD_TO_TIER.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        if color_key in normalized:
            tiers.add(tier)
    return tiers


def _get_main_object_target_tiers(construction_list):
    for obj in construction_list or []:
        if obj.get("is_main") is not True:
            continue
        return _extract_target_tiers_from_color_text(obj.get("color"))
    return set()


def _get_allowed_tiers(target_tiers):
    """
    Build a looser tier set for color matching.
    - +/-2 around the queried tier
    - merge visually confusable groups
    """
    if not target_tiers:
        return set()

    allowed = set()
    for tier in target_tiers:
        allowed.update(range(max(1, tier - 2), min(11, tier + 2) + 1))

        if tier in {1, 2, 3, 11}:
            allowed.update({1, 2, 3, 11})
        if tier in {7, 8}:
            allowed.update({7, 8})
        if tier in {4, 5}:
            allowed.update({4, 5})
        if tier in {10, 11, 3}:
            allowed.update({10, 11, 3})
    return allowed


def _main_object_color_missing(construction_list):
    """
    True if the first construction entry with is_main=True has color None or "".
    CityRefer ND: when the main object has no color annotation, skip tier filtering.
    """
    for obj in construction_list or []:
        if obj.get("is_main") is not True:
            continue
        c = obj.get("color")
        if c is None:
            return True
        if isinstance(c, str) and not c.strip():
            return True
        return False
    return False


def load_color_map(color_jsonl_path):
    """Load (scene_id, object_id) -> final_tier from tier2 jsonl."""
    color_map = {}
    if not os.path.exists(color_jsonl_path):
        print(f"警告：颜色文件不存在: {color_jsonl_path}")
        return color_map
    with open(color_jsonl_path, "r", encoding="utf-8") as cf:
        for line in cf:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            scene_id = item.get("scene_id")
            obj_id = str(item.get("object_id"))
            pred_tier = item.get("final_tier")
            if scene_id is not None and obj_id:
                color_map[(scene_id, obj_id)] = pred_tier
    return color_map


def process_item_to_candidates(color_map, bbox_dir, item, verbose=False):
    """
    One jsonl sample -> stage-1 candidates (category + landmark + optional color tier).
    Same schema for CityRefer ND / NO jsonl.
    """
    try:
        return _process_item_to_candidates_impl(color_map, bbox_dir, item, verbose)
    except Exception as e:
        print(e)
        return {"scene_id": None, "gt_id": None, "candidates": []}


def _process_item_to_candidates_impl(color_map, bbox_dir, item, verbose=False):
    scene_id = item["scene_id"]
    gt_id = str(item["object_id"])
    required_category = item["object_name"]
    construction_list = item.get("construction") or []

    # Landmarks: all non-empty landmark fields in construction
    raw_landmarks = []
    for obj in construction_list:
        lm = obj.get("landmark")
        if lm is not None and str(lm).strip():
            raw_landmarks.append(str(lm).strip())
    raw_landmarks = list(dict.fromkeys(raw_landmarks))

    norm_required_category = _normalize_text(required_category)

    bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
    landmark_path = os.path.join(bbox_dir, f"{scene_id}_bbox_landmark2.json")

    if not os.path.exists(bbox_path):
        return {"scene_id": scene_id, "gt_id": gt_id, "candidates": []}
    if not os.path.exists(landmark_path):
        return {"scene_id": scene_id, "gt_id": gt_id, "candidates": []}

    with open(bbox_path, "r", encoding="utf-8") as bf:
        bbox_json = json.load(bf)
        bbox_list = bbox_json.get("bboxes", [])

    existing_landmarks_in_scene = set()
    for obj in bbox_list:
        lm = obj.get("landmark")
        if lm:
            existing_landmarks_in_scene.add(_normalize_text(lm))

    norm_input_landmarks = [_normalize_text(lm) for lm in raw_landmarks if lm]
    valid_landmarks_to_check = []
    for lm in norm_input_landmarks:
        if not lm:
            continue
        if any(lm in exist_lm for exist_lm in existing_landmarks_in_scene):
            valid_landmarks_to_check.append(lm)

    with open(landmark_path, "r", encoding="utf-8") as lf:
        landmark_json = json.load(lf)
        landmark_list = landmark_json.get("objects", [])

    landmark_dict = {}
    for lm_item in landmark_list:
        obj_id = str(lm_item["object_id"])
        nearest_lms = lm_item.get("nearest_landmarks", [])
        landmark_dict[obj_id] = [_normalize_text(lm) for lm in nearest_lms]

    candidates = []
    for obj in bbox_list:
        obj_id = str(obj["object_id"])
        obj_name = obj["object_name"]
        norm_obj_name = _normalize_text(obj_name)
        if norm_obj_name != norm_required_category:
            continue

        if valid_landmarks_to_check:
            obj_nearest_lms = landmark_dict.get(obj_id, [])
            is_match = any(
                any(req_lm in obj_lm for obj_lm in obj_nearest_lms)
                for req_lm in valid_landmarks_to_check
            )
            if not is_match:
                continue

        candidates.append(obj_id)

    # Color: infer target tier(s) from the query color text, then apply a loose tier
    # filter. If the filter becomes too restrictive or tier info is invalid, skip it.
    if not _main_object_color_missing(construction_list):
        target_tiers = _get_main_object_target_tiers(construction_list)
        if target_tiers and len(target_tiers) == 1:
            allowed_tiers = _get_allowed_tiers(target_tiers)
            color_filtered = []
            invalid_tier_found = False
            for cand_id in candidates:
                cand_tier = _tier_to_int(color_map.get((scene_id, cand_id)))
                if cand_tier is None or cand_tier == -1:
                    invalid_tier_found = True
                    continue
                if cand_tier in allowed_tiers:
                    color_filtered.append(cand_id)
            # Skip color filtering when tier predictions are unreliable or the query
            # itself is multi-color / ambiguous.
            if not invalid_tier_found and len(color_filtered) >= max(3, len(candidates) // 5):
                candidates = color_filtered
        elif verbose:
            print("Skip color filter: query color is unmapped or multi-tier")

    return {
        "scene_id": scene_id,
        "gt_id": gt_id,
        "candidates": candidates,
    }


def process_files_to_candidates(bbox_dir, data_jsonl_path, color_jsonl_path, index):
    """
    Stage-1 candidates: same category + landmarks from construction (non-empty landmark
    fields in CityRefer jsonl), validated against *_bbox_landmark2.json; then filter
    by query color mapped to tier(s), allowing +/-1 neighboring tier, unless the main
    construction object (is_main: true) has empty color — then no tier filter.

    data_jsonl_path: e.g. CityRefer_val_ND_0324.jsonl or CityRefer_val_NO_0324.jsonl —
    one line per sample; landmarks are collected from item['construction'][*]['landmark'].
    """
    color_map = load_color_map(color_jsonl_path)

    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num != index:
                continue
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                return process_item_to_candidates(color_map, bbox_dir, item, verbose=False)
            except Exception as e:
                print(e)
                return {"scene_id": None, "gt_id": None, "candidates": []}

    return {"scene_id": None, "gt_id": None, "candidates": []}


def save_all_candidates_jsonl(
    bbox_dir,
    data_jsonl_path,
    color_jsonl_path,
    output_jsonl_path,
):
    """
    Walk all lines in data_jsonl_path, compute stage-1 candidates per line, write one JSON
    object per line to output_jsonl_path. Works for ND and NO splits (same item schema).
    """
    color_map = load_color_map(color_jsonl_path)
    out_dir = os.path.dirname(os.path.abspath(output_jsonl_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = 0
    with open(data_jsonl_path, "r", encoding="utf-8") as fin, open(
        output_jsonl_path, "w", encoding="utf-8"
    ) as fout:
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                rec = {
                    "line_idx": line_idx,
                    "scene_id": None,
                    "object_id": None,
                    "ann_id": None,
                    "object_name": None,
                    "candidates": [],
                    "num_candidates": 0,
                    "error": f"json: {e}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                continue

            res = process_item_to_candidates(color_map, bbox_dir, item, verbose=False)
            rec = {
                "line_idx": line_idx,
                "scene_id": res.get("scene_id"),
                "object_id": res.get("gt_id"),
                "ann_id": item.get("ann_id"),
                "object_name": item.get("object_name"),
                "candidates": res.get("candidates") or [],
                "num_candidates": len(res.get("candidates") or []),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"已保存 {n} 条候选结果到: {output_jsonl_path}")
    return n


# Default data paths (ND / NO 与 stage1_pipeline2 可对照修改)
DEFAULT_JSONL_ND = _bundle_or_legacy(
    ("Cityrefer", "meta_data", "0311data", "CityRefer_val_ND_0324.jsonl"),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0324.jsonl",
)
DEFAULT_JSONL_NO = _bundle_or_legacy(
    ("Cityrefer", "meta_data", "0311data", "CityRefer_val_NO_0324.jsonl"),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_NO_0324.jsonl",
)
DEFAULT_BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d"
DEFAULT_COLOR_JSONL = _bundle_or_legacy(
    ("Cityrefer", "0311data", "feature", "all_objects_color_per_image_tier2.jsonl"),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature/all_objects_color_per_image_tier2.jsonl",
)
DEFAULT_SAVE_DIR = _bundle_or_legacy(
    ("Cityrefer", "meta_data", "0311data"),
    "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data",
)


def _default_save_path_for_jsonl(data_jsonl_path):
    base = os.path.splitext(os.path.basename(data_jsonl_path))[0]
    if base == "CityRefer_val_ND_0324":
        return DEFAULT_OUTPUT_JSONL_ND
    if base == "CityRefer_val_NO_0324":
        return DEFAULT_OUTPUT_JSONL_NO
    return os.path.join(DEFAULT_SAVE_DIR, f"{base}_stage1_candidates.jsonl")


def save_both_nd_no_splits(bbox_dir, color_jsonl_path):
    """
    分别处理 ND / NO 两个 jsonl，各写入一份默认路径的 *_stage1_candidates.jsonl。
    """
    pairs = (
        ("ND", DEFAULT_JSONL_ND),
        ("NO", DEFAULT_JSONL_NO),
    )
    for tag, data_path in pairs:
        if not os.path.exists(data_path):
            print(f"[{tag}] 跳过（文件不存在）: {data_path}")
            continue
        out_path = _default_save_path_for_jsonl(data_path)
        print(f"[{tag}] 输入: {data_path}\n      保存: {out_path}")
        save_all_candidates_jsonl(bbox_dir, data_path, color_jsonl_path, out_path)


def run_rolling_eval(bbox_dir, data_jsonl_path, color_jsonl_path):
    """逐行评估：平均候选数、GT 命中率，以及 GT 不在候选中的比例。"""
    if not os.path.exists(data_jsonl_path):
        print(f"文件不存在，跳过评估: {data_jsonl_path}")
        return

    def get_all_line_indices(jsonl_path):
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return list(range(count))

    all_indices = get_all_line_indices(data_jsonl_path)
    n = len(all_indices)
    print(f"数据集: {data_jsonl_path}")
    print(f"总行数: {n}")
    if n == 0:
        print("平均候选数: —")
        print("准确率 (GT ∈ candidates): —")
        print("GT 不在 candidates 的比例: —")
        return

    summ = 0
    good = 0
    for i in range(n):
        results = process_files_to_candidates(
            bbox_dir, data_jsonl_path, color_jsonl_path, i
        )
        summ += len(results["candidates"])
        gt_id = results.get("gt_id")
        if gt_id and gt_id in results["candidates"]:
            good += 1
    print(f"平均候选数: {summ / n}")
    print(f"准确率 (GT ∈ candidates): {good / n}")
    print(f"GT 不在 candidates 的比例: {(n - good) / n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage-1 semantic+landmark(+color) candidates; optional dump all lines to jsonl."
    )
    parser.add_argument(
        "--dataset",
        choices=("nd", "no"),
        default="nd",
        help="Shortcut: nd -> ND jsonl, no -> NO jsonl (overridden by --jsonl).",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Input jsonl. If omitted (仅评估时): 依次评估 ND 与 NO 两个默认 jsonl 并分别打印准确率；"
        "与 --save 联用时仍用 --dataset 选择要保存的那份。",
    )
    parser.add_argument(
        "--bbox-dir",
        default=DEFAULT_BBOX_DIR,
        help="Directory of *_bbox.json and *_bbox_landmark2.json",
    )
    parser.add_argument(
        "--color-jsonl",
        default=DEFAULT_COLOR_JSONL,
        help="all_objects_color_per_image_tier2.jsonl",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="OUT.jsonl",
        help="保存当前所选数据集的全部候选。省略路径时写到 meta_data 下 <stem>_stage1_candidates.jsonl",
    )
    parser.add_argument(
        "--save-both",
        action="store_true",
        help="分别处理 ND 与 NO 两个 jsonl，各保存一份（默认路径，见 --save 说明）。",
    )
    parser.add_argument(
        "--also-eval",
        action="store_true",
        help="保存结束后仍跑滚动评估；与 --save-both 联用时对 ND、NO 各跑一轮。",
    )
    args = parser.parse_args()

    data_jsonl_path = args.jsonl
    if not data_jsonl_path:
        data_jsonl_path = DEFAULT_JSONL_ND if args.dataset == "nd" else DEFAULT_JSONL_NO

    if args.save_both:
        save_both_nd_no_splits(args.bbox_dir, args.color_jsonl)
        if args.also_eval:
            print("\n========== 评估 ND ==========")
            run_rolling_eval(args.bbox_dir, DEFAULT_JSONL_ND, args.color_jsonl)
            print("\n========== 评估 NO ==========")
            run_rolling_eval(args.bbox_dir, DEFAULT_JSONL_NO, args.color_jsonl)
        sys.exit(0)

    if args.save is not None:
        out_path = args.save if args.save != "__auto__" else _default_save_path_for_jsonl(
            data_jsonl_path
        )
        save_all_candidates_jsonl(
            args.bbox_dir,
            data_jsonl_path,
            args.color_jsonl,
            out_path,
        )
        if not args.also_eval:
            sys.exit(0)

    # 未指定 --jsonl 时：分别对 ND / NO 两个 jsonl 评估并打印各自准确率
    if args.jsonl is None:
        print("\n========== CityRefer ND ==========")
        run_rolling_eval(args.bbox_dir, DEFAULT_JSONL_ND, args.color_jsonl)
        print("\n========== CityRefer NO ==========")
        run_rolling_eval(args.bbox_dir, DEFAULT_JSONL_NO, args.color_jsonl)
    else:
        run_rolling_eval(args.bbox_dir, data_jsonl_path, args.color_jsonl)

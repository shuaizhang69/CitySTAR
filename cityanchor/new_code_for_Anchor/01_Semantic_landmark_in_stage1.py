import argparse
import json
import math
import os
import sys

SOFT_TIER_GROUPS = (
    {1, 2, 3, 11},
    {4, 5},
    {6, 11},
    {7, 8},
    {10, 3, 11},
)

DEFAULT_JSONL_ND = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_ND_0324.jsonl"
DEFAULT_JSONL_NO_CANDIDATES = (
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_0324_new.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new_0324_new.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_0324.jsonl",
)
DEFAULT_BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/bbox"
DEFAULT_COLOR_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_colors2_per_image_tier2.jsonl"
DEFAULT_QUERY_COLOR_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm_rerun_object_name.json"
DEFAULT_SAVE_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates/support_object_proximity_shrink"


def _resolve_existing_path(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0] if paths else None


DEFAULT_JSONL_NO = _resolve_existing_path(*DEFAULT_JSONL_NO_CANDIDATES)


def _norm_tier(val):
    if val is None or val == "" or val == "-1":
        return None
    return str(val).strip()


def _tier_to_int(val):
    norm = _norm_tier(val)
    if norm is None:
        return None
    try:
        return int(float(norm))
    except (TypeError, ValueError):
        return None


def _normalize_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = text.replace("the ", "")
    text = text.replace(" building", "")
    text = text.replace(" street", "")
    text = text.replace(" road", "")
    return text.strip()


def _category_soft_match(norm_obj_name, norm_required_category):
    if norm_obj_name == norm_required_category:
        return True
    if {norm_obj_name, norm_required_category} == {"vehicle", "truck"}:
        return True
    return False


def _norm_jsonl_path(path):
    if not path:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))


def _color_fields_from_record(rec):
    """Support cleaned records (color_phrase, mapped_tiers) and legacy llm_* keys."""
    raw_tiers = rec.get("mapped_tiers")
    if raw_tiers is None:
        raw_tiers = rec.get("llm_mapped_tiers", []) or []
    labels = rec.get("color_labels")
    if labels is None:
        labels = rec.get("llm_matched_color_labels", []) or []
    phrase = rec.get("color_phrase")
    if phrase is None:
        phrase = rec.get("llm_main_object_color_phrase")
    if phrase is None:
        phrase = ""
    src = rec.get("color_source")
    if src is None:
        src = rec.get("llm_color_source")
    if not src:
        src = "unknown"
    return {
        "mapped_tiers": [int(x) for x in raw_tiers if str(x).strip()],
        "color_phrase": str(phrase),
        "color_source": str(src),
        "matched_color_labels": list(labels) if labels is not None else [],
    }


def load_query_color_map(query_color_json_path):
    """
    index["by_path_line"] — key (abs_jsonl_path, line_idx), disambiguates duplicate scene/object.
    index["by_id"] — (scene_id, object_id) last-wins, for calls without line context.
    """
    index = {"by_path_line": {}, "by_id": {}}
    if not os.path.exists(query_color_json_path):
        print(f"警告：查询颜色文件不存在: {query_color_json_path}")
        return index

    with open(query_color_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for rec in data.get("records", []):
        scene_id = rec.get("scene_id")
        obj_id = str(rec.get("object_id"))
        if not scene_id or not obj_id:
            continue
        info = _color_fields_from_record(rec)
        index["by_id"][(scene_id, obj_id)] = info
        src = rec.get("source_path")
        line_idx = rec.get("line_idx")
        if src is not None and line_idx is not None:
            k = (_norm_jsonl_path(src), int(line_idx))
            index["by_path_line"][k] = info
    return index


def _get_query_color_info(query_index, scene_id, gt_id, jsonl_path=None, line_idx=None):
    info = None
    if query_index and jsonl_path is not None and line_idx is not None:
        k = (_norm_jsonl_path(jsonl_path), int(line_idx))
        by_pl = query_index.get("by_path_line") or {}
        info = by_pl.get(k)
    if not info and query_index:
        by_id = query_index.get("by_id") or {}
        info = by_id.get((scene_id, str(gt_id)))
    return info


def _get_main_object_target_tiers(
    scene_id,
    gt_id,
    query_index,
    jsonl_path=None,
    line_idx=None,
):
    info = _get_query_color_info(
        query_index, scene_id, gt_id, jsonl_path=jsonl_path, line_idx=line_idx
    )
    if not info:
        return set(), "query_color_missing", "", []
    tiers = {int(x) for x in info.get("mapped_tiers", [])}
    return (
        tiers,
        info.get("color_source") or "query_color_loaded",
        info.get("color_phrase") or "",
        info.get("matched_color_labels") or [],
    )


def _get_allowed_tiers(target_tiers):
    if not target_tiers:
        return set()

    allowed = set()
    for tier in target_tiers:
        allowed.update(range(max(1, tier - 2), min(11, tier + 2) + 1))
        for group in SOFT_TIER_GROUPS:
            if tier in group:
                allowed.update(group)
    return allowed


def _min_color_distance(cand_tier, target_tiers):
    if cand_tier is None or not target_tiers:
        return 999
    return min(abs(cand_tier - target_tier) for target_tier in target_tiers)


def _color_bucket(cand_tier, target_tiers):
    if not target_tiers:
        return 0
    if cand_tier is None or cand_tier == -1:
        return 3
    if cand_tier in target_tiers:
        return 0
    if cand_tier in _get_allowed_tiers(target_tiers):
        return 1
    return 2


def _soft_color_match_filter(
    candidate_records,
    target_tiers,
):
    if not target_tiers or not candidate_records:
        return list(candidate_records), {
            "color_filter_attempted": False,
            "color_filter_applied": False,
            "color_exact_count": 0,
            "color_soft_count": 0,
            "color_unknown_count": 0,
            "color_far_count": 0,
            "color_match_ratio": 0.0,
        }

    exact_records = []
    soft_records = []
    unknown_records = []
    far_records = []
    allowed_tiers = _get_allowed_tiers(target_tiers)

    for rec in candidate_records:
        cand_tier = rec["cand_tier"]
        if cand_tier is None or cand_tier == -1:
            unknown_records.append(rec)
        elif cand_tier in target_tiers:
            exact_records.append(rec)
        elif cand_tier in allowed_tiers:
            soft_records.append(rec)
        else:
            far_records.append(rec)

    matched_records = exact_records + soft_records
    match_ratio = len(matched_records) / len(candidate_records)
    should_apply = bool(matched_records)

    return (
        matched_records if should_apply else list(candidate_records),
        {
            "color_filter_attempted": True,
            "color_filter_applied": should_apply,
            "color_exact_count": len(exact_records),
            "color_soft_count": len(soft_records),
            "color_unknown_count": len(unknown_records),
            "color_far_count": len(far_records),
            "color_match_ratio": round(match_ratio, 4),
        },
    )


def _bbox_center_xy(obj):
    bbox = obj.get("bbox") or []
    if len(bbox) < 2:
        return None
    return float(bbox[0]), float(bbox[1])


def _euclidean_xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _collect_support_categories(construction_list):
    support_categories = []
    seen = set()
    for obj in construction_list or []:
        if obj.get("is_main") is True:
            continue
        label = obj.get("category") or obj.get("category2") or ""
        norm = _normalize_text(label)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        support_categories.append(norm)
    return support_categories


def _topk_keep_count(num_candidates, num_support_groups, keep_ratio, min_keep):
    ratio_keep = int(math.ceil(num_candidates * keep_ratio))
    support_keep = num_support_groups * 5
    return max(1, min(num_candidates, max(min_keep, ratio_keep, support_keep)))


def _support_object_proximity_shrink(
    candidates,
    bbox_list,
    construction_list,
    keep_ratio=0.35,
    min_keep=8,
):
    if len(candidates) <= 1:
        return list(candidates), {
            "support_proximity_attempted": False,
            "support_proximity_applied": False,
            "support_object_groups": 0,
        }

    support_categories = _collect_support_categories(construction_list)
    if not support_categories:
        return list(candidates), {
            "support_proximity_attempted": False,
            "support_proximity_applied": False,
            "support_object_groups": 0,
        }

    bbox_by_id = {str(obj.get("object_id")): obj for obj in bbox_list}
    support_groups = []
    for support_category in support_categories:
        group = []
        for obj in bbox_list:
            norm_obj_name = _normalize_text(obj.get("object_name"))
            if _category_soft_match(norm_obj_name, support_category):
                center = _bbox_center_xy(obj)
                if center is not None:
                    group.append((str(obj.get("object_id")), center))
        if group:
            support_groups.append(group)

    if not support_groups:
        return list(candidates), {
            "support_proximity_attempted": False,
            "support_proximity_applied": False,
            "support_object_groups": 0,
        }

    keep_count = _topk_keep_count(
        num_candidates=len(candidates),
        num_support_groups=len(support_groups),
        keep_ratio=keep_ratio,
        min_keep=min_keep,
    )
    if keep_count >= len(candidates):
        return list(candidates), {
            "support_proximity_attempted": True,
            "support_proximity_applied": False,
            "support_object_groups": len(support_groups),
        }

    scored = []
    for cand_id in candidates:
        cand_obj = bbox_by_id.get(str(cand_id))
        cand_center = _bbox_center_xy(cand_obj or {})
        if cand_center is None:
            continue
        per_group_dist = []
        for group in support_groups:
            dists = [
                _euclidean_xy(cand_center, support_center)
                for support_id, support_center in group
                if support_id != str(cand_id)
            ]
            if dists:
                per_group_dist.append(min(dists))
        if per_group_dist:
            scored.append((sum(per_group_dist) / len(per_group_dist), str(cand_id)))

    if len(scored) < keep_count:
        return list(candidates), {
            "support_proximity_attempted": True,
            "support_proximity_applied": False,
            "support_object_groups": len(support_groups),
        }

    scored.sort(key=lambda x: (x[0], x[1]))
    shrunk = [cand_id for _, cand_id in scored[:keep_count]]
    return shrunk, {
        "support_proximity_attempted": True,
        "support_proximity_applied": True,
        "support_object_groups": len(support_groups),
    }


def load_color_map(color_jsonl_path):
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


def process_item_to_candidates(
    color_map,
    query_color_map,
    bbox_dir,
    item,
    verbose=False,
    keep_ratio=0.35,
    min_keep=8,
    enable_support_shrink=False,
    jsonl_path=None,
    line_idx=None,
):
    try:
        return _process_item_to_candidates_impl(
            color_map=color_map,
            query_color_map=query_color_map,
            bbox_dir=bbox_dir,
            item=item,
            verbose=verbose,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
            enable_support_shrink=enable_support_shrink,
            jsonl_path=jsonl_path,
            line_idx=line_idx,
        )
    except Exception as e:
        print(e)
        return {"scene_id": None, "gt_id": None, "candidates": [], "meta": {}}


def _process_item_to_candidates_impl(
    color_map,
    query_color_map,
    bbox_dir,
    item,
    verbose=False,
    keep_ratio=0.35,
    min_keep=8,
    enable_support_shrink=False,
    jsonl_path=None,
    line_idx=None,
):
    scene_id = item["scene_id"]
    gt_id = str(item["object_id"])
    required_category = item["object_name"]
    construction_list = item.get("construction") or []

    norm_required_category = _normalize_text(required_category)
    bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
    if not os.path.exists(bbox_path):
        return {
            "scene_id": scene_id,
            "gt_id": gt_id,
            "candidates": [],
            "meta": {"missing_bbox": True},
        }

    with open(bbox_path, "r", encoding="utf-8") as bf:
        bbox_json = json.load(bf)
    bbox_list = bbox_json.get("bboxes", [])

    candidate_records = []
    for obj in bbox_list:
        obj_id = str(obj["object_id"])
        norm_obj_name = _normalize_text(obj["object_name"])
        if not _category_soft_match(norm_obj_name, norm_required_category):
            continue
        candidate_records.append(
            {
                "candidate_id": obj_id,
                "cand_tier": _tier_to_int(color_map.get((scene_id, obj_id))),
            }
        )

    target_tiers, color_source, color_phrase, color_labels = _get_main_object_target_tiers(
        scene_id=scene_id,
        gt_id=gt_id,
        query_index=query_color_map,
        jsonl_path=jsonl_path,
        line_idx=line_idx,
    )
    candidate_records, color_filter_meta = _soft_color_match_filter(
        candidate_records=candidate_records,
        target_tiers=target_tiers,
    )

    candidate_records.sort(
        key=lambda rec: (
            _color_bucket(rec["cand_tier"], target_tiers),
            _min_color_distance(rec["cand_tier"], target_tiers),
            int(rec["candidate_id"]) if str(rec["candidate_id"]).isdigit() else 10**18,
        )
    )
    candidates_before_support = [rec["candidate_id"] for rec in candidate_records]
    if enable_support_shrink:
        candidates_after_support, support_meta = _support_object_proximity_shrink(
            candidates=candidates_before_support,
            bbox_list=bbox_list,
            construction_list=construction_list,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
        )
    else:
        candidates_after_support = list(candidates_before_support)
        support_meta = {
            "support_proximity_attempted": False,
            "support_proximity_applied": False,
            "support_object_groups": 0,
            "support_shrink_enabled": False,
        }

    return {
        "scene_id": scene_id,
        "gt_id": gt_id,
        "candidates": candidates_after_support,
        "meta": {
            "missing_bbox": False,
            "num_same_category": len([1 for obj in bbox_list if _category_soft_match(_normalize_text(obj["object_name"]), norm_required_category)]),
            "num_after_color": len(candidates_before_support),
            "num_after_support_proximity": len(candidates_after_support),
            "query_color_source": color_source,
            "query_color_phrase": color_phrase,
            "query_color_labels": color_labels,
            "query_target_tiers": sorted(target_tiers),
            **color_filter_meta,
            "support_shrink_enabled": bool(enable_support_shrink),
            **support_meta,
        },
    }


def save_all_candidates_jsonl(
    bbox_dir,
    data_jsonl_path,
    color_jsonl_path,
    query_color_json_path,
    output_jsonl_path,
    keep_ratio=0.35,
    min_keep=8,
    enable_support_shrink=False,
):
    color_map = load_color_map(color_jsonl_path)
    query_color_map = load_query_color_map(query_color_json_path)
    out_dir = os.path.dirname(os.path.abspath(output_jsonl_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = 0
    with open(data_jsonl_path, "r", encoding="utf-8") as fin, open(output_jsonl_path, "w", encoding="utf-8") as fout:
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

            res = process_item_to_candidates(
                color_map=color_map,
                bbox_dir=bbox_dir,
                item=item,
                verbose=False,
                query_color_map=query_color_map,
                keep_ratio=keep_ratio,
                min_keep=min_keep,
                enable_support_shrink=enable_support_shrink,
                jsonl_path=data_jsonl_path,
                line_idx=line_idx,
            )
            rec = {
                "line_idx": line_idx,
                "scene_id": res.get("scene_id"),
                "object_id": res.get("gt_id"),
                "ann_id": item.get("ann_id"),
                "object_name": item.get("object_name"),
                "candidates": res.get("candidates") or [],
                "num_candidates": len(res.get("candidates") or []),
                "meta": res.get("meta") or {},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"已保存 {n} 条候选结果到: {output_jsonl_path}")
    return n


def _default_save_path_for_jsonl(data_jsonl_path):
    base = os.path.splitext(os.path.basename(data_jsonl_path))[0]
    return os.path.join(DEFAULT_SAVE_DIR, f"{base}_stage1_candidates_support_object_proximity_shrink.jsonl")


def run_rolling_eval(
    bbox_dir,
    data_jsonl_path,
    color_jsonl_path,
    query_color_json_path,
    keep_ratio=0.35,
    min_keep=8,
    enable_support_shrink=False,
):
    if not os.path.exists(data_jsonl_path):
        print(f"文件不存在，跳过评估: {data_jsonl_path}")
        return

    count = 0
    summ = 0
    good = 0
    color_attempted = 0
    color_applied = 0
    support_attempted = 0
    support_applied = 0

    color_map = load_color_map(color_jsonl_path)
    query_color_map = load_query_color_map(query_color_json_path)
    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            count += 1
            item = json.loads(line.strip())
            results = process_item_to_candidates(
                color_map=color_map,
                query_color_map=query_color_map,
                bbox_dir=bbox_dir,
                item=item,
                verbose=False,
                keep_ratio=keep_ratio,
                min_keep=min_keep,
                enable_support_shrink=enable_support_shrink,
                jsonl_path=data_jsonl_path,
                line_idx=line_idx,
            )
            candidates = results.get("candidates") or []
            meta = results.get("meta") or {}
            summ += len(candidates)
            gt_id = results.get("gt_id")
            if gt_id and gt_id in candidates:
                good += 1
            color_attempted += int(bool(meta.get("color_filter_attempted")))
            color_applied += int(bool(meta.get("color_filter_applied")))
            support_attempted += int(bool(meta.get("support_proximity_attempted")))
            support_applied += int(bool(meta.get("support_proximity_applied")))

    print(f"数据集: {data_jsonl_path}")
    print(f"总行数: {count}")
    if count == 0:
        print("平均候选数: —")
        print("准确率 (GT ∈ candidates): —")
        return

    print(f"平均候选数: {summ / count}")
    print(f"准确率 (GT ∈ candidates): {good / count}")
    print(f"main-color soft-match attempted: {color_attempted}")
    print(f"main-color soft-match applied: {color_applied}")
    print(f"support-object proximity attempted: {support_attempted}")
    print(f"support-object proximity applied: {support_applied}")


def print_query_gt_tier_mismatches(data_jsonl_path, color_jsonl_path, query_color_json_path, limit=12):
    color_map = load_color_map(color_jsonl_path)
    query_color_map = load_query_color_map(query_color_json_path)
    mismatch_count = 0
    total_with_both = 0
    examples = []

    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line.strip())
            scene_id = item["scene_id"]
            gt_id = str(item["object_id"])
            target_tiers, color_source, color_phrase, color_labels = _get_main_object_target_tiers(
                scene_id=scene_id,
                gt_id=gt_id,
                query_index=query_color_map,
                jsonl_path=data_jsonl_path,
                line_idx=line_idx,
            )
            gt_pred_tier = _tier_to_int(color_map.get((scene_id, gt_id)))
            if not target_tiers or gt_pred_tier is None:
                continue
            total_with_both += 1
            if gt_pred_tier in _get_allowed_tiers(target_tiers):
                continue
            mismatch_count += 1
            if len(examples) < limit:
                examples.append(
                    {
                        "scene_id": scene_id,
                        "object_id": gt_id,
                        "object_name": item.get("object_name"),
                        "description": item.get("description"),
                        "query_color_source": color_source,
                        "query_color_phrase": color_phrase,
                        "query_color_labels": color_labels,
                        "query_target_tiers": sorted(target_tiers),
                        "gt_pred_tier": gt_pred_tier,
                    }
                )

    print(f"query-tier vs GT-tier soft-mismatch: {mismatch_count}/{total_with_both}")
    for example in examples:
        print(json.dumps(example, ensure_ascii=False))


def save_both_nd_no_splits(
    bbox_dir,
    color_jsonl_path,
    query_color_json_path,
    keep_ratio=0.35,
    min_keep=8,
    enable_support_shrink=False,
):
    pairs = (("ND", DEFAULT_JSONL_ND), ("NO", DEFAULT_JSONL_NO))
    for tag, data_path in pairs:
        if not data_path or not os.path.exists(data_path):
            print(f"[{tag}] 跳过（文件不存在）: {data_path}")
            continue
        out_path = _default_save_path_for_jsonl(data_path)
        print(f"[{tag}] 输入: {data_path}\n      保存: {out_path}")
        save_all_candidates_jsonl(
            bbox_dir=bbox_dir,
            data_jsonl_path=data_path,
            color_jsonl_path=color_jsonl_path,
            query_color_json_path=query_color_json_path,
            output_jsonl_path=out_path,
            keep_ratio=keep_ratio,
            min_keep=min_keep,
            enable_support_shrink=enable_support_shrink,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage-1 semantic candidates with main-object color ranking and optional support-object proximity shrink."
    )
    parser.add_argument("--dataset", choices=("nd", "no"), default="nd")
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--bbox-dir", default=DEFAULT_BBOX_DIR)
    parser.add_argument("--color-jsonl", default=DEFAULT_COLOR_JSONL)
    parser.add_argument("--query-color-json", default=DEFAULT_QUERY_COLOR_JSON)
    parser.add_argument("--keep-ratio", type=float, default=0.35)
    parser.add_argument("--min-keep", type=int, default=8)
    parser.add_argument(
        "--enable-support-shrink",
        action="store_true",
        help="启用 support-object proximity shrink。默认关闭，优先保证 GT 保留率。",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="OUT.jsonl",
        help="保存当前所选数据集的全部候选。省略路径时写到 support_object_proximity_shrink 默认目录。",
    )
    parser.add_argument("--save-both", action="store_true")
    parser.add_argument("--also-eval", action="store_true")
    args = parser.parse_args()

    data_jsonl_path = args.jsonl
    if not data_jsonl_path:
        data_jsonl_path = DEFAULT_JSONL_ND if args.dataset == "nd" else DEFAULT_JSONL_NO

    if args.save_both:
        save_both_nd_no_splits(
            bbox_dir=args.bbox_dir,
            color_jsonl_path=args.color_jsonl,
            query_color_json_path=args.query_color_json,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            enable_support_shrink=args.enable_support_shrink,
        )
        if args.also_eval:
            print("\n========== 评估 ND ==========")
            run_rolling_eval(
                args.bbox_dir,
                DEFAULT_JSONL_ND,
                args.color_jsonl,
                args.query_color_json,
                keep_ratio=args.keep_ratio,
                min_keep=args.min_keep,
                enable_support_shrink=args.enable_support_shrink,
            )
            print_query_gt_tier_mismatches(DEFAULT_JSONL_ND, args.color_jsonl, args.query_color_json)
            print("\n========== 评估 NO ==========")
            run_rolling_eval(
                args.bbox_dir,
                DEFAULT_JSONL_NO,
                args.color_jsonl,
                args.query_color_json,
                keep_ratio=args.keep_ratio,
                min_keep=args.min_keep,
                enable_support_shrink=args.enable_support_shrink,
            )
            print_query_gt_tier_mismatches(DEFAULT_JSONL_NO, args.color_jsonl, args.query_color_json)
        sys.exit(0)

    if args.save is not None:
        out_path = args.save if args.save != "__auto__" else _default_save_path_for_jsonl(data_jsonl_path)
        save_all_candidates_jsonl(
            bbox_dir=args.bbox_dir,
            data_jsonl_path=data_jsonl_path,
            color_jsonl_path=args.color_jsonl,
            query_color_json_path=args.query_color_json,
            output_jsonl_path=out_path,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            enable_support_shrink=args.enable_support_shrink,
        )
        if not args.also_eval:
            sys.exit(0)

    if args.jsonl is None:
        print("\n========== CityRefer ND ==========")
        run_rolling_eval(
            args.bbox_dir,
            DEFAULT_JSONL_ND,
            args.color_jsonl,
            args.query_color_json,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            enable_support_shrink=args.enable_support_shrink,
        )
        print_query_gt_tier_mismatches(DEFAULT_JSONL_ND, args.color_jsonl, args.query_color_json)
        print("\n========== CityRefer NO ==========")
        run_rolling_eval(
            args.bbox_dir,
            DEFAULT_JSONL_NO,
            args.color_jsonl,
            args.query_color_json,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            enable_support_shrink=args.enable_support_shrink,
        )
        print_query_gt_tier_mismatches(DEFAULT_JSONL_NO, args.color_jsonl, args.query_color_json)
    else:
        run_rolling_eval(
            args.bbox_dir,
            data_jsonl_path,
            args.color_jsonl,
            args.query_color_json,
            keep_ratio=args.keep_ratio,
            min_keep=args.min_keep,
            enable_support_shrink=args.enable_support_shrink,
        )
        print_query_gt_tier_mismatches(data_jsonl_path, args.color_jsonl, args.query_color_json)

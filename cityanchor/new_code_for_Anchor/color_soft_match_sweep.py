#!/usr/bin/env python3
"""
Sweep color soft-match strategies on ND/NO: mine GT vs query-tier gaps, simulate avg candidates + recall.
Run: python color_soft_match_sweep.py

Main script wiring (±1 无 groups):
  --color-tier-half-width 1 --no-color-soft-groups --color-sparse-pm1-max-pool 400

Sweep 仅在「有 query 颜色且 GT 有 single_image tier」的 214 条上模拟颜色过滤；
主脚本 run_rolling_eval 会统计整条 jsonl（含无 query 颜色等），故平均候选数会略高。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter, defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_STAGE1 = os.path.join(_SCRIPT_DIR, "01_Semantic_color_in_stage1_support_object_proximity_shrink.py")


def _load_stage1():
    spec = importlib.util.spec_from_file_location("stage1_color", _STAGE1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage1_color"] = mod
    spec.loader.exec_module(mod)
    return mod


def _allowed_pm1(target_tiers: set[int]) -> set[int]:
    s = set()
    for t in target_tiers:
        t = int(t)
        for d in (-1, 0, 1):
            x = t + d
            if 1 <= x <= 11:
                s.add(x)
    return s


def _allowed_pm2_groups(mod, target_tiers: set[int]) -> set[int]:
    mod.set_color_stage_runtime(
        tier_half_width=2, use_soft_groups=True, color_soft_policy="legacy"
    )
    return mod._get_allowed_tiers(target_tiers)


def _allowed_confusion_default(mod, target_tiers: set[int]) -> set[int]:
    mod.set_color_stage_runtime(color_soft_policy="confusion_supplement")
    return mod._get_allowed_tiers(set(target_tiers))


def _allowed_exact(target_tiers: set[int]) -> set[int]:
    return set(int(t) for t in target_tiers)


def _filter_count(records, allowed: set[int], target_tiers: set[int]):
    n = 0
    for rec in records:
        c = rec["cand_tier"]
        if c is None or c == -1:
            continue
        if c in target_tiers or c in allowed:
            n += 1
    return n


def _mine_pm1_failures(mod, data_path, color_map, query_map):
    """Lines where GT tier is not in PM1-allowed set (query has tiers)."""
    fails = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = mod.json.loads(line.strip())
            scene_id = item["scene_id"]
            gt_id = str(item["object_id"])
            targets, _, phrase, labels = mod._get_main_object_target_tiers(
                scene_id, gt_id, query_map, jsonl_path=data_path, line_idx=line_idx
            )
            if not targets:
                continue
            g = mod._tier_to_int(color_map.get((scene_id, gt_id)))
            if g is None:
                continue
            allow = _allowed_pm1(targets)
            if g in allow:
                continue
            fails.append(
                {
                    "scene_id": scene_id,
                    "gt_id": gt_id,
                    "targets": frozenset(targets),
                    "gt_tier": g,
                    "phrase": phrase,
                    "labels": tuple(labels or ()),
                }
            )
    return fails


def _pairs_from_failures(fails):
    """Undirected tier pairs (min,max) for each (t in targets, gt_tier)."""
    pairs = Counter()
    for row in fails:
        g = row["gt_tier"]
        for t in row["targets"]:
            a, b = (int(t), int(g)) if int(t) <= int(g) else (int(g), int(t))
            pairs[(a, b)] += 1
    return pairs


def _allowed_edge_bridge(target_tiers: set[int], pair_set: set[tuple[int, int]]) -> set[int]:
    """Exact tiers + any c that forms a mined pair with some t in T."""
    s = set(int(t) for t in target_tiers)
    for t in target_tiers:
        t = int(t)
        for c in range(1, 12):
            a, b = (t, c) if t <= c else (c, t)
            if (a, b) in pair_set:
                s.add(c)
    return s


def _allowed_exact_plus_pm1_if_sparse(
    target_tiers: set[int], records: list, gt_id: str, k_sparse: int
) -> set[int]:
    """If same-category count <= k, use PM1; else exact only."""
    n = len(records)
    if n <= k_sparse:
        return _allowed_pm1(target_tiers)
    return _allowed_exact(target_tiers)


def _label_hint_extra(labels: tuple) -> set[int]:
    """From PM1-failure cases: small tier blobs per color word (heuristic, not exhaustive)."""
    s = set()
    flat = " ".join(str(x).lower() for x in labels)
    if "gray" in flat or "grey" in flat:
        s.update({10, 11, 6})
    if "black" in flat:
        s.update({1, 3, 11})
    if "white" in flat:
        s.update({10, 11, 1})
    if "green" in flat or "dark green" in flat:
        s.update({3, 6, 10, 11})
    if "brown" in flat or "reddish" in flat:
        s.update({3, 6, 10})
    return s


def _allowed_pm1_intersect_label_hint(
    target_tiers: set[int], records: list, gt_id: str, labels: tuple
) -> set[int]:
    """PM1 ring ∩ (T ∪ label hints): tighter than PM1 when hints small."""
    pm = _allowed_pm1(target_tiers)
    hint = set(target_tiers) | _label_hint_extra(labels)
    return pm & hint


def _allowed_pm1_union_label_hint(
    target_tiers: set[int], records: list, gt_id: str, labels: tuple
) -> set[int]:
    """PM1 ∪ label hints (can be looser than PM1)."""
    return _allowed_pm1(target_tiers) | _label_hint_extra(labels) | set(target_tiers)


def _allowed_exact_union_label_hint(
    target_tiers: set[int], records: list, gt_id: str, labels: tuple
) -> set[int]:
    """精准 ∪ 标签启发（无 ±1 环）；比 PM1 窄但试图用短语补洞。"""
    return set(int(t) for t in target_tiers) | _label_hint_extra(labels)


def main():
    mod = _load_stage1()
    bbox_dir = mod.DEFAULT_BBOX_DIR
    color_path = mod.DEFAULT_COLOR_JSONL
    query_path = mod.DEFAULT_QUERY_COLOR_JSON
    paths = [("ND", mod.DEFAULT_JSONL_ND), ("NO", mod.DEFAULT_JSONL_NO)]

    color_map = mod.load_color_map(color_path)
    query_map = mod.load_query_color_map(query_path)

    all_fails = []
    for tag, data_path in paths:
        if not data_path or not os.path.exists(data_path):
            continue
        f = _mine_pm1_failures(mod, data_path, color_map, query_map)
        for row in f:
            row["split"] = tag
        all_fails.extend(f)

    print("=== PM1 仍失败的 GT（用于挖 bridge pairs）===")
    print(f"count={len(all_fails)}")
    for row in all_fails:
        sp = row["split"]
        print(
            f"  [{sp}] {row['scene_id']} gt={row['gt_tier']} targets={sorted(row['targets'])} labels={row['labels']}"
        )

    pair_counts = _pairs_from_failures(all_fails)
    pair_set = set(pair_counts.keys())
    print("\n=== 从失败样本得到的 (tier_a,tier_b) 频次（无序）===")
    for (a, b), c in sorted(pair_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  ({a},{b}): {c}")

    strategies = [
        (
            "confusion_supplement(default)",
            lambda T, recs, gt, lb: _allowed_confusion_default(mod, T),
        ),
        ("exact", lambda T, recs, gt, lb: _allowed_exact(T)),
        ("pm1", lambda T, recs, gt, lb: _allowed_pm1(T)),
        ("edge_bridge", lambda T, recs, gt, lb: _allowed_edge_bridge(T, pair_set)),
        ("sparse200_pm1_else_exact", lambda T, recs, gt, lb: _allowed_exact_plus_pm1_if_sparse(T, recs, gt, 200)),
        ("sparse400_pm1_else_exact", lambda T, recs, gt, lb: _allowed_exact_plus_pm1_if_sparse(T, recs, gt, 400)),
        (
            "pm1_intersect_label_hint",
            lambda T, recs, gt, lb: _allowed_pm1_intersect_label_hint(T, recs, gt, lb),
        ),
        (
            "pm1_union_label_hint",
            lambda T, recs, gt, lb: _allowed_pm1_union_label_hint(T, recs, gt, lb),
        ),
        (
            "exact_union_label_hint",
            lambda T, recs, gt, lb: _allowed_exact_union_label_hint(T, recs, gt, lb),
        ),
    ]

    def eval_strategy(name, allow_fn):
        tot_lines = 0
        summ_cand = 0
        good = 0
        with_gt_color = 0
        for tag, data_path in paths:
            if not data_path or not os.path.exists(data_path):
                continue
            with open(data_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    item = mod.json.loads(line.strip())
                    scene_id = item["scene_id"]
                    gt_id = str(item["object_id"])
                    norm_req = mod._normalize_text(item["object_name"])
                    bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
                    if not os.path.exists(bbox_path):
                        continue
                    with open(bbox_path, "r", encoding="utf-8") as bf:
                        bbox_json = mod.json.load(bf)
                    bbox_list = bbox_json.get("bboxes", [])
                    records = []
                    for obj in bbox_list:
                        oid = str(obj["object_id"])
                        if not mod._category_soft_match(
                            mod._normalize_text(obj["object_name"]), norm_req
                        ):
                            continue
                        records.append(
                            {
                                "candidate_id": oid,
                                "cand_tier": mod._tier_to_int(
                                    color_map.get((scene_id, oid))
                                ),
                            }
                        )
                    targets, _, _, labels = mod._get_main_object_target_tiers(
                        scene_id, gt_id, query_map, jsonl_path=data_path, line_idx=line_idx
                    )
                    if not targets:
                        continue
                    g = mod._tier_to_int(color_map.get((scene_id, gt_id)))
                    if g is None:
                        continue
                    with_gt_color += 1
                    lb = tuple(labels or ())
                    allowed = allow_fn(set(targets), records, gt_id, lb)
                    # 与 _soft_color_match_filter 一致：unknown tier 的 bbox 仍保留
                    n_after = 0
                    kept_gt = False
                    for rec in records:
                        c = rec["cand_tier"]
                        oid = rec["candidate_id"]
                        if c is None or c == -1:
                            n_after += 1
                            if oid == gt_id:
                                kept_gt = True
                        elif c in targets or c in allowed:
                            n_after += 1
                            if oid == gt_id:
                                kept_gt = True
                    tot_lines += 1
                    summ_cand += n_after
                    if kept_gt:
                        good += 1
        avg = summ_cand / tot_lines if tot_lines else 0
        rec = good / tot_lines if tot_lines else 0
        return tot_lines, with_gt_color, avg, rec

    print("\n=== 策略对比（同 category 池 + 颜色过滤；unknown tier 全保留）===")
    print(f"{'strategy':<28} {'lines':>6} {'avg_cand':>10} {'GT_recall':>10}")
    mod.set_color_stage_runtime(
        tier_half_width=2, use_soft_groups=True, color_soft_policy="legacy"
    )
    n, w, avg, rec = eval_strategy(
        "pm2+groups(ref)", lambda T, r, g, lb: _allowed_pm2_groups(mod, T)
    )
    print(f"{'pm2+groups(ref)':<28} {n:>6} {avg:>10.4f} {rec:>10.4f}")

    for name, fn in strategies:
        n, w, avg, rec = eval_strategy(name, fn)
        print(f"{name:<28} {n:>6} {avg:>10.4f} {rec:>10.4f}")

    # LOO: mine pairs from ND only, test NO (and reverse) to see overfit
    print("\n=== 留一 split：用一侧失败样本的 pairs 做 edge_bridge，测另一侧 ===")

    def mine_pairs_for_path(data_path):
        fails = _mine_pm1_failures(mod, data_path, color_map, query_map)
        return set(_pairs_from_failures(fails).keys())

    nd_path, no_path = mod.DEFAULT_JSONL_ND, mod.DEFAULT_JSONL_NO
    pairs_nd = mine_pairs_for_path(nd_path) if os.path.exists(nd_path) else set()
    pairs_no = mine_pairs_for_path(no_path) if os.path.exists(no_path) else set()

    def eval_bridge_on_path(data_path, pair_set_local, label):
        tot = summ = hit = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = mod.json.loads(line.strip())
                scene_id = item["scene_id"]
                gt_id = str(item["object_id"])
                norm_req = mod._normalize_text(item["object_name"])
                bbox_path = os.path.join(bbox_dir, f"{scene_id}_bbox.json")
                if not os.path.exists(bbox_path):
                    continue
                with open(bbox_path, "r", encoding="utf-8") as bf:
                    bbox_list = mod.json.load(bf).get("bboxes", [])
                records = []
                for obj in bbox_list:
                    oid = str(obj["object_id"])
                    if not mod._category_soft_match(
                        mod._normalize_text(obj["object_name"]), norm_req
                    ):
                        continue
                    records.append(
                        {
                            "candidate_id": oid,
                            "cand_tier": mod._tier_to_int(color_map.get((scene_id, oid))),
                        }
                    )
                targets, _, _, _ = mod._get_main_object_target_tiers(
                    scene_id, gt_id, query_map, jsonl_path=data_path, line_idx=line_idx
                )
                if not targets:
                    continue
                g = mod._tier_to_int(color_map.get((scene_id, gt_id)))
                if g is None:
                    continue
                allow = _allowed_edge_bridge(set(targets), pair_set_local)
                n_after = kept = 0
                for rec in records:
                    c, oid = rec["cand_tier"], rec["candidate_id"]
                    if c is None or c == -1:
                        n_after += 1
                        if oid == gt_id:
                            kept = 1
                    elif c in targets or c in allow:
                        n_after += 1
                        if oid == gt_id:
                            kept = 1
                tot += 1
                summ += n_after
                hit += kept
        print(
            f"  {label}: N={tot} avg_cand={summ/tot:.4f} recall={hit/tot:.4f} (pairs={len(pair_set_local)})"
        )

    if os.path.exists(nd_path):
        eval_bridge_on_path(no_path, pairs_nd, "pairs mined ND → eval NO")
    if os.path.exists(no_path):
        eval_bridge_on_path(nd_path, pairs_no, "pairs mined NO → eval ND")


if __name__ == "__main__":
    main()

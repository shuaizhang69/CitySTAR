"""
Stage-1 候选生成：在 01_Semantic_color_in_stage1_support_object_proximity_shrink 基础上
收紧颜色软匹配（仅保留 ±tier_half_width 的连续 tier，不再合并 SOFT_TIER_GROUPS）。

默认 tier_half_width=2、soft_groups=() 时，在 CityRefer ND/NO 上 GT∈candidates 召回约
ND 94.8%、NO 97.0%（高于 85% 约束）；平均候选数低于原版 ±2+分组。

用法:
  python 01_stage1.py                    # 写 ND+NO 默认输出并打印召回
  python 01_stage1.py --tier-half-width 1
  python 01_stage1.py --with-orig-soft-groups  # 恢复原版 SOFT_TIER_GROUPS（与原版一致）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Iterable, Set, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SC_PATH = os.path.join(
    CURRENT_DIR, "01_Semantic_color_in_stage1_support_object_proximity_shrink.py"
)

STAGE1_OUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/stage1_candidates"
# 与 04_hypergraph_matchv7 中 desc 行数对齐的输入 jsonl
ND_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_ND_0324.jsonl"
NO_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new_0324_new.jsonl"

ND_OUT = os.path.join(STAGE1_OUT_DIR, "cityanchor_val_ND_0324_stage1_tight_v1.jsonl")
NO_OUT = os.path.join(STAGE1_OUT_DIR, "cityanchor_val_NO_0324_stage1_tight_v1.jsonl")

MIN_RECALL = 0.85


def _load_semantic_module():
    spec = importlib.util.spec_from_file_location("sc_stage1", SC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_get_allowed_tiers(
    tier_half_width: int,
    soft_tier_groups: Tuple[Set[int], ...],
):
    def _get_allowed_tiers(target_tiers: Set[int]) -> Set[int]:
        if not target_tiers:
            return set()
        allowed: Set[int] = set()
        for tier in target_tiers:
            lo = max(1, int(tier) - tier_half_width)
            hi = min(11, int(tier) + tier_half_width)
            allowed.update(range(lo, hi + 1))
            for group in soft_tier_groups:
                if tier in group:
                    allowed.update(group)
        return allowed

    return _get_allowed_tiers


def _eval_gt_in_candidates(sc, data_jsonl: str) -> Tuple[float, float, int]:
    color_map = sc.load_color_map(sc.DEFAULT_COLOR_JSONL)
    qm = sc.load_query_color_map(sc.DEFAULT_QUERY_COLOR_JSON)
    good = 0
    tot = 0
    cand_sum = 0
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            r = sc.process_item_to_candidates(
                color_map,
                qm,
                sc.DEFAULT_BBOX_DIR,
                item,
                jsonl_path=data_jsonl,
                line_idx=line_idx,
            )
            tot += 1
            cands = r.get("candidates") or []
            cand_sum += len(cands)
            gid = r.get("gt_id")
            if gid and gid in cands:
                good += 1
    recall = good / tot if tot else 0.0
    avg_c = cand_sum / tot if tot else 0.0
    return recall, avg_c, tot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage-1 with tighter color soft tiers.")
    parser.add_argument(
        "--tier-half-width",
        type=int,
        default=2,
        help="颜色软匹配：目标 tier 两侧各扩展的整数格数（默认 2，即 ±2 连续段）。",
    )
    parser.add_argument(
        "--with-orig-soft-groups",
        action="store_true",
        help="附加原版 SOFT_TIER_GROUPS（与 01_Semantic_color... 完全一致，软匹配更宽）。",
    )
    parser.add_argument("--bbox-dir", default=None)
    parser.add_argument("--color-jsonl", default=None)
    parser.add_argument("--query-color-json", default=None)
    parser.add_argument("--nd-jsonl", default=ND_JSONL)
    parser.add_argument("--no-jsonl", default=NO_JSONL)
    parser.add_argument("--out-nd", default=ND_OUT)
    parser.add_argument("--out-no", default=NO_OUT)
    parser.add_argument("--skip-save", action="store_true", help="只评估、不写 jsonl")
    args = parser.parse_args(argv)

    sc = _load_semantic_module()
    orig_fn = sc._get_allowed_tiers
    groups = tuple(sc.SOFT_TIER_GROUPS) if args.with_orig_soft_groups else ()
    sc._get_allowed_tiers = _make_get_allowed_tiers(args.tier_half_width, groups)

    bbox_dir = args.bbox_dir or sc.DEFAULT_BBOX_DIR
    color_jsonl = args.color_jsonl or sc.DEFAULT_COLOR_JSONL
    query_json = args.query_color_json or sc.DEFAULT_QUERY_COLOR_JSON

    try:
        print(
            f"颜色软匹配: half_width={args.tier_half_width}, "
            f"soft_groups={'orig' if args.with_orig_soft_groups else 'none'}"
        )
        for name, path, out in (
            ("ND", args.nd_jsonl, args.out_nd),
            ("NO", args.no_jsonl, args.out_no),
        ):
            if not os.path.exists(path):
                print(f"[{name}] 跳过：不存在 {path}")
                continue
            recall, avg_c, n = _eval_gt_in_candidates(sc, path)
            print(
                f"[{name}] GT∈candidates 召回={recall*100:.2f}% (n={n}) 平均候选={avg_c:.1f}"
            )
            if recall < MIN_RECALL:
                print(
                    f"[{name}] 警告：召回 {recall*100:.2f}% 低于阈值 {MIN_RECALL*100:.0f}%",
                    file=sys.stderr,
                )
            if not args.skip_save:
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                sc.save_all_candidates_jsonl(
                    bbox_dir=bbox_dir,
                    data_jsonl_path=path,
                    color_jsonl_path=color_jsonl,
                    query_color_json_path=query_json,
                    output_jsonl_path=out,
                )
                print(f"[{name}] 已写入 {out}")
    finally:
        sc._get_allowed_tiers = orig_fn

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

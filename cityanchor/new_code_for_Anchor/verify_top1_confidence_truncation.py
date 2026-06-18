#!/usr/bin/env python3
"""验证 `infer_confident_top1_only`（Top1 置信截断）：规则自洽、与 object_id/GT 无关；可选对已保存 rerank JSON 重算一致。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List


def load_r08():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, "08reranking_Reranking.py")
    spec = importlib.util.spec_from_file_location("r08", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_thresholds():
    """与 08 argparse 默认一致。"""
    return dict(
        api_margin_strong=0.12,
        api_margin_weak=0.04,
        api_margin_short=0.05,
        fused_margin_strong=0.15,
        short_pool_max=15,
    )


def mk_scored(pairs: List[tuple[str, float]], reasons: str = "qwen3-vl-rerank") -> List[Dict[str, Any]]:
    out = []
    for cid, sc in pairs:
        out.append(
            {
                "candidate_bbox_id": cid,
                "score": float(sc),
                "reason": reasons,
                "image_path": "/fake",
            }
        )
    return out


def run_synthetic(r08) -> int:
    th = default_thresholds()
    errors = 0

    def call(item, scored, truncate_k):
        return r08.infer_confident_top1_only(
            item,
            scored,
            truncate_k=truncate_k,
            api_margin_strong=th["api_margin_strong"],
            api_margin_weak=th["api_margin_weak"],
            api_margin_short=th["api_margin_short"],
            fused_margin_strong=th["fused_margin_strong"],
            short_pool_max=th["short_pool_max"],
        )

    # 1) API 分差足够大 → confident
    ok, _ = call({"all_fused_scores": {"a": 1.0, "b": 0.9}}, mk_scored([("x", 0.9), ("y", 0.7)]), 20)
    assert ok is True, "strong api margin expected True"

    # 2) 分差极小、fused 也近 → False
    ok, _ = call({"all_fused_scores": {"a": 5.1, "b": 5.09}}, mk_scored([("x", 0.51), ("y", 0.50)]), 20)
    assert ok is False, "ambiguous scores expected False"

    # 3) single rerank candidate → True
    ok, _ = call({}, mk_scored([("x", 0.8)]), 10)
    assert ok is True, "single candidate expected True"

    # 4) short pool：truncate_k≤15 + 分差≥0.05 → True（弱分差）
    ok, _ = call({}, mk_scored([("x", 0.2), ("y", 0.14)]), 15)
    assert ok is True, "short_truncated_pool expected True"

    # 5) GT / object_id 不应影响布尔：同一 scored+fused，改 object_id
    base_item = {"all_fused_scores": {"1": 3.0, "2": 2.7}, "object_id": "999"}
    alt_item = dict(base_item)
    alt_item["object_id"] = "000"
    scored = mk_scored([("a", 1.0), ("b", 0.92)])
    a = call(base_item, scored, 12)[0]
    b = call({k: v for k, v in base_item.items() if k != "object_id"}, scored, 12)[0]
    c = call(alt_item, scored, 12)[0]
    assert a == b == c, "confident_top1_only must ignore object_id / GT label"

    print("[synthetic] infer_confident_top1_only: OK (5 checks)")
    return errors


def recompute_record(r08, row: Dict[str, Any], th: Dict[str, Any]) -> bool:
    scored = row.get("api_rerank_top5_scores") or []
    truncate_k = int(row.get("hg_rerank_truncate_k") or row.get("api_rerank_topk") or 0)
    item_keys = ("all_fused_scores", "all_scores")
    item_sub = {k: row[k] for k in item_keys if k in row and row[k]}
    ok, _ = r08.infer_confident_top1_only(
        item_sub,
        scored,
        truncate_k=truncate_k,
        api_margin_strong=th["api_margin_strong"],
        api_margin_weak=th["api_margin_weak"],
        api_margin_short=th["api_margin_short"],
        fused_margin_strong=th["fused_margin_strong"],
        short_pool_max=th["short_pool_max"],
    )
    stored = row.get("confident_top1_only")
    if stored is None:
        return True
    return bool(ok) == bool(stored)


def run_rerank_json(path: str, r08) -> int:
    th = default_thresholds()
    if not os.path.isfile(path):
        print(f"[rerank-json] SKIP (不存在): {path}")
        return 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mismatch = 0
    checked = 0
    for row in data:
        if row.get("api_rerank_status") != "ok":
            continue
        scored = row.get("api_rerank_top5_scores") or []
        if not scored:
            continue
        checked += 1
        if not recompute_record(r08, row, th):
            mismatch += 1
            print(f"  mismatch key={row.get('scene_id')},{row.get('object_id')}: stored={row.get('confident_top1_only')}")
    print(f"[rerank-json] {path}: recomputed vs stored mismatch {mismatch}/{checked} (ok rows)")
    return mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rerank-json",
        nargs="*",
        default=[],
        help="已跑过的 08 输出 JSON，可选多个；核对 confident_top1_only 与重算一致",
    )
    args = ap.parse_args()

    r08 = load_r08()
    run_synthetic(r08)

    mism = 0
    default_paths = [
        "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_v7_geom_tie_v3_api_rerank_trunc20.json",
        "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_NO_0324_stage1_tight_v1_hgmatch_v7_geom_tie_v3_api_rerank_trunc30.json",
    ]
    paths = list(args.rerank_json) if args.rerank_json else default_paths
    for p in paths:
        mism += run_rerank_json(p, r08)

    if mism:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

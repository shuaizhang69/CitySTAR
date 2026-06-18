#!/usr/bin/env python3
"""
错例与「构造图区分度」分析报告（只读，不改数据）。

结合：
- 超图匹配结果 JSON（如 04_* 输出的 hgmatch_*.json）
- 描述超图 jsonl（*_desc_hypergraphs*.jsonl）
- 可选：bbox 候选超图 jsonl（逐候选一行）

输出：
1. 描述侧：边数分布、空图比例、含 weak_anchor 边的比例（若存在）。
2. 匹配侧：预测错误样本的描述边数分布 vs Top-1 正确样本。
3. （可选）bbox 侧：同一查询下不同候选的「结构指纹」碰撞率——量化 v2 式全 pairwise 导致的同质化。

用法示例：

  python3 analyze_supergraph_construction_errors.py \\
    --match-json /path/to/cityanchor_val_ND_*_hgmatch_v7_geom_tie_v5.json \\
    --desc-jsonl /path/to/cityanchor_val_ND_desc_hypergraphs.jsonl \\
    --bbox-jsonl-sample /path/to/cityanchor_val_ND_*_bbox_only_hypergraphs.jsonl \\
    --max-bbox-scan-lines 8000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def _key(rec: dict) -> Tuple[str, str, str]:
    return (
        str(rec.get("scene_id", "")),
        str(rec.get("object_id", "")),
        str(rec.get("ann_id", 0)),
    )


def load_desc_index(path: str) -> Dict[Tuple[str, str, str], dict]:
    idx: Dict[Tuple[str, str, str], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            idx[_key(data)] = data
    return idx


def desc_edge_stats(hg: dict) -> Tuple[int, bool]:
    edges = (hg.get("hypergraph") or {}).get("edges") or []
    n = len(edges)
    weak = any((e.get("info") or {}).get("weak_anchor") == "true" for e in edges)
    return n, weak


def fingerprint_hypergraph(edges: List[dict]) -> tuple:
    parts = []
    for e in edges:
        parts.append(
            (
                e.get("relation"),
                e.get("from"),
                e.get("to"),
                round(float(e.get("score") or 0), 2),
            )
        )
    return tuple(sorted(parts))


def scan_bbox_collisions(
    path: str,
    max_lines: int,
) -> None:
    """同一 (scene, query, ann) 下，候选之间结构指纹相同的比例。"""
    by_q: Dict[Tuple[str, str, str], List[Tuple[str, Tuple[Any, ...]]]] = defaultdict(list)
    n_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if n_lines >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            k = (
                str(rec.get("scene_id", "")),
                str(rec.get("query_object_id", "")),
                str(rec.get("ann_id", 0)),
            )
            bid = str(rec.get("bbox_id", ""))
            edges = (rec.get("hypergraph") or {}).get("edges") or []
            by_q[k].append((bid, fingerprint_hypergraph(edges)))
            n_lines += 1

    collision_queries = 0
    total_queries = 0
    for k, lst in by_q.items():
        if len(lst) < 2:
            continue
        total_queries += 1
        fps = [x[1] for x in lst]
        uniq = len(set(fps))
        if uniq < len(fps):
            collision_queries += 1

    print("\n--- Bbox 候选结构同质化（采样） ---")
    print(f"  扫描行数: {n_lines}（上限 {max_lines}）")
    print(f"  至少 2 候选的查询数: {total_queries}")
    if total_queries > 0:
        print(
            f"  存在「不同候选同结构指纹」的查询数: {collision_queries} "
            f"({collision_queries / total_queries * 100:.1f}%)"
        )
    else:
        print("  （数据不足，跳过碰撞统计）")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析构造超图导致的错例与区分度")
    parser.add_argument("--match-json", type=str, default="", help="匹配评估输出的 JSON")
    parser.add_argument("--desc-jsonl", type=str, required=True)
    parser.add_argument("--bbox-jsonl-sample", type=str, default="")
    parser.add_argument("--max-bbox-scan-lines", type=int, default=8000)
    args = parser.parse_args()

    print(f"加载描述超图索引: {args.desc_jsonl}")
    desc_index = load_desc_index(args.desc_jsonl)
    print(f"  条数: {len(desc_index)}")

    ec_bucket = Counter()
    weak_ct = 0
    total_desc = 0
    for rec in desc_index.values():
        total_desc += 1
        n, weak = desc_edge_stats(rec)
        ec_bucket[n] += 1
        if weak:
            weak_ct += 1

    print("\n--- 描述超图总体 ---")
    print(f"  样本数: {total_desc}")
    print(f"  无边（edge_count=0）: {ec_bucket.get(0, 0)} ({ec_bucket.get(0, 0) / max(total_desc,1) * 100:.1f}%)")
    print("  边数直方图（前 15 档）:")
    for k in sorted(ec_bucket.keys())[:15]:
        print(f"    {k} 条: {ec_bucket[k]}")
    print(f"  至少一条 weak_anchor 边（若有标记）的样本: {weak_ct}")

    if args.match_json and os.path.isfile(args.match_json):
        with open(args.match_json, "r", encoding="utf-8") as f:
            matches = json.load(f)
        wrong_ec = Counter()
        right_ec = Counter()
        wrong_n = right_n = 0
        missing_desc = 0
        for m in matches:
            if m.get("status") != "matched":
                continue
            k = (
                str(m.get("scene_id", "")),
                str(m.get("object_id", "")),
                str(m.get("ann_id", 0)),
            )
            desc_rec = desc_index.get(k)
            if not desc_rec:
                missing_desc += 1
                continue
            n, _ = desc_edge_stats(desc_rec)
            if m.get("is_correct"):
                right_ec[n] += 1
                right_n += 1
            else:
                wrong_ec[n] += 1
                wrong_n += 1

        print("\n--- 与匹配结果交叉（需 scene/object/ann 对齐） ---")
        print(f"  匹配记录中找不到描述行的数量: {missing_desc}")
        if wrong_n + right_n > 0:
            print(f"  Top-1 正确: {right_n}  错误: {wrong_n}")
            print("  错误样本的描述边数分布（前 10 档）:")
            for k in sorted(wrong_ec.keys())[:10]:
                print(f"    {k} 条: {wrong_ec[k]}")
            print("  正确样本的描述边数分布（前 10 档）:")
            for k in sorted(right_ec.keys())[:10]:
                print(f"    {k} 条: {right_ec[k]}")
    else:
        print("\n（未提供 --match-json 或文件不存在，跳过匹配交叉）")

    if args.bbox_jsonl_sample and os.path.isfile(args.bbox_jsonl_sample):
        scan_bbox_collisions(args.bbox_jsonl_sample, args.max_bbox_scan_lines)
    else:
        print("\n（未提供 --bbox-jsonl-sample，跳过候选碰撞分析）")

    print("\n--- 构造链路常见问题（对照 1_NER / 2 / 3 脚本） ---")
    print(
        "  1) NER 阶段四类 category 强约束 → 大量物体被强制归类，客体非四类被丢弃，语义损失。\n"
        "  2) 描述边归一化：短词优先匹配已改进于 desc_hypergraph_v3；旧逻辑易受「in」等子串干扰。\n"
        "  3) bbox 图：邻域内实例两两全连接时，不同候选易得到相似边集 → 建议用 3b main_incident。\n"
    )


if __name__ == "__main__":
    main()

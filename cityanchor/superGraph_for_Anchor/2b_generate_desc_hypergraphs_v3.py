"""
从 1_NER_multiprocess.py 输出的 *_infer_result.jsonl 生成描述超图（**schema v3**）。

使用同目录下 `desc_hypergraph_v3.py`：更长短语优先、方向词回退、代词降权、object 节点 id 修正。

默认输出文件名：`*_desc_hypergraphs_v3.jsonl`（与 2_generate_desc_hypergraphs.py 并存，不覆盖旧图）。
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from tqdm import tqdm

from desc_hypergraph_v3 import build_description_hypergraph


DEFAULT_INPUTS = [
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc/cityanchor_val_ND_infer_result.jsonl",
    "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc/cityanchor_val_NO_new_infer_result.jsonl",
]
DEFAULT_OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/desc"


def optimize_hypergraph_structure(graph):
    """
    优化超图结构，增加索引便于快速匹配。
    """
    if graph is None:
        return None

    result = {
        "schema_version": "desc_hypergraph_v3",
        "scene_id": graph.scene_id,
        "object_id": graph.object_id,
        "ann_id": getattr(graph, "ann_id", None),
        "object_name": getattr(graph, "object_name", ""),
        "description": graph.description,
        "hypergraph": {
            "nodes": [],
            "edges": [],
            "node_by_category": defaultdict(list),
            "edges_by_relation": defaultdict(list),
            "main_node": None,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        },
    }

    for node in graph.nodes:
        node_dict = {
            "id": node.id,
            "category": node.category,
            "is_main": node.is_main,
            "properties": node.properties,
        }
        result["hypergraph"]["nodes"].append(node_dict)
        result["hypergraph"]["node_by_category"][node.category].append(node.id)

        if node.is_main:
            result["hypergraph"]["main_node"] = node_dict
            result["hypergraph"]["main_category"] = node.category

    for edge in graph.edges:
        edge_dict = {
            "from": edge.from_node,
            "to": edge.to_node,
            "relation": edge.relation,
            "raw_relation": edge.raw_relation,
            "score": edge.score,
            "info": edge.info,
        }
        result["hypergraph"]["edges"].append(edge_dict)
        result["hypergraph"]["edges_by_relation"][edge.relation].append(edge_dict)

    result["hypergraph"]["node_by_category"] = dict(result["hypergraph"]["node_by_category"])
    result["hypergraph"]["edges_by_relation"] = dict(result["hypergraph"]["edges_by_relation"])
    return result


def _default_output_path(input_jsonl, output_dir):
    name = os.path.splitext(os.path.basename(input_jsonl))[0]
    if name.endswith("_infer_result"):
        name = name[: -len("_infer_result")]
    return os.path.join(output_dir, f"{name}_desc_hypergraphs_v3.jsonl")


def generate_hypergraphs(input_jsonl, output_jsonl):
    """从单个 infer_result.jsonl 生成描述超图 jsonl。"""
    print(f"Reading from: {input_jsonl}")

    stats = {
        "total": 0,
        "with_edges": 0,
        "relation_types": defaultdict(int),
        "category_pairs": defaultdict(int),
    }
    results = []

    with open(input_jsonl, "r", encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    for line in tqdm(lines, desc=f"Generating {os.path.basename(input_jsonl)}"):
        data = json.loads(line)
        graph = build_description_hypergraph(data)
        optimized = optimize_hypergraph_structure(graph)

        if not optimized:
            continue

        results.append(optimized)
        stats["total"] += 1
        if optimized["hypergraph"]["edge_count"] > 0:
            stats["with_edges"] += 1

        for rel_type in optimized["hypergraph"]["edges_by_relation"].keys():
            stats["relation_types"][rel_type] += 1

        main_cat = optimized["hypergraph"].get("main_category", "unknown")
        for node in optimized["hypergraph"]["nodes"]:
            if not node["is_main"]:
                pair = f"{main_cat}->{node['category']}"
                stats["category_pairs"][pair] += 1

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    print(f"\nSaving to: {output_jsonl}")
    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print("生成统计:")
    print(f"  总描述数: {stats['total']}")
    if stats["total"] > 0:
        ratio = stats["with_edges"] / stats["total"] * 100
        print(f"  含边的描述: {stats['with_edges']} ({ratio:.1f}%)")
    else:
        print("  含边的描述: 0 (0.0%)")
    print("\n  关系类型分布 (Top 10):")
    for rel, count in sorted(stats["relation_types"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {rel}: {count}")
    print("\n  主-客类别对分布 (Top 10):")
    for pair, count in sorted(stats["category_pairs"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate description hypergraphs (v3 builder) from infer_result jsonl."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input *_infer_result.jsonl files.",
    )
    parser.add_argument(
        "--glob-input",
        default="",
        help="Optional glob for input files, e.g. '/path/*_infer_result.jsonl'.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated *_desc_hypergraphs.jsonl files.",
    )
    args = parser.parse_args()

    input_paths = list(args.inputs)
    if args.glob_input:
        input_paths.extend(sorted(glob.glob(args.glob_input)))

    seen = set()
    unique_inputs = []
    for path in input_paths:
        if path not in seen:
            unique_inputs.append(path)
            seen.add(path)

    if not unique_inputs:
        raise ValueError("No input files provided.")

    os.makedirs(args.output_dir, exist_ok=True)

    for input_path in unique_inputs:
        if not os.path.exists(input_path):
            print(f"Skip missing input: {input_path}")
            continue
        output_path = _default_output_path(input_path, args.output_dir)
        generate_hypergraphs(input_path, output_path)


if __name__ == "__main__":
    main()

"""
从 CityRefer_val_infer_result.jsonl 生成描述超图
输出结构优化，便于快速匹配
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm
from desc_hypergraph import build_description_hypergraph


def optimize_hypergraph_structure(graph):
    """
    优化超图结构，增加索引便于快速匹配
    
    输入: DescriptionHyperGraph 对象
    输出: 优化后的字典结构
    """
    if graph is None:
        return None
    
    # 基础结构
    result = {
        "scene_id": graph.scene_id,
        "object_id": graph.object_id,
        "ann_id": getattr(graph, "ann_id", None),
        "object_name": getattr(graph, "object_name", ""),
        "description": graph.description,
        "hypergraph": {
            "nodes": [],
            "edges": [],
            # 索引结构
            "node_by_category": defaultdict(list),
            "edges_by_relation": defaultdict(list),
            "main_node": None,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
        }
    }
    
    # 处理节点
    for node in graph.nodes:
        node_dict = {
            "id": node.id,
            "category": node.category,
            "is_main": node.is_main,
            "properties": node.properties
        }
        result["hypergraph"]["nodes"].append(node_dict)
        
        # 按类别索引
        result["hypergraph"]["node_by_category"][node.category].append(node.id)
        
        # 记录主节点
        if node.is_main:
            result["hypergraph"]["main_node"] = node_dict
            result["hypergraph"]["main_category"] = node.category
    
    # 处理边
    for edge in graph.edges:
        edge_dict = {
            "from": edge.from_node,
            "to": edge.to_node,
            "relation": edge.relation,
            "raw_relation": edge.raw_relation,
            "score": edge.score,
            "info": edge.info
        }
        result["hypergraph"]["edges"].append(edge_dict)
        
        # 按关系类型索引
        result["hypergraph"]["edges_by_relation"][edge.relation].append(edge_dict)
    
    # 将 defaultdict 转为普通 dict (便于 JSON 序列化)
    result["hypergraph"]["node_by_category"] = dict(result["hypergraph"]["node_by_category"])
    result["hypergraph"]["edges_by_relation"] = dict(result["hypergraph"]["edges_by_relation"])
    
    return result


def generate_hypergraphs(input_jsonl, output_jsonl):
    """批量生成超图"""
    print(f"Reading from: {input_jsonl}")
    
    # 统计信息
    stats = {
        "total": 0,
        "with_edges": 0,
        "relation_types": defaultdict(int),
        "category_pairs": defaultdict(int)
    }
    
    results = []
    
    with open(input_jsonl, 'r') as f_in:
        lines = f_in.readlines()
    
    for line in tqdm(lines, desc="Generating hypergraphs"):
        data = json.loads(line.strip())
        
        # 构建超图
        graph = build_description_hypergraph(data)
        optimized = optimize_hypergraph_structure(graph)
        
        if optimized:
            results.append(optimized)
            
            # 更新统计
            stats["total"] += 1
            if optimized["hypergraph"]["edge_count"] > 0:
                stats["with_edges"] += 1
            
            # 统计关系类型
            for rel_type in optimized["hypergraph"]["edges_by_relation"].keys():
                stats["relation_types"][rel_type] += 1
            
            # 统计类别对 (主类别 -> 客类别)
            main_cat = optimized["hypergraph"].get("main_category", "unknown")
            for node in optimized["hypergraph"]["nodes"]:
                if not node["is_main"]:
                    pair = f"{main_cat}->{node['category']}"
                    stats["category_pairs"][pair] += 1
    
    # 保存结果
    print(f"\nSaving to: {output_jsonl}")
    with open(output_jsonl, 'w') as f_out:
        for result in results:
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"生成统计:")
    print(f"  总描述数: {stats['total']}")
    print(f"  含边的描述: {stats['with_edges']} ({stats['with_edges']/stats['total']*100:.1f}%)")
    print(f"\n  关系类型分布 (Top 10):")
    for rel, count in sorted(stats["relation_types"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {rel}: {count}")
    print(f"\n  主-客类别对分布 (Top 10):")
    for pair, count in sorted(stats["category_pairs"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {pair}: {count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_val_infer_result.jsonl")
    parser.add_argument('--output', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_desc_hypergraphs.jsonl")
    
    args = parser.parse_args()
    
    generate_hypergraphs(args.input, args.output)


if __name__ == "__main__":
    main()

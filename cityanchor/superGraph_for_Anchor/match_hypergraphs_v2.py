"""
超图匹配 v2：正确的图匹配算法（find_best_mapping），先建立节点映射再验证边匹配。

融合细排、扫参与完整评估与 match_hypergraphs.py 对齐；日常实验可优先使用主脚本。
"""

import json
import os
from collections import defaultdict
from itertools import product
from typing import Dict, Set, Tuple

from tqdm import tqdm

from match_hypergraphs import (
    get_desc_main_category,
    max_main_category_num_points,
    rank_prior_from_index,
)


# 关系语义映射表
RELATION_SEMANTIC_MAP = {
    'front_of': ['front_of', 'facing', 'towards', 'opposite'],
    'behind': ['behind', 'opposite'],
    'left_of': ['left_of'],
    'right_of': ['right_of'],
    'belonging': ['belonging', 'adjacent', 'next_to', 'on_edge', 'on_side', 'connected_to'],
    'adjacent': ['adjacent', 'next_to', 'on_edge', 'on_side', 'connected_to', 'belonging'],
    'next_to': ['next_to', 'adjacent', 'on_edge', 'on_side', 'connected_to'],
    'inside': ['inside', 'surrounded_by'],
    'on_surface': ['on_surface', 'adjacent', 'belonging'],
    'between': ['between'],
    'at_corner': ['at_corner', 'near_corner'],
    'near_corner': ['near_corner', 'at_corner'],
    'at_end': ['at_end', 'on_edge'],
    'opposite': ['opposite', 'far_from', 'front_of', 'behind'],
    'facing': ['facing', 'front_of', 'towards'],
    'above': ['above'],
    'below': ['below'],
    'far_from': ['far_from', 'opposite'],
    'closest_to': ['closest_to'],
    'along': ['along', 'connected_to', 'on_edge'],
    'outside': ['outside'],
    'connected_to': ['connected_to', 'adjacent', 'next_to'],
    'on_side': ['on_side', 'on_edge', 'adjacent'],
    'towards': ['towards', 'facing', 'front_of'],
}


def load_desc_hypergraphs(desc_jsonl: str) -> dict:
    """加载描述超图"""
    desc_graphs = {}
    with open(desc_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            key = (str(data['scene_id']), str(data['object_id']))
            desc_graphs[key] = data
    return desc_graphs


def load_desc_hypergraphs_with_ann(desc_jsonl: str) -> dict:
    """加载描述超图，key=(scene_id, object_id, ann_id)"""
    desc_graphs = {}
    with open(desc_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            key = (
                str(data['scene_id']),
                str(data['object_id']),
                str(data.get('ann_id', 0)),
            )
            desc_graphs[key] = data
    return desc_graphs


def load_bbox_hypergraphs_jsonl(bbox_jsonl: str) -> dict:
    """加载 bbox-only 超图 jsonl，key=(scene_id, bbox_id, ann_id)"""
    bbox_graphs = {}
    with open(bbox_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            key = (
                str(data['scene_id']),
                str(data['bbox_id']),
                str(data.get('ann_id', 0)),
            )
            bbox_graphs[key] = data
    return bbox_graphs


def load_stage1_candidates_jsonl(candidates_jsonl: str) -> list:
    items = []
    with open(candidates_jsonl, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_bbox_hypergraph(hypergraph_path: str) -> dict:
    """加载单个 bbox 超图"""
    if not os.path.exists(hypergraph_path):
        return None
    with open(hypergraph_path, 'r') as f:
        return json.load(f)


def build_node_category_map(desc_graph: dict) -> dict:
    """构建节点ID到类别的映射"""
    node_categories = {}
    main_cat = desc_graph.get('hypergraph', {}).get('main_category', '')
    
    for node in desc_graph.get('hypergraph', {}).get('nodes', []):
        node_id = node['id']
        if node_id == 'main':
            node_categories[node_id] = main_cat
        else:
            node_categories[node_id] = node.get('category', '')
    return node_categories


def get_instances_by_category(bbox_nodes: list) -> dict:
    """按类别分组 bbox 中的实例"""
    by_cat = defaultdict(list)
    for node in bbox_nodes:
        by_cat[node['category']].append(node['id'])
    return dict(by_cat)


def check_edge_match(desc_edge: dict, bbox_edges: list, node_mapping: dict) -> float:
    """
    检查在特定节点映射下，描述边是否在 bbox 中有匹配
    
    Args:
        desc_edge: 描述超图的一条边
        bbox_edges: bbox 超图的所有边
        node_mapping: 描述节点到 bbox 实例的映射 {desc_node: bbox_instance}
    
    Returns:
        匹配分数 (0.0 ~ 1.0+)
    """
    rel_type = desc_edge['relation']
    from_node = desc_edge['from']
    to_node = desc_edge['to']
    
    # 获取映射后的 bbox 实例
    bbox_from = node_mapping.get(from_node)
    bbox_to = node_mapping.get(to_node)
    
    if not bbox_from or not bbox_to:
        return 0.0
    
    # 获取语义等价关系列表
    equivalent_rels = RELATION_SEMANTIC_MAP.get(rel_type, [rel_type])
    
    best_score = 0.0
    for bbox_edge in bbox_edges:
        # 检查节点是否匹配
        if bbox_edge['from'] == bbox_from and bbox_edge['to'] == bbox_to:
            # 检查关系是否匹配
            if bbox_edge['relation'] in equivalent_rels:
                edge_score = bbox_edge.get('score', 1.0)
                # 精确匹配给奖励
                if bbox_edge['relation'] == rel_type:
                    edge_score = min(1.0, edge_score * 1.2)
                best_score = max(best_score, edge_score)
    
    return best_score


def evaluate_mapping(desc_edges: list, bbox_edges: list, node_mapping: dict) -> float:
    """
    评估一个节点映射的质量
    
    Returns:
        总匹配分数
    """
    total_score = 0.0
    matched_count = 0
    
    for desc_edge in desc_edges:
        score = check_edge_match(desc_edge, bbox_edges, node_mapping)
        if score > 0:
            total_score += score
            matched_count += 1
    
    # 归一化
    if len(desc_edges) > 0:
        match_ratio = matched_count / len(desc_edges)
        avg_score = total_score / len(desc_edges)
        return match_ratio * 0.6 + avg_score * 0.4
    
    return 0.0


def find_best_mapping(desc_graph: dict, bbox_graph: dict) -> tuple:
    """
    找到描述超图和 bbox 超图之间的最佳节点映射
    
    Returns:
        (best_score, best_mapping)
    """
    desc_edges = desc_graph.get('hypergraph', {}).get('edges', [])
    bbox_edges = bbox_graph.get('hypergraph', {}).get('edges', [])
    bbox_nodes = bbox_graph.get('hypergraph', {}).get('nodes', [])
    
    if not desc_edges or not bbox_edges:
        return 0.0, {}
    
    # 获取描述超图的节点类别
    desc_node_cats = build_node_category_map(desc_graph)
    
    # 获取主类别
    main_cat = desc_graph.get('hypergraph', {}).get('main_category', '')
    if not main_cat:
        for node in desc_graph.get('hypergraph', {}).get('nodes', []):
            if node.get('is_main', False):
                main_cat = node.get('category', '')
                break
    
    # 按类别分组 bbox 实例
    bbox_by_cat = get_instances_by_category(bbox_nodes)
    
    # 按类别分组描述节点（不包括 main）
    desc_nodes_by_cat = defaultdict(list)
    for node_id, cat in desc_node_cats.items():
        if node_id != 'main':
            desc_nodes_by_cat[cat].append(node_id)
    
    # 检查 main 类别是否有实例
    if main_cat not in bbox_by_cat or not bbox_by_cat[main_cat]:
        return 0.0, {}
    
    # 检查每个非主类别是否有足够实例
    valid_categories = True
    for cat, desc_nodes in desc_nodes_by_cat.items():
        if cat not in bbox_by_cat or len(bbox_by_cat[cat]) < len(desc_nodes):
            valid_categories = False
            break
    
    if not valid_categories:
        return 0.0, {}
    
    # 枚举所有可能的映射
    # main 节点可以选择任意一个同类别 bbox 实例
    main_instances = bbox_by_cat[main_cat]
    
    best_score = 0.0
    best_mapping = {}
    
    # 对于每个可能的 main 实例
    for main_instance in main_instances:
        # 构建基础映射
        base_mapping = {'main': main_instance}
        
        # 对于其他类别，需要枚举所有排列
        # 收集需要映射的节点和可选的实例
        category_choices = []
        for cat, desc_nodes in desc_nodes_by_cat.items():
            bbox_insts = bbox_by_cat.get(cat, [])
            # 为每个描述节点选择一个 bbox 实例（排列）
            # 使用 product 枚举所有组合
            category_choices.append((cat, desc_nodes, bbox_insts))
        
        # 生成所有可能的映射组合
        # 如果类别太多或实例太多，限制枚举数量
        def generate_mappings(cat_idx, current_mapping):
            nonlocal best_score, best_mapping
            
            if cat_idx == len(category_choices):
                # 评估这个完整映射
                score = evaluate_mapping(desc_edges, bbox_edges, current_mapping)
                if score > best_score:
                    best_score = score
                    best_mapping = current_mapping.copy()
                return
            
            cat, desc_nodes, bbox_insts = category_choices[cat_idx]
            
            # 限制：如果实例太多，只取前5个
            if len(bbox_insts) > 5:
                bbox_insts = bbox_insts[:5]
            
            # 如果描述节点数 > bbox 实例数，无法映射
            if len(desc_nodes) > len(bbox_insts):
                return
            
            # 枚举所有排列
            from itertools import permutations
            for perm in permutations(bbox_insts, len(desc_nodes)):
                mapping = current_mapping.copy()
                for i, desc_node in enumerate(desc_nodes):
                    mapping[desc_node] = perm[i]
                generate_mappings(cat_idx + 1, mapping)
        
        # 从基础映射开始
        generate_mappings(0, base_mapping)
    
    return best_score, best_mapping


def compute_match_score(desc_graph: dict, bbox_graph: dict) -> float:
    """计算描述超图与 bbox 超图的匹配分数"""
    score, _ = find_best_mapping(desc_graph, bbox_graph)
    return score


def match_single_query(scene_id: str, object_id: str, candidates: list,
                       desc_graph: dict, bbox_hypergraphs_dir: str,
                       score_eps: float = 1e-9,
                       fusion_lambda: float = 1.0) -> tuple:
    """与 match_hypergraphs.match_single_query 相同融合逻辑，仅 compute_match_score 不同。"""
    if not candidates:
        return None, -1.0, {}, {}, False, False

    rank_by_id: Dict[str, int] = {}
    for i, cid in enumerate(candidates):
        cs = str(cid)
        if cs not in rank_by_id:
            rank_by_id[cs] = i

    ordered_unique: list = []
    seen: Set[str] = set()
    for cid in candidates:
        cs = str(cid)
        if cs not in seen:
            seen.add(cs)
            ordered_unique.append(cs)

    n_unique = len(ordered_unique)
    desc_main_cat = get_desc_main_category(desc_graph)
    all_scores: Dict[str, float] = {}
    cached_graphs: Dict = {}

    for bbox_id in ordered_unique:
        hypergraph_path = os.path.join(
            bbox_hypergraphs_dir,
            f"{scene_id}_bbox{bbox_id}_hypergraph.json"
        )
        bbox_graph = load_bbox_hypergraph(hypergraph_path)
        if bbox_graph is None:
            continue
        cached_graphs[bbox_id] = bbox_graph
        score = compute_match_score(desc_graph, bbox_graph)
        all_scores[bbox_id] = score

    if not all_scores:
        return None, -1.0, {}, {}, False, False

    best_hg = max(all_scores.values())
    hg_tops = [
        bid for bid, s in all_scores.items()
        if abs(s - best_hg) <= score_eps
    ]
    hg_tie = len(hg_tops) > 1

    fl = max(0.0, min(1.0, float(fusion_lambda)))
    all_fused_scores: Dict[str, float] = {}
    for bid in all_scores.keys():
        r = rank_by_id.get(bid, 10**9)
        rp = rank_prior_from_index(r, n_unique)
        hg = all_scores[bid]
        all_fused_scores[bid] = fl * hg + (1.0 - fl) * rp

    best_fused = max(all_fused_scores.values())
    fused_tops = [
        bid for bid, s in all_fused_scores.items()
        if abs(s - best_fused) <= score_eps
    ]
    tie_break_used = len(fused_tops) > 1

    def pick_key(bid: str):
        r = rank_by_id.get(bid, 10**9)
        pts = max_main_category_num_points(cached_graphs[bid], desc_main_cat)
        return (r, -pts, bid)

    best_bbox_id = min(fused_tops, key=pick_key)
    return (
        best_bbox_id,
        best_fused,
        all_scores,
        all_fused_scores,
        tie_break_used,
        hg_tie,
    )


def evaluate_matching(evaluation_log: str, desc_hypergraphs_jsonl: str,
                     bbox_hypergraphs_dir: str, val_json: str,
                     output_json: str, score_eps: float = 1e-9,
                     fusion_lambda: float = 1.0):
    """评估匹配效果；参数与 match_hypergraphs.evaluate_matching 对齐。"""
    
    print("Loading data...")
    desc_graphs = load_desc_hypergraphs(desc_hypergraphs_jsonl)
    
    with open(evaluation_log, 'r') as f:
        eval_data = json.load(f)
    
    print(f"Loaded {len(desc_graphs)} description graphs")
    print(f"Loaded {len(eval_data)} evaluation queries")
    print(f"fusion_lambda={fusion_lambda}")
    
    stats = {
        'fusion_lambda': fusion_lambda,
        'total': 0,
        'matched': 0,
        'correct_top1': 0,
        'correct_top5': 0,
        'correct_top10': 0,
        'queries_with_score_tie': 0,
        'queries_with_fused_tie': 0,
        'tie_break_used': 0,
        'correct_top1_when_tie_break': 0,
    }
    
    results = []
    
    for query in tqdm(eval_data, desc="Matching queries"):
        scene_id = query['scene_id']
        object_id = str(query['object_id'])
        ann_id = query.get('ann_id', 0)
        candidates_30 = query.get('candidates_30', [])
        
        if not candidates_30:
            continue
        
        stats['total'] += 1
        
        desc_key = (scene_id, object_id)
        desc_graph = desc_graphs.get(desc_key)
        
        if desc_graph is None:
            results.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'status': 'no_desc_graph',
                'predicted_bbox_id': None,
                'match_score': 0.0
            })
            continue
        
        (
            best_bbox_id,
            best_fused_score,
            all_scores,
            all_fused_scores,
            tie_break_used,
            hg_tie,
        ) = match_single_query(
            scene_id, object_id, candidates_30,
            desc_graph, bbox_hypergraphs_dir,
            score_eps=score_eps,
            fusion_lambda=fusion_lambda,
        )
        
        if best_bbox_id is not None:
            stats['matched'] += 1

        if all_scores:
            max_s = max(all_scores.values())
            tied_ids = [k for k, v in all_scores.items() if abs(v - max_s) <= score_eps]
            if len(tied_ids) > 1:
                stats['queries_with_score_tie'] += 1

        if all_fused_scores:
            max_f = max(all_fused_scores.values())
            fused_tied = [k for k, v in all_fused_scores.items() if abs(v - max_f) <= score_eps]
            if len(fused_tied) > 1:
                stats['queries_with_fused_tie'] += 1

        if tie_break_used:
            stats['tie_break_used'] += 1
        
        is_correct = (str(best_bbox_id) == object_id)
        
        if is_correct:
            stats['correct_top1'] += 1
        if tie_break_used and is_correct:
            stats['correct_top1_when_tie_break'] += 1
        
        sorted_fused = sorted(all_fused_scores.items(), key=lambda x: -x[1])
        top5_ids = [x[0] for x in sorted_fused[:5]]
        top10_ids = [x[0] for x in sorted_fused[:10]]
        
        if str(object_id) in top5_ids:
            stats['correct_top5'] += 1
        if str(object_id) in top10_ids:
            stats['correct_top10'] += 1
        
        hg_chosen = all_scores.get(best_bbox_id) if best_bbox_id is not None else None
        results.append({
            'scene_id': scene_id,
            'object_id': object_id,
            'ann_id': ann_id,
            'status': 'matched' if best_bbox_id else 'failed',
            'predicted_bbox_id': best_bbox_id,
            'fused_score': best_fused_score,
            'hypergraph_score': hg_chosen,
            'fusion_lambda': fusion_lambda,
            'is_correct': is_correct,
            'tie_break_used': tie_break_used,
            'hg_tie': hg_tie,
            'candidates_30': candidates_30,
            'all_scores': all_scores,
            'all_fused_scores': all_fused_scores,
        })
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("匹配结果统计:")
    print(f"  总查询数: {stats['total']}")
    print(f"  匹配成功: {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    print(f"  Top-1 正确: {stats['correct_top1']} ({stats['correct_top1']/stats['total']*100:.1f}%)")
    print(f"  Top-5 包含: {stats['correct_top5']} ({stats['correct_top5']/stats['total']*100:.1f}%)")
    print(f"  Top-10 包含: {stats['correct_top10']} ({stats['correct_top10']/stats['total']*100:.1f}%)")
    print(f"  纯超图分并列: {stats['queries_with_score_tie']}")
    print(f"  融合分仍并列: {stats['queries_with_fused_tie']}")
    tb = stats['tie_break_used']
    print(f"  融合分并列后用 rank/num_points 打破: {tb} ({tb/stats['total']*100:.1f}%)")
    if tb > 0:
        print(f"  其中 Top-1 正确: {stats['correct_top1_when_tie_break']} "
              f"({stats['correct_top1_when_tie_break']/tb*100:.1f}%)")
    print(f"\n结果保存至: {output_json}")


def evaluate_matching_pipeline2(
    candidates_jsonl: str,
    desc_hypergraphs_jsonl: str,
    bbox_hypergraphs_jsonl: str,
    output_json: str,
    score_eps: float = 1e-9,
    fusion_lambda: float = 1.0,
):
    """直接使用 pipeline2 的 stage1 candidates + desc jsonl + bbox jsonl。"""
    print("Loading pipeline2 data...")
    desc_graphs = load_desc_hypergraphs_with_ann(desc_hypergraphs_jsonl)
    bbox_graphs = load_bbox_hypergraphs_jsonl(bbox_hypergraphs_jsonl)
    eval_data = load_stage1_candidates_jsonl(candidates_jsonl)

    print(f"Loaded {len(desc_graphs)} description graphs")
    print(f"Loaded {len(bbox_graphs)} bbox hypergraphs")
    print(f"Loaded {len(eval_data)} candidate queries")
    print(f"fusion_lambda={fusion_lambda}")

    stats = {
        'fusion_lambda': fusion_lambda,
        'total': 0,
        'matched': 0,
        'correct_top1': 0,
        'correct_top5': 0,
        'correct_top10': 0,
        'correct_top20': 0,
        'queries_with_score_tie': 0,
        'queries_with_fused_tie': 0,
        'tie_break_used': 0,
        'correct_top1_when_tie_break': 0,
    }
    results = []

    for query in tqdm(eval_data, desc="Matching pipeline2 queries"):
        scene_id = str(query['scene_id'])
        object_id = str(query['object_id'])
        ann_id = str(query.get('ann_id', 0))
        candidates = [str(x) for x in query.get('candidates', [])]
        if not candidates:
            continue
        stats['total'] += 1

        desc_graph = desc_graphs.get((scene_id, object_id, ann_id))
        if desc_graph is None:
            results.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'status': 'no_desc_graph',
                'predicted_bbox_id': None,
                'fused_score': 0.0,
            })
            continue

        rank_by_id: Dict[str, int] = {}
        for i, cid in enumerate(candidates):
            if cid not in rank_by_id:
                rank_by_id[cid] = i

        ordered_unique = []
        seen = set()
        for cid in candidates:
            if cid not in seen:
                seen.add(cid)
                ordered_unique.append(cid)

        n_unique = len(ordered_unique)
        desc_main_cat = get_desc_main_category(desc_graph)
        all_scores: Dict[str, float] = {}
        cached_graphs: Dict[str, dict] = {}

        for bbox_id in ordered_unique:
            bbox_graph = bbox_graphs.get((scene_id, bbox_id, ann_id))
            if bbox_graph is None:
                continue
            cached_graphs[bbox_id] = bbox_graph
            all_scores[bbox_id] = compute_match_score(desc_graph, bbox_graph)

        if not all_scores:
            results.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'status': 'failed',
                'predicted_bbox_id': None,
                'fused_score': 0.0,
            })
            continue

        stats['matched'] += 1
        max_s = max(all_scores.values())
        tied_ids = [k for k, v in all_scores.items() if abs(v - max_s) <= score_eps]
        if len(tied_ids) > 1:
            stats['queries_with_score_tie'] += 1

        all_fused_scores: Dict[str, float] = {}
        for bid in all_scores.keys():
            r = rank_by_id.get(bid, 10**9)
            rp = rank_prior_from_index(r, n_unique)
            hg = all_scores[bid]
            all_fused_scores[bid] = fusion_lambda * hg + (1.0 - fusion_lambda) * rp

        max_f = max(all_fused_scores.values())
        fused_tied = [k for k, v in all_fused_scores.items() if abs(v - max_f) <= score_eps]
        tie_break_used = len(fused_tied) > 1
        if tie_break_used:
            stats['queries_with_fused_tie'] += 1
            stats['tie_break_used'] += 1

        # No GT-aware tie-break: only use visible features / stable IDs.
        def pick_key(bid: str):
            r = rank_by_id.get(bid, 10**9)
            pts = max_main_category_num_points(cached_graphs[bid], desc_main_cat)
            return (r, -pts, bid)

        best_bbox_id = min(fused_tied, key=pick_key)
        best_fused_score = all_fused_scores[best_bbox_id]

        sorted_fused = sorted(
            all_fused_scores.items(),
            key=lambda x: (-x[1], pick_key(x[0])),
        )
        top5_ids = [x[0] for x in sorted_fused[:5]]
        top10_ids = [x[0] for x in sorted_fused[:10]]
        top20_ids = [x[0] for x in sorted_fused[:20]]
        is_correct = (best_bbox_id == object_id)

        if is_correct:
            stats['correct_top1'] += 1
        if tie_break_used and is_correct:
            stats['correct_top1_when_tie_break'] += 1
        if object_id in top5_ids:
            stats['correct_top5'] += 1
        if object_id in top10_ids:
            stats['correct_top10'] += 1
        if object_id in top20_ids:
            stats['correct_top20'] += 1

        results.append({
            'scene_id': scene_id,
            'object_id': object_id,
            'ann_id': ann_id,
            'status': 'matched',
            'predicted_bbox_id': best_bbox_id,
            'fused_score': best_fused_score,
            'hypergraph_score': all_scores.get(best_bbox_id),
            'fusion_lambda': fusion_lambda,
            'is_correct': is_correct,
            'tie_break_used': tie_break_used,
            'candidates': candidates,
            'top5_ids': top5_ids,
            'top10_ids': top10_ids,
            'top20_ids': top20_ids,
            'all_scores': all_scores,
            'all_fused_scores': all_fused_scores,
        })

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("pipeline2 输入结果统计:")
    print(f"  总查询数: {stats['total']}")
    print(f"  匹配成功: {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    print(f"  Top-1 正确: {stats['correct_top1']} ({stats['correct_top1']/stats['total']*100:.1f}%)")
    print(f"  Top-5 包含: {stats['correct_top5']} ({stats['correct_top5']/stats['total']*100:.1f}%)")
    print(f"  Top-10 包含: {stats['correct_top10']} ({stats['correct_top10']/stats['total']*100:.1f}%)")
    print(f"  Top-20 包含: {stats['correct_top20']} ({stats['correct_top20']/stats['total']*100:.1f}%)")
    print(f"  纯超图分并列: {stats['queries_with_score_tie']}")
    print(f"  融合分仍并列: {stats['queries_with_fused_tie']}")
    tb = stats['tie_break_used']
    print(f"  融合分并列后用 rank/num_points 打破: {tb} ({tb/stats['total']*100:.1f}%)")
    if tb > 0:
        print(f"  其中 Top-1 正确: {stats['correct_top1_when_tie_break']} "
              f"({stats['correct_top1_when_tie_break']/tb*100:.1f}%)")
    print(f"\n结果保存至: {output_json}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation_log', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/evaluation_results_log.json")
    parser.add_argument('--desc_hypergraphs', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_desc_hypergraphs_dedup.jsonl")
    parser.add_argument('--bbox_hypergraphs_dir', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/bbox_hypergraphs_v2")
    parser.add_argument('--val_json', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_val.json")
    parser.add_argument('--output', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/matching_results_v2.json")
    parser.add_argument('--fusion_lambda', type=float, default=1.0,
                       help='与 match_hypergraphs.py 一致')
    parser.add_argument('--pipeline2', action='store_true',
                       help='直接使用 pipeline2 的 stage1 candidates/desc jsonl/bbox jsonl 输入')
    
    args = parser.parse_args()
    if args.pipeline2:
        evaluate_matching_pipeline2(
            args.evaluation_log,
            args.desc_hypergraphs,
            args.bbox_hypergraphs_dir,
            args.output,
            fusion_lambda=args.fusion_lambda,
        )
    else:
        evaluate_matching(
            args.evaluation_log,
            args.desc_hypergraphs,
            args.bbox_hypergraphs_dir,
            args.val_json,
            args.output,
            fusion_lambda=args.fusion_lambda,
        )


if __name__ == "__main__":
    main()

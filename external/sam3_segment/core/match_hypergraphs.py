"""
超图匹配：从 candidates_30 中筛选 top1
并与 CityRefer_val.json 对比计算准确率
"""

import json
import os
from collections import defaultdict
from typing import Dict, Optional, Set

from tqdm import tqdm


# 关系语义映射表：描述超图关系 -> bbox 超图等价关系列表
RELATION_SEMANTIC_MAP = {
    # 方向类
    'front_of': ['front_of', 'facing', 'towards', 'opposite'],
    'behind': ['behind', 'opposite'],
    'left_of': ['left_of'],
    'right_of': ['right_of'],
    
    # 拓扑/相邻类
    'belonging': ['belonging', 'adjacent', 'next_to', 'on_edge', 'on_side', 'connected_to'],
    'adjacent': ['adjacent', 'next_to', 'on_edge', 'on_side', 'connected_to', 'belonging'],
    'next_to': ['next_to', 'adjacent', 'on_edge', 'on_side', 'connected_to'],
    
    # 包含类
    'inside': ['inside', 'surrounded_by'],
    'on_surface': ['on_surface', 'adjacent', 'belonging'],
    
    # 位置类
    'between': ['between'],
    'at_corner': ['at_corner', 'near_corner'],
    'near_corner': ['near_corner', 'at_corner'],
    'at_end': ['at_end', 'on_edge'],
    
    # 相对类
    'opposite': ['opposite', 'far_from', 'front_of', 'behind'],
    'facing': ['facing', 'front_of', 'towards'],
    
    # 序数类
    'nth_from_left': ['nth_from_left'],
    'nth_from_right': ['nth_from_right'],
    'nth_from_front': ['nth_from_front'],
    'nth_from_back': ['nth_from_back'],
    
    # 其他
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
    """加载描述超图，按 scene_id + object_id 索引（统一为字符串）"""
    desc_graphs = {}
    with open(desc_jsonl, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # 统一 key 格式为 (str, str)
            key = (str(data['scene_id']), str(data['object_id']))
            desc_graphs[key] = data
    return desc_graphs


def load_bbox_hypergraph(hypergraph_path: str) -> dict:
    """加载单个 bbox 超图"""
    if not os.path.exists(hypergraph_path):
        return None
    with open(hypergraph_path, 'r') as f:
        return json.load(f)


def build_node_category_map(desc_graph: dict) -> dict:
    """构建节点ID到类别的映射"""
    node_categories = {}
    for node in desc_graph.get('hypergraph', {}).get('nodes', []):
        node_categories[node['id']] = node.get('category', '')
    return node_categories


def build_bbox_edge_index(bbox_edges: list) -> dict:
    """
    构建 bbox 边的索引：((from_cat, to_cat, rel_type)) -> list of edges
    """
    index = defaultdict(list)
    for edge in bbox_edges:
        from_cat = edge['from'].split('_')[0]
        to_cat = edge['to'].split('_')[0]
        rel_type = edge['relation']
        index[(from_cat, to_cat, rel_type)].append(edge)
    return index


def greedy_node_mapping(desc_graph: dict, bbox_graph: dict, 
                        main_cat: str, main_instance: str) -> dict:
    """
    贪心算法建立节点映射（优化版）
    """
    desc_edges = desc_graph.get('hypergraph', {}).get('edges', [])
    bbox_edges = bbox_graph.get('hypergraph', {}).get('edges', [])
    bbox_nodes = bbox_graph.get('hypergraph', {}).get('nodes', [])
    
    if not desc_edges:
        return {'main': main_instance}
    
    # 构建节点类别映射
    node_to_cat = build_node_category_map(desc_graph)
    
    # 按类别分组 bbox 实例
    bbox_by_cat = defaultdict(list)
    for node in bbox_nodes:
        bbox_by_cat[node['category']].append(node['id'])
    
    # 初始化映射
    mapping = {'main': main_instance}
    used_bbox_instances = {main_instance}
    
    # 收集所有非 main 的描述节点，按类别分组
    desc_nodes_by_cat = defaultdict(list)
    for node_id, cat in node_to_cat.items():
        if node_id != 'main':
            desc_nodes_by_cat[cat].append(node_id)
    
    # 预构建 bbox 边索引: (from, to) -> list of (relation, score)
    bbox_edge_index = {}
    for edge in bbox_edges:
        key = (edge['from'], edge['to'])
        if key not in bbox_edge_index:
            bbox_edge_index[key] = []
        bbox_edge_index[key].append((edge['relation'], edge.get('score', 1.0)))
    
    # 为每个类别贪心分配实例
    for cat, desc_nodes in desc_nodes_by_cat.items():
        available_instances = [i for i in bbox_by_cat.get(cat, [])
                               if i not in used_bbox_instances]

        # 宽松匹配：即使实例不够也尝试部分匹配
        # 按得分排序，优先匹配能带来高分的节点
        node_scores = []
        for desc_node in desc_nodes:
            best_score = -1
            best_inst = None
            for bbox_inst in available_instances:
                score = 0
                for edge in desc_edges:
                    if edge['from'] == desc_node or edge['to'] == desc_node:
                        mapped_from = mapping.get(edge['from'], bbox_inst if edge['from'] == desc_node else None)
                        mapped_to = mapping.get(edge['to'], bbox_inst if edge['to'] == desc_node else None)
                        if mapped_from and mapped_to:
                            key = (mapped_from, mapped_to)
                            if key in bbox_edge_index:
                                score += max(s for _, s in bbox_edge_index[key])
                if score > best_score:
                    best_score = score
                    best_inst = bbox_inst
            node_scores.append((best_score, desc_node, best_inst))

        # 按分数降序排序，优先匹配高分节点
        node_scores.sort(reverse=True)

        for score, desc_node, best_inst in node_scores:
            if best_inst and best_inst in available_instances:
                mapping[desc_node] = best_inst
                used_bbox_instances.add(best_inst)
                available_instances.remove(best_inst)
            elif available_instances:
                # 如果最佳实例已被使用，退而求其次选择可用实例
                fallback_inst = available_instances[0]
                mapping[desc_node] = fallback_inst
                used_bbox_instances.add(fallback_inst)
                available_instances.remove(fallback_inst)
            # 如果没有可用实例，跳过此节点（部分匹配）

    return mapping


def compute_match_score_with_main(desc_graph: dict, bbox_graph: dict, 
                                   main_instance: str, main_cat: str) -> float:
    """
    计算匹配分数（使用指定的 main 实例）
    
    使用贪心算法建立节点映射，然后评估匹配质量
    """
    desc_edges = desc_graph.get('hypergraph', {}).get('edges', [])
    bbox_edges = bbox_graph.get('hypergraph', {}).get('edges', [])
    
    if not desc_edges or not bbox_edges:
        return 0.0
    
    # 使用贪心算法建立节点映射
    mapping = greedy_node_mapping(desc_graph, bbox_graph, main_cat, main_instance)
    
    if not mapping:
        return 0.0
    
    # 评估这个映射下的边匹配
    total_score = 0.0
    matched_count = 0
    
    for desc_edge in desc_edges:
        rel_type = desc_edge['relation']
        from_node = desc_edge['from']
        to_node = desc_edge['to']
        
        # 获取映射后的实例
        bbox_from = mapping.get(from_node)
        bbox_to = mapping.get(to_node)
        
        if not bbox_from or not bbox_to:
            continue
        
        # 获取语义等价关系
        equivalent_rels = RELATION_SEMANTIC_MAP.get(rel_type, [rel_type])
        
        best_match_score = 0.0
        for bbox_edge in bbox_edges:
            if bbox_edge['from'] == bbox_from and bbox_edge['to'] == bbox_to:
                if bbox_edge['relation'] in equivalent_rels:
                    edge_score = bbox_edge.get('score', 1.0)
                    if bbox_edge['relation'] == rel_type:
                        edge_score = min(1.0, edge_score * 1.2)
                    best_match_score = max(best_match_score, edge_score)
        
        if best_match_score > 0:
            total_score += best_match_score
            matched_count += 1
    
    # 归一化
    if len(desc_edges) > 0:
        match_ratio = matched_count / len(desc_edges)
        avg_score = total_score / len(desc_edges)
        return match_ratio * 0.6 + avg_score * 0.4
    
    return 0.0


def get_desc_main_category(desc_graph: dict) -> str:
    """描述超图主物体类别（与 compute_match_score 一致）"""
    desc_main_cat = desc_graph.get('hypergraph', {}).get('main_category', '')
    if not desc_main_cat:
        nodes = desc_graph.get('hypergraph', {}).get('nodes', [])
        for node in nodes:
            if node.get('is_main', False):
                desc_main_cat = node.get('category', '')
                break
    return desc_main_cat or ''


def max_main_category_num_points(bbox_graph: dict, main_cat: str) -> int:
    """主类别在 bbox 超图节点中的最大 num_points（用于同分同 rank 时第三关键字）"""
    if not main_cat:
        return 0
    best = 0
    for node in bbox_graph.get('hypergraph', {}).get('nodes', []):
        if node.get('category') == main_cat:
            best = max(best, int(node.get('num_points', 0) or 0))
    return best


def compute_match_score(desc_graph: dict, bbox_graph: dict) -> float:
    """
    计算描述超图与 bbox 超图的匹配分数
    
    策略：尝试把 bbox 中每个同类实例都当作 main 节点，取最高分
    """
    desc_main_cat = get_desc_main_category(desc_graph)
    if not desc_main_cat:
        return 0.0
    
    bbox_edges = bbox_graph.get('hypergraph', {}).get('edges', [])
    
    # 在 bbox 超图中找到所有主类别的实例
    main_instances = set()
    for edge in bbox_edges:
        from_cat = edge['from'].split('_')[0]
        to_cat = edge['to'].split('_')[0]
        if from_cat == desc_main_cat:
            main_instances.add(edge['from'])
        if to_cat == desc_main_cat:
            main_instances.add(edge['to'])
    
    if not main_instances:
        return 0.0
    
    # 尝试每个主实例，取最高分
    best_score = 0.0
    for main_instance in main_instances:
        score = compute_match_score_with_main(
            desc_graph, bbox_graph, main_instance, desc_main_cat
        )
        best_score = max(best_score, score)
    
    return best_score


def rank_prior_from_index(rank: int, n_unique: int) -> float:
    """
    将首次出现下标 rank 映射到 [0,1]，rank=0 最优 -> 1.0。
    n_unique 为去重后的候选个数。
    """
    if n_unique <= 1:
        return 1.0
    return (n_unique - 1 - rank) / float(n_unique - 1)


def match_single_query(scene_id: str, object_id: str, candidates: list,
                       desc_graph: dict, bbox_hypergraphs_dir: str,
                       score_eps: float = 1e-9,
                       fusion_lambda: float = 1.0,
                       fusion_on_tie_only: bool = True,
                       near_tie_delta: float = 0.0) -> tuple:
    """
    为单个查询匹配最佳 bbox。

    - fusion_lambda=1.0：仅用超图分（与旧版一致）。
    - fusion_lambda<1.0：final = λ * hg_score + (1-λ) * rank_prior。
    - fusion_on_tie_only=True：仅在超图分并列/近并列时启用融合，避免全量重排掉点。

    若融合分仍并列，则按更小 rank、更大主类 num_points、更小 id 打破平局。

    Returns:
        (best_bbox_id, best_fused_score, all_scores, all_fused_scores,
         tie_break_used, hg_tie)
        all_scores: 纯超图分；all_fused_scores: 融合分（λ=1 时与超图分相同）。
        tie_break_used: 融合分仍存在并列而启用次级规则。
        hg_tie: 纯超图分存在并列（用于诊断）。
    """
    if not candidates:
        return None, -1.0, {}, {}, False, False

    # 首次出现下标 = stage2 排名（0 最优）
    rank_by_id: Dict[str, int] = {}
    for i, cid in enumerate(candidates):
        cs = str(cid)
        if cs not in rank_by_id:
            rank_by_id[cs] = i

    # 去重且保持首次出现顺序，避免重复加载同一 bbox
    ordered_unique: list[str] = []
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

    # 超图分并列（诊断）
    best_hg = max(all_scores.values())
    hg_tops = [
        bid for bid, s in all_scores.items()
        if abs(s - best_hg) <= score_eps
    ]
    hg_tie = len(hg_tops) > 1

    # 融合分（可只对超图分并列/近并列集合做 gated 融合）
    fl = float(fusion_lambda)
    fl = max(0.0, min(1.0, fl))
    all_fused_scores: Dict[str, float] = dict(all_scores)

    # near tie: 距离 best_hg 不超过 near_tie_delta 视为近并列
    gated_ids = [
        bid for bid, s in all_scores.items()
        if abs(best_hg - s) <= max(score_eps, near_tie_delta)
    ]

    apply_global = not fusion_on_tie_only
    if apply_global:
        target_ids = list(all_scores.keys())
    else:
        # 仅在并列/近并列子集上融合，其他候选保持原超图分
        target_ids = gated_ids if len(gated_ids) > 1 else []

    for bid in target_ids:
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


def load_val_ground_truth(val_json: str) -> dict:
    """加载 val.json 的 ground truth"""
    gt = {}
    with open(val_json, 'r') as f:
        data = json.load(f)
    
    for item in data:
        key = (item['scene_id'], str(item['object_id']))
        if key not in gt:
            gt[key] = []
        gt[key].append({
            'ann_id': item.get('ann_id', 0),
            'bbox': item.get('bbox', []),
            'description': item.get('description', '')
        })
    
    return gt


def evaluate_matching(evaluation_log: str, desc_hypergraphs_jsonl: str,
                     bbox_hypergraphs_dir: str, val_json: str,
                     output_json: str,
                     summary_json: Optional[str] = None,
                     score_eps: float = 1e-6,
                     fusion_lambda: float = 1.0,
                     fusion_on_tie_only: bool = True,
                     near_tie_delta: float = 0.0,
                     save_results: bool = True):
    """评估匹配效果；fusion_lambda 见 match_single_query。"""
    
    # 加载数据
    print("Loading data...")
    desc_graphs = load_desc_hypergraphs(desc_hypergraphs_jsonl)
    
    with open(evaluation_log, 'r') as f:
        eval_data = json.load(f)
    
    gt_data = load_val_ground_truth(val_json)
    
    print(f"Loaded {len(desc_graphs)} description graphs")
    print(f"Loaded {len(eval_data)} evaluation queries")
    print(f"Loaded {len(gt_data)} ground truth entries")
    print(f"fusion_lambda={fusion_lambda} (1.0=纯超图分)")
    print(f"fusion_on_tie_only={fusion_on_tie_only}, near_tie_delta={near_tie_delta}")

    # 匹配统计
    stats = {
        'fusion_lambda': fusion_lambda,
        'fusion_on_tie_only': fusion_on_tie_only,
        'near_tie_delta': near_tie_delta,
        'total': 0,
        'matched': 0,
        'correct_top1': 0,
        'correct_top5': 0,
        'correct_top10': 0,
        'queries_with_score_tie': 0,  # 纯超图分并列
        'queries_with_fused_tie': 0,  # 融合分仍并列
        'tie_vote_correct': 0,  # 超图最高分并列集合中含 GT
        'tie_break_used': 0,  # 融合分并列时用 rank/num_points 打破
        'correct_top1_when_tie_break': 0,
    }
    
    results = [] if save_results else None
    
    for query in tqdm(eval_data, desc="Matching queries"):
        scene_id = query['scene_id']
        object_id = str(query['object_id'])
        ann_id = query.get('ann_id', 0)
        candidates_30 = query.get('candidates_30', [])
        
        if not candidates_30:
            continue
        
        stats['total'] += 1
        
        # 获取描述超图
        desc_key = (scene_id, object_id)
        desc_graph = desc_graphs.get(desc_key)
        
        if desc_graph is None:
            if save_results:
                results.append({
                    'scene_id': scene_id,
                    'object_id': object_id,
                    'ann_id': ann_id,
                    'status': 'no_desc_graph',
                    'predicted_bbox_id': None,
                    'match_score': 0.0
                })
            continue
        
        # 执行匹配（融合分优先，并列再按 rank / num_points）
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
            fusion_on_tie_only=fusion_on_tie_only,
            near_tie_delta=near_tie_delta,
        )
        
        if best_bbox_id is not None:
            stats['matched'] += 1

        if all_scores:
            max_s = max(all_scores.values())
            tied_ids = [k for k, v in all_scores.items() if abs(v - max_s) <= score_eps]
            if len(tied_ids) > 1:
                stats['queries_with_score_tie'] += 1
                if str(object_id) in map(str, tied_ids):
                    stats['tie_vote_correct'] += 1

        if all_fused_scores:
            max_f = max(all_fused_scores.values())
            fused_tied = [k for k, v in all_fused_scores.items() if abs(v - max_f) <= score_eps]
            if len(fused_tied) > 1:
                stats['queries_with_fused_tie'] += 1

        if tie_break_used:
            stats['tie_break_used'] += 1
        
        # 检查是否正确
        is_correct = (str(best_bbox_id) == object_id)
        
        if is_correct:
            stats['correct_top1'] += 1
        if tie_break_used and is_correct:
            stats['correct_top1_when_tie_break'] += 1
        
        # Top5/Top10 按融合分排序（λ=1 时等价于超图分）
        sorted_fused = sorted(all_fused_scores.items(), key=lambda x: -x[1])
        top5_ids = [x[0] for x in sorted_fused[:5]]
        top10_ids = [x[0] for x in sorted_fused[:10]]
        
        if str(object_id) in top5_ids:
            stats['correct_top5'] += 1
        if str(object_id) in top10_ids:
            stats['correct_top10'] += 1
        
        hg_chosen = all_scores.get(best_bbox_id) if best_bbox_id is not None else None
        if save_results:
            results.append({
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'status': 'matched' if best_bbox_id else 'failed',
                'predicted_bbox_id': best_bbox_id,
                'fused_score': best_fused_score,
                'hypergraph_score': hg_chosen,
                'fusion_lambda': fusion_lambda,
                'fusion_on_tie_only': fusion_on_tie_only,
                'near_tie_delta': near_tie_delta,
                'is_correct': is_correct,
                'tie_break_used': tie_break_used,
                'hg_tie': hg_tie,
                'candidates_30': candidates_30,
                'all_scores': all_scores,
                'all_fused_scores': all_fused_scores,
            })
    
    # 保存结果（可选：仅 summary 以加速）
    if save_results:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("匹配结果统计:")
    print(f"  总查询数: {stats['total']}")
    print(f"  匹配成功: {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    print(f"  Top-1 正确: {stats['correct_top1']} ({stats['correct_top1']/stats['total']*100:.1f}%)")
    print(f"  Top-5 包含: {stats['correct_top5']} ({stats['correct_top5']/stats['total']*100:.1f}%)")
    print(f"  Top-10 包含: {stats['correct_top10']} ({stats['correct_top10']/stats['total']*100:.1f}%)")
    print(f"  纯超图分并列: {stats['queries_with_score_tie']} "
          f"({stats['queries_with_score_tie']/stats['total']*100:.1f}%)")
    print(f"  融合分仍并列: {stats['queries_with_fused_tie']} "
          f"({stats['queries_with_fused_tie']/stats['total']*100:.1f}%)")
    print(f"  超图最高分并列集合中含 GT: {stats['tie_vote_correct']}")
    tb = stats['tie_break_used']
    print(f"  融合分并列后用 rank/num_points 打破: {tb} ({tb/stats['total']*100:.1f}%)")
    if tb > 0:
        print(f"  其中 Top-1 正确: {stats['correct_top1_when_tie_break']} "
              f"({stats['correct_top1_when_tie_break']/tb*100:.1f}%)")
    if save_results:
        print(f"\n结果保存至: {output_json}")

    if summary_json:
        summary = {
            'stats': stats,
            'bbox_hypergraphs_dir': bbox_hypergraphs_dir,
            'desc_hypergraphs': desc_hypergraphs_jsonl,
            'evaluation_log': evaluation_log,
            'val_json': val_json,
        }
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"摘要保存至: {summary_json}")


def run_fusion_sweep(
    evaluation_log: str,
    desc_hypergraphs_jsonl: str,
    bbox_hypergraphs_dir: str,
    val_json: str,
    output_prefix: str,
    summary_path: Optional[str],
    lambdas: list,
    score_eps: float = 1e-6,
) -> dict:
    """对多个 fusion_lambda 依次评估，写入 fusion_sweep.json。"""
    sweep = {'lambdas': lambdas, 'runs': []}
    n = len(lambdas)
    for i, lam in enumerate(lambdas):
        out_json = output_prefix.replace('.json', f'_lam{lam}.json')
        sum_json: Optional[str] = None
        if summary_path:
            sum_json = summary_path if i == n - 1 else summary_path.replace('.json', f'_lam{lam}.json')
        evaluate_matching(
            evaluation_log,
            desc_hypergraphs_jsonl,
            bbox_hypergraphs_dir,
            val_json,
            out_json,
            summary_json=sum_json,
            score_eps=score_eps,
            fusion_lambda=lam,
            fusion_on_tie_only=True,
            near_tie_delta=0.0,
        )
        if sum_json and os.path.isfile(sum_json):
            with open(sum_json) as f:
                sweep['runs'].append({'lambda': lam, 'summary': json.load(f)})
    base = summary_path or output_prefix
    sweep_path = base.replace('.json', '_fusion_sweep.json') if base else None
    if sweep_path:
        with open(sweep_path, 'w') as f:
            json.dump(sweep, f, indent=2)
        print(f"\n融合扫参汇总: {sweep_path}")
    return sweep


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
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/matching_results.json")
    parser.add_argument('--summary', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/matching_eval_summary.json",
                       help='平局与统计摘要 JSON；传空字符串可关闭')
    parser.add_argument('--fusion_lambda', type=float, default=1.0,
                       help='超图分权重；1.0 为纯超图，<1 时混入 stage2 排名先验')
    parser.add_argument('--fusion_on_tie_only', type=int, default=1,
                       help='1: 仅超图分并列/近并列时融合；0: 全量融合')
    parser.add_argument('--near_tie_delta', type=float, default=0.0,
                       help='超图分近并列阈值，>0 时扩大融合触发集合')
    parser.add_argument('--save_results', type=int, default=1,
                       help='1: 保存每个查询的匹配明细 json；0: 仅保存 summary（更快）')
    parser.add_argument('--fusion_sweep', type=str, default=None,
                       help='逗号分隔多个 λ，如 0.5,0.6,0.7,1.0；设置则对每个 λ 跑一遍并写 fusion_sweep.json')
    
    args = parser.parse_args()
    
    if args.fusion_sweep:
        parts = [float(x.strip()) for x in args.fusion_sweep.split(',') if x.strip()]
        run_fusion_sweep(
            args.evaluation_log,
            args.desc_hypergraphs,
            args.bbox_hypergraphs_dir,
            args.val_json,
            args.output,
            args.summary if args.summary else None,
            parts,
        )
    else:
        evaluate_matching(
            args.evaluation_log,
            args.desc_hypergraphs,
            args.bbox_hypergraphs_dir,
            args.val_json,
            args.output,
            summary_json=args.summary if args.summary else None,
            fusion_lambda=args.fusion_lambda,
            fusion_on_tie_only=bool(args.fusion_on_tie_only),
            near_tie_delta=args.near_tie_delta,
            save_results=bool(args.save_results),
        )


if __name__ == "__main__":
    main()

"""
超图匹配算法
计算描述超图与实例超图的匹配度
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from desc_hypergraph import DescriptionHyperGraph, HyperNode, HyperEdge
from inst_hypergraph import InstanceHyperGraph, InstNode, InstEdge


@dataclass
class MatchResult:
    """匹配结果"""
    desc_graph_id: str
    candidate_bbox_id: int
    match_score: float
    matched_edges: List[Tuple[HyperEdge, InstEdge]]
    node_mapping: Dict[str, str]  # desc_node_id -> inst_node_id
    
    def __repr__(self):
        return f"MatchResult({self.desc_graph_id}, bbox_{self.candidate_bbox_id}, score={self.match_score:.3f})"


def match_nodes(desc_node: HyperNode, inst_nodes: List[InstNode]) -> List[InstNode]:
    """
    匹配描述节点到实例节点（按类别筛选）
    
    Args:
        desc_node: 描述图中的节点
        inst_nodes: 候选实例节点列表
    
    Returns:
        匹配的实例节点列表
    """
    return [n for n in inst_nodes if n.category == desc_node.category]


def match_edge(desc_edge: HyperEdge, inst_edge: InstEdge) -> float:
    """
    匹配两条边
    
    Returns:
        匹配分数 (0.0 ~ 1.0)
    """
    # 关系类型必须相同
    if desc_edge.relation != inst_edge.relation:
        return 0.0
    
    # 基础匹配分数
    base_score = 1.0
    
    # 结合实例边的置信度分数
    score = base_score * inst_edge.score
    
    return score


def compute_graph_match_score(desc_graph: DescriptionHyperGraph,
                               inst_subgraph: InstanceHyperGraph) -> Tuple[float, Dict[str, str], List]:
    """
    计算描述超图与实例子图的匹配分数
    
    Args:
        desc_graph: 描述超图
        inst_subgraph: 实例子图（来自候选 bbox）
    
    Returns:
        (匹配分数, 节点映射, 匹配的边列表)
    """
    if len(desc_graph.edges) == 0:
        # 没有边的描述，直接返回0（无法匹配空间关系）
        return 0.0, {}, []
    
    # 第一步：建立节点映射候选
    # 对每个描述节点，找出所有匹配的实例节点（同类别）
    node_candidates = {}
    for desc_node in desc_graph.nodes:
        candidates = match_nodes(desc_node, inst_subgraph.nodes)
        if not candidates:
            # 关键类别缺失，匹配失败
            return 0.0, {}, []
        node_candidates[desc_node.id] = candidates
    
    # 第二步：尝试找到最佳节点映射
    # 简化版：基于边的匹配度贪婪选择
    
    # 初始化映射：主体节点映射到最近的实例
    main_node = desc_graph.get_main_node()
    if main_node is None:
        return 0.0, {}, []
    
    # 选择匹配主体类别的所有实例节点
    main_candidates = node_candidates[main_node.id]
    if not main_candidates:
        return 0.0, {}, []
    
    # 第三步：对每条描述边，在实例图中找最佳匹配
    total_score = 0.0
    matched_edges = []
    
    for desc_edge in desc_graph.edges:
        best_edge_score = 0.0
        best_match = None
        
        # 获取描述边的起点和终点
        from_desc = desc_graph.get_node_by_id(desc_edge.from_node)
        to_desc = desc_graph.get_node_by_id(desc_edge.to_node)
        
        if from_desc is None or to_desc is None:
            continue
        
        # 遍历实例图的所有边，找关系类型匹配的
        for inst_edge in inst_subgraph.edges:
            if inst_edge.relation != desc_edge.relation:
                continue
            
            # 检查节点类别是否对应
            from_inst = inst_subgraph.get_node_by_id(inst_edge.from_node)
            to_inst = inst_subgraph.get_node_by_id(inst_edge.to_node)
            
            if from_inst is None or to_inst is None:
                continue
            
            if from_inst.category == from_desc.category and \
               to_inst.category == to_desc.category:
                # 边类型和节点类别都匹配
                edge_score = inst_edge.score
                if edge_score > best_edge_score:
                    best_edge_score = edge_score
                    best_match = (desc_edge, inst_edge)
        
        if best_match:
            total_score += best_edge_score
            matched_edges.append(best_match)
    
    # 归一化：按描述图的边数归一化
    match_score = total_score / len(desc_graph.edges) if desc_graph.edges else 0.0
    
    # 构建节点映射（简化版：取第一个匹配的）
    node_mapping = {}
    if main_candidates:
        node_mapping[main_node.id] = main_candidates[0].id
    
    return match_score, node_mapping, matched_edges


def match_hypergraphs(desc_graph: DescriptionHyperGraph,
                      instance_graph: InstanceHyperGraph,
                      candidate_bboxes: List[Tuple[int, np.ndarray, np.ndarray]],
                      instances_dict: Dict) -> Optional[MatchResult]:
    """
    为主查询匹配最佳候选 bbox
    
    Args:
        desc_graph: 描述超图
        instance_graph: 完整实例超图
        candidate_bboxes: [(bbox_id, bbox_min, bbox_max), ...]
        instances_dict: 原始实例字典（用于提取子图）
    
    Returns:
        最佳匹配结果
    """
    from inst_hypergraph import extract_subgraph
    
    best_result = None
    best_score = 0.0
    
    for bbox_id, bbox_min, bbox_max in candidate_bboxes:
        # 提取子图
        subgraph = extract_subgraph(instance_graph, bbox_min, bbox_max, instances_dict)
        
        # 计算匹配分数
        score, node_mapping, matched_edges = compute_graph_match_score(desc_graph, subgraph)
        
        print(f"  Candidate {bbox_id}: match_score = {score:.3f}, "
              f"nodes = {len(subgraph.nodes)}, edges = {len(subgraph.edges)}")
        
        if score > best_score:
            best_score = score
            best_result = MatchResult(
                desc_graph_id=f"{desc_graph.scene_id}_obj{desc_graph.object_id}",
                candidate_bbox_id=bbox_id,
                match_score=score,
                matched_edges=matched_edges,
                node_mapping=node_mapping
            )
    
    return best_result


def batch_match(query_results: List[Dict],
                desc_graphs: List[DescriptionHyperGraph],
                scene_graphs: Dict[str, InstanceHyperGraph],
                instances_dict: Dict[str, Dict]) -> List[MatchResult]:
    """
    批量匹配所有查询
    
    Args:
        query_results: evaluation_results_log.json 中的结果列表
        desc_graphs: 所有描述超图
        scene_graphs: {scene_id: InstanceHyperGraph}
        instances_dict: {scene_id: {category: [Instance]}}
    
    Returns:
        匹配结果列表
    """
    results = []
    
    for query in query_results:
        scene_id = query.get('scene_id')
        object_id = query.get('object_id')
        
        print(f"\nMatching {scene_id}_obj{object_id}...")
        
        # 找到对应的描述超图
        desc_graph = None
        for g in desc_graphs:
            if g.scene_id == scene_id and g.object_id == object_id:
                desc_graph = g
                break
        
        if desc_graph is None:
            print(f"  Warning: No description graph found")
            continue
        
        # 获取场景的实例超图
        if scene_id not in scene_graphs:
            print(f"  Warning: No instance graph for scene {scene_id}")
            continue
        
        instance_graph = scene_graphs[scene_id]
        instances = instances_dict.get(scene_id, {})
        
        # 构建候选 bbox 列表（从 top30 中取）
        # TODO: 从 evaluation_results_log.json 读取具体候选
        candidate_bboxes = []
        
        # 如果没有候选，跳过
        if not candidate_bboxes:
            print(f"  Warning: No candidate bboxes")
            continue
        
        # 执行匹配
        result = match_hypergraphs(desc_graph, instance_graph, candidate_bboxes, instances)
        
        if result:
            results.append(result)
            print(f"  Best match: {result}")
        else:
            print(f"  No valid match found")
    
    return results


if __name__ == "__main__":
    # 测试
    print("Hypergraph matching module loaded")
    print("Use batch_match() or match_hypergraphs() for matching")

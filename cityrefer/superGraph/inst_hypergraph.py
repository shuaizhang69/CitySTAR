"""
从场景实例构建实例超图
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from instance_extractor import Instance, load_scene_instances, get_all_instances_in_bbox
from spatial_relations import compute_relation, SpatialObject


@dataclass
class InstNode:
    """实例超图节点"""
    id: str
    category: str
    instance: Instance
    properties: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"InstNode({self.id}, {self.category})"


@dataclass
class InstEdge:
    """实例超图边"""
    from_node: str
    to_node: str
    relation: str
    score: float
    
    def __repr__(self):
        return f"InstEdge({self.from_node} --[{self.relation}:{self.score:.2f}]--> {self.to_node})"


@dataclass
class InstanceHyperGraph:
    """实例超图"""
    scene_id: str
    nodes: List[InstNode] = field(default_factory=list)
    edges: List[InstEdge] = field(default_factory=list)
    
    def get_node_by_id(self, node_id: str) -> Optional[InstNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_nodes_by_category(self, category: str) -> List[InstNode]:
        return [n for n in self.nodes if n.category == category]
    
    def get_edges_from(self, node_id: str) -> List[InstEdge]:
        return [e for e in self.edges if e.from_node == node_id]
    
    def get_edges_to(self, node_id: str) -> List[InstEdge]:
        return [e for e in self.edges if e.to_node == node_id]
    
    def get_edges_between(self, from_id: str, to_id: str) -> List[InstEdge]:
        return [e for e in self.edges if e.from_node == from_id and e.to_node == to_id]
    
    def __repr__(self):
        return f"InstGraph({self.scene_id}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


def instance_to_spatial_object(node: InstNode) -> SpatialObject:
    """将 InstNode 转换为 SpatialObject"""
    return SpatialObject(
        id=node.id,
        category=node.category,
        center=node.instance.center,
        bbox_min=node.instance.bbox_min,
        bbox_max=node.instance.bbox_max
    )


def build_instance_hypergraph(scene_id: str, 
                              instances: Dict[str, List[Instance]]) -> InstanceHyperGraph:
    """
    从场景实例构建完整的实例超图
    
    Args:
        scene_id: 场景名称
        instances: {category: [Instance, ...]}
    
    Returns:
        InstanceHyperGraph 对象
    """
    graph = InstanceHyperGraph(scene_id=scene_id)
    
    # 第一步：创建所有节点
    node_id_map = {}  # (category, inst_id) -> node_id
    for category, inst_list in instances.items():
        for inst in inst_list:
            node_id = f"{category}_{inst.id}"
            node_id_map[(category, inst.id)] = node_id
            
            node = InstNode(
                id=node_id,
                category=category,
                instance=inst
            )
            graph.nodes.append(node)
    
    # 第二步：计算所有实例对的空间关系
    nodes_list = graph.nodes
    
    for i, node_a in enumerate(nodes_list):
        for j, node_b in enumerate(nodes_list):
            if i == j:
                continue
            
            obj_a = instance_to_spatial_object(node_a)
            obj_b = instance_to_spatial_object(node_b)
            
            # 测试所有可能的关系，保留满足条件且分数最高的
            relations_to_test = [
                'left_of', 'right_of', 'front_of', 'behind',
                'adjacent', 'opposite', 'inside', 'on_surface'
            ]
            
            for rel_name in relations_to_test:
                is_valid, score = compute_relation(rel_name, obj_a, obj_b)
                
                if is_valid and score > 0.5:  # 只保留高置信度的关系
                    edge = InstEdge(
                        from_node=node_a.id,
                        to_node=node_b.id,
                        relation=rel_name,
                        score=score
                    )
                    graph.edges.append(edge)
    
    return graph


def extract_subgraph(graph: InstanceHyperGraph, 
                     bbox_min: np.ndarray,
                     bbox_max: np.ndarray,
                     instances: Dict[str, List[Instance]]) -> InstanceHyperGraph:
    """
    提取 bbox 范围内的子图
    
    Args:
        graph: 完整实例超图
        bbox_min: bbox 最小点
        bbox_max: bbox 最大点
        instances: 原始实例数据
    
    Returns:
        子图 InstanceHyperGraph
    """
    # 获取 bbox 内的所有实例
    filtered_instances = get_all_instances_in_bbox(instances, bbox_min, bbox_max)
    
    # 构建子图
    subgraph = InstanceHyperGraph(scene_id=graph.scene_id)
    
    # 只保留在范围内的节点
    node_ids_in_bbox = set()
    for inst in filtered_instances:
        node_id = f"{inst.category}_{inst.id}"
        if node_id in [n.id for n in graph.nodes]:
            node_ids_in_bbox.add(node_id)
    
    # 复制节点
    for node in graph.nodes:
        if node.id in node_ids_in_bbox:
            subgraph.nodes.append(node)
    
    # 复制边（两个端点都在范围内）
    for edge in graph.edges:
        if edge.from_node in node_ids_in_bbox and edge.to_node in node_ids_in_bbox:
            subgraph.edges.append(edge)
    
    return subgraph


if __name__ == "__main__":
    # 测试
    import os
    
    instance_root = "/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances"
    scene_name = "birmingham_block_1"
    
    print(f"Building instance hypergraph for {scene_name}...")
    
    # 加载实例
    instances = load_scene_instances(scene_name, instance_root)
    
    # 构建超图
    graph = build_instance_hypergraph(scene_name, instances)
    print(f"\nFull graph: {graph}")
    
    # 测试子图提取
    bbox_min = np.array([200, 650, 0])
    bbox_max = np.array([350, 800, 50])
    
    subgraph = extract_subgraph(graph, bbox_min, bbox_max, instances)
    print(f"Subgraph: {subgraph}")
    
    # 显示部分边
    print("\nSample edges:")
    for edge in subgraph.edges[:5]:
        print(f"  {edge}")

"""
从 CityRefer construction 构建描述超图
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from relation_mapper import map_relation, parse_spatial_relation


@dataclass
class HyperNode:
    """超图节点"""
    id: str
    category: str
    is_main: bool = False
    properties: Dict = field(default_factory=dict)
    
    def __repr__(self):
        main_flag = "[MAIN]" if self.is_main else ""
        return f"Node({self.id}, {self.category}){main_flag}"


@dataclass
class HyperEdge:
    """超图边"""
    from_node: str
    to_node: str
    relation: str
    raw_relation: str
    score: float = 1.0
    info: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return f"Edge({self.from_node} --[{self.relation}]--> {self.to_node})"


@dataclass
class DescriptionHyperGraph:
    """描述超图"""
    scene_id: str
    object_id: str
    description: str
    nodes: List[HyperNode] = field(default_factory=list)
    edges: List[HyperEdge] = field(default_factory=list)
    
    def get_main_node(self) -> Optional[HyperNode]:
        """获取主体节点"""
        for node in self.nodes:
            if node.is_main:
                return node
        return None
    
    def get_node_by_id(self, node_id: str) -> Optional[HyperNode]:
        """通过 ID 获取节点"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_edges_from(self, node_id: str) -> List[HyperEdge]:
        """获取从某节点出发的所有边"""
        return [e for e in self.edges if e.from_node == node_id]
    
    def get_edges_to(self, node_id: str) -> List[HyperEdge]:
        """获取指向某节点的所有边"""
        return [e for e in self.edges if e.to_node == node_id]
    
    def __repr__(self):
        return f"DescGraph({self.scene_id}_obj{self.object_id}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


def build_description_hypergraph(data: Dict) -> Optional[DescriptionHyperGraph]:
    """
    从 CityRefer 数据构建描述超图
    
    Args:
        data: {
            'scene_id': 'birmingham_block_12',
            'object_id': '1',
            'description': '...',
            'construction': [...]
        }
    
    Returns:
        DescriptionHyperGraph 对象
    """
    scene_id = data.get('scene_id', '')
    object_id = data.get('object_id', '')
    description = data.get('description', '')
    construction = data.get('construction', [])
    
    if not construction:
        return None
    
    graph = DescriptionHyperGraph(
        scene_id=scene_id,
        object_id=object_id,
        description=description
    )
    
    # 第一步：创建节点
    main_node = None
    other_nodes = []
    
    for i, obj in enumerate(construction):
        category = obj.get('category', '').lower()
        is_main = obj.get('is_main', False)
        
        if is_main:
            node_id = 'main'
            main_node = HyperNode(
                id=node_id,
                category=category,
                is_main=True,
                properties={
                    'category2': obj.get('category2', ''),
                    'color': obj.get('color', ''),
                    'landmark': obj.get('landmark', ''),
                    'identity_feature': obj.get('identity_feature', ''),
                }
            )
            graph.nodes.append(main_node)
        else:
            node_id = f"obj{i}"
            other_nodes.append(HyperNode(
                id=node_id,
                category=category,
                is_main=False,
                properties={
                    'category2': obj.get('category2', ''),
                    'color': obj.get('color', ''),
                    'landmark': obj.get('landmark', ''),
                    'identity_feature': obj.get('identity_feature', ''),
                    'sub_index': obj.get('sub_index', i),
                }
            ))
    
    # 添加其他节点
    graph.nodes.extend(other_nodes)
    
    # 第二步：创建边
    if main_node is None:
        print(f"Warning: No main node found in {scene_id}_obj{object_id}")
        return graph
    
    for i, obj in enumerate(construction):
        if obj.get('is_main', False):
            continue  # 跳过主体
        
        node_id = f"obj{i}"
        raw_relation = obj.get('spatial_relation', '')
        reference_anchor = obj.get('reference_anchor', '')
        
        # 解析空间关系
        relation, info = parse_spatial_relation(raw_relation, reference_anchor)
        
        if relation is None:
            # 无法解析的关系，跳过
            continue
        
        # 确定边的方向
        # 默认：客体 -> 主体 (如 "car next to building" -> car --next_to--> building)
        from_node = node_id
        to_node = 'main'
        
        # 特殊情况：关系是从主体指向客体
        # 如 "building to the right of road" -> building --right_of--> road
        if reference_anchor and 'main' in reference_anchor.lower():
            # 检查是否需要反转
            # 如果 raw_relation 包含 "to the right of" 等，可能需要反转
            if any(kw in raw_relation.lower() for kw in ['to the', 'of']):
                from_node = 'main'
                to_node = node_id
        
        edge = HyperEdge(
            from_node=from_node,
            to_node=to_node,
            relation=relation,
            raw_relation=raw_relation,
            info=info
        )
        graph.edges.append(edge)
    
    return graph


def load_description_hypergraphs(jsonl_path: str) -> List[DescriptionHyperGraph]:
    """
    从 JSONL 文件加载所有描述超图
    
    Args:
        jsonl_path: CityRefer_val_infer_result.jsonl 路径
    
    Returns:
        DescriptionHyperGraph 列表
    """
    graphs = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            graph = build_description_hypergraph(data)
            if graph:
                graphs.append(graph)
    
    return graphs


def get_hypergraph_by_id(graphs: List[DescriptionHyperGraph], 
                         scene_id: str, 
                         object_id: str) -> Optional[DescriptionHyperGraph]:
    """通过 scene_id 和 object_id 查找超图"""
    for graph in graphs:
        if graph.scene_id == scene_id and graph.object_id == object_id:
            return graph
    return None


if __name__ == "__main__":
    # 测试
    jsonl_path = "/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_val_infer_result.jsonl"
    
    print("Loading description hypergraphs...")
    graphs = load_description_hypergraphs(jsonl_path)
    print(f"Loaded {len(graphs)} graphs")
    
    # 显示第一个例子
    if graphs:
        g = graphs[0]
        print(f"\nExample: {g}")
        print(f"Description: {g.description}")
        print("\nNodes:")
        for node in g.nodes:
            print(f"  {node}")
        print("\nEdges:")
        for edge in g.edges:
            print(f"  {edge}")

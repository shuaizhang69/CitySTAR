"""
改进版：扩大bbox范围 + 根据描述选择性构建边
"""

from __future__ import annotations

import json
import os
import numpy as np
from typing import Optional
from tqdm import tqdm
from collections import defaultdict

from instance_extractor import Instance, load_scene_instances
from spatial_relations import SpatialObject, compute_relation


def load_desc_hypergraphs(desc_path: str) -> dict:
    """加载描述超图，按 (scene_id, object_id) 索引"""
    desc_graphs = {}
    with open(desc_path) as f:
        for line in f:
            data = json.loads(line)
            key = (data['scene_id'], data['object_id'])
            desc_graphs[key] = data
    return desc_graphs


def get_required_relations(desc_graphs: dict, scene_id: str, obj_id: int) -> set:
    """获取指定场景和物体需要的关系类型"""
    key = (scene_id, str(obj_id))  # object_id 转为字符串匹配
    if key not in desc_graphs:
        return set()
    
    edges = desc_graphs[key].get('hypergraph', {}).get('edges', [])
    relations = set(e['relation'] for e in edges)
    return relations


# 描述缺失或无边时，用弱拓扑/方向关系作为回退，避免实例超图只有点无边
FALLBACK_RELATIONS = frozenset({
    'left_of', 'right_of', 'front_of', 'behind',
    'north_of', 'south_of', 'adjacent', 'belonging',
})


def load_box3d(bbox_path: str) -> list:
    """加载 box3d 文件"""
    with open(bbox_path) as f:
        data = json.load(f)
    return data.get('bboxes', [])


def bbox_to_bounds(
    bbox: list,
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
) -> tuple:
    """
    从 bbox [x, y, z, w, h, d, ...] 提取 bounds，并扩大范围

    Args:
        expand_factor: 水平面 (w, h) 扩大倍数，默认 3
        expand_z_factor: 高度 d 的扩大倍数；默认 None 表示与 expand_factor 相同（旧行为）。
            设为小于 expand_factor 可减少垂直方向无关实例（如仅 XY 扩得大、Z 扩得小）。
    """
    x, y, z, w, h, d = bbox[:6]

    if expand_z_factor is None:
        expand_z_factor = expand_factor

    w_expanded = w * expand_factor
    h_expanded = h * expand_factor
    d_expanded = d * expand_z_factor

    bbox_min = np.array([x - w_expanded/2, y - h_expanded/2, z - d_expanded/2])
    bbox_max = np.array([x + w_expanded/2, y + h_expanded/2, z + d_expanded/2])
    center = np.array([x, y, z])
    
    return center, bbox_min, bbox_max


def _bbox_intersection_volume(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
) -> float:
    """计算两个 AABB 的交集体积（若不相交返回 0）。"""
    inter_min = np.maximum(bbox_min_a, bbox_min_b)
    inter_max = np.minimum(bbox_max_a, bbox_max_b)
    diff = inter_max - inter_min
    diff = np.maximum(diff, 0.0)
    return float(np.prod(diff))


def find_instances_in_bbox(
    instances: dict,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    include_mode: str = "center",
    overlap_ratio: float = 0.1,
) -> list:
    """查找 bbox 邻域内的实例。

    include_mode:
      - center: 实例中心在 bbox 内
      - overlap: 实例 bbox 与查询 bbox 交集体积 / 实例体积 >= overlap_ratio
    """
    result = []
    for _, inst_list in instances.items():
        for inst in inst_list:
            if include_mode == "center":
                if np.all(inst.center >= bbox_min) and np.all(inst.center <= bbox_max):
                    result.append(inst)
            elif include_mode == "overlap":
                inst_vol = _bbox_intersection_volume(inst.bbox_min, inst.bbox_max, inst.bbox_min, inst.bbox_max)
                if inst_vol <= 1e-9:
                    continue
                inter_vol = _bbox_intersection_volume(inst.bbox_min, inst.bbox_max, bbox_min, bbox_max)
                ratio = inter_vol / inst_vol
                if ratio >= overlap_ratio:
                    result.append(inst)
            else:
                raise ValueError(f"Unknown include_mode: {include_mode}")

    return result


def compute_instance_relations_selective(
    instance_list: list, 
    required_relations: set,
    target_bbox_center: np.ndarray = None,
    max_instances: int = 100,
    score_threshold: float = 0.5,
) -> list:
    """
    选择性计算实例关系
    
    Args:
        instance_list: 实例列表
        required_relations: 需要计算的关系集合
        target_bbox_center: 目标bbox中心（用于优先选择近实例）
        max_instances: 最大处理实例数
    """
    edges = []
    
    if len(instance_list) < 2 or not required_relations:
        return edges
    
    # 如果实例太多，优先选择离目标中心近的
    if len(instance_list) > max_instances and target_bbox_center is not None:
        instance_list = sorted(
            instance_list, 
            key=lambda x: np.linalg.norm(x.center - target_bbox_center)
        )[:max_instances]
    elif len(instance_list) > max_instances:
        instance_list = sorted(instance_list, key=lambda x: -x.num_points)[:max_instances]
    
    # 转换为 SpatialObject
    spatial_objs = []
    for inst in instance_list:
        obj = SpatialObject(
            id=f"{inst.category}_{inst.id}",
            category=inst.category,
            center=inst.center,
            bbox_min=inst.bbox_min,
            bbox_max=inst.bbox_max
        )
        spatial_objs.append((inst, obj))
    
    # 只计算需要的关系（由 score_threshold 参数控制边过滤）
    
    # 双向关系
    bidirectional_rels = [
        'left_of', 'right_of', 'front_of', 'behind',
        'north_of', 'south_of', 'above', 'below',
        'inside', 'on_surface', 'belonging',
        'adjacent', 'opposite', 'at_corner', 'near_corner',
        'at_end', 'on_edge', 'along', 'outside',
        'surrounded_by', 'connected_to', 'on_side',
        'towards', 'facing', 'far_from'
    ]
    
    # 检查所有实例对
    for i in range(len(spatial_objs)):
        for j in range(i + 1, len(spatial_objs)):
            inst_a, obj_a = spatial_objs[i]
            inst_b, obj_b = spatial_objs[j]
            
            # 计算双向关系
            for rel_name in bidirectional_rels:
                if rel_name not in required_relations:
                    continue
                    
                # A -> B
                is_valid, score = compute_relation(rel_name, obj_a, obj_b)
                if is_valid and score >= score_threshold:
                    edges.append({
                        "from": f"{inst_a.category}_{inst_a.id}",
                        "to": f"{inst_b.category}_{inst_b.id}",
                        "relation": rel_name,
                        "score": round(score, 3)
                    })
                
                # B -> A
                is_valid_rev, score_rev = compute_relation(rel_name, obj_b, obj_a)
                if is_valid_rev and score_rev >= score_threshold:
                    edges.append({
                        "from": f"{inst_b.category}_{inst_b.id}",
                        "to": f"{inst_a.category}_{inst_a.id}",
                        "relation": rel_name,
                        "score": round(score_rev, 3)
            })
    
    # 计算 between（三元关系）
    if 'between' in required_relations:
        # between(A, B, C): A 在 B 和 C 之间
        # 在超图中表示为: B --[between]--> A 和 C --[between]--> A
        for i, (inst_a, obj_a) in enumerate(spatial_objs):
            for j, (inst_b, obj_b) in enumerate(spatial_objs):
                if i == j:
                    continue
                for k, (inst_c, obj_c) in enumerate(spatial_objs):
                    if i == k or j == k:
                        continue
                    
                    # 检查 A 是否在 B 和 C 之间
                    from spatial_relations import between
                    is_valid, score = between(obj_a, obj_b, obj_c)
                    if is_valid and score >= score_threshold:
                        # B -> A (between)
                        edges.append({
                            "from": f"{inst_b.category}_{inst_b.id}",
                            "to": f"{inst_a.category}_{inst_a.id}",
                            "relation": "between",
                            "score": round(score, 3),
                            "anchors": [f"{inst_c.category}_{inst_c.id}"]  # 记录第三个节点
                        })
                        # C -> A (between)
                        edges.append({
                            "from": f"{inst_c.category}_{inst_c.id}",
                            "to": f"{inst_a.category}_{inst_a.id}",
                            "relation": "between",
                            "score": round(score, 3),
                            "anchors": [f"{inst_b.category}_{inst_b.id}"]  # 记录第三个节点
                        })
    
    # 计算 closest_to（如果需要）
    if 'closest_to' in required_relations:
        by_category = {}
        for inst, obj in spatial_objs:
            if inst.category not in by_category:
                by_category[inst.category] = []
            by_category[inst.category].append((inst, obj))
        
        for cat, objs in by_category.items():
            if len(objs) < 2:
                continue
            for i, (inst_a, obj_a) in enumerate(objs):
                min_dist = float('inf')
                closest_inst = None
                for j, (inst_b, obj_b) in enumerate(objs):
                    if i == j:
                        continue
                    dist = np.linalg.norm(obj_a.center - obj_b.center)
                    if dist < min_dist:
                        min_dist = dist
                        closest_inst = inst_b
                
                if closest_inst:
                    edges.append({
                        "from": f"{inst_a.category}_{inst_a.id}",
                        "to": f"{closest_inst.category}_{closest_inst.id}",
                        "relation": "closest_to",
                        "score": 1.0
                    })
    
    return edges


def build_bbox_hypergraph_v2(
    scene_id: str, 
    bbox: dict, 
    instances: dict,
    desc_graphs: dict,
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "center",
    instance_overlap_ratio: float = 0.1,
    instance_edge_score_threshold: float = 0.5,
) -> dict:
    """
    为单个 bbox 构建实例超图（改进版）
    """
    bbox_id = bbox['object_id']
    object_name = bbox['object_name'].lower()
    bbox_center, bbox_min, bbox_max = bbox_to_bounds(
        bbox['bbox'], expand_factor, expand_z_factor
    )
    
    # 查找 bbox 内的所有实例（扩大后的范围）
    instance_list = find_instances_in_bbox(
        instances,
        bbox_min,
        bbox_max,
        include_mode=instance_inclusion_mode,
        overlap_ratio=instance_overlap_ratio,
    )
    
    # 构建节点
    nodes = []
    node_by_category = defaultdict(list)
    
    for inst in instance_list:
        node_id = f"{inst.category}_{inst.id}"
        
        node = {
            "id": node_id,
            "category": inst.category,
            "center": inst.center.tolist(),
            "bbox_min": inst.bbox_min.tolist(),
            "bbox_max": inst.bbox_max.tolist(),
            "num_points": inst.num_points
        }
        
        nodes.append(node)
        node_by_category[inst.category].append(node_id)
    
    # 获取需要的关系（描述缺失或无边时用弱回退关系）
    required_relations = get_required_relations(desc_graphs, scene_id, bbox_id)
    if not required_relations:
        required_relations = set(FALLBACK_RELATIONS)
    
    # 选择性计算边
    edges = compute_instance_relations_selective(
        instance_list,
        required_relations,
        target_bbox_center=bbox_center,
        score_threshold=instance_edge_score_threshold,
    )
    
    # 按关系类型索引
    edges_by_relation = defaultdict(list)
    for edge in edges:
        edges_by_relation[edge['relation']].append(edge)
    
    return {
        "scene_id": scene_id,
        "bbox_id": bbox_id,
        "object_name": object_name,
        "expand_factor": expand_factor,
        "expand_z_factor": expand_z_factor if expand_z_factor is not None else expand_factor,
        "instance_inclusion_mode": instance_inclusion_mode,
        "instance_overlap_ratio": instance_overlap_ratio,
        "instance_edge_score_threshold": instance_edge_score_threshold,
        "bbox_center": bbox_center.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "hypergraph": {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "required_relations": list(required_relations),
            "node_by_category": dict(node_by_category),
            "edges_by_relation": dict(edges_by_relation)
        }
    }


def process_scene_v2(
    scene_id: str, 
    box3d_root: str, 
    instance_root: str, 
    output_dir: str,
    desc_graphs: dict,
    expand_factor: float = 3.0,
    expand_z_factor: Optional[float] = None,
    instance_inclusion_mode: str = "center",
    instance_overlap_ratio: float = 0.1,
    instance_edge_score_threshold: float = 0.5,
):
    """处理单个场景的所有 bbox（改进版）"""
    
    # 加载 box3d
    bbox_path = os.path.join(box3d_root, f"{scene_id}_bbox.json")
    if not os.path.exists(bbox_path):
        print(f"Box3d not found: {bbox_path}")
        return 0
    
    bboxes = load_box3d(bbox_path)
    print(f"  {scene_id}: {len(bboxes)} bboxes")
    
    # 加载实例
    instances = load_scene_instances(scene_id, instance_root)
    
    # 为每个 bbox 构建超图
    count = 0
    for bbox in bboxes:
        try:
            hypergraph = build_bbox_hypergraph_v2(
                scene_id,
                bbox,
                instances,
                desc_graphs,
                expand_factor,
                expand_z_factor,
                instance_inclusion_mode=instance_inclusion_mode,
                instance_overlap_ratio=instance_overlap_ratio,
                instance_edge_score_threshold=instance_edge_score_threshold,
            )
            
            # 保存
            bbox_id = bbox['object_id']
            output_path = os.path.join(output_dir, f"{scene_id}_bbox{bbox_id}_hypergraph.json")
            with open(output_path, 'w') as f:
                json.dump(hypergraph, f, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            
            count += 1
            
        except Exception as e:
            print(f"  Error processing bbox {bbox.get('object_id', '?')}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return count


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--box3d_root', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/box3d")
    parser.add_argument('--instance_root', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances")
    parser.add_argument('--desc_hypergraphs', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data/cityrefer/meta_data/CityRefer_desc_hypergraphs_dedup.jsonl")
    parser.add_argument('--output_dir', type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/bbox_hypergraphs_v2")
    parser.add_argument('--expand_factor', type=float, default=3.0,
                       help='水平面 w,h 扩大倍数')
    parser.add_argument('--expand_z_factor', type=float, default=None,
                       help='高度 d 扩大倍数；默认与 expand_factor 相同。可设为较小值以减少垂直无关实例')
    parser.add_argument('--scenes', type=str, nargs='+',
                       help='指定要处理的场景，不指定则处理所有')
    parser.add_argument('--instance_edge_score_threshold', type=float, default=0.5,
                        help='实例超图中边的 score_threshold（过滤低置信度边）')
    parser.add_argument('--instance_inclusion_mode', type=str, default='center',
                        choices=['center', 'overlap'],
                        help='实例包含标准：center=实例中心在 bbox 内；overlap=交集体积/实例体积>=overlap_ratio')
    parser.add_argument('--instance_overlap_ratio', type=float, default=0.1,
                        help='仅当 instance_inclusion_mode=overlap 时生效：交集体积/实例体积 阈值')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载描述超图
    print("Loading description hypergraphs...")
    desc_graphs = load_desc_hypergraphs(args.desc_hypergraphs)
    print(f"  Loaded {len(desc_graphs)} description graphs")
    
    # 获取场景列表
    if args.scenes:
        scenes = args.scenes
    else:
        import glob
        pattern = os.path.join(args.box3d_root, "*_bbox.json")
        files = glob.glob(pattern)
        scenes = sorted([os.path.basename(f).replace("_bbox.json", "") for f in files])
    
    ez = args.expand_z_factor if args.expand_z_factor is not None else args.expand_factor
    print(
        f"\nProcessing {len(scenes)} scenes with expand_factor={args.expand_factor}, "
        f"expand_z_factor={ez}, instance_inclusion_mode={args.instance_inclusion_mode}, "
        f"instance_overlap_ratio={args.instance_overlap_ratio}, "
        f"instance_edge_score_threshold={args.instance_edge_score_threshold}..."
    )
    
    total_hypergraphs = 0
    
    for scene_id in tqdm(scenes, desc="Processing scenes"):
        try:
            count = process_scene_v2(
                scene_id, 
                args.box3d_root, 
                args.instance_root, 
                args.output_dir,
                desc_graphs,
                args.expand_factor,
                args.expand_z_factor,
                instance_inclusion_mode=args.instance_inclusion_mode,
                instance_overlap_ratio=args.instance_overlap_ratio,
                instance_edge_score_threshold=args.instance_edge_score_threshold,
            )
            total_hypergraphs += count
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"生成统计:")
    print(f"  总场景数: {len(scenes)}")
    print(f"  总 bbox 超图: {total_hypergraphs}")


if __name__ == "__main__":
    main()

"""
从 PLY 文件提取单个实例的中心点和 bbox
"""

import numpy as np
import plyfile
import os
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from grid_50m_metadata import load_grid_metadata_offsets, resolve_tile_origin_xy

# 多瓦片合并时把 (tile_index, local_instance_id) 编码为全局 id，避免不同块里重复的 1,2,3… 冲突。
# 取 8192=2^13：在 int32 下可支持约 26 万块；要求单瓦片内 local instance_id < 8192。
TILE_INSTANCE_ID_MULTIPLIER = 8192

# 分块 PLY 形如 birmingham_block_12_x1200_y1100_building_instances.ply，场景 ID 仍为 birmingham_block_12
_SCENE_FROM_BUILDING_PLY = re.compile(
    r"^(?P<scene>birmingham_block_\d+|cambridge_block_\d+)(?:_x\d+_y\d+)?_building_instances\.ply$"
)


def discover_scene_ids(instance_root: str) -> List[str]:
    """
    扫描 instance_root 下所有 *_building_instances.ply，得到 scene_id 列表。
    支持扁平结构和分块子目录结构（如 scene/xNN_yMM/*.ply）。
    分块文件名中的 xNN_yMM 不会计入 scene_id。
    """
    if not os.path.isdir(instance_root):
        return []
    import glob
    scene_ids = set()
    pattern = os.path.join(instance_root, '**', '*_building_instances.ply')
    for path in glob.glob(pattern, recursive=True):
        name = os.path.basename(path)
        m = _SCENE_FROM_BUILDING_PLY.match(name)
        if m:
            scene_ids.add(m.group("scene"))
    return sorted(scene_ids)


def tile_origin_xy_from_ply_filename(
    filename: str,
    tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    瓦片世界 XY 原点：优先 grid_50m_3dqa/metadata.json 的 offset；否则用文件名 _xNNN_yMMM_。
    """
    return resolve_tile_origin_xy(filename, tile_xy_offsets)


def discover_categories_for_scene(scene_name: str, instance_root: str) -> List[str]:
    """
    列出某场景下所有 {scene}_{category}_instances.ply 或 {scene}_xNN_yMM_{category}_instances.ply 中的 category 名。
    支持递归扫描分块子目录。
    """
    if not os.path.isdir(instance_root):
        return []
    import re
    import glob
    cats = set()
    # 递归匹配所有该 scene 的 PLY
    pattern = os.path.join(instance_root, '**', f'{scene_name}_*_instances.ply')
    for path in glob.glob(pattern, recursive=True):
        name = os.path.basename(path)
        # 匹配 pattern: {scene}_{cat}_instances.ply 或 {scene}_xNN_yMM_{cat}_instances.ply
        m = re.match(rf'{re.escape(scene_name)}_(?:x\d+_y\d+_)?(.+?)_instances\.ply$', name)
        if m:
            cats.add(m.group(1))
    return sorted(cats)


@dataclass
class Instance:
    """单个实例"""
    id: int
    category: str
    center: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    num_points: int
    points: np.ndarray = None  # 可选：保存所有点
    
    def __repr__(self):
        return f"Instance(id={self.id}, cat='{self.category}', center=[{self.center[0]:.1f}, {self.center[1]:.1f}, {self.center[2]:.1f}], n={self.num_points})"


def load_instances_from_ply(
    ply_path: str,
    category: str,
    min_points: int = 100,
    tile_origin_xy: Optional[Tuple[float, float]] = None,
) -> List[Instance]:
    """
    从 PLY 文件加载所有实例
    
    Args:
        ply_path: PLY 文件路径
        category: 类别名称 (如 'building', 'car')
        min_points: 最小点数阈值，过滤太小的实例
        tile_origin_xy: 若给定 (ox, oy)，将 bbox/center 的 XY 平移到世界坐标（Z 不变）
    
    Returns:
        Instance 列表
    """
    if not os.path.exists(ply_path):
        return []
    
    try:
        plydata = plyfile.PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        points = np.stack([
            np.array(vertex['x']),
            np.array(vertex['y']),
            np.array(vertex['z'])
        ], axis=1)
        
        instance_ids = np.array(vertex['instance_id'])
        
        # 按 instance_id 分组
        unique_ids = np.unique(instance_ids)
        instances = []
        
        for inst_id in unique_ids:
            mask = instance_ids == inst_id
            inst_points = points[mask]
            
            # 过滤小实例
            if len(inst_points) < min_points:
                continue
            
            # 计算中心和 bbox
            center = np.mean(inst_points, axis=0).copy()
            bbox_min = np.min(inst_points, axis=0).copy()
            bbox_max = np.max(inst_points, axis=0).copy()
            if tile_origin_xy is not None:
                ox, oy = tile_origin_xy
                center[0] += ox
                center[1] += oy
                bbox_min[0] += ox
                bbox_min[1] += oy
                bbox_max[0] += ox
                bbox_max[1] += oy
            
            instance = Instance(
                id=int(inst_id),
                category=category,
                center=center,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                num_points=len(inst_points),
                # points=inst_points  # 可选：占用内存大，默认不保存
            )
            instances.append(instance)
        
        return instances
        
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        return []


def load_scene_instances(scene_name: str, instance_root: str, 
                         categories: Optional[List[str]] = None,
                         discover_all: bool = False,
                         recursive: bool = False,
                         tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
                         ) -> Dict[str, List[Instance]]:
    """
    加载场景的所有类别实例
    
    Args:
        scene_name: 场景名称，如 'birmingham_block_1'
        instance_root: 实例文件根目录
        categories: 类别列表，默认 ['building', 'car', 'parking', 'ground']
        discover_all: 为 True 时忽略 categories，扫描目录加载该场景全部 *_instances.ply 类别
        recursive: 为 True 时递归扫描子目录（用于 fusion_by_class 分块结构）
        tile_xy_offsets: 瓦片 XY 偏移表（metadata.json）；None 时若存在默认 metadata 则自动加载；
            传入空 dict 表示禁用 metadata、仅用文件名解析 x/y。
    
    Returns:
        {category: [Instance, ...]}
    """
    if discover_all:
        categories = discover_categories_for_scene(scene_name, instance_root)
    elif categories is None:
        categories = ['building', 'car', 'parking', 'ground']
    
    scene_instances = {}

    resolved_offsets = tile_xy_offsets
    if recursive and resolved_offsets is None:
        loaded = load_grid_metadata_offsets(None)
        resolved_offsets = loaded if loaded is not None else {}

    for category in categories:
        instances = []
        
        if recursive:
            # 递归扫描所有匹配文件（支持 xNN_yMM 分块）
            import glob
            import re
            pattern = os.path.join(
                instance_root, '**', 
                f"{scene_name}_*{category}_instances.ply"
            )
            ply_files = glob.glob(pattern, recursive=True)
            # 过滤确保 category 匹配正确（避免 building 匹配到 parking 等）
            ply_files = [
                p for p in ply_files 
                if re.search(rf'{re.escape(scene_name)}_(?:x\d+_y\d+_)?{re.escape(category)}_instances\.ply$', 
                            os.path.basename(p))
            ]
            for tidx, ply_path in enumerate(sorted(ply_files)):
                fn = os.path.basename(ply_path)
                ox, oy = tile_origin_xy_from_ply_filename(fn, resolved_offsets)
                origin = (ox, oy) if (ox != 0.0 or oy != 0.0) else None
                insts = load_instances_from_ply(ply_path, category, tile_origin_xy=origin)
                for inst in insts:
                    if inst.id != 0:
                        inst.id = tidx * TILE_INSTANCE_ID_MULTIPLIER + inst.id
                instances.extend(insts)
        else:
            # 原有扁平结构
            ply_path = os.path.join(
                instance_root,
                f"{scene_name}_{category}_instances.ply"
            )
            instances = load_instances_from_ply(ply_path, category)
        
        scene_instances[category] = instances
        print(f"  {category}: {len(instances)} instances")
    
    return scene_instances


def get_all_instances_in_bbox(instances: Dict[str, List[Instance]], 
                               bbox_min: np.ndarray, 
                               bbox_max: np.ndarray) -> List[Instance]:
    """
    获取 bbox 范围内的所有实例
    
    Args:
        instances: 场景实例字典
        bbox_min: bbox 最小点 [x, y, z]
        bbox_max: bbox 最大点 [x, y, z]
    
    Returns:
        在 bbox 内的实例列表
    """
    result = []
    
    for category, inst_list in instances.items():
        for inst in inst_list:
            # 检查实例中心是否在 bbox 内
            if (np.all(inst.center >= bbox_min) and 
                np.all(inst.center <= bbox_max)):
                result.append(inst)
    
    return result


def compute_instance_distances(inst1: Instance, inst2: Instance) -> Dict[str, float]:
    """
    计算两个实例之间的距离信息
    
    Returns:
        {
            'euclidean': 欧氏距离,
            'dx': X方向差,
            'dy': Y方向差,
            'dz': Z方向差,
        }
    """
    diff = inst2.center - inst1.center
    return {
        'euclidean': np.linalg.norm(diff),
        'dx': diff[0],
        'dy': diff[1],
        'dz': diff[2],
    }


if __name__ == "__main__":
    # 测试
    import sys
    
    instance_root = "/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances"
    scene_name = "birmingham_block_1"
    
    print(f"Loading instances for {scene_name}...")
    instances = load_scene_instances(scene_name, instance_root)
    
    total = sum(len(v) for v in instances.values())
    print(f"\nTotal instances: {total}")
    
    # 显示每个类别的前3个实例
    for category, inst_list in instances.items():
        print(f"\n{category} instances:")
        for inst in inst_list[:3]:
            print(f"  {inst}")

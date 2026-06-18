"""
实例过分割合并模块
基于实例 bbox 的 IOU 进行合并
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from instance_extractor import Instance
import plyfile


def compute_bbox_iou(bbox_min_a: np.ndarray, bbox_max_a: np.ndarray,
                     bbox_min_b: np.ndarray, bbox_max_b: np.ndarray,
                     use_2d: bool = True) -> float:
    """
    计算两个 BBox 的 IOU
    
    Args:
        bbox_min_a, bbox_max_a: 第一个 bbox 的最小/最大坐标
        bbox_min_b, bbox_max_b: 第二个 bbox 的最小/最大坐标
        use_2d: 是否只用 XY 平面（航拍点云适用）
    
    Returns:
        IOU 值 [0, 1]
    """
    if use_2d:
        # 只用 XY 平面（航拍点云）
        inter_min = np.maximum(bbox_min_a[:2], bbox_min_b[:2])
        inter_max = np.minimum(bbox_max_a[:2], bbox_max_b[:2])
        inter_size = np.maximum(inter_max - inter_min, 0)
        inter_area = np.prod(inter_size)
        
        if inter_area <= 0:
            return 0.0
        
        # 计算各自面积
        area_a = np.prod(bbox_max_a[:2] - bbox_min_a[:2])
        area_b = np.prod(bbox_max_b[:2] - bbox_min_b[:2])
        
        # 计算并集面积
        union_area = area_a + area_b - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    else:
        # 3D IOU（完整体积）
        inter_min = np.maximum(bbox_min_a, bbox_min_b)
        inter_max = np.minimum(bbox_max_a, bbox_max_b)
        inter_size = np.maximum(inter_max - inter_min, 0)
        inter_volume = np.prod(inter_size)
        
        if inter_volume <= 0:
            return 0.0
        
        vol_a = np.prod(bbox_max_a - bbox_min_a)
        vol_b = np.prod(bbox_max_b - bbox_min_b)
        union_volume = vol_a + vol_b - inter_volume
        
        if union_volume <= 0:
            return 0.0
        
        return inter_volume / union_volume


def merge_instance_group(instances: List[Instance]) -> Instance:
    """
    合并一组实例
    
    Args:
        instances: 要合并的实例列表
    
    Returns:
        合并后的新实例
    """
    if len(instances) == 1:
        return instances[0]
    
    # 使用第一个实例的 ID 和类别
    base_inst = instances[0]
    
    # 计算合并后的属性
    centers = np.array([inst.center for inst in instances])
    new_center = np.mean(centers, axis=0)
    
    bbox_mins = np.array([inst.bbox_min for inst in instances])
    bbox_maxs = np.array([inst.bbox_max for inst in instances])
    new_bbox_min = np.min(bbox_mins, axis=0)
    new_bbox_max = np.max(bbox_maxs, axis=0)
    
    total_points = sum(inst.num_points for inst in instances)
    
    # 合并后的实例（不保存点数据以节省内存，保存时再加载）
    merged = Instance(
        id=base_inst.id,  # 保留第一个实例的 ID
        category=base_inst.category,
        center=new_center,
        bbox_min=new_bbox_min,
        bbox_max=new_bbox_max,
        num_points=total_points,
        points=None  # 不保存点数据
    )
    
    # 记录合并信息
    merged.merged_from = [inst.id for inst in instances]
    merged.merged_count = len(instances)
    
    return merged


def merge_instances_by_iou(instances: List[Instance],
                           iou_threshold: float = 0.3,
                           category_specific: bool = True,
                           iterative: bool = False,
                           use_2d: bool = True) -> List[Instance]:
    """
    基于 IOU 合并过分割的实例
    
    Args:
        instances: 原始实例列表
        iou_threshold: IOU 阈值，超过则合并
        category_specific: 是否只合并同类实例
        iterative: 是否迭代合并直到收敛
        use_2d: 是否只用 XY 平面（航拍点云适用，默认 True）
    
    Returns:
        合并后的实例列表
    """
    if not instances:
        return []
    
    if not iterative:
        # 单次合并
        n = len(instances)
        merged = []
        used = set()
        
        # 按类别分组处理
        if category_specific:
            categories = set(inst.category for inst in instances)
            
            for category in categories:
                cat_indices = [i for i, inst in enumerate(instances) 
                              if inst.category == category and i not in used]
                
                # 对该类别的实例进行 IOU 合并
                cat_merged = _merge_single_category(
                    [instances[i] for i in cat_indices],
                    iou_threshold,
                    use_2d=use_2d
                )
                
                merged.extend(cat_merged)
                used.update(cat_indices)
        else:
            # 不考虑类别，直接合并
            merged = _merge_single_category(instances, iou_threshold, use_2d=use_2d)
        
        return merged
    else:
        # 迭代合并直到收敛
        return _merge_iterative(instances, iou_threshold, category_specific, use_2d=use_2d)


def _merge_iterative(instances: List[Instance],
                     iou_threshold: float,
                     category_specific: bool,
                     use_2d: bool = True) -> List[Instance]:
    """
    迭代合并直到没有实例可以合并
    """
    current = instances.copy()
    iteration = 0
    total_merged = 0
    
    print(f"  开始迭代合并 (IOU≥{iou_threshold})...")
    
    while True:
        iteration += 1
        before_count = len(current)
        
        # 执行一次合并
        categories = set(inst.category for inst in current) if category_specific else {None}
        new_instances = []
        
        for category in categories:
            if category_specific:
                cat_instances = [inst for inst in current if inst.category == category]
            else:
                cat_instances = current
            
            if len(cat_instances) <= 1:
                new_instances.extend(cat_instances)
                continue
            
            # 对该类别进行一次合并
            merged_cat = _merge_single_category(cat_instances, iou_threshold, use_2d=use_2d)
            new_instances.extend(merged_cat)
        
        after_count = len(new_instances)
        merged_this_round = before_count - after_count
        total_merged += merged_this_round
        
        print(f"    迭代 {iteration}: {before_count} -> {after_count} (合并 {merged_this_round})")
        
        if merged_this_round == 0:
            # 没有更多合并发生，收敛
            print(f"  收敛！总共合并 {total_merged} 个实例")
            break
        
        current = new_instances
    
    return current


def _merge_single_category(instances: List[Instance], 
                           iou_threshold: float,
                           use_2d: bool = True) -> List[Instance]:
    """
    对单一类别的实例进行 IOU 合并
    
    Args:
        instances: 实例列表
        iou_threshold: IOU 阈值
        use_2d: 是否只用 XY 平面（航拍点云适用）
    """
    if not instances:
        return []
    
    n = len(instances)
    parent = list(range(n))  # 并查集
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 计算所有实例对的 IOU
    for i in range(n):
        for j in range(i + 1, n):
            inst_a = instances[i]
            inst_b = instances[j]
            
            iou = compute_bbox_iou(
                inst_a.bbox_min, inst_a.bbox_max,
                inst_b.bbox_min, inst_b.bbox_max,
                use_2d=use_2d
            )
            
            if iou >= iou_threshold:
                union(i, j)
    
    # 按连通分量分组
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    # 合并每组
    merged = []
    for group_indices in groups.values():
        group_instances = [instances[i] for i in group_indices]
        merged_inst = merge_instance_group(group_instances)
        merged.append(merged_inst)
    
    return merged


def merge_instances_by_proximity(instances: List[Instance],
                                  distance_threshold: float = 2.0,
                                  category_specific: bool = True) -> List[Instance]:
    """
    基于中心点距离合并实例（备选方案）
    
    Args:
        instances: 原始实例列表
        distance_threshold: 中心点距离阈值（米）
        category_specific: 是否只合并同类实例
    """
    if not instances:
        return []
    
    n = len(instances)
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # 计算距离
    for i in range(n):
        for j in range(i + 1, n):
            inst_a = instances[i]
            inst_b = instances[j]
            
            # 类别检查
            if category_specific and inst_a.category != inst_b.category:
                continue
            
            dist = np.linalg.norm(inst_a.center - inst_b.center)
            
            if dist <= distance_threshold:
                union(i, j)
    
    # 按连通分量分组合并
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)
    
    merged = []
    for group_indices in groups.values():
        group_instances = [instances[i] for i in group_indices]
        merged_inst = merge_instance_group(group_instances)
        merged.append(merged_inst)
    
    return merged


# 类别特定的默认 IOU 阈值
DEFAULT_IOU_THRESHOLDS = {
    'car': 0.1,        # 车很紧凑，IOU 高才应该合并
    'building': 0.3,   # 建筑可能相邻，需要较高阈值
    'ground': 0.01,    # 地面：极小阈值，约 50 个实例
    'parking': 0.01,   # 停车场：极小阈值
}


def compute_bbox_distance(bbox_min_a: np.ndarray, bbox_max_a: np.ndarray,
                          bbox_min_b: np.ndarray, bbox_max_b: np.ndarray) -> float:
    """
    计算两个 BBox 在 XY 平面的距离
    如果相交返回负数（相交深度）
    """
    # XY 平面
    inter_min = np.maximum(bbox_min_a[:2], bbox_min_b[:2])
    inter_max = np.minimum(bbox_max_a[:2], bbox_max_b[:2])
    
    # 检查是否相交
    if np.all(inter_max > inter_min):
        # 相交，返回负的相交面积
        inter_size = inter_max - inter_min
        return -np.prod(inter_size)
    
    # 不相交，计算最近距离
    dist_x = max(0, inter_min[0] - inter_max[0]) if inter_min[0] > inter_max[0] else 0
    dist_y = max(0, inter_min[1] - inter_max[1]) if inter_min[1] > inter_max[1] else 0
    
    # 返回欧氏距离
    return np.sqrt(dist_x**2 + dist_y**2)


def compute_bbox_area(inst: Instance, use_2d: bool = True) -> float:
    """
    计算实例 bbox 的面积（2D）或体积（3D）
    """
    if use_2d:
        size = inst.bbox_max[:2] - inst.bbox_min[:2]
        return float(np.prod(size))
    else:
        size = inst.bbox_max - inst.bbox_min
        return float(np.prod(size))


def is_center_contained(small_inst: Instance, large_inst: Instance, 
                        expand_ratio: float = 1.0) -> bool:
    """
    检查 small_inst 的中心是否落在 large_inst 的 bbox 内
    支持扩展 large_inst 的 bbox 来检测
    
    Args:
        small_inst: 小实例
        large_inst: 大实例
        expand_ratio: bbox 扩展比例（1.0 表示不扩展，1.2 表示扩展 20%）
    
    Returns:
        如果中心点被包含返回 True
    """
    center = small_inst.center[:2]  # XY平面
    
    # 扩展 large_inst 的 bbox
    large_size = large_inst.bbox_max[:2] - large_inst.bbox_min[:2]
    expanded_min = large_inst.bbox_min[:2] - large_size * (expand_ratio - 1) / 2
    expanded_max = large_inst.bbox_max[:2] + large_size * (expand_ratio - 1) / 2
    
    return np.all(center >= expanded_min) and np.all(center <= expanded_max)


def merge_instances_advanced(instances: List[Instance],
                             category: str,
                             iou_threshold: float = 0.5,
                             min_area: float = 50.0,
                             center_expand_ratio: float = 1.2) -> List[Instance]:
    """
    改进的合并函数：IOU + 小实例过滤（不使用中心点包含，避免过合并）

    Args:
        instances: 原始实例列表
        category: 类别名称
        iou_threshold: IOU 合并阈值
        min_area: 最小面积阈值，小于此值的实例会被删除（如果未被合并）
        center_expand_ratio: 中心点包含检测时的 bbox 扩展比例（已禁用）

    Returns:
        合并后的实例列表
    """
    if not instances:
        return []

    if len(instances) == 1:
        # 检查是否需要删除
        area = compute_bbox_area(instances[0])
        if area < min_area:
            return []
        return instances

    n = len(instances)
    parent = list(range(n))  # 并查集

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Step 1: IOU 合并（只合并 IOU 高的实例，避免建筑群被合并）
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_bbox_iou(
                instances[i].bbox_min, instances[i].bbox_max,
                instances[j].bbox_min, instances[j].bbox_max,
                use_2d=True
            )
            if iou >= iou_threshold:
                union(i, j)

    # Step 2: 按连通分量分组
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Step 3: 执行合并并过滤小实例
    merged = []
    for group_indices in groups.values():
        group_instances = [instances[i] for i in group_indices]

        # 如果组内只有一个实例且面积小于阈值，则删除
        if len(group_instances) == 1:
            area = compute_bbox_area(group_instances[0])
            if area < min_area:
                continue  # 删除小实例

        merged_inst = merge_instance_group(group_instances)
        merged.append(merged_inst)

    return merged


# 与 car 相同策略：不合并（保存阶段按合并结果重映射 instance_id）
NO_MERGE_CATEGORIES = frozenset({
    'car', 'vegetation', 'footpath', 'bike', 'rail',
})

# 改进的合并配置
MERGE_CONFIG = {
    'building': {
        'iou_threshold': 0.5,  # IOU 阈值，只合并高度重叠的实例
        'min_area': 50.0,  # 50 m²，小于此值的孤立实例会被删除
        'center_expand_ratio': 1.0,  # 不扩展 bbox（中心点包含已禁用）
    },
    'car': {
        'iou_threshold': 0.5,
        'min_area': 10.0,  # 10 m²
        'center_expand_ratio': 1.0,
    },
    'ground': {
        # ground 使用密度聚类
        'use_density_clustering': True,
        'eps': 20.0,
        'max_cluster_size': 50,
    },
    'parking': {
        # parking 使用密度聚类
        'use_density_clustering': True,
        'eps': 10.0,
        'max_cluster_size': 50,
    },
}


def merge_instances_by_density_clustering(instances: List[Instance],
                                           eps: float = 5.0,
                                           min_samples: int = 1,
                                           max_cluster_size: int = 50) -> List[Instance]:
    """
    基于密度聚类合并实例（DBSCAN 思想，适用于 ground/parking）
    
    策略：
    1. 用中心点坐标进行密度聚类
    2. 距离小于 eps 的实例聚在一起
    3. 限制每个聚类的最大大小，避免过度合并
    
    Args:
        instances: 实例列表
        eps: 邻域半径（米），默认 5.0 米
        min_samples: 最小样本数（默认 1，即所有点都分配到一个簇）
        max_cluster_size: 每个簇的最大实例数，避免过度合并
    
    Returns:
        合并后的实例列表
    """
    if not instances:
        return []
    
    n = len(instances)
    if n <= 1:
        return instances
    
    # 提取中心点坐标（XY 平面）
    centers = np.array([inst.center[:2] for inst in instances])
    
    # 简单的 DBSCAN 实现
    visited = np.zeros(n, dtype=bool)
    clusters = []
    
    def get_neighbors(i):
        """获取点 i 的 eps 邻域内的所有点"""
        dists = np.linalg.norm(centers - centers[i], axis=1)
        return np.where(dists <= eps)[0]
    
    for i in range(n):
        if visited[i]:
            continue
        
        # 找到所有邻居
        neighbors = get_neighbors(i)
        
        if len(neighbors) < min_samples:
            # 标记为噪声（单独一个簇）
            visited[i] = True
            clusters.append([i])
        else:
            # 开始一个新簇
            cluster = []
            queue = list(neighbors)
            visited[neighbors] = True
            
            while queue and len(cluster) < max_cluster_size:
                j = queue.pop(0)
                cluster.append(j)
                
                # 获取 j 的邻居
                j_neighbors = get_neighbors(j)
                for nb in j_neighbors:
                    if not visited[nb] and len(cluster) < max_cluster_size:
                        visited[nb] = True
                        queue.append(nb)
            
            clusters.append(cluster)
    
    # 合并每个簇
    merged = []
    for cluster_indices in clusters:
        group_instances = [instances[i] for i in cluster_indices]
        merged_inst = merge_instance_group(group_instances)
        merged.append(merged_inst)
    
    return merged


def load_and_merge_scene_instances(scene_id: str, 
                                   instance_root: str,
                                   iou_threshold: float = None,
                                   iou_thresholds: Dict[str, float] = None,
                                   categories: List[str] = None,
                                   use_all_categories: bool = False,
                                   recursive: bool = False,
                                   tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
                                   ) -> Dict[str, List[Instance]]:
    """
    加载场景实例并进行合并（支持类别特定的 IOU 阈值）
    
    Args:
        scene_id: 场景 ID
        instance_root: 实例根目录
        iou_threshold: 全局 IOU 合并阈值（如果设置，覆盖所有类别）
        iou_thresholds: 类别特定的 IOU 阈值字典，如 {'car': 0.1, 'building': 0.3}
        categories: 类别列表，默认 ['building', 'car', 'parking', 'ground']
        use_all_categories: 为 True 时加载该场景目录下全部 *_instances.ply 类别
        recursive: 为 True 时递归扫描子目录（用于 fusion_by_class 分块结构）
        tile_xy_offsets: 见 instance_extractor.load_scene_instances；None 表示分块时自动读默认 metadata.json
    
    Returns:
        按类别组织的合并后实例字典
    """
    from instance_extractor import load_scene_instances, discover_categories_for_scene
    
    if use_all_categories:
        categories = None
    elif categories is None:
        categories = ['building', 'car', 'parking', 'ground']
    
    if use_all_categories:
        cat_list = discover_categories_for_scene(scene_id, instance_root)
    else:
        cat_list = categories
    
    if iou_threshold is not None:
        thresholds = {cat: iou_threshold for cat in cat_list}
    elif iou_thresholds is not None:
        thresholds = {cat: iou_thresholds.get(cat, DEFAULT_IOU_THRESHOLDS.get(cat, 0.3)) 
                     for cat in cat_list}
    else:
        thresholds = {cat: DEFAULT_IOU_THRESHOLDS.get(cat, 0.3) for cat in cat_list}
    
    raw_instances = load_scene_instances(
        scene_id, instance_root,
        categories=categories,
        discover_all=use_all_categories,
        recursive=recursive,
        tile_xy_offsets=tile_xy_offsets,
    )
    
    ground_cfg = MERGE_CONFIG.get('ground', {})
    parking_cfg = MERGE_CONFIG.get('parking', {})

    merged_instances = {}
    for category, instances in raw_instances.items():
        if len(instances) <= 1:
            merged_instances[category] = instances
            continue
        
        config = MERGE_CONFIG.get(category, {})

        if category in NO_MERGE_CATEGORIES:
            merged = instances
            print(f"  {category} (不合并, 与 car 同策略): {len(instances)} instances")
        elif category == 'ground':
            merged = merge_instances_by_density_clustering(
                instances,
                eps=config.get('eps', ground_cfg.get('eps', 20.0)),
                min_samples=1,
                max_cluster_size=config.get('max_cluster_size', ground_cfg.get('max_cluster_size', 50))
            )
            print(f"  {category} (密度聚类 eps={config.get('eps', 20.0)}m, max={config.get('max_cluster_size', 50)}): "
                  f"{len(instances)} -> {len(merged)} instances "
                  f"(merged {len(instances) - len(merged)})")
        elif category == 'parking':
            merged = merge_instances_by_density_clustering(
                instances,
                eps=config.get('eps', parking_cfg.get('eps', 10.0)),
                min_samples=1,
                max_cluster_size=config.get('max_cluster_size', parking_cfg.get('max_cluster_size', 50))
            )
            print(f"  {category} (密度聚类 eps={config.get('eps', 10.0)}m, max={config.get('max_cluster_size', 50)}): "
                  f"{len(instances)} -> {len(merged)} instances "
                  f"(merged {len(instances) - len(merged)})")
        elif category == 'building':
            merged = merge_instances_advanced(
                instances,
                category=category,
                iou_threshold=config.get('iou_threshold', 0.5),
                min_area=config.get('min_area', 50.0),
                center_expand_ratio=config.get('center_expand_ratio', 1.2)
            )
            print(f"  {category} (改进合并 IOU≥{config.get('iou_threshold', 0.5)}, "
                  f"min_area={config.get('min_area', 50.0)}m²): "
                  f"{len(instances)} -> {len(merged)} instances "
                  f"(merged {len(instances) - len(merged)})")
        else:
            unk_cfg = MERGE_CONFIG.get(category, ground_cfg)
            eps_u = unk_cfg.get('eps', ground_cfg.get('eps', 20.0))
            max_u = unk_cfg.get('max_cluster_size', ground_cfg.get('max_cluster_size', 50))
            merged = merge_instances_by_density_clustering(
                instances,
                eps=eps_u,
                min_samples=1,
                max_cluster_size=max_u
            )
            print(f"  {category} (未知类, 密度聚类 eps={eps_u}m, max={max_u}): "
                  f"{len(instances)} -> {len(merged)} instances "
                  f"(merged {len(instances) - len(merged)})")
        
        merged_instances[category] = merged
    
    return merged_instances


def load_and_merge_scene_instances_iterative(scene_id: str,
                                              instance_root: str,
                                              iou_threshold: float = 0.3,
                                              categories: List[str] = None) -> Dict[str, List[Instance]]:
    """
    加载场景实例并进行迭代合并（直到收敛）
    
    Args:
        scene_id: 场景 ID
        instance_root: 实例根目录
        iou_threshold: IOU 合并阈值
        categories: 类别列表
    
    Returns:
        按类别组织的合并后实例字典
    """
    from instance_extractor import load_scene_instances
    
    if categories is None:
        categories = ['building', 'car', 'parking', 'ground']
    
    # 加载原始实例
    raw_instances = load_scene_instances(scene_id, instance_root, categories)
    
    # 对每个类别的实例进行迭代合并
    merged_instances = {}
    for category, instances in raw_instances.items():
        if len(instances) <= 1:
            merged_instances[category] = instances
            continue
        
        # 迭代合并直到收敛
        merged = merge_instances_by_iou(instances, iou_threshold, category_specific=True, iterative=True)
        merged_instances[category] = merged
        
        print(f"  {category}: {len(instances)} -> {len(merged)} instances "
              f"(总共合并 {len(instances) - len(merged)})")
    
    return merged_instances


def save_merged_instances_to_ply(merged_instances: Dict[str, List[Instance]],
                                  scene_id: str,
                                  output_dir: str,
                                  original_instance_root: str,
                                  recursive: bool = False,
                                  tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
                                  ):
    """
    将合并后的实例保存为 PLY 文件
    
    从原始 PLY 文件读取点数据，按照合并后的分组重新分配 instance_id
    
    Args:
        merged_instances: 按类别组织的合并后实例字典
        scene_id: 场景 ID
        output_dir: 输出目录
        original_instance_root: 原始实例根目录（用于读取点数据）
        recursive: 为 True 时递归扫描原始目录读取分块 PLY
        tile_xy_offsets: 与 load_scene_instances 一致；None 时自动读默认 metadata.json
    """
    import os
    import numpy as np
    import plyfile
    import glob
    import re

    from grid_50m_metadata import load_grid_metadata_offsets
    
    os.makedirs(output_dir, exist_ok=True)

    resolved_save_offsets = tile_xy_offsets
    if recursive and resolved_save_offsets is None:
        _ld = load_grid_metadata_offsets(None)
        resolved_save_offsets = _ld if _ld is not None else {}
    
    for category, instances in merged_instances.items():
        if not instances:
            continue
        
        if recursive:
            # 递归扫描所有分块 PLY
            pattern = os.path.join(
                original_instance_root, '**',
                f"{scene_id}_*{category}_instances.ply"
            )
            ply_files = sorted(glob.glob(pattern, recursive=True))
            # 过滤确保 category 匹配正确
            ply_files = [
                p for p in ply_files 
                if re.search(rf'{re.escape(scene_id)}_(?:x\d+_y\d+_)?{re.escape(category)}_instances\.ply$', 
                            os.path.basename(p))
            ]
            from instance_extractor import (
                tile_origin_xy_from_ply_filename,
                TILE_INSTANCE_ID_MULTIPLIER,
            )
            
            # 收集所有原始点和 instance_id（瓦片局部坐标 -> 世界 XY；局部 instance_id -> 全局 id）
            all_points = []
            all_instance_ids = []
            for tidx, ply_path in enumerate(ply_files):
                try:
                    plydata = plyfile.PlyData.read(ply_path)
                    vertex = plydata['vertex']
                    pts = np.stack([
                        np.array(vertex['x']),
                        np.array(vertex['y']),
                        np.array(vertex['z'])
                    ], axis=1).astype(np.float64, copy=True)
                    ox, oy = tile_origin_xy_from_ply_filename(
                        os.path.basename(ply_path), resolved_save_offsets
                    )
                    pts[:, 0] += ox
                    pts[:, 1] += oy
                    iids = np.array(vertex['instance_id'], dtype=np.int64)
                    nz = iids != 0
                    iids_g = iids.copy()
                    iids_g[nz] = tidx * TILE_INSTANCE_ID_MULTIPLIER + iids[nz]
                    all_points.append(pts)
                    all_instance_ids.append(iids_g)
                except Exception as e:
                    print(f"  Warning: error reading {ply_path}: {e}")
                    continue
            
            if not all_points:
                print(f"  Warning: no PLY files found for {scene_id} {category}")
                continue
            
            points = np.vstack(all_points)
            original_instance_ids = np.concatenate(all_instance_ids)
        else:
            # 读取原始 PLY 文件
            original_ply_path = os.path.join(
                original_instance_root,
                f"{scene_id}_{category}_instances.ply"
            )
            
            if not os.path.exists(original_ply_path):
                print(f"  Warning: {original_ply_path} not found")
                continue
            
            # 读取原始点数据
            plydata = plyfile.PlyData.read(original_ply_path)
            vertex = plydata['vertex']
            
            points = np.stack([
                np.array(vertex['x']),
                np.array(vertex['y']),
                np.array(vertex['z'])
            ], axis=1)
            original_instance_ids = np.array(vertex['instance_id'])
        
        # 构建原始 ID 到新 ID 的映射
        id_mapping = {}
        for new_id, inst in enumerate(instances):
            if hasattr(inst, 'merged_from'):
                for old_id in inst.merged_from:
                    id_mapping[old_id] = new_id
            else:
                id_mapping[inst.id] = new_id
        
        # 重新分配 instance_id
        new_instance_ids = np.array([id_mapping.get(old_id, -1) for old_id in original_instance_ids])
        
        # 过滤掉未映射的点（不应该发生）
        valid_mask = new_instance_ids >= 0
        if not np.all(valid_mask):
            print(f"  Warning: {np.sum(~valid_mask)} points not mapped")
            points = points[valid_mask]
            new_instance_ids = new_instance_ids[valid_mask]
        
        # 创建 PLY 数据
        vertex_data = np.zeros(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('instance_id', 'i4')
        ])
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['instance_id'] = new_instance_ids
        
        # 保存 PLY
        ply_data = plyfile.PlyData([
            plyfile.PlyElement.describe(vertex_data, 'vertex')
        ])
        
        output_path = os.path.join(output_dir, f"{scene_id}_{category}_instances.ply")
        ply_data.write(output_path)
        print(f"  Saved: {output_path} ({len(instances)} instances, {len(points)} points)")


def process_and_save_scene(scene_id: str,
                           instance_root: str,
                           output_root: str,
                           iou_threshold: float = None,
                           categories: List[str] = None,
                           use_all_categories: bool = False,
                           recursive: bool = False,
                           tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
                           ):
    """
    处理单个场景：加载、合并、保存
    
    Args:
        scene_id: 场景 ID
        instance_root: 原始实例根目录
        output_root: 输出根目录
        iou_threshold: IOU 合并阈值（None 则使用类别特定阈值）
        categories: 类别列表
        use_all_categories: 为 True 时处理该场景下全部 *_instances.ply 类别
        recursive: 为 True 时递归扫描子目录（用于 fusion_by_class 分块结构）
        tile_xy_offsets: 瓦片偏移表；由 prepare_tile_xy_offsets 生成或传 None 自动加载
    """
    print(f"\n处理场景: {scene_id}")
    if use_all_categories:
        print(f"  模式: 全类别（目录扫描）")
    if recursive:
        print(f"  模式: 递归扫描分块目录")
    if iou_threshold is not None:
        print(f"  IOU 阈值: {iou_threshold} (全局)")
    else:
        print(f"  IOU 阈值: 类别特定")
        for cat, thr in DEFAULT_IOU_THRESHOLDS.items():
            print(f"    {cat}: {thr}")
    
    # 加载并合并
    merged = load_and_merge_scene_instances(
        scene_id, instance_root,
        iou_threshold=iou_threshold,
        categories=categories,
        use_all_categories=use_all_categories,
        recursive=recursive,
        tile_xy_offsets=tile_xy_offsets,
    )
    
    # 保存（传递原始实例根目录用于读取点数据）
    save_merged_instances_to_ply(
        merged, scene_id, output_root, instance_root,
        recursive=recursive, tile_xy_offsets=tile_xy_offsets,
    )
    
    total = sum(len(insts) for insts in merged.values())
    print(f"  完成: {total} instances saved")
    
    return merged


def batch_process_scenes(scene_ids: List[str],
                         instance_root: str,
                         output_root: str,
                         iou_threshold: float = None,
                         categories: List[str] = None,
                         use_all_categories: bool = False,
                         skip_existing: bool = False):
    """
    批量处理多个场景
    
    Args:
        scene_ids: 场景 ID 列表
        instance_root: 原始实例根目录
        output_root: 输出根目录
        iou_threshold: IOU 合并阈值（None 为类别特定默认）
        categories: 类别列表
        use_all_categories: 每场景处理目录下全部类别
        skip_existing: 若输出中已存在 {scene}_building_instances.ply 则跳过
    """
    print(f"批量处理 {len(scene_ids)} 个场景")
    print(f"输出目录: {output_root}")
    print(f"IOU 阈值: {iou_threshold if iou_threshold is not None else '类别特定'}")
    print(f"全类别模式: {use_all_categories}")
    print("=" * 60)
    
    for scene_id in scene_ids:
        if skip_existing:
            marker = os.path.join(output_root, f"{scene_id}_building_instances.ply")
            if os.path.isfile(marker):
                print(f"\n跳过（已存在）: {scene_id}")
                continue
        try:
            process_and_save_scene(
                scene_id, instance_root, output_root,
                iou_threshold=iou_threshold,
                categories=categories,
                use_all_categories=use_all_categories,
            )
        except Exception as e:
            print(f"  Error processing {scene_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("批量处理完成!")


if __name__ == "__main__":
    import argparse
    from instance_extractor import discover_scene_ids
    from grid_50m_metadata import prepare_tile_xy_offsets, DEFAULT_GRID_METADATA_PATH
    
    parser = argparse.ArgumentParser(description="实例过分割合并工具")
    parser.add_argument("--scene", type=str, default="birmingham_block_0",
                       help="场景 ID")
    parser.add_argument("--input_root", type=str, 
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances",
                       help="原始实例根目录")
    parser.add_argument("--output_root", type=str,
                       default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances_merged_all",
                       help="合并后实例输出目录")
    parser.add_argument("--iou", type=float, default=None,
                       help="全局 IOU 合并阈值（不传则各类别用默认策略）")
    parser.add_argument("--save", action="store_true",
                       help="保存合并后的实例到新文件夹")
    parser.add_argument("--test", action="store_true",
                       help="运行对比测试")
    parser.add_argument("--all_categories", action="store_true",
                       help="处理该场景 input_root 下全部 *_instances.ply 类别（与默认四类互斥于扫描范围）")
    parser.add_argument("--discover_all_scenes", action="store_true",
                       help="扫描 input_root 下全部 *_building_instances.ply 并批量处理（需配合 --save）")
    parser.add_argument("--skip_existing", action="store_true",
                       help="若 output_root 已存在 {scene}_building_instances.ply 则跳过该场景")
    
    parser.add_argument("--recursive", action="store_true",
                       help="递归扫描子目录（用于 fusion_by_class scene/xNN_yMM/ 分块结构）")
    parser.add_argument(
        "--grid_metadata",
        type=str,
        default=None,
        help=(
            "grid_50m_3dqa/metadata.json 路径；默认在存在时使用 "
            f"{DEFAULT_GRID_METADATA_PATH}"
        ),
    )
    parser.add_argument(
        "--no_grid_metadata",
        action="store_true",
        help="不使用 metadata.json，仅用 instances 文件名中的 _xNNN_yMMM_",
    )

    args = parser.parse_args()

    def _cli_tile_offsets():
        return prepare_tile_xy_offsets(
            disabled=args.no_grid_metadata,
            metadata_path=args.grid_metadata,
        )
    
    if args.test:
        # 运行对比测试
        print(f"Testing instance merging for {args.scene}...")
        print("=" * 60)
        
        # 测试 1: 单次合并 IOU=0.3
        print("\n[1] 单次合并 (IOU≥0.3):")
        print("-" * 60)
        merged_single = load_and_merge_scene_instances(
            args.scene, args.input_root, iou_threshold=0.3, recursive=args.recursive,
            tile_xy_offsets=_cli_tile_offsets(),
        )
        total_single = sum(len(insts) for insts in merged_single.values())
        print(f"Total: {total_single} instances")
        
        # 测试 2: 单次合并 IOU=0.1
        print("\n[2] 单次合并 (IOU≥0.1):")
        print("-" * 60)
        merged_01 = load_and_merge_scene_instances(
            args.scene, args.input_root, iou_threshold=0.1, recursive=args.recursive,
            tile_xy_offsets=_cli_tile_offsets(),
        )
        total_01 = sum(len(insts) for insts in merged_01.values())
        print(f"Total: {total_01} instances")
        
        print("\n" + "=" * 60)
        print("对比总结:")
        original = 675
        print(f"  原始实例数: {original}")
        print(f"  单次 IOU=0.3: {total_single} instances (减少 {original - total_single}, {(original - total_single)/original*100:.1f}%)")
        print(f"  单次 IOU=0.1: {total_01} instances (减少 {original - total_01}, {(original - total_01)/original*100:.1f}%)")
    
    elif args.save:
        tile_ov = _cli_tile_offsets()
        if tile_ov is None and args.recursive:
            print("瓦片偏移: 使用默认 metadata.json（若存在）")
        elif args.no_grid_metadata:
            print("瓦片偏移: 已禁用 metadata，仅用文件名 x/y")
        elif args.grid_metadata is not None:
            print(f"瓦片偏移: {args.grid_metadata}")
        print(f"输出目录: {args.output_root}")
        print("类别特定阈值 (DEFAULT_IOU_THRESHOLDS):")
        for cat, thr in DEFAULT_IOU_THRESHOLDS.items():
            print(f"  {cat}: IOU≥{thr}")
        print("=" * 60)
        
        if args.discover_all_scenes:
            scene_ids = discover_scene_ids(args.input_root)
            print(f"扫描到 {len(scene_ids)} 个场景 (经 *_building_instances.ply)")
            print(f"递归模式: {args.recursive}")
            for scene_id in scene_ids:
                try:
                    process_and_save_scene(
                        scene_id, args.input_root, args.output_root,
                        iou_threshold=args.iou,
                        use_all_categories=True,
                        recursive=args.recursive,
                        tile_xy_offsets=tile_ov,
                    )
                except Exception as e:
                    print(f"  Error processing {scene_id}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"处理场景: {args.scene}")
            print(f"递归模式: {args.recursive}")
            process_and_save_scene(
                args.scene, args.input_root, args.output_root,
                iou_threshold=args.iou,
                use_all_categories=args.all_categories,
                recursive=args.recursive,
                tile_xy_offsets=tile_ov,
            )
        
        print(f"\n完成！合并后的实例已保存到: {args.output_root}")
    
    else:
        # 默认：使用类别特定阈值测试
        print(f"Testing instance merging for {args.scene}...")
        print("=" * 60)
        print("类别特定阈值:")
        for cat, thr in DEFAULT_IOU_THRESHOLDS.items():
            print(f"  {cat}: IOU≥{thr}")
        print("-" * 60)
        
        # 使用默认的类别特定阈值（不传入 iou_threshold）
        merged = load_and_merge_scene_instances(
            args.scene, args.input_root,
            tile_xy_offsets=_cli_tile_offsets(),
        )
        total = sum(len(insts) for insts in merged.values())
        
        print(f"\n合并完成: {total} instances")
        print(f"使用 --save 参数保存结果")
        print(f"使用 --test 参数运行对比测试")

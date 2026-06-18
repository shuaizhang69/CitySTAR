"""
Instance Merger - 参数化的实例合并模块
支持按类别配置不同的合并策略
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from instance_splitter import InstanceData


def compute_bbox_iou_3d(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray
) -> float:
    """
    计算两个 3D bbox 的 IoU
    """
    inter_min = np.maximum(bbox_min_a, bbox_min_b)
    inter_max = np.minimum(bbox_max_a, bbox_max_b)

    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    vol_a = np.prod(bbox_max_a - bbox_min_a)
    vol_b = np.prod(bbox_max_b - bbox_min_b)
    union_vol = vol_a + vol_b - inter_vol

    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def compute_bbox_iou_2d(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray
) -> float:
    """
    计算两个 bbox 在 XY 平面的 2D IoU
    """
    inter_min = np.maximum(bbox_min_a[:2], bbox_min_b[:2])
    inter_max = np.minimum(bbox_max_a[:2], bbox_max_b[:2])

    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_area = np.prod(inter_size)

    area_a = np.prod(bbox_max_a[:2] - bbox_min_a[:2])
    area_b = np.prod(bbox_max_b[:2] - bbox_min_b[:2])
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_bbox_area_2d(inst: InstanceData) -> float:
    """计算实例 bbox 的 XY 面积"""
    size = inst.bbox_max[:2] - inst.bbox_min[:2]
    return float(np.prod(size))


def compute_bbox_volume(inst: InstanceData) -> float:
    """计算实例 bbox 的体积"""
    size = inst.bbox_max - inst.bbox_min
    return float(np.prod(size))


def check_z_overlap(inst_a: InstanceData, inst_b: InstanceData, min_overlap: float = 0.1) -> bool:
    """
    检查两个实例在 Z 方向是否有重叠

    Args:
        inst_a, inst_b: 实例
        min_overlap: 最小重叠比例（相对于较小的 Z 范围）

    Returns:
        是否有足够重叠
    """
    z_min_a, z_max_a = inst_a.bbox_min[2], inst_a.bbox_max[2]
    z_min_b, z_max_b = inst_b.bbox_min[2], inst_b.bbox_max[2]

    overlap_min = max(z_min_a, z_min_b)
    overlap_max = min(z_max_a, z_max_b)

    if overlap_max <= overlap_min:
        return False

    overlap = overlap_max - overlap_min
    range_a = z_max_a - z_min_a
    range_b = z_max_b - z_min_b

    min_range = min(range_a, range_b)
    if min_range == 0:
        return True

    return (overlap / min_range) >= min_overlap


def merge_instance_group(instances: List[InstanceData], new_id: int = 0) -> InstanceData:
    """
    合并一组实例

    Args:
        instances: 要合并的实例列表
        new_id: 合并后实例的 ID

    Returns:
        合并后的新实例
    """
    if len(instances) == 0:
        return InstanceData(instance_id=new_id, points=np.zeros((0, 3)))

    if len(instances) == 1:
        return InstanceData(
            instance_id=new_id,
            points=instances[0].points.copy(),
            category=instances[0].category
        )

    # 合并所有点
    all_points = np.vstack([inst.points for inst in instances])
    category = instances[0].category

    return InstanceData(
        instance_id=new_id,
        points=all_points,
        category=category
    )


class InstanceMerger:
    """实例合并器，支持参数化配置"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 合并配置字典，按类别组织
            {
                'building': {
                    'iou_threshold': 0.3,
                    'distance_threshold': 5.0,
                    'require_z_overlap': True,
                    'min_area': 50.0,
                    'use_2d_iou': True,
                },
                ...
            }
        """
        self.config = config

    def should_merge_pair(
        self,
        inst_a: InstanceData,
        inst_b: InstanceData,
        cat_config: Dict
    ) -> bool:
        """
        判断两个实例是否应该合并

        Args:
            inst_a, inst_b: 待判断的实例
            cat_config: 该类别的合并配置

        Returns:
            是否应该合并
        """
        # 检查 IoU
        iou_thr = cat_config.get('iou_threshold', 0.0)
        use_2d = cat_config.get('use_2d_iou', True)

        if use_2d:
            iou = compute_bbox_iou_2d(
                inst_a.bbox_min, inst_a.bbox_max,
                inst_b.bbox_min, inst_b.bbox_max
            )
        else:
            iou = compute_bbox_iou_3d(
                inst_a.bbox_min, inst_a.bbox_max,
                inst_b.bbox_min, inst_b.bbox_max
            )

        if iou >= iou_thr and iou_thr > 0:
            return True

        # 检查中心点距离
        dist_thr = cat_config.get('distance_threshold', 0.0)
        if dist_thr > 0:
            center_dist = np.linalg.norm(inst_a.center[:2] - inst_b.center[:2])
            if center_dist <= dist_thr:
                # 检查 Z 重叠（如果需要）
                if cat_config.get('require_z_overlap', False):
                    if check_z_overlap(inst_a, inst_b, min_overlap=0.1):
                        return True
                else:
                    return True

        return False

    def merge_instances(
        self,
        instances: List[InstanceData],
        category: str
    ) -> List[InstanceData]:
        """
        合并同类别的实例

        Args:
            instances: 实例列表
            category: 类别名

        Returns:
            合并后的实例列表
        """
        if len(instances) <= 1:
            return instances

        cat_config = self.config.get(category, {})
        if not cat_config:
            # 无配置则不合并
            return instances

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

        # 用空间索引加速候选对查找，避免 O(n^2)
        dist_thr = cat_config.get('distance_threshold', 0.0)
        iou_thr = cat_config.get('iou_threshold', 0.0)
        use_spatial_index = False
        neighbor_pairs = []

        if dist_thr > 0 or iou_thr > 0:
            centers = np.array([inst.center[:2] for inst in instances])
            sizes = np.array([inst.bbox_max[:2] - inst.bbox_min[:2] for inst in instances])
            half_diags = np.linalg.norm(sizes / 2, axis=1)
            p95_half_diag = float(np.percentile(half_diags, 95)) if len(half_diags) > 0 else 0.0
            # 保守搜索半径：覆盖 distance_threshold 和可能的 IoU 重叠
            # 用 p95 代替 max，避免被单个异常大实例拖垮
            search_radius = dist_thr + 2.0 * p95_half_diag

            if search_radius > 0 and search_radius < float('inf'):
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(centers)
                    pairs_set = set()
                    for i, neighbors in enumerate(tree.query_ball_point(centers, r=search_radius)):
                        for j in neighbors:
                            if j > i:
                                pairs_set.add((i, j))
                    neighbor_pairs = list(pairs_set)
                    use_spatial_index = True
                except ImportError:
                    pass

        if use_spatial_index:
            for i, j in neighbor_pairs:
                if self.should_merge_pair(instances[i], instances[j], cat_config):
                    union(i, j)
        else:
            # Fallback: O(n^2)
            for i in range(n):
                for j in range(i + 1, n):
                    if self.should_merge_pair(instances[i], instances[j], cat_config):
                        union(i, j)

        # 按连通分量分组
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        # 合并每组
        merged = []
        new_id = 0
        for group_indices in groups.values():
            group_instances = [instances[i] for i in group_indices]
            merged_inst = merge_instance_group(group_instances, new_id)

            # 检查最小面积/点数过滤
            min_area = cat_config.get('min_area', 0.0)
            min_points = cat_config.get('min_points', 0)

            area = compute_bbox_area_2d(merged_inst)
            num_points = merged_inst.num_points

            if area >= min_area and num_points >= min_points:
                merged.append(merged_inst)
                new_id += 1

        return merged


def merge_by_density_clustering(
    instances: List[InstanceData],
    eps: float,
    max_cluster_size: int = 50,
    min_points: int = 1
) -> List[InstanceData]:
    """
    基于中心点密度聚类合并实例（适用于 ground/parking）
    使用 cKDTree 优化邻域查询，避免 O(n^2)

    Args:
        instances: 实例列表
        eps: 邻域半径（米）
        max_cluster_size: 每个簇的最大实例数
        min_points: 最小核心点数

    Returns:
        合并后的实例列表
    """
    if len(instances) <= 1:
        return instances

    n = len(instances)
    centers = np.array([inst.center[:2] for inst in instances])

    # 用 cKDTree 加速邻域查询
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centers)
        neighbor_lists = tree.query_ball_point(centers, r=eps)
    except ImportError:
        # fallback: 简单循环（但仍是 O(n^2)，仅用于无 scipy 环境）
        neighbor_lists = []
        for i in range(n):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            neighbor_lists.append(np.where(dists <= eps)[0].tolist())

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        neighbors = neighbor_lists[i]

        if len(neighbors) < min_points:
            # 噪声点，单独成簇
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

                j_neighbors = neighbor_lists[j]
                for nb in j_neighbors:
                    if not visited[nb] and len(cluster) < max_cluster_size:
                        visited[nb] = True
                        queue.append(nb)

            clusters.append(cluster)

    # 合并每个簇
    merged = []
    for cluster_indices in clusters:
        group_instances = [instances[i] for i in cluster_indices]
        merged_inst = merge_instance_group(group_instances, len(merged))
        merged.append(merged_inst)

    return merged


def merge_instances_by_category(
    instances: List[InstanceData],
    category: str,
    config: Dict
) -> List[InstanceData]:
    """
    根据类别配置合并实例

    Args:
        instances: 实例列表
        category: 类别名
        config: 合并配置

    Returns:
        合并后的实例列表
    """
    if len(instances) <= 1:
        return instances

    # 检查是否使用密度聚类
    if config.get('use_density_clustering', False):
        eps = config.get('eps', 10.0)
        max_size = config.get('max_cluster_size', 50)
        return merge_by_density_clustering(instances, eps, max_size)

    # 使用 IoU/距离合并
    merger = InstanceMerger({category: config})
    return merger.merge_instances(instances, category)


# 默认合并配置（尽量与原始 merge_oversegmented_instances.py 对齐）
DEFAULT_MERGE_CONFIG = {
    'building': {
        'iou_threshold': 0.5,
        'distance_threshold': 0.0,  # 默认不启用距离合并
        'require_z_overlap': True,
        'min_area': 50.0,
        'min_points': 50,
        'use_2d_iou': True,
    },
    'car': {
        'iou_threshold': 1.0,  # 设为 1.0 表示不合并（与 NO_MERGE_CATEGORIES 对齐）
        'distance_threshold': 0.0,
        'require_z_overlap': False,
        'min_area': 0.0,
        'min_points': 0,
        'use_2d_iou': True,
    },
    'ground': {
        'use_density_clustering': True,
        'eps': 20.0,
        'max_cluster_size': 50,
    },
    'parking': {
        'use_density_clustering': True,
        'eps': 10.0,
        'max_cluster_size': 50,
    },
}


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)

    # 创建测试实例
    # 两个应该合并的 building（距离近、Z 重叠）
    building_a = InstanceData(
        instance_id=0,
        points=np.random.randn(100, 3) * 2 + np.array([0, 0, 5]),
        category='building'
    )
    building_b = InstanceData(
        instance_id=1,
        points=np.random.randn(100, 3) * 2 + np.array([3, 0, 5]),
        category='building'
    )

    # 两个不应该合并的 building（Z 不重叠）
    building_c = InstanceData(
        instance_id=2,
        points=np.random.randn(100, 3) * 2 + np.array([0, 0, 20]),
        category='building'
    )

    instances = [building_a, building_b, building_c]

    print(f"Original: {len(instances)} instances")

    # 合并
    merged = merge_instances_by_category(
        instances, 'building', DEFAULT_MERGE_CONFIG['building']
    )

    print(f"After merging: {len(merged)} instances")
    for i, inst in enumerate(merged):
        print(f"  Instance {i}: {inst.num_points} points, center={inst.center}")

"""
Instance Splitter - 基于 DBSCAN 的实例分割模块
用于将误桥接的大实例切分成多个独立实例
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class InstanceData:
    """实例数据容器"""
    instance_id: int
    points: np.ndarray  # (N, 3) xyz 坐标
    category: str = ""

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0) if len(self.points) > 0 else np.zeros(3)

    @property
    def bbox_min(self) -> np.ndarray:
        return self.points.min(axis=0) if len(self.points) > 0 else np.zeros(3)

    @property
    def bbox_max(self) -> np.ndarray:
        return self.points.max(axis=0) if len(self.points) > 0 else np.zeros(3)

    @property
    def num_points(self) -> int:
        return len(self.points)


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    轻量级 DBSCAN 实现（不依赖 sklearn）
    对大数据集自动使用体素下采样加速。

    Args:
        points: (N, D) 点云坐标
        eps: 邻域半径
        min_samples: 核心点最小邻居数

    Returns:
        labels: (N,) 整数数组，-1 表示噪声
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int32)
    if n < min_samples:
        return np.full(n, -1, dtype=np.int32)

    # 大数据集：先体素下采样，再对下采样点做 DBSCAN，最后将原近点分配
    if n > 5000:
        return _dbscan_with_downsample(points, eps, min_samples, max_points=5000)

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(dists <= eps)[0]

        if len(neighbors) < min_samples:
            visited[i] = True
            continue

        visited[i] = True
        labels[i] = cluster_id
        queue = list(neighbors)
        labels[neighbors] = cluster_id
        visited[neighbors] = True
        seed_idx = 0

        while seed_idx < len(queue):
            j = queue[seed_idx]
            seed_idx += 1

            j_dists = np.linalg.norm(points - points[j], axis=1)
            j_neighbors = np.where(j_dists <= eps)[0]

            if len(j_neighbors) >= min_samples:
                for nb in j_neighbors:
                    if not visited[nb]:
                        visited[nb] = True
                        labels[nb] = cluster_id
                        queue.append(nb)

        cluster_id += 1

    return labels


def _dbscan_with_downsample(points: np.ndarray, eps: float, min_samples: int, max_points: int = 5000) -> np.ndarray:
    """
    先体素下采样，再做 DBSCAN，最后将原全量点分配到最近的簇
    """
    n = len(points)
    # 体素下采样
    voxel_size = eps / 2.0
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    unique_coords, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    
    # 计算每个体素的代表点（均值）
    downsampled = []
    for idx in range(len(unique_coords)):
        mask = inverse == idx
        downsampled.append(points[mask].mean(axis=0))
    downsampled = np.array(downsampled)
    
    # 如果下采样后仍然太大，随机采样
    if len(downsampled) > max_points:
        indices = np.random.choice(len(downsampled), max_points, replace=False)
        downsampled = downsampled[indices]
    
    # 对下采样点做 DBSCAN
    ds_labels = _dbscan(downsampled, eps, min_samples)
    
    # 将原全量点分配到最近的簇（基于下采样点）
    labels = np.full(n, -1, dtype=np.int32)
    
    # 先根据体素反向映射
    for i in range(n):
        voxel_idx = inverse[i]
        ds_idx = voxel_idx
        if len(downsampled) < len(unique_coords):
            # 随机采样后需要重新找最近代表点
            distances = np.linalg.norm(downsampled - points[i], axis=1)
            ds_idx = int(np.argmin(distances))
        
        l = ds_labels[ds_idx]
        if l >= 0:
            labels[i] = l
    
    return labels


def split_instance_by_dbscan(
    points: np.ndarray,
    eps: float,
    min_samples: int = 3,
    metric: str = 'euclidean'
) -> List[np.ndarray]:
    """
    对单个实例的点云做 DBSCAN 分割

    Args:
        points: (N, 3) 点云坐标
        eps: DBSCAN 邻域半径
        min_samples: 核心点最小邻居数
        metric: 距离度量（仅兼容参数，只用 euclidean）

    Returns:
        分割后的点云 mask 列表，每个元素是布尔数组表示属于该子实例的点
    """
    if len(points) < min_samples:
        return [np.ones(len(points), dtype=bool)]

    labels = _dbscan(points, eps, min_samples)

    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) <= 1:
        return [np.ones(len(points), dtype=bool)]

    masks = []
    for label in sorted(unique_labels):
        mask = labels == label
        masks.append(mask)

    # 噪声点分配给最近的簇
    noise_mask = labels == -1
    if np.any(noise_mask):
        noise_points = points[noise_mask]
        cluster_centers = [points[m].mean(axis=0) for m in masks]

        noise_indices = np.where(noise_mask)[0]
        for idx, pt in zip(noise_indices, noise_points):
            distances = [np.linalg.norm(pt - c) for c in cluster_centers]
            nearest_cluster = int(np.argmin(distances))
            masks[nearest_cluster][idx] = True

    return masks


def split_instance_by_2d_dbscan(
    points: np.ndarray,
    eps: float,
    min_samples: int = 3
) -> List[np.ndarray]:
    """
    在 XY 平面上做 DBSCAN 分割
    """
    xy_points = points[:, :2].copy()
    return split_instance_by_dbscan(xy_points, eps, min_samples)


def split_instance_by_height(
    points: np.ndarray,
    z_gap_threshold: float = 2.0
) -> List[np.ndarray]:
    """
    按 Z 轴高度间隙分割实例
    """
    z_values = points[:, 2]
    sorted_indices = np.argsort(z_values)
    sorted_z = z_values[sorted_indices]

    z_diffs = np.diff(sorted_z)
    gap_indices = np.where(z_diffs > z_gap_threshold)[0]

    if len(gap_indices) == 0:
        return [np.ones(len(points), dtype=bool)]

    masks = []
    start_idx = 0
    for gap_idx in gap_indices:
        end_idx = gap_idx + 1
        mask = np.zeros(len(points), dtype=bool)
        mask[sorted_indices[start_idx:end_idx]] = True
        masks.append(mask)
        start_idx = end_idx

    mask = np.zeros(len(points), dtype=bool)
    mask[sorted_indices[start_idx:]] = True
    masks.append(mask)

    return masks


def split_instances_in_category(
    instances: List[InstanceData],
    eps: float,
    min_points: int = 3,
    min_split_size: int = 10,
    use_2d: bool = True
) -> List[InstanceData]:
    """
    对一个类别的所有实例进行分割
    """
    if eps is None or eps <= 0:
        return instances

    new_instances = []
    new_id = 0

    for inst in instances:
        if inst.num_points < min_split_size * 2:
            new_inst = InstanceData(
                instance_id=new_id,
                points=inst.points.copy(),
                category=inst.category
            )
            new_instances.append(new_inst)
            new_id += 1
            continue

        if use_2d:
            masks = split_instance_by_2d_dbscan(inst.points, eps, min_points)
        else:
            masks = split_instance_by_dbscan(inst.points, eps, min_points)

        if len(masks) == 1:
            new_inst = InstanceData(
                instance_id=new_id,
                points=inst.points.copy(),
                category=inst.category
            )
            new_instances.append(new_inst)
            new_id += 1
        else:
            for mask in masks:
                sub_points = inst.points[mask]
                if len(sub_points) >= min_split_size:
                    new_inst = InstanceData(
                        instance_id=new_id,
                        points=sub_points,
                        category=inst.category
                    )
                    new_instances.append(new_inst)
                    new_id += 1

    return new_instances


def split_instances_by_category_config(
    instances: List[InstanceData],
    category: str,
    config: Dict
) -> List[InstanceData]:
    """
    根据类别配置分割实例
    """
    eps = config.get('split_eps', None)
    min_points = config.get('split_min_points', 3)
    min_split_size = config.get('min_split_size', 10)
    use_2d = config.get('split_use_2d', True)

    if eps is None or eps <= 0:
        return instances

    return split_instances_in_category(
        instances, eps, min_points, min_split_size, use_2d
    )


if __name__ == "__main__":
    np.random.seed(42)

    center_a = np.array([0, 0, 0])
    points_a = center_a + np.random.randn(100, 3) * 0.5

    center_b = np.array([3, 0, 0])
    points_b = center_b + np.random.randn(100, 3) * 0.5

    all_points = np.vstack([points_a, points_b])

    print(f"Original instance: {len(all_points)} points")

    masks = split_instance_by_dbscan(all_points, eps=1.0, min_samples=3)
    print(f"Split into {len(masks)} sub-instances")
    for i, mask in enumerate(masks):
        print(f"  Sub-instance {i}: {mask.sum()} points")

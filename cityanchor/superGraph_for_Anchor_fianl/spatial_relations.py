"""
空间关系硬编码计算模块
基于地图/鸟瞰坐标系：
  -X (西) = 左
  +X (东) = 右
  +Y (南) = 前
  -Y (北) = 后
"""

import numpy as np
from typing import List, Tuple, Optional

# 阈值配置（收紧版本）
DIRECTION_THRESHOLD = 3.0  # 方向判断的最小距离差(米) - 从5.0收紧到3.0
ADJACENT_THRESHOLD = 5.0   # 相邻/靠近的最大距离(米) - 从10.0收紧到5.0
FAR_THRESHOLD = 30.0       # 远离的最小距离(米) - 从50.0收紧到30.0
INSIDE_THRESHOLD = 0.9     # inside判断的bbox重叠比例 - 从0.8收紧到0.9


class SpatialObject:
    """空间对象，包含中心点和bbox"""
    def __init__(self, id: str, category: str, center: np.ndarray, 
                 bbox_min: np.ndarray = None, bbox_max: np.ndarray = None):
        self.id = id
        self.category = category
        self.center = np.array(center)  # [x, y, z]
        self.bbox_min = bbox_min if bbox_min is not None else center
        self.bbox_max = bbox_max if bbox_max is not None else center
        
    def distance_to(self, other: 'SpatialObject') -> float:
        """计算到另一个对象的距离"""
        return np.linalg.norm(self.center - other.center)
    
    def is_inside_bbox(self, point: np.ndarray) -> bool:
        """检查点是否在自己的bbox内"""
        return np.all(point >= self.bbox_min) and np.all(point <= self.bbox_max)


# ==================== 方向类关系 ====================

def left_of(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """
    A 在 B 的左边（西）
    条件: A.x < B.x - threshold
    返回: (是否满足, 置信度分数)
    """
    diff = B.center[0] - A.center[0]  # B.x - A.x
    if diff > threshold:
        # 距离差越大，置信度越高（饱和到1.0）
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


def right_of(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """
    A 在 B 的右边（东）
    条件: A.x > B.x + threshold
    """
    diff = A.center[0] - B.center[0]  # A.x - B.x
    if diff > threshold:
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


def front_of(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """
    A 在 B 的前面（南）
    条件: A.y > B.y + threshold
    """
    diff = A.center[1] - B.center[1]  # A.y - B.y
    if diff > threshold:
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


def behind(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """
    A 在 B 的后面（北）
    条件: A.y < B.y - threshold
    """
    diff = B.center[1] - A.center[1]  # B.y - A.y
    if diff > threshold:
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


def north_of(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """A 在 B 的北边（同 behind）"""
    return behind(A, B, threshold)


def south_of(A: SpatialObject, B: SpatialObject, threshold: float = DIRECTION_THRESHOLD) -> Tuple[bool, float]:
    """A 在 B 的南边（同 front_of）"""
    return front_of(A, B, threshold)


# ==================== 拓扑类关系 ====================

def inside(A: SpatialObject, B: SpatialObject, ratio_threshold: float = INSIDE_THRESHOLD) -> Tuple[bool, float]:
    """
    A 在 B 的内部
    判断: A 的中心在 B 的 bbox 内，且 A 的 bbox 大部分在 B 内
    """
    # 中心点必须在 B 内
    if not B.is_inside_bbox(A.center):
        return False, 0.0
    
    # 计算 A 的 bbox 与 B 的 bbox 重叠比例
    overlap_min = np.maximum(A.bbox_min, B.bbox_min)
    overlap_max = np.minimum(A.bbox_max, B.bbox_max)
    
    if np.any(overlap_max < overlap_min):
        return False, 0.0
    
    overlap_volume = np.prod(overlap_max - overlap_min)
    A_volume = np.prod(A.bbox_max - A.bbox_min)
    
    if A_volume == 0:
        return False, 0.0
    
    overlap_ratio = overlap_volume / A_volume
    
    if overlap_ratio >= ratio_threshold:
        return True, min(1.0, overlap_ratio)
    return False, 0.0


def on_surface(A: SpatialObject, B: SpatialObject, dist_threshold: float = 2.0) -> Tuple[bool, float]:
    """
    A 在 B 的表面上（如车在 parking 上）
    判断: A 与 B 的距离很近，且 A 的 z 接近 B 的顶部
    """
    dist = A.distance_to(B)
    
    # 水平距离检查
    if dist > dist_threshold:
        return False, 0.0
    
    # z 高度检查: A 应该接近 B 的顶部
    height_diff = abs(A.center[2] - B.bbox_max[2])
    if height_diff < dist_threshold:
        score = 1.0 - (dist / dist_threshold) * 0.5
        return True, max(0.0, score)
    
    return False, 0.0


def between(A: SpatialObject, B: SpatialObject, C: SpatialObject) -> Tuple[bool, float]:
    """
    A 在 B 和 C 之间
    判断: A 到 B-C 连线中点的距离很小，且 A 在 B 和 C 的 bbox 包围盒内
    """
    # B-C 中点
    mid_point = (B.center + C.center) / 2
    dist_to_mid = np.linalg.norm(A.center - mid_point)
    
    # B-C 距离
    BC_dist = B.distance_to(C)
    
    if BC_dist == 0:
        return False, 0.0
    
    # A 到中点的距离应小于 B-C 距离的 1/3
    if dist_to_mid < BC_dist / 3:
        # 检查 A 是否在 B 和 C 的包围盒内
        bbox_min = np.minimum(B.bbox_min, C.bbox_min)
        bbox_max = np.maximum(B.bbox_max, C.bbox_max)
        
        if np.all(A.center >= bbox_min) and np.all(A.center <= bbox_max):
            score = 1.0 - (dist_to_mid / (BC_dist / 3))
            return True, max(0.0, score)
    
    return False, 0.0


def belonging(A: SpatialObject, B: SpatialObject, dist_threshold: float = ADJACENT_THRESHOLD) -> Tuple[bool, float]:
    """
    A 属于 B（如 parking of building）
    简化为: A 靠近 B 且在 B 的范围内
    """
    return adjacent(A, B, dist_threshold)


# ==================== 相邻类关系 ====================

def adjacent(A: SpatialObject, B: SpatialObject, threshold: float = ADJACENT_THRESHOLD) -> Tuple[bool, float]:
    """
    A 与 B 相邻/靠近
    判断: A 和 B 的距离 < threshold
    """
    dist = A.distance_to(B)
    
    if dist < threshold:
        # 距离越近，分数越高
        score = 1.0 - (dist / threshold)
        return True, score
    return False, 0.0


# next_to 是 adjacent 的别名
def next_to(A: SpatialObject, B: SpatialObject, threshold: float = ADJACENT_THRESHOLD) -> Tuple[bool, float]:
    return adjacent(A, B, threshold)


# ==================== 相对类关系 ====================

def opposite(A: SpatialObject, B: SpatialObject, 
             min_dist: float = FAR_THRESHOLD,
             max_dist: float = 200.0) -> Tuple[bool, float]:
    """
    A 在 B 的对面
    判断: A 和 B 距离较远，但面向相同方向
    """
    dist = A.distance_to(B)
    
    # 距离应该在 min_dist 和 max_dist 之间
    if min_dist <= dist <= max_dist:
        score = 1.0 - abs(dist - (min_dist + max_dist) / 2) / ((max_dist - min_dist) / 2)
        return True, max(0.0, score)
    return False, 0.0


def facing(A: SpatialObject, B: SpatialObject, 
           dist_threshold: float = FAR_THRESHOLD) -> Tuple[bool, float]:
    """
    A 面向 B
    简化版: A 在 B 的前方（假设 A 的正面朝向 B）
    """
    # 简化为 front_of 的反向: B 在 A 的前面 <=> A 面向 B
    return front_of(B, A, dist_threshold)


# ==================== 距离类关系 ====================

def closest_to(A: SpatialObject, B_list: List[SpatialObject], 
               target: SpatialObject) -> Tuple[bool, float]:
    """
    A 是 B_list 中距离 target 最近的
    返回: (A 是否是最近的, 相对优势分数)
    """
    if len(B_list) == 0:
        return False, 0.0
    
    # 计算所有距离
    distances = [(b, b.distance_to(target)) for b in B_list]
    distances.sort(key=lambda x: x[1])
    
    # A 是否是最近的
    if distances[0][0].id == A.id:
        # 如果有多个，计算相对优势
        if len(distances) > 1:
            advantage = distances[1][1] - distances[0][1]
            score = min(1.0, advantage / 10.0)  # 优势越大分数越高
        else:
            score = 1.0
        return True, score
    
    return False, 0.0


def far_from(A: SpatialObject, B: SpatialObject, threshold: float = FAR_THRESHOLD) -> Tuple[bool, float]:
    """
    A 远离 B
    判断: A 和 B 的距离 > threshold
    """
    dist = A.distance_to(B)
    
    if dist > threshold:
        # 距离越远，分数越高（饱和）
        score = min(1.0, (dist - threshold) / threshold)
        return True, score
    return False, 0.0


# ==================== 垂直类关系 ====================

def above(A: SpatialObject, B: SpatialObject, threshold: float = 2.0) -> Tuple[bool, float]:
    """
    A 在 B 的上方
    判断: A.z > B.z + threshold
    """
    diff = A.center[2] - B.center[2]
    if diff > threshold:
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


def below(A: SpatialObject, B: SpatialObject, threshold: float = 2.0) -> Tuple[bool, float]:
    """
    A 在 B 的下方
    判断: A.z < B.z - threshold
    """
    diff = B.center[2] - A.center[2]
    if diff > threshold:
        score = min(1.0, diff / (threshold * 3))
        return True, score
    return False, 0.0


# ==================== 序数类关系 ====================

def nth_from_left(A: SpatialObject, candidates: List[SpatialObject], n: int) -> Tuple[bool, float]:
    """
    A 是从左数第 n 个
    判断: 将 candidates 按 X 排序，A 是第 n 个（从1开始计数）
    """
    if len(candidates) < n:
        return False, 0.0
    
    # 按 X 坐标排序（从左到右）
    sorted_candidates = sorted(candidates, key=lambda x: x.center[0])
    
    # 检查 A 是否是第 n 个
    if sorted_candidates[n-1].id == A.id:
        # 分数基于位置准确性
        score = 1.0
        return True, score
    
    return False, 0.0


def nth_from_right(A: SpatialObject, candidates: List[SpatialObject], n: int) -> Tuple[bool, float]:
    """
    A 是从右数第 n 个
    """
    if len(candidates) < n:
        return False, 0.0
    
    # 按 X 坐标降序排序（从右到左）
    sorted_candidates = sorted(candidates, key=lambda x: x.center[0], reverse=True)
    
    if sorted_candidates[n-1].id == A.id:
        return True, 1.0
    
    return False, 0.0


def nth_from_front(A: SpatialObject, candidates: List[SpatialObject], n: int) -> Tuple[bool, float]:
    """
    A 是从前数第 n 个（Y 坐标从大到小）
    """
    if len(candidates) < n:
        return False, 0.0
    
    sorted_candidates = sorted(candidates, key=lambda x: x.center[1], reverse=True)
    
    if sorted_candidates[n-1].id == A.id:
        return True, 1.0
    
    return False, 0.0


def nth_from_back(A: SpatialObject, candidates: List[SpatialObject], n: int) -> Tuple[bool, float]:
    """
    A 是从后数第 n 个（Y 坐标从小到大）
    """
    if len(candidates) < n:
        return False, 0.0
    
    sorted_candidates = sorted(candidates, key=lambda x: x.center[1])
    
    if sorted_candidates[n-1].id == A.id:
        return True, 1.0
    
    return False, 0.0


# ==================== 角落/边缘类关系 ====================

def at_corner(A: SpatialObject, B: SpatialObject, threshold: float = 2.0) -> Tuple[bool, float]:
    """
    A 在 B 的角落
    判断: A 靠近 B 的 bbox 角落
    """
    # 获取 B 的四个角落点（XY平面）
    corners = [
        np.array([B.bbox_min[0], B.bbox_min[1], B.center[2]]),  # 左后
        np.array([B.bbox_max[0], B.bbox_min[1], B.center[2]]),  # 右后
        np.array([B.bbox_min[0], B.bbox_max[1], B.center[2]]),  # 左前
        np.array([B.bbox_max[0], B.bbox_max[1], B.center[2]]),  # 右前
    ]
    
    # 检查 A 是否靠近任何一个角落
    min_dist = min(np.linalg.norm(A.center - corner) for corner in corners)
    
    if min_dist < threshold:
        score = 1.0 - (min_dist / threshold)
        return True, max(0.0, score)
    
    return False, 0.0


def near_corner(A: SpatialObject, B: SpatialObject, threshold: float = 3.0) -> Tuple[bool, float]:
    """
    A 在 B 的角落附近（比 at_corner 更宽松）
    """
    return at_corner(A, B, threshold)


def at_end(A: SpatialObject, B: SpatialObject, threshold: float = 3.0) -> Tuple[bool, float]:
    """
    A 在 B 的尽头/末端
    判断: A 靠近 B 的 Y 方向末端（前或后）
    """
    # 检查距离 B 的前边或后边的距离
    dist_to_front = abs(A.center[1] - B.bbox_max[1])
    dist_to_back = abs(A.center[1] - B.bbox_min[1])
    
    min_dist = min(dist_to_front, dist_to_back)
    
    if min_dist < threshold:
        score = 1.0 - (min_dist / threshold)
        return True, max(0.0, score)
    
    return False, 0.0


def on_edge(A: SpatialObject, B: SpatialObject, threshold: float = 5.0) -> Tuple[bool, float]:
    """
    A 在 B 的边缘上
    判断: A 在 B 的 bbox 边界附近
    """
    # 检查是否在边界上
    on_x_min = abs(A.center[0] - B.bbox_min[0]) < threshold
    on_x_max = abs(A.center[0] - B.bbox_max[0]) < threshold
    on_y_min = abs(A.center[1] - B.bbox_min[1]) < threshold
    on_y_max = abs(A.center[1] - B.bbox_max[1]) < threshold
    
    if on_x_min or on_x_max or on_y_min or on_y_max:
        return True, 0.9
    
    return False, 0.0


# ==================== 沿线类关系 ====================

def along(A: SpatialObject, B: SpatialObject, threshold: float = 15.0) -> Tuple[bool, float]:
    """
    A 沿着 B（如沿着街道）
    判断: A 与 B 的距离较近，且 A 在 B 的延长线上
    """
    dist = A.distance_to(B)
    
    if dist < threshold:
        score = 1.0 - (dist / threshold)
        return True, max(0.0, score)
    
    return False, 0.0


# ==================== 外部类关系 ====================

def outside(A: SpatialObject, B: SpatialObject, threshold: float = 5.0) -> Tuple[bool, float]:
    """
    A 在 B 的外部
    判断: A 的中心在 B 的 bbox 外部
    """
    if not B.is_inside_bbox(A.center):
        # 计算到 B 边界的距离
        dx = max(B.bbox_min[0] - A.center[0], 0, A.center[0] - B.bbox_max[0])
        dy = max(B.bbox_min[1] - A.center[1], 0, A.center[1] - B.bbox_max[1])
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > threshold:
            return True, min(1.0, dist / (threshold * 2))
    
    return False, 0.0


# ==================== 包围类关系 ====================

def surrounded_by(A: SpatialObject, B: SpatialObject, threshold: float = 30.0) -> Tuple[bool, float]:
    """
    A 被 B 包围/环绕
    判断: A 在 B 的范围内，且 B 比 A 大很多
    """
    # 检查 A 是否在 B 内
    if not B.is_inside_bbox(A.center):
        return False, 0.0
    
    # 检查 B 是否比 A 大很多
    B_volume = np.prod(B.bbox_max - B.bbox_min)
    A_volume = np.prod(A.bbox_max - A.bbox_min)
    
    if B_volume > A_volume * 5:  # B 至少是 A 的 5 倍大
        return True, min(1.0, B_volume / (A_volume * 10))
    
    return False, 0.0


# ==================== 连接类关系 ====================

def connected_to(A: SpatialObject, B: SpatialObject, threshold: float = 5.0) -> Tuple[bool, float]:
    """
    A 连接到 B
    判断: A 和 B 的 bbox 相连或重叠
    """
    # 检查 bbox 是否重叠或相邻
    overlap_min = np.maximum(A.bbox_min, B.bbox_min)
    overlap_max = np.minimum(A.bbox_max, B.bbox_max)
    
    # 检查是否接触（允许小间隙）
    gap = np.maximum(0, overlap_min - overlap_max)
    max_gap = np.max(gap)
    
    if max_gap < threshold:
        score = 1.0 - (max_gap / threshold)
        return True, max(0.0, score)
    
    return False, 0.0


# ==================== 侧面类关系 ====================

def on_side(A: SpatialObject, B: SpatialObject, threshold: float = 10.0) -> Tuple[bool, float]:
    """
    A 在 B 的侧面
    简化为: adjacent 或 on_edge
    """
    # 先检查 on_edge
    is_on_edge, edge_score = on_edge(A, B, threshold)
    if is_on_edge:
        return True, edge_score
    
    # 再检查 adjacent
    return adjacent(A, B, threshold)


# ==================== 朝向类关系 ====================

def towards(A: SpatialObject, B: SpatialObject, threshold: float = FAR_THRESHOLD) -> Tuple[bool, float]:
    """
    A 朝向 B
    简化为: A 在 B 的某个方向上且距离适中
    """
    return facing(A, B, threshold)


# ==================== 关系函数映射表 ====================

RELATION_FUNCTIONS = {
    'left_of': left_of,
    'right_of': right_of,
    'front_of': front_of,
    'behind': behind,
    'north_of': north_of,
    'south_of': south_of,
    'above': above,
    'below': below,
    'inside': inside,
    'on_surface': on_surface,
    'between': between,
    'belonging': belonging,
    'adjacent': adjacent,
    'next_to': next_to,
    'opposite': opposite,
    'facing': facing,
    'closest_to': closest_to,
    'far_from': far_from,
    'nth_from_left': nth_from_left,
    'nth_from_right': nth_from_right,
    'nth_from_front': nth_from_front,
    'nth_from_back': nth_from_back,
    'at_corner': at_corner,
    'near_corner': near_corner,
    'at_end': at_end,
    'on_edge': on_edge,
    'along': along,
    'outside': outside,
    'surrounded_by': surrounded_by,
    'connected_to': connected_to,
    'on_side': on_side,
    'towards': towards,
}


def compute_relation(relation_name: str, *args, **kwargs) -> Tuple[bool, float]:
    """
    通用关系计算接口
    
    Args:
        relation_name: 关系名称
        *args: 参数 (SpatialObject 或列表)
        **kwargs: 额外参数 (如 n=3 表示第3个)
    
    Returns:
        (是否满足, 置信度分数)
    """
    if relation_name not in RELATION_FUNCTIONS:
        return False, 0.0
    
    func = RELATION_FUNCTIONS[relation_name]
    
    # 特殊处理序数关系
    if relation_name in ['nth_from_left', 'nth_from_right', 'nth_from_front', 'nth_from_back']:
        # 需要额外参数 n
        n = kwargs.get('n', 1)
        return func(*args, n=n)
    
    return func(*args)

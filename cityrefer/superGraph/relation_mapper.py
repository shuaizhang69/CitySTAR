"""
关键词到核心空间关系的映射模块
将 1233 种描述映射到 14 种核心关系
"""

from typing import Optional, Tuple
import re

# ==================== 关键词映射表 ====================
# 按优先级排序，优先匹配更具体的短语

RELATION_KEYWORDS = {
    # ========== 序数类 (Ordinal) ==========
    'nth_from_left': [
        'first from the left',
        'second from the left',
        'third from the left',
        'fourth from the left',
        'fifth from the left',
        'sixth from the left',
        'seventh from the left',
        'eighth from the left',
        'ninth from the left',
        'tenth from the left',
        '1st from the left',
        '2nd from the left',
        '3rd from the left',
        '4th from the left',
        '5th from the left',
        '6th from the left',
        '7th from the left',
        '8th from the left',
        '9th from the left',
        '10th from the left',
        'from the left',  # 通用
    ],
    
    'nth_from_right': [
        'first from the right',
        'second from the right',
        'third from the right',
        'fourth from the right',
        'fifth from the right',
        'sixth from the right',
        'seventh from the right',
        'eighth from the right',
        'ninth from the right',
        'tenth from the right',
        '1st from the right',
        '2nd from the right',
        '3rd from the right',
        '4th from the right',
        '5th from the right',
        '6th from the right',
        '7th from the right',
        '8th from the right',
        '9th from the right',
        '10th from the right',
        'from the right',  # 通用
    ],
    
    'nth_from_front': [
        'first from the front',
        'second from the front',
        'third from the front',
        'fourth from the front',
        'fifth from the front',
        '1st from the front',
        '2nd from the front',
        '3rd from the front',
        '4th from the front',
        '5th from the front',
        'from the front',
    ],
    
    'nth_from_back': [
        'first from the back',
        'second from the back',
        'third from the back',
        'fourth from the back',
        'fifth from the back',
        '1st from the back',
        '2nd from the back',
        '3rd from the back',
        '4th from the back',
        '5th from the back',
        'from the back',
    ],
    
    'in_the_row_of': [
        'in the row of',
        'in the middle row of',
        'in the middle',
        'in the third row',
        'in the second row',
    ],
    
    # ========== 居中/位置 ==========
    'between': [
        'between',
        'in between',
        'in the middle of',
        'in the middle',
        'middle of',
    ],
    
    # ========== 最近 ==========
    'closest_to': [
        'closest to',
        'closest to the intersection with',
        'closest',
        'nearest to',
        'nearest',
    ],
    
    # ========== 前后方向 ==========
    'front_of': [
        'parked in front of',
        'in front of',
        'in front',
        'at the front of',
        'at the front',
        'facing',
        'facing the',
        'directly in front of',
        'on the front',
        'at the front',
    ],
    
    'behind': [
        'behind it',
        'behind',
        'at the back of',
        'at the back',
        'back of',
        'back',
        'against the back of',
    ],
    
    # ========== 左右方向 ==========
    'left_of': [
        'on the left side of',
        'at the left side of',
        'to the left of',
        'on the left of',
        'on the left',
        'on its left',
        'on it\'s left',
        'to the left',
        'left of',
        'left',
    ],
    
    'right_of': [
        'on the right side of',
        'at the right side of',
        'to the right of',
        'on the right of',
        'on the right',
        'on its right',
        'on it\'s right',
        'to the right',
        'right of',
        'right',
    ],
    
    # ========== 南北方向 ==========
    'north_of': [
        'north of',
        'northern',
        'to the north of',
        'northern side of',
    ],
    
    'south_of': [
        'south of',
        'southern',
        'to the south of',
        'southern side of',
    ],
    
    # ========== 垂直方向 ==========
    'above': [
        'above',
        'over',
        'on top of',
        'higher than',
    ],
    
    'below': [
        'below',
        'under',
        'beneath',
        'lower than',
    ],
    
    # ========== 内部 ==========
    'inside': [
        'parked in the',
        'in the',
        'parked in',
        'located in',
        'within',
        'inside',
        'into',
        'in the driveway of',
        'in the corner of',
        'in the row of',
    ],
    
    # ========== 表面 ==========
    'on_surface': [
        'parked on the',
        'located on the',
        'on the',
        'parked on',
        'located on',
        'on',
    ],
    
    # ========== 所属 ==========
    'belonging': [
        'off of',
        'off',
        'of the',
        'of',
        'for',
    ],
    
    # ========== 相邻/靠近 ==========
    'adjacent': [
        'parked next to',
        'right next to',
        'next to',
        'adjacent to',
        'beside',
        'by',
        'near the corner of',
        'near',
        'close to',
        'closed to',
        'parked near',
        'bordered by',
    ],
    
    # ========== 对面 ==========
    'opposite': [
        'across the street from',
        'on the opposite side of',
        'across from',
        'opposite',
        'across to',
        'across',
        'on the other side of',
        'on the other side',
        'on the other',
        'on both sides of',
        'on one side of',
        'on one side',
    ],
    
    # ========== 远离 ==========
    'far_from': [
        'away from',
        'far from',
        'distant from',
    ],
    
    # ========== 角落 ==========
    'at_corner': [
        'at the corner of',
        'on the corner of',
        'at the intersection of',
        'at the intersection with',
        'corner of',
        'from the corner of',
    ],
    
    'near_corner': [
        'near the corner of',
        'around the corner of',
    ],
    
    # ========== 沿线 ==========
    'along': [
        'along',
        'along the',
        'alongside',
    ],
    
    # ========== 尽头/边缘 ==========
    'at_end': [
        'at the end of',
        'on the end of',
        'end of',
    ],
    
    'on_edge': [
        'on the edge of',
        'at the edge of',
        'edge of',
    ],
    
    # ========== 外部 ==========
    'outside': [
        'outside',
        'outside of',
        'out of',
    ],
    
    # ========== 包围/周围 ==========
    'surrounded_by': [
        'surrounded by',
        'surrounding',
        'around',
        'all around',
    ],
    
    # ========== 连接 ==========
    'connected_to': [
        'connected to',
        'attached to',
        'linked to',
        'joining',
    ],
    
    # ========== 侧面 ==========
    'on_side': [
        'on the side of',
        'at the side of',
        'side of',
    ],
    
    # ========== 朝向 ==========
    'towards': [
        'towards',
        'toward',
        'facing towards',
    ],
    
    # ========== 来自 ==========
    'coming_from': [
        'coming from',
        'from',
        'originating from',
    ],
}

# 可以忽略的关系（模糊/无效）
AMBIGUOUS_KEYWORDS = [
    'and',
    'for',
    'with',
    'from',
    'at',
    'at the',
    'to',
    'along',
    'around',
    'coming from',
    'towards',
    'through',
    'up',
    'down',
    'via',
    'perpendicular to',
    'parallel to',
    'on one side',
    'on one side of',
]


def map_relation(raw_relation: str) -> Optional[str]:
    """
    将原始关系字符串映射到核心关系
    
    Args:
        raw_relation: 原始描述中的关系，如 "in front of"
    
    Returns:
        核心关系名称，如 "front_of"；无法映射返回 None
    """
    if not raw_relation:
        return None
    
    # 标准化：小写 + 去除首尾空格
    normalized = raw_relation.lower().strip()
    
    # 先检查是否在模糊列表中
    for ambig in AMBIGUOUS_KEYWORDS:
        if normalized == ambig or normalized.startswith(ambig + ' '):
            return None
    
    # 按优先级匹配关键词
    for relation, keywords in RELATION_KEYWORDS.items():
        for keyword in keywords:
            # 完全匹配或开头匹配
            if normalized == keyword or normalized.startswith(keyword + ' '):
                return relation
    
    # 未匹配到任何关系
    return None


def parse_spatial_relation(raw_relation: str, reference_anchor: str) -> Tuple[Optional[str], dict]:
    """
    解析完整的空间关系
    
    Args:
        raw_relation: 原始关系字符串
        reference_anchor: 参照物锚点
    
    Returns:
        (核心关系名称, 额外信息字典)
    """
    relation = map_relation(raw_relation)
    
    info = {
        'raw': raw_relation,
        'anchor': reference_anchor,
        'parsed': relation,
    }
    
    # 特殊处理：提取额外的序数信息
    if relation == 'closest_to':
        # 提取 "nth from the left/right" 信息
        ordinal_pattern = r'(\w+) from the (left|right)'
        match = re.search(ordinal_pattern, raw_relation.lower())
        if match:
            info['ordinal'] = match.group(1)  # e.g., "ninth"
            info['direction'] = match.group(2)  # e.g., "left"
    
    # 特殊处理：between 关系
    if relation == 'between' and reference_anchor:
        # 解析 "A and B" 或 "A, B and C"
        anchors = [a.strip() for a in re.split(r'\s+and\s+|,\s*', reference_anchor)]
        info['anchors'] = anchors
    
    return relation, info


def get_supported_relations() -> list:
    """返回所有支持的核心关系列表"""
    return list(RELATION_KEYWORDS.keys())


def get_relation_coverage() -> dict:
    """
    统计映射表覆盖率（用于调试）
    返回每种关系覆盖的关键词数量
    """
    return {k: len(v) for k, v in RELATION_KEYWORDS.items()}


if __name__ == "__main__":
    # 测试
    test_cases = [
        "in front of",
        "to the left of",
        "between",
        "next to",
        "of",
        "for",
        "parked in front of",
        "ninth from the left",
        "at the intersection with",
    ]
    
    print("=== 关系映射测试 ===")
    for case in test_cases:
        result = map_relation(case)
        print(f"'{case}' -> {result}")
    
    print("\n=== 关系覆盖率 ===")
    for rel, count in sorted(get_relation_coverage().items(), key=lambda x: -x[1]):
        print(f"{rel}: {count} keywords")

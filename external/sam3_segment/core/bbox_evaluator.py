"""
BBox Evaluator - 计算 3D bbox 的 IoU、recall、precision
用于评估生成的 bbox 与 GT bbox 的匹配程度
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict


def compute_iou_3d(box_a: List[float], box_b: List[float]) -> float:
    """
    计算两个 3D bbox 的 IoU

    Args:
        box_a: [cx, cy, cz, dx, dy, dz, ...]
        box_b: [cx, cy, cz, dx, dy, dz, ...]

    Returns:
        IoU 值 [0, 1]
    """
    ca, sa = np.array(box_a[:3]), np.array(box_a[3:6])
    cb, sb = np.array(box_b[:3]), np.array(box_b[3:6])

    min_a = ca - sa / 2
    max_a = ca + sa / 2
    min_b = cb - sb / 2
    max_b = cb + sb / 2

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)

    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter_size)

    vol_a = np.prod(sa)
    vol_b = np.prod(sb)
    union_vol = vol_a + vol_b - inter_vol

    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def compute_iou_2d(box_a: List[float], box_b: List[float]) -> float:
    """
    计算两个 3D bbox 在 XY 平面的 2D IoU

    Args:
        box_a: [cx, cy, cz, dx, dy, dz, ...]
        box_b: [cx, cy, cz, dx, dy, dz, ...]

    Returns:
        2D IoU 值 [0, 1]
    """
    ca, sa = np.array(box_a[:2]), np.array(box_a[3:5])
    cb, sb = np.array(box_b[:2]), np.array(box_b[3:5])

    min_a = ca - sa / 2
    max_a = ca + sa / 2
    min_b = cb - sb / 2
    max_b = cb + sb / 2

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)

    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_area = np.prod(inter_size)

    area_a = np.prod(sa)
    area_b = np.prod(sb)
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_bbox_center_distance(box_a: List[float], box_b: List[float], use_2d: bool = True) -> float:
    """
    计算两个 bbox 中心点的距离

    Args:
        box_a: [cx, cy, cz, dx, dy, dz, ...]
        box_b: [cx, cy, cz, dx, dy, dz, ...]
        use_2d: 是否只计算 XY 平面距离

    Returns:
        欧氏距离
    """
    if use_2d:
        return float(np.linalg.norm(np.array(box_a[:2]) - np.array(box_b[:2])))
    else:
        return float(np.linalg.norm(np.array(box_a[:3]) - np.array(box_b[:3])))


def match_bboxes_greedy(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    iou_threshold: float = 0.5,
    use_2d: bool = False
) -> Tuple[int, int, int]:
    """
    使用 greedy 匹配计算 recall 和 precision

    每个 GT 框匹配最佳 pred 框，匹配成功后 pred 框被移除（保证 1-to-1）

    Args:
        gt_boxes: GT bbox 列表，每个元素包含 'bbox' 和 'object_name'
        pred_boxes: pred bbox 列表
        iou_threshold: IoU 阈值
        use_2d: 是否使用 2D IoU

    Returns:
        (matched_count, gt_count, pred_count)
    """
    if not gt_boxes or not pred_boxes:
        return 0, len(gt_boxes), len(pred_boxes)

    matched = 0
    used_pred = set()

    iou_func = compute_iou_2d if use_2d else compute_iou_3d

    for gt in gt_boxes:
        best_iou = 0.0
        best_idx = -1

        for idx, pred in enumerate(pred_boxes):
            if idx in used_pred:
                continue

            iou = iou_func(gt['bbox'], pred['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            matched += 1
            used_pred.add(best_idx)

    return matched, len(gt_boxes), len(pred_boxes)


def evaluate_bboxes(
    pred_json_path: str,
    gt_json_path: str,
    iou_threshold: float = 0.5,
    use_2d: bool = False,
    categories: Optional[List[str]] = None,
    eval_categories: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    评估预测 bbox 与 GT bbox 的匹配情况

    Args:
        pred_json_path: 预测 bbox JSON 路径
        gt_json_path: GT bbox JSON 路径
        iou_threshold: IoU 阈值
        use_2d: 是否使用 2D IoU
        categories: 要评估的类别列表，None 表示评估所有 GT 中存在的类别

    Returns:
        按类别组织的结果字典：
        {
            "Car": {"recall": 0.65, "precision": 0.40, "gt_count": 100, "pred_count": 160, "matched": 65},
            ...
        }
    """
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)

    gt_bboxes = gt_data.get('bboxes', [])
    pred_bboxes = pred_data.get('bboxes', [])

    # 按 object_name 分组
    gt_by_cat = defaultdict(list)
    pred_by_cat = defaultdict(list)

    for b in gt_bboxes:
        gt_by_cat[b['object_name']].append(b)

    for b in pred_bboxes:
        pred_by_cat[b['object_name']].append(b)

    # 确定要评估的类别
    if eval_categories is not None:
        eval_cats = set(eval_categories)
    elif categories is not None:
        eval_cats = set(categories)
    else:
        eval_cats = set(gt_by_cat.keys())

    results = {}
    for cat in sorted(eval_cats):
        gt_boxes = gt_by_cat.get(cat, [])
        pred_boxes = pred_by_cat.get(cat, [])

        matched, gt_count, pred_count = match_bboxes_greedy(
            gt_boxes, pred_boxes, iou_threshold, use_2d
        )

        results[cat] = {
            'recall': matched / gt_count if gt_count > 0 else 0.0,
            'precision': matched / pred_count if pred_count > 0 else 0.0,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'matched': matched,
        }

    return results


def evaluate_multiple_scenes(
    pred_dir: str,
    gt_dir: str,
    scene_ids: List[str],
    iou_threshold: float = 0.5,
    use_2d: bool = False,
    categories: Optional[List[str]] = None,
    eval_categories: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    评估多个场景的 bbox，汇总各类别的总体指标

    Args:
        pred_dir: 预测 bbox JSON 目录
        gt_dir: GT bbox JSON 目录
        scene_ids: 场景 ID 列表
        iou_threshold: IoU 阈值
        use_2d: 是否使用 2D IoU
        categories: 要评估的类别列表

    Returns:
        汇总后的结果字典
    """
    aggregated = defaultdict(lambda: {
        'gt_count': 0,
        'pred_count': 0,
        'matched': 0
    })

    for scene_id in scene_ids:
        pred_path = Path(pred_dir) / f"{scene_id}_bbox.json"
        gt_path = Path(gt_dir) / f"{scene_id}_bbox.json"

        if not pred_path.exists():
            print(f"Warning: pred file not found: {pred_path}")
            continue

        if not gt_path.exists():
            print(f"Warning: GT file not found: {gt_path}")
            continue

        results = evaluate_bboxes(
            str(pred_path), str(gt_path),
            iou_threshold, use_2d, categories
        )

        for cat, metrics in results.items():
            aggregated[cat]['gt_count'] += metrics['gt_count']
            aggregated[cat]['pred_count'] += metrics['pred_count']
            aggregated[cat]['matched'] += metrics['matched']

    # 计算最终的 recall 和 precision
    final_results = {}
    for cat, data in aggregated.items():
        gt_count = data['gt_count']
        pred_count = data['pred_count']
        matched = data['matched']

        final_results[cat] = {
            'recall': matched / gt_count if gt_count > 0 else 0.0,
            'precision': matched / pred_count if pred_count > 0 else 0.0,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'matched': matched,
        }

    return final_results


def print_evaluation_results(results: Dict[str, Dict], title: str = "Evaluation Results"):
    """
    打印评估结果表格
    """
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")
    print(f"{'Category':<20} {'GT':>8} {'Pred':>8} {'Matched':>8} {'Recall':>8} {'Precision':>10}")
    print("-" * 70)

    for cat in sorted(results.keys()):
        r = results[cat]
        print(f"{cat:<20} {r['gt_count']:>8} {r['pred_count']:>8} {r['matched']:>8} "
              f"{r['recall']:>8.2f} {r['precision']:>10.2f}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BBox Evaluator")
    parser.add_argument("--pred", type=str, required=True, help="Predicted bbox JSON path or directory")
    parser.add_argument("--gt", type=str, required=True, help="GT bbox JSON path or directory")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--use_2d", action="store_true", help="Use 2D IoU instead of 3D")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated categories to evaluate")

    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else None

    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    if pred_path.is_file() and gt_path.is_file():
        results = evaluate_bboxes(
            str(pred_path), str(gt_path),
            args.iou, args.use_2d, categories
        )
        print_evaluation_results(results, "Single Scene Evaluation")
    elif pred_path.is_dir() and gt_path.is_dir():
        # 找到两个目录共有的场景
        pred_scenes = {p.stem.replace('_bbox', '') for p in pred_path.glob("*_bbox.json")}
        gt_scenes = {p.stem.replace('_bbox', '') for p in gt_path.glob("*_bbox.json")}
        common_scenes = sorted(pred_scenes & gt_scenes)

        print(f"Found {len(common_scenes)} common scenes")

        results = evaluate_multiple_scenes(
            str(pred_path), str(gt_path), common_scenes,
            args.iou, args.use_2d, categories
        )
        print_evaluation_results(results, "Multi-Scene Evaluation")
    else:
        print("Error: Both pred and gt should be files or directories")
        exit(1)

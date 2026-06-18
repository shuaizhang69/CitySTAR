"""
CityRefer 集成脚本
复用现有的 inference.py 和 fusion.py
只需要按需渲染网格块 + 运行已有 pipeline
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Set, Dict
import numpy as np


def load_box3d(scene_name: str, box3d_root: str) -> Dict:
    """加载 box3d 数据"""
    path = os.path.join(box3d_root, f"{scene_name}_bbox.json")
    with open(path, 'r') as f:
        return json.load(f)


def candidates_to_grid_blocks(
    candidates: List[str],
    box3d_data: Dict,
    grid_size: int = 50
) -> Set[str]:
    """
    从候选 object_id 映射到中心网格块
    renderer.py 会自动加载周围 450 米的上下文
    """
    blocks = set()
    obj_to_bbox = {obj['object_id']: obj for obj in box3d_data['bboxes']}

    for obj_id_str in candidates:
        obj_id = int(obj_id_str)
        if obj_id not in obj_to_bbox:
            continue

        obj = obj_to_bbox[obj_id]
        x, y = obj['bbox'][:2]

        # 只计算中心网格块（renderer 会自动加载周围）
        gx = int(np.floor(x / grid_size)) * grid_size
        gy = int(np.floor(y / grid_size)) * grid_size
        blocks.add(f"x{gx}_y{gy}")

    return blocks


def run_renderer(input_folder: str, output_root: str, blocks: Set[str], gpu: str = "0"):
    """
    运行 renderer，只渲染指定的网格块
    通过修改 metadata.json 来临时指定要处理的块
    """
    # 读取原始 metadata
    meta_path = os.path.join(input_folder, "metadata.json")
    with open(meta_path, 'r') as f:
        full_metadata = json.load(f)

    # 过滤只保留需要的块
    filtered_metadata = {}
    for fname, info in full_metadata.items():
        ox, oy = info['offset'][:2]
        block_id = f"x{int(ox)}_y{int(oy)}"
        if block_id in blocks:
            filtered_metadata[fname] = info

    if not filtered_metadata:
        print("[Warning] No blocks to render after filtering")
        return

    # 保存临时 metadata
    temp_meta_path = os.path.join(input_folder, "metadata_temp.json")
    with open(temp_meta_path, 'w') as f:
        json.dump(filtered_metadata, f)

    print(f"[Renderer] Processing {len(filtered_metadata)} blocks: {blocks}")

    # 调用 renderer
    cmd = [
        "python", "core/renderer.py",
        "-i", input_folder,
        "-o", output_root,
        "--gpus", gpu,
        "--num_workers", "1"
    ]

    # 临时替换 metadata
    orig_meta_path = os.path.join(input_folder, "metadata.json")
    backup_meta_path = os.path.join(input_folder, "metadata_backup.json")

    try:
        # 备份原 metadata
        os.rename(orig_meta_path, backup_meta_path)
        # 使用过滤后的 metadata
        os.rename(temp_meta_path, orig_meta_path)

        # 运行 renderer
        subprocess.run(cmd, check=True)

    finally:
        # 恢复原 metadata
        if os.path.exists(backup_meta_path):
            if os.path.exists(orig_meta_path):
                os.remove(orig_meta_path)
            os.rename(backup_meta_path, orig_meta_path)


def run_inference(render_root: str, mask_root: str, mode: str = "all", gpu: str = "0"):
    """
    运行 inference.py
    """
    cmd = [
        "python", "core/inference.py",
        "--render_root", render_root,
        "--output_root", mask_root,
        "--mode", mode,
        "--gpu", gpu,
        "--score_thresh", "0.35"
    ]

    print(f"[Inference] Running SAM3 inference...")
    subprocess.run(cmd, check=True)


def run_fusion(cluster_path: str, snap_root: str, mask_root: str, output_root: str, gpu: str = "0"):
    """
    运行 fusion.py
    """
    cmd = [
        "python", "core/fusion.py",
        "--parallel",
        "--cluster_path", cluster_path,
        "--snap_root", snap_root,
        "--mask_root", mask_root,
        "--raw_ply_folder", "dummy",  # fusion 需要这个参数但不实际使用
        "--output_root", output_root,
        "--gpus", gpu
    ]

    print(f"[Fusion] Running fusion...")
    subprocess.run(cmd, check=True)


def compute_spatial_relation_score(
    candidate_id: str,
    construction: List[Dict],
    box3d_data: Dict
) -> float:
    """
    基于 GT box3d 计算空间关系匹配分数
    （简化版，后续可以加入 SAM3 实例）
    """
    obj_to_bbox = {obj['object_id']: obj for obj in box3d_data['bboxes']}

    cand_id = int(candidate_id)
    if cand_id not in obj_to_bbox:
        return 0.0

    cand_obj = obj_to_bbox[cand_id]
    cand_center = np.array(cand_obj['bbox'][:3])

    # 提取 construction 中的关系
    relations_matched = 0
    total_relations = 0

    for item in construction:
        if item.get('is_main', False):
            continue

        spatial_rel = item.get('spatial_relation', '')
        if not spatial_rel:
            continue

        total_relations += 1

        # 找到参考物体
        ref_category = item.get('category2', item.get('category', ''))

        # 在周围找同类物体
        for other_id, other_obj in obj_to_bbox.items():
            if other_id == cand_id:
                continue
            if other_obj['object_name'].lower() != ref_category.lower():
                continue

            # 计算相对位置
            other_center = np.array(other_obj['bbox'][:3])
            delta = cand_center - other_center

            # 简单的方向判断
            if 'front' in spatial_rel.lower() and delta[1] > 0:
                relations_matched += 1
                break
            elif 'behind' in spatial_rel.lower() and delta[1] < 0:
                relations_matched += 1
                break
            elif 'left' in spatial_rel.lower() and delta[0] < 0:
                relations_matched += 1
                break
            elif 'right' in spatial_rel.lower() and delta[0] > 0:
                relations_matched += 1
                break
            elif 'near' in spatial_rel.lower() or 'close' in spatial_rel.lower():
                dist = np.linalg.norm(delta)
                if dist < 20:  # 20米内算 near
                    relations_matched += 1
                    break

    if total_relations == 0:
        return 1.0

    return relations_matched / total_relations


def process_annotation(
    ann: Dict,
    input_folder: str,
    box3d_root: str,
    render_root: str,
    mask_root: str,
    cluster_path: str,
    fusion_output_root: str,
    gpu: str = "0",
    skip_render: bool = False,
    skip_inference: bool = False,
    skip_fusion: bool = False
) -> Dict:
    """
    处理单条 CityRefer 标注
    """
    scene_id = ann['scene_id']
    object_id = ann['object_id']
    ann_id = ann['ann_id']
    candidates = ann.get('candidates_30', [])
    construction = ann.get('construction', [])

    print(f"\n{'='*60}")
    print(f"Processing: {scene_id} / {object_id} / ann_{ann_id}")
    print(f"Description: {ann.get('description', '')}")
    print(f"Candidates: {candidates}")

    # 1. 候选 -> 网格块 (renderer 会自动加载周围 450m 上下文)
    box3d_data = load_box3d(scene_id, box3d_root)
    blocks = candidates_to_grid_blocks(candidates, box3d_data)
    print(f"[Center Blocks]: {blocks}")
    print(f"  (renderer will load surrounding 450m context automatically)")

    # 2. 渲染
    if not skip_render and blocks:
        run_renderer(input_folder, render_root, blocks, gpu)
    else:
        print("[Skip] Renderer")

    # 3. SAM3 推理
    if not skip_inference:
        run_inference(render_root, mask_root, mode="all", gpu=gpu)
    else:
        print("[Skip] Inference")

    # 4. 融合
    if not skip_fusion:
        run_fusion(cluster_path, render_root, mask_root, fusion_output_root, gpu)
    else:
        print("[Skip] Fusion")

    # 5. 空间关系验证（基于 GT box3d）
    scores = {}
    for cand_id in candidates:
        score = compute_spatial_relation_score(cand_id, construction, box3d_data)
        scores[cand_id] = score
        print(f"  Candidate {cand_id}: score = {score:.3f}")

    # 6. 选择 Top1
    if scores:
        best_id = max(scores, key=scores.get)
        best_score = scores[best_id]
    else:
        best_id = candidates[0] if candidates else None
        best_score = 0.0

    is_correct = (best_id == object_id)

    print(f"[Result] Predicted: {best_id}, GT: {object_id}, Correct: {is_correct}")

    return {
        'scene_id': scene_id,
        'object_id': object_id,
        'ann_id': ann_id,
        'predicted_object_id': best_id,
        'match_score': best_score,
        'is_correct': is_correct,
        'all_scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description="CityRefer Pipeline")
    parser.add_argument('--jsonl', type=str, required=True,
                        help='CityRefer JSONL 文件路径')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='预处理后的网格块 PLY 目录')
    parser.add_argument('--box3d_root', type=str, required=True,
                        help='box3d JSON 目录')
    parser.add_argument('--render_root', type=str, default='./cityrefer_render',
                        help='渲染输出目录')
    parser.add_argument('--mask_root', type=str, default='./cityrefer_mask',
                        help='mask 输出目录')
    parser.add_argument('--cluster_path', type=str, required=True,
                        help='聚类结果目录 (_nag.pt 文件)')
    parser.add_argument('--fusion_output', type=str, default='./cityrefer_fusion',
                        help='融合输出目录')
    parser.add_argument('--output_results', type=str, default='./cityrefer_results.json',
                        help='最终结果文件')
    parser.add_argument('--max_samples', type=int, default=10,
                        help='处理的最大样本数')
    parser.add_argument('--gpu', type=str, default='0',
                        help='使用的 GPU')
    parser.add_argument('--skip_render', action='store_true',
                        help='跳过渲染')
    parser.add_argument('--skip_inference', action='store_true',
                        help='跳过推理')
    parser.add_argument('--skip_fusion', action='store_true',
                        help='跳过融合')
    args = parser.parse_args()

    # 加载标注
    print(f"Loading annotations from {args.jsonl}")
    annotations = []
    with open(args.jsonl, 'r') as f:
        for line in f:
            annotations.append(json.loads(line.strip()))
    print(f"Loaded {len(annotations)} annotations")

    # 处理样本
    results = []
    for i, ann in enumerate(annotations[:args.max_samples]):
        try:
            result = process_annotation(
                ann=ann,
                input_folder=args.input_folder,
                box3d_root=args.box3d_root,
                render_root=args.render_root,
                mask_root=args.mask_root,
                cluster_path=args.cluster_path,
                fusion_output_root=args.fusion_output,
                gpu=args.gpu,
                skip_render=args.skip_render,
                skip_inference=args.skip_inference,
                skip_fusion=args.skip_fusion
            )
            results.append(result)
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    with open(args.output_results, 'w') as f:
        json.dump(results, f, indent=2)

    # 统计
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Results saved to {args.output_results}")


if __name__ == '__main__':
    main()

"""
CityRefer 开放词汇定位 Pipeline
- 只渲染候选 bbox 周围的网格块
- SAM3 实例分割 + 空间关系验证
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# sam3_segment 模块
import re
from PIL import Image

from core.renderer import (
    load_ply_block_local, 
    get_safe_camera_R_T,
    get_ortho_scale,
    convert_RT_to_C2W_Global,
    get_scene_bounds,
    render_task,
    GLOBAL_CAM_DIST, DETAIL_CAM_DIST, SIDE_CAM_DIST, IMG_SIZE
)
from configs.prompts import get_prompts_strict


# ==========================================
# 1. 数据结构
# ==========================================

@dataclass
class CityReferAnnotation:
    """CityRefer 单条标注数据"""
    scene_id: str
    object_id: str
    object_name: str
    ann_id: int
    description: str
    construction: List[Dict]  # 超图结构
    candidates_30: List[str]  # 候选 object_id


@dataclass
class SpatialRelation:
    """空间关系"""
    type: str  # "directional", "topological", "distance"
    relation: str  # "in front of", "on the left", "near", "inside"
    from_obj: str
    to_obj: str
    confidence: float = 1.0


# ==========================================
# 2. 候选 → 网格块映射
# ==========================================

def load_box3d(scene_name: str, box3d_root: str) -> Dict:
    """加载场景的 box3d 数据"""
    path = os.path.join(box3d_root, f"{scene_name}_bbox.json")
    with open(path, 'r') as f:
        return json.load(f)


def candidates_to_grid_blocks(
    candidates: List[str], 
    box3d_data: Dict, 
    grid_size: int = 50, 
    context_margin: float = 25.0
) -> Set[str]:
    """
    从候选 object_id 映射到需要渲染的网格块
    
    Args:
        candidates: object_id 列表
        box3d_data: box3d 数据
        grid_size: 网格大小（米）
        context_margin: 额外扩展范围，用于捕捉周围客体
    
    Returns:
        需要渲染的网格块 ID 集合
    """
    blocks = set()
    
    # 构建 object_id -> bbox 映射
    obj_to_bbox = {obj['object_id']: obj for obj in box3d_data['bboxes']}
    
    for obj_id_str in candidates:
        obj_id = int(obj_id_str)
        if obj_id not in obj_to_bbox:
            continue
        
        obj = obj_to_bbox[obj_id]
        x, y, z, w, h, d = obj['bbox'][:6]
        
        # 计算覆盖范围（加 margin 捕捉周围物体）
        min_x = x - w/2 - context_margin
        max_x = x + w/2 + context_margin
        min_y = y - h/2 - context_margin
        max_y = y + h/2 + context_margin
        
        # 枚举覆盖的所有网格块
        gx_start = int(np.floor(min_x / grid_size)) * grid_size
        gx_end = int(np.ceil(max_x / grid_size)) * grid_size
        gy_start = int(np.floor(min_y / grid_size)) * grid_size
        gy_end = int(np.ceil(max_y / grid_size)) * grid_size
        
        for gx in range(gx_start, gx_end + 1, grid_size):
            for gy in range(gy_start, gy_end + 1, grid_size):
                blocks.add(f"x{gx}_y{gy}")
    
    return blocks


# ==========================================
# 3. 网格块渲染
# ==========================================

class GridBlockRenderer:
    """网格块渲染器"""
    
    def __init__(self, scene_name: str, input_folder: str, output_root: str, device='cuda'):
        self.scene_name = scene_name
        self.input_folder = input_folder
        self.output_root = output_root
        self.device = torch.device(device)
        
        # 加载 metadata
        meta_path = os.path.join(input_folder, "metadata.json")
        with open(meta_path, 'r') as f:
            self.metadata = json.load(f)
    
    def render_blocks(self, block_ids: Set[str]):
        """渲染指定的网格块"""
        for block_id in block_ids:
            self._render_single_block(block_id)
    
    def _render_single_block(self, block_id: str):
        """渲染单个网格块（简化版，复用 renderer.py 逻辑）"""
        # 找到对应的 PLY 文件
        ply_file = None
        for fname, info in self.metadata.items():
            ox, oy = info['offset'][:2]
            expected_id = f"x{int(ox)}_y{int(oy)}"
            if expected_id == block_id:
                ply_file = fname
                break
        
        if ply_file is None:
            print(f"[Warning] Block {block_id} not found in metadata")
            return
        
        # 加载点云
        path = os.path.join(self.input_folder, ply_file)
        verts_local, verts_rgb = load_ply_block_local(path)
        if verts_local is None:
            return
        
        # 创建输出目录
        scene_dir = os.path.join(self.output_root, self.scene_name, block_id)
        for subdir in ["dev", "image", "pose", "intrinsic", "depth"]:
            os.makedirs(os.path.join(scene_dir, subdir), exist_ok=True)
        
        # 计算场景边界
        z_min, z_max = get_scene_bounds(verts_local)
        
        # 重心调整
        search_min, search_max = -20.0, 70.0
        local_mask = (
            (verts_local[:, 0] > search_min) & (verts_local[:, 0] < search_max) &
            (verts_local[:, 1] > search_min) & (verts_local[:, 1] < search_max)
        )
        
        if local_mask.sum() > 50:
            local_points = verts_local[local_mask]
            center_x = torch.median(local_points[:, 0]).item()
            center_y = torch.median(local_points[:, 1]).item()
        else:
            center_x, center_y = 25.0, 25.0
        
        center_z = (z_min + z_max) / 2.0
        fixed_origin_center = np.array([center_x, center_y, center_z])
        
        # 解析网格偏移
        match = re.search(r"x(-?\d+)_y(-?\d+)", block_id)
        if match:
            off_x, off_y = float(match.group(1)), float(match.group(2))
            global_offset_vec = np.array([off_x, off_y, 0])
        else:
            global_offset_vec = np.array([0, 0, 0])
        
        # 保存中心点
        np.save(
            os.path.join(scene_dir, "dev", "center.npy"),
            np.array([center_x, center_y, z_min]) + global_offset_vec
        )
        
        # 构建渲染任务
        tasks = []
        safe_z_global = z_max + 50.0
        R_g, T_g = get_safe_camera_R_T(
            fixed_origin_center, -89.9, 0, 1, GLOBAL_CAM_DIST, safe_z_global
        )
        tasks.append({
            "name": f"{block_id}_Global",
            "pitch": -89.9,
            "R": R_g,
            "T": T_g,
            "origin": fixed_origin_center,
            "folder_group": "dev",
            "fit_radius": 65.0,
            "render_radius": 0.004
        })
        
        safe_z_detail = z_max + 30.0
        R_d, T_d = get_safe_camera_R_T(
            fixed_origin_center, -89.9, 0, 1, DETAIL_CAM_DIST, safe_z_detail
        )
        tasks.append({
            "name": f"{block_id}_Detail",
            "pitch": -89.9,
            "R": R_d,
            "T": T_d,
            "origin": fixed_origin_center,
            "folder_group": "dev",
            "fit_radius": 35.0,
            "render_radius": 0.0035
        })
        
        # 执行渲染
        for task in tasks:
            res = render_task(task, verts_local, verts_rgb, None, self._args(), self.device)
            if res is None:
                continue
            
            img_np, depth_np, K_out, _ = res
            c2w_global = convert_RT_to_C2W_Global(task["R"], task["T"], global_offset_vec)
            save_name = task["name"]
            
            # 保存结果
            from PIL import Image
            pil_img = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
            
            if task["folder_group"] == "dev":
                base = os.path.join(scene_dir, "dev")
                pil_img.save(os.path.join(base, f"{save_name}_real.png"))
                np.save(os.path.join(base, f"intrinsic_{save_name}.npy"), K_out)
                np.save(os.path.join(base, f"pose_{save_name}.npy"), c2w_global)
                np.save(os.path.join(base, f"depth_{save_name}.npy"), depth_np)
        
        print(f"[Rendered] {block_id}")
    
    def _args(self):
        """创建模拟 args 对象"""
        class Args:
            width = IMG_SIZE
            height = IMG_SIZE
        return Args()


# ==========================================
# 4. 空间关系计算
# ==========================================

def compute_3d_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算点云的 3D 边界框
    
    Returns:
        center: [x, y, z]
        size: [w, h, d]
    """
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt) / 2
    size = max_pt - min_pt
    return center, size


def compute_spatial_relation(
    main_bbox: Tuple[np.ndarray, np.ndarray],
    ref_bbox: Tuple[np.ndarray, np.ndarray],
    angle_threshold: float = 30.0
) -> List[SpatialRelation]:
    """
    计算两个 bbox 之间的空间关系
    
    Args:
        main_bbox: (center, size) of main object
        ref_bbox: (center, size) of reference object
        angle_threshold: 角度阈值（度）
    
    Returns:
        空间关系列表
    """
    main_center, main_size = main_bbox
    ref_center, ref_size = ref_bbox
    
    relations = []
    
    # 相对位置向量
    delta = main_center - ref_center
    dist = np.linalg.norm(delta)
    
    # 1. 方向关系 (基于主方向)
    if dist > 0.5:  # 忽略非常近的
        # 计算角度
        angle_xy = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        
        # 定义方向区间
        directions = [
            (-45, 45, "in front of"),      # 东
            (45, 135, "on the left of"),   # 北
            (135, 225, "behind"),          # 西
            (225, 315, "on the right of"), # 南
        ]
        
        # 归一化到 0-360
        if angle_xy < 0:
            angle_xy += 360
        
        for min_a, max_a, rel_name in directions:
            if min_a <= angle_xy < max_a:
                relations.append(SpatialRelation(
                    type="directional",
                    relation=rel_name,
                    from_obj="main",
                    to_obj="ref",
                    confidence=min(1.0, dist / 10.0)  # 距离越近置信度越高
                ))
                break
    
    # 2. 距离关系
    if dist < 5.0:
        relations.append(SpatialRelation(
            type="distance",
            relation="very close to",
            from_obj="main",
            to_obj="ref",
            confidence=1.0
        ))
    elif dist < 15.0:
        relations.append(SpatialRelation(
            type="distance",
            relation="near",
            from_obj="main",
            to_obj="ref",
            confidence=0.8
        ))
    
    # 3. 包含关系
    # 检查 main 是否在 ref 内部
    ref_min = ref_center - ref_size / 2
    ref_max = ref_center + ref_size / 2
    main_min = main_center - main_size / 2
    main_max = main_center + main_size / 2
    
    if np.all(main_min >= ref_min) and np.all(main_max <= ref_max):
        relations.append(SpatialRelation(
            type="topological",
            relation="inside",
            from_obj="main",
            to_obj="ref",
            confidence=1.0
        ))
    
    # 检查 main 是否包含 ref
    if np.all(ref_min >= main_min) and np.all(ref_max <= main_max):
        relations.append(SpatialRelation(
            type="topological",
            relation="contains",
            from_obj="main",
            to_obj="ref",
            confidence=1.0
        ))
    
    return relations


def match_construction_relations(
    computed_relations: List[SpatialRelation],
    construction: List[Dict]
) -> float:
    """
    计算计算出的关系与 construction 中关系的匹配分数
    
    Args:
        computed_relations: SAM3 实例间计算的关系
        construction: CityRefer 标注的关系
    
    Returns:
        匹配分数 (0-1)
    """
    if not construction:
        return 1.0
    
    # 提取 construction 中的关系
    target_relations = []
    for item in construction:
        if not item.get('is_main', False) and item.get('spatial_relation'):
            target_relations.append({
                'category': item.get('category2', item.get('category', '')),
                'relation': item['spatial_relation'],
                'reference': item.get('reference_anchor', '')
            })
    
    if not target_relations:
        return 1.0
    
    # 计算匹配
    matched = 0
    total_score = 0.0
    
    for target in target_relations:
        target_rel = target['relation'].lower().strip()
        
        # 模糊匹配
        for comp in computed_relations:
            comp_rel = comp.relation.lower().strip()
            
            # 精确匹配
            if target_rel == comp_rel:
                matched += 1
                total_score += comp.confidence
                break
            # 模糊匹配
            elif target_rel in comp_rel or comp_rel in target_rel:
                matched += 0.5
                total_score += comp.confidence * 0.5
                break
    
    # 归一化分数
    if len(target_relations) > 0:
        return total_score / len(target_relations)
    return 1.0


# ==========================================
# 5. 主 Pipeline
# ==========================================

class CityReferPipeline:
    """CityRefer 开放词汇定位 Pipeline"""
    
    def __init__(
        self,
        input_folder: str,      # 预处理后的网格块 PLY 目录
        box3d_root: str,        # box3d JSON 目录
        output_root: str,       # 渲染输出目录
        cluster_path: str,      # 聚类结果目录
        mask_root: str,         # mask 输出目录
        device: str = 'cuda'
    ):
        self.input_folder = input_folder
        self.box3d_root = box3d_root
        self.output_root = output_root
        self.cluster_path = cluster_path
        self.mask_root = mask_root
        self.device = device
    
    def process_annotation(self, ann: CityReferAnnotation) -> Dict:
        """
        处理单条 CityRefer 标注
        
        Returns:
            {
                'scene_id': str,
                'object_id': str,
                'ann_id': int,
                'predicted_object_id': str,  # Top1 预测
                'match_score': float,        # 匹配分数
                'is_correct': bool           # 是否正确
            }
        """
        print(f"\n{'='*60}")
        print(f"Processing: {ann.scene_id} / {ann.object_id} / ann_{ann.ann_id}")
        print(f"Description: {ann.description}")
        print(f"Candidates: {ann.candidates_30}")
        
        # 1. 加载 box3d
        box3d_data = load_box3d(ann.scene_id, self.box3d_root)
        
        # 2. 候选 → 网格块
        blocks = candidates_to_grid_blocks(
            ann.candidates_30, 
            box3d_data,
            grid_size=50,
            context_margin=25.0
        )
        print(f"[Blocks to render]: {len(blocks)} blocks")
        print(f"  {blocks}")
        
        # 3. 渲染网格块
        renderer = GridBlockRenderer(
            ann.scene_id, 
            self.input_folder, 
            self.output_root,
            self.device
        )
        renderer.render_blocks(blocks)
        
        # 4. SAM3 分割（为每个块单独运行）
        # TODO: 这里需要调用 inference.py 的逻辑
        # 暂时跳过，假设已有 mask 结果
        
        # 5. 融合（为每个块单独运行）
        # TODO: 这里需要调用 fusion.py 的逻辑
        
        # 6. 空间关系验证
        # 获取候选 object_id 的 3D bbox
        obj_to_bbox = {obj['object_id']: obj for obj in box3d_data['bboxes']}
        
        scores = {}
        for cand_id in ann.candidates_30:
            obj_id = int(cand_id)
            if obj_id not in obj_to_bbox:
                scores[cand_id] = 0.0
                continue
            
            # 获取候选物体的 bbox
            cand_obj = obj_to_bbox[obj_id]
            cand_center = np.array(cand_obj['bbox'][:3])
            cand_size = np.array(cand_obj['bbox'][3:6])
            cand_bbox = (cand_center, cand_size)
            
            # 查找周围的客体
            computed_relations = []
            for other_id, other_obj in obj_to_bbox.items():
                if other_id == obj_id:
                    continue
                
                other_center = np.array(other_obj['bbox'][:3])
                other_size = np.array(other_obj['bbox'][3:6])
                other_bbox = (other_center, other_size)
                
                # 计算距离
                dist = np.linalg.norm(cand_center - other_center)
                if dist < 50:  # 只考虑50米内的
                    rels = compute_spatial_relation(cand_bbox, other_bbox)
                    computed_relations.extend(rels)
            
            # 匹配 construction 中的关系
            match_score = match_construction_relations(
                computed_relations, 
                ann.construction
            )
            scores[cand_id] = match_score
            print(f"  Candidate {cand_id}: score = {match_score:.3f}")
        
        # 7. 选择 Top1
        if scores:
            best_id = max(scores, key=scores.get)
            best_score = scores[best_id]
        else:
            best_id = ann.candidates_30[0] if ann.candidates_30 else None
            best_score = 0.0
        
        is_correct = (best_id == ann.object_id)
        
        print(f"[Result] Predicted: {best_id}, GT: {ann.object_id}, Correct: {is_correct}")
        
        return {
            'scene_id': ann.scene_id,
            'object_id': ann.object_id,
            'ann_id': ann.ann_id,
            'predicted_object_id': best_id,
            'match_score': best_score,
            'is_correct': is_correct,
            'all_scores': scores
        }


def load_cityrefer_annotations(jsonl_path: str) -> List[CityReferAnnotation]:
    """加载 CityRefer 标注"""
    annotations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            ann = CityReferAnnotation(
                scene_id=data['scene_id'],
                object_id=data['object_id'],
                object_name=data['object_name'],
                ann_id=data['ann_id'],
                description=data['description'],
                construction=data['construction'],
                candidates_30=data.get('candidates_30', [])
            )
            annotations.append(ann)
    return annotations


# ==========================================
# 6. 主函数
# ==========================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=str, required=True, help='CityRefer JSONL 文件')
    parser.add_argument('--input_folder', type=str, required=True, help='网格块 PLY 目录')
    parser.add_argument('--box3d_root', type=str, required=True, help='box3d JSON 目录')
    parser.add_argument('--output_root', type=str, default='./cityrefer_output')
    parser.add_argument('--cluster_path', type=str, required=True)
    parser.add_argument('--mask_root', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=10, help='处理的最大样本数')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # 加载标注
    print(f"Loading annotations from {args.jsonl}")
    annotations = load_cityrefer_annotations(args.jsonl)
    print(f"Loaded {len(annotations)} annotations")
    
    # 创建 pipeline
    pipeline = CityReferPipeline(
        input_folder=args.input_folder,
        box3d_root=args.box3d_root,
        output_root=args.output_root,
        cluster_path=args.cluster_path,
        mask_root=args.mask_root,
        device=args.device
    )
    
    # 处理样本
    results = []
    for i, ann in enumerate(annotations[:args.max_samples]):
        try:
            result = pipeline.process_annotation(ann)
            results.append(result)
        except Exception as e:
            print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
    
    # 保存结果
    output_file = os.path.join(args.output_root, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 统计
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()

"""
Step 5: BBox-level Instance Segmentation
以 CityRefer bbox 为单位，生成跨网格实例分割结果
"""

import torch
import numpy as np
import json
import os
import glob
import argparse
from collections import defaultdict
from tqdm import tqdm
import cv2
import re

# 类别定义
CLASS_NAME_TO_ID = {
    "Ground": 0, "Vegetation": 1, "Building": 2, "Wall": 3,
    "Bridge": 4, "Parking": 5, "Rail": 6, "Traffic Road": 7,
    "Street Furniture": 8, "Car": 9, "Footpath": 10, "Bike": 11, "Water": 12
}
ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


class BBoxInstanceSegmentor:
    def __init__(self, args):
        self.args = args
        self.fusion_root = args.fusion_root
        self.mask_root = args.mask_root
        self.grid_root = args.grid_root
        self.output_root = args.output_root
        os.makedirs(self.output_root, exist_ok=True)
        
    def parse_grid_id(self, grid_id):
        """解析网格 ID，如 x0_y500 -> (0, 500)"""
        match = re.search(r"x(-?\d+)_y(-?\d+)", grid_id)
        if match:
            return float(match.group(1)), float(match.group(2))
        return 0, 0
    
    def load_bbox_data(self, box3d_path):
        """加载 CityRefer bbox 数据"""
        with open(box3d_path) as f:
            data = json.load(f)
        return data.get('bboxes', [])
    
    def get_bbox_bounds(self, bbox):
        """计算 bbox 的边界"""
        x, y, z, w, h, d = bbox['bbox'][:6]
        return {
            'min_x': x - w/2, 'max_x': x + w/2,
            'min_y': y - h/2, 'max_y': y + h/2,
            'min_z': z - d/2, 'max_z': z + d/2,
            'center': np.array([x, y, z]),
            'size': np.array([w, h, d])
        }
    
    def find_intersecting_grids(self, bbox_bounds):
        """找出与 bbox 相交的所有网格块"""
        grid_size = 50  # 网格大小
        grids = []
        
        # 计算 bbox 覆盖的网格范围
        min_gx = int(np.floor(bbox_bounds['min_x'] / grid_size) * grid_size)
        max_gx = int(np.floor(bbox_bounds['max_x'] / grid_size) * grid_size)
        min_gy = int(np.floor(bbox_bounds['min_y'] / grid_size) * grid_size)
        max_gy = int(np.floor(bbox_bounds['max_y'] / grid_size) * grid_size)
        
        print(f"    Looking for grids in x[{min_gx}-{max_gx}], y[{min_gy}-{max_gy}]")
        
        for gx in range(min_gx, max_gx + grid_size, grid_size):
            for gy in range(min_gy, max_gy + grid_size, grid_size):
                grid_id = f"x{gx}_y{gy}"
                # 检查该网格是否存在 (fusion 结果)
                fusion_path = os.path.join(self.fusion_root, f"{self.scene_name}_{grid_id}_fusion.ply")
                if os.path.exists(fusion_path):
                    grids.append({
                        'id': grid_id,
                        'offset': np.array([float(gx), float(gy), 0.0])
                    })
                    print(f"      Found grid: {grid_id}")
        
        return grids
    
    def load_grid_points(self, grid_id):
        """加载网格的点云 (从 PLY)，不使用超点"""
        import plyfile
        
        ply_path = os.path.join(self.fusion_root, f"{self.scene_name}_{grid_id}_fusion.ply")
        if not os.path.exists(ply_path):
            return None
        
        try:
            plydata = plyfile.PlyData.read(ply_path)
            vertex = plydata['vertex']
            
            # 读取点坐标
            points = np.stack([
                vertex['x'],
                vertex['y'],
                vertex['z']
            ], axis=1)
            
            # 读取语义标签 (pred 属性)
            if 'pred' in vertex:
                labels = np.array(vertex['pred'])
            else:
                print(f"  Warning: No 'pred' in {ply_path}")
                return None
            
            return {
                'points': points,
                'labels': labels
            }
            
        except Exception as e:
            print(f"  Error loading {ply_path}: {e}")
            return None
    
    def decode_mask(self, mask_buffer):
        """解码压缩的 mask"""
        if mask_buffer is None:
            return None
        mask_np = cv2.imdecode(np.frombuffer(mask_buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask_np is None:
            return None
        return mask_np > 127
    
    def load_grid_masks(self, grid_id):
        """加载网格的 SAM3 mask"""
        mask_path = os.path.join(self.mask_root, self.scene_name, grid_id, f"{grid_id}_strict_bev.pt")
        if not os.path.exists(mask_path):
            return []
        
        data = torch.load(mask_path, map_location='cpu', weights_only=False)
        masks = []
        
        for img_idx, img_name in enumerate(data['img_names']):
            # 只处理 Detail 视图
            if "Detail" not in img_name:
                continue
            
            mask_buffers = data['masks'][img_idx]
            labels = data['labels'][img_idx].numpy()
            scores = data['scores'][img_idx].numpy()
            
            for mask_idx, (buf, lbl, scr) in enumerate(zip(mask_buffers, labels, scores)):
                mask = self.decode_mask(buf)
                if mask is not None:
                    masks.append({
                        'image_name': img_name,
                        'mask': mask,
                        'label': int(lbl),
                        'confidence': float(scr),
                        'grid_id': grid_id
                    })
        
        return masks
    
    def extract_bbox_semantic_points(self, grids, bbox_bounds, target_label):
        """在 bbox 区域内提取目标语义的原始点"""
        all_points = []
        all_labels = []
        
        for grid in grids:
            data = self.load_grid_points(grid['id'])
            if data is None:
                continue
            
            points = data['points'] + grid['offset']
            labels = data['labels']
            
            # 筛选在 bbox 内且标签匹配的点
            in_bbox = (
                (points[:, 0] >= bbox_bounds['min_x']) & (points[:, 0] <= bbox_bounds['max_x']) &
                (points[:, 1] >= bbox_bounds['min_y']) & (points[:, 1] <= bbox_bounds['max_y']) &
                (points[:, 2] >= bbox_bounds['min_z']) & (points[:, 2] <= bbox_bounds['max_z'])
            )
            
            label_match = labels == target_label
            valid = in_bbox & label_match
            
            if valid.any():
                all_points.extend(points[valid])
                all_labels.extend(labels[valid])
        
        if len(all_points) == 0:
            return None
        
        return {
            'points': np.array(all_points),
            'labels': np.array(all_labels),
            'center': np.mean(all_points, axis=0),
            'bbox': {
                'min': np.min(all_points, axis=0),
                'max': np.max(all_points, axis=0)
            }
        }
    
    def project_mask_to_points(self, mask, pose, intrinsic, points_global, img_size=(1024, 1024)):
        """将 2D mask 投影到 3D 点云，返回被 mask 覆盖的点索引"""
        H, W = img_size
        
        # 获取 mask 中的像素坐标
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.array([]), np.array([])
        
        # 构建像素点 (u, v, 1)
        pixels = np.stack([xs, ys, np.ones(len(xs))], axis=1).astype(np.float32)
        
        # 转换为 NDC
        u_ndc = (pixels[:, 0] / (W / 2.0)) - 1.0
        v_ndc = 1.0 - (pixels[:, 1] / (H / 2.0))
        
        # 从 NDC 恢复相机坐标（假设深度为 1）
        scale = intrinsic[0, 0]
        x_cam = -u_ndc / scale
        y_cam = v_ndc / scale
        z_cam = np.ones_like(x_cam)
        
        # 相机坐标 -> 世界坐标
        c2w = pose
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones(len(x_cam))], axis=1)
        pts_world = (c2w @ pts_cam.T).T[:, :3]
        
        # 找到最近的点
        from scipy.spatial import cKDTree
        tree = cKDTree(points_global)
        distances, indices = tree.query(pts_world, k=1, distance_upper_bound=2.0)
        
        # 过滤有效匹配
        valid = distances < 2.0
        hit_indices = indices[valid]
        
        # 去重
        unique_indices = np.unique(hit_indices)
        
        return unique_indices, pts_world[valid]
    
    def compute_mask_center(self, mask, pose, intrinsic, img_size=(1024, 1024)):
        """计算 mask 的 3D 中心点"""
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        
        # mask 中心像素
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # 转换到 NDC
        H, W = img_size
        u_ndc = (cx / (W / 2.0)) - 1.0
        v_ndc = 1.0 - (cy / (H / 2.0))
        
        # 相机坐标 -> 世界坐标
        scale = intrinsic[0, 0]
        x_cam = -u_ndc / scale
        y_cam = v_ndc / scale
        z_cam = 1.0
        
        c2w = pose
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_world = (c2w @ pt_cam)[:3]
        
        return pt_world
    
    def load_camera_params(self, grid_id, image_name):
        """加载相机参数"""
        # 从 render_out 加载 pose 和 intrinsic
        scene_dir = os.path.join(self.args.render_root, self.scene_name, grid_id, "dev")
        
        prefix = image_name.replace("_real.png", "").replace(".png", "")
        pose_path = os.path.join(scene_dir, f"pose_{prefix}.npy")
        intr_path = os.path.join(scene_dir, f"intrinsic_{prefix}.npy")
        
        if not os.path.exists(pose_path) or not os.path.exists(intr_path):
            return None, None
        
        pose = np.load(pose_path)
        intr = np.load(intr_path)
        
        return pose, intr
    
    def match_masks_to_points(self, masks, semantic_points, bbox_bounds, grids):
        """匹配 masks 到语义点云，包含完整投影逻辑"""
        if semantic_points is None or len(masks) == 0:
            return []
        
        target_label = semantic_points['labels'][0]
        
        # 为每个 mask 计算 3D 中心
        mask_3d_info = []
        
        for m in masks:
            if m['label'] != target_label:
                continue
            
            # 加载相机参数
            pose, intr = self.load_camera_params(m['grid_id'], m['image_name'])
            if pose is None or intr is None:
                continue
            
            # 获取网格偏移
            grid_offset = None
            for g in grids:
                if g['id'] == m['grid_id']:
                    grid_offset = g['offset']
                    break
            
            if grid_offset is None:
                continue
            
            # 加载该网格的点云
            data = self.load_grid_points(m['grid_id'])
            if data is None:
                continue
            
            points_global = data['points'] + grid_offset
            
            # 投影 mask 到点云
            hit_indices, hit_pts = self.project_mask_to_points(
                m['mask'], pose, intr, points_global
            )
            
            if len(hit_indices) == 0:
                continue
            
            # 计算 mask 的 3D 中心
            center_3d = self.compute_mask_center(m['mask'], pose, intr)
            if center_3d is None:
                continue
            
            mask_3d_info.append({
                'mask': m,
                'center_3d': center_3d,
                'hit_indices': hit_indices,
                'num_hit_points': len(hit_indices)
            })
        
        if len(mask_3d_info) == 0:
            return []
        
        # 计算每个 mask 与语义点云的匹配度
        for info in mask_3d_info:
            # 使用中心点距离作为匹配度
            dist_to_semantic = np.linalg.norm(
                info['center_3d'] - semantic_points['center']
            )
            # 归一化距离（距离越近，匹配度越高）
            info['match_score'] = max(0, 1 - dist_to_semantic / 10.0)
            info['score'] = info['match_score'] * info['mask']['confidence']
        
        # 按 score 排序
        mask_3d_info.sort(key=lambda x: x['score'], reverse=True)
        
        # 距离聚类合并（中心点距离 < 5m 的合并）
        clusters = []
        used = set()
        
        for i, info_i in enumerate(mask_3d_info):
            if i in used:
                continue
            
            cluster = [info_i]
            used.add(i)
            
            for j, info_j in enumerate(mask_3d_info[i+1:], start=i+1):
                if j in used:
                    continue
                
                dist = np.linalg.norm(info_i['center_3d'] - info_j['center_3d'])
                if dist < 5.0:  # 5米内合并
                    cluster.append(info_j)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def process_bbox(self, bbox, box3d_path):
        """处理单个 bbox"""
        object_id = bbox['object_id']
        object_name = bbox['object_name']
        target_label = CLASS_NAME_TO_ID.get(object_name, -1)
        
        if target_label == -1:
            print(f"  Warning: Unknown class {object_name}")
            return None
        
        bbox_bounds = self.get_bbox_bounds(bbox)
        
        # 1. 找出相交的网格
        grids = self.find_intersecting_grids(bbox_bounds)
        if len(grids) == 0:
            print(f"  Warning: No grids found for bbox {object_id}")
            return None
        
        print(f"  BBox {object_id} ({object_name}): {len(grids)} grids")
        
        # 2. 提取语义点
        semantic_points = self.extract_bbox_semantic_points(grids, bbox_bounds, target_label)
        if semantic_points is None:
            print(f"  Warning: No semantic points found for bbox {object_id}")
            return None
        
        print(f"    Found {len(semantic_points['points'])} semantic points")
        
        # 3. 加载 masks
        all_masks = []
        for grid in grids:
            masks = self.load_grid_masks(grid['id'])
            all_masks.extend(masks)
        
        print(f"    Found {len(all_masks)} masks")
        
        # 4. 匹配 masks（包含投影逻辑）
        mask_clusters = self.match_masks_to_points(all_masks, semantic_points, bbox_bounds, grids)
        
        # 5. 生成结果
        result = {
            'object_id': object_id,
            'object_name': object_name,
            'gt_bbox': bbox['bbox'],
            'semantic_center': semantic_points['center'].tolist(),
            'semantic_bbox': {
                'min': semantic_points['bbox']['min'].tolist(),
                'max': semantic_points['bbox']['max'].tolist()
            },
            'num_semantic_points': len(semantic_points['points']),
            'num_mask_clusters': len(mask_clusters),
            'mask_clusters': [
                {
                    'num_masks': len(cluster),
                    'avg_confidence': np.mean([info['mask']['confidence'] for info in cluster]),
                    'avg_score': np.mean([info['score'] for info in cluster]),
                    'center_3d': np.mean([info['center_3d'] for info in cluster], axis=0).tolist(),
                    'masks': [
                        {
                            'image': info['mask']['image_name'],
                            'grid': info['mask']['grid_id'],
                            'conf': info['mask']['confidence'],
                            'score': info['score'],
                            'center_3d': info['center_3d'].tolist(),
                            'num_hit_points': info['num_hit_points']
                        }
                        for info in cluster
                    ]
                }
                for cluster in mask_clusters
            ]
        }
        
        return result, semantic_points
    
    def process_scene(self, scene_name, box3d_path):
        """处理整个场景"""
        self.scene_name = scene_name
        print(f"\n{'='*60}")
        print(f"Processing Scene: {scene_name}")
        print(f"{'='*60}")
        
        # 加载 bbox 数据
        bboxes = self.load_bbox_data(box3d_path)
        print(f"Total bboxes: {len(bboxes)}")
        
        # 如果指定了 object_id，只处理一个
        if hasattr(self.args, 'object_id') and self.args.object_id is not None:
            bboxes = [b for b in bboxes if b['object_id'] == self.args.object_id]
            if len(bboxes) == 0:
                print(f"Error: object_id {self.args.object_id} not found")
                return
            print(f"Processing single bbox: object_id={self.args.object_id}")
        
        results = []
        all_components = []
        
        for bbox in tqdm(bboxes, desc="Processing bboxes"):
            result = self.process_bbox(bbox, box3d_path)
            if result:
                json_data, component = result
                results.append(json_data)
                all_components.append({
                    'object_id': bbox['object_id'],
                    'component': component
                })
        
        # 保存 JSON
        json_path = os.path.join(self.output_root, f"{scene_name}_bbox_instances.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved {len(results)} results to {json_path}")
        
        # 保存可视化 PLY（可选）
        if self.args.save_ply:
            self.save_visualization(scene_name, all_components)
    
    def save_visualization(self, scene_name, components):
        """保存可视化 PLY"""
        # 这里可以实现 PLY 保存逻辑
        print(f"Visualization PLY saved (placeholder)")


def main():
    parser = argparse.ArgumentParser(description="Step 5: BBox-level Instance Segmentation")
    parser.add_argument('--scene_name', type=str, required=True, help="Scene name, e.g., birmingham_block_1")
    parser.add_argument('--box3d_path', type=str, required=True, help="Path to box3d JSON file")
    parser.add_argument('--fusion_root', type=str, 
                        default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/sensaturban/fusion_results_final",
                        help="Root directory of fusion results")
    parser.add_argument('--mask_root', type=str,
                        default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/sensaturban/2d_mask_sam3_v75_vis",
                        help="Root directory of SAM3 masks")
    parser.add_argument('--render_root', type=str,
                        default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/sensaturban/render_out",
                        help="Root directory of render output (for camera poses)")
    parser.add_argument('--grid_root', type=str,
                        default="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/sensaturban/val_grid_50m",
                        help="Root directory of grid blocks")
    parser.add_argument('--output_root', type=str, default="./step5_output", help="Output directory")
    parser.add_argument('--object_id', type=int, default=None, help="Process single bbox by object_id")
    parser.add_argument('--save_ply', action='store_true', help="Save visualization PLY")
    
    args = parser.parse_args()
    
    segmentor = BBoxInstanceSegmentor(args)
    segmentor.process_scene(args.scene_name, args.box3d_path)


if __name__ == "__main__":
    main()

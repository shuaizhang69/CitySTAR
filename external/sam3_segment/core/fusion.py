import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import os
import glob
import cv2
import argparse
import plyfile
from tqdm import tqdm
import re
import traceback
import random
from collections import Counter

# ================= 0. 配置与常量 =================

# 类别定义
CLASS_NAME_TO_ID = {
    "Ground": 0, "Vegetation": 1, "Building": 2, "Wall": 3,
    "Bridge": 4, "Parking": 5, "Rail": 6, "Traffic Road": 7,
    "Street Furniture": 8, "Car": 9, "Footpath": 10, "Bike": 11, "Water": 12
}
NUM_CLASSES = 13

# 投票阈值
VOTE_IOU_THRESH = 0.28   
VOTE_SCORE_THRESH = 0.35 

# 深度一致性容差 (米)
DEPTH_TOLERANCE = 1.5 

# 优先级类别
FOREGROUND_CLASSES = [3, 6, 8, 9, 11]

# ================= [策略配置] 高置信度直接锁定 =================
# 1. 允许直接锁定的类别
DIRECT_HIT_THRESH = 0.65
DIRECT_HIT_CLASSES = [
    1,  # Vegetation
    2,  # Building 
    3,  # Wall
    6,  # Rail
    4,  # Bridge
    9   # Car
]

# 2. [新增] 大尺度类别 (Large Scale Context)
# 定义：这些类别在“远景(Global)”中看更准确，不能被近景“无脑覆盖”
CONTEXT_CLASSES = [5, 7] # Parking, Traffic Road

class SceneIntegrityError(Exception):
    pass

def validate_file(path, description="File"):
    if not os.path.exists(path):
        raise SceneIntegrityError(f"Missing {description}: {path}")
    if os.path.getsize(path) == 0:
        raise SceneIntegrityError(f"Empty {description}: {path}")
    return path

def safe_load_torch(path, device='cpu'):
    validate_file(path, "Torch file")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise SceneIntegrityError(f"Corrupt Torch file {path}: {str(e)}")

def safe_load_numpy(path):
    validate_file(path, "Numpy file")
    try:
        return np.load(path)
    except Exception as e:
        raise SceneIntegrityError(f"Corrupt Numpy file {path}: {str(e)}")

# ================= 1. 几何投影 (核心修复版) =================

def project_points_ortho(points_3d, pose, intrinsic, img_size=(1024, 1024)):
    """ 
    [FIXED] 修正后的投影函数
    解决 Y 轴翻转问题 (World North -> Image Top)
    """
    device = points_3d.device
    H, W = img_size
    
    # 1. World -> Camera
    c2w = torch.from_numpy(pose).float().to(device)
    w2c = torch.inverse(c2w)
    
    # P_cam = P_world @ R.T + T
    pts_cam = torch.matmul(points_3d, w2c[:3, :3].T) + w2c[:3, 3]
    
    scale = torch.tensor(intrinsic[0, 0], device=device).float()
    
    # [关键修改] Y轴正号，修正南北翻转
    x_ndc = -pts_cam[:, 0] * scale 
    y_ndc =  pts_cam[:, 1] * scale 
    
    # NDC -> Pixel
    u = (x_ndc + 1.0) * (W / 2.0)
    v = (1.0 - y_ndc) * (H / 2.0)
    
    depth = pts_cam[:, 2]
    return u, v, depth

def check_occlusion(u, v, z, d_map, device, tol=1.0):
    H, W = d_map.shape
    valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    visible_mask = torch.zeros_like(z, dtype=torch.bool)
    
    if not valid_uv.any(): return visible_mask
    
    u_valid = u[valid_uv].long()
    v_valid = v[valid_uv].long()
    z_valid = z[valid_uv]
    
    d_buf = d_map[v_valid, u_valid]
    is_vis = (z_valid <= (d_buf + tol)) & (d_buf > 0)
    visible_mask[valid_uv] = is_vis
    
    return visible_mask

# ================= 2. 融合管线 (几何投影版) =================

class PixelPreciseFusionPipeline:
    def __init__(self, scene_id, cluster_path, snap_root, mask_root, raw_ply_folder, output_root, device='cuda'):
        self.raw_scene_id = scene_id
        self.device = torch.device(device)
        self.output_root = output_root
        
        match = re.search(r"^(.*)_(x\d+_y\d+)$", scene_id)
        if match:
            self.scene_name = match.group(1) 
            self.grid_id = match.group(2)    
            self.snap_scene_dir = os.path.join(snap_root, self.scene_name, self.grid_id)
            self.mask_scene_dir = os.path.join(mask_root, self.scene_name, self.grid_id)
        else:
            self.scene_name = scene_id
            self.grid_id = scene_id
            self.snap_scene_dir = os.path.join(snap_root, scene_id)
            self.mask_scene_dir = os.path.join(mask_root, scene_id)

        self.paths = {
            'cluster': cluster_path,
            'mask_bev': os.path.join(self.mask_scene_dir, f"{self.grid_id}_strict_bev.pt"),
            'mask_tiles': os.path.join(self.mask_scene_dir, f"{self.grid_id}_strict_tiles.pt"),
            'raw_ply': os.path.join(raw_ply_folder, f"{scene_id}.ply")
        }
        
        os.makedirs(output_root, exist_ok=True)

    def get_file_paths(self, img_name):
        prefix = img_name.replace("_real.png", "").replace(".png", "")
        if "Global" in img_name or "Detail" in img_name:
            base = os.path.join(self.snap_scene_dir, "dev")
            return os.path.join(base, f"pose_{prefix}.npy"), \
                   os.path.join(base, f"intrinsic_{prefix}.npy"), \
                   os.path.join(base, f"depth_{prefix}.npy")
        else:
            base = self.snap_scene_dir
            p_path = os.path.join(base, "pose", f"pose_{prefix}.npy")
            if not os.path.exists(p_path): p_path = os.path.join(base, "dev", f"pose_{prefix}.npy")
            i_path = os.path.join(base, "intrinsic", f"intrinsic_{prefix}.npy")
            if not os.path.exists(i_path): i_path = os.path.join(base, "dev", f"intrinsic_{prefix}.npy")
            d_path = os.path.join(base, "depth", f"depth_{prefix}.npy")
            if not os.path.exists(d_path): d_path = os.path.join(base, "dev", f"depth_{prefix}.npy")
            return p_path, i_path, d_path

    def decode_mask_data(self, mask_data):
        if isinstance(mask_data, torch.Tensor):
            return mask_data.to(self.device)
        if isinstance(mask_data, list):
            if len(mask_data) == 0: return torch.zeros((0, 0, 0), dtype=torch.bool, device=self.device)
            decoded_masks = []
            for buffer in mask_data:
                if buffer is None: 
                    decoded_masks.append(torch.zeros((1024,1024), dtype=torch.bool, device=self.device))
                    continue
                mask_np = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask_np is None:
                     decoded_masks.append(torch.zeros((1024,1024), dtype=torch.bool, device=self.device))
                else:
                    mask_bool = torch.from_numpy(mask_np > 127).to(self.device)
                    decoded_masks.append(mask_bool)
            if not decoded_masks: return torch.zeros((0, 0, 0), dtype=torch.bool, device=self.device)
            return torch.stack(decoded_masks)
        return mask_data

    def load_data(self):
        self.nag = safe_load_torch(self.paths['cluster'], self.device)
        self.sp_pos = self.nag['sp_pos'].to(self.device)
        self.num_sp = self.sp_pos.shape[0]

        # 坐标还原
        if 'pos_offset' in self.nag:
            raw_offset = self.nag['pos_offset']
            if isinstance(raw_offset, np.ndarray):
                pos_offset = torch.from_numpy(raw_offset).to(self.device)
            else:
                pos_offset = raw_offset.to(self.device)
            if pos_offset.dim() == 2: pos_offset = pos_offset[0]
            print(f"    -> Applying Cluster Offset: {pos_offset.tolist()}")
            self.sp_pos = self.sp_pos + pos_offset

        match = re.search(r"x(-?\d+)_y(-?\d+)", self.grid_id)
        if match:
            off_x, off_y = float(match.group(1)), float(match.group(2))
            grid_offset = torch.tensor([off_x, off_y, 0.0], device=self.device).float()
            print(f"    -> Applying Grid Offset: {grid_offset.tolist()}")
            self.sp_pos = self.sp_pos + grid_offset
        else:
            print("    -> [Warning] Could not parse grid offset from filename!")
        
        if 'edge_weight' in self.nag:
            self.edge_weight = self.nag['edge_weight'].to(self.device)
            self.edge_index = self.nag['edge_index'].cpu().numpy()
        else:
            self.edge_weight = None
            self.edge_index = None

        self.sub_pointers = self.nag['sub_pointers'].cpu().numpy()
        self.sub_points = self.nag['sub_points'].cpu().numpy() 
        self.final_labels = torch.full((self.num_sp,), -1, dtype=torch.long, device=self.device)

    def get_hit_sps_precise(self, img_name, masks, pose, intr):
        _, _, d_path = self.get_file_paths(img_name)
        try:
            if os.path.exists(d_path):
                d_map_np = safe_load_numpy(d_path)
                d_map = torch.from_numpy(d_map_np).float().to(self.device)
            else:
                d_map = torch.zeros((1024, 1024), device=self.device)

            H, W = masks.shape[1], masks.shape[2]
            u, v, z = project_points_ortho(self.sp_pos, pose, intr, (H, W))
            visible = check_occlusion(u, v, z, d_map, self.device, tol=DEPTH_TOLERANCE)
            
            if not visible.any(): return [None]*len(masks)
            
            u_vis = u[visible].long()
            v_vis = v[visible].long()
            sp_indices_vis = torch.where(visible)[0]
            
            results = []
            for k in range(len(masks)):
                mask = masks[k]
                hits = mask[v_vis, u_vis]
                if not hits.any():
                    results.append(None)
                else:
                    hit_sps = sp_indices_vis[hits]
                    if len(hit_sps) < 3:
                        results.append(None)
                    else:
                        results.append(hit_sps)
            return results
        except Exception as e: 
            print(f"Projection Error in {img_name}: {e}")
            return [None]*len(masks)

    def step1_generate_instances_from_bev(self):
        if not os.path.exists(self.paths['mask_bev']): return []
        data = safe_load_torch(self.paths['mask_bev'], 'cpu')
        
        candidates = [] 
        print("    -> Generating Instances (Geometric Projection)...")
        
        for i, fname in enumerate(data['img_names']):
            try:
                p_path, i_path, _ = self.get_file_paths(fname)
                pose = safe_load_numpy(p_path)
                intr = safe_load_numpy(i_path)
            except: continue

            masks = self.decode_mask_data(data['masks'][i])
            if len(masks) == 0: continue
            
            labels = data['labels'][i].to(self.device)
            scores = data['scores'][i].to(self.device)
            
            hit_results = self.get_hit_sps_precise(fname, masks, pose, intr)
            
            # 识别视角类型
            if "Detail" in fname:
                view_type = "detail"
            elif "Global" in fname:
                view_type = "global"
            else:
                view_type = "other"
            
            for k, sp_indices in enumerate(hit_results):
                if sp_indices is None: continue
                lbl = labels[k].item()
                scr = scores[k].item()
                priority = scr + (2.0 if lbl in FOREGROUND_CLASSES else 0.0)
                
                # [修改] 实例分割：保留完整的实例信息
                candidates.append({
                    'sps': set(sp_indices.cpu().numpy()),
                    'label': lbl,
                    'score': scr,
                    'priority': priority,
                    'view_type': view_type,
                    'source_view': fname,
                    'mask_idx': k
                })
        
        candidates.sort(key=lambda x: x['priority'], reverse=True)
        
        # [修改] 实例分割：使用更严格的 NMS，只移除高度重叠的重复检测
        final_instances = [] 
        
        print(f"    -> NMS on {len(candidates)} candidates...")
        for cand in candidates:
            cand_sps = cand['sps']
            keep = True
            
            for exist in final_instances:
                exist_sps = exist['sps']
                inter = cand_sps.intersection(exist_sps)
                if not inter: continue
                
                # 计算 IoU
                union = cand_sps.union(exist_sps)
                iou = len(inter) / (len(union) + 1e-6)
                
                # 同类且 IoU 高：认为是重复检测，保留置信度高的
                if iou > 0.75 and cand['label'] == exist['label']:
                    keep = False
                    break
                # 不同类但重叠严重：根据优先级决定
                elif iou > 0.5 and cand['label'] != exist['label']:
                    if cand['priority'] < exist['priority']:
                        # 移除重叠部分
                        cand_sps -= inter
                        if len(cand_sps) < 5: 
                            keep = False
                            break
            
            if keep and len(cand_sps) >= 5:
                final_instances.append({
                    'sps': cand_sps,
                    'label': cand['label'],
                    'score': cand['score'],
                    'view_type': cand['view_type']
                })

        print(f"    -> Created {len(final_instances)} 3D instances.")
        return final_instances

    # ================= 核心 Step 2: 投票打标 (智能视角覆盖) =================

    def step2_instance_voting(self, instances):
        """
        [修改] 实例分割版本：跨视图投票，但按实例ID投票而非类别
        同一3D实例在多个视图中被检测，合并这些检测结果
        """
        if not instances: return
        
        cameras = []
        if os.path.exists(self.paths['mask_bev']):
            d = safe_load_torch(self.paths['mask_bev'], 'cpu')
            for i, n in enumerate(d['img_names']): cameras.append({'idx': i, 'name': n, 'data': d})
        if os.path.exists(self.paths['mask_tiles']):
            d = safe_load_torch(self.paths['mask_tiles'], 'cpu')
            for i, n in enumerate(d['img_names']): cameras.append({'idx': i, 'name': n, 'data': d})
        
        num_inst = len(instances)
        instance_votes = torch.zeros((num_inst, NUM_CLASSES), device=self.device)
        
        sp_to_inst = torch.full((self.num_sp,), -1, dtype=torch.long, device=self.device)
        for i, inst in enumerate(instances):
            indices = torch.tensor(list(inst['sps']), dtype=torch.long, device=self.device)
            sp_to_inst[indices] = i
        
        # [修改] 实例分割：记录每个实例的视图来源数量（用于验证实例稳定性）
        instance_view_count = torch.zeros(num_inst, device=self.device)
        
        print(f"    -> Instance Voting across {len(cameras)} views...")
        
        for cam in tqdm(cameras, desc=f"Voting {self.raw_scene_id}"):
            try:
                p_path, i_path, _ = self.get_file_paths(cam['name'])
                pose = safe_load_numpy(p_path)
                intr = safe_load_numpy(i_path)
            except: continue

            masks = self.decode_mask_data(cam['data']['masks'][cam['idx']])
            if len(masks) == 0: continue
            
            labels = cam['data']['labels'][cam['idx']].to(self.device)
            scores = cam['data']['scores'][cam['idx']].to(self.device)
            
            filtered_hits = self.get_hit_sps_precise(cam['name'], masks, pose, intr)
            
            # 计算可见性
            u, v, z = project_points_ortho(self.sp_pos, pose, intr, (1024, 1024))
            _, _, d_path = self.get_file_paths(cam['name'])
            d_map = safe_load_numpy(d_path) if os.path.exists(d_path) else np.zeros((1024,1024))
            d_map = torch.from_numpy(d_map).float().to(self.device)
            visible = check_occlusion(u, v, z, d_map, self.device, tol=DEPTH_TOLERANCE)
            vis_sps = torch.where(visible)[0]
            
            vis_inst_ids = sp_to_inst[vis_sps]
            vis_inst_ids = vis_inst_ids[vis_inst_ids != -1]
            if vis_inst_ids.numel() == 0: continue
            total_vis_counts = torch.bincount(vis_inst_ids, minlength=num_inst)

            for k in range(len(masks)):
                hit_sps = filtered_hits[k]
                if hit_sps is None: continue
                
                hit_inst_ids = sp_to_inst[hit_sps]
                hit_inst_ids = hit_inst_ids[hit_inst_ids != -1]
                if hit_inst_ids.numel() == 0: continue
                
                inter_counts = torch.bincount(hit_inst_ids, minlength=num_inst)
                valid_ids = torch.unique(hit_inst_ids)
                
                for inst_id in valid_ids:
                    inter = inter_counts[inst_id].item()
                    total = total_vis_counts[inst_id].item()
                    if total == 0: continue
                    ratio = inter / total
                    
                    lbl = labels[k].item()
                    score = scores[k].item()
                    
                    # [修改] 实例分割：基于IoU的投票
                    if ratio > VOTE_IOU_THRESH and score > VOTE_SCORE_THRESH:
                        instance_votes[inst_id, lbl] += ratio * score
                        instance_view_count[inst_id] += 1

        # 应用投票结果
        max_w, best_cls = torch.max(instance_votes, dim=1)
        valid = max_w > 0.001
        
        cnt = 0
        for i in range(num_inst):
            if valid[i]:
                indices = torch.tensor(list(instances[i]['sps']), dtype=torch.long, device=self.device)
                self.final_labels[indices] = best_cls[i]
                cnt += 1
        
        print(f"    -> Labeled {cnt}/{num_inst} instances.")
        print(f"    -> Avg views per instance: {instance_view_count[valid].mean():.1f}")

    def step4_graph_diffusion(self):
        if self.edge_weight is None: return
        print("    -> Graph Diffusion...")
        probs = torch.zeros((self.num_sp, NUM_CLASSES), device=self.device)
        known = torch.where(self.final_labels != -1)[0]
        probs[known, self.final_labels[known]] = 1.0
        
        src, dst = torch.from_numpy(self.edge_index).long().to(self.device)
        w = self.edge_weight
        mask = w > 0.05
        src, dst, w = src[mask], dst[mask], w[mask]
        
        for _ in range(3):
            msg = probs[src] * w.unsqueeze(1)
            upd = torch.zeros_like(probs)
            upd.index_add_(0, dst, msg)
            probs = 0.5 * probs + 0.5 * upd
            probs[known] = 0
            probs[known, self.final_labels[known]] = 1.0
            probs /= (probs.sum(1, keepdim=True) + 1e-6)
            
        scores, preds = torch.max(probs, dim=1)
        fill = (self.final_labels == -1) & (scores > 0.01)
        self.final_labels[fill] = preds[fill]

    def save_ply(self):
        try:
            with open(self.paths['raw_ply'], 'rb') as f: plydata = plyfile.PlyData.read(f)
        except: return
        N = plydata['vertex'].count
        pred_labels = np.full(N, -1, dtype=np.int32)
        sp_lbls = self.final_labels.cpu().numpy()
        for i in range(self.num_sp):
            if sp_lbls[i] == -1: continue
            s, e = self.sub_pointers[i], self.sub_pointers[i+1]
            pred_labels[self.sub_points[s:e]] = sp_lbls[i]

        new_data = []
        for prop in plydata['vertex'].properties:
            if prop.name in ['label', 'pred']: continue
            new_data.append((prop.name, plydata['vertex'][prop.name]))
        new_data.append(('pred', pred_labels.astype('i4')))
        arr = np.empty(N, dtype=[(n, d.dtype.str) for n, d in new_data])
        for n, d in new_data: arr[n] = d
        
        with open(os.path.join(self.output_root, f"{self.raw_scene_id}_fusion.ply"), 'wb') as f:
            plyfile.PlyData([plyfile.PlyElement.describe(arr, 'vertex')], text=False).write(f)

    def run(self):
        print(f"🚀 Processing {self.raw_scene_id} (Smart View Priority)...")
        self.load_data()
        instances = self.step1_generate_instances_from_bev()
        self.step2_voting(instances)
        self.step4_graph_diffusion()
        self.save_ply()

def process_scene_batch(scene_list, gpu_id, args):
    torch.cuda.set_device(gpu_id)
    for sid in scene_list:
        try:
            cluster_file = os.path.join(args.cluster_path, f"{sid}_nag.pt")
            pipe = PixelPreciseFusionPipeline(
                sid, cluster_file, args.snap_root, 
                args.mask_root, args.raw_ply_folder, args.output_root, 
                device=f"cuda:{gpu_id}"
            )
            pipe.run()
        except Exception as e:
            print(f"❌ Error {sid}: {e}")
            traceback.print_exc()

def scan_scenes(cluster_path):
    # 扫描所有 .pt 文件
    return sorted([os.path.basename(f).replace("_nag.pt", "") 
                   for f in glob.glob(os.path.join(cluster_path, "*_nag.pt"))])

if __name__ == "__main__":
    try: mp.set_start_method('spawn', force=True)
    except: pass
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene_id', type=str)
    group.add_argument('--parallel', action='store_true')
    parser.add_argument('--cluster_path', type=str, required=True)
    parser.add_argument('--snap_root', type=str, required=True)
    parser.add_argument('--mask_root', type=str, required=True)
    parser.add_argument('--raw_ply_folder', type=str, required=True)
    parser.add_argument('--output_root', type=str, default="./fusion_results")
    parser.add_argument('--gpus', type=str, default="0")
    args = parser.parse_args()

    if args.scene_id:
        process_scene_batch([args.scene_id], int(args.gpus.split(',')[0]), args)
    else:
        # 1. 扫描所有 Block
        scenes = scan_scenes(args.cluster_path)
        
        # 2. 随机打乱，实现“全局抽样预览”
        print(f"🔀 Found {len(scenes)} blocks. Shuffling for random inference order...")
        random.shuffle(scenes)
        
        gpus = [int(x) for x in args.gpus.split(',')]
        buckets = [[] for _ in range(len(gpus))]
        for i, s in enumerate(scenes): buckets[i % len(gpus)].append(s)
        procs = []
        for i, g in enumerate(gpus):
            if not buckets[i]: continue
            p = mp.Process(target=process_scene_batch, args=(buckets[i], g, args))
            p.start()
            procs.append(p)
        for p in procs: p.join()
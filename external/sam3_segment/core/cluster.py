import os
import glob
import torch
import numpy as np
import plyfile
import open3d as o3d 
import torch.multiprocessing as mp
import colorsys
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import argparse

from src.data import Data
from src.data.cluster import Cluster
from src.transforms import (
    KNN, 
    PointFeatures, 
    AdjacencyGraph, 
    CutPursuitPartition
)
from torchvision.transforms import Compose

try:
    from torch_scatter import scatter_mean
    HAS_SCATTER = True
except ImportError:
    HAS_SCATTER = False

# =================【配置加载逻辑】=================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 初始化参数解析
parser = argparse.ArgumentParser(description="Batch Cluster Processing")
parser.add_argument('--config', type=str, default='./config/cluster/sum.yaml', help='Path to the yaml config file')
parser.add_argument('--input', type=str, default=None, help='Single PLY file (overrides config input_folder when used with --output)')
parser.add_argument('--output', type=str, default=None, help='Single NAG output path (overrides config output_root when used with --input)')
args, _ = parser.parse_known_args()

# 加载全局配置 CFG
if os.path.exists(args.config):
    CFG = load_config(args.config)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, args.config)
    if os.path.exists(config_path):
        CFG = load_config(config_path)
    else:
        if __name__ == "__main__":
             # 如果是直接运行脚本且找不到配置，给一个默认空字典防报错，但在 process 中会失败
             # 建议用户确保路径正确
             raise FileNotFoundError(f"Config file not found at {args.config}")
        else:
             CFG = {} 

# =========================================================

def generate_distinct_colors(n_colors=100):
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    h = np.random.random()
    for i in range(n_colors):
        h += golden_ratio_conjugate
        h %= 1.0
        s = 0.5 + np.random.random() * 0.5 
        v = 0.6 + np.random.random() * 0.4 
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([int(r*255), int(g*255), int(b*255)])
    return np.array(colors, dtype=np.uint8)

def get_smart_colors(num_sp, edge_index):
    palette_size = max(200, num_sp // 10)
    base_palette = generate_distinct_colors(palette_size)
    if num_sp <= 1: return np.array([[255, 255, 255]], dtype=np.uint8)
    if isinstance(edge_index, torch.Tensor): edge_index = edge_index.cpu().numpy()
    adj = {}
    if edge_index.shape[1] > 0:
        src, tgt = edge_index[0], edge_index[1]
        for s, t in zip(src, tgt):
            if s == t: continue
            if s not in adj: adj[s] = []
            if t not in adj: adj[t] = []
            adj[s].append(t)
            adj[t].append(s)
    color_indices = np.full(num_sp, -1, dtype=int)
    for i in range(num_sp):
        neighbors = adj.get(i, [])
        used_indices = set()
        for n in neighbors:
            if n < num_sp and color_indices[n] != -1: used_indices.add(color_indices[n])
        c = 0
        while c in used_indices: c += 1
        color_indices[i] = c
    return base_palette[color_indices % len(base_palette)]

def process_one_scene(ply_path, output_dir, device_id):
    device = torch.device(f"cuda:{device_id}")
    
    # === 参数提取区 ===
    params = CFG['clustering']
    p_voxel   = params['voxel_size']
    p_radius  = params['normal_radius']
    p_knn     = params['knn_k']
    p_reg     = params['regularization']
    p_spatial = params['spatial_weight']
    
    weights = params['weights']
    w_color  = weights['color_boost']
    w_height = weights['height_boost']
    w_geom   = weights['geom_boost']
    w_pos    = weights['pos_boost']
    
    out_cfg = CFG['output']
    s_color = out_cfg['sigma_color']
    s_pos   = out_cfg['sigma_pos']
    n_pow   = out_cfg['normal_pow']
    # ===============================

    scene_name = os.path.splitext(os.path.basename(ply_path))[0]
    nag_save_path = os.path.join(output_dir, f"{scene_name}_nag.pt")
    vis_save_path = os.path.join(output_dir, f"{scene_name}_vis.ply")
    
    if os.path.exists(nag_save_path): return

    # === 1. 读取数据 ===
    try:
        plydata = plyfile.PlyData.read(ply_path)
    except Exception as e:
        print(f"Error reading {ply_path}: {e}")
        return

    vertex_data = plydata['vertex']
    raw_pos = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    
    # 基础数量检查
    if len(raw_pos) < p_knn + 50: 
        tqdm.write(f"[{scene_name}] Skipped: Too few points ({len(raw_pos)})")
        return

    # =========================================================
    # 【步骤 A】 强力离群点去除 (Outlier Removal)
    # =========================================================
    try:
        # 构建 Open3D 点云用于去噪
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(raw_pos)
        
        # 1. 统计式去噪 (Statistical Outlier Removal)
        _, ind_stat = pcd_temp.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        
        # 如果去噪删除了点，应用掩码
        if len(ind_stat) < len(raw_pos):
            raw_pos = raw_pos[ind_stat]
            vertex_data = vertex_data[ind_stat] # 关键：属性也要对应删除
            pcd_temp = pcd_temp.select_by_index(ind_stat)

        # 2. 半径去噪 (Radius Outlier Removal)
        _, ind_rad = pcd_temp.remove_radius_outlier(nb_points=10, radius=p_voxel * 5.0)
        
        if len(ind_rad) < len(raw_pos):
            raw_pos = raw_pos[ind_rad]
            vertex_data = vertex_data[ind_rad]
            pcd_temp = pcd_temp.select_by_index(ind_rad)
            
    except Exception as e:
        tqdm.write(f"[{scene_name}] Warning: Outlier removal failed ({e}), using raw data.")

    # 去噪后二次检查
    if len(raw_pos) < p_knn + 1:
        return

    # === 1.5 【显存优化】点数超限时体素下采样，避免 OOM ===
    max_points = CFG.get('clustering', {}).get('max_points', None)  # 不设则不下采样
    if max_points is not None and len(raw_pos) > max_points:
        n_orig = len(raw_pos)
        # 目标点数约 max_points，反推体素边长（每体素保留 1 点）
        vol = np.prod(raw_pos.max(axis=0) - raw_pos.min(axis=0) + 1e-6)
        voxel_size_down = (vol / float(max_points)) ** (1.0 / 3.0)
        voxel_size_down = max(voxel_size_down, p_voxel * 1.5)  # 不小于基础 voxel_size
        voxel_ids = np.floor(raw_pos / voxel_size_down).astype(np.int32)
        # 每个体素保留第一个点的索引
        dtype_view = np.dtype((np.void, voxel_ids.dtype.itemsize * voxel_ids.shape[1]))
        _, keep_idx = np.unique(np.ascontiguousarray(voxel_ids).view(dtype_view), return_index=True)
        keep_idx = np.sort(keep_idx)
        raw_pos = raw_pos[keep_idx]
        vertex_data = vertex_data[keep_idx]
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(raw_pos)
        tqdm.write(f"[{scene_name}] Downsampled {n_orig} -> {len(raw_pos)} points (voxel={voxel_size_down:.3f}m) for GPU memory")

    # === 2. 计算法向量 ===
    pcd_temp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=p_radius, max_nn=30))
    raw_normal = np.asarray(pcd_temp.normals)

    # === 3. 数据去中心化与清洗 (CRITICAL FIX) ===
    
    # [FIX] 禁用去中心化，直接使用原始 PLY 坐标
    # 这样做是为了保证生成的 _nag.pt 坐标系与 Render 管线 (基于 Grid/Global) 完全一致
    # 避免出现 min() 引起的随机偏移 (25m问题)
    pos_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pos_centered = raw_pos.astype(np.float32)

    # [FIX] 再次清洗 NaN/Inf (双重保险)
    valid_mask = ~np.isnan(pos_centered).any(axis=1) & ~np.isinf(pos_centered).any(axis=1) & \
                 ~np.isnan(raw_normal).any(axis=1) & ~np.isinf(raw_normal).any(axis=1)

    if np.sum(~valid_mask) > 0:
        pos_centered = pos_centered[valid_mask]
        raw_normal = raw_normal[valid_mask]
        vertex_data = vertex_data[valid_mask]

    # 获取颜色
    field_names = vertex_data.dtype.names if vertex_data.dtype.names else []
    has_color = 'red' in field_names and 'green' in field_names and 'blue' in field_names
    if has_color:
        raw_rgb = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
        real_rgb = raw_rgb.astype(np.float32) / 255.0
    else:
        z_vals = pos_centered[:, 2]
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-6)
        real_rgb = np.stack([np.sin(z_norm*50), np.cos(z_norm*30), z_norm], axis=1)

    # [FIX] 最终数据验证
    rgb_valid_mask = ~np.isnan(real_rgb).any(axis=1) & ~np.isinf(real_rgb).any(axis=1)
    
    if np.sum(~rgb_valid_mask) > 0:
        pos_centered = pos_centered[rgb_valid_mask]
        raw_normal = raw_normal[rgb_valid_mask]
        real_rgb = real_rgb[rgb_valid_mask]
    
    if len(pos_centered) < p_knn + 1:
        tqdm.write(f"[{scene_name}] Skipped: Too few valid points after cleaning ({len(pos_centered)})")
        return
    
    # 确保数据范围合理
    pos_centered = np.clip(pos_centered, -1e6, 1e6).astype(np.float32)
    real_rgb = np.clip(real_rgb, -10.0, 10.0).astype(np.float32)
    raw_normal = np.clip(raw_normal, -1.0, 1.0).astype(np.float32)
    
    # 归一化法向量
    normal_norm = np.linalg.norm(raw_normal, axis=1, keepdims=True)
    normal_norm = np.where(normal_norm > 1e-6, normal_norm, 1.0)
    raw_normal = raw_normal / normal_norm

    # === 4. 构造 Tensor (强制 Contiguous) ===
    try:
        data = Data(
            pos=torch.from_numpy(pos_centered).contiguous(), 
            rgb=torch.from_numpy(real_rgb).contiguous(), 
            normal=torch.from_numpy(raw_normal).float().contiguous(), 
            original_point_idx=torch.arange(len(pos_centered), dtype=torch.long)
        )
        data.raw_mixed_rgb = data.rgb.clone()
        
        # 验证 tensor 有效性
        if torch.isnan(data.pos).any() or torch.isinf(data.pos).any():
            raise ValueError("Invalid values in pos tensor")
        if torch.isnan(data.rgb).any() or torch.isinf(data.rgb).any():
            raise ValueError("Invalid values in rgb tensor")
        if torch.isnan(data.normal).any() or torch.isinf(data.normal).any():
            raise ValueError("Invalid values in normal tensor")
        
        # 移动到 GPU
        data = data.to(device)
        torch.cuda.synchronize()
        
    except (RuntimeError, ValueError) as e:
        tqdm.write(f"[{scene_name}] Error creating/transferring data to GPU: {e}")
        if 'data' in locals():
            try: del data
            except: pass
        try:
            torch.cuda.empty_cache()
        except:
            pass  # 避免在CUDA错误状态下调用empty_cache导致二次错误
        return
    
    # === 5. 预处理管道 ===
    pre_transform = Compose([
        KNN(k=p_knn, r_max=p_voxel * 1.5),  
        PointFeatures(keys=('linearity', 'planarity', 'verticality', 'scattering')),
        AdjacencyGraph(k=p_knn)
    ])

    # === 6. 聚类器定义 ===
    partitioner = CutPursuitPartition(
        regularization=[p_reg], 
        spatial_weight=[p_spatial], 
        cutoff=[0],     
        iterations=100
    )

    try:
        torch.cuda.synchronize()
        
        # A. 几何特征
        data = pre_transform(data)

        # B. 特征注入
        feat_color = data.raw_mixed_rgb * w_color   
        feat_geom = data.normal * w_geom            
        feat_pos = data.pos * w_pos
        feat_z = data.pos[:, 2:3] * w_height 
        
        custom_features = torch.cat([feat_color, feat_geom, feat_pos, feat_z], dim=1)

        if data.x is not None:
            data.x = torch.cat([data.x, custom_features], dim=1)
        else:
            data.x = custom_features
            
        # C. 聚类
        nag = partitioner(data)

    except RuntimeError as e:
        tqdm.write(f"[GPU {device_id}] Runtime Error in {scene_name}: {e}")
        if 'data' in locals():
            try: 
                del data
            except: 
                pass
        try: 
            torch.cuda.empty_cache() 
        except: 
            pass
        return
    
    # === 结果处理与保存 ===
    sp_data = nag[0]
    super_index = sp_data.super_index
    num_sp = super_index.max().item() + 1
    num_original = len(pos_centered)
    
    ratio = num_original / num_sp if num_sp > 0 else 0
    tqdm.write(f"[{scene_name}] Pts: {num_original} -> SPs: {num_sp} (Ratio: {ratio:.1f}x)")

    # 1. 聚合超点中心
    sp_pos = torch.zeros((num_sp, 3), device=device)
    sp_normal = torch.zeros((num_sp, 3), device=device)
    
    if HAS_SCATTER:
        sp_pos = scatter_mean(sp_data.pos, super_index, dim=0, dim_size=num_sp)
        if hasattr(sp_data, 'normal'):
            sp_normal = scatter_mean(sp_data.normal, super_index, dim=0, dim_size=num_sp)
    else:
        for i in range(num_sp):
            mask = (super_index == i)
            if mask.any():
                sp_pos[i] = sp_data.pos[mask].mean(dim=0)
                if hasattr(sp_data, 'normal'):
                     sp_normal[i] = sp_data.normal[mask].mean(dim=0)
    
    norms = torch.norm(sp_normal, dim=1, keepdim=True)
    sp_normal = sp_normal / (norms + 1e-6)

    # 2. 构建超点邻接图
    if hasattr(sp_data, 'edge_index'):
        raw_edge_index = sp_data.edge_index
        sp_src = super_index[raw_edge_index[0]]
        sp_tgt = super_index[raw_edge_index[1]]
        mask = sp_src != sp_tgt
        sp_edges = torch.stack([sp_src[mask], sp_tgt[mask]], dim=0)
        edge_index = torch.unique(sp_edges, dim=1) if sp_edges.shape[1] > 0 else torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # 3. 计算边权重
    edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
    if edge_index.shape[1] > 0:
        s, d = edge_index[0], edge_index[1]
        
        raw_rgb_gpu = data.raw_mixed_rgb
        if HAS_SCATTER:
             sp_real_rgb = scatter_mean(raw_rgb_gpu, super_index, dim=0, dim_size=num_sp)
        else:
             sp_real_rgb = torch.zeros((num_sp, 3), device=device)
             for i in range(num_sp):
                 mask = (super_index == i)
                 if mask.any(): sp_real_rgb[i] = raw_rgb_gpu[mask].mean(dim=0)

        c_diff = torch.norm(sp_real_rgb[s] - sp_real_rgb[d], dim=1)
        w_color = torch.exp(-c_diff / s_color)
        p_diff = torch.norm(sp_pos[s] - sp_pos[d], dim=1)
        w_pos = torch.exp(-p_diff / s_pos)
        n_dot = (sp_normal[s] * sp_normal[d]).sum(dim=1).clamp(-1.0, 1.0)
        w_normal = ((n_dot + 1.0) / 2.0).pow(n_pow)
        edge_weight = w_color * w_pos * w_normal

    # 4. 构建 Sub-Cluster 结构
    sp_to_points_list = [[] for _ in range(num_sp)]
    cpu_super_index = super_index.cpu().numpy()
    for idx, sp_id in enumerate(cpu_super_index):
        sp_to_points_list[sp_id].append(idx)
        
    flat_indices = []
    ptr_list = [0]
    for sp_id in range(num_sp):
        flat_indices.extend(sp_to_points_list[sp_id])
        ptr_list.append(len(flat_indices))
        
    sp_data.sub = Cluster(
        torch.tensor(ptr_list, dtype=torch.long, device=device),
        torch.tensor(flat_indices, dtype=torch.long, device=device),
        dense=False
    )

    # === 保存 NAG (.pt) ===
    save_dict = {
        "super_index": super_index.cpu(),
        "edge_index": edge_index.cpu(),    
        "edge_weight": edge_weight.cpu(),   
        "sp_pos": sp_pos.cpu(),             
        "sp_normal": sp_normal.cpu(),       
        "pos_offset": pos_min, # 现在是 [0,0,0]             
        "voxel_pos": sp_data.pos.cpu(),     
        "sub_pointers": sp_data.sub.pointers.cpu(),
        "sub_points": sp_data.sub.points.cpu()
    }
    torch.save(save_dict, nag_save_path)
    
    # === 保存可视化 (.ply) ===
    palette = get_smart_colors(num_sp, edge_index)
    sp_vis_colors = palette[super_index.cpu().numpy()]
    
    out_x = (sp_data.pos[:, 0].cpu().numpy() + pos_min[0]).astype(np.float32)
    out_y = (sp_data.pos[:, 1].cpu().numpy() + pos_min[1]).astype(np.float32)
    out_z = (sp_data.pos[:, 2].cpu().numpy() + pos_min[2]).astype(np.float32)
    
    elements = np.empty(len(out_x), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('sp_id', 'i4')])
    elements['x'] = out_x; elements['y'] = out_y; elements['z'] = out_z
    elements['red'] = sp_vis_colors[:, 0]; elements['green'] = sp_vis_colors[:, 1]; elements['blue'] = sp_vis_colors[:, 2]
    elements['sp_id'] = super_index.cpu().numpy()
    
    el = plyfile.PlyElement.describe(elements, 'vertex')
    plyfile.PlyData([el], text=False).write(vis_save_path)
    
    del data, nag, sp_data
    try: torch.cuda.empty_cache()
    except: pass

def worker_func(gpu_id, file_subset, worker_idx):
    torch.set_num_threads(2)
    # 显式设置CUDA设备
    try:
        torch.cuda.set_device(gpu_id)
        print(f"Worker {worker_idx} set to GPU {gpu_id}")
    except Exception as e:
        print(f"Worker {worker_idx} failed to set GPU {gpu_id}: {e}")
        return
    
    output_root = CFG['paths']['output_root']
    desc = f"Worker {worker_idx} (GPU {gpu_id})"
    for ply_path in tqdm(file_subset, desc=desc, position=worker_idx):
        try:
            process_one_scene(ply_path, output_root, gpu_id)
        except Exception as e:
            tqdm.write(f"[Worker {worker_idx}] Unexpected error processing {ply_path}: {e}")
            try:
                torch.cuda.empty_cache()
            except:
                pass

def main():
    mp.set_start_method('spawn', force=True)
    
    # 调试模式：添加CUDA启动阻塞以获取准确的错误位置
    if os.environ.get('CUDA_LAUNCH_BLOCKING') != '1':
        print("⚠️  Tip: Set CUDA_LAUNCH_BLOCKING=1 for better error location when debugging CUDA illegal memory access")
        print("     Example: CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=6,7 python batch_cluster.py --config ./config/cluster/sensaturban.yaml")
    
    # 单文件模式：--input + --output 覆盖 config 路径
    if args.input and args.output:
        ply_files = [args.input]
        output_root = os.path.dirname(args.output)
        input_folder = os.path.dirname(args.input)
        if not os.path.isfile(args.input):
            print(f"Error: --input file not found: {args.input}")
            return
    else:
        input_folder = CFG['paths']['input_folder']
        output_root = CFG['paths']['output_root']
        ply_files = glob.glob(os.path.join(input_folder, "*.ply"))
        ply_files.sort()

    processes_per_gpu = CFG['system']['processes_per_gpu']

    os.makedirs(output_root, exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: 
        print("Error: No GPU found.")
        return

    total_workers = num_gpus * processes_per_gpu
    print(f"Loaded config from: {args.config}")
    print(f"Input Folder: {input_folder}")
    print(f"Output Root: {output_root}")
    print(f"Found {len(ply_files)} files.")
    print(f"Launching {total_workers} processes on {num_gpus} GPUs.")
    
    files_chunks = np.array_split(ply_files, total_workers)
    processes = []
    
    for i in range(total_workers):
        subset = files_chunks[i].tolist()
        if not subset: continue
        assigned_gpu_id = i % num_gpus
        p = mp.Process(target=worker_func, args=(assigned_gpu_id, subset, i))
        p.start()
        processes.append(p)
    
    for p in processes: p.join()

if __name__ == "__main__":
    main()
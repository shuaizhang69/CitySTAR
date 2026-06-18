import argparse
import os
import json
import numpy as np
import torch
import torch.multiprocessing as mp
import plyfile
from PIL import Image

# PyTorch3D Imports
from pytorch3d.renderer import (
    OrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer, AlphaCompositor,
    look_at_view_transform
)
from pytorch3d.structures import Pointclouds

# ==========================================
# 0. 全局物理配置
# ==========================================
SEARCH_RADIUS_METER = 450.0   
GLOBAL_CAM_DIST = 500.0   
DETAIL_CAM_DIST = 200.0   
SIDE_CAM_DIST = 300.0
IMG_SIZE = 1024
POINTS_PER_PIXEL = 15     
MAX_POINTS_PER_BIN = 8000000 

# ==========================================
# 1. 数据加载与工具函数
# ==========================================

def load_ply_block_local(filepath):
    try:
        with open(filepath, "rb") as f:
            plydata = plyfile.PlyData.read(f)
        data = plydata["vertex"].data
        x = data['x'].astype(np.float32)
        y = data['y'].astype(np.float32)
        z = data['z'].astype(np.float32)
        verts_local = torch.from_numpy(np.stack([x, y, z], axis=-1))
        
        if 'red' in data.dtype.names:
            r = data['red'].astype(np.float32) / 255.0
            g = data['green'].astype(np.float32) / 255.0
            b = data['blue'].astype(np.float32) / 255.0
            verts_rgb = torch.from_numpy(np.stack([r, g, b], axis=-1))
        else:
            verts_rgb = torch.ones_like(verts_local) * 0.5
        return verts_local, verts_rgb
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None, None

def downsample_with_ids(xyz, rgb, ids, max_points=200000000):
    """
    [修改] 支持 ID 为 None 的情况
    """
    N = xyz.shape[0]
    if N > max_points:
        step = N // max_points
        xyz = xyz[::step][:max_points]
        rgb = rgb[::step][:max_points]
        if ids is not None:
            ids = ids[::step][:max_points]
    return xyz, rgb, ids

def load_wide_context_with_tracking(target_grid_id, metadata, input_folder, device, generate_ids=False):
    """
    [修改] 增加 generate_ids 开关
    """
    target_info = metadata[target_grid_id]
    tgt_ox, tgt_oy, _ = target_info['offset']
    target_scene = target_info.get('original_scene') 
    
    files_to_load = []
    for fname, info in metadata.items():
        if info.get('original_scene') != target_scene:
            continue
        ox, oy, _ = info['offset']
        dist_x = abs(float(ox) - float(tgt_ox))
        dist_y = abs(float(oy) - float(tgt_oy))
        if dist_x < SEARCH_RADIUS_METER and dist_y < SEARCH_RADIUS_METER:
            files_to_load.append((fname, info['offset']))
            
    neighbors_xyz, neighbors_rgb, neighbors_ids = [], [], []
    
    for fname, offset in files_to_load:
        path = os.path.join(input_folder, fname)
        if not os.path.exists(path): continue
        
        v_local, c = load_ply_block_local(path)
        if v_local is not None:
            N = v_local.shape[0]
            
            # === [开关控制] 只有在需要 ID Map 时才生成 ID ===
            ids = None
            if generate_ids:
                if fname == target_grid_id:
                    ids = torch.arange(N, dtype=torch.int32)
                else:
                    ids = torch.full((N,), -1, dtype=torch.int32)
            
            shift = torch.tensor([float(offset[0]) - float(tgt_ox), 
                                 float(offset[1]) - float(tgt_oy), 
                                 float(offset[2])], dtype=torch.float32)
            
            neighbors_xyz.append(v_local + shift)
            neighbors_rgb.append(c)
            if generate_ids:
                neighbors_ids.append(ids)
            
    if not neighbors_xyz: return None, None, None
    
    full_xyz = torch.cat(neighbors_xyz, dim=0)
    full_rgb = torch.cat(neighbors_rgb, dim=0)
    full_ids = torch.cat(neighbors_ids, dim=0) if generate_ids else None
    
    # 下采样
    full_xyz, full_rgb, full_ids = downsample_with_ids(full_xyz, full_rgb, full_ids)
    
    # 转到 GPU
    full_xyz = full_xyz.to(device)
    full_rgb = full_rgb.to(device)
    if full_ids is not None:
        full_ids = full_ids.to(device)
    
    return full_xyz, full_rgb, full_ids

def get_scene_bounds(verts_xyz):
    N = verts_xyz.shape[0]
    if N < 100: return 0.0, 10.0
    step = max(1, N // 100000)
    samples = verts_xyz[::step, 2]
    z_min = torch.quantile(samples, 0.05).item()
    z_max = torch.quantile(samples, 0.98).item()
    return z_min, z_max

# ==========================================
# 2. 核心数学逻辑
# ==========================================
def get_safe_camera_R_T(target_pos, pitch_deg, azim_idx, total_views, radius, min_safe_z):
    phi_rad = np.deg2rad(90 + pitch_deg)
    azim_rad = np.deg2rad((azim_idx / total_views) * 360.0)
    z_rel = radius * np.cos(phi_rad)
    r_xy  = radius * np.sin(phi_rad)
    x_rel = r_xy * np.cos(azim_rad)
    y_rel = r_xy * np.sin(azim_rad)
    camera_pos = target_pos + np.array([x_rel, y_rel, z_rel])
    if camera_pos[2] < min_safe_z: camera_pos[2] = min_safe_z
    dx, dy = camera_pos[0] - target_pos[0], camera_pos[1] - target_pos[1]
    dist_xy = np.sqrt(dx*dx + dy*dy)
    if dist_xy < 1.0: up_vector = ((0, 1, 0),) 
    else: up_vector = ((0, 0, 1),)
    R, T = look_at_view_transform(eye=torch.tensor([camera_pos], dtype=torch.float32), at=torch.tensor([target_pos], dtype=torch.float32), up=up_vector)
    return R, T

def get_ortho_scale(fit_radius, image_size):
    physical_diameter = fit_radius * 2.0
    margin_factor = 1.1 
    scale = 2.0 / (physical_diameter * margin_factor)
    return scale

def convert_RT_to_C2W_Global(R, T, global_offset_vec):
    R_np = R[0].cpu().numpy()
    T_np = T[0].cpu().numpy()
    c2w = np.eye(4)
    c2w[:3, :3] = R_np.T
    c2w[:3, 3] = -np.dot(R_np.T, T_np)
    c2w_global = c2w.copy()
    c2w_global[:3, 3] += global_offset_vec 
    return c2w_global

# ==========================================
# 3. 渲染任务逻辑 (修改：可选 ID Map)
# ==========================================

def render_task(task, verts_xyz, verts_rgb, full_ids, args, device):
    """
    [修改] 根据 full_ids 是否为 None 来决定是否渲染 ID Map
    """
    origin = task["origin"]
    radius_crop = 400.0 if ("Global" in task["name"] or "obl" in task["name"]) else 250.0
    
    mask = (torch.abs(verts_xyz[:, 0] - origin[0]) < radius_crop) & (torch.abs(verts_xyz[:, 1] - origin[1]) < radius_crop)
    if mask.sum() < 50: return None
    
    verts_active = verts_xyz[mask]
    rgb_active = verts_rgb[mask]
    
    # 只有当开启了 ID Buffer 且 full_ids 存在时，才处理 ID
    ids_active = None
    if full_ids is not None:
        ids_active = full_ids[mask] 
    
    # 光照处理
    grayscale = rgb_active[:, 0] * 0.299 + rgb_active[:, 1] * 0.587 + rgb_active[:, 2] * 0.114
    grayscale = grayscale.unsqueeze(1)
    rgb_active = grayscale + (rgb_active - grayscale) * 1.3
    contrast_factor = 1.1  
    rgb_active = (rgb_active - 0.5) * contrast_factor + 0.5
    brightness_factor = 1.05 
    rgb_active = rgb_active * brightness_factor
    rgb_active = torch.clamp(rgb_active, 0.0, 1.0)
    
    R, T = task["R"], task["T"]
    H, W = args.height, args.width
    fit_r = task.get("fit_radius", 35.0)
    ortho_scale = get_ortho_scale(fit_r, (H, W))
    
    camera = OrthographicCameras(
        focal_length=((ortho_scale, ortho_scale),), 
        principal_point=((0.0, 0.0),), 
        R=R.to(device), T=T.to(device),
        image_size=torch.tensor([[H, W]], device=device), 
        in_ndc=True, 
        device=device
    )
    
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W), 
        radius=task.get("render_radius", 0.0015), 
        points_per_pixel=POINTS_PER_PIXEL,
        max_points_per_bin=MAX_POINTS_PER_BIN
    )
    
    rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor(background_color=(0, 0, 0)))
    
    try:
        pcd = Pointclouds(points=[verts_active], features=[rgb_active])
        image = renderer(pcd)
        fragments = rasterizer(pcd)
        
        # === [核心修改] ID Map 映射逻辑 (可选) ===
        final_id_map_np = None
        
        if ids_active is not None:
            # fragments.idx 的形状: (N, H, W, K)
            local_idx_map = fragments.idx[0, ..., 0] 
            
            final_id_map = torch.full_like(local_idx_map, -1, dtype=torch.int32)
            valid_mask = local_idx_map != -1
            hit_active_indices = local_idx_map[valid_mask]
            
            # 查表映射
            final_id_map[valid_mask] = ids_active[hit_active_indices]
            final_id_map_np = final_id_map.cpu().numpy()
        
        depth = fragments.zbuf[0, ..., 0].cpu().numpy()
        depth[depth == -1] = 0 
        
        K_dummy = np.eye(3, dtype=np.float32)
        K_dummy[0,0] = ortho_scale
        K_dummy[1,1] = ortho_scale
        
        return (image[0, ..., :3].cpu().numpy(), 
                depth, 
                K_dummy, 
                final_id_map_np) # 如果没开启，这里返回 None
                
    except Exception as e: 
        print(f"Render Error: {e}")
        return None

# ==========================================
# 4. 主流程
# ==========================================

def process_grid_block(block_key, metadata, global_center, input_folder, output_root, args, worker_id, device):
    info = metadata[block_key]
    tgt_ox = float(info['offset'][0])
    tgt_oy = float(info['offset'][1])
    grid_id = f"x{int(tgt_ox)}_y{int(tgt_oy)}"
    scene_name = os.path.basename(info.get('original_scene', 'scene')).replace(".ply", "")
    
    scene_dir = os.path.join(output_root, scene_name)
    tile_dir = os.path.join(scene_dir, grid_id)

    # 基础文件夹
    for f in ["dev", "image", "pose", "intrinsic", "depth"]:
        os.makedirs(os.path.join(tile_dir, f), exist_ok=True)
    
    # [修改] 只有开启时才创建 id_map 文件夹
    if args.use_id_buffer:
        os.makedirs(os.path.join(tile_dir, "id_map"), exist_ok=True)

    # [修改] 传入 args.use_id_buffer 开关
    v_wide_xyz, v_wide_rgb, v_wide_ids = load_wide_context_with_tracking(
        block_key, metadata, input_folder, device, generate_ids=args.use_id_buffer
    )
    
    if v_wide_xyz is None: return
    
    z_min, z_max = get_scene_bounds(v_wide_xyz)

    # --- 重心调整 ---
    search_min, search_max = -20.0, 70.0
    local_mask = (v_wide_xyz[:, 0] > search_min) & (v_wide_xyz[:, 0] < search_max) & \
                 (v_wide_xyz[:, 1] > search_min) & (v_wide_xyz[:, 1] < search_max)
    
    if local_mask.sum() > 50:
        local_points = v_wide_xyz[local_mask]
        center_x = torch.median(local_points[:, 0]).item()
        center_y = torch.median(local_points[:, 1]).item()
        print(f"[Worker {worker_id}] {grid_id} Adjusted Center: ({center_x:.1f}, {center_y:.1f})")
    else:
        center_x, center_y = 25.0, 25.0

    center_z = (z_min + z_max) / 2.0
    fixed_origin_center = np.array([center_x, center_y, center_z])
    global_offset_vec = np.array([tgt_ox, tgt_oy, 0])
    
    np.save(os.path.join(tile_dir, "dev", "center.npy"), np.array([center_x, center_y, z_min]) + global_offset_vec)
    
    tasks = []
    
    # --- Task A: Global ---
    safe_z_global = z_max + 50.0
    R_g, T_g = get_safe_camera_R_T(fixed_origin_center, -89.9, 0, 1, GLOBAL_CAM_DIST, safe_z_global)
    tasks.append({
        "name": f"{grid_id}_Global", "pitch": -89.9, "R": R_g, "T": T_g, 
        "origin": fixed_origin_center, "folder_group": "dev", 
        "fit_radius": 65.0, "render_radius": 0.004
    })
    
    # --- Task B: Detail ---
    safe_z_detail = z_max + 30.0
    R_d, T_d = get_safe_camera_R_T(fixed_origin_center, -89.9, 0, 1, DETAIL_CAM_DIST, safe_z_detail)
    tasks.append({
        "name": f"{grid_id}_Detail", "pitch": -89.9, "R": R_d, "T": T_d, 
        "origin": fixed_origin_center, "folder_group": "dev", 
        "fit_radius": 35.0, "render_radius": 0.0035
    })
    
    # --- Task C: Oblique/Side Views ---
    safe_z_side = z_max + 20.0
    for i in range(12):
        R, T = get_safe_camera_R_T(fixed_origin_center, -85, i, 12, SIDE_CAM_DIST, safe_z_side)
        tasks.append({
            "name": f"{grid_id}_obl_{i}", "pitch": -85, "R": R, "T": T, 
            "origin": fixed_origin_center, "folder_group": "image_set",
            "fit_radius": 45.0, "render_radius": 0.0035
        })
    for i in range(12):
        idx = 12 + i
        R, T = get_safe_camera_R_T(fixed_origin_center, -75, i, 12, SIDE_CAM_DIST, safe_z_side)
        tasks.append({
            "name": f"{grid_id}_obl_{idx}", "pitch": -75, "R": R, "T": T, 
            "origin": fixed_origin_center, "folder_group": "image_set",
            "fit_radius": 45.0, "render_radius": 0.0035
        })

    # 执行渲染
    for task in tasks:
        # [修改] 传入 v_wide_ids (可能是 None)
        res = render_task(task, v_wide_xyz, v_wide_rgb, v_wide_ids, args, device)
        if res is None: continue
        
        img_np, depth_np, K_out, id_map_np = res
        pil_img_real = Image.fromarray((np.clip(img_np, 0, 1) * 255).astype(np.uint8))
        c2w_global = convert_RT_to_C2W_Global(task["R"], task["T"], global_offset_vec)
        save_name = task["name"]
        
        if task["folder_group"] == "dev":
            base_path = os.path.join(tile_dir, "dev")
            pil_img_real.save(os.path.join(base_path, f"{save_name}_real.png"))
            np.save(os.path.join(base_path, f"intrinsic_{save_name}.npy"), K_out)
            np.save(os.path.join(base_path, f"pose_{save_name}.npy"), c2w_global)
            np.save(os.path.join(base_path, f"depth_{save_name}.npy"), depth_np)
            # [修改] 仅在有 ID Map 时保存
            if id_map_np is not None:
                np.save(os.path.join(base_path, f"id_map_{save_name}.npy"), id_map_np)
            
        elif task["folder_group"] == "image_set":
            pil_img_real.save(os.path.join(tile_dir, "image", f"{save_name}_real.png"))
            np.save(os.path.join(tile_dir, "intrinsic", f"intrinsic_{save_name}.npy"), K_out)
            np.save(os.path.join(tile_dir, "pose", f"pose_{save_name}.npy"), c2w_global)
            np.save(os.path.join(tile_dir, "depth", f"depth_{save_name}.npy"), depth_np)
            # [修改] 仅在有 ID Map 时保存
            if id_map_np is not None:
                np.save(os.path.join(tile_dir, "id_map", f"id_map_{save_name}.npy"), id_map_np)
    
    print(f"[Worker {worker_id}] Finished {grid_id}")

def worker_entry(queue, metadata, global_center, input_folder, output_root, args, worker_id, gpu_ids):
    gpu_id = gpu_ids[worker_id % len(gpu_ids)]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    while not queue.empty():
        try:
            block_key = queue.get_nowait()
            process_grid_block(block_key, metadata, global_center, input_folder, output_root, args, worker_id, device)
        except Exception: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", "-i", type=str, required=True)
    parser.add_argument("--output_root", "-o", type=str, default="output_render_ortho_final")
    parser.add_argument("--num_workers", type=int, default=6) 
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    
    # [新增] 开启 ID Buffer 生成的开关
    parser.add_argument("--use_id_buffer", action='store_true', help="Enable ID Map generation (Slows down loading)")
    
    args = parser.parse_args()
    
    meta_path = os.path.join(args.input_folder, "metadata.json")
    if not os.path.exists(meta_path): return
    
    with open(meta_path, 'r') as f: metadata = json.load(f)
    keys = list(metadata.keys())
    
    print(f"--- Capture GPU (Vertical Light) ---\n")
    print(f"ID Buffer Generation: {'ENABLED' if args.use_id_buffer else 'DISABLED'}")
    print(f"Lighting: Sat=1.3, Cont=1.1, Bright=1.05")
    
    task_queue = mp.Queue()
    for k in keys: task_queue.put(k)
    
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    procs = [mp.Process(target=worker_entry, args=(task_queue, metadata, None, args.input_folder, args.output_root, args, i, gpu_ids)) for i in range(args.num_workers)]
    
    for p in procs: p.start()
    for p in procs: p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
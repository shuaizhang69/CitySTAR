import os
import numpy as np
import json
import time
from plyfile import PlyData, PlyElement
from tqdm import tqdm

def voxel_downsample_fast(x, y, z, voxel_size):
    """
    使用 NumPy 哈希加速实现体素下采样。
    """
    import time
    start_time = time.time()
    print(f"  - Executing Fast Voxel Downsampling (size={voxel_size}m)...")
    
    # 1. 计算体素索引
    # 避免 np.stack 产生大内存占用，直接分步计算
    print(f"    Computing min values...")
    min_x, min_y, min_z = x.min(), y.min(), z.min()
    print(f"    Min bounds: x={min_x:.2f}, y={min_y:.2f}, z={min_z:.2f}")
    
    print(f"    Computing voxel indices for {len(x):,} points...")
    vx = ((x - min_x) / voxel_size).astype(np.int32)
    vy = ((y - min_y) / voxel_size).astype(np.int32)
    vz = ((z - min_z) / voxel_size).astype(np.int32)
    print(f"    Voxel index range: x=[{vx.min()}, {vx.max()}], y=[{vy.min()}, {vy.max()}], z=[{vz.min()}, {vz.max()}]")
    
    # 2. 将 3D 索引转换为 1D 哈希值。
    print("    Computing voxel hashes...")
    # 使用 int64 线性哈希，优化计算顺序
    # 使用更大的乘数避免哈希冲突
    hash_mult_start = time.time()
    hash_idx = vx.astype(np.int64)
    hash_idx = hash_idx * 100000000000 + vy.astype(np.int64) * 1000000 + vz.astype(np.int64)
    hash_mult_time = time.time() - hash_mult_start
    print(f"    Hash multiplication took {hash_mult_time:.2f} seconds")
    
    # 释放中间变量
    del vx, vy, vz
    
    # 3. 寻找唯一体素
    print(f"    Finding unique voxels among {len(hash_idx):,} points...")
    _, unique_indices = np.unique(hash_idx, return_index=True)
    print(f"    Found {len(unique_indices):,} unique voxels")
    
    # 4. 排序以保持原始顺序
    print(f"    Sorting indices...")
    unique_indices.sort()
    
    elapsed = time.time() - start_time
    print(f"    Downsampling completed in {elapsed:.2f} seconds")
    
    return unique_indices

def split_ply_grid(input_folder, output_folder, grid_size=50.0, voxel_size=0.1):
    """
    [增强版] 
    1. 先进行体素下采样 (Voxel Downsampling)
    2. 显示保留点云百分比
    3. 按网格无缝切分 (保留稀疏块)
    """
    import time
    total_start_time = time.time()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    metadata = {}
    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]
    
    if not ply_files:
        print(f"Warning: No .ply files found in {input_folder}")
        return
    
    print(f"Found {len(ply_files)} PLY files in {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Grid size: {grid_size}m, Voxel size: {voxel_size}m")

    for filename in ply_files:
        src_path = os.path.join(input_folder, filename)
        file_size = os.path.getsize(src_path) / (1024*1024)  # MB
        print(f"\nProcessing {filename} ({file_size:.1f} MB)...")
        file_start_time = time.time()
        
        # 1. 读取数据
        try:
            print(f"  - Reading {filename} (this may take a while for large files)...")
            plydata = PlyData.read(src_path)
            vertex = plydata['vertex']
            print(f"  - Loaded {len(vertex)} points")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        
        # 获取属性列表
        property_names = [p.name for p in vertex.properties]
        
        # 2. 加载数据 (Float64)
        print("  - Loading data into Float64 memory...")
        import time
        start_load = time.time()
        x = vertex['x'].astype(np.float64)
        y = vertex['y'].astype(np.float64)
        z = vertex['z'].astype(np.float64)
        load_time = time.time() - start_load
        print(f"  - XYZ loaded in {load_time:.2f} seconds")
        
        # 动态加载其他属性
        has_color = 'red' in property_names
        r, g, b = (None, None, None)
        if has_color:
            print("  - Loading color data...")
            r = vertex['red']
            g = vertex['green']
            b = vertex['blue']
            print(f"  - RGB loaded")
            
        class_field_name = None
        if 'class' in property_names: class_field_name = 'class'
        elif 'label' in property_names: class_field_name = 'label'
        
        has_class = class_field_name is not None
        raw_labels, label_dtype = (None, None)
        if has_class:
            print(f"  - Loading class data ('{class_field_name}')...")
            raw_labels = vertex[class_field_name]
            label_dtype = raw_labels.dtype
            print(f"  - Class labels loaded")
            
        # 释放原始 plydata 以节省内存
        del plydata
        print("  - Raw plydata released.")
            
        # --- [新增功能] 下采样逻辑 ---
        original_count = len(x)
        if voxel_size is not None and voxel_size > 0:
            # 使用哈希加速的下采样
            keep_indices = voxel_downsample_fast(x, y, z, voxel_size)
            
            # 应用筛选
            x = x[keep_indices]
            y = y[keep_indices]
            z = z[keep_indices]
            
            if has_color:
                r, g, b = r[keep_indices], g[keep_indices], b[keep_indices]
            if has_class:
                raw_labels = raw_labels[keep_indices]
                
            final_count = len(x)
            retention_rate = (final_count / original_count) * 100.0
            
            print(f"  - [Downsample Stats]")
            print(f"    Original : {original_count}")
            print(f"    Kept     : {final_count}")
            print(f"    Retention: {retention_rate:.2f}% (Removed {original_count - final_count} points)")
        else:
            print(f"  - Skipping downsample (voxel_size={voxel_size})")

        # 3. 计算整数网格边界
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        
        min_gx_idx = int(np.floor(min_x / grid_size))
        max_gx_idx = int(np.ceil(max_x / grid_size))
        min_gy_idx = int(np.floor(min_y / grid_size))
        max_gy_idx = int(np.ceil(max_y / grid_size))
        
        total_grids = (max_gx_idx - min_gx_idx) * (max_gy_idx - min_gy_idx)
        pbar = tqdm(total=total_grids, desc="  - Gridding")
        
        for gx_idx in range(min_gx_idx, max_gx_idx):
            for gy_idx in range(min_gy_idx, max_gy_idx):
                
                # 计算当前网格理论坐标
                gx = float(gx_idx) * grid_size
                gy = float(gy_idx) * grid_size
                
                # 4. 筛选点
                mask = (x >= gx) & (x < gx + grid_size) & \
                       (y >= gy) & (y < gy + grid_size)
                
                if not np.any(mask):
                    pbar.update(1)
                    continue
                
                pts_count = np.count_nonzero(mask)
                
                # [核心修复] 只要有点就保存，防止空洞
                if pts_count == 0: 
                    pbar.update(1)
                    continue
                
                # 5. 坐标归一化
                offset = np.array([gx, gy, 0.0], dtype=np.float64)
                
                pts_x_local = x[mask] - offset[0]
                pts_y_local = y[mask] - offset[1]
                pts_z_local = z[mask] - offset[2]
                
                # 6. 构建保存数据
                dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
                if has_color: dtype_list += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
                if has_class: dtype_list.append((class_field_name, label_dtype))
                
                vertex_data = np.empty(pts_count, dtype=dtype_list)
                
                vertex_data['x'] = pts_x_local.astype('f4')
                vertex_data['y'] = pts_y_local.astype('f4')
                vertex_data['z'] = pts_z_local.astype('f4')
                
                if has_color:
                    vertex_data['red'] = r[mask]
                    vertex_data['green'] = g[mask]
                    vertex_data['blue'] = b[mask]
                    
                if has_class:
                    vertex_data[class_field_name] = raw_labels[mask]
                
                # 7. 保存
                block_name = f"{os.path.splitext(filename)[0]}_x{int(gx)}_y{int(gy)}"
                save_path = os.path.join(output_folder, f"{block_name}.ply")
                
                el = PlyElement.describe(vertex_data, 'vertex')
                PlyData([el], text=False).write(save_path)
                
                metadata[f"{block_name}.ply"] = {
                    "offset": offset.tolist(),
                    "grid_size": grid_size,
                    "original_scene": filename,
                    "has_class": has_class
                }
                
                pbar.update(1)
        
        pbar.close()
        file_end_time = time.time()
        file_time = file_end_time - file_start_time
        print(f"  - File {filename} processed in {file_time:.2f} seconds")
        print(f"  - Generated {len([k for k in metadata.keys() if k.startswith(os.path.splitext(filename)[0])])} grid blocks")

    # 8. 保存 Metadata
    metadata_path = os.path.join(output_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\n{'='*60}")
    print(f"Process completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Total grid blocks generated: {len(metadata)}")
    print(f"Output folder: {output_folder}")
    print(f"Metadata saved: {metadata_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # --- 配置参数 ---
    split_ply_grid(
        input_folder="/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/hkust",
        output_folder="./hkust/grid_50m", 
        grid_size=50.0,
        voxel_size=0.1  # [设置] 下采样间距，单位为米。设为 0 或 None 则不采样
    )   
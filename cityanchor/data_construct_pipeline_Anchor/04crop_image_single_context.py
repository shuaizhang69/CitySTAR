import os
import json
import numpy as np
import rasterio
import torch
import cv2  # 用于绘图和缩放
import sys
import re


def sanitize_filename(name):
    """清理文件名中的非法字符"""
    if not name:
        return "unknown"
    name = name.replace(' ', '_')
    # 替换所有文件系统非法字符
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # 压缩连续的下划线
    name = re.sub(r'_+', '_', name)
    # 去除首尾的下划线或点
    name = name.strip('_.')
    return name
def process_and_crop(tif_path, pth_path, json_path, output_dir, num_instances=20):
    if not os.path.exists(pth_path):
        print(f"❌ PTH 文件不存在: {pth_path}")
        return
    
    # 1. 加载 PTH
    try:
        coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes_dict, \
            landmark_names, landmark_ids, globalShift = torch.load(pth_path)
        
        valid_ids = sorted([k for k in instance_bboxes_dict.keys() if k != -100])
        bboxes_list = np.stack([instance_bboxes_dict[k] for k in valid_ids])
        id_map = {old_id: new_idx for new_idx, old_id in enumerate(valid_ids)}
        print(f"✅ PTH 加载成功，实例数：{len(bboxes_list)}")
    except Exception as e:
        print(f"❌ PTH 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if not os.path.exists(json_path):
        print(f"❌ JSON 文件不存在")
        return
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    scene_id = data.get("scene_id", "unknown")
    json_objects = data.get("bboxes", [])
    
    if not os.path.exists(tif_path):
        print(f"❌ TIF 文件不存在")
        return

    # 读取 TIF
    with rasterio.open(tif_path) as src:
        image = src.read()
        c, img_height, img_width = image.shape
        
        print(f"🚀 处理场景：{scene_id}")
        print(f"   影像尺寸：Height={img_height}, Width={img_width}")
        print(f"   📌 策略：当前物体红框 + 邻域窗口(2x bbox + 50m), 输出固定 1024x1024")

        scale_factor = 0.1
        OUTPUT_SIZE = 1024
        CONTEXT_SCALE = 2.0
        CONTEXT_MARGIN_METERS = 50.0

        os.makedirs(output_dir, exist_ok=True)
        count = 0
        processed_ids = set()

        for i, obj in enumerate(json_objects):
            if count >= num_instances: break
            
            obj_id = obj.get("object_id") or obj.get("id")
            landmark = obj.get("landmark", "")
            safe_landmark = sanitize_filename(landmark)
        
            # 限制文件名长度 (可选，防止某些系统文件名过长报错)
            if len(safe_landmark) > 50:
                safe_landmark = safe_landmark[:50]
                
            
            
            # if obj_id is None or obj_id not in id_map or obj_id in processed_ids:
            #     print("pass")
            #     continue
            if obj_id is None:
                # print(f"[PASS] 原因: obj_id 为 None | scene: {scene_id},")
                # continue
                obj_id=0
            
            if obj_id not in id_map:
                print(f"[PASS] 原因: obj_id '{obj_id}' 不在 id_map 中 (id_map 长度: {len(id_map)}) | scene: {scene_id}")
                continue
            
            if obj_id in processed_ids:
                print(f"[PASS] 原因: obj_id '{obj_id}' 已经被处理过 (processed_ids: {processed_ids}) | scene: {scene_id}")
                continue
            processed_ids.add(obj_id)

            idx = id_map[obj_id]
            box = bboxes_list[idx]
            
            # 解析原始数据 (米)
            x_raw, y_raw = box[0], box[1]      # 中心点
            w_raw, h_raw = box[3], box[4]      # 宽高
            
            # 转换为像素坐标
            x_px = x_raw / scale_factor
            y_px = y_raw / scale_factor
            w_px = w_raw / scale_factor
            h_px = h_raw / scale_factor
            
            # --- 核心逻辑：强制 Fit 1 ---
            # Col = X
            # Row = Height - Y (GIS 转图像坐标系)
            center_col = int(x_px)
            center_row = int(img_height - y_px)
            
            # 边界检查 (中心点必须在图内)
            if not (0 <= center_row < img_height and 0 <= center_col < img_width):
                print(f"   ❌ 实例 {obj_id} 中心点越界 ({center_row}, {center_col}), 跳过。")
                continue

            # --- 计算裁剪区域 (Crop Region) [修改为长方形裁剪] ---
            
            # 1. 获取物体的原始宽高
            obj_w = w_px
            obj_h = h_px
            
            # 防止除以零或过小
            if obj_w < 1: obj_w = 1
            if obj_h < 1: obj_h = 1
            
            # 邻域窗口: 宽高 = 2 * bbox + 50m
            context_margin_px = CONTEXT_MARGIN_METERS / scale_factor
            crop_h = int(obj_h * CONTEXT_SCALE + context_margin_px)
            crop_w = int(obj_w * CONTEXT_SCALE + context_margin_px)
            
            # 设置最小尺寸限制 (防止物体太小时裁出来看不清)
            # min_dim = 64
            # if crop_h < min_dim: crop_h = min_dim
            # if crop_w < min_dim: crop_w = min_dim
            
            half_h = crop_h // 2
            half_w = crop_w // 2
            
            # 4. 计算边界 (分别用 half_h 和 half_w)
            r_start = max(center_row - half_h, 0)
            r_end = min(center_row + half_h, img_height)
            c_start = max(center_col - half_w, 0)
            c_end = min(center_col + half_w, img_width)
            
            # 如果裁剪区域为空，跳过
            if r_end <= r_start or c_end <= c_start:
                print("!!!empty")
                continue

            # 执行裁剪 (此时 ROI 是长方形的！)
            roi = image[:, r_start:r_end, c_start:c_end]
            
            # --- 记录物体框在 ROI 中的相对坐标 (逻辑不变，但 ROI 尺寸变了) ---
            rel_center_row = center_row - r_start
            rel_center_col = center_col - c_start
            rel_w = w_px
            rel_h = h_px
            
            box_roi_x1 = int(rel_center_col - rel_w / 2)
            box_roi_y1 = int(rel_center_row - rel_h / 2)
            box_roi_x2 = int(rel_center_col + rel_w / 2)
            box_roi_y2 = int(rel_center_row + rel_h / 2)

            # --- 缩放至 OUTPUT_SIZE (注意这里的变化) ---
            h_roi, w_roi = roi.shape[1], roi.shape[2]
            
            if h_roi == 0 or w_roi == 0:
                continue

            # 转为 HWC
            roi_hwc = np.transpose(roi, (1, 2, 0))
            if roi_hwc.dtype != np.uint8:
                min_v, max_v = roi_hwc.min(), roi_hwc.max()
                if max_v > min_v:
                    roi_hwc = ((roi_hwc - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    roi_hwc = np.zeros_like(roi_hwc, dtype=np.uint8)
            
            if roi_hwc.shape[2] == 1:
                roi_hwc = cv2.cvtColor(roi_hwc, cv2.COLOR_GRAY2RGB)
            elif roi_hwc.shape[2] > 3:
                roi_hwc = roi_hwc[:, :, :3]
            
            # 【重要决策点】
            # 选项 A: 保持长方形输出 (文件名可能冲突，且后续模型需支持变长输入)
            # 选项 B: 依然缩放到正方形 (512x512)，但内容不再是“居中+黑边”，而是“拉伸填充”或“Letterbox”
            
            # 如果你的目的是“只要框里的内容”，通常意味着你想去掉背景。
            # 如果你直接 resize 长方形 ROI 到 512x512，物体会被“压扁”或“拉长” (变形)。
            
            # 👇 推荐做法：保持纵横比缩放 (Letterbox)，即缩放到 512x512 范围内，不足的部分补黑边/白边
            # 这样既保留了长方形物体的比例，又满足了模型输入 512x512 的要求
            
            target_size = OUTPUT_SIZE # 512
            
            # 计算缩放比例，确保能放进 512x512
            scale = min(target_size / w_roi, target_size / h_roi)
            new_w = int(w_roi * scale)
            new_h = int(h_roi * scale)
            
            # 执行 Resize (保持比例)
            resized_content = cv2.resize(roi_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建 512x512 的画布 (黑色背景)
            resized_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            
            # 将内容粘贴到画布中心
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            resized_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_content
            
            # 注意：此时 draw_x1... 等坐标需要重新计算 (相对于新的 512x512 画布)
            # 原来的 box 坐标也要跟着缩放和平移
            scale_x_final = new_w / w_roi
            scale_y_final = new_h / h_roi
            
            draw_x1 = int(box_roi_x1 * scale_x_final) + x_offset
            draw_y1 = int(box_roi_y1 * scale_y_final) + y_offset
            draw_x2 = int(box_roi_x2 * scale_x_final) + x_offset
            draw_y2 = int(box_roi_y2 * scale_y_final) + y_offset
            
            # 再次确保框在输出范围内
            draw_x1 = max(0, min(draw_x1, OUTPUT_SIZE))
            draw_y1 = max(0, min(draw_y1, OUTPUT_SIZE))
            draw_x2 = max(0, min(draw_x2, OUTPUT_SIZE))
            draw_y2 = max(0, min(draw_y2, OUTPUT_SIZE))

            # 当前物体加红框，邻域内容作为上下文保留。
            cv2.rectangle(
                resized_img,
                (draw_x1, draw_y1),
                (draw_x2, draw_y2),
                (255, 0, 0),
                thickness=max(2, OUTPUT_SIZE // 256),
            )

            # --- 保存 ---
            save_name = f"{scene_id}_obj{obj_id}.jpg"
            save_path = os.path.join(output_dir, save_name)
            
            # 直接使用原始的 resized_img (假设它是 RGB 格式) 进行保存
            # 因为移除了画图操作，不需要再转换颜色空间
            
            with rasterio.open(
                save_path, 
                'w', 
                driver='JPEG', 
                height=OUTPUT_SIZE, 
                width=OUTPUT_SIZE, 
                count=3, 
                dtype='uint8',
                photometric='rgb',
                QUALITY=95
            ) as dst:
                # 写入 (C, H, W)
                # 确保 resized_img 是 RGB 格式且维度为 (H, W, C)
                dst.write(np.transpose(resized_img, (2, 0, 1)))
            
            print(
                f"   ✅ 保存：{save_name} | 原中心：({center_row}, {center_col}) | "
                f"邻域窗口：{crop_h}x{crop_w}px (2xbbox+50m) -> {OUTPUT_SIZE}"
            )
            count += 1

if __name__ == "__main__":
    # 基础数据目录
    base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/a10c6e205104486cb806ea94956039e4/SensatUrban/feature/random-50_crop-250"
    base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/map"
    # CityRefer JSON 标注文件所在的目录 (假设所有 bbox json 都在这个目录下)
    # 根据你的路径推断：/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d/
    json_base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d/"
    json_base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"
    # 最终输出根目录 (每个场景会在此目录下创建一个子文件夹)
    output_root = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/single_image/"
    output_root = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image"
    os.makedirs(output_root, exist_ok=True)

    # 获取目录下所有 .tif 文件
    tif_files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]
    
    if not tif_files:
        print(f"❌ 在 {base_dir} 中未找到任何 .tif 文件")
        sys.exit(1)
        
    print(f"🚀 发现 {len(tif_files)} 个 TIF 文件，开始批量处理...")

    processed_count = 0
    skipped_count = 0

    for tif_filename in sorted(tif_files):
        # 提取场景 ID (去掉 .tif 后缀)
        # 例如: "birmingham_block_1.tif" -> "birmingham_block_1"
        scene_id = os.path.splitext(tif_filename)[0]
        # 构建完整路径
        tif_path = os.path.join(base_dir, tif_filename)
        pth_path = os.path.join(base_dir, f"{scene_id}.pth")
        
        # 构建 JSON 路径 (假设命名规则为 {scene_id}_bbox.json)
        # 如果不确定后缀，可以尝试查找包含 scene_id 的 json 文件
        json_filename = f"{scene_id}_bbox.json"
        json_path = os.path.join(json_base_dir, json_filename)
        
        # 如果默认命名找不到，尝试在 json_base_dir 中模糊匹配 (容错处理)
        if not os.path.exists(json_path):
            # 尝试查找以 scene_id 开头的 json 文件
            potential_jsons = [f for f in os.listdir(json_base_dir) if f.startswith(scene_id) and f.endswith('.json')]
            if potential_jsons:
                json_filename = potential_jsons[0]
                json_path = os.path.join(json_base_dir, json_filename)
                print(f"   💡 自动修正 JSON 文件名: {json_filename}")
            else:
                print(f"⚠️ 跳过 {scene_id}: 找不到对应的 PTH 或 JSON 文件")
                print(f"   - PTH 检查: {os.path.exists(pth_path)} ({pth_path})")
                print(f"   - JSON 检查: {os.path.exists(json_path)} ({json_path})")
                skipped_count += 1
                continue

        # 定义该场景的输出文件夹
        scene_output_folder = os.path.join(output_root, scene_id)
        
        print(f"\n{'='*20} 处理场景: {scene_id} {'='*20}")
        
        # 执行处理
        # 注意：process_and_crop 内部已经做了 output_folder 的 makedirs
        process_and_crop(tif_path, pth_path, json_path, scene_output_folder, num_instances=999999)
        
        processed_count += 1

    print(f"\n🎉 批量处理完成!")
    print(f"   ✅ 成功处理: {processed_count} 个场景")
    print(f"   ⏭️  跳过: {skipped_count} 个场景")
    print(f"   📂 结果保存至: {output_root}")

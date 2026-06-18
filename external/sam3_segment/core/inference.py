import torch
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import sys
import cv2

# ================= 1. 环境与导入 (保持不变) =================

try:
    from ..sam3.model_builder import build_sam3_image_model
    from ..sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("❌ Error: Could not import sam3.")
    sys.exit(1)

try:
    from ..configs import prompts as P
except ImportError:
    print("❌ Error: Could not import prompts.py.")
    sys.exit(1)

# ================= 2. 可视化配置 (保持不变) =================
# ... (此处省略 CLASS_NAME_TO_ID 和 CLASS_COLORS 的定义，与原文件保持一致即可) ...
CLASS_NAME_TO_ID = {
    "Ground": 0, "Vegetation": 1, "Building": 2, "Wall": 3,
    "Bridge": 4, "Parking": 5, "Rail": 6, "Traffic Road": 7,
    "Street Furniture": 8, "Car": 9, "Footpath": 10, "Bike": 11, "Water": 12
}
ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

CLASS_COLORS = {
    0:  (128, 128, 128), 1:  (34, 139, 34), 2:  (0, 165, 255), 3:  (105, 105, 105),
    4:  (255, 255, 0), 5:  (203, 192, 255), 6:  (0, 0, 255), 7:  (50, 50, 50),
    8:  (255, 0, 255), 9:  (255, 0, 0), 10: (0, 255, 255), 11: (0, 255, 127), 12: (200, 0, 0)
}

# ================= 新增：压缩核心函数 =================

def encode_masks(masks_tensor):
    """
    将 mask tensor (N, H, W) 压缩为 PNG 格式的字节流列表。
    压缩率极高且无损。
    """
    compressed_list = []
    
    # 转为 numpy
    if isinstance(masks_tensor, torch.Tensor):
        masks_np = masks_tensor.cpu().numpy()
    else:
        masks_np = masks_tensor
        
    # 确保格式为 uint8 (0, 255)，这是 cv2 编码需要的格式
    if masks_np.dtype == bool:
        masks_np = masks_np.astype(np.uint8) * 255
    elif masks_np.dtype == np.uint8:
        # 确保二值化，防止原本只有0/1导致图像全黑看不清，统一转为0/255
        # 如果已经是0/255则保持，如果是0/1则乘255
        if masks_np.max() <= 1:
            masks_np = masks_np * 255

    # 遍历每一层 mask 进行单独压缩
    for i in range(masks_np.shape[0]):
        # cv2.imencode 返回 (success, buffer)
        # 使用 PNG 最高压缩等级 9
        success, buffer = cv2.imencode('.png', masks_np[i], [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if success:
            compressed_list.append(buffer)
        else:
            print("⚠️ Mask compression failed, skipping one mask.")
            # 极少情况会失败，如果失败可以选择存原图或跳过，这里防止崩溃选择存None占位
            compressed_list.append(None) 
            
    return compressed_list

# ================= 3. 核心策略逻辑 (BEV) (保持不变) =================

def resolve_overlaps_user_strategy(masks, labels, scores, subclasses, merge_iou_thresh=0.7):
    """
    [修改] 实例分割版本：保留所有独立实例，不做同类合并
    只解决不同类别之间的重叠竞争
    """
    if len(masks) == 0:
        return masks, labels, scores, subclasses

    device = masks.device
    
    # --- [删除] 阶段一：同类合并 (Intra-class Union) ---
    # 实例分割模式下，每个 mask 就是一个独立实例，不再合并同类
    
    # 直接保留原始 masks（按置信度排序以便后续处理）
    sort_idx = torch.argsort(scores, descending=True)
    g_masks = masks[sort_idx]
    g_labels = labels[sort_idx]
    g_scores = scores[sort_idx]
    g_subs = [subclasses[i] for i in sort_idx.cpu().numpy()]

    # --- 阶段二：异类竞争 (Inter-class Logic) ---
    sort_idx = torch.argsort(g_scores, descending=True)
    sorted_masks = g_masks[sort_idx]
    sorted_labels = g_labels[sort_idx]
    sorted_scores = g_scores[sort_idx]
    sorted_subs = [g_subs[i] for i in sort_idx.cpu().numpy()]

    final_masks = []
    final_labels = []
    final_scores = []
    final_subs = []

    for i in range(len(sorted_masks)):
        curr_mask = sorted_masks[i]
        curr_label = sorted_labels[i]
        curr_score = sorted_scores[i]
        curr_sub = sorted_subs[i]

        if curr_mask.sum() < 10: continue

        is_fully_eaten = False
        
        for k in range(len(final_masks)):
            prev_mask = final_masks[k]
            inter = (curr_mask & prev_mask)
            inter_area = inter.sum().item()
            if inter_area == 0: continue

            union = (curr_mask | prev_mask).sum().item()
            iou = inter_area / (union + 1e-6)

            if iou > merge_iou_thresh:
                final_masks[k] = final_masks[k] | curr_mask
                is_fully_eaten = True
                break
            else:
                curr_mask = curr_mask & (~prev_mask)
                if curr_mask.sum() < 10:
                    is_fully_eaten = True
                    break
        
        if not is_fully_eaten:
            final_masks.append(curr_mask)
            final_labels.append(curr_label)
            final_scores.append(curr_score)
            final_subs.append(curr_sub)

    return torch.stack(final_masks), \
           torch.stack(final_labels), \
           torch.stack(final_scores), \
           final_subs

# ================= 4. 可视化功能 (保持不变) =================
# ... (save_visualization 函数保持不变，直接复制原代码即可) ...
def save_visualization(image_path, masks, labels, scores, subclasses, save_dir):
    try:
        img = cv2.imread(image_path)
        if img is None: return
        vis = img.copy()
        overlay = img.copy()
        
        for i in range(len(masks)):
            mask = masks[i]
            label_id = labels[i].item()
            
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = mask.astype(bool)
            
            if not mask_np.any(): continue
            color = CLASS_COLORS.get(label_id, (255, 255, 255))
            overlay[mask_np] = color

        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        for i in range(len(masks)):
            mask = masks[i]
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = mask.astype(bool)
            
            if not mask_np.any() or mask_np.sum() < 100: continue

            label_id = labels[i].item()
            score = scores[i].item()
            sub_name = subclasses[i]
            
            ys, xs = np.where(mask_np)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            major_name = ID_TO_CLASS_NAME.get(label_id, str(label_id))
            
            if len(sub_name) > 15: sub_name = sub_name[:12] + "..."
            text = f"{major_name}: {sub_name} ({score:.2f})"
            
            font_scale = 0.5
            cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        base_name = os.path.basename(image_path)
        cv2.imwrite(os.path.join(save_dir, f"vis_{base_name}"), vis)
    except Exception as e:
        print(f"⚠️ Vis Error: {e}")

# ================= 5. 主程序 Worker =================

class SAM3InferenceWorker:
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 [Worker Start] GPU: {args.gpu} | Vis: {args.vis}")
        
        self.model = build_sam3_image_model()
        self.model.to(self.device)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.35)

    def resolve_prompt_mode(self, img_name, current_mode):
        if current_mode == 'bev':
            if "Global" in img_name: return 'global'
            elif "Detail" in img_name: return 'closeup'
            return 'all'
        return 'all'

    def process_image(self, img_path, current_mode):
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except: return None, None, None, None

        prompt_mode = self.resolve_prompt_mode(os.path.basename(img_path), current_mode)
        flat_prompts, text_to_class_map = P.get_prompts_strict(prompt_mode)
        if not flat_prompts: return None, None, None, None

        inference_state = self.processor.set_image(image_pil)
        
        raw_masks = []
        raw_labels = []
        raw_scores = []
        raw_subclasses = []

        # 1. 原始推理
        for p_text in flat_prompts:
            output = self.processor.set_text_prompt(state=inference_state, prompt=p_text)
            masks = output["masks"]
            scores = output["scores"]

            if isinstance(masks, torch.Tensor):
                if masks.ndim == 4: masks = masks.squeeze(1)
                if masks.ndim == 2: masks = masks.unsqueeze(0)
            
            if isinstance(scores, torch.Tensor):
                if scores.ndim == 2: scores = scores.squeeze(1)
                if scores.ndim == 0: scores = scores.unsqueeze(0)

            keep = scores > self.args.score_thresh
            if not keep.any(): continue

            valid_masks = masks[keep]
            valid_scores = scores[keep]
            
            class_name = text_to_class_map.get(p_text)
            if class_name not in CLASS_NAME_TO_ID: continue
            class_id = CLASS_NAME_TO_ID[class_name]

            for k in range(len(valid_masks)):
                raw_masks.append(valid_masks[k] > 0) 
                raw_labels.append(class_id)
                raw_scores.append(valid_scores[k].item())
                raw_subclasses.append(p_text)

        if not raw_masks: return None, None, None, None

        masks_tensor = torch.stack(raw_masks).to(self.device)
        labels_tensor = torch.tensor(raw_labels, dtype=torch.long, device=self.device)
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)

        # 2. 分支逻辑
        if current_mode == 'bev':
            final_masks, final_labels, final_scores, final_subs = resolve_overlaps_user_strategy(
                masks_tensor, labels_tensor, scores_tensor, raw_subclasses, merge_iou_thresh=0.7
            )
            # [修改]：不再转为 uint8，保持 bool 或原格式，交给 run 中的压缩函数处理
            return final_masks.cpu(), final_labels.cpu(), final_scores.cpu(), final_subs
        else:
            return masks_tensor.cpu(), labels_tensor.cpu(), scores_tensor.cpu(), raw_subclasses

    def run(self):
        all_scenes = sorted(glob.glob(os.path.join(self.args.render_root, "*")))
        my_scenes = [s for i, s in enumerate(all_scenes) if i % self.args.total_parts == self.args.part]
        
        if self.args.mode == 'all':
            modes_to_run = ['bev', 'tiles']
        else:
            modes_to_run = [self.args.mode]

        print(f"🔄 Processing modes: {modes_to_run}")
        
        for scene_dir in tqdm(my_scenes, desc=f"GPU {self.args.gpu}"):
            scene_name = os.path.basename(scene_dir)
            grid_dirs = sorted(glob.glob(os.path.join(scene_dir, "x*_y*")))
            
            for grid_dir in grid_dirs:
                grid_id = os.path.basename(grid_dir)
                
                for current_mode in modes_to_run:
                    
                    if current_mode == 'bev':
                        target_folder = os.path.join(grid_dir, 'dev')
                        out_filename = f"{grid_id}_strict_bev.pt"
                        vis_subfolder = "vis_bev"
                    else:
                        target_folder = os.path.join(grid_dir, 'image')
                        out_filename = f"{grid_id}_strict_tiles.pt"
                        vis_subfolder = "vis_tiles"
                    
                    if not os.path.exists(target_folder): continue
                    
                    out_grid_dir = os.path.join(self.args.output_root, scene_name, grid_id)
                    os.makedirs(out_grid_dir, exist_ok=True)
                    out_pt_path = os.path.join(out_grid_dir, out_filename)
                    
                    if os.path.exists(out_pt_path) and not self.args.overwrite: continue

                    if self.args.vis:
                        vis_dir = os.path.join(out_grid_dir, vis_subfolder)
                        os.makedirs(vis_dir, exist_ok=True)
                    else:
                        vis_dir = None

                    img_paths = sorted(glob.glob(os.path.join(target_folder, "*_real.png")))
                    if not img_paths: continue

                    data_pack = {'img_names': [], 'masks': [], 'labels': [], 'scores': []}
                    has_data = False

                    for img_path in img_paths:
                        masks, labels, scores, subclasses = self.process_image(img_path, current_mode)
                        
                        if masks is not None and len(masks) > 0:
                            img_name = os.path.basename(img_path)
                            
                            # [可视化] 仍使用原始 masks 进行绘制
                            if self.args.vis and vis_dir:
                                save_visualization(img_path, masks, labels, scores, subclasses, vis_dir)

                            # [关键修改]：保存前进行压缩
                            compressed_masks = encode_masks(masks) # 压缩为 list of bytes
                            
                            data_pack['img_names'].append(img_name)
                            data_pack['masks'].append(compressed_masks) # 存入的是压缩后的列表
                            data_pack['labels'].append(labels)
                            data_pack['scores'].append(scores)
                            has_data = True

                    if has_data:
                        # 此时保存的 data_pack 极小
                        torch.save(data_pack, out_pt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['bev', 'tiles', 'all'], required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--total_parts', type=int, default=1)
    parser.add_argument('--score_thresh', type=float, default=0.20)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    worker = SAM3InferenceWorker(args)
    worker.run()
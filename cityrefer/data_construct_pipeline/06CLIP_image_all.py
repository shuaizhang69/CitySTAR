import sys
sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")

import os
# ⚠️ 关键：必须在 import torch 之前设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import os
import sys
import json
import torch
import glob
from PIL import Image
from tqdm import tqdm

# 路径配置
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code"

# 输入/输出目录配置
INPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d/"
IMAGE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/single_image_new2/"
# 主路径读图失败（不存在或损坏）时，按相同文件名在此目录下重试
IMAGE_FALLBACK_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/single_image"
OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature"
# IMAGE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/Context_image_new2/"
# OUTPUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/context_feature2"

# 添加项目路径 (虽然纯图像模式不需要 tokenizer，但保留以防 open_clip 内部依赖)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 优先处理的场景（顺序越前越先跑；数字为参考物体量，仅作注释）
# cambridge_block_25:25, cambridge_block_33:50, birmingham_block_12:3, birmingham_block_3:91,
# birmingham_block_4:95, birmingham_block_9:50, cambridge_block_10:62, cambridge_block_2:27,
# cambridge_block_20:132, cambridge_block_3:42
PRIORITY_SCENE_ORDER = [
    "cambridge_block_25",
    "cambridge_block_33",
    "birmingham_block_12",
    "birmingham_block_3",
    "birmingham_block_4",
    "birmingham_block_9",
    "cambridge_block_10",
    "cambridge_block_2",
    "cambridge_block_20",
    "cambridge_block_3",
]
PRIORITY_SCENE_INDEX = {s: i for i, s in enumerate(PRIORITY_SCENE_ORDER)}

# 已验证标注/特征正确，无需再跑 CLIP，直接跳过（与优先队列无关，一律不处理）
SKIP_VERIFIED_COMPLETE_FILENAMES = {
    "birmingham_block_9_bbox.json",
    "birmingham_block_4_bbox.json",
    "birmingham_block_12_bbox.json",
    "birmingham_block_5_bbox.json",
    "cambridge_block_10_bbox.json",
    "cambridge_block_2_bbox.json",
    "cambridge_block_3_bbox.json",
    "cambridge_block_21_bbox.json",
}

# 历史上跳过的问题场景；若出现在 PRIORITY_SCENE_ORDER 中则仍参与处理（优先排队）
LEGACY_SKIP_FILENAMES = set()

# ===========================================

def get_scene_id_from_json_filename(filename):
    """从文件名提取 scene_id，例如 birmingham_block_1_bbox.json -> birmingham_block_1"""
    if filename.endswith("_bbox.json"):
        return filename[:-10] # 去掉 _bbox.json
    return None

def iter_candidate_image_paths(
    scene_id, obj_id, current_img_root, image_root, fallback_root
):
    """按顺序生成候选图片路径。

    single_image_new2 常用：{scene_id}_{object_id}.png
    single_image 备用目录常用：{scene_id}_obj{object_id}.jpg（场景子目录下），与主路径命名不一致时需都尝试。
    """
    stem_flat = f"{scene_id}_{obj_id}"
    stem_obj = f"{scene_id}_obj{obj_id}"
    exts = (".png", ".jpg", ".jpeg")
    seen = set()

    def emit(*parts):
        if None in parts:
            return
        p = os.path.join(*parts)
        if p not in seen:
            seen.add(p)
            yield p

    for ext in exts:
        for rel in (
            (current_img_root, stem_flat + ext),
            (image_root, stem_flat + ext) if current_img_root != image_root else None,
            (fallback_root, scene_id, stem_flat + ext),
            (fallback_root, stem_flat + ext),
            (fallback_root, scene_id, stem_obj + ext),
            (fallback_root, stem_obj + ext),
        ):
            if rel is None:
                continue
            yield from emit(*rel)

def output_is_complete(output_path, input_bbox_path=None):
    """判断 feature 输出是否已与当前输入对齐且整场景已跑完一轮，可安全跳过。

    不能仅用 processed_count >= total_objects：部分物体缺图失败时 proc 永远小于 tot，
    但 proc + failed == total 表示已遍历完，应视为完成（否则会像 cambridge_block_21 那样反复重跑）。

    以下视为未完成：输入 bbox 条数与输出不一致；features 条数与 processed_count 不一致；
    proc+failed != total；或全场均失败（可补图后重试，如 cambridge_block_33）。
    """
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, "r") as f:
            ex = json.load(f)
    except Exception:
        return False

    feats = ex.get("features") or []
    n_feat = len(feats)

    n_expected = None
    if input_bbox_path and os.path.exists(input_bbox_path):
        try:
            with open(input_bbox_path, "r") as f:
                src = json.load(f)
            n_expected = len(src.get("bboxes") or [])
        except Exception:
            n_expected = None

    tot = ex.get("total_objects")
    proc = ex.get("processed_count")
    failed = ex.get("failed_count")

    if n_expected is None:
        n_expected = tot

    if n_expected is None:
        return n_feat > 0

    if n_expected == 0:
        return True

    if tot is not None and tot != n_expected:
        return False

    if proc is None:
        proc = n_feat

    if failed is None:
        ref = tot if tot is not None else n_expected
        failed = max(0, ref - proc)

    if n_feat != proc:
        return False

    if proc + failed != n_expected:
        return False

    if proc == 0 and failed == n_expected:
        return False

    return True

def main():
    print(f"🚀 开始批量场景图像特征提取...")
    
    # --- 1. 创建输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 获取所有待处理的 JSON 文件 ---
    all_json_files = glob.glob(os.path.join(INPUT_DIR, "*_bbox.json"))
    
    if not all_json_files:
        print(f"❌ 在 {INPUT_DIR} 中未找到任何 *_bbox.json 文件")
        sys.exit(1)
    
    print(f"   📂 发现 {len(all_json_files)} 个标注文件")

    # --- 3. 过滤已处理的文件 ---
    # 完成条件见 output_is_complete（与当前输入 bbox 条数对齐 + 整轮已跑完）。
    tasks = []
    for json_path in all_json_files:
        filename = os.path.basename(json_path)
        scene_id = get_scene_id_from_json_filename(filename)
        if filename in SKIP_VERIFIED_COMPLETE_FILENAMES:
            print(f"   ⏭️  跳过 (已验证正确，无需处理): {filename}")
            continue
        if filename in LEGACY_SKIP_FILENAMES and (
            not scene_id or scene_id not in PRIORITY_SCENE_INDEX
        ):
            print(f"   ⏭️  跳过 (黑名单): {filename}")
            continue
        output_path = os.path.join(OUTPUT_DIR, filename)

        if output_is_complete(output_path, input_bbox_path=json_path):
            print(f"   ⏭️  跳过 (已完成): {filename}")
            continue
        if os.path.exists(output_path):
            print(f"   🔁  将重跑 (输出不完整或失败): {filename}")
        tasks.append((json_path, output_path))

    def task_sort_key(item):
        json_path, _ = item
        sid = get_scene_id_from_json_filename(os.path.basename(json_path))
        if sid and sid in PRIORITY_SCENE_INDEX:
            return (0, PRIORITY_SCENE_INDEX[sid])
        return (1, sid or "")

    tasks.sort(key=task_sort_key)

    if not tasks:
        print(f"✅ 所有场景均已处理完毕，无需执行任何操作。")
        return

    n_pri = sum(
        1
        for jp, _ in tasks
        if get_scene_id_from_json_filename(os.path.basename(jp)) in PRIORITY_SCENE_INDEX
    )
    print(f"   ⚡ 待处理场景: {len(tasks)} 个（其中优先队列: {n_pri} 个）")

    # --- 4. 加载模型 (全局加载一次) ---
    try:
        import open_clip
        print("   ✅ 依赖导入成功")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   🖥️  使用设备: {device}")

    print(f"   ⏳ 正在加载模型: {MODEL_PATH} ...")
    try:
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            model_name="EVA02-E-14-plus", 
            pretrained=MODEL_PATH, 
            device=device
        )
        clip_model.eval()
        print("   ✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 5. 逐个场景处理 ---
    total_success = 0
    total_fail = 0

    for json_path, output_path in tqdm(tasks, desc="Scenes"):
        scene_filename = os.path.basename(json_path)
        
        try:
            # A. 读取 JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            scene_id = data.get("scene_id", get_scene_id_from_json_filename(scene_filename))
            bboxes = data.get("bboxes", [])
            
            if not scene_id:
                print(f"   ⚠️ 警告: {scene_filename} 中缺少 scene_id 且无法从文件名推断，跳过。")
                continue

            # B. 确定该场景的图片子目录
            # 策略：优先尝试 {IMAGE_ROOT}/{scene_id}/
            # 如果不存在，尝试直接在 {IMAGE_ROOT}/ 下查找
            img_dir_candidate_1 = os.path.join(IMAGE_ROOT, scene_id)
            img_dir_candidate_2 = IMAGE_ROOT
            
            if os.path.isdir(img_dir_candidate_1):
                current_img_root = img_dir_candidate_1
            elif os.path.isdir(img_dir_candidate_2):
                # 检查里面是否有对应的图片，防止误用
                test_img = os.path.join(img_dir_candidate_2, f"{scene_id}_0.png")
                if os.path.exists(test_img) or len(bboxes) == 0:
                    current_img_root = img_dir_candidate_2
                else:
                    # 如果根目录下没有该场景图片，可能还是需要在子目录找，或者报错
                    # 这里假设如果子目录存在就用子目录，否则用根目录尝试匹配
                    current_img_root = img_dir_candidate_1 # 即使不存在，后续会报文件找不到
            else:
                current_img_root = img_dir_candidate_1

            features_list = []
            scene_success = 0
            scene_fail = 0

            # C. 遍历物体提取特征
            num_objects = len(bboxes)
            
            # 使用 tqdm 包裹内层循环，desc 显示当前场景 ID，unit 显示为 object
            for obj in tqdm(bboxes, desc=f"Processing {scene_id}", unit="obj", leave=False):
                obj_id = obj.get("object_id")
                img_filename = f"{scene_id}_{obj_id}.png"
                loaded = False
                for try_path in iter_candidate_image_paths(
                    scene_id, obj_id, current_img_root, IMAGE_ROOT, IMAGE_FALLBACK_ROOT
                ):
                    if not os.path.exists(try_path):
                        continue
                    try:
                        image_raw = Image.open(try_path).convert("RGB")
                        image_input = preprocess(image_raw).unsqueeze(0).to(device, non_blocking=True)
                        
                        with torch.no_grad():
                            image_feat = clip_model.encode_image(image_input)
                            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                            image_feat = image_feat.clone().detach()
                        
                        feat_np = image_feat.squeeze(0).cpu().numpy()
                        
                        features_list.append({
                            "object_id": obj_id,
                            "feature": feat_np.tolist()
                        })
                        scene_success += 1
                        loaded = True
                        break
                    except Exception:
                        continue
                
                if not loaded:
                    scene_fail += 1
                    print(os.path.join(current_img_root, img_filename))
                    continue

            # D. 保存结果
            output_data = {
                "scene_id": scene_id,
                "source_json": json_path,
                "model_used": "EVA02-E-14-plus",
                "total_objects": len(bboxes),
                "processed_count": scene_success,
                "failed_count": scene_fail,
                "features": features_list
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f) # 不使用 indent 以节省空间，如需调试可加 indent=2
            
            total_success += scene_success
            total_fail += scene_fail
            # print(f"   ✅ {scene_filename}: 成功 {scene_success}, 失败 {scene_fail}")

        except Exception as e:
            print(f"\n❌ 场景 {scene_filename} 处理严重错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*40)
    print(f"🎉 全部完成!")
    print(f"   处理场景数: {len(tasks)}")
    print(f"   总成功物体: {total_success}")
    print(f"   总失败物体: {total_fail}")
    print(f"   结果保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
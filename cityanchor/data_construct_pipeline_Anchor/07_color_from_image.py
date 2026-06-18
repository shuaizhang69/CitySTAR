import argparse
import sys
sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")

import os
# ⚠️ 关键：必须在 import torch 之前设置显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple

# ================= 配置区域 =================
# 1. 06CLIP_image_all.py 输出的聚合特征 JSON
CLIP_FEATURE_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_same_scene_same_category_clip_features.json"

# 2. bbox 文件目录（用于获取 object_name，但不保存）
BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/bbox"

# 3. 模型路径
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"

# 4. 项目根目录
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 5. 输出路径
OUTPUT_JSONL_PATH = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/color/all_objects_color.jsonl"

# ================= 颜色与档位定义 =================
# 定义档位映射：颜色 -> 档位ID (1-11)
# 注意：这里将用户定义的11个类别映射为整数ID
COLOR_TO_TIER = {
    # 1. 白色系
    "White": 1, "Cream": 1, "Off-White": 1,
    # 2. 银色、米色与浅灰色系
    "Silver": 2, "Beige": 2, "Tan": 2, "Light Gray": 2, "Light Grey": 2, "Gold": 2, "Light Yellow": 2,
    # 3. 中灰色系
    "Gray": 3, "Grey": 3, "Stone": 3,
    # 4. 红色系
    "Red": 4, "Maroon": 4, "Burgundy": 4, "Dark Red": 4, "Pink": 4,
    # 5. 橙色与黄色系
    "Orange": 5, "Yellow": 5, "Rust": 5,
    # 6. 绿色系
    "Green": 6, "Light Green": 6,  "Teal": 6, "Turquoise": 6,
    # 7. 蓝色系
    "Blue": 7, "Light Blue": 7, "Sky Blue": 7, "Bright Blue": 7,
    # 8. 深蓝色系
    "Dark Blue": 8, "Navy Blue": 8, "Deep Blue": 8,
    # 9. 紫色系
    "Purple": 9, "Violet": 9, "Magenta": 9,
    # 10. 棕色与土色系
    "Brown": 10, "Dark Brown": 10, "Light Brown": 10, "Dirt": 10, "Tan-Brown": 10,
    # 11. 深灰与黑色系 (合并了原11和12)
    "Dark Gray": 11, "Dark Grey": 11, "Charcoal": 11, "Dim": 11, "Black": 11,"Dark Green": 11,    # 复杂情况
    #"Multi-color": -1, "Mixed color": -1, "Unknown color": -1, "Various color": -1,
}

# 保持原有的颜色列表顺序，用于生成特征
COLORS = list(COLOR_TO_TIER.keys())

# ================= 全局变量 =================
device_global = None
clip_model_global = None
tokenizer_global = None
# object_name -> [num_colors, D]，同一类别只编码一次颜色提示词（否则每物体都跑一遍 CLIP 文本侧，极慢）
_color_features_cache: Dict[str, torch.Tensor] = {}

# ================= 核心功能函数 =================

def encode_text(text: str) -> torch.Tensor:
    """编码单个文本"""
    global device_global, clip_model_global, tokenizer_global
    
    with torch.no_grad():
        tokens = tokenizer_global([text]).to(device_global, non_blocking=True)
        if len(tokens.shape) < 2: 
            tokens = tokens[None, ...]
        
        emb = clip_model_global.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze(0)

def batch_encode_texts(texts: List[str]) -> torch.Tensor:
    """批量编码文本"""
    global device_global, clip_model_global, tokenizer_global
    
    with torch.no_grad():
        tokens = tokenizer_global(texts).to(device_global, non_blocking=True)
        emb = clip_model_global.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

def load_bbox_data(scene_id: str) -> Dict[str, str]:
    """
    从 bbox 文件加载 object_id -> object_name 的映射
    文件：{BBOX_DIR}/{scene_id}_bbox.json
    """
    bbox_path = os.path.join(BBOX_DIR, f"{scene_id}_bbox.json")
    
    if not os.path.exists(bbox_path):
        return {}
    
    try:
        with open(bbox_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 构建映射：object_id -> object_name
        name_map = {}
        for item in data.get("bboxes", []):
            obj_id = str(item.get("object_id"))
            obj_name = item.get("object_name", "object")
            if obj_id:
                name_map[obj_id] = obj_name
        
        return name_map
    except Exception as e:
        print(f"Warning: Failed to load bbox for {scene_id}: {e}")
        return {}

def get_color_features_for_object(object_name: str) -> torch.Tensor:
    """
    为特定物体生成所有颜色的特征 [num_colors, D]
    prompts: "a photo of a {color} {object_name}."
    同一 object_name 复用缓存，避免对每个物体重复跑文本编码（主要性能瓶颈）。
    """
    name = object_name.lower().strip() if object_name else "object"
    if name in _color_features_cache:
        return _color_features_cache[name]
    color_prompts = [f"a photo of a {color} {name}." for color in COLORS]
    emb = batch_encode_texts(color_prompts)
    _color_features_cache[name] = emb
    return emb

def predict_image_color(image_feature: torch.Tensor, object_name: str = "object") -> Dict:
    """
    预测单张图片的颜色（使用特定物体的颜色提示词）
    输入: 
        image_feature: [D] 已归一化的特征向量
        object_name: 物体类别名称（仅用于构建提示词，不保存）
    输出: {
        "predicted_color": "red",
        "predicted_tier": 4,  # 新增：颜色所属档位
        "confidence": 0.85,
        "all_scores": {"red": 0.85, "blue": 0.12, ...}
    }
    """
    # 获取该物体类别的颜色特征
    color_features = get_color_features_for_object(object_name)
    
    # 图像特征归一化（与颜色文本特征做点积相似度；此前误用 color_features 的范数）
    img_feat_norm = image_feature / (image_feature.norm() + 1e-8)
    
    # 计算与所有颜色的相似度 [num_colors]
    similarities = torch.matmul(color_features, img_feat_norm)
    
    # 获取最高分的颜色
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    predicted_color = COLORS[best_idx]
    
    # 【新增】获取该颜色对应的档位
    predicted_tier = COLOR_TO_TIER.get(predicted_color, -1)
    
    # 构建所有颜色的分数字典
    all_scores = {
        COLORS[i]: round(similarities[i].item(), 4) 
        for i in range(len(COLORS))
    }
    
    return {
        "predicted_color": predicted_color,
        "predicted_tier": predicted_tier,  # 新增字段
        "confidence": round(best_score, 4),
        "all_scores": all_scores
    }

def load_clip_feature_groups(
    clip_feature_json: str,
    only_scenes: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    读取 06CLIP_image_all.py 输出 JSON，并返回 groups 列表。
    支持按 scene 过滤。
    """
    if not os.path.exists(clip_feature_json):
        print(f"❌ 聚合特征文件不存在：{clip_feature_json}")
        return []

    try:
        with open(clip_feature_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取聚合特征文件失败：{e}")
        return []

    groups = data.get("groups", [])
    if not isinstance(groups, list):
        print("❌ 聚合特征 JSON 格式错误：缺少合法的 groups 列表")
        return []

    if only_scenes is not None:
        groups = [g for g in groups if str(g.get("scene_id", "")) in only_scenes]

    return groups

def load_existing_color_results(
    path: str, only_scenes: Optional[Set[str]] = None
) -> Tuple[List[Dict], Set[Tuple[str, str]]]:
    """
    读取已有 JSONL 结果，用于断点续跑：返回已有记录列表与 (scene_id, object_id) 集合。
    only_scenes 非空时只保留这些 scene 的行（减少内存；仍须顺序读完整文件）。
    """
    records: List[Dict] = []
    keys: Set[Tuple[str, str]] = set()
    if not os.path.exists(path):
        return records, keys
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    sid = str(r.get("scene_id", ""))
                    oid = str(r.get("object_id", ""))
                    if only_scenes is not None and sid not in only_scenes:
                        continue
                    records.append(r)
                    keys.add((sid, oid))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: 读取已有结果失败（将从头写入新键）: {e}")
        return [], set()
    return records, keys

def parse_args():
    p = argparse.ArgumentParser(
        description="从 06CLIP_image_all.py 的聚合特征 JSON 预测物体颜色 / tier"
    )
    p.add_argument(
        "--clip-feature-json",
        type=str,
        default=CLIP_FEATURE_JSON,
        help="06CLIP_image_all.py 的输出 JSON 路径",
    )
    p.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        metavar="SCENE_ID",
        help="只处理这些 scene_id；不传则处理聚合 JSON 中全部 scene",
    )
    p.add_argument(
        "--bbox-dir",
        type=str,
        default=BBOX_DIR,
        help="bbox 目录，用于回退读取 object_name",
    )
    p.add_argument(
        "--output-jsonl",
        type=str,
        default=OUTPUT_JSONL_PATH,
        help="颜色预测输出 jsonl 路径",
    )
    return p.parse_args()

def main():
    global device_global, clip_model_global, tokenizer_global

    args = parse_args()
    output_jsonl_path = args.output_jsonl

    print("🚀 开始批量预测图片颜色（使用物体类别构建提示词）")
    
    # 0. 初始化模型
    print("📦 正在加载 CLIP 模型...")
    device_global = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import open_clip
        from utils.tokenizer import SimpleTokenizer
        
        clip_model_global, _ = open_clip.create_model_from_pretrained(
            model_name="EVA02-E-14-plus", pretrained=MODEL_PATH, device=device_global
        )
        tokenizer_global = SimpleTokenizer()
        print("   ✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        sys.exit(1)

    only_scenes: Optional[Set[str]] = None
    if args.scenes:
        only_scenes = set(args.scenes)

    groups = load_clip_feature_groups(args.clip_feature_json, only_scenes=only_scenes)
    if len(groups) == 0:
        print("❌ 没有可处理的 group 数据")
        return
    scene_ids = sorted({str(g.get("scene_id", "")) for g in groups if g.get("scene_id") is not None})
    if args.scenes:
        missing = only_scenes - set(scene_ids) if only_scenes is not None else set()
        if missing:
            print(f"⚠️ 以下 scene 不在聚合特征 JSON 中，已忽略: {sorted(missing)}")
        print(f"📁 指定场景 {len(args.scenes)} 个，聚合 JSON 内命中 {len(scene_ids)} 个场景，{len(groups)} 个 group")
    else:
        print(f"📁 聚合 JSON 内共 {len(scene_ids)} 个场景，{len(groups)} 个 group")

    global BBOX_DIR
    BBOX_DIR = args.bbox_dir

    existing_records, processed_keys = load_existing_color_results(
        output_jsonl_path, only_scenes=only_scenes
    )
    print(f"📂 已有结果（用于续跑跳过）{len(processed_keys)} 条，已处理物体将跳过；新结果将追加写入")

    # 2. 遍历所有场景，预测颜色
    new_results: List[Dict] = []
    failed_groups = 0
    skipped_without_feature = 0

    for group in tqdm(groups, desc="Processing groups"):
        scene_id = str(group.get("scene_id", ""))
        objects = group.get("objects", [])
        if not isinstance(objects, list):
            failed_groups += 1
            continue

        # 仅做回退：当 object_name 缺失时尝试从 bbox 文件查
        name_map = load_bbox_data(scene_id)

        for obj in objects:
            obj_id = str(obj.get("object_id", ""))
            if not obj_id or not scene_id:
                continue
            if (scene_id, obj_id) in processed_keys:
                continue

            feat_val = obj.get("feature")
            if not isinstance(feat_val, list) or len(feat_val) == 0:
                skipped_without_feature += 1
                continue

            try:
                img_feat = torch.tensor(feat_val, dtype=torch.float32, device=device_global)
                img_feat = img_feat / (img_feat.norm() + 1e-8)

                object_name = (
                    obj.get("object_name")
                    or name_map.get(obj_id)
                    or "object"
                )

                color_result = predict_image_color(img_feat, object_name)
                record = {
                    "scene_id": scene_id,
                    "object_id": obj_id,
                    "predicted_color": color_result["predicted_color"],
                    "predicted_tier": color_result["predicted_tier"],
                    "confidence": color_result["confidence"],
                    "all_color_scores": color_result["all_scores"],
                }
                new_results.append(record)

            except Exception as e:
                print(f"⚠️ 处理失败 {scene_id}/{obj_id}: {e}")
    
    all_results = existing_records + new_results
    print(f"\n✅ 本轮新处理 {len(new_results)} 个物体，累计 {len(all_results)} 条")
    print(f"   失败 group：{failed_groups} 个")
    print(f"   缺失 feature 跳过：{skipped_without_feature} 个")

    # 3. 追加新结果到 JSONL（已有记录不重复写入）
    output_dir = os.path.dirname(output_jsonl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    if new_results:
        print(f"💾 追加 {len(new_results)} 条新结果到：{output_jsonl_path}")
        with open(output_jsonl_path, "a", encoding="utf-8") as f:
            for record in new_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✅ 完成！本轮追加 {len(new_results)} 条，文件中共 {len(all_results)} 条记录")
    else:
        print(f"💾 无新结果需写入，已有文件：{output_jsonl_path}（共 {len(all_results)} 条）")

    # 4. 打印统计信息
    if all_results:
        colors_count = {}
        tiers_count = {}
        
        for r in all_results:
            c = r["predicted_color"]
            t = r["predicted_tier"]
            
            colors_count[c] = colors_count.get(c, 0) + 1
            tiers_count[t] = tiers_count.get(t, 0) + 1
        
        print("\n📊 颜色分布统计 (Top 10):")
        for color, count in sorted(colors_count.items(), key=lambda x: -x[1])[:10]:
            print(f"   {color}: {count}")
            
        print("\n📊 档位分布统计:")
        for tier in sorted(tiers_count.keys()):
            print(f"   Tier {tier}: {tiers_count[tier]}")

    # 【新增】============================== 按 Tier 统计所有颜色属性平均分 ==============================
    print("\n" + "="*60)
    print("📊 按 Tier 分组统计所有颜色属性的平均分数")
    print("="*60)
    
        # 【新增】============================== 按 Tier 统计颜色属性平均分 ==============================
    print("\n" + "="*60)
    print("📊 按 Tier 分组统计：计算每个 Tier 内所属颜色的平均分数")
    print("="*60)
    
        # 【新增】============================== 为每张图片计算各 Tier 平均分并重新判定 ==============================
    print("\n" + "="*60)
    print("📊 为每张图片计算各 Tier 的平均分，取最高作为最终 Tier")
    print("="*60)
    
    if all_results:
        from collections import defaultdict
        
        # 建立 tier -> [属于该tier的所有颜色名] 的反向映射
        tier_to_colors = defaultdict(list)
        for color_name, tier_id in COLOR_TO_TIER.items():
            tier_to_colors[tier_id].append(color_name)
        
        print(f"\n各 Tier 包含的颜色数:")
        for tier in sorted(tier_to_colors.keys()):
            print(f"   Tier {tier:3d}: {len(tier_to_colors[tier])} 个颜色")
        
        # 处理每张图片
        final_results = []
        tier_change_stats = defaultdict(int)  # (original, final) -> count
        
        for r in all_results:
            all_scores = r["all_color_scores"]  # {"White": 0.32, "Blue": 0.45, ...}
            original_tier = r["predicted_tier"]
            
            # 计算该图片在每个 tier 上的平均分
            tier_avg_scores = {}  # tier -> avg_score
            
            for tier, colors in tier_to_colors.items():
                scores_in_tier = []
                for color_name in colors:
                    if color_name in all_scores:
                        scores_in_tier.append(all_scores[color_name])
                
                if scores_in_tier:  # 如果该 tier 有颜色分数
                    avg_score = sum(scores_in_tier) / len(scores_in_tier)
                    tier_avg_scores[tier] = {
                        "avg_score": avg_score,
                        "color_count": len(scores_in_tier),
                        "all_scores": scores_in_tier
                    }
            
            # 找出平均分最高的 tier
            if tier_avg_scores:
                best_tier = max(tier_avg_scores.items(), key=lambda x: x[1]["avg_score"])
                final_tier = best_tier[0]
                final_avg_score = best_tier[1]["avg_score"]
                final_color_count = best_tier[1]["color_count"]
            else:
                final_tier = original_tier
                final_avg_score = 0.0
                final_color_count = 0
            
            # 记录变化统计
            tier_change_stats[(original_tier, final_tier)] += 1
            
            final_results.append({
                "scene_id": r["scene_id"],
                "object_id": r["object_id"],
                "original_tier": original_tier,
                "original_color": r["predicted_color"],
                "final_tier": final_tier,
                "final_tier_avg_score": round(final_avg_score, 4),
                "final_tier_color_count": final_color_count,
                "all_tier_avg_scores": {t: round(s["avg_score"], 4) for t, s in tier_avg_scores.items()}
            })
        
        # 打印部分样本的详细对比（前5个变化的）
        print("\n样本示例（前5个 Tier 发生变化的）:")
        changed_samples = [r for r in final_results if r["original_tier"] != r["final_tier"]]
        for i, sample in enumerate(changed_samples[:5]):
            print(f"\n   样本 {i+1}: {sample['scene_id']}/obj_{sample['object_id']}")
            print(f"      原始: Tier {sample['original_tier']} ({sample['original_color']})")
            print(f"      最终: Tier {sample['final_tier']} (avg_score={sample['final_tier_avg_score']:.4f})")
            print(f"      各 Tier 平均分: {sample['all_tier_avg_scores']}")
        
        # 统计最终 tier 分布
        final_tier_dist = defaultdict(int)
        for item in final_results:
            final_tier_dist[item["final_tier"]] += 1
        
        print("\n最终 Tier 分布:")
        for tier in sorted(final_tier_dist.keys()):
            count = final_tier_dist[tier]
            pct = count / len(final_results) * 100
            print(f"   Tier {tier:3d}: {count:5d} ({pct:5.2f}%)")
        
        # 对比原始分布
        original_tier_dist = defaultdict(int)
        for r in all_results:
            original_tier_dist[r["predicted_tier"]] += 1
        
        print("\n原始 Tier 分布:")
        for tier in sorted(original_tier_dist.keys()):
            count = original_tier_dist[tier]
            pct = count / len(all_results) * 100
            print(f"   Tier {tier:3d}: {count:5d} ({pct:5.2f}%)")
        
        # 统计变化
        changed = sum(1 for item in final_results if item["original_tier"] != item["final_tier"])
        print(f"\nTier 变化统计: {changed}/{len(final_results)} ({changed/len(final_results)*100:.2f}%)")
        
        # 打印主要的迁移方向
        print("\n主要的 Tier 迁移方向 (原始 -> 最终) [Top 10]:")
        sorted_changes = sorted(tier_change_stats.items(), key=lambda x: -x[1])
        for (orig, final), count in sorted_changes[:10]:
            if orig != final:
                print(f"   Tier {orig:3d} -> Tier {final:3d}: {count:4d} 个")
        
        # 保存结果
        final_path = output_jsonl_path.replace(".jsonl", "_per_image_tier2.jsonl")
        with open(final_path, 'w', encoding='utf-8') as f:
            for item in final_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"\n💾 每张图片的 Tier 判定结果已保存到: {final_path}")

if __name__ == "__main__":
    main()

import sys
sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import json
import time
from typing import List, Tuple, Optional, Dict

# ================= 路径配置 =================
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入模块
import Semantic_landmark_in_stage1 as stage1_module
import Indentity_in_stage1_new as stage2_module

# ================= 数据路径配置 =================
# 与 Stage 1 共用同一份 ND jsonl：construction 内地标 + 评估行一一对应
JSONL_PATH = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0324.jsonl"
COLOR_TIER_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature/all_objects_color_per_image_tier2.jsonl"

BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d"

# 🔥 新增：定义结果保存路径
OUTPUT_JSON_PATH = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2/pipeline/evaluation_results_log.json"

# Stage 2 排序 Recall@20 / @30 / @40（单次取 top MAX_K）
EVAL_TOP_K_LIST = [20, 30, 40]
MAX_K = max(EVAL_TOP_K_LIST)
# ================================================================

def run_full_evaluation():
    """
    遍历 JSONL 所有条目，评估 Recall@20, @30, @40（Stage 2 编码主体 category2+color+identity_feat）并保存结果到 JSON
    """
    
    print("="*70)
    print(f"🚀 开始全量评估 | 数据集：{os.path.basename(JSONL_PATH)}")
    print(f"🎯 评估指标：Recall@{EVAL_TOP_K_LIST}")
    print(f"💾 结果保存至：{OUTPUT_JSON_PATH}")
    print("="*70)

    # 0. 预初始化模型
    print("\n⏳ 正在初始化模型...")
    try:
        # 假设 stage2 只需要初始化一次即可支持不同的 top_k 查询
        stage2_module.init_model(device_id="0")
        print("   ✅ 模型就绪")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    # 统计变量 (改为字典以支持多个 K)
    total_lines = 0
    processed_count = 0
    
    # 初始化统计字典
    stats = {
        'success_count': {k: 0 for k in EVAL_TOP_K_LIST},
        'failed_count': {k: 0 for k in EVAL_TOP_K_LIST},
        'stage1_direct_hit': {k: 0 for k in EVAL_TOP_K_LIST},
        'stage2_rank_hit': {k: 0 for k in EVAL_TOP_K_LIST},
        'stage1_miss_gt': 0,     # 这个对所有 K 都一样，Stage 1 丢了就全丢了
        'stage2_miss_topk': {k: 0 for k in EVAL_TOP_K_LIST},
        'error_count': 0
    }

    # 🔥 新增：用于存储每条详细结果的列表
    results_log = []

    start_time = time.time()

    if not os.path.exists(JSONL_PATH):
        print(f"❌ 文件不存在: {JSONL_PATH}")
        return
    
    good=0
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            # 初始化当前条目的记录变量
            current_scene_id = None
            current_gt_id = None
            current_ann_id = None # 假设 ann_id 等同于 line_idx 或者需要从 item 中获取，这里暂用 line_idx 或 item 中的特定字段
            current_is_success = False
            current_candidates_40 = []
            current_candidates_30 = []
            current_candidates_20 = []

            try:
                item = json.loads(line)
                scene_id = item.get('scene_id')
                gt_id = str(item.get('object_id'))
                
                # 假设 ann_id 可以使用行索引，或者如果数据中有特定字段如 'id' 或 'ann_id' 请替换此处
                # 这里暂时使用 line_idx 作为 ann_id，如果有具体字段请修改：item.get('ann_id')
                ann_id = item.get('ann_id', line_idx) 

                current_scene_id = scene_id
                current_gt_id = gt_id
                current_ann_id = ann_id

                category = item.get('object_name')
                construction_list = item.get('construction', [])
                
                category_for_stage2 = category
                main_color = ''
                identity_feat = ''

                for obj in construction_list:
                    if obj.get('is_main') is True:
                        category_for_stage2 = obj.get('category2', category)
                        main_color = obj.get('color', '') or ''
                        if isinstance(main_color, str):
                            main_color = main_color.strip()
                        else:
                            main_color = str(main_color).strip()
                        identity_feat = obj.get('identity_feature', '') or ''
                        break
                
                # --- Stage 1: 粗筛（地标来自当前行 construction 非空 landmark；颜色为 final_tier 与 GT 一致）---
                final_results = stage1_module.process_files_to_candidates(
                    BBOX_DIR, JSONL_PATH, COLOR_TIER_JSONL, line_idx
                )
                
                candidate_ids = final_results["candidates"]
                gt_id = final_results["gt_id"] # 更新 gt_id 以防 stage1 中做了处理
                
                # 更新当前记录的 GT ID (以 stage1 返回的为准)
                current_gt_id = gt_id

                if gt_id in final_results["candidates"]:
                    good+=1

                # 处理 Stage 1 丢失 GT 的情况
                if not candidate_ids or gt_id not in candidate_ids:
                    stats['stage1_miss_gt'] += 1
                    for k in EVAL_TOP_K_LIST:
                        stats['failed_count'][k] += 1
                    
                    # 记录失败情况：候选列表为空或不包含GT
                    # 即使失败，也保存当前的候选列表（可能是空的）
                    current_candidates_40 = candidate_ids[:40]
                    current_candidates_30 = candidate_ids[:30]
                    current_candidates_20 = candidate_ids[:20]
                    current_is_success = False # 明确标记为失败

                    # 构建结果字典并加入日志
                    results_log.append({
                        "scene_id": current_scene_id,
                        "object_id": current_gt_id,
                        "ann_id": current_ann_id,
                        "is_success": current_is_success,
                        "candidates_40": current_candidates_40,
                        "candidates_30": current_candidates_30,
                        "candidates_20": current_candidates_20,
                    })
                    continue

                processed_count += 1

                # --- 判定逻辑 (针对每个 K) ---
                num_candidates = len(candidate_ids)
                
                # 为了效率，如果需要跑 Stage 2，我们只跑一次取最大的 K (MAX_K)
                ranked_results = None
                topk_ids_map = {} # {k: [id_list]}

                # 用于判断最终是否成功 (以最高 K=40 是否命中作为整体成功标志)
                hit_at_max_k = False

                for k in EVAL_TOP_K_LIST:
                    is_hit = False
                    
                    # 规则 1: 如果候选数 <= k，直接命中
                    if num_candidates <= k:
                        is_hit = True
                        stats['stage1_direct_hit'][k] += 1
                    else:
                        # 规则 2: 候选数 > k，需要 Stage 2 排序
                        if ranked_results is None:
                            ranked_results = stage2_module.get_topk_candidates(
                                candidate_ids=candidate_ids,
                                scene_id=scene_id,
                                category2=category_for_stage2,
                                color=main_color,
                                identity_feat=identity_feat,
                                top_k=MAX_K,
                            )
                            # 预处理出不同 K 的 ID 列表
                            all_ids = [x[0] for x in ranked_results]
                            for mk in EVAL_TOP_K_LIST:
                                topk_ids_map[mk] = all_ids[:mk]
                        
                        # 检查 GT 是否在 Top-K 中
                        if gt_id in topk_ids_map[k]:
                            is_hit = True
                            stats['stage2_rank_hit'][k] += 1
                        else:
                            stats['stage2_miss_topk'][k] += 1

                    if is_hit:
                        stats['success_count'][k] += 1
                        if k == MAX_K:
                            hit_at_max_k = True
                    else:
                        stats['failed_count'][k] += 1

                # 🔥 记录当前条目的详细结果
                # 获取各个 K 对应的候选列表
                # 注意：如果 num_candidates <= k，candidate_ids 本身就是前 k 个（因为全部都在）
                # 如果 num_candidates > k，则使用 stage2 排序后的 topk_ids_map[k]
                
                if num_candidates <= 40:
                    current_candidates_40 = candidate_ids
                else:
                    current_candidates_40 = topk_ids_map.get(40, candidate_ids[:40])

                if num_candidates <= 30:
                    current_candidates_30 = candidate_ids
                else:
                    current_candidates_30 = topk_ids_map.get(30, candidate_ids[:30])

                if num_candidates <= 20:
                    current_candidates_20 = candidate_ids
                else:
                    current_candidates_20 = topk_ids_map.get(20, candidate_ids[:20])

                current_is_success = hit_at_max_k

                results_log.append({
                    "scene_id": current_scene_id,
                    "object_id": current_gt_id,
                    "ann_id": current_ann_id,
                    "is_success": current_is_success,
                    "candidates_40": current_candidates_40,
                    "candidates_30": current_candidates_30,
                    "candidates_20": current_candidates_20,
                })

                # 进度打印
                if (line_idx + 1) % 100 == 0:
                    msg_parts = []
                    for k in EVAL_TOP_K_LIST:
                        r = (stats['success_count'][k] / processed_count * 100) if processed_count > 0 else 0
                        msg_parts.append(f"@{k}:{r:.1f}%")
                    print(f"   ... 已处理 {line_idx + 1} 条，当前 Recall: {' | '.join(msg_parts)}")

            except Exception as e:
                print(f"Error processing line {line_idx}: {e}")
                stats['error_count'] += 1
                
                # 即使出错，也尝试记录一条错误日志（可选）
                results_log.append({
                    "scene_id": current_scene_id,
                    "object_id": current_gt_id,
                    "ann_id": current_ann_id,
                    "is_success": False,
                    "error": str(e),
                    "candidates_40": [],
                    "candidates_30": [],
                    "candidates_20": [],
                })
                continue

    # 🔥 保存结果到 JSON 文件
    try:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as out_f:
            json.dump(results_log, out_f, ensure_ascii=False, indent=2)
        print(f"\n✅ 详细结果已保存至：{OUTPUT_JSON_PATH}")
        print(f"   共保存 {len(results_log)} 条记录")
    except Exception as e:
        print(f"\n❌ 保存结果文件失败：{e}")

    end_time = time.time()
    duration = end_time - start_time

    # ================= 打印最终报告 =================
    print("\n" + "="*70)
    print("📊 评估结果汇总")
    print("="*70)
    print(f"总数据行数：{total_lines}")
    print(f"有效处理数：{processed_count} (跳过 {total_lines - processed_count} 条，含错误 {stats['error_count']} 条)")
    print(f"Stage 1 丢失 GT 总数：{stats['stage1_miss_gt']} (这对所有 K 都是失败)")
    print("-" * 70)

    for k in EVAL_TOP_K_LIST:
        print(f"\n🔹 指标详情：Recall@{k}")
        print(f"   ✅ 成功总数：{stats['success_count'][k]}")
        print(f"      ├─ Stage 1 直接命中 (候选≤{k}): {stats['stage1_direct_hit'][k]}")
        print(f"      └─ Stage 2 排序命中 (候选>{k}): {stats['stage2_rank_hit'][k]}")
        print(f"   ❌ 失败总数：{stats['failed_count'][k]}")
        print(f"      ├─ Stage 1 未包含 GT: {stats['stage1_miss_gt']}")
        print(f"      └─ Stage 2 未排进 Top{k}: {stats['stage2_miss_topk'][k]}")
        
        if processed_count > 0:
            final_recall = stats['success_count'][k] / processed_count * 100
            print(f"   🏆 最终 Recall@{k}: {final_recall:.2f}%")
        else:
            print("   ⚠️ 没有有效数据")

    print("\n" + "="*70)
    print(f"⏱️ 耗时：{duration:.2f} 秒 ({processed_count/duration:.2f} 条/秒)")
    print("="*70)

if __name__ == "__main__":
    run_full_evaluation()
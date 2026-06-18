"""
Stage1 粗筛 + Stage2 末位淘汰（无 Top-K 截断）。

对同一批 Stage1 成功样本跑两种 Stage2：
- 池人数 < STOP_BELOW_THRESHOLD 时不再删减（提前结束）；
- 不提前停止，按非空字段跑完各轮。
"""
import sys

sys.path.append("/hpc2hdd/home/yxiao224/Henry/code/Uni3D")
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import time
from typing import List, Optional

# ================= 路径配置 =================
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import Semantic_landmark_in_stage1 as stage1_module
import Indentity_in_stage1_elim as stage2_module

# ================= 数据路径配置 =================
JSONL_PATH = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0324.jsonl"
COLOR_TIER_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0311data/feature/all_objects_color_per_image_tier2.jsonl"

BBOX_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/box3d"

OUTPUT_JSON_PATH = (
    "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2/pipeline/evaluation_results_elim.json"
)

# 方案「提前停」：当前池人数 < 该值则不再进行后续轮次的末位淘汰
STOP_BELOW_THRESHOLD = 25


def run_elim_evaluation():
    print("=" * 70)
    print(f"🚀 末位淘汰 Pipeline（无 Top-K）| 数据集：{os.path.basename(JSONL_PATH)}")
    print(f"💾 结果保存至：{OUTPUT_JSON_PATH}")
    print("=" * 70)

    print("\n⏳ 正在初始化模型...")
    try:
        stage2_module.init_model(device_id="0")
        print("   ✅ 模型就绪")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return

    total_lines = 0
    processed_count = 0
    stats = {
        "stage1_miss_gt": 0,
        "error_count": 0,
    }
    # 方案 A：池 < STOP_BELOW_THRESHOLD 即停止后续删减
    stop_sum_n = 0
    stop_hit = 0
    stop_final_ns: List[int] = []
    # 方案 B：不提前停止（stop_elimination_when_below=None）
    full_sum_n = 0
    full_hit = 0
    full_final_ns: List[int] = []

    results_log = []
    start_time = time.time()

    if not os.path.exists(JSONL_PATH):
        print(f"❌ 文件不存在: {JSONL_PATH}")
        return

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            current_scene_id: Optional[str] = None
            current_gt_id: Optional[str] = None
            current_ann_id = None

            try:
                item = json.loads(line)
                scene_id = item.get("scene_id")
                gt_id = str(item.get("object_id"))
                ann_id = item.get("ann_id", line_idx)

                current_scene_id = scene_id
                current_gt_id = gt_id
                current_ann_id = ann_id

                category = item.get("object_name")
                construction_list = item.get("construction", [])

                category_for_stage2 = category
                main_color = ""
                identity_feat = ""

                for obj in construction_list:
                    if obj.get("is_main") is True:
                        category_for_stage2 = obj.get("category2", category)
                        main_color = obj.get("color", "") or ""
                        if isinstance(main_color, str):
                            main_color = main_color.strip()
                        else:
                            main_color = str(main_color).strip()
                        identity_feat = obj.get("identity_feature", "") or ""
                        break

                final_results = stage1_module.process_files_to_candidates(
                    BBOX_DIR, JSONL_PATH, COLOR_TIER_JSONL, line_idx
                )

                candidate_ids = final_results["candidates"]
                gt_id = final_results["gt_id"]
                current_gt_id = gt_id

                if not candidate_ids or gt_id not in candidate_ids:
                    stats["stage1_miss_gt"] += 1
                    results_log.append(
                        {
                            "scene_id": current_scene_id,
                            "object_id": current_gt_id,
                            "ann_id": current_ann_id,
                            "is_success": False,
                            "reason": "stage1_miss_gt",
                            "final_candidate_count": len(candidate_ids or []),
                            "final_candidates": candidate_ids or [],
                        }
                    )
                    continue

                processed_count += 1

                ranked_stop, ranked_full = stage2_module.get_eliminated_candidates_pair(
                    candidate_ids=candidate_ids,
                    scene_id=scene_id,
                    category2=category_for_stage2,
                    color=main_color,
                    identity_feat=identity_feat,
                    stop_elimination_when_below=STOP_BELOW_THRESHOLD,
                )

                ids_stop = [x[0] for x in ranked_stop]
                ids_full = [x[0] for x in ranked_full]
                n_stop = len(ids_stop)
                n_full = len(ids_full)

                stop_sum_n += n_stop
                stop_final_ns.append(n_stop)
                if n_stop > 0 and gt_id in set(ids_stop):
                    stop_hit += 1

                full_sum_n += n_full
                full_final_ns.append(n_full)
                if n_full > 0 and gt_id in set(ids_full):
                    full_hit += 1

                results_log.append(
                    {
                        "scene_id": current_scene_id,
                        "object_id": current_gt_id,
                        "ann_id": current_ann_id,
                        "stage1_candidate_count": len(candidate_ids),
                        f"scheme_stop_below_{STOP_BELOW_THRESHOLD}": {
                            "final_candidate_count": n_stop,
                            "final_candidates": ids_stop,
                            "hit": gt_id in set(ids_stop) if n_stop > 0 else False,
                        },
                        "scheme_no_early_stop": {
                            "final_candidate_count": n_full,
                            "final_candidates": ids_full,
                            "hit": gt_id in set(ids_full) if n_full > 0 else False,
                        },
                    }
                )

                if (line_idx + 1) % 100 == 0 and processed_count > 0:
                    acc_stop = stop_hit / processed_count * 100
                    avg_stop = stop_sum_n / processed_count
                    acc_full = full_hit / processed_count * 100
                    avg_full = full_sum_n / processed_count
                    print(
                        f"   ... 已处理 {line_idx + 1} 行 | "
                        f"<{STOP_BELOW_THRESHOLD}停: 准确率 {acc_stop:.1f}% 平均 {avg_stop:.2f} | "
                        f"不提前停: 准确率 {acc_full:.1f}% 平均 {avg_full:.2f}"
                    )

            except Exception as e:
                print(f"Error processing line {line_idx}: {e}")
                stats["error_count"] += 1
                results_log.append(
                    {
                        "scene_id": current_scene_id,
                        "object_id": current_gt_id,
                        "ann_id": current_ann_id,
                        "error": str(e),
                        f"scheme_stop_below_{STOP_BELOW_THRESHOLD}": {},
                        "scheme_no_early_stop": {},
                    }
                )
                continue

    try:
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as out_f:
            json.dump(results_log, out_f, ensure_ascii=False, indent=2)
        print(f"\n✅ 详细结果已保存至：{OUTPUT_JSON_PATH}")
        print(f"   共保存 {len(results_log)} 条记录")
    except Exception as e:
        print(f"\n❌ 保存结果文件失败：{e}")

    end_time = time.time()
    duration = end_time - start_time

    def _pct(vals: List[int], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        i = int(round((len(s) - 1) * p))
        return float(s[i])

    print("\n" + "=" * 70)
    print("📊 末位淘汰评估汇总（两种方案，同一批 Stage1 成功样本）")
    print("=" * 70)
    print(f"总数据行数：{total_lines}")
    print(f"Stage1 含 GT 且进入 Stage2：{processed_count}")
    print(f"Stage1 丢失 GT：{stats['stage1_miss_gt']}")
    print(f"解析/运行错误：{stats['error_count']}")
    print("-" * 70)
    print(
        f"方案 A：当前池人数 < {STOP_BELOW_THRESHOLD} 时停止后续删减"
    )
    if processed_count > 0:
        print(
            f"   准确率（GT ∈ 幸存者）：{stop_hit} / {processed_count} = "
            f"{stop_hit / processed_count * 100:.2f}%"
        )
        print(f"   平均幸存者数量：{stop_sum_n / processed_count:.4f}")
        if stop_final_ns:
            print(
                f"   数量范围：min={min(stop_final_ns)} max={max(stop_final_ns)} "
                f"p50={_pct(stop_final_ns, 0.5):.4f} p90={_pct(stop_final_ns, 0.9):.4f}"
            )
    print("-" * 70)
    print("方案 B：不提前停止（字段非空则照常跑各轮，空字段仍跳过该轮）")
    if processed_count > 0:
        print(
            f"   准确率（GT ∈ 幸存者）：{full_hit} / {processed_count} = "
            f"{full_hit / processed_count * 100:.2f}%"
        )
        print(f"   平均幸存者数量：{full_sum_n / processed_count:.4f}")
        if full_final_ns:
            print(
                f"   数量范围：min={min(full_final_ns)} max={max(full_final_ns)} "
                f"p50={_pct(full_final_ns, 0.5):.4f} p90={_pct(full_final_ns, 0.9):.4f}"
            )
    print("-" * 70)
    print(f"⏱️ 耗时：{duration:.2f} 秒")
    if processed_count > 0:
        print(f"   约 {processed_count / duration:.2f} 条/秒（含 Stage1+Stage2 单次双策略仿真）")
    print("=" * 70)


if __name__ == "__main__":
    run_elim_evaluation()

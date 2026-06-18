# 颜色模块改进记录

## 第一步目标

先提升 `GT ∈ candidates` 的准确率，再考虑缩小候选集。

## 问题定位

原脚本 `01_Semantic_color_in_stage1_support_object_proximity_shrink.py` 的 ND 基线结果为：

- 平均候选数：`75.3043`
- 准确率：`0.6696`

离线检查后发现：

- 同类别候选本身基本都能保住 GT。
- 主要掉点来自 `support_object_proximity_shrink` 的裁剪。
- 颜色模块还会从 `description` 抽颜色，容易把配角颜色误当成主体颜色，进一步增加误裁剪风险。

## 本次修改

1. 颜色来源改为只看 `construction` 中 `is_main=True` 的主体颜色。
2. 如果主体的 `category/category2` 和当前目标类别不一致，则直接跳过颜色约束。
3. 新增主体颜色软匹配：
   先把同类候选分成 `exact / soft / unknown / far` 四类。
   `soft` 定义为落在 tier 邻域和混淆组内。
4. 当前版本按你的要求改成无阈值：
   只要主体颜色能映射出 tier，就直接用 `allowed_tiers` 过滤候选；
   如果过滤后为空，则回退到原始同类候选，避免输出空集合。
5. 当前已回退到 `+/-2` 的无阈值版本，并保留混淆颜色组。
6. `support_object_proximity_shrink` 仍默认关闭，通过 `--enable-support-shrink` 显式开启。

## 当前阶段结论

这一步是“主体颜色软匹配”的无阈值版本。

- 默认运行：ND / NO 都直接启用 `allowed_tiers` 过滤。
- 评估结果：
  ND：平均候选数 `131.0`，准确率 `0.6957`
  NO：平均候选数 `146.7328`，准确率 `0.6947`
- 结论：`+/-2 + 混淆组` 的无阈值过滤会显著压缩候选，但会误杀大量 GT，需要结合具体 case 判断是描述问题还是颜色预测问题。

## 2026-04-27 验证：替换颜色 tier 文件

验证脚本：

- `01_Semantic_color_in_stage1_support_object_proximity_shrink.py`

对比的颜色文件：

- 默认：`/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/color/all_objects_color_per_image_tier2.jsonl`
- 替换：`/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_colors_per_image_tier2.jsonl`

验证结果：

- 两次评估输出完全一致。
- ND：平均候选数 `131.0`，准确率 `0.6957`
- NO：平均候选数 `146.7328`，准确率 `0.6947`

进一步检查：

- 两个文件的 `md5` 不同。
- 但 `(scene_id, object_id) -> final_tier` 映射完全一致：
  `len1=13489`, `len2=13489`, `common=13489`, `diff=0`

结论：

- 对当前脚本来说，加载 `single_image_colors_per_image_tier2.jsonl` 不会带来提升。
- 原因不是脚本没有切换，而是这两个颜色文件在 `final_tier` 映射上完全一样。

# 超图匹配后续工作（阶段二 / 三）

## 融合细排（已实现）

主入口 [`core/match_hypergraphs.py`](../core/match_hypergraphs.py)：

- `--fusion_lambda 1.0`：纯超图分（默认）。
- `--fusion_lambda 0.7`：\( \lambda\cdot\text{hg} + (1-\lambda)\cdot\text{rank\_prior} \)。
- `--fusion_sweep 0.5,0.6,0.7,1.0`：多组 λ，输出 `matching_eval_summary_fusion_sweep.json`（路径由 `--summary` 推导）。

[`match_hypergraphs_v2.py`](../core/match_hypergraphs_v2.py) 支持 `--fusion_lambda`，扫参请用主脚本。

---

本文件对应中长期项（未编码或持续迭代）。

## 阶段二：描述与关系侧

- 对失败 query 抽样，统计描述边类型与实例边不一致的模式。
- 迭代 [`core/relation_mapper.py`](../core/relation_mapper.py) 关键词表。
- 按数据集 BEV 约定微调 [`core/spatial_relations.py`](../core/spatial_relations.py) 阈值。
- 若 `construction` 漏边，检查 `02NER` 提示词与 [`CityRefer_desc_hypergraphs_dedup.jsonl`](../../data/cityrefer/meta_data/CityRefer_desc_hypergraphs_dedup.jsonl) 生成流程。

## 阶段三：跨网格实例

- 评估 SAM3/fusion 在 [`core/fusion.py`](../core/fusion.py) 中按 grid 切分导致的实例断裂。
- 设计跨 tile 合并策略后再生成实例 PLY / 超图，复测 `match_hypergraphs.py` 的 Top-1 与平局率。

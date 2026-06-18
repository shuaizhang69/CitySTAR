# Hypergraph Match V7

这版保留了实验里稳定有效、且便于迁移的策略，去掉了依赖细碎权重的小技巧。

## 数据构造

1. 类别空间和 bbox 图对齐  
文本侧统一使用 `ground / building / parking / vehicle / truck / highvegetation / fence`，避免 `car` 和 `vehicle`、`tree` 和 `highvegetation` 这种类别错位。

2. 锚点类别清洗  
`reference_anchor` 不再把代词当有效锚点。  
属性描述优先识别主物体，例如 `building with brown walls` 仍然优先映到 `building`，而不是误映成 `fence`。

3. 保守的 fallback 提取  
当原始结构化抽取太空时，允许从原句中补少量显式类别词和关系短语，但只补高置信、可直接摘抄的关系。

4. 不再强行猜关系  
无法稳定归一的 `directional / topological` 短语不再硬回退到 `front_of / inside`，避免错误边污染整张描述图。

5. bbox 子图按描述裁剪  
候选局部图只保留描述里相关的类别，并且每类只保留靠近候选框的少量实例，降低大场景同类噪声。

## 匹配

1. main 节点强绑定当前 candidate  
匹配时主节点不再在候选图里任意挑同类实例，而是固定到当前候选框本身。

2. 类别归一后再匹配  
文本图和 bbox 图统一做类别 canonicalization，减少本来应当匹配却因为类别名不同而丢分。

3. 复杂描述先裁弱边  
当描述边过多时，只保留强边或最有判别力的少量边，避免噪声边把 GT 从前列挤掉。

4. 安全关系才用连续几何  
`adjacent / front_of / left_of / right_of` 这类稳定关系，额外使用简单几何兼容分；不把这套规则硬套到 `inside / on_surface / facing` 这类更脆的关系上。

5. 用简单证据相加，不做复杂加权  
最终分数只组合几类可解释证据：
   - 关系匹配
   - 节点类别覆盖
   - anchor 类别支持
   - 数量提示支持
   - 安全关系下的连续几何支持

## 迁移内容

- `our_data/superGraph_for_Anchor_new/1_NER_multiprocess.py`
- `our_data/superGraph_for_Anchor_new/2_generate_desc_hypergraphs.py`
- `our_data/superGraph_for_Anchor_new/3_generate_bbox_hypergraphs_v2.py`
- `our_data/superGraph_for_Anchor_new/desc_hypergraph.py`
- `new_code_for_Anchor/04_hypergraph_matchv7.py`

## 使用顺序

1. 跑 `1_NER_multiprocess.py`
2. 跑 `2_generate_desc_hypergraphs.py`
3. 跑 `3_generate_bbox_hypergraphs_v2.py`
4. 跑 `04_hypergraph_matchv7.py`

## 注意

这版目标偏向稳定提升 `top10 / top20`，不是单独优化 `top1`。

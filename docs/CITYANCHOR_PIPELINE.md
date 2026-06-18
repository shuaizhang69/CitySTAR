# CityAnchor Pipeline

Config: `configs/cityanchor.pipeline.json`

Remote audit note: the zhangshuai server has STPLS3D raw/txt data under
`/mnt/shuaizhang_data/STPLS3D*` and a small
`/mnt/shuaizhang_data/CityRefer_1/data/cityrefer/meta_data/cityanchor_val_ND.json`
file, but the full final CityAnchor scripts and generated ND/NO artifacts were
not found under the audited zhangshuai paths. This repository's local
`cityanchor/` code is therefore the canonical final pipeline unless the missing
remote path is identified.

## Inputs

Bundled inputs:

```text
cityanchor/data/CityAnchor/cityanchor_val_ND_0324.jsonl
cityanchor/data/city_Anchor/cityanchor_val_NO_0324.jsonl
cityanchor/data/city_Anchor/cityanchor_val_NO_new_0324_new.jsonl
cityanchor/data/city_Anchor/single_image_colors2_per_image_tier2.jsonl
cityanchor/data/city_Anchor/main_object_color_llm_rerun_object_name.json
```

External inputs for from-scratch runs:

```text
data/raw/cityanchor/bbox/
data/raw/cityanchor/desc/cityanchor_val_{ND,NO}_infer_result.jsonl
data/raw/cityanchor/context_images/
```

## Stages

1. `stage1_candidates`

   Runs
   `cityanchor/new_code_for_Anchor/01_Semantic_color_in_stage1_support_object_proximity_shrink.py`.
   It filters candidates by category and color tiers, using the main-object
   query color map.

2. `desc_hypergraphs`

   Runs `cityanchor/superGraph_for_Anchor_fianl/2_generate_desc_hypergraphs.py`
   over CityAnchor LLM NER outputs.

3. `bbox_hypergraphs`

   Runs `cityanchor/superGraph_for_Anchor_fianl/3_generate_bbox_hypergraphs_v2.py`.
   It builds candidate bbox-only hypergraphs aligned with the stage1 candidate
   list.

4. `geometry_mentions`

   Runs `cityanchor/new_code_for_Anchor/06_extract_geometry_mentions.py`.
   It uses heuristics by default and can optionally use an OpenAI-compatible LLM.

5. `hypergraph_match_v7`

   Runs
   `cityanchor/superGraph_for_Anchor_fianl/04_hypergraph_matchv7_with_geometrey_awrev5.py`.
   This final matcher combines stage1 order, hypergraph matching, and geometry
   tie-breaking. The script now accepts overrides for candidates, desc
   hypergraphs, bbox hypergraphs, geometry mentions, bbox directory, and output.

6. `dashscope_rerank`

   Optional. Uses `cityanchor/new_code_for_Anchor/08reranking_Reranking_new.py`
   and `DASHSCOPE_API_KEY`.

## Example

```bash
python scripts/run_pipeline.py doctor --dataset cityanchor --split ND
python scripts/run_pipeline.py plan --dataset cityanchor --split ND
python scripts/run_pipeline.py run --dataset cityanchor --split ND --from-stage stage1_candidates --to-stage hypergraph_match_v7
```

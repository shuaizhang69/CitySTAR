# Remote Server Audit

Audit target:

```text
ssh -l zhangshuai codex-lqh
/home/zhangshuai/workshop/open_vocabulary/LqhSpace
```

Audit date: 2026-06-18.

## Executive Status

| Area | Status | Evidence |
| --- | --- | --- |
| SAM3 / SensatUrban upstream | Mostly present on server | Full `LqhSpace/sam3` workspace, pipeline docs, scripts, and generated artifacts. Raw SensatUrban data exists under `/mnt/shuaizhang_data/SensatUrban` and CityRefer copy under `/mnt/shuaizhang_data/CityRefer_1/data/sensaturban`. |
| CityRefer downstream | Present enough for from-scratch/replay work | Official CityRefer code/data under `/mnt/shuaizhang_data/CityRefer_1`, stage1 and hypergraph artifacts under `LqhSpace/Stage1_code_result`. |
| CityAnchor downstream | Partial only | STPLS3D raw/txt data exists, and one `cityanchor_val_ND.json` exists. The final local CityAnchor scripts and full generated candidate/bbox/geometry/hgmatch artifacts were not found under zhangshuai paths. |

The important correction is that `LqhSpace/Stage1_code_result/data` is a stale
symlink:

```text
/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/data
  -> /home/zhangshuai/workshop/CityRefer_1/data
```

The real data location found during the audit is:

```text
/mnt/shuaizhang_data/CityRefer_1/data
```

Use that path, or recreate the symlink before running remote scripts from
`Stage1_code_result`.

## CityRefer And SAM3 Evidence

### CityRefer Repository

Remote path:

```text
/mnt/shuaizhang_data/CityRefer_1
```

Observed contents:

```text
README.md
requirements.txt
scripts/train.py
scripts/eval.py
scripts/train_3dqa.py
models/
utils/
data/cityrefer/
data/sensaturban/
outputs/sensaturban/*/checkpoints
```

Key dataset counts:

| File | Count |
| --- | ---: |
| `CityRefer_train.json` | 23586 |
| `CityRefer_val.json` | 5934 |
| `CityRefer_val_ND.json` | 502 |
| `CityRefer_train_missing_6scenes.json` | 7518 |
| `CityRefer_desc_hypergraphs_dedup.jsonl` | present |

Large data observed:

| Path | Size / count |
| --- | --- |
| `/mnt/shuaizhang_data/CityRefer_1/data/sensaturban/scans` | 79G, 69 top-level entries |
| `/mnt/shuaizhang_data/CityRefer_1/data/cityrefer/redbox` | 1.2G |
| `/mnt/shuaizhang_data/CityRefer_1/data/cityrefer/box3d` | 29 top-level entries |
| `/mnt/shuaizhang_data/CityRefer_1/outputs/sensaturban/3dqa_final/checkpoints` | 100M |

### SAM3 Workspace

Remote path:

```text
/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3
```

Observed pipeline documentation:

```text
PIPELINE_COMMANDS.md
SAM3_PIPELINE_NOTES.md
```

Observed upstream stages in those docs:

```text
raw PLY
  -> grid_split_downsampling.py
  -> batch_cluster.py
  -> capture_get_idbuffer_sensaturban.py
  -> sam3_mask_generation_sensaturban.py
  -> superpoint_fusion_sensaturban.py
  -> extract_instances_from_masks.py / by-class flow
  -> batch_superpoint_fusion.py
  -> merge_fusion_results_instances.py
```

No large model checkpoint files were found directly under `LqhSpace/sam3`
during this audit. Treat SAM3 model weights as an external/downloaded/cached
dependency when documenting a clean machine run.

### Stage1 / Hypergraph Workspace

Remote path:

```text
/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result
```

Observed relevant code/assets:

```text
02NER*.py
04crop_image.py
05VLm2_pro.py
06CLIP_image_all.py
06_CLIP_text.py
bev_render_bbox_batch4_context.py
convert_candidates_to_eval_log.py
instances_ply_to_sensaturban_bbox.py
sam3_segment/
cityrefer_env/
```

Observed generated artifacts:

| Path | Size / count |
| --- | --- |
| `bbox_hypergraphs_v2_overlap01_thr0.5_bestfull_fixed` | 2.1G, 5871 top-level entries |
| `fusion_by_class_3dqa_merged` | 3.6G, 119 top-level entries |
| `instances_final` | 6.7G, 45 top-level entries |
| `sensaturban_val_bbox` | 3.6G, 5 top-level entries |
| `matching_eval_summary_*.json` | present |
| `evaluation_results_log*.json` | present |

This is the strongest evidence that CityRefer plus SAM3 bbox/hypergraph logic
can be reconstructed on the server, once the stale data symlink is corrected.

## CityAnchor Evidence

CityAnchor-related raw data found:

```text
/mnt/shuaizhang_data/STPLS3D
/mnt/shuaizhang_data/STPLS3D_txt
/mnt/shuaizhang_data/CityRefer_1/data/cityrefer/meta_data/cityanchor_val_ND.json
```

Observed sizes:

| Path | Size |
| --- | ---: |
| `/mnt/shuaizhang_data/STPLS3D` | 35G |
| `/mnt/shuaizhang_data/STPLS3D_txt` | 141G |

The sample `cityanchor_val_ND.json` scene id `12_points_GTv3` maps to:

```text
/mnt/shuaizhang_data/STPLS3D/Synthetic_v3/12_points_GTv3.ply
/mnt/shuaizhang_data/STPLS3D_txt/V3txt_instance/12_points_GTv3.txt
/mnt/shuaizhang_data/STPLS3D_txt/v3_building/12_points_GTv3_building.txt
/mnt/shuaizhang_data/STPLS3D_txt/v3_tree/12_points_GTv3_tree.txt
```

Only one CityAnchor metadata file was found in the audited zhangshuai paths:

| File | Count |
| --- | ---: |
| `cityanchor_val_ND.json` | 115 |

The following local final-pipeline scripts were not found under
`/home/zhangshuai`, `/mnt/shuaizhang_data`, or `/mnt/datasets` in the audit:

```text
01_Semantic_color_in_stage1_support_object_proximity_shrink.py
04_hypergraph_matchv7_with_geometrey_awrev5.py
*hypergraph_matchv7*
*stage1_support_object*
```

`LqhSpace/3D-City-LLM/utils/cityanchor_dataset.py` exists, but it is a dataset
loader with hardcoded Colab-style paths, not the complete CityAnchor final
pipeline:

```text
/content/drive/MyDrive/CityAnchor_data/meta_data/cityanchor_meta_train_all_v1.json
/content/drive/MyDrive/CityAnchor_data/meta_data/cityanchor_meta_val_97.json
```

## Runability Decision

CityRefer from zero on this server is feasible, with path cleanup:

1. Use `/mnt/shuaizhang_data/CityRefer_1/data` as the data root.
2. Use `LqhSpace/sam3` for raw SensatUrban to masks/fusion/instances.
3. Use `LqhSpace/Stage1_code_result/sam3_segment` and sibling scripts for
   bbox conversion, bbox hypergraphs, and hypergraph matching.
4. Provide external LLM/API credentials only for regenerated NER/rerank stages.

CityAnchor from zero is not fully confirmed on this server:

1. STPLS3D raw/txt data exists.
2. A small ND metadata file exists.
3. The full final CityAnchor code and generated ND/NO candidates,
   bbox-hypergraphs, geometry mentions, and v7 match outputs were not found.

For a complete CityAnchor remote run, either copy this repository's CityAnchor
pipeline to the server and prepare `data/raw/cityanchor/*`, or locate the
missing original path that contains the final v7 pipeline and full CityAnchor
artifacts.

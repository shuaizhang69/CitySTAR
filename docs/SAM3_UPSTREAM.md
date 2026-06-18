# SAM3 / SensatUrban Upstream

Config: `configs/sam3_sensaturban.pipeline.json`

The SAM3 upstream code was explored on:

```text
/home/zhangshuai/workshop/open_vocabulary/LqhSpace
```

The lightweight wrappers copied into this repository came from:

```text
LqhSpace/Stage1_code_result/sam3_segment/
```

Important remote artifact directories observed:

```text
LqhSpace/sam3/
Stage1_code_result/sam3_segment/
Stage1_code_result/instances_merged_all/
Stage1_code_result/fusion_by_class_3dqa_merged/
Stage1_code_result/fusion_by_class_3dqa_bbox/
Stage1_code_result/sensaturban_bbox/
Stage1_code_result/bbox_hypergraphs_v2_overlap01_thr0.5_bestfull_fixed/
/mnt/shuaizhang_data/SensatUrban/
/mnt/shuaizhang_data/CityRefer_1/data/
```

Path caveat: `Stage1_code_result/data` is a stale symlink to
`/home/zhangshuai/workshop/CityRefer_1/data`. The data found during the audit is
under `/mnt/shuaizhang_data/CityRefer_1/data`.

## Stages

```text
raw SensatUrban point cloud
  -> grid_split
  -> superpoints
  -> render_views
  -> sam3_masks
  -> semantic_fusion
  -> merge_instances
  -> bbox_json
```

The model package and checkpoints are external dependencies. The repository
does not vendor the full SAM3 source tree, generated masks, PLY files, or
point-cloud data.

During the remote audit, no large SAM3 checkpoint files were found directly
inside `LqhSpace/sam3`; treat model weights as a download/cache dependency for a
clean setup.

Run:

```bash
python scripts/run_pipeline.py doctor --config configs/sam3_sensaturban.pipeline.json --split ALL
python scripts/run_pipeline.py plan --config configs/sam3_sensaturban.pipeline.json --split ALL
```

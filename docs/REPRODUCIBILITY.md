# Reproducibility

## Code-Level Reproducibility

The repository contains machine-readable pipeline configs and a small CLI:

```text
configs/cityrefer.pipeline.json
configs/cityanchor.pipeline.json
configs/sam3_sensaturban.pipeline.json
city_pipeline/
scripts/run_pipeline.py
```

Use these commands to verify the code-level contract:

```bash
python scripts/check_project.py
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
python scripts/run_pipeline.py plan --dataset cityanchor --split ND --from-stage stage1_candidates --to-stage hypergraph_match_v7
```

## Data-Level Reproducibility

Full from-zero runs require external assets:

- CityRefer and CityAnchor metadata and bbox JSON roots.
- SensatUrban or STPLS3D point clouds.
- CityAnchor instance-segmentation outputs can be converted to bbox JSON with
  `scripts/instances_to_bbox.py`.
- SAM3 model package and weights.
- Rendered context images for VLM/rerank stages.
- API keys for regenerated LLM extraction or DashScope reranking.

Keep those assets under `data/raw/` or pass equivalent paths through pipeline
config edits.

## Remote Server Notes

The zhangshuai server audit is recorded in `docs/REMOTE_SERVER_AUDIT.md`.
CityRefer/SAM3 assets are mostly available there. CityAnchor is only partially
mapped there and should be treated as requiring local raw asset preparation
unless the missing original final workspace is found.

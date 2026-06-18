# Complete Pipeline

This repository now exposes three configured pipelines:

- `configs/sam3_sensaturban.pipeline.json`: raw SensatUrban / SAM3 upstream.
- `configs/cityrefer.pipeline.json`: CityRefer downstream retrieval and
  hypergraph matching.
- `configs/cityanchor.pipeline.json`: CityAnchor downstream retrieval,
  geometry-aware matching, and reranking.

## Shared SAM3 Upstream

```text
raw SensatUrban point cloud
  -> 50m grid tiles
  -> NAG / superpoints
  -> rendered BEV and tile views
  -> SAM3 masks
  -> semantic fusion PLY
  -> merged instance PLY
  -> CityRefer-style bbox JSON
```

See `docs/SAM3_UPSTREAM.md`.

Remote availability and path caveats are summarized in
`docs/REMOTE_SERVER_AUDIT.md`.

## CityRefer

```text
metadata + bbox + color tiers
  -> semantic / landmark / color stage1 candidates
  -> evaluation log
  -> description hypergraphs
  -> candidate bbox hypergraphs
  -> hypergraph matching
  -> optional VLM / DashScope rerank
```

See `docs/CITYREFER_PIPELINE.md`.

Quick bundled replay:

```bash
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
python scripts/run_pipeline.py run --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```

## CityAnchor

```text
metadata + bbox + color tiers + query color map
  -> color-aware stage1 candidates
  -> description hypergraphs
  -> bbox-only candidate hypergraphs
  -> geometry mention extraction
  -> v7 geometry-aware hypergraph matching
  -> optional VLM / DashScope rerank
```

See `docs/CITYANCHOR_PIPELINE.md`.

## External Assets

From-scratch runs still require large, non-committed assets under `data/raw/`:

- CityRefer / CityAnchor bbox JSON roots.
- LLM NER outputs used to build description hypergraphs.
- SAM3 point clouds, masks, instance PLY files, and checkpoints.
- Context images for VLM/rerank stages.

Use `python scripts/run_pipeline.py doctor ...` to see exactly which assets are
missing for a selected dataset, split, and stage.

Remote audit summary: CityRefer/SAM3 assets were found on the zhangshuai server,
but the `Stage1_code_result/data` symlink points to an old location. Use
`/mnt/shuaizhang_data/CityRefer_1/data`. CityAnchor is only partially present
remotely: STPLS3D raw/txt data and a small ND metadata file exist, while the
full final CityAnchor generated artifacts and v7 matcher workspace were not
found under the audited paths.

# Pipeline Overview

The project is organized around three pipeline configs:

- `configs/sam3_sensaturban.pipeline.json`: shared upstream segmentation and
  bbox generation.
- `configs/cityrefer.pipeline.json`: CityRefer downstream retrieval and
  hypergraph matching.
- `configs/cityanchor.pipeline.json`: CityAnchor downstream retrieval,
  geometry-aware matching, and reranking.

The config files are declarative. Each stage records:

- `script`: source file to run.
- `inputs`: required input files or directories.
- `command`: exact command rendered by `city-pipeline plan`.
- `optional`: whether missing inputs/env vars should be reported without
  failing `doctor`.

Use this command to inspect any pipeline:

```bash
python scripts/run_pipeline.py plan --dataset cityrefer --split ND
```

## Shared Upstream

The upstream SAM3 path is:

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

The output bbox JSON can be consumed by CityRefer and, after dataset alignment,
by CityAnchor-style bbox-hypergraph builders.

## Dataset Downstream

CityRefer emphasizes semantic/landmark stage1 filtering and hypergraph matching
against description and bbox hypergraphs. The current bundle includes enough
precomputed artifacts to replay bundled hypergraph matching without raw bbox
data.

CityAnchor adds stronger main-object color handling, geometry mention
extraction, and a final v7 matcher that fuses stage1 rank, hypergraph score, and
geometry tie-breaking.

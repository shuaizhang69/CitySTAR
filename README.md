# CityRefer / CityAnchor Pipeline

This repository packages the CityRefer and CityAnchor experiment code into a
reproducible project layout. It keeps the original research scripts under
`cityrefer/` and `cityanchor/`, adds a small orchestration CLI, and mirrors the
lightweight SAM3/SensatUrban upstream wrappers under `external/sam3_segment/`.

Large assets are intentionally not committed: raw point clouds, bbox roots,
rendered images, SAM3 masks, instance PLY files, checkpoints, and private API
keys must be provided outside the repository.

## Current Status

- Code-level pipeline is closed: CityRefer, CityAnchor, and SAM3 stages are
  represented by config files and runnable through one CLI.
- CityRefer bundled replay is self-contained and is used as the smoke test.
- Full data-level runs still require external raw assets under `data/raw/`.
- Remote server findings are documented in `docs/REMOTE_SERVER_AUDIT.md`.

## What Is Included

- `cityrefer/`: CityRefer candidate generation, hypergraph construction,
  matching, and reranking scripts.
- `cityanchor/`: CityAnchor color-aware candidate generation, bbox/description
  hypergraph logic, geometry matching, and reranking scripts.
- `external/sam3_segment/`: copied lightweight upstream wrappers from
  `/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result`.
- `configs/*.pipeline.json`: machine-readable pipeline definitions.
- `city_pipeline/`: CLI used to inspect, plan, and run configured stages.
- `docs/`: pipeline, artifact, and release documentation.
- `tests/`: lightweight contract tests for pipeline wiring.
- `.github/workflows/ci.yml`: CI smoke checks for open-source publishing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Optional dependencies:

- DashScope rerank: `pip install dashscope`
- SAM3 / point cloud stages: install the SAM3 environment, PyTorch, CUDA,
  `plyfile`, and any renderer/Open3D dependencies required by your machine.

## Environment

Do not hardcode keys in code. Copy `.env.example` for your own environment and
export the variables in your shell:

```bash
export DEEPSEEK_API_KEY=...
export DASHSCOPE_API_KEY=...
export SGLANG_BASE_URL=http://127.0.0.1:8001/v1
```

If you install the CLI outside an editable source checkout, set:

```bash
export CITY_PIPELINE_ROOT=/path/to/CityRefer_CityAnchor_code_bundle
```

## Data Layout

Put non-committed raw data under `data/raw/`:

```text
data/raw/
  cityrefer/
    bbox/
    desc/
    context_images/
  cityanchor/
    bbox/
    desc/
    context_images/
  sensaturban/
    point_clouds/
    instances/
```

Generated outputs go under `outputs/` and are ignored by git.

## CLI

List available pipeline configs:

```bash
python scripts/run_pipeline.py list
```

List stages for one dataset:

```bash
python scripts/run_pipeline.py list --dataset cityrefer
python scripts/run_pipeline.py list --dataset cityanchor
```

Check whether scripts, inputs, and required environment variables are present.
The full configs will report missing external raw data until you populate
`data/raw/`:

```bash
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND
python scripts/run_pipeline.py doctor --dataset cityanchor --split NO
```

Check the bundled CityRefer replay stage only:

```bash
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```

Print commands without running:

```bash
python scripts/run_pipeline.py plan --dataset cityrefer --split ND
python scripts/run_pipeline.py plan --dataset cityanchor --split NO
```

Run selected stages:

```bash
python scripts/run_pipeline.py run --dataset cityrefer --split ND --stage bundle_replay_hgmatch
python scripts/run_pipeline.py run --dataset cityanchor --split ND --from-stage stage1_candidates --to-stage hypergraph_match_v7
```

Generate CityAnchor bbox JSON from instance-segmentation point outputs:

```bash
python scripts/instances_to_bbox.py \
  --input-root data/raw/cityanchor/instances \
  --output-dir data/raw/cityanchor/bbox \
  --recursive \
  --xyz-cols 0,1,2 \
  --instance-col -1
```

Installed console script equivalent:

```bash
city-pipeline list
city-pipeline doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```

## Pipeline Summary

Shared upstream segmentation path:

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

CityRefer downstream path:

```text
metadata + bbox + color features
  -> stage1 candidates
  -> evaluation log
  -> description hypergraphs
  -> bbox hypergraphs
  -> hypergraph matching
  -> optional VLM / DashScope rerank
```

CityAnchor downstream path:

```text
metadata + instance outputs
  -> bbox JSON
  -> color features + query color map
  -> color-aware stage1 candidates
  -> description hypergraphs
  -> bbox-only hypergraphs
  -> geometry mentions
  -> v7 geometry-aware hypergraph matching
  -> optional VLM / DashScope rerank
```

See `docs/` for details.

## Verification

```bash
python scripts/check_project.py
python scripts/verify_bundle.py
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
python -m py_compile city_pipeline/config.py city_pipeline/runner.py city_pipeline/cli.py scripts/run_pipeline.py scripts/verify_bundle.py
```

`verify_bundle.py` fails if a Python bytecode cache or a token matching
`sk-*` is present in tracked source-like files.

## Documentation Map

- `docs/QUICKSTART.md`: minimal local verification.
- `COMPLETE_PIPELINE.md`: end-to-end pipeline summary.
- `docs/PIPELINE_OVERVIEW.md`: stage-level overview.
- `docs/CITYREFER_PIPELINE.md`: CityRefer details.
- `docs/CITYANCHOR_PIPELINE.md`: CityAnchor details.
- `docs/SAM3_UPSTREAM.md`: SAM3/SensatUrban upstream details.
- `docs/DATA_LAYOUT.md`: expected raw/generated asset layout.
- `docs/REMOTE_SERVER_AUDIT.md`: zhangshuai server audit.
- `docs/REPRODUCIBILITY.md`: code-level and data-level reproducibility notes.

## License

Project orchestration code and documentation are released under the MIT
License. See `THIRD_PARTY_NOTICES.md` for dataset, checkpoint, and legacy-script
provenance notes.

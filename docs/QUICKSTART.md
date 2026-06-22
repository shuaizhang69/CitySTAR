# Quickstart

This quickstart verifies the code-level pipeline without requiring large raw
point-cloud assets.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## 2. Verify The Bundle

```bash
python scripts/check_project.py
```

This runs JSON validation, unit tests, CLI listing, a CityRefer bundled replay
doctor check, cache cleanup, and secret/cache verification.

## 3. Inspect Pipelines

```bash
python scripts/run_pipeline.py list
python scripts/run_pipeline.py list --dataset cityrefer
python scripts/run_pipeline.py list --dataset cityanchor
python scripts/run_pipeline.py list --dataset sam3_sensaturban
```

## 4. Run The Self-Contained Replay

```bash
python scripts/run_pipeline.py run --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```

This stage uses bundled JSON/JSONL artifacts and writes
`outputs/cityrefer/ND/bundle_hgmatch.json`.

## 5. Prepare Full Runs

Populate `data/raw/` before running full CityRefer, CityAnchor, or SAM3 stages.
See `docs/DATA_LAYOUT.md`, `docs/CITYREFER_PIPELINE.md`,
`docs/CITYANCHOR_PIPELINE.md`, and `docs/SAM3_UPSTREAM.md`.

For CityAnchor, bbox JSON can be generated directly from instance-segmentation
outputs:

```bash
python scripts/instances_to_bbox.py \
  --input-root data/raw/cityanchor/instances \
  --output-dir data/raw/cityanchor/bbox \
  --recursive
```

# Contributing

## Development Setup

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

## Checks

Run the project check before submitting changes:

```bash
python scripts/check_project.py
```

For targeted checks:

```bash
python -m unittest discover -s tests
python scripts/verify_bundle.py
python scripts/run_pipeline.py doctor --dataset cityrefer --split ND --stage bundle_replay_hgmatch
```

## Rules For Changes

- Keep raw data, generated outputs, checkpoints, masks, and point clouds out of
  git.
- Keep API keys in environment variables. Do not commit `.env`.
- Prefer adding pipeline stages through `configs/*.pipeline.json` and the
  shared CLI instead of adding ad hoc shell instructions.
- Keep legacy research scripts intact unless a change is required to make a
  configured stage runnable.
- Document any new external asset requirements in `docs/DATA_LAYOUT.md`.

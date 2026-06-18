# Open Source Checklist

Before publishing:

- Keep `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CONTRIBUTING.md`, and
  `SECURITY.md` in the release bundle.
- Replace personal/HPC absolute paths in legacy scripts when they become active
  entry points. The new configs avoid relying on those defaults.
- Keep all API keys in environment variables.
- Keep raw data and generated outputs outside git.
- Run `python scripts/check_project.py`.
- Run `python scripts/run_pipeline.py doctor --dataset cityrefer --split ND`.
- Run `python scripts/run_pipeline.py doctor --dataset cityanchor --split ND`.
- Document exact external dataset download/preparation steps.
- Document SAM3 checkpoint sources and compatible CUDA/PyTorch versions.
- Keep CI green with `python scripts/check_project.py`.

from __future__ import annotations

import shutil
import subprocess
import sys
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=str(ROOT), check=True)


def clean_bytecode() -> None:
    for path in ROOT.rglob("__pycache__"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in ROOT.rglob("*.pyc"):
        if path.is_file():
            path.unlink()


def validate_json(relative_path: str) -> None:
    path = ROOT / relative_path
    print(f"+ validate-json {relative_path}", flush=True)
    with path.open("r", encoding="utf-8") as f:
        json.load(f)


def main() -> int:
    py = sys.executable
    clean_bytecode()
    validate_json("configs/cityrefer.pipeline.json")
    validate_json("configs/cityanchor.pipeline.json")
    validate_json("configs/sam3_sensaturban.pipeline.json")
    run([py, "-m", "unittest", "discover", "-s", "tests"])
    run([py, "scripts/run_pipeline.py", "list"])
    run(
        [
            py,
            "scripts/run_pipeline.py",
            "doctor",
            "--dataset",
            "cityrefer",
            "--split",
            "ND",
            "--stage",
            "bundle_replay_hgmatch",
        ]
    )
    clean_bytecode()
    run([py, "scripts/verify_bundle.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

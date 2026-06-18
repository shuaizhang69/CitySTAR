from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAM3_CORE = ROOT / "external" / "sam3_segment" / "core"
if str(SAM3_CORE) not in sys.path:
    sys.path.insert(0, str(SAM3_CORE))

from preprocessor import split_ply_grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Split raw PLY files into 50m SensatUrban tiles.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--grid-size", type=float, default=50.0)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    args = parser.parse_args()

    split_ply_grid(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        grid_size=args.grid_size,
        voxel_size=args.voxel_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

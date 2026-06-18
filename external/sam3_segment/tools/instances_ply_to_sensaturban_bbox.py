#!/usr/bin/env python3
"""Build per-scene bbox JSON from SensatUrban *_instances.ply (vertex instance_id)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sam3_segment.core.grid_50m_metadata import (
    DEFAULT_GRID_METADATA_PATH,
    load_grid_metadata_offsets,
    prepare_tile_xy_offsets,
    resolve_tile_origin_xy,
)

# Filename token -> (object_name, SensatUrban semantic class id as in Cityrefer bbox index 6)
_CATEGORY_TO_NAME_AND_CLASS: dict[str, tuple[str, float]] = {
    "ground": ("Ground", 0.0),
    "vegetation": ("HighVegetation", 1.0),
    "building": ("Building", 2.0),
    "wall": ("Wall", 3.0),
    "bridge": ("Bridge", 4.0),
    "parking": ("Parking", 5.0),
    "rail": ("Rail", 6.0),
    "car": ("Car", 9.0),
    "footpath": ("Footpath", 10.0),
    "bike": ("Bike", 11.0),
    "water": ("Water", 12.0),
    # fusion_by_class 等分块输出中的额外类别（class 编号仅用于 JSON 第 7 维，与 CityRefer 对齐时可再改）
    "street_furniture": ("StreetFurniture", 13.0),
    "traffic_road": ("TrafficRoad", 14.0),
}

_DT = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("iid", "i4")])


def _read_ply_vertices(path: Path) -> tuple[int, np.ndarray]:
    with path.open("rb") as f:
        header_lines: list[bytes] = []
        while True:
            line = f.readline()
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        header = b"".join(header_lines).decode("ascii", errors="replace")
        n_vertices = 0
        for hl in header.splitlines():
            if hl.startswith("element vertex"):
                n_vertices = int(hl.split()[-1])
        offset = f.tell()
    # memmap avoids loading whole cloud into RAM (large PLYs).
    data = np.memmap(path, dtype=_DT, mode="r", offset=offset, shape=(n_vertices,))
    return n_vertices, data


def _aabb_rows(xyz: np.ndarray, iid: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return list of (instance_id, center(3), size(3)) for iid != 0 and valid coords."""
    m = (iid != 0) & np.all(np.isfinite(xyz), axis=1)
    if not np.any(m):
        return []
    xyz = np.asarray(xyz[m], dtype=np.float64)
    iid = np.asarray(iid[m], dtype=np.int32)
    order = np.argsort(iid, kind="mergesort")
    xyz = xyz[order]
    iid = iid[order]
    uniq, idx0 = np.unique(iid, return_index=True)
    n = len(xyz)
    idx1 = np.append(idx0[1:], n)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for u, a, b in zip(uniq, idx0, idx1):
        sl = xyz[a:b]
        lo = sl.min(axis=0)
        hi = sl.max(axis=0)
        center = (lo + hi) * 0.5
        size = hi - lo
        out.append((int(u), center, size))
    return out


def _parse_scene_category(name: str) -> tuple[str, str] | None:
    """解析 {scene}_{cat}_instances.ply 或 {scene}_xNxNyM_{cat}_instances.ply（分块融合目录）。"""
    m = re.match(
        r"^(?P<scene>birmingham_block_\d+|cambridge_block_\d+)_(?:x\d+_y\d+_)?(?P<cat>.+)_instances\.ply$",
        name,
    )
    if not m:
        return None
    return m.group("scene"), m.group("cat")


def build_scene(
    scene_id: str,
    ply_paths: list[Path],
    tile_xy_offsets: Optional[dict[str, tuple[float, float]]] = None,
) -> dict[str, Any]:
    records: list[
        tuple[float, float, float, float, float, float, str, float, int, str, str]
    ] = []
    # sort for deterministic global ordering: class id, name, ply path, instance id
    for p in sorted(ply_paths, key=lambda x: x.name):
        parsed = _parse_scene_category(p.name)
        if not parsed:
            continue
        sc, cat = parsed
        if sc != scene_id:
            continue
        if cat not in _CATEGORY_TO_NAME_AND_CLASS:
            raise KeyError(f"Unknown category {cat!r} in {p}")
        oname, class_f = _CATEGORY_TO_NAME_AND_CLASS[cat]
        _, data = _read_ply_vertices(p)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(
            np.float64, copy=True
        )
        ox, oy = resolve_tile_origin_xy(p.name, tile_xy_offsets)
        xyz[:, 0] += ox
        xyz[:, 1] += oy
        iid = data["iid"]
        for inst_id, center, size in _aabb_rows(xyz, iid):
            records.append(
                (
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                    oname,
                    class_f,
                    inst_id,
                    cat,
                    str(p),
                )
            )

    records.sort(
        key=lambda r: (r[7], r[6], r[0], r[1], r[2], r[8], r[9], r[10]),
    )
    bboxes: list[dict[str, Any]] = []
    for oid, r in enumerate(records):
        cx, cy, cz, dx, dy, dz, oname, class_f, _iid, _cat, _path = r
        bboxes.append(
            {
                "object_id": oid,
                "object_name": oname,
                "landmark": "",
                "bbox": [cx, cy, cz, dx, dy, dz, class_f, float(oid)],
            }
        )
    return {"scene_id": scene_id, "bboxes": bboxes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/instances_merged_all"
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/zhangshuai/workshop/open_vocabulary/LqhSpace/Stage1_code_result/sensaturban_bbox"),
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录下所有 *_instances.ply（用于 fusion_by_class_3dqa 等 scene/xNN_yMM/ 结构）",
    )
    ap.add_argument(
        "--grid_metadata",
        type=str,
        default=None,
        help=(
            "grid_50m_3dqa/metadata.json；默认在存在时使用 "
            f"{DEFAULT_GRID_METADATA_PATH}"
        ),
    )
    ap.add_argument(
        "--no_grid_metadata",
        action="store_true",
        help="不使用 metadata，仅用文件名 _xNNN_yMMM_",
    )
    args = ap.parse_args()
    ind: Path = args.input_dir
    outd: Path = args.output_dir
    outd.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[Path]] = {}
    ply_iter = sorted(ind.rglob("*_instances.ply")) if args.recursive else sorted(ind.glob("*.ply"))
    for p in ply_iter:
        parsed = _parse_scene_category(p.name)
        if not parsed:
            continue
        scene_id, _cat = parsed
        by_scene.setdefault(scene_id, []).append(p)

    prep = prepare_tile_xy_offsets(
        disabled=args.no_grid_metadata,
        metadata_path=args.grid_metadata,
    )
    if prep is None:
        prep = load_grid_metadata_offsets(None) or {}
    for scene_id in sorted(by_scene.keys()):
        data = build_scene(scene_id, by_scene[scene_id], tile_xy_offsets=prep)
        outp = outd / f"{scene_id}_bbox.json"
        with outp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"wrote {outp} ({len(data['bboxes'])} objects)")


if __name__ == "__main__":
    main()

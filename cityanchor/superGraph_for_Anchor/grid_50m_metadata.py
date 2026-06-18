from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple


_TILE_XY_RE = re.compile(r"_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)_")
_DEFAULT_METADATA_CANDIDATES = (
    "/hpc2hdd/home/yxiao224/Henry/dataset/grid_50m_3dqa/metadata.json",
    "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/grid_50m_3dqa/metadata.json",
)


def _normalize_offsets(data: object) -> Dict[str, Tuple[float, float]]:
    offsets: Dict[str, Tuple[float, float]] = {}
    if isinstance(data, dict):
        iterable = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        return offsets

    for key, value in iterable:
        if isinstance(value, dict):
            x = value.get("offset_x", value.get("x", value.get("origin_x")))
            y = value.get("offset_y", value.get("y", value.get("origin_y")))
            tile_key = (
                value.get("tile_name")
                or value.get("tile_id")
                or value.get("tile")
                or value.get("filename")
                or str(key)
            )
            if x is not None and y is not None:
                offsets[str(tile_key)] = (float(x), float(y))
    return offsets


def load_grid_metadata_offsets(metadata_path: Optional[str]) -> Optional[Dict[str, Tuple[float, float]]]:
    candidates = [metadata_path] if metadata_path else list(_DEFAULT_METADATA_CANDIDATES)
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            return _normalize_offsets(json.load(f))
    return None


def resolve_tile_origin_xy(
    filename: str,
    tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    if tile_xy_offsets:
        base = os.path.basename(filename)
        for key in (base, os.path.splitext(base)[0]):
            if key in tile_xy_offsets:
                return tile_xy_offsets[key]

    match = _TILE_XY_RE.search(os.path.basename(filename))
    if match:
        return float(match.group("x")), float(match.group("y"))
    return 0.0, 0.0

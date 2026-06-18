"""
grid_50m_3dqa 切块 metadata.json：按瓦片 PLY 键查询世界坐标 offset（比从 instances 文件名猜 x/y 更准确）。
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

# 与 sensatUrban 切块脚本输出一致；若不存在则回退为仅按文件名解析 x/y。
DEFAULT_GRID_METADATA_PATH = (
    "/home/zhangshuai/workshop/open_vocabulary/LqhSpace/sam3/sensatUrban/"
    "grid_50m_3dqa/metadata.json"
)

# birmingham_block_12_x1200_y800_building_instances.ply -> birmingham_block_12_x1200_y800.ply
_INSTANCES_BASENAME_TO_TILE_KEY = re.compile(
    r"^(.+_x\d+_y\d+)_(.+)_instances\.ply$"
)


def load_tile_xy_offsets(metadata_path: str) -> Dict[str, Tuple[float, float]]:
    """读取 metadata.json，返回 { 'scene_xNN_yMM.ply': (ox, oy), ... }。"""
    with open(metadata_path, encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in raw.items():
        off = v.get("offset")
        if isinstance(off, (list, tuple)) and len(off) >= 2:
            out[str(k)] = (float(off[0]), float(off[1]))
    return out


def load_grid_metadata_offsets(metadata_path: Optional[str] = None) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    metadata_path:
      None — 若 DEFAULT_GRID_METADATA_PATH 存在则加载，否则返回 None；
      "" — 显式禁用，返回 None；
      其他 — 按给定路径加载（文件不存在则返回 None）。
    """
    if metadata_path == "":
        return None
    path = metadata_path or DEFAULT_GRID_METADATA_PATH
    if not path or not os.path.isfile(path):
        return None
    return load_tile_xy_offsets(path)


def instances_ply_basename_to_tile_metadata_key(basename: str) -> Optional[str]:
    """由 *_{category}_instances.ply  basename 得到 metadata 中的键（*.ply）。"""
    m = _INSTANCES_BASENAME_TO_TILE_KEY.match(basename)
    if not m:
        return None
    return m.group(1) + ".ply"


def resolve_tile_origin_xy(
    filename: str,
    tile_xy_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    优先用 metadata 中该瓦片的 offset[0:2]；若无表项则回退为文件名中的 _xNNN_yMMM_。
    """
    if tile_xy_offsets:
        key = instances_ply_basename_to_tile_metadata_key(filename)
        if key is not None and key in tile_xy_offsets:
            return tile_xy_offsets[key]
    m = re.search(r"_x(\d+)_y(\d+)_", filename)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


def prepare_tile_xy_offsets(
    *,
    disabled: bool = False,
    metadata_path: Optional[str] = None,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    供 CLI / process_and_save 一次解析。

    返回:
      None — 在 recursive 加载/写 PLY 时由调用方自动加载默认 metadata.json（若存在）；
      {} — 禁用 metadata，仅用文件名解析 x/y；
      非空 dict — 使用该偏移表。
    """
    if disabled:
        return {}
    if metadata_path is not None:
        if not os.path.isfile(metadata_path):
            print(
                f"Warning: grid metadata 文件不存在，将仅用文件名 x/y: {metadata_path!r}"
            )
            return {}
        return load_tile_xy_offsets(metadata_path)
    return None

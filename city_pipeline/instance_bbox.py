from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


CATEGORY_ALIASES: dict[str, tuple[str, float]] = {
    "ground": ("Ground", 0.0),
    "terrain": ("Ground", 0.0),
    "building": ("Building", 1.0),
    "buildings": ("Building", 1.0),
    "tree": ("Tree", 2.0),
    "trees": ("Tree", 2.0),
    "vegetation": ("Tree", 2.0),
    "highvegetation": ("Tree", 2.0),
    "vehicle": ("Vehicle", 3.0),
    "vehicles": ("Vehicle", 3.0),
    "car": ("Vehicle", 3.0),
    "cars": ("Vehicle", 3.0),
    "truck": ("Truck", 4.0),
    "trucks": ("Truck", 4.0),
    "fence": ("Fence", 5.0),
    "fences": ("Fence", 5.0),
    "lightpole": ("LightPole", 6.0),
    "light_pole": ("LightPole", 6.0),
    "pole": ("LightPole", 6.0),
    "parking": ("Parking", 7.0),
    "road": ("Road", 8.0),
    "street": ("Road", 8.0),
    "footpath": ("Footpath", 9.0),
    "sidewalk": ("Footpath", 9.0),
    "water": ("Water", 10.0),
}

DEFAULT_CATEGORY_SUFFIXES = sorted(CATEGORY_ALIASES.keys(), key=len, reverse=True)


@dataclass
class BBoxAccumulator:
    scene_id: str
    object_name: str
    class_id: float
    source_instance_id: str
    source_file: str
    count: int = 0
    min_xyz: np.ndarray | None = None
    max_xyz: np.ndarray | None = None

    def update(self, xyz: np.ndarray) -> None:
        if not np.all(np.isfinite(xyz)):
            return
        xyz = xyz.astype(float, copy=False)
        if self.count == 0:
            self.min_xyz = xyz.copy()
            self.max_xyz = xyz.copy()
        else:
            assert self.min_xyz is not None and self.max_xyz is not None
            self.min_xyz = np.minimum(self.min_xyz, xyz)
            self.max_xyz = np.maximum(self.max_xyz, xyz)
        self.count += 1

    def bbox_center_size(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0 or self.min_xyz is None or self.max_xyz is None:
            raise ValueError("empty accumulator")
        center = (self.min_xyz + self.max_xyz) * 0.5
        size = self.max_xyz - self.min_xyz
        return center, size


def parse_col_index(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "-"}:
        return None
    return int(text)


def parse_xyz_cols(text: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--xyz-cols must contain exactly three comma-separated indexes")
    return parts[0], parts[1], parts[2]


def normalize_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def normalize_category_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def resolve_category(category: str | None, default_object_name: str = "Object") -> tuple[str, float]:
    if category:
        key = normalize_category_token(category)
        if key in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[key]
        return category.strip(), -1.0
    return default_object_name, -1.0


def infer_scene_category(path: Path) -> tuple[str, str | None]:
    stem = path.stem
    if stem.endswith("_instances"):
        stem = stem[: -len("_instances")]

    for suffix in DEFAULT_CATEGORY_SUFFIXES:
        token = f"_{suffix}"
        if stem.lower().endswith(token):
            return stem[: -len(token)], suffix
    return stem, None


def iter_input_files(input_roots: Iterable[Path], recursive: bool) -> Iterator[Path]:
    patterns = ("*.txt", "*.csv", "*.npy", "*.ply")
    for root in input_roots:
        if root.is_file():
            yield root
            continue
        for pattern in patterns:
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            yield from sorted(iterator)


def split_text_line(line: str, delimiter: str | None) -> list[str]:
    line = line.strip().lstrip("\ufeff")
    if not line or line.startswith("#"):
        return []
    if delimiter:
        return [part.strip() for part in line.split(delimiter)]
    if "," in line:
        return next(csv.reader([line]))
    return line.split()


def value_at(parts: list[str], col: int | None) -> str | None:
    if col is None:
        return None
    idx = col if col >= 0 else len(parts) + col
    if idx < 0 or idx >= len(parts):
        return None
    return parts[idx]


def update_acc(
    accs: dict[tuple[str, str, str], BBoxAccumulator],
    scene_id: str,
    object_name: str,
    class_id: float,
    instance_id: str,
    source_file: Path,
    xyz: np.ndarray,
) -> None:
    key = (scene_id, object_name, instance_id)
    if key not in accs:
        accs[key] = BBoxAccumulator(
            scene_id=scene_id,
            object_name=object_name,
            class_id=class_id,
            source_instance_id=instance_id,
            source_file=str(source_file),
        )
    accs[key].update(xyz)


def ingest_text_file(
    path: Path,
    accs: dict[tuple[str, str, str], BBoxAccumulator],
    *,
    xyz_cols: tuple[int, int, int],
    instance_col: int | None,
    semantic_col: int | None,
    delimiter: str | None,
    category_override: str | None,
    default_object_name: str,
    min_points: int,
) -> None:
    scene_id, filename_category = infer_scene_category(path)
    category = category_override or filename_category
    fallback_object_name, fallback_class_id = resolve_category(category, default_object_name)
    fallback_instance_id = path.stem

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = split_text_line(line, delimiter)
            if not parts:
                continue
            try:
                xyz = np.array([float(value_at(parts, col)) for col in xyz_cols], dtype=float)
            except (TypeError, ValueError):
                continue

            instance_value = value_at(parts, instance_col)
            instance_id = normalize_token(instance_value) if instance_value is not None else fallback_instance_id
            semantic_value = value_at(parts, semantic_col)
            if semantic_value is not None and category_override is None and filename_category is None:
                object_name, class_id = resolve_category(normalize_token(semantic_value), default_object_name)
            else:
                object_name, class_id = fallback_object_name, fallback_class_id
            update_acc(accs, scene_id, object_name, class_id, instance_id, path, xyz)


def ingest_npy_file(
    path: Path,
    accs: dict[tuple[str, str, str], BBoxAccumulator],
    *,
    xyz_cols: tuple[int, int, int],
    instance_col: int | None,
    semantic_col: int | None,
    category_override: str | None,
    default_object_name: str,
    min_points: int,
) -> None:
    scene_id, filename_category = infer_scene_category(path)
    category = category_override or filename_category
    fallback_object_name, fallback_class_id = resolve_category(category, default_object_name)
    fallback_instance_id = path.stem
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D array")

    for row in arr:
        xyz = np.array([float(row[col]) for col in xyz_cols], dtype=float)
        instance_id = normalize_token(row[instance_col]) if instance_col is not None else fallback_instance_id
        if semantic_col is not None and category_override is None and filename_category is None:
            object_name, class_id = resolve_category(normalize_token(row[semantic_col]), default_object_name)
        else:
            object_name, class_id = fallback_object_name, fallback_class_id
        update_acc(accs, scene_id, object_name, class_id, instance_id, path, xyz)


def read_ply_with_plyfile(path: Path) -> dict[str, np.ndarray] | None:
    try:
        from plyfile import PlyData  # type: ignore
    except ImportError:
        return None
    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    return {name: np.asarray(vertex[name]) for name in vertex.dtype.names or []}


def read_ascii_ply(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header: list[str] = []
        for line in f:
            header.append(line.rstrip("\n"))
            if line.strip() == "end_header":
                break
        if not header or header[0].strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")
        if not any(line.strip() == "format ascii 1.0" for line in header):
            raise ValueError(f"{path} is not ASCII PLY and plyfile is not installed")

        properties: list[str] = []
        in_vertex = False
        vertex_count = 0
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                in_vertex = True
                vertex_count = int(parts[2])
                continue
            if len(parts) >= 2 and parts[:2] == ["element", "face"]:
                in_vertex = False
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        columns = {name: [] for name in properties}
        for _ in range(vertex_count):
            parts = f.readline().split()
            if len(parts) < len(properties):
                continue
            for name, value in zip(properties, parts):
                columns[name].append(float(value))
        return {name: np.asarray(values) for name, values in columns.items()}


def pick_ply_column(columns: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray | None:
    lower = {key.lower(): key for key in columns.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key:
            return columns[key]
    return None


def ingest_ply_file(
    path: Path,
    accs: dict[tuple[str, str, str], BBoxAccumulator],
    *,
    category_override: str | None,
    default_object_name: str,
    min_points: int,
) -> None:
    scene_id, filename_category = infer_scene_category(path)
    category = category_override or filename_category
    fallback_object_name, fallback_class_id = resolve_category(category, default_object_name)
    columns = read_ply_with_plyfile(path) or read_ascii_ply(path)

    x = pick_ply_column(columns, ("x",))
    y = pick_ply_column(columns, ("y",))
    z = pick_ply_column(columns, ("z",))
    iid = pick_ply_column(columns, ("instance_id", "instance", "iid", "label", "object_id"))
    semantic = pick_ply_column(columns, ("semantic_id", "semantic", "class_id", "category"))
    if x is None or y is None or z is None:
        raise ValueError(f"{path} must contain x/y/z vertex properties")
    if iid is None:
        iid = np.zeros_like(x)

    for idx in range(len(x)):
        xyz = np.array([float(x[idx]), float(y[idx]), float(z[idx])], dtype=float)
        instance_id = normalize_token(iid[idx])
        if semantic is not None and category_override is None and filename_category is None:
            object_name, class_id = resolve_category(normalize_token(semantic[idx]), default_object_name)
        else:
            object_name, class_id = fallback_object_name, fallback_class_id
        update_acc(accs, scene_id, object_name, class_id, instance_id, path, xyz)


def remove_small_instances(accs: dict[tuple[str, str, str], BBoxAccumulator], min_points: int) -> None:
    if min_points <= 1:
        return
    for key in [key for key, acc in accs.items() if acc.count < min_points]:
        del accs[key]


def numeric_sort_key(text: str) -> tuple[int, float | str]:
    try:
        return 0, float(text)
    except ValueError:
        return 1, text


def build_scene_json(
    scene_id: str,
    accs: list[BBoxAccumulator],
    *,
    object_id_mode: str,
    duplicate_id_policy: str,
) -> dict[str, Any]:
    rows = sorted(
        accs,
        key=lambda acc: (
            acc.object_name.lower(),
            numeric_sort_key(acc.source_instance_id),
            acc.source_file,
        ),
    )

    used_ids: set[int] = set()
    bboxes: list[dict[str, Any]] = []
    next_id = 0
    for acc in rows:
        center, size = acc.bbox_center_size()
        if object_id_mode == "preserve":
            try:
                object_id = int(float(acc.source_instance_id))
            except ValueError as exc:
                raise ValueError(
                    f"Cannot preserve non-numeric instance id {acc.source_instance_id!r}; "
                    "use --object-id-mode enumerate"
                ) from exc
            if object_id in used_ids:
                if duplicate_id_policy == "error":
                    raise ValueError(f"duplicate object_id {object_id} in scene {scene_id}")
                if duplicate_id_policy == "enumerate":
                    while next_id in used_ids:
                        next_id += 1
                    object_id = next_id
                elif duplicate_id_policy == "keep":
                    pass
        else:
            while next_id in used_ids:
                next_id += 1
            object_id = next_id

        used_ids.add(object_id)
        bboxes.append(
            {
                "object_id": object_id,
                "object_name": acc.object_name,
                "landmark": "",
                "bbox": [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    float(size[0]),
                    float(size[1]),
                    float(size[2]),
                    float(acc.class_id),
                    float(object_id),
                ],
                "source_instance_id": acc.source_instance_id,
                "num_points": acc.count,
            }
        )
    bboxes.sort(key=lambda item: int(item["object_id"]))
    return {"scene_id": scene_id, "bboxes": bboxes}


def convert_instances_to_bbox(
    *,
    input_roots: list[Path],
    output_dir: Path,
    recursive: bool,
    xyz_cols: tuple[int, int, int],
    instance_col: int | None,
    semantic_col: int | None,
    delimiter: str | None,
    category: str | None,
    default_object_name: str,
    min_points: int,
    object_id_mode: str,
    duplicate_id_policy: str,
) -> list[Path]:
    accs: dict[tuple[str, str, str], BBoxAccumulator] = {}
    files = list(iter_input_files(input_roots, recursive=recursive))
    if not files:
        raise FileNotFoundError("no instance files found")

    for path in files:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".csv"}:
            ingest_text_file(
                path,
                accs,
                xyz_cols=xyz_cols,
                instance_col=instance_col,
                semantic_col=semantic_col,
                delimiter=delimiter,
                category_override=category,
                default_object_name=default_object_name,
                min_points=min_points,
            )
        elif suffix == ".npy":
            ingest_npy_file(
                path,
                accs,
                xyz_cols=xyz_cols,
                instance_col=instance_col,
                semantic_col=semantic_col,
                category_override=category,
                default_object_name=default_object_name,
                min_points=min_points,
            )
        elif suffix == ".ply":
            ingest_ply_file(
                path,
                accs,
                category_override=category,
                default_object_name=default_object_name,
                min_points=min_points,
            )

    by_scene: dict[str, list[BBoxAccumulator]] = {}
    for acc in accs.values():
        if acc.count >= min_points:
            by_scene.setdefault(acc.scene_id, []).append(acc)
    if not by_scene:
        raise ValueError("no valid instances after filtering")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scene_id in sorted(by_scene.keys()):
        data = build_scene_json(
            scene_id,
            by_scene[scene_id],
            object_id_mode=object_id_mode,
            duplicate_id_policy=duplicate_id_policy,
        )
        out_path = output_dir / f"{scene_id}_bbox.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written.append(out_path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert instance-segmentation point outputs into CityRefer/CityAnchor bbox JSON."
    )
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--xyz-cols", default="0,1,2", help="Text/NPY xyz column indexes, default: 0,1,2")
    parser.add_argument("--instance-col", default="-1", help="Text/NPY instance id column, or none")
    parser.add_argument("--semantic-col", default="", help="Optional text/NPY semantic/category column")
    parser.add_argument("--delimiter", default="", help="Optional text delimiter; default auto-detects comma/whitespace")
    parser.add_argument("--category", default="", help="Force one object category for all files")
    parser.add_argument("--default-object-name", default="Object")
    parser.add_argument("--min-points", type=int, default=1)
    parser.add_argument("--object-id-mode", choices=["preserve", "enumerate"], default="preserve")
    parser.add_argument("--duplicate-id-policy", choices=["error", "enumerate", "keep"], default="error")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    written = convert_instances_to_bbox(
        input_roots=args.input_root,
        output_dir=args.output_dir,
        recursive=args.recursive,
        xyz_cols=parse_xyz_cols(args.xyz_cols),
        instance_col=parse_col_index(args.instance_col),
        semantic_col=parse_col_index(args.semantic_col),
        delimiter=args.delimiter or None,
        category=args.category or None,
        default_object_name=args.default_object_name,
        min_points=args.min_points,
        object_id_mode=args.object_id_mode,
        duplicate_id_policy=args.duplicate_id_policy,
    )
    for path in written:
        with path.open("r", encoding="utf-8") as f:
            count = len(json.load(f).get("bboxes", []))
        print(f"wrote {path} ({count} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
CityAnchor：对 knn(for_render) 中出现、且为 Building/Fence、且 XY 底面积 **>=** knn 的
median_remaining_bbox_xy_area_m2 的物体（与 BEV 中「需面积 < median 才渲染」互补，即「不渲 BEV」侧）
从 city_Anchor/single_image 读图，用 CLIP 提特征；**每个场景只写一个 .npy**（numpy 结构化数组，
字段 `object_id` (int64) 与 `feature` (float32, D)，便于按 id 查找）。

读取示例::
    rec = np.load("xxx.npy")
    # rec["object_id"]  shape (N,)  ;  rec["feature"]  shape (N, D)
    oid_to_feat = {int(rec["object_id"][i]): rec["feature"][i].copy() for i in range(len(rec))}

与 bev_render 中 MEDIAN_FILTER_OBJECT_NAMES（Building、Fence）一致。
"""
import os
import sys
import json
import glob

# 必须在 import torch 之前（若需可 export 覆盖）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── 路径（可按需改 argparse）────────────────────────────────────────
MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code"
_DATASET = "/hpc2hdd/home/yxiao224/Henry/dataset"

BBOX_DIR = os.path.join(_DATASET, "city_Anchor", "bbox")
KNN_FOR_RENDER_DIR = os.path.join(_DATASET, "Our_cityG3D", "knn", "for_render")
# 仅从该目录下读图：{IMAGE_SINGLE_ROOT}/{scene_id}/{scene_id}_obj{object_id}.jpg
IMAGE_SINGLE_ROOT = os.path.join(_DATASET, "city_Anchor", "single_image")
OUTPUT_DIR = os.path.join(
    _DATASET, "city_Anchor", "clip_features_above_median"
)

MEDIAN_FILTER_NAMES = frozenset(("Building", "Fence"))


def object_ids_from_knn_clusters(knn_data: dict) -> set[int]:
    ids: set[int] = set()
    for cl in knn_data.get("clusters") or []:
        cid = cl.get("center_object_id")
        if cid is not None:
            ids.add(int(cid))
        for nid in cl.get("neighbor_object_ids") or []:
            ids.add(int(nid))
        for nb in cl.get("neighbors") or []:
            oid = nb.get("object_id")
            if oid is not None:
                ids.add(int(oid))
    return ids


def bbox_xy_area_m2(bbox_raw: list) -> float | None:
    if not bbox_raw or len(bbox_raw) < 5:
        return None
    sx, sy = float(bbox_raw[3]), float(bbox_raw[4])
    return float(abs(sx * sy))


def load_knn_for_scene(knn_dir: str, scene_id: str) -> dict | None:
    safe = str(scene_id).replace("/", "_").replace("\\", "_")
    path = os.path.join(knn_dir, f"{safe}_knn.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_single_image_cityanchor(scene_id: str, object_id) -> str | None:
    """仅 city_Anchor/single_image/{scene_id}/{scene_id}_obj{id}.(jpg|png|jpeg)"""
    sid = str(scene_id)
    oid = str(int(object_id))
    base = os.path.join(IMAGE_SINGLE_ROOT, sid, f"{sid}_obj{oid}")
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        p = base + ext
        if os.path.isfile(p):
            return p
    return None


def filter_above_median_building_fence(
    bboxes: list,
    knn_ids: set[int],
    median_m2: float | None,
) -> list[dict]:
    """Building/Fence 且在 knn 中且 area >= median（BEV 不渲面积过大）。"""
    if median_m2 is None:
        return []
    out: list[dict] = []
    for item in bboxes:
        name = str(item.get("object_name") or item.get("label") or "")
        if name not in MEDIAN_FILTER_NAMES:
            continue
        oid = item.get("object_id")
        if oid is None:
            continue
        try:
            oid_i = int(float(oid))
        except (TypeError, ValueError):
            continue
        if oid_i not in knn_ids:
            continue
        bbox_raw = item.get("bbox") or []
        area = bbox_xy_area_m2(bbox_raw)
        if area is None:
            continue
        try:
            med = float(median_m2)
        except (TypeError, ValueError):
            continue
        if area >= med:
            out.append(item)
    return out


def main():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bbox-dir", type=str, default=BBOX_DIR)
    p.add_argument("--knn-dir", type=str, default=KNN_FOR_RENDER_DIR)
    p.add_argument("--image-root", type=str, default=IMAGE_SINGLE_ROOT)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--model", type=str, default=MODEL_PATH)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bbox_files = sorted(glob.glob(os.path.join(args.bbox_dir, "*_bbox.json")))
    if not bbox_files:
        print(f"未找到 *_bbox.json: {args.bbox_dir}")
        sys.exit(1)

    try:
        import open_clip
    except ImportError as e:
        print(f"需要 open_clip: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    with tqdm(total=1, desc="Loading CLIP", unit="step") as load_pbar:
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            model_name="EVA02-E-14-plus",
            pretrained=args.model,
            device=device,
        )
        load_pbar.update(1)
    clip_model.eval()

    for json_path in tqdm(
        bbox_files, desc="scenes", unit="scene", total=len(bbox_files)
    ):
        fname = os.path.basename(json_path)
        if not fname.endswith("_bbox.json"):
            continue
        scene_id = fname.replace("_bbox.json", "")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        scene_id = data.get("scene_id", scene_id)
        knn_data = load_knn_for_scene(args.knn_dir, scene_id)
        if not knn_data:
            tqdm.write(f"[skip] 无 knn: {scene_id}")
            continue
        median_m2 = knn_data.get("median_remaining_bbox_xy_area_m2")
        if median_m2 is None:
            tqdm.write(f"[skip] knn 无 median: {scene_id}")
            continue
        knn_ids = object_ids_from_knn_clusters(knn_data)
        bboxes = data.get("bboxes") or []
        targets = filter_above_median_building_fence(bboxes, knn_ids, float(median_m2))
        if not targets:
            tqdm.write(f"[skip] 无 Building/Fence 且 area>=median 且在 knn: {scene_id}")
            continue

        feats_list: list[np.ndarray] = []
        ids_list: list[int] = []

        for obj in tqdm(
            targets,
            desc=f"Processing {scene_id}",
            unit="obj",
            leave=False,
        ):
            oid = obj.get("object_id")
            try:
                oid_i = int(float(oid))
            except (TypeError, ValueError):
                continue
            img_path = find_single_image_cityanchor(scene_id, oid_i)
            if not img_path:
                tqdm.write(f"  [missing] {scene_id} obj {oid_i}")
                continue
            try:
                image_raw = Image.open(img_path).convert("RGB")
                image_input = preprocess(image_raw).unsqueeze(0).to(device, non_blocking=True)
                with torch.no_grad():
                    image_feat = clip_model.encode_image(image_input)
                    image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                feat_np = image_feat.squeeze(0).float().cpu().numpy()
                feats_list.append(feat_np)
                ids_list.append(oid_i)
            except Exception as e:
                tqdm.write(f"  [fail] {img_path}: {e}")
                continue

        if not feats_list:
            tqdm.write(f"[skip] 无有效图: {scene_id}")
            continue

        feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
        ids_arr = np.array(ids_list, dtype=np.int64)
        n, d = feats_arr.shape

        safe = str(scene_id).replace("/", "_").replace("\\", "_")
        out_path = os.path.join(
            args.output_dir, f"{safe}_clip_BuildingFence_above_median.npy"
        )

        dt = np.dtype(
            [
                ("object_id", np.int64),
                ("feature", np.float32, (d,)),
            ]
        )
        rec = np.empty(n, dtype=dt)
        rec["object_id"] = ids_arr
        rec["feature"] = feats_arr
        np.save(out_path, rec)

        tqdm.write(f"[ok] {scene_id}: {n} rows (object_id, feature) -> {out_path}")

    print(f"完成。输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

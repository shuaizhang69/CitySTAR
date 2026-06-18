"""
CityAnchor：从 city_Anchor/bev_crops 读 BEV 裁剪图，用 CLIP（EVA02-E-14-plus）提特征。

选取规则（与 bev_render 一致地 **不处理** SKIP_BEV 三类：HighVegetation / Bike / LightPole）：
  - for_render knn 中所有簇的 center + neighbor object_id；
  - 在 bbox JSON 中能查到该 id 的条目，且类名不在 SKIP_BEV；
  - 在 bev_crops 路径下存在对应图片（{scene_id}/{scene_id}_{object_id}.png 等）。

每个场景只写一个 .npy：结构化数组 dtype [('object_id', int64), ('feature', float32, (D,))]。

读取示例::
    rec = np.load("xxx.npy", allow_pickle=False)
    oid_to_feat = {int(rec["object_id"][i]): rec["feature"][i].copy() for i in range(len(rec))}
"""
import os
import sys
import json
import glob

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code"
_DATASET = "/hpc2hdd/home/yxiao224/Henry/dataset"

BBOX_DIR = os.path.join(_DATASET, "city_Anchor", "bbox")
KNN_FOR_RENDER_DIR = os.path.join(_DATASET, "Our_cityG3D", "knn", "for_render")
# BEV 输出：bev_crops/<scene_id>/<scene_id>_<object_id>.png（见 bev_render_bbox_stpls3d_synthetic_v3）
BEV_CROPS_ROOT = os.path.join(_DATASET, "city_Anchor", "bev_crops")
OUTPUT_DIR = os.path.join(_DATASET, "city_Anchor", "clip_features_bev_crops_knn")

# 与 bev_render_bbox_stpls3d_synthetic_v3 中 SKIP_BEV_OBJECT_NAMES 一致
SKIP_BEV_OBJECT_NAMES = frozenset(("HighVegetation", "Bike", "LightPole"))
GROUND_OBJECT_ID = -100


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


def object_name_str(item: dict) -> str:
    return str(item.get("object_name") or item.get("label") or "")


def bbox_by_object_id(bboxes: list) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for item in bboxes:
        oid = item.get("object_id")
        if oid is None:
            continue
        try:
            oi = int(float(oid))
        except (TypeError, ValueError):
            continue
        if oi not in out:
            out[oi] = item
    return out


def find_bev_crop_image(bev_root: str, scene_id: str, object_id: int) -> str | None:
    """bev_crops/{scene_id}/{scene_id}_{object_id}.(png|jpg|...)"""
    safe = str(scene_id).replace("/", "_").replace("\\", "_")
    oid = int(object_id)
    base = os.path.join(bev_root, safe, f"{safe}_{oid}")
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".JPG", ".JPEG"):
        p = base + ext
        if os.path.isfile(p):
            return p
    return None


def main():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--bbox-dir", type=str, default=BBOX_DIR)
    p.add_argument("--knn-dir", type=str, default=KNN_FOR_RENDER_DIR)
    p.add_argument("--bev-crops-root", type=str, default=BEV_CROPS_ROOT)
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

        knn_ids = object_ids_from_knn_clusters(knn_data)
        if not knn_ids:
            tqdm.write(f"[skip] knn 无 clusters/object_id: {scene_id}")
            continue

        bboxes = data.get("bboxes") or []
        oid_to_bbox = bbox_by_object_id(bboxes)

        scene_bev_dir = os.path.join(
            args.bev_crops_root,
            str(scene_id).replace("/", "_").replace("\\", "_"),
        )
        if not os.path.isdir(scene_bev_dir):
            tqdm.write(f"[skip] 无 bev_crops 目录: {scene_bev_dir}")
            continue

        feats_list: list[np.ndarray] = []
        ids_list: list[int] = []

        ordered = sorted(knn_ids)
        for oid_i in tqdm(
            ordered,
            desc=f"Processing {scene_id}",
            unit="obj",
            leave=False,
        ):
            if oid_i == GROUND_OBJECT_ID:
                continue
            item = oid_to_bbox.get(oid_i)
            if item is None:
                continue
            name = object_name_str(item)
            if name in SKIP_BEV_OBJECT_NAMES:
                continue

            img_path = find_bev_crop_image(args.bev_crops_root, scene_id, oid_i)
            if not img_path:
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
            tqdm.write(f"[skip] 无有效 bev 图: {scene_id}")
            continue

        feats_arr = np.stack(feats_list, axis=0).astype(np.float32)
        ids_arr = np.array(ids_list, dtype=np.int64)
        n, d = feats_arr.shape

        safe = str(scene_id).replace("/", "_").replace("\\", "_")
        out_path = os.path.join(
            args.output_dir, f"{safe}_clip_bev_crops_knn.npy"
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

        tqdm.write(f"[ok] {scene_id}: {n} rows -> {out_path}")

    print(f"完成。输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

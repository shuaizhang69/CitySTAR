"""
对“同场景 + 同类别”分组后的 crop_all 图片提取 CLIP 特征，并保存为 JSON。

输入:
  cityanchor_val_same_scene_same_category_objects.json

输出:
  cityanchor_val_same_scene_same_category_clip_features.json
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


MODEL_PATH = "/hpc2hdd/home/yxiao224/Henry/checkpoints/CLIP_eav02/open_clip_pytorch_model.bin"
PROJECT_ROOT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code"
_DATASET = "/hpc2hdd/home/yxiao224/Henry/dataset"

GROUP_JSON = os.path.join(
    _DATASET,
    "city_Anchor",
    "cityanchor_val_same_scene_same_category_objects.json",
)
OUTPUT_JSON = os.path.join(
    _DATASET,
    "city_Anchor",
    "cityanchor_val_same_scene_same_category_clip_features.json",
)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sliced_groups(groups: list[dict], rank: int, world_size: int) -> list[dict]:
    return [group for idx, group in enumerate(groups) if idx % world_size == rank]


def output_path_for_rank(output_json: str, rank: int, world_size: int) -> str:
    if world_size <= 1:
        return output_json
    root, ext = os.path.splitext(output_json)
    return f"{root}_rank{rank}of{world_size}{ext}"


def encode_image_feature(
    image_path: str,
    preprocess,
    clip_model,
    device: torch.device,
) -> np.ndarray:
    image_raw = Image.open(image_path).convert("RGB")
    image_input = preprocess(image_raw).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        image_feat = clip_model.encode_image(image_input)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    return image_feat.squeeze(0).float().cpu().numpy()


def main():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--group-json", type=str, default=GROUP_JSON)
    parser.add_argument("--output-json", type=str, default=OUTPUT_JSON)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    if args.world_size < 1:
        raise ValueError("--world-size 必须 >= 1")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank 必须满足 0 <= rank < world_size")

    data = load_json(args.group_json)
    all_groups = data.get("groups") or []
    groups = sliced_groups(all_groups, rank=args.rank, world_size=args.world_size)
    if args.limit_groups and args.limit_groups > 0:
        groups = groups[: args.limit_groups]
    output_json = output_path_for_rank(
        args.output_json,
        rank=args.rank,
        world_size=args.world_size,
    )

    try:
        import open_clip
    except ImportError as e:
        print(f"需要 open_clip: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(
        f"分片: rank={args.rank}/{args.world_size}, "
        f"assigned_groups={len(groups)}, total_groups={len(all_groups)}"
    )
    with tqdm(total=1, desc="Loading CLIP", unit="step") as load_pbar:
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            model_name="EVA02-E-14-plus",
            pretrained=args.model,
            device=device,
        )
        clip_model.eval()
        load_pbar.update(1)

    feature_cache: dict[str, list[float]] = {}
    output_groups = []
    total_processed = 0
    total_failed = 0
    feature_dim = None

    for group in tqdm(groups, desc="groups", unit="group"):
        scene_id = group.get("scene_id")
        category = group.get("category")
        referenced_ids = {
            int(ref["object_id"])
            for ref in group.get("referenced_by") or []
            if ref.get("object_id") is not None
        }

        group_objects_out = []
        group_processed = 0
        group_failed = 0

        for obj in tqdm(
            group.get("objects") or [],
            desc=f"Processing {scene_id}/{category}",
            unit="obj",
            leave=False,
        ):
            image_path = obj.get("image_path")
            if not image_path or not os.path.isfile(image_path):
                group_failed += 1
                total_failed += 1
                continue

            cache_key = image_path
            try:
                if cache_key not in feature_cache:
                    feat_np = encode_image_feature(
                        image_path=image_path,
                        preprocess=preprocess,
                        clip_model=clip_model,
                        device=device,
                    )
                    feature_cache[cache_key] = feat_np.tolist()
                    if feature_dim is None:
                        feature_dim = int(feat_np.shape[0])

                group_objects_out.append(
                    {
                        "object_id": int(obj["object_id"]),
                        "object_name": obj.get("object_name"),
                        "landmark": obj.get("landmark", ""),
                        "bbox": obj.get("bbox"),
                        "image_path": image_path,
                        "is_referenced_object": int(obj["object_id"]) in referenced_ids,
                        "feature": feature_cache[cache_key],
                    }
                )
                group_processed += 1
                total_processed += 1
            except Exception as e:
                tqdm.write(f"[fail] {image_path}: {e}")
                group_failed += 1
                total_failed += 1

        output_groups.append(
            {
                "scene_id": scene_id,
                "category": category,
                "bbox_json": group.get("bbox_json"),
                "crop_dir": group.get("crop_dir"),
                "reference_count": len(group.get("referenced_by") or []),
                "object_count": len(group.get("objects") or []),
                "processed_count": group_processed,
                "failed_count": group_failed,
                "referenced_by": group.get("referenced_by") or [],
                "objects": group_objects_out,
            }
        )

    output = {
        "source_group_json": args.group_json,
        "model_used": "EVA02-E-14-plus",
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_group_count": len(groups),
        "total_group_count": len(all_groups),
        "feature_dim": feature_dim,
        "group_count": len(output_groups),
        "processed_object_count": total_processed,
        "failed_object_count": total_failed,
        "groups": output_groups,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(
        "完成: "
        f"groups={len(output_groups)}, "
        f"processed={total_processed}, "
        f"failed={total_failed} -> {output_json}"
    )


if __name__ == "__main__":
    main()

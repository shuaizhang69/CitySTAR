"""
根据场景 bbox JSON（无 .pth）裁剪正射影像。
JSON 中每条 bboxes[].bbox 为轴对齐框（米）：[cx, cy, cz, w, h, ...]，与旧版 PTH 中
instance_bboxes_dict 的约定一致。
裁剪窗口：将框四角 (cx±w/2, cy±h/2) 用 TIF 仿射变换投到像素行/列，再按 zoom_ratio
外扩；不再用单一 scale_factor 换算宽高，避免 x/y 分辨率不一致或中心 int 截断导致「框未完全包住」。
贴图边界时仍会裁掉越界部分（与旧逻辑相同）。
"""
import json
import os
import sys
from typing import Optional, Set, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.transform import rowcol


def parse_axis_aligned_bbox_meters(bbox) -> Optional[Tuple[float, float, float, float]]:
    """从 JSON bbox 列表解析中心与宽高（米）。横平竖直，对应索引 0,1,3,4。"""
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 5:
        return None
    cx, cy = float(bbox[0]), float(bbox[1])
    w, h = float(bbox[3]), float(bbox[4])
    return cx, cy, w, h


def bbox_meters_to_pixel_roi(
    transform,
    cx_m: float,
    cy_m: float,
    w_m: float,
    h_m: float,
    *,
    context_scale: float,
    context_margin_m: float,
    img_height: int,
    img_width: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    用仿射变换把米制轴对齐框四角投到像素，再按
    邻域窗口 = context_scale * bbox + context_margin_m（米）
    扩展，得到 [r_start:r_end, c_start:c_end)（行/列切片）。

    比「中心 + w/scale、h/scale」更稳：与 TIF 的 x/y 分辨率一致，且不因 int 截断中心或奇数宽高少 1 像素而裁小。
    """
    x_lo = cx_m - 0.5 * w_m
    x_hi = cx_m + 0.5 * w_m
    y_lo = cy_m - 0.5 * h_m
    y_hi = cy_m + 0.5 * h_m
    corners = ((x_lo, y_lo), (x_hi, y_lo), (x_lo, y_hi), (x_hi, y_hi))
    rows: list[float] = []
    cols: list[float] = []
    for x, y in corners:
        r, c = rowcol(transform, x, y)
        rows.append(float(r))
        cols.append(float(c))
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    # 覆盖四角所占像素行/列区间（半开区间）
    r0 = int(np.floor(r_min))
    r1 = int(np.floor(r_max)) + 1
    c0 = int(np.floor(c_min))
    c1 = int(np.floor(c_max)) + 1
    rh = r1 - r0
    rw = c1 - c0
    if rh < 1 or rw < 1:
        return None
    r_center = 0.5 * (r0 + r1)
    c_center = 0.5 * (c0 + c1)

    res_x = abs(float(transform.a))
    res_y = abs(float(transform.e))
    margin_cols = context_margin_m / res_x if res_x > 0 else 0.0
    margin_rows = context_margin_m / res_y if res_y > 0 else 0.0

    new_rh = max(rh * context_scale + margin_rows, 1.0)
    new_rw = max(rw * context_scale + margin_cols, 1.0)
    r_start = int(np.floor(r_center - new_rh / 2.0))
    r_end = int(np.ceil(r_center + new_rh / 2.0))
    c_start = int(np.floor(c_center - new_rw / 2.0))
    c_end = int(np.ceil(c_center + new_rw / 2.0))
    r_start = max(r_start, 0)
    r_end = min(r_end, img_height)
    c_start = max(c_start, 0)
    c_end = min(c_end, img_width)
    if r_end <= r_start or c_end <= c_start:
        return None
    return r_start, r_end, c_start, c_end


def process_scene_from_bbox_json(
    tif_path: str,
    json_path: str,
    output_dir: str,
    *,
    output_size: int = 1024,
    context_scale: float = 2.0,
    context_margin_m: float = 50.0,
    max_instances: Optional[int] = None,
    skip_object_ids: Optional[Set[int]] = None,
) -> int:
    """
    读取 json 中 bboxes，按轴对齐 bbox 裁剪并保存 JPEG。
    返回成功写入的实例数。
    """
    skip_object_ids = skip_object_ids if skip_object_ids is not None else {-100}

    if not os.path.isfile(json_path):
        print(f"❌ JSON 不存在: {json_path}")
        return 0
    if not os.path.isfile(tif_path):
        print(f"❌ TIF 不存在: {tif_path}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene_id = data.get("scene_id") or os.path.splitext(os.path.basename(json_path))[0]
    json_objects = data.get("bboxes", [])

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    processed_ids: set = set()

    with rasterio.open(tif_path) as src:
        image = src.read()
        _, img_height, img_width = image.shape

        print(
            f"🚀 场景：{scene_id} | 影像 {img_width}x{img_height} | 标注条数 {len(json_objects)} "
            f"| 邻域窗口={context_scale}xbbox+{context_margin_m}m"
        )

        for obj in json_objects:
            if max_instances is not None and count >= max_instances:
                break

            obj_id = obj.get("object_id")
            if obj_id is None:
                obj_id = obj.get("id")

            parsed = parse_axis_aligned_bbox_meters(obj.get("bbox"))
            if parsed is None:
                continue
            cx_m, cy_m, w_m, h_m = parsed

            if obj_id in skip_object_ids:
                continue
            if obj_id in processed_ids:
                continue
            processed_ids.add(obj_id)

            r_c = rowcol(src.transform, cx_m, cy_m)
            cr, cc = float(r_c[0]), float(r_c[1])
            if not (0 <= cr < img_height and 0 <= cc < img_width):
                print(f"   ⚠️ 实例 {obj_id} 中心越界 ({int(cr)}, {int(cc)})，跳过")
                continue

            roi = bbox_meters_to_pixel_roi(
                src.transform,
                cx_m,
                cy_m,
                w_m,
                h_m,
                context_scale=context_scale,
                context_margin_m=context_margin_m,
                img_height=img_height,
                img_width=img_width,
            )
            if roi is None:
                continue
            r_start, r_end, c_start, c_end = roi

            roi = image[:, r_start:r_end, c_start:c_end]
            h_roi, w_roi = roi.shape[1], roi.shape[2]
            if h_roi == 0 or w_roi == 0:
                continue

            roi_hwc = np.transpose(roi, (1, 2, 0))
            if roi_hwc.dtype != np.uint8:
                min_v, max_v = roi_hwc.min(), roi_hwc.max()
                if max_v > min_v:
                    roi_hwc = ((roi_hwc - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    roi_hwc = np.zeros_like(roi_hwc, dtype=np.uint8)

            if roi_hwc.shape[2] == 1:
                roi_hwc = cv2.cvtColor(roi_hwc, cv2.COLOR_GRAY2RGB)
            elif roi_hwc.shape[2] > 3:
                roi_hwc = roi_hwc[:, :, :3]

            target_size = output_size
            scale = min(target_size / w_roi, target_size / h_roi)
            new_w = int(w_roi * scale)
            new_h = int(h_roi * scale)
            resized_content = cv2.resize(roi_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            resized_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_content

            box_r0, box_r1, box_c0, box_c1 = bbox_meters_to_pixel_roi(
                src.transform,
                cx_m,
                cy_m,
                w_m,
                h_m,
                context_scale=context_scale,
                context_margin_m=0.0,
                img_height=img_height,
                img_width=img_width,
            )
            rel_left = box_c0 - c_start
            rel_top = box_r0 - r_start
            rel_right = box_c1 - c_start
            rel_bottom = box_r1 - r_start

            draw_x1 = int(rel_left * (new_w / w_roi)) + x_offset
            draw_y1 = int(rel_top * (new_h / h_roi)) + y_offset
            draw_x2 = int(rel_right * (new_w / w_roi)) + x_offset
            draw_y2 = int(rel_bottom * (new_h / h_roi)) + y_offset

            draw_x1 = max(0, min(draw_x1, output_size - 1))
            draw_y1 = max(0, min(draw_y1, output_size - 1))
            draw_x2 = max(0, min(draw_x2, output_size - 1))
            draw_y2 = max(0, min(draw_y2, output_size - 1))
            cv2.rectangle(
                resized_img,
                (draw_x1, draw_y1),
                (draw_x2, draw_y2),
                (255, 0, 0),
                thickness=max(2, output_size // 256),
            )

            save_name = f"{scene_id}_obj{obj_id}.jpg"
            save_path = os.path.join(output_dir, save_name)
            with rasterio.open(
                save_path,
                "w",
                driver="JPEG",
                height=output_size,
                width=output_size,
                count=3,
                dtype="uint8",
                photometric="rgb",
                QUALITY=95,
            ) as dst:
                dst.write(np.transpose(resized_img, (2, 0, 1)))

            print(f"   ✅ {save_name}")
            count += 1

    return count


def main():
    # 与 04crop_image_single.py 保持一致的默认路径，可按需修改
    tif_base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/map"
    json_base_dir = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/bbox"
    output_root = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_context"

    os.makedirs(output_root, exist_ok=True)

    # 优先：遍历 bbox 目录下所有 json，用 data['scene_id'] 找 TIF
    json_files = sorted(
        f for f in os.listdir(json_base_dir) if f.endswith(".json")
    )
    if not json_files:
        print(f"❌ {json_base_dir} 下没有 json")
        sys.exit(1)

    ok, skip = 0, 0
    for idx, jf in enumerate(json_files):
        if idx == 0:
            print(f"⚠️ 跳过第一个场景: {jf}")
            skip += 1
            continue
        json_path = os.path.join(json_base_dir, jf)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        scene_id = meta.get("scene_id")
        if not scene_id:
            scene_id = jf.replace("_bbox.json", "").replace(".json", "")

        tif_path = os.path.join(tif_base_dir, f"{scene_id}.tif")
        if not os.path.isfile(tif_path):
            print(f"⚠️ 跳过 {jf}: 无对应 TIF {tif_path}")
            skip += 1
            continue

        out_dir = os.path.join(output_root, scene_id)
        print(f"\n{'=' * 20} {scene_id} {'=' * 20}")
        n = process_scene_from_bbox_json(
            tif_path,
            json_path,
            out_dir,
            max_instances=None,
        )
        if n:
            ok += 1
        else:
            skip += 1

    print(f"\n🎉 完成：有输出的场景约 {ok} 个，跳过/无输出 {skip}")


if __name__ == "__main__":
    main()

import importlib.util
import json
import os
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


COLOR_SCRIPT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code_for_Anchor/01_Semantic_color_in_stage1_support_object_proximity_shrink.py"
SINGLE_IMAGE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image"
OUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/query_color_conflict_viz"
OUT_JSON = os.path.join(OUT_DIR, "query_color_conflicts.json")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module(COLOR_SCRIPT, "color_stage1")


def collect_conflicts(data_jsonl_path, split_tag):
    color_map = MOD.load_color_map(MOD.DEFAULT_COLOR_JSONL)
    query_color_map = MOD.load_query_color_map(MOD.DEFAULT_QUERY_COLOR_JSON)
    conflicts = []

    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            scene_id = item["scene_id"]
            gt_id = str(item["object_id"])
            target_tiers, color_source, color_phrase, color_labels = MOD._get_main_object_target_tiers(
                scene_id=scene_id,
                gt_id=gt_id,
                query_color_map=query_color_map,
            )
            gt_pred_tier = MOD._tier_to_int(color_map.get((scene_id, gt_id)))
            if not target_tiers or gt_pred_tier is None:
                continue
            if gt_pred_tier in MOD._get_allowed_tiers(target_tiers):
                continue
            conflicts.append(
                {
                    "split": split_tag,
                    "scene_id": scene_id,
                    "object_id": gt_id,
                    "object_name": item.get("object_name"),
                    "description": item.get("description", ""),
                    "query_color_source": color_source,
                    "query_color_phrase": color_phrase,
                    "query_color_labels": color_labels,
                    "query_target_tiers": sorted(target_tiers),
                    "gt_pred_tier": gt_pred_tier,
                }
            )
    return conflicts


def resolve_crop_path(scene_id, object_id):
    base_dir = os.path.join(SINGLE_IMAGE_ROOT, scene_id)
    candidates = [
        os.path.join(base_dir, f"{scene_id}_obj{object_id}.jpg"),
        os.path.join(base_dir, f"{scene_id}_obj{object_id}.png"),
        os.path.join(base_dir, f"{scene_id}_obj{object_id}.jpeg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def make_canvas(image, lines, footer_lines):
    font = ImageFont.load_default()
    line_h = 16
    top_h = 12 + len(lines) * line_h
    bottom_h = 8 + len(footer_lines) * line_h if footer_lines else 0
    canvas = Image.new("RGB", (image.width, top_h + image.height + bottom_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, top_h), fill=(32, 32, 32))
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    canvas.paste(image, (0, top_h))
    if footer_lines:
        footer_y = top_h + image.height + 4
        for line in footer_lines:
            draw.text((8, footer_y), line, fill=(20, 20, 20), font=font)
            footer_y += line_h
    return canvas


def annotate_conflict(record):
    crop_path = resolve_crop_path(record["scene_id"], record["object_id"])
    record["crop_path"] = crop_path
    if crop_path is None:
        record["viz_path"] = None
        return record

    image = Image.open(crop_path).convert("RGB")
    header = [
        f"{record['split']} | {record['scene_id']} | obj {record['object_id']} | {record['object_name']}",
        f"query_color_phrase={record['query_color_phrase'] or 'NONE'}",
        f"query_target_tiers={record['query_target_tiers']} | gt_pred_tier={record['gt_pred_tier']} | source={record['query_color_source']}",
    ]
    footer_text = f"desc: {record['description']}"
    footer = wrap(footer_text, width=max(30, image.width // 8))
    canvas = make_canvas(image, header, footer[:6])

    out_name = f"{record['split']}_{record['scene_id']}_obj{record['object_id']}.jpg"
    out_path = os.path.join(OUT_DIR, out_name)
    canvas.save(out_path, quality=95)
    record["viz_path"] = out_path
    return record


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    conflicts = []
    conflicts.extend(collect_conflicts(MOD.DEFAULT_JSONL_ND, "ND"))
    conflicts.extend(collect_conflicts(MOD.DEFAULT_JSONL_NO, "NO"))

    annotated = [annotate_conflict(rec) for rec in conflicts]
    output = {
        "count": len(annotated),
        "output_dir": OUT_DIR,
        "records": annotated,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"saved_dir: {OUT_DIR}")
    print(f"saved_json: {OUT_JSON}")
    print(f"conflict_count: {len(annotated)}")
    print(f"with_viz: {sum(1 for x in annotated if x.get('viz_path'))}")


if __name__ == "__main__":
    main()

import importlib.util
import json
import os
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


QUERY_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm_rerun_object_name.json"
COLOR_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_colors2_per_image_tier2.jsonl"
COLOR_SCRIPT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code_for_Anchor/01_Semantic_color_in_stage1_support_object_proximity_shrink.py"
SINGLE_IMAGE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image"
OUT_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/final_query_vs_single_image_colors2_conflicts_viz"
OUT_JSON = os.path.join(OUT_DIR, "final_query_vs_single_image_colors2_conflicts.json")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module(COLOR_SCRIPT, "color_stage1")


def load_color_map(path: str):
    return MOD.load_color_map(path)


def resolve_crop_path(scene_id: str, object_id: str):
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


def make_canvas(image: Image.Image, header_lines: list[str], footer_lines: list[str]) -> Image.Image:
    font = ImageFont.load_default()
    line_h = 16
    top_h = 10 + line_h * len(header_lines)
    bottom_h = 8 + line_h * len(footer_lines) if footer_lines else 0
    canvas = Image.new("RGB", (image.width, top_h + image.height + bottom_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, top_h), fill=(30, 30, 30))
    y = 6
    for line in header_lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    canvas.paste(image, (0, top_h))
    if footer_lines:
        y = top_h + image.height + 4
        for line in footer_lines:
            draw.text((8, y), line, fill=(0, 0, 0), font=font)
            y += line_h
    return canvas


def collect_conflicts():
    with open(QUERY_JSON, "r", encoding="utf-8") as f:
        query_data = json.load(f)
    color_map = load_color_map(COLOR_JSONL)

    conflicts = []
    for rec in query_data.get("records", []):
        llm_tiers = rec.get("llm_mapped_tiers") or []
        if not llm_tiers:
            continue
        scene_id = rec.get("scene_id")
        object_id = str(rec.get("object_id"))
        gt_pred_tier = MOD._tier_to_int(color_map.get((scene_id, object_id)))
        if gt_pred_tier is None:
            continue
        if gt_pred_tier in MOD._get_allowed_tiers(set(llm_tiers)):
            continue
        conflicts.append(
            {
                "split": rec.get("split"),
                "scene_id": scene_id,
                "object_id": object_id,
                "object_name": rec.get("object_name"),
                "description": rec.get("description"),
                "llm_main_object_name": rec.get("llm_main_object_name"),
                "query_color_phrase": rec.get("llm_main_object_color_phrase"),
                "query_color_source": rec.get("llm_color_source"),
                "query_color_labels": rec.get("llm_matched_color_labels") or [],
                "query_target_tiers": llm_tiers,
                "gt_main_color": rec.get("gt_main_color"),
                "gt_field_tiers": rec.get("gt_mapped_tiers") or [],
                "single_image_final_tier": gt_pred_tier,
            }
        )
    return conflicts


def visualize_conflicts(conflicts: list[dict]):
    os.makedirs(OUT_DIR, exist_ok=True)
    output_records = []
    for rec in conflicts:
        crop_path = resolve_crop_path(rec["scene_id"], rec["object_id"])
        viz_path = None
        if crop_path:
            image = Image.open(crop_path).convert("RGB")
            header = [
                f"{rec['split']} | {rec['scene_id']} | obj {rec['object_id']} | target={rec['object_name']}",
                f"llm_main_object_name={rec.get('llm_main_object_name') or 'NONE'}",
                f"query_color_phrase={rec.get('query_color_phrase') or 'NONE'}",
                f"query_tiers={rec.get('query_target_tiers') or []} | single_image_final_tier={rec.get('single_image_final_tier')}",
                f"gt_field_color={rec.get('gt_main_color') or 'NONE'} | gt_field_tiers={rec.get('gt_field_tiers') or []}",
            ]
            footer = wrap(f"description: {rec.get('description') or ''}", width=max(36, image.width // 8))[:8]
            canvas = make_canvas(image, header, footer)
            out_name = f"{rec['split']}_{rec['scene_id']}_obj{rec['object_id']}.jpg"
            viz_path = os.path.join(OUT_DIR, out_name)
            canvas.save(viz_path, quality=95)
        rec = dict(rec)
        rec["crop_path"] = crop_path
        rec["viz_path"] = viz_path
        output_records.append(rec)
    return output_records


def main():
    conflicts = collect_conflicts()
    visualized = visualize_conflicts(conflicts)
    output = {
        "query_json": QUERY_JSON,
        "color_jsonl": COLOR_JSONL,
        "count": len(visualized),
        "records": visualized,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"saved_dir: {OUT_DIR}")
    print(f"saved_json: {OUT_JSON}")
    print(f"conflict_count: {len(visualized)}")
    print(f"with_viz: {sum(1 for x in visualized if x.get('viz_path'))}")


if __name__ == "__main__":
    main()

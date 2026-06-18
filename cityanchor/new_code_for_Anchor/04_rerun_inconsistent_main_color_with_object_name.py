import importlib.util
import json
import os
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


COLOR_LLM_SCRIPT = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/new_code_for_Anchor/02_extract_main_object_color_llm.py"
INPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm.json"
OUTPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm_rerun_object_name.json"
VIZ_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm_rerun_object_name_viz"
VIZ_INDEX_JSON = os.path.join(VIZ_DIR, "remaining_soft_mismatches.json")
SINGLE_IMAGE_ROOT = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image"


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_module(COLOR_LLM_SCRIPT, "color_llm")


def is_soft_mismatch(record: dict) -> bool:
    llm_tiers = record.get("llm_mapped_tiers") or []
    gt_tiers = record.get("gt_mapped_tiers") or []
    if not llm_tiers or not gt_tiers:
        return False
    return not MOD.soft_tier_match(llm_tiers, gt_tiers)


def rerun_records(records: list[dict]) -> list[dict]:
    results = []
    for rec in records:
        results.append(
            MOD.infer_single(
                rec,
                api_key=MOD.API_KEY,
                base_url=MOD.BASE_URL,
                model_name=MOD.MODEL_NAME,
            )
        )
    return results


def resolve_crop_path(scene_id: str, object_id: str) -> str | None:
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
    draw.rectangle((0, 0, image.width, top_h), fill=(24, 24, 24))
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


def visualize_remaining_mismatches(records: list[dict]) -> list[dict]:
    os.makedirs(VIZ_DIR, exist_ok=True)
    output_records = []
    for rec in records:
        crop_path = resolve_crop_path(rec["scene_id"], str(rec["object_id"]))
        viz_path = None
        if crop_path:
            image = Image.open(crop_path).convert("RGB")
            header = [
                f"{rec['split']} | {rec['scene_id']} | obj {rec['object_id']} | target={rec['object_name']}",
                f"llm_main_object_name={rec.get('llm_main_object_name') or 'NONE'}",
                f"extract={rec.get('llm_main_object_color_phrase') or 'NONE'} | llm_tiers={rec.get('llm_mapped_tiers') or []}",
                f"gt_color={rec.get('gt_main_color') or 'NONE'} | gt_tiers={rec.get('gt_mapped_tiers') or []}",
            ]
            footer = wrap(f"description: {rec.get('description') or ''}", width=max(36, image.width // 8))[:8]
            canvas = make_canvas(image, header, footer)
            out_name = f"{rec['split']}_{rec['scene_id']}_obj{rec['object_id']}.jpg"
            viz_path = os.path.join(VIZ_DIR, out_name)
            canvas.save(viz_path, quality=95)
        rec = dict(rec)
        rec["viz_path"] = viz_path
        output_records.append(rec)

    with open(VIZ_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump({"count": len(output_records), "records": output_records}, f, ensure_ascii=False)
    return output_records


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    mismatches = [rec for rec in records if is_soft_mismatch(rec)]
    rerun_inputs = []
    rerun_keys = set()
    for rec in mismatches:
        rerun_inputs.append(
            {
                "split": rec["split"],
                "source_path": rec["source_path"],
                "line_idx": rec["line_idx"],
                "scene_id": rec["scene_id"],
                "object_id": str(rec["object_id"]),
                "object_name": rec["object_name"],
                "ann_id": rec["ann_id"],
                "description": rec["description"],
                "gt_main_color": rec["gt_main_color"],
            }
        )
        rerun_keys.add((rec["split"], rec["scene_id"], str(rec["object_id"]), rec.get("ann_id")))

    rerun_results = rerun_records(rerun_inputs) if rerun_inputs else []
    rerun_map = {
        (rec["split"], rec["scene_id"], str(rec["object_id"]), rec.get("ann_id")): rec
        for rec in rerun_results
    }

    merged_records = []
    for rec in records:
        key = (rec["split"], rec["scene_id"], str(rec["object_id"]), rec.get("ann_id"))
        if key in rerun_map:
            merged_records.append(rerun_map[key])
        else:
            merged_records.append(rec)

    output = {
        "source_files": data.get("source_files", []),
        "model_name": data.get("model_name"),
        "record_count": len(merged_records),
        "summary": MOD.build_summary(merged_records),
        "rerun_target_count": len(rerun_inputs),
        "records": merged_records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    remaining = [rec for rec in merged_records if is_soft_mismatch(rec)]
    visualize_remaining_mismatches(remaining)

    print(f"saved_json: {OUTPUT_JSON}")
    print(json.dumps(output["summary"], ensure_ascii=False))
    print(f"rerun_target_count: {len(rerun_inputs)}")
    print(f"remaining_soft_mismatch_count: {len(remaining)}")
    print(f"saved_viz_dir: {VIZ_DIR}")
    print(f"saved_viz_index: {VIZ_INDEX_JSON}")


if __name__ == "__main__":
    main()

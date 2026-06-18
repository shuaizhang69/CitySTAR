import json
import os
import re
import time
from functools import partial
from multiprocessing import Pool
from typing import Any, Dict, List, Optional


PROMPT_TEMPLATE = """
# Role
You are a careful extractor for the MAIN target object's color.

# Task
Given one referring expression, do these steps in order:
1. First identify what the main target object is.
2. Then extract only the color phrase describing that main target object.
3. Then map the main target color to the predefined color labels and tier ids below.

# Important constraints
0. The target object's category is explicitly given as: {object_name}
   You must use this category as the main disambiguation signal.
   If the sentence mentions multiple objects, prefer the color that belongs to the object matching this category.
1. Ignore colors of surrounding objects, distractors, nearby vehicles, nearby trees, nearby buildings, roads, fences, and background objects.
2. If the description mentions multiple objects with different colors, only keep the color of the main target object.
3. For buildings, use the ROOF color as primary whenever roof color is explicitly mentioned.
   If both wall color and roof color are mentioned for a building, prefer the roof color.
4. Return only information explicitly supported by the description.
5. If the main target color is not explicit, return empty fields.
6. If the main target has multiple colors, keep the short original phrase and include all relevant labels and tiers.

# Color label to tier mapping
1. White family:
White -> 1, Cream -> 1, Off-White -> 1
2. Silver / beige / light gray family:
Silver -> 2, Beige -> 2, Tan -> 2, Light Gray -> 2, Light Grey -> 2, Gold -> 2, Light Yellow -> 2
3. Medium gray family:
Gray -> 3, Grey -> 3, Stone -> 3
4. Red family:
Red -> 4, Maroon -> 4, Burgundy -> 4, Dark Red -> 4, Pink -> 4
5. Orange / yellow family:
Orange -> 5, Yellow -> 5, Rust -> 5
6. Green family:
Green -> 6, Light Green -> 6, Teal -> 6, Turquoise -> 6
7. Blue family:
Blue -> 7, Light Blue -> 7, Sky Blue -> 7, Bright Blue -> 7
8. Dark blue family:
Dark Blue -> 8, Navy Blue -> 8, Deep Blue -> 8
9. Purple family:
Purple -> 9, Violet -> 9, Magenta -> 9
10. Brown / earth family:
Brown -> 10, Dark Brown -> 10, Light Brown -> 10, Dirt -> 10, Tan-Brown -> 10
11. Dark gray / black family:
Dark Gray -> 11, Dark Grey -> 11, Charcoal -> 11, Dim -> 11, Black -> 11, Dark Green -> 11

# Output JSON
{
  "main_object_name": "",
  "main_object_color_phrase": "",
  "evidence_span": "",
  "predicted_color_labels": [],
  "predicted_color_tiers": []
}

# Description
{description}
"""


API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = "deepseek-chat"
NUM_WORKERS = 8

INPUT_FILES = [
    ("ND", "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_ND_0324.jsonl"),
    ("NO", "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new_0324_new.jsonl"),
]
OUTPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/main_object_color_llm.json"

COLOR_TO_TIER = {
    "white": 1,
    "cream": 1,
    "off white": 1,
    "off-white": 1,
    "silver": 2,
    "beige": 2,
    "tan": 2,
    "light gray": 2,
    "light grey": 2,
    "gold": 2,
    "light yellow": 2,
    "gray": 3,
    "grey": 3,
    "stone": 3,
    "red": 4,
    "maroon": 4,
    "burgundy": 4,
    "dark red": 4,
    "pink": 4,
    "orange": 5,
    "yellow": 5,
    "rust": 5,
    "green": 6,
    "light green": 6,
    "teal": 6,
    "turquoise": 6,
    "blue": 7,
    "light blue": 7,
    "sky blue": 7,
    "bright blue": 7,
    "dark blue": 8,
    "navy blue": 8,
    "deep blue": 8,
    "purple": 9,
    "violet": 9,
    "magenta": 9,
    "brown": 10,
    "dark brown": 10,
    "light brown": 10,
    "dirt": 10,
    "tan brown": 10,
    "tan-brown": 10,
    "reddish brown": 10,
    "red brown": 10,
    "orange red": 5,
    "orange-red": 5,
    "dark gray": 11,
    "dark grey": 11,
    "charcoal": 11,
    "dim": 11,
    "black": 11,
    "dark green": 11,
}


def extract_json_content(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    clean_text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()

    if not clean_text.startswith("{"):
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(0)
        else:
            return None

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return None


def normalize_color_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = text.replace("/", " ")
    text = text.replace("&", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z\s,]", " ", text)
    return " ".join(text.split())


def extract_color_labels_and_tiers(color_text: str) -> Dict[str, Any]:
    normalized = normalize_color_text(color_text)
    if not normalized:
        return {
            "normalized_phrase": "",
            "matched_color_labels": [],
            "mapped_tiers": [],
        }

    padded = f" {normalized} "
    matches = []
    for color_key, tier in sorted(COLOR_TO_TIER.items(), key=lambda kv: len(kv[0]), reverse=True):
        normalized_key = normalize_color_text(color_key)
        if f" {normalized_key} " in padded:
            matches.append((normalized_key, tier))

    dedup_labels = []
    seen_labels = set()
    dedup_tiers = []
    seen_tiers = set()
    for label, tier in matches:
        if label not in seen_labels:
            seen_labels.add(label)
            dedup_labels.append(label)
        if tier not in seen_tiers:
            seen_tiers.add(tier)
            dedup_tiers.append(tier)

    return {
        "normalized_phrase": normalized,
        "matched_color_labels": dedup_labels,
        "mapped_tiers": dedup_tiers,
    }


def get_gt_main_color(item: Dict[str, Any]) -> str:
    for obj in item.get("construction") or []:
        if obj.get("is_main") is True:
            return str(obj.get("color") or "")
    return ""


def load_input_records() -> List[Dict[str, Any]]:
    records = []
    for split, path in INPUT_FILES:
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                records.append(
                    {
                        "split": split,
                        "source_path": path,
                        "line_idx": line_idx,
                        "scene_id": item.get("scene_id"),
                        "object_id": str(item.get("object_id")),
                        "object_name": item.get("object_name"),
                        "ann_id": item.get("ann_id"),
                        "description": item.get("description", ""),
                        "gt_main_color": get_gt_main_color(item),
                    }
                )
    return records


def infer_single(item_data: Dict[str, Any], api_key: str, base_url: str, model_name: str) -> Dict[str, Any]:
    from openai import OpenAI
    import httpx

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(trust_env=False),
    )
    prompt_text = PROMPT_TEMPLATE.replace("{description}", item_data.get("description", ""))
    prompt_text = prompt_text.replace("{object_name}", str(item_data.get("object_name") or ""))
    messages = [{"role": "user", "content": prompt_text}]

    parsed_data = None
    raw_text = ""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            raw_text = (response.choices[0].message.content or "").strip()
            parsed_data = extract_json_content(raw_text)
            if parsed_data is not None:
                break
        except Exception as e:
            raw_text = f"ERROR: {e}"
        time.sleep(1)

    llm_main_object_name = ""
    llm_phrase = ""
    evidence_span = ""
    llm_predicted_labels: List[str] = []
    llm_predicted_tiers: List[int] = []
    if isinstance(parsed_data, dict):
        llm_main_object_name = str(parsed_data.get("main_object_name") or "").strip()
        llm_phrase = str(parsed_data.get("main_object_color_phrase") or "").strip()
        evidence_span = str(parsed_data.get("evidence_span") or "").strip()
        raw_labels = parsed_data.get("predicted_color_labels") or []
        raw_tiers = parsed_data.get("predicted_color_tiers") or []
        if isinstance(raw_labels, list):
            llm_predicted_labels = [str(x).strip() for x in raw_labels if str(x).strip()]
        if isinstance(raw_tiers, list):
            for x in raw_tiers:
                try:
                    llm_predicted_tiers.append(int(x))
                except (TypeError, ValueError):
                    continue

    color_source = "llm"
    if not llm_phrase:
        gt_main_color = str(item_data.get("gt_main_color") or "").strip()
        if gt_main_color:
            llm_phrase = gt_main_color
            evidence_span = gt_main_color
            color_source = "gt_main_color_fallback"
        else:
            color_source = "none"

    llm_mapping = extract_color_labels_and_tiers(llm_phrase)
    if not llm_predicted_labels:
        llm_predicted_labels = list(llm_mapping["matched_color_labels"])
    if not llm_predicted_tiers:
        llm_predicted_tiers = list(llm_mapping["mapped_tiers"])
    gt_mapping = extract_color_labels_and_tiers(item_data.get("gt_main_color", ""))

    result = dict(item_data)
    result.update(
        {
            "llm_main_object_name": llm_main_object_name,
            "llm_main_object_color_phrase": llm_phrase,
            "llm_evidence_span": evidence_span,
            "llm_color_source": color_source,
            "llm_normalized_phrase": llm_mapping["normalized_phrase"],
            "llm_matched_color_labels": llm_predicted_labels,
            "llm_mapped_tiers": llm_predicted_tiers,
            "gt_normalized_main_color": gt_mapping["normalized_phrase"],
            "gt_matched_color_labels": gt_mapping["matched_color_labels"],
            "gt_mapped_tiers": gt_mapping["mapped_tiers"],
            "llm_raw_response": raw_text,
        }
    )
    return result


def get_allowed_tiers(target_tiers: List[int]) -> set[int]:
    soft_groups = (
        {1, 2, 3, 11},
        {4, 5},
        {6, 11},
        {7, 8},
        {10, 3, 11},
    )
    target_set = {int(x) for x in target_tiers}
    allowed = set()
    for tier in target_set:
        allowed.update(range(max(1, tier - 2), min(11, tier + 2) + 1))
        for group in soft_groups:
            if tier in group:
                allowed.update(group)
    return allowed


def soft_tier_match(llm_tiers: List[int], gt_tiers: List[int]) -> bool:
    llm_set = {int(x) for x in llm_tiers}
    gt_set = {int(x) for x in gt_tiers}
    if not llm_set or not gt_set:
        return False
    gt_allowed = get_allowed_tiers(list(gt_set))
    llm_allowed = get_allowed_tiers(list(llm_set))
    return bool((llm_set & gt_allowed) or (gt_set & llm_allowed))


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    llm_non_empty = sum(bool(x.get("llm_main_object_color_phrase")) for x in results)
    llm_direct_non_empty = sum(x.get("llm_color_source") == "llm" and bool(x.get("llm_main_object_color_phrase")) for x in results)
    fallback_non_empty = sum(x.get("llm_color_source") == "gt_main_color_fallback" for x in results)
    gt_non_empty = sum(bool(x.get("gt_main_color")) for x in results)
    both_non_empty = sum(
        bool(x.get("llm_main_object_color_phrase")) and bool(x.get("gt_main_color"))
        for x in results
    )
    exact_phrase_match = sum(
        normalize_color_text(x.get("llm_main_object_color_phrase", ""))
        == normalize_color_text(x.get("gt_main_color", ""))
        and bool(x.get("llm_main_object_color_phrase"))
        and bool(x.get("gt_main_color"))
        for x in results
    )
    tier_overlap = sum(
        bool(set(x.get("llm_mapped_tiers", [])) & set(x.get("gt_mapped_tiers", [])))
        and bool(x.get("llm_mapped_tiers"))
        and bool(x.get("gt_mapped_tiers"))
        for x in results
    )
    soft_tier_match_count = sum(
        soft_tier_match(x.get("llm_mapped_tiers", []), x.get("gt_mapped_tiers", []))
        for x in results
        if x.get("llm_mapped_tiers") and x.get("gt_mapped_tiers")
    )
    soft_tier_compare_count = sum(
        bool(x.get("llm_mapped_tiers")) and bool(x.get("gt_mapped_tiers"))
        for x in results
    )
    return {
        "total": total,
        "llm_non_empty": llm_non_empty,
        "llm_direct_non_empty": llm_direct_non_empty,
        "gt_fallback_non_empty": fallback_non_empty,
        "gt_non_empty": gt_non_empty,
        "both_non_empty": both_non_empty,
        "exact_phrase_match": exact_phrase_match,
        "tier_overlap": tier_overlap,
        "soft_tier_match": soft_tier_match_count,
        "soft_tier_compare_count": soft_tier_compare_count,
    }


def main() -> None:
    all_data = load_input_records()
    infer_with_config = partial(
        infer_single,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
    )

    with Pool(processes=NUM_WORKERS) as pool:
        results = list(pool.imap(infer_with_config, all_data))

    output = {
        "source_files": [path for _, path in INPUT_FILES],
        "model_name": MODEL_NAME,
        "record_count": len(results),
        "summary": build_summary(results),
        "records": results,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"saved: {OUTPUT_JSON}")
    print(json.dumps(output["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()

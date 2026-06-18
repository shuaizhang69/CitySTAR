#!/usr/bin/env python3
"""
Extract geometry-related object mentions from CityAnchor descriptions.

Features:
1. Supports both JSON list and JSONL inputs.
2. Uses a regex/heuristic extractor by default.
3. Optionally supports an OpenAI-compatible LLM endpoint for harder cases.
4. Outputs one JSON object per input item with structured geometry mentions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


TASK_CONFIGS = {
    "ND": {
        "input": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_ND.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/geometry_final/cityanchor_val_ND_geometry_mentions.jsonl",
    },
    "NO": {
        "input": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/geometry_final/cityanchor_val_NO_geometry_mentions.jsonl",
    },
}


WORD_NUMBER_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "dozen": 12,
}


CATEGORY_SYNONYMS = {
    "Building": [
        "building",
        "buildings",
        "structure",
        "structures",
        "house",
        "houses",
        "edifice",
        "roofed building",
        "residential area",
        "parking lot",
    ],
    "Vehicle": [
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "bus",
        "buses",
        "sedan",
        "van",
        "vans",
        "helicopter",
    ],
    "Truck": [
        "truck",
        "trucks",
        "lorry",
        "lorries",
    ],
    "Bike": [
        "bike",
        "bicycle",
        "bicycles",
        "motorcycle",
        "motorcycles",
    ],
    "Fence": [
        "fence",
        "fences",
        "wall",
        "walls",
    ],
    "LightPole": [
        "light pole",
        "light poles",
        "street lamp",
        "street lamps",
        "streetlight",
        "streetlight pole",
        "utility pole",
        "lamp post",
        "pole",
    ],
    "HighVegetation": [
        "tree",
        "trees",
        "forest",
        "forests",
        "vegetation",
        "woodland",
        "woodland area",
        "grass",
        "grassy area",
        "lawn",
        "field",
    ],
}


DIMENSION_PATTERNS = {
    "height": [
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*(?:high|tall)\b", re.I),
        re.compile(r"height\s*(?:of\s*)?(?:about\s*|approximately\s*)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*in\s*height\b", re.I),
    ],
    "length": [
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*long\b", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*in\s*length\b", re.I),
        re.compile(r"length\s*(?:of\s*)?(?:about\s*|approximately\s*)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", re.I),
    ],
    "width": [
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*wide\b", re.I),
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\s*in\s*width\b", re.I),
        re.compile(r"width\s*(?:of\s*)?(?:about\s*|approximately\s*)?(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", re.I),
    ],
}


STORY_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen)[-\s]?(?:story|stories|floor|floors)\b",
    re.I,
)


SHAPE_PATTERNS = {
    "rectangular": re.compile(r"\brectangular\b", re.I),
    "square": re.compile(r"\bsquare\b", re.I),
    "irregular": re.compile(r"\birregular(?:ly shaped| shape)?\b", re.I),
    "u-shaped": re.compile(r"\bu[- ]shaped\b", re.I),
    "l-shaped": re.compile(r"\bl[- ]shaped\b", re.I),
    "f-shaped": re.compile(r"\bf[- ]shaped\b", re.I),
    "t-shaped": re.compile(r"\bt[- ]shaped\b", re.I),
    "slender": re.compile(r"\bslender\b", re.I),
    "round": re.compile(r"\bround\b|\bcircular\b", re.I),
}


SIZE_PATTERNS = {
    "short": re.compile(r"\bshort\b", re.I),
    "low": re.compile(r"\blow\b", re.I),
    "tall": re.compile(r"\btall\b|\bvery high\b|\bquite high\b", re.I),
    "long": re.compile(r"\blong\b|\bvery long\b", re.I),
    "large": re.compile(r"\blarge\b|\bvery large\b|\bbig\b|\bspacious\b", re.I),
    "small": re.compile(r"\bsmall\b|\btiny\b|\blittle\b", re.I),
}


SPLIT_PATTERN = re.compile(r"[.;]|, while |, with | and there is | and there are ", re.I)

DEFAULT_NER_CONFIG_SOURCE = "/hpc2hdd/home/yxiao224/Henry/code/CityAnchor/newcode2/data_construct_pipeline_Anchor/02NER.py"


LLM_PROMPT_TEMPLATE = """
You extract geometry mentions from a single CityAnchor description.

Known target category: {target_category}
Description: {description}

Return JSON only with this schema:
{{
  "mentions": [
    {{
      "source_text": "",
      "mention_role": "",
      "category_guess": "",
      "numeric_constraints": {{
        "height_m": null,
        "length_m": null,
        "width_m": null,
        "stories": null
      }},
      "shape_constraints": [],
      "size_constraints": [],
      "geometry_types": []
    }}
  ]
}}

Rules:
1. Extract every object mention that has geometry or size info.
2. For each mention, set mention_role to one of: "main", "other", "unknown".
3. "main" means the geometry is describing the referred target object of this sample.
4. "other" means the geometry is describing a context object, not the referred target.
5. If the text is too ambiguous to decide, use "unknown".
6. The target object category is usually the sample's object_name, but you must judge from sentence semantics rather than forcing every mention to main.
7. A description may contain both main and other mentions; keep them all and label them correctly.
8. Keep source_text short and local to the mention.
9. category_guess should be a dataset-like category if possible:
   Building, Vehicle, Truck, Bike, Fence, LightPole, HighVegetation.
10. geometry_types can contain: height, length, width, stories, shape, size.
11. Use null when a numeric value is absent.
12. Return an empty mentions array if nothing geometric exists.
"""


@dataclass
class ExtractorConfig:
    use_llm: bool = False
    api_key: str = ""
    base_url: str = ""
    model: str = "deepseek-chat"
    max_retries: int = 2
    sleep_seconds: float = 1.0


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_items(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def iter_tasks(task_arg: str) -> Iterable[Tuple[str, Dict[str, str]]]:
    if task_arg == "BOTH":
        for name in ("ND", "NO"):
            yield name, TASK_CONFIGS[name]
        return
    yield task_arg, TASK_CONFIGS[task_arg]


def canonical_category(text: str) -> str:
    text_norm = (text or "").strip().lower()
    if not text_norm:
        return ""
    for category, synonyms in CATEGORY_SYNONYMS.items():
        if text_norm == category.lower():
            return category
        if text_norm in synonyms:
            return category
    return text.strip()


def find_category_in_text(text: str, target_category: str = "") -> str:
    text_norm = (text or "").lower()
    if not text_norm:
        return canonical_category(target_category)

    for category, synonyms in CATEGORY_SYNONYMS.items():
        for synonym in sorted(synonyms, key=len, reverse=True):
            if re.search(rf"\b{re.escape(synonym)}\b", text_norm):
                return category

    return canonical_category(target_category)


def word_to_number(text: str) -> Optional[int]:
    if not text:
        return None
    text = text.strip().lower()
    if text.isdigit():
        return int(text)
    return WORD_NUMBER_MAP.get(text)


def unique_keep_order(items: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def extract_numeric_constraints(text: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "height_m": None,
        "length_m": None,
        "width_m": None,
        "stories": None,
    }
    for dim_name, patterns in DIMENSION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                value = float(match.group(1))
                out[f"{dim_name}_m"] = value
                break

    story_match = STORY_PATTERN.search(text)
    if story_match:
        out["stories"] = float(word_to_number(story_match.group(1)) or 0)

    return out


def extract_shape_constraints(text: str) -> List[str]:
    shapes = [shape for shape, pattern in SHAPE_PATTERNS.items() if pattern.search(text)]
    return unique_keep_order(shapes)


def extract_size_constraints(text: str) -> List[str]:
    sizes = [size for size, pattern in SIZE_PATTERNS.items() if pattern.search(text)]
    return unique_keep_order(sizes)


def geometry_types_from_parts(
    numeric_constraints: Dict[str, Optional[float]],
    shape_constraints: List[str],
    size_constraints: List[str],
) -> List[str]:
    types: List[str] = []
    if numeric_constraints.get("height_m") is not None:
        types.append("height")
    if numeric_constraints.get("length_m") is not None:
        types.append("length")
    if numeric_constraints.get("width_m") is not None:
        types.append("width")
    if numeric_constraints.get("stories") is not None:
        types.append("stories")
    if shape_constraints:
        types.append("shape")
    if size_constraints:
        types.append("size")
    return types


def clean_fragment(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text.strip(" ,.")


def split_description(description: str) -> List[str]:
    parts = [clean_fragment(x) for x in SPLIT_PATTERN.split(description or "")]
    return [x for x in parts if x]


def guess_mention_role_regex(fragment: str, target_category: str, description: str) -> str:
    fragment_norm = (fragment or "").lower()
    description_norm = (description or "").lower()
    target_norm = (target_category or "").lower()
    target_synonyms = set(CATEGORY_SYNONYMS.get(canonical_category(target_category), []))
    target_synonyms.add(target_norm)

    if any(s and re.search(rf"\b{re.escape(s)}\b", fragment_norm) for s in target_synonyms):
        return "main"
    if fragment_norm.startswith(("this ", "the ", "it ", "its ")) and description_norm.startswith(fragment_norm):
        return "main"
    if re.search(r"\b(next to|behind|in front of|between|near|beside)\b", fragment_norm) and any(
        s and re.search(rf"\b{re.escape(s)}\b", fragment_norm) for s in target_synonyms
    ):
        return "main"
    return "unknown"


def load_default_llm_config_from_source(source_path: str = DEFAULT_NER_CONFIG_SOURCE) -> Dict[str, str]:
    if not os.path.exists(source_path):
        return {}

    text = open(source_path, "r", encoding="utf-8").read()
    api_key_match = re.search(r'api_key\s*=\s*"([^"]+)"', text)
    base_url_match = re.search(r'base_url\s*=\s*"([^"]+)"', text)
    model_match = re.search(r'LLM_MODEL\s*=\s*"([^"]+)"', text)
    out: Dict[str, str] = {}
    if api_key_match:
        out["api_key"] = api_key_match.group(1)
    if base_url_match:
        out["base_url"] = base_url_match.group(1)
    if model_match:
        out["model"] = model_match.group(1)
    return out


def extract_mentions_regex(description: str, target_category: str) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    fragments = split_description(description)
    if not fragments and description:
        fragments = [clean_fragment(description)]

    for fragment in fragments:
        numeric_constraints = extract_numeric_constraints(fragment)
        shape_constraints = extract_shape_constraints(fragment)
        size_constraints = extract_size_constraints(fragment)
        geometry_types = geometry_types_from_parts(
            numeric_constraints=numeric_constraints,
            shape_constraints=shape_constraints,
            size_constraints=size_constraints,
        )
        if not geometry_types:
            continue

        mentions.append(
            {
                "source_text": fragment,
                "mention_role": guess_mention_role_regex(fragment, target_category=target_category, description=description),
                "category_guess": find_category_in_text(fragment, target_category=target_category),
                "numeric_constraints": numeric_constraints,
                "shape_constraints": shape_constraints,
                "size_constraints": size_constraints,
                "geometry_types": geometry_types,
            }
        )

    if mentions:
        return mentions

    numeric_constraints = extract_numeric_constraints(description)
    shape_constraints = extract_shape_constraints(description)
    size_constraints = extract_size_constraints(description)
    geometry_types = geometry_types_from_parts(
        numeric_constraints=numeric_constraints,
        shape_constraints=shape_constraints,
        size_constraints=size_constraints,
    )
    if geometry_types:
        mentions.append(
            {
                "source_text": clean_fragment(description),
                "mention_role": "unknown",
                "category_guess": canonical_category(target_category),
                "numeric_constraints": numeric_constraints,
                "shape_constraints": shape_constraints,
                "size_constraints": size_constraints,
                "geometry_types": geometry_types,
            }
        )

    return mentions


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def call_llm_extract(description: str, target_category: str, config: ExtractorConfig) -> Optional[Dict[str, Any]]:
    if not config.api_key or not config.base_url:
        return None

    try:
        from openai import OpenAI
        import httpx
    except Exception:
        return None

    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        http_client=httpx.Client(trust_env=False, timeout=30.0),
    )
    prompt = LLM_PROMPT_TEMPLATE.format(
        target_category=target_category,
        description=description,
    )

    for attempt in range(config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                top_p=0.8,
                timeout=30,
            )
            text = response.choices[0].message.content or ""
            obj = extract_json_object(text)
            if obj is not None:
                return obj
        except Exception:
            pass
        time.sleep(config.sleep_seconds)
    return None


def normalize_mentions(
    mentions: List[Dict[str, Any]],
    description: str,
    target_category: str,
    method: str,
) -> Dict[str, Any]:
    normalized_mentions: List[Dict[str, Any]] = []
    all_types: List[str] = []
    role_counter: Dict[str, int] = {"main": 0, "other": 0, "unknown": 0}

    for idx, raw in enumerate(mentions):
        numeric_constraints = raw.get("numeric_constraints") or {}
        normalized_numeric = {
            "height_m": float(numeric_constraints["height_m"]) if numeric_constraints.get("height_m") is not None else None,
            "length_m": float(numeric_constraints["length_m"]) if numeric_constraints.get("length_m") is not None else None,
            "width_m": float(numeric_constraints["width_m"]) if numeric_constraints.get("width_m") is not None else None,
            "stories": float(numeric_constraints["stories"]) if numeric_constraints.get("stories") is not None else None,
        }
        shape_constraints = unique_keep_order([str(x).strip().lower() for x in raw.get("shape_constraints") or [] if str(x).strip()])
        size_constraints = unique_keep_order([str(x).strip().lower() for x in raw.get("size_constraints") or [] if str(x).strip()])
        geometry_types = geometry_types_from_parts(
            numeric_constraints=normalized_numeric,
            shape_constraints=shape_constraints,
            size_constraints=size_constraints,
        )
        if not geometry_types:
            continue
        mention_role = str(raw.get("mention_role") or "unknown").strip().lower()
        if mention_role not in {"main", "other", "unknown"}:
            mention_role = "unknown"
        normalized_mentions.append(
            {
                "mention_id": idx,
                "source_text": clean_fragment(raw.get("source_text") or description),
                "mention_role": mention_role,
                "category_guess": canonical_category(raw.get("category_guess") or target_category),
                "numeric_constraints": normalized_numeric,
                "shape_constraints": shape_constraints,
                "size_constraints": size_constraints,
                "geometry_types": geometry_types,
            }
        )
        all_types.extend(geometry_types)
        role_counter[mention_role] = role_counter.get(mention_role, 0) + 1

    return {
        "method": method,
        "has_geometry": bool(normalized_mentions),
        "geometry_types": unique_keep_order(all_types),
        "mention_role_counter": role_counter,
        "mentions": normalized_mentions,
    }


def extract_geometry_for_item(item: Dict[str, Any], config: ExtractorConfig) -> Dict[str, Any]:
    description = str(item.get("description", "")).strip()
    target_category = str(item.get("object_name", "")).strip()

    llm_obj: Optional[Dict[str, Any]] = None
    if config.use_llm:
        llm_obj = call_llm_extract(description, target_category, config)
    if llm_obj and isinstance(llm_obj.get("mentions"), list):
        geometry_extraction = normalize_mentions(
            mentions=llm_obj["mentions"],
            description=description,
            target_category=target_category,
            method="llm",
        )
    else:
        mentions = extract_mentions_regex(description, target_category)
        geometry_extraction = normalize_mentions(
            mentions=mentions,
            description=description,
            target_category=target_category,
            method="regex",
        )

    out = dict(item)
    out["geometry_extraction"] = geometry_extraction
    return out


def process_file(input_path: str, output_path: str, config: ExtractorConfig, limit: int = -1) -> Dict[str, Any]:
    items = load_items(input_path)
    ensure_parent_dir(output_path)

    total = 0
    with_geometry = 0
    geometry_type_counter: Dict[str, int] = {}

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            if limit >= 0 and idx >= limit:
                break
            out = extract_geometry_for_item(item, config)
            geometry_extraction = out.get("geometry_extraction") or {}
            total += 1
            if geometry_extraction.get("has_geometry"):
                with_geometry += 1
            for geometry_type in geometry_extraction.get("geometry_types") or []:
                geometry_type_counter[geometry_type] = geometry_type_counter.get(geometry_type, 0) + 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary = {
        "input_path": input_path,
        "output_path": output_path,
        "total_items": total,
        "with_geometry": with_geometry,
        "without_geometry": total - with_geometry,
        "geometry_type_counter": geometry_type_counter,
        "method": "llm" if config.use_llm else "regex",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract geometry mentions from CityAnchor descriptions.")
    parser.add_argument("--task", choices=["ND", "NO", "BOTH"], default="ND")
    parser.add_argument("--input", default="", help="Optional custom input path.")
    parser.add_argument("--output", default="", help="Optional custom output path.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--api-key", default=os.environ.get("CITYANCHOR_LLM_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("CITYANCHOR_LLM_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("CITYANCHOR_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_llm and (not args.api_key or not args.base_url):
        defaults = load_default_llm_config_from_source()
        if not args.api_key:
            args.api_key = defaults.get("api_key", "")
        if not args.base_url:
            args.base_url = defaults.get("base_url", "")
        if args.model == os.environ.get("CITYANCHOR_LLM_MODEL", "deepseek-chat"):
            args.model = defaults.get("model", args.model)

    config = ExtractorConfig(
        use_llm=args.use_llm,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )

    summaries: List[Dict[str, Any]] = []
    for task_name, task_cfg in iter_tasks(args.task):
        input_path = args.input or task_cfg["input"]
        output_path = args.output or task_cfg["output"]
        if args.task == "BOTH" and not args.input and not args.output:
            input_path = task_cfg["input"]
            output_path = task_cfg["output"]

        summary = process_file(
            input_path=input_path,
            output_path=output_path,
            config=config,
            limit=args.limit,
        )
        summary["task"] = task_name
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(summaries) > 1:
        total_items = sum(x["total_items"] for x in summaries)
        with_geometry = sum(x["with_geometry"] for x in summaries)
        merged_counter: Dict[str, int] = {}
        for summary in summaries:
            for key, value in summary["geometry_type_counter"].items():
                merged_counter[key] = merged_counter.get(key, 0) + int(value)
        print(
            json.dumps(
                {
                    "task": "BOTH",
                    "total_items": total_items,
                    "with_geometry": with_geometry,
                    "without_geometry": total_items - with_geometry,
                    "geometry_type_counter": merged_counter,
                    "method": "llm" if config.use_llm else "regex",
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

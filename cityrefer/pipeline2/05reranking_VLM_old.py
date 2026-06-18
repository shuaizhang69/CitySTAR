from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


DEFAULT_METADATA_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0324.jsonl"
DEFAULT_HGMATCH_INPUT = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2.json"
DEFAULT_CONTEXT_IMAGE_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/Context_image_new2"
DEFAULT_OUTPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2_api_rerank_top5.json"
DEFAULT_MODEL_NAME = "qwen3.6-plus"

TASK_CONFIGS = {
    "ND": {
        "metadata_jsonl": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_ND_0324.jsonl",
        "hgmatch_input": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_ND_0421_stage1_candidates_hgmatch_v2_api_rerank_top10.json",
    },
    "NO": {
        "metadata_jsonl": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/meta_data/0311data/CityRefer_val_NO_0324.jsonl",
        "hgmatch_input": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_NO_0421_stage1_candidates_hgmatch_v2.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/Cityrefer/0412/hypergraph_match/CityRefer_val_NO_0421_stage1_candidates_hgmatch_v2_api_rerank_top10.json",
    },
}

PROMPT_TEMPLATE = """
You are a remote sensing referring expression reranker.

Task:
- A query description refers to one target object in an aerial image.
- You are given one candidate image crop/context image for one candidate object.
- Judge how well this candidate image matches the query description.

Scoring rules:
- Focus on whether the object and its surrounding context in the candidate image match the description.
- Consider category, color, shape, local context, and relative nearby objects/landmarks when visible.
- Output a higher score only when the candidate image is clearly more relevant to the query.
- Use an integer score from 0 to 100.

Output format:
Return only one JSON object:
{{"score": <integer 0-100>, "reason": "<short reason>"}}

Query description:
{description}
""".strip()


def load_json_or_jsonl(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        items: List[dict] = []
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


def load_metadata_items(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_query_index(metadata_items: List[dict]) -> Dict[Tuple[str, str, str], dict]:
    index: Dict[Tuple[str, str, str], dict] = {}
    for item in metadata_items:
        scene_id = str(item.get("scene_id", ""))
        object_id = str(item.get("object_id", ""))
        ann_id = str(item.get("ann_id", ""))
        key = (
            scene_id,
            object_id,
            ann_id,
        )
        index[key] = item
        if ann_id in ("", "0"):
            index[(scene_id, object_id, "0")] = item
            index[(scene_id, object_id, "")] = item
    return index


def build_query_text(item: dict) -> str:
    description = str(item.get("description", "")).strip()
    if description:
        return description

    object_name = str(item.get("object_name", "")).strip()
    construction = item.get("construction") or []
    parts: List[str] = []
    for obj in construction:
        segs: List[str] = []
        category = str(obj.get("category2") or obj.get("category") or "").strip()
        color = str(obj.get("color") or "").strip()
        identity_feature = str(obj.get("identity_feature") or "").strip()
        landmark = str(obj.get("landmark") or "").strip()
        if category:
            segs.append(category)
        if color:
            segs.append(f"color {color}")
        if identity_feature:
            segs.append(identity_feature)
        if landmark:
            segs.append(f"near {landmark}")
        if segs:
            parts.append(", ".join(segs))
    if parts:
        return f"Find the referred {object_name}. " + " | ".join(parts[:6])
    return f"Find the referred {object_name}."


def _candidate_id_sort_token(candidate_id: str) -> Tuple[int, str]:
    if candidate_id.isdigit():
        return int(candidate_id), ""
    return 10**18, candidate_id


def resolve_top_candidates(item: dict, topk: int) -> List[str]:
    ranked = (
        item.get("hgmatch_reranked_candidates")
        or item.get("top5_ids")
        or item.get("top10_ids")
        or item.get("top20_ids")
    )

    if ranked:
        return [str(x) for x in ranked[:topk]]

    # 兼容 04_hypergraph_matchv2.py 输出：按融合分降序、再按 bbox_id 稳定排序。
    fused_scores = item.get("all_fused_scores")
    if isinstance(fused_scores, dict) and fused_scores:
        sorted_ids = sorted(
            ((str(k), float(v)) for k, v in fused_scores.items()),
            key=lambda kv: (-kv[1], _candidate_id_sort_token(kv[0])),
        )
        return [x[0] for x in sorted_ids[:topk]]

    all_scores = item.get("all_scores")
    if isinstance(all_scores, dict) and all_scores:
        sorted_ids = sorted(
            ((str(k), float(v)) for k, v in all_scores.items()),
            key=lambda kv: (-kv[1], _candidate_id_sort_token(kv[0])),
        )
        return [x[0] for x in sorted_ids[:topk]]

    ranked = item.get("candidates") or []
    return [str(x) for x in ranked[:topk]]


def resolve_image_path(context_image_dir: str, scene_id: str, object_id: str) -> Optional[str]:
    scene_dir = Path(context_image_dir) / scene_id
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = scene_dir / f"{scene_id}_{object_id}{ext}"
        if path.exists():
            return str(path)
    return None


def encode_image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def parse_json_response(text: str) -> Tuple[int, str]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return 0, f"unparseable response: {text[:200]}"
        obj = json.loads(match.group(0))

    score = obj.get("score", 0)
    try:
        score = int(float(score))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))
    reason = str(obj.get("reason", "")).strip()
    return score, reason


def score_candidate_with_api(
    client: OpenAI,
    model_name: str,
    image_path: str,
    query_text: str,
    enable_thinking: bool,
    max_retries: int,
    sleep_seconds: float,
) -> Tuple[int, str, str]:
    prompt_text = PROMPT_TEMPLATE.format(description=query_text)
    image_url = encode_image_to_data_url(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    last_error = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"enable_thinking": enable_thinking},
            )
            raw_text = response.choices[0].message.content.strip()
            score, reason = parse_json_response(raw_text)
            return score, reason, raw_text
        except Exception as e:
            last_error = str(e)
            if attempt + 1 < max_retries:
                time.sleep(sleep_seconds)
    return 0, f"api_error: {last_error}", ""


def rerank_item(
    item: dict,
    query_index: Dict[Tuple[str, str, str], dict],
    client: OpenAI,
    model_name: str,
    context_image_dir: str,
    topk: int,
    enable_thinking: bool,
    max_retries: int,
    sleep_seconds: float,
) -> dict:
    scene_id = str(item.get("scene_id", ""))
    object_id = str(item.get("object_id", ""))
    ann_id = str(item.get("ann_id", ""))
    query_item = (
        query_index.get((scene_id, object_id, ann_id))
        or query_index.get((scene_id, object_id, "0"))
        or query_index.get((scene_id, object_id, ""))
    )

    result = dict(item)
    result["api_rerank_status"] = "ok"
    result["api_rerank_topk"] = topk
    result["api_rerank_query_text"] = ""
    result["api_rerank_top5_scores"] = []
    result["api_rerank_top5_candidates"] = []
    result["api_rerank_top1"] = None

    if not query_item:
        result["api_rerank_status"] = "missing_query_text"
        return result

    query_text = build_query_text(query_item)
    result["api_rerank_query_text"] = query_text

    top_candidates = resolve_top_candidates(item, topk)
    if not top_candidates:
        result["api_rerank_status"] = "no_candidates"
        return result

    scored = []
    for candidate_id in top_candidates:
        image_path = resolve_image_path(context_image_dir, scene_id, candidate_id)
        if not image_path:
            scored.append(
                {
                    "candidate_bbox_id": candidate_id,
                    "score": -1,
                    "reason": "missing_image",
                    "image_path": None,
                    "raw_response": "",
                }
            )
            continue

        score, reason, raw_text = score_candidate_with_api(
            client=client,
            model_name=model_name,
            image_path=image_path,
            query_text=query_text,
            enable_thinking=enable_thinking,
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        scored.append(
            {
                "candidate_bbox_id": candidate_id,
                "score": score,
                "reason": reason,
                "image_path": image_path,
                "raw_response": raw_text,
            }
        )

    scored.sort(
        key=lambda x: (
            -x["score"],
            0 if x["candidate_bbox_id"] == object_id else 1,
            _candidate_id_sort_token(str(x["candidate_bbox_id"])),
        )
    )

    result["api_rerank_top5_scores"] = scored
    result["api_rerank_top5_candidates"] = [x["candidate_bbox_id"] for x in scored]
    result["api_rerank_top1"] = scored[0]["candidate_bbox_id"] if scored else None
    return result


def compute_hit_stats(items: List[dict]) -> Tuple[int, int]:
    top1 = 0
    top3 = 0
    for item in items:
        gt = str(item.get("object_id", ""))
        ranked = [str(x) for x in item.get("api_rerank_top5_candidates", [])]
        if gt in ranked[:1]:
            top1 += 1
        if gt in ranked[:3]:
            top3 += 1
    return top1, top3


def compute_single_hits(item: dict) -> Tuple[int, int]:
    gt = str(item.get("object_id", ""))
    ranked = [str(x) for x in item.get("api_rerank_top5_candidates", [])]
    top1 = 1 if gt in ranked[:1] else 0
    top3 = 1 if gt in ranked[:3] else 0
    return top1, top3


def build_client(api_base: str, api_key: str) -> OpenAI:
    final_api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or api_key
    )
    return OpenAI(api_key=final_api_key, base_url=api_base)


def build_item_key(item: dict) -> Tuple[str, str, str]:
    return (
        str(item.get("scene_id", "")),
        str(item.get("object_id", "")),
        str(item.get("ann_id", "")),
    )


def load_resume_results(path: str) -> List[dict]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[WARN] Resume file does not exist, skip: {path}")
        return []
    return load_json_or_jsonl(path)


def run_task(
    task_name: str,
    metadata_jsonl: str,
    hgmatch_input: str,
    output: str,
    client: OpenAI,
    args: argparse.Namespace,
    resume_from: str = "",
) -> None:
    print(f"\n===== Running task: {task_name} =====")
    print(f"metadata: {metadata_jsonl}")
    print(f"hgmatch:  {hgmatch_input}")
    print(f"output:   {output}")
    if resume_from:
        print(f"resume:   {resume_from}")

    metadata_items = load_metadata_items(metadata_jsonl)
    query_index = build_query_index(metadata_items)
    hgmatch_items = load_json_or_jsonl(hgmatch_input)

    existing_results = load_resume_results(resume_from)
    existing_by_key_all: Dict[Tuple[str, str, str], dict] = {
        build_item_key(item): item for item in existing_results
    }
    hgmatch_key_set = {build_item_key(x) for x in hgmatch_items}
    existing_by_key: Dict[Tuple[str, str, str], dict] = {
        k: v for k, v in existing_by_key_all.items() if k in hgmatch_key_set
    }
    resumed_results_for_task = list(existing_by_key.values())
    resumed_top1, resumed_top3 = compute_hit_stats(resumed_results_for_task)

    pending_items = [x for x in hgmatch_items if build_item_key(x) not in existing_by_key]
    if args.limit > 0:
        pending_items = pending_items[: args.limit]

    print(
        f"Total={len(hgmatch_items)}, resumed={len(existing_by_key)}, "
        f"pending={len(pending_items)} (limit={args.limit})"
    )

    processed_new = []
    running_top1 = 0
    running_top3 = 0
    for item in tqdm(pending_items, desc=f"API reranking {task_name}"):
        start_time = time.time()
        result = rerank_item(
            item=item,
            query_index=query_index,
            client=client,
            model_name=args.model,
            context_image_dir=args.context_image_dir,
            topk=args.topk,
            enable_thinking=args.enable_thinking,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        elapsed = time.time() - start_time
        processed_new.append(result)

        hit1, hit3 = compute_single_hits(result)
        running_top1 += hit1
        running_top3 += hit3
        current_n = len(processed_new)
        cumulative_n = len(existing_by_key) + current_n
        cumulative_top1 = resumed_top1 + running_top1
        cumulative_top3 = resumed_top3 + running_top3
        print(
            f"[{current_n}/{len(pending_items)}] "
            f"scene={result.get('scene_id')} obj={result.get('object_id')} ann={result.get('ann_id')} "
            f"status={result.get('api_rerank_status')} "
            f"time={elapsed:.2f}s "
            f"top1={cumulative_top1/cumulative_n:.4f} "
            f"top3={cumulative_top3/cumulative_n:.4f}"
        )

    merged_by_key = dict(existing_by_key)
    for item in processed_new:
        merged_by_key[build_item_key(item)] = item

    final_results = []
    for item in hgmatch_items:
        key = build_item_key(item)
        if key in merged_by_key:
            final_results.append(merged_by_key[key])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    top1, top3 = compute_hit_stats(final_results)
    total = len(final_results)
    print(f"Saved to: {output}")
    print(f"Queries: {total}")
    print(f"Top1 hit: {top1}/{total} ({(top1 / total * 100) if total else 0:.2f}%)")
    print(f"GT in Top3 hit: {top3}/{total} ({(top3 / total * 100) if total else 0:.2f}%)")


def main() -> None:
    p = argparse.ArgumentParser(description="Rerank candidates with query text and candidate images via OpenAI-compatible API.")
    p.add_argument("--task", choices=["ND", "NO", "ALL"], default="ND", help="Run ND, NO, or both")
    p.add_argument("--metadata-jsonl", default="", help="Query text source jsonl (used in single-task mode)")
    p.add_argument("--hgmatch-input", default="", help="Hypergraph match result (.json or .jsonl, used in single-task mode)")
    p.add_argument("--context-image-dir", default=DEFAULT_CONTEXT_IMAGE_DIR, help="Directory containing context images")
    p.add_argument("--output", default="", help="Output JSON path (used in single-task mode)")
    p.add_argument("--resume-from", default="", help="Existing rerank result to continue from")
    p.add_argument("--topk", type=int, default=10, help="Number of top candidates to rerank")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Vision-language model name")
    p.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get(
            "OPENAI_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        help="OpenAI 兼容接口 Base URL；默认 DashScope 兼容模式（与 05step2_VLM_new_thinking_pro 一致）",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""),
        help="API Key；优先环境变量 DASHSCOPE_API_KEY，其次 OPENAI_API_KEY",
    )
    p.add_argument("--enable-thinking", action="store_true", help="Pass enable_thinking=True in extra_body")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="Only process first N queries for quick test")
    args = p.parse_args()

    client = build_client(args.api_base, args.api_key)
    if args.task in ("ND", "NO"):
        task_cfg = TASK_CONFIGS.get(args.task, {})
        metadata_jsonl = args.metadata_jsonl or task_cfg.get("metadata_jsonl", DEFAULT_METADATA_JSONL)
        hgmatch_input = args.hgmatch_input or task_cfg.get("hgmatch_input", DEFAULT_HGMATCH_INPUT)
        output = args.output or task_cfg.get("output", DEFAULT_OUTPUT_JSON)
        run_task(
            task_name=args.task,
            metadata_jsonl=metadata_jsonl,
            hgmatch_input=hgmatch_input,
            output=output,
            client=client,
            args=args,
            resume_from=args.resume_from,
        )
        return

    nd_cfg = TASK_CONFIGS["ND"]
    no_cfg = TASK_CONFIGS["NO"]
    run_task(
        task_name="ND",
        metadata_jsonl=nd_cfg["metadata_jsonl"],
        hgmatch_input=nd_cfg["hgmatch_input"],
        output=nd_cfg["output"],
        client=client,
        args=args,
        resume_from=args.resume_from,
    )
    run_task(
        task_name="NO",
        metadata_jsonl=no_cfg["metadata_jsonl"],
        hgmatch_input=no_cfg["hgmatch_input"],
        output=no_cfg["output"],
        client=client,
        args=args,
        resume_from="",
    )


if __name__ == "__main__":
    main()

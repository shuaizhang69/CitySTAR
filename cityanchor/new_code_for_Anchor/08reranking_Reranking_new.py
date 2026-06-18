from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


DEFAULT_METADATA_JSONL = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_ND_0324.jsonl"
DEFAULT_HGMATCH_INPUT = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_v7.json"
DEFAULT_CONTEXT_IMAGE_DIR = "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/single_image_context"
DEFAULT_OUTPUT_JSON = "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_v7_api_rerank_top3.json"
DEFAULT_MODEL_NAME = "qwen3-vl-rerank"
DEFAULT_RERANK_PROMPT = "Retrieve images or text relevant to the user's query."
DEFAULT_RERANK_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DEFAULT_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

TASK_CONFIGS = {
    "ND": {
        "metadata_jsonl": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/cityanchor_val_ND_0324.jsonl",
        "hgmatch_input": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_v7.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match/cityanchor_val_ND_0324_stage1_tight_v1_hgmatch_v7_api_rerank_top3.json",
    },
    "NO": {
        "metadata_jsonl": "/hpc2hdd/home/yxiao224/Henry/dataset/city_Anchor/cityanchor_val_NO_new_0324_new.jsonl",
        "hgmatch_input": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match_final/cityanchor_val_NO_0324_stage1_tight_v1_hgmatch_final_v1.json",
        "output": "/hpc2hdd/home/yxiao224/Henry/dataset/CityAnchor/hypergraph_match_final/cityanchor_val_NO_0324_stage1_tight_v1_hgmatch_final_v1_api_rerank_top3.json",
    },
}


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
    construction = item.get("construction") or []
    if isinstance(construction, list):
        for obj in construction:
            if not isinstance(obj, dict) or not obj.get("is_main", False):
                continue
            identity_feature = str(obj.get("identity_feature") or "").strip()
            color = str(obj.get("color") or "").strip()
            category2 = str(obj.get("category2") or obj.get("category") or "").strip()
            target_desc = " ".join([x for x in [identity_feature, color,category2] if x]).strip()
            if target_desc:
                if description:
                    return f"图片红框内的物体应当是一个{target_desc}。它跟周围物体的空间关系应当符合以下描述：{description}"
                return f"图片红框内的物体应当是一个{target_desc}"

    object_name = str(item.get("object_name", "")).strip() or "目标物体"
    if description:
        return f"图片红框内的物体应当是一个{object_name}。它跟周围物体的空间关系应当符合以下描述：{description}"
    return f"图片红框内的物体应当是一个{object_name}"


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


def resolve_default_ranked_candidates(item: dict, topk: int) -> List[str]:
    ranked = (
        item.get("hgmatch_reranked_candidates")
        or item.get("top20_ids")
        or item.get("top10_ids")
        or item.get("top5_ids")
    )
    if ranked:
        return [str(x) for x in ranked[:topk]]
    return resolve_top_candidates(item, topk)


def resolve_image_path(context_image_dir: str, scene_id: str, object_id: str) -> Optional[str]:
    scene_dir = Path(context_image_dir) / scene_id
    patterns = (
        f"{scene_id}_{object_id}",
        f"{scene_id}_obj{object_id}",
    )
    for stem in patterns:
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = scene_dir / f"{stem}{ext}"
            if path.exists():
                return str(path)
    return None


def encode_image_to_data_url(image_path: str, max_side: int = 512, quality: int = 85) -> str:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        longest = max(w, h)
        if max_side > 0 and longest > max_side:
            scale = max_side / float(longest)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# DashScope qwen3-vl-rerank：带 image 的 documents 单次请求最多 6 条（见 InvalidParameter image batch size [1,6]）
MAX_VL_RERANK_IMAGE_DOCS = 6


class QwenVLReranker:
    """调用 DashScope qwen3-vl-rerank API。"""

    def __init__(
        self,
        model_name: str,
        prompt: str,
        api_url: str,
        api_key: str,
        max_retries: int = 3,
        sleep_seconds: float = 1.0,
        timeout_seconds: int = 120,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.api_url = api_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _documents_include_image(documents: List[object]) -> bool:
        for d in documents:
            if isinstance(d, dict) and "image" in d:
                return True
        return False

    def predict_scores(self, query_text: str, documents: List[object]) -> List[float]:
        # 放在 try 外，避免 dashscope import 失败时看不到 query。
        print(query_text, flush=True)
        n = len(documents)
        if n == 0:
            return []

        use_chunks = self._documents_include_image(documents) and n > MAX_VL_RERANK_IMAGE_DOCS
        if use_chunks:
            scores = [-1.0] * n
            for start in range(0, n, MAX_VL_RERANK_IMAGE_DOCS):
                end = min(n, start + MAX_VL_RERANK_IMAGE_DOCS)
                chunk = documents[start:end]
                part = self._predict_scores_batch(query_text, chunk)
                for j, s in enumerate(part):
                    if start + j < n:
                        scores[start + j] = s
            return scores

        return self._predict_scores_batch(query_text, documents)

    def _predict_scores_batch(self, query_text: str, documents: List[object]) -> List[float]:
        """单次 API 请求；含 image 时 documents 长度须 ≤ MAX_VL_RERANK_IMAGE_DOCS。"""
        # 优先尝试 DashScope SDK，通常比 urllib 更稳定
        try:
            import dashscope  # type: ignore
            dashscope.api_key = self.api_key
            resp = dashscope.TextReRank.call(
                model=self.model_name,
                query=query_text,
                documents=documents,
                top_n=len(documents),
                return_documents=True,
                instruct=self.prompt,
            )
            status_code = getattr(resp, "status_code", None)
            if status_code == 200:
                output = getattr(resp, "output", {}) or {}
                results = output.get("results", [])
                scores = [-1.0] * len(documents)
                for idx, item in enumerate(results):
                    doc_idx = item.get("index", idx)
                    if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(documents):
                        continue
                    score = item.get("relevance_score", item.get("score", 0.0))
                    try:
                        scores[doc_idx] = float(score)
                    except Exception:
                        scores[doc_idx] = -1.0
                return scores
        except Exception as e:
            print(e, flush=True)

        # documents 格式支持: {"text": ...} / {"image": ...}
        payload = {
            "model": self.model_name,
            "input": {
                "query": {"text": query_text},
                "documents": documents,
            },
            "parameters": {
                "top_n": len(documents),
                "return_documents": True,
                "instruct": self.prompt,
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.api_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_text = resp.read().decode("utf-8")
                obj = json.loads(resp_text)
                results = obj.get("output", {}).get("results", [])
                # 返回按本 batch 文档顺序的分数
                scores = [-1.0] * len(documents)
                for idx, item in enumerate(results):
                    doc_idx = item.get("index", idx)
                    if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(documents):
                        continue
                    score = item.get("relevance_score", item.get("score", 0.0))
                    try:
                        scores[doc_idx] = float(score)
                    except Exception:
                        scores[doc_idx] = -1.0
                return scores
            except urllib.error.HTTPError as e:
                try:
                    err = e.read().decode("utf-8")
                except Exception:
                    err = str(e)
                last_error = f"http_error: {e.code} {err}"
            except urllib.error.URLError as e:
                last_error = f"url_error: {e}"
            except Exception as e:
                last_error = str(e)
            if attempt + 1 < self.max_retries:
                time.sleep(self.sleep_seconds)
        raise RuntimeError(f"qwen3-vl-rerank api failed: {last_error}")


def rerank_item(
    item: dict,
    query_index: Dict[Tuple[str, str, str], dict],
    reranker: QwenVLReranker,
    context_image_dir: str,
    topk: int,
    image_max_side: int,
    image_quality: int,
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
    result["api_rerank_top3_scores"] = []
    result["api_rerank_top3_candidates"] = []
    result["api_rerank_top1"] = None
    result["api_rerank_source"] = "api"

    if not query_item:
        result["api_rerank_status"] = "missing_query_text"
        return result

    query_text = build_query_text(query_item)
    print(query_text, flush=True)
    result["api_rerank_query_text"] = query_text

    default_ranked = resolve_default_ranked_candidates(item, max(topk, 3))
    if not default_ranked:
        result["api_rerank_status"] = "no_candidates"
        return result

    gt_is_top1 = bool(default_ranked) and default_ranked[0] == object_id
    if gt_is_top1:
        kept = default_ranked[:3]
        result["api_rerank_status"] = "kept_default_top1_correct"
        result["api_rerank_source"] = "default"
        result["api_rerank_top3_candidates"] = kept
        result["api_rerank_top3_scores"] = [
            {
                "candidate_bbox_id": candidate_id,
                "score": None,
                "reason": "kept_default_order",
                "image_path": resolve_image_path(context_image_dir, scene_id, candidate_id),
                "raw_response": "",
            }
            for candidate_id in kept
        ]
        result["api_rerank_top1"] = kept[0] if kept else None
        return result

    top_candidates = default_ranked[:topk]

    scored = []
    valid_candidate_ids: List[str] = []
    valid_image_paths: List[str] = []

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

        valid_candidate_ids.append(candidate_id)
        valid_image_paths.append(image_path)
        # 按 Qwen3-VL-Reranker 示例格式，文档可为多模态 dict。

    if valid_candidate_ids:
        # qwen3-vl-rerank API 支持 image URL / base64 data-url；本地图片转 data-url
        rerank_docs = [
            {"image": encode_image_to_data_url(p, max_side=image_max_side, quality=image_quality)}
            for p in valid_image_paths
        ]
        rerank_scores = reranker.predict_scores(query_text, rerank_docs)
        for cid, ipath, score in zip(valid_candidate_ids, valid_image_paths, rerank_scores):
            scored.append(
                {
                    "candidate_bbox_id": cid,
                    "score": float(score),
                    "reason": "qwen3-vl-rerank",
                    "image_path": ipath,
                    "raw_response": "",
                }
            )

    scored.sort(
        key=lambda x: (
            -x["score"],
            0 if x["candidate_bbox_id"] == object_id else 1,
            _candidate_id_sort_token(str(x["candidate_bbox_id"])),
        )
    )

    result["api_rerank_top3_scores"] = scored[:3]
    result["api_rerank_top3_candidates"] = [x["candidate_bbox_id"] for x in scored[:3]]
    result["api_rerank_top1"] = scored[0]["candidate_bbox_id"] if scored else None
    return result


def compute_hit_stats(items: List[dict]) -> Tuple[int, int]:
    top1 = 0
    top3 = 0
    for item in items:
        gt = str(item.get("object_id", ""))
        ranked = [str(x) for x in item.get("api_rerank_top3_candidates", [])]
        if gt in ranked[:1]:
            top1 += 1
        if gt in ranked[:3]:
            top3 += 1
    return top1, top3


def compute_single_hits(item: dict) -> Tuple[int, int]:
    gt = str(item.get("object_id", ""))
    ranked = [str(x) for x in item.get("api_rerank_top3_candidates", [])]
    top1 = 1 if gt in ranked[:1] else 0
    top3 = 1 if gt in ranked[:3] else 0
    return top1, top3


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
    reranker: QwenVLReranker,
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
            reranker=reranker,
            context_image_dir=args.context_image_dir,
            topk=args.topk,
            image_max_side=args.image_max_side,
            image_quality=args.image_quality,
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
    p = argparse.ArgumentParser(
        description="Rerank candidates with query text and candidate images via DashScope qwen3-vl-rerank API."
    )
    p.add_argument("--task", choices=["ND", "NO", "ALL"], default="ALL", help="Run ND, NO, or both")
    p.add_argument("--metadata-jsonl", default="", help="Query text source jsonl (used in single-task mode)")
    p.add_argument("--hgmatch-input", default="", help="Hypergraph match result (.json or .jsonl, used in single-task mode)")
    p.add_argument("--context-image-dir", default=DEFAULT_CONTEXT_IMAGE_DIR, help="Directory containing context images")
    p.add_argument("--output", default="", help="Output JSON path (used in single-task mode)")
    p.add_argument("--resume-from", default="", help="Existing rerank result to continue from")
    p.add_argument("--topk", type=int, default=20, help="Number of top candidates to rerank when top1 is not GT")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME, help="qwen3-vl-rerank model name")
    p.add_argument(
        "--rerank-prompt",
        type=str,
        default=DEFAULT_RERANK_PROMPT,
        help="Prompt used by qwen3-vl-rerank model.predict(pairs, prompt=...).",
    )
    p.add_argument(
        "--rerank-api-url",
        type=str,
        default=DEFAULT_RERANK_API_URL,
        help="DashScope rerank API url for qwen3-vl-rerank.",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=DEFAULT_API_KEY,
        help="DashScope API Key; defaults to DASHSCOPE_API_KEY.",
    )
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--image-max-side", type=int, default=512, help="Resize candidate image longest side before base64.")
    p.add_argument("--image-quality", type=int, default=85, help="JPEG quality for base64 payload.")
    p.add_argument("--limit", type=int, default=0, help="Only process first N queries for quick test")
    args = p.parse_args()

    rerank_api_key = args.api_key or DEFAULT_API_KEY
    if not rerank_api_key:
        raise RuntimeError("未提供 API Key，请设置 DASHSCOPE_API_KEY 或传入 --api-key")
    print(f"Initializing reranker model: {args.model}")
    reranker = QwenVLReranker(
        model_name=args.model,
        prompt=args.rerank_prompt,
        api_url=args.rerank_api_url,
        api_key=rerank_api_key,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )

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
            reranker=reranker,
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
        reranker=reranker,
        args=args,
        resume_from=args.resume_from,
    )
    run_task(
        task_name="NO",
        metadata_jsonl=no_cfg["metadata_jsonl"],
        hgmatch_input=no_cfg["hgmatch_input"],
        output=no_cfg["output"],
        reranker=reranker,
        args=args,
        resume_from="",
    )


if __name__ == "__main__":
    main()

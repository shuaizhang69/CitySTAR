import argparse
import json
from collections import Counter, defaultdict


def canonicalize_category(category: str) -> str:
    category = (category or "").strip().lower()
    alias = {
        "car": "vehicle",
        "cars": "vehicle",
        "tree": "highvegetation",
        "trees": "highvegetation",
        "bush": "highvegetation",
        "bushes": "highvegetation",
        "wall": "fence",
        "walls": "fence",
        "gate": "fence",
        "gates": "fence",
    }
    return alias.get(category, category)


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_desc_map(path: str) -> dict[tuple[str, str, str], dict]:
    out = {}
    for item in load_jsonl(path):
        key = (
            str(item["scene_id"]),
            str(item["object_id"]),
            str(item.get("ann_id", 0)),
        )
        out[key] = item
    return out


def load_bbox_map(path: str) -> dict[tuple[str, str, str, str], dict]:
    out = {}
    for item in load_jsonl(path):
        query_object_id = str(
            item.get("query_object_id", item.get("object_id", item["bbox_id"]))
        )
        key = (
            str(item["scene_id"]),
            query_object_id,
            str(item["bbox_id"]),
            str(item.get("ann_id", 0)),
        )
        out[key] = item
    return out


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ratio(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return numer / denom * 100.0


def summarize_desc(desc_map: dict[tuple[str, str, str], dict]) -> None:
    edge_counter = Counter()
    rel_counter = Counter()
    node_counter = Counter()
    main_counter = Counter()
    other_counter = Counter()

    for item in desc_map.values():
        hg = item.get("hypergraph", {})
        edge_counter[hg.get("edge_count", len(hg.get("edges", [])))] += 1
        node_counter[hg.get("node_count", len(hg.get("nodes", [])))] += 1
        main_counter[hg.get("main_category", "")] += 1
        for node in hg.get("nodes", []):
            if node.get("is_main"):
                continue
            other_counter[node.get("category", "")] += 1
        for edge in hg.get("edges", []):
            rel_counter[edge.get("relation", "")] += 1

    print("\n[Description Graph]")
    print("  edge_count dist:", dict(edge_counter.most_common()))
    print("  main categories:", dict(main_counter.most_common()))
    print("  other categories:", dict(other_counter.most_common()))
    print("  relation types:", dict(rel_counter.most_common()))


def summarize_results(results: list[dict]) -> None:
    matched = [x for x in results if x.get("status") == "matched"]
    top1 = sum(bool(x.get("is_correct")) for x in matched)
    top10 = sum(
        str(x["object_id"]) in {str(v) for v in x.get("top10_ids", [])}
        for x in matched
    )
    top20 = sum(
        str(x["object_id"]) in {str(v) for v in x.get("top20_ids", [])}
        for x in matched
    )
    gt_in_pool = 0
    gt_zero = 0
    all_zero = 0
    gt_not_top20 = 0
    gt_positive_not_top20 = 0
    tie_break_not_top20 = 0

    for item in matched:
        object_id = str(item["object_id"])
        scores = {str(k): float(v) for k, v in (item.get("all_scores") or {}).items()}
        in_top20 = object_id in {str(v) for v in item.get("top20_ids", [])}
        if object_id in scores:
            gt_in_pool += 1
            if scores[object_id] == 0.0:
                gt_zero += 1
            if not in_top20:
                gt_not_top20 += 1
                if scores[object_id] > 0.0:
                    gt_positive_not_top20 += 1
                if item.get("tie_break_used"):
                    tie_break_not_top20 += 1
        if scores and max(scores.values()) == 0.0:
            all_zero += 1

    print("\n[Result Summary]")
    print(f"  total results: {len(results)}")
    print(f"  matched: {len(matched)}")
    print(f"  top1: {top1} ({ratio(top1, len(matched)):.1f}%)")
    print(f"  top10: {top10} ({ratio(top10, len(matched)):.1f}%)")
    print(f"  top20: {top20} ({ratio(top20, len(matched)):.1f}%)")
    print(f"  gt_in_pool: {gt_in_pool}")
    print(f"  gt_zero_score: {gt_zero} ({ratio(gt_zero, gt_in_pool):.1f}%)")
    print(f"  all_zero_queries: {all_zero} ({ratio(all_zero, len(matched)):.1f}%)")
    print(f"  gt_not_top20: {gt_not_top20} ({ratio(gt_not_top20, gt_in_pool):.1f}%)")
    print(
        "  gt_positive_but_not_top20:"
        f" {gt_positive_not_top20} ({ratio(gt_positive_not_top20, gt_not_top20):.1f}%)"
    )
    print(
        f"  tie_break_used_among_gt_not_top20: {tie_break_not_top20}"
        f" ({ratio(tie_break_not_top20, gt_not_top20):.1f}%)"
    )


def summarize_gt_alignment(
    desc_map: dict[tuple[str, str, str], dict],
    bbox_map: dict[tuple[str, str, str, str], dict],
    candidates: list[dict],
) -> None:
    usable = 0
    gt_missing_bbox_graph = 0
    bbox_edge_zero = 0
    required_relation_missing = 0
    target_category_missing = 0
    target_category_over_8 = 0

    for cand in candidates:
        scene_id = str(cand["scene_id"])
        object_id = str(cand["object_id"])
        ann_id = str(cand.get("ann_id", 0))
        desc_item = desc_map.get((scene_id, object_id, ann_id))
        bbox_item = bbox_map.get((scene_id, object_id, object_id, ann_id))
        if desc_item is None:
            continue
        if bbox_item is None:
            gt_missing_bbox_graph += 1
            continue

        usable += 1
        desc_edges = desc_item.get("hypergraph", {}).get("edges", [])
        bbox_edges = bbox_item.get("hypergraph", {}).get("edges", [])
        bbox_nodes = bbox_item.get("hypergraph", {}).get("nodes", [])
        if not bbox_edges:
            bbox_edge_zero += 1

        required_relations = {
            (edge.get("relation") or "").strip()
            for edge in desc_edges
            if (edge.get("relation") or "").strip()
        }
        bbox_relations = {
            (edge.get("relation") or "").strip()
            for edge in bbox_edges
            if (edge.get("relation") or "").strip()
        }
        if required_relations and required_relations.isdisjoint(bbox_relations):
            required_relation_missing += 1

        bbox_cat_counts = defaultdict(int)
        for node in bbox_nodes:
            bbox_cat_counts[canonicalize_category(node.get("category", ""))] += 1

        target_categories = []
        for edge in desc_edges:
            target_category = canonicalize_category(
                (edge.get("info") or {}).get("target_category", "")
            )
            if target_category:
                target_categories.append(target_category)

        if target_categories and any(bbox_cat_counts.get(cat, 0) == 0 for cat in target_categories):
            target_category_missing += 1
        if target_categories and any(bbox_cat_counts.get(cat, 0) > 8 for cat in target_categories):
            target_category_over_8 += 1

    print("\n[GT Graph Alignment]")
    print(f"  usable_gt: {usable}")
    print(f"  gt_missing_bbox_graph: {gt_missing_bbox_graph}")
    print(
        f"  gt_bbox_edge_zero: {bbox_edge_zero}"
        f" ({ratio(bbox_edge_zero, usable):.1f}%)"
    )
    print(
        f"  required_relation_missing: {required_relation_missing}"
        f" ({ratio(required_relation_missing, usable):.1f}%)"
    )
    print(
        f"  target_category_missing: {target_category_missing}"
        f" ({ratio(target_category_missing, usable):.1f}%)"
    )
    print(
        f"  target_category_count_gt_8: {target_category_over_8}"
        f" ({ratio(target_category_over_8, usable):.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize where hypergraph retrieval failures come from."
    )
    parser.add_argument("--desc", required=True, help="desc_hypergraphs jsonl")
    parser.add_argument("--bbox", required=True, help="bbox_hypergraphs jsonl")
    parser.add_argument("--results", required=True, help="matching result json")
    parser.add_argument("--candidates", required=True, help="stage1 candidates jsonl")
    args = parser.parse_args()

    desc_map = load_desc_map(args.desc)
    bbox_map = load_bbox_map(args.bbox)
    results = load_results(args.results)
    candidates = load_jsonl(args.candidates)

    print(f"desc: {args.desc}")
    print(f"bbox: {args.bbox}")
    print(f"results: {args.results}")
    print(f"candidates: {args.candidates}")

    summarize_desc(desc_map)
    summarize_results(results)
    summarize_gt_alignment(desc_map, bbox_map, candidates)


if __name__ == "__main__":
    main()

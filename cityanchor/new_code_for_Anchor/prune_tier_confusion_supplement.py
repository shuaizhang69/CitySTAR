#!/usr/bin/env python3
"""Greedy prune TIER_CONFUSION_SUPPLEMENT; require ND & NO GT recall >= 0.9."""
import importlib.util
import os
import sys

_ST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "01_Semantic_color_in_stage1_support_object_proximity_shrink.py",
)


def load_mod():
    spec = importlib.util.spec_from_file_location("st1", _ST)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def deepcopy_supp(s):
    return {int(k): set(v) for k, v in s.items()}


def feasible(mod, supp, min_recall=0.9):
    mod.TIER_CONFUSION_SUPPLEMENT = deepcopy_supp(supp)
    mod.set_color_stage_runtime(
        color_soft_policy="confusion_supplement",
        tier_half_width=2,
        use_soft_groups=True,
        sparse_pm1_max_pool=None,
    )
    ok = True
    metrics = []
    for path in (mod.DEFAULT_JSONL_ND, mod.DEFAULT_JSONL_NO):
        if not path or not os.path.exists(path):
            continue
        met = mod.run_rolling_eval(
            mod.DEFAULT_BBOX_DIR,
            path,
            mod.DEFAULT_COLOR_JSONL,
            mod.DEFAULT_QUERY_COLOR_JSON,
            quiet=True,
        )
        metrics.append(met)
        r = met["recall_gt_in_candidates"]
        if r is None or r < min_recall:
            ok = False
    return ok, metrics


def combined_avg_cand(metrics):
    s = 0
    n = 0
    for m in metrics:
        s += m["summ_candidates"]
        n += m["count"]
    return s / n if n else 0.0


def main():
    mod = load_mod()
    initial = deepcopy_supp(mod.TIER_CONFUSION_SUPPLEMENT)
    ok, base_m = feasible(mod, initial)
    if not ok:
        print("baseline fails constraint", base_m)
        sys.exit(1)
    print(
        "baseline: ND recall=%.4f avg_cand=%.2f | NO recall=%.4f avg_cand=%.2f | combined_avg=%.2f"
        % (
            base_m[0]["recall_gt_in_candidates"],
            base_m[0]["avg_candidates"],
            base_m[1]["recall_gt_in_candidates"],
            base_m[1]["avg_candidates"],
            combined_avg_cand(base_m),
        )
    )

    current = deepcopy_supp(initial)
    removed = []

    changed = True
    while changed:
        changed = False
        for t in sorted(current.keys()):
            extras = sorted(current[t])
            for extra in extras:
                trial = deepcopy_supp(current)
                trial[t].discard(extra)
                if not trial[t] and t not in trial:
                    pass
                ok, met = feasible(mod, trial)
                if not ok:
                    continue
                current = trial
                removed.append((t, extra))
                changed = True
                print(
                    "remove (%d->%d) combined_avg=%.4f ND_r=%.4f NO_r=%.4f"
                    % (
                        t,
                        extra,
                        combined_avg_cand(met),
                        met[0]["recall_gt_in_candidates"],
                        met[1]["recall_gt_in_candidates"],
                    )
                )
                break
            if changed:
                break

    print("\n=== greedy done (local optimum) ===")
    print("removed count:", len(removed))
    ok, final_m = feasible(mod, current)
    print(
        "final: ND recall=%.4f avg=%.2f | NO recall=%.4f avg=%.2f | combined_avg=%.2f"
        % (
            final_m[0]["recall_gt_in_candidates"],
            final_m[0]["avg_candidates"],
            final_m[1]["recall_gt_in_candidates"],
            final_m[1]["avg_candidates"],
            combined_avg_cand(final_m),
        )
    )
    print("\nTIER_CONFUSION_SUPPLEMENT = {")
    for k in sorted(current.keys()):
        vs = ", ".join(str(x) for x in sorted(current[k]))
        print(f"    {k}: {{{vs}}},")
    print("}")


if __name__ == "__main__":
    main()

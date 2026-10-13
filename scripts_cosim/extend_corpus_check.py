#!/usr/bin/env python3
"""Checks on an extend_corpus_slate.py run, once: census, the 1e-9 re-score check on every extended dataset, old rows contained in new rows for every
extended dataset, the merged sweeps complete, and how often (and by how much) the wider slate found a strictly better plan than the old best.

Regret of the old best = (old best - new best) / new best, in %, over the extended datasets (zero where the old best stands); pooled, and by batch
size (tasks in the dataset). Untouched datasets (over the cap, already full) are listed by count.

  extend_corpus_check.py --src <corpus> --out <extended corpus> [--tol 1e-9] [--json f]
"""
import argparse
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_cosim.extend_corpus_slate import best_of, read_rows  # noqa: E402
from src.placement.sweep_status import sweep_status  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = {}
    for f in sorted(glob.glob(str(a.out / "extend_manifest*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs[r["dataset"]] = r        # the last line of a dataset wins (a rerun replaces a failure)
    src_sets = {p.name for p in a.src.iterdir() if p.name.startswith("ds_") and p.is_dir()}
    census = Counter(r["status"] for r in recs.values())
    problems = []
    if set(recs) != src_sets:
        problems.append(f"{len(src_sets - set(recs))} source datasets missing from the manifests, {len(set(recs) - src_sets)} unknown")
    for r in recs.values():
        if r["status"] == "failed":
            problems.append(f"{r['dataset']} failed: {r.get('error')}")
    ext = [r for r in recs.values() if r["status"] == "extended"]
    reg, per_n, worst_verify = [], defaultdict(list), 0.0
    contained = complete = 0
    for r in ext:
        d = r["dataset"]
        v = r["verified"]
        worst_verify = max(worst_verify, v["max_abs_diff"])
        if v["n"] < min(20, r["plans_before"]) or v["max_abs_diff"] > a.tol:
            problems.append(f"{d}: re-score check n={v['n']} max diff {v['max_abs_diff']}")
        old = [l for l, _ in read_rows(a.src / d / "placements" / "placements.jsonl")]
        new_rows = read_rows(a.out / d / "placements" / "placements.jsonl")
        new = {l for l, _ in new_rows}
        if set(old) <= new and len(new_rows) == r["plans_after"] and len(new) == len(new_rows):
            contained += 1
        else:
            problems.append(f"{d}: old rows not contained in the new rows (or counts differ)")
        done, why = sweep_status(a.out / d)
        if done:
            complete += 1
        else:
            problems.append(f"{d}: merged sweep not complete ({why})")
        ob, nb = r["best_before"], r["best_after"]
        pct = 100.0 * (ob - nb) / nb if nb > 0 else 0.0
        reg.append(pct)
        per_n[len(json.loads((a.out / d / "slate_extension.json").read_text())["per_task_candidates_after"])].append((pct, r["plans_added"]))

    def summ(xs):
        xs = sorted(xs)
        n = len(xs)
        better = [x for x in xs if x > 1e-9]
        return {"n": n, "strictly_better": len(better), "share_better": len(better) / n if n else None,
                "mean_regret_pct": sum(xs) / n if n else None, "median_regret_pct": st.median(xs) if n else None,
                "p90_regret_pct": xs[int(0.9 * (n - 1))] if n else None, "max_regret_pct": xs[-1] if n else None,
                "mean_regret_pct_where_better": sum(better) / len(better) if better else None}

    res = {"census": dict(census), "extended": len(ext), "reconstruction_max_abs_diff": worst_verify, "old_rows_contained": contained,
           "merged_sweeps_complete": complete, "plans_before": sum(r["plans_before"] for r in ext), "plans_added": sum(r["plans_added"] for r in ext),
           "plans_after": sum(r["plans_after"] for r in ext), "pooled": summ(reg),
           "by_batch_size": {str(k): summ([p for p, _ in v]) for k, v in sorted(per_n.items())}, "problems": problems[:50], "n_problems": len(problems)}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

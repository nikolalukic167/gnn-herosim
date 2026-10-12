#!/usr/bin/env python3
"""What drives the datasets where the wider slate found a strictly better plan? CPU only, reads the source corpus, the extended corpus and the manifests; no simulation.

For each extended dataset whose new best RTT is below the old best (the tie-break is the sweep's), compare the new-best plan with the old-best plan, per task position
(plan keys are positions in workload.json trace_task_ids; the raw snapshot's candidates carry the facts):

 (a) cold mispricing vs the cap. The new-best plan is not in the old rows, so it uses at least one candidate the declared slate dropped. For each such candidate, its rank among
     the task's full candidate set under (i) the DECLARED standalone cost (declared_slate.standalone_cost: cold = 0 when `initialized`) and (ii) the SIM standalone cost, the same
     drain + exec + latency with the sim's cold charge (the table cold start unless the platform's previous task has the task's type: warmth.sandbox_is_warm, from
     fidelity.platforms[queue_key].prev). A dropped candidate has declared rank > 5 by construction. Dataset class:
       cold-mispricing  every dropped candidate the new best uses has SIM rank <= 5 (a top-5 under the correct cost would have contained the new-best plan's candidates);
       mixed            some do, some do not;
       cap              none does (the plan needs a candidate outside the correct top-5 as well).
 (b) what changed: tasks whose (node, platform) differs between the two plans; a dataset is `node-change` when at least one such task moves node, `platform-only` when every change
     keeps the node. Co-location = number of task pairs on one node, new vs old plan (more / same / fewer).
 (c) regret of the old best in seconds (old best RTT - new best RTT): mean, median, p90, max, pooled and per batch size.

  extend_corpus_drivers.py --src <corpus> --out <extended corpus> [--json f]
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts_cosim.extend_corpus_slate import best_of, read_rows, slate_view  # noqa: E402
from src.placement.declared_slate import TOP_K, standalone_cost  # noqa: E402

TIE = 1e-12


def sim_cost(cand, task_type, prev):
    cold = 0.0 if (prev is not None and prev == task_type) else float(cand.get("cold_start_time", 0.0) or 0.0)
    return (float(cand.get("queue_drain_seconds", 0.0) or 0.0) + cold + float(cand.get("execution_time", 0.0) or 0.0) + float(cand.get("network_latency", 0.0) or 0.0))


def ranks(cands, cost):
    order = sorted(cands, key=lambda c: (cost(c), int(c["node_id"]), int(c["platform_id"])))
    return {(int(c["node_id"]), int(c["platform_id"])): i + 1 for i, c in enumerate(order)}


def quantile(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] if s else None


def colocated(combo):
    return sum(1 for a, b in combinations(combo, 2) if a[0] == b[0])


def one(src_ds, out_ds):
    old_rows = read_rows(src_ds / "placements" / "placements.jsonl")
    new_rows = read_rows(out_ds / "placements" / "placements.jsonl")
    (old_rtt, old_combo), (new_rtt, new_combo) = best_of(old_rows), best_of(new_rows)
    if not new_rtt < old_rtt - TIE:
        return None
    v = slate_view(src_ds)
    tids = [int(g) for g in v["wl"]["trace_task_ids"]]
    raw_by = {int(t["task_id"]): t for t in v["raw"]["tasks"]}
    plats = v["raw"]["fidelity"]["platforms"]
    changed, dropped_info = [], []
    for pos, g in enumerate(tids):
        if old_combo[pos] == new_combo[pos]:
            continue
        changed.append((old_combo[pos], new_combo[pos]))
        t = raw_by[g]
        cands = t["candidates"]
        by_key = {(int(c["node_id"]), int(c["platform_id"])): c for c in cands}
        decl = ranks(cands, standalone_cost)
        sim = ranks(cands, lambda c: sim_cost(c, t["task_type"], (plats.get(c["queue_key"]) or {}).get("prev")))
        k = new_combo[pos]
        if decl[k] > TOP_K:
            dropped_info.append({"declared_rank": decl[k], "sim_rank": sim[k], "n_candidates": len(cands),
                                 "initialized": bool(by_key[k].get("initialized")), "same_node_as_old": old_combo[pos][0] == k[0]})
    if not dropped_info:
        cls = "no-dropped-candidate"          # an RTT tie broken against the sweep's order would land here; reported, not expected
    else:
        low = [d["sim_rank"] <= TOP_K for d in dropped_info]
        cls = "cold-mispricing" if all(low) else ("mixed" if any(low) else "cap")
    moved_node = any(a[0] != b[0] for a, b in changed)
    co_old, co_new = colocated(old_combo), colocated(new_combo)
    return {"n_tasks": len(tids), "regret_s": old_rtt - new_rtt, "regret_pct": 100 * (old_rtt - new_rtt) / new_rtt, "class": cls,
            "kind": "node-change" if moved_node else "platform-only", "n_changed": len(changed),
            "colocation": "more" if co_new > co_old else ("fewer" if co_new < co_old else "same"), "dropped": dropped_info}


def summarize(rows):
    n = len(rows)
    out = {"n": n}
    if not n:
        return out
    for key in ("class", "kind", "colocation"):
        c = Counter(r[key] for r in rows)
        out[key] = {k: [v, round(v / n, 4)] for k, v in sorted(c.items())}
    rs = [r["regret_s"] for r in rows]
    out["regret_s"] = {"mean": st.fmean(rs), "median": st.median(rs), "p90": quantile(rs, 0.9), "max": max(rs)}
    out["n_changed_tasks_mean"] = st.fmean(r["n_changed"] for r in rows)
    ds = [d for r in rows for d in r["dropped"]]
    out["dropped_candidates"] = {"n": len(ds), "sim_rank_le_5": sum(d["sim_rank"] <= TOP_K for d in ds), "sim_rank_6_10": sum(TOP_K < d["sim_rank"] <= 10 for d in ds),
                                 "sim_rank_gt_10": sum(d["sim_rank"] > 10 for d in ds), "initialized": sum(d["initialized"] for d in ds)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    manifest = [json.loads(l) for f in sorted(a.out.glob("extend_manifest.*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    ext = sorted(r["dataset"] for r in manifest if r["status"] == "extended")
    rows = []
    for i, d in enumerate(ext):
        r = one(a.src / d, a.out / d)
        if r:
            r["dataset"] = d
            rows.append(r)
    print(f"{a.out.name}: {len(ext)} extended, {len(rows)} strictly better")
    res = {"extended": len(ext), "strictly_better": len(rows), "pooled": summarize(rows),
           "by_batch_size": {str(n): summarize([r for r in rows if r["n_tasks"] == n]) for n in sorted({r["n_tasks"] for r in rows})}, "rows": rows}
    p = res["pooled"]
    print(json.dumps({k: p[k] for k in p}, indent=1))
    for n, s in res["by_batch_size"].items():
        print(f"  n={n}: {s['n']} datasets; class {s['class']}; kind {s['kind']}; regret_s median {s['regret_s']['median']:.4f} p90 {s['regret_s']['p90']:.4f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

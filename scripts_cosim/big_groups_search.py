#!/usr/bin/env python3
"""big_groups_s0_v1 reference search: plans scored by their replayed true batch latency (rollout_replay batch-only mode).

  big_groups_search.py check OUT.jsonl STATES.txt [--n-states 40] [--per-state 10] [--workers 31]

check: the scorer must reproduce the corpus label. For each state (a held-out fidelity dataset with a sweep) the label-optimal row
and a seeded draw of other sweep rows are replayed in ONE worker process, in sequence (the search scores a state's plans that way),
and the first plan is replayed again at the end; the row records |replayed - label| / label per plan and whether the repeat is
bit-identical. Bar (registered): within 0.1 % on the sampled sweep plans.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))


def score(ds: str, plan) -> dict:
    """True batch latency of `plan` ([(node_id, platform_id)] in dataset task order) at the dataset's state."""
    import rollout_replay

    r = rollout_replay._one(dict(ds=ds, plan=[list(map(int, p)) for p in plan], H=-1.0, continuation="live", table=None, eps=0.0))
    if "error" in r:
        raise RuntimeError(r["error"] + "\n" + r.get("tb", ""))
    if r["plan"] != [list(map(int, p)) for p in plan]:
        raise RuntimeError(f"replay served {r['plan']}, not the forced {plan}")
    return r


def _check_state(job):
    t0 = time.time()
    ds = job["ds"]
    try:
        rows = [json.loads(l) for l in open(Path(ds) / "placements" / "placements.jsonl") if l.strip()]
        rows = [r for r in rows if "placement_plan" in r and r.get("rtt") is not None]
        n = len(rows[0]["placement_plan"])
        best = min(rows, key=lambda r: float(r["rtt"]))
        rng = random.Random(f"{job['seed']}:{Path(ds).name}")
        others = [r for r in rows if r is not best]
        pick = [best] + rng.sample(others, min(job["per_state"] - 1, len(others)))
        out = []
        for r in pick:
            plan = [r["placement_plan"][str(i)] for i in range(n)]
            s = score(ds, plan)
            lab = float(r["rtt"])
            out.append(dict(plan=plan, label=lab, replay=s["q_batch"], rel=(s["q_batch"] - lab) / lab, wall=s["wall"]))
        again = score(ds, out[0]["plan"])["q_batch"]
        return dict(ds=ds, n_tasks=n, n_rows=len(rows), plans=out, repeat_identical=again == out[0]["replay"],
                    repeat=again, wall=time.time() - t0)
    except Exception as e:
        import traceback

        return dict(ds=ds, error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:])


def check(a):
    states = [l.strip() for l in open(a.states) if l.strip()]
    rng = random.Random(a.seed)
    states = sorted(rng.sample(states, min(a.n_states, len(states))))
    jobs = [dict(ds=d, per_state=a.per_state, seed=a.seed) for d in states]
    res = []
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_check_state, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            res.append(r)
    ok = [r for r in res if "error" not in r]
    rel = sorted(abs(p["rel"]) for r in ok for p in r["plans"])
    walls = sorted(p["wall"] for r in ok for p in r["plans"])
    print(f"check: {len(ok)}/{len(res)} states without error, {len(rel)} plans")
    if rel:
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        print(f"  |replay - label| / label: median {q(rel, .5):.2e} p95 {q(rel, .95):.2e} max {rel[-1]:.2e}; "
              f"within 0.1 %: {sum(x <= 1e-3 for x in rel)}/{len(rel)}; exact (<1e-9): {sum(x < 1e-9 for x in rel)}/{len(rel)}")
        print(f"  repeat in-process bit-identical: {sum(r['repeat_identical'] for r in ok)}/{len(ok)}")
        print(f"  wall per plan: median {q(walls, .5):.2f} s p95 {q(walls, .95):.2f} s")
        print("  PASS" if rel[-1] <= 1e-3 and all(r["repeat_identical"] for r in ok) and len(ok) == len(res) else "  FAIL")
    for r in res:
        if "error" in r:
            print("  ERROR", r["ds"], r["error"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check",))
    ap.add_argument("out")
    ap.add_argument("states")
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--per-state", type=int, default=10)
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20261011)
    a = ap.parse_args()
    check(a)


if __name__ == "__main__":
    main()

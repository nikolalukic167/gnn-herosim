#!/usr/bin/env python3
"""Offline regret of the CD greedy's own cost model on a cache's VAL graphs (docs/lineages/small_batch_v1.md).

  small_batch_rules_eval.py --cache <graph cache> --split <split json> --out <json>

Per VAL graph: the plan that coordinate descent finds on the CD cost (service + peer exchange + standing backlog +
in-batch wait, the terms queue_gap_diag_eval decomposes), started from the per-task independent argmin and swept in task-id
order until no single-task move lowers it, and the plan of the independent argmin alone (a pointwise rule with the same
columns). Both are scored by the full sweep, so their regret is on the learned arms' scale and size. It is a replica
of the CD cost, not the live CD arm; it answers "is the batch size changing how hard the problem is for a rule".
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))
sys.path.insert(0, str(REPO / "scripts_cosim"))


def plan_cost(ctx, plan):
    n = len(plan)
    svc = exch = back = wait = 0.0
    load = {}
    for t in range(n):
        key = plan[t]
        e = 0.0
        for j in range(n):
            b = ctx.peer_pairs.get((t, j))
            if b is None or j == t:
                continue
            pb, lat = ctx.node_exchange[(ctx.node_of[key], ctx.node_of[plan[j]])]
            e += b * pb + (lat if pb > 0.0 else 0.0)
        charge = float(ctx.service_s[(t, key)]) + e
        svc += float(ctx.service_s[(t, key)])
        exch += e
        back += float(ctx.backlog_s[key])
        wait += load.get(key, 0.0)
        load[key] = load.get(key, 0.0) + charge
    return svc + exch + back + wait


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import queue_gap_diag_eval as D
    from non_unique_lib.training_contract import canonical_parent_id, load_split_artifact
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    cache = Path(a.cache)
    graphs = pickle.load(open(cache / "graphs.pkl", "rb"))
    ids = pickle.load(open(cache / "dataset_ids.pkl", "rb"))
    opt_rtt = pickle.load(open(cache / "optimal_rtt.pkl", "rb"))
    split, _ = load_split_artifact(Path(a.split))
    val_parents = set(split["val"])
    val = [(g, gid) for g, gid in zip(graphs, ids)
           if canonical_parent_id(getattr(g, "parent_dataset_id", None) or gid) in val_parents]
    sweep = D._val_sweep({gid.split("@seq", 1)[0] for _, gid in val}, cache)
    rows = []
    for g, gid in val:
        key = gid.split("@seq", 1)[0]
        ctx = build_partial_state_context_from_graph(g)
        n = int(g.n_tasks)
        cands = [[tuple(int(v) for v in c) for c in g.task_logit_to_placement[t]] for t in range(n)]
        plan = [min(cs, key=lambda c, t=t: float(ctx.service_s[(t, c)]) + float(ctx.backlog_s[c]))
                for t, cs in enumerate(cands)]
        indep = list(plan)
        for _ in range(10):
            moved = False
            for t in range(n):
                best, best_c = plan_cost(ctx, plan), plan[t]
                for c in cands[t]:
                    trial = plan[:t] + [c] + plan[t + 1:]
                    v = plan_cost(ctx, trial)
                    if v < best - 1e-12:
                        best, best_c = v, c
                if best_c != plan[t]:
                    plan[t] = best_c
                    moved = True
            if not moved:
                break
        opt = float(opt_rtt.get(gid, opt_rtt.get(key)))
        r = {"id": gid, "n_tasks": n}
        for name, p in (("cd", plan), ("indep", indep)):
            combo = tuple(p)
            if combo not in sweep[key]:
                raise RuntimeError(f"FAIL LOUD: {name} plan of {gid} is not in its full sweep")
            r[f"regret_{name}"] = sweep[key][combo] - opt
        rows.append(r)
    for nm in ("cd", "indep"):
        print(f"[rules] {nm}: mean val regret {sum(r[f'regret_{nm}'] for r in rows) / len(rows):.3f} s over {len(rows)} graphs")
    json.dump({"rows": rows}, open(a.out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

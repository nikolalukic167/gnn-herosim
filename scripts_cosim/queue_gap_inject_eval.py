#!/usr/bin/env python3
"""Information-injection test, one checkpoint (docs/lineages/raw_plan_v2.md, after the queue-gap diagnostic).

  queue_gap_inject_eval.py --ckpt models/<name>.pt --out <json>

Question: is the raw GNN short of *information* or of *use of the information it has*? At decode time only, the model's
own logits get two hand-computed terms subtracted, in seconds from the cached context:
  exch  peer transfer from the candidate to every committed partner on another node (route-aware)
  load  service + exchange already committed by batch-mates on the candidate's platform (the CD greedy's in-batch load)
  mass  expected transfer to partners not yet placed, averaged over each partner's candidate nodes (the lookahead
        column of the engineered context)
  back  standing backlog seconds on the candidate's platform
  logit' = logit - lam_exch * exch - lam_load * load - lam_mass * mass - lam_back * back
Stage 1 grids (lam_exch, lam_load); stage 2 fixes those at (0.3, 0.1), the stage-1 optimum, and grids (lam_mass,
lam_back). The model is not retrained. If injecting a term collapses the regret, the model
lacked it; if regret barely moves, the lack is elsewhere (training signal, use of context). Tuning on the even-indexed
half of VAL and reporting the odd-indexed half is done by queue_gap_inject_read.py.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))
sys.path.insert(0, str(REPO / "scripts_cosim"))

GRID = (0.0, 0.1, 0.3, 1.0, 3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    a = ap.parse_args()
    os.environ["GNN_PREFIX_SELF_REFINE"] = "0"

    import queue_gap_diag_eval as D
    from non_unique_lib.training_contract import canonical_parent_id, load_split_artifact
    from src.policy.gnn import prefix_serving as PS
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    graphs = pickle.load(open(D.CACHE / "graphs.pkl", "rb"))
    ids = pickle.load(open(D.CACHE / "dataset_ids.pkl", "rb"))
    opt_rtt = pickle.load(open(D.CACHE / "optimal_rtt.pkl", "rb"))
    split, _ = load_split_artifact(D.SPLIT)
    val_parents = set(split["val"])
    val = [(g, gid) for g, gid in zip(graphs, ids)
           if canonical_parent_id(getattr(g, "parent_dataset_id", None) or gid) in val_parents]
    sweep = D._val_sweep({gid.split("@seq", 1)[0] for _, gid in val})
    model, options, _sc = PS.load_prefix_conditioned_gnn(Path(a.ckpt), adopt_env=True)

    lam = {"exch": 0.0, "load": 0.0, "mass": 0.0, "back": 0.0}
    base_factory = PS.make_partial_state_score_fn

    def injected(model_, graph, ctx):
        base = base_factory(model_, graph, ctx)
        tl = graph.task_logit_to_placement

        def fn(task_idx, committed):
            logits = base(task_idx, committed)
            if not any(lam.values()):
                return logits
            cands = [tuple(int(v) for v in c) for c in tl[task_idx]]
            comm = {j: tuple(int(v) for v in p) for j, p in committed.items()}
            load_by_key: dict = {}
            if lam["load"] != 0.0:
                for t, key in comm.items():
                    e = 0.0
                    for j, kj in comm.items():
                        b = ctx.peer_pairs.get((t, j))
                        if b is None or j == t:
                            continue
                        pb, lt = ctx.node_exchange[(ctx.node_of[key], ctx.node_of[kj])]
                        e += b * pb + (lt if pb > 0.0 else 0.0)
                    load_by_key[key] = load_by_key.get(key, 0.0) + float(ctx.service_s[(t, key)]) + e
            unplaced = [(j, b) for (i_, j), b in ctx.peer_pairs.items()
                        if i_ == task_idx and j not in comm and j != task_idx] if lam["mass"] != 0.0 else []
            cost = []
            for cand in cands:
                x = 0.0
                for j, kj in comm.items():
                    b = ctx.peer_pairs.get((task_idx, j))
                    if b is None:
                        continue
                    pb, lt = ctx.node_exchange[(ctx.node_of[cand], ctx.node_of[kj])]
                    x += b * pb + (lt if pb > 0.0 else 0.0)
                m = 0.0
                for j, b in unplaced:
                    nodes_j = ctx.cand_nodes.get(j) or ()
                    if nodes_j:
                        m += sum(b * ctx.node_exchange[(ctx.node_of[cand], nj)][0]
                                 + (ctx.node_exchange[(ctx.node_of[cand], nj)][1]
                                    if ctx.node_exchange[(ctx.node_of[cand], nj)][0] > 0.0 else 0.0)
                                 for nj in nodes_j) / len(nodes_j)
                bk = float(ctx.backlog_s[cand]) if lam["back"] != 0.0 else 0.0
                cost.append(lam["exch"] * x + lam["load"] * load_by_key.get(cand, 0.0) + lam["mass"] * m
                            + lam["back"] * bk)
            return logits - torch.tensor(cost, dtype=logits.dtype, device=logits.device)

        return fn

    PS.make_partial_state_score_fn = injected
    res = {}
    t0 = time.time()
    with torch.no_grad():
        for g1, g2 in itertools.product(GRID, GRID):
            if a.stage == 1:
                lam["exch"], lam["load"] = g1, g2
            else:
                lam["exch"], lam["load"], lam["mass"], lam["back"] = 0.3, 0.1, g1, g2
            le, ll = g1, g2
            reg = []
            for g, gid in val:
                key = gid.split("@seq", 1)[0]
                combo = PS.decode_prefix_conditioned(model, g, options)
                rmap = sweep[key]
                if combo not in rmap:
                    raise RuntimeError(f"FAIL LOUD: decoded combo of {gid} is not in its full sweep")
                reg.append(rmap[combo] - float(opt_rtt.get(gid, opt_rtt.get(key))))
            res[f"{le}|{ll}"] = reg
            print(f"[inject] {os.path.basename(a.ckpt)} stage{a.stage} a={le} b={ll}: {sum(reg) / len(reg):.3f}s "
                  f"({time.time() - t0:.0f}s)", flush=True)
    tmp = a.out + ".partial"
    json.dump({"ckpt": os.path.basename(a.ckpt), "stage": a.stage, "grid": GRID, "ids": [gid for _, gid in val], "regret": res},
              open(tmp, "w"))
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

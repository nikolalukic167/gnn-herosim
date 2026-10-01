#!/usr/bin/env python3
"""Queue-gap diagnostic, one checkpoint (docs/lineages/raw_plan_v2.md, diagnosis before the next build).

  queue_gap_diag_eval.py --ckpt models/<name>.pt --refine 0|3 --out <json>

Decodes every VAL graph of the backlog_corpus_v1 split with the served decoder
(prefix_serving.decode_prefix_conditioned, self-refine passes per --refine) and records, per dataset:
  regret         true full-sweep RTT of the decoded plan minus the optimum (the trainer's selection metric)
  n_tasks, n_choice
  q_*            queue length (tasks) at the chosen / label / shortest candidate, summed over choice tasks
  agree          choice tasks where the decode equals the first tied-optimal plan
  backlog        mean queue length over every candidate of every choice task (how loaded the state is)
  plat/node_*    most tasks stacked on one platform / node, for the decode and for the label plan
  coloc_*        peer pairs placed on one node, for the decode and for the label plan (n_pairs in total)
  agree_peer/solo  agreement with the label on choice tasks that have a peer partner in the batch / that have none
  cost_*_dec/lab   the CD greedy's cost terms of the decoded / label plan, in seconds, from the cached context:
                   svc (service of every task at its candidate), exch (peer transfer to partners on other nodes),
                   back (standing backlog at each chosen platform), wait (in-batch stacking: each task waits for the
                   service + exchange of earlier batch-mates on its platform, in task-id order)
Everything is read from the cache; nothing is simulated.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

CACHE = REPO / "simulation_data" / "graphs_cache_backlog_corpus_v1_psv4_inf"
SPLIT = REPO / "experiments" / "backlog_corpus_v1_split.json"


def _val_sweep(val_ids: set) -> Dict[str, Dict[tuple, float]]:
    from non_unique_lib.cache_io import _rtt_chunks_meta

    num_chunks, total = _rtt_chunks_meta(CACHE)
    out: Dict[str, Dict[tuple, float]] = {}
    n = 0
    for i in range(num_chunks):
        with open(CACHE / f"rtt_chunk_{i}.pkl", "rb") as f:
            chunk = pickle.load(f)
        for (ds, combo), rtt in chunk.items():
            n += 1
            if ds in val_ids:
                out.setdefault(ds, {})[combo] = float(rtt)
        chunk.clear()
    if n != total:
        raise RuntimeError(f"FAIL LOUD: read {n} sweep rows, metadata declares {total}")
    missing = val_ids - set(out)
    if missing:
        raise RuntimeError(f"FAIL LOUD: {len(missing)} val parents have no sweep rows")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--refine", type=int, choices=(0, 3), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["GNN_PREFIX_SELF_REFINE"] = str(a.refine)

    from non_unique_lib.training_contract import canonical_parent_id, load_split_artifact
    from src.policy.gnn.prefix_serving import decode_prefix_conditioned, load_prefix_conditioned_gnn

    graphs = pickle.load(open(CACHE / "graphs.pkl", "rb"))
    ids = pickle.load(open(CACHE / "dataset_ids.pkl", "rb"))
    opt_rtt = pickle.load(open(CACHE / "optimal_rtt.pkl", "rb"))
    split, _ = load_split_artifact(SPLIT)
    val_parents = set(split["val"])
    parent_of = [canonical_parent_id(getattr(g, "parent_dataset_id", None) or gid) for g, gid in zip(graphs, ids)]
    val = [(g, gid, p) for g, gid, p in zip(graphs, ids, parent_of) if p in val_parents]
    sweep = _val_sweep({gid.split("@seq", 1)[0] for _, gid, _ in val})
    print(f"[diag] {len(val)} val graphs, {sum(len(v) for v in sweep.values()):,} sweep rows", flush=True)

    model, options, sidecar = load_prefix_conditioned_gnn(Path(a.ckpt), adopt_env=True)
    alpha = options.alpha_key
    rows: List[dict] = []
    t0 = time.time()
    with torch.no_grad():
        for g, gid, _ in val:
            key = gid.split("@seq", 1)[0]
            combo = decode_prefix_conditioned(model, g, options)
            rmap = sweep[key]
            if combo not in rmap:
                raise RuntimeError(f"FAIL LOUD: decoded combo of {gid} is not in its full sweep")
            snap = getattr(g, "queue_snapshot", None)
            qkeys = getattr(g, "task_logit_to_queue_key", None)
            if not snap or not qkeys:
                raise RuntimeError(f"FAIL LOUD: {gid} carries no queue_snapshot/task_logit_to_queue_key")
            tl = g.task_logit_to_placement
            label = g.tied_optimal_logit_plans[alpha][0]
            n = int(g.n_tasks)
            qs = {"chosen": 0.0, "label": 0.0, "min": 0.0}
            lab_combo = [tuple(int(v) for v in tl[t][int(label[t])]) for t in range(n)]
            dec_combo = [tuple(int(v) for v in combo[t]) for t in range(n)]

            def stack(plan, idx):
                counts: Dict[tuple, int] = {}
                for c in plan:
                    k = c if idx is None else c[idx]
                    counts[k] = counts.get(k, 0) + 1
                return max(counts.values())

            pe = getattr(g, "peer_edge_index", None)
            pairs = sorted({tuple(sorted((int(x), int(y)))) for x, y in pe.t().tolist()}) if pe is not None and pe.numel() else []

            def coloc(plan):
                return sum(plan[x][0] == plan[y][0] for x, y in pairs)

            from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

            ctx = build_partial_state_context_from_graph(g)

            def terms(plan):
                svc = exch = back = wait = 0.0
                load: Dict[tuple, float] = {}
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
                return svc, exch, back, wait

            cost = {"dec": terms(dec_combo), "lab": terms(lab_combo)}
            peer_tasks = {x for pr in pairs for x in pr}
            ag = {"peer": [0, 0], "solo": [0, 0]}
            cand_q, n_choice, agree = [], 0, 0
            for t in range(n):
                if len(tl[t]) < 2:
                    continue
                n_choice += 1
                qrow = [float(snap[qkeys[t][i]]) for i in range(len(tl[t]))]
                chosen = [tuple(int(v) for v in c) for c in tl[t]].index(tuple(int(v) for v in combo[t]))
                qs["chosen"] += qrow[chosen]
                qs["label"] += qrow[int(label[t])]
                qs["min"] += min(qrow)
                agree += int(chosen == int(label[t]))
                kind = "peer" if t in peer_tasks else "solo"
                ag[kind][0] += int(chosen == int(label[t]))
                ag[kind][1] += 1
                cand_q.extend(qrow)
            opt = opt_rtt.get(gid, opt_rtt.get(key))
            if opt is None:
                raise RuntimeError(f"FAIL LOUD: no optimal RTT for {gid}")
            rows.append({"id": gid, "regret": rmap[combo] - float(opt), "n_tasks": n, "n_choice": n_choice,
                         "agree": agree,
                         "agree_peer": ag["peer"][0], "n_peer": ag["peer"][1],
                         "agree_solo": ag["solo"][0], "n_solo": ag["solo"][1],
                         "plat_dec": stack(dec_combo, None), "plat_lab": stack(lab_combo, None),
                         "node_dec": stack(dec_combo, 0), "node_lab": stack(lab_combo, 0),
                         **{f"cost_{nm}_{w}": cost[w][i] for w in ("dec", "lab")
                            for i, nm in enumerate(("svc", "exch", "back", "wait"))},
                         "n_pairs": len(pairs), "coloc_dec": coloc(dec_combo), "coloc_lab": coloc(lab_combo),
                         "q_chosen": qs["chosen"], "q_label": qs["label"], "q_min": qs["min"],
                         "backlog": sum(cand_q) / max(1, len(cand_q))})
    mean = sum(r["regret"] for r in rows) / len(rows)
    print(f"[diag] {os.path.basename(a.ckpt)} refine={a.refine}: val regret {mean:.3f}s over {len(rows)} graphs "
          f"in {time.time() - t0:.0f}s", flush=True)
    tmp = a.out + ".partial"
    json.dump({"ckpt": os.path.basename(a.ckpt), "refine": a.refine, "mean_regret": mean, "rows": rows}, open(tmp, "w"))
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

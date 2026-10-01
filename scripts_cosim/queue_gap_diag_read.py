#!/usr/bin/env python3
"""Reader for the queue-gap diagnostic (docs/lineages/raw_plan_v2.md).

  queue_gap_diag_read.py <dir of <arm>_s<seed>_r<refine>.json> [--out read.json]

Arms: rawgnn / rawmlp (raw_plan_v1), psignn (engineered context, xs1load), psimlp (its MP-OFF twin).
All figures are per VAL graph, averaged over the four seeds first, then over graphs (the seeds are the replicates).

Tables:
  1. mean val regret (s) per arm and refine setting;
  2. where the extra regret of rawgnn over psignn sits, by backlog quartile of the state (mean candidate queue length);
  3. queue behaviour on choice tasks: queue length at the chosen candidate minus the label's and minus the shortest;
  4. agreement with the first tied-optimal plan;
  5. by batch size (number of choice tasks);
  6. plan shape against the label plan: tasks stacked on the busiest platform / node, and peer pairs on one node;
  7. the CD greedy's own cost terms (service, exchange, backlog, in-batch wait) of the decoded plan minus the label
     plan: which term the extra regret sits in. (Every choice task here has a peer partner, so a with/without split
     of agreement carries no information and was dropped.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, List

ARMS = ("rawgnn", "rawmlp", "psignn", "psimlp")


def load(d: str) -> Dict[tuple, Dict[str, dict]]:
    out: Dict[tuple, Dict[str, dict]] = {}
    for f in glob.glob(os.path.join(d, "*.json")):
        arm, s, r = os.path.basename(f)[:-5].split("_")
        if arm not in ARMS:
            continue
        rows = {x["id"]: x for x in json.load(open(f))["rows"]}
        out.setdefault((arm, int(r[1:])), {})[s] = rows
    for k, seeds in out.items():
        if len(seeds) != 4:
            raise SystemExit(f"FAIL LOUD: {k} has seeds {sorted(seeds)}, expected 4")
    return out


def per_graph(seeds: Dict[str, Dict[str, dict]], field: str) -> Dict[str, float]:
    ids = set.intersection(*[set(v) for v in seeds.values()])
    return {i: mean(seeds[s][i][field] for s in seeds) for i in ids}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = load(a.dir)
    res: dict = {"regret": {}, "by_backlog": {}, "queue": {}, "agree": {}, "by_choice": {}}
    for refine in (0, 3):
        ids = None
        pg: Dict[str, Dict[str, Dict[str, float]]] = {}
        for arm in ARMS:
            seeds = data[(arm, refine)]
            pg[arm] = {f: per_graph(seeds, f) for f in
                       ("regret", "q_chosen", "q_label", "q_min", "agree", "n_choice", "backlog", "n_tasks",
                        "plat_dec", "plat_lab", "node_dec", "node_lab", "n_pairs", "coloc_dec", "coloc_lab",
                        "agree_peer", "n_peer", "agree_solo", "n_solo",
                        *[f"cost_{nm}_{w}" for nm in ("svc", "exch", "back", "wait") for w in ("dec", "lab")])}
            ids = set(pg[arm]["regret"]) if ids is None else ids & set(pg[arm]["regret"])
        ids = sorted(ids)
        res["regret"][refine] = {arm: mean(pg[arm]["regret"][i] for i in ids) for arm in ARMS}
        backlog = {i: pg["psignn"]["backlog"][i] for i in ids}
        order = sorted(ids, key=lambda i: backlog[i])
        q = len(order) // 4
        buckets = {f"Q{k + 1}": order[k * q:(k + 1) * q if k < 3 else len(order)] for k in range(4)}
        res["by_backlog"][refine] = {
            b: {"n": len(g), "backlog": mean(backlog[i] for i in g),
                **{arm: mean(pg[arm]["regret"][i] for i in g) for arm in ARMS}}
            for b, g in buckets.items()}
        res["queue"][refine] = {}
        res["agree"][refine] = {}
        for arm in ARMS:
            nc = sum(pg[arm]["n_choice"][i] for i in ids)
            res["queue"][refine][arm] = {
                "chosen_minus_label": sum(pg[arm]["q_chosen"][i] - pg[arm]["q_label"][i] for i in ids) / nc,
                "chosen_minus_min": sum(pg[arm]["q_chosen"][i] - pg[arm]["q_min"][i] for i in ids) / nc,
                "label_minus_min": sum(pg[arm]["q_label"][i] - pg[arm]["q_min"][i] for i in ids) / nc}
            res["agree"][refine][arm] = sum(pg[arm]["agree"][i] for i in ids) / nc
            res.setdefault("cost", {}).setdefault(refine, {})[arm] = {
                nm: mean(pg[arm][f"cost_{nm}_dec"][i] - pg[arm][f"cost_{nm}_lab"][i] for i in ids)
                for nm in ("svc", "exch", "back", "wait")}
            res["cost"][refine][arm]["total"] = sum(res["cost"][refine][arm].values())
            pairs = sum(pg[arm]["n_pairs"][i] for i in ids)
            res.setdefault("agree_split", {}).setdefault(refine, {})[arm] = {
                "peer_tasks": sum(pg[arm]["agree_peer"][i] for i in ids) / max(1, sum(pg[arm]["n_peer"][i] for i in ids)),
                "solo_tasks": sum(pg[arm]["agree_solo"][i] for i in ids) / max(1, sum(pg[arm]["n_solo"][i] for i in ids)),
                "n_peer": sum(pg[arm]["n_peer"][i] for i in ids), "n_solo": sum(pg[arm]["n_solo"][i] for i in ids)}
            res.setdefault("shape", {}).setdefault(refine, {})[arm] = {
                "platform_stack_decode": mean(pg[arm]["plat_dec"][i] for i in ids),
                "platform_stack_label": mean(pg[arm]["plat_lab"][i] for i in ids),
                "node_stack_decode": mean(pg[arm]["node_dec"][i] for i in ids),
                "node_stack_label": mean(pg[arm]["node_lab"][i] for i in ids),
                "peer_colocated_decode": sum(pg[arm]["coloc_dec"][i] for i in ids) / max(1, pairs),
                "peer_colocated_label": sum(pg[arm]["coloc_lab"][i] for i in ids) / max(1, pairs)}
        sizes = defaultdict(list)
        for i in ids:
            sizes[min(int(round(pg["psignn"]["n_choice"][i])), 12) // 4 * 4].append(i)
        res["by_choice"][refine] = {
            f"{k}-{k + 3} choice tasks": {"n": len(g), **{arm: mean(pg[arm]["regret"][i] for i in g) for arm in ARMS}}
            for k, g in sorted(sizes.items())}
    for refine in (0, 3):
        print(f"\n=== self-refine {refine}")
        print("mean val regret (s):", {k: round(v, 2) for k, v in res["regret"][refine].items()})
        print("by backlog quartile (regret s):")
        for b, r in res["by_backlog"][refine].items():
            print(f"  {b} n={r['n']:4d} backlog={r['backlog']:6.2f}  " + "  ".join(f"{arm}={r[arm]:6.2f}" for arm in ARMS))
        print("queue length on choice tasks (tasks), chosen-label | chosen-shortest | label-shortest:")
        for arm in ARMS:
            x = res["queue"][refine][arm]
            print(f"  {arm:7s} {x['chosen_minus_label']:+7.3f} | {x['chosen_minus_min']:+7.3f} | {x['label_minus_min']:+7.3f}"
                  f"   agree {100 * res['agree'][refine][arm]:5.1f} %")
        print("plan shape (decode | label): tasks on busiest platform, on busiest node, peer pairs on one node:")
        for arm in ARMS:
            x = res["shape"][refine][arm]
            print(f"  {arm:7s} platform {x['platform_stack_decode']:.2f} | {x['platform_stack_label']:.2f}   "
                  f"node {x['node_stack_decode']:.2f} | {x['node_stack_label']:.2f}   "
                  f"coloc {100 * x['peer_colocated_decode']:5.1f} % | {100 * x['peer_colocated_label']:5.1f} %")
        print("CD-model cost of the decoded plan minus the label plan, per graph (s):  service | exchange | backlog | "
              "in-batch wait | total")
        for arm in ARMS:
            x = res["cost"][refine][arm]
            print(f"  {arm:7s} {x['svc']:+7.2f} | {x['exch']:+7.2f} | {x['back']:+7.2f} | {x['wait']:+7.2f} | {x['total']:+7.2f}")
        print("by number of choice tasks (regret s):")
        for b, r in res["by_choice"][refine].items():
            print(f"  {b:22s} n={r['n']:4d}  " + "  ".join(f"{arm}={r[arm]:6.2f}" for arm in ARMS))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

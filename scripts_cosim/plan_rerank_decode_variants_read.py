#!/usr/bin/env python3
"""plan_rerank_v1 offline screen of decode variants (descriptive; orders work, closes nothing). Reads NEAR_RTT_EVAL_VARIANTS dumps.

Arms per table (single-decision tables of the plan_rerank tables; rung from there):
  served        the masked_topo decode (the record's own combo)
  branch_ps     (a)  step-0 branching, keep the completion with the highest GNN plan score
  branch_sl     (a') step-0 branching, keep the completion with the highest summed step logits (the decoder's own criterion)
  beam2_ps / beam5_ps   beam over all steps (pruned by cumulative step logit), final pick by plan score
  beam2_sl / beam5_sl   the same beams, final pick by summed step logits (descriptive)
  ss_top1       self-search top-1 (argmax plan score over the full sweep; plan_rerank table top10[0])
  argmin_s      argmin-S (best-label tie), the bar
  oracle_branches  best true RTT among the step-0 branch completions (reference ceiling of (a)/(a'))
Ties in a selection break by branch / beam order (candidate order; beam rank).
Selection among the four candidate variants (branch_ps, branch_sl, beam2_ps, beam5_ps) is on TRAIN pooled regret only; held-out is
read once per rung: pooled, top-3-trimmed pooled, median and mean per-table relative regret, per-table better/equal/worse vs argmin-S.
Bar for proposing a live arm: held-out pooled AND trimmed regret below argmin-S's at both rungs.
Integrity, fail loud: the branch whose first choice equals the served decode's first choice must reproduce the served combo exactly;
every variant plan must be in the sweep (an RTT).

  plan_rerank_decode_variants_read.py --dump-dir DIR --heldout-table-dir DIR --train-table-dir DIR --out read.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

CKS = {"g4": "accel-replica-v1-gnn-eng-g4-seed1", "physmp_g3": "accel-replica-v1-gnn-eng-physmp-g3-seed1"}
CANDIDATES = ("branch_ps", "branch_sl", "beam2_ps", "beam5_ps")
ARMS = ("served", "branch_ps", "branch_sl", "beam2_ps", "beam5_ps", "beam2_sl", "beam5_sl", "ss_top1", "argmin_s", "oracle_branches")
EPS = 1e-9


def pick(rows, field):
    return max(range(len(rows)), key=lambda i: (rows[i][field], -i))


def arms_of(rec, trow):
    v = rec["variants"]
    for key in ("branches", "beam2", "beam5"):
        for r in v[key]:
            if r["rtt"] is None:
                raise SystemExit(f"FAIL LOUD: {rec['dataset_id']} {key} plan not in the sweep")
    served = rec["opt_rtt"] + rec["regret"]
    first = [tuple(c) for c in rec["combo"]]
    same = [r for r in v["branches"] if [tuple(c) for c in r["plan"]] == first]
    if not same:
        raise SystemExit(f"FAIL LOUD: {rec['dataset_id']} no branch reproduces the served combo")
    br = v["branches"]
    return {"served": served,
            "branch_ps": br[pick(br, "plan_score")]["rtt"], "branch_sl": br[pick(br, "step_logit_sum")]["rtt"],
            "beam2_ps": v["beam2"][pick(v["beam2"], "plan_score")]["rtt"], "beam5_ps": v["beam5"][pick(v["beam5"], "plan_score")]["rtt"],
            "beam2_sl": v["beam2"][pick(v["beam2"], "step_logit_sum")]["rtt"], "beam5_sl": v["beam5"][pick(v["beam5"], "step_logit_sum")]["rtt"],
            "ss_top1": trow["top10"][0]["rtt"], "argmin_s": trow["argmin_s_min"], "oracle_branches": min(r["rtt"] for r in br)}


def summary(tabs, arm):
    so = sum(t["opt"] for t in tabs)
    ex = sorted((t["arms"][arm] - t["opt"] for t in tabs), reverse=True)
    rel = [(t["arms"][arm] - t["opt"]) / t["opt"] for t in tabs]
    b = sum(t["arms"][arm] < t["arms"]["argmin_s"] * (1 - EPS) for t in tabs)
    w = sum(t["arms"][arm] > t["arms"]["argmin_s"] * (1 + EPS) for t in tabs)
    return dict(n=len(tabs), pooled=sum(ex) / so, trim3=sum(ex[3:]) / so, median=st.median(rel), mean=st.mean(rel),
                better=b, equal=len(tabs) - b - w, worse=w)


def load(dump: Path, table: Path):
    trows = {json.loads(l)["ds"]: json.loads(l) for l in open(table)}
    tabs = []
    for rec in json.load(open(dump))["records"]:
        tr = trows.get(rec["dataset_id"])
        if tr is None:
            continue
        if abs(tr["opt"] - rec["opt_rtt"]) > 1e-6 * max(1.0, tr["opt"]):
            raise SystemExit(f"FAIL LOUD: {rec['dataset_id']} opt differs between dump and table")
        tabs.append({"ds": rec["dataset_id"], "rung": tr["rung"], "opt": tr["opt"], "arms": arms_of(rec, tr)})
    if len(tabs) != len(trows):
        raise SystemExit(f"FAIL LOUD: {table}: {len(trows)} tables, {len(tabs)} with a variants record")
    return tabs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--heldout-table-dir", required=True)
    ap.add_argument("--train-table-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    pct = lambda x: f"{100 * x:6.2f}%"
    for ck, stem in CKS.items():
        tr = load(Path(a.dump_dir) / f"{stem}-train.json", Path(a.train_table_dir) / f"table_{stem}-best.jsonl")
        ho = load(Path(a.dump_dir) / f"{stem}-val.json", Path(a.heldout_table_dir) / f"table_{stem}-best.jsonl")
        train_pooled = {arm: summary(tr, arm)["pooled"] for arm in ARMS}
        chosen = min(CANDIDATES, key=lambda arm: (train_pooled[arm], CANDIDATES.index(arm)))
        res = {"n_train": len(tr), "n_heldout": len(ho), "train_pooled": train_pooled, "chosen_on_train": chosen, "heldout": {}, "bar": {}}
        print(f"== {ck}: train n {len(tr)}, held-out n {len(ho)}")
        print("   train pooled: " + "  ".join(f"{arm} {pct(x)}" for arm, x in train_pooled.items()) + f"  -> chosen on train: {chosen}")
        for rung in ("moderate", "heavy", "all"):
            sub = [t for t in ho if rung == "all" or t["rung"] == rung]
            res["heldout"][rung] = {arm: summary(sub, arm) for arm in ARMS}
            for arm in ARMS:
                m = res["heldout"][rung][arm]
                print(f"   held-out {rung:8s} {arm:16s} n {m['n']:3d} | pooled {pct(m['pooled'])} trim-3 {pct(m['trim3'])} | median {pct(m['median'])} "
                      f"mean {pct(m['mean'])} | vs argmin-S better {m['better']} equal {m['equal']} worse {m['worse']}")
        for arm in CANDIDATES:
            res["bar"][arm] = all(res["heldout"][r][arm]["pooled"] < res["heldout"][r]["argmin_s"]["pooled"]
                                  and res["heldout"][r][arm]["trim3"] < res["heldout"][r]["argmin_s"]["trim3"] for r in ("moderate", "heavy"))
        print("   bar (held-out pooled AND trim-3 below argmin-S at both rungs): " + ", ".join(f"{arm} {res['bar'][arm]}" for arm in CANDIDATES))
        out[ck] = res
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

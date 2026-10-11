#!/usr/bin/env python3
"""link_bound_s0_v1 screen read (docs/lineages/link_bound_s0_v1.md), committed before any capture.

Input: the reference-search rows (big_groups_search.py search) of every state. A state's (bandwidth, rung) comes from its dataset's
source tag, which carries the rung name and a `bw<core MB/s>` token. Per (bandwidth, rung), each a median over states:
  L1  link wait / batch latency AT THE REFERENCE PLAN (the best-known plan by replayed batch latency); link wait = the batch's
      summed linkWaitTime, the queueing for the input route's link pipes (peer exchange never queues in this physics)
  L2  argmin-S regret (q_argmin_s - q_best) / q_best, per state
  L3  argmin-S' regret, the same with S5's input-route term, variant `hold` (HEROSIM_PG_INPUT_LINK=hold, aa208b1d; Amendment 2)
      [descriptive: the `core` variant, and big_groups M4's peer-pair S']
  L4  (q of the best plan whose latency is distinct from the best's by > 1e-6 relative - q_best) / q_best; a state whose
      replayed plans are all tied with its best has no L4 and is counted
GO: at one admissible bandwidth below the default (--admissible, S7's load pre-check), at BOTH rungs, L1 >= 10 %, L2 >= 2 % and
L3 >= L2 / 2. A (bandwidth, rung) with an error row, or with no rows, cannot pass (unknown is not a pass). Otherwise NO-GO.

  link_bound_read.py --rows rows_0.jsonl rows_1.jsonl ... --default 1000 --admissible 500 250 --out read.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

RUNGS = ("moderate", "heavy")
L1_BAR, L2_BAR, L3_FRAC = 0.10, 0.02, 0.5


def cell_of(ds: str):
    tag = json.load(open(Path(ds) / "warm_snapshot.json"))["provenance"]["source_tag"]
    rung = next((r for r in RUNGS if re.search(rf"(^|_){r}(_|$)", tag)), None)
    bw = re.search(r"(^|_)bw(\d+(?:\.\d+)?)(_|$)", tag)
    if rung is None or bw is None:
        raise ValueError(f"FAIL LOUD: source tag {tag!r} of {ds} names no rung or no bw<MB/s> token")
    return float(bw.group(2)), rung


def measures(v: list) -> dict:
    rel = lambda x, k: (x[k] - x["q_best"]) / x["q_best"]
    l4 = [rel(x, "q_second_distinct") for x in v if x.get("q_second_distinct") is not None]
    return dict(
        n=len(v),
        L1=median(x["best_link_wait"] / x["q_best"] for x in v),
        L2=median(rel(x, "q_argmin_s") for x in v),
        L3=median(rel(x, "q_argmin_s_hold") for x in v),
        L3_core=median(rel(x, "q_argmin_s_core") for x in v), L3_peer_pair=median(rel(x, "q_argmin_s2") for x in v),
        hold_wait_zero=sum(x["hold_wait_cd"] == 0 for x in v) / len(v),
        hold_wait_cd_med=median(x["hold_wait_cd"] for x in v), core_wait_cd_med=median(x["core_wait_cd"] for x in v),
        selfcheck_fail=sum(x[f"{il}_zero_equal"] is False for x in v for il in ("hold", "core")),
        hold_moves=sum(x["s_hold_moves"] for x in v) / len(v), core_moves=sum(x["s_core_moves"] for x in v) / len(v),
        L4=median(l4) if l4 else None, L4_all_tied=len(v) - len(l4),
        # context, not in the bar
        L2_pooled=(sum(x["q_argmin_s"] for x in v) - sum(x["q_best"] for x in v)) / sum(x["q_best"] for x in v),
        cd_exacts=median(rel(x, "q_cd_exacts") for x in v),
        argmin_s_misses=sum(x["q_argmin_s"] > x["q_best"] * (1 + 1e-9) for x in v) / len(v),
        L1_at_argmin_s=median(x["argmin_s_link_wait"] / x["q_argmin_s"] for x in v),
        link_transfer_share=median(x["best_link_transfer"] / x["q_best"] for x in v),
        peer_exchange_share=median(x["best_peer_exchange"] / x["q_best"] for x in v),
        s_exact=sum(x["s_exact"] for x in v) / len(v), converged=sum(x["converged"] for x in v) / len(v),
        replays=median(x["replays"] for x in v), last20=sum(x["improved_last20"] for x in v) / len(v),
        exhausted=sum(x["exhausted"] for x in v) / len(v))


def read(rows: list, default: float, admissible: list) -> dict:
    cells, errors = defaultdict(list), defaultdict(int)
    for r in rows:
        key = cell_of(r["ds"])
        if "error" in r:
            errors[key] += 1
        else:
            cells[key].append(r)
    res = {}
    for key in sorted(set(cells) | set(errors)):
        m = measures(cells[key]) if cells[key] else {"n": 0}
        m["errors"] = errors[key]
        res[key] = m
    verdict = {}
    for bw in sorted({k[0] for k in res}, reverse=True):
        if bw >= default:
            continue
        ks = [(bw, r) for r in RUNGS]
        why = []
        if bw not in admissible:
            why.append("not admissible (S7's load pre-check)")
        for k in ks:
            m = res.get(k)
            if m is None or m["n"] == 0:
                why.append(f"{k[1]}: no rows")
                continue
            if m["selfcheck_fail"]:
                why.append(f"{k[1]}: {m['selfcheck_fail']} S' == S self-check failures")
            if m["errors"]:
                why.append(f"{k[1]}: {m['errors']} error rows")
            if m["L1"] < L1_BAR:
                why.append(f"{k[1]}: L1 {100*m['L1']:.2f} % < 10 %")
            if m["L2"] < L2_BAR:
                why.append(f"{k[1]}: L2 {100*m['L2']:.2f} % < 2 %")
            if m["L3"] < L3_FRAC * m["L2"]:
                why.append(f"{k[1]}: L3 {100*m['L3']:.2f} % < L2 / 2")
        verdict[bw] = {"pass": not why, "why": why}
    go = any(v["pass"] for v in verdict.values())
    return {"cells": {f"{k[0]:g}|{k[1]}": v for k, v in res.items()}, "per_bandwidth": {f"{k:g}": v for k, v in verdict.items()},
            "verdict": "GO" if go else "NO-GO", "default": default, "admissible": admissible}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--default", type=float, required=True, help="the default core bandwidth (MB/s); never a screen bandwidth")
    ap.add_argument("--admissible", type=float, nargs="*", default=[], help="S7's admissible bandwidths below the default")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.rows for l in open(f) if l.strip()]
    r = read(rows, a.default, a.admissible)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    pct = lambda x: "-" if x is None else f"{100*x:6.2f} %"
    print(f"{len(rows)} rows; default {a.default:g}, admissible {a.admissible}")
    for k, m in r["cells"].items():
        if not m["n"]:
            print(f"  {k:16s} n 0, errors {m['errors']}")
            continue
        print(f"  {k:16s} n {m['n']:3d} err {m['errors']} | L1 {pct(m['L1'])} L2 {pct(m['L2'])} L3 {pct(m['L3'])} L4 {pct(m['L4'])} "
              f"(all tied {m['L4_all_tied']}) | L3 core {pct(m['L3_core'])}, peer-pair {pct(m['L3_peer_pair'])}, S'-hold moves argmin-S in "
              f"{100*m['hold_moves']:.0f} % (core {100*m['core_moves']:.0f} %); S'-hold term at the cd plan zero in {100*m['hold_wait_zero']:.0f} %, "
              f"median {m['hold_wait_cd_med']:.3f} s (core {m['core_wait_cd_med']:.3f} s), S' == S self-check failures {m['selfcheck_fail']} | L2 pooled {pct(m['L2_pooled'])}, cd_exactS {pct(m['cd_exacts'])}, argmin-S misses "
              f"{100*m['argmin_s_misses']:.0f} %, L1 at argmin-S {pct(m['L1_at_argmin_s'])}, link transfer {pct(m['link_transfer_share'])}, "
              f"peer exchange {pct(m['peer_exchange_share'])} | S exact {100*m['s_exact']:.0f} %, converged {100*m['converged']:.0f} %, "
              f"replays {m['replays']:.0f}, last-20 % {100*m['last20']:.0f} %, exhausted {100*m['exhausted']:.0f} %")
    for bw, v in r["per_bandwidth"].items():
        print(f"  bw {bw}: {'PASS' if v['pass'] else 'fail'} {'; '.join(v['why'])}")
    print(f"VERDICT {r['verdict']}" + ("" if r["verdict"] == "GO" else " -- the record states: search on S is the bar in the accel seat"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

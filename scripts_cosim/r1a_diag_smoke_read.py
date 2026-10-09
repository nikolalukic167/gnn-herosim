#!/usr/bin/env python3
"""Read the r1a_diag_smoke.sbatch output (DESCRIPTIVE ONLY: a mid-training snapshot on training topologies, never a result).

Per cell (topology x rung, window g0): average task latency of the learned arm, CD and reactive, the paired % of the arm vs CD (and vs reactive),
queue share, and what the arm's decode did: spread (distinct nodes per decoded batch, mean and as a share of batch size), co-location rate (share of
in-batch peer pairs the decoded plan puts on one node), and the refine / sibling-move counters. CD's own books: partners joined / known.
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import statistics
from pathlib import Path


def load_summary(path: str):
    return json.load(open(path))


def traces(path: Path):
    out = []
    if not path.is_file():
        return out
    with open(path, "rb") as fh:
        while True:
            try:
                out.append(pickle.load(fh))
            except EOFError:
                break
    return out


def pairs_of(rec):
    g = rec.get("graph")
    ctx = getattr(g, "partial_state_ctx", None) or {}
    pp = ctx.get("peer_pairs")
    if pp is None:
        return None
    items = list(pp.keys()) if isinstance(pp, dict) else [tuple(x[:2]) for x in pp]
    n = len(rec["task_ids"])
    return [(int(i), int(j)) for i, j in items if 0 <= int(i) < n and 0 <= int(j) < n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", default="gnn_eng")
    ap.add_argument("--topos", default="9101,9103")
    a = ap.parse_args()
    print("DESCRIPTIVE ONLY -- mid-training checkpoint, training topologies; not a result.\n")
    print(f"{'cell':18s} {'arm lat':>8s} {'CD lat':>8s} {'react':>8s} {'%vsCD':>7s} {'%vsReact':>9s} {'q-share arm/CD/react':>22s} | spread  nodes/batch  coloc  batches  refine  sibling")
    for T in a.topos.split(","):
        for R in ("light", "moderate", "heavy"):
            tag = f"cc40s{T}__g0{R}"
            def one(pattern):
                fs = glob.glob(str(a.out / pattern))
                return load_summary(fs[0]) if fs else None
            arm = one(f"learned_{T}_{R}/{tag}__ra_{a.arm}_s0.summary.json")
            cd = one(f"classical/{tag}__cd_s0.summary.json")
            re = one(f"classical/{tag}__reactive_s0.summary.json")
            if not (arm and cd and re):
                print(f"{T}/{R:9s} MISSING summaries: arm={bool(arm)} cd={bool(cd)} reactive={bool(re)}")
                continue
            la, lc, lr = (float(x["averageElapsedTime"]) for x in (arm, cd, re))
            qs = "/".join(f"{float(x.get('queue_share') or 0):.2f}" for x in (arm, cd, re))
            recs = traces(a.out / f"trace_{T}_{R}.pkl")
            spread = per = coloc = None
            if recs:
                nodes = [len({int(c[0]) for c in r["combo"]}) for r in recs]
                share = [n / len(r["combo"]) for n, r in zip(nodes, recs)]
                spread, per = statistics.fmean(share), statistics.fmean(nodes)
                hit = tot = 0
                for r in recs:
                    ps = pairs_of(r)
                    if ps is None:
                        tot = None
                        break
                    for i, j in ps:
                        tot += 1
                        hit += int(r["combo"][i][0] == r["combo"][j][0])
                coloc = None if not tot else hit / tot
            sc = arm.get("schedulerCounters") or {}
            f = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(f"{T}/{R:9s}      {la:8.3f} {lc:8.3f} {lr:8.3f} {100 * (la - lc) / lc:+7.1f} {100 * (la - lr) / lr:+9.1f} {qs:>22s} | {f(spread):>6s} {f(per):>11s} {f(coloc):>6s} {len(recs):7d} "
                  f"{int(sc.get('prefix_self_refine_moves') or 0):6d} {int(sc.get('prefix_sibling_moves') or 0):7d}")
            ccd = cd.get("schedulerCounters") or {}
            if ccd.get("pg_partners_known"):
                print(f"{'':18s} CD joined a known partner's node on {int(ccd.get('pg_joined_partner') or 0)} of {int(ccd.get('pg_decisions') or 0)} decisions "
                      f"({int(ccd.get('pg_partners_known'))} known partners), CD refine moves {int(ccd.get('pg_cd_moves') or 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

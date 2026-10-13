#!/usr/bin/env python3
"""warm_surge_s0_v1 leak checks 1-3 (docs/lineages/warm_surge_s0_v1.md), fail loud.

  workload   --src <inputs>/wf1_<rung> --out <inputs'> --rung heavy --t-switch 1200 [--ks 1 2 3] [--wl grounded_g0_n50000]
             check 1: groups anchored before t_s byte-identical; post-switch offsets = source / k (<= 1e-9); type/qos/client/peer_group/peer_exchange
             identical; no group straddles; the sha chain matches the source; cfg files byte-identical.
  pre        --cells <dir_k1> <dir_k2> ... --stem cc40s19420__g0heavy{tag}__cd_s0 --t-switch 1200
             check 2: the (dispatched, done) rows with done < t_s are identical across k.
  parity     --k1 <dir> --ref <gate_B dir> --rung heavy --arm cd [--ks k1]
             check 3: averageElapsedTime of the k = 1 cell equals the reference cell per topology.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_workload(src_wl: dict, out_wl: dict, t_s: float, k: float, src_sha: str) -> dict:
    a, b = src_wl["events"], out_wl["events"]
    assert len(a) == len(b), "event count changed"
    anchor = {}
    for e in a:
        anchor[e.get("peer_group")] = min(anchor.get(e.get("peer_group"), e["timestamp"]), e["timestamp"])
    last = {}
    for e in a:
        last[e.get("peer_group")] = max(last.get(e.get("peer_group"), e["timestamp"]), e["timestamp"])
    straddle = sum(1 for g in anchor if anchor[g] < t_s <= last[g])
    n_pre = n_post = 0
    max_err = 0.0
    for x, y in zip(a, b):
        if {kk: v for kk, v in x.items() if kk != "timestamp"} != {kk: v for kk, v in y.items() if kk != "timestamp"}:
            raise SystemExit("FAIL LOUD: a non-timestamp event field changed")
        if anchor[x.get("peer_group")] < t_s:
            if x["timestamp"] != y["timestamp"]:
                raise SystemExit("FAIL LOUD: a pre-switch group moved")
            n_pre += 1
        else:
            max_err = max(max_err, abs((y["timestamp"] - t_s) * k - (x["timestamp"] - t_s)))
            n_post += 1
    if max_err > 1e-9 * max(1.0, k):
        raise SystemExit(f"FAIL LOUD: post-switch offset error {max_err}")
    if straddle:
        raise SystemExit(f"FAIL LOUD: {straddle} groups straddle the switch")
    if src_wl.get("peer_exchange") != out_wl.get("peer_exchange"):
        raise SystemExit("FAIL LOUD: peer_exchange changed")
    rec = out_wl.get("warm_surge_s0_v1")
    if k != 1:
        if not rec or rec["source_sha256"] != src_sha or (rec["n_pre"], rec["n_post"]) != (n_pre, n_post):
            raise SystemExit(f"FAIL LOUD: sha chain / counts wrong: {rec}")
    return {"n_pre": n_pre, "n_post": n_post, "max_offset_err": max_err, "straddling_groups": straddle}


def cmd_workload(a):
    for k in a.ks:
        name = a.wl + ".json"
        src = os.path.join(a.src, "wl", name)
        outd = os.path.join(a.out, f"wf1_{a.rung}k{k}")
        out = os.path.join(outd, "wl", name)
        if k == 1:
            if _sha(src) != _sha(out):
                raise SystemExit("FAIL LOUD: k = 1 is not a byte copy")
            print(f"{a.rung} k=1: byte copy OK")
        else:
            r = check_workload(json.load(open(src)), json.load(open(out)), a.t_switch, float(k), _sha(src))
            print(f"{a.rung} k={k}: {r}")
        for c in sorted(glob.glob(os.path.join(a.src, "cfg", "*.json"))):
            if _sha(c) != _sha(os.path.join(outd, "cfg", os.path.basename(c))):
                raise SystemExit(f"FAIL LOUD: cfg changed {c}")
    print("CHECK1_OK")


def cmd_pre(a):
    import numpy as np

    ref = None
    for d, tag in zip(a.cells, a.tags):
        r = np.load(os.path.join(d, a.stem.format(tag=tag) + ".tasks.npy"))
        pre = r[r[:, 1] < a.t_switch]
        pre = pre[np.lexsort((pre[:, 1], pre[:, 0]))]
        if ref is None:
            ref = pre
        elif pre.shape != ref.shape or not (pre == ref).all():
            raise SystemExit(f"FAIL LOUD: pre-switch rows differ at {tag} ({pre.shape} vs {ref.shape})")
        print(f"  {tag}: {len(pre)} tasks done before {a.t_switch}")
    print("CHECK2_OK")


def cmd_parity(a):
    bad = 0
    for p in sorted(glob.glob(os.path.join(a.k1, f"*__g0{a.rung}k1__{a.arm}_s0.summary.json"))):
        topo = os.path.basename(p).split("__")[0]
        ref = os.path.join(a.ref, f"{topo}__g0{a.rung}__{a.arm}_s0.summary.json")
        x, y = json.load(open(p)), json.load(open(ref))
        same = x["averageElapsedTime"] == y["averageElapsedTime"] and x["endTime"] == y["endTime"]
        bad += not same
        print(f"  {topo} {a.rung}: k1 {x['averageElapsedTime']!r} ref {y['averageElapsedTime']!r} {'OK' if same else 'DIFF'}")
    if bad:
        raise SystemExit(f"FAIL LOUD: {bad} cells differ from the reference; physics moved. Do not proceed")
    print("CHECK3_OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("workload")
    w.add_argument("--src", required=True); w.add_argument("--out", required=True); w.add_argument("--rung", required=True)
    w.add_argument("--t-switch", type=float, default=1200.0); w.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    w.add_argument("--wl", default="grounded_g0_n50000"); w.set_defaults(fn=cmd_workload)
    p = sub.add_parser("pre")
    p.add_argument("--cells", nargs="+", required=True); p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--stem", required=True); p.add_argument("--t-switch", type=float, default=1200.0); p.set_defaults(fn=cmd_pre)
    q = sub.add_parser("parity")
    q.add_argument("--k1", required=True); q.add_argument("--ref", required=True); q.add_argument("--rung", required=True)
    q.add_argument("--arm", default="cd"); q.set_defaults(fn=cmd_parity)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())

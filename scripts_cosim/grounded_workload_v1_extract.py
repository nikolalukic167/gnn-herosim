#!/usr/bin/env python3
"""grounded_workload_v1 -- reduce one Alibaba 2021 microservice call-graph shard to a fan-out library
(docs/lineages/grounded_workload_v1.md).

Source: alibaba/clusterdata cluster-trace-microservices-v2021, MSCallGraph_<k>.csv (Luo et al., SoCC'21).
A request is a `traceid`; its arrival is the earliest `timestamp` (ms) of any of its calls. A fan-out is the
set of calls in one trace that share an rpcid parent (rpcid minus its last segment) -- siblings dispatched by
the same caller. For the first --n-traces requests in arrival order, one fan-out is drawn uniformly (seeded)
among the trace's fan-outs of size >= 2; a trace with none is a singleton. The fan-out keeps its real
sibling offsets (ms, relative to its first sibling), truncated to the first --max-size siblings by time.

  grounded_workload_v1_extract.py --csv MSCallGraph_0.csv --out lib_0.json --seed 1 [--n-traces 40000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-traces", type=int, default=40000)
    ap.add_argument("--max-size", type=int, default=10)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists; a library is frozen once minted")

    df = pd.read_csv(a.csv, usecols=["traceid", "timestamp", "rpcid"],
                     dtype={"traceid": "str", "timestamp": "int64", "rpcid": "str"})
    n_rows = len(df)
    start = df.groupby("traceid", sort=False)["timestamp"].min().sort_values(kind="stable")
    if len(start) < a.n_traces:
        raise SystemExit(f"FAIL LOUD: shard has {len(start)} traces < --n-traces {a.n_traces}")
    keep = start.iloc[: a.n_traces]
    sub = df[df["traceid"].isin(set(keep.index))].dropna(subset=["rpcid"])
    sub = sub[sub["rpcid"].str.fullmatch(r"\d+(\.\d+)+")]
    sub = sub.assign(parent=sub["rpcid"].str.rsplit(".", n=1).str[0])

    fanouts = {}
    for (tid, _parent), g in sub.groupby(["traceid", "parent"], sort=False):
        if len(g) >= 2:
            fanouts.setdefault(tid, []).append(np.sort(g["timestamp"].to_numpy()))

    rng = random.Random(a.seed)
    groups = []
    n_trunc = 0
    raw_sizes = []
    for tid, t0 in keep.items():
        cands = fanouts.get(tid)
        if not cands:
            groups.append({"t0_ms": int(t0), "offsets_ms": [0], "size_raw": 1})
            raw_sizes.append(1)
            continue
        ts = cands[rng.randrange(len(cands))]
        size_raw = len(ts)
        raw_sizes.append(size_raw)
        if size_raw > a.max_size:
            n_trunc += 1
            ts = ts[: a.max_size]
        groups.append({"t0_ms": int(t0), "offsets_ms": [int(x - ts[0]) for x in ts], "size_raw": size_raw})

    sizes = np.array([len(g["offsets_ms"]) for g in groups])
    spans = np.array([g["offsets_ms"][-1] for g in groups if len(g["offsets_ms"]) > 1], dtype=float)
    doc = {
        "source": {"dataset": "alibaba/clusterdata cluster-trace-microservices-v2021", "file": os.path.basename(a.csv),
                   "md5": _md5(a.csv), "rows": n_rows, "traces_in_shard": int(len(start))},
        "rule": {"seed": a.seed, "n_traces": a.n_traces, "max_size": a.max_size,
                 "fanout": "uniform among the trace's rpcid-parent sibling sets of size >= 2; none -> singleton"},
        "summary": {"singleton_share": float((sizes == 1).mean()), "mean_size": float(sizes.mean()),
                    "size_counts": {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))},
                    "truncated_share": n_trunc / len(groups),
                    "span_ms_p50": float(np.percentile(spans, 50)), "span_ms_p90": float(np.percentile(spans, 90)),
                    "span_ms_p99": float(np.percentile(spans, 99)),
                    "arrival_span_s": (int(keep.iloc[-1]) - int(keep.iloc[0])) / 1000.0},
        "groups": groups,
    }
    tmp = a.out + ".partial"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(json.dumps({k: doc[k] for k in ("source", "rule", "summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

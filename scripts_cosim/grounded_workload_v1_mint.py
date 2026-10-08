#!/usr/bin/env python3
"""grounded_workload_v1 -- mint a study workload whose group structure comes from Alibaba's call graphs
(docs/lineages/grounded_workload_v1.md).

From the fan-out library (grounded_workload_v1_extract.py) and a pre-burst study window (drainable_*_n50000):
  * groups, in request-arrival order, take the library's fan-out sizes until --n-tasks tasks are placed
    (the last group is cut to fit);
  * a group's dispatch time is its request's arrival, with the library's inter-request times stretched by one
    factor so the task rate equals the base window's (the admissible x1 load of the study);
  * a member's time is its group's dispatch time plus its REAL sibling offset (ms, never stretched);
  * each task takes the k-th base event's type, demand scale, QoS and client, so the type mix and sources are
    the study's;
  * peers are drawn with the study's rule (make_peer_affinity_production_workload.py): each member draws
    min(2, size - 1) distinct partners in its group, payload 200 MB * 10**U(-1, 1).
Events are sorted by time (stable) and peer indices refer to the sorted order.

  grounded_workload_v1_mint.py --lib fanout_lib_0.json --base drainable_f4000_n50000.json --seed 11 --out g0.json

--merge-k K (scale_sweep_v1): K consecutive library groups become one group, dispatched at the first one's arrival;
each member keeps its request's raw ms lag behind that arrival plus its sibling offset (neither is stretched), so
the task rate is unchanged and a decision batch holds ~K times the tasks. Partners are drawn over the merged group,
and one bridging pair joins any components the draw leaves apart, so the batched seat's peer closure is the whole
group. K = 1 is byte-identical to the original mint.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import statistics as st
from typing import Dict, List, Tuple

from src.placement.workload_payloads import (
    PAYLOAD_SAMPLERS, payload_rng, payload_sampler_meta, require_sampler, sample_payload_bytes,
)

PARTNERS = 2
X_SCALE_BYTES = 200e6
LOG10_SPREAD = 1.0


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def merge_groups(groups: List[dict], k: int) -> List[dict]:
    if k == 1:
        return groups
    out = []
    for i in range(0, len(groups), k):
        chunk = groups[i:i + k]
        t0 = int(chunk[0]["t0_ms"])
        offs = sorted(int(g["t0_ms"]) - t0 + int(o) for g in chunk for o in g["offsets_ms"])
        out.append({"t0_ms": t0, "offsets_ms": offs})
    return out


def plan_groups(groups: List[dict], n_tasks: int) -> List[Tuple[int, List[int]]]:
    out: List[Tuple[int, List[int]]] = []
    placed = 0
    for g in groups:
        offs = list(g["offsets_ms"])[: n_tasks - placed]
        out.append((int(g["t0_ms"]), offs))
        placed += len(offs)
        if placed == n_tasks:
            return out
    raise SystemExit(f"FAIL LOUD: library holds only {placed} tasks < {n_tasks}")


def mint(lib: dict, base: dict, seed: int, n_tasks: int, merge_k: int = 1,
         payload_sampler: str = "legacy") -> Tuple[dict, dict]:
    require_sampler(payload_sampler)
    plan = plan_groups(merge_groups(lib["groups"], merge_k), n_tasks)
    bev = base["events"][:n_tasks]
    if len(bev) < n_tasks:
        raise SystemExit("FAIL LOUD: base window shorter than --n-tasks")
    bts = [float(e["timestamp"]) for e in bev]
    rate = n_tasks / (bts[-1] - bts[0])
    rng = random.Random(seed)
    payload_rng_ = payload_rng(seed) if payload_sampler == "wf1_v1" else None

    def payload() -> float:
        legacy = X_SCALE_BYTES * (10.0 ** rng.uniform(-LOG10_SPREAD, LOG10_SPREAD))
        # wf1_v1 still consumes the legacy draw so the partner draws after it do not move: the stages differ in
        # payloads only, on an identical pair graph.
        return legacy if payload_rng_ is None else sample_payload_bytes(payload_rng_)

    # The trace stores whole milliseconds; a stretch of ~4000x would put every arrival on a ~4 s lattice and
    # dispatch same-millisecond requests at one instant. Spread each arrival uniformly inside its millisecond.
    t0s = sorted(t0 + rng.random() for t0, _offs in plan)
    t_first = t0s[0]
    arrival_span_s = (t0s[-1] - t_first) / 1000.0
    if arrival_span_s <= 0:
        raise SystemExit("FAIL LOUD: degenerate arrival span")
    stretch = (n_tasks / rate) / arrival_span_s

    raw: List[dict] = []
    k = 0
    for gi, (t0, (_t0_int, offs)) in enumerate(zip(t0s, plan)):
        anchor = bts[0] + (t0 - t_first) / 1000.0 * stretch
        for o in offs:
            src = bev[k]
            ev = {"timestamp": anchor + o / 1000.0,
                  "application": copy.deepcopy(src["application"]),
                  "qos": copy.deepcopy(src["qos"]), "node_name": src["node_name"], "peer_group": gi}
            raw.append(ev)
            k += 1
    order = sorted(range(len(raw)), key=lambda i: (raw[i]["timestamp"], i))
    new_idx = {old: new for new, old in enumerate(order)}
    events = [raw[i] for i in order]

    pairs: Dict[Tuple[int, int], float] = {}
    k = 0
    for gi, (_t0, offs) in enumerate(plan):
        members = [new_idx[k + j] for j in range(len(offs))]
        k += len(offs)
        p = min(PARTNERS, len(members) - 1)
        for i in members:
            for j in (rng.sample([m for m in members if m != i], p) if p > 0 else []):
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    pairs[key] = payload()
        if merge_k > 1 and len(members) > 1:
            root = {m: m for m in members}

            def find(x: int) -> int:
                while root[x] != x:
                    root[x] = root[root[x]]
                    x = root[x]
                return x
            for (a, b) in pairs:
                if a in root and b in root:
                    root[find(a)] = find(b)
            comps = sorted({find(m) for m in members})
            for a, b in zip(comps, comps[1:]):
                key = (min(a, b), max(a, b))
                pairs[key] = payload()
                root[find(a)] = find(b)

    sizes = [len(offs) for _t, offs in plan]
    spans = [offs[-1] for _t, offs in plan if len(offs) > 1]
    interleaved = sum(1 for a, b in zip(order, order[1:]) if raw[a]["peer_group"] > raw[b]["peer_group"])
    ts = [e["timestamp"] for e in events]
    meta = {
        "n_tasks": n_tasks, "n_groups": len(plan), "n_pairs": len(pairs), "seed": seed,
        "rate_per_s": n_tasks / (ts[-1] - ts[0]), "base_rate_per_s": rate, "arrival_stretch": stretch,
        "mean_group_size": st.mean(sizes), "singleton_share": sizes.count(1) / len(sizes),
        "group_span_ms": {"median": st.median(spans), "p90": sorted(spans)[int(0.9 * (len(spans) - 1))],
                          "max": max(spans)},
        "interleaved_boundaries": interleaved,
        "peer_rule": {"partners": PARTNERS, "x_scale_bytes": X_SCALE_BYTES, "log10_spread": LOG10_SPREAD},
    }
    if merge_k > 1:
        meta["merge_k"] = merge_k
    if payload_sampler != "legacy":
        meta["payload_sampler"] = payload_sampler_meta()
    doc = {"rps": base["rps"], "duration": base["duration"], "events": events,
           "peer_exchange": [[i, j, b] for (i, j), b in sorted(pairs.items())]}
    return doc, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tasks", type=int, default=50000)
    ap.add_argument("--merge-k", type=int, default=1)
    ap.add_argument("--payload-sampler", choices=PAYLOAD_SAMPLERS, default="legacy",
                    help="legacy: 200 MB * 10**U(-1,1). wf1_v1: workload_fix_v1 W2 (log-normal 4 MB + heavy tier)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists; a gate workload is frozen once minted")
    lib = json.load(gzip.open(a.lib, "rt") if a.lib.endswith(".gz") else open(a.lib))
    base = json.load(open(a.base))
    if a.merge_k < 1:
        raise SystemExit("FAIL LOUD: --merge-k must be >= 1")
    doc, meta = mint(lib, base, a.seed, a.n_tasks, a.merge_k, a.payload_sampler)
    doc["grounded_workload_v1"] = {**meta, "library": os.path.basename(a.lib), "library_sha256": _sha(a.lib),
                                   "library_source": lib["source"], "base": os.path.basename(a.base),
                                   "base_sha256": _sha(a.base)}
    tmp = a.out + ".partial"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    os.replace(tmp, a.out)
    print(json.dumps(doc["grounded_workload_v1"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

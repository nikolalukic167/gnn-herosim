#!/usr/bin/env python3
"""peer_affinity_v1 stage 3: put the corpus's peer structure onto a real production trace.

The production traces (`data/nofs-ids/traces/workload-<rps>-<duration>.json`, ~450-560k
single-task events at ~2,650 arrivals/s) carry no `peer_exchange`, so under
HEROSIM_PEER_EXCHANGE=1 the physics the peer_affinity checkpoints were fitted on is inert
on them. This script adds it, with the SAME generative rule the corpus used
(`generate_gnn_datasets_fast.py`, grid key `peer_exchange`): consecutive arrivals are
grouped into batches of `--group-size` (= the co-sim batch, k = 10), every task in a group
draws `--partners` distinct peers inside its group, and each pair's payload is
`x_scale * 10**U(-spread, spread)` bytes. Per-event `demand_scale` is drawn the way the
corpus's `demand_spread` draws it (uniform [0.5, 2.0]) so the decode-side cap binds the
way it did in training. One `random.Random(seed)` for everything; global task ids are the
event index (every event is a single-task application — asserted).

Nothing else about the trace changes: timestamps, task types, source clients, QoS.

Usage:
  PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH=. pipenv run python3 \
    scripts_cosim/make_peer_affinity_production_workload.py \
      --trace data/nofs-ids/traces/workload-150-150.json --seed 7300 \
      --output data/nofs-ids/traces/workload-150-150-peer_p2_x200.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--group-size", type=int, default=10, help="tasks per peer group (the corpus batch, k)")
    ap.add_argument("--partners", type=int, default=2, help="peers drawn per task (corpus GO cell: 2)")
    ap.add_argument("--x-scale-bytes", type=float, default=200e6, help="corpus GO cell: 200 MB")
    ap.add_argument("--log10-spread", type=float, default=1.0)
    ap.add_argument("--demand-spread", type=float, nargs=2, default=(0.5, 2.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--max-events", type=int, default=None, help="truncate the trace (smoke runs only)")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output} — a gate workload is frozen once a run depends on it.")
    if not (1 <= args.partners < args.group_size):
        raise SystemExit(f"--partners must be in [1, group_size-1], got {args.partners}")

    raw = args.trace.read_bytes()
    trace = json.loads(raw)
    events = trace["events"]
    if args.max_events:
        events = events[: args.max_events]
    for i, ev in enumerate(events):
        dag = ev["application"]["dag"]
        if not isinstance(dag, dict) or len(dag) != 1 or any(dag.values()):
            raise SystemExit(f"event {i}: expected a single-task application, got dag={dag!r}")
    ts = [float(ev["timestamp"]) for ev in events]
    if any(b < a for a, b in zip(ts, ts[1:])):
        raise SystemExit("trace timestamps are not non-decreasing; group-by-arrival needs that")

    rng = random.Random(args.seed)
    lo, hi = args.demand_spread
    pairs: Dict[Tuple[int, int], float] = {}
    n_groups = 0
    for g0 in range(0, len(events), args.group_size):
        members = list(range(g0, min(g0 + args.group_size, len(events))))
        n_groups += 1
        for i in members:
            ev = events[i]
            ttype = next(iter(ev["application"]["dag"]))
            ev["application"]["demand_scale"] = {ttype: rng.uniform(lo, hi)}
            ev["peer_group"] = n_groups - 1
        if len(members) <= args.partners:
            # a short tail group cannot host `partners` distinct peers; leave it peer-free
            # (counted below) rather than change the rule for it
            continue
        for i in members:
            for j in rng.sample([j for j in members if j != i], args.partners):
                key = (min(i, j), max(i, j))
                if key not in pairs:
                    pairs[key] = args.x_scale_bytes * (10.0 ** rng.uniform(-args.log10_spread, args.log10_spread))

    peer_exchange: List[List[float]] = [[i, j, b] for (i, j), b in sorted(pairs.items())]
    group_spans = []
    for g0 in range(0, len(events), args.group_size):
        seg = ts[g0: g0 + args.group_size]
        group_spans.append(seg[-1] - seg[0])
    out = {
        "rps": trace["rps"],
        "duration": trace["duration"],
        "events": events,
        "peer_exchange": peer_exchange,
        "peer_augmentation": {
            "source_trace": str(args.trace),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_events": len(trace["events"]),
            "events_used": len(events),
            "seed": args.seed,
            "group_size": args.group_size,
            "partners": args.partners,
            "x_scale_bytes": args.x_scale_bytes,
            "log10_spread": args.log10_spread,
            "demand_spread": {"dist": "uniform", "params": [lo, hi]},
            "n_groups": n_groups,
            "n_pairs": len(pairs),
            "group_span_seconds": {
                "median": st.median(group_spans), "p90": sorted(group_spans)[int(0.9 * (len(group_spans) - 1))],
                "max": max(group_spans),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":")))
    b = [p[2] for p in peer_exchange]
    print(
        f"[peer-workload] {args.output}: {len(events)} events, {n_groups} groups of {args.group_size}, "
        f"{len(pairs)} pairs, bytes median {st.median(b)/1e6:.1f} MB (min {min(b)/1e6:.1f}, max {max(b)/1e6:.1f}); "
        f"group span median {st.median(group_spans)*1000:.2f} ms, max {max(group_spans)*1000:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

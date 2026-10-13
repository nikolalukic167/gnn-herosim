"""Arm 1 post-hoc cache check: on every built graph, the partial-state demand table equals the candidate set, and the route table's nodes
are the demand table's nodes. Fails loud; prints the number of graphs checked and failed."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path


def check_graph(g) -> list:
    ctx = g.partial_state_ctx
    D = set(ctx["demand"].keys())
    C = {(t, tuple(int(v) for v in c)) for t in range(int(g.n_tasks)) for c in g.task_logit_to_placement[t]}
    bad = []
    if D != C:
        bad.append(f"demand != candidates: {len(D - C)} demand keys outside, {len(C - D)} candidates without demand")
    d_nodes = sorted({int(p[0]) for _, p in D})
    r_nodes = sorted({int(a) for a, _b in ctx["route_hops_bneck"]})
    if d_nodes != r_nodes:
        bad.append(f"route table nodes {len(r_nodes)} != demand nodes {len(d_nodes)}")
    return bad


def main() -> int:
    cache = Path(sys.argv[1])
    graphs = pickle.load(open(cache / "graphs.pkl", "rb"))
    ids = pickle.load(open(cache / "dataset_ids.pkl", "rb"))
    failed = []
    for gid, g in zip(ids, graphs):
        bad = check_graph(g)
        if bad:
            failed.append((gid, bad))
    print(f"graphs checked: {len(graphs)}; failed: {len(failed)}")
    for gid, bad in failed[:10]:
        print(" ", gid, bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

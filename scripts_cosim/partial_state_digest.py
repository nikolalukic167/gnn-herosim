#!/usr/bin/env python3
"""sha256 of the partial-state columns of every (task, committed-prefix) of an existing cache, under its own contract.

Run it in two checkouts on the same cache; equal digests mean the code between them leaves that contract's features
byte for byte unchanged (partial_state_v5 prep: v3 and v4 must not move).

  PARTIAL_STATE_CONTRACT=partial_state_v4 python3 scripts_cosim/partial_state_digest.py <cache dir> [--limit N]
"""
import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.policy.tabular.reduced_features import (  # noqa: E402
    build_partial_state_context_from_graph, partial_state_columns, resolve_partial_state_contract,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    graphs = pickle.load(open(a.cache_dir / "graphs.pkl", "rb"))
    if a.limit:
        graphs = graphs[: a.limit]
    h = hashlib.sha256()
    rows = 0
    for g in graphs:
        ctx = build_partial_state_context_from_graph(g)
        tl = g.task_logit_to_placement
        committed = {}
        for t in range(int(g.n_tasks)):
            cands = [tuple(c) for c in tl[t]]
            cols = np.ascontiguousarray(partial_state_columns(ctx, t, cands, dict(committed)))
            h.update(cols.tobytes())
            rows += cols.shape[0]
            if cands:
                committed[t] = cands[0]
    print(f"{resolve_partial_state_contract()} graphs={len(graphs)} rows={rows} width={cols.shape[1]} sha256={h.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

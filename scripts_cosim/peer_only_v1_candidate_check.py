"""Cache guard: every candidate of every task must have a demand entry (i.e. appear in the
placement sweep the partial-state context was built from).

    python3 scripts_cosim/peer_only_v1_candidate_check.py --cache DIR --out offenders.json

The partial-state demand table is built from the sweep rows (prepare_graphs_cache
attach_dag_partial_state_block), while the graph's candidate set lists every replica of the
task's type. When a replica exists that no sweep row ever used, the trainer dies at the first
prefix-conditioned score with "candidate absent from the partial-state context"
(peer_only_v1 Phase B, job 782166: 3 of 1,141 new train2 datasets, candidate (22, 118) on
tasks 1-2). Exits 1 and lists the offenders so a corpus job can set them aside BEFORE training.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Sequence


def offenders(cache: Path) -> dict:
    graphs = pickle.load(open(cache / "graphs.pkl", "rb"))
    out = {}
    for g in graphs:
        dem = g.partial_state_ctx["demand"]
        missing = []
        for t in range(int(g.n_tasks)):
            for c in g.task_logit_to_placement[t]:
                c = tuple(int(x) for x in c)
                if (t, c) not in dem:
                    missing.append([t, list(c)])
        if missing:
            out[str(g.dataset_id)] = missing
    return {"n_graphs": len(graphs), "n_offenders": len(out), "offenders": out}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    res = offenders(a.cache)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"[candidate-check] graphs={res['n_graphs']} offenders={res['n_offenders']}"
          + (f" e.g. {sorted(res['offenders'])[:3]}" if res["offenders"] else ""))
    return 1 if res["offenders"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

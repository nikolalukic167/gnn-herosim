#!/usr/bin/env python3
"""cluster_scale_v1 -- mint the rungs' cells and compute S0.d, the candidate-support bar.

Cells differ from the base in EXACTLY two fields, asserted field-by-field: the server node
count and the topology seed. Nothing else -- no physics, no payload, no scheduler setting.

S0.d needs no simulation. With `replicas.<type>.per_server = 1` every server hosts every task
type, so a task's candidate set is the set of servers its client can reach, which the topology
generator settles deterministically. The corpus maximum is 5 candidates per task
(graphs_cache_drainable_objective_v1_v1, 516 datasets, 5,160 task slots: p50 3, p90 4, p99 5,
max 5), so a rung whose reachable-server count exceeds it is OUT-OF-SUPPORT before it runs and
S1 on that rung is CONFOUNDED-CANDIDATE-SUPPORT whatever its latency.

    python3 scripts_cosim/cluster_scale_v1_cells.py --base <cell>.json --sim-inputs data/nofs-ids \\
        --rungs 6,24,80 --seeds 9001,9002,9003,9004 --out-dir <configs> --manifest <out>.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.scheduler_residence_v1_read import cell_structure  # noqa: E402

# Measured, not guessed. See the module docstring and docs/lineages/cluster_scale_v1.md.
S0_CORPUS_CANDIDATE_MAX = 5.0
SEED_PATH = "/network/topology/seed"
SERVERS_PATH = "/nodes/server_nodes/count"


class RungMintError(RuntimeError):
    """Fail loud: a cell that differs in more than the two declared fields is a different
    experiment wearing the same name."""


def _flatten(doc: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(doc, dict):
        for k, v in doc.items():
            out.update(_flatten(v, f"{prefix}/{k}"))
    elif isinstance(doc, list):
        out[prefix] = json.dumps(doc)
    else:
        out[prefix] = doc
    return out


def mint(base: Dict[str, Any], servers: int, seed: int) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base))
    cfg["nodes"]["server_nodes"]["count"] = int(servers)
    cfg.setdefault("network", {}).setdefault("topology", {})["seed"] = int(seed)
    fb, fc = _flatten(base), _flatten(cfg)
    differing = sorted(k for k in set(fb) | set(fc) if fb.get(k) != fc.get(k))
    allowed = {SEED_PATH, SERVERS_PATH}
    unexpected = [k for k in differing if k not in allowed]
    if unexpected:
        raise RungMintError(
            f"servers={servers} seed={seed}: minted config also differs in {unexpected}; "
            f"only {sorted(allowed)} may change")
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--sim-inputs", type=Path, required=True)
    ap.add_argument("--rungs", required=True, help="server counts, e.g. 6,24,80")
    ap.add_argument("--seeds", required=True, help="topology seeds, e.g. 9001,9002,9003,9004")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--prefix", default="cs")
    args = ap.parse_args(argv)

    base = json.loads(args.base.read_text())
    rungs = [int(x) for x in args.rungs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out: Dict[str, Any] = {"lineage": "cluster_scale_v1", "stage": "S0.d",
                           "base": str(args.base),
                           "corpus_candidate_max": S0_CORPUS_CANDIDATE_MAX, "rungs": []}
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[S0.d] candidates per task = reachable servers (per_server=1). "
          f"Corpus max {S0_CORPUS_CANDIDATE_MAX:.0f}.")
    print(f"  {'servers':>8s} {'cells':>6s} {'mean cand':>10s} {'min':>5s} {'max':>5s} "
          f"{'vs corpus':>10s}  support")
    for servers in rungs:
        cells = []
        for seed in seeds:
            cfg = mint(base, servers, seed)
            name = f"{args.prefix}{servers}s{seed}"
            if args.out_dir:
                (args.out_dir / f"{name}.json").write_text(json.dumps(cfg, indent=1))
            cells.append({"cell": name, "servers": servers, "seed": seed,
                          "structure": cell_structure(cfg, sim_input_path=args.sim_inputs)})
        means = [c["structure"]["mean_reachable_servers"] for c in cells
                 if c["structure"]["mean_reachable_servers"] is not None]
        mins = [c["structure"]["min_reachable_servers"] for c in cells
                if c["structure"]["min_reachable_servers"] is not None]
        mean_c = st.fmean(means) if means else None
        in_support = bool(mean_c is not None and mean_c <= S0_CORPUS_CANDIDATE_MAX)
        row = {"servers": servers, "cells": cells, "mean_candidates": mean_c,
               "min_candidates": min(mins) if mins else None,
               "max_candidates": max(means) if means else None,
               "in_support": in_support,
               "support": "IN-SUPPORT" if in_support else "OUT-OF-SUPPORT"}
        out["rungs"].append(row)
        ratio = (mean_c / S0_CORPUS_CANDIDATE_MAX) if mean_c else float("nan")
        print(f"  {servers:8d} {len(cells):6d} {mean_c:10.2f} {row['min_candidates']:5.0f} "
              f"{row['max_candidates']:5.2f} {ratio:9.2f}x  {row['support']}")

    out["any_out_of_support"] = any(not r["in_support"] for r in out["rungs"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote] {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

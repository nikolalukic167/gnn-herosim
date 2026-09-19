"""partial_state_v3 P0 -- the instrument bar, measured on every dataset of the corpus.

A cache stores the partial-state INGREDIENTS (node ranks, caps, routes, peer table); the
columns are computed under the contract at train and serve time. So P0 asks two things of
the v2 and v3 caches built over the same parents:

  1. the ingredients are identical dataset for dataset (a representation change must not
     move the cache), and
  2. computing the columns under each contract from its own cache, the 10 base and 4 linkrank
     columns are bit-identical and the v2 one-hot rank is recovered from v3's
     (rank_frac, inv_n) on 100 % of edges -- with an empty prefix AND along the first
     tied-optimal plan's teacher-forced prefix, so the prefix-dependent columns are covered.

Bars are the module constants in partial_state_v3_read.py. Fails loud on anything but
INSTRUMENT-PASS, so a training job that requires the P0 artefact cannot start on a cache
that is not a pure representation change.

    python3 scripts_cosim/partial_state_v3_p0.py --v2-cache DIR --v3-cache DIR --out p0.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from scripts_cosim.partial_state_v3_read import V_P0_PASS, p0_dataset, read_p0
from src.policy.tabular.reduced_features import (
    KRANK_FEATURE_DIM, KRANK_TYPES, KRANK_V3_FEATURE_DIM, LINKRANK_FEATURE_DIM,
    PARTIAL_STATE_BASE_DIM, PARTIAL_STATE_CONTRACT_ENV, PARTIAL_STATE_CONTRACT_V2,
    PARTIAL_STATE_CONTRACT_V3, build_partial_state_context_from_graph, partial_state_columns,
)

INGREDIENTS = ("node_rank", "node_caps", "demand", "parents", "route_hops_bneck",
               "payload_bytes", "transfer_norm", "ingress_links", "core_links",
               "task_type_index", "peer_pairs", "node_exchange", "peer_norm", "cand_nodes")


def _load(cache: Path) -> Dict[str, Any]:
    graphs = pickle.load(open(cache / "graphs.pkl", "rb"))
    out = {}
    for g in graphs:
        gid = str(getattr(g, "dataset_id", None) or getattr(g, "parent_dataset_id", None))
        if gid in out:
            raise ValueError(f"{cache}: dataset id {gid} appears twice")
        out[gid] = g
    return out


def _canon(x: Any) -> Any:
    """A comparable, order-free form of an ingredient (dict keys may be tuples)."""
    if isinstance(x, dict):
        return sorted((repr(k), _canon(v)) for k, v in x.items())
    if isinstance(x, (list, tuple, set, frozenset)):
        return sorted(repr(_canon(v)) for v in x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _ctx_under(g: Any, contract: str):
    saved = os.environ.get(PARTIAL_STATE_CONTRACT_ENV)
    os.environ[PARTIAL_STATE_CONTRACT_ENV] = contract
    try:
        return build_partial_state_context_from_graph(g)
    finally:
        if saved is None:
            os.environ.pop(PARTIAL_STATE_CONTRACT_ENV, None)
        else:
            os.environ[PARTIAL_STATE_CONTRACT_ENV] = saved


def _first_plan(g: Any) -> Optional[Sequence[int]]:
    plans = getattr(g, "tied_optimal_logit_plans", None)
    if plans is None:
        return None
    if isinstance(plans, dict):
        key = getattr(g, "dag_primary_alpha_key", None)
        plans = plans.get(key) if key in plans else next(iter(plans.values()), None)
    if not plans:
        return None
    plan = plans[0]
    if len(plan) != int(g.n_tasks):
        return None
    return [int(p) for p in plan]


def compare_dataset(g2: Any, g3: Any) -> dict:
    """One dataset: ingredient equality plus p0_dataset over every (task, prefix)."""
    p2, p3 = g2.partial_state_ctx, g3.partial_state_ctx
    moved = [k for k in INGREDIENTS if _canon(p2.get(k)) != _canon(p3.get(k))]
    if moved:
        raise ValueError(f"cache ingredients differ on {moved}: not a representation change")
    ctx2 = _ctx_under(g2, PARTIAL_STATE_CONTRACT_V2)
    ctx3 = _ctx_under(g3, PARTIAL_STATE_CONTRACT_V3)
    tl2, tl3 = g2.task_logit_to_placement, g3.task_logit_to_placement
    n_tasks = int(g2.n_tasks)
    plan = _first_plan(g2)
    blocks2: List[np.ndarray] = []
    blocks3: List[np.ndarray] = []
    for t in range(n_tasks):
        cands2 = [tuple(c) for c in tl2[t]]
        cands3 = [tuple(c) for c in tl3[t]]
        if cands2 != cands3:
            raise ValueError(f"task {t}: candidate sets differ between the caches")
        prefixes = [{}]
        if plan is not None:
            prefixes.append({j: tuple(tl2[j][plan[j]]) for j in range(t)})
        for committed in prefixes:
            blocks2.append(partial_state_columns(ctx2, t, cands2, committed))
            blocks3.append(partial_state_columns(ctx3, t, cands3, committed))
    v2 = np.vstack(blocks2) if blocks2 else np.zeros((0, 38))
    v3 = np.vstack(blocks3) if blocks3 else np.zeros((0, 22))
    row = p0_dataset(v2, v3, base_dim=PARTIAL_STATE_BASE_DIM, v2_krank_dim=KRANK_FEATURE_DIM,
                     v3_krank_dim=KRANK_V3_FEATURE_DIM, types=KRANK_TYPES,
                     link_dim=LINKRANK_FEATURE_DIM)
    row["n_nodes"] = len(p2["node_rank"])
    row["prefix_covered"] = plan is not None
    return row


def compare_caches(v2_cache: Path, v3_cache: Path) -> dict:
    m2 = json.load(open(v2_cache / "metadata.json"))
    m3 = json.load(open(v3_cache / "metadata.json"))
    g2, g3 = _load(v2_cache), _load(v3_cache)
    if set(g2) != set(g3):
        raise ValueError(f"dataset sets differ: {len(set(g2) ^ set(g3))} ids not shared")
    rows = {}
    for gid in sorted(g2):
        rows[gid] = compare_dataset(g2[gid], g3[gid])
    verdict = read_p0(list(rows.values()))
    return {**verdict, "v2_cache": str(v2_cache), "v3_cache": str(v3_cache),
            "v2_contract": m2.get("partial_state_contract"), "v3_contract": m3.get("partial_state_contract"),
            "n_prefix_covered": sum(1 for r in rows.values() if r["prefix_covered"]),
            "max_nodes": max(r["n_nodes"] for r in rows.values()),
            "per_dataset": rows}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v2-cache", required=True, type=Path)
    ap.add_argument("--v3-cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    res = compare_caches(args.v2_cache, args.v3_cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"[P0] {res['verdict']}  datasets={res['n_datasets']}  "
          f"max_column_diff={res.get('max_column_diff')}  min_rank_recovery={res.get('min_rank_recovery')}  "
          f"prefix_covered={res['n_prefix_covered']}  max_nodes={res['max_nodes']}")
    if res["verdict"] != V_P0_PASS:
        print(f"FAIL LOUD: P0 is {res['verdict']} ({res.get('reason', '')}); see {args.out}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

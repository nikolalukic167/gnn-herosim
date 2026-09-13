#!/usr/bin/env python3
"""Does the live decoder get offered candidates the training corpus never contained?

Audit 2026-09-13. The live candidate set is built by `feature_builder` from whatever
replicas the autoscaler has created by that moment; the cache's candidate set was frozen
when the co-sim dataset captured its state. Nothing has ever compared the two. A local
3,000-event smoke says they differ in kind, not just in size:

    cache  (516 datasets): rpiCpu, xavierCpu, pynqFpga          max candidate demand 0.213
    live   (24 batches)  : rpiCpu, xavierCpu, pynqFpga, xavierGpu   max candidate demand 1.739

`xavierGpu` appears in **zero** training datasets. Two things follow and both bite:

  1. Its platform-type one-hot column is on for a candidate the platform encoder was never
     fitted on, so its embedding is whatever initialisation and weight decay left there.
  2. `node_caps[n] = alpha x max single candidate demand on n` over THIS batch. One GPU
     candidate raises that node's cap ~8x, so the capacity mask -- the decoder's only
     concentration control before `GNN_PREFIX_PLATFORM_CAP` existed -- stops binding on
     that node for the whole batch.

This probe measures both on the real production decode traces (`serving_gap_v1/traces_h1`,
every arm and seed of all three corpora), and reports the one statistic that can make it an
ARM asymmetry rather than a shared handicap: the share of decoded placements that land on a
platform type absent from the checkpoint's own training corpus.

Support is derived from the cache, never hardcoded, so a corpus that does contain GPU
replicas reports zero rather than a false positive.
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src" / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARMS = ("gnn", "mpoff")


def _ptype_by_pid(graph: Any) -> Dict[int, str]:
    return {int(m["platform_id"]): str(m["platform_type"])
            for m in graph.queue_key_to_platform_meta.values()}


def cache_support(cache_dir: Path) -> Tuple[Set[str], float, float]:
    """(platform types that appear as a CANDIDATE, max candidate demand, max node cap)."""
    with open(cache_dir / "graphs.pkl", "rb") as fh:
        graphs = pickle.load(fh)
    types: Set[str] = set()
    max_demand = 0.0
    max_cap = 0.0
    for g in graphs:
        pt = _ptype_by_pid(g)
        psc = g.partial_state_ctx
        for (_t, pl), v in psc["demand"].items():
            types.add(pt.get(int(pl[1]), "?"))
            max_demand = max(max_demand, float(v))
        for cap in psc["node_caps"].values():
            max_cap = max(max_cap, float(cap))
    return types, max_demand, max_cap


def _iter_records(path: Path):
    with open(path, "rb") as fh:
        while True:
            try:
                yield pickle.load(fh)
            except EOFError:
                return


def scan_trace(path: Path, support: Set[str], cache_max_cap: float) -> Dict[str, Any]:
    n_batches = n_graphless = 0
    cand_total = cand_oos = 0
    plc_total = plc_oos = 0
    batches_with_oos = 0
    caps_over = caps_total = 0
    max_cap = 0.0
    for rec in _iter_records(path):
        graph = rec.get("graph")
        if graph is None:
            n_graphless += 1
            continue
        n_batches += 1
        pt = _ptype_by_pid(graph)
        seen_oos = False
        for _t, cands in graph.task_logit_to_placement.items():
            for c in cands:
                cand_total += 1
                if pt.get(int(c[1]), "?") not in support:
                    cand_oos += 1
                    seen_oos = True
        if seen_oos:
            batches_with_oos += 1
        for cap in (graph.partial_state_ctx.get("node_caps") or {}).values():
            caps_total += 1
            max_cap = max(max_cap, float(cap))
            if float(cap) > cache_max_cap:
                caps_over += 1
        combo = rec.get("combo")
        if combo:
            for pl in combo:
                plc_total += 1
                if pt.get(int(pl[1]), "?") not in support:
                    plc_oos += 1
    return {
        "n_batches": n_batches, "n_graphless_records": n_graphless,
        "frac_batches_with_oos_candidate": (batches_with_oos / n_batches) if n_batches else None,
        "frac_candidates_oos": (cand_oos / cand_total) if cand_total else None,
        "frac_placements_oos": (plc_oos / plc_total) if plc_total else None,
        "frac_node_caps_above_cache_max": (caps_over / caps_total) if caps_total else None,
        "max_node_cap": max_cap, "n_placements": plc_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--traces", type=Path, required=True, help="dir holding <arm>_s<seed>.pkl")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    support, max_demand, max_cap = cache_support(args.cache_dir)
    print(f"[support] {args.corpus}: candidate platform types {sorted(support)} "
          f"max_demand={max_demand:.4f} max_node_cap={max_cap:.4f}", flush=True)

    per: Dict[str, Any] = {}
    for arm in ARMS:
        for seed in args.seeds:
            f = args.traces / f"{arm}_s{seed}.pkl"
            if not f.is_file():
                continue
            row = scan_trace(f, support, max_cap)
            row["arm"], row["seed"] = arm, seed
            per[f"{arm}_s{seed}"] = row
            print(f"[scan] {arm}_s{seed}: batches={row['n_batches']} "
                  f"oos_cand={row['frac_candidates_oos']:.4f} "
                  f"oos_placed={row['frac_placements_oos']:.4f} "
                  f"caps_over={row['frac_node_caps_above_cache_max']:.4f}", flush=True)

    summary: Dict[str, Any] = {}
    for key in ("frac_candidates_oos", "frac_placements_oos", "frac_node_caps_above_cache_max"):
        vals = {a: {r["seed"]: r[key] for r in per.values() if r["arm"] == a and r[key] is not None}
                for a in ARMS}
        entry: Dict[str, Any] = {a: (st.median(list(vals[a].values())) if vals[a] else None) for a in ARMS}
        seeds = sorted(set(vals["gnn"]) & set(vals["mpoff"]))
        diffs = [vals["gnn"][s] - vals["mpoff"][s] for s in seeds]
        entry["n_pairs"] = len(diffs)
        if diffs:
            entry["gnn_minus_mpoff_median"] = st.median(diffs)
            entry["gnn_higher_seeds"] = sum(1 for d in diffs if d > 0)
            if len(diffs) > 1 and any(d != 0 for d in diffs):
                from scipy.stats import wilcoxon
                entry["p_exact_wilcoxon"] = float(
                    wilcoxon(diffs, alternative="two-sided", mode="exact").pvalue)
        summary[key] = entry

    out = {"probe": "candidate_support", "corpus": args.corpus,
           "cache_dir": str(args.cache_dir), "traces": str(args.traces),
           "training_support": sorted(support),
           "cache_max_candidate_demand": max_demand, "cache_max_node_cap": max_cap,
           "per_trace": per, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

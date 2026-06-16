#!/usr/bin/env python3
"""Quantify how much *irreducible joint coupling* exists in the co-sim oracle labels.

Central question for "is a GNN necessary":
  A pointwise edge scorer f(task, platform) -> logit picks argmax per task,
  INDEPENDENTLY. A GNN can (in principle) make the score of one edge depend on
  the rest of the batch. The GNN can only beat a pointwise model when the OPTIMAL
  placement is NOT recoverable by independent per-task decisions.

We measure three label-grounded quantities per dataset, from placements/placements.jsonl
(the full (placement_plan, rtt) brute-force sweep):

  (M1) marginal-greedy regret:
       pi(t) = argmin_p [ min RTT over all combos with task t on platform p ]
       regret = RTT(joint combo of pi) - RTT(optimal).
       If ~0 everywhere, each task's best platform is independent of the others
       => separable => pointwise MLP suffices.

  (M2) identical-task symmetry:
       tasks with identical (type, source_node) get IDENTICAL features, hence a
       pointwise model (and a frozen-decode GNN) MUST assign them the same platform.
       - frac of datasets with >=2 identical tasks
       - among those, does the optimum SPREAD them?
       - regret of forcing identical tasks to co-locate (pointwise lower bound):
         min RTT over combos where every identical group shares one platform,
         minus optimal RTT.

  (M3) collision in optimum:
       fraction of optimal combos that place 2+ tasks on the SAME platform
       (i.e., the optimum itself "double books") vs. spreads.

Run:
  pipenv run python3 scripts_cosim/separability_diagnostic.py <corpus_dir> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_workload_task_sigs(ds_dir: Path) -> Optional[List[Tuple[str, str]]]:
    """Return per-task (type_name, source_node) in event order = task index order."""
    wl = ds_dir / "workload.json"
    if not wl.exists():
        return None
    try:
        data = json.loads(wl.read_text())
    except Exception:
        return None
    sigs: List[Tuple[str, str]] = []
    for ev in data.get("events", []):
        app = ev.get("application", {})
        # task type is the dag key, e.g. {"dnn2": []}
        dag = app.get("dag", {})
        ttype = next(iter(dag.keys()), app.get("name", "?"))
        src = str(ev.get("node_name", "?"))
        sigs.append((str(ttype), src))
    return sigs


def load_combos(ds_dir: Path) -> Optional[List[Tuple[Dict[int, Tuple[int, int]], float]]]:
    jp = ds_dir / "placements" / "placements.jsonl"
    if not jp.exists():
        return None
    combos: List[Tuple[Dict[int, Tuple[int, int]], float]] = []
    with jp.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            plan = rec.get("placement_plan")
            rtt = rec.get("rtt")
            if plan is None or rtt is None:
                continue
            pp: Dict[int, Tuple[int, int]] = {}
            ok = True
            for k, v in plan.items():
                try:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        pp[int(k)] = (int(v[0]), int(v[1]))
                    else:
                        ok = False
                        break
                except Exception:
                    ok = False
                    break
            if ok and pp:
                combos.append((pp, float(rtt)))
    return combos or None


def analyze_dataset(ds_dir: Path) -> Optional[dict]:
    combos = load_combos(ds_dir)
    if not combos:
        return None
    sigs = load_workload_task_sigs(ds_dir)

    n_tasks = max(len(pp) for pp, _ in combos)
    # task indices present
    task_ids = sorted({t for pp, _ in combos for t in pp.keys()})
    if len(task_ids) < 1:
        return None

    # optimal
    opt_plan, opt_rtt = min(combos, key=lambda x: x[1])
    if opt_rtt <= 0:
        return None

    # --- M1: marginal greedy ---
    # marginal_min[t][p] = min rtt over combos with task t -> p
    marginal_min: Dict[int, Dict[Tuple[int, int], float]] = defaultdict(lambda: defaultdict(lambda: float("inf")))
    combo_lookup: Dict[Tuple[Tuple[int, int], ...], float] = {}
    for pp, rtt in combos:
        key = tuple(pp[t] for t in task_ids)
        # keep min rtt if duplicate combo keys
        if key not in combo_lookup or rtt < combo_lookup[key]:
            combo_lookup[key] = rtt
        for t in task_ids:
            p = pp[t]
            if rtt < marginal_min[t][p]:
                marginal_min[t][p] = rtt

    greedy_choice: Dict[int, Tuple[int, int]] = {}
    for t in task_ids:
        greedy_choice[t] = min(marginal_min[t].items(), key=lambda kv: kv[1])[0]
    greedy_key = tuple(greedy_choice[t] for t in task_ids)
    greedy_rtt = combo_lookup.get(greedy_key)  # may be None if combo not enumerated
    m1_regret = None
    m1_regret_rel = None
    greedy_in_sweep = greedy_rtt is not None
    if greedy_rtt is not None:
        m1_regret = greedy_rtt - opt_rtt
        m1_regret_rel = m1_regret / opt_rtt

    # --- M2: identical-task symmetry (needs sigs) ---
    m2 = {
        "has_identical": False,
        "n_identical_groups": 0,
        "max_group_size": 0,
        "opt_spreads_identical": None,
        "colocate_regret_rel": None,
    }
    if sigs is not None and len(sigs) >= len(task_ids):
        groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for t in task_ids:
            if t < len(sigs):
                groups[sigs[t]].append(t)
        ident_groups = [g for g in groups.values() if len(g) >= 2]
        m2["n_identical_groups"] = len(ident_groups)
        m2["has_identical"] = len(ident_groups) > 0
        m2["max_group_size"] = max((len(g) for g in ident_groups), default=0)
        if ident_groups:
            # does optimum place each identical group on a single platform?
            opt_spreads = False
            for g in ident_groups:
                plats = {opt_plan[t] for t in g}
                if len(plats) > 1:
                    opt_spreads = True
                    break
            m2["opt_spreads_identical"] = opt_spreads
            # pointwise lower bound: best combo where each identical group co-locates
            best_colo = float("inf")
            for key, rtt in combo_lookup.items():
                ok = True
                for g in ident_groups:
                    plats = {key[task_ids.index(t)] for t in g}
                    if len(plats) > 1:
                        ok = False
                        break
                if ok and rtt < best_colo:
                    best_colo = rtt
            if best_colo < float("inf"):
                m2["colocate_regret_rel"] = (best_colo - opt_rtt) / opt_rtt

    # --- M3: collision in optimum ---
    opt_plats = [opt_plan[t] for t in task_ids]
    opt_has_collision = len(set(opt_plats)) < len(opt_plats)
    opt_unique_plats = len(set(opt_plats))

    return {
        "n_tasks": len(task_ids),
        "n_combos": len(combos),
        "opt_rtt": opt_rtt,
        "greedy_in_sweep": greedy_in_sweep,
        "m1_regret_rel": m1_regret_rel,
        "m1_greedy_eq_opt": (greedy_key == tuple(opt_plan[t] for t in task_ids)),
        "m2": m2,
        "opt_has_collision": opt_has_collision,
        "opt_unique_plats": opt_unique_plats,
    }


def pctl(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=str)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.corpus_dir)
    ds_dirs = sorted(d for d in base.glob("ds_*") if d.is_dir())
    if args.limit:
        ds_dirs = ds_dirs[: args.limit]
    if not ds_dirs:
        print(f"No ds_* dirs under {base}", file=sys.stderr)
        return 1

    results = []
    for d in ds_dirs:
        r = analyze_dataset(d)
        if r is not None:
            results.append(r)

    n = len(results)
    if n == 0:
        print("No analyzable datasets.")
        return 1

    # aggregate
    m1_rel = [r["m1_regret_rel"] for r in results if r["m1_regret_rel"] is not None]
    greedy_eq = sum(1 for r in results if r["m1_greedy_eq_opt"])
    greedy_in = sum(1 for r in results if r["greedy_in_sweep"])
    has_ident = sum(1 for r in results if r["m2"]["has_identical"])
    spreads = sum(1 for r in results if r["m2"]["opt_spreads_identical"] is True)
    colo_rel = [r["m2"]["colocate_regret_rel"] for r in results if r["m2"]["colocate_regret_rel"] is not None]
    opt_coll = sum(1 for r in results if r["opt_has_collision"])
    multitask = [r for r in results if r["n_tasks"] >= 2]

    print(f"\n===== Separability diagnostic: {base.name} =====")
    print(f"Datasets analyzed: {n} (multi-task >=2: {len(multitask)})")
    print(f"Mean n_combos: {sum(r['n_combos'] for r in results)/n:.0f}")

    print("\n--- M1: marginal-greedy (independent per-task best) vs joint optimum ---")
    print(f"  greedy combo present in sweep: {greedy_in}/{n} ({100*greedy_in/n:.1f}%)")
    print(f"  greedy == optimum (exact):     {greedy_eq}/{n} ({100*greedy_eq/n:.1f}%)")
    if m1_rel:
        print(f"  regret_rel (greedy vs opt): mean={sum(m1_rel)/len(m1_rel)*100:.2f}%  "
              f"median={pctl(m1_rel,0.5)*100:.2f}%  p90={pctl(m1_rel,0.9)*100:.2f}%  max={max(m1_rel)*100:.1f}%")
        frac_big = sum(1 for v in m1_rel if v > 0.05) / len(m1_rel)
        print(f"  frac datasets with greedy regret > 5%: {frac_big*100:.1f}%")

    print("\n--- M2: identical (type,src) tasks => pointwise MUST co-assign ---")
    print(f"  datasets with >=2 identical tasks: {has_ident}/{n} ({100*has_ident/n:.1f}%)")
    if has_ident:
        print(f"  among identical: optimum SPREADS them: {spreads}/{has_ident} ({100*spreads/has_ident:.1f}%)")
    if colo_rel:
        print(f"  forced-colocation regret_rel (pointwise floor): mean={sum(colo_rel)/len(colo_rel)*100:.2f}%  "
              f"median={pctl(colo_rel,0.5)*100:.2f}%  p90={pctl(colo_rel,0.9)*100:.2f}%  max={max(colo_rel)*100:.1f}%")

    print("\n--- M3: does the OPTIMUM itself collide (2+ tasks same platform)? ---")
    print(f"  optimal combo has collision: {opt_coll}/{n} ({100*opt_coll/n:.1f}%)")
    avg_uniq = sum(r["opt_unique_plats"] for r in multitask) / max(len(multitask), 1)
    avg_nt = sum(r["n_tasks"] for r in multitask) / max(len(multitask), 1)
    print(f"  avg unique platforms in optimum: {avg_uniq:.2f} / avg n_tasks {avg_nt:.2f} (multi-task)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

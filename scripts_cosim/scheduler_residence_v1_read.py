#!/usr/bin/env python3
"""scheduler_residence_v1 -- R0 (the decomposition) and R1 (what separates the good cell).

Registered in docs/lineages/scheduler_residence_v1.md on 2026-09-15. Every bar below is a
module constant and was committed before the arms it reads had been submitted.

`averageWaitTime` is `scheduled_time - dispatched_time` -- the WHOLE scheduler-side residence,
not a batch timer -- and queue_range_v1 wrote its 6.87 s up as "batch wait". It contains head-
of-line blocking behind a batch the task is not part of, the collection window itself, re-queue
after deferral, and store/mutex serialization. R0 splits it. The split decides whether the fix
is scheduler concurrency (untested) or the batch window (swept and closed by
drainable_serving_config_v1, where zero window reads -1731 %).

R0 blocking -- and the reconstruction sub-bar is blocking too: the three components must add
     back up to `averageWaitTime`, or the decomposition is a story.
R1 nearly free -- s9001 has no queue problem under identical policy, trace and load, so the
     difference is in the cell. Registered expectation NEGATIVE.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- BARS
R0_RECONSTRUCTION_TOL = 0.01     # |sum of parts - averageWaitTime| / averageWaitTime
R0_HOL_MIN_S = 2.0
R0_MIN_CELLS = 2
R0_MIN_SEEDS = 12
R0_REGISTERED_EXPECTATION = "UNCERTAIN"

R1_MIN_SEPARATING = 1
R1_REGISTERED_EXPECTATION = "NEGATIVE"
R1_GOOD_CELL = "cell_s9001_f4000_pg16"

CELLS = ("cell_s7901_f4000_pg16", "cell_s9001_f4000_pg16", "cell_s9002_f4000_pg16")
POLICIES = ("gnn", "mpoff")

KNOWN_UNSERVABLE: Tuple[Tuple[str, int], ...] = (("gnn", 3),)
BURNED: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "cell_s7901_f4000_pg16": (("gnn", 8), ("gnn", 14)),
}


class ResidenceReadError(RuntimeError):
    """Fail loud: a quietly dropped arm is a quietly different experiment."""


# ------------------------------------------------------- per-arm summarisation (gate side)
def residence_summary(
    tasks: Sequence[Sequence[float]],
    batches: Sequence[Sequence[float]],
    bounds: Sequence[float],
    average_wait_time: float,
    unstamped: int,
) -> Dict[str, Any]:
    """Collapse ~50,000 per-task rows into the numbers R0 reads, in the gate's own step.

    `tasks` rows are [dispatched, head_of_line, collection, placement]; `bounds` are
    `decile_queue_summary`'s edges, so a task lands in the same decile in both summaries.
    """
    if not tasks:
        raise ResidenceReadError(
            "no per-task residence rows -- no task was scheduled through the instrumented "
            "path, so this arm cannot be read against R0"
        )
    n = len(bounds) + 1
    buckets: List[List[Sequence[float]]] = [[] for _ in range(n)]
    for row in tasks:
        d = float(row[0])
        idx = n - 1
        for i, b in enumerate(bounds):
            if d < b:
                idx = i
                break
        buckets[idx].append(row)

    def part(rows: Sequence[Sequence[float]], i: int) -> float:
        return st.fmean(float(r[i]) for r in rows)

    deciles = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            deciles.append({"decile": i + 1, "n": 0})
            continue
        deciles.append({
            "decile": i + 1, "n": len(bucket),
            "head_of_line_s": part(bucket, 1),
            "collection_s": part(bucket, 2),
            "placement_s": part(bucket, 3),
        })
    hol, coll, place = part(tasks, 1), part(tasks, 2), part(tasks, 3)
    total = hol + coll + place
    aw = float(average_wait_time)
    return {
        "deciles": deciles,
        "n_tasks": len(tasks),
        "n_batches": len(batches),
        "head_of_line_s": hol,
        "collection_s": coll,
        "placement_s": place,
        "sum_of_parts_s": total,
        "average_wait_time_s": aw,
        # R0's blocking sub-bar. Reported, never silently corrected.
        "reconstruction_error": (abs(total - aw) / aw) if aw > 0 else None,
        "unstamped": int(unstamped),
        "mean_batch_size": (st.fmean(float(b[2]) for b in batches) if batches else None),
    }


# --------------------------------------------------------------------------- loading
def seeds_for(arms: Dict[str, Dict[str, Any]], cell: str, policy: str) -> List[int]:
    burned = set(BURNED.get(cell, ()))
    keep: List[int] = []
    for seed in range(1, 17):
        if (policy, seed) in burned or (policy, seed) in KNOWN_UNSERVABLE:
            continue
        if f"{cell}__residence_{policy}_s{seed}" in arms:
            keep.append(seed)
    return keep


def load(results: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results.glob("*.summary.json")):
        doc = json.loads(path.read_text())
        name = doc.get("arm")
        if not name:
            raise ResidenceReadError(f"{path}: no arm name")
        out[name] = doc
    if not out:
        raise ResidenceReadError(f"{results}: no summaries")
    return out


def _res(doc: Dict[str, Any]) -> Dict[str, Any]:
    r = doc.get("residence_summary")
    if not r:
        raise ResidenceReadError(
            f"{doc.get('arm')}: no residence_summary -- produced by a tree without the "
            "instrument and cannot be read against R0"
        )
    return r


# --------------------------------------------------------------------------- R0
def read_r0(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    firing = 0
    for cell in CELLS:
        seeds = seeds_for(arms, cell, "gnn")
        if len(seeds) < R0_MIN_SEEDS:
            per_cell[cell] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(seeds)}
            continue
        rows = [_res(arms[f"{cell}__residence_gnn_s{s}"]) for s in seeds]
        errs = [r["reconstruction_error"] for r in rows if r["reconstruction_error"] is not None]
        unstamped = max(r["unstamped"] for r in rows)
        reconstructs = bool(errs) and max(errs) <= R0_RECONSTRUCTION_TOL and unstamped == 0
        if not reconstructs:
            per_cell[cell] = {
                "verdict": "VOID-DOES-NOT-RECONSTRUCT", "n": len(seeds),
                "max_reconstruction_error": max(errs) if errs else None,
                "max_unstamped": unstamped,
            }
            continue
        hol = st.median(r["head_of_line_s"] for r in rows)
        coll = st.median(r["collection_s"] for r in rows)
        place = st.median(r["placement_s"] for r in rows)
        fires = hol >= R0_HOL_MIN_S and hol > coll
        per_cell[cell] = {
            "n": len(seeds), "head_of_line_s": hol, "collection_s": coll,
            "placement_s": place,
            "average_wait_time_s": st.median(r["average_wait_time_s"] for r in rows),
            "max_reconstruction_error": max(errs), "fires": fires,
        }
        firing += int(fires)
    return {
        "per_cell": per_cell, "cells_firing": firing, "fires": firing >= R0_MIN_CELLS,
        "verdict": ("HEAD-OF-LINE-DOMINATES" if firing >= R0_MIN_CELLS
                    else "COLLECTION-DOMINATES"),
        "registered_expectation": R0_REGISTERED_EXPECTATION,
        "bars": {"hol_min_s": R0_HOL_MIN_S, "min_cells": R0_MIN_CELLS,
                 "min_seeds": R0_MIN_SEEDS, "reconstruction_tol": R0_RECONSTRUCTION_TOL},
    }


# --------------------------------------------------------------------------- R1
R1_STATS = ("server_nodes", "client_nodes", "mean_reachable_servers",
            "min_reachable_servers", "replicas_per_task_type", "hosting_node_spread")


def cell_structure(config: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Structural statistics of one live cell, from its own config.

    Reads only what the config declares; a statistic the config cannot answer is None and is
    skipped by the bar rather than imputed.
    """
    infra = config.get("infrastructure") or config
    nodes = infra.get("nodes") or []
    out: Dict[str, Optional[float]] = {k: None for k in R1_STATS}
    if not nodes:
        return out
    clients, servers = [], []
    for node in nodes:
        (clients if node.get("is_client") or node.get("client") else servers).append(node)
    out["server_nodes"] = float(len(servers))
    out["client_nodes"] = float(len(clients))
    reach: List[float] = []
    server_names = {str(n.get("node_name") or n.get("name")) for n in servers}
    for node in clients:
        nm = node.get("network_map") or {}
        seen = {str(k) for k, v in nm.items() if v} if isinstance(nm, dict) else set()
        reach.append(float(len(seen & server_names)))
    if reach:
        out["mean_reachable_servers"] = st.fmean(reach)
        out["min_reachable_servers"] = min(reach)
    plats = [len(n.get("platforms") or []) for n in servers]
    if plats:
        out["replicas_per_task_type"] = st.fmean(float(p) for p in plats)
        out["hosting_node_spread"] = float(sum(1 for p in plats if p))
    return out


def read_r1(structures: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Any]:
    others = [c for c in CELLS if c != R1_GOOD_CELL]
    rows: Dict[str, Any] = {}
    separating: List[str] = []
    for stat in R1_STATS:
        good = structures.get(R1_GOOD_CELL, {}).get(stat)
        rest = [structures.get(c, {}).get(stat) for c in others]
        if good is None or any(v is None for v in rest):
            rows[stat] = {"verdict": "NOT-MEASURABLE"}
            continue
        lo, hi = min(rest), max(rest)          # type: ignore[type-var]
        sep = good < lo or good > hi
        rows[stat] = {"good": good, "others": rest, "separates": bool(sep)}
        if sep:
            separating.append(stat)
    fires = len(separating) >= R1_MIN_SEPARATING
    return {"per_stat": rows, "separating": separating, "fires": fires,
            "verdict": ("STRUCTURE-SEPARATES" if fires else "STRUCTURE-DOES-NOT-SEPARATE"),
            "registered_expectation": R1_REGISTERED_EXPECTATION,
            "good_cell": R1_GOOD_CELL, "bars": {"min_separating": R1_MIN_SEPARATING}}


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="R0: the residence arms' summaries")
    ap.add_argument("--configs", type=Path, help="R1: directory holding the cells' configs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not args.results and not args.configs:
        raise SystemExit("nothing to read: pass --results (R0) and/or --configs (R1)")

    result: Dict[str, Any] = {"lineage": "scheduler_residence_v1"}

    if args.results:
        r0 = read_r0(load(args.results))
        result["R0"] = r0
        print(f"\n[R0] where the scheduler-side time goes  (bar: head-of-line >= "
              f"{R0_HOL_MIN_S} s and > collection, on >= {R0_MIN_CELLS} of {len(CELLS)} cells; "
              f"registered expectation {R0_REGISTERED_EXPECTATION})")
        for cell, row in r0["per_cell"].items():
            if "verdict" in row:
                print(f"     {cell}  {row['verdict']} (n={row['n']})")
                continue
            print(f"     {cell}  n={row['n']:2d}  waitTime {row['average_wait_time_s']:6.3f} s"
                  f"  =  head-of-line {row['head_of_line_s']:6.3f}"
                  f"  + collection {row['collection_s']:6.3f}"
                  f"  + placement {row['placement_s']:6.3f}"
                  f"   (err {row['max_reconstruction_error']:.5f})"
                  f"  -> {'FIRES' if row['fires'] else 'no'}")
        print(f"     {r0['verdict']}  ({r0['cells_firing']}/{len(CELLS)} cells)")

    if args.configs:
        structures = {}
        for cell in CELLS:
            path = args.configs / f"{cell}.json"
            if not path.exists():
                raise ResidenceReadError(f"{path} missing -- R1 cannot compare the cells")
            structures[cell] = cell_structure(json.loads(path.read_text()))
        r1 = read_r1(structures)
        result["R1"] = r1
        result["R1"]["structures"] = structures
        print(f"\n[R1] does anything structural separate {R1_GOOD_CELL}?  "
              f"(registered expectation {R1_REGISTERED_EXPECTATION})")
        for stat, row in r1["per_stat"].items():
            if "verdict" in row:
                print(f"     {stat:24s} {row['verdict']}")
                continue
            print(f"     {stat:24s} good {row['good']:10.3f}   others "
                  f"{', '.join(f'{v:.3f}' for v in row['others'])}"
                  f"  -> {'SEPARATES' if row['separates'] else 'no'}")
        print(f"     {r1['verdict']}  ({len(r1['separating'])} separating)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\n[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

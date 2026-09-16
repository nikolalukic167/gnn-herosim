"""partial_state_v3 -- turn the P2 / P3 gate summaries into the registered reads.

    python3 scripts_cosim/partial_state_v3_gate_read.py p2 --dir .../results/psv3_p2
    python3 scripts_cosim/partial_state_v3_gate_read.py p3 --dir .../results/psv3_p3

The bars and the verdict logic live in partial_state_v3_read.py; this file only maps the
summary files the gate arrays write onto those functions, and prints what each rung/cell
contributed so a reader can see every number behind a verdict.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

from scripts_cosim.partial_state_v3_read import (
    P2_CELLS, P3_CHECKPOINT_SEEDS, P3_RUNGS, median, read_p2, read_p3, read_p4,
)


def load_summaries(directory: str) -> List[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.summary.json"))):
        doc = json.load(open(path))
        if "arm" not in doc:
            raise ValueError(f"FAIL LOUD: {path} has no arm name")
        out.append(doc)
    return out


# --- P2 ----------------------------------------------------------------------------------

def p2_inputs(summaries: Sequence[dict]) -> dict:
    """cell -> {seed -> mean elapsed} per (representation, arm_kind); plus the v2 gnn medians."""
    table: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {}
    for s in summaries:
        cell, rep, kind, seed = s["cell"], s["representation"], s["arm_kind"], int(s["training_seed"])
        table.setdefault(kind, {}).setdefault(rep, {}).setdefault(cell, {})[seed] = float(s["averageElapsedTime"])
    v2_gnn_medians = {cell: median(list(seeds.values()))
                      for cell, seeds in table.get("gnn", {}).get("v2", {}).items()}
    return {"table": table, "v2_gnn_medians": v2_gnn_medians}


def read_p2_dir(directory: str) -> dict:
    inp = p2_inputs(load_summaries(directory))
    out = {"n_arms": sum(len(v) for k in inp["table"].values() for r in k.values() for v in r.values())}
    for kind in ("gnn", "mpoff"):
        v3 = inp["table"].get(kind, {}).get("v3", {})
        v2 = inp["table"].get(kind, {}).get("v2", {})
        out[kind] = read_p2(v3, v2, inp["v2_gnn_medians"])
    out["v2_gnn_medians"] = inp["v2_gnn_medians"]
    return out


def format_p2(res: dict) -> str:
    lines = ["P2 -- live tie at 6 servers, v3 vs v2 paired by training seed",
             f"  arms read: {res['n_arms']}",
             "  P2-a control (v2 gnn median per cell vs queue_range_v1 plain):"]
    for cell, (ok, got, want) in res["gnn"].get("control", {}).items():
        lines.append(f"    {cell:28s} got={got if got is None else f'{got:.3f}'} want={want:.3f} -> {'ok' if ok else 'MOVED'}")
    for kind in ("gnn", "mpoff"):
        r = res[kind]
        lines.append(f"  {kind}: {r['verdict']}")
        for cell, c in r.get("cells", {}).items():
            if c.get("verdict") == "UNREADABLE":
                lines.append(f"    {cell:28s} {c['verdict']} ({c.get('reason')})")
            else:
                lines.append(f"    {cell:28s} n={c['n']:2d} median={c['median']:+6.2f}%  p={c['p']:.4f}  v3 ahead {c['v3_ahead']}/{c['n']} -> {c['verdict']}")
    return "\n".join(lines)


# --- P3 ----------------------------------------------------------------------------------

def p3_inputs(summaries: Sequence[dict]) -> Dict[str, dict]:
    """rung -> {"cells": {cell: {...}}, "wallclock_min": [...]} in read_p3's shape.

    A cell without its reactive arm is attrition (the documented starved-client spin hangs
    every policy) and is omitted. A cell WITH reactive but missing learned arms is kept with
    completed < expected, which is P3-a's STILL-PINNED signal."""
    by_rung: Dict[str, Dict[str, dict]] = {}
    walls: Dict[str, List[float]] = {}
    for s in summaries:
        rung, cell, kind = s["rung"], s["cell"], s["arm_kind"]
        c = by_rung.setdefault(rung, {}).setdefault(cell, {"gnn": {}, "mpoff": {}, "reactive": None})
        if kind == "reactive":
            c["reactive"] = float(s["averageElapsedTime"])
        else:
            c[kind][int(s["checkpoint_seed"])] = float(s["averageElapsedTime"])
        walls.setdefault(rung, []).append(float(s["wallclock_s"]) / 60.0)
    out = {}
    n_ck = len(P3_CHECKPOINT_SEEDS)
    for rung, cells in by_rung.items():
        kept = {}
        for cell, c in cells.items():
            if c["reactive"] is None:
                continue
            kept[cell] = {**c, "completed": {"gnn": len(c["gnn"]), "mpoff": len(c["mpoff"])},
                          "expected": {"gnn": n_ck, "mpoff": n_ck}}
        out[rung] = {"cells": kept, "wallclock_min": walls.get(rung, [])}
    return out


def read_p3_dir(directory: str) -> dict:
    p3 = read_p3(p3_inputs(load_summaries(directory)))
    return {"p3": p3, "p4": read_p4(p3)}


def format_p3(res: dict) -> str:
    p3, p4 = res["p3"], res["p4"]
    lines = ["P3 -- does a 6-server checkpoint survive 12 / 24 / 80 servers, live",
             f"  registered expectation: {p3['expected']}",
             f"  {'rung':>4} {'srv':>4} {'arr/s':>6} {'cells':>5} {'d_gnn':>8} {'d_mpoff':>8} {'wall/min':>8}  incomplete"]
    for tag, servers, _, rate in P3_RUNGS:
        r = p3["rungs"].get(tag, {})
        if not r.get("readable") and r.get("n_cells", 0) == 0:
            lines.append(f"  {tag:>4} {servers:>4} {rate:>6.3f} {'0':>5}  ({r.get('reason', 'no cells')})")
            continue
        flag = "" if r.get("readable") else "  UNREADABLE"
        inc = ", ".join(f"{c}:{a}" for c, a in r.get("incomplete_arms", [])) or "-"
        lines.append(f"  {tag:>4} {servers:>4} {rate:>6.3f} {r['n_cells']:>5} {r['d_gnn']:>+8.3f} {r['d_mpoff']:>+8.3f} "
                     f"{r['wallclock_min']:>8.1f}  {inc}{flag}")
    lines.append("")
    lines.append(f"  P3 VERDICT: {p3['verdict']}" + (f"  (first break at {p3['first_break']})" if p3.get("first_break") else "")
                 + (f"  reason: {p3['reason']}" if p3.get("reason") else "")
                 + (f"  pinned: {p3['pinned']}" if p3.get("pinned") else ""))
    if p3.get("r3_affordable") is False:
        lines.append("  R3: RUNG-UNAFFORDABLE (median wall-clock above the bar) -- recorded, not failed")
    lines.append(f"  P4: {p4['verdict']}" + (f"  d={p4['d']}" if "d" in p4 else f"  ({p4.get('reason')})"))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["p2", "p3"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = read_p2_dir(args.dir) if args.stage == "p2" else read_p3_dir(args.dir)
    print(format_p2(res) if args.stage == "p2" else format_p3(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

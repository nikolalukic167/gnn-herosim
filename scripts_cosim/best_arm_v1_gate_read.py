"""best_arm_v1 — map the gate summaries onto the registered F1/F2/F3 reads.

    python3 scripts_cosim/best_arm_v1_gate_read.py \\
        --po-dir simulation_data/peer_affinity_live_gate/results/po_v1 \\
        --p3-dir simulation_data/peer_affinity_live_gate/results/psv3_p3 \\
        --client-dir simulation_data/peer_affinity_live_gate/results/po_v1_clients

The 20-client rung **is** the R0 server cell (6 servers, 20 clients), so it is read from the
SERVER gate's results and the other two from the client gate's. That is the same cells, not a
parallel measurement — and it is the one thing about this read that is easy to get wrong, so it
is asserted rather than assumed: the R0 cells must be the `cs6s*` family.

Bars live in `scripts_cosim/best_arm_v1_read.py`, committed before either missing arm-set was
served. Nothing here decides anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.best_arm_v1_read import (  # noqa: E402
    F_CLIENTS, F_MIN_SEEDS, read_f1, read_f2, read_f3, saturated,
)
from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE, collapse_to_seed, read_pair_pct,
)
from scripts_cosim.peer_only_v1_gate_read import load as load_server, tables as server_tables  # noqa: E402
from scripts_cosim.peer_only_v1_client_read import (  # noqa: E402
    load as load_clients, tables as client_tables,
)

GRAPH = "be1670_gnnedge0"      # the repaired arm: mean aggregation, attributes zeroed
POINTWISE = "516_mpoff"        # the best pointwise arm (v3ext, relabelled by both readers)
REACTIVE = "reactive"


def _cells(table: Mapping[str, dict], label: str) -> list:
    return sorted({c for (c, _s) in table.get(label, {})})


def _collapse(table, label, cells) -> Optional[Dict[int, float]]:
    arm = table.get(label)
    if not arm:
        return None
    try:
        return collapse_to_seed(arm, rung_cells=cells)
    except ValueError:
        return None


def _reactive_like(table, cells, seeds) -> Optional[Dict[int, float]]:
    """Reactive is deterministic and keyed at seed 0; intersecting seed sets with it empties the
    baseline out of a table that plainly contains one (peer_only_v1 C4). Collapse, then replicate."""
    arm = table.get(REACTIVE)
    if not arm:
        return None
    try:
        one = collapse_to_seed(arm, rung_cells=cells)[0]
    except (ValueError, KeyError):
        return None
    return {int(s): one for s in seeds}


def _queue_share(metrics: Mapping[str, dict], cells: Sequence[str]) -> Optional[float]:
    q, e = metrics.get("queue", {}).get(REACTIVE), metrics.get("elapsed", {}).get(REACTIVE)
    if not q or not e:
        return None
    qs = [v for k, v in q.items() if k[0] in cells]
    es = [v for k, v in e.items() if k[0] in cells]
    if not qs or not es or sum(es) <= 0:
        return None
    return sum(qs) / sum(es)


def _rung(metrics: Mapping[str, dict], label: str) -> dict:
    tab = metrics["elapsed"]
    cells = sorted(set(_cells(tab, GRAPH)) & set(_cells(tab, POINTWISE)))
    if not cells:
        return {"rung": label, "verdict": V_UNREADABLE,
                "why": f"no cell carries both {GRAPH} and {POINTWISE} "
                       f"(graph={len(_cells(tab, GRAPH))} cells, pointwise={len(_cells(tab, POINTWISE))})"}
    g, p = _collapse(tab, GRAPH, cells), _collapse(tab, POINTWISE, cells)
    if g is None or p is None:
        return {"rung": label, "verdict": V_UNREADABLE,
                "why": "an arm is ragged -- some checkpoint is missing a cell"}
    out = {**read_f1(g, p), "rung": label, "cells": cells, "n_graph": len(g), "n_point": len(p)}
    qs = _queue_share(metrics, cells)
    out["queue_share"], out["saturated"] = qs, saturated(qs)
    # F4: context only, never a bar.
    react = _reactive_like(tab, cells, sorted(g))
    if react:
        out["graph_vs_reactive"] = read_pair_pct(g, react, tol=5.0, alpha=0.05, min_seeds=len(g))
        out["pointwise_vs_reactive"] = read_pair_pct(p, react, tol=5.0, alpha=0.05, min_seeds=len(p))
    return out


def _dedupe_r0_mpoff(po_rows, p3_rows):
    """`mpoff_516` at R0 exists TWICE and both are correct.

    partial_state_v3 P3 served it at seeds 1, 2, 4, 5; this lineage then served all 16 as
    `v3ext`, because 20 clients is the R0 cell and the arm had to cover every checkpoint. B4
    avoided the overlap at R3 by running only the other 12 seeds; here the overlap is real, and
    peer_only_v1's `tables()` refuses a collision rather than letting one silently win -- which
    is the correct behaviour and is why this function exists instead of a looser loader.

    They are the SAME checkpoint on the SAME cell, so they must agree exactly. That is asserted,
    not assumed: if a re-serve of one checkpoint on one cell ever stopped being bit-identical,
    every paired comparison in this record would be resting on sand. Measured 16/16 identical,
    max|delta| = 0.000000 -- the same check A0 ran at R3.
    """
    v3ext = {(r["cell"], int(r["checkpoint_seed"])): float(r["averageElapsedTime"])
             for r in po_rows
             if r.get("rung") == "R0" and r.get("corpus") == "v3ext" and r.get("arm_kind") == "mpoff"}
    kept, dropped = [], 0
    for r in p3_rows:
        key = (r.get("cell"), int(r.get("checkpoint_seed", -1)))
        if r.get("rung") == "R0" and r.get("arm_kind") == "mpoff" and key in v3ext:
            got, want = float(r["averageElapsedTime"]), v3ext[key]
            if got != want:
                raise ValueError(
                    f"FAIL LOUD: mpoff_516 at {key} reads {got} in psv3_p3 and {want} in po_v1. "
                    "A re-serve of one checkpoint on one cell is not reproducing; do not "
                    "compare anything in this record until that is explained."
                )
            dropped += 1
            continue
        kept.append(r)
    return kept, dropped


def read(po_dir: str, p3_dir: str, client_dir: str) -> dict:
    po_rows = load_server(po_dir)
    p3_rows, dropped = _dedupe_r0_mpoff(po_rows, load_server(p3_dir))
    srv = server_tables(po_rows, p3_rows)
    cli = client_tables(load_clients(client_dir))

    per_rung: Dict[int, dict] = {}
    for nc in F_CLIENTS:
        if nc == 20:
            metrics = srv.get("R0")
            if not metrics:
                per_rung[nc] = {"rung": "C20", "verdict": V_UNREADABLE, "why": "R0 absent"}
                continue
            # The 20-client rung IS the R0 server cell. Assert it rather than trust the mapping.
            bad = [c for c in _cells(metrics["elapsed"], GRAPH) if not c.startswith("cs6s")]
            if bad:
                raise ValueError(
                    f"FAIL LOUD: R0 is meant to be the 6-server / 20-client cells, got {bad}"
                )
            per_rung[nc] = _rung(metrics, "C20 (= R0 cells)")
        else:
            metrics = cli.get(nc)
            per_rung[nc] = (_rung(metrics, f"C{nc}") if metrics
                            else {"rung": f"C{nc}", "verdict": V_UNREADABLE, "why": "rung absent"})

    return {"bar": {"separate_pct": 5.0, "alpha": 0.05, "min_seeds": F_MIN_SEEDS},
            "r0_mpoff_duplicates_verified_identical": dropped,
            "per_rung": per_rung,
            "F2": read_f2(per_rung),
            "F3": read_f3(per_rung)}


def _line(r: Mapping[str, object]) -> str:
    if r.get("verdict") == V_UNREADABLE:
        return f"  {r['rung']:18s} {V_UNREADABLE} -- {r.get('why')}"
    sat = r.get("saturated")
    tag = ("SATURATED" if sat else "unsaturated") if sat is not None else "saturation UNKNOWN"
    s = (f"  {r['rung']:18s} {r['verdict']:20s} median {float(r['median']):+7.2f}%  "
         f"p={float(r['p']):.4f}  {r['v3_ahead']}/{r['n']}"
         f"  [reactive queue {float(r['queue_share']):.0%}, {tag}]"
         if r.get("queue_share") is not None else
         f"  {r['rung']:18s} {r['verdict']:20s} median {float(r['median']):+7.2f}%  "
         f"p={float(r['p']):.4f}  {r['v3_ahead']}/{r['n']}  [{tag}]")
    for key, name in (("graph_vs_reactive", "gnnedge0 "), ("pointwise_vs_reactive", "mpoff_516")):
        sub = r.get(key)
        if sub and sub.get("verdict") != V_UNREADABLE:
            s += (f"\n      {name} vs reactive: {float(sub['median']):+7.2f}%  "
                  f"p={float(sub['p']):.4f}  {sub['v3_ahead']}/{sub['n']}")
    return s


def report(res: Mapping[str, object]) -> str:
    out = ["best_arm_v1 -- F1 (per rung), F2 (the composite), F3 (descriptive)",
           f"bar: |median| >= {res['bar']['separate_pct']}%, p < {res['bar']['alpha']}, "
           f"n >= {res['bar']['min_seeds']} checkpoints   |   negative = gnnedge0 faster",
           f"(mpoff_516 at R0 was served twice; {res['r0_mpoff_duplicates_verified_identical']} "
           f"overlapping runs verified BIT-IDENTICAL before deduping)", ""]
    out += [_line(res["per_rung"][c]) for c in F_CLIENTS]
    f2, f3 = res["F2"], res["F3"]
    out += ["", f"F2: {f2['verdict']}", f"    {f2.get('why', '')}"]
    out += [f"F3: {f3['verdict']} (descriptive only; three rungs cannot establish a trend)"]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--po-dir", required=True)
    ap.add_argument("--p3-dir", required=True)
    ap.add_argument("--client-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = read(a.po_dir, a.p3_dir, a.client_dir)
    print(report(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2, default=str)
        print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

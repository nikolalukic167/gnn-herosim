"""bipartite_edge_v1 -- map the gate summaries onto the registered D1/D2/D3/D4 reads.

    python3 scripts_cosim/bipartite_edge_v1_gate_read.py \\
        --po-dir simulation_data/peer_affinity_live_gate/results/po_v1 \\
        --p3-dir simulation_data/peer_affinity_live_gate/results/psv3_p3 \\
        --client-dir simulation_data/peer_affinity_live_gate/results/po_v1_clients \\
        [--out d.json]

The D arms were run inside peer_only_v1's two gate scripts on purpose, so they share cells,
workloads, window and summary schema with the `gnn`, `peeronly` and reactive arms they are
compared against. That means this file builds NO loader of its own: it reuses the two readers
that already produced every published number on these rungs, and only collapses and compares.

Bars and verdicts live in scripts_cosim/bipartite_edge_v1_read.py, committed before any arm was
trained. Nothing here decides anything.

Two disciplines inherited from the parent lineage, each because it was got wrong once:
  * the unit is the CHECKPOINT, never the (cell, seed) pair;
  * an arm lost to a resource kill NEVER relaxes a registered bar. The registered read reports
    UNREADABLE and a DISCLOSED read prints beside it with the exclusion named -- both, always,
    so a reader cannot see the disclosed number without the registered one.
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

from scripts_cosim.bipartite_edge_v1_read import (  # noqa: E402
    D_CLIENTS, D_MIN_SEEDS, D_RUNGS,
    read_d1, read_d2, read_d3, read_d4, read_lineage,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE, collapse_to_seed  # noqa: E402
from scripts_cosim.peer_only_v1_gate_read import load as load_server, tables as server_tables  # noqa: E402
from scripts_cosim.peer_only_v1_client_read import (  # noqa: E402
    load as load_clients, tables as client_tables,
)

EDGE, EDGE0 = "be1670_gnnedge", "be1670_gnnedge0"
GNN, PEERONLY, REACTIVE = "1670_gnn", "1670_peeronly", "reactive"


def _cells(table: Mapping[str, dict], label: str) -> list:
    return sorted({c for (c, _s) in table.get(label, {})})


def _collapse(table: Mapping[str, dict], label: str, cells: Sequence[str]) -> Optional[Dict[int, float]]:
    """Checkpoint-level, over the cells EVERY arm in the comparison carries.

    Returns None rather than raising when the arm is absent, so a missing arm is reported as
    UNREADABLE by its bar instead of taking the whole read down -- the failure mode that made
    three OOM'd C2/C4 arms crash a read of 125 healthy ones.
    """
    arm = table.get(label)
    if not arm:
        return None
    try:
        return collapse_to_seed(arm, rung_cells=cells)
    except ValueError:
        return None


def _disclosed(table: Mapping[str, dict], label: str, cells: Sequence[str]) -> Optional[Dict[int, float]]:
    """The same collapse over only the seeds that carry every cell. NEVER used for a registered
    verdict -- the caller must label it disclosed and name what was excluded."""
    arm = table.get(label)
    if not arm:
        return None
    by_seed: Dict[int, Dict[str, float]] = {}
    for (cell, seed), v in arm.items():
        by_seed.setdefault(int(seed), {})[cell] = float(v)
    keep = {s: cs for s, cs in by_seed.items() if all(c in cs for c in cells)}
    return collapse_to_seed(
        {(c, s): v for s, cs in keep.items() for c, v in cs.items() if c in cells},
        rung_cells=cells,
    ) if keep else None


def _reactive_like(table: Mapping[str, dict], cells: Sequence[str], seeds: Sequence[int]) -> Optional[Dict[int, float]]:
    """Reactive is deterministic and keyed at seed 0, so intersecting seed sets with it yields
    the EMPTY set and the baseline vanishes from a table that plainly contains it (C4,
    2026-09-18). Collapse it on its own and replicate it across the arm's seeds."""
    arm = table.get(REACTIVE)
    if not arm:
        return None
    try:
        one = collapse_to_seed(arm, rung_cells=cells)[0]
    except (ValueError, KeyError):
        return None
    return {int(s): one for s in seeds}


def _read_one(table: Mapping[str, dict], label: str, queue_share: Optional[float]) -> dict:
    """Every D bar at one rung, registered and disclosed side by side."""
    cells = sorted(set(_cells(table, EDGE)) & set(_cells(table, EDGE0)) & set(_cells(table, GNN)))
    if not cells:
        return {"rung": label, "verdict": V_UNREADABLE,
                "why": "no cell carries gnnedge, gnnedge0 and gnn together"}

    out: dict = {"rung": label, "cells": cells}
    for tag, get in (("registered", _collapse), ("disclosed", _disclosed)):
        edge = get(table, EDGE, cells)
        edge0 = get(table, EDGE0, cells)
        gnn = get(table, GNN, cells)
        peeronly = get(table, PEERONLY, cells)
        if edge is None:
            # Distinguish the two causes: an arm that was never run, and one that ran but is
            # RAGGED because a cell died. "no gnnedge arm" on a table containing 63 gnnedge
            # summaries is the kind of message that costs an hour, so say which it is -- and
            # report every bar separately rather than collapsing four into one.
            why = ("gnnedge is absent from these cells" if not table.get(EDGE)
                   else "gnnedge is ragged -- some checkpoint is missing a cell")
            out[tag] = {"verdict": V_UNREADABLE, "why": why, "n_checkpoints": 0,
                        **{b: {"verdict": V_UNREADABLE, "why": why} for b in ("D1", "D2", "D3", "D4")},
                        "lineage": {"verdict": V_UNREADABLE, "why": why}}
            continue
        ms = None if tag == "registered" else len(edge)
        d1 = read_d1(edge, gnn, min_seeds=ms) if gnn else {"verdict": V_UNREADABLE, "why": "no gnn arm"}
        d2 = (read_d2(edge, edge0, d1_verdict=d1.get("verdict"), min_seeds=ms)
              if edge0 else {"verdict": V_UNREADABLE, "why": "no gnnedge0 control"})
        d3 = (read_d3(edge, peeronly, d1_verdict=d1.get("verdict"), min_seeds=ms)
              if peeronly else {"verdict": V_UNREADABLE, "why": "no peeronly arm"})
        reactive = _reactive_like(table, cells, sorted(edge))
        d4 = (read_d4(edge, reactive, queue_share=queue_share, min_seeds=ms)
              if reactive else {"verdict": V_UNREADABLE, "why": "no reactive baseline"})
        out[tag] = {"n_checkpoints": len(edge), "D1": d1, "D2": d2, "D3": d3, "D4": d4,
                    "lineage": read_lineage(d1, d2)}
    return out


def _queue_share(table: Mapping[str, dict], cells: Sequence[str]) -> Optional[float]:
    """Reactive's queue as a fraction of its elapsed: the registered saturation classification.
    Unknown stays None -- 'unknown is not a pass' (PARITY.md), so it must never read as False."""
    q, e = table.get("queue", {}).get(REACTIVE), table.get("elapsed", {}).get(REACTIVE)
    if not q or not e:
        return None
    qs = [q[k] for k in q if k[0] in cells]
    es = [e[k] for k in e if k[0] in cells]
    if not qs or not es or sum(es) <= 0:
        return None
    return sum(qs) / sum(es)


def read(po_dir: str, p3_dir: str, client_dir: Optional[str]) -> dict:
    res: dict = {"bar": {"separate_pct": 5.0, "alpha": 0.05, "min_seeds": D_MIN_SEEDS},
                 "rungs": []}

    srv = server_tables(load_server(po_dir), load_server(p3_dir))
    for rung in D_RUNGS:
        tab = srv.get(rung)
        if not tab:
            res["rungs"].append({"rung": rung, "verdict": V_UNREADABLE, "why": "rung absent"})
            continue
        cells = sorted(set(_cells(tab["elapsed"], EDGE)) & set(_cells(tab["elapsed"], GNN)))
        res["rungs"].append(_read_one(tab["elapsed"], rung, _queue_share(tab, cells)))

    if client_dir:
        cli = client_tables(load_clients(client_dir))
        for nc in D_CLIENTS:
            tab = cli.get(nc)
            if not tab:
                res["rungs"].append({"rung": f"C{nc}", "verdict": V_UNREADABLE, "why": "rung absent"})
                continue
            cells = sorted(set(_cells(tab["elapsed"], EDGE)) & set(_cells(tab["elapsed"], GNN)))
            res["rungs"].append(_read_one(tab["elapsed"], f"C{nc}", _queue_share(tab, cells)))
    return res


def _fmt(r: Mapping[str, object]) -> str:
    v = r.get("verdict")
    if v == V_UNREADABLE:
        return f"{V_UNREADABLE} ({r.get('why', '')})"
    med, p, ahead, n = r.get("median"), r.get("p"), r.get("ahead"), r.get("n")
    s = f"{v:32s} median {float(med):+7.2f}%  p={float(p):.4f}  {ahead}/{n}"
    if r.get("saturated") is not None:
        s += f"  [reactive queue {float(r['queue_share']):.0%}, " \
             f"{'SATURATED' if r['saturated'] else 'unsaturated'}]"
    elif "queue_share" in r:
        s += "  [saturation UNKNOWN -- not a pass]"
    return s


def report(res: Mapping[str, object]) -> str:
    out = ["bipartite_edge_v1 -- D1/D2/D3/D4",
           f"bar: |median| >= {res['bar']['separate_pct']}%, p < {res['bar']['alpha']}, "
           f"n >= {res['bar']['min_seeds']} checkpoints", ""]
    for rung in res["rungs"]:
        out.append(f"--- {rung['rung']} " + "-" * 60)
        if rung.get("verdict") == V_UNREADABLE:
            out.append(f"  {V_UNREADABLE}: {rung.get('why')}")
            out.append("")
            continue
        out.append(f"  cells: {', '.join(rung['cells'])}")
        for tag in ("registered", "disclosed"):
            block = rung.get(tag) or {}
            if block.get("verdict") == V_UNREADABLE:
                out.append(f"  [{tag}] {V_UNREADABLE}: {block.get('why')}")
                continue
            out.append(f"  [{tag}] n = {block['n_checkpoints']} checkpoints")
            for bar, what in (("D1", "gnnedge vs gnn      "), ("D2", "gnnedge vs gnnedge0 "),
                              ("D3", "gnnedge vs peeronly "), ("D4", "gnnedge vs reactive ")):
                out.append(f"      {bar} {what} {_fmt(block[bar])}")
            out.append(f"      lineage: {block['lineage']['verdict']} -- {block['lineage']['why']}")
        out.append("")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--po-dir", required=True)
    ap.add_argument("--p3-dir", required=True)
    ap.add_argument("--client-dir", default=None)
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

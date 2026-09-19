"""corpus_matched_v1 — map the gate summaries onto the registered G1/G2/G3/G4 reads.

    python3 scripts_cosim/corpus_matched_v1_gate_read.py \\
        --po-dir simulation_data/peer_affinity_live_gate/results/po_v1 \\
        --p3-dir simulation_data/peer_affinity_live_gate/results/psv3_p3 \\
        --client-dir simulation_data/peer_affinity_live_gate/results/po_v1_clients

Every arm G needs is ALREADY on disk, from `peer_only_v1` B7 and `best_arm_v1` F. That is the
point of the lineage and also its one hazard: a free contrast is easy to compute privately and
report selectively, so the bars live in `scripts_cosim/corpus_matched_v1_read.py`, committed
before this file was ever run. Nothing here decides anything.

The 20-client rung **is** the R0 server cell (6 servers, 20 clients), read from the SERVER
gate's results; the assertion that the cells are the `cs6s*` family is carried over from
`best_arm_v1_gate_read` rather than re-derived, because getting it wrong is silent.
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

from scripts_cosim.corpus_matched_v1_read import (  # noqa: E402
    G_CLIENTS, G_MIN_SEEDS, read_g1, read_g2, read_g3, read_g3_composite, read_g4, saturated,
)
from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    V_UNREADABLE, collapse_to_seed, read_pair_pct,
)
from scripts_cosim.best_arm_v1_gate_read import _dedupe_r0_mpoff  # noqa: E402
from scripts_cosim.peer_only_v1_gate_read import load as load_server, tables as server_tables  # noqa: E402
from scripts_cosim.peer_only_v1_client_read import (  # noqa: E402
    load as load_clients, tables as client_tables,
)

GRAPH_1670 = "be1670_gnnedge0"   # the repaired arm, 1,670 datasets
POINT_1670 = "1670_mpoff"        # its MP-OFF twin, SAME 1,670 datasets  -> G1 is matched
POINT_516 = "516_mpoff"          # the best pointwise arm, 516 datasets  -> G3's other corpus
GRAPH_516 = "cm516_gnnedge0"     # the repaired arm at 516 -> H, absent until trained
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
    baseline out of a table that plainly contains one. Collapse, then replicate."""
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


def _pair(metrics: Mapping[str, dict], a_label: str, b_label: str, reader, label: str) -> dict:
    """One rung of one registered contrast, on the cells that carry BOTH arms."""
    tab = metrics["elapsed"]
    cells = sorted(set(_cells(tab, a_label)) & set(_cells(tab, b_label)))
    if not cells:
        return {"rung": label, "verdict": V_UNREADABLE,
                "why": f"no cell carries both {a_label} and {b_label} "
                       f"(a={len(_cells(tab, a_label))} cells, b={len(_cells(tab, b_label))})"}
    a, b = _collapse(tab, a_label, cells), _collapse(tab, b_label, cells)
    if a is None or b is None:
        return {"rung": label, "verdict": V_UNREADABLE,
                "why": "an arm is ragged -- some checkpoint is missing a cell"}
    out = {**reader(a, b), "rung": label, "cells": cells, "n_a": len(a), "n_b": len(b),
           "arms": (a_label, b_label)}
    qs = _queue_share(metrics, cells)
    out["queue_share"], out["saturated"] = qs, saturated(qs)
    # Context only, never a bar: where does each arm stand against reactive on these cells?
    react = _reactive_like(tab, cells, sorted(a))
    if react:
        out["a_vs_reactive"] = read_pair_pct(a, react, tol=5.0, alpha=0.05, min_seeds=len(a))
        out["b_vs_reactive"] = read_pair_pct(b, react, tol=5.0, alpha=0.05, min_seeds=len(b))
    return out


def _rung_metrics(srv, cli, nc) -> Optional[dict]:
    """The 20-client rung IS the R0 server cell. Assert the mapping rather than trust it."""
    if nc != 20:
        return cli.get(nc)
    metrics = srv.get("R0")
    if not metrics:
        return None
    bad = [c for c in _cells(metrics["elapsed"], GRAPH_1670) if not c.startswith("cs6s")]
    if bad:
        raise ValueError(f"FAIL LOUD: R0 is meant to be the 6-server / 20-client cells, got {bad}")
    return metrics


def read(po_dir: str, p3_dir: str, client_dir: str) -> dict:
    po_rows = load_server(po_dir)
    p3_rows, dropped = _dedupe_r0_mpoff(po_rows, load_server(p3_dir))
    srv = server_tables(po_rows, p3_rows)
    cli = client_tables(load_clients(client_dir))

    g1: Dict[int, dict] = {}
    g3: Dict[int, dict] = {}
    h1: Dict[int, dict] = {}
    for nc in G_CLIENTS:
        metrics = _rung_metrics(srv, cli, nc)
        lab = "C20 (= R0 cells)" if nc == 20 else f"C{nc}"
        if not metrics:
            absent = {"verdict": V_UNREADABLE, "why": "rung absent"}
            g1[nc] = {"rung": lab, **absent}
            g3[nc] = {"rung": lab, **absent}
            h1[nc] = {"rung": lab, **absent}
            continue
        g1[nc] = _pair(metrics, GRAPH_1670, POINT_1670, read_g1, lab)
        g3[nc] = _pair(metrics, POINT_1670, POINT_516, read_g3, lab)
        h1[nc] = _pair(metrics, GRAPH_516, POINT_516, read_g1, lab)

    g2 = read_g2(g1)
    h2 = read_g2(h1)
    return {"bar": {"separate_pct": 5.0, "alpha": 0.05, "min_seeds": G_MIN_SEEDS},
            "r0_mpoff_duplicates_verified_identical": dropped,
            "G1": g1, "G2": g2,
            "G3": g3, "G3_composite": read_g3_composite(g3),
            "H1": h1, "H2": h2,
            "G4": read_g4(g2, None if h2.get("verdict") == V_UNREADABLE else h2)}


def _line(r: Mapping[str, object]) -> str:
    if r.get("verdict") == V_UNREADABLE:
        return f"  {r['rung']:18s} {V_UNREADABLE} -- {r.get('why')}"
    sat = r.get("saturated")
    tag = ("SATURATED" if sat else "unsaturated") if sat is not None else "saturation UNKNOWN"
    qs = r.get("queue_share")
    head = (f"  {r['rung']:18s} {r['verdict']:36s} median {float(r['median']):+7.2f}%  "
            f"p={float(r['p']):.4f}  {r['v3_ahead']}/{r['n']}")
    s = head + (f"  [reactive queue {float(qs):.0%}, {tag}]" if qs is not None else f"  [{tag}]")
    for key, arm in (("a_vs_reactive", 0), ("b_vs_reactive", 1)):
        sub = r.get(key)
        if sub and sub.get("verdict") != V_UNREADABLE:
            s += (f"\n      {r['arms'][arm]:18s} vs reactive: {float(sub['median']):+7.2f}%  "
                  f"p={float(sub['p']):.4f}  {sub['v3_ahead']}/{sub['n']}")
    return s


def report(res: Mapping[str, object]) -> str:
    out = ["corpus_matched_v1 -- G1/G2 (model class, corpus held fixed), G3 (the confound), "
           "G4 (the 2x2)",
           f"bar: |median| >= {res['bar']['separate_pct']}%, p < {res['bar']['alpha']}, "
           f"n >= {res['bar']['min_seeds']} checkpoints",
           f"(mpoff_516 at R0 was served twice; {res['r0_mpoff_duplicates_verified_identical']} "
           f"overlapping runs verified BIT-IDENTICAL before deduping)",
           "",
           f"G1  {GRAPH_1670} vs {POINT_1670}   (BOTH at 1,670 datasets; negative = graph faster)"]
    out += [_line(res["G1"][c]) for c in G_CLIENTS]
    g2 = res["G2"]
    out += ["", f"G2: {g2['verdict']}", f"    {g2.get('why', '')}", "",
            f"G3  {POINT_1670} vs {POINT_516}   (SAME architecture; negative = 1,670 faster)"]
    out += [_line(res["G3"][c]) for c in G_CLIENTS]
    g3c = res["G3_composite"]
    out += ["", f"G3: {g3c['verdict']}", f"    {g3c.get('why', '')}", "",
            f"H1  {GRAPH_516} vs {POINT_516}   (BOTH at 516 datasets)"]
    out += [_line(res["H1"][c]) for c in G_CLIENTS]
    h2, g4 = res["H2"], res["G4"]
    out += ["", f"H2: {h2['verdict']}", f"    {h2.get('why', '')}",
            "", f"G4: {g4['verdict']} -- {g4.get('why', g4.get('note', ''))}"]
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

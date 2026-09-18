#!/usr/bin/env python3
"""peer_only_v1 — the CLIENT axis read: B7, B8 and C2, from the gate's own summaries.

Why this file exists. B7 and B8 were read from a terminal with inline Python and only their
conclusions were written down. That is the one thing this lineage keeps being burned by: a
number quoted from a read nobody can re-run. This is that read, in the tree, with the bars it
uses imported from `peer_only_v1_read.py` rather than restated.

  B7  the client ladder: each learned arm against reactive Knative, per rung, plus the
      registered saturation / load-vs-dispersion classification of the BASELINE
  B8  peeronly_1670 vs mpoff_516 at the unsaturated rungs (AMENDMENT 5)
  C2  gnn_1670 against reactive AND against peeronly, at the unsaturated rungs (AMENDMENT 7)

Every contrast is CHECKPOINT-LEVEL: one value per checkpoint seed, the median over that
checkpoint's cells, via `collapse_to_seed`. The unit for a claim about an architecture is the
checkpoint, not the (cell, seed) pair — B3 cost this lineage a halved headline to learn that.

Usage:
  python3 scripts_cosim/peer_only_v1_client_read.py \
      --dir simulation_data/peer_affinity_live_gate/results/po_v1_clients [--out x.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.peer_only_v1_read import (  # noqa: E402
    B7_CLIENTS,
    B8_CLIENTS,
    C2_CLIENTS,
    C2_DISCLOSED_MIN_SEEDS,
    V_UNREADABLE,
    classify_b7_rungs,
    collapse_to_seed,
    read_b3,
    read_b4,
    read_b8,
    read_c1,
    read_c2,
    read_c2_rung,
    read_vs_reactive,
)


def load(directory: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "*.summary.json"))):
        d = json.load(open(p))
        if "arm" not in d or "clients" not in d:
            raise ValueError(f"FAIL LOUD: {p} is not a client-axis summary")
        out.append(d)
    if not out:
        raise ValueError(f"FAIL LOUD: no summaries under {directory}")
    return out


def tables(rows: Sequence[dict]) -> Dict[int, Dict[str, Dict[str, dict]]]:
    """clients -> metric -> arm label -> {(cell, seed): value}.

    `v3ext` carries the `516_*` label: it IS partial_state_v3's mpoff, re-served, and A0 proves
    that re-serve bit-identical — so it extends the same arm rather than naming a new one.
    """
    out: Dict[int, Dict[str, Dict[str, dict]]] = {}
    for d in rows:
        # Loud here as well as in load(): tables() is importable on its own, and a summary
        # from the SERVER ladder has every other field this function reads.
        if "clients" not in d or "arm_kind" not in d:
            raise ValueError(
                f"FAIL LOUD: {d.get('arm', d)!r} carries no client count / arm kind -- this is "
                "not a client-axis summary. The server ladder is read by peer_only_v1_gate_read."
            )
        n = int(d["clients"])
        t = out.setdefault(n, {"elapsed": {}, "queue": {}})
        kind = d["arm_kind"]
        if kind == "reactive":
            label, seed = "reactive", 0
        else:
            corpus = "516" if d["corpus"] == "v3ext" else d["corpus"]
            label, seed = f"{corpus}_{kind}", int(d["checkpoint_seed"])
        key = (d["cell"], seed)
        if key in t["elapsed"].get(label, {}):
            raise ValueError(
                f"FAIL LOUD: two summaries for {label} at {key}, {n} clients -- an arm name is "
                "colliding, do not let one silently win"
            )
        t["elapsed"].setdefault(label, {})[key] = float(d["averageElapsedTime"])
        t["queue"].setdefault(label, {})[key] = float(d["averageQueueTime"])
    return out


def _cells(table: Mapping[str, dict], label: str) -> List[str]:
    return sorted({c for (c, _s) in table.get(label, {})})


def _by_seed(table: Mapping[str, dict], label: str, cells: Sequence[str]) -> Optional[dict]:
    arm = table.get(label)
    if not arm:
        return None
    return collapse_to_seed(arm, rung_cells=cells)


def _by_seed_disclosed(table: Mapping[str, dict], label: str,
                       cells: Sequence[str]) -> Tuple[Optional[dict], List[int]]:
    """`_by_seed`, but dropping the checkpoints that do not carry every cell.

    For the DISCLOSED read only, when an arm was lost to a resource kill. Returns the
    collapsed values and the excluded seeds, so the exclusion is printed rather than absorbed.
    `collapse_to_seed` is deliberately strict and stays strict -- this filters its INPUT.
    """
    arm = table.get(label)
    if not arm:
        return None, []
    by_seed: Dict[int, set] = {}
    for (cell, seed) in arm:
        by_seed.setdefault(int(seed), set()).add(cell)
    complete = sorted(s for s, cs in by_seed.items() if set(cells) <= cs)
    excluded = sorted(s for s in by_seed if s not in complete)
    if not complete:
        return None, excluded
    kept = {k: v for k, v in arm.items() if int(k[1]) in set(complete)}
    return collapse_to_seed(kept, rung_cells=cells), excluded


def _reactive_like(reactive: Mapping[tuple, float], cells: Sequence[str],
                   seeds: Sequence[int]) -> dict:
    """Reactive replicated across the learned arm's seed keys.

    Reactive is deterministic and carries no checkpoint seed, so the honest pairing is every
    checkpoint against the SAME baseline on the SAME cells. Replicating it makes that explicit
    instead of hiding it in a one-sample test.
    """
    base = collapse_to_seed({(c, 0): reactive[(c, 0)] for c in cells if (c, 0) in reactive},
                            rung_cells=cells)[0]
    return {s: base for s in seeds}


def read_ladder(tab: Mapping[int, dict]) -> dict:
    """B7 + B8 + C2 over whatever the summaries actually contain."""
    result: Dict[str, object] = {}

    # --- B7: classify the BASELINE first, before any learned arm is read ------------------
    reactive_metrics = {}
    for n, t in tab.items():
        cells = _cells(t["elapsed"], "reactive")
        if not cells:
            continue
        el = collapse_to_seed({(c, 0): t["elapsed"]["reactive"][(c, 0)] for c in cells},
                              rung_cells=cells)[0]
        q = collapse_to_seed({(c, 0): t["queue"]["reactive"][(c, 0)] for c in cells},
                             rung_cells=cells)[0]
        reactive_metrics[n] = {"elapsed": el, "queue": q}
    result["b7_classification"] = classify_b7_rungs(reactive_metrics)
    result["reactive"] = reactive_metrics

    # --- every learned arm against reactive, per rung -------------------------------------
    vs_reactive: Dict[str, Dict[int, dict]] = {}
    for n in sorted(tab):
        t = tab[n]
        if "reactive" not in t["elapsed"]:
            continue
        for label in sorted(t["elapsed"]):
            if label == "reactive":
                continue
            cells = _cells(t["elapsed"], label)
            try:
                arm = _by_seed(t["elapsed"], label, cells)
            except ValueError as exc:      # an arm died -- name it, do not average around it
                vs_reactive.setdefault(label, {})[n] = {"verdict": V_UNREADABLE,
                                                        "reason": str(exc)}
                continue
            if not arm:
                continue
            base = _reactive_like(t["elapsed"]["reactive"], cells, sorted(arm))
            vs_reactive.setdefault(label, {})[n] = read_vs_reactive(arm, base)
    result["vs_reactive"] = vs_reactive

    # --- peeronly vs its own twin, per rung (B7's architecture contrast) -------------------
    twin: Dict[int, dict] = {}
    for n in sorted(tab):
        t = tab[n]
        cells = _cells(t["elapsed"], "1670_peeronly")
        try:
            po = _by_seed(t["elapsed"], "1670_peeronly", cells)
            mp = _by_seed(t["elapsed"], "1670_mpoff", cells)
        except ValueError as exc:
            twin[n] = {"verdict": V_UNREADABLE, "reason": str(exc)}
            continue
        if po and mp:
            twin[n] = read_b3(po, mp)
    result["peeronly_vs_mpoff"] = twin

    # --- B8: peeronly_1670 vs mpoff_516 at the unsaturated rungs --------------------------
    b8: Dict[int, dict] = {}
    for n in B8_CLIENTS:
        t = tab.get(n)
        if not t:
            continue
        cells = _cells(t["elapsed"], "1670_peeronly")
        try:
            po = _by_seed(t["elapsed"], "1670_peeronly", cells)
            m516 = _by_seed(t["elapsed"], "516_mpoff", cells)
        except ValueError as exc:
            b8[n] = {"verdict": V_UNREADABLE, "reason": str(exc)}
            continue
        if po and m516:
            b8[n] = read_b4(po, m516)
    result["b8"] = read_b8(b8) if len(b8) == len(B8_CLIENTS) else {
        "verdict": V_UNREADABLE, "reason": f"have {sorted(b8)}, need {list(B8_CLIENTS)}",
        "per_rung": b8}

    # --- C2: the bipartite arm, vs reactive and vs its peeronly sibling -------------------
    c2_react, c2_sib = {}, {}
    disclosed_react, disclosed_sib, excluded_by_rung = {}, {}, {}
    for n in C2_CLIENTS:
        t = tab.get(n)
        if not t:
            continue
        cells = _cells(t["elapsed"], "1670_gnn")
        try:
            gnn = _by_seed(t["elapsed"], "1670_gnn", cells)
        except ValueError as exc:
            # An arm died. The REGISTERED read refuses and names the cause; the disclosed read
            # below still prints, so the surviving checkpoints are not thrown away.
            gnn = None
            c2_react[n] = {"verdict": V_UNREADABLE, "reason": str(exc)}
            c2_sib[n] = {"verdict": V_UNREADABLE, "reason": str(exc)}
        if gnn and "reactive" in t["elapsed"]:
            base = _reactive_like(t["elapsed"]["reactive"], cells, sorted(gnn))
            c2_react[n] = read_c2_rung(gnn, base)
        if gnn:
            po = _by_seed(t["elapsed"], "1670_peeronly", cells)
            if po:
                c2_sib[n] = read_c1(po, gnn)
        # --- DISCLOSED, always computed so a refusal is never the only thing printed --------
        g_d, excluded = _by_seed_disclosed(t["elapsed"], "1670_gnn", cells)
        excluded_by_rung[n] = excluded
        if g_d and "reactive" in t["elapsed"]:
            base_d = _reactive_like(t["elapsed"]["reactive"], cells, sorted(g_d))
            disclosed_react[n] = read_c2_rung(g_d, base_d, min_seeds=C2_DISCLOSED_MIN_SEEDS)
        if g_d:
            po_d, _ = _by_seed_disclosed(t["elapsed"], "1670_peeronly", cells)
            if po_d:
                shared = sorted(set(g_d) & set(po_d))
                disclosed_sib[n] = read_c1({s: po_d[s] for s in shared},
                                           {s: g_d[s] for s in shared},
                                           min_seeds=C2_DISCLOSED_MIN_SEEDS)
    result["c2_disclosed"] = {"vs_reactive": disclosed_react, "vs_peeronly": disclosed_sib,
                              "excluded_seeds": excluded_by_rung,
                              "min_seeds": C2_DISCLOSED_MIN_SEEDS}
    result["c2"] = (read_c2(c2_react, c2_sib)
                    if len(c2_react) == len(C2_CLIENTS) and len(c2_sib) == len(C2_CLIENTS)
                    else {"verdict": V_UNREADABLE,
                          "reason": f"gnn read at {sorted(c2_react)}, need {list(C2_CLIENTS)}",
                          "vs_reactive": c2_react, "vs_peeronly": c2_sib})
    return result


def _fmt(r: dict) -> str:
    if not r or r.get("verdict") == V_UNREADABLE:
        return f"{'UNREADABLE':>28}"
    return (f"{r['median']:+7.2f}%  p={r['p']:.4f}  {r.get('v3_ahead', '?')}/{r['n']:<3} "
            f"{r['verdict']}")


def report(res: dict) -> str:
    out: List[str] = []
    cl = res["b7_classification"]
    out.append("=== B7 — the baseline, classified before any learned arm ===")
    out.append(f"  verdict          : {cl['verdict']}")
    out.append(f"  reactive spread  : {cl.get('reactive_elapsed_spread_pct', float('nan')):.1f}% "
               f"(flat band {cl['bar']['load_flat_pct']}%)")
    for n in sorted(res["reactive"]):
        share = cl["queue_share"][n]
        tag = "SATURATED" if n in cl["saturated"] else "unsaturated"
        out.append(f"    {n:>3} clients: elapsed {res['reactive'][n]['elapsed']:7.2f}s  "
                   f"queue share {share:5.1%}  {tag}")
    out.append(f"  primary rung     : {cl.get('primary_rung')} clients (largest unsaturated)")

    out.append("\n=== each learned arm vs reactive Knative (checkpoint-level) ===")
    for label in sorted(res["vs_reactive"]):
        for n in sorted(res["vs_reactive"][label]):
            out.append(f"  {n:>3} clients  {label:>16}  {_fmt(res['vs_reactive'][label][n])}")

    out.append("\n=== peeronly vs its MP-OFF twin (both 1670) ===")
    for n in sorted(res["peeronly_vs_mpoff"]):
        out.append(f"  {n:>3} clients  {_fmt(res['peeronly_vs_mpoff'][n])}")

    out.append("\n=== B8 — peeronly_1670 vs mpoff_516 (clause 3) ===")
    b8 = res["b8"]
    out.append(f"  verdict: {b8['verdict']}"
               + (f"   scoped_to_saturated={b8['scoped_to_saturated']}"
                  if "scoped_to_saturated" in b8 else f"   ({b8.get('reason', '')})"))
    for n in sorted(b8.get("per_rung", {})):
        out.append(f"  {n:>3} clients  {_fmt(b8['per_rung'][n])}")

    out.append("\n=== C2 — the bipartite arm at the unsaturated rungs ===")
    c2 = res["c2"]
    out.append(f"  verdict: {c2['verdict']}"
               + (f"   works at {c2['rungs_where_it_works']} clients"
                  if "rungs_where_it_works" in c2 else f"   ({c2.get('reason', '')})"))
    for n in sorted(c2.get("vs_reactive", {})):
        out.append(f"  {n:>3} clients  gnn vs reactive   {_fmt(c2['vs_reactive'][n])}")
    for n in sorted(c2.get("vs_peeronly", {})):
        out.append(f"  {n:>3} clients  peeronly vs gnn   {_fmt(c2['vs_peeronly'][n])}")
    d = res.get("c2_disclosed") or {}
    if d.get("vs_reactive") or d.get("vs_peeronly"):
        out.append(f"\n  --- DISCLOSED (>= {d.get('min_seeds')} complete checkpoints; NOT the "
                   f"registered bar) ---")
        for n in sorted(d.get("excluded_seeds", {})):
            ex = d["excluded_seeds"][n]
            out.append(f"  {n:>3} clients  excluded seeds: {ex if ex else 'none'}")
        for n in sorted(d.get("vs_reactive", {})):
            out.append(f"  {n:>3} clients  gnn vs reactive   {_fmt(d['vs_reactive'][n])}")
        for n in sorted(d.get("vs_peeronly", {})):
            out.append(f"  {n:>3} clients  peeronly vs gnn   {_fmt(d['vs_peeronly'][n])}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    rows = load(args.dir)
    tab = tables(rows)
    missing = [c for c in B7_CLIENTS if c not in tab]
    if missing:
        print(f"[note] no summaries at {missing} clients "
              f"(the 5- and 10-client rungs are UNSERVABLE by every policy -- recorded)")
    res = read_ladder(tab)
    print(report(res))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

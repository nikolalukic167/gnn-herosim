"""peer_only_v1 -- map the gate summaries onto the registered reads.

    python3 scripts_cosim/peer_only_v1_gate_read.py --po-dir results/po_v1 --p3-dir results/psv3_p3 \
        [--phase a|b|b3] [--out x.json]

Sources, per rung (R0 = 6 servers, R3 = 80):
  * reactive, gnn(516) and mpoff(516) arms come from partial_state_v3 P3's summaries
    (results/psv3_p3), reused under A0;
  * A0 re-serves, peeronly(516) and every 1670 arm come from results/po_v1.
Pairing key everywhere is (cell, checkpoint seed). Bars and verdicts live in
peer_only_v1_read.py; this file only builds the inputs and prints every number.
--phase b5 adds AMENDMENT 3, the cluster-size ladder.
--phase b3 adds AMENDMENT 1: the same B2 contrast collapsed to ONE VALUE PER CHECKPOINT
over all 16 trained seeds, printed beside the registered pair-level test, never instead of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from scripts_cosim.peer_only_v1_read import (
    A2_RUNGS, B3_RUNG, B4_RUNG, B5_RUNGS, B5_SERVERS, V_UNREADABLE, collapse_to_seed,
    headline, read_a0, read_a2_rung, read_a3_rung, read_a4_rung, read_b1_rung,
)
from scripts_cosim.peer_only_v1_read import read_b3 as read_b3_bar
from scripts_cosim.peer_only_v1_read import read_b4 as read_b4_bar
from scripts_cosim.peer_only_v1_read import read_b5 as read_b5_bar

PairKey = Tuple[str, int]
ArmTable = Dict[str, Dict[str, Dict[PairKey, float]]]     # metric -> arm label -> {(cell, seed): value}


def load(directory: str) -> List[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "*.summary.json"))):
        d = json.load(open(p))
        if "arm" not in d:
            raise ValueError(f"FAIL LOUD: {p} has no arm name")
        out.append(d)
    return out


def _peer_per_task(d: dict) -> float:
    n = float(d.get("num_tasks") or 0) or 1.0
    return (float(d.get("totalPeerExchangeTime") or 0.0) + float(d.get("totalPeerRendezvousWait") or 0.0)) / n


def tables(po: Sequence[dict], p3: Sequence[dict]) -> Dict[str, ArmTable]:
    """rung -> metric -> arm label ('516_gnn', '516_mpoff', '516_peeronly', '1670_*',
    'v3_gnn' (A0 re-serve), 'reactive') -> {(cell, seed): value}."""
    out: Dict[str, ArmTable] = {}

    def put(rung, label, cell, seed, d):
        t = out.setdefault(rung, {"elapsed": {}, "queue": {}, "peer": {}})
        if (cell, seed) in t["elapsed"].get(label, {}):
            raise ValueError(f"FAIL LOUD: two summaries for {label} at {(cell, seed)} in {rung} "
                             "-- an arm name is colliding, do not let one silently win")
        t["elapsed"].setdefault(label, {})[(cell, seed)] = float(d["averageElapsedTime"])
        t["queue"].setdefault(label, {})[(cell, seed)] = float(d["averageQueueTime"])
        t["peer"].setdefault(label, {})[(cell, seed)] = _peer_per_task(d)

    for d in p3:
        if d["rung"] not in A2_RUNGS:
            continue
        if d["arm_kind"] == "reactive":
            put(d["rung"], "reactive", d["cell"], 0, d)
        else:
            put(d["rung"], f"516_{d['arm_kind']}", d["cell"], int(d["checkpoint_seed"]), d)
    for d in po:
        # v3ext is B4's extension of the 516-corpus mpoff arm: same checkpoints partial_state_v3
        # trained, whose re-serve A0 proves bit-identical, so it carries the 516_* label.
        corpus = "516" if d["corpus"] == "v3ext" else d["corpus"]
        put(d["rung"], f"{corpus}_{d['arm_kind']}", d["cell"], int(d["checkpoint_seed"]), d)
    return out


def read_a0_from(tab: Mapping[str, ArmTable]) -> dict:
    """The v3_* re-serves against P3's 516_* arms on the same (cell, seed)."""
    pairs = {}
    for rung, t in tab.items():
        for arm in ("gnn", "mpoff"):
            now = t["elapsed"].get(f"v3_{arm}", {})
            then = t["elapsed"].get(f"516_{arm}", {})
            for key, v in now.items():
                if key in then:
                    pairs[f"{rung}:{key[0]}:{arm}:s{key[1]}"] = (v, then[key])
    return read_a0(pairs)


def read_phase(tab: Mapping[str, ArmTable], corpus: str) -> dict:
    """A2/A3/A4 (or B2 with corpus='1670') per rung, plus the headline."""
    out = {"corpus": corpus, "rungs": {}}
    for rung in A2_RUNGS:
        t = tab.get(rung)
        if not t:
            out["rungs"][rung] = {"a2": {"verdict": V_UNREADABLE, "reason": "no summaries"}}
            continue
        po, mp, gn = (t["elapsed"].get(f"{corpus}_peeronly", {}), t["elapsed"].get(f"{corpus}_mpoff", {}),
                      t["elapsed"].get(f"{corpus}_gnn", {}))
        r = {"a2": read_a2_rung(po, mp),
             "a3": read_a3_rung(t["queue"].get(f"{corpus}_peeronly", {}), t["queue"].get(f"{corpus}_gnn", {}),
                                t["queue"].get(f"{corpus}_mpoff", {})),
             "a4": read_a4_rung(t["peer"].get(f"{corpus}_peeronly", {}), t["peer"].get(f"{corpus}_mpoff", {})),
             "n": {"peeronly": len(po), "mpoff": len(mp), "gnn": len(gn)}}
        # descriptive: each arm relative to reactive on its cell (the partial_state_v3 statistic)
        react = t["elapsed"].get("reactive", {})
        for arm, table in (("peeronly", po), ("mpoff", mp), ("gnn", gn)):
            rel = [v / react[(c, 0)] - 1.0 for (c, s), v in table.items() if (c, 0) in react]
            r[f"d_{arm}"] = (sorted(rel)[len(rel) // 2] if rel else None)
        out["rungs"][rung] = r
    out["headline"] = headline({k: v["a2"] for k, v in out["rungs"].items()})
    return out


def read_b1(tab: Mapping[str, ArmTable]) -> dict:
    out = {}
    for arm in ("gnn", "mpoff", "peeronly"):
        out[arm] = {rung: read_b1_rung(tab.get(rung, {}).get("elapsed", {}).get(f"1670_{arm}", {}),
                                       tab.get(rung, {}).get("elapsed", {}).get(f"516_{arm}", {}))
                    for rung in A2_RUNGS}
    return out


def read_b3(tab: Mapping[str, ArmTable]) -> dict:
    """AMENDMENT 1: B2's contrast with the CHECKPOINT as the unit, at B3_RUNG.

    Prints beside B2 rather than replacing it -- the pair-level test is what was registered,
    the seed-level test is what an architectural claim needs. See gate-tools 2026-09-17.
    """
    t = tab.get(B3_RUNG)
    if not t:
        return {"verdict": V_UNREADABLE, "reason": f"no summaries at {B3_RUNG}"}
    po_pairs = t["elapsed"].get("1670_peeronly", {})
    mp_pairs = t["elapsed"].get("1670_mpoff", {})
    cells = sorted({c for c, _ in po_pairs} & {c for c, _ in mp_pairs})
    if not cells:
        return {"verdict": V_UNREADABLE, "reason": "no shared cells"}
    po = collapse_to_seed(po_pairs, rung_cells=cells)
    mp = collapse_to_seed(mp_pairs, rung_cells=cells)
    r = read_b3_bar(po, mp)
    return {**r, "rung": B3_RUNG, "cells": cells,
            "per_seed_pct": {s: 100.0 * (po[s] / mp[s] - 1.0) for s in sorted(set(po) & set(mp))}}


def complete_seeds(tab: Mapping[str, ArmTable], rung: str) -> set:
    """Checkpoint seeds carrying EVERY shared cell for both arms at this rung."""
    t = tab.get(rung) or {}
    po_pairs, mp_pairs = t.get("elapsed", {}).get("1670_peeronly", {}), t.get("elapsed", {}).get("1670_mpoff", {})
    cells = {c for c, _ in po_pairs} & {c for c, _ in mp_pairs}
    if not cells:
        return set()
    out = set()
    for arm_seeds in ({s for _, s in po_pairs},):
        for s in arm_seeds:
            if all((c, s) in po_pairs and (c, s) in mp_pairs for c in cells):
                out.add(s)
    return out


def _b3_at(tab: Mapping[str, ArmTable], rung: str, seeds: Optional[Sequence[int]] = None) -> dict:
    """The B3 statistic at one rung: one value per checkpoint, peeronly_1670 vs mpoff_1670."""
    t = tab.get(rung)
    if not t:
        return {"verdict": V_UNREADABLE, "reason": f"no summaries at {rung}"}
    po_pairs, mp_pairs = t["elapsed"].get("1670_peeronly", {}), t["elapsed"].get("1670_mpoff", {})
    if seeds is not None:
        keep = set(seeds)
        po_pairs = {k: v for k, v in po_pairs.items() if k[1] in keep}
        mp_pairs = {k: v for k, v in mp_pairs.items() if k[1] in keep}
    cells = sorted({c for c, _ in po_pairs} & {c for c, _ in mp_pairs})
    if not cells:
        return {"verdict": V_UNREADABLE, "reason": f"no shared cells at {rung}"}
    try:
        po = collapse_to_seed(po_pairs, rung_cells=cells)
        mp = collapse_to_seed(mp_pairs, rung_cells=cells)
    except ValueError as exc:                      # an arm died; name it, do not average around it
        return {"verdict": V_UNREADABLE, "reason": str(exc)}
    shared = sorted(set(po) & set(mp))
    r = read_b3_bar({s: po[s] for s in shared}, {s: mp[s] for s in shared})
    return {**r, "rung": rung, "cells": cells,
            "per_seed_pct": {s: 100.0 * (po[s] / mp[s] - 1.0) for s in shared}}


def read_b5(tab: Mapping[str, ArmTable]) -> dict:
    """AMENDMENT 3: the B3 contrast across the whole cluster-size ladder.

    The REGISTERED read requires all 16 checkpoints at every rung. When an arm is lost to a
    resource kill it reports UNREADABLE -- the bar is never relaxed. A second DISCLOSED read
    over the checkpoints complete at EVERY rung is attached beside it, so 15 good checkpoints
    are not thrown away and the exclusion is visible. Read by cause, not by count.
    """
    res = read_b5_bar({rung: _b3_at(tab, rung) for rung in B5_RUNGS})
    common = set.intersection(*(complete_seeds(tab, r) for r in B5_RUNGS)) if tab else set()
    excluded = sorted(set(range(1, 17)) - common)
    if excluded and common:
        res["disclosed"] = {
            "excluded_seeds": excluded, "n_seeds": len(common),
            "note": "not the registered read: B5_MIN_SEEDS is not relaxed",
            "per_rung": {rung: _b3_at(tab, rung, seeds=sorted(common)) for rung in B5_RUNGS},
        }
    return res


def format_b5(res: dict) -> str:
    lines = [f"B5 (AMENDMENT 3) -- peeronly vs mpoff (1670) by CLUSTER SIZE, one value per "
             f"checkpoint -> {res['verdict']}"]
    for rung in B5_RUNGS:
        r = res["per_rung"][rung]
        head = f"    {rung} ({B5_SERVERS[rung]:2d} servers): "
        lines.append(head + (f"{r['verdict']} ({r.get('reason')})" if r.get("verdict") == V_UNREADABLE
                             else fmt_pair(r)))
    if res.get("crossover_servers"):
        lines.append(f"    first rung clearing the bar: {res['crossover_rung']} "
                     f"({res['crossover_servers']} servers)")
    elif res.get("verdict") != V_UNREADABLE:
        lines.append("    no rung clears the bar")
    d = res.get("disclosed")
    if d:
        lines.append(f"  DISCLOSED (NOT the registered read) -- {d['n_seeds']} checkpoints complete at "
                     f"every rung; excluded {d['excluded_seeds']} (resource kill, read by cause)")
        for rung in B5_RUNGS:
            r = d["per_rung"][rung]
            head = f"    {rung} ({B5_SERVERS[rung]:2d} servers): "
            lines.append(head + (f"{r['verdict']} ({r.get('reason')})" if r.get("verdict") == V_UNREADABLE
                                 else fmt_pair(r)))
    return "\n".join(lines)


def read_b4(tab: Mapping[str, ArmTable]) -> dict:
    """AMENDMENT 2: clause 3 -- peeronly_1670 vs mpoff_516 -- with the checkpoint as the unit."""
    t = tab.get(B4_RUNG)
    if not t:
        return {"verdict": V_UNREADABLE, "reason": f"no summaries at {B4_RUNG}"}
    po_pairs, mp_pairs = t["elapsed"].get("1670_peeronly", {}), t["elapsed"].get("516_mpoff", {})
    cells = sorted({c for c, _ in po_pairs} & {c for c, _ in mp_pairs})
    if not cells:
        return {"verdict": V_UNREADABLE, "reason": "no shared cells"}
    po = collapse_to_seed(po_pairs, rung_cells=cells)
    mp = collapse_to_seed(mp_pairs, rung_cells=cells)
    shared = sorted(set(po) & set(mp))
    r = read_b4_bar({s: po[s] for s in shared}, {s: mp[s] for s in shared})
    return {**r, "rung": B4_RUNG, "cells": cells,
            "per_seed_pct": {s: 100.0 * (po[s] / mp[s] - 1.0) for s in shared}}


def format_b4(res: dict) -> str:
    if res.get("verdict") == V_UNREADABLE:
        return f"B4 -- vs the best pointwise arm: {res['verdict']} ({res.get('reason')})"
    lines = [f"B4 (AMENDMENT 2) -- peeronly_1670 vs mpoff_516 at {res['rung']}, ONE VALUE PER "
             f"CHECKPOINT, bar {res['bar']}  (positive = the pointwise arm is still ahead)",
             f"    {fmt_pair(res)}"]
    lines.append("    per checkpoint: " + "  ".join(f"s{s}:{v:+.1f}%" for s, v in sorted(res["per_seed_pct"].items())))
    return "\n".join(lines)


def format_b3(res: dict) -> str:
    if res.get("verdict") == V_UNREADABLE:
        return f"B3 -- checkpoint-level: {res['verdict']} ({res.get('reason')})"
    lines = [f"B3 (AMENDMENT 1) -- peeronly vs mpoff at {res['rung']}, ONE VALUE PER CHECKPOINT "
             f"(median over {len(res['cells'])} cells), bar {res['bar']}",
             f"    {fmt_pair(res)}"]
    per = res["per_seed_pct"]
    lines.append("    per checkpoint: " + "  ".join(f"s{s}:{v:+.1f}%" for s, v in sorted(per.items())))
    return "\n".join(lines)


def fmt_pair(r: dict) -> str:
    if r.get("verdict") == V_UNREADABLE:
        return f"{r['verdict']} ({r.get('reason')})"
    return f"median {r['median']:+6.2f}%  p={r['p']:.4f}  ahead {r.get('v3_ahead')}/{r['n']} -> {r['verdict']}"


def format_phase(res: dict, label: str) -> str:
    lines = [f"{label} -- corpus {res['corpus']}, peeronly vs mpoff paired by (cell, seed)"]
    for rung, r in res["rungs"].items():
        lines.append(f"  {rung}: n={r.get('n')}  d vs reactive: peeronly={r.get('d_peeronly')!s:>7.7} "
                     f"mpoff={r.get('d_mpoff')!s:>7.7} gnn={r.get('d_gnn')!s:>7.7}")
        lines.append(f"      A2 elapsed : {fmt_pair(r['a2'])}")
        a3 = r["a3"]
        if a3.get("verdict") == V_UNREADABLE:
            lines.append(f"      A3 queue   : {a3['verdict']}")
        else:
            lines.append(f"      A3 queue   : vs gnn median {a3['vs_gnn']['median']:+6.2f}% p={a3['vs_gnn']['p']:.4f}; "
                         f"vs mpoff median {a3['vs_mpoff']['median']:+6.2f}% p={a3['vs_mpoff']['p']:.4f} -> {a3['verdict']}")
        lines.append(f"      A4 peer    : {fmt_pair(r['a4'])}")
    lines.append(f"  HEADLINE: {res['headline']}")
    return "\n".join(lines)


def format_a0(res: dict) -> str:
    lines = [f"A0 -- re-served v3 arms vs partial_state_v3 P3 (tol {res['tol']}): {res['verdict']}"]
    for k, (a, b, d) in res["arms"].items():
        lines.append(f"    {k:32s} now={a:.4f} then={b:.4f} |d|={d:.5f}")
    return "\n".join(lines)


def format_b1(res: dict) -> str:
    lines = ["B1 -- corpus lever, 1670 vs 516 per arm, paired by (cell, seed)"]
    for arm, rungs in res.items():
        for rung, r in rungs.items():
            lines.append(f"    {arm:9s} {rung}: {fmt_pair(r)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--po-dir", required=True)
    ap.add_argument("--p3-dir", required=True)
    ap.add_argument("--phase", choices=["a", "b", "b3", "b5"], default="a")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    tab = tables(load(args.po_dir), load(args.p3_dir))
    res = {"a0": read_a0_from(tab)}
    print(format_a0(res["a0"]))
    res["phase_a"] = read_phase(tab, "516")
    print(); print(format_phase(res["phase_a"], "Phase A (A2/A3/A4)"))
    if args.phase in ("b", "b3"):
        res["b1"] = read_b1(tab); res["phase_b"] = read_phase(tab, "1670")
        print(); print(format_b1(res["b1"])); print(); print(format_phase(res["phase_b"], "Phase B (B2, A2-bars)"))
    if args.phase in ("b3", "b5"):
        res["b3"] = read_b3(tab); res["b4"] = read_b4(tab)
        print(); print(format_b3(res["b3"])); print(); print(format_b4(res["b4"]))
    if args.phase == "b5":
        res["b5"] = read_b5(tab)
        print(); print(format_b5(res["b5"]))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, default=str)
        print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

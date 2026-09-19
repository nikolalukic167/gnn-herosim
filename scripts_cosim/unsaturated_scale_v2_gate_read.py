"""unsaturated_scale_v2 -- the gate read. Bars live in `unsaturated_scale_v2_read.py`; this
file only finds the summaries, pairs the arms and prints.

    # M0: the screen -> the admissible map and the 4 study topologies
    python3 scripts_cosim/unsaturated_scale_v2_gate_read.py m0 \
        simulation_data/peer_affinity_live_gate/results/us_v2_screen \
        --write-selection simulation_data/unsaturated_scale_v2/selected.json

    # L: the study
    python3 scripts_cosim/unsaturated_scale_v2_gate_read.py study \
        simulation_data/peer_affinity_live_gate/results/us_v2_screen \
        simulation_data/peer_affinity_live_gate/results/us_v2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, pstdev
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.unsaturated_scale_v2_read import (  # noqa: E402
    M_ARMS, M_BASELINES, M_GNN, M_GRAPH, M_REACTIVE, M_TWIN, M_WINDOWS, V_UNREADABLE,
    checkpoint_stats, pair_checkpoint_stats, read_l1, read_l2, read_l3, read_l4, read_l5,
    read_l6, read_m0, select_topologies,
)

RELABEL = {"v3ext": "516"}


def _load(d: str, pattern: str) -> list:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pattern)))]


# --- M0 -------------------------------------------------------------------------------------

def screen_shares(screen_dir: str) -> Dict[tuple, Optional[float]]:
    out: Dict[tuple, Optional[float]] = {}
    for r in _load(screen_dir, "cs80s*__w*__reactive_s0.summary.json"):
        if int(r.get("num_tasks") or 0) != 50000:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; inadmissible",
                  file=sys.stderr)
            continue
        out[(int(r["topology"]), r["window"])] = float(r["queue_share"])
    return out


def report_m0(screen_dir: str, write_selection: Optional[str] = None) -> dict:
    shares = screen_shares(screen_dir)
    m0 = read_m0(shares)
    sel = select_topologies(m0)
    rows = m0["environments"]
    topos = sorted({t for t, _ in rows})
    print("M0 -- reactive queue share, 80 servers (admissible = share <= 0.80)\n")
    print(f"   {'topology':>9} " + " ".join(f"{w:>8}" for w in M_WINDOWS) + "   all 4?")
    for t in topos:
        cells = []
        for w in M_WINDOWS:
            qs = rows[(t, w)]["queue_share"]
            cells.append("     n/a" if qs is None else f"{qs:8.3f}")
        ok = all(rows[(t, w)]["admissible"] for w in M_WINDOWS)
        print(f"   {t:>9} " + " ".join(cells) + f"   {'YES' if ok else 'no'}")
    print(f"\n   admissible environments: {m0['n_admissible']} / {m0['n_screened']}")
    print(f"   SELECTION: {sel['verdict']}  topologies={sel.get('topologies')}")
    print(f"   {sel['why']}")
    if write_selection and sel["verdict"] != "TOO-FEW-UNSATURATED-ENVIRONMENTS":
        Path(write_selection).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"topologies": sel["topologies"], "windows": list(M_WINDOWS),
                   "environments": [list(e) for e in sel["environments"]],
                   "verdict": sel["verdict"],
                   "queue_share": {f"{t}_{w}": rows[(t, w)]["queue_share"]
                                   for t in sel["topologies"] for w in M_WINDOWS}},
                  open(write_selection, "w"), indent=1)
        print(f"   wrote {write_selection}")
    return {"m0": m0, "selection": sel}


# --- the study --------------------------------------------------------------------------------

def study_tables(study_dir: str, screen_dir: str):
    """label -> {(environment, seed): elapsed}; reactive comes from the screen."""
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    for r in _load(study_dir, "cs80s*__w*__*.summary.json"):
        if int(r.get("num_tasks") or 0) != 50000:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; dropped", file=sys.stderr)
            continue
        env = (int(r["topology"]), r["window"])
        kind = r["arm_kind"]
        if kind in M_BASELINES:
            arms[kind][(env, 0)] = float(r["averageElapsedTime"])
        else:
            corpus = RELABEL.get(r["corpus"], r["corpus"])
            arms[f"{corpus}_{kind}"][(env, int(r["checkpoint_seed"]))] = float(r["averageElapsedTime"])
    reactive = {(int(r["topology"]), r["window"]): float(r["averageElapsedTime"])
                for r in _load(screen_dir, "cs80s*__w*__reactive_s0.summary.json")
                if int(r.get("num_tasks") or 0) == 50000}
    return arms, reactive


def report_study(screen_dir: str, study_dir: str) -> dict:
    m0 = report_m0(screen_dir)
    sel = m0["selection"]
    if sel["verdict"] == "TOO-FEW-UNSATURATED-ENVIRONMENTS":
        raise SystemExit("FAIL LOUD: M0 did not produce a design; the study must not be read")
    envs = [tuple(e) for e in sel["environments"]]
    arms, reactive = study_tables(study_dir, screen_dir)
    missing = [e for e in envs if e not in reactive]
    if missing:
        raise SystemExit(f"FAIL LOUD: no reactive baseline for {missing}")
    print(f"\nL -- {len(envs)} environments ({sel['topologies']} x {list(M_WINDOWS)}), "
          f"reactive median {median(reactive[e] for e in envs):.3f} s")

    stats: Dict[str, Dict[int, float]] = {}
    l1: Dict[str, dict] = {}
    print(f"\n   {'arm':<18} {'L1 verdict':<42} {'median':>9} {'p':>8} {'ahead':>7} {'sd':>7}")
    for a in M_ARMS:
        try:
            stats[a] = checkpoint_stats(arms.get(a, {}), reactive, envs)
        except ValueError as e:
            l1[a] = {"verdict": V_UNREADABLE, "why": str(e)}
            print(f"   {a:<18} UNREADABLE  {e}")
            continue
        r = read_l1(stats[a])
        l1[a] = r
        if r["verdict"] == V_UNREADABLE:
            print(f"   {a:<18} UNREADABLE  n={r['n']}")
            continue
        print(f"   {a:<18} {r['verdict']:<42} {r['median']:+8.2f} % {r['p']:8.4f} "
              f"{r['ahead']:4d}/{r['n']:<3} {pstdev(list(stats[a].values())):6.2f}")

    l5 = read_l5(stats)
    print(f"\n   L5 {l5['verdict']}: {l5['why']}")
    out = {"m0": m0, "l1": l1, "l5": l5}

    for name, fn, a, b, first in (("L2", read_l2, M_GRAPH, M_TWIN, "graph"),
                                  ("L3", read_l3, M_GNN, M_GRAPH, "gnn")):
        if a in arms and b in arms:
            try:
                paired = pair_checkpoint_stats(arms[a], arms[b], envs)
            except ValueError as e:
                print(f"   {name} UNREADABLE: {e}")
                continue
            r = fn(paired)
            out[name.lower()] = r
            print(f"   {name} {a} vs {b} ({first} first): {r['verdict']}  "
                  f"median {r.get('median', float('nan')):+.2f} % p={r.get('p', float('nan')):.4f}")

    per_window = {}
    for a in M_ARMS:
        if a not in arms:
            continue
        pw = {}
        for w in M_WINDOWS:
            wenvs = [e for e in envs if e[1] == w]
            try:
                s = checkpoint_stats(arms[a], reactive, wenvs)
            except ValueError:
                continue
            pw[w] = median(s.values())
        per_window[a] = pw
    l6 = read_l6(per_window)
    out["l6"] = l6
    print(f"\n   L6 per-window median vs reactive (%):")
    print(f"   {'arm':<18} " + " ".join(f"{w:>8}" for w in M_WINDOWS) + "   verdict")
    for a in M_ARMS:
        r = l6["arms"].get(a, {})
        if r.get("verdict") == V_UNREADABLE:
            print(f"   {a:<18} UNREADABLE")
            continue
        pw = r.get("per_window", {})
        print(f"   {a:<18} " + " ".join(f"{pw.get(w, float('nan')):+8.2f}" for w in M_WINDOWS)
              + f"   {r.get('verdict')}")
    print(f"   {l6['why']}")

    print("\n   baselines (median elapsed over the study environments, s):")
    for b in M_BASELINES:
        if b == M_REACTIVE:
            print(f"     {b:<22} {median(reactive[e] for e in envs):8.3f}   (the reference)")
        elif b in arms:
            vals = [v for (e, _s), v in arms[b].items() if e in envs]
            if vals:
                rk = median(reactive[e] for e in envs)
                print(f"     {b:<22} {median(vals):8.3f}   {100*(median(vals)/rk-1):+7.1f} % vs reactive")

    l4 = read_l4(l1, l5)
    out["l4"] = l4
    print(f"\n   L4: {l4['verdict']}")
    print(f"     winners={l4.get('winners')} losers={l4.get('losers')} missing={l4.get('missing')}"
          f" interpretable={l4.get('interpretable', True)}")
    print(f"     {l4.get('why', '')}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("m0", "study"))
    ap.add_argument("screen_dir")
    ap.add_argument("study_dir", nargs="?")
    ap.add_argument("--write-selection")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.step == "m0":
        res = report_m0(a.screen_dir, a.write_selection)
    else:
        if not a.study_dir:
            ap.error("study needs study_dir")
        res = report_study(a.screen_dir, a.study_dir)
    if a.json:
        # Environment keys are (topology, window) tuples, which json refuses. Stringify keys
        # rather than dropping them -- the map of which environment was admissible is the M0
        # record, not a debug aid.
        def _keys(o):
            if isinstance(o, dict):
                return {("_".join(map(str, k)) if isinstance(k, tuple) else str(k)): _keys(v)
                        for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_keys(v) for v in o]
            return o
        json.dump(_keys(res), open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

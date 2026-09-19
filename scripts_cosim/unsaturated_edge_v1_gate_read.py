"""unsaturated_edge_v1 -- the gate read. Bars live in `unsaturated_edge_v1_read.py`; this file
only finds the summaries, pairs the arms and prints.

    # M0: the screen -> the admissible map and the 4 study topologies PER RUNG
    python3 scripts_cosim/unsaturated_edge_v1_gate_read.py m0 \
        simulation_data/peer_affinity_live_gate/results/ue_v1_screen \
        --write-selection simulation_data/unsaturated_edge_v1/selected.json

    # E: the study
    python3 scripts_cosim/unsaturated_edge_v1_gate_read.py study \
        simulation_data/peer_affinity_live_gate/results/ue_v1_screen \
        simulation_data/peer_affinity_live_gate/results/ue_v1
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

from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_ARMS_BY_RUNG, E_BASELINES, E_GNN, E_GRAPH, E_RANDOM, E_REACTIVE, E_RUNGS, E_TWIN,
    E_WINDOWS, V_DESIGN_READY, V_UNREADABLE, checkpoint_stats, pair_checkpoint_stats, read_e1,
    read_e2, read_e3, read_e4, read_e5, read_e6, read_e7, read_m0, select_topologies,
)

RELABEL = {"v3ext": "516"}


def _load(d: str, pattern: str) -> list:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pattern)))]


def _env(r) -> tuple:
    return (int(r["clients"]), int(r["topology"]), r["window"])


# --- M0 -------------------------------------------------------------------------------------

def screen_shares(screen_dir: str) -> Dict[tuple, Optional[float]]:
    out: Dict[tuple, Optional[float]] = {}
    for r in _load(screen_dir, "cc*__w*__reactive_s0.summary.json"):
        if int(r.get("num_tasks") or 0) != 50000:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; inadmissible", file=sys.stderr)
            continue
        out[_env(r)] = float(r["queue_share"])
    return out


def report_m0(screen_dir: str, write_selection: Optional[str] = None) -> dict:
    m0 = read_m0(screen_shares(screen_dir))
    rows = m0["environments"]
    sel = {nc: select_topologies(m0, nc) for nc in E_RUNGS}
    print("M0 -- reactive queue share, 6 servers (admissible = share <= 0.80)")
    for nc in E_RUNGS:
        topos = sorted({t for (r, t, _) in rows if r == nc})
        print(f"\n   C{nc}  {'topology':>9} " + " ".join(f"{w:>8}" for w in E_WINDOWS) + "   all 4?")
        for t in topos:
            cells = []
            for w in E_WINDOWS:
                qs = rows[(nc, t, w)]["queue_share"]
                cells.append("     n/a" if qs is None else f"{qs:8.3f}")
            ok = all(rows[(nc, t, w)]["admissible"] for w in E_WINDOWS)
            print(f"        {t:>9} " + " ".join(cells) + f"   {'YES' if ok else 'no'}")
        print(f"        SELECTION: {sel[nc]['verdict']}  topologies={sel[nc].get('topologies')}")
        print(f"        {sel[nc]['why']}")
    print(f"\n   admissible environments: {m0['n_admissible']} / {m0['n_screened']}")
    if write_selection and all(s["verdict"] == V_DESIGN_READY for s in sel.values()):
        Path(write_selection).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"verdict": V_DESIGN_READY, "windows": list(E_WINDOWS),
                   "rungs": {str(nc): {"topologies": sel[nc]["topologies"],
                                       "environments": [list(e) for e in sel[nc]["environments"]],
                                       "queue_share": {f"{t}_{w}": rows[(nc, t, w)]["queue_share"]
                                                       for t in sel[nc]["topologies"] for w in E_WINDOWS}}
                             for nc in E_RUNGS}},
                  open(write_selection, "w"), indent=1)
        print(f"   wrote {write_selection}")
    elif write_selection:
        print("   NOT writing a selection: a rung is not DESIGN-READY", file=sys.stderr)
    return {"m0": m0, "selection": sel}


# --- the study --------------------------------------------------------------------------------

def study_tables(study_dir: str, screen_dir: str):
    """label -> {(environment, seed): elapsed}; reactive comes from the screen."""
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    for r in _load(study_dir, "cc*__w*__*.summary.json"):
        if int(r.get("num_tasks") or 0) != 50000:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; dropped", file=sys.stderr)
            continue
        kind = r["arm_kind"]
        if kind in E_BASELINES:
            arms[kind][(_env(r), 0)] = float(r["averageElapsedTime"])
        else:
            corpus = RELABEL.get(r["corpus"], r["corpus"])
            arms[f"{corpus}_{kind}"][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageElapsedTime"])
    reactive = {_env(r): float(r["averageElapsedTime"])
                for r in _load(screen_dir, "cc*__w*__reactive_s0.summary.json")
                if int(r.get("num_tasks") or 0) == 50000}
    return arms, reactive


def _print_row(label, r):
    if r.get("verdict") == V_UNREADABLE:
        print(f"   {label:<18} UNREADABLE  {r.get('reason') or r.get('why') or ''}")
    else:
        print(f"   {label:<18} {r['verdict']:<40} {r['median']:+8.2f} % {r['p']:8.4f} "
              f"{r['ahead']:4d}/{r['n']:<3}")


def report_study(screen_dir: str, study_dir: str) -> dict:
    m0 = report_m0(screen_dir)
    sel = m0["selection"]
    if any(s["verdict"] != V_DESIGN_READY for s in sel.values()):
        raise SystemExit("FAIL LOUD: M0 did not produce a design on every rung; the study must not be read")
    arms, reactive = study_tables(study_dir, screen_dir)
    out = {"m0": m0, "rungs": {}}
    e1_by_rung, e5_by_rung = {}, {}
    for nc in E_RUNGS:
        envs = [tuple(e) for e in sel[nc]["environments"]]
        missing = [e for e in envs if e not in reactive]
        if missing:
            raise SystemExit(f"FAIL LOUD: no reactive baseline for {missing}")
        rk = median(reactive[e] for e in envs)
        print(f"\n=== C{nc} -- 16 environments ({sel[nc]['topologies']} x {list(E_WINDOWS)}), "
              f"reactive median {rk:.3f} s ===")
        stats, e1 = {}, {}
        print(f"\n   {'arm':<18} {'E1 vs reactive':<40} {'median':>9} {'p':>8} {'ahead':>7}   sd")
        for a in E_ARMS_BY_RUNG[nc]:
            try:
                stats[a] = checkpoint_stats(arms.get(a, {}), reactive, envs)
            except ValueError as e:
                e1[a] = {"verdict": V_UNREADABLE, "why": str(e)}
                _print_row(a, e1[a]); continue
            e1[a] = read_e1(stats[a])
            _print_row(a, e1[a])
            if e1[a]["verdict"] != V_UNREADABLE:
                print(f"{'':>90}{pstdev(list(stats[a].values())):6.2f}")
        e5 = read_e5(stats, nc)
        print(f"\n   E5 {e5['verdict']}: {e5['why']}")
        e1_by_rung[nc], e5_by_rung[nc] = e1, e5
        rung_out = {"e1": e1, "e5": e5, "environments": envs}

        # E4: every registered arm vs random, paired on the environment
        random_by_env = {e: v for (e, _s), v in arms.get(E_RANDOM, {}).items()}
        print(f"\n   {'arm':<18} {'E4 vs random':<40} {'median':>9} {'p':>8} {'ahead':>7}")
        e4 = {}
        for a in E_ARMS_BY_RUNG[nc]:
            try:
                e4[a] = read_e4(checkpoint_stats(arms.get(a, {}), random_by_env, envs))
            except ValueError as e:
                e4[a] = {"verdict": V_UNREADABLE, "why": str(e)}
            _print_row(a, e4[a])
        rung_out["e4"] = e4

        if nc == 40:
            for name, fn, a, b in (("E2", read_e2, E_GRAPH, E_TWIN), ("E3", read_e3, E_GNN, E_GRAPH)):
                if a in arms and b in arms:
                    try:
                        r = fn(pair_checkpoint_stats(arms[a], arms[b], envs))
                    except ValueError as e:
                        print(f"   {name} UNREADABLE: {e}"); continue
                    rung_out[name.lower()] = r
                    print(f"   {name} {a} vs {b}: {r['verdict']}  median {r.get('median', float('nan')):+.2f} % "
                          f"p={r.get('p', float('nan')):.4f} {r.get('ahead')}/{r.get('n')}")

        per_window = {}
        for a in E_ARMS_BY_RUNG[nc]:
            if a not in arms:
                continue
            pw = {}
            for w in E_WINDOWS:
                try:
                    pw[w] = median(checkpoint_stats(arms[a], reactive, [e for e in envs if e[2] == w]).values())
                except ValueError:
                    continue
            per_window[a] = pw
        e6 = read_e6(per_window, nc)
        rung_out["e6"] = e6
        print(f"\n   E6 per-window median vs reactive (%):")
        print(f"   {'arm':<18} " + " ".join(f"{w:>8}" for w in E_WINDOWS) + "   verdict")
        for a in E_ARMS_BY_RUNG[nc]:
            r = e6["arms"].get(a, {})
            if r.get("verdict") == V_UNREADABLE:
                print(f"   {a:<18} UNREADABLE"); continue
            pw = r.get("per_window", {})
            print(f"   {a:<18} " + " ".join(f"{pw.get(w, float('nan')):+8.2f}" for w in E_WINDOWS)
                  + f"   {r.get('verdict')}")
        print(f"   {e6['why']}")

        print("\n   baselines (median elapsed over the study environments, s):")
        for b in E_BASELINES:
            if b == E_REACTIVE:
                print(f"     {b:<22} {rk:8.3f}   (the reference)")
            elif b in arms:
                vals = [v for (e, _s), v in arms[b].items() if e in envs]
                if vals:
                    print(f"     {b:<22} {median(vals):8.3f}   {100*(median(vals)/rk-1):+7.1f} % vs reactive")
        for a in E_ARMS_BY_RUNG[nc]:
            if a in arms:
                vals = [v for (e, _s), v in arms[a].items() if e in envs]
                print(f"     {a:<22} {median(vals):8.3f}   {100*(median(vals)/rk-1):+7.1f} % vs reactive (raw)")
        out["rungs"][nc] = rung_out

    e7 = read_e7(e1_by_rung, e5_by_rung)
    out["e7"] = e7
    print(f"\n   E7: {e7['verdict']}  interpretable={e7.get('interpretable', True)}")
    print(f"     {e7.get('why', '')}")
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

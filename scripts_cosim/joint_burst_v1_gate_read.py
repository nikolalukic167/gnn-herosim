"""joint_burst_v1 -- the reads. Bars live in `joint_burst_v1_read.py`.

    # J0a: the pool's screen -> the study selection (both serving paths must finish)
    python3 scripts_cosim/joint_burst_v1_gate_read.py screen <screen_dir> \
        --write-selection simulation_data/joint_burst_v1/selected.json \
        --capture-dir simulation_data/joint_burst_v1

    # the gate
    python3 scripts_cosim/joint_burst_v1_gate_read.py study <screen_dir> <gate_dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.joint_burst_v1_read import (  # noqa: E402
    J_BATCHED, J_GRAPH_BT, J_GRAPH_COLD, J_IMMEDIATE, J_RANDOM, J_REACTIVE, J_RUNG,
    J_STUDY_CANDIDATES, J_TRAIN_SEEDS, J_TWIN_BT, J_WINDOWS, V_DESIGN_READY, broadcast_rule,
    checkpoint_stats, env_stats, pair_checkpoint_stats, read_j0, read_j1, read_j2, read_j3,
    read_j4, read_j5, read_j6, select_burst_topologies,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N_TASKS = 50000


def _load(d: str, pattern: str) -> list:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pattern)))]


def _env(r) -> tuple:
    return (int(r["clients"]), int(r["topology"]), r["window"])


def _per_task(r: dict) -> dict:
    n = float(r["num_tasks"])
    return {"elapsed": float(r["averageElapsedTime"]), "wait": float(r["averageWaitTime"] or 0.0),
            "queue": float(r["averageQueueTime"]), "exchange": float(r["totalPeerExchangeTime"]) / n,
            "rendezvous": float(r["totalPeerRendezvousWait"]) / n}


def _label(r: dict) -> str:
    kind = r["arm_kind"]
    if kind == "reactive":
        return J_REACTIVE
    if kind == "batched":
        return J_BATCHED
    if kind in (J_REACTIVE, J_RANDOM, J_IMMEDIATE, J_BATCHED):
        return kind
    return f"{r.get('corpus')}_{kind}"


def tables(*dirs: str):
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    rows: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    shares: Dict[tuple, Optional[float]] = {}
    done: Dict[tuple, bool] = {}
    for d in dirs:
        for r in _load(d, "cc*__w*__*.summary.json"):
            if int(r.get("num_tasks") or 0) != N_TASKS:
                continue
            label = _label(r); seed = int(r.get("checkpoint_seed") or 0); e = _env(r)
            arms[label][(e, seed)] = float(r["averageElapsedTime"])
            rows[label][(e, seed)] = _per_task(r)
            if label == J_REACTIVE:
                shares[e] = float(r["queue_share"])
            if label == J_BATCHED:
                done[e] = True
    return arms, rows, shares, done


def report_screen(screen_dir: str, write_selection: Optional[str], capture_dir: Optional[str]) -> dict:
    _a, _r, shares, done = tables(screen_dir)
    sel = select_burst_topologies(shares, done)
    print(f"=== joint_burst_v1 screen at C{J_RUNG}: {sel['verdict']} ===")
    print(f"   {'topology':>9} " + " ".join(f"{w:>14}" for w in J_WINDOWS) + "   (reactive share / batch path)")
    for t in J_STUDY_CANDIDATES:
        cells = []
        for w in J_WINDOWS:
            s = shares.get((J_RUNG, t, w)); b = done.get((J_RUNG, t, w), False)
            cells.append(f"{'hang' if s is None else f'{s:.3f}':>8} {'ok' if b else 'spin':>5}")
        print(f"   {t:>9} " + " ".join(cells))
    print(f"   {sel.get('why')}")
    n_cells, n_snap = 0, 0
    if capture_dir:
        for s in J_TRAIN_SEEDS:
            for w in ("w0", "w1"):
                f = os.path.join(capture_dir, "capture_summaries", f"cc40s{s}_{w}.summary.json")
                if os.path.exists(f):
                    c = json.load(open(f)); n_snap += int(c.get("snapshots") or 0); n_cells += 1
        print(f"   capture: {n_cells}/48 training runs finished, {n_snap} snapshots")
    out = {"verdict": sel["verdict"], "topologies": sel.get("topologies"), "environments": sel.get("environments"),
           "qualified": sel.get("qualified"), "capture_runs": n_cells, "snapshots": n_snap,
           "queue_share": {f"{t}_{w}": shares.get((J_RUNG, t, w)) for t in J_STUDY_CANDIDATES for w in J_WINDOWS},
           "batch_path_done": {f"{t}_{w}": done.get((J_RUNG, t, w), False) for t in J_STUDY_CANDIDATES for w in J_WINDOWS}}
    if write_selection:
        os.makedirs(os.path.dirname(write_selection) or ".", exist_ok=True)
        json.dump(out, open(write_selection, "w"), indent=1)
        print(f"   wrote {write_selection}")
    return out


def _print_row(label, r):
    if r.get("verdict") == V_UNREADABLE:
        print(f"   {label:<40} UNREADABLE  {r.get('reason') or r.get('why') or ''}")
    else:
        print(f"   {label:<40} {r['verdict']:<44} {r['median']:+8.2f} % {r['p']:8.4f} {r['ahead']:4d}/{r['n']:<3}")


def _complete(arm: Dict[tuple, float], envs) -> set:
    return {s for s in {s for (_e, s) in arm} if all((e, s) in arm for e in envs)}


def _pair(a: Dict[tuple, float], b: Dict[tuple, float], envs, fn, label):
    """Registered on every checkpoint; disclosed on the checkpoints complete in both."""
    try:
        r = fn(pair_checkpoint_stats(a, b, envs))
    except ValueError as ex:
        r = {"verdict": V_UNREADABLE, "reason": str(ex)}
    if r["verdict"] == V_UNREADABLE:
        comp = _complete(a, envs) & _complete(b, envs)
        if comp:
            d = fn(pair_checkpoint_stats({k: v for k, v in a.items() if k[1] in comp},
                                         {k: v for k, v in b.items() if k[1] in comp}, envs), min_n=len(comp))
            d["disclosed"] = True; d["n_complete"] = len(comp); r["disclosed"] = d
    _print_row(label, r)
    if "disclosed" in r:
        _print_row(f"   disclosed ({r['disclosed']['n_complete']} complete ckpts)", r["disclosed"])
    return r


def report_study(screen_dir: str, gate_dir: str, selection_path: str = "simulation_data/joint_burst_v1/selected.json") -> dict:
    sel = json.load(open(selection_path))
    if sel.get("verdict") != V_DESIGN_READY:
        raise SystemExit(f"FAIL LOUD: {selection_path} verdict {sel.get('verdict')!r}")
    envs = [tuple(e) for e in sel["environments"]]
    arms, rows, _s, _d = tables(screen_dir, gate_dir)
    react = {e: v for (e, s), v in arms[J_REACTIVE].items() if s == 0}
    missing = [e for e in envs if e not in react]
    if missing:
        raise SystemExit(f"FAIL LOUD: no reactive baseline for {missing}")
    rules = {a: {e: v for (e, s), v in arms.get(a, {}).items()} for a in (J_IMMEDIATE, J_BATCHED, J_RANDOM)}
    seeds = sorted({s for (_e, s) in arms.get(J_GRAPH_BT, {})}) or list(range(1, 17))
    print(f"\n=== joint_burst_v1 -- {len(envs)} environments ({sel['topologies']} x {list(J_WINDOWS)}), "
          f"reactive median {median(react[e] for e in envs):.3f} s ===")
    for a in (J_GRAPH_BT, J_TWIN_BT, J_GRAPH_COLD, J_IMMEDIATE, J_BATCHED, J_RANDOM):
        n = sum(1 for k in arms.get(a, {}) if k[0] in envs)
        print(f"   [{a}] {n} arms on disk")
    print(f"\n   {'read':<40} {'verdict':<44} {'median':>9} {'p':>8} {'ahead':>7}")
    R: dict = {"environments": envs}
    g = arms.get(J_GRAPH_BT, {})
    R["j1"] = _pair(g, broadcast_rule(rules[J_BATCHED], seeds), envs, read_j1, "J1 bt-gnnedge0 vs batched greedy")
    R["j2"] = _pair(g, broadcast_rule(rules[J_IMMEDIATE], seeds), envs, read_j2, "J2 bt-gnnedge0 vs immediate rule")

    def _ckpt(a, fn, label):
        try:
            r = fn(checkpoint_stats(arms.get(a, {}), react, envs))
        except ValueError as ex:
            r = {"verdict": V_UNREADABLE, "reason": str(ex)}
        if r["verdict"] == V_UNREADABLE:
            comp = _complete(arms.get(a, {}), envs)
            if comp:
                d = fn(checkpoint_stats({k: v for k, v in arms[a].items() if k[1] in comp}, react, envs), min_n=len(comp))
                d["disclosed"] = True; d["n_complete"] = len(comp); r["disclosed"] = d
        _print_row(label, r)
        if "disclosed" in r:
            _print_row(f"   disclosed ({r['disclosed']['n_complete']} complete ckpts)", r["disclosed"])
        return r

    R["j3"] = _ckpt(J_GRAPH_BT, read_j3, "J3 bt-gnnedge0 vs reactive")
    R["j3_twin"] = _ckpt(J_TWIN_BT, read_j3, "   bt-mpoff vs reactive")
    R["j3_cold"] = _ckpt(J_GRAPH_COLD, read_j3, "   cold gnnedge0 vs reactive")
    R["j4"] = _pair(g, arms.get(J_TWIN_BT, {}), envs, read_j4, "J4 bt-gnnedge0 vs bt-mpoff")
    R["j5"] = _pair(g, arms.get(J_GRAPH_COLD, {}), envs, read_j5, "J5 bt-gnnedge0 vs cold gnnedge0")
    for a, lab in ((J_IMMEDIATE, "   immediate rule vs reactive (env)"), (J_BATCHED, "   batched greedy vs reactive (env)"),
                   (J_RANDOM, "   random vs reactive (env)")):
        try:
            from scripts_cosim.unsaturated_scale_v2_read import _one_sample
            st = env_stats(rules[a], react, envs)
            R[f"env_{a}"] = _one_sample(st, min_n=len(st), faster_a="RULE-BEATS-REACTIVE", faster_b="REACTIVE-FASTER", tie="NOT-SEPARATED")
        except ValueError as ex:
            R[f"env_{a}"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
        _print_row(lab, R[f"env_{a}"])
    R["j6"] = read_j6(R["j1"] if R["j1"]["verdict"] != V_UNREADABLE else R["j1"].get("disclosed", R["j1"]),
                      R["j3"] if R["j3"]["verdict"] != V_UNREADABLE else R["j3"].get("disclosed", R["j3"]))
    print(f"\n   J6 {R['j6']['verdict']}  (J1 {R['j6'].get('j1')}, J3 {R['j6'].get('j3')})"
          + ("  [composed on disclosed reads]" if R["j1"]["verdict"] == V_UNREADABLE or R["j3"]["verdict"] == V_UNREADABLE else ""))
    print(f"\n   per-task decomposition (medians over environments and checkpoints, s):")
    print(f"   {'arm':<28} {'wait':>7} {'queue':>7} {'exch':>7} {'rendez':>7} {'elapsed':>8}")
    R["decomposition"] = {}
    for a in (J_REACTIVE, J_RANDOM, J_IMMEDIATE, J_BATCHED, J_GRAPH_COLD, J_TWIN_BT, J_GRAPH_BT):
        ks = [k for k in rows.get(a, {}) if k[0] in envs]
        if not ks:
            continue
        m = {c: median(rows[a][k][c] for k in ks) for c in ("wait", "queue", "exchange", "rendezvous", "elapsed")}
        R["decomposition"][a] = m
        print(f"   {a:<28} {m['wait']:7.2f} {m['queue']:7.2f} {m['exchange']:7.2f} {m['rendezvous']:7.2f} {m['elapsed']:8.2f}")
    return R


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "study"))
    ap.add_argument("screen_dir")
    ap.add_argument("gate_dir", nargs="?")
    ap.add_argument("--write-selection")
    ap.add_argument("--capture-dir")
    ap.add_argument("--selection", default="simulation_data/joint_burst_v1/selected.json")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.step == "screen":
        res = report_screen(a.screen_dir, a.write_selection, a.capture_dir)
    else:
        if not a.gate_dir:
            ap.error("study needs gate_dir")
        res = report_study(a.screen_dir, a.gate_dir, a.selection)
    if a.json:
        def _keys(o):
            if isinstance(o, dict):
                return {("_".join(map(str, k)) if isinstance(k, tuple) else str(k)): _keys(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_keys(v) for v in o]
            return o
        json.dump(_keys(res), open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

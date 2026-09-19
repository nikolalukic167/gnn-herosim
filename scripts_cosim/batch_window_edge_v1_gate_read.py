"""batch_window_edge_v1 -- the gate read. Bars live in `batch_window_edge_v1_read.py`.

    G=simulation_data/peer_affinity_live_gate/results
    # the screen -> the chosen window (or WINDOW-NOT-THE-LEVER)
    python3 scripts_cosim/batch_window_edge_v1_gate_read.py screen $G/ue_v1_screen $G/ue_v1 $G/bw_v1 \
        --write-selection simulation_data/batch_window_edge_v1/selected_window.json
    # the held-out confirmation
    python3 scripts_cosim/batch_window_edge_v1_gate_read.py confirm $G/ue_v1_screen $G/ue_v1 $G/bw_v1
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

from scripts_cosim.batch_window_edge_v1_read import (  # noqa: E402
    V_WINDOW_SELECTED, W_BASE_WINDOW_S, W_GRAPH, W_RUNG, W_TWIN, checkpoint_stats,
    pair_checkpoint_stats, read_w1, read_w2, read_w3, read_w4, read_w5, read_w6, select_window,
    split_environments,
)
from scripts_cosim.unsaturated_edge_v1_read import E_RANDOM, E_WINDOWS, V_UNREADABLE  # noqa: E402

SELECTION = "simulation_data/unsaturated_edge_v1/selected.json"


def _load(d: str, pattern: str) -> list:
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, pattern)))]


def _env(r) -> tuple:
    return (int(r["clients"]), int(r["topology"]), r["window"])


def tables(screen_dir: str, study_dir: str, bw_dir: str):
    """reactive/random by env; elapsed and wait by (window_s, label) -> {(env, seed): value}."""
    reactive = {_env(r): float(r["averageElapsedTime"])
                for r in _load(screen_dir, "cc40*__w*__reactive_s0.summary.json")
                if int(r.get("num_tasks") or 0) == 50000}
    random_ = {_env(r): float(r["averageElapsedTime"])
               for r in _load(study_dir, "cc40*__w*__random_network_s0.summary.json")
               if int(r.get("num_tasks") or 0) == 50000}
    elapsed: Dict[tuple, Dict[tuple, float]] = defaultdict(dict)
    wait: Dict[tuple, Dict[tuple, float]] = defaultdict(dict)
    for r in _load(study_dir, "cc40*__w*__*_s*.summary.json"):       # the 16 s incumbents
        if r.get("policy_name") != "gnn" or int(r.get("num_tasks") or 0) != 50000:
            continue
        key = (W_BASE_WINDOW_S, f"{r['corpus']}_{r['arm_kind']}")
        elapsed[key][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageElapsedTime"])
        wait[key][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageWaitTime"] or 0)
    for r in _load(bw_dir, "cc40*__w*__bt*__*_s*.summary.json"):
        if int(r.get("num_tasks") or 0) != 50000:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; dropped", file=sys.stderr)
            continue
        key = (float(r["batch_timeout_s"]), f"{r['corpus']}_{r['arm_kind']}")
        elapsed[key][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageElapsedTime"])
        wait[key][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageWaitTime"] or 0)
    return reactive, random_, elapsed, wait


def _split():
    sel = json.load(open(SELECTION))
    if sel.get("verdict") != "DESIGN-READY":
        raise SystemExit("FAIL LOUD: unsaturated_edge_v1's selection is not DESIGN-READY")
    return split_environments(sel["rungs"][str(W_RUNG)]["topologies"])


def _row(label, r):
    if r.get("verdict") == V_UNREADABLE:
        print(f"   {label:<34} UNREADABLE  {r.get('reason') or r.get('why') or ''}")
    else:
        print(f"   {label:<34} {r['verdict']:<44} {r['median']:+8.2f} % {r['p']:8.4f} {r['ahead']:3d}/{r['n']}")


def report_screen(screen_dir, study_dir, bw_dir, write_selection=None) -> dict:
    sp = _split()
    reactive, _rand, elapsed, wait = tables(screen_dir, study_dir, bw_dir)
    envs = sp["screen"]
    print(f"SCREEN -- gnnedge0 vs reactive on topology {sp['screen_topologies']} x {list(E_WINDOWS)}")
    stats, waits = {}, {}
    for w in sorted({w for (w, a) in elapsed if a == W_GRAPH}):
        try:
            stats[w] = checkpoint_stats(elapsed[(w, W_GRAPH)], reactive, envs)
        except ValueError as e:
            print(f"   {w:>5g} s  UNREADABLE: {e}"); continue
        waits[w] = median(v for (e, _s), v in wait[(w, W_GRAPH)].items() if e in envs)
        print(f"   {w:>5g} s  median {median(stats[w].values()):+7.2f} %  n={len(stats[w])}  "
              f"batch wait {waits[w]:.3f} s")
    sel = select_window(stats)
    print(f"\n   {sel['verdict']}: {sel['why']}")
    out = {"split": sp, "screen": {str(w): median(s.values()) for w, s in stats.items()},
           "batch_wait": {str(w): v for w, v in waits.items()}, "selection": sel}
    if write_selection and sel["verdict"] != V_UNREADABLE:
        Path(write_selection).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"verdict": sel["verdict"], "window_s": sel.get("window_s"),
                   "screen_medians": {str(w): m for w, m in sel["medians"].items()},
                   "screen_topologies": sp["screen_topologies"],
                   "held_out_topologies": sp["held_out_topologies"]},
                  open(write_selection, "w"), indent=1)
        print(f"   wrote {write_selection}")
    return out


def report_confirm(screen_dir, study_dir, bw_dir) -> dict:
    sp = _split()
    wsel = json.load(open("simulation_data/batch_window_edge_v1/selected_window.json"))
    if wsel.get("verdict") != V_WINDOW_SELECTED:
        raise SystemExit(f"FAIL LOUD: screen verdict {wsel.get('verdict')!r}; there is nothing to confirm")
    bt = float(wsel["window_s"])
    reactive, random_, elapsed, wait = tables(screen_dir, study_dir, bw_dir)
    envs = sp["held_out"]
    rk = median(reactive[e] for e in envs)
    print(f"CONFIRM -- window {bt:g} s on the 12 HELD-OUT environments "
          f"(topologies {sp['held_out_topologies']}), reactive median {rk:.3f} s\n")
    out = {"window_s": bt, "held_out": envs}
    g_short, g_base, t_short = elapsed.get((bt, W_GRAPH), {}), elapsed.get((W_BASE_WINDOW_S, W_GRAPH), {}), elapsed.get((bt, W_TWIN), {})

    def _try(fn, *a):
        try:
            return fn(*a)
        except ValueError as e:
            return {"verdict": V_UNREADABLE, "why": str(e)}

    out["w1"] = _try(lambda: read_w1(checkpoint_stats(g_short, reactive, envs)))
    _row(f"W1 gnnedge0@{bt:g}s vs reactive", out["w1"])
    out["w1_base"] = _try(lambda: read_w1(checkpoint_stats(g_base, reactive, envs)))
    _row("    (gnnedge0@16s vs reactive, same 12)", out["w1_base"])
    out["w2"] = _try(lambda: read_w2(pair_checkpoint_stats(g_short, g_base, envs)))
    _row(f"W2 gnnedge0@{bt:g}s vs gnnedge0@16s", out["w2"])
    out["w3"] = _try(lambda: read_w3(pair_checkpoint_stats(g_short, t_short, envs)))
    _row(f"W3 gnnedge0@{bt:g}s vs mpoff@{bt:g}s", out["w3"])
    out["w3_twin_vs_reactive"] = _try(lambda: read_w1(checkpoint_stats(t_short, reactive, envs)))
    _row(f"    (mpoff@{bt:g}s vs reactive)", out["w3_twin_vs_reactive"])
    out["w4"] = _try(lambda: read_w4(checkpoint_stats(g_short, random_, envs)))
    _row(f"W4 gnnedge0@{bt:g}s vs random", out["w4"])

    per = {}
    for w in E_WINDOWS:
        try:
            per[w] = median(checkpoint_stats(g_short, reactive, [e for e in envs if e[2] == w]).values())
        except ValueError:
            pass
    out["w5"] = read_w5(per)
    print(f"\n   W5 per-window median vs reactive: " + " ".join(f"{w} {per.get(w, float('nan')):+.2f}" for w in E_WINDOWS)
          + f"   {out['w5']['verdict']}")
    ws = {k: v for k, v in wait.get((bt, W_GRAPH), {}).items() if k[0] in envs}
    wb = {k: v for k, v in wait.get((W_BASE_WINDOW_S, W_GRAPH), {}).items() if k[0] in envs}
    out["w6"] = read_w6(ws, wb)
    print(f"   W6 {out['w6']['verdict']}: batch wait {out['w6'].get('median_wait_short_s', float('nan')):.3f} s "
          f"at {bt:g} s vs {out['w6'].get('median_wait_base_s', float('nan')):.3f} s at 16 s")
    for label, tab in ((f"gnnedge0@{bt:g}s", g_short), ("gnnedge0@16s", g_base), (f"mpoff@{bt:g}s", t_short)):
        vals = [v for (e, _s), v in tab.items() if e in envs]
        if vals:
            print(f"     {label:<16} median elapsed {median(vals):8.3f} s  ({100*(median(vals)/rk-1):+6.1f} % vs reactive, raw)")
    rv = [v for e, v in random_.items() if e in envs]
    if rv:
        print(f"     {'random':<16} median elapsed {median(rv):8.3f} s  ({100*(median(rv)/rk-1):+6.1f} % vs reactive)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "confirm"))
    ap.add_argument("screen_dir"); ap.add_argument("study_dir"); ap.add_argument("bw_dir")
    ap.add_argument("--write-selection"); ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = (report_screen(a.screen_dir, a.study_dir, a.bw_dir, a.write_selection) if a.step == "screen"
           else report_confirm(a.screen_dir, a.study_dir, a.bw_dir))
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

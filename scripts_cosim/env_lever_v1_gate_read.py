"""env_lever_v1 -- the gate read for burst_groups_v1 / payload_scale_v1 / backbone_sparsity_v1.
Bars live in `env_lever_v1_read.py`; this file finds the summaries, pairs the arms and prints.

    # the lever's own screen -> its 4 study topologies
    python3 scripts_cosim/env_lever_v1_gate_read.py screen <lever> \
        simulation_data/peer_affinity_live_gate/results/el_v1/<lever> \
        --write-selection simulation_data/env_lever_v1/selected_<lever>.json

    # the study
    python3 scripts_cosim/env_lever_v1_gate_read.py study <lever> \
        simulation_data/peer_affinity_live_gate/results/el_v1/<lever> [--json out.json]

    # payload_scale_v1's regime read across pk0.1 / the study (ue_v1, pg_v1) / pk10
    python3 scripts_cosim/env_lever_v1_gate_read.py regime \
        simulation_data/peer_affinity_live_gate/results [--json out.json]
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

from scripts_cosim.env_lever_v1_read import (  # noqa: E402
    L_ARMS, L_BATCHED, L_GRAPH, L_IMMEDIATE, L_LEVERS, L_LINEAGE_OF, L_PAYLOAD_SCALES, L_RANDOM,
    L_REACTIVE, L_RUNG, L_TWIN, L_WINDOWS, V_DESIGN_READY, V_RULE_BEATS_REACTIVE,
    V_ARM_BEATS_REACTIVE, broadcast_rule, checkpoint_stats, env_stats, pair_checkpoint_stats,
    read_l0, read_l1, read_l2, read_l3, read_l4, read_l5, read_l6,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import E_TOPOLOGY_CANDIDATES  # noqa: E402

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
    if kind in (L_REACTIVE, L_RANDOM, L_IMMEDIATE, L_BATCHED, "knative_network_ect", "reactive"):
        return L_REACTIVE if kind == "reactive" else kind
    return f"{r['corpus']}_{kind}"


def tables(result_dir: str):
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    rows: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    shares: Dict[tuple, Optional[float]] = {}
    for r in _load(result_dir, "cc*__w*__*.summary.json"):
        if int(r.get("num_tasks") or 0) != N_TASKS:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; dropped", file=sys.stderr)
            continue
        label = _label(r)
        seed = int(r.get("checkpoint_seed") or 0)
        arms[label][(_env(r), seed)] = float(r["averageElapsedTime"])
        rows[label][(_env(r), seed)] = _per_task(r)
        if label == L_REACTIVE:
            shares[_env(r)] = float(r["queue_share"])
    return arms, rows, shares


def report_screen(lever: str, result_dir: str, write_selection: Optional[str]) -> dict:
    _arms, _rows, shares = tables(result_dir)
    full = {(L_RUNG, t, w): shares.get((L_RUNG, t, w)) for t in E_TOPOLOGY_CANDIDATES for w in L_WINDOWS}
    r = read_l0(full)
    print(f"=== {lever} ({L_LINEAGE_OF[lever]}) screen at C{L_RUNG}: {r['verdict']} ===")
    print(f"   {'topology':>9} " + " ".join(f"{w:>8}" for w in L_WINDOWS))
    for t in E_TOPOLOGY_CANDIDATES:
        cells = []
        for w in L_WINDOWS:
            v = full[(L_RUNG, t, w)]
            cells.append("  hang  " if v is None else f"{v:8.3f}")
        print(f"   {t:>9} " + " ".join(cells))
    print(f"   {r['selection'].get('why')}")
    out = {"lever": lever, "lineage": L_LINEAGE_OF[lever], "verdict": r["verdict"],
           "topologies": r["selection"].get("topologies"), "environments": r["selection"].get("environments"),
           "queue_share": {f"{t}_{w}": full[(L_RUNG, t, w)] for t in E_TOPOLOGY_CANDIDATES for w in L_WINDOWS},
           "n_admissible": r["m0"]["n_admissible"]}
    if write_selection:
        os.makedirs(os.path.dirname(write_selection) or ".", exist_ok=True)
        json.dump(out, open(write_selection, "w"), indent=1)
        print(f"   wrote {write_selection}")
    return out


def _print_row(label, r):
    if r.get("verdict") == V_UNREADABLE:
        print(f"   {label:<36} UNREADABLE  {r.get('reason') or r.get('why') or ''}")
    else:
        print(f"   {label:<36} {r['verdict']:<42} {r['median']:+8.2f} % {r['p']:8.4f} {r['ahead']:4d}/{r['n']:<3}")


def _complete(graph: Dict[tuple, float], envs) -> set:
    seeds = {s for (_e, s) in graph}
    return {s for s in seeds if all((e, s) in graph for e in envs)}


def report_study(lever: str, result_dir: str, selection_path: Optional[str] = None) -> dict:
    sel_path = selection_path or f"simulation_data/env_lever_v1/selected_{lever}.json"
    sel = json.load(open(sel_path))
    if sel.get("verdict") != V_DESIGN_READY:
        raise SystemExit(f"FAIL LOUD: {sel_path} verdict {sel.get('verdict')!r}; the study must not be read")
    envs = [tuple(e) for e in sel["environments"]]
    amended = sel.get("amendment")
    if amended:
        print(f"\n   AMENDED DESIGN ({len(envs)} environments): {amended}")
    arms, rows, _shares = tables(result_dir)
    react = {e: v for (e, s), v in arms[L_REACTIVE].items() if s == 0}
    missing = [e for e in envs if e not in react]
    if missing:
        raise SystemExit(f"FAIL LOUD: no reactive baseline for {missing}")
    rk = median(react[e] for e in envs)
    print(f"\n=== {lever} ({L_LINEAGE_OF[lever]}) -- {len(envs)} environments ({sel['topologies']} x {list(L_WINDOWS)}), "
          f"reactive median {rk:.3f} s ===")
    R: dict = {"lever": lever, "environments": envs}
    for a in L_ARMS[lever]:
        want = 16 if a in (L_GRAPH, L_TWIN) else 1
        n = sum(1 for e in envs for s in range(0, 17) if (e, s) in arms.get(a, {}))
        print(f"   [{a}] {n}/{len(envs) * want} arms on disk")
    print(f"\n   {'read':<36} {'verdict':<42} {'median':>9} {'p':>8} {'ahead':>7}")

    def _ckpt(a, ref_by_env, fn, label):
        """Registered: every checkpoint on every environment. Disclosed: the checkpoints
        complete on every environment (a hung arm drops its checkpoint, never an environment)."""
        try:
            r = fn(checkpoint_stats(arms.get(a, {}), ref_by_env, envs))
        except ValueError as ex:
            r = {"verdict": V_UNREADABLE, "reason": str(ex)}
        if r["verdict"] == V_UNREADABLE:
            comp = _complete(arms.get(a, {}), envs)
            if comp:
                try:
                    d = fn(checkpoint_stats({k: v for k, v in arms[a].items() if k[1] in comp}, ref_by_env, envs),
                           min_n=len(comp))
                    d["disclosed"] = True; d["n_complete"] = len(comp); r["disclosed"] = d
                except ValueError as ex:
                    r["disclosed"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
        _print_row(label, r)
        if "disclosed" in r:
            _print_row(f"   disclosed ({r['disclosed'].get('n_complete', '?')} complete ckpts)", r["disclosed"])
        return r

    R["l1_graph"] = _ckpt(L_GRAPH, react, read_l1, "L1 gnnedge0 vs reactive")
    if lever == "burst":
        R["l1_twin"] = _ckpt(L_TWIN, react, read_l1, "   mpoff vs reactive")
    rules = {a: {e: v for (e, s), v in arms.get(a, {}).items()} for a in (L_IMMEDIATE, L_BATCHED, L_RANDOM)}
    for a, lab in ((L_IMMEDIATE, "L2 immediate rule vs reactive"), (L_BATCHED, "   batched rule vs reactive"),
                   (L_RANDOM, "   random vs reactive")):
        try:
            st = env_stats(rules[a], react, envs)
            R[f"l2_{a}"] = read_l2(st)
            if R[f"l2_{a}"]["verdict"] == V_UNREADABLE and len(st) >= 8:
                # a signed 12-environment amendment: the same bar at n = 12, DISCLOSED
                from scripts_cosim.unsaturated_scale_v2_read import _one_sample
                from scripts_cosim.env_lever_v1_read import V_REACTIVE_FASTER_THAN_RULE, V_NOT_SEP
                d = _one_sample(st, min_n=len(st), faster_a=V_RULE_BEATS_REACTIVE,
                                faster_b=V_REACTIVE_FASTER_THAN_RULE, tie=V_NOT_SEP)
                d["disclosed"] = True
                R[f"l2_{a}"]["disclosed"] = d
        except ValueError as ex:
            R[f"l2_{a}"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
        _print_row(lab, R[f"l2_{a}"])
        if "disclosed" in R[f"l2_{a}"]:
            _print_row(f"   disclosed (n = {R[f'l2_{a}']['disclosed']['n']} env)", R[f"l2_{a}"]["disclosed"])
    graph = arms.get(L_GRAPH, {})
    if graph and all(e in rules[L_IMMEDIATE] for e in envs):
        seeds = sorted({s for (_e, s) in graph})
        try:
            R["l3"] = read_l3(pair_checkpoint_stats(graph, broadcast_rule(rules[L_IMMEDIATE], seeds), envs))
        except ValueError as ex:
            R["l3"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
        if R["l3"]["verdict"] == V_UNREADABLE:
            comp = _complete(graph, envs)
            if comp:
                d = read_l3(pair_checkpoint_stats({k: v for k, v in graph.items() if k[1] in comp},
                                                  broadcast_rule(rules[L_IMMEDIATE], sorted(comp)), envs),
                            min_n=len(comp))
                d["disclosed"] = True; d["n_complete"] = len(comp); R["l3"]["disclosed"] = d
    else:
        R["l3"] = {"verdict": V_UNREADABLE, "reason": "gnnedge0 or the immediate rule absent"}
    _print_row("L3 gnnedge0 vs immediate rule", R["l3"])
    if "disclosed" in R["l3"]:
        _print_row(f"   disclosed ({R['l3']['disclosed']['n_complete']} complete ckpts)", R["l3"]["disclosed"])
    if lever == "burst":
        waits = {k: rows[L_GRAPH][k]["wait"] for k in rows.get(L_GRAPH, {}) if k[0] in envs}
        R["l4"] = read_l4(waits)
        print(f"   L4 {R['l4']['verdict']}: gnnedge0 median scheduler wait {R['l4'].get('median_wait_s', float('nan')):.3f} s "
              f"against {R['l4'].get('bar_s')} s")
        twin = arms.get(L_TWIN, {})
        if graph and twin:
            try:
                R["l5"] = read_l5(pair_checkpoint_stats(graph, twin, envs))
            except ValueError as ex:
                R["l5"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
            if R["l5"]["verdict"] == V_UNREADABLE:
                comp = _complete(graph, envs) & _complete(twin, envs)
                if comp:
                    d = read_l5(pair_checkpoint_stats({k: v for k, v in graph.items() if k[1] in comp},
                                                      {k: v for k, v in twin.items() if k[1] in comp}, envs),
                                min_n=len(comp))
                    d["disclosed"] = True; d["n_complete"] = len(comp); R["l5"]["disclosed"] = d
            _print_row("L5 gnnedge0 vs mpoff", R["l5"])
            if "disclosed" in R["l5"]:
                _print_row(f"   disclosed ({R['l5']['disclosed']['n_complete']} complete ckpts)", R["l5"]["disclosed"])

    print(f"\n   per-task decomposition (medians over environments and checkpoints, s):")
    print(f"   {'arm':<28} {'wait':>7} {'queue':>7} {'exch':>7} {'rendez':>7} {'elapsed':>8}")
    R["decomposition"] = {}
    for a in (L_REACTIVE, L_RANDOM, L_IMMEDIATE, L_BATCHED, L_GRAPH, L_TWIN):
        ks = [k for k in rows.get(a, {}) if k[0] in envs]
        if not ks:
            continue
        m = {c: median(rows[a][k][c] for k in ks) for c in ("wait", "queue", "exchange", "rendezvous", "elapsed")}
        R["decomposition"][a] = m
        print(f"   {a:<28} {m['wait']:7.2f} {m['queue']:7.2f} {m['exchange']:7.2f} {m['rendezvous']:7.2f} {m['elapsed']:8.2f}")
    return R


def report_regime(results_root: str) -> dict:
    """payload_scale_v1: compose L1 (gnnedge0) and L2 (immediate rule) across the three scales.
    k = 1 is the study itself: reactive from ue_v1_screen, gnnedge0 from ue_v1, the rule from
    pg_v1, on unsaturated_edge_v1's C40 environments."""
    out = {"by_scale": {}}
    # k = 1
    sel = json.load(open("simulation_data/unsaturated_edge_v1/selected.json"))
    envs = [tuple(e) for e in sel["rungs"]["40"]["environments"]]
    a_s, _r, _ = tables(os.path.join(results_root, "ue_v1_screen"))
    a_u, _r, _ = tables(os.path.join(results_root, "ue_v1"))
    a_p, _r, _ = tables(os.path.join(results_root, "pg_v1"))
    react = {e: v for (e, s), v in a_s[L_REACTIVE].items() if s == 0}
    graph = a_u.get(L_GRAPH, {})
    comp = _complete(graph, envs)
    l1 = read_l1(checkpoint_stats({k: v for k, v in graph.items() if k[1] in comp}, react, envs), min_n=len(comp))
    rule = {e: v for (e, s), v in a_p.get(L_IMMEDIATE, {}).items()}
    l2 = read_l2(env_stats(rule, react, envs)) if all(e in rule for e in envs) else {"verdict": V_UNREADABLE}
    out["by_scale"][1.0] = {"l1": l1, "l2": l2, "n_complete": len(comp)}
    for k, lever in ((0.1, "pk0.1"), (3.0, "pk3"), (10.0, "pk10")):
        sel_path = f"simulation_data/env_lever_v1/selected_{lever}.json"
        d = os.path.join(results_root, "el_v1", lever)
        if not os.path.exists(sel_path) or json.load(open(sel_path)).get("verdict") != V_DESIGN_READY:
            out["by_scale"][k] = {"l1": {"verdict": V_UNREADABLE, "reason": "no design"}, "l2": {"verdict": V_UNREADABLE, "reason": "no design"}}
            continue
        r = report_study(lever, d, sel_path)
        out["by_scale"][k] = {"l1": r["l1_graph"], "l2": r[f"l2_{L_IMMEDIATE}"]}
    print("\n=== payload_scale_v1 regime read ===")
    for who, key, beats in (("immediate rule", "l2", V_RULE_BEATS_REACTIVE), ("gnnedge0", "l1", V_ARM_BEATS_REACTIVE)):
        vb = {k: out["by_scale"][k][key].get("verdict") for k in L_PAYLOAD_SCALES}
        r = read_l6(vb, beats=beats)
        out[f"regime_{key}"] = r
        print(f"   {who:<16} " + "  ".join(f"x{k:g}: {vb[k]} ({out['by_scale'][k][key].get('median', float('nan')):+.1f} %)" for k in L_PAYLOAD_SCALES))
        print(f"   {'':<16} -> {r['verdict']}")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("screen", "study", "regime"))
    ap.add_argument("lever_or_root")
    ap.add_argument("result_dir", nargs="?")
    ap.add_argument("--write-selection")
    ap.add_argument("--selection", help="study: an amended selection file instead of selected_<lever>.json")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.step == "regime":
        res = report_regime(a.lever_or_root)
    else:
        if a.lever_or_root not in L_LEVERS:
            ap.error(f"lever must be one of {L_LEVERS}")
        if not a.result_dir:
            ap.error("result_dir needed")
        res = report_screen(a.lever_or_root, a.result_dir, a.write_selection) if a.step == "screen" \
            else report_study(a.lever_or_root, a.result_dir, a.selection)
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

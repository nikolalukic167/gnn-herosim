"""peer_greedy_live_v1 -- the gate read. Bars live in `peer_greedy_live_v1_read.py`; this file
only finds the summaries, pairs the arms and prints.

    python3 scripts_cosim/peer_greedy_live_v1_gate_read.py \
        simulation_data/peer_affinity_live_gate/results/ue_v1_screen \
        simulation_data/peer_affinity_live_gate/results/ue_v1 \
        simulation_data/peer_affinity_live_gate/results/pg_v1 [--json out.json]

Reactive comes from the ue_v1 screen (`reactive_s0`), ECT / random / gnnedge0 from the ue_v1
study, the rule arms from pg_v1; the environments are the ones `unsaturated_edge_v1` selected
(`simulation_data/unsaturated_edge_v1/selected.json`).
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

from scripts_cosim.peer_greedy_live_v1_read import (  # noqa: E402
    P_BATCHED, P_DRAIN, P_ECT, P_GRAPH, P_IMMEDIATE, P_MIN_CHECKPOINTS, P_RANDOM, P_REACTIVE,
    P_RULES, P_RUNGS, P_TWIN, P_WINDOWS, V_UNREADABLE, broadcast_rule, env_stats,
    pair_checkpoint_stats, read_g1, read_g2, read_g3, read_g4, read_g5, read_g6, read_g7, read_g8,
)

SELECTION = "simulation_data/unsaturated_edge_v1/selected.json"
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


def tables(screen_dir: str, study_dir: str, rule_dir: str):
    """arms[label][(env, seed)] = elapsed; rows[label][env] = per-task decomposition + counters."""
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    rows: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for r in _load(screen_dir, "cc*__w*__reactive_s0.summary.json"):
        if int(r.get("num_tasks") or 0) != N_TASKS:
            continue
        arms[P_REACTIVE][(_env(r), 0)] = float(r["averageElapsedTime"])
        rows[P_REACTIVE][_env(r)] = _per_task(r)
    for r in _load(study_dir, "cc*__w*__*.summary.json"):
        if int(r.get("num_tasks") or 0) != N_TASKS:
            continue
        kind = r["arm_kind"]
        if kind in (P_ECT, P_RANDOM):
            arms[kind][(_env(r), 0)] = float(r["averageElapsedTime"])
            rows[kind][_env(r)] = _per_task(r)
        elif f"{r['corpus']}_{kind}" in (P_GRAPH, P_TWIN):
            label = f"{r['corpus']}_{kind}"
            arms[label][(_env(r), int(r["checkpoint_seed"]))] = float(r["averageElapsedTime"])
    for r in _load(rule_dir, "cc*__w*__*.summary.json"):
        if int(r.get("num_tasks") or 0) != N_TASKS:
            print(f"[WARN] {r['arm']}: num_tasks={r.get('num_tasks')!r}; dropped", file=sys.stderr)
            continue
        kind = r["arm_kind"]
        if kind not in P_RULES:
            raise SystemExit(f"FAIL LOUD: {r['arm']} is not a rule arm of this lineage")
        arms[kind][(_env(r), 0)] = float(r["averageElapsedTime"])
        row = _per_task(r)
        c = r.get("schedulerCounters") or {}
        if "pg_decisions" not in c:
            raise SystemExit(f"FAIL LOUD: {r['arm']} carries no pg_* counters; the instrument is off")
        row["counters"] = {k: v for k, v in c.items() if k.startswith("pg_")}
        rows[kind][_env(r)] = row
    return arms, rows


def _by_env(arm: Dict[tuple, float]) -> Dict[tuple, float]:
    return {e: v for (e, _s), v in arm.items()}


def _print_row(label, r):
    if r.get("verdict") == V_UNREADABLE:
        print(f"   {label:<34} UNREADABLE  {r.get('reason') or r.get('why') or ''}")
    else:
        print(f"   {label:<34} {r['verdict']:<38} {r['median']:+8.2f} % {r['p']:8.4f} "
              f"{r['ahead']:4d}/{r['n']:<3}")


def report(screen_dir: str, study_dir: str, rule_dir: str) -> dict:
    sel = json.load(open(SELECTION))
    if sel.get("verdict") != "DESIGN-READY":
        raise SystemExit(f"FAIL LOUD: selection verdict {sel.get('verdict')!r}")
    arms, rows = tables(screen_dir, study_dir, rule_dir)
    out: dict = {"rungs": {}}
    for nc in P_RUNGS:
        envs = [tuple(e) for e in sel["rungs"][str(nc)]["environments"]]
        react = _by_env(arms[P_REACTIVE])
        missing = [e for e in envs if e not in react]
        if missing:
            raise SystemExit(f"FAIL LOUD: no reactive baseline for {missing}")
        rk = median(react[e] for e in envs)
        print(f"\n=== C{nc} -- 16 environments ({sel['rungs'][str(nc)]['topologies']} x {list(P_WINDOWS)}), "
              f"reactive median {rk:.3f} s ===")
        have = {a: _by_env(arms[a]) for a in P_RULES if a in arms}
        for a in P_RULES:
            n = sum(1 for e in envs if e in have.get(a, {}))
            if n != len(envs):
                print(f"   [{a}] {n}/{len(envs)} environments on disk")
        R: dict = {"environments": envs}

        def _stat(a, b_by_env):
            try:
                return env_stats(have[a], b_by_env, envs)
            except (KeyError, ValueError) as ex:
                return {"__error__": str(ex)}

        def _read(fn, s, *, a=None, b_by_env=None):
            if "__error__" not in s:
                return fn(s)
            r = {"verdict": V_UNREADABLE, "reason": s["__error__"]}
            if a is not None and b_by_env is not None and a in have:
                envs_have = [e for e in envs if e in have[a] and e in b_by_env]
                if len(envs_have) >= 8:
                    d = _one_sample_env(fn, env_stats(have[a], b_by_env, envs_have))
                    d.update(disclosed=True, n_environments=len(envs_have))
                    r["disclosed"] = d
            return r

        def _one_sample_env(fn, stats):
            # the read's own verdict vocabulary at the disclosed n: same bar, fewer environments
            from scripts_cosim.unsaturated_scale_v2_read import _one_sample
            import scripts_cosim.peer_greedy_live_v1_read as m
            names = {read_g1: (m.V_IMMEDIATE_BEATS_REACTIVE, m.V_REACTIVE_FASTER_THAN_IMMEDIATE, m.V_NOT_SEP),
                     read_g2: (m.V_BATCHED_BEATS_REACTIVE, m.V_REACTIVE_FASTER_THAN_BATCHED, m.V_NOT_SEP),
                     read_g4: (m.V_IMMEDIATE_BEATS_RANDOM, m.V_RANDOM_FASTER_THAN_IMMEDIATE, m.V_RANDOM_NOT_SEP),
                     read_g5: (m.V_IMMEDIATE_FASTER_THAN_BATCHED, m.V_BATCHED_FASTER_THAN_IMMEDIATE, m.V_NOT_SEP),
                     read_g6: (m.V_EXCHANGE_HELPS, m.V_EXCHANGE_HURTS, m.V_EXCHANGE_NOT_SEP)}[fn]
            return _one_sample(stats, min_n=len(stats), faster_a=names[0], faster_b=names[1], tie=names[2])

        print(f"\n   {'read':<34} {'verdict':<38} {'median':>9} {'p':>8} {'ahead':>7}")
        R["g1"] = _read(read_g1, _stat(P_IMMEDIATE, react)); _print_row("G1 immediate vs reactive", R["g1"])
        R["g2"] = _read(read_g2, _stat(P_BATCHED, react), a=P_BATCHED, b_by_env=react); _print_row("G2 batched vs reactive", R["g2"])
        if "disclosed" in R["g2"]: _print_row(f"   disclosed ({R['g2']['disclosed']['n_environments']} env)", R["g2"]["disclosed"])
        R["drain_vs_reactive"] = _read(read_g1, _stat(P_DRAIN, react)); _print_row("   drain-only vs reactive", R["drain_vs_reactive"])
        graph = arms.get(P_GRAPH, {})
        seeds = sorted({s for (_e, s) in graph})

        def _rule_vs_graph(rule_by_env):
            """Registered: all 16 environments x all 16 checkpoints. Disclosed: the environments
            the rule has x the checkpoints complete on them -- printed as such, never in the
            registered slot."""
            out = {}
            try:
                out["registered"] = read_g3(pair_checkpoint_stats(broadcast_rule(rule_by_env, seeds), graph, envs))
            except ValueError as ex:
                out["registered"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
            if out["registered"]["verdict"] == V_UNREADABLE:
                envs_have = [e for e in envs if e in rule_by_env]
                complete = sorted(s for s in seeds if all((e, s) in graph for e in envs_have))
                if envs_have and complete:
                    st = pair_checkpoint_stats(broadcast_rule(rule_by_env, complete),
                                               {k: v for k, v in graph.items() if k[1] in complete}, envs_have)
                    d = read_g3(st, min_n=len(complete))
                    d.update(disclosed=True, n_environments=len(envs_have), n_checkpoints=len(complete))
                    out["disclosed"] = d
            return out

        if P_BATCHED in have and graph:
            g3 = _rule_vs_graph(have[P_BATCHED])
            R["g3"] = g3["registered"]
            if "disclosed" in g3:
                R["g3_disclosed"] = g3["disclosed"]
        else:
            R["g3"] = {"verdict": V_UNREADABLE, "reason": "batched rule or gnnedge0 absent"}
        _print_row("G3 batched rule vs gnnedge0 (ckpt)", R["g3"])
        if "g3_disclosed" in R:
            d = R["g3_disclosed"]
            _print_row(f"   disclosed ({d['n_environments']} env x {d['n_checkpoints']} ckpt)", d)
        if P_IMMEDIATE in have and graph:
            ig = _rule_vs_graph(have[P_IMMEDIATE])
            R["immediate_vs_graph"] = ig["registered"]
            _print_row("   immediate rule vs gnnedge0 (ckpt)", R["immediate_vs_graph"])
            if "disclosed" in ig:
                R["immediate_vs_graph_disclosed"] = ig["disclosed"]
                d = ig["disclosed"]
                _print_row(f"   disclosed ({d['n_environments']} env x {d['n_checkpoints']} ckpt)", d)
        R["g4"] = _read(read_g4, _stat(P_IMMEDIATE, _by_env(arms.get(P_RANDOM, {})))); _print_row("G4 immediate vs random", R["g4"])
        R["g5"] = _read(read_g5, _stat(P_IMMEDIATE, have.get(P_BATCHED, {})), a=P_IMMEDIATE, b_by_env=have.get(P_BATCHED, {})); _print_row("G5 immediate vs batched", R["g5"])
        if "disclosed" in R["g5"]: _print_row(f"   disclosed ({R['g5']['disclosed']['n_environments']} env)", R["g5"]["disclosed"])
        R["g6"] = _read(read_g6, _stat(P_IMMEDIATE, have.get(P_DRAIN, {}))); _print_row("G6 immediate vs drain-only", R["g6"])
        R["immediate_vs_ect"] = _read(read_g1, _stat(P_IMMEDIATE, _by_env(arms.get(P_ECT, {})))); _print_row("   immediate vs ECT", R["immediate_vs_ect"])

        if P_IMMEDIATE in have:
            ex = {e: (rows[P_IMMEDIATE][e]["exchange"], rows[P_REACTIVE][e]["exchange"]) for e in envs if e in rows[P_IMMEDIATE]}
            q = {e: (rows[P_IMMEDIATE][e]["queue"], rows[P_REACTIVE][e]["queue"]) for e in envs if e in rows[P_IMMEDIATE]}
            R["g7"] = read_g7(ex, q)
        else:
            R["g7"] = {"verdict": V_UNREADABLE, "reason": "immediate rule absent"}
        print(f"\n   G7 {R['g7']['verdict']}: {R['g7'].get('why') or R['g7'].get('reason')}")
        R["g8"] = read_g8(R["g1"], R["g3"], R["g5"])
        print(f"   G8 {R['g8']['verdict']}: {R['g8'].get('why', '')}")

        print(f"\n   per-task decomposition (medians over the environments, s):")
        print(f"   {'arm':<28} {'wait':>7} {'queue':>7} {'exch':>7} {'rendez':>7} {'elapsed':>8}   vs reactive (median of per-env %)")
        for a in (P_REACTIVE, P_ECT, P_RANDOM, P_DRAIN, P_IMMEDIATE, P_BATCHED):
            if a not in rows or not all(e in rows[a] for e in envs):
                continue
            m = {k: median(rows[a][e][k] for e in envs) for k in ("wait", "queue", "exchange", "rendezvous", "elapsed")}
            rel = median(100 * (rows[a][e]["elapsed"] / react[e] - 1) for e in envs)
            print(f"   {a:<28} {m['wait']:7.2f} {m['queue']:7.2f} {m['exchange']:7.2f} {m['rendezvous']:7.2f} {m['elapsed']:8.2f}   {rel:+7.2f} %")
        for a in (P_IMMEDIATE, P_BATCHED, P_DRAIN):
            if a in rows and all(e in rows[a] for e in envs):
                c = {k: median(rows[a][e]["counters"].get(k, 0) for e in envs)
                     for k in ("pg_decisions", "pg_partners_known", "pg_partners_unknown", "pg_joined_partner", "pg_moved_by_exchange", "pg_batches")}
                d = max(1.0, c["pg_decisions"])
                print(f"   {a:<28} joined a partner's node on {100*c['pg_joined_partner']/d:5.1f} % of decisions, "
                      f"exchange moved the argmin on {100*c['pg_moved_by_exchange']/d:5.1f} %; partners known/unknown "
                      f"{c['pg_partners_known']:.0f}/{c['pg_partners_unknown']:.0f}; batches {c['pg_batches']:.0f}")

        print(f"\n   per-environment elapsed (s) and % vs reactive:")
        print(f"   {'environment':<18} {'reactive':>9} " + " ".join(f"{a[:14]:>16}" for a in (P_DRAIN, P_IMMEDIATE, P_BATCHED)))
        per_env = {}
        for e in envs:
            cells = []
            for a in (P_DRAIN, P_IMMEDIATE, P_BATCHED):
                v = have.get(a, {}).get(e)
                cells.append("        --      " if v is None else f"{v:7.2f} ({100*(v/react[e]-1):+6.1f})")
            per_env[e] = {a: have.get(a, {}).get(e) for a in P_RULES}
            print(f"   {str(e):<18} {react[e]:9.3f} " + " ".join(cells))
        R["per_env"] = per_env
        R["decomposition"] = {a: {str(e): rows[a][e] for e in envs if e in rows[a]} for a in rows}
        out["rungs"][nc] = R
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("screen_dir")
    ap.add_argument("study_dir")
    ap.add_argument("rule_dir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = report(a.screen_dir, a.study_dir, a.rule_dir)
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

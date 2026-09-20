#!/usr/bin/env python3
"""joint_burst_v2 -- read the gate and fire K1-K9. docs/lineages/joint_burst_v2.md.

Reads the v2 result summaries (results/jb_v2/gate) plus reactive from the v1 screen/gate (the
v2 gate reuses reactive on the identical cells). Builds paired per-(env,seed) tables and calls
the signed bars in scripts_cosim/joint_burst_v2_read.py. Registered on every checkpoint;
discloses a fallback on the checkpoints complete across the paired arms when an arm hangs.

Usage (datalab, micromamba gnn):
  PYTHONPATH=. python3 scripts_cosim/joint_burst_v2_gate_read.py \
      --gate-dir  simulation_data/peer_affinity_live_gate/results/jb_v2/gate \
      --v1-dirs   simulation_data/peer_affinity_live_gate/results/jb_v1/gate \
                  simulation_data/peer_affinity_live_gate/results/jb_v1/screen \
      --selection simulation_data/joint_burst_v1/selected.json \
      --out       simulation_data/joint_burst_v2/read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from statistics import median
from typing import Dict, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts_cosim.peer_greedy_live_v1_read import broadcast_rule, env_stats  # noqa: E402
from scripts_cosim.unsaturated_edge_v1_read import pair_checkpoint_stats  # noqa: E402
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402
from scripts_cosim.joint_burst_v2_read import (  # noqa: E402
    K_GRAPH, K_TWIN, K_V1_UNCAPPED, K_CD, K_BATCHED, K_REACTIVE, K_RANDOM, K_X2,
    read_k1, read_k2, read_k3, read_k4, read_k5, read_k6, read_k7, read_k8, read_k9,
)

N_TASKS = 50000
SEEDS = tuple(range(1, 17))


def _env(r) -> tuple:
    return (int(r["clients"]), int(r["topology"]), r["window"])


def _label(r: dict) -> str:
    """Map a summary to its v2 arm label."""
    name = str(r.get("arm") or "")
    kind = r.get("arm_kind")
    corpus = r.get("corpus")
    if "x2scale" in name:
        return K_X2
    if kind == "reactive" or kind == K_REACTIVE:
        return K_REACTIVE
    if kind == "random" or kind == "random_network":
        return K_RANDOM
    if kind == "peer_greedy_network":
        return "peer_greedy_network_immediate"
    if kind == "peer_greedy_network_cd":
        return K_CD
    if kind in ("batched", "peer_greedy_network_batch"):
        return K_BATCHED
    if corpus and kind in ("gnnedge0", "mpoff"):
        return f"{corpus}_{kind}"          # jb2_gnnedge0, jb2_mpoff, jb1uncapped_gnnedge0
    return f"{corpus}_{kind}"


def _per_task(r: dict) -> dict:
    n = float(r["num_tasks"])
    return {"elapsed": float(r["averageElapsedTime"]), "wait": float(r["averageWaitTime"] or 0.0),
            "queue": float(r["averageQueueTime"]),
            "exchange": float(r["totalPeerExchangeTime"]) / n,
            "rendezvous": float(r["totalPeerRendezvousWait"]) / n}


def tables(gate_dirs, v1_dirs):
    arms: Dict[str, Dict[tuple, float]] = defaultdict(dict)
    rows: Dict[str, Dict[tuple, dict]] = defaultdict(dict)
    for d in list(gate_dirs) + list(v1_dirs):
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "cc*__w*__*.summary.json"))):
            r = json.load(open(f))
            if int(r.get("num_tasks") or 0) != N_TASKS:
                continue
            label = _label(r)
            seed = int(r.get("checkpoint_seed") or 0)
            e = _env(r)
            # v1 dirs only contribute reactive (the reused arm); ignore their learned/other arms
            if d in v1_dirs and label != K_REACTIVE:
                continue
            arms[label][(e, seed)] = float(r["averageElapsedTime"])
            rows[label][(e, seed)] = _per_task(r)
    return arms, rows


def _rule_by_env(arms, label) -> dict:
    """A per-arrival rule / reactive is stored at seed 0; collapse to {env: elapsed}."""
    return {e: v for (e, s), v in arms.get(label, {}).items()}


def _complete_seeds(arm, envs) -> set:
    return {s for s in {s for (_e, s) in arm} if all((e, s) in arm for e in envs)}


def _pair(a, b, envs, fn, min_n_full=16):
    """fn on pair_checkpoint_stats(a, b); disclose a fallback on complete seeds if unreadable."""
    try:
        r = fn(pair_checkpoint_stats(a, b, envs))
        if r["verdict"] != V_UNREADABLE:
            return r, None
    except ValueError as ex:
        r = {"verdict": V_UNREADABLE, "reason": str(ex)}
    comp = _complete_seeds(a, envs) & _complete_seeds(b, envs)
    if comp:
        d = fn(pair_checkpoint_stats({k: v for k, v in a.items() if k[1] in comp},
                                     {k: v for k, v in b.items() if k[1] in comp}, envs),
               min_n=len(comp))
        return r, {"disclosed_on": sorted(comp), **d}
    return r, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--v1-dirs", nargs="*", default=[])
    ap.add_argument("--selection", default="simulation_data/joint_burst_v1/selected.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    sel = json.load(open(a.selection))
    envs = [tuple(e) for e in sel["environments"]]
    arms, rows = tables([a.gate_dir], a.v1_dirs)

    graph = arms.get(K_GRAPH, {})
    twin = arms.get(K_TWIN, {})
    v1unc = arms.get(K_V1_UNCAPPED, {})
    cd = _rule_by_env(arms, K_CD)
    batched = _rule_by_env(arms, K_BATCHED)
    reactive = _rule_by_env(arms, K_REACTIVE)
    x2 = _rule_by_env(arms, K_X2)

    def present(name, d):
        miss = [e for e in envs if e not in d]
        return f"{name}: {len(d)} cells" + (f"  MISSING {miss}" if miss else "")

    print("=== joint_burst_v2 gate ===")
    for nm, d in (("reactive", reactive), ("batched greedy", batched), ("CD greedy", cd),
                  ("x2 probe", x2)):
        print("  " + present(nm, d))
    for nm, d in (("jb2 gnnedge0", graph), ("jb2 mpoff", twin), ("v1-uncapped gnnedge0", v1unc)):
        seeds = _complete_seeds(d, envs)
        print(f"  {nm}: {len(seeds)}/16 complete checkpoints")

    R: Dict[str, object] = {"environments": [list(e) for e in envs]}
    batched_bc = broadcast_rule(batched, SEEDS)
    cd_bc = broadcast_rule(cd, SEEDS)
    react_bc = broadcast_rule(reactive, SEEDS)

    R["k1"], R["k1_disc"] = _pair(graph, batched_bc, envs, read_k1)
    R["k2"], R["k2_disc"] = _pair(graph, cd_bc, envs, read_k2)
    R["k3"], R["k3_disc"] = _pair(graph, react_bc, envs, read_k3)
    R["k4"], R["k4_disc"] = _pair(graph, twin, envs, read_k4)
    R["k5"], R["k5_disc"] = _pair(graph, v1unc, envs, read_k5)
    R["k6"], R["k6_disc"] = _pair(v1unc, batched_bc, envs, read_k6)
    # K7/K8 are per-environment rule contrasts
    try:
        R["k7"] = read_k7(env_stats(cd, batched, envs))
    except ValueError as ex:
        R["k7"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
    try:
        R["k8"] = read_k8(env_stats(x2, batched, envs)) if x2 else {"verdict": V_UNREADABLE, "reason": "no x2 arm"}
    except ValueError as ex:
        R["k8"] = {"verdict": V_UNREADABLE, "reason": str(ex)}
    R["k9"] = read_k9(R["k1"], R["k2"])

    print(f"\n   {'read':<42} {'verdict':<34} {'median':>8} {'p':>8} {'ahead':>7}")
    def show(key, label):
        r = R.get(key) or {}
        disc = R.get(key + "_disc")
        if r.get("verdict") == V_UNREADABLE and disc:
            r2 = disc
            print(f"   {label:<42} {'UNREADABLE -> disclosed':<34}")
            print(f"   {'  (' + str(len(disc.get('disclosed_on', []))) + ' complete)':<42} "
                  f"{r2['verdict']:<34} {r2['median']:+7.2f}% {r2['p']:8.4f} {r2['ahead']:3d}/{r2['n']:<3}")
        elif r.get("verdict") == V_UNREADABLE:
            print(f"   {label:<42} UNREADABLE {r.get('reason','')[:50]}")
        else:
            print(f"   {label:<42} {r['verdict']:<34} {r['median']:+7.2f}% {r['p']:8.4f} {r['ahead']:3d}/{r['n']:<3}")
    show("k1", "K1 jb2 gnnedge0 vs 1-pass greedy [reachable]")
    show("k2", "K2 jb2 gnnedge0 vs CD greedy [honest]")
    show("k3", "K3 jb2 gnnedge0 vs reactive")
    show("k4", "K4 jb2 gnnedge0 vs mpoff twin")
    show("k5", "K5 jb2 vs v1-uncapped (corpus beyond cap)")
    show("k6", "K6 v1-uncapped vs 1-pass greedy")
    for key, label in (("k7", "K7 CD vs 1-pass greedy [env]"), ("k8", "K8 x2 vs 1-pass greedy [env]")):
        r = R[key]
        if r.get("verdict") == V_UNREADABLE:
            print(f"   {label:<42} UNREADABLE {r.get('reason','')[:40]}")
        else:
            print(f"   {label:<42} {r['verdict']:<34} {r['median']:+7.2f}% {r['p']:8.4f} {r['ahead']:3d}/{r['n']:<3}")
    print(f"\n   K9 COMPOSITE: {R['k9']['verdict']}   (K1={R['k9'].get('k1')}, K2={R['k9'].get('k2')})")

    # per-task decomposition (medians over env x seed)
    print(f"\n   per-task medians (s): {'arm':<26} {'queue':>7} {'exch':>7} {'elapsed':>8}")
    for label in (K_REACTIVE, K_RANDOM, "peer_greedy_network_immediate", K_BATCHED, K_CD, K_X2,
                  K_GRAPH, K_TWIN, K_V1_UNCAPPED):
        rr = rows.get(label)
        if not rr:
            continue
        m = {c: median(v[c] for v in rr.values()) for c in ("queue", "exchange", "elapsed")}
        print(f"   {'':<26}{label:<26} {m['queue']:7.3f} {m['exchange']:7.3f} {m['elapsed']:8.3f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(R, open(a.out, "w"), indent=1, default=list)
        print(f"\n   wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

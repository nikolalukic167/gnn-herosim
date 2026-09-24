#!/usr/bin/env python3
"""selfpredict_burst_v1 -- read the live gate: the self-predict rule vs gnnedge0 in the burst seat.
See docs/lineages/selfpredict_burst_v1.md.

joint_burst_v2's uncapped gnnedge0 beat the 1-pass greedy in its own (burst) seat -11.9 % (13/13)
and lost to the CD greedy +12.5 %; selfpredict_bar_v1 then made the self-predict rule the bar on the
unburst rungs. This gate puts that rule on the SAME 16 burst environments, fresh at one commit with
every arm it is compared against.

Units as in joint_burst_v2: gnnedge0 per CHECKPOINT (the 13 jb2 checkpoints that exist: seeds
{1-5, 7-10, 12, 14-16}; 6/11/13 were never trained), a rule per ENVIRONMENT (n = 16), rule vs
learned by broadcasting the rule over checkpoint seeds. Metric averageElapsedTime (joint_burst_v2's).
Bars: |median| >= 5 %, p < 0.05, two-sided signed-rank (signed 2026-09-24, before any data):

  S1 gnnedge0 vs selfpredict, GRAPH FIRST, n = 13 ckpt   GRAPH-BEATS-SELFPREDICT / SELFPREDICT-FASTER / NOT-SEPARATED
  S2 selfpredict vs CD greedy, n = 16 env                SELFPREDICT-BEATS-CD / CD-FASTER / NOT-SEPARATED
  S3 selfpredict vs 1-pass batched greedy, n = 16 env    SELFPREDICT-BEATS-1PASS / 1PASS-FASTER / NOT-SEPARATED
  R  replication (disclosed): fresh gnnedge0 vs 1-pass / vs CD against joint_burst_v2's -11.90 % / +12.48 %,
     and every fresh (arm, env, seed) elapsed against the jb2 gate's summary when --jb2-dir is given
  -- disclosed: selfpredict vs immediate rule, vs knative_network, vs random; per-task decomposition
  VERDICT:
    GNN-BEATS-SELFPREDICT        if S1 fires graph-faster
    SELFPREDICT-AHEAD-OF-GNN     if S1 fires selfpredict-faster
    GNN-TIES-SELFPREDICT         otherwise
  BURST-SEAT BAR: CD if S2 is CD-FASTER, SELFPREDICT if S2 is SELFPREDICT-BEATS-CD, else CD~SELFPREDICT

Usage (datalab, micromamba gnn):
  PYTHONPATH=. python3 scripts_cosim/selfpredict_burst_v1_read.py \\
      --gate-dir  simulation_data/peer_affinity_live_gate/results/selfpredict_burst_v1_gate \\
      --jb2-dir   simulation_data/peer_affinity_live_gate/results/jb_v2/gate \\
      --selection simulation_data/joint_burst_v1/selected.json \\
      --out       simulation_data/selfpredict_burst_v1/read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from statistics import median
from typing import Dict, Mapping, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts_cosim.peer_greedy_live_v1_read import P_MIN_ENVIRONMENTS, broadcast_rule, env_stats  # noqa: E402
from scripts_cosim.unsaturated_scale_v2_read import _one_sample, pair_checkpoint_stats  # noqa: E402

SELFPREDICT = "peer_greedy_selfpredict_network"
IMMEDIATE = "peer_greedy_network"
BATCHED = "peer_greedy_network_batch"
CD = "peer_greedy_network_cd"
REACTIVE = "knative_network"
RANDOM = "random_network"
GRAPH = "gnnedge0"
RULES = (SELFPREDICT, IMMEDIATE, BATCHED, CD, REACTIVE, RANDOM)
SEEDS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)
N_TASKS = 50000
JB2 = {"vs_1pass": -11.90, "vs_cd": 12.48}
PARTS = ("wait", "queue", "exchange", "rendezvous")

V_NOT_SEP = "NOT-SEPARATED"


def _per_task(r: dict) -> dict:
    n = float(r["num_tasks"])
    return {"elapsed": float(r["averageElapsedTime"]), "wait": float(r["averageWaitTime"] or 0.0),
            "queue": float(r["averageQueueTime"]),
            "exchange": float(r["totalPeerExchangeTime"] or 0.0) / n,
            "rendezvous": float(r["totalPeerRendezvousWait"] or 0.0) / n}


def _load(d: str, *, jb2: bool = False) -> Dict[str, Dict[tuple, dict]]:
    out: Dict[str, Dict[tuple, dict]] = {}
    for f in sorted(glob.glob(os.path.join(d, "cc*__w*__*.summary.json"))):
        r = json.load(open(f))
        if int(r.get("num_tasks") or 0) != N_TASKS:
            continue
        kind = r.get("arm_kind")
        if jb2:
            if "x2scale" in str(r.get("arm")) or (kind in ("gnnedge0", "mpoff") and r.get("corpus") != "jb2"):
                continue
            if kind == "mpoff":
                continue
        if kind not in RULES + (GRAPH,):
            continue
        env = (int(r["clients"]), int(r["topology"]), r["window"])
        out.setdefault(kind, {})[(env, int(r.get("checkpoint_seed") or 0))] = _per_task(r)
    return out


def _metric(arm: Mapping[tuple, dict], m: str = "elapsed") -> Dict[tuple, float]:
    return {k: v[m] for k, v in arm.items()}


def _by_env(arm: Mapping[tuple, dict], m: str = "elapsed") -> Dict[tuple, float]:
    return {e: v[m] for (e, _s), v in arm.items()}


def _parts(a: Mapping[tuple, dict], b: Mapping[tuple, dict], envs) -> dict:
    """Median over environments of the per-task difference A - B, in seconds (rules only)."""
    A, B = ({e: v for (e, _s), v in x.items()} for x in (a, b))
    return {m: median(A[e][m] - B[e][m] for e in envs) for m in PARTS}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--jb2-dir", default=None)
    ap.add_argument("--selection", default="simulation_data/joint_burst_v1/selected.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    sel = json.load(open(a.selection))
    envs = [tuple(e) for e in sel["environments"]]
    envs = [(int(c), int(t), str(w)) for c, t, w in envs]
    if len(envs) != 16:
        print(f"FAIL: {len(envs)} environments in {a.selection}, the registered design is 16")
        return 1
    keep = set(envs)
    arms = {k: {key: v for key, v in arm.items() if key[0] in keep} for k, arm in _load(a.gate_dir).items()}
    missing = [k for k in RULES if {e for e, _s in arms.get(k, {})} != keep]
    graph = arms.get(GRAPH, {})
    graph_missing = [(e, s) for e in envs for s in SEEDS if (e, s) not in graph]
    if missing or graph_missing:
        print(f"FAIL: incomplete gate -- rule arms short: {missing}; gnnedge0 cells missing: "
              f"{len(graph_missing)} (first {graph_missing[:3]})")
        return 1

    el = {k: _by_env(arms[k]) for k in RULES}
    g = _metric(graph)
    R: dict = {"gate_dir": a.gate_dir, "environments": [list(e) for e in envs], "seeds": list(SEEDS)}

    def ckpt(ref):
        return pair_checkpoint_stats(g, broadcast_rule(el[ref], SEEDS), envs)

    R["S1"] = _one_sample(ckpt(SELFPREDICT), min_n=len(SEEDS), faster_a="GRAPH-BEATS-SELFPREDICT",
                          faster_b="SELFPREDICT-FASTER", tie=V_NOT_SEP)
    R["S2"] = _one_sample(env_stats(el[SELFPREDICT], el[CD], envs), min_n=P_MIN_ENVIRONMENTS,
                          faster_a="SELFPREDICT-BEATS-CD", faster_b="CD-FASTER", tie=V_NOT_SEP)
    R["S3"] = _one_sample(env_stats(el[SELFPREDICT], el[BATCHED], envs), min_n=P_MIN_ENVIRONMENTS,
                          faster_a="SELFPREDICT-BEATS-1PASS", faster_b="1PASS-FASTER", tie=V_NOT_SEP)
    R["R_graph_vs_1pass"] = _one_sample(ckpt(BATCHED), min_n=len(SEEDS), faster_a="GRAPH-BEATS-GREEDY",
                                        faster_b="GREEDY-FASTER", tie=V_NOT_SEP)
    R["R_graph_vs_cd"] = _one_sample(ckpt(CD), min_n=len(SEEDS), faster_a="GRAPH-BEATS-CD",
                                     faster_b="CD-FASTER", tie=V_NOT_SEP)
    R["R_graph_vs_reactive"] = _one_sample(ckpt(REACTIVE), min_n=len(SEEDS), faster_a="GRAPH-BEATS-REACTIVE",
                                           faster_b="REACTIVE-FASTER", tie=V_NOT_SEP)
    for ref, label in ((IMMEDIATE, "immediate"), (REACTIVE, "reactive"), (RANDOM, "random")):
        R[f"D_selfpredict_vs_{label}"] = _one_sample(
            env_stats(el[SELFPREDICT], el[ref], envs), min_n=P_MIN_ENVIRONMENTS,
            faster_a=f"SELFPREDICT-BEATS-{label.upper()}", faster_b=f"{label.upper()}-FASTER", tie=V_NOT_SEP)
    R["D_cd_vs_1pass"] = _one_sample(env_stats(el[CD], el[BATCHED], envs), min_n=P_MIN_ENVIRONMENTS,
                                     faster_a="CD-BEATS-1PASS", faster_b="1PASS-FASTER", tie=V_NOT_SEP)
    R["parts_s"] = {f"selfpredict-{ref}": _parts(arms[SELFPREDICT], arms[ref], envs)
                    for ref in (CD, BATCHED, IMMEDIATE, REACTIVE)}
    gp = {m: median(v[m] for v in graph.values()) for m in ("elapsed",) + PARTS}
    R["median_per_task_s"] = {k: {m: median(v[m] for v in arms[k].values()) for m in ("elapsed",) + PARTS}
                              for k in RULES}
    R["median_per_task_s"][GRAPH] = gp

    if a.jb2_dir:
        old = _load(a.jb2_dir, jb2=True)
        diffs, n = [], 0
        for k, arm in arms.items():
            for key, v in arm.items():
                if k in old and key in old[k]:
                    n += 1
                    diffs.append(abs(v["elapsed"] - old[k][key]["elapsed"]))
        R["R_cells"] = {"compared": n, "max_abs_elapsed_diff_s": max(diffs) if diffs else None,
                        "identical": sum(d == 0.0 for d in diffs)}

    s1 = R["S1"]["verdict"]
    R["verdict"] = {"GRAPH-BEATS-SELFPREDICT": "GNN-BEATS-SELFPREDICT",
                    "SELFPREDICT-FASTER": "SELFPREDICT-AHEAD-OF-GNN"}.get(s1, "GNN-TIES-SELFPREDICT")
    R["burst_seat_bar"] = {"CD-FASTER": "CD", "SELFPREDICT-BEATS-CD": "SELFPREDICT"}.get(
        R["S2"]["verdict"], "CD~SELFPREDICT")

    print("=== selfpredict_burst_v1 LIVE GATE (burst seat, 16 env, C40) ===")
    print(f"  gnnedge0 checkpoints: {len(SEEDS)} x 16 env; rule arms 16 env each")
    rows = (("S1", "gnnedge0 vs selfpredict [ckpt]"), ("S2", "selfpredict vs CD greedy [env]"),
            ("S3", "selfpredict vs 1-pass greedy [env]"),
            ("R_graph_vs_1pass", f"R gnnedge0 vs 1-pass (jb2 {JB2['vs_1pass']:+.2f}%)"),
            ("R_graph_vs_cd", f"R gnnedge0 vs CD (jb2 {JB2['vs_cd']:+.2f}%)"),
            ("R_graph_vs_reactive", "-- gnnedge0 vs reactive [ckpt]"),
            ("D_selfpredict_vs_immediate", "-- selfpredict vs immediate rule"),
            ("D_selfpredict_vs_reactive", "-- selfpredict vs knative_network"),
            ("D_selfpredict_vs_random", "-- selfpredict vs random"),
            ("D_cd_vs_1pass", "-- CD vs 1-pass greedy"))
    for key, label in rows:
        r = R[key]
        print(f"  {label:<40} {r['verdict']:<28} median {r['median']:+7.2f}%  p {r['p']:.4g}  "
              f"ahead {r['ahead']}/{r['n']}")
    print("  per-task deltas, selfpredict minus arm (s): " + "; ".join(
        f"{k.split('-', 1)[1]}: " + " ".join(f"{m} {v:+.2f}" for m, v in d.items())
        for k, d in R["parts_s"].items()))
    print("  median per task (s): " + "  ".join(f"{k}={v['elapsed']:.2f}" for k, v in R["median_per_task_s"].items()))
    if "R_cells" in R:
        c = R["R_cells"]
        print(f"  R cells vs jb2 gate: {c['compared']} compared, {c['identical']} identical, "
              f"max |d elapsed| {c['max_abs_elapsed_diff_s']}")
    print(f"\nVERDICT: {R['verdict']}   BURST-SEAT BAR: {R['burst_seat_bar']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        tmp = a.out + ".partial"
        json.dump(R, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""selfpredict_bar_v1 -- read the live gate that sizes the new hand-rule bar.
See docs/lineages/selfpredict_bar_v1.md.

Pairs peer_greedy_selfpredict_network against each other arm PER (rung, env) over the 16
unsaturated_edge_v1 environments per rung, primary metric total_rtt, with the per-task
decomposition (wait / queue / exchange / rendezvous) alongside.

Registered bars (signed 2026-09-24, before any data; chain 5 % / p<0.05 / n=16):
  B1 vs reactive (knative_network, peer_greedy_live_v1's baseline)
                                        SELFPREDICT-BEATS-REACTIVE / REACTIVE-FASTER / NOT-SEPARATED
  -- vs ECT (knative_network_ect)       disclosed; rollout_imitation_v1 called this arm "reactive"
  B2 vs rule (peer_greedy_network)      REPLICATES if <= -5 % and p < 0.05 on BOTH rungs (P0b in a
                                        fresh run at one commit), else NOT-REPLICATED
  B3 vs CD greedy (peer_greedy_network_cd, batched seat)
                                        SELFPREDICT-BEATS-CD / CD-FASTER / NOT-SEPARATED
  -- vs random (random_network)         disclosed
  sanity (disclosed): rule vs reactive against peer_greedy_live_v1's -12.96 % / -16.01 %
  BAR verdict:
    NOT-REPLICATED                       if B2 fails -- the bar stays peer_greedy_network
    BAR=SELFPREDICT                      if B2 replicates and B3 is never CD-FASTER
    BAR=CD-BATCHED/SELFPREDICT-PER-ARRIVAL  if B2 replicates and B3 is CD-FASTER on a rung

Usage:
  python3 scripts_cosim/selfpredict_bar_v1_read.py \\
      --gate-dir simulation_data/peer_affinity_live_gate/results/selfpredict_bar_v1_gate \\
      --out      simulation_data/selfpredict_bar_v1/read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from statistics import median
from typing import Dict, Optional, Sequence, Tuple

PRIMARY = "peer_greedy_selfpredict_network"
RULE = "peer_greedy_network"
CD = "peer_greedy_network_cd"
REACTIVE = "knative_network"
ECT = "knative_network_ect"
RANDOM = "random_network"
ARMS = (PRIMARY, RULE, CD, REACTIVE, ECT, RANDOM)
THRESHOLD = 5.0
N_TASKS = 50000
PER_TASK = {"totalPeerExchangeTime", "totalPeerRendezvousWait"}
PARTS = ("averageWaitTime", "averageQueueTime", "totalPeerExchangeTime", "totalPeerRendezvousWait")
SANITY = {"C40": -12.96, "C80": -16.01}


def _load(gate_dir: str) -> Dict[str, Dict[Tuple[str, str, str], dict]]:
    out: Dict[str, Dict[Tuple[str, str, str], dict]] = {}
    for f in sorted(glob.glob(os.path.join(gate_dir, "cc*__*__*_s0.summary.json"))):
        s = json.load(open(f))
        if s.get("arm_kind") in ARMS:
            out.setdefault(s["arm_kind"], {})[(str(s["rung"]), str(s["topology"]), str(s["window"]))] = s
    return out


def _p(k: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    kk = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(kk + 1)) / (2 ** n))


def _val(s: dict, m: str) -> Optional[float]:
    v = s.get(m)
    return None if v is None else (float(v) / N_TASKS if m in PER_TASK else float(v))


def _contrast(a: Dict, b: Dict, m: str) -> dict:
    pct, absd, wins = [], [], 0
    for key in sorted(set(a) & set(b)):
        x, y = _val(a[key], m), _val(b[key], m)
        if x is None or y is None or y == 0.0:
            continue
        pct.append(100.0 * (x - y) / y)
        absd.append(x - y)
        wins += 1 if x < y else 0
    return {"n": len(pct), "median_pct": median(pct) if pct else None,
            "median_abs": median(absd) if absd else None, "a_faster": wins, "sign_p": _p(wins, len(pct))}


def _verdict(c: dict, names: Tuple[str, str, str]) -> str:
    mp, p = c["median_pct"], c["sign_p"]
    if mp is None or c["n"] == 0:
        return "NO-DATA"
    if mp <= -THRESHOLD and p is not None and p < 0.05:
        return names[0]
    if mp >= THRESHOLD and p is not None and p < 0.05:
        return names[1]
    return names[2]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    arms = _load(a.gate_dir)
    missing = [k for k in ARMS if k not in arms]
    if missing:
        print(f"FAIL: no summaries for {missing} under {a.gate_dir}")
        return 1

    report: dict = {"gate_dir": a.gate_dir, "rungs": {}}
    b2_ok, b3_cd_faster = [], []
    print("=== selfpredict_bar_v1 LIVE GATE ===")
    for rung in ("C40", "C80"):
        sub = {k: {e: s for e, s in arms[k].items() if e[0] == rung} for k in ARMS}
        rr: dict = {"n_env": {k: len(v) for k, v in sub.items()}}
        spec = (("B1", "reactive", REACTIVE, ("SELFPREDICT-BEATS-REACTIVE", "REACTIVE-FASTER", "NOT-SEPARATED")),
                ("B2", "rule", RULE, ("SELFPREDICT-BEATS-RULE", "RULE-FASTER", "NOT-SEPARATED")),
                ("B3", "cd", CD, ("SELFPREDICT-BEATS-CD", "CD-FASTER", "NOT-SEPARATED")),
                ("--", "ect", ECT, ("SELFPREDICT-BEATS-ECT", "ECT-FASTER", "NOT-SEPARATED")),
                ("--", "random", RANDOM, ("SELFPREDICT-BEATS-RANDOM", "RANDOM-FASTER", "NOT-SEPARATED")))
        print(f"\n-- {rung} (envs: {rr['n_env']}) --")
        for bar, label, kind, names in spec:
            c = _contrast(sub[PRIMARY], sub[kind], "total_rtt")
            c["verdict"] = _verdict(c, names)
            c["parts"] = {m: _contrast(sub[PRIMARY], sub[kind], m) for m in PARTS}
            rr[label] = c
            parts = "  ".join(f"{m.replace('average','').replace('totalPeer','').replace('Time','')} "
                              f"{c['parts'][m]['median_abs']:+.2f}s" for m in PARTS
                              if c["parts"][m]["median_abs"] is not None)
            print(f"  {bar} vs {label:9s} n={c['n']:2d} median={c['median_pct']:+7.2f}% "
                  f"faster={c['a_faster']}/{c['n']} p={c['sign_p']:.4f} -> {c['verdict']}   [{parts}]")
        b2_ok.append(rr["rule"]["verdict"] == "SELFPREDICT-BEATS-RULE")
        b3_cd_faster.append(rr["cd"]["verdict"] == "CD-FASTER")
        san = _contrast(sub[RULE], sub[REACTIVE], "total_rtt")
        rr["sanity_rule_vs_reactive"] = san
        print(f"  sanity rule vs reactive median={san['median_pct']:+.2f}% (peer_greedy_live_v1: {SANITY[rung]:+.2f}%) "
              f"faster={san['a_faster']}/{san['n']}")
        rr["elapsed_median_s"] = {k: median(float(s["averageElapsedTime"]) for s in sub[k].values()) for k in ARMS}
        print("  median elapsed s/task: " + "  ".join(f"{k}={v:.2f}" for k, v in rr["elapsed_median_s"].items()))
        report["rungs"][rung] = rr

    if not all(b2_ok):
        verdict = "NOT-REPLICATED"
    elif any(b3_cd_faster):
        verdict = "BAR=CD-BATCHED/SELFPREDICT-PER-ARRIVAL"
    else:
        verdict = "BAR=SELFPREDICT"
    report["verdict"] = verdict
    print(f"\nVERDICT: {verdict}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

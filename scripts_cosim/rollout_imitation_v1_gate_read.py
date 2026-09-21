#!/usr/bin/env python3
"""rollout_imitation_v1 -- read the LIVE GATE and fire its signed bars. See
docs/lineages/rollout_imitation_v1.md.

Pairs the learned arm (peer_greedy_learned_network) against each other arm PER (rung, env) over
the 16 unsaturated_edge_v1 environments per rung, on the primary metric total_rtt (and reports
averageElapsedTime alongside). For each contrast: the paired median %Delta (learned - other)/other,
the sign count (envs where learned is faster) out of n, and a two-sided exact binomial sign-test p.

Bars (node R2-R4; chain 5% / p<0.05 / n=16):
  R2 learned vs rule (peer_greedy_network) : MODEL-BEATS-RULE / RULE-FASTER / NOT-SEPARATED
  R3 learned vs reactive (knative_network_ect)
  --  learned vs CD greedy (peer_greedy_network_cd, the honest ceiling, disclosed)
  --  learned vs random (random_network)

Usage (datalab or local after rsync, PYTHONPATH=.):
  python3 scripts_cosim/rollout_imitation_v1_gate_read.py \
      --gate-dir simulation_data/peer_affinity_live_gate/results/rollout_imitation_v1_gate \
      --out      simulation_data/rollout_imitation_v1/gate_read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

LEARNED = "peer_greedy_learned_network"
CONTRASTS = [
    ("rule", "peer_greedy_network", "R2", ("MODEL-BEATS-RULE", "RULE-FASTER", "NOT-SEPARATED")),
    ("reactive", "knative_network_ect", "R3", ("MODEL-BEATS-REACTIVE", "REACTIVE-FASTER", "NOT-SEPARATED")),
    ("cd_greedy", "peer_greedy_network_cd", "--", ("MODEL-BEATS-CD", "CD-FASTER", "NOT-SEPARATED")),
    ("random", "random_network", "--", ("MODEL-BEATS-RANDOM", "RANDOM-FASTER", "NOT-SEPARATED")),
]
METRICS = ("total_rtt", "averageElapsedTime")


def _arm_of(fname: str) -> Optional[Tuple[str, str, str]]:
    """(cell, window, arm_kind) from '<cell>__<window>__<kind>_s0.summary.json'."""
    base = os.path.basename(fname)
    if not base.endswith("_s0.summary.json"):
        return None
    stem = base[: -len(".summary.json")]
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    cell, window, kind_s0 = parts
    if not kind_s0.endswith("_s0"):
        return None
    return cell, window, kind_s0[: -len("_s0")]


def _load(gate_dir: str) -> Dict[str, Dict[Tuple[str, str, str], dict]]:
    """arm_kind -> {(rung, topo, window) -> summary}."""
    out: Dict[str, Dict[Tuple[str, str, str], dict]] = {}
    for f in sorted(glob.glob(os.path.join(gate_dir, "cc*__*__*_s0.summary.json"))):
        parsed = _arm_of(f)
        if parsed is None:
            continue
        _cell, _window, kind = parsed
        s = json.load(open(f))
        key = (str(s.get("rung")), str(s.get("topology")), str(s.get("window")))
        out.setdefault(kind, {})[key] = s
    return out


def _binom_two_sided_p(k: int, n: int) -> Optional[float]:
    """Exact two-sided sign test against p=0.5."""
    if n == 0:
        return None
    def cdf_le(x):
        return sum(math.comb(n, i) for i in range(0, x + 1)) / (2 ** n)
    kk = min(k, n - k)
    p = 2.0 * cdf_le(kk)
    return min(1.0, p)


def _contrast(learned: Dict, other: Dict, metric: str) -> dict:
    keys = sorted(set(learned) & set(other))
    deltas, wins, n = [], 0, 0
    for key in keys:
        lo, ot = learned[key].get(metric), other[key].get(metric)
        if lo is None or ot is None or float(ot) == 0.0:
            continue
        lo, ot = float(lo), float(ot)
        deltas.append(100.0 * (lo - ot) / ot)
        wins += 1 if lo < ot else 0
        n += 1
    return {"n": n, "median_pct": median(deltas) if deltas else None,
            "learned_faster": wins, "sign_p": _binom_two_sided_p(wins, n)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=5.0, help="percent magnitude for a signed verdict")
    a = ap.parse_args(argv)

    arms = _load(a.gate_dir)
    if LEARNED not in arms:
        print(f"FAIL: no {LEARNED} summaries under {a.gate_dir}"); return 1

    report = {"gate_dir": a.gate_dir, "rungs": {}}
    print("=== rollout_imitation_v1 LIVE GATE ===")
    for rung in ("C40", "C80"):
        learned_rung = {k: v for k, v in arms[LEARNED].items() if k[0] == rung}
        if not learned_rung:
            continue
        print(f"\n-- {rung} (learned arm envs: {len(learned_rung)}) --")
        rr = {}
        for label, kind, bar, verdicts in CONTRASTS:
            other = {k: v for k, v in arms.get(kind, {}).items() if k[0] == rung}
            per_metric = {}
            for metric in METRICS:
                c = _contrast(learned_rung, other, metric)
                # verdict on the primary metric only
                if metric == METRICS[0]:
                    mp, p, n = c["median_pct"], c["sign_p"], c["n"]
                    if mp is None or n == 0:
                        v = "NO-DATA"
                    elif mp <= -a.threshold and p is not None and p < 0.05:
                        v = verdicts[0]
                    elif mp >= a.threshold and p is not None and p < 0.05:
                        v = verdicts[1]
                    else:
                        v = verdicts[2]
                    c["verdict"] = v
                per_metric[metric] = c
            rr[label] = {"kind": kind, "bar": bar, **per_metric}
            pm = per_metric[METRICS[0]]
            print(f"  {label:9s} vs {kind:28s} [{METRICS[0]}] n={pm['n']:2d} "
                  f"median={pm['median_pct'] if pm['median_pct'] is None else round(pm['median_pct'],2)}% "
                  f"faster={pm['learned_faster']}/{pm['n']} "
                  f"p={pm['sign_p'] if pm['sign_p'] is None else round(pm['sign_p'],4)} -> {pm['verdict']}")
        report["rungs"][rung] = rr

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

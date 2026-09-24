#!/usr/bin/env python3
"""lookahead_mp_v1 P0 -- read the live headroom gate. See docs/lineages/lookahead_mp_v1.md.

Pairs the arms PER (rung, env) over the 16 unsaturated_edge_v1 environments per rung, primary
metric total_rtt; reports the RTT decomposition alongside (queue / exchange / rendezvous per task).

Registered bars (signed 2026-09-24, before any P0 data; chain 5 % / p<0.05 / n=16):
  H1 oracle vs rule    -- HEADROOM              median <= -5 % and sign p < 0.05
                          ORACLE-SLOWER          median >= +5 % and sign p < 0.05
                          NO-HEADROOM-FOR-A-5%-GAP  otherwise
  P0 verdict           -- GO-P1 if H1 reads HEADROOM on at least one rung, else STOP
                          (lookahead cannot carry the 5 % model-class bar the lineage is built on)
  H2 lookahead vs rule -- same thresholds (HAND-BEATS-RULE / RULE-FASTER / NOT-SEPARATED), plus
                          recovered = median(lookahead - rule) / median(oracle - rule) on total_rtt;
                          HAND-RECOVERS if recovered >= 0.80 on a HEADROOM rung (the room left for
                          message passing over a pointwise two-step feature is then <= 20 % of it)
  mechanism (reported, not gated) -- the oracle's gain should sit in exchange per task

P0b (signed 2026-09-24, before any P0b data; read only when selfpredict summaries exist):
  S1 selfpredict vs rule -- same thresholds (COORD-BEATS-RULE / RULE-FASTER / NOT-SEPARATED), plus
                            recovered_b = median(selfpredict - rule) / median(oracle - rule)
  P1 verdict             -- STOP-P1 (HAND-COORDINATION-RECOVERS) if recovered_b >= 0.80 on EVERY
                            HEADROOM rung; otherwise P1-GO, with selfpredict as a P4 control

Usage (local after rsync of the summaries, or on datalab):
  python3 scripts_cosim/lookahead_mp_v1_p0_read.py \\
      --gate-dir simulation_data/peer_affinity_live_gate/results/lookahead_mp_v1_p0 \\
      --out      simulation_data/lookahead_mp_v1/p0_read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from statistics import median
from typing import Dict, Optional, Sequence, Tuple

RULE = "peer_greedy_network"
LOOKAHEAD = "peer_greedy_lookahead_network"
ORACLE = "peer_greedy_oracle_network"
SELFPREDICT = "peer_greedy_selfpredict_network"
THRESHOLD = 5.0
RECOVERS = 0.80
N_TASKS = 50000
PER_TASK = {"totalPeerExchangeTime", "totalPeerRendezvousWait"}
COMPONENTS = ("averageQueueTime", "totalPeerExchangeTime", "totalPeerRendezvousWait")


def _load(gate_dir: str) -> Dict[str, Dict[Tuple[str, str, str], dict]]:
    out: Dict[str, Dict[Tuple[str, str, str], dict]] = {}
    for f in sorted(glob.glob(os.path.join(gate_dir, "cc*__*__*_s0.summary.json"))):
        s = json.load(open(f))
        kind = s.get("arm_kind")
        if kind not in (RULE, LOOKAHEAD, ORACLE, SELFPREDICT):
            continue
        key = (str(s["rung"]), str(s["topology"]), str(s["window"]))
        out.setdefault(kind, {})[key] = s
    return out


def _binom_two_sided_p(k: int, n: int) -> Optional[float]:
    if n == 0:
        return None
    kk = min(k, n - k)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(kk + 1)) / (2 ** n))


def _val(s: dict, metric: str) -> Optional[float]:
    v = s.get(metric)
    if v is None:
        return None
    return float(v) / N_TASKS if metric in PER_TASK else float(v)


def _contrast(a: Dict, b: Dict, metric: str) -> dict:
    """a - b, paired on env: median %Delta, median absolute Delta, a-faster count, sign p."""
    pct, absd, wins = [], [], 0
    for key in sorted(set(a) & set(b)):
        x, y = _val(a[key], metric), _val(b[key], metric)
        if x is None or y is None or y == 0.0:
            continue
        pct.append(100.0 * (x - y) / y)
        absd.append(x - y)
        wins += 1 if x < y else 0
    n = len(pct)
    return {"n": n, "median_pct": median(pct) if pct else None,
            "median_abs": median(absd) if absd else None,
            "a_faster": wins, "sign_p": _binom_two_sided_p(wins, n)}


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
    for kind in (RULE, LOOKAHEAD, ORACLE):
        if kind not in arms:
            print(f"FAIL: no {kind} summaries under {a.gate_dir}")
            return 1

    report: dict = {"gate_dir": a.gate_dir, "rungs": {}}
    headroom_rungs, recovered_on_headroom, recovered_b_on_headroom = [], [], []
    print("=== lookahead_mp_v1 P0 LIVE HEADROOM GATE ===")
    for rung in ("C40", "C80"):
        sub = {k: {e: s for e, s in arms[k].items() if e[0] == rung} for k in arms}
        n_env = {k: len(v) for k, v in sub.items() if k != SELFPREDICT}
        if not all(n_env.values()):
            continue
        rr: dict = {"n_env": n_env}
        h1 = _contrast(sub[ORACLE], sub[RULE], "total_rtt")
        h1["verdict"] = _verdict(h1, ("HEADROOM", "ORACLE-SLOWER", "NO-HEADROOM-FOR-A-5%-GAP"))
        h2 = _contrast(sub[LOOKAHEAD], sub[RULE], "total_rtt")
        h2["verdict"] = _verdict(h2, ("HAND-BEATS-RULE", "RULE-FASTER", "NOT-SEPARATED"))
        ol = _contrast(sub[ORACLE], sub[LOOKAHEAD], "total_rtt")
        recovered = None
        if h1["median_abs"] and h1["median_abs"] < 0 and h2["median_abs"] is not None:
            recovered = h2["median_abs"] / h1["median_abs"]
        h2["recovered_fraction"] = recovered
        if h1["verdict"] == "HEADROOM":
            headroom_rungs.append(rung)
            recovered_on_headroom.append(recovered)
        rr["H1_oracle_vs_rule"] = h1
        rr["H2_lookahead_vs_rule"] = h2
        rr["oracle_vs_lookahead"] = ol
        rr["components"] = {
            label: {m: _contrast(sub[x], sub[RULE], m) for m in COMPONENTS}
            for label, x in (("oracle", ORACLE), ("lookahead", LOOKAHEAD))
        }
        rr["counters_median"] = {
            k: {c: median([float((s.get("schedulerCounters") or {}).get(c, 0)) for s in sub[k].values()])
                for c in ("pg_joined_partner", "pg_moved_by_exchange", "pg_lookahead_priced", "pg_lookahead_blind")}
            for k in (RULE, LOOKAHEAD, ORACLE)
        }
        if sub.get(SELFPREDICT):
            s1 = _contrast(sub[SELFPREDICT], sub[RULE], "total_rtt")
            s1["verdict"] = _verdict(s1, ("COORD-BEATS-RULE", "RULE-FASTER", "NOT-SEPARATED"))
            rb = None
            if h1["median_abs"] and h1["median_abs"] < 0 and s1["median_abs"] is not None:
                rb = s1["median_abs"] / h1["median_abs"]
            s1["recovered_fraction"] = rb
            s1["n_env"] = len(sub[SELFPREDICT])
            rr["S1_selfpredict_vs_rule"] = s1
            rr["selfpredict_vs_oracle"] = _contrast(sub[SELFPREDICT], sub[ORACLE], "total_rtt")
            rr["components"]["selfpredict"] = {m: _contrast(sub[SELFPREDICT], sub[RULE], m) for m in COMPONENTS}
            if h1["verdict"] == "HEADROOM":
                recovered_b_on_headroom.append(rb)
        report["rungs"][rung] = rr

        def line(tag, c):
            mp = "n/a" if c["median_pct"] is None else f"{c['median_pct']:+.2f}%"
            p = "n/a" if c["sign_p"] is None else f"{c['sign_p']:.4f}"
            return f"  {tag:24s} n={c['n']:2d} median={mp:>8s} faster={c['a_faster']}/{c['n']} p={p}"
        print(f"\n-- {rung} (envs: {n_env}) --")
        print(line("H1 oracle vs rule", h1) + f" -> {h1['verdict']}")
        print(line("H2 lookahead vs rule", h2) + f" -> {h2['verdict']}"
              + ("" if recovered is None else f"  recovered={recovered:.2f}"))
        print(line("oracle vs lookahead", ol))
        for label in ("oracle", "lookahead"):
            comp = rr["components"][label]
            parts = "  ".join(
                f"{m.replace('averageQueueTime','queue').replace('totalPeerExchangeTime','exch').replace('totalPeerRendezvousWait','rendez')}"
                f" {comp[m]['median_abs']:+.3f}s/task" for m in COMPONENTS if comp[m]["median_abs"] is not None)
            print(f"  [{label} - rule] {parts}")
        if "S1_selfpredict_vs_rule" in rr:
            s1 = rr["S1_selfpredict_vs_rule"]
            print(line("S1 selfpredict vs rule", s1) + f" -> {s1['verdict']}"
                  + ("" if s1["recovered_fraction"] is None else f"  recovered_b={s1['recovered_fraction']:.2f}"))
            print(line("selfpredict vs oracle", rr["selfpredict_vs_oracle"]))
            comp = rr["components"]["selfpredict"]
            print("  [selfpredict - rule] " + "  ".join(
                f"{m} {comp[m]['median_abs']:+.3f}" for m in COMPONENTS if comp[m]["median_abs"] is not None))
        for k in (RULE, LOOKAHEAD, ORACLE):
            print(f"  counters {k:30s} {rr['counters_median'][k]}")

    verdict = "GO-P1" if headroom_rungs else "STOP"
    hand = "HAND-RECOVERS" if any(r is not None and r >= RECOVERS - 1e-9 for r in recovered_on_headroom) else None
    report["P0_verdict"] = verdict
    report["headroom_rungs"] = headroom_rungs
    report["hand_flag"] = hand
    print(f"\nP0 VERDICT: {verdict}  (HEADROOM on {headroom_rungs or 'no rung'})" + (f"  [{hand}]" if hand else ""))
    if SELFPREDICT in arms and headroom_rungs:
        stop = all(r is not None and r >= RECOVERS - 1e-9 for r in recovered_b_on_headroom) \
            and len(recovered_b_on_headroom) == len(headroom_rungs)
        report["P1_verdict"] = "STOP-P1 (HAND-COORDINATION-RECOVERS)" if stop else "P1-GO"
        print(f"P1 VERDICT (P0b): {report['P1_verdict']}  recovered_b on HEADROOM rungs = "
              + str([None if r is None else round(r, 3) for r in recovered_b_on_headroom]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

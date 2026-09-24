#!/usr/bin/env python3
"""fresh_topo_burst_v1 -- select the study topologies from the screen, then read the gate.
See docs/lineages/fresh_topo_burst_v1.md.

  fresh_topo_burst_v1_read.py select --screen DIR --out selected.json
  fresh_topo_burst_v1_read.py gate   --gate DIR --selection selected.json [--out read.json]

The unit is the TOPOLOGY. Per environment (topology, window) a learned arm's value is the median over
its 13 checkpoints of the paired % against the reference (gnnedge0 vs mpoff pairs the same seed);
per topology it is the median over its 4 windows; the test is a two-sided exact Wilcoxon signed-rank
over topologies. Negative = the first-named arm is faster.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from statistics import median
from typing import Dict, List, Optional, Tuple

from scipy.stats import wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts_cosim"))
from fresh_topo_burst_v1_gate import CANDIDATES, GATE_RULES, SEEDS, WINDOWS  # noqa: E402

TARGET_SHARE = 0.80
N_STUDY = 12
MIN_TOPOLOGIES = 8
BAR = 5.0
ALPHA = 0.05


def _summaries(d: str) -> Dict[Tuple[int, str, str, int], dict]:
    out = {}
    for f in glob.glob(os.path.join(d, "*.summary.json")):
        r = json.load(open(f))
        kind = r["arm"].split("__")[2].rsplit("_s", 1)[0]
        out[(int(r["topology"]), r["window"], kind, int(r["checkpoint_seed"]))] = r
    return out


def select(screen: str) -> dict:
    s = _summaries(screen)

    def ok(t: int) -> bool:
        for w in WINDOWS:
            r, b = s.get((t, w, "reactive", 0)), s.get((t, w, "batched", 0))
            if r is None or b is None or r.get("queue_share") is None or float(r["queue_share"]) > TARGET_SHARE:
                return False
        return True

    qualified = [t for t in CANDIDATES if ok(t)]
    failed = sorted(os.path.basename(f) for f in glob.glob(os.path.join(screen, "*.failed.json")))
    base = {"qualified": qualified, "n_candidates": len(CANDIDATES), "failed_runs": failed,
            "rule": f"reactive queue share <= {TARGET_SHARE} and the batch path finishes, all 4 windows; "
                    f"the {N_STUDY} lowest qualified ids are the study (unknown is not a pass)"}
    if len(qualified) < MIN_TOPOLOGIES:
        return dict(base, verdict="DESIGN-SHORT")
    return dict(base, verdict="DESIGN-READY", topologies=sorted(qualified)[:N_STUDY])


def _pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def _el(r: dict) -> float:
    return float(r["averageElapsedTime"])


def _read(vals: Dict[int, float], a: str, b: str) -> dict:
    xs = [vals[t] for t in sorted(vals)]
    p = float(wilcoxon(xs, method="exact").pvalue) if any(xs) else 1.0
    med = median(xs)
    faster = sum(x < 0 for x in xs)
    if p < ALPHA and med < 0:
        verdict = f"{a}-FASTER" + ("" if med <= -BAR else " (direction only, |median| < 5 %)")
    elif p < ALPHA and med > 0:
        verdict = f"{b}-FASTER" + ("" if med >= BAR else " (direction only, |median| < 5 %)")
    else:
        verdict = "NOT-SEPARATED"
    return {"median_pct": med, "p": p, "n": len(xs), "a_faster": faster, "verdict": verdict,
            "per_topology": {str(t): vals[t] for t in sorted(vals)}}


def gate(gate_dir: str, selection: dict) -> dict:
    s = _summaries(gate_dir)
    topos = selection["topologies"]
    complete, dropped = [], {}
    for t in topos:
        missing = [(w, k) for w in WINDOWS for k in GATE_RULES if (t, w, k, 0) not in s]
        missing += [(w, k, sd) for w in WINDOWS for k in ("gnnedge0", "mpoff") for sd in SEEDS if (t, w, k, sd) not in s]
        if missing:
            dropped[str(t)] = missing[:6] + (["..."] if len(missing) > 6 else [])
        else:
            complete.append(t)
    codes = {(r["code"]["commit"], r["code"]["dirty"], r["code"].get("diff_sha256")) for r in s.values()}
    out = {"topologies": complete, "dropped": dropped, "code_states": [list(c) for c in codes]}
    if len(codes) != 1 or next(iter(codes))[1]:
        out["verdict"] = "INVALID-CODE-STATE"
        return out
    if len(complete) < MIN_TOPOLOGIES:
        out["verdict"] = "DESIGN-SHORT"
        return out

    def rule(t, w, k):
        return _el(s[(t, w, k, 0)])

    def env_learned(t, w, arm, ref_rule=None, ref_arm=None):
        ds = []
        for sd in SEEDS:
            ref = rule(t, w, ref_rule) if ref_rule else _el(s[(t, w, ref_arm, sd)])
            ds.append(_pct(_el(s[(t, w, arm, sd)]), ref))
        return median(ds)

    def topo_vals(fn) -> Dict[int, float]:
        return {t: median(fn(t, w) for w in WINDOWS) for t in complete}

    out["F1_gnnedge0_vs_mpoff"] = _read(topo_vals(lambda t, w: env_learned(t, w, "gnnedge0", ref_arm="mpoff")),
                                        "GNNEDGE0", "MPOFF")
    out["F2_gnnedge0_vs_selfpredict"] = _read(topo_vals(lambda t, w: env_learned(t, w, "gnnedge0", ref_rule="selfpredict")),
                                              "GNNEDGE0", "SELFPREDICT")
    out["F3_mpoff_vs_selfpredict"] = _read(topo_vals(lambda t, w: env_learned(t, w, "mpoff", ref_rule="selfpredict")),
                                           "MPOFF", "SELFPREDICT")
    out["D_gnnedge0_vs_cd"] = _read(topo_vals(lambda t, w: env_learned(t, w, "gnnedge0", ref_rule="cd")), "GNNEDGE0", "CD")
    out["D_selfpredict_vs_cd"] = _read(topo_vals(lambda t, w: _pct(rule(t, w, "selfpredict"), rule(t, w, "cd"))),
                                       "SELFPREDICT", "CD")
    out["D_gnnedge0_vs_1pass"] = _read(topo_vals(lambda t, w: env_learned(t, w, "gnnedge0", ref_rule="batched")),
                                       "GNNEDGE0", "1PASS")
    out["D_gnnedge0_vs_reactive"] = _read(topo_vals(lambda t, w: env_learned(t, w, "gnnedge0", ref_rule="reactive")),
                                          "GNNEDGE0", "REACTIVE")
    f1 = out["F1_gnnedge0_vs_mpoff"]
    if f1["verdict"].startswith("GNNEDGE0-FASTER"):
        out["verdict"] = "MP-EDGE-GENERALISES" + ("" if f1["median_pct"] <= -BAR else " (direction only)")
    elif f1["verdict"].startswith("MPOFF-FASTER"):
        out["verdict"] = "MPOFF-FASTER-ON-FRESH-TOPOLOGIES"
    else:
        out["verdict"] = "MP-EDGE-DOES-NOT-GENERALISE"
    return out


def _print(r: dict) -> None:
    print(f"topologies {r.get('topologies')}  dropped {list(r.get('dropped', {}))}")
    for k, v in r.items():
        if isinstance(v, dict) and "median_pct" in v:
            print(f"  {k:30s} median {v['median_pct']:+6.2f} %  p={v['p']:.4f}  first faster "
                  f"{v['a_faster']}/{v['n']}  -> {v['verdict']}")
            print("      per topology: " + ", ".join(f"{t}:{x:+.1f}" for t, x in v["per_topology"].items()))
    print(f"VERDICT: {r.get('verdict')}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("select", "gate"))
    ap.add_argument("--screen")
    ap.add_argument("--gate")
    ap.add_argument("--selection")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.phase == "select":
        r = select(a.screen)
        print(json.dumps({k: v for k, v in r.items() if k != "failed_runs"}, indent=1))
        print(f"failed screen runs: {len(r['failed_runs'])}")
    else:
        r = gate(a.gate, json.load(open(a.selection)))
        _print(r)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

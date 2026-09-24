#!/usr/bin/env python3
"""rollout_imitation_v1 Phase A read (R0, blocking): pool the per-topology label files and decide
whether the forced-rollout label is a property of the decision or downstream chaos.

R0 PASS iff over >= 300 pooled decisions: median Spearman(N=20 vs N=100) >= 0.80 AND the argmin
(the chosen candidate) agrees between N=20 and N=100 on >= 80% of decisions. Otherwise the label is
downstream chaos and the lineage closes LABEL-IS-CHAOS (as objective_pivot_v1 P3 did), nothing trained.

Also reports R1 preview (label-vs-rule differ rate + gain) where computable, for the A.2 gate.

Usage (datalab): python3 scripts_cosim/rollout_imitation_v1_phaseA_read.py \
    --labels-dir results/rollout_imitation_v1/labels --out results/rollout_imitation_v1/phaseA.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import median
from typing import Optional, Sequence

R0_MIN_DECISIONS = 300
R0_MIN_RHO = 0.80
R0_MIN_ARGMIN_AGREE = 0.80

# R1 (Phase A.2): the label must differ from the rule's own choice on >= 10% of decisions with a
# median realised gain >= 0.3 s where it differs, else the greedy is one-step optimal here.
R1_MIN_DIFFER_FRAC = 0.10
R1_MIN_MEDIAN_GAIN = 0.30
R1_HORIZON = 100

V_STABLE = "LABEL-IS-A-PROPERTY-OF-THE-DECISION"
V_CHAOS = "LABEL-IS-CHAOS"
V_UNDERPOWERED = "UNDERPOWERED"
V_DIFFERS = "LABEL-DIFFERS-FROM-RULE"
V_ONE_STEP_OPT = "GREEDY-IS-ONE-STEP-OPTIMAL"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True, nargs="+",
                    help="one or more dirs of s*.json label files (pooled)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    files = sorted(f for d in a.labels_dir for f in glob.glob(os.path.join(d, "s*.json")))
    decisions = []
    per_topo = {}
    for f in files:
        d = json.load(open(f))
        per_topo[d["seed"]] = d["n_decisions"]
        decisions.extend(d["decisions"])

    n = len(decisions)
    rhos_20_100 = [x["spearman"]["20_100"] for x in decisions if x["spearman"]["20_100"] is not None]
    rhos_50_100 = [x["spearman"]["50_100"] for x in decisions if x["spearman"]["50_100"] is not None]
    argmin_agree = [1 for x in decisions if x["argmins"]["20"] == x["argmins"]["100"]]
    agree_frac = (sum(argmin_agree) / n) if n else 0.0
    med_rho = median(rhos_20_100) if rhos_20_100 else None

    # R1 (Phase A.2): label vs the rule's OWN choice at the label horizon. differ + realised gain.
    have_rule = [x for x in decisions if x.get("rule_idx", -1) >= 0]
    gains, differ = [], 0
    for x in have_rule:
        c = x["costs"][str(R1_HORIZON)] if str(R1_HORIZON) in x["costs"] else x["costs"].get(R1_HORIZON)
        ri, ai = x["rule_idx"], x["argmins"][str(R1_HORIZON)] if str(R1_HORIZON) in x["argmins"] else x["argmins"][R1_HORIZON]
        if c is None:
            continue
        if ai != ri:
            differ += 1
            gains.append(c[ri] - c[ai])   # rule cost - label cost (>=0, the realised gain)
    nr = len(have_rule)
    differ_frac = (differ / nr) if nr else 0.0
    med_gain = median(gains) if gains else 0.0
    r1_pass = (differ_frac >= R1_MIN_DIFFER_FRAC and med_gain >= R1_MIN_MEDIAN_GAIN)
    r1_verdict = V_DIFFERS if r1_pass else V_ONE_STEP_OPT

    passed = (n >= R0_MIN_DECISIONS and med_rho is not None
              and med_rho >= R0_MIN_RHO and agree_frac >= R0_MIN_ARGMIN_AGREE)
    if n < R0_MIN_DECISIONS:
        verdict = V_UNDERPOWERED
    else:
        verdict = V_STABLE if passed else V_CHAOS

    R = {
        "verdict": verdict, "n_decisions": n, "n_topologies": len(files),
        "median_rho_20_100": med_rho,
        "median_rho_50_100": median(rhos_50_100) if rhos_50_100 else None,
        "argmin_agree_frac": agree_frac,
        "thresholds": {"min_decisions": R0_MIN_DECISIONS, "min_rho": R0_MIN_RHO,
                       "min_argmin_agree": R0_MIN_ARGMIN_AGREE},
        "r1": {"verdict": r1_verdict, "n_with_rule": nr, "differ_frac": differ_frac,
               "median_gain_s": med_gain, "n_differ": differ,
               "thresholds": {"min_differ_frac": R1_MIN_DIFFER_FRAC,
                              "min_median_gain_s": R1_MIN_MEDIAN_GAIN, "horizon": R1_HORIZON}},
        "per_topology_n": per_topo,
    }

    print("=== rollout_imitation_v1 Phase A (R0 rank stability) ===")
    print(f"  topologies: {len(files)}   pooled decisions: {n}")
    print(f"  median Spearman(20,100): {med_rho if med_rho is None else round(med_rho,3)}  (>= {R0_MIN_RHO})")
    print(f"  argmin(20==100) agreement: {agree_frac:.3f}  (>= {R0_MIN_ARGMIN_AGREE})")
    print(f"  R0 VERDICT: {verdict}")
    print("=== Phase A.2 (R1 label-vs-rule) ===")
    print(f"  decisions with a known rule choice: {nr}")
    print(f"  label differs from rule: {differ}/{nr} = {differ_frac:.3f}  (>= {R1_MIN_DIFFER_FRAC})")
    print(f"  median realised gain where differ: {med_gain:.3f}s  (>= {R1_MIN_MEDIAN_GAIN})")
    print(f"  R1 VERDICT: {r1_verdict}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

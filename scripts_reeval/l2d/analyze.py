"""literature_reeval_v1 -- registered statistics for the L2D arms (see docs/lineages/literature_reeval_v1.md).

Per size and checkpoint protocol:
  * per-seed mean makespan on the frozen `test` split for every arm;
  * Delta% = (arm - gnn) / gnn on those per-seed means, positive = arm worse than the GNN;
  * primary: two-sided exact Mann-Whitney U on per-seed means, gnn vs each other arm;
  * secondary: per-instance paired Wilcoxon on seed-averaged makespans (100 instances);
  * reading per the registered rules (GNN-NEEDED / TIE / POINTWISE-BETTER / INDETERMINATE).
Also tabulates PDRs and the upstream SavedNetwork on the same split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ARMS = ("gnn", "mpoff", "mlp_t1", "mlp_t1x")
DELTA_BAR = 1.0  # percent of GNN mean makespan
ALPHA = 0.05


def reading(delta_med, p):
    sig = p < ALPHA
    if sig and delta_med >= DELTA_BAR:
        return "GNN-NEEDED"
    if sig and delta_med <= -DELTA_BAR:
        return "POINTWISE-BETTER"
    if not sig and abs(delta_med) < DELTA_BAR:
        return "TIE"
    return "INDETERMINATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5, help="only arms trained at this lr (upstream default 2e-5)")
    a = ap.parse_args()
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(a.eval_dir, f"*__{a.split}.json"))]
    if not recs:
        raise SystemExit(f"no *__{a.split}.json under {a.eval_dir}")
    n_j, n_m = recs[0]["n_j"], recs[0]["n_m"]
    report = {"size": f"{n_j}x{n_m}", "split": a.split, "lr": a.lr, "delta_bar_pct": DELTA_BAR, "alpha": ALPHA,
              "reference": {}, "protocols": {}}
    for r in recs:
        if r["kind"] in ("pdr", "upstream_checkpoint"):
            report["reference"][r["subject"]] = r["mean_makespan"]
    for ck in ("best_val", "final"):
        per_arm = defaultdict(dict)  # arm -> seed -> makespans array
        for r in recs:
            if r["kind"] == "arm" and r["checkpoint"] == ck and abs(r.get("lr", 2e-5) - a.lr) < 1e-12:
                per_arm[r["arm"]][r["seed"]] = np.array(r["makespans"])
        if "gnn" not in per_arm:
            continue
        gnn_seeds = sorted(per_arm["gnn"])
        gnn_means = np.array([per_arm["gnn"][s].mean() for s in gnn_seeds])
        gnn_inst = np.mean([per_arm["gnn"][s] for s in gnn_seeds], axis=0)
        proto = {"n_seeds": {arm: len(v) for arm, v in per_arm.items()},
                 "per_seed_mean": {arm: {str(s): float(v[s].mean()) for s in sorted(v)} for arm, v in per_arm.items()},
                 "arm_mean": {arm: float(np.mean([v[s].mean() for s in v])) for arm, v in per_arm.items()},
                 "arm_sd_over_seeds": {arm: float(np.std([v[s].mean() for s in v], ddof=1)) if len(v) > 1 else None
                                       for arm, v in per_arm.items()},
                 "contrasts": {}}
        for arm in ARMS:
            if arm == "gnn" or arm not in per_arm:
                continue
            seeds = sorted(per_arm[arm])
            means = np.array([per_arm[arm][s].mean() for s in seeds])
            delta = (means - gnn_means.mean()) / gnn_means.mean() * 100.0
            mw = stats.mannwhitneyu(means, gnn_means, alternative="two-sided", method="exact")
            inst = np.mean([per_arm[arm][s] for s in seeds], axis=0)
            wil = stats.wilcoxon(inst, gnn_inst, alternative="two-sided") if np.any(inst != gnn_inst) else None
            proto["contrasts"][f"{arm}_vs_gnn"] = {
                "delta_pct_median": float(np.median(delta)), "delta_pct_mean": float(delta.mean()),
                "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
                "per_instance_wilcoxon_p": None if wil is None else float(wil.pvalue),
                "per_instance_frac_arm_better": float(np.mean(inst < gnn_inst)),
                "reading": reading(float(np.median(delta)), float(mw.pvalue)),
            }
        report["protocols"][ck] = proto
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=1)
    # human summary
    print(f"== {report['size']} split={a.split}")
    for k, v in sorted(report["reference"].items()):
        print(f"  ref {k:22s} {v:9.2f}")
    for ck, proto in report["protocols"].items():
        print(f"  -- checkpoint protocol: {ck}")
        for arm, m in proto["arm_mean"].items():
            sd = proto["arm_sd_over_seeds"][arm]
            print(f"     {arm:8s} n={proto['n_seeds'][arm]} mean {m:9.2f} sd {sd if sd is None else round(sd,2)}")
        for name, c in proto["contrasts"].items():
            print(f"     {name:16s} delta_med {c['delta_pct_median']:+6.2f}%  MW p={c['mannwhitney_p']:.4f} "
                  f"inst-Wilcoxon p={c['per_instance_wilcoxon_p']}  -> {c['reading']}")


if __name__ == "__main__":
    main()

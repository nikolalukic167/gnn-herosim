"""literature_reeval_v1 Amendment A1 -- pick each arm's operating point on the VALIDATION split only,
then re-read the registered contrasts on the frozen test split at that lr. Never select on test.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_j", type=int, required=True); ap.add_argument("--n_m", type=int, required=True)
    ap.add_argument("--eval_dir", required=True, help="dir with *__val.json and *__test.json from evaluate.py")
    ap.add_argument("--lrs", default="2e-5,1e-4,5e-4")
    ap.add_argument("--checkpoint", default="best_val")
    a = ap.parse_args()
    lrs = [float(x) for x in a.lrs.split(",")]
    arms = ("gnn", "mpoff", "mlp_t1", "mlp_t1x")

    def mean_val_makespan(arm, lr, seed):
        tag = "" if abs(lr - 2e-5) < 1e-12 else f"_lr{lr:g}"
        name = f"l2d_{a.n_j}x{a.n_m}_{arm}{tag}_s{seed}__{a.checkpoint}__val.json"
        p = os.path.join(a.eval_dir, name)
        if not os.path.exists(p):
            return None
        return json.load(open(p))["mean_makespan"]

    picks = {}
    for arm in arms:
        rows = []
        for lr in lrs:
            vals = [mean_val_makespan(arm, lr, s) for s in range(8)]
            vals = [v for v in vals if v is not None]
            if len(vals) < 8:
                print(f"  {arm} lr={lr}: only {len(vals)}/8 seeds found, skipping"); continue
            rows.append((lr, float(np.mean(vals)), float(np.std(vals, ddof=1))))
        rows.sort(key=lambda r: r[1])
        print(f"{arm}: " + ", ".join(f"lr={lr:g} val_mean={m:.2f} sd={sd:.2f}" for lr, m, sd in rows))
        picks[arm] = rows[0][0]
    print("Selected operating points (by validation mean, never test):", picks)
    with open(os.path.join(a.eval_dir, "selected_lr.json"), "w") as fh:
        json.dump(picks, fh, indent=1)

if __name__ == "__main__":
    main()

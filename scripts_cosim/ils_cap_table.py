"""ils_cap_s.json for the budget-capped ILS arm: per payload and rung, per size bucket, the median over groups of the refined GNN's charged decision time (seconds, at k = 1).
usage: ils_cap_table.py <final_run dir> <out.json> [pay ...]   (default pays rand_p48c50 rand_p120c50; the key is the gate's R1A_S7_DL_PAY row "rand_s20_<pay>|<rung>")
A group of n tasks is charged n x (plan-equivalents per task of its bucket, s7_dl_gnnr_evals_per_task.json) x the refined GNN's bucket rate (decision_cost_w.json, ms per plan) / 1000.
The groups are those of the stored refined-GNN cells final_run/out/<pay>__sr2_s4 (tasks_bsz column 3 = the group size of each task; a group of n tasks appears n times, so it counts 1/n)."""
import glob, gzip, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = (("le10", 0, 10), ("11_50", 11, 50), ("gt50", 51, 10 ** 9))
GNNR = "ra_gnn_eng_combfl_linkstatic_e40_nosplit"


def wmedian(sizes: np.ndarray) -> float:
    """Median over groups from per-task group sizes (each group of n appears n times)."""
    groups = np.sort(np.concatenate([np.full(int(round((sizes == n).sum() / n)), n) for n in np.unique(sizes)]))
    return float(groups[(len(groups) - 1) // 2]) if len(groups) % 2 else float((groups[len(groups) // 2 - 1] + groups[len(groups) // 2]) / 2)


def main() -> None:
    root, out = sys.argv[1], sys.argv[2]
    pays = sys.argv[3:] or ["rand_p48c50", "rand_p120c50"]
    w = json.load(open(os.path.join(HERE, "decision_cost_w.json")))["w_ms_per_plan_by_bucket"][GNNR]
    mg = json.load(open(os.path.join(HERE, "s7_dl_gnnr_evals_per_task.json")))
    res = {}
    for pay in pays:
        for rung in ("moderate", "heavy"):
            files = sorted(glob.glob(f"{root}/out/{pay}__sr2_s4/cc40s*__g?{rung}__*_sr2_k100_s4.tasks_bsz.npy.gz"))
            if not files:
                sys.exit(f"FAIL LOUD: no refined-GNN cells for {pay} {rung} under {root}/out")
            sizes = np.concatenate([np.load(gzip.open(f))[:, 3] for f in files]).astype(int)
            row = {"_cells": len(files), "_groups": int(sum(round((sizes == n).sum() / n) for n in np.unique(sizes)))}
            for b, lo, hi in SIZE:
                sel = sizes[(sizes >= lo) & (sizes <= hi)]
                key = f"rand_s20_{pay[5:]}|{rung}"
                n_med = wmedian(sel)
                row[b] = n_med * mg[key][b] * float(w[rung][b]) / 1000.0
                row[f"_median_group_size_{b}"] = n_med
            res[f"rand_s20_{pay[5:]}|{rung}"] = row
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

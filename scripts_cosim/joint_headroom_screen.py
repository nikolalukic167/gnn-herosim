"""Descriptive offline screen on a finished co-sim corpus (read-only): per dataset, the best pointwise plan's regret vs the joint optimum,
the gap between the best and second-best plan, and whether the optimum spans >1 node.
Pointwise plan = argmin of the per-slot additive (indicator least-squares) fit over the enumerated plans, exactly
scripts_cosim/separability_diagnostic.py:283-289 (`additive_choice`, `additive_regret_rel`); datasets with fewer plans than
parameters+2 are 'underdetermined' there (:265-266) and are flagged here, not scored (the fit interpolates, regret is vacuous).
usage: joint_headroom_screen.py OUT.jsonl DATASET_ROOT [--workers N]"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np


def sweep_complete(d):
    try:
        meta = json.loads((d / "placement_metadata.json").read_text())
        if meta.get("sweep_complete") is not True or not (d / "best.json").stat().st_size:
            return False
        rows = sum(1 for l in (d / "placements" / "placements.jsonl").open("rb") if l.strip())
        return rows == int(meta.get("num_placements", -1))
    except (OSError, json.JSONDecodeError):
        return False


def indicator_matrix(rows):
    n_slots = len(rows[0])
    vocab = [sorted({r[s] for r in rows}) for s in range(n_slots)]
    index = [{v: i for i, v in enumerate(vs)} for vs in vocab]
    offsets, width = [], 1
    for vs in vocab:
        offsets.append(width); width += len(vs)
    m = np.zeros((len(rows), width)); m[:, 0] = 1.0
    for i, r in enumerate(rows):
        for s, v in enumerate(r):
            m[i, offsets[s] + index[s][v]] = 1.0
    return m, width


def one(path):
    d = Path(path)
    try:
        if not sweep_complete(d):
            return dict(ds=d.name, status="incomplete")
        rung = json.load(open(d / "warm_snapshot.json"))["provenance"]["cell_config"].split("/inputs/")[1].split("/")[0].replace("wf1_", "")
        plans, y = [], []
        for l in open(d / "placements" / "placements.jsonl"):
            r = json.loads(l)
            if "placement_plan" in r:
                p = r["placement_plan"]
                plans.append(tuple(tuple(p[k]) for k in sorted(p, key=int))); y.append(r["rtt"])
        y = np.array(y, float)
        out = dict(ds=d.name, rung=rung, n_plans=len(y))
        if len(y) < 2:
            return dict(out, status="single_plan")
        order = np.argsort(y, kind="stable")
        best = float(y[order[0]])
        out["gap_rel"] = (float(y[order[1]]) - best) / best if best > 0 else None
        out["multi_node_opt"] = len({n for n, _ in plans[order[0]]}) > 1
        out["n_opt_ties"] = int((y == best).sum())
        m, width = indicator_matrix(plans)
        if len(y) < width + 2:
            return dict(out, status="underdetermined")
        if float(((y - y.mean()) ** 2).sum()) <= 0:
            return dict(out, status="degenerate")
        beta, *_ = np.linalg.lstsq(m, y, rcond=None)
        choice = int(np.argmin(m @ beta))
        out["regret_rel"] = (float(y[choice]) - best) / best if best > 0 else None
        return dict(out, status="ok")
    except Exception as e:
        return dict(ds=d.name, status="error", err=repr(e)[:200])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("root"); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ds = sorted(str(p) for p in Path(a.root).glob("ds_*"))
    with Pool(a.workers) as pool, open(a.out, "w") as f:
        for i, r in enumerate(pool.imap(one, ds, chunksize=8)):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                print(i, len(ds), flush=True)

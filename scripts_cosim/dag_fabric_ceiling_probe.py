#!/usr/bin/env python3
"""dag_fabric_contention_v1 -- Phase 0a paper screen (2026-09-08). NO SIMULATION IS RUN.

Question: if parent->child OUTPUT payloads (the 800 MB route_b lever) were served
store-and-forward over the contended link fabric (capacity-1 pipe per link, the
`link_contention_v1` model that today only carries the client->node INPUT), would link
waiting become a material, non-count-shaped share of the coupled tasks' RTT -- and would it
bind AT THE OPTIMUM, where the label lives?

Three stages, all computed from the stored arm_s sweep
(`simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s`, diamond4: 0=dnn1 root,
1=dnn2, 2=rf, 3=cnn sink; fabric = 12-router ring, attach_degree 1, 6 servers):

  ceiling   -- per plan, the max number k of concurrent parent->child transfers sharing one
               link (fan-out phase 0->1,0->2; fan-in phase 1->3,2->3); bandwidth-free ceiling
               wait/(wait+transfer) = ((k-1)/2)/(1+(k-1)/2). `any` counts every link (the
               parent's / sink's own ACCESS link, which >=2 remote children/parents always
               share -- a node-indexed count); `core` counts core|core links only, the only
               segment shared between DIFFERENT (parent, child) node pairs and therefore the
               only non-count-shaped part. Also a 300-pair sample of two instances (8 tasks).
  synthetic -- rtt'(plan) = rtt + W(plan, bw), W = sum over phases and links of
               T_hop * k(k-1)/2 with T_hop = 800 MB / bw (simultaneous starts; an UPPER
               bound). Manipulation share = W / (children's RTT + W) from task_times; does the
               alpha=2.0 optimum move, and does the NEW optimum carry any wait; additive-LS
               argmin regret on rtt' with and without a per-parent remote-children count.
  joint     -- 8 tasks: two feasible 4-task plans merged under the alpha=4.0 equal-tightness
               cap (route_b s9d), exhaustive over feas x feas (subsampled), synthetic waits
               on the union of concurrent transfers; avoidance premium = best zero-wait joint
               plan vs joint optimum; core-only variant.

Reading (measured 2026-09-08, written into docs/lineages/dag_fabric_contention_v1.md):
4-task, alpha=2.0 feasible set -- median ceiling `any` 0.333 (GO bar 0.25 passes) but `core`
0.000 (37.5% of plans, 9/204 optima); synthetic at bw 100/25 MB/s the manipulation share
passes (0.17/0.45 of children's RTT) yet the optimum carries ZERO wait in 98-100% of
datasets at every hosting-node count (incl. the 17 squeezed to 2 nodes). 8-task joint --
the optimum cannot avoid some wait in ~50% of datasets (premium >5% in 48%) but the binding
wait is access-link (count-shaped); core wait binds at the optimum in 10-12%. NO-GO.

Usage:
  dag_fabric_ceiling_probe.py --out simulation_data/dag_fabric_ceiling_probe.json
  dag_fabric_ceiling_probe.py --stages ceiling synthetic --out ...
"""
from __future__ import annotations
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts_cosim"))
from src.placement.network_fabric import route_links, is_core_link  # noqa: E402
import score_route_b_contention as S  # noqa: E402

CORPUS_DEFAULT = ROOT / "simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_s"
TT = S.load_task_types(ROOT / "data/nofs-ids/task-types.json")
PAYLOAD = 800e6
FANOUT = [(0, 1), (0, 2)]
FANIN = [(1, 3), (2, 3)]
PHASES = [FANOUT, FANIN]
BWS = [1000.0, 100.0, 25.0]


def median(xs):
    xs = sorted(x for x in xs if not (isinstance(x, float) and math.isnan(x)))
    n = len(xs)
    if n == 0:
        return float("nan")
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def ceiling(k):
    return ((k - 1) / 2) / (1 + (k - 1) / 2) if k >= 1 else 0.0


def ds_dirs(corpus):
    return sorted(p for p in corpus.glob("ds_*") if (p / "placements/placements.jsonl").exists())


# ----------------------------------------------------------------------------- stage: ceiling
def phase_k(routes, nodes_by_task, edges):
    any_c, core_c = Counter(), Counter()
    n_transfers = 0
    for p, c in edges:
        a, b = nodes_by_task[p], nodes_by_task[c]
        if a == b:
            continue
        n_transfers += 1
        for key in route_links(routes, a, b):
            any_c[key] += 1
            if is_core_link(key):
                core_c[key] += 1
    return (max(any_c.values()) if any_c else 0, max(core_c.values()) if core_c else 0, n_transfers)


def plan_k(ds, plan):
    nodes = {t: ds.node_of(p) for t, p in plan.items()}
    ka1, kc1, n1 = phase_k(ds.routes, nodes, FANOUT)
    ka2, kc2, n2 = phase_k(ds.routes, nodes, FANIN)
    return max(ka1, ka2), max(kc1, kc2), n1 + n2


def stage_ceiling(corpus, alpha):
    rng = random.Random(0)
    per_ds = []
    for ds_dir in ds_dirs(corpus):
        ds = S.Dataset(ds_dir, TT, "rtt")
        caps = ds.node_caps(alpha)
        feas = [(plan, v) for plan, v in ds.rows if ds.plan_feasible(plan, caps)]
        if not feas:
            continue
        ks = [plan_k(ds, plan) for plan, _ in feas]
        opt_any, opt_core, opt_n = plan_k(ds, min(feas, key=lambda r: r[1])[0])
        k8_any, k8_core = [], []
        for _ in range(300):
            pa, _ = feas[rng.randrange(len(feas))]
            pb, _ = feas[rng.randrange(len(feas))]
            merged = {t: ds.node_of(p) for t, p in pa.items()}
            merged.update({t + 10: ds.node_of(p) for t, p in pb.items()})
            best_a = best_c = 0
            for edges in PHASES:
                ka, kc, _n = phase_k(ds.routes, merged, edges + [(p + 10, c + 10) for p, c in edges])
                best_a, best_c = max(best_a, ka), max(best_c, kc)
            k8_any.append(best_a); k8_core.append(best_c)
        per_ds.append({
            "ds": ds_dir.name, "n_rows": len(ds.rows), "n_feasible": len(feas),
            "hosting_nodes": len({ds.node_of(p) for plan, _ in ds.rows for p in plan.values()}),
            "frac_plans_any_share": sum(k[0] >= 2 for k in ks) / len(ks),
            "frac_plans_core_share": sum(k[1] >= 2 for k in ks) / len(ks),
            "frac_plans_no_transfer": sum(k[2] == 0 for k in ks) / len(ks),
            "median_ceiling_any": median([ceiling(k[0]) for k in ks]),
            "median_ceiling_core": median([ceiling(k[1]) for k in ks]),
            "opt_k_any": opt_any, "opt_k_core": opt_core, "opt_n_transfers": opt_n,
            "k8_median_ceiling_any": median([ceiling(k) for k in k8_any]),
            "k8_median_ceiling_core": median([ceiling(k) for k in k8_core]),
            "k8_frac_core_share": sum(k >= 2 for k in k8_core) / len(k8_core),
        })

    def agg(key):
        vals = [d[key] for d in per_ds]
        return {"median": median(vals), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    return {
        "alpha": alpha, "n_datasets": len(per_ds),
        "hosting_nodes_hist": dict(Counter(d["hosting_nodes"] for d in per_ds)),
        "four_task": {k: agg(k) for k in ["frac_plans_any_share", "frac_plans_core_share",
                                           "frac_plans_no_transfer", "median_ceiling_any", "median_ceiling_core"]},
        "optimum": {"k_any_hist": dict(Counter(d["opt_k_any"] for d in per_ds)),
                    "k_core_hist": dict(Counter(d["opt_k_core"] for d in per_ds)),
                    "n_transfers_hist": dict(Counter(d["opt_n_transfers"] for d in per_ds))},
        "eight_task_sampled": {k: agg(k) for k in ["k8_median_ceiling_any", "k8_median_ceiling_core", "k8_frac_core_share"]},
        "per_dataset": per_ds,
    }


# --------------------------------------------------------------------------- stage: synthetic
def waits(routes, nodes, bw, core_only=False):
    T = PAYLOAD / (bw * 1024 * 1024)
    total = 0.0
    remote_children = Counter()
    for edges in PHASES:
        c = Counter()
        for p, ch in edges:
            a, b = nodes[p], nodes[ch]
            if a == b:
                continue
            remote_children[p] += 1
            for key in route_links(routes, a, b):
                if core_only and not is_core_link(key):
                    continue
                c[key] += 1
        total += sum(T * k * (k - 1) / 2 for k in c.values())
    return total, remote_children


def load_children_rtt(ds_dir):
    out = {}
    with open(ds_dir / "placements/placements.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            tt = r.get("task_times")
            if not tt:
                return None
            key = tuple(sorted((int(k), int(v[1])) for k, v in r["placement_plan"].items()))
            out[key] = sum(t[2] - t[1] for t in tt if t[0] in (1, 2, 3))
    return out


def additive_fit_argmin(rows, extra_cols=None):
    keys = sorted({(t, p) for plan, _ in rows for t, p in plan.items()})
    idx = {k: i for i, k in enumerate(keys)}
    n_extra = extra_cols.shape[1] if extra_cols is not None else 0
    X = np.zeros((len(rows), len(keys) + n_extra))
    for r, (plan, _) in enumerate(rows):
        for t, p in plan.items():
            X[r, idx[(t, p)]] = 1.0
    if extra_cols is not None:
        X[:, len(keys):] = extra_cols
    y = np.array([v for _, v in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return int(np.argmin(pred)), float(r2)


def stage_synthetic(corpus, alpha, ceiling_per_ds):
    hosting = {d["ds"]: d["hosting_nodes"] for d in ceiling_per_ds} if ceiling_per_ds else {}
    res = {bw: [] for bw in BWS}
    for ds_dir in ds_dirs(corpus):
        ds = S.Dataset(ds_dir, TT, "rtt")
        caps = ds.node_caps(alpha)
        feas = [(plan, v) for plan, v in ds.rows if ds.plan_feasible(plan, caps)]
        if len(feas) < 4:
            continue
        tt = load_children_rtt(ds_dir)
        nodes_l = [{t: ds.node_of(p) for t, p in plan.items()} for plan, _ in feas]
        base_opt = min(range(len(feas)), key=lambda i: feas[i][1])
        for bw in BWS:
            W = [waits(ds.routes, n, bw) for n in nodes_l]
            Wc = [waits(ds.routes, n, bw, core_only=True)[0] for n in nodes_l]
            rows2 = [(plan, v + w[0]) for (plan, v), w in zip(feas, W)]
            new_opt = min(range(len(rows2)), key=lambda i: rows2[i][1])
            share = float("nan")
            if tt is not None:
                shares = []
                for (plan, _v), w in zip(feas, W):
                    ch = tt.get(tuple(sorted((t, p[1]) for t, p in plan.items())))
                    if ch is not None and ch + w[0] > 0:
                        shares.append(w[0] / (ch + w[0]))
                share = median(shares)
            ai, r2 = additive_fit_argmin(rows2)
            regret = (rows2[ai][1] - rows2[new_opt][1]) / rows2[new_opt][1] * 100
            cnt = np.array([[w[1].get(0, 0), w[1].get(1, 0) + w[1].get(2, 0)] for w in W], float)
            ai_c, r2_c = additive_fit_argmin(rows2, cnt)
            regret_c = (rows2[ai_c][1] - rows2[new_opt][1]) / rows2[new_opt][1] * 100
            res[bw].append({"ds": ds_dir.name, "hosting_nodes": hosting.get(ds_dir.name),
                            "n_feasible": len(feas), "share_median": share,
                            "opt_moved": new_opt != base_opt,
                            "new_opt_wait": W[new_opt][0], "new_opt_core_wait": Wc[new_opt],
                            "additive_r2": r2, "additive_regret_pct": regret,
                            "count_repaired_r2": r2_c, "count_repaired_regret_pct": regret_c})
    summary = {}
    for bw in BWS:
        r = res[bw]; n = len(r)
        by_h = {}
        for d in r:
            by_h.setdefault(str(d["hosting_nodes"]), []).append(d)
        summary[str(bw)] = {
            "n": n,
            "manipulation_share_median_of_medians": median([d["share_median"] for d in r]),
            "manipulation_share_ge_0.10_frac": sum(d["share_median"] >= 0.10 for d in r) / n,
            "opt_moved_frac": sum(d["opt_moved"] for d in r) / n,
            "new_opt_has_any_wait_frac": sum(d["new_opt_wait"] > 0 for d in r) / n,
            "new_opt_has_core_wait_frac": sum(d["new_opt_core_wait"] > 0 for d in r) / n,
            "additive_r2_median": median([d["additive_r2"] for d in r]),
            "additive_regret_gt5_frac": sum(d["additive_regret_pct"] > 5 for d in r) / n,
            "count_repaired_regret_gt5_frac": sum(d["count_repaired_regret_pct"] > 5 for d in r) / n,
            "count_repaired_r2_median": median([d["count_repaired_r2"] for d in r]),
            "by_hosting_nodes": {h: {"n": len(v),
                                     "opt_has_any_wait": sum(x["new_opt_wait"] > 0 for x in v) / len(v),
                                     "opt_has_core_wait": sum(x["new_opt_core_wait"] > 0 for x in v) / len(v)}
                                 for h, v in sorted(by_h.items())},
        }
    return {"alpha": alpha, "summary": summary, "per_dataset": res}


# ------------------------------------------------------------------------------- stage: joint
def link_counts(routes, nodes, edges, core_only):
    c = Counter()
    for p, ch in edges:
        a, b = nodes[p], nodes[ch]
        if a == b:
            continue
        for key in route_links(routes, a, b):
            if core_only and not is_core_link(key):
                continue
            c[key] += 1
    return c


def joint_wait(routes, nA, nB, T, core_only=False):
    total = 0.0
    for edges in PHASES:
        c = link_counts(routes, nA, edges, core_only) + link_counts(routes, nB, edges, core_only)
        total += sum(T * k * (k - 1) / 2 for k in c.values())
    return total


def stage_joint(corpus, alpha4, alpha8, n_datasets, bws=(100.0, 25.0)):
    rng = random.Random(1)
    dirs = ds_dirs(corpus)
    rng.shuffle(dirs)
    out8 = {bw: [] for bw in bws}
    for ds_dir in dirs[:n_datasets]:
        ds = S.Dataset(ds_dir, TT, "rtt")
        caps8 = ds.node_caps(alpha8)
        feas4 = [(plan, v) for plan, v in ds.rows if ds.plan_feasible(plan, ds.node_caps(alpha4))]
        if len(feas4) < 4:
            continue
        if len(feas4) > 400:
            feas4 = rng.sample(feas4, 400)
        nodes = [{t: ds.node_of(p) for t, p in plan.items()} for plan, _ in feas4]
        loads = []
        for plan, _ in feas4:
            L = Counter()
            for t, p in plan.items():
                L[ds.node_of(p)] += ds.demand[(t, p)]
            loads.append(L)
        pairs = [(i, j) for i in range(len(feas4)) for j in range(len(feas4))
                 if all(tot <= caps8.get(n, math.inf) + 1e-9 for n, tot in (loads[i] + loads[j]).items())]
        if len(pairs) < 10:
            continue
        if len(pairs) > 60000:
            pairs = rng.sample(pairs, 60000)
        base = np.array([feas4[i][1] + feas4[j][1] for i, j in pairs])
        for bw in bws:
            T = PAYLOAD / (bw * 1024 * 1024)
            W = np.array([joint_wait(ds.routes, nodes[i], nodes[j], T) for i, j in pairs])
            Wc = np.array([joint_wait(ds.routes, nodes[i], nodes[j], T, True) for i, j in pairs])
            y = base + W
            opt = int(np.argmin(y)); base_opt = int(np.argmin(base))
            zero = np.where(W == 0)[0]
            best_zero = float(y[zero].min()) if len(zero) else float("inf")
            premium = (best_zero - y[opt]) / y[opt] * 100 if math.isfinite(best_zero) else float("inf")
            keys = sorted({(t, p) for i, j in pairs for t, p in list(feas4[i][0].items()) + [(t + 10, p) for t, p in feas4[j][0].items()]})
            idx = {k: n for n, k in enumerate(keys)}
            X = np.zeros((len(pairs), len(keys)))
            for r, (i, j) in enumerate(pairs):
                for t, p in feas4[i][0].items():
                    X[r, idx[(t, p)]] = 1
                for t, p in feas4[j][0].items():
                    X[r, idx[(t + 10, p)]] = 1
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            ai = int(np.argmin(pred))
            out8[bw].append({"ds": ds_dir.name, "n_pairs": len(pairs),
                             "opt_has_any_wait": bool(W[opt] > 0), "opt_has_core_wait": bool(Wc[opt] > 0),
                             "opt_moved": opt != base_opt, "zero_wait_frac": float(len(zero) / len(pairs)),
                             "avoidance_premium_pct": premium, "wait_share_at_opt": float(W[opt] / y[opt]),
                             "additive_r2": float(r2), "additive_regret_pct": float((y[ai] - y[opt]) / y[opt] * 100)})
    summary = {}
    for bw, r in out8.items():
        n = len(r)
        summary[str(bw)] = {
            "n_datasets": n,
            "opt_has_any_wait_frac": sum(d["opt_has_any_wait"] for d in r) / n,
            "opt_has_core_wait_frac": sum(d["opt_has_core_wait"] for d in r) / n,
            "opt_moved_frac": sum(d["opt_moved"] for d in r) / n,
            "zero_wait_pairs_frac_median": median([d["zero_wait_frac"] for d in r]),
            "avoidance_premium_gt1pct_frac": sum(d["avoidance_premium_pct"] > 1 for d in r) / n,
            "avoidance_premium_gt5pct_frac": sum(d["avoidance_premium_pct"] > 5 for d in r) / n,
            "wait_share_at_opt_median": median([d["wait_share_at_opt"] for d in r]),
            "additive_r2_median": median([d["additive_r2"] for d in r]),
            "additive_regret_gt5_frac": sum(d["additive_regret_pct"] > 5 for d in r) / n,
        }
    return {"alpha4": alpha4, "alpha8": alpha8, "summary": summary, "per_dataset": out8}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    ap.add_argument("--alpha", type=float, default=2.0, help="4-task cap (route_b registered primary)")
    ap.add_argument("--alpha8", type=float, default=4.0, help="8-task equal-tightness cap (route_b s9d)")
    ap.add_argument("--joint-datasets", type=int, default=60)
    ap.add_argument("--stages", nargs="+", default=["ceiling", "synthetic", "joint"],
                    choices=["ceiling", "synthetic", "joint"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {"corpus": str(args.corpus), "payload_bytes": PAYLOAD}
    if "ceiling" in args.stages:
        report["ceiling"] = stage_ceiling(args.corpus, args.alpha)
    if "synthetic" in args.stages:
        report["synthetic"] = stage_synthetic(args.corpus, args.alpha, report.get("ceiling", {}).get("per_dataset"))
    if "joint" in args.stages:
        report["joint"] = stage_joint(args.corpus, args.alpha, args.alpha8, args.joint_datasets)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    brief = {k: ({kk: vv for kk, vv in v.items() if kk != "per_dataset"} if isinstance(v, dict) else v)
             for k, v in report.items()}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

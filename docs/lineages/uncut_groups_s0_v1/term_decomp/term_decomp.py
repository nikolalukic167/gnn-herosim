#!/usr/bin/env python3
"""Per-term latency breakdown (Eq. 2), DESCRIPTIVE, filed cells only, no new cells.
usage: term_decomp.py <set a|b> <outdir>
Per (set, rung): whole run, windows g0-g3 pooled; per topology value = mean over windows (and GNN seeds 1-2);
delta = GNN - bar; faster/12 = topologies with GNN term < bar term; raw two-sided Wilcoxon signed-rank p over the 12 paired topology means.
Field mapping (src/placement/infrastructure.py, orchestrator.py):
  W  wait for scheduler      averageWaitTime            (scheduled - dispatched: batch collection + charged decision time)
  QNP queue field            averageQueueTime           (arrived - scheduled: FIFO queue + ingress propagation/transfer/link wait + image pull)
     N_prop                  averageNetworkLatency      (ingress propagation only)
     P  pull                 averagePullTime
     Q+N_xfer (residual)     averageQueueTime - averageNetworkLatency - averagePullTime  (queue and ingress transfer/link wait: NOT separable)
  K  cold start              averageColdStartTime       (init field averageInitializationTime = K + R)
  R  rendezvous              totalPeerRendezvousWait / num_tasks
  X  exchange                totalPeerExchangeTime / num_tasks
  I+O input read + output    averageCommunicationsTime - X   (storage read and write: NOT separable from each other)
  L  compute-lock wait       lock_wait.mean
  E  execution               averageExecutionTime
  elapsed                    averageElapsedTime
  compute-field residual     averageComputeTime - (comm + L + E)   (output storage wait etc.)
  decision wall (informational, real seconds, not simulated) decisionTiming.total_s / num_tasks
"""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

U = "/share/nikola.lukic/uncut_groups"
SET, OUT = sys.argv[1], sys.argv[2]
GNN, BAR = "ra_gnn_eng_combfl_nosplit_k100", "cd_ib0_inflightfull_k100"
if SET == "a":
    TOPOS = list(range(19537, 19549)); BD = f"{U}/comb/cells_bar_g8"; GD = f"{U}/comb/cells_comb_fl"; IN = f"{U}/comb/in_comball"
    NAME = "gate topologies 19537-19548 (combfl s1-2 vs cd_ib0_inflightfull)"
else:
    TOPOS = list(range(30129, 30141)); BD = f"{U}/size64/cells_bar"; GD = f"{U}/size64/cells_learned"; IN = f"{U}/size64/in_size64"
    NAME = "size-64 topologies 30129-30140 (FL gnn s1-2 vs cd_ib0_inflightfull)"
WINDOWS = ("g0", "g1", "g2", "g3"); SEEDS = (1, 2)
TERMS = ["W_sched_wait", "queue_field", "N_prop", "P_pull", "Q_resid", "K_cold", "R_rendezvous", "X_exchange", "IO_input_output", "L_lock_wait", "E_exec", "compute_resid", "elapsed"]
EXTRA = ["decision_wall_s", "coloc_share", "replica_peak", "replica_time_mean", "cold_starts", "scale_ups"]


def one(p):
    s = json.load(open(p + ".summary.json")); n = s["num_tasks"]
    X = s["totalPeerExchangeTime"] / n; comm = s["averageCommunicationsTime"]; L = s["lock_wait"]["mean"]; E = s["averageExecutionTime"]
    q = s["averageQueueTime"]; N = s["averageNetworkLatency"]; P = s["averagePullTime"]
    tr = sum(v["transfers"] for v in s["peerExchangeByAccessClass"].values())
    so = s["scaleOut"]["scale_ups_by_cause"]
    return {"W_sched_wait": s["averageWaitTime"], "queue_field": q, "N_prop": N, "P_pull": P, "Q_resid": q - N - P, "K_cold": s["averageColdStartTime"],
            "R_rendezvous": s["totalPeerRendezvousWait"] / n, "X_exchange": X, "IO_input_output": comm - X, "L_lock_wait": L, "E_exec": E,
            "compute_resid": s["averageComputeTime"] - (comm + L + E), "elapsed": s["averageElapsedTime"],
            "decision_wall_s": s["decisionTiming"]["total_s"] / n, "transfers": tr, "replica_peak": s["replica_count_series"]["peak"],
            "replica_time_mean": s["replica_count_series"]["time_mean"], "cold_starts": s["cold_start_pct"] / 100 * n, "scale_ups": sum(so.values())}


def main():
    res, L = {}, [f"=== {NAME}; whole run, windows g0-g3, mean s per task; delta = GNN - bar; faster = GNN < bar; Wilcoxon raw two-sided over 12 topology means"]
    for r in ("moderate", "heavy"):
        rel = {w: 2 * len(json.load(open(f"{IN}/wf1_{r}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in WINDOWS}
        per = {"gnn": [], "bar": []}
        for t in TOPOS:
            for a in per:
                rows = []
                for w in WINDOWS:
                    if a == "bar":
                        d = one(f"{BD}/cc40s{t}__{w}{r}__{BAR}_s0")
                        d["coloc_share"] = 1 - d["transfers"] / rel[w]; rows.append(d)
                    else:
                        ds = [one(f"{GD}/cc40s{t}__{w}{r}__{GNN}_s{s}") for s in SEEDS]
                        d = {k: float(np.mean([x[k] for x in ds])) for k in ds[0]}
                        d["coloc_share"] = 1 - d["transfers"] / rel[w]; rows.append(d)
                per[a].append({k: float(np.mean([x[k] for x in rows])) for k in TERMS + EXTRA})
        R = {"n_topologies": len(TOPOS), "terms": {}}
        L.append(f"--- {r}: {'term':16s} {'GNN':>9s} {'bar':>9s} {'delta':>9s} {'faster/12':>10s} {'wilcoxon p':>11s}")
        for k in TERMS + EXTRA:
            g = np.array([x[k] for x in per["gnn"]]); b = np.array([x[k] for x in per["bar"]]); dlt = g - b
            try:
                p = float(wilcoxon(g, b).pvalue) if np.any(dlt != 0) else 1.0
            except ValueError:
                p = float("nan")
            R["terms"][k] = {"gnn": float(g.mean()), "bar": float(b.mean()), "delta": float(dlt.mean()), "faster_of_12": int((dlt < 0).sum()), "wilcoxon_p": p}
            L.append(f"        {k:16s} {g.mean():9.4f} {b.mean():9.4f} {dlt.mean():+9.4f} {int((dlt < 0).sum()):7d}/12 {p:11.4f}")
        res[r] = R
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/term_decomp_{SET}.json", "w"), indent=1); open(f"{OUT}/term_decomp_{SET}.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


main()

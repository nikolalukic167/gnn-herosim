"""Where does GNN self-search lose to cd_exactS? An offline decomposition of the paired mean-latency gap from the gate summaries alone.

The gates deleted every per-task result after summarising it (KEEP_RAW_DIR unset), so only the summary fields exist. What they allow:

(c) latency component, exact. A task's elapsed time is wait (scheduled - dispatched) + queue (arrived - scheduled) + initialization (started - arrived)
    + compute (done - started); the means are in the summary and are checked to add up. Inside them: lock wait (compute start - io end, inside compute),
    cold start and pull (inside initialization), execution, storage communications, peer exchange and rendezvous. Buckets reported, seconds per task
    A - B, with a residual so the buckets sum to the gap:
      queue    = wait + queue + lock wait
      cold     = cold start + pull
      exec     = execution
      exchange = peer exchange + rendezvous (per task)
      other    = elapsed gap - the four above (initialization beyond cold/pull, storage communications, compute not otherwise named)
(a) time phase, coarse. The summary keeps the waiting part of a task (wait + queue + lock wait, `backlog_profile_v2`) as the mean over each quarter of the
    tasks in arrival order, so the waiting gap splits exactly over four quarters (12,500 tasks each). The rest of the gap (elapsed gap minus the waiting
    gap) has no phase. The opening burst t < 120 s is about 3 % of the tasks, inside quarter 1, and cannot be isolated.
(b) batch size and (d) plan disagreement need per-task / per-batch records (dispatched time, batch or peer-group id, chosen node, stage times) that the
    summaries do not hold; this script says so rather than guessing.

Per topology the gap is taken over its four windows (sums of per-cell differences). Reported per rung: the paired % (median over topologies, as the gate
reader), each bucket's contribution in percentage points of B's latency (median over topologies; they sum to the paired percentage up to the median), and
each bucket's share of the gap (pooled over all cells, and the median of the per-topology shares).

  selfsearch_gap_decomp.py --gate <dir> [<dir>...] --a ra_gnn_eng_selfsearch:1 --b cd_exactS:0 --topos <ids|auto> [--out f.json]
"""
import argparse
import json
import statistics as st
import sys

import r1_attribution_v1_read as R

RUNGS = ("moderate", "heavy")
BUCKETS = ("queue", "cold", "exec", "exchange", "other")


def comps(s):
    n = float(s["num_tasks"])
    wait, queue, init, comp = (float(s[k]) for k in ("averageWaitTime", "averageQueueTime", "averageInitializationTime", "averageComputeTime"))
    el = float(s["averageElapsedTime"])
    lock = float((s.get("lock_wait") or {}).get("mean") or 0.0)
    q = {"queue": wait + queue + lock,
         "cold": float(s.get("averageColdStartTime") or 0.0) + float(s.get("averagePullTime") or 0.0),
         "exec": float(s.get("averageExecutionTime") or 0.0),
         "exchange": (float(s.get("totalPeerExchangeTime") or 0.0) + float(s.get("totalPeerRendezvousWait") or 0.0)) / n}
    q["other"] = el - sum(q.values())
    bl = (s.get("backlog_profile_v2") or {}).get("quarter_mean_backlog")
    return {"elapsed": el, "sum_stages_resid": el - (wait + queue + init + comp), **q, "wait_quarters": bl}


def cell_pairs(cells, a, b, rung, topos):
    out = {}
    for t in topos:
        for w in sorted({k[3] for k in cells if k[2] == t and k[4] == rung}):
            ka, kb = (a[0], a[1], t, w, rung), (b[0], b[1], t, w, rung)
            if ka in cells and kb in cells:
                out.setdefault(t, []).append((comps(cells[ka]), comps(cells[kb])))
    return out


def decompose(cells, a, b, rung, topos):
    pairs = cell_pairs(cells, a, b, rung, topos)
    per, pool = {}, {k: 0.0 for k in ("gap", *BUCKETS, "wait_q1", "wait_q2", "wait_q3", "wait_q4", "base")}
    worst_resid = 0.0
    for t, ps in sorted(pairs.items()):
        agg = {k: 0.0 for k in pool}
        for ca, cb in ps:
            worst_resid = max(worst_resid, abs(ca["sum_stages_resid"]), abs(cb["sum_stages_resid"]))
            agg["gap"] += ca["elapsed"] - cb["elapsed"]
            agg["base"] += cb["elapsed"]
            for k in BUCKETS:
                agg[k] += ca[k] - cb[k]
            for i in range(4):
                agg[f"wait_q{i + 1}"] += (ca["wait_quarters"][i] - cb["wait_quarters"][i]) / 4.0
        per[t] = agg
        for k in pool:
            pool[k] += agg[k]
    tops = sorted(per)
    pct = {t: 100.0 * per[t]["gap"] / per[t]["base"] for t in tops}
    pp = {k: st.median(100.0 * per[t][k] / per[t]["base"] for t in tops) for k in (*BUCKETS, "wait_q1", "wait_q2", "wait_q3", "wait_q4")}
    nonwait = {t: per[t]["gap"] - sum(per[t][f"wait_q{i}"] for i in range(1, 5)) for t in tops}
    pp["no_phase_rest"] = st.median(100.0 * nonwait[t] / per[t]["base"] for t in tops)
    share_pool = {k: (pool[k] / pool["gap"] if pool["gap"] else None) for k in (*BUCKETS, "wait_q1", "wait_q2", "wait_q3", "wait_q4")}
    share_pool["no_phase_rest"] = (sum(nonwait.values()) / pool["gap"]) if pool["gap"] else None
    share_med = {k: st.median(per[t][k] / per[t]["gap"] for t in tops if per[t]["gap"] > 0) if any(per[t]["gap"] > 0 for t in tops) else None
                 for k in (*BUCKETS, "wait_q1", "wait_q2", "wait_q3", "wait_q4")}
    share_med["no_phase_rest"] = st.median(nonwait[t] / per[t]["gap"] for t in tops if per[t]["gap"] > 0) if any(per[t]["gap"] > 0 for t in tops) else None
    return {"a": list(a), "b": list(b), "rung": rung, "n_topologies": len(tops), "cells": sum(len(v) for v in pairs.values()),
            "paired_pct_median": st.median(pct.values()) if pct else None, "topologies_a_slower": sum(1 for v in pct.values() if v > 0),
            "pp_of_b_median": pp, "share_of_gap_pooled": share_pool, "share_of_gap_median_over_topologies": share_med,
            "gap_s_per_task_pooled_mean": pool["gap"] / max(1, sum(len(v) for v in pairs.values())),
            "worst_stage_sum_residual_s": worst_resid, "per_topology_pct": {str(t): round(v, 3) for t, v in pct.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--a", required=True, help="kind:seed of the first arm, e.g. ra_gnn_eng_selfsearch:1")
    ap.add_argument("--b", required=True, help="kind:seed of the reference arm, e.g. cd_exactS:0")
    ap.add_argument("--topos", default="auto", help="comma-separated ids, or auto = every topology with both arms")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    cells, _failed = R.load(x.gate)
    a, b = (tuple([y.split(":")[0], int(y.split(":")[1])]) for y in (x.a, x.b))
    have = lambda arm: {k[2] for k in cells if (k[0], k[1]) == arm}
    topos = sorted(have(a) & have(b)) if x.topos == "auto" else [int(t) for t in x.topos.split(",")]
    res = {r: decompose(cells, a, b, r, topos) for r in RUNGS}
    if x.out:
        json.dump(res, open(x.out, "w"), indent=1)
    f = lambda v, n=2: "-" if v is None else f"{v:+.{n}f}"
    for r, d in res.items():
        print(f"== {x.a} vs {x.b}, {r}: {d['n_topologies']} topologies, {d['cells']} cells, paired median {f(d['paired_pct_median'])} %, "
              f"A slower on {d['topologies_a_slower']}/{d['n_topologies']}; mean gap {f(d['gap_s_per_task_pooled_mean'], 4)} s/task; "
              f"worst |stage-sum residual| {d['worst_stage_sum_residual_s']:.2e} s")
        print(f"   {'bucket':14s} {'pp of B (median)':>17s} {'share pooled':>13s} {'share median':>13s}")
        for k in (*BUCKETS, "wait_q1", "wait_q2", "wait_q3", "wait_q4", "no_phase_rest"):
            print(f"   {k:14s} {f(d['pp_of_b_median'][k]):>17s} {f(100 * d['share_of_gap_pooled'][k] if d['share_of_gap_pooled'][k] is not None else None, 1):>12s}% "
                  f"{f(100 * d['share_of_gap_median_over_topologies'][k] if d['share_of_gap_median_over_topologies'][k] is not None else None, 1):>12s}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

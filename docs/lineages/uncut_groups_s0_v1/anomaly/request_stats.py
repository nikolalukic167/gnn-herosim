#!/usr/bin/env python3
"""S4. Request-level completion time, descriptive, under SCIENTIST's six conditions (2026-10-22). Written once.
Roots and pins (Table I's): bar = comb/cells_bar_g8 (cd_ib0_inflightfull_k100 s0, pin a459dd98); learned = comb/cells_comb (FL gnn ra_gnn_eng_combfl_nosplit_k100 and
FL MLP ra_mlp_same_combfl_nosplit_k100, seeds 1-2, pin a8681537); 12 topologies 19537-19548 x g0-g3 x moderate/heavy, 24 servers. The reader reproduces Table I's four
task-mean medians (comb read, b3_13c9343d/combfl_cells_comb.json) from these cells before any request number is written.
Definitions. Request = peer_group (the workload's `events[i].peer_group`; a fan-out = a group of >= 2 tasks). tasks.npy col 0 is the task's arrival (it equals the workload event
timestamp, asserted per cell), col 1 its completion time; request completion time = max over members of col 1 minus min over members of col 0 (first arrival).
after360: a request counts iff its FIRST member arrives at t >= 360 s; a request whose first member arrives before 360 s is excluded whole, including one that straddles 360 s.
Statistic (Table I's): per topology the median over the 4 windows of the paired % difference to the comparator in the same cell, learned cell = the mean over the seeds of the
per-seed cell statistic; then the median over the 12 topologies, the faster count, and the exact two-sided Wilcoxon (r1_attribution_v1_read.exact_wilcoxon). Raw p, no Holm.
Per-seed (s1, s2) rows are read the same way alone."""
import glob, hashlib, json, os, statistics as st, sys
import numpy as np
sys.path.insert(0, "/home/nikola.lukic/gnn-herosim-wt/s4_reader_six_44481b82/scripts_cosim")
from r1_attribution_v1_read import exact_wilcoxon  # noqa: E402

U = "/share/nikola.lukic/uncut_groups"; C = U + "/comb"; INP = C + "/in_comball"
BAR, GNN, MLP = "cd_ib0_inflightfull_k100", "ra_gnn_eng_combfl_nosplit_k100", "ra_mlp_same_combfl_nosplit_k100"
ROOT = {BAR: C + "/cells_bar_g8", GNN: C + "/cells_comb", MLP: C + "/cells_comb"}
PIN = {BAR: "a459dd98", GNN: "a8681537", MLP: "a8681537"}
SEEDS = {BAR: (0,), GNN: (1, 2), MLP: (1, 2)}
TOPOS = list(range(19537, 19549)); W = ("g0", "g1", "g2", "g3"); RUNGS = ("moderate", "heavy"); T = 360.0
OUT = U + "/anomaly/request_stats.json"; TXT = U + "/anomaly/request_stats.txt"
assert not os.path.exists(OUT), "written once"
wl = {}
def workload(rung, w):
    if (rung, w) not in wl:
        ev = json.load(open(f"{INP}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["events"]
        ts = np.array([e["timestamp"] for e in ev]); g = np.array([e["peer_group"] for e in ev])
        u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
        wl[(rung, w)] = (ts, inv, cnt, cnt[inv])
    return wl[(rung, w)]
chk = {"cells": 0, "arrival_eq_workload_max_abs": 0.0, "groupsize_eq_workload": 0, "bsz_time_cols_eq_max_abs": 0.0, "incomplete_tasks": 0, "incomplete_groups": 0,
       "completion_before_arrival": 0, "after_endTime": 0}
def cell(kind, sd, t, w, rung):
    stem = f"{ROOT[kind]}/cc40s{t}__{w}{rung}__{kind}_s{sd}"
    s = json.load(open(stem + ".summary.json"))
    assert s["code"]["commit"][:8] == PIN[kind] and not s["code"]["dirty"] and s["servers"] == 24, (stem, s["code"]["commit"])
    f32 = np.load(stem + ".tasks.npy").astype(np.float64); b = np.load(stem + ".tasks_bsz.npy").astype(np.float64)
    ts, inv, cnt, size = workload(rung, w)
    n = s["num_tasks"]; assert len(f32) == len(b) == len(ts) == n, stem
    tol = lambda x: 1.2e-7 * np.abs(x) + 1e-9  # float32 half-ulp
    ids = b[:, 0].astype(int)  # col 0 is the workload event index; rows are permuted within a group (membership and timestamps are checked below)
    assert np.array_equal(np.sort(ids), np.arange(n)), (stem, "task ids are not a permutation of the workload events")
    ts, size, inv = ts[ids], size[ids], inv[ids]
    assert np.array_equal(b[:, 3].astype(int), size), (stem, "groupSize != workload group size")
    assert np.all(np.abs(f32[:, 0] - ts) <= tol(ts)), (stem, "tasks.npy col 0 != workload arrival timestamp")
    assert np.all(np.abs(b[:, 1] - ts) <= 1e-6), (stem, "tasks_bsz col 1 != workload arrival timestamp")
    d = float(np.max(np.abs(f32[:, 0] - ts))); d2 = float(max(np.max(np.abs(f32[:, 0] - b[:, 1])), np.max(np.abs(f32[:, 1] - b[:, 2]))))
    assert np.all(np.abs(f32[:, 1] - b[:, 2]) <= tol(b[:, 2])), (stem, "tasks.npy col 1 != tasks_bsz col 2")
    a = np.stack([b[:, 1], b[:, 2]], 1)  # float64 arrival and completion
    fin = np.isfinite(a[:, 1]); bad_t = int((~fin).sum()); early = int((a[:, 1] < a[:, 0]).sum()); late = int((a[:, 1] > s["endTime"] + 1e-3).sum())
    chk["cells"] += 1; chk["arrival_eq_workload_max_abs"] = max(chk["arrival_eq_workload_max_abs"], d); chk["bsz_time_cols_eq_max_abs"] = max(chk["bsz_time_cols_eq_max_abs"], d2)
    chk["groupsize_eq_workload"] += 1; chk["incomplete_tasks"] += bad_t; chk["completion_before_arrival"] += early; chk["after_endTime"] += late
    ng = len(cnt)
    inc = np.zeros(ng, bool); np.logical_or.at(inc, inv, ~fin | (a[:, 1] > s["endTime"] + 1e-3)); chk["incomplete_groups"] += int(inc.sum())
    first = np.full(ng, np.inf); last = np.full(ng, -np.inf)
    np.minimum.at(first, inv, a[:, 0]); np.maximum.at(last, inv, a[:, 1])
    req = last - first; el = a[:, 1] - a[:, 0]
    assert abs(el.mean() - s["averageElapsedTime"]) < 1e-7 * max(1.0, s["averageElapsedTime"]), stem
    after_t = a[:, 0] >= T
    tasks = {"whole": el, "after360": el[after_t]}
    out = {"n_tasks": n, "n_groups": ng, "n_fanout": int((cnt >= 2).sum()), "ss_mean": s["steady_state"]["after_360s"]["mean"], "mean_whole": s["averageElapsedTime"]}
    fan = cnt >= 2; late360 = first >= T
    sc = {"fanout": {"whole": req[fan], "after360": req[fan & late360]}, "all": {"whole": req, "after360": req[late360]}}
    for scope, d_ in sc.items():
        for win, v in d_.items():
            out[f"req_{scope}_{win}_n"] = int(len(v))
            for nm, f in (("mean", np.mean), ("p50", lambda x: np.percentile(x, 50)), ("p90", lambda x: np.percentile(x, 90)), ("p99", lambda x: np.percentile(x, 99))):
                out[f"req_{scope}_{win}_{nm}"] = float(f(v))
    for win, v in tasks.items():
        for nm, f in (("mean", np.mean), ("p50", lambda x: np.percentile(x, 50)), ("p90", lambda x: np.percentile(x, 90)), ("p99", lambda x: np.percentile(x, 99))):
            out[f"task_{win}_{nm}"] = float(f(v))
    return out
cells = {}
for kind in (BAR, GNN, MLP):
    for sd in SEEDS[kind]:
        for t in TOPOS:
            for w in W:
                for r in RUNGS:
                    cells[(kind, sd, t, w, r)] = cell(kind, sd, t, w, r)
for t in TOPOS:  # the same population in every arm: identical task and group counts per (topology, window, rung)
    for w in W:
        for r in RUNGS:
            sig = {(c["n_tasks"], c["n_groups"], c["n_fanout"]) for k, c in cells.items() if k[2:] == (t, w, r)}
            assert len(sig) == 1, (t, w, r, sig)
KEYS = [k for k in cells[(BAR, 0, 19537, "g0", "moderate")] if k.startswith(("req_", "task_")) and not k.endswith("_n")]
def val(kind, seeds, t, w, r, key):
    return float(np.mean([cells[(kind, sd, t, w, r)][key] for sd in seeds]))
def test(a, a_seeds, b, b_seeds, rung, key):
    per = {}
    for t in TOPOS:
        v = [100.0 * (val(a, a_seeds, t, w, rung, key) - val(b, b_seeds, t, w, rung, key)) / val(b, b_seeds, t, w, rung, key) for w in W]
        per[t] = st.median(v)
    x = list(per.values())
    return {"median_pct": st.median(x), "faster": sum(1 for y in x if y < 0), "n_topologies": len(x), "p": exact_wilcoxon(x), "per_topology": {str(t): round(v, 3) for t, v in per.items()}}
COMP = {"gnn_vs_bar": (GNN, BAR), "mlp_vs_bar": (MLP, BAR), "gnn_vs_mlp": (GNN, MLP)}
SEEDSETS = {"seed_mean": (1, 2), "s1": (1,), "s2": (2,)}
res = {"definitions": __doc__, "checks": chk, "table_I_reproduced": {}, "tests": {}}
tI = json.load(open(U + "/conf2/b3_13c9343d/combfl_cells_comb.json"))["F1"]["tests"]
for e in tI:
    key = "mean_whole" if e["stat"] == "whole" else "ss_mean"
    vals = []
    for t in TOPOS:
        v = []
        for w in W:
            a = np.mean([cells[(GNN, sd, t, w, e["rung"])][key] for sd in (1, 2)]); b = cells[(BAR, 0, t, w, e["rung"])][key]
            v.append(100.0 * (a - b) / b)
        vals.append(st.median(v))
    mine = st.median(vals); assert abs(mine - e["median_pct"]) < 1e-9, (e["rung"], e["stat"], mine, e["median_pct"])
    res["table_I_reproduced"][f"{e['rung']} {e['stat']}"] = {"mine": mine, "filed_read": e["median_pct"]}
for comp, (a, b) in COMP.items():
    for ss, sdset in SEEDSETS.items():
        if comp == "mlp_vs_bar" and False: continue
        for r in RUNGS:
            for key in KEYS:
                if key.startswith(("req_all", "task_")) and ss != "seed_mean" and not key.startswith("task_whole_mean"): continue
                res["tests"][f"{comp}|{ss}|{r}|{key}"] = test(a, sdset, b, (0,) if b == BAR else sdset, r, key)
res["counts"] = {"requests_fanout_per_cell_mean": float(np.mean([c["n_fanout"] for c in cells.values()])),
                 "requests_all_per_cell_mean": float(np.mean([c["n_groups"] for c in cells.values()]))}
txt = json.dumps(res, indent=1, default=str); open(OUT + ".tmp", "w").write(txt); os.replace(OUT + ".tmp", OUT)
L = [f"request_stats md5-of-json written to {OUT}", f"checks: {json.dumps(chk)}", "Table I reproduced: " + json.dumps(res["table_I_reproduced"]), ""]
for scope in ("fanout",):
    for r in RUNGS:
        for win in ("whole", "after360"):
            L.append(f"=== {r} {win}, fan-out requests (>=2 tasks); FL gnn vs bar, seed mean | s1 | s2 ; median % (faster/12; raw p)")
            for nm in ("mean", "p50", "p90", "p99"):
                row = []
                for ss in SEEDSETS:
                    e = res["tests"][f"gnn_vs_bar|{ss}|{r}|req_{scope}_{win}_{nm}"]
                    row.append(f"{e['median_pct']:+.2f} ({e['faster']}/12; {e['p']:.3g})")
                L.append(f"   request {nm:4s} " + "  |  ".join(row))
            for nm in ("mean", "p50", "p90", "p99"):
                e = res["tests"][f"gnn_vs_bar|seed_mean|{r}|task_{win}_{nm}"] if f"gnn_vs_bar|seed_mean|{r}|task_{win}_{nm}" in res["tests"] else None
                if e: L.append(f"   task    {nm:4s} {e['median_pct']:+.2f} ({e['faster']}/12; {e['p']:.3g})")
open(TXT, "w").write("\n".join(L) + "\n"); print("\n".join(L))

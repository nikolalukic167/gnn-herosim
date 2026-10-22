#!/usr/bin/env python3
"""S4. Request-level completion time at 64 and 128 servers (usage: request_stats_size.py 64|128), descriptive, the signed request_stats.py (md5 9264eff6) generalised to the size's roots; SCIENTIST's six conditions of 2026-10-22. Written once per size.
Roots and pins are those of the size's filed primary/secondary read (bar cd_ib0_inflightfull_k100 s0 a459dd98; FL gnn and FL MLP _k100 s1-2 74ddf70e); the filed read's task-mean medians (F1 tests and the secondary's descriptive mlp_vs_bar / fl_vs_mlp) are reproduced to 1e-9 before any request number is written.
The text below is the 24-server docstring, kept for the definitions.
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

U = "/share/nikola.lukic/uncut_groups"
SIZE = int(sys.argv[1]); assert SIZE in (64, 128)
BAR, GNN, MLP = "cd_ib0_inflightfull_k100", "ra_gnn_eng_combfl_nosplit_k100", "ra_mlp_same_combfl_nosplit_k100"
if SIZE == 64:
    R = U + "/size64"; INP = R + "/in_size64"; ROOT = {BAR: R + "/cells_bar", GNN: R + "/cells_learned", MLP: R + "/cells_desc"}
    TOPOS = list(range(30129, 30141)); READ = R + "/reads/read_size64_secondary.json"
else:
    R = U + "/size128"; INP = R + "/in_size128"; ROOT = {BAR: R + "/cells_bar", GNN: R + "/cells_learned", MLP: R + "/mlp_cells"}
    TOPOS = list(range(30241, 30253)); READ = R + "/reads/read_s128_secondary.json"
PIN = {BAR: "a459dd98", GNN: "74ddf70e", MLP: "74ddf70e"}
SEEDS = {BAR: (0,), GNN: (1, 2), MLP: (1, 2)}
W = ("g0", "g1", "g2", "g3"); RUNGS = ("moderate", "heavy"); T = 360.0
OUT = f"{U}/anomaly/request_stats_s{SIZE}.json"; TXT = f"{U}/anomaly/request_stats_s{SIZE}.txt"
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
    assert s["code"]["commit"][:8] == PIN[kind] and not s["code"]["dirty"] and s["servers"] == SIZE, (stem, s["code"]["commit"])
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
filed = json.load(open(READ)); assert [int(x) for x in filed["topos"]] == TOPOS, "topology list != the filed read's"
for t in TOPOS:  # the same population in every arm: identical task and group counts per (topology, window, rung)
    for w in W:
        for r in RUNGS:
            sig = {(c["n_tasks"], c["n_groups"], c["n_fanout"]) for k, c in cells.items() if k[2:] == (t, w, r)}
            assert len(sig) == 1, (t, w, r, sig)
KEYS = [k for k in cells[(BAR, 0, TOPOS[0], "g0", "moderate")] if k.startswith(("req_", "task_")) and not k.endswith("_n")]
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
def filed_tests():
    out = {}
    def walk(o, p):
        if isinstance(o, dict):
            if "a" in o and "b" in o and "median_pct" in o and "stat" in o:
                out.setdefault((o["a"], tuple(o["a_seeds"]), o["b"], o["rung"], o["stat"]), (p, o["median_pct"]))
                return
            for k, v in o.items():
                if k not in ("cotenancy", "vs_cd_robust", "seeds_alone"): walk(v, p + "/" + k)
        elif isinstance(o, list):
            for i, v in enumerate(o): walk(v, p + f"[{i}]")
    walk(filed, ""); return out
FT = filed_tests()
NAME = {GNN: "ra_gnn_eng_combfl_nosplit", MLP: "ra_mlp_same_combfl_nosplit", BAR: "cd_ib0_inflightfull"}
res["reproduced_n"] = {}
for comp, (ca, cb) in COMP.items():
    n = 0
    for r in RUNGS:
        for stat, key in (("whole", "mean_whole"), ("after360", "ss_mean")):
            mine = test(ca, (1, 2), cb, (0,) if cb == BAR else (1, 2), r, key)["median_pct"]
            fk = (NAME[ca], (1, 2), NAME[cb], r, stat)
            if fk in FT:
                assert abs(mine - FT[fk][1]) < 1e-9, (comp, r, stat, mine, FT[fk])
                res["table_I_reproduced"][f"{comp} {r} {stat}"] = {"mine": mine, "filed_read": FT[fk][1], "where": FT[fk][0]}; n += 1
            else:
                res["table_I_reproduced"][f"{comp} {r} {stat}"] = {"mine": mine, "filed_read": None}
    res["reproduced_n"][comp] = n
assert res["reproduced_n"]["gnn_vs_bar"] == 4, "the filed F1 must reproduce on all four tests"
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
L = [f"request_stats_size {SIZE} servers; json written to {OUT}", f"checks: {json.dumps(chk)}", "filed task-mean medians reproduced (1e-9): " + json.dumps(res["reproduced_n"]), json.dumps(res["table_I_reproduced"]), ""]
for comp in COMP:
    for r in RUNGS:
        for win in ("whole", "after360"):
            L.append(f"=== {SIZE} servers | {comp} | {r} {win} | fan-out requests (>=2 tasks) ; seed mean | s1 | s2 ; median % (faster/12; raw p)")
            for nm in ("mean", "p50", "p90", "p99"):
                row = []
                for ss in SEEDSETS:
                    e = res["tests"][f"{comp}|{ss}|{r}|req_fanout_{win}_{nm}"]
                    row.append(f"{e['median_pct']:+.2f} ({e['faster']}/12; {e['p']:.3g})")
                L.append(f"   request {nm:4s} " + "  |  ".join(row))
            for nm in ("mean", "p50", "p90", "p99"):
                k = f"{comp}|seed_mean|{r}|task_{win}_{nm}"
                if k in res["tests"]:
                    e = res["tests"][k]; L.append(f"   task    {nm:4s} {e['median_pct']:+.2f} ({e['faster']}/12; {e['p']:.3g})")
open(TXT, "w").write("\n".join(L) + "\n"); print("\n".join(L))

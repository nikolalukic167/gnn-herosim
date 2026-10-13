"""decision_cost_s0_v1 reader (docs/lineages/decision_cost_s0_v1.md, Amendments 2 and 3). Written before the data lands.

Cells: 8 arms x k in {0, 0.25, 0.5, 1, 2} x 12 topologies (19405-19416) x windows g0,g1 x rungs moderate,heavy. A cell's decision cost is k x w[arm][rung] x plans_evaluated.
k = 0 is the unsuffixed kind (cd, cd_exactS, cd_exactS_ib0 come from the classical gate directory, the other five from the grid). Whole-run mean latency only:
the summaries carry no latency by arrival time (disclosed; a steady-state split is not available from these cells).
Unit: topology; per topology the median over windows of the paired % of mean latency; median over topologies; sign count; exact Wilcoxon; negative = first arm faster.
Bar at each k: the best HAND arm, chosen per (k, rung) at the arm level by the lower median-over-topologies latency, among arms with all 12 topologies finished.
WIN (primary family: self-search and cdxapply, Holm over arm x rung at each k > 0): median <= -2 % and Holm p < .05 at BOTH rungs for the same arm.
Crossover k = the smallest grid k with a WIN, with the linear interpolation of the worse rung's median to -2 % between the previous and that k. Decision line: a WIN counts only if
the crossover <= the largest anchor k of the hand arm it beats (k_anchor = B / (w x plans per task of that hand arm), B in {1, 5, 10} ms, largest B used); else NO-GO.
nosplit and the contrasts against cd_exactS are descriptive. Utilisation = sched_busy_seconds / endTime per cell, above 1 is unstable and flagged.
Counts first; a failed.json is a count, never a rerun.

  decision_cost_read.py --gate <grid dir> <classical dir> [--w scripts_cosim/decision_cost_w.json] [--out read.json]
"""
import argparse
import json
import os
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

HAND = ("cd", "cd_ib0", "cd_exactS", "cd_exactS_ib0", "cd_exactS_ib0_prune")
GNN_PRIMARY = ("ra_gnn_eng_selfsearch", "ra_gnn_eng_cdxapply")
GNN_DESC = ("ra_gnn_eng_nosplit",)
ARMS = HAND + GNN_PRIMARY + GNN_DESC
KS = (0.0, 0.25, 0.5, 1.0, 2.0)
SUFFIX = {"_k025": 0.25, "_k050": 0.5, "_k100": 1.0, "_k200": 2.0}
TOPOS = tuple(range(19405, 19417))
WINDOWS = ("g0", "g1")
RUNGS = ("moderate", "heavy")
WIN_PCT, WIN_P = -2.0, 0.05
ANCHOR_B_MS = (1.0, 5.0, 10.0)
DEFAULT_W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_cost_w.json")


def split_kind(kind):
    """(arm, k) for a result kind; an unsuffixed kind is k = 0."""
    for tag, k in SUFFIX.items():
        if kind.endswith(tag):
            return kind[: -len(tag)], k
    return kind, 0.0


def name(arm, k):
    return f"{arm}@{k:g}"


def load(gates):
    raw, failed_raw = R.load(list(gates))
    cells, failed = {}, {}
    for src, dst in ((raw, cells), (failed_raw, failed)):
        for (kind, seed, topo, win, rung), v in src.items():
            arm, k = split_kind(kind)
            if arm in ARMS and topo in TOPOS and win in WINDOWS and rung in RUNGS:
                dst[(name(arm, k), seed, topo, win, rung)] = v
    return cells, failed


def per_topology_latency(cells, key, rung):
    out = {}
    for t in TOPOS:
        v = [float(s["averageElapsedTime"]) for (kd, sd, tp, w, r), s in cells.items() if kd == key and tp == t and r == rung]
        if v:
            out[t] = st.median(v)
    return out


def best_hand(cells, k, rung):
    """The hand arm with the lowest median-over-topologies latency at (k, rung), among arms with every topology finished."""
    best = None
    for arm in HAND:
        lat = per_topology_latency(cells, name(arm, k), rung)
        if len(lat) < len(TOPOS):
            continue
        m = st.median(lat.values())
        if best is None or m < best[1]:
            best = (arm, m)
    return best[0] if best else None


def plans_per_task(cells, arm, rung):
    p = n = 0.0
    for (kd, sd, tp, w, r), s in cells.items():
        if kd.split("@")[0] != arm:
            continue
        if r != rung:
            continue
        c = s.get("schedulerCounters") or {}
        if "sched_plan_evals" in c:
            p += float(c["sched_plan_evals"])
            n += float(s["num_tasks"])
    return p / n if n else None


def utilisation(cells):
    out = {}
    for arm in ARMS:
        for k in KS:
            for rung in RUNGS:
                u, qw = [], []
                for (kd, sd, tp, w, r), s in cells.items():
                    if kd != name(arm, k) or r != rung:
                        continue
                    c = s.get("schedulerCounters") or {}
                    if "sched_busy_seconds" in c and s.get("endTime"):
                        u.append(float(c["sched_busy_seconds"]) / float(s["endTime"]))
                        qw.append(float(c.get("sched_queue_wait_seconds", 0)) / float(s["num_tasks"]))
                if u:
                    out[f"{arm}|k={k:g}|{rung}"] = {"cells": len(u), "median_util": st.median(u), "max_util": max(u), "unstable_cells": sum(1 for x in u if x > 1.0),
                                                   "median_queue_wait_s_per_task": st.median(qw)}
    return out


def contrasts_at(cells, k):
    bar = {r: best_hand(cells, k, r) for r in RUNGS}
    prim = {}
    for arm in GNN_PRIMARY:
        for r in RUNGS:
            if bar[r] is None:
                prim[f"{arm}|{r}"] = {"bar": None, "n_topologies": 0, "median_pct": None, "p": None, "wins": None}
                continue
            c = R.contrast(cells, name(arm, k), name(bar[r], k), r, list(TOPOS))
            c["bar"] = bar[r]
            prim[f"{arm}|{r}"] = c
    adj = R.holm({key: c["p"] for key, c in prim.items()}) if k > 0 else {key: c["p"] for key, c in prim.items()}
    for key, c in prim.items():
        c["holm_p"] = adj[key]
    wins = {}
    for arm in GNN_PRIMARY:
        ok = True
        for r in RUNGS:
            c = prim[f"{arm}|{r}"]
            ok = ok and c["median_pct"] is not None and c["median_pct"] <= WIN_PCT and c["holm_p"] is not None and c["holm_p"] < WIN_P
        wins[arm] = ok
    desc = {}
    for arm in GNN_PRIMARY + GNN_DESC:
        for r in RUNGS:
            for ref in ("cd_exactS", "cd_exactS_ib0"):
                desc[f"{arm} vs {ref}|{r}"] = R.contrast(cells, name(arm, k), name(ref, k), r, list(TOPOS))
    for arm in GNN_DESC:
        for r in RUNGS:
            if bar[r]:
                desc[f"{arm} vs best hand ({bar[r]})|{r}"] = R.contrast(cells, name(arm, k), name(bar[r], k), r, list(TOPOS))
    return {"bar": bar, "primary": prim, "win": wins, "descriptive": desc}


def self_cost(cells):
    """Descriptive: each arm at k vs the same arm at k = 0 (what its own decision time costs it), paired % per rung."""
    return {f"{arm} k={k:g} vs k=0|{r}": R.contrast(cells, name(arm, k), name(arm, 0.0), r, list(TOPOS)) for arm in ARMS for k in KS[1:] for r in RUNGS}


def crossover(by_k, arm):
    """Smallest grid k > 0 with a WIN; interpolated k where the worse-rung median crosses -2 % between the previous k and that k."""
    prev = None
    for k in KS[1:]:
        c = by_k[k]["primary"]
        worst = max((c[f"{arm}|{r}"]["median_pct"] for r in RUNGS), default=None) if all(c[f"{arm}|{r}"]["median_pct"] is not None for r in RUNGS) else None
        if by_k[k]["win"][arm]:
            interp = None
            if prev is not None and prev[1] is not None and worst is not None and prev[1] > WIN_PCT >= worst and prev[1] != worst:
                interp = prev[0] + (k - prev[0]) * (prev[1] - WIN_PCT) / (prev[1] - worst)
            return {"grid_k": k, "interpolated_k": interp}
        prev = (k, worst)
    return {"grid_k": None, "interpolated_k": None}


def in_budget(cells, anc):
    """GNN primary arms vs the best IN-BUDGET hand arm at each k and rung (k_anchor at the largest B, per rung, >= k; complete arms only), plus each GNN arm's own k_anchor."""
    out = {}
    top = f"B={max(ANCHOR_B_MS):g}ms"
    for k in KS[1:]:
        row = {}
        for r in RUNGS:
            best = None
            for arm in HAND:
                ka = anc[arm]["per_rung"][r]["k_anchor"][top]
                lat = per_topology_latency(cells, name(arm, k), r)
                if ka is None or ka < k or len(lat) < len(TOPOS):
                    continue
                m = st.median(lat.values())
                if best is None or m < best[1]:
                    best = (arm, m)
            for g in GNN_PRIMARY:
                c = R.contrast(cells, name(g, k), name(best[0], k), r, list(TOPOS)) if best else None
                row[f"{g}|{r}"] = {"bar": best[0] if best else None, "own_k_anchor": anc[g]["per_rung"][r]["k_anchor"][top], "contrast": c}
        out[f"{k:g}"] = row
    return out


def anchors(cells, w):
    """k_anchor per hand arm and rung: B / (w x plans per task), B in ANCHOR_B_MS. Largest B and the larger rung value are what the decision line uses."""
    out = {}
    for arm in ARMS:
        per = {}
        for r in RUNGS:
            ppt = plans_per_task(cells, arm, r)
            wv = (w.get(arm) or {}).get(r)
            per[r] = {"plans_per_task": ppt, "w_ms_per_plan": wv,
                      "k_anchor": {f"B={b:g}ms": (b / (wv * ppt) if ppt and wv else None) for b in ANCHOR_B_MS}}
        vals = [per[r]["k_anchor"][f"B={max(ANCHOR_B_MS):g}ms"] for r in RUNGS if per[r]["k_anchor"][f"B={max(ANCHOR_B_MS):g}ms"] is not None]
        out[arm] = {"per_rung": per, "largest_anchor_k": max(vals) if vals else None}
    return out


def read(gates, w_path=DEFAULT_W):
    cells, failed = load(gates)
    w = json.load(open(w_path))["w_ms_per_plan"]
    counts = {f"{arm}@{k:g}|{r}": {"finished": sum(1 for x in cells if x[0] == name(arm, k) and x[4] == r),
                                   "failed": sum(1 for x in failed if x[0] == name(arm, k) and x[4] == r), "expected": len(TOPOS) * len(WINDOWS)}
              for arm in ARMS for k in KS for r in RUNGS}
    by_k = {k: contrasts_at(cells, k) for k in KS}
    anc = anchors(cells, w)
    cross = {arm: crossover(by_k, arm) for arm in GNN_PRIMARY}
    decision = {}
    for arm in GNN_PRIMARY:
        ck = cross[arm]["grid_k"]
        if ck is None:
            decision[arm] = {"result": "NO-GO", "why": "no grid k shows a WIN (crossover above the largest k = 2)"}
            continue
        beaten = {r: by_k[ck]["bar"][r] for r in RUNGS}
        ref_k = cross[arm]["interpolated_k"] if cross[arm]["interpolated_k"] is not None else ck
        lim = {r: (anc[beaten[r]]["largest_anchor_k"] if beaten[r] else None) for r in RUNGS}
        known = [v for v in lim.values() if v is not None]
        strict = bool(known) and len(known) == len(RUNGS) and all(ref_k <= v for v in known)  # each rung's beaten hand arm must be in budget at the crossover
        permissive = bool(known) and ref_k <= max(known)  # the largest anchor over the arms beaten
        decision[arm] = {"result": "GO" if strict else "NO-GO (strict); GO under the permissive reading" if permissive else "NO-GO", "crossover_k": ref_k, "grid_k": ck,
                         "hand_arms_beaten": beaten, "largest_anchor_k_per_rung": lim, "strict": strict, "permissive": permissive,
                         "why": "the node says 'the hand arm it beats' (singular); with a different bar per rung both readings are reported and the ruling is the coordinator's"}
    return {"gate": list(gates), "counts": counts, "failed_why": sorted({str(f.get("why", "?"))[:90] for f in failed.values()}),
            "by_k": {f"{k:g}": v for k, v in by_k.items()}, "crossover": cross, "anchors": anc, "decision": decision, "in_budget": in_budget(cells, anc), "utilisation": utilisation(cells), "self_cost": self_cost(cells)}


def print_report(r):
    f = lambda x: "-" if x is None else f"{x:.4f}"
    print("-- counts (finished / failed / expected), arms x k x rung")
    for key, c in r["counts"].items():
        if c["finished"] or c["failed"]:
            print(f"  {key:44s} {c['finished']}/{c['failed']}/{c['expected']}")
    print(f"failed why: {r['failed_why']}")
    for k, v in r["by_k"].items():
        print(f"== k = {k}: best hand arm {v['bar']}  WIN {v['win']}")
        for key, c in v["primary"].items():
            m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  {key:36s} bar {str(c.get('bar')):22s} median {m:>9s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])} holm={f(c.get('holm_p'))}")
        for key, c in v["descriptive"].items():
            m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  (desc) {key:56s} median {m:>9s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])}")
    print("-- crossover", r["crossover"])
    print("-- anchors (k_anchor = B / (w x plans per task))")
    for arm, a in r["anchors"].items():
        print(f"  {arm:22s} largest {f(a['largest_anchor_k'])}  " + "  ".join(f"{rg}: ppt {f(v['plans_per_task'])} " + ",".join(f"{b}->{f(x)}" for b, x in v["k_anchor"].items()) for rg, v in a["per_rung"].items()))
    print("-- utilisation (busy / end time; above 1 = unstable)")
    for key, u in r["utilisation"].items():
        flag = "  UNSTABLE" if u["unstable_cells"] else ""
        print(f"  {key:46s} cells {u['cells']:3d} median {u['median_util']:.3f} max {u['max_util']:.3f} qwait/task {u['median_queue_wait_s_per_task']:.4f}s{flag}")
    print("-- vs the best IN-BUDGET hand arm (k_anchor at B = 10 ms >= k, per rung) and the GNN arm's own k_anchor")
    for k, row in r["in_budget"].items():
        for key, v in row.items():
            c = v["contrast"]
            m = "-" if not c or c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  k={k:5s} {key:34s} bar {str(v['bar']):14s} median {m:>9s} wins={c['wins'] if c else None} p={f(c['p']) if c else '-'}  own k_anchor {f(v['own_k_anchor'])}")
    print("-- self cost (arm at k vs the same arm at k = 0; positive = slower because of its own decision time)")
    for key, c in r["self_cost"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {key:46s} median {m:>9s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])}")
    print("-- decision", json.dumps(r["decision"], indent=1))


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--w", default=DEFAULT_W)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate, a.w)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

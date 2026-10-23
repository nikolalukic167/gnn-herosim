#!/usr/bin/env python3
"""Label search and exact search under the decision charge: the read (uncut_groups_s0_v1, registered 2026-10-20 "Label search and exact search under the
decision charge", amendment A1). A validity control, descriptive, no label.

Arms (24 servers, topologies 19537-19548, windows g0-g3, moderate and heavy, frozen _k100, TCP and fair-share off):
  bar    cd_ib0_inflightfull_k100 s0     filed cells dose_7cb483c5/cells_ref_{mod,heavy}070   (pin 7cb483c5)
  gnn    ra_gnn_eng_combfl_nosplit_k100 s1, s2   filed cells dose_7cb483c5/cells_gnn_{mod,heavy}070   (pin 7cb483c5)
  ils<B> cd_ib0_ils[_p<B>]_k100 s0       ils_live/cells_ils_b<B>  (ILS-live at budget B in {200, 50, 10}: the bar's plan, then the label engine's ILS on the bar's S,
                                          charged at w_S)
  exact  cd_exactS_ib0_live_k100 s0      ils_live/cells_exact  (exact-live: cd_exactS_ib0 on groups of at most G = 8, the bar's plan above)
Amendment A2: the ILS arm is read at a budget per load, fixed by the filed feasibility projection (ils_live_projection.py): the largest budget in {200, 50, 10} whose
projected heavy rho_sched < 1 (the primary budget; read at both loads when it is also feasible at moderate); a budget with rho_sched >= 1 at a load is "infeasible
live at that load", is not run, and its tests enter with p = 1 (counted in the Holm 4). If no budget is feasible at heavy, the heavy tests are infeasible and moderate
is read at the largest budget feasible there. Every other feasible budget is a descriptive ladder row (contrast vs the bar and vs FL gnn, raw p, no Holm).
FREEZE FILE (--freeze): one entry per arm root with its pin, cell list (path, md5) and the md5 of its w entries in decision_cost_w.json, plus the projection file (path,
md5). Each entry's md5 is filed in the Record with its pin before its cells run; the reader code (this file) is frozen once, before any arm cell.

PRIMARY is the frozen reader's own statistic, imported and blob-checked (uncut_arm_gate_read.contrast, the Table I statistic): per cell 100 (arm - cmp) / cmp, a
learned arm the mean of its two seeds, the median over the four windows per topology, the median over the 12 topologies, exact two-sided Wilcoxon over
topologies. Holm 4 (2 loads x whole / after 360 s) in each of three families: F-ils-bar (ILS-live vs bar), F-ils-gnn (FL gnn vs ILS-live; negative = FL gnn
faster), F-exact-bar (exact-live vs bar). No correction across families. The window-mean statistic is printed beside the primary and never Holm-ed.

DESCRIPTIVE, per arm and load (mean over the 12 topologies of the per-topology mean over windows): mean latency whole / after 360 s; charged decision seconds per
task (sched_busy_seconds / tasks) and the decision share of latency, pooled ((queue wait + busy) / tasks / latency) and felt (sched_felt_task_seconds / tasks /
latency: every task of a batch waits the batch's whole charge); ILS-live: batches the ILS ran on, batches of at most 10 left to the bar's plan, share of ILS
batches whose plan the ILS changed, plans evaluated per ILS batch, ILS batches stopped on the budget vs converged; exact-live: batches on the bar's plan
(more than G = 8 tasks, pg_exact_g_skipped), exhaustive batches (pg_exact_batches) and expanded batches (pg_exact_fallbacks, the slate product above
100,000 plans); the plan-quality gap with the charge removed: mean latency less the pooled decision seconds per task, per cell, then the same median and
Wilcoxon over topologies (the bar's cells predate the felt counter, so the removal is the pooled one for every arm; the felt-removed gap is printed for the
two new arms against each other only where both carry the counter). The exact-live gap is also printed on groups of at most G: not derivable per cell
(the summary has no per-size latency), so it is reported as the gap on the whole cell and the exhaustive / expanded / bar-plan batch counts beside it.

Per cell, asserted (FAIL LOUD): the exact grid (every cell present, none failed, no extra cell in a root); _k100 only; code.commit is the arm's pin and clean (new
arms: the freeze entry's pin; filed arms: 7cb483c5); servers 24 and the topology; scaleOut.target 0.7 and no knative_panic; no HEROSIM_SNAPSHOT_FIDELITY /
PG_ORACLE / KPA_KNATIVE_PANIC / EXCHANGE_TCP / EXCHANGE_FAIRSHARE / LIVE_AUDIT_* in env; the arm's own instrument and charge env equal the frozen
decision_cost_w.json (the freeze entry's w md5) times k = 1 (HEROSIM_DECISION_COST_MS or the bucket object; for ILS-live also HEROSIM_DECISION_COST_MS_S_BY_BUCKET); the
instrument counters ran; FL gnn cells' checkpoint basename and md5 against comb/md5_comball.json. Latency keys only; never total_rtt_plus_inference.
The R1A_STALL_S = 7200 limit of the two new arms is a driver setting the summary does not record; a cell that tripped it has a .failed.json and fails the read.
Output is written once."""
import argparse
import hashlib
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FROZEN_BLOBS = {"uncut_arm_gate_read.py": "04e368af8fbd9eb462ae68945676cc850ea808a9",
                "r1_attribution_v1_read.py": "f72bbb8fbbc423b70b073b7e6f3cc431b68bc5fc"}


def git_blob(path):
    b = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


# each frozen module is loaded at its blob (git object store when the working file has since changed), never the working file
from frozen_import import load_frozen  # noqa: E402
G = load_frozen(HERE, "uncut_arm_gate_read.py", FROZEN_BLOBS["uncut_arm_gate_read.py"])
exact_wilcoxon = load_frozen(HERE, "r1_attribution_v1_read.py", FROZEN_BLOBS["r1_attribution_v1_read.py"]).exact_wilcoxon

FILED_PIN = "7cb483c5"
BUDGETS = ("200", "50", "10")
ILS_KIND = {"200": "cd_ib0_ils", "50": "cd_ib0_ils_p50", "10": "cd_ib0_ils_p10"}
ILS_ARMS = tuple(f"ils{b}" for b in BUDGETS)
ARMS = ("bar", "gnn", "exact") + ILS_ARMS
SEEDS = {"bar": (0,), "gnn": (1, 2), "exact": (0,), **{a: (0,) for a in ILS_ARMS}}
ARM_NAME = {"bar": "cd_ib0_inflightfull_k100", "gnn": "ra_gnn_eng_combfl_nosplit_k100", "exact": "cd_exactS_ib0_live_k100",
            **{f"ils{b}": ILS_KIND[b] + "_k100" for b in BUDGETS}}
ARM_KIND = {"exact": "cd_exactS_ib0_live", **{f"ils{b}": ILS_KIND[b] for b in BUDGETS}}
ARM_ROOT = {"exact": "cells_exact", **{f"ils{b}": f"cells_ils_b{b}" for b in BUDGETS}}
TOPOS = list(range(19537, 19549))
RUNGS, WINDOWS, STATS = G.RUNGS, G.WINDOWS, G.STATS
BAD_ENV = ("HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_ORACLE", "HEROSIM_KPA_KNATIVE_PANIC", "HEROSIM_EXCHANGE_TCP", "HEROSIM_EXCHANGE_FAIRSHARE")
CKPT_FILE = "uncut-comb-fl-gnn-eng-seed{}.pt"
G_CAP = 8
ILS_PER = {"200": "200", "50": "50", "10": "10"}


def md5_file(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def w_entries_md5(w_file, arm):
    """md5 of the canonical JSON of the w entries an arm's cells are charged from (so freezing one arm's w leaves the other's entry md5 unchanged)."""
    w = json.load(open(w_file))
    k = ARM_KIND[arm]
    ent = {"w_ms_per_plan": w.get("w_ms_per_plan", {}).get(k), "w_ms_per_plan_by_bucket": w.get("w_ms_per_plan_by_bucket", {}).get(k)}
    if arm.startswith("ils"):
        ent["w_S_ms_per_plan_by_bucket"] = w.get("w_S_ms_per_plan_by_bucket", {}).get(k)
    if arm == "exact":
        ent["exact_live_G"] = w.get("exact_live_G")
    return hashlib.md5(json.dumps(ent, sort_keys=True).encode()).hexdigest()


def load_freeze(U, path, w_file):
    """The freeze file: {"arms": {arm: {"pin", "list", "list_md5", "w_md5"}}, "projection": {"file", "md5"}}; every md5 is recomputed here."""
    fz = json.load(open(path))
    proj = fz["projection"] if "projection" in fz else None
    for arm, e in fz["arms"].items():
        if arm not in ARM_ROOT:
            raise SystemExit(f"FAIL LOUD: freeze entry for unknown arm {arm}")
        if len(e["pin"]) != 40:
            raise SystemExit(f"FAIL LOUD: {arm}: pin must be a full sha")
        if md5_file(f"{U}/{e['list']}") != e["list_md5"]:
            raise SystemExit(f"FAIL LOUD: {arm}: cell list md5 differs from the freeze entry")
        if w_entries_md5(w_file, arm) != e["w_md5"]:
            raise SystemExit(f"FAIL LOUD: {arm}: its w entries in decision_cost_w.json differ from the freeze entry")
    rho = None
    if proj is not None:
        if md5_file(f"{U}/{proj['file']}") != proj["md5"]:
            raise SystemExit("FAIL LOUD: projection file md5 differs from the freeze file")
        rho = json.load(open(f"{U}/{proj['file']}"))["rho"]
    return fz, rho


def ils_budget_per_load(rho):
    """A2's rule. {load: budget or None}. The primary budget is the largest in {200, 50, 10} whose projected heavy rho_sched < 1; it is read at moderate too when
    feasible there. If none is feasible at heavy, heavy is infeasible and moderate is read at the largest budget feasible at moderate."""
    ok = lambda b, r: rho[r][b]["rho_median"] < 1.0
    bh = next((b for b in BUDGETS if ok(b, "heavy")), None)
    if bh is not None:
        return {"heavy": bh, "moderate": bh if ok(bh, "moderate") else None}, bh
    return {"heavy": None, "moderate": next((b for b in BUDGETS if ok(b, "moderate")), None)}, None


def feasible_loads(rho, b):
    return [r for r in RUNGS if rho[r][b]["rho_median"] < 1.0]


def roots(U, fz):
    out = {}
    for r, short in (("moderate", "mod"), ("heavy", "heavy")):
        A = f"{U}/anomaly/dose_7cb483c5"
        out[("bar", r)] = (f"{A}/cells_ref_{short}070", FILED_PIN)
        out[("gnn", r)] = (f"{A}/cells_gnn_{short}070", FILED_PIN)
    for arm, e in fz["arms"].items():
        for r in RUNGS:
            out[(arm, r)] = (f"{U}/ils_live/{ARM_ROOT[arm]}", e["pin"])
    return out


def required_arms(fz, rho):
    """Which (arm, load) pairs the grid holds: bar, gnn and exact at both loads; each ILS budget in the freeze at its feasible loads."""
    req = [(a, r) for a in ("bar", "gnn") for r in RUNGS]
    if "exact" in fz["arms"]:
        req += [("exact", r) for r in RUNGS]
    for arm in fz["arms"]:
        if arm.startswith("ils"):
            if rho is None:
                raise SystemExit("FAIL LOUD: an ILS arm in the freeze file without a projection")
            fl = feasible_loads(rho, arm[3:])
            if not fl:
                raise SystemExit(f"FAIL LOUD: {arm} is in the freeze file but is infeasible live at both loads")
            req += [(arm, r) for r in fl]
    return req


def stem_of(rt, arm, rung, t, w, sd):
    return f"{rt[(arm, rung)][0]}/cc40s{t}__{w}{rung}__{ARM_NAME[arm]}_s{sd}"


def grid(rt, req):
    return {(arm, sd, t, w, r): stem_of(rt, arm, r, t, w, sd) for arm, r in req for t in TOPOS for w in WINDOWS for sd in SEEDS[arm]}


def check_lists(U, fz, g):
    """Each new arm's cells of the grid equal its list's cell entries (identity entries never opened)."""
    n_cells = n_ident = 0
    for arm, e in fz["arms"].items():
        listed = set()
        for line in open(f"{U}/{e['list']}").read().splitlines():
            root, name = line.split("\t")[:2]
            if "/ident_" in root:
                n_ident += 1
                continue
            listed.add((f"{U}/{root}", name))
        want = {(os.path.dirname(stem), os.path.basename(stem)) for (a, *_), stem in g.items() if a == arm}
        # a list names both loads; a budget infeasible at a load drops those cells from the grid and from the run, so compare on the grid's loads
        listed = {x for x in listed if any(f"__{w}{r}__" in x[1] for w in WINDOWS for r in {k[4] for k in g if k[0] == arm})}
        if listed != want:
            raise SystemExit(f"FAIL LOUD: {arm}: cell list and grid differ: {len(listed - want)} listed only {sorted(listed - want)[:3]}, "
                             f"{len(want - listed)} grid only {sorted(want - listed)[:3]}")
        n_cells += len(listed)
    return n_cells, n_ident


def expected_charge(w_file, kind, rung):
    w = json.load(open(w_file))
    out = {}
    byb = w.get("w_ms_per_plan_by_bucket", {}).get(kind, {})
    if rung in byb:
        out["HEROSIM_DECISION_COST_MS_BY_BUCKET"] = {b: float(v) for b, v in byb[rung].items()}
    else:
        out["HEROSIM_DECISION_COST_MS"] = float(w["w_ms_per_plan"][kind][rung])
    if kind in ILS_KIND.values():
        out["HEROSIM_DECISION_COST_MS_S_BY_BUCKET"] = {b: float(v) for b, v in w["w_S_ms_per_plan_by_bucket"][kind][rung].items()}
    return out


def same_charge(got, want):
    """The served env string equals the frozen w times k = 1 (a number, or a JSON object keyed by the size buckets); both absent also agree."""
    if got is None or want is None:
        return got is None and want is None
    got = json.loads(got) if got.startswith("{") else float(got)
    if isinstance(want, dict):
        return isinstance(got, dict) and set(got) == set(want) and all(abs(got[b] - want[b]) <= 1e-12 for b in want)
    return not isinstance(got, dict) and abs(got - want) <= 1e-12


def one(rt, key, stem, ckpt, w_file):
    arm, sd, t, w, r = key
    if os.path.exists(stem + ".failed.json"):
        raise SystemExit(f"FAIL LOUD: failed cell {stem}")
    if not os.path.exists(stem + ".summary.json"):
        raise SystemExit(f"FAIL LOUD: missing cell {stem}")
    s = json.load(open(stem + ".summary.json"))
    s["_path"] = stem + ".summary.json"
    pin = rt[(arm, r)][1]
    c = s["code"]
    if not c["commit"].startswith(pin) or c["dirty"]:
        raise SystemExit(f"FAIL LOUD: pin {stem}: {c['commit'][:8]} dirty={c['dirty']}, want {pin[:8]}")
    if s["servers"] != 24 or s["topology"] != t:
        raise SystemExit(f"FAIL LOUD: servers/topology {stem}")
    if "knative_panic" in s["scaleOut"] or s["scaleOut"]["target"] != 0.7:
        raise SystemExit(f"FAIL LOUD: scaleOut {stem}")
    bad = [k for k in s["env"] if k in BAD_ENV or k.startswith("LIVE_AUDIT_")]
    if bad:
        raise SystemExit(f"FAIL LOUD: env {stem}: {bad}")
    if s.get("peerExchangeTcp") is not None:
        raise SystemExit(f"FAIL LOUD: TCP block in {stem}")
    if arm == "gnn":
        path = s["env"].get("GNN_MODEL_PATH")
        if path is None or os.path.basename(path) != CKPT_FILE.format(sd):
            raise SystemExit(f"FAIL LOUD: model path {stem}: {path}")
        if path not in ckpt["seen"]:
            ckpt["seen"][path] = hashlib.md5(open(path, "rb").read()).hexdigest()
        if ckpt["seen"][path] != ckpt["md5"][f"ra_gnn_eng_combfl_nosplit|s{sd}"]:
            raise SystemExit(f"FAIL LOUD: checkpoint md5 {stem}")
    is_ils = arm in ILS_ARMS
    if arm in ARM_KIND:
        e, cnt = s["env"], s["schedulerCounters"]
        need = {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_INFLIGHT_FULL": "1"}
        need.update({"HEROSIM_PG_ILS_PER": ILS_PER[arm[3:]], "HEROSIM_PG_ILS_SEED": "0"} if is_ils else
                    {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_EXACT_G": str(G_CAP)})
        for k, v in need.items():
            if e.get(k) != v:
                raise SystemExit(f"FAIL LOUD: {stem}: {k} served {e.get(k)!r}, want {v!r}")
        want = expected_charge(w_file, ARM_KIND[arm], r)
        for var in ("HEROSIM_DECISION_COST_MS", "HEROSIM_DECISION_COST_MS_BY_BUCKET", "HEROSIM_DECISION_COST_MS_S_BY_BUCKET"):
            if not same_charge(e.get(var), want.get(var)):
                raise SystemExit(f"FAIL LOUD: {stem}: {var} served {e.get(var)!r}, want {want.get(var)!r}")
        if is_ils and not (cnt.get("pg_ils_batches", 0) > 0 and cnt.get("sched_plan_evals_s", 0) > 0):
            raise SystemExit(f"FAIL LOUD: ILS instrument off {stem}")
        if arm == "exact" and not cnt.get("pg_exact_batches", 0) > 0:
            raise SystemExit(f"FAIL LOUD: exact instrument off {stem}")
        if arm == "exact" and cnt.get("pg_ils_batches"):
            raise SystemExit(f"FAIL LOUD: exact-live cell ran the ILS {stem}")
        if is_ils and (cnt.get("pg_exact_batches") or cnt.get("pg_expand_batches")):
            raise SystemExit(f"FAIL LOUD: ILS-live cell ran the exact search or the expansion {stem}")
        if not cnt.get("sched_busy_seconds", 0) > 0 or "sched_felt_task_seconds" not in cnt:
            raise SystemExit(f"FAIL LOUD: decision charge / felt counter missing {stem}")
    n = int(s["num_tasks"])
    cnt = s["schedulerCounters"]
    lat = float(s["averageElapsedTime"])
    pooled = (float(cnt.get("sched_queue_wait_seconds", 0.0)) + float(cnt.get("sched_busy_seconds", 0.0))) / n
    row = {"whole": lat, "after360": float(s["steady_state"]["after_360s"]["mean"]),
           "busy_per_task": float(cnt.get("sched_busy_seconds", 0.0)) / n, "share_pooled": 100.0 * pooled / lat,
           "share_felt": (100.0 * float(cnt["sched_felt_task_seconds"]) / n / lat) if "sched_felt_task_seconds" in cnt else None,
           "whole_nocharge": lat - pooled,
           "tail": float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > G.TAIL_S}
    for k in ("pg_ils_batches", "pg_ils_small_batches", "pg_ils_improved", "pg_ils_evals", "pg_ils_stop_evals", "pg_ils_stop_converged", "pg_ils_stop_other",
              "pg_exact_g_skipped", "pg_exact_batches", "pg_exact_fallbacks"):
        row[k] = int(cnt.get(k, 0))
    return s, row


def load_all(U, fz, rho, w_file):
    rt = roots(U, fz)
    req = required_arms(fz, rho)
    g = grid(rt, req)
    ckpt = {"seen": {}, "md5": json.load(open(f"{U}/comb/md5_comball.json"))}
    cells, rows = {}, {}
    for key, stem in g.items():
        s, row = one(rt, key, stem, ckpt, w_file)
        cells[key], rows[key] = s, row
    for arm in fz["arms"]:
        d = rt[(arm, "moderate")][0]
        extra = {f for f in os.listdir(d) if f.endswith(".summary.json")} - {os.path.basename(v) + ".summary.json" for k, v in g.items() if k[0] == arm}
        if extra:
            raise SystemExit(f"FAIL LOUD: {len(extra)} cells in {d} outside the grid: {sorted(extra)[:3]}")
    return cells, rows, g, req


def topo_vals(rows, a, b, rung, key):
    out = {}
    for t in TOPOS:
        va = [sum(rows[(a, sd, t, w, rung)][key] for sd in SEEDS[a]) / len(SEEDS[a]) for w in WINDOWS]
        vb = [sum(rows[(b, sd, t, w, rung)][key] for sd in SEEDS[b]) / len(SEEDS[b]) for w in WINDOWS]
        out[t] = 100.0 * (sum(va) / len(va) - sum(vb) / len(vb)) / (sum(vb) / len(vb))
    return out


def infeasible_entry(a, b, rung, stat, why):
    return {"a": a, "b": b, "rung": rung, "stat": stat, "n_topologies": len(TOPOS), "n_cells": 0, "median_pct": None, "faster": 0, "p": 1.0,
            "infeasible": why, "wm_median": None, "wm_faster": 0}


def entry(cells, rows, a, b, rung, stat):
    e = G.contrast(cells, a, SEEDS[a], b, SEEDS[b], rung, TOPOS, stat)
    wm = topo_vals(rows, a, b, rung, "whole" if stat == "whole" else "after360")
    e["wm_median"], e["wm_faster"] = st.median(wm.values()), sum(1 for x in wm.values() if x < 0)
    return e


def build(cells, rows, req, rho):
    have = set(req)
    bud, primary = ils_budget_per_load(rho) if rho is not None else ({}, None)
    res = {"budget": bud, "primary": primary, "fam": {}, "ladder": {}, "nocharge": {}}
    for name, a, b in (("F-ils-bar", "ils", "bar"), ("F-ils-gnn", "gnn", "ils"), ("F-exact-bar", "exact", "bar")):
        if a == "exact" and ("exact", "moderate") not in have:
            continue
        if "ils" in (a, b) and not any(x[0].startswith("ils") for x in have):
            continue
        es = []
        for r in RUNGS:
            for stat in STATS:
                if "ils" in (a, b):
                    bd = bud.get(r)
                    if bd is None or (f"ils{bd}", r) not in have:
                        es.append(infeasible_entry(a, b, r, stat, "infeasible live at this load (projected rho_sched >= 1 at every budget, or none cut)"))
                        continue
                    aa = f"ils{bd}" if a == "ils" else a
                    bb = f"ils{bd}" if b == "ils" else b
                    e = entry(cells, rows, aa, bb, r, stat)
                    e["budget"] = bd
                else:
                    e = entry(cells, rows, a, b, r, stat)
                es.append(e)
        G.family(es)
        res["fam"][name] = es
        for r in RUNGS:
            if "ils" in (a, b) and bud.get(r) is None:
                continue
            aa = f"ils{bud[r]}" if a == "ils" else a
            bb = f"ils{bud[r]}" if b == "ils" else b
            if (aa, r) in have and (bb, r) in have:
                v = topo_vals(rows, aa, bb, r, "whole_nocharge")
                res["nocharge"][(name, r)] = (st.median(v.values()), sum(1 for x in v.values() if x < 0), exact_wilcoxon(list(v.values())))
    for arm in ILS_ARMS:
        for r in RUNGS:
            if (arm, r) in have:
                for other in ("bar", "gnn"):
                    res["ladder"][(arm, r, other)] = [entry(cells, rows, arm, other, r, stat) for stat in STATS]
    return res


def per_arm(rows, cells, req):
    out = {}
    for arm, r in req:
        def tm(k):
            vals = []
            for t in TOPOS:
                cell = [rows[(arm, sd, t, w, r)][k] for w in WINDOWS for sd in SEEDS[arm]]
                if any(c is None for c in cell):
                    return None
                vals.append(sum(cell) / len(cell))
            return sum(vals) / len(vals)
        d = {k: tm(k) for k in ("whole", "after360", "busy_per_task", "share_pooled", "share_felt", "pg_ils_batches", "pg_ils_small_batches", "pg_ils_improved",
                                "pg_ils_evals", "pg_ils_stop_evals", "pg_ils_stop_converged", "pg_exact_g_skipped", "pg_exact_batches", "pg_exact_fallbacks")}
        d["tail"] = sum(rows[(arm, sd, t, w, r)]["tail"] for t in TOPOS for w in WINDOWS for sd in SEEDS[arm])
        d["collapse"] = sum(any(G.means(cells[(arm, sd, t, w, r)])[k] > G.COLLAPSE_X * G.cell_mean(cells, "bar", (0,), t, w, r)[k] for k in STATS)
                            for t in TOPOS for w in WINDOWS for sd in SEEDS[arm])
        out[(arm, r)] = d
    return out


def fp(p):
    return "n/a" if p is None else format(p, ".4g")


def fmt_e(e):
    if e.get("infeasible"):
        return f"      {e['rung']:8s} {e['stat']:8s} INFEASIBLE LIVE ({e['infeasible']}); p = 1 enters the Holm 4  holm {fp(e.get('p_holm'))}"
    return (f"      {e['rung']:8s} {e['stat']:8s} median {e['median_pct']:+8.3f} % ({e['faster']:2d}/{e['n_topologies']})  p {fp(e['p'])}"
            + (f"  holm {fp(e['p_holm'])}" if "p_holm" in e else "") + (f"  [budget {e['budget']}]" if "budget" in e else "")
            + f"   | window-mean {e['wm_median']:+8.3f} % ({e['wm_faster']}/{e['n_topologies']})")


def report(fz, rho, res, pa, ncells, nlist, req):
    L = [__doc__.strip(), "", f"=== cells read and asserted: {ncells}; cell-list entries {nlist[0]} (identity entries {nlist[1]} never opened)",
         "=== freeze entries: " + json.dumps(fz["arms"], sort_keys=True), "=== projection: " + json.dumps(fz.get("projection")),
         "=== budget per load (A2 rule): " + json.dumps(res["budget"]) + f"; primary budget {res['primary']}"]
    for name, es in res["fam"].items():
        L.append(f"\n  -- {name}   (primary (Table I), Holm within the family of 4 | window-mean)")
        L += [fmt_e(e) for e in es]
    L.append("\n  -- plan quality with the charge removed (mean latency less the pooled decision seconds per task; median over topologies, topologies faster, raw exact Wilcoxon p; descriptive)")
    for (name, r), (m, f, p) in res["nocharge"].items():
        L.append(f"      {name:12s} {r:8s} {m:+8.3f} % ({f}/12)  p {fp(p)}")
    if res["ladder"]:
        L.append("\n  -- budget ladder (descriptive, raw p, no Holm): ILS-live at each feasible budget and load vs the bar and vs FL gnn")
        for (arm, r, other), es in res["ladder"].items():
            for e in es:
                L.append(f"      {arm:6s} vs {other:3s} {r:8s} {e['stat']:8s} median {e['median_pct']:+8.3f} % ({e['faster']}/{e['n_topologies']})  p {fp(e['p'])}")
    L.append("\n  -- per arm and load, mean over 12 topologies of the per-topology window mean")
    for (arm, r), d in pa.items():
        L.append(f"    {arm:6s} {r:8s} latency whole {d['whole']:.3f} / after360 {d['after360']:.3f} s; charged decision s/task {d['busy_per_task']:.4f}; "
                 f"decision share pooled {d['share_pooled']:.3f} %" + (f", felt {d['share_felt']:.3f} %" if d["share_felt"] is not None else ", felt n/a (filed cell)")
                 + f"; tail flags {d['tail']}, collapse cells (> {G.COLLAPSE_X:g} x the bar) {d['collapse']}")
        if arm in ILS_ARMS:
            b = max(d["pg_ils_batches"], 1)
            L.append(f"             ILS batches {d['pg_ils_batches']:.0f} (batches of <= 10 left to the bar's plan {d['pg_ils_small_batches']:.0f}); plan changed in "
                     f"{100 * d['pg_ils_improved'] / b:.1f} % of ILS batches; plans per ILS batch {d['pg_ils_evals'] / b:.0f}; stopped on the budget {d['pg_ils_stop_evals']:.0f}, "
                     f"converged {d['pg_ils_stop_converged']:.0f}")
        if arm == "exact":
            L.append(f"             batches on the bar's plan (> {G_CAP} tasks) {d['pg_exact_g_skipped']:.0f}; exhaustive {d['pg_exact_batches']:.0f}; "
                     f"expanded (pg_exact_fallbacks) {d['pg_exact_fallbacks']:.0f}")
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv[1:])
    U = os.environ.get("ILS_LIVE_U", "/share/nikola.lukic/uncut_groups")
    w_file = os.path.join(HERE, "decision_cost_w.json")
    out = a.out or f"{U}/ils_live/ils_live_read.txt"
    fz, rho = load_freeze(U, a.freeze, w_file)
    cells, rows, g, req = load_all(U, fz, rho, w_file)
    nlist = check_lists(U, fz, g)
    res = build(cells, rows, req, rho)
    txt = report(fz, rho, res, per_arm(rows, cells, req), len(g), nlist, req)
    print(txt)
    if os.path.exists(out):
        raise SystemExit("FAIL LOUD: output exists, the read is written once")
    open(out + ".partial", "w").write(txt + "\n")
    os.replace(out + ".partial", out)


if __name__ == "__main__":
    main(sys.argv)

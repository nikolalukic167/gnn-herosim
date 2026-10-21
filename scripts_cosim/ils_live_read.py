#!/usr/bin/env python3
"""Label search and exact search under the decision charge: the read (uncut_groups_s0_v1, registered 2026-10-20 "Label search and exact search under the
decision charge", amendment A1). A validity control, descriptive, no label.

Arms (24 servers, topologies 19537-19548, windows g0-g3, moderate and heavy, frozen _k100, TCP and fair-share off):
  bar    cd_ib0_inflightfull_k100 s0     filed cells dose_7cb483c5/cells_ref_{mod,heavy}070   (pin 7cb483c5)
  gnn    ra_gnn_eng_combfl_nosplit_k100 s1, s2   filed cells dose_7cb483c5/cells_gnn_{mod,heavy}070   (pin 7cb483c5)
  ils    cd_ib0_ils_k100 s0              ils_live/cells_ils    (ILS-live: the bar's plan, then the label engine's ILS on the bar's S, charged at w_S)
  exact  cd_exactS_ib0_live_k100 s0      ils_live/cells_exact  (exact-live: cd_exactS_ib0 on groups of at most G = 8, the bar's plan above)

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
arms: the frozen ARM_PIN; filed arms: 7cb483c5); servers 24 and the topology; scaleOut.target 0.7 and no knative_panic; no HEROSIM_SNAPSHOT_FIDELITY /
PG_ORACLE / KPA_KNATIVE_PANIC / EXCHANGE_TCP / EXCHANGE_FAIRSHARE / LIVE_AUDIT_* in env; the arm's own instrument and charge env equal the frozen
decision_cost_w.json (md5 W_MD5) times k = 1 (HEROSIM_DECISION_COST_MS or the bucket object; for ILS-live also HEROSIM_DECISION_COST_MS_S_BY_BUCKET); the
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


for _f, _want in FROZEN_BLOBS.items():
    _got = git_blob(os.path.join(HERE, _f))
    if _got != _want:
        raise SystemExit(f"FAIL LOUD: frozen {_f} blob {_got}, want {_want}")
import uncut_arm_gate_read as G  # noqa: E402
from r1_attribution_v1_read import exact_wilcoxon  # noqa: E402

# Set when the calibration is filed and the arm pin is cut; None = not frozen, the reader refuses to run.
ARM_PIN = None          # the w-only pin (full sha) the ils / exact cells run at
W_MD5 = None            # md5 of scripts_cosim/decision_cost_w.json at ARM_PIN
CELLS_MD5 = None        # md5 of the cell list ils_live_cells.txt

FILED_PIN = "7cb483c5"
ARMS = ("bar", "gnn", "ils", "exact")
SEEDS = {"bar": (0,), "gnn": (1, 2), "ils": (0,), "exact": (0,)}
ARM_NAME = {"bar": "cd_ib0_inflightfull_k100", "gnn": "ra_gnn_eng_combfl_nosplit_k100", "ils": "cd_ib0_ils_k100", "exact": "cd_exactS_ib0_live_k100"}
ARM_KIND = {"ils": "cd_ib0_ils", "exact": "cd_exactS_ib0_live"}
TOPOS = list(range(19537, 19549))
RUNGS, WINDOWS, STATS = G.RUNGS, G.WINDOWS, G.STATS
CONTRASTS = (("F-ils-bar", "ils", "bar"), ("F-ils-gnn", "gnn", "ils"), ("F-exact-bar", "exact", "bar"))
BAD_ENV = ("HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_ORACLE", "HEROSIM_KPA_KNATIVE_PANIC", "HEROSIM_EXCHANGE_TCP", "HEROSIM_EXCHANGE_FAIRSHARE")
CKPT_FILE = "uncut-comb-fl-gnn-eng-seed{}.pt"
BUCKETS = ("le10", "11_50", "gt50")
G_CAP = 8
ILS_PER = "200"


def roots(U):
    A = f"{U}/anomaly/dose_7cb483c5"
    out = {}
    for r, short in (("moderate", "mod"), ("heavy", "heavy")):
        out[("bar", r)] = (f"{A}/cells_ref_{short}070", FILED_PIN)
        out[("gnn", r)] = (f"{A}/cells_gnn_{short}070", FILED_PIN)
        for arm in ("ils", "exact"):
            out[(arm, r)] = (f"{U}/ils_live/cells_{arm}", None)       # pin is ARM_PIN, resolved at read time
    return out


def stem_of(U, arm, rung, t, w, sd):
    d = roots(U)[(arm, rung)][0]
    return f"{d}/cc40s{t}__{w}{rung}__{ARM_NAME[arm]}_s{sd}"


def grid(U):
    return {(arm, sd, t, w, r): stem_of(U, arm, r, t, w, sd)
            for arm in ARMS for r in RUNGS for t in TOPOS for w in WINDOWS for sd in SEEDS[arm]}


def check_list(U, cells_file):
    """The new arms' cells of the grid equal the cell list's entries (the bar identity entry is never opened)."""
    if CELLS_MD5 is None or ARM_PIN is None or W_MD5 is None:
        raise SystemExit("FAIL LOUD: the reader is not frozen (CELLS_MD5, ARM_PIN and W_MD5 are set when the calibration is filed)")
    got = hashlib.md5(open(cells_file, "rb").read()).hexdigest()
    if got != CELLS_MD5:
        raise SystemExit(f"FAIL LOUD: cell list md5 {got}, want {CELLS_MD5}")
    listed, ident = set(), 0
    for line in open(cells_file).read().splitlines():
        root, name, flag = line.split("\t")[:3]
        if root.startswith("ils_live/ident"):
            ident += 1
            continue
        listed.add((f"{U}/{root}", name))
    want = {(os.path.dirname(stem), os.path.basename(stem)) for (arm, *_), stem in grid(U).items() if arm in ARM_KIND}
    if listed != want:
        raise SystemExit(f"FAIL LOUD: cell list and grid differ: {len(listed - want)} listed only {sorted(listed - want)[:3]}, "
                         f"{len(want - listed)} grid only {sorted(want - listed)[:3]}")
    return len(listed), ident


def expected_charge(w_file, kind, rung):
    """(env var -> value) the cell of `kind` at `rung` must carry at k = 1, from the frozen w file (the gate's dc_cost_ms / dc_s_cost_ms)."""
    w = json.load(open(w_file))
    out = {}
    byb = w.get("w_ms_per_plan_by_bucket", {}).get(kind, {})
    if rung in byb:
        out["HEROSIM_DECISION_COST_MS_BY_BUCKET"] = {b: float(v) for b, v in byb[rung].items()}
    else:
        out["HEROSIM_DECISION_COST_MS"] = float(w["w_ms_per_plan"][kind][rung])
    if kind == "cd_ib0_ils":
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


def one(U, key, stem, ckpt, w_file):
    arm, sd, t, w, r = key
    if os.path.exists(stem + ".failed.json"):
        raise SystemExit(f"FAIL LOUD: failed cell {stem}")
    if not os.path.exists(stem + ".summary.json"):
        raise SystemExit(f"FAIL LOUD: missing cell {stem}")
    s = json.load(open(stem + ".summary.json"))
    s["_path"] = stem + ".summary.json"
    pin = roots(U)[(arm, r)][1] or ARM_PIN
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
    if arm in ARM_KIND:
        e, cnt = s["env"], s["schedulerCounters"]
        kind = ARM_KIND[arm]
        need = {"HEROSIM_PG_BATCH_ALPHA": "0", "HEROSIM_PG_INFLIGHT_FULL": "1"}
        need.update({"HEROSIM_PG_ILS_PER": ILS_PER, "HEROSIM_PG_ILS_SEED": "0"} if arm == "ils" else
                    {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_EXACT_G": str(G_CAP)})
        for k, v in need.items():
            if e.get(k) != v:
                raise SystemExit(f"FAIL LOUD: {stem}: {k} served {e.get(k)!r}, want {v!r}")
        want = expected_charge(w_file, kind, r)
        for var in ("HEROSIM_DECISION_COST_MS", "HEROSIM_DECISION_COST_MS_BY_BUCKET", "HEROSIM_DECISION_COST_MS_S_BY_BUCKET"):
            if not same_charge(e.get(var), want.get(var)):
                raise SystemExit(f"FAIL LOUD: {stem}: {var} served {e.get(var)!r}, want {want.get(var)!r}")
        if arm == "ils" and not (cnt.get("pg_ils_batches", 0) > 0 and cnt.get("sched_plan_evals_s", 0) > 0):
            raise SystemExit(f"FAIL LOUD: ILS instrument off {stem}")
        if arm == "exact" and not cnt.get("pg_exact_batches", 0) > 0:
            raise SystemExit(f"FAIL LOUD: exact instrument off {stem}")
        if arm == "exact" and cnt.get("pg_ils_batches"):
            raise SystemExit(f"FAIL LOUD: exact-live cell ran the ILS {stem}")
        if arm == "ils" and (cnt.get("pg_exact_batches") or cnt.get("pg_expand_batches")):
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
           "whole_nocharge_felt": (lat - float(cnt["sched_felt_task_seconds"]) / n) if "sched_felt_task_seconds" in cnt else None,
           "tail": float(s["endTime"]) - float(s["arrival_end"]["last_arrival_s"]) > G.TAIL_S}
    for k in ("pg_ils_batches", "pg_ils_small_batches", "pg_ils_improved", "pg_ils_evals", "pg_ils_stop_evals", "pg_ils_stop_converged", "pg_ils_stop_other",
              "pg_exact_g_skipped", "pg_exact_batches", "pg_exact_fallbacks"):
        row[k] = int(cnt.get(k, 0))
    return s, row


def load_all(U, w_file):
    ckpt = {"seen": {}, "md5": json.load(open(f"{U}/comb/md5_comball.json"))}
    cells, rows = {}, {}
    g = grid(U)
    for key, stem in g.items():
        s, row = one(U, key, stem, ckpt, w_file)
        cells[key], rows[key] = s, row
    for (arm, r), (d, _p) in roots(U).items():
        if arm in ARM_KIND:
            extra = {f for f in os.listdir(d) if f.endswith(".summary.json")} - {os.path.basename(v) + ".summary.json" for k, v in g.items() if k[0] == arm and k[4] == r}
            if extra:
                raise SystemExit(f"FAIL LOUD: {len(extra)} cells in {d} outside the grid: {sorted(extra)[:3]}")
    return cells, rows, len(g)


def topo_vals(rows, a, b, rung, key):
    out = {}
    for t in TOPOS:
        va = [sum(rows[(a, sd, t, w, rung)][key] for sd in SEEDS[a]) / len(SEEDS[a]) for w in WINDOWS]
        vb = [sum(rows[(b, sd, t, w, rung)][key] for sd in SEEDS[b]) / len(SEEDS[b]) for w in WINDOWS]
        out[t] = (100.0 * (sum(va) / len(va) - sum(vb) / len(vb)) / (sum(vb) / len(vb)))
    return out


def build(cells, rows):
    res = {}
    for name, a, b in CONTRASTS:
        es = [G.contrast(cells, a, SEEDS[a], b, SEEDS[b], r, TOPOS, s) for r in RUNGS for s in STATS]
        G.family(es)
        for e in es:
            wm = topo_vals(rows, a, b, e["rung"], "whole" if e["stat"] == "whole" else "after360")
            e["wm_median"], e["wm_faster"] = st.median(wm.values()), sum(1 for x in wm.values() if x < 0)
        res[name] = es
    nc = {}
    for name, a, b in CONTRASTS:
        for r in RUNGS:
            v = topo_vals(rows, a, b, r, "whole_nocharge")
            nc[(name, r)] = (st.median(v.values()), sum(1 for x in v.values() if x < 0), exact_wilcoxon(list(v.values())))
    return res, nc


def per_arm(rows, cells):
    out = {}
    for arm in ARMS:
        for r in RUNGS:
            def tm(k):
                vals = []
                for t in TOPOS:
                    cell = [rows[(arm, sd, t, w, r)][k] for w in WINDOWS for sd in SEEDS[arm]]
                    if any(c is None for c in cell):
                        return None
                    vals.append(sum(cell) / len(cell))
                return sum(vals) / len(vals)
            out[(arm, r)] = {k: tm(k) for k in ("whole", "after360", "busy_per_task", "share_pooled", "share_felt", "pg_ils_batches", "pg_ils_small_batches",
                                                  "pg_ils_improved", "pg_ils_evals", "pg_ils_stop_evals", "pg_ils_stop_converged", "pg_exact_g_skipped",
                                                  "pg_exact_batches", "pg_exact_fallbacks")}
            out[(arm, r)]["tail"] = sum(rows[(arm, sd, t, w, r)]["tail"] for t in TOPOS for w in WINDOWS for sd in SEEDS[arm])
            out[(arm, r)]["collapse"] = sum(
                any(G.means(cells[(arm, sd, t, w, r)])[k] > G.COLLAPSE_X * G.cell_mean(cells, "bar", (0,), t, w, r)[k] for k in STATS)
                for t in TOPOS for w in WINDOWS for sd in SEEDS[arm])
    return out


def fp(p):
    return 'n/a' if p is None else format(p, '.4g')


def report(res, nc, pa, nlist, ncells, cells):
    L = [__doc__.strip(), "", f"=== cells read and asserted: {ncells}; new-arm cell list entries {nlist[0]} (identity entries {nlist[1]} never opened)",
         f"=== frozen: ARM_PIN {ARM_PIN}, W_MD5 {W_MD5}, CELLS_MD5 {CELLS_MD5}"]
    for name, a, b in CONTRASTS:
        L.append(f"\n  -- {name}   ({a} vs {b}; primary (Table I), Holm within the family of 4 | window-mean)")
        for e in res[name]:
            L.append(f"      {e['rung']:8s} {e['stat']:8s} median {e['median_pct']:+8.3f} % ({e['faster']:2d}/{e['n_topologies']})  p {fp(e['p'])}  holm {fp(e['p_holm'])}"
                     f"   | window-mean {e['wm_median']:+8.3f} % ({e['wm_faster']}/{e['n_topologies']})")
    L.append("\n  -- plan quality with the charge removed (mean latency less the pooled decision seconds per task; median over topologies, topologies faster, exact Wilcoxon raw p; descriptive)")
    for (name, r), (m, f, p) in nc.items():
        L.append(f"      {name:12s} {r:8s} {m:+8.3f} % ({f}/12)  p {fp(p)}")
    L.append("\n  -- per arm and load, mean over 12 topologies of the per-topology window mean")
    for (arm, r), d in pa.items():
        L.append(f"    {arm:5s} {r:8s} latency whole {d['whole']:.3f} / after360 {d['after360']:.3f} s; charged decision s/task {d['busy_per_task']:.4f}; "
                 f"decision share pooled {d['share_pooled']:.3f} %" + (f", felt {d['share_felt']:.3f} %" if d["share_felt"] is not None else ", felt n/a (filed cell)") + f"; tail flags {d['tail']}, collapse cells (> {G.COLLAPSE_X:g} x the bar) {d['collapse']}")
        if arm == "ils":
            b = max(d["pg_ils_batches"], 1)
            L.append(f"            ILS batches {d['pg_ils_batches']:.0f} (batches of <= 10 left to the bar's plan {d['pg_ils_small_batches']:.0f}); plan changed in "
                     f"{100 * d['pg_ils_improved'] / b:.1f} % of ILS batches; plans per ILS batch {d['pg_ils_evals'] / b:.0f}; stopped on the budget {d['pg_ils_stop_evals']:.0f}, converged {d['pg_ils_stop_converged']:.0f}")
        if arm == "exact":
            L.append(f"            batches on the bar's plan (> {G_CAP} tasks) {d['pg_exact_g_skipped']:.0f}; exhaustive {d['pg_exact_batches']:.0f}; expanded (pg_exact_fallbacks) {d['pg_exact_fallbacks']:.0f}")
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--out")
    a = ap.parse_args(argv[1:])
    U = os.environ.get("ILS_LIVE_U", "/share/nikola.lukic/uncut_groups")
    w_file = os.path.join(HERE, "decision_cost_w.json")
    out = a.out or f"{U}/ils_live/ils_live_read.txt"
    nlist = check_list(U, a.cells)
    if hashlib.md5(open(w_file, "rb").read()).hexdigest() != W_MD5:
        raise SystemExit("FAIL LOUD: decision_cost_w.json is not the frozen one (W_MD5)")
    cells, rows, ncells = load_all(U, w_file)
    res, nc = build(cells, rows)
    txt = report(res, nc, per_arm(rows, cells), nlist, ncells, cells)
    print(txt)
    if os.path.exists(out):
        raise SystemExit("FAIL LOUD: output exists, the read is written once")
    open(out + ".partial", "w").write(txt + "\n")
    os.replace(out + ".partial", out)


if __name__ == "__main__":
    main(sys.argv)

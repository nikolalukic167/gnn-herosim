"""plan_rerank_v1 Phase 0 reader (live oracle screen). Written before the data lands; the bars are the node's.

Arms on the 6 fresh topologies 19300-19305, windows g0 and g1, both rungs, all seed 0:
  O  cd_oracle   cd_exactS with the exact search scored by the real single-batch RTT of a fidelity replay (batches of <= 625 slate plans; larger
                 batches run cd_exactS unchanged and are counted)
  E  cd_exactS   the hand bar            C  cd
Counts first (summaries and failures per arm and rung against the expected 12 cells; a failed.json is a count, never a rerun). Then the oracle's
books per rung: batches seen / served / over the cap (and the task share of the over-cap batches), plans per served batch, how often it changed
cd_exactS's plan, the single-batch RTT it saved per served batch, decision wall seconds per batch. Then, per rung, the paired % of mean latency per
topology (median over its windows), its median over topologies, the sign count (wins = topologies where the first arm is faster) and the exact
two-sided Wilcoxon (n = 6 gives p >= 0.031; p is reported and is not part of the bar). Negative = the first arm faster.

Bar (node, signed before data): GO = O vs E median <= -4 % AND O faster on >= 5 of 6 topologies, at BOTH rungs. Otherwise the lineage closes
SUPERVISED-CEILING-AT-BAR. The flags are printed, the node holds the rule.

Descriptive readouts for O vs E (per rung): the latency buckets as pp of E's mean latency (median over cells of the paired difference: elapsed,
cold start, peer exchange, lock wait, rest), the stacking of the oracle's plan against cd_exactS's plan on the very same states (--gate cells'
oracle_diag/<cell>/*.jsonl: mean tasks per occupied platform, share of tasks sharing a platform with a batch-mate), and with --raw-dir the share
of the paired gap that falls in the first quarter of the run (tasks in id order).

  plan_oracle_read.py --gate <dir> [<dir> ...] --topos t1,...,t6 [--raw-dir <dir>] [--out read.json]
"""
import argparse
import glob
import json
import os
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

O, E, C = "cd_oracle", "cd_exactS", "cd"
ARMS = (O, E, C)
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1")
CONTRASTS = ((O, E), (O, C), (E, C))
GO_MEDIAN_PCT = -4.0
GO_MIN_WINS = 5
BOOKS = ("pg_batches", "pg_exact_batches", "pg_exact_fallbacks", "pg_oracle_batches", "pg_oracle_tasks", "pg_oracle_plans", "pg_oracle_changed",
         "pg_oracle_tasks_moved", "pg_oracle_gain_seconds", "pg_oracle_seconds", "pg_oracle_seen_batches", "pg_oracle_seen_tasks",
         "pg_oracle_over_cap_batches", "pg_oracle_over_cap_tasks")


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    bad_w = sorted({k[3] for k in list(cells) + list(failed)} - set(WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")
    bad_k = sorted({k[0] for k in list(cells) + list(failed)} - set(ARMS))
    if bad_k:
        raise SystemExit(f"FAIL LOUD: arms outside {ARMS}: {bad_k}")


def counts(cells, failed, topos):
    exp = len(topos) * len(WINDOWS)
    out = {}
    for kind in ARMS:
        for rung in RUNGS:
            fin = sum(1 for k in cells if k[0] == kind and k[4] == rung)
            bad = sum(1 for k in failed if k[0] == kind and k[4] == rung)
            out[f"{kind}|{rung}"] = {"finished": fin, "failed": bad, "expected": exp, "complete": fin == exp}
    return out


def _tot(cs, name):
    return sum(float(c.get(name) or 0) for c in cs)


def books(cells, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == O and k[4] == rung]
    t = {n: _tot(cs, n) for n in BOOKS}
    served, seen = t["pg_oracle_batches"], t["pg_oracle_seen_batches"]
    return {"cells": len(cs), **t,
            "books_close": seen == served + t["pg_oracle_over_cap_batches"],
            "over_cap_share_of_batches": t["pg_oracle_over_cap_batches"] / seen if seen else None,
            "over_cap_share_of_tasks": t["pg_oracle_over_cap_tasks"] / t["pg_oracle_seen_tasks"] if t["pg_oracle_seen_tasks"] else None,
            "plans_per_served_batch": t["pg_oracle_plans"] / served if served else None,
            "changed_share": t["pg_oracle_changed"] / served if served else None,
            "saved_single_batch_rtt_s_per_served_batch": t["pg_oracle_gain_seconds"] / served if served else None,
            "decision_wall_s_per_served_batch": t["pg_oracle_seconds"] / served if served else None}


def _per_task(s):
    n = float(s["num_tasks"])
    elapsed = float(s["averageElapsedTime"])
    cold = float(s.get("averageColdStartTime") or 0.0)
    exch = float(s.get("totalPeerExchangeTime") or 0.0) / n
    lock = float((s.get("lock_wait") or {}).get("mean") or 0.0)
    return {"elapsed": elapsed, "cold_start": cold, "exchange": exch, "lock_wait": lock, "rest": elapsed - cold - exch - lock}


def buckets(cells, rung, topos):
    """Paired O - E per cell, in pp of E's mean latency, median over the cells of the rung."""
    rows = {k: [] for k in ("elapsed", "cold_start", "exchange", "lock_wait", "rest")}
    for t in topos:
        for w in WINDOWS:
            o, e = cells.get((O, 0, t, w, rung)), cells.get((E, 0, t, w, rung))
            if o is None or e is None:
                continue
            po, pe = _per_task(o), _per_task(e)
            for k in rows:
                rows[k].append(100.0 * (po[k] - pe[k]) / pe["elapsed"])
    return {k: (st.median(v) if v else None) for k, v in rows.items()} | {"n_cells": len(rows["elapsed"])}


def _stack(plan):
    occ = {}
    for nid_pid in plan:
        occ[tuple(nid_pid)] = occ.get(tuple(nid_pid), 0) + 1
    n = len(plan)
    shared = sum(v for v in occ.values() if v > 1)
    return n / len(occ), shared / n


def stacking(diag_rows):
    """Per-batch stacking of the oracle's plan against cd_exactS's plan on the same state, task-weighted."""
    acc = {"best": [0.0, 0.0, 0], "s_plan": [0.0, 0.0, 0]}
    for r in diag_rows:
        n = int(r["n"])
        for key in acc:
            per, shared = _stack(r[key])
            acc[key][0] += per * n
            acc[key][1] += shared * n
            acc[key][2] += n
    out = {"batches": len(diag_rows)}
    for key, (per, shared, n) in acc.items():
        out[key] = {"tasks_per_occupied_platform": per / n if n else None, "share_tasks_sharing_a_platform": shared / n if n else None}
    return out


def diag_rows(gate_dirs, rung, topos):
    rows = []
    for g in gate_dirs:
        for d in glob.glob(os.path.join(g, "oracle_diag", f"cc40s*__g?{rung}__{O}_s0")):
            if int(os.path.basename(d).split("__")[0][len("cc40s"):]) not in topos:
                continue
            for f in glob.glob(os.path.join(d, "*.jsonl")):
                with open(f) as fh:
                    rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def quarter_gap(raw_a, raw_b):
    """Share of the paired gap sum(elapsed_a - elapsed_b) that falls in the first quarter of the tasks (by task id), from two raw results."""
    def elapsed(path):
        with open(path) as fh:
            doc = json.load(fh)
        trs = (doc.get("stats") or doc)["taskResults"]
        return {int(t["taskId"]): float(t["doneTime"]) - float(t["scheduledTime"]) for t in trs if t.get("taskId", -1) >= 0}
    a, b = elapsed(raw_a), elapsed(raw_b)
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    d = [a[i] - b[i] for i in ids]
    q1 = d[: max(1, len(d) // 4)]
    total = sum(d)
    return {"tasks": len(ids), "gap_s": total, "first_quarter_gap_s": sum(q1), "first_quarter_share": sum(q1) / total if total else None}


def read(gates, topos, raw_dir=None):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos)
    sel = sorted(topos)
    contrasts = {f"{a} vs {b}|{r}": R.contrast(cells, a, b, r, sel) for a, b in CONTRASTS for r in RUNGS}
    prim = [contrasts[f"{O} vs {E}|{r}"] for r in RUNGS]
    go = {r: (c["median_pct"] is not None and c["median_pct"] <= GO_MEDIAN_PCT and (c["wins"] or 0) >= GO_MIN_WINS) for r, c in zip(RUNGS, prim)}
    out = {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed),
           "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
           "counts": counts(cells, failed, sel), "oracle_books": {r: books(cells, r) for r in RUNGS},
           "contrasts": contrasts, "buckets_O_minus_E_pp": {r: buckets(cells, r, sel) for r in RUNGS},
           "stacking": {r: stacking(diag_rows(gates, r, set(sel))) for r in RUNGS},
           "GO_per_rung": go, "GO": all(go.values())}
    if raw_dir:
        out["first_quarter"] = {}
        for r in RUNGS:
            gaps = []
            for t in sel:
                for w in WINDOWS:
                    pa = os.path.join(raw_dir, f"cc40s{t}__{w}{r}__{O}_s0.raw.json")
                    pb = os.path.join(raw_dir, f"cc40s{t}__{w}{r}__{E}_s0.raw.json")
                    if os.path.exists(pa) and os.path.exists(pb):
                        g = quarter_gap(pa, pb)
                        if g:
                            gaps.append(g)
            tot = sum(g["gap_s"] for g in gaps)
            out["first_quarter"][r] = {"cells": len(gaps), "first_quarter_share_of_gap": (sum(g["first_quarter_gap_s"] for g in gaps) / tot) if tot else None}
    return out


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:24s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- oracle books")
    for rung, b in r["oracle_books"].items():
        keep = {n: (round(v, 4) if isinstance(v, float) else v) for n, v in b.items()
                if n in ("cells", "pg_oracle_seen_batches", "pg_oracle_batches", "books_close", "over_cap_share_of_batches", "over_cap_share_of_tasks",
                         "plans_per_served_batch", "changed_share", "saved_single_batch_rtt_s_per_served_batch", "decision_wall_s_per_served_batch")}
        print(f"  {rung:9s} {keep}")
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {k:30s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print("-- O - E buckets, pp of E's latency (median over cells)")
    for rung, b in r["buckets_O_minus_E_pp"].items():
        print(f"  {rung:9s} " + " ".join(f"{k}={f(v, 2)}" for k, v in b.items()))
    print("-- stacking on the same states (oracle plan vs cd_exactS plan)")
    for rung, s in r["stacking"].items():
        print(f"  {rung:9s} {s}")
    if "first_quarter" in r:
        print(f"-- first-quarter share of the O-E gap: {r['first_quarter']}")
    print(f"-- GO per rung {r['GO_per_rung']}   GO {r['GO']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True, help="comma-separated topology ids (the 6 fresh ones)")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    if len(topos) != 6:
        raise SystemExit(f"FAIL LOUD: expected 6 topologies, got {len(topos)}")
    r = read(a.gate, topos, a.raw_dir)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

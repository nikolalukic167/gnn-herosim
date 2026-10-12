"""ib0_confirm_v1 reader (classical + GNN-seed stage). Written before the data lands; the bars are the node's (docs/lineages/ib0_confirm_v1.md).

12 fresh topologies, both rungs of the accel seat. Arms (cell kinds):
  IB0  cd_exactS_ib0        exact search over S with batch alpha 0           E  cd_exactS (the reference)         C  cd
  CDXH cdx_ib0              hand-seed control: the cd_expand flow on the ib0 surrogate (greedy first pass, passes, one expansion, passes)
  CDXI ra_gnn_eng_cdxib0    the GNN seed (seed 1) refined by GnnCdRefiner on the ib0 surrogate (same passes, same expansion limits)
  descriptive (g0-g1): A05 cd_exactS_a05 (batch alpha 0.5), A0 cd_exactS_a0 (whole queue term to 0), TWIN ra_twin_eng_cdxib0 (the MP-OFF seed twin)
Windows g0-g3 for IB0 / E / C / CDXH / CDXI, g0-g1 for the descriptive arms. The unit is the topology: per topology the median over its windows of the paired
% of mean latency; median over the 12 topologies; sign count; exact Wilcoxon (n = 12). Negative = the first arm faster.

Primary family (Holm over the 2 rungs): IB0 vs E. WIN: median <= -2 % and Holm p < .05 and IB0 faster on >= 9 of 12 topologies, at BOTH rungs, and no rung where E is
Holm-faster. A rung where E is Holm-faster is a LOSS for IB0; otherwise TIES.
Secondary family (one Holm over 4 tests: 2 contrasts x 2 rungs): CDXI vs CDXH (credits the GNN seed) and CDXI vs IB0. GNN-SEED WIN: CDXI vs CDXH median <= -2 % and
Holm p < .05 at both rungs. No MP claim is available here (the twin is descriptive).
Descriptive (no bar): A05 vs IB0, A0 vs IB0, TWIN vs CDXI, IB0 vs C, E vs C, CDXH vs IB0; the g0+g1 sensitivity of the primary contrast; latency buckets (pp of the
reference's mean latency); the search books of the two expansion arms (passes, expansion batches / moves / sweeps) so the equal budget is on the page.
Counts first: summaries and failures against the expected cells per arm and rung; a failed.json is a count, never a rerun.

  ib0_confirm_read.py --gate <dir> [<dir> ...] --topos t1,...,t12 [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

IB0, E, C = "cd_exactS_ib0", "cd_exactS", "cd"
CDXH, CDXI = "cdx_ib0", "ra_gnn_eng_cdxib0"
A05, A0, TWIN = "cd_exactS_a05", "cd_exactS_a0", "ra_twin_eng_cdxib0"
ARMS = (IB0, E, C, CDXH, CDXI, A05, A0, TWIN)
RUNGS = ("moderate", "heavy")
WINDOWS4, WINDOWS2 = ("g0", "g1", "g2", "g3"), ("g0", "g1")
EXPECTED_WINDOWS = {IB0: WINDOWS4, E: WINDOWS4, C: WINDOWS4, CDXH: WINDOWS4, CDXI: WINDOWS4, A05: WINDOWS2, A0: WINDOWS2, TWIN: WINDOWS2}
WIN_MEDIAN_PCT, WIN_MIN_WINS, HOLM_ALPHA = -2.0, 9, 0.05
PRIMARY = ((IB0, E),)
SECONDARY = ((CDXI, CDXH), (CDXI, IB0))
DESCRIPTIVE = ((A05, IB0), (A0, IB0), (TWIN, CDXI), (IB0, C), (E, C), (CDXH, IB0), (A0, E), (A05, E))
SEARCH_BOOKS = ("pg_cd_passes", "pg_cd_moves", "pg_expand_batches", "pg_expand_moves", "pg_expand_sweeps", "pg_expand_evals", "pg_expand_labels_skipped")


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    bad_k = sorted({k[0] for k in list(cells) + list(failed)} - set(ARMS))
    if bad_k:
        raise SystemExit(f"FAIL LOUD: arms outside {ARMS}: {bad_k}")
    for k in list(cells) + list(failed):
        if k[3] not in EXPECTED_WINDOWS[k[0]]:
            raise SystemExit(f"FAIL LOUD: {k} is a window outside {EXPECTED_WINDOWS[k[0]]} for {k[0]}")


def counts(cells, failed, topos):
    out = {}
    for kind in ARMS:
        exp = len(topos) * len(EXPECTED_WINDOWS[kind])
        for rung in RUNGS:
            fin = sum(1 for k in cells if k[0] == kind and k[4] == rung)
            bad = sum(1 for k in failed if k[0] == kind and k[4] == rung)
            out[f"{kind}|{rung}"] = {"finished": fin, "failed": bad, "expected": exp, "complete": fin == exp}
    return out


def _per_task(s):
    n = float(s["num_tasks"])
    elapsed = float(s["averageElapsedTime"])
    cold = float(s.get("averageColdStartTime") or 0.0)
    exch = float(s.get("totalPeerExchangeTime") or 0.0) / n
    lock = float((s.get("lock_wait") or {}).get("mean") or 0.0)
    return {"elapsed": elapsed, "cold_start": cold, "exchange": exch, "lock_wait": lock, "rest": elapsed - cold - exch - lock}


def buckets(cells, a, b, rung, topos):
    """Paired a - b per cell, in pp of b's mean latency, median over the cells of the rung (cells where both finished)."""
    rows = {k: [] for k in ("elapsed", "cold_start", "exchange", "lock_wait", "rest")}
    for (kind, seed, t, w, r), s in cells.items():
        if kind != a or r != rung or t not in topos:
            continue
        ref = cells.get((b, seed, t, w, r)) or cells.get((b, 0, t, w, r))
        if ref is None:
            continue
        pa, pb = _per_task(s), _per_task(ref)
        for k in rows:
            rows[k].append(100.0 * (pa[k] - pb[k]) / pb["elapsed"])
    return {k: (st.median(v) if v else None) for k, v in rows.items()} | {"n_cells": len(rows["elapsed"])}


def search_books(cells, kind, rung):
    cs = [s.get("schedulerCounters") or {} for k, s in cells.items() if k[0] == kind and k[4] == rung]
    tot = {n: sum(float(c.get(n) or 0) for c in cs) for n in SEARCH_BOOKS}
    batches = sum(float(c.get("cdr_batches") or c.get("pg_batches") or 0) for c in cs)
    return {"cells": len(cs), "batches": batches, **tot}


def verdicts(contrasts, holm_primary):
    out = {}
    wins = {}
    for rung in RUNGS:
        c = contrasts[f"{IB0} vs {E}|{rung}"]
        hp = holm_primary[f"{rung}"]
        ok = c["median_pct"] is not None and c["median_pct"] <= WIN_MEDIAN_PCT and hp is not None and hp < HOLM_ALPHA and (c["wins"] or 0) >= WIN_MIN_WINS
        loses = c["median_pct"] is not None and c["median_pct"] > 0 and hp is not None and hp < HOLM_ALPHA
        wins[rung] = {"win": ok, "e_holm_faster": loses}
    out["primary_per_rung"] = wins
    out["PRIMARY"] = "WIN" if all(w["win"] for w in wins.values()) else ("LOSES" if any(w["e_holm_faster"] for w in wins.values()) else "TIES")
    return out


def read(gates, topos):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS and k[0] in ARMS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS and k[0] in ARMS}
    check(cells, failed, topos)
    sel = sorted(topos)
    contrasts = {f"{a} vs {b}|{r}": R.contrast(cells, a, b, r, sel) for a, b in PRIMARY + SECONDARY + DESCRIPTIVE for r in RUNGS}
    holm_primary = R.holm({r: contrasts[f"{IB0} vs {E}|{r}"]["p"] for r in RUNGS})
    sec_ps = {f"{a} vs {b}|{r}": contrasts[f"{a} vs {b}|{r}"]["p"] for a, b in SECONDARY for r in RUNGS}
    holm_secondary = R.holm(sec_ps)
    g01 = {k: v for k, v in cells.items() if k[3] in WINDOWS2}
    sens = {f"{IB0} vs {E}|{r}|g0+g1": R.contrast(g01, IB0, E, r, sel) for r in RUNGS}
    verdict = verdicts(contrasts, holm_primary)
    seed_win = {}
    for r in RUNGS:
        c = contrasts[f"{CDXI} vs {CDXH}|{r}"]
        hp = holm_secondary[f"{CDXI} vs {CDXH}|{r}"]
        seed_win[r] = c["median_pct"] is not None and c["median_pct"] <= WIN_MEDIAN_PCT and hp is not None and hp < HOLM_ALPHA
    verdict["GNN_SEED_WIN_per_rung"] = seed_win
    verdict["GNN_SEED_WIN"] = all(seed_win.values())
    return {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed),
            "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
            "counts": counts(cells, failed, sel), "contrasts": contrasts, "sensitivity": sens,
            "holm_primary": holm_primary, "holm_secondary": holm_secondary,
            "buckets_pp": {f"{a} vs {b}|{r}": buckets(cells, a, b, r, set(sel)) for a, b in ((IB0, E), (CDXI, CDXH)) for r in RUNGS},
            "search_books": {f"{k}|{r}": search_books(cells, k, r) for k in (CDXH, CDXI) for r in RUNGS},
            "verdict": verdict}


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:30s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in {**r["contrasts"], **r["sensitivity"]}.items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {k:46s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- Holm primary {r['holm_primary']}   Holm secondary {r['holm_secondary']}")
    for k, b in r["buckets_pp"].items():
        print(f"  buckets {k:34s} " + " ".join(f"{n}={f(v, 2)}" for n, v in b.items()))
    for k, b in r["search_books"].items():
        print(f"  search books {k:30s} {b}")
    print(f"-- verdict {r['verdict']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True, help="comma-separated topology ids (the 12 fresh ones)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    if len(topos) != 12:
        raise SystemExit(f"FAIL LOUD: expected 12 topologies, got {len(topos)}")
    r = read(a.gate, topos)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

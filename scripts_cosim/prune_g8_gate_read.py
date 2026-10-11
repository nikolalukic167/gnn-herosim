"""prune_g8_s0_v1 live gate reader (rule 6, G = 8). Written before the data lands.

Arms on the 12 fresh topologies, 4 windows, both rungs (seed 1 for the learned arms, seed 0 for the rules):
  A  ra_gnn_eng_cdxpra   cdxprune, GNN order (gnn_eng checkpoint)      B  ra_gnn_eng_cdxprb   cdxprune, cost order (same shell and seed plan)
  T  ra_twin_eng_cdxpra  cdxprune, order from the MP-OFF twin           E  cd_exactS           exact search, expansion fallback        C  cd
Counts first: summaries and failures per arm and rung against the expected 48 cells, and any cell the progress watchdog killed (a failed.json is a
count, never a rerun). Then the prune books per arm and rung (over-cap batches and their share of batches, pruned optimum vs the expansion plan:
differs / better / worse, S gain per over-cap batch against the expansion and against the pass plan, how often the GNN top-k set differs from
the cost top-k). Then, per rung, the paired % of mean latency per topology (median over the topology's windows), its median over the 12
topologies, the sign count (wins = topologies where the first arm is faster) and an exact two-sided Wilcoxon. Negative = the first arm faster.

Registered families (Holm within each family, exact Wilcoxon p; every other contrast is descriptive and labelled so):
  PRIMARY    A vs E at moderate and heavy                                  Holm over 2
  SECONDARY  A vs B (learned order vs cost order), A vs T (message passing vs MP-OFF), at both rungs     Holm over 4
Descriptive: A, B, T, E each vs C; B vs E; T vs E; T vs B.
Flags (printed, not interpreted here; the node holds the rules): PRIMARY_BOTH_RUNGS_FASTER = A's median < 0 against E at both rungs,
PRIMARY_HOLM_SIG = both Holm-adjusted p < 0.05 AND A faster at both rungs (a significant loss is not a signal). A topology outside --topos is refused, and so is a window set other than g0-g3.

  prune_g8_gate_read.py --gate <dir> [<dir> ...] --topos t1,...,t12 [--out read.json]
"""
import argparse
import json
import signal
import sys

import r1_attribution_v1_read as R

A, B, T = "ra_gnn_eng_cdxpra", "ra_gnn_eng_cdxprb", "ra_twin_eng_cdxpra"
E, C = "cd_exactS", "cd"
ARMS = (A, B, T, E, C)
PRUNE_ARMS = (A, B, T)
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
PRIMARY = ((A, E),)
SECONDARY = ((A, B), (A, T))
DESCRIPTIVE = ((A, C), (B, C), (T, C), (E, C), (B, E), (T, E), (T, B))
ALPHA = 0.05
PRUNE_BOOKS = ("pg_batches", "pg_exact_batches", "pg_exact_fallbacks", "pg_exact_prune_batches", "pg_exact_prune_plans", "pg_exact_prune_differs",
               "pg_exact_prune_better", "pg_exact_prune_worse", "pg_exact_prune_kept_pass", "pg_exact_prune_tasks_moved",
               "pg_exact_prune_gain_fb_seconds", "pg_exact_prune_gain_pass_seconds", "pg_exact_prune_set_differs",
               "pg_exact_prune_k3", "pg_exact_prune_k4", "pg_exact_prune_kother")


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    wins = {k[3] for k in list(cells) + list(failed)}
    bad_w = sorted(w for w in wins if not any(w.startswith(x) for x in WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")


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
    return sum(int(c.get(name) or 0) for c in cs)


def books(cells, kind, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == kind and k[4] == rung]
    t = {n: _tot(cs, n) for n in PRUNE_BOOKS}
    over = t["pg_exact_fallbacks"]
    out = {"cells": len(cs), **t, "over_cap_share_of_batches": over / t["pg_batches"] if t["pg_batches"] else None}
    pb = t["pg_exact_prune_batches"]
    if kind in PRUNE_ARMS:
        out.update({"prune_equals_fallbacks": pb == over,
                    "differs_share": t["pg_exact_prune_differs"] / pb if pb else None,
                    "better": t["pg_exact_prune_better"], "worse": t["pg_exact_prune_worse"],
                    "gain_vs_expansion_s_per_batch": t["pg_exact_prune_gain_fb_seconds"] / pb if pb else None,
                    "gain_vs_pass_s_per_batch": t["pg_exact_prune_gain_pass_seconds"] / pb if pb else None,
                    "topk_set_differs_share": t["pg_exact_prune_set_differs"] / pb if pb else None,
                    "plans_per_over_cap_batch": t["pg_exact_prune_plans"] / pb if pb else None})
    return out


def holm_family(pairs, contrasts):
    adj = R.holm({f"{a} vs {b}|{r}": contrasts[f"{a} vs {b}|{r}"]["p"] for a, b in pairs for r in RUNGS})
    for k, v in adj.items():
        contrasts[k]["p_holm"] = v
    return adj


def read(gates, topos):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos)
    sel = sorted(topos)
    contrasts = {}
    for a, b in PRIMARY + SECONDARY + DESCRIPTIVE:
        for r in RUNGS:
            contrasts[f"{a} vs {b}|{r}"] = R.contrast(cells, a, b, r, sel)
    for pairs in (PRIMARY, SECONDARY):
        holm_family(pairs, contrasts)
    fam = {f"{a} vs {b}|{r}": name for name, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)) for a, b in pairs for r in RUNGS}
    for k in contrasts:
        contrasts[k]["family"] = fam.get(k, "descriptive")
    prim = [contrasts[f"{A} vs {E}|{r}"] for r in RUNGS]
    out = {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed),
           "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
           "counts": counts(cells, failed, topos), "books": {f"{k}|{r}": books(cells, k, r) for k in ARMS for r in RUNGS},
           "contrasts": contrasts,
           "PRIMARY_BOTH_RUNGS_FASTER": all((c["median_pct"] is not None and c["median_pct"] < 0) for c in prim),
           "PRIMARY_HOLM_SIG": all((c.get("p_holm") is not None and c["p_holm"] < ALPHA and c["median_pct"] < 0) for c in prim)}
    return out


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:36s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- over-cap batches and the prune books")
    for k, b in r["books"].items():
        keep = {n: (round(v, 4) if isinstance(v, float) else v) for n, v in b.items()
                if n in ("cells", "pg_batches", "pg_exact_fallbacks", "over_cap_share_of_batches", "prune_equals_fallbacks", "differs_share", "better",
                         "worse", "gain_vs_expansion_s_per_batch", "gain_vs_pass_s_per_batch", "topk_set_differs_share", "plans_per_over_cap_batch")}
        print(f"  {k:36s} {keep}")
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        hp = f" p_holm={f(c.get('p_holm'))}" if "p_holm" in c else ""
        print(f"  [{c['family']:11s}] {k:44s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}{hp}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- PRIMARY_BOTH_RUNGS_FASTER {r['PRIMARY_BOTH_RUNGS_FASTER']}   PRIMARY_HOLM_SIG {r['PRIMARY_HOLM_SIG']}")


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

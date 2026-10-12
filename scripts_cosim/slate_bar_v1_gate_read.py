"""slate_bar_v1 live gate reader (rule 6). Written before the data lands; the node (docs/lineages/slate_bar_v1.md) holds the design.

Arms on the 24 selfsearch_confirm_v1 topologies, windows g0-g3, both rungs:
  S  cd_exactS_slate      cd_exactS restricted to the declared top-5 slate, decoded whole (seed 0)           NEW
  K  cd_exactS_top5cd     the same, the slate ranked by CD's own standalone cost (seed 0), descriptive         NEW
  G  ra_gnn_eng_selfsearch  the GNN own-score self-search (checkpoint seeds 1 and 2)                          reused
  E  cd_exactS (seed 0)   C  cd (seed 0)                                                                      reused
Counts first (summaries and failures per arm and rung against the expected cells, any watchdog kill counted, never rerun), then the instrument books
(the slate served on every batch, pg_declared_out_of_slate which must be 0 in S and K), then per rung the paired % of mean latency per topology (median over
windows and seeds), its median over the 24 topologies, the sign count (wins = topologies where the first arm is faster) and an exact two-sided Wilcoxon.
Negative = the first arm faster.

Registered families (Holm within each; every other contrast is descriptive and labelled so):
  PRIMARY    G vs S at moderate and heavy         Holm over 2
  SECONDARY  S vs E at moderate and heavy         Holm over 2   (the size of the handicap)
Descriptive: K vs S, K vs E, G vs K, G vs E, S vs C, G vs C.
Label per rung (the node label is the heavy one):
  SLATE-IS-DISCLOSED-LIMIT  G slower than S with Holm p < 0.05
  SLATE-HANDICAP            G vs S median <= 0 or p_holm >= 0.05, AND S slower than E with Holm p < 0.05
  SLATE-IRRELEVANT          G vs S median <= 0 or p_holm >= 0.05, and S does not lose to E with Holm p < 0.05 (the slate costs cd_exactS nothing; the parent's verdicts stand)

  slate_bar_v1_gate_read.py --gate <dir> [<dir> ...] --topos t1,...,t24 [--out read.json]
"""
import argparse
import json
import signal
import sys

import r1_attribution_v1_read as R

S, K, G = "cd_exactS_slate", "cd_exactS_top5cd", "ra_gnn_eng_selfsearch"
E, C = "cd_exactS", "cd"
SEEDS = {S: 1, K: 1, G: 2, E: 1, C: 1}   # seeds per cell of the arm (S, K, E, C run one seed, 0; G runs checkpoint seeds 1 and 2)
ARMS = (S, K, G, E, C)
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
PRIMARY = ((G, S),)
SECONDARY = ((S, E),)
DESCRIPTIVE = ((K, S), (K, E), (G, K), (G, E), (S, C), (G, C))
ALPHA = 0.05
NTOPO = 24
BOOKS = ("pg_batches", "pg_declared_batches", "pg_declared_groups", "pg_declared_sub_batched", "pg_declared_out_of_slate", "pg_exact_batches",
         "pg_exact_fallbacks", "pg_exact_kept_pass", "pg_exact_tasks_moved", "pg_exact_gain_seconds", "pg_expand_batches")


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    bad_w = sorted({k[3] for k in list(cells) + list(failed)} - set(WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")
    unknown = sorted({k[0] for k in list(cells) + list(failed)} - set(ARMS))
    if unknown:
        raise SystemExit(f"FAIL LOUD: arms outside the registered five: {unknown}")


def counts(cells, failed, topos):
    out = {}
    for kind in ARMS:
        exp = len(topos) * len(WINDOWS) * SEEDS[kind]
        for rung in RUNGS:
            fin = sum(1 for k in cells if k[0] == kind and k[4] == rung)
            bad = sum(1 for k in failed if k[0] == kind and k[4] == rung)
            out[f"{kind}|{rung}"] = {"finished": fin, "failed": bad, "expected": exp, "complete": fin == exp}
    return out


def books(cells, kind, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == kind and k[4] == rung]
    t = {n: sum(float(c.get(n) or 0) for c in cs) for n in BOOKS}
    out = {"cells": len(cs), **{n: (int(v) if float(v).is_integer() else v) for n, v in t.items()}}
    if kind in (S, K):
        out["slate_on_every_batch"] = t["pg_declared_batches"] > 0 and t["pg_declared_groups"] == t["pg_declared_batches"]
        out["out_of_slate_zero"] = t["pg_declared_out_of_slate"] == 0
    return out


def holm_family(pairs, contrasts):
    adj = R.holm({f"{a} vs {b}|{r}": contrasts[f"{a} vs {b}|{r}"]["p"] for a, b in pairs for r in RUNGS})
    for k, v in adj.items():
        contrasts[k]["p_holm"] = v


def label(contrasts, rung):
    g, h = contrasts[f"{G} vs {S}|{rung}"], contrasts[f"{S} vs {E}|{rung}"]
    if g["median_pct"] is None or h["median_pct"] is None:
        return "NOT-READABLE"
    sig = lambda c: c.get("p_holm") is not None and c["p_holm"] < ALPHA   # no p (every paired difference zero) is not significant
    if g["median_pct"] > 0 and sig(g):
        return "SLATE-IS-DISCLOSED-LIMIT"
    return "SLATE-HANDICAP" if (h["median_pct"] > 0 and sig(h)) else "SLATE-IRRELEVANT"


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
    bk = {f"{k}|{r}": books(cells, k, r) for k in ARMS for r in RUNGS}
    instrument_ok = all(bk[f"{k}|{r}"]["slate_on_every_batch"] and bk[f"{k}|{r}"]["out_of_slate_zero"] for k in (S, K) for r in RUNGS)
    return {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed),
            "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
            "counts": counts(cells, failed, topos), "books": bk, "contrasts": contrasts, "INSTRUMENT_OK": instrument_ok,
            "labels": {r: label(contrasts, r) for r in RUNGS}, "LABEL": label(contrasts, "heavy")}


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:36s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- instrument books")
    for k, b in r["books"].items():
        print(f"  {k:36s} {b}")
    print("-- paired % per topology (median over windows and seeds), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        hp = f" p_holm={f(c.get('p_holm'))}" if "p_holm" in c else ""
        print(f"  [{c['family']:11s}] {k:44s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}{hp}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- INSTRUMENT_OK {r['INSTRUMENT_OK']}   labels {r['labels']}   LABEL (heavy) {r['LABEL']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True, help="comma-separated topology ids (the 24 selfsearch_confirm_v1 ones)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    if len(topos) != NTOPO:
        raise SystemExit(f"FAIL LOUD: expected {NTOPO} topologies, got {len(topos)}")
    r = read(a.gate, topos)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

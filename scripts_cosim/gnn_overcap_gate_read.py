"""gnn_overcap_v1 live gate reader (rule 6, G = 8 primary; the current group size as the descriptive secondary seat). Written before the data lands.

Arms on 12 fresh topologies, both rungs (seed 1 for the learned arms, seed 0 for the rules):
  H  ra_gnn_eng_ovg   cd_exactS on batches within the exact cap, the GNN self-search plan (gnn_eng s1) served unchanged on over-cap batches (no expansion fallback)
  T  ra_twin_eng_ovg  the same with the MP-OFF twin's self-search plan on over-cap batches
  E  cd_exactS        exact search over S on the top-5 slate, expansion fallback above 100,000 plans          C  cd
Primary seat G = 8: 4 windows (g0-g3), 48 cells per arm and rung. The secondary seat (--windows g0) runs H, E, C on one window; its counts are the cells it names.
Counts first (summaries and failures against the expected cells; a failed.json is a count, never a rerun). Then the over-cap books per arm and rung: over-cap batches
served and their share of refined batches and of tasks (H, T), the exact fallbacks (E), and the instrument identities (served == predicted, fallbacks 0 for H/T).
Then, per rung, the paired % of mean latency per topology (median over the topology's windows), its median over the 12 topologies, the sign count (wins = topologies where
the first arm is faster) and an exact two-sided Wilcoxon. Negative = the first arm faster.

Registered families (Holm within each family):
  PRIMARY    H vs E at moderate and heavy                         Holm over 2
  SECONDARY  H vs T (message passing vs MP-OFF), H vs C at both rungs   Holm over 4
Descriptive: T vs E, T vs C, E vs C.
Flags: WIN = H's median <= -2 % against E at both rungs AND both Holm-adjusted p < 0.05; LOSS = at either rung H slower than E with Holm p < 0.05 (a significant loss is not a signal);
E_FASTER_RUNGS lists the rungs where H's median is above 0. A topology outside --topos is refused, and so is a window set outside g0-g3.

  gnn_overcap_gate_read.py --gate <dir> [<dir> ...] --topos t1,...,t12 [--windows g0,g1,g2,g3] [--out read.json]
"""
import argparse
import json
import signal
import sys

import r1_attribution_v1_read as R

H, T = "ra_gnn_eng_ovg", "ra_twin_eng_ovg"
E, C = "cd_exactS", "cd"
ARMS = (H, T, E, C)
OVERCAP_ARMS = (H, T)
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
PRIMARY = ((H, E),)
SECONDARY = ((H, T), (H, C))
DESCRIPTIVE = ((T, E), (T, C), (E, C))
ALPHA = 0.05
WIN_PCT = -2.0
BOOKS = ("pg_batches", "cdr_batches", "cdr_tasks", "pg_exact_batches", "pg_exact_fallbacks", "pg_overcap_pred", "pg_overcap_gnn_batches", "pg_overcap_gnn_tasks")


def check(cells, failed, topos, windows):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    wins = {k[3] for k in list(cells) + list(failed)}
    bad_w = sorted(w for w in wins if not any(w.startswith(x) for x in WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")
    off = sorted(w for w in wins if not any(w.startswith(x) for x in windows))
    if off:
        raise SystemExit(f"FAIL LOUD: windows {off} are outside the windows this read was told to expect {windows}")


def counts(cells, failed, topos, windows, arms):
    exp = len(topos) * len(windows)
    out = {}
    for kind in arms:
        for rung in RUNGS:
            fin = sum(1 for k in cells if k[0] == kind and k[4] == rung)
            bad = sum(1 for k in failed if k[0] == kind and k[4] == rung)
            out[f"{kind}|{rung}"] = {"finished": fin, "failed": bad, "expected": exp, "complete": fin == exp}
    return out


def _tot(cs, name):
    return sum(int(c.get(name) or 0) for c in cs)


def books(cells, kind, rung):
    ks = [(s["schedulerCounters"], int(s["num_tasks"])) for k, s in cells.items() if k[0] == kind and k[4] == rung]
    cs = [c for c, _n in ks]
    t = {n: _tot(cs, n) for n in BOOKS}
    out = {"cells": len(cs), **t}
    if kind in OVERCAP_ARMS:
        refined = t["cdr_batches"]
        out.update({"over_cap_share_of_batches": t["pg_overcap_gnn_batches"] / refined if refined else None,
                    "over_cap_share_of_tasks": t["pg_overcap_gnn_tasks"] / sum(n for _c, n in ks) if ks else None,
                    "served_equals_predicted": t["pg_overcap_gnn_batches"] == t["pg_overcap_pred"],
                    "no_fallback_reached": t["pg_exact_fallbacks"] == 0})
    elif kind == E:
        out.update({"over_cap_share_of_batches": t["pg_exact_fallbacks"] / t["pg_batches"] if t["pg_batches"] else None,
                    "predicted_equals_fallbacks": t["pg_overcap_pred"] == t["pg_exact_fallbacks"]})
    return out


def holm_family(pairs, contrasts):
    adj = R.holm({f"{a} vs {b}|{r}": contrasts[f"{a} vs {b}|{r}"]["p"] for a, b in pairs for r in RUNGS})
    for k, v in adj.items():
        contrasts[k]["p_holm"] = v
    return adj


def read(gates, topos, windows=WINDOWS):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos, windows)
    sel = sorted(topos)
    arms = [a for a in ARMS if any(k[0] == a for k in list(cells) + list(failed))]
    contrasts = {}
    for a, b in PRIMARY + SECONDARY + DESCRIPTIVE:
        if a in arms and b in arms:
            for r in RUNGS:
                contrasts[f"{a} vs {b}|{r}"] = R.contrast(cells, a, b, r, sel)
    for pairs in (PRIMARY, SECONDARY):
        if all(a in arms and b in arms for a, b in pairs):
            holm_family(pairs, contrasts)
    fam = {f"{a} vs {b}|{r}": name for name, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)) for a, b in pairs for r in RUNGS}
    for k in contrasts:
        contrasts[k]["family"] = fam.get(k, "descriptive")
    prim = [contrasts.get(f"{H} vs {E}|{r}") for r in RUNGS]
    have = all(c is not None and c.get("p_holm") is not None and c["median_pct"] is not None for c in prim)
    out = {"gate": list(gates), "topologies": sel, "windows": list(windows), "n_summaries": len(cells), "n_failed": len(failed),
           "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
           "counts": counts(cells, failed, topos, windows, arms), "books": {f"{k}|{r}": books(cells, k, r) for k in arms for r in RUNGS},
           "contrasts": contrasts,
           "WIN": bool(have and all(c["median_pct"] <= WIN_PCT and c["p_holm"] < ALPHA for c in prim)),
           "LOSS": bool(have and any(c["median_pct"] > 0 and c["p_holm"] < ALPHA for c in prim)),
           "E_FASTER_RUNGS": [r for r, c in zip(RUNGS, prim) if c is not None and c["median_pct"] is not None and c["median_pct"] > 0]}
    return out


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies, windows {r['windows']}; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:36s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- over-cap books")
    for k, b in r["books"].items():
        keep = {n: (round(v, 4) if isinstance(v, float) else v) for n, v in b.items()
                if n in ("cells", "over_cap_share_of_batches", "over_cap_share_of_tasks", "served_equals_predicted", "no_fallback_reached", "predicted_equals_fallbacks",
                         "pg_overcap_gnn_batches", "pg_exact_fallbacks")}
        print(f"  {k:36s} {keep}")
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        hp = f" p_holm={f(c.get('p_holm'))}" if "p_holm" in c else ""
        print(f"  [{c['family']:11s}] {k:44s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}{hp}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- WIN {r['WIN']}   LOSS {r['LOSS']}   E_FASTER_RUNGS {r['E_FASTER_RUNGS']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True, help="comma-separated topology ids (the 12 fresh ones)")
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    if len(topos) != 12:
        raise SystemExit(f"FAIL LOUD: expected 12 topologies, got {len(topos)}")
    windows = tuple(w for w in a.windows.split(",") if w)
    r = read(a.gate, topos, windows)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

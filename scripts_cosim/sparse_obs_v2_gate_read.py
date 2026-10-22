"""sparse_obs_s0_v1 stage-2 reader (warmth; written before the data lands; docs/lineages/archive/sparse_obs_s0_v1.md, "Stage 2 design").

Arms (g0, 12 fresh topologies 19440-19451, seed 0): A0 = cd_exactS_ib0 (stage 1, same pinned head or identity-checked), A0+ = cd_obs_a0plus (cold priced against the type the task follows),
W1-NAIVE = cd_obs_w1_naive and W1-EC = cd_obs_w1_ec (A0+'s warm bit, degraded cold seconds). Per rung the paired % of mean latency per topology, median over the 12, exact Wilcoxon.
Negative = the first arm faster.

  PRIMARY      A0+ vs A0          the hand-bar correction (Holm over the two rungs)
  VoI_W1       W1-NAIVE vs A0+    R_W1  W1-EC vs A0+       (Holm within each family over the two rungs)
  descriptive  W1-EC vs W1-NAIVE, W1-NAIVE vs A0, W1-EC vs A0
Flags: VOI_GE_10, R_GE_5, W1_PASS (both, some rung); the kill rule is HAND-RECOVERS-UNDER-SPARSITY for warmth when neither rung passes. LEAK: any W1 arm faster than A0+ by >= 1 % with Holm < 0.05
(the degraded arm beating its own reference) is audited before VoI is read. A0P_GE_1: A0+ faster than A0 by >= 1 % (the hand-bar correction is material).

Steady state (descriptive, labelled, NOT in any family or flag): beside every primary/VoI/R contrast the same paired statistic on the mean latency of the tasks dispatched at or after 360 s
(`steady_state.after_360s`) and after the 5th-percentile dispatch time (`after_p5_dispatch`), because the live mean carries a whole-system cold start and A0+ is a cold-start correction.
The A0 cells are re-run in this stage at the same pin (the stage-1 A0 summaries predate the steady-state keys), and the two A0 sets are compared by sparse_obs_ident.py.

  sparse_obs_v2_gate_read.py --gate <dir> [<dir> ...] --topos t1,...,t12 [--out read.json]
"""
import argparse
import json
import signal
import sys

import r1_attribution_v1_read as R

A0, AP, WN, WE = "cd_exactS_ib0", "cd_obs_a0plus", "cd_obs_w1_naive", "cd_obs_w1_ec"
ARMS = (A0, AP, WN, WE)
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0",)
VOI_BAR, R_BAR, LEAK_BAR, ALPHA = 10.0, 5.0, -1.0, 0.05
BOOKS = ("pg_cold_tail_calls", "obs_cold_degraded", "obs_cold_exact", "obs_drain_degraded", "obs_exec_degraded", "obs_partner_hidden", "obs_inflight_degraded")


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    if seen - set(topos):
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {sorted(seen - set(topos))}")
    arms = {k[0] for k in list(cells) + list(failed)}
    if arms - set(ARMS):
        raise SystemExit(f"FAIL LOUD: arms outside the registered {ARMS}: {sorted(arms - set(ARMS))}")
    bad_w = sorted({k[3] for k in list(cells) + list(failed)} - set(WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")


def books(cells, kind, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == kind and k[4] == rung]
    return {"cells": len(cs), **{n: sum(int(c.get(n) or 0) for c in cs) for n in BOOKS}}


def instrument_ok(b, kind):
    other = b["obs_drain_degraded"] == 0 and b["obs_exec_degraded"] == 0 and b["obs_partner_hidden"] == 0 and b["obs_inflight_degraded"] == 0
    if kind == A0:
        return other and b["pg_cold_tail_calls"] == 0 and b["obs_cold_degraded"] == 0
    if kind == AP:
        return other and b["pg_cold_tail_calls"] > 0 and b["obs_cold_degraded"] == 0
    return other and b["pg_cold_tail_calls"] > 0 and b["obs_cold_degraded"] > 0 and b["obs_cold_exact"] == 0


def read(gates, topos):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos)
    sel = sorted(topos)
    pairs = {"primary": [(AP, A0)], "VoI_W1": [(WN, AP)], "R_W1": [(WE, AP)], "descriptive": [(WE, WN), (WN, A0), (WE, A0)]}
    contrasts = {}
    for fam, ps in pairs.items():
        for a, b in ps:
            for r in RUNGS:
                c = R.contrast(cells, a, b, r, sel)
                c["family"] = fam
                contrasts[f"{a} vs {b}|{r}"] = c
        if fam != "descriptive":
            adj = R.holm({f"{a} vs {b}|{r}": contrasts[f"{a} vs {b}|{r}"]["p"] for a, b in ps for r in RUNGS})
            for k, v in adj.items():
                contrasts[k]["p_holm"] = v
    steady = {}
    for label, key in (("after_360s", "after_360s"), ("after_p5", "after_p5_dispatch")):
        ss = {}
        for k, v in cells.items():
            x = (v.get("steady_state") or {}).get(key) or {}
            if x.get("mean") is not None:
                ss[k] = {**v, "averageElapsedTime": x["mean"]}
        for fam, ps in pairs.items():
            if fam == "descriptive":
                continue
            for a, b in ps:
                for r in RUNGS:
                    c = R.contrast(ss, a, b, r, sel)
                    c["family"] = f"steady:{label}"
                    steady[f"{a} vs {b}|{r}|{label}"] = c
    m = lambda a, b, r: contrasts[f"{a} vs {b}|{r}"]["median_pct"]
    flags = {}
    for r in RUNGS:
        voi, res = m(WN, AP, r), m(WE, AP, r)
        f = {"VOI_GE_10": voi is not None and voi >= VOI_BAR, "R_GE_5": res is not None and res >= R_BAR}
        f["W1_PASS"] = f["VOI_GE_10"] and f["R_GE_5"]
        f["A0P_GE_1"] = m(AP, A0, r) is not None and m(AP, A0, r) <= -1.0
        f["LEAK"] = {w: (m(w, AP, r) is not None and m(w, AP, r) <= LEAK_BAR and (contrasts[f"{w} vs {AP}|{r}"].get("p_holm") or 1.0) < ALPHA) for w in (WN, WE)}
        flags[r] = f
    bk = {f"{k}|{r}": books(cells, k, r) for k in ARMS for r in RUNGS}
    inst = {k: instrument_ok(b, k.split("|")[0]) for k, b in bk.items()}
    exp = len(topos) * len(WINDOWS)
    cnt = {f"{k}|{r}": {"finished": sum(1 for c in cells if c[0] == k and c[4] == r),
                        "failed": sum(1 for c in failed if c[0] == k and c[4] == r), "expected": exp} for k in ARMS for r in RUNGS}
    return {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed), "counts": cnt, "books": bk,
            "instrument_ok": inst, "INSTRUMENT_OK": all(inst.values()), "contrasts": contrasts, "flags": flags, "steady_state_descriptive": steady,
            "W1_PASS": any(f["W1_PASS"] for f in flags.values()), "LEAK": any(any(f["LEAK"].values()) for f in flags.values())}


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    for k, c in r["counts"].items():
        print(f"  {k:30s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['finished'] == c['expected'] else '   INCOMPLETE'}")
    for k, b in r["books"].items():
        print(f"  {k:30s} ok={r['instrument_ok'][k]} " + " ".join(f"{n}={b[n]}" for n in BOOKS))
    for k, c in r["contrasts"].items():
        mm = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        hp = f" p_holm={f(c.get('p_holm'))}" if "p_holm" in c else ""
        print(f"  [{c['family']:11s}] {k:38s} median {mm:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}{hp}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print("-- steady state (descriptive, labelled; not in any family or flag)")
    for k, c in r["steady_state_descriptive"].items():
        mm = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  [{c['family']:18s}] {k:46s} median {mm:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")
    for rr, v in r["flags"].items():
        print(f"  flags {rr:9s} {v}")
    print(f"-- INSTRUMENT_OK {r['INSTRUMENT_OK']}   W1_PASS {r['W1_PASS']}   LEAK {r['LEAK']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True)
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

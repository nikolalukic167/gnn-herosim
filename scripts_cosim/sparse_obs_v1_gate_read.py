"""sparse_obs_s0_v1 stage-1 reader (written before the data lands; docs/lineages/archive/sparse_obs_s0_v1.md).

Arms (g0, 12 fresh topologies 19440-19451, seed 0): A0 = cd_exactS_ib0 (full information); per degradation D in {queue, bundle} a NAIVE and an EC arm
(cd_obs_<D>_naive, cd_obs_<D>_ec). Per rung the paired % of mean latency per topology (median over the topology's windows), its median over the 12 topologies
and an exact two-sided Wilcoxon (r1_attribution_v1_read.contrast). Negative = the first arm faster.

  VoI(D) = NAIVE(D) vs A0   how much the channel is worth (a positive number = the degraded naive arm is slower)
  R(D)   = EC(D) vs A0      what a hand rule on the same observation cannot recover (the residual a learned scorer could take)
  descriptive: EC(D) vs NAIVE(D)

Registered flags per D and rung: VOI_GE_10 (VoI median >= +10 %), R_GE_5 (R median >= +5 %), D_PASS (both). The kill rule applies per CHANNEL (drain here; exec in stage 2): STAGE1_PASS =
the queue channel passes at some rung; otherwise the drain channel closes HAND-RECOVERS-UNDER-SPARSITY. The bundle (queue + exec noise + the batch plan hidden) is a MIXED read dominated by
the hidden batch plan: its VoI is the measured price of removing the batch plan (how much of the hand search's margin is in-batch co-location), reported and not a pass/kill input; its EC arm
is vacuous on partners (cross-batch pairs are ~0.1-0.4 % of pairs, so the two-step estimate has no inputs) and the partner row is NOT-TESTABLE-IN-SEAT. Holm over the (D x rung) VoI family and over the R family, reported with the exact p. Counts first (12 per arm and rung), then the instrument
books (degraded channels used, exact queue walks 0 where the queue is degraded, every other book 0).

  sparse_obs_v1_gate_read.py --gate <dir> [<dir> ...] --topos t1,...,t12 [--out read.json]
"""
import argparse
import json
import signal
import sys

import r1_attribution_v1_read as R

A0 = "cd_exactS_ib0"
DEGRADATIONS = ("queue", "bundle")
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0",)
VOI_BAR, R_BAR = 10.0, 5.0
OBS_BOOKS = ("obs_drain_degraded", "obs_drain_exact", "obs_exec_degraded", "obs_inflight_degraded", "obs_partner_hidden", "obs_partner_predicted")


def arm(d, est):
    return f"cd_obs_{d}_{est}"


ARMS = (A0,) + tuple(arm(d, e) for d in DEGRADATIONS for e in ("naive", "ec"))


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    arms = {k[0] for k in list(cells) + list(failed)}
    if arms - set(ARMS):
        raise SystemExit(f"FAIL LOUD: arms outside the registered {ARMS}: {sorted(arms - set(ARMS))}")
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


def books(cells, kind, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == kind and k[4] == rung]
    return {"cells": len(cs), **{n: sum(int(c.get(n) or 0) for c in cs) for n in OBS_BOOKS},
            "decision_s_per_task": None}


def instrument_ok(b, kind):
    if kind == A0:
        return all(b[n] == 0 for n in OBS_BOOKS)
    d = kind.split("_")[2]
    ok = b["obs_exec_degraded"] > 0 if d == "bundle" else b["obs_exec_degraded"] == 0
    ok = ok and b["obs_drain_degraded"] > 0 and b["obs_drain_exact"] == 0 and b["obs_inflight_degraded"] == 0
    ok = ok and (b["obs_partner_hidden"] > 0 if d == "bundle" else b["obs_partner_hidden"] == 0)
    if kind.endswith("_ec") and d == "bundle":
        ok = ok and b["obs_partner_predicted"] > 0
    return ok


def read(gates, topos):
    cells, failed = R.load(list(gates))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos)
    sel = sorted(topos)
    contrasts = {}
    for d in DEGRADATIONS:
        for r in RUNGS:
            contrasts[f"VoI {d}|{r}"] = R.contrast(cells, arm(d, "naive"), A0, r, sel)
            contrasts[f"R {d}|{r}"] = R.contrast(cells, arm(d, "ec"), A0, r, sel)
            contrasts[f"EC vs NAIVE {d}|{r}"] = R.contrast(cells, arm(d, "ec"), arm(d, "naive"), r, sel)
    for fam in ("VoI", "R"):
        adj = R.holm({k: v["p"] for k, v in contrasts.items() if k.startswith(fam + " ")})
        for k, v in adj.items():
            contrasts[k]["p_holm"] = v
    flags = {}
    for d in DEGRADATIONS:
        for r in RUNGS:
            voi, res = contrasts[f"VoI {d}|{r}"]["median_pct"], contrasts[f"R {d}|{r}"]["median_pct"]
            f = {"VOI_GE_10": voi is not None and voi >= VOI_BAR, "R_GE_5": res is not None and res >= R_BAR}
            f["D_PASS"] = f["VOI_GE_10"] and f["R_GE_5"]
            flags[f"{d}|{r}"] = f
    bk = {f"{k}|{r}": books(cells, k, r) for k in ARMS for r in RUNGS}
    inst = {k: instrument_ok(b, k.split("|")[0]) for k, b in bk.items()}
    return {"gate": list(gates), "topologies": sel, "n_summaries": len(cells), "n_failed": len(failed),
            "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
            "counts": counts(cells, failed, topos), "books": bk, "instrument_ok": inst, "INSTRUMENT_OK": all(inst.values()),
            "contrasts": contrasts, "flags": flags, "STAGE1_PASS": any(v["D_PASS"] for k, v in flags.items() if k.startswith("queue|")),
            "BUNDLE_VOI_PCT": {r: contrasts[f"VoI bundle|{r}"]["median_pct"] for r in RUNGS}}


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies; failed why: {r['failed_why']}")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:34s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- instrument books")
    for k, b in r["books"].items():
        print(f"  {k:34s} ok={r['instrument_ok'][k]} " + " ".join(f"{n[4:]}={b[n]}" for n in OBS_BOOKS))
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        hp = f" p_holm={f(c.get('p_holm'))}" if "p_holm" in c else ""
        print(f"  {k:34s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}{hp}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    for k, v in r["flags"].items():
        print(f"  flags {k:16s} {v}")
    print(f"-- INSTRUMENT_OK {r['INSTRUMENT_OK']}   STAGE1_PASS {r['STAGE1_PASS']}   BUNDLE_VOI_PCT {r['BUNDLE_VOI_PCT']}")


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

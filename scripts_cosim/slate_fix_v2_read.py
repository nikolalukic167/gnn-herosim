#!/usr/bin/env python3
"""slate_fix_v2 reader (docs/lineages/slate_fix_v2.md). Written before the data.

All gate cells ran at one commit (6a5c922c), so learned kinds keep their harness names; the bar is cd_exactS_ib0. Topologies
19420-19431 and windows g0-g1 only. Statistic as slate_fix_v1 (r1_attribution_v1_read): per topology the median over (window,
seed) of the paired %, then the median over topologies and an exact two-sided Wilcoxon; + = the reference is faster.

Latency fields: whole = averageElapsedTime; s360 = steady_state.after_360s.mean; p5 = steady_state.after_p5_dispatch.mean.

  * identity: every gate cell that also exists in the slate_fix_v1 directories (--ref) must have a bit-equal averageElapsedTime;
    a mismatched cell leaves every steady-state family (its pairs go with it) and is listed;
  * PRIMARY, Holm over 4: new self-search and no-split vs cd_exactS_ib0 x 2 rungs on s360 (seeds 1-2), with the whole-run % and
    delta = steady % - whole % (pp, per topology then median, Wilcoxon) beside each. Transient label per contrast and rung: HIDES
    (delta >= +1 pp, p < .05), INFLATES (delta <= -1 pp, p < .05), else NEUTRAL;
  * SECONDARY, Holm over 4: the same contrasts on p5;
  * MP, Holm over 4: gnn no-split vs twin no-split, seeds 1-4 paired by seed, x 2 rungs x {whole, s360}. Per metric: CONFIRMED
    (median <= -2 % and Holm < .05 at both rungs), TWIN-FASTER (twin Holm-faster at either rung), DIRECTION-ONLY, NOT-SEPARATED;
  * descriptive: MP on seeds 1-2 and 3-4 separately, latency deciles per arm and rung, served-slate counters, self-search vs
    no-split on s360; the bar's own transient share (expert's signature addition): per rung, the median over cd_exactS_ib0 cells of
    the whole-run mean, the after-360 s mean and 1 - s360 / whole.

  slate_fix_v2_read.py --gate <dir> [<dir> ...] --ref <slate_fix_v1 dir> [<dir> ...] [--out read.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_attribution_v1_read as R  # noqa: E402
import slate_fix_v1_gate_read as V1  # noqa: E402

RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1")
TOPO_RANGE = (19420, 19431)
IB0 = "cd_exactS_ib0"
SS, NS, TWIN_NS = "ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit", "ra_twin_eng_nosplit"
BAR_PCT, ALPHA, DELTA_PP = -2.0, 0.05, 1.0
METRICS = {"whole": ("averageElapsedTime",), "s360": ("steady_state", "after_360s", "mean"),
           "p5": ("steady_state", "after_p5_dispatch", "mean")}


def value(summary: dict, metric: str) -> Optional[float]:
    x = summary
    for k in METRICS[metric]:
        x = (x or {}).get(k) if isinstance(x, dict) else None
    return None if x is None else float(x)


def remap(cells: dict, metric: str, seeds: Optional[Sequence[int]] = None, drop: frozenset = frozenset()) -> dict:
    """Cells re-keyed so R.contrast runs on `metric`; cells without it, outside `seeds` (hand cells, seed 0, always kept) or in
    `drop` are left out."""
    out = {}
    for k, s in cells.items():
        if k in drop or (seeds is not None and k[1] != 0 and k[1] not in seeds):
            continue
        v = value(s, metric)
        if v is not None:
            out[k] = {"averageElapsedTime": v}
    return out


def load(dirs: Sequence[str]):
    cells, failed = R.load(list(dirs))
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    return cells, failed


def identity(cells: dict, ref: dict) -> dict:
    checked, mismatched = 0, []
    for k, s in cells.items():
        r = ref.get(k)
        if r is None:
            continue
        checked += 1
        if float(s["averageElapsedTime"]) != float(r["averageElapsedTime"]):
            mismatched.append({"cell": list(k), "gate": s["averageElapsedTime"], "slate_fix_v1": r["averageElapsedTime"]})
    return {"checked": checked, "mismatched": mismatched}


def holm_family(tests: Dict[str, dict]) -> Dict[str, dict]:
    adj = R.holm({k: c["p"] for k, c in tests.items()})
    for k, c in tests.items():
        c["holm_p"] = adj[k]
    return tests


def delta(cells: dict, a: str, b: str, rung: str, topos: List[int], metric: str, drop: frozenset, seeds) -> dict:
    steady = R.paired(remap(cells, metric, seeds, drop), a, b, rung, topos, set())
    whole = R.paired(remap(cells, "whole", seeds, drop), a, b, rung, topos, set())
    d = [steady[t]["median_pct"] - whole[t]["median_pct"] for t in topos if t in steady and t in whole]
    if not d:
        return {"n_topologies": 0, "median_pp": None, "p": None, "label": "NEUTRAL"}
    med, p = st.median(d), R.exact_wilcoxon(d)
    lab = "HIDES" if med >= DELTA_PP and p is not None and p < ALPHA else \
        "INFLATES" if med <= -DELTA_PP and p is not None and p < ALPHA else "NEUTRAL"
    return {"n_topologies": len(d), "median_pp": med, "p": p, "label": lab}


def mp_label(tests: Sequence[dict]) -> str:
    if any(c["median_pct"] is None or c.get("holm_p") is None for c in tests):
        return "NOT-SEPARATED"
    if any(c["median_pct"] > 0 and c["holm_p"] < ALPHA for c in tests):
        return "TWIN-FASTER"
    if all(c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA for c in tests):
        return "CONFIRMED"
    return "DIRECTION-ONLY" if all(c["median_pct"] < 0 for c in tests) else "NOT-SEPARATED"


def deciles(cells: dict) -> dict:
    out = {}
    for arm in sorted({k[0] for k in cells}):
        for r in RUNGS:
            ds = [s["steady_state"]["decile_mean"] for k, s in cells.items() if k[0] == arm and k[4] == r and (s.get("steady_state") or {}).get("decile_mean")]
            if ds:
                out[f"{arm}|{r}"] = [st.median(x[i] for x in ds) for i in range(10)]
    return out


def hand_transient(cells: dict, arm: str = IB0) -> dict:
    out = {}
    for r in RUNGS:
        pairs = [(value(s, "whole"), value(s, "s360")) for k, s in cells.items() if k[0] == arm and k[4] == r]
        pairs = [(w, x) for w, x in pairs if w is not None and x is not None and w > 0]
        if pairs:
            out[r] = {"cells": len(pairs), "whole_s": st.median(w for w, _ in pairs), "s360_s": st.median(x for _, x in pairs),
                      "transient_share": st.median(1 - x / w for w, x in pairs)}
    return out


def read(gate: Sequence[str], ref: Sequence[str]) -> dict:
    gate, ref = ([d] if isinstance(d, str) else list(d) for d in (gate, ref))
    cells, failed = load(gate)
    topos = V1.check_grid(list(cells) + list(failed))
    ref_cells, _ = load(ref)
    ident = identity(cells, ref_cells)
    drop = frozenset(tuple(m["cell"]) for m in ident["mismatched"])
    s12 = (1, 2)
    prim = {f"{n}|{r}": R.contrast(remap(cells, "s360", s12, drop), a, IB0, r, topos)
            for a, n in ((SS, "self-search"), (NS, "no-split")) for r in RUNGS}
    holm_family(prim)
    for key, c in prim.items():
        n, r = key.split("|")
        a = SS if n == "self-search" else NS
        c["whole"] = R.contrast(remap(cells, "whole", s12), a, IB0, r, topos)
        c["delta"] = delta(cells, a, IB0, r, topos, "s360", drop, s12)
        c["label"] = V1.rung_label(c)
    sec = holm_family({f"{n}|{r}": R.contrast(remap(cells, "p5", s12, drop), a, IB0, r, topos)
                       for a, n in ((SS, "self-search"), (NS, "no-split")) for r in RUNGS})
    for key, c in sec.items():
        n, r = key.split("|")
        c["delta"] = delta(cells, SS if n == "self-search" else NS, IB0, r, topos, "p5", drop, s12)
    s14 = (1, 2, 3, 4)
    mp = holm_family({f"{m}|{r}": R.contrast(remap(cells, m, s14, drop if m != "whole" else frozenset()), NS, TWIN_NS, r, topos)
                      for m in ("whole", "s360") for r in RUNGS})
    mp_labels = {m: mp_label([mp[f"{m}|{r}"] for r in RUNGS]) for m in ("whole", "s360")}
    desc = {}
    for seeds, tag in (((1, 2), "seeds 1-2"), ((3, 4), "seeds 3-4")):
        for m in ("whole", "s360"):
            for r in RUNGS:
                desc[f"mp {tag} {m}|{r}"] = R.contrast(remap(cells, m, seeds, drop if m != "whole" else frozenset()), NS, TWIN_NS, r, topos)
    for r in RUNGS:
        desc[f"self-search vs no-split s360|{r}"] = R.contrast(remap(cells, "s360", s12, drop), SS, NS, r, topos)
    return {"gate": list(gate), "ref": list(ref), "code_commits": V1.code_commits(list(gate)), "topologies": topos,
            "n_summaries": len(cells), "n_failed": len(failed), "arm_table": R.arm_table(cells, failed, {}), "identity": ident,
            "primary_s360": prim, "secondary_p5": sec, "mp": {"tests": mp, "labels": mp_labels}, "descriptive": desc,
            "deciles": deciles(cells), "hand_transient": hand_transient(cells), "slate": V1.slate_counts({(f"g:{k[0]}",) + k[1:]: v for k, v in cells.items() if k[0].startswith("ra_")})}


def print_report(r: dict) -> None:
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- code: " + "; ".join(f"{os.path.basename(k.rstrip('/'))} {','.join(c[:8] for c in v)}" for k, v in r["code_commits"].items()))
    i = r["identity"]
    print(f"-- whole-run identity vs slate_fix_v1: {i['checked']} cells checked, {len(i['mismatched'])} mismatched"
          + ("" if not i["mismatched"] else f" (dropped from the steady-state families: {[m['cell'] for m in i['mismatched']][:6]} ...)"))
    print("-- cells per arm and rung (finished/failed, median wall s)")
    for k, a in r["arm_table"].items():
        print(f"  {k:36s} {a['finished']:4d}/{a['failed']:<3d} wall {f(a['median_wallclock_s'], 0)}")
    print("-- PRIMARY: vs cd_exactS_ib0 on latency after 360 s (Holm over 4), whole-run beside, delta = steady - whole")
    for k, c in r["primary_s360"].items():
        d = c["delta"]
        print(f"  {k:22s} steady {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) Holm {f(c['holm_p'], 4)} {c['label']:15s} "
              f"| whole {f(c['whole']['median_pct'])} % | delta {f(d['median_pp'])} pp p {f(d['p'], 4)} {d['label']}")
    print("-- SECONDARY: the same on latency after the 5th-percentile dispatch (Holm over 4)")
    for k, c in r["secondary_p5"].items():
        print(f"  {k:22s} {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) Holm {f(c['holm_p'], 4)} | delta {f(c['delta']['median_pp'])} pp {c['delta']['label']}")
    print("-- MP: gnn no-split vs twin no-split, seeds 1-4 (Holm over 4)")
    for k, c in r["mp"]["tests"].items():
        print(f"  {k:22s} {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) p {f(c['p'], 4)} Holm {f(c['holm_p'], 4)}")
    print(f"  MP labels: {r['mp']['labels']}  (model-class result, never a hand win)")
    print("-- descriptive")
    for k, c in r["descriptive"].items():
        print(f"  {k:36s} {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) p {f(c['p'], 4)}")
    print("-- the bar's own transient (cd_exactS_ib0; median over cells): whole-run, after 360 s, share of the whole-run mean in the opening")
    for r_, h in r["hand_transient"].items():
        print(f"  {r_:9s} whole {f(h['whole_s'], 3)} s  after-360 {f(h['s360_s'], 3)} s  share {f(100 * h['transient_share'], 1)} %  (n {h['cells']})")
    print("-- latency deciles in dispatch order (median over cells, s)")
    for k, d in r["deciles"].items():
        print(f"  {k:36s} " + " ".join(f(x) for x in d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = read(a.gate, a.ref)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(r, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

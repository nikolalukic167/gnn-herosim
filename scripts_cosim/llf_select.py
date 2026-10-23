#!/usr/bin/env python3
"""live_link_features_v1, amendment A1's selection rule: which variant (as built, (a) static, (b) re-encode) goes on to seeds 2-3 and confirmation.

Run ONCE, after the last 20-epoch seed-1 selection cell of all three variants is complete (A4). It refuses to run while any cell is missing,
and refuses to overwrite its --out.

  llf_select.py --as-built DIR --static DIR --conv DIR --flags FLAGS.json --out SELECT.json

DIR holds that variant's cells (gnn and its control twin, seed 1): cc40s<topo>__<window><rung>__<kind>_nosplit_k100_s1.summary.json for topologies
30176-30181, windows g0 g1, rungs moderate heavy. Per variant, gnn kind / control-twin kind:
  as_built  ra_gnn_eng_combfl_link        / ra_twin_eng_combfl_link
  static    ra_gnn_eng_combfl_linkstatic  / ra_twin_eng_combfl_linkstatic
  conv      ra_gnn_eng_combfl_linkconv    / ra_twin_eng_combfl_link_gpu   (A1 extension: (b)'s seed-1 twin is GPU-trained)

FLAGS.json: {"as_built": {"parity": bool, "collapse": bool, "curves": bool, "source": str}, "static": {...}, "conv": {...}}. A variant is eligible only if
all three are literally true; a missing variant, key or non-bool is not a pass. A4: a (b) that never became trainable is entered with its flags false.

Score (A1): cell value = whole-run averageElapsedTime. Paired % = 100 (gnn - twin) / twin per (topology, window, rung). Runaway: a cell > 10x the
median of its arm's 12 cells at that rung; "without runaways" drops a (topology, window, rung) pair when either cell is flagged. Per topology, the
median over its windows left; the score is the median over the (topology, rung) values left (12 when nothing is dropped). Lower is better.
Choice among the eligible: the lowest score; variants within 0.5 pp of it are tied, and the tie goes to the lowest all-cells score, again with 0.5 pp
ties, then to the simpler variant (as_built < static < conv). No eligible variant, or an eligible variant with no score: HOLD (exit 2), report to the user.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

TOPOS = (30176, 30177, 30178, 30179, 30180, 30181)
WINDOWS = ("g0", "g1")
RUNGS = ("moderate", "heavy")
SEED = 1
CELL_SUFFIX = "_nosplit_k100"
VARIANTS = ("as_built", "static", "conv")  # also the simplicity order
PAIRS = {"as_built": ("ra_gnn_eng_combfl_link", "ra_twin_eng_combfl_link"),
         "static": ("ra_gnn_eng_combfl_linkstatic", "ra_twin_eng_combfl_linkstatic"),
         "conv": ("ra_gnn_eng_combfl_linkconv", "ra_twin_eng_combfl_link_gpu")}
FLAG_KEYS = ("parity", "collapse", "curves")
RUNAWAY_X = 10.0
TIE_PP = 0.5


def cell_path(d, kind, t, w, r):
    return os.path.join(d, f"cc40s{t}__{w}{r}__{kind}{CELL_SUFFIX}_s{SEED}.summary.json")


def check_option_d(s, where):
    fs, tcp = s.get("peerExchangeFairshare"), s.get("peerExchangeTcp")
    if not isinstance(fs, dict) or fs.get("partners") != "serial" or not isinstance(tcp, dict) or s.get("accessCellularOutMbps") is not None:
        raise SystemExit(f"FAIL LOUD: {where} is not an option D cell (fair share serial + TCP on, default cellular)")


def load_variant(name, d):
    """{(arm, t, w, r): averageElapsedTime}; refuses unless all 48 cells exist, are option D and carry a finite latency."""
    missing, cells, ntasks = [], {}, {}
    for arm, kind in zip(("gnn", "twin"), PAIRS[name]):
        for t in TOPOS:
            for w in WINDOWS:
                for r in RUNGS:
                    p = cell_path(d, kind, t, w, r)
                    if not os.path.exists(p):
                        missing.append(os.path.basename(p))
                        continue
                    with open(p) as fh:
                        s = json.load(fh)
                    check_option_d(s, p)
                    v = s.get("averageElapsedTime")
                    if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                        raise SystemExit(f"FAIL LOUD: {p} has no finite averageElapsedTime: {v!r}")
                    n = s.get("num_tasks")
                    if ntasks.setdefault((t, w, r), n) != n:
                        raise SystemExit(f"FAIL LOUD: {name} {t} {w} {r}: gnn and twin served different task counts ({ntasks[(t, w, r)]} vs {n})")
                    cells[(arm, t, w, r)] = float(v)
    if missing:
        raise SystemExit(f"FAIL LOUD: variant {name} is not complete in {d}: {len(missing)} of 48 cells missing, e.g. {missing[:4]}. The selection runs only when all three variants are complete (A4).")
    return cells


def runaways(cells):
    out = set()
    for arm in ("gnn", "twin"):
        for r in RUNGS:
            vals = [cells[(arm, t, w, r)] for t in TOPOS for w in WINDOWS]
            med = statistics.median(vals)
            out |= {(arm, t, w, r) for t in TOPOS for w in WINDOWS if cells[(arm, t, w, r)] > RUNAWAY_X * med}
    return out


def score(cells, drop):
    """Median over (topology, rung) of the per-topology window-median paired %; `drop` = flagged cells (pair dropped if either is flagged)."""
    vals, per = [], {}
    for r in RUNGS:
        for t in TOPOS:
            ws = [100.0 * (cells[("gnn", t, w, r)] - cells[("twin", t, w, r)]) / cells[("twin", t, w, r)]
                  for w in WINDOWS if ("gnn", t, w, r) not in drop and ("twin", t, w, r) not in drop]
            if ws:
                per[f"{r}:{t}"] = statistics.median(ws)
                vals.append(per[f"{r}:{t}"])
    return (statistics.median(vals) if vals else None), len(vals), per


def eligible(flags, name):
    f = flags.get(name)
    return isinstance(f, dict) and all(f.get(k) is True for k in FLAG_KEYS)


def choose(res):
    """res: {variant: {"eligible", "score", "score_all"}} -> (choice or None, reason)."""
    el = [v for v in VARIANTS if res[v]["eligible"]]
    if not el:
        return None, "HOLD: no variant is eligible"
    noscore = [v for v in el if res[v]["score"] is None or res[v]["score_all"] is None]
    if noscore:
        return None, f"HOLD: eligible variant(s) {noscore} have no score"
    best = min(res[v]["score"] for v in el)
    tied = [v for v in el if res[v]["score"] <= best + TIE_PP]
    if len(tied) == 1:
        return tied[0], "lowest score without runaways"
    best_all = min(res[v]["score_all"] for v in tied)
    tied2 = [v for v in tied if res[v]["score_all"] <= best_all + TIE_PP]
    if len(tied2) == 1:
        return tied2[0], f"tie within {TIE_PP} pp among {tied}; lowest all-cells score"
    return tied2[0], f"tie within {TIE_PP} pp among {tied}, all-cells tie among {tied2}; simpler variant"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-built", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists; the selection is read once")
    dirs = {"as_built": a.as_built, "static": a.static, "conv": a.conv}
    with open(a.flags) as fh:
        flags = json.load(fh)
    cells = {v: load_variant(v, dirs[v]) for v in VARIANTS}  # every variant complete before anything is computed
    res = {}
    for v in VARIANTS:
        run = runaways(cells[v])
        s, n, per = score(cells[v], run)
        s_all, n_all, per_all = score(cells[v], set())
        res[v] = {"gnn": PAIRS[v][0], "twin": PAIRS[v][1], "dir": dirs[v], "flags": flags.get(v), "eligible": eligible(flags, v),
                  "score": s, "n": n, "score_all": s_all, "n_all": n_all, "per_topology": per, "per_topology_all": per_all,
                  "runaway_cells": sorted(f"{arm}:{t}:{w}:{r}" for arm, t, w, r in run)}
    choice, reason = choose(res)
    out = {"rule": "live_link_features_v1 A1 (A4: all three variants complete)", "choice": choice, "reason": reason, "variants": res}
    for v in VARIANTS:
        x = res[v]
        sc = "none" if x["score"] is None else f"{x['score']:+.2f}%"
        print(f"{v:9s} eligible={x['eligible']!s:5s} score {sc} (n={x['n']})  all-cells {x['score_all']:+.2f}% (n={x['n_all']})  runaways {len(x['runaway_cells'])}  flags {x['flags']}")
    print(f"CHOICE: {choice}  ({reason})")
    with open(a.out, "x") as fh:
        json.dump(out, fh, indent=1)
    return 0 if choice else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""uncut_groups_s0_v1 TE3 read (topology extension for finding (c), registered 2026-10-17, 1d5d0243).

  uncut_te3_read.py --new-bar <cells_bar> --new-cells arm3=<dir>,arm1b=<dir>,k0=<dir>
                    [--old-bar <869828 bar dir> --old-cells arm3=<dir>,arm1b=<dir>,k0=<dir>] --out te3.json

Statistic: the filed reader's (uncut_arm_gate_read.load / contrast, imported, not copied). Cell = mean over seeds 1 and 2; topology = median over the
four windows of 100 (arm - bar) / bar against cd_ib0_k100 (seed 0); exact two-sided Wilcoxon over the topologies. A topology with any missing cell
(bar or either seed of the arm, any window) at a rung is dropped from that rung's tests and listed.

TE3: heavy {whole, after360} x {Arm 3, Arm 1b, K0} = 6 tests, Holm within each charge (frozen _k100 and co-tenancy _kr). Per test and charge:
  CLAIM (Holm p < .05, median < 0), REVERSED (Holm p < .05, median > 0), CLEAN-NEGATIVE (neither, and the sign-test 95 % interval for the median,
  [x(k), x(n+1-k)], lies above -1 %, i.e. excludes an arm-faster effect of 1 % or more), UNRESOLVED (otherwise).
Combined over the two charges: the label stands only if both charges carry it, else UNRESOLVED. A rung's finding stands only if both statistics carry
the same combined label, else UNRESOLVED. Moderate is descriptive (median, mean, n ahead; no label). The pooled line over the old and the new
topologies is descriptive (no label, no p used) and mixes two bar venues.
"""
import argparse
import json
import os
import statistics as st
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uncut_arm_gate_read as R  # noqa: E402

ARMS = {"arm3": "ra_gnn_eng_arm3_nosplit", "arm1b": "ra_gnn_eng_arm1b_nosplit", "k0": "ra_gnn_eng_arm2k0_nosplit"}
CHARGES = ("k100", "kr")
BAR = R.BAR
CLEAN_MARGIN = -1.0


def sign_interval_index(n, cover=0.95):
    """Largest k with 1 - 2 P(Bin(n, .5) <= k - 1) >= cover; the interval is [x(k), x(n+1-k)] (1-based order statistics)."""
    best = None
    for k in range(1, n // 2 + 1):
        cov = 1 - 2 * sum(comb(n, j) for j in range(k)) / 2 ** n
        if cov >= cover:
            best = (k, cov)
    return best


def label(p_holm, median, vals):
    if p_holm is not None and p_holm < R.ALPHA:
        return "CLAIM" if median < 0 else "REVERSED"
    idx = sign_interval_index(len(vals))
    if idx is not None:
        k, _ = idx
        lo = sorted(vals)[k - 1]
        if lo > CLEAN_MARGIN:
            return "CLEAN-NEGATIVE"
    return "UNRESOLVED"


def combine(labels):
    return labels[0] if len(set(labels)) == 1 else "UNRESOLVED"


def complete_topos(cells, arm, seeds, rung, topos):
    keep, dropped = [], []
    for t in topos:
        ok = all((arm, sd, t, w, rung) in cells for sd in seeds for w in R.WINDOWS) and all((BAR, 0, t, w, rung) in cells for w in R.WINDOWS)
        (keep if ok else dropped).append(t)
    return keep, dropped


def merged(bar_dirs, arm_dirs, charge):
    bar, bfail = R.load(bar_dirs, "_k100")
    arms, afail = R.load(arm_dirs, "_" + charge)
    cells = dict(bar)
    for k, v in arms.items():
        if k in cells:
            raise SystemExit(f"FAIL LOUD: cell {k} is in both the bar and the arm directories")
        cells[k] = v
    return cells, {**bfail, **afail}


def descr(cells, arm, rung, stat, topos):
    c = R.contrast(cells, arm, R.SEEDS, BAR, (0,), rung, topos, stat)
    vals = list(c["per_topology"].values())
    return {"n_topologies": c["n_topologies"], "median_pct": c["median_pct"], "mean_pct": st.mean(vals) if vals else None, "n_ahead": c["faster"]}


def read(new_bar, new_cells, old_bar=None, old_cells=None):
    out = {"label": "TE3 (registered 1d5d0243)", "charges": {}}
    for charge in CHARGES:
        cells, failed = merged(new_bar, ",".join(new_cells.values()), charge)
        topos = sorted({k[2] for k in cells if k[0] == BAR})
        entry = {"n_new_topologies": len(topos), "failed_cells": len(failed), "tests": {}, "dropped": {}, "moderate": {}}
        tests = []
        for name, arm in ARMS.items():
            for stat in R.STATS:
                keep, dropped = complete_topos(cells, arm, R.SEEDS, "heavy", topos)
                c = R.contrast(cells, arm, R.SEEDS, BAR, (0,), "heavy", keep, stat)
                c["name"] = f"{name}/heavy/{stat}"
                tests.append(c)
                entry["dropped"][f"{name}/heavy"] = dropped
        R.family(tests)
        for c in tests:
            vals = list(c["per_topology"].values())
            c["label"] = label(c["p_holm"], c["median_pct"], vals) if vals else "UNRESOLVED"
            c["sign_interval"] = None
            idx = sign_interval_index(len(vals)) if vals else None
            if idx:
                k, cov = idx
                sv = sorted(vals)
                c["sign_interval"] = {"k": k, "coverage": round(cov, 4), "lo": sv[k - 1], "hi": sv[len(sv) - k]}
            entry["tests"][c["name"]] = c
        for name, arm in ARMS.items():
            for rung in ("moderate",):
                for stat in R.STATS:
                    keep, _ = complete_topos(cells, arm, R.SEEDS, rung, topos)
                    entry["moderate"][f"{name}/{rung}/{stat}"] = descr(cells, arm, rung, stat, keep)
        out["charges"][charge] = entry
    out["combined"] = {}
    for name in ARMS:
        row = {}
        for stat in R.STATS:
            row[stat] = combine([out["charges"][ch]["tests"][f"{name}/heavy/{stat}"]["label"] for ch in CHARGES])
        row["heavy_rung"] = combine([row[s] for s in R.STATS])
        out["combined"][name] = row
    if old_bar and old_cells:
        out["pooled_descriptive"] = {"note": "old 12 topologies (bar 869828, SMT-on) + new 36; two bar venues; no label", "charges": {}}
        for charge in CHARGES:
            cells, _ = merged(",".join([new_bar, old_bar]), ",".join(list(new_cells.values()) + list(old_cells.values())), charge)
            topos = sorted({k[2] for k in cells if k[0] == BAR})
            row = {}
            for name, arm in ARMS.items():
                for rung in ("heavy", "moderate"):
                    for stat in R.STATS:
                        keep, _ = complete_topos(cells, arm, R.SEEDS, rung, topos)
                        row[f"{name}/{rung}/{stat}"] = descr(cells, arm, rung, stat, keep)
            out["pooled_descriptive"]["charges"][charge] = {"n_topologies": len(topos), **row}
    return out


def pairs(spec):
    return {a.split("=")[0]: a.split("=")[1] for a in spec.split(",") if a}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-bar", required=True)
    ap.add_argument("--new-cells", required=True)
    ap.add_argument("--old-bar")
    ap.add_argument("--old-cells")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    nc = pairs(a.new_cells)
    if set(nc) != set(ARMS):
        raise SystemExit(f"FAIL LOUD: --new-cells needs exactly {sorted(ARMS)}; got {sorted(nc)}")
    oc = pairs(a.old_cells) if a.old_cells else None
    if oc is not None and set(oc) != set(ARMS):
        raise SystemExit(f"FAIL LOUD: --old-cells needs exactly {sorted(ARMS)}; got {sorted(oc)}")
    res = read(a.new_bar, nc, a.old_bar, oc)
    txt = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n")
    for ch in CHARGES:
        e = res["charges"][ch]
        print(f"[TE3] charge {ch}: {e['n_new_topologies']} new topologies, {e['failed_cells']} failed cells")
        for n, t in e["tests"].items():
            print(f"[TE3]   {n}: median {t['median_pct']:+.2f} % n {t['n_topologies']} ahead {t['faster']} p {t['p']:.4g} holm {t['p_holm']:.4g} -> {t['label']}")
    for name, row in res["combined"].items():
        print(f"[TE3] combined {name}: whole {row['whole']}, after360 {row['after360']}, heavy rung {row['heavy_rung']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

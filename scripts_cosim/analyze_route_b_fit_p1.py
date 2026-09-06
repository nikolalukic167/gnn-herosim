#!/usr/bin/env python3
"""route_b fit-ceiling Phase 1 — paired read of the held-out ordering across seeds.

Exploration, not a registration. The 2026-09-03 probe read "held-out favours the pointwise
arms" off ONE seed and a MEAN over 31 test parents in which three datasets carried half
the sum (Phase 0, 2026-09-06). This script reads the same quantity as a paired statistic:

  * unit = training seed; each seed has one checkpoint per arm on the same split;
  * per (arm, seed): mean and median decode-regret % over the held-out parents;
  * per arm pair: paired per-seed differences (Wilcoxon signed-rank, exact sign test) and
    per-(seed, dataset) win counts.

Inputs are the per-checkpoint reports `eval_route_b_stage2_arm.py --report` writes.

    analyze_route_b_fit_p1.py \\
        --arm gnn=simulation_data/route_b_fit_p1/eval_gnn_seed{1..8}.json \\
        --arm mpoff=... --arm mlp=... [--split test|heldout]

`--split test` (default) is for val-selected checkpoints; `--split heldout` (val+test) is
only valid for last-epoch checkpoints, which never saw the val split's regret.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = {"test": ("test",), "heldout": ("val", "test")}


def load_report(path: Path, splits: Tuple[str, ...]) -> Dict[str, float]:
    d = json.load(open(path))
    rows = d.get("per_dataset") or d.get("datasets") or d.get("rows")
    if not rows:
        raise SystemExit(f"FAIL LOUD: {path} has no per-dataset rows")
    out = {}
    for r in rows:
        if r["split"] not in splits:
            continue
        rg = r["decode_regret_pct"]
        if rg is None:
            raise SystemExit(f"FAIL LOUD: {path}: {r['dataset_id']} has no decode (infeasible?)")
        out[r["dataset_id"]] = float(rg["registered"])
    return out


def wilcoxon_signed_rank(diffs: List[float]) -> Tuple[float, float]:
    """Two-sided p via scipy if present, else an exact sign-flip enumeration (n<=20)."""
    d = [x for x in diffs if abs(x) > 1e-12]
    if len(d) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon  # type: ignore
        res = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        pass
    ranks = _midranks([abs(x) for x in d])
    w_obs = sum(r for x, r in zip(d, ranks) if x > 0)
    n = len(d)
    if n > 20:
        raise SystemExit("FAIL LOUD: n>20 without scipy; install scipy")
    ge = 0
    total = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = sum(r for s, r in zip(signs, ranks) if s)
        total += 1
        if abs(w - n * (n + 1) / 4) >= abs(w_obs - n * (n + 1) / 4) - 1e-12:
            ge += 1
    return w_obs, ge / total


def _midranks(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def sign_test_p(diffs: List[float]) -> float:
    from math import comb
    d = [x for x in diffs if abs(x) > 1e-12]
    n, k = len(d), sum(1 for x in d if x > 0)
    if n == 0:
        return float("nan")
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="label=glob of per-seed eval reports; seed is read from the filename's seed{N}")
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    splits = SPLITS[args.split]
    arms: Dict[str, Dict[int, Dict[str, float]]] = {}
    for spec in args.arm:
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"FAIL LOUD: no reports match {pattern}")
        for f in files:
            stem = Path(f).stem
            seed = int(stem.split("seed")[-1].split("-")[0].split("_")[0])
            arms.setdefault(label, {})[seed] = load_report(Path(f), splits)

    seeds = sorted(set.intersection(*(set(v) for v in arms.values())))
    if not seeds:
        raise SystemExit("FAIL LOUD: no seed present in every arm")
    datasets = sorted(set.intersection(*(set(arms[a][s]) for a in arms for s in seeds)))
    print(f"split={args.split}  seeds common to all arms: {seeds}  held-out parents: {len(datasets)}")

    per_seed: Dict[str, Dict[int, Tuple[float, float]]] = {}
    print(f"\n{'arm':<8} {'seed':>4} {'mean%':>8} {'median%':>8} {'zero%':>6}")
    for a in arms:
        for s in seeds:
            v = [arms[a][s][d] for d in datasets]
            per_seed.setdefault(a, {})[s] = (st.mean(v), st.median(v))
            print(f"{a:<8} {s:>4} {st.mean(v):8.2f} {st.median(v):8.2f} {100*sum(1 for x in v if x==0)/len(v):6.0f}")
    print(f"\n{'arm':<8} {'mean-of-means':>14} {'median-of-medians':>18} {'sd(mean)':>9}")
    for a in arms:
        ms = [per_seed[a][s][0] for s in seeds]
        md = [per_seed[a][s][1] for s in seeds]
        print(f"{a:<8} {st.mean(ms):14.2f} {st.median(md):18.2f} {(st.pstdev(ms) if len(ms)>1 else 0):9.2f}")

    summary = {"split": args.split, "seeds": seeds, "n_datasets": len(datasets), "pairs": {}}
    print("\nPaired per-seed differences (B - A, negative = B better):")
    for a, b in itertools.combinations(arms, 2):
        dm = [per_seed[b][s][0] - per_seed[a][s][0] for s in seeds]
        dd = [per_seed[b][s][1] - per_seed[a][s][1] for s in seeds]
        _, p_mean = wilcoxon_signed_rank(dm)
        _, p_med = wilcoxon_signed_rank(dd)
        wins_b = sum(1 for s in seeds for d in datasets if arms[b][s][d] < arms[a][s][d] - 1e-9)
        wins_a = sum(1 for s in seeds for d in datasets if arms[a][s][d] < arms[b][s][d] - 1e-9)
        ties = len(seeds) * len(datasets) - wins_a - wins_b
        print(f"  {b} - {a}: mean-diff {st.mean(dm):+7.2f}pp (median of seeds {st.median(dm):+7.2f}; "
              f"{sum(1 for x in dm if x<0)}/{len(dm)} seeds B better; Wilcoxon p={p_mean:.3f}, sign p={sign_test_p(dm):.3f}) | "
              f"median-diff {st.mean(dd):+7.2f}pp (p={p_med:.3f}) | per-(seed,dataset) wins B {wins_b} / A {wins_a} / ties {ties}")
        summary["pairs"][f"{b}-{a}"] = {
            "mean_diff_pp": dm, "median_diff_pp": dd, "p_wilcoxon_mean": p_mean,
            "p_wilcoxon_median": p_med, "p_sign_mean": sign_test_p(dm),
            "wins_b": wins_b, "wins_a": wins_a, "ties": ties,
        }
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

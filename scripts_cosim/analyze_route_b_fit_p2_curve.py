"""route_b fit-ceiling Phase 2: the learning curve and the registered rung-3 reading.

Reads the per-rung verdict JSONs written by `analyze_route_b_fit_p1.py`
(`simulation_data/route_b_fit_p2_r{1,2,3}_verdict.json`, produced by
`route_b_fit_p2_eval.sh`) and prints, per rung and per arm pair, the mean of the per-seed
median-paired differences, the exact Wilcoxon p on those medians, and the per-(seed,
parent) win/tie/loss counts. Then applies the registered rule (docs/lineages/route_b_v1.md,
Phase 2, 2026-09-06) to the rung-3 GNN-vs-MP-OFF contrast:

    D_s = median(GNN MP-ON) - median(GNN MP-OFF), 8 paired seeds, exact Wilcoxon, two-sided 0.05
    DATA-RESCUE   D < 0, p < 0.05      GAP-PERSISTS   D > 0, p < 0.05      else INDETERMINATE

The verdict files store pair `mpoff-gnn` as (mpoff - gnn), i.e. -D; the sign is flipped here.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

RUNG_TRAIN = {1: 204, 2: 612, 3: 1020}
PRIMARY_PAIR = "mpoff-gnn"  # stored as mpoff - gnn = -D


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdicts", nargs="+", default=[
        f"simulation_data/route_b_fit_p2_r{r}_verdict.json" for r in (1, 2, 3)])
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = []
    for path in args.verdicts:
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"FAIL LOUD: missing {p}")
        m = re.search(r"_r(\d+)_verdict$", p.stem)
        if not m:
            raise SystemExit(f"FAIL LOUD: {p.name} is not a *_r<N>_verdict.json file")
        rung = int(m.group(1))
        d = json.loads(p.read_text())
        rows.append((rung, d))
    rows.sort()

    print(f"{'rung':>4} {'train':>6} {'test':>5} {'seeds':>5}  pair          "
          f"{'mean(med diff) pp':>18} {'p_wilcoxon(med)':>15} {'mean(mean diff)':>15} {'p(mean)':>8}  wins_a/ties/wins_b")
    for rung, d in rows:
        for pair, v in d["pairs"].items():
            md, mn = v["median_diff_pp"], v["mean_diff_pp"]
            print(f"{rung:>4} {RUNG_TRAIN.get(rung, '?'):>6} {d['n_datasets']:>5} {len(d['seeds']):>5}  {pair:<13} "
                  f"{statistics.mean(md):>+18.2f} {v['p_wilcoxon_median']:>15.4f} {statistics.mean(mn):>+15.2f} "
                  f"{v['p_wilcoxon_mean']:>8.4f}  {v['wins_a']}/{v['ties']}/{v['wins_b']}")

    top = [d for r, d in rows if r == 3]
    if not top:
        print("\nno rung-3 verdict yet — registered reading not applicable")
        return
    v = top[0]["pairs"].get(PRIMARY_PAIR)
    if v is None:
        raise SystemExit(f"FAIL LOUD: rung-3 verdict lacks pair {PRIMARY_PAIR}")
    D = [-x for x in v["median_diff_pp"]]  # GNN - MP-OFF per seed
    p = v["p_wilcoxon_median"]
    med_D, mean_D = statistics.median(D), statistics.mean(D)
    if p < args.alpha and med_D < 0 and mean_D < 0:
        verdict = "DATA-RESCUE"
    elif p < args.alpha and med_D > 0 and mean_D > 0:
        verdict = "GAP-PERSISTS"
    else:
        verdict = "INDETERMINATE"
    print(f"\nREGISTERED rung-3 contrast D = median(GNN) - median(MP-OFF), n={len(D)} seeds: "
          f"median {med_D:+.2f} pp, mean {mean_D:+.2f} pp, exact Wilcoxon p = {p:.4f}")
    print(f"per-seed D: {[round(x, 2) for x in D]}")
    print(f"READING: {verdict}")


if __name__ == "__main__":
    main()

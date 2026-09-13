#!/usr/bin/env python3
"""D3b power computation for the route_b_v1 stage 2 pre-registration (§6).

Every number cited in §6 of docs/lineages/route_b_v1/stage2-preregistration.md comes out of this script:
the effect-size table, the compound PASS-condition power, and the f=0 null false-pass row.

Effect model (assumptions registered in the doc):
  - The MLP(T1) per-dataset floor is the per-dataset residual named by --floor-column:
      * planning draft: r_exact_repaired_kint_pct (count-augmented SEPARABLE surrogate —
        obsolete for powering the moment the §9a T1 repair runs);
      * post-§9a: r_exact_repaired_t1_pct (the T1-expressible surrogate's residual).
  - The GNN captures a fraction f of that closable gap: per-dataset true effect
    delta_i = f * floor_i.
  - Per-dataset paired-difference noise after 8-draw medians: N(0, sigma^2). sigma is
    --sigma; the planning draft used 2.5 (B0's separable-physics residual ceiling — a
    stand-in), superseded by the §9 pre-probe measurement per the calibrate-then-freeze
    step.
  - Holdout composition i.i.d. from the empirical stage-1 per-dataset distribution, zeros
    included (the firing stratum is a definition applied post hoc, so power comes from the
    all-datasets view).

Test simulated: one-sided paired t (z approx, n >= 100) on delta > 0, alpha = 0.05, plus
the compound §8 condition-1 (CI excludes 0 AND mean >= floor M). The f=0 row is the null
false-pass rate of the compound condition.
"""
import argparse
import json
import math
import random
import statistics

T95 = 1.645  # one-sided z; every n here is >= 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report",
                    default="simulation_data/route_b_stage2_preprobe_t1_rtt.json",
                    help="frozen §9a pre-probe report (must contain per_dataset with the "
                         "t1 repair column); the stage-1 report carries no t1 column")
    ap.add_argument("--alpha-key", default="2.0")
    ap.add_argument("--floor-column", default="r_exact_repaired_t1_pct",
                    help="per-dataset MLP(T1) floor; the kint column is the obsolete "
                         "planning-draft floor (§9a declared it superseded)")
    ap.add_argument("--sigma", type=float, default=2.5,
                    help="per-dataset paired-difference noise sd (%%); calibrated by §9")
    ap.add_argument("--materiality-floor", type=float, default=0.25,
                    help="M: compound condition-1 point-estimate floor (%%)")
    ap.add_argument("--n-grid", default="204,300,504,804")
    ap.add_argument("--f-grid", default="0,0.3,0.5,0.7",
                    help="captured-fraction grid; 0 is the null row")
    ap.add_argument("--nsim", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    random.seed(args.seed)
    rep = json.load(open(args.report))[0]
    rows = rep["per_dataset"][args.alpha_key]
    floor = [r[args.floor_column] for r in rows if r.get(args.floor_column) is not None]
    if len(floor) != len(rows):
        raise RuntimeError(
            f"{args.floor_column}: {len(rows) - len(floor)} datasets have no value "
            "(saturated repairs?) — refusing to power on a silently reduced corpus")

    def frac_gt(xs, t):
        return sum(1 for x in xs if x > t) / len(xs)

    print(f"report={args.report}  alpha={args.alpha_key}  n_datasets={len(floor)}")
    print(f"floor column: {args.floor_column}  mean={statistics.mean(floor):.3f}  "
          f"median={statistics.median(floor):.3f}  frac>5%={frac_gt(floor, 5):.3f}  "
          f"max={max(floor):.1f}")
    print(f"sigma={args.sigma}%  M={args.materiality_floor}%  one-sided alpha=0.05  "
          f"{args.nsim} sims/cell\n")

    header = (f"{'n':>5} {'f':>5} {'mean_eff':>8} {'power(t)':>9} "
              f"{'power(CI>0 & mean>=M)':>22} {'det_margin(med)':>15}")
    print(header)
    for n in (int(t) for t in args.n_grid.split(",")):
        for f in (float(t) for t in args.f_grid.split(",")):
            t_hits = comp_hits = 0
            margins = []
            for _ in range(args.nsim):
                d = [f * random.choice(floor) + random.gauss(0, args.sigma)
                     for _ in range(n)]
                m = statistics.mean(d)
                se = statistics.stdev(d) / math.sqrt(n)
                if m / se > T95:
                    t_hits += 1
                if m - T95 * se > 0 and m >= args.materiality_floor:
                    comp_hits += 1
                margins.append(m - T95 * se)
            label = "null" if f == 0 else f"{f:.1f}"
            print(f"{n:>5} {label:>5} {f * statistics.mean(floor):>8.3f} "
                  f"{t_hits / args.nsim:>9.3f} {comp_hits / args.nsim:>22.3f} "
                  f"{statistics.median(margins):>15.3f}")
    print("\nnote: at f=0 the compound column IS the null false-pass rate of §8 "
          "condition 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

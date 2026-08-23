"""Score any {condition} x {arm} x cell live-gate matrix laid out as <prefix>_<cond>_<arm>.

Generalizes score_draw_robustness_gate.py so the draw-robustness, tempfix-promotion and
backbone-config gates are all read by one tool instead of three near-identical forks (the
per-experiment script fork is what CLAUDE.md's rule 2 exists to prevent).

Every arm's margin is computed against the `knative` arm of the SAME condition, so a
condition that changes absolute cost by 10x -- which a binding backbone does -- does not
leak into the comparison.

Refuses to print a table whose arms disagree on INFERENCE_FEATURE_LAYOUT or
warmth_physics: the first is the confound that sat underneath the 2026-08-22 lottery
result, and a scorer that averages over it produces a number that means nothing.

Usage:
  score_live_gate_matrix.py --prefix drawgate --conditions nobackbone,backbone \\
      --arms deployed,prefixctl,tempfix
  score_live_gate_matrix.py --prefix bbrob \\
      --conditions bb_core8_bw1p5,bb_core4_bw0p5 --arms deployed,tempfix
"""

import argparse
import json
import sys
from pathlib import Path

NOISE_FLOOR_PCT = 0.4  # measured run-to-run spread; see PARITY.md


def load(root: Path, prefix: str, cond: str, arm: str, cell: str) -> dict:
    suffix = "knative" if arm == "knative" else "gnn"
    p = root / f"{prefix}_{cond}_{arm}" / "results" / f"{cell}_s0_{suffix}.json"
    if not p.is_file():
        raise SystemExit(f"FAIL LOUD: missing result {p}")
    d = json.loads(p.read_text())
    if not d.get("total_rtt"):
        raise SystemExit(f"FAIL LOUD: {p} has no usable total_rtt")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("simulation_data/normal_sim_sweeps"))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--arms", required=True, help="GNN arms; knative is always the baseline")
    ap.add_argument("--cells", default=None,
                    help="default: every cell present in the first condition's knative arm")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not conds or not arms:
        raise SystemExit("FAIL LOUD: --conditions and --arms must each name at least one")

    if args.cells:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    else:
        kn_dir = args.root / f"{args.prefix}_{conds[0]}_knative" / "results"
        cells = sorted(
            p.name.replace("_s0_knative.json", "")
            for p in kn_dir.glob("*_s0_knative.json")
        )
        if not cells:
            raise SystemExit(f"FAIL LOUD: no knative results under {kn_dir}")

    report = {"prefix": args.prefix, "conditions": {}, "noise_floor_pct": NOISE_FLOOR_PCT}
    layouts, physics = set(), set()

    for cond in conds:
        print(f"\n=== {cond} ===")
        head = f"{'cell':22s} {'knative':>16s}" + "".join(f" {a:>14s}" for a in arms)
        print(head)
        wins = {a: 0 for a in arms}
        margins = {a: [] for a in arms}
        rows = {}

        for cell in cells:
            kn = load(args.root, args.prefix, cond, "knative", cell)
            layouts.add(kn["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT"))
            physics.add(kn["run_provenance"].get("warmth_physics"))
            kn_rtt = kn["total_rtt"]
            line = f"{cell:22s} {kn_rtt:16,.0f}"
            rows[cell] = {"knative": kn_rtt}
            for a in arms:
                d = load(args.root, args.prefix, cond, a, cell)
                layouts.add(d["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT"))
                physics.add(d["run_provenance"].get("warmth_physics"))
                pct = 100.0 * (d["total_rtt"] - kn_rtt) / kn_rtt
                margins[a].append(pct)
                if pct < -NOISE_FLOOR_PCT:
                    wins[a] += 1
                line += f" {pct:+13.1f}%"
                rows[cell][a] = {"total_rtt": d["total_rtt"], "vs_knative_pct": pct}
            print(line)

        print(f"{'mean margin':22s} {'':16s}"
              + "".join(f" {sum(margins[a]) / len(margins[a]):+13.1f}%" for a in arms))
        print(f"{'wins vs knative':22s} {'':16s}"
              + "".join(f" {str(wins[a]) + '/' + str(len(cells)):>14s}" for a in arms))

        report["conditions"][cond] = {
            "cells": rows,
            "wins_vs_knative": wins,
            "n_cells": len(cells),
            "mean_margin_pct": {a: sum(margins[a]) / len(margins[a]) for a in arms},
        }

    if len(layouts) > 1:
        raise SystemExit(f"FAIL LOUD: arms mix INFERENCE_FEATURE_LAYOUT {layouts}")
    if len(physics) > 1:
        raise SystemExit(f"FAIL LOUD: arms mix warmth_physics {physics}")
    report["inference_feature_layout"] = layouts.pop() if layouts else None
    report["warmth_physics"] = physics.pop() if physics else None
    print(f"\nall arms served layout={report['inference_feature_layout']!r} "
          f"physics={report['warmth_physics']!r}")

    print("\n=== summary ===")
    for a in arms:
        per = ", ".join(
            f"{c}: {report['conditions'][c]['mean_margin_pct'][a]:+.1f}% "
            f"({report['conditions'][c]['wins_vs_knative'][a]}/{len(cells)})"
            for c in conds
        )
        print(f"  {a:12s} {per}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score the backbone draw-robustness gate: regime-level win, or draw-level luck?

Reads the drawgate_{cond}_{arm} sweeps and prints, per cell, each GNN arm's total_rtt and
its margin vs Knative in the SAME condition. The question this answers is not "is the
deployed checkpoint good" (already known) but "do draws that LOSE without a backbone win
with one" -- i.e. whether the binding-bandwidth regime, not the training draw, is what
produces the GNN's advantage.

Fails loud rather than reporting a partial table: a missing cell, a mixed warmth_physics,
or an arm serving a different INFERENCE_FEATURE_LAYOUT than the others would each make the
comparison mean something other than it appears to.
"""

import argparse
import json
import sys
from pathlib import Path

CELLS = [
    "cell01_p25_s9001",
    "cell02_p35_s9002",
    "cell03_p15_s9003",
    "cell04_p50_s9004",
    "cell05_p20_s9005",
]
DEFAULT_GNN_ARMS = ["deployed", "prefixctl", "tempfix"]
NOISE_FLOOR_PCT = 0.4  # measured local run-to-run spread, LINEAGES/PARITY


def result_path(root: Path, cond: str, arm: str, cell: str) -> Path:
    suffix = "knative" if arm == "knative" else "gnn"
    return root / f"drawgate_{cond}_{arm}" / "results" / f"{cell}_s0_{suffix}.json"


def load(root: Path, cond: str, arm: str, cell: str) -> dict:
    p = result_path(root, cond, arm, cell)
    if not p.is_file():
        raise SystemExit(f"FAIL LOUD: missing result {p}")
    with open(p) as fh:
        d = json.load(fh)
    rtt = d.get("total_rtt")
    if not rtt:
        raise SystemExit(f"FAIL LOUD: {p} has no usable total_rtt ({rtt!r})")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("simulation_data/normal_sim_sweeps"))
    ap.add_argument("--arms", default=",".join(DEFAULT_GNN_ARMS),
                    help="comma-separated GNN arms to score; use this to report before a "
                         "slower arm has landed (default: %(default)s)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arms:
        raise SystemExit("FAIL LOUD: --arms selected no arms")

    report = {"conditions": {}, "noise_floor_pct": NOISE_FLOOR_PCT}
    layouts, physics = set(), set()

    for cond in ("nobackbone", "backbone"):
        print(f"\n=== {cond} ===")
        header = f"{'cell':22s} {'knative':>16s}"
        for arm in arms:
            header += f" {arm:>16s}"
        print(header)

        cond_rows, wins = {}, {arm: 0 for arm in arms}
        margins = {arm: [] for arm in arms}

        for cell in CELLS:
            kn = load(args.root, cond, "knative", cell)
            layouts.add(kn["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT"))
            physics.add(kn["run_provenance"].get("warmth_physics"))
            kn_rtt = kn["total_rtt"]
            line = f"{cell:22s} {kn_rtt:16,.0f}"
            cell_row = {"knative": kn_rtt}

            for arm in arms:
                d = load(args.root, cond, arm, cell)
                layouts.add(d["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT"))
                physics.add(d["run_provenance"].get("warmth_physics"))
                rtt = d["total_rtt"]
                pct = 100.0 * (rtt - kn_rtt) / kn_rtt
                margins[arm].append(pct)
                if pct < -NOISE_FLOOR_PCT:
                    wins[arm] += 1
                line += f" {pct:+15.1f}%"
                cell_row[arm] = {"total_rtt": rtt, "vs_knative_pct": pct}

            print(line)
            cond_rows[cell] = cell_row

        print(f"{'':22s} {'':16s}", end="")
        for arm in arms:
            mean = sum(margins[arm]) / len(margins[arm])
            print(f" {mean:+15.1f}%", end="")
        print("   <- mean margin")
        print(f"{'':22s} {'wins vs knative:':>16s}", end="")
        for arm in arms:
            print(f" {str(wins[arm]) + '/5':>16s}", end="")
        print()

        report["conditions"][cond] = {
            "cells": cond_rows,
            "wins_vs_knative": wins,
            "mean_margin_pct": {a: sum(margins[a]) / len(margins[a]) for a in arms},
        }

    # Comparability guards -- a table that silently mixes these is not a comparison.
    if len(layouts) > 1:
        raise SystemExit(f"FAIL LOUD: arms mix INFERENCE_FEATURE_LAYOUT {layouts}")
    if len(physics) > 1:
        raise SystemExit(f"FAIL LOUD: arms mix warmth_physics {physics}")
    report["inference_feature_layout"] = layouts.pop() if layouts else None
    report["warmth_physics"] = physics.pop() if physics else None
    print(f"\nall arms served layout={report['inference_feature_layout']!r} "
          f"physics={report['warmth_physics']!r}")

    nb = report["conditions"]["nobackbone"]["wins_vs_knative"]
    bb = report["conditions"]["backbone"]["wins_vs_knative"]
    alts = [a for a in arms if a != "deployed"]
    print("\n=== verdict ===")
    if not alts:
        print("no alternate draws selected — nothing to decide")
        alts = []
    else:
        print(f"alternate draws vs knative: no-backbone "
              f"{'/'.join(str(nb[a]) for a in alts)} -> backbone "
              f"{'/'.join(str(bb[a]) for a in alts)} (of 5 cells each)")
    if alts and all(bb[a] >= 4 for a in alts):
        print("REGIME-LEVEL: every alternate draw wins >=4/5 under the backbone.")
    elif alts and all(bb[a] > nb[a] for a in alts):
        print("PARTIAL: the backbone improves every alternate draw, but not to >=4/5.")
    else:
        print("DRAW-LEVEL: the backbone does not rescue the losing draws.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

`mlp` is a valid arm: the pointwise verification baseline CLAUDE.md keeps a GNN honest
against. It reads from <prefix>_<cond>_mlp/ like any other arm, only under the MLP result
suffix.

Usage:
  score_live_gate_matrix.py --prefix drawgate --conditions nobackbone,backbone \\
      --arms deployed,prefixctl,tempfix,mlp
  score_live_gate_matrix.py --prefix bbrob \\
      --conditions bb_core8_bw1p5,bb_core4_bw0p5 --arms deployed,tempfix,mlp
"""

import argparse
import json
import sys
from pathlib import Path

NOISE_FLOOR_PCT = 0.4  # measured run-to-run spread; see PARITY.md

# Result-file suffix per arm. Anything not listed is a GNN draw, which is the common case:
# an arm name like `deployed` or `tempfix` names a checkpoint, not a policy. `mlp` is the
# pointwise verification baseline and run_full_corpus_siv1_live_gate.sh writes it as
# `<cell>_s0_mlp_dim22.json`.
#
# `mlptempfix` is the SAME policy under a different checkpoint (the corrected batch cache),
# so it shares the `mlp_dim22` suffix. The runner names the result file from the policy and
# not from the checkpoint, which means the two MLP arms are only kept apart by their sweep
# dirs -- <prefix>_<cond>_mlp/ vs <prefix>_<cond>_mlptempfix/. Pointing a second MLP arm at
# the first one's SWEEP_DIR silently overwrites it; see mlp_tempfix_arm_all_gates.sbatch.
#
# `mlpcandrel` / `mlpcandreltf` are the P5b candidate-relative arms (program_verdict_v1):
# the same policy again, under a dim25cr checkpoint, so they share the suffix too and are
# likewise kept apart only by their sweep dirs.
ARM_SUFFIX = {
    "knative": "knative",
    "mlp": "mlp_dim22",
    "mlptempfix": "mlp_dim22",
    "mlpcandrel": "mlp_dim22",
    "mlpcandreltf": "mlp_dim22",
}


def load(root: Path, prefix: str, cond: str, arm: str, cell: str) -> dict:
    suffix = ARM_SUFFIX.get(arm, "gnn")
    p = root / f"{prefix}_{cond}_{arm}" / "results" / f"{cell}_s0_{suffix}.json"
    if not p.is_file():
        raise SystemExit(f"FAIL LOUD: missing result {p}")
    d = json.loads(p.read_text())
    if not d.get("total_rtt"):
        raise SystemExit(f"FAIL LOUD: {p} has no usable total_rtt")
    return d


def _parse_expect_layouts(spec):
    """`arm=layout,...` → {arm: layout}, or None when the flag was not given."""
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"FAIL LOUD: --expect-layouts entry {item!r} is not arm=layout")
        arm, layout = item.split("=", 1)
        out[arm.strip()] = layout.strip().lower()
    if not out:
        raise SystemExit("FAIL LOUD: --expect-layouts was empty")
    return out


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
    ap.add_argument(
        "--expect-layouts",
        default=None,
        help=(
            "arm=layout,... — declare the INFERENCE_FEATURE_LAYOUT each arm was served "
            "under, for the case where the difference IS the intervention (P5b's "
            "candidate-relative arms serve dim25cr against dim22 baselines). Every arm "
            "must be declared and must match what run_provenance recorded. Omit this and "
            "the arms must all agree, which is the right default."
        ),
    )
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

    expect_layouts = _parse_expect_layouts(args.expect_layouts)
    if expect_layouts is not None:
        undeclared = sorted(set(["knative"] + arms) - set(expect_layouts))
        if undeclared:
            raise SystemExit(
                f"FAIL LOUD: --expect-layouts must declare every arm; missing {undeclared}. "
                f"Declaring only some arms would let an unintended layout through under the "
                f"cover of an intended one."
            )

    report = {"prefix": args.prefix, "conditions": {}, "noise_floor_pct": NOISE_FLOOR_PCT}
    layouts_by_arm: dict = {}
    physics = set()

    for cond in conds:
        print(f"\n=== {cond} ===")
        head = f"{'cell':22s} {'knative':>16s}" + "".join(f" {a:>14s}" for a in arms)
        print(head)
        wins = {a: 0 for a in arms}
        margins = {a: [] for a in arms}
        rows = {}

        for cell in cells:
            kn = load(args.root, args.prefix, cond, "knative", cell)
            layouts_by_arm.setdefault("knative", set()).add(
                kn["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT")
            )
            physics.add(kn["run_provenance"].get("warmth_physics"))
            kn_rtt = kn["total_rtt"]
            line = f"{cell:22s} {kn_rtt:16,.0f}"
            rows[cell] = {"knative": kn_rtt}
            for a in arms:
                d = load(args.root, args.prefix, cond, a, cell)
                layouts_by_arm.setdefault(a, set()).add(
                    d["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT")
                )
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

    # An arm that is internally inconsistent is always a bug, declared or not.
    for arm, seen in sorted(layouts_by_arm.items()):
        if len(seen) > 1:
            raise SystemExit(
                f"FAIL LOUD: arm {arm!r} mixes INFERENCE_FEATURE_LAYOUT {seen} across its own cells"
            )
    observed = {arm: next(iter(seen)) for arm, seen in layouts_by_arm.items()}

    if expect_layouts is None:
        distinct = set(observed.values())
        if len(distinct) > 1:
            raise SystemExit(
                f"FAIL LOUD: arms mix INFERENCE_FEATURE_LAYOUT {observed}. If the difference "
                f"is the intervention rather than an accident, declare it: "
                f"--expect-layouts "
                + ",".join(f"{a}={l}" for a, l in sorted(observed.items()))
            )
        report["inference_feature_layout"] = distinct.pop() if distinct else None
        print(f"\nall arms served layout={report['inference_feature_layout']!r} ", end="")
    else:
        wrong = {
            arm: (observed[arm], expect_layouts[arm])
            for arm in observed
            if str(observed[arm] or "").lower() != expect_layouts[arm]
        }
        if wrong:
            raise SystemExit(
                "FAIL LOUD: served layout does not match the declaration "
                + "; ".join(f"{a}: served {s!r}, declared {d!r}" for a, (s, d) in sorted(wrong.items()))
            )
        report["inference_feature_layout"] = observed
        report["inference_feature_layout_declared"] = expect_layouts
        print("\nlayouts served as declared: "
              + ", ".join(f"{a}={observed[a]}" for a in sorted(observed)) + " ", end="")

    if len(physics) > 1:
        raise SystemExit(f"FAIL LOUD: arms mix warmth_physics {physics}")
    report["warmth_physics"] = physics.pop() if physics else None
    print(f"physics={report['warmth_physics']!r}")

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

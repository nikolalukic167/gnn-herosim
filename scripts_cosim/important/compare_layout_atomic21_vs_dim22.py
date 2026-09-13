"""Measure the live effect of INFERENCE_FEATURE_LAYOUT on a siv1 dim14 checkpoint.

The prefixctl and tempfix live gates served `atomic21` (their sidecars declare no layout
and the runner does not export the variable, so load_gnn_model's default applied), while
every deployed-checkpoint gate served `dim22` from its sidecar. This compares two sweeps
of the SAME checkpoint on the SAME cells, trace and venue, differing only in that layout.

If the delta is large relative to the 0.1-0.4% simulation noise floor, then the
2026-08-22 "training-draw lottery" table -- deployed@dim22 vs prefixctl@atomic21 -- was
measuring a serving difference on top of training nondeterminism.

Usage:
  compare_layout_atomic21_vs_dim22.py --old <sweep_dir> --new <sweep_dir>
"""

import argparse
import json
import sys
from pathlib import Path


def layout_of(d: dict):
    return d["run_provenance"]["env"].get("INFERENCE_FEATURE_LAYOUT")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True, help="sweep dir served as atomic21")
    ap.add_argument("--new", type=Path, required=True, help="sweep dir served as dim22")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    old_results = sorted((args.old / "results").glob("*_gnn.json"))
    old_results = [p for p in old_results if not p.name.endswith(".decode_stats.json")]
    if not old_results:
        raise SystemExit(f"FAIL LOUD: no *_gnn.json under {args.old}/results")

    print(f"{'cell':24s} {'old':>16s} {'new':>16s} {'delta':>9s}   layouts")
    rows, deltas = {}, []
    old_layouts, new_layouts = set(), set()

    for op in old_results:
        np_ = args.new / "results" / op.name
        if not np_.is_file():
            raise SystemExit(f"FAIL LOUD: no counterpart for {op.name} under {args.new}")
        o = json.loads(op.read_text())
        n = json.loads(np_.read_text())
        ol, nl = layout_of(o), layout_of(n)
        old_layouts.add(ol)
        new_layouts.add(nl)
        o_rtt, n_rtt = o["total_rtt"], n["total_rtt"]
        delta = 100.0 * (n_rtt - o_rtt) / o_rtt
        deltas.append(delta)
        cell = op.name.replace("_s0_gnn.json", "")
        rows[cell] = {"old_total_rtt": o_rtt, "new_total_rtt": n_rtt,
                      "delta_pct": delta, "old_layout": ol, "new_layout": nl}
        print(f"{cell:24s} {o_rtt:16,.0f} {n_rtt:16,.0f} {delta:+8.2f}%   {ol} -> {nl}")

    if len(old_layouts) > 1 or len(new_layouts) > 1:
        raise SystemExit(
            f"FAIL LOUD: a sweep mixes layouts (old={old_layouts}, new={new_layouts})"
        )
    if old_layouts == new_layouts:
        raise SystemExit(
            f"FAIL LOUD: both sweeps served the same layout {old_layouts} — "
            "this comparison would measure nothing"
        )

    worst = max(abs(d) for d in deltas)
    mean = sum(deltas) / len(deltas)
    print(f"\nmean delta {mean:+.2f}%, worst |delta| {worst:.2f}% "
          f"(simulation noise floor is 0.1-0.4%)")
    if worst <= 0.4:
        print("VERDICT: layout is INERT here — within the noise floor on every cell.")
    else:
        print(f"VERDICT: layout MATTERS — up to {worst:.1f}% of total_rtt, "
              f"{worst / 0.4:.0f}x the noise floor. Any comparison of a "
              "sidecar-declared checkpoint against a sidecar-silent one is confounded.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"cells": rows, "mean_delta_pct": mean, "worst_abs_delta_pct": worst,
             "old_layout": old_layouts.pop(), "new_layout": new_layouts.pop()},
            indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

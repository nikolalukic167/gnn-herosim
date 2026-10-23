#!/usr/bin/env python3
"""live_link_features_v1: the per-arm decision-charge w of the four v7 kinds from their calibration series (cot_link_steps.sh: one exclusive SMT-off node, k = 0, 19461 g0,
both rungs, seed 1, PAR 1, 1 thread), the H3 drift stop of that node, and the node-vs-node H3 check against the filed node-B series.

  v7_freeze_w.py --series <dir holding C_<tag>_B, H3pre_B, H3end_B> --filed-h3 <filed series dir holding H3pre_B, H3end_B> [--out table.json]

Per kind (tags gnn twin mlp wide) and rung: w = 1000 x sum(decision seconds) / sum(plan evals) pooled over the kind's cells (uncut_charge_share.pooled_w), the same per served-batch bucket;
the decision seconds of a v7 call include its link capture, and the capture seconds are reported beside it (decisionTiming.link_capture_s_*). The H3 control (cd_ib0, no exchange
switches) is read at the start and end of the series: STOP if they differ by more than 2 % at a rung (the drift stop); and against the filed node-B H3 of the series that froze the v6 kinds: the
series is VOID if either H3 differs from the filed one by more than 2 % at a rung (the expert's ruling for a different node, 2026-10-23)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uncut_charge_share as U  # noqa: E402

KINDS = {"gnn": "ra_gnn_eng_combfl_link_nosplit", "twin": "ra_twin_eng_combfl_link_nosplit", "mlp": "ra_mlp_same_combfl_link_nosplit", "wide": "ra_mlp_wide_combfl_link_nosplit"}
RUNGS = ("moderate", "heavy")
LIMIT = U.DRIFT


def kind_row(series, tag, arm):
    cs = U.cells(os.path.join(series, f"C_{tag}_B"), arm)
    pooled = U.pooled_w(cs)
    if sorted(pooled) != sorted(RUNGS) or any(sum(1 for s in cs if s["_rung"] == r) != 1 for r in RUNGS):
        raise SystemExit(f"FAIL LOUD: {arm}: want exactly one cell per rung, got {[s['_rung'] for s in cs]}")
    row = {"n_cells": len(cs), "w": {}, "by_bucket": {}, "link_capture": {}, "host": sorted({(s["host"]["cpu_model"], s["host"]["smt_control"]) for s in cs})}
    for r in RUNGS:
        p = pooled[r]
        if any(p[b] is None for b in ("all",) + U.BUCKETS):
            raise SystemExit(f"FAIL LOUD: {arm} {r}: a size bucket has no plan evaluations: {p}")
        row["w"][r] = round(p["all"], 6)
        row["by_bucket"][r] = {b: round(p[b], 6) for b in U.BUCKETS}
        d = next(s for s in cs if s["_rung"] == r)["decisionTiming"]
        row["link_capture"][r] = {"calls": d["link_capture_calls"], "total_s": round(d["link_capture_s_total"], 6), "mean_s_per_call": round(d["link_capture_s_per_call_mean"], 9),
                                  "max_s_per_call": round(d["link_capture_s_per_call_max"], 6), "share_of_decision_seconds": round(d["link_capture_s_total"] / d["total_s"], 6)}
    return row


def h3(series_dir):
    return U.pooled_w(U.cells(series_dir, "cd_ib0"))


def rel(a, b):
    return abs(a - b) / b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--filed-h3", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    pre, end = h3(os.path.join(a.series, "H3pre_B")), h3(os.path.join(a.series, "H3end_B"))
    fpre, fend = h3(os.path.join(a.filed_h3, "H3pre_B")), h3(os.path.join(a.filed_h3, "H3end_B"))
    drift = {r: rel(end[r]["all"], pre[r]["all"]) for r in RUNGS}
    vs_filed = {r: {"pre": rel(pre[r]["all"], fpre[r]["all"]), "end": rel(end[r]["all"], fend[r]["all"])} for r in RUNGS}
    out = {"h3": {"pre": {r: pre[r]["all"] for r in RUNGS}, "end": {r: end[r]["all"] for r in RUNGS}, "filed_pre": {r: fpre[r]["all"] for r in RUNGS},
                  "filed_end": {r: fend[r]["all"] for r in RUNGS}, "drift_start_to_end": drift, "vs_filed_node_B": vs_filed},
           "drift_stop_fired": any(v > LIMIT for v in drift.values()),
           "void_vs_filed_node_B": any(v > LIMIT for r in RUNGS for v in vs_filed[r].values())}
    if out["drift_stop_fired"] or out["void_vs_filed_node_B"]:
        print(json.dumps(out, indent=1))
        print("STOP: the H3 control " + ("drifted over the series" if out["drift_stop_fired"] else "differs from the filed node B") + f" by more than {LIMIT:.0%}; the series is void")
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)
        return 1
    out["kinds"] = {arm: kind_row(a.series, tag, arm) for tag, arm in KINDS.items()}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

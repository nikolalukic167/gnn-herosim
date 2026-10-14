#!/usr/bin/env python3
"""uncut_groups_s0_v1 INTERIM: write a PROVISIONAL-W into this branch's decision_cost_w.json (rp/uncut-interim only, never the pin chain).

  uncut_interim_w.py --tools <checkout at 557d84bf> --cells <calibration dir> --arm ra_twin_eng_arm1b_nosplit --job <id>

The calibration is S4's recipe on the interim venue (k = 0, 19461 g0, both rungs, checkpoint seed 1, PAR = 1, one torch thread). The rates
are uncut_charge_share.pooled_w at 557d84bf (imported): per rung the pooled ms/plan (w_ms_per_plan) and the measured ms/plan per served-batch
bucket (w_ms_per_plan_by_bucket, which the gate charges); a bucket with no calibration plan evals takes the rung's pooled rate, and is named
in the note. Every cell must carry the gate's host block, in one CPU model and SMT state. Existing keys are never changed.
"""
import argparse
import json
import os
import sys

W_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_cost_w.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--w-file", default=W_FILE)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(a.tools, "scripts_cosim"))
    import uncut_charge_share as C

    cs = C.cells(a.cells, a.arm)
    if {s["_rung"] for s in cs} != {"moderate", "heavy"} or len(cs) != 2:
        raise SystemExit(f"FAIL LOUD: want one moderate and one heavy cell of {a.arm}, got {[s['_path'] for s in cs]}")
    h = cs[0]["host"]
    w = C.pooled_w(cs)
    f = json.load(open(a.w_file))
    if a.arm in f["w_ms_per_plan"] or a.arm in f.get("w_ms_per_plan_by_bucket", {}):
        raise SystemExit(f"FAIL LOUD: {a.arm} already has a w in {a.w_file}")
    filled = [f"{r}/{b}" for r in w for b in C.BUCKETS if w[r][b] is None]
    f["w_ms_per_plan"][a.arm] = {r: round(w[r]["all"], 6) for r in ("moderate", "heavy")}
    f.setdefault("w_ms_per_plan_by_bucket", {})[a.arm] = {
        r: {b: round(w[r][b] if w[r][b] is not None else w[r]["all"], 6) for b in C.BUCKETS} for r in ("moderate", "heavy")}
    f[f"_source_interim_{a.arm}"] = (
        f"PROVISIONAL-W, INTERIM only (70022963 / 190e5771); never on rp/uncut-arm1-pin, never in a w-only pin. Job {a.job}, S4's recipe "
        f"(k = 0, 19461 g0, both rungs, seed 1, PAR = 1, 1 torch thread) on {h.get('hostname')} ({h.get('cpu_model')}, SMT {h.get('smt_control')}, "
        f"cpuset {h.get('cpuset')}, nproc {h.get('nproc')}, physical cores {h.get('physical_cores')}). Cells {a.cells}. "
        f"Rates: uncut_charge_share.pooled_w at 557d84bf; by bucket = measured ms/plan per bucket"
        + (f"; buckets with no calibration evals take the rung's pooled rate: {', '.join(filled)}" if filled else "") + ".")
    with open(a.w_file, "w") as fh:
        fh.write(json.dumps(f, indent=1) + "\n")
    print(json.dumps({"arm": a.arm, "w": f["w_ms_per_plan"][a.arm], "by_bucket": f["w_ms_per_plan_by_bucket"][a.arm], "filled": filled,
                      "host": h}, indent=1))


if __name__ == "__main__":
    main()

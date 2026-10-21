#!/usr/bin/env python3
"""Freeze the decision-charge w of the two live-search arms from their calibration series (cot_ils_steps.sh: one exclusive SMT-off node per arm, k = 0, 19461 g0,
both rungs, seed 0). Pooled per rung and size bucket: w = 1000 x sum(seconds) / sum(plans).
  exact-live  cd_exactS_ib0_live: decisionTiming.by_bucket total_s / plan_evals  -> w_ms_per_plan_by_bucket (scalar w_ms_per_plan = le10, as for every hand arm)
  ILS-live    cd_ib0_ils: the ILS phase's own wall / plans per bucket (pg_ils_wall_<b> / pg_ils_evals_<b>) -> w_S_ms_per_plan_by_bucket; the bar's plans keep the
              bar's filed w (w_ms_per_plan[cd_ib0_ils] = cd_ib0_inflightfull's, unchanged). The ILS never runs on a batch of at most 10 tasks, so le10 has no
              evals; it is set to the 11_50 value and never charges.
usage: ils_live_freeze_w.py [--exact <ser_cd_exactS_ib0_live>] [--ils <ser_cd_ib0_ils_p50>] [--write scripts_cosim/decision_cost_w.json]   (series dir holds ARM/)"""
import glob
import json
import sys

BUCKETS = ("le10", "11_50", "gt50")
RUNGS = ("moderate", "heavy")


def cells(series):
    out = {r: [] for r in RUNGS}
    for f in sorted(glob.glob(f"{series}/ARM/*.summary.json")):
        s = json.load(open(f))
        r = "heavy" if "g0heavy" in f else "moderate" if "g0moderate" in f else None
        if r is None:
            raise SystemExit(f"FAIL LOUD: unexpected cell {f}")
        out[r].append(s)
    for r in RUNGS:
        if len(out[r]) != 1:
            raise SystemExit(f"FAIL LOUD: {series}: {len(out[r])} cells at {r}, want 1")
    return out


def exact_w(series):
    res = {}
    for r, cs in cells(series).items():
        acc = {b: [0.0, 0] for b in BUCKETS}
        for s in cs:
            for b in BUCKETS:
                acc[b][0] += s["decisionTiming"]["by_bucket"][b]["total_s"]
                acc[b][1] += s["decisionTiming"]["by_bucket"][b]["plan_evals"]
        if any(n == 0 for _t, n in acc.values()):
            raise SystemExit(f"FAIL LOUD: exact-live {r}: a size bucket has no plan evaluations: {acc}")
        res[r] = {b: round(1000.0 * t / n, 6) for b, (t, n) in acc.items()}
    return res


def ils_w(series):
    res = {}
    for r, cs in cells(series).items():
        acc = {b: [0.0, 0] for b in ("11_50", "gt50")}
        for s in cs:
            c = s["schedulerCounters"]
            for b in acc:
                acc[b][0] += float(c.get(f"pg_ils_wall_{b}", 0.0))
                acc[b][1] += int(c.get(f"pg_ils_evals_{b}", 0))
        if any(n == 0 for _t, n in acc.values()):
            raise SystemExit(f"FAIL LOUD: ILS-live {r}: an ILS size bucket has no plans: {acc}")
        w = {b: round(1000.0 * t / n, 6) for b, (t, n) in acc.items()}
        res[r] = {"le10": w["11_50"], **w}
    return res


def main(argv):
    """--exact <series dir> and/or --ils <series dir> [--write decision_cost_w.json]. --ils takes the PER 50 series (A2: w_S = the PER 50 value) and sets it for every
    ILS budget kind (cd_ib0_ils, cd_ib0_ils_p50, cd_ib0_ils_p10), which all charge the same w_S."""
    arg = lambda k: argv[argv.index(k) + 1] if k in argv else None
    tab = {}
    if arg("--exact"):
        tab["exact"] = exact_w(arg("--exact"))
    if arg("--ils"):
        tab["ils"] = ils_w(arg("--ils"))
    print(json.dumps(tab, indent=1))
    p = arg("--write")
    if not p:
        return
    d = json.load(open(p))
    if "exact" in tab:
        d["w_ms_per_plan_by_bucket"]["cd_exactS_ib0_live"] = tab["exact"]
        d["w_ms_per_plan"]["cd_exactS_ib0_live"] = {r: tab["exact"][r]["le10"] for r in RUNGS}
        d["_source_exact_live_w"] = ("FROZEN (calibration series ils_cal/ser_cd_exactS_ib0_live, the filed hand recipe on one exclusive SMT-off node: k = 0, 19461 g0, both rungs, seed 0, "
                                     "H3 drift references cd_ib0 at 114cc0e7, drift -0.25 / -0.03 %). w_ms_per_plan_by_bucket[cd_exactS_ib0_live] = decisionTiming wall per plan, pooled; "
                                     "its scalar is the le10 bucket.")
    if "ils" in tab:
        for k in ("cd_ib0_ils", "cd_ib0_ils_p50", "cd_ib0_ils_p10"):
            d["w_S_ms_per_plan_by_bucket"][k] = tab["ils"]
        d["_source_ils_w_S"] = ("FROZEN (A2): w_S_ms_per_plan_by_bucket = the ILS phase's wall per _pg_plan_cost evaluation measured at PER = 50 (calibration series "
                                "ils_cal/ser_cd_ib0_ils_p50, the filed hand recipe, exclusive SMT-off node, k = 0, 19461 g0, both rungs, seed 0, H3 references at 114cc0e7); le10 = the 11_50 "
                                "value, never charged (the ILS runs on batches of more than 10 tasks only). The bar's plans before the ILS keep the bar's filed w (w_ms_per_plan).")
    if "ils" in tab:
        d.pop("_source_ils_exact_live", None)
    else:
        d["_source_ils_exact_live"] = ("PROVISIONAL, NOT FROZEN (2026-10-21): w_S_ms_per_plan_by_bucket (every cd_ib0_ils budget kind) copies cd_exactS_ib0's bucketed whole-plan w until A2's "
                                       "PER 10 / PER 50 calibration is filed; exact-live's w is frozen (_source_exact_live_w). The ILS cells serve only after w_S is frozen.")
    open(p, "w").write(json.dumps(d, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)

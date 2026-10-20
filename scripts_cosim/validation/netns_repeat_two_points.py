#!/usr/bin/env python3
"""Repeat two missed shaped-link points with 10 reps, on the filed harness unchanged (netns_transfer_check.py, md5 b6538092c8ce0d5d526399feea5bc969, imported not edited).
usage (root): netns_repeat_two_points.py --cfg <cc40s19537.json> --out <new json> [--reps 10]
Points (S6 / SCIENTIST ruling): (a) cellular_to_wired 250 MB n=4; (b) wired_to_cellular 50 MB n=4. Pass rule fixed before the repeat: the median of the 10 makespans is
within 10 % of the fair-share-with-TCP-start-up model: (a) 253.44 s -> [228.1, 278.8] s, (b) 3.75 s -> [3.38, 4.13] s. Nothing else is re-measured."""
import argparse, importlib.util, json, os, statistics, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
H = "b6538092c8ce0d5d526399feea5bc969"
spec = importlib.util.spec_from_file_location("netns_transfer_check", f"{HERE}/netns_transfer_check.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
POINTS = (("cellular_to_wired", 250, 4, 253.44, (228.1, 278.8)), ("wired_to_cellular", 50, 4, 3.75, (3.38, 4.13)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    if os.geteuid() != 0 or os.path.exists(a.out): raise SystemExit("needs root and a new --out")
    if h.md5(f"{HERE}/netns_transfer_check.py") != H: raise SystemExit("FAIL LOUD: harness md5")
    meta, cases, dropped = h.charge_cases(a.cfg); cases = {c["case"]: c for c in cases}
    prov = {"harness_md5": H, "driver_md5": h.md5(os.path.abspath(__file__)), "cfg_md5": h.md5(a.cfg), "reps": a.reps,
            "head": h.sh(["git", "-C", h.ROOT, "rev-parse", "HEAD"]).stdout.strip(), "kernel": h.platform.release(), "host": h.platform.node()}
    runs = []; lab = h.Lab()
    def save(table=None): json.dump({"provenance": prov, "points": [p[:4] + (p[4],) for p in POINTS], "runs": runs, "table": table}, open(a.out, "w"), indent=1)
    try:
        lab.up(); prov["sysctls"] = lab.sysctls()
        with tempfile.TemporaryDirectory() as tmp:
            for case, s, n, model, (lo, hi) in POINTS:
                c = cases[case]; lab.shape(c["bottleneck_mbs"], c["latency_s"]); pr = h.predictions(c, s, n)
                for rep in range(a.reps):
                    fl, rt = h.run_point(lab, s, n, tmp, 120 + 3 * pr["fifo_s"][-1]); comp = [f["completion_s"] for f in fl]
                    runs.append({"case": case, "size_mb": s, "flows": n, "rep": rep, "completions_s": comp, "makespan_s": comp[-1], "all_digests_ok": all(f["ok"] for f in fl), "retransmits": rt})
                    print(case, s, "MB x", n, "rep", rep, round(comp[-1], 2), "retrans", rt, flush=True); save()
        table = []
        for case, s, n, model, (lo, hi) in POINTS:
            m = [r["makespan_s"] for r in runs if r["case"] == case]
            med = statistics.median(m)
            table.append({"case": case, "size_mb": s, "flows": n, "reps": len(m), "median_s": med, "min_s": min(m), "max_s": max(m), "model_s": model, "err_pct": 100 * (model / med - 1),
                          "bound_s": [lo, hi], "pass": lo <= med <= hi, "flagged_retransmit_runs": sum(1 for r in runs if r["case"] == case and r["retransmits"] > 0),
                          "digests_ok": all(r["all_digests_ok"] for r in runs if r["case"] == case)})
        save(table)
        for t in table: print(t, flush=True)
    finally:
        lab.down()


main()

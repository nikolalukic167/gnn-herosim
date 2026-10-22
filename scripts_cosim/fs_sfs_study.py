#!/usr/bin/env python3
"""S_fs test (direction only, nothing is trained on it): does a label search whose surrogate prices fair-share find better labels than the blind one, and how much slower is it?

Every replay runs under HEROSIM_EXCHANGE_FAIRSHARE=1 (serial partners) + HEROSIM_EXCHANGE_TCP=1, set by the sbatch. Per state of the seeded sample (fs_label_disagreement.py sample):
  BLIND  today's candidates: the stored ILS on S -> the 8 lowest-S distinct plans + the current label (plan spaces <= 600 or no stored trajectory: the 8 lowest-S plans of the enumeration + the label);
  FS     the ILS re-run with S_fs (same start = the cd_ib0 hand plan, same seed, same budget 200 x n x mean candidates) -> the 8 lowest-S_fs distinct plans + the label;
  EXACT  plan spaces <= 600: every plan replayed, the optimum under fair-share physics is the reference.
The BLIND search is also re-run on S (same seed, budget) so both walls are measured on the same machine; the re-run is checked against the stored trajectory.

  run  : --split SAMPLE_SPLIT --corpus CORPUS --shard K/N --out ROWS.jsonl --workers 4
  read : --rows ROWS.jsonl ... --out READ.json"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import uncut_label_screen as L  # noqa: E402


def study_state(corpus, st):
    import gzip

    import uncut_ils as U

    ds = L._dataset_dir(corpus, st["gid"])
    head, rows = L._stored_rows(ds)
    n = int(head["n_tasks"])
    trace, gids = [int(x) for x in head["trace_task_ids"]], [int(x) for x in head["snapshot_gids_sorted"]]
    label, stored_rtt = L._plan_from_row(rows[0], n), float(rows[0]["rtt"])
    small = bool(st["small"])
    ils_rows = []
    if head.get("evaluated_plans_file"):
        ils_rows = [{"S": r["S"], "eval": r["eval"], "plan": L.ds_order_from_gid_order(r["plan"], trace, gids)}
                    for r in (json.loads(line) for line in gzip.open(f"{ds}/{head['evaluated_plans_file']}", "rt"))]
    job = {"dataset": ds, "sim_input": L.SIM_INPUT}
    if small or not ils_rows:
        job["enum_cap"] = L.ENUM_CAP
    d = U.decision_pass(job, "cd_ib0", 0, 1, 1, None, check_arm=False)             # the hand plan, the enumeration (small), the ILS start
    enum = d["enum"] or []
    blind_plans = L.plan_list(label, False, enum, [] if small else ils_rows)       # top 8 by S + the label (small: the enumeration's 8 lowest S)
    all_plans = L.plan_list(label, True, enum, ils_rows) if small else None
    budget_job = {"dataset": ds, "sim_input": L.SIM_INPUT, "keep_plans": True}
    t = time.time()
    b = U.decision_pass(dict(budget_job), "cd_ib0", 0, None, 200, None, check_arm=False)
    blind_wall_measured = b["search_wall"]
    f = U.decision_pass({**budget_job, "sfs": True}, "cd_ib0", 0, None, 200, None, check_arm=False)
    fs_rows = [{"S": s_, "eval": e, "plan": pl} for e, s_, _it, pl in f["evals_full"]]
    fs_plans = L.plan_list(label, False, (), fs_rows)
    union, seen = [], {}
    for p in (all_plans or []) + blind_plans + fs_plans:
        k = L.plan_key(p["plan"])
        if k not in seen:
            seen[k] = len(union)
            union.append(p)
    rt, _setup, _runs = U.replay_rtts(d["spec"], d["types"], [p["plan"] for p in union])
    if rt[seen[L.plan_key(label)]] <= 0:
        raise SystemExit(f"FAIL LOUD: {st['gid']}: non-positive replayed RTT for the label")
    rtt_of = lambda plans: [rt[seen[L.plan_key(p["plan"])]] for p in plans]
    br, fr = rtt_of(blind_plans), rtt_of(fs_plans)
    bi, fi = min(range(len(br)), key=lambda i: (br[i], i)), min(range(len(fr)), key=lambda i: (fr[i], i))
    out = {"gid": st["gid"], "bucket": st["bucket"], "small": small, "n_tasks": n, "plan_space": int(head["plan_space_size"]),
           "rtt_label": rt[seen[L.plan_key(label)]], "rtt_blind": br[bi], "rtt_fs": fr[fi], "plan_blind": blind_plans[bi]["plan"], "plan_fs": fs_plans[fi]["plan"],
           "blind_cands": len(blind_plans), "fs_cands": len(fs_plans), "cands_shared": len({L.plan_key(p["plan"]) for p in blind_plans} & {L.plan_key(p["plan"]) for p in fs_plans}),
           "blind_wall_s": blind_wall_measured, "fs_wall_s": f["search_wall"], "blind_s_calls": b["s_calls"], "fs_s_calls": f["s_calls"], "budget": f["budget"],
           "blind_start_S": b["start_S"], "blind_best_S": b["best_S"], "fs_start_S": f["start_S"], "fs_best_S": f["best_S"],
           "stored_top_best_S": min(r["S"] for r in ils_rows) if ils_rows else None, "wall_s": time.time() - t}
    if all_plans is not None:
        ar = rtt_of(all_plans)
        out["rtt_exact"] = min(ar)
        out["n_exact_plans"] = len(all_plans)
    return out


def _work(args):
    corpus, st = args
    return study_state(corpus, st)


def cmd_run(a):
    states = L.all_states(a.split, a.corpus)
    i, n = (int(x) for x in a.shard.split("/"))
    mine = states[i::n]
    done = L.read_done(a.out)
    todo = [st for st in mine if st["gid"] not in done]
    import concurrent.futures as cf
    import multiprocessing as mp

    size = a.workers * 2
    t0 = time.time()
    with open(a.out, "a") as fh:
        for lo in range(0, len(todo), size):
            chunk = todo[lo:lo + size]
            pool = cf.ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=L._init_worker)
            fs = [pool.submit(_work, (a.corpus, st)) for st in chunk]
            try:
                for st, f in zip(chunk, fs):
                    try:
                        r = f.result(timeout=a.state_timeout)
                    except Exception as err:
                        raise SystemExit(f"FAIL LOUD: state {st['gid']}: {type(err).__name__}: {err}")
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"{r['gid']} {r['bucket']} blind {r['blind_wall_s']:.0f}s fs {r['fs_wall_s']:.0f}s {time.time() - t0:.0f}s", flush=True)
            finally:
                procs = list((getattr(pool, "_processes", None) or {}).values())
                for f in fs:
                    f.cancel()
                pool.shutdown(wait=False, cancel_futures=True)
                for p in procs:
                    p.terminate()


def _q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(math.ceil(p * len(x))) - 1)] if x else None


def cmd_read(a):
    rows = {}
    for f in a.rows:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                rows[r["gid"]] = r
    ref = {}
    for f in a.ref_rows:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                ref[r["gid"]] = r
    res = {"n": len(rows), "by_bucket": {}}
    if ref:
        bad = []
        for r in rows.values():       # BLIND on S must reproduce the fs-label test: large states the same best plan and RTT, small states the exact optimum
            t = ref[r["gid"]]
            same = (abs(r["rtt_exact"] - t["rtt_sim"]) <= 1e-9 * abs(t["rtt_sim"])) if "rtt_exact" in r else \
                (r["plan_blind"] == t["plans"][0] and abs(r["rtt_blind"] - t["rtt_sim"]) <= 1e-9 * abs(t["rtt_sim"]))
            if not same:
                bad.append(r["gid"])
        res["blind_reproduces_fs_label_test"] = f"{len(rows) - len(bad)} of {len(rows)}"
        res["blind_not_reproduced"] = bad[:10]
    rows = list(rows.values())
    for key, sel in [("ALL", lambda r: True)] + [(b, (lambda b: lambda r: r["bucket"] == b)(b)) for b in L.__dict__.get("BUCKET_NAMES", ("le10", "11_50", "51_100", "101_256"))] + \
            [("small (exact reference)", lambda r: "rtt_exact" in r), ("large (no exact reference)", lambda r: "rtt_exact" not in r)]:
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        gain = [1.0 - r["rtt_fs"] / r["rtt_blind"] for r in R]          # > 0: the FS label is faster than the blind one under fair-share physics
        better = [g for g in gain if g > 1e-9]
        e = {"n": len(R), "share_fs_faster_gt_1e-9": len(better) / len(R), "share_fs_faster_ge_5pct": sum(g >= 0.05 for g in gain) / len(R),
             "share_fs_slower_gt_1e-9": sum(g < -1e-9 for g in gain) / len(R), "median_gain_where_faster": statistics.median(better) if better else None,
             "mean_gain_all": sum(gain) / len(gain),
             "blind_search_wall_s": {"median": statistics.median(r["blind_wall_s"] for r in R), "p90": _q([r["blind_wall_s"] for r in R], 0.9)},
             "fs_search_wall_s": {"median": statistics.median(r["fs_wall_s"] for r in R), "p90": _q([r["fs_wall_s"] for r in R], 0.9)},
             "median_wall_ratio_fs_over_blind": statistics.median(r["fs_wall_s"] / r["blind_wall_s"] for r in R if r["blind_wall_s"] > 0)}
        sm = [r for r in R if "rtt_exact" in r]
        if sm:
            rb = [r["rtt_blind"] / r["rtt_exact"] - 1.0 for r in sm]
            rf = [r["rtt_fs"] / r["rtt_exact"] - 1.0 for r in sm]
            e["exact"] = {"n": len(sm), "regret_blind_mean": sum(rb) / len(rb), "regret_fs_mean": sum(rf) / len(rf),
                          "blind_finds_exact_optimum": sum(x <= 1e-9 for x in rb) / len(rb), "fs_finds_exact_optimum": sum(x <= 1e-9 for x in rf) / len(rf)}
        res["by_bucket"][key] = e
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--shard", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--state-timeout", type=float, default=14400.0)
    q = sub.add_parser("read")
    q.add_argument("--rows", nargs="+", required=True)
    q.add_argument("--ref-rows", nargs="*", default=[], help="the fs-label test's fair-share rows (fs_*_of_4.jsonl): BLIND must reproduce them")
    q.add_argument("--out", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        for k in ("HEROSIM_EXCHANGE_FAIRSHARE", "HEROSIM_EXCHANGE_TCP"):
            if os.environ.get(k) != "1":
                raise SystemExit(f"FAIL LOUD: {k}=1 is required (every replay of the study runs under fair-share + TCP)")
        cmd_run(a)
    else:
        cmd_read(a)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

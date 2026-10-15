#!/usr/bin/env python3
"""W3 pre-check (spec: docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-15 "W3 self-prefix pre-check"). Offline diagnostic, MYOPIC-REPLAY, INTERIM-DESCRIPTIVE.

  uncut_w3_prefix.py draw   --src replay_diag_draw.json --out w3_draw.json      the 257 val groups of 51-100 and 101-256
  uncut_w3_prefix.py points --shards D --decode-dir D --out points.json          per group: n, label plan, topological commit order, each checkpoint's decoded plan, prefix tasks at t = floor(.25 n), floor(.5 n)
  uncut_w3_prefix.py ils    --points points.json --task K --stride S --out rows   ILS completion of each model prefix (prefix frozen, budget x (n - t)/n, start = the model's own plan), replayed
  uncut_w3_prefix.py analyze --rows-dir D --out out.json [--md out.md]            the PASS / FAIL read

Decode, prep, replay and the ILS engine are S5's (`uncut_replay_diag.py`, `uncut_ils.py`), imported unchanged. The only new logic is the freeze (the prefix tasks are added to the
engine's `queued` set, which the engine already pins) and the budget scaling."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts_cosim")):
    if p not in sys.path:
        sys.path.insert(0, p)

RUNS = ("gnn_eng_s1_final", "gnn_eng_s2_final")
FRACS = (0.25, 0.5)
BUCKETS = ("51_100", "101_256")
SEED = 0                    # the engine's ILS seed, one for every run
FIT_A, FIT_K = 0.0072059, 2.0689      # S5 job 874078, 724 runs of groups > 50 tasks: full-budget search wall = FIT_A * n ** FIT_K seconds
WALL_CAP = 3600.0           # safety only; a run it stops is CAPPED and reported


def prefix_len(n: int, frac: float) -> int:
    return int(math.floor(frac * n))


def cmd_draw(a):
    src = json.load(open(a.src))
    val = sorted(g for g in src["val"] if src["meta"][g]["bucket"] in BUCKETS)
    out = {"split_sha256": src["split_sha256"], "source": a.src, "train": [], "val": val, "meta": {g: src["meta"][g] for g in val}}
    json.dump(out, open(a.out, "w"), indent=1)
    from collections import Counter
    print(dict(Counter(f"{src['meta'][g]['bucket']}/{src['meta'][g]['label']}" for g in val)))


def cmd_points(a):
    import pickle
    import uncut_replay_diag as R
    from src.policy.gnn.seq_decode import topological_task_order

    draw = json.load(open(a.draw))
    dec = R.load_decodes(a.decode_dir)
    out: Dict[str, Any] = {}
    for f in sorted(Path(a.shards).glob("g_*.pkl")):
        for g in pickle.load(open(f, "rb")):
            gid, n = str(g.dataset_id), int(g.n_tasks)
            order = [int(x) for x in topological_task_order(n, g.dag_parents)]
            label = [[int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[R.ALPHA][0][t])]] for t in range(n)]
            runs = {}
            for run in RUNS:
                d = dec[gid][run]
                runs[run] = {"decoded": None if d["decoded"] is None else [[int(v) for v in c] for c in d["decoded"]], "agree": d.get("agree"), "exact": d.get("exact"),
                             "points": {str(fr): order[:prefix_len(n, fr)] for fr in FRACS}}
            out[gid] = {"n": n, "bucket": draw["meta"][gid]["bucket"], "label_type": draw["meta"][gid]["label"], "order": order, "label": label, "runs": runs}
    miss = sorted(set(draw["val"]) - set(out))
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} drawn groups are missing from the shards, e.g. {miss[:3]}")
    json.dump(out, open(a.out, "w"))
    print(f"points: {len(out)} groups; decode failed (None): {sum(1 for v in out.values() for r in v['runs'].values() if r['decoded'] is None)}")


def cmd_ils(a):
    import uncut_ils as U
    import uncut_replay_diag as R

    pts = json.load(open(a.points))
    groups = sorted(pts, key=lambda g: (-pts[g]["n"], g))[a.task::a.stride]
    out: Dict[str, Any] = {}
    for gid in groups:
        P = pts[gid]
        n, ds = P["n"], R.dataset_dir(gid)
        head, rows = R._label_rows(ds)
        trace = [int(x) for x in head["trace_task_ids"]]
        if len(trace) != n:
            raise SystemExit(f"FAIL LOUD: {gid} trace {len(trace)} != n {n}")
        stored = {json.dumps([list(map(int, r["placement_plan"][str(i)])) for i in range(n)]): float(r["rtt"]) for r in rows}
        plan_of: Dict[str, List[List[int]]] = {}
        spec = types = None
        recs: List[Dict[str, Any]] = []
        t_group = time.time()

        # the unfrozen default budget of this group (200 x n x mean candidates): one pass whose budget is recorded and set to 1 evaluation
        box: Dict[str, int] = {}
        real_budget = U.default_budget

        def probe_budget(ckeys, per=200):
            box["b0"] = real_budget(ckeys, per)
            return 1
        U.default_budget = probe_budget
        try:
            d0 = U.decision_pass({"dataset": ds, "sim_input": R.SIM_INPUT}, "cd_ib0", SEED, None, 200, WALL_CAP, check_arm=False)
        finally:
            U.default_budget = real_budget
        b0 = box["b0"]
        spec, types = d0["spec"], d0["types"]

        for run in RUNS:
            R_ = P["runs"][run]
            model = R_["decoded"]
            if model is None:
                recs.append({"run": run, "decode_failed": True})
                continue
            plan_of[run] = model
            for fr in FRACS:
                prefix = R_["points"][str(fr)]
                t = len(prefix)
                frozen_idx = set(prefix)
                orig = U._search_at_decision

                def wrapped(sched, batch_tasks, system_state, res, local_of, queued, *args, _model=model, _frozen=frozen_idx, **kw):
                    res2 = dict(res)
                    for i, tk in enumerate(batch_tasks):
                        if int(tk.id) in local_of:
                            res2[i] = tuple(_model[local_of[int(tk.id)]])
                    fz = {int(tk.id) for tk in batch_tasks if int(tk.id) in local_of and local_of[int(tk.id)] in _frozen}
                    return orig(sched, batch_tasks, system_state, res2, local_of, set(queued) | fz, *args, **kw)

                budget = max(1, int(round(b0 * (n - t) / n)))
                t1 = time.time()
                U._search_at_decision = wrapped
                try:
                    d = U.decision_pass({"dataset": ds, "sim_input": R.SIM_INPUT}, "cd_ib0", SEED, budget, 200, WALL_CAP, check_arm=False)
                finally:
                    U._search_at_decision = orig
                start_ok = [list(map(int, x)) for x in d["hand"]] == model
                best = [list(map(int, x)) for x in d["best"]]
                kept = all(best[k] == model[k] for k in prefix)
                key = f"{run}|{fr}"
                plan_of[key] = best
                recs.append({"run": run, "frac": fr, "t": t, "budget": budget, "budget_unfrozen": b0, "n_evals": d["n_evals"], "stopped": d["stopped"], "capped": d["stopped"] == "seconds",
                             "search_wall": d["search_wall"], "fit_wall": FIT_A * n ** FIT_K * (n - t) / n, "wall": time.time() - t1, "start_S": d["start_S"], "best_S": d["best_S"], "n_pinned": d["n_pinned"],
                             "start_is_model": start_ok, "prefix_kept": kept, "changed_tasks": sum(1 for k in range(n) if best[k] != model[k]), "plan_key": key})
                recs[-1]["over2x"] = recs[-1]["search_wall"] > 2 * recs[-1]["fit_wall"]
                if recs[-1]["over2x"]:
                    print(f"OVER2X {gid} {key} n={n} search_wall={recs[-1]['search_wall']:.0f}s fit={recs[-1]['fit_wall']:.0f}s", flush=True)
                if not (start_ok and kept):
                    raise SystemExit(f"FAIL LOUD: {gid} {key}: start_is_model={start_ok} prefix_kept={kept}")

        names = ["label"] + list(plan_of)
        rt, _s, _r = U.replay_rtts(spec, types, [P["label"]] + [plan_of[k] for k in plan_of])
        rtt = dict(zip(names, rt))
        lk = json.dumps(P["label"])
        out[gid] = {"n": n, "bucket": P["bucket"], "label_type": P["label_type"], "rtt": rtt, "label_in_stored_rows": lk in stored, "stored_label_rtt": stored.get(lk),
                    "records": recs, "group_wall": time.time() - t_group}
        json.dump(out, open(a.out, "w"))
    print(f"ils: {len(out)} groups")


# ---- analysis (pure python) ------------------------------------------------------------------------------------------------------------------

def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _boot(groups: Sequence[Sequence[float]], stat, reps: int = 2000, seed: int = 20261015) -> List[float]:
    """95 % group-bootstrap interval of stat(pooled points): resample groups with replacement, keep each group's points together."""
    rng = random.Random(seed)
    k = len(groups)
    vals = []
    for _ in range(reps):
        pool = [x for j in (rng.randrange(k) for _ in range(k)) for x in groups[j]]
        vals.append(stat(pool))
    vals.sort()
    return [vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]]


def gains_of(rows: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for gid, r in rows.items():
        for rec in r["records"]:
            if rec.get("decode_failed"):
                continue
            rm, rc = r["rtt"][rec["run"]], r["rtt"][rec["plan_key"]]
            g = (rm - rc) / rm
            sb = rec["best_S"] < rec["start_S"] - 1e-9
            out.append({"group": gid, "bucket": r["bucket"], "label_type": r["label_type"], "run": rec["run"], "frac": rec["frac"], "gain": g, "s_better": sb,
                        "s_better_replay_worse": sb and g < 0, "s_better_replay_ge5_worse": sb and g <= -0.05, "changed": rec["changed_tasks"] > 0})
    return out


def summarize(pts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not pts:
        return {"n_points": 0}
    by_g: Dict[str, List[Dict[str, Any]]] = {}
    for p in pts:
        by_g.setdefault(p["group"], []).append(p)
    gl = [[p["gain"] for p in v] for v in by_g.values()]
    ge = [[1.0 if p["gain"] >= 0.05 else 0.0 for p in v] for v in by_g.values()]
    le = [[1.0 if p["gain"] <= -0.05 else 0.0 for p in v] for v in by_g.values()]
    allg = [p["gain"] for p in pts]
    mean = lambda xs: sum(xs) / len(xs)
    share_ge, share_le = mean([x for v in ge for x in v]), mean([x for v in le for x in v])

    def diff_stat(pool):
        return mean(pool)
    # the difference share_ge - share_le, bootstrapped on groups
    dif = [[a - b for a, b in zip(g1, g2)] for g1, g2 in zip(ge, le)]
    sb = [p for p in pts if p["s_better"]]
    return {"n_points": len(pts), "n_groups": len(by_g), "median_gain": _median(allg), "median_gain_ci": _boot(gl, _median),
            "mean_gain": mean(allg), "mean_gain_ci": _boot(gl, mean),
            "share_gain_ge5": share_ge, "share_gain_ge5_ci": _boot(ge, mean), "share_gain_le_m5": share_le, "share_gain_le_m5_ci": _boot(le, mean),
            "share_diff": share_ge - share_le, "share_diff_ci": _boot(dif, diff_stat),
            "ils_s_better": len(sb), "ils_s_better_replay_worse": sum(1 for p in sb if p["gain"] < 0), "ils_s_better_replay_worse_ge5": sum(1 for p in sb if p["gain"] <= -0.05),
            "ils_s_better_replay_better": sum(1 for p in sb if p["gain"] > 0), "ils_unchanged_plan": sum(1 for p in pts if not p["changed"])}


def cmd_analyze(a):
    rows: Dict[str, Any] = {}
    for f in sorted(Path(a.rows_dir).glob("i_*.json")):
        rows.update(json.load(open(f)))
    pts = gains_of(rows)
    # integrity
    id_err = [abs(r["rtt"]["label"] - r["stored_label_rtt"]) / max(1.0, abs(r["stored_label_rtt"])) for r in rows.values() if r["label_in_stored_rows"]]
    capped = sum(1 for r in rows.values() for rec in r["records"] if rec.get("capped"))
    failed = sum(1 for r in rows.values() for rec in r["records"] if rec.get("decode_failed"))
    res: Dict[str, Any] = {"groups": len(rows), "points": len(pts), "decode_failed_runs": failed, "capped_runs": capped,
                           "label_replay_identity": {"checked": len(id_err), "max_rel_err": max(id_err) if id_err else None, "fail_1e-9": sum(1 for e in id_err if e > 1e-9)}, "cells": {}}
    classes = {"pooled": lambda p: True, "ils_labelled": lambda p: p["label_type"] == "ils", "hand_fallback": lambda p: p["label_type"] == "hand"}
    for b in BUCKETS:
        for cname, cf in classes.items():
            for run in (None,) + RUNS:
                for fr in (None,) + FRACS:
                    sel = [p for p in pts if p["bucket"] == b and cf(p) and (run is None or p["run"] == run) and (fr is None or p["frac"] == fr)]
                    res["cells"][f"{b}|{cname}|{run or 'both_seeds'}|{fr if fr is not None else 'both_points'}"] = summarize(sel)
    verdict = {}
    for b in BUCKETS:
        c = res["cells"][f"{b}|pooled|both_seeds|both_points"]
        if c["n_points"]:
            verdict[b] = {"share_ge5": c["share_gain_ge5"], "share_le_m5": c["share_gain_le_m5"], "i_share_ge_30pct": c["share_gain_ge5"] >= 0.30,
                          "ii_diff_ge_15pp": c["share_diff"] >= 0.15, "both": c["share_gain_ge5"] >= 0.30 and c["share_diff"] >= 0.15}
    res["verdict_by_bucket"] = verdict
    res["PASS"] = any(v["both"] for v in verdict.values())
    json.dump(res, open(a.out, "w"), indent=1)
    if a.md:
        L = ["# W3 self-prefix pre-check (offline, MYOPIC-REPLAY, INTERIM-DESCRIPTIVE)", "", f"PASS: {res['PASS']}  groups {res['groups']}  points {res['points']}  decode-failed runs {failed}  capped runs {capped}",
             f"label replay identity: {res['label_replay_identity']}", "", "| bucket | labels | seed | point | n pts | share >=5% [CI] | share <=-5% [CI] | diff [CI] | median gain [CI] | mean gain [CI] | ILS S-better / replay-worse |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        f = lambda x, ci=None: f"{100 * x:.1f}%" + (f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else "")
        for k, c in res["cells"].items():
            if not c["n_points"]:
                continue
            b, cn, run, fr = k.split("|")
            L.append(f"| {b} | {cn} | {run} | {fr} | {c['n_points']} | {f(c['share_gain_ge5'], c['share_gain_ge5_ci'])} | {f(c['share_gain_le_m5'], c['share_gain_le_m5_ci'])} | {f(c['share_diff'], c['share_diff_ci'])} | "
                     f"{f(c['median_gain'], c['median_gain_ci'])} | {f(c['mean_gain'], c['mean_gain_ci'])} | {c['ils_s_better']} / {c['ils_s_better_replay_worse']} |")
        Path(a.md).write_text("\n".join(L) + "\n")
    print(json.dumps({"PASS": res["PASS"], "verdict_by_bucket": verdict}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw"); d.add_argument("--src", required=True); d.add_argument("--out", required=True)
    p = sub.add_parser("points"); p.add_argument("--shards", required=True); p.add_argument("--decode-dir", required=True); p.add_argument("--draw", required=True); p.add_argument("--out", required=True)
    i = sub.add_parser("ils"); i.add_argument("--points", required=True); i.add_argument("--task", type=int, required=True); i.add_argument("--stride", type=int, required=True); i.add_argument("--out", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--rows-dir", required=True); an.add_argument("--out", required=True); an.add_argument("--md", default="")
    a = ap.parse_args(argv)
    {"draw": cmd_draw, "points": cmd_points, "ils": cmd_ils, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""D0 for the expected move-regret lever (spec: docs/lineages/uncut_groups_s0_v1.md, Record 2026-10-15 "Expected move-regret lever: REGISTERED-DESIGN").
Diagnostic, no training: how much does each teacher-forced MISS of the existing Arm 1b checkpoints cost, and how concentrated is that cost?

  uncut_d0_regret.py draw    --src replay_diag_draw.json --out d0_draw.json            ILS-labelled val groups, every bucket
  (prep: S5's `uncut_replay_diag.py prep --draw d0_draw.json --out SHARDS`)
  uncut_d0_regret.py regret  --shards D --draw d0_draw.json --out regret.json         per group: the label plan's single-task move costs under the labeller's S
  uncut_d0_regret.py tf      --shards D --arm gnn_eng|twin_eng --out tf_<arm>.json    per group x checkpoint: the teacher-forced argmax of every choice step
  uncut_d0_regret.py combine --regret regret.json --tf tf_gnn_eng.json [--tf-twin tf_twin_eng.json] --out d0.json [--md d0.md]

Move cost: D_c(t) = S(label with task t moved to c) - S(label), S the labeller's own plan cost (`_pg_plan_cost`, ib0); d_c = D_c / (S(label) / n), clipped to [0, D_MAX].
The decode and the teacher-forced pass are S5's (`uncut_replay_diag`), unchanged; `tf` restates the teacher-forced loop of `decode_group` to keep the argmax, and its hit count is asserted equal to S5's `tf_hit`."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts_cosim")):
    if p not in sys.path:
        sys.path.insert(0, p)

D_MAX = 10.0
TOP_SHARE = 0.20
NO_SIGNAL_C20 = 0.50
MIN_SCORED = 0.80
BUCKETS = ("51_100", "101_256")
BANNED_TOPOLOGIES = {16226} | set(range(16251, 16271)) | set(range(19537, 19549))   # accel_replica_v1 test topology and held-out sets; the Arm 1/1b/2 gate block
SPLIT = "experiments/accel_replica_v1_split.json"


def assert_not_test(gid: str, seed: Optional[int], test_ids: Sequence[str]) -> None:
    if gid in set(test_ids):
        raise SystemExit(f"FAIL LOUD: {gid} is one of the 57 accel_replica_v1 test ids")
    if seed is not None and seed in BANNED_TOPOLOGIES:
        raise SystemExit(f"FAIL LOUD: {gid} comes from topology {seed}, which is a test / held-out / gate topology")


def move_cost(dS: float, S0: float, n: int) -> Tuple[float, float]:
    """(clipped, raw) move cost in units of the label's mean per-task cost."""
    raw = max(0.0, dS) / (S0 / n)
    return min(raw, D_MAX), raw


def c20(costs: Sequence[float]) -> Optional[float]:
    """Share of the total cost carried by the costliest ceil(0.2 m) of the m misses; None when there are no misses or no cost."""
    m = len(costs)
    tot = sum(costs)
    if m == 0 or tot <= 0:
        return None
    k = max(1, math.ceil(TOP_SHARE * m))
    return sum(sorted(costs, reverse=True)[:k]) / tot


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


# ---- draw ------------------------------------------------------------------------------------------------------------------------------------

def cmd_draw(a):
    src = json.load(open(a.src))
    val = sorted(g for g in src["val"] if src["meta"][g]["label"] == "ils")
    out = {"split_sha256": src["split_sha256"], "source": a.src, "train": [], "val": val, "meta": {g: src["meta"][g] for g in val}}
    json.dump(out, open(a.out, "w"), indent=1)
    from collections import Counter
    print(dict(Counter(src["meta"][g]["bucket"] for g in val)))


# ---- regret table (engine only) --------------------------------------------------------------------------------------------------------------

def cmd_regret(a):
    import uncut_ils as U
    import uncut_replay_diag as R

    split = json.load(open(REPO / SPLIT))
    draw = json.load(open(a.draw))
    out: Dict[str, Any] = {}
    for f in sorted(Path(a.shards).glob("g_*.pkl")):
        for g in pickle.load(open(f, "rb")):
            gid, n = str(g.dataset_id), int(g.n_tasks)
            ds = R.dataset_dir(gid)
            seed = json.load(open(f"{ds}/generation_provenance.json")).get("seed")
            assert_not_test(gid, seed, split["test"])
            label = [[int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[R.ALPHA][0][t])]] for t in range(n)]
            box: Dict[str, Any] = {}
            orig = U._search_at_decision

            def table(sched, batch_tasks, system_state, res, local_of, queued, *args, _label=label, **kw):
                placements = {}
                for i, tk in enumerate(batch_tasks):
                    placements[i] = tuple(_label[local_of[int(tk.id)]]) if int(tk.id) in local_of else (int(res[i][0]), int(res[i][1]))
                cands_o, couple, _keys, _plan_o, pinned = sched._pg_plan_objects(batch_tasks, system_state, placements)
                for i, tk in enumerate(batch_tasks):
                    if int(tk.id) in queued:
                        pinned.add(i)
                ckeys = [[(int(x.id), int(y.id)) for x, y in cs] if i not in pinned else [placements[i]] for i, cs in enumerate(cands_o)]
                orch, nodes = sched._pg_orchestrator(), sched.nodes.items
                start = tuple(placements[i] for i in range(len(batch_tasks)))
                saved = (sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha)
                sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = True, 1.0, 0.0       # ib0, as the labeller
                memo: Dict[str, float] = {}
                try:
                    def S(plan):
                        return float(sched._pg_plan_cost(batch_tasks, [couple[k] for k in plan], orch, memo, nodes)[0])
                    S0 = S(start)
                    moves = {}
                    for i, tk in enumerate(batch_tasks):
                        if i in pinned or int(tk.id) not in local_of:
                            continue
                        row = []
                        for c in ckeys[i]:
                            if c == start[i]:
                                continue
                            row.append([list(c), S(start[:i] + (c,) + start[i + 1:]) - S0])
                        moves[local_of[int(tk.id)]] = row
                finally:
                    sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = saved
                return {"S0": S0, "moves": moves, "n_batch": len(batch_tasks), "n_pinned": len(pinned)}

            U._search_at_decision = table
            try:
                d = U.decision_pass({"dataset": ds, "sim_input": R.SIM_INPUT}, "cd_ib0", 0, 1, 1, None, check_arm=False)
            finally:
                U._search_at_decision = orig
            out[gid] = {"n": n, "bucket": draw["meta"][gid]["bucket"], "topology": seed, "S0": d["S0"], "moves": {str(k): v for k, v in d["moves"].items()}, "n_pinned": d["n_pinned"],
                        "label": label}
    miss = sorted(set(draw["val"]) - set(out))
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} drawn groups missing from the shards, e.g. {miss[:3]}")
    json.dump(out, open(a.out, "w"))
    print(f"regret: {len(out)} groups; test-id / banned-topology assertion passed on all")


# ---- teacher-forced argmax (models only) -----------------------------------------------------------------------------------------------------

def cmd_tf(a):
    import torch
    import uncut_replay_diag as R
    from src.policy.gnn.partial_state_edges import make_partial_state_score_fn
    from src.policy.gnn.seq_decode import topological_task_order
    from src.policy.tabular.reduced_features import build_partial_state_context_from_graph

    names = [n for n in R.RUNS if n.startswith(a.arm) and n.endswith("_final")]
    models = R._load_models(names)
    out: Dict[str, Any] = {"runs": names, "groups": {}}
    for f in sorted(Path(a.shards).glob("g_*.pkl")):
        for g in pickle.load(open(f, "rb")):
            gid, n = str(g.dataset_id), int(g.n_tasks)
            ctx = build_partial_state_context_from_graph(g)
            ctx.node_caps = g.partial_state_ctx["node_caps_by_alpha"][R.ALPHA]
            lp = [tuple(int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[R.ALPHA][0][t])]) for t in range(n)]
            choice = [t for t in range(n) if len(g.task_logit_to_placement[t]) > 1]
            rec = {}
            for name, m in models.items():
                tf: Dict[str, List[int]] = {}
                with torch.no_grad():
                    sf = make_partial_state_score_fn(m, g, ctx)
                    try:
                        committed: Dict[int, Any] = {}
                        hit = 0
                        for t in topological_task_order(n, g.dag_parents):
                            lg = sf(t, committed)
                            if t in choice:
                                cands = [tuple(int(v) for v in c) for c in g.task_logit_to_placement[t]]
                                am = cands[int(torch.argmax(lg))]
                                hit += int(am == lp[t])
                                tf[str(t)] = list(am)
                            committed[t] = lp[t]
                    finally:
                        if hasattr(sf, "close"):
                            sf.close()
                s5 = R.decode_group(m, g)["tf_hit"]
                if hit != s5:
                    raise SystemExit(f"FAIL LOUD: {gid} {name}: teacher-forced hits {hit} != S5's decode_group tf_hit {s5}")
                rec[name] = {"argmax": tf, "hit": hit, "n_choice": len(choice)}
            out["groups"][gid] = rec
    json.dump(out, open(a.out, "w"))
    print(f"tf: {len(out['groups'])} groups x {len(names)} checkpoints; hit counts equal S5's tf_hit on all")


# ---- combine (pure python) -------------------------------------------------------------------------------------------------------------------

def group_rows(reg: Dict[str, Any], tf: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (group, checkpoint, choice step): hit / miss, the move cost of the model's argmax (None if the engine never scored that candidate), and the step's clear-label flag."""
    rows = []
    for gid, r in reg.items():
        n, S0 = r["n"], r["S0"]
        for name, rec in tf["groups"][gid].items():
            for t_s, am in rec["argmax"].items():
                lab = r["label"][int(t_s)]
                moves = r["moves"].get(t_s)
                if moves is None:
                    continue
                costs = {tuple(k): move_cost(dS, S0, n) for k, dS in moves}
                row = {"group": gid, "bucket": r["bucket"], "run": name, "task": int(t_s), "miss": list(am) != list(lab), "scored": False, "d": None, "d_raw": None,
                       "clear": bool(costs) and min(c[0] for c in costs.values()) >= 1.0, "alts": len(costs)}
                if row["miss"] and tuple(am) in costs:
                    row["scored"], (row["d"], row["d_raw"]) = True, costs[tuple(am)]
                rows.append(row)
    return rows


def summarize(rows: Sequence[Dict[str, Any]], reps: int = 2000, seed: int = 20261015) -> Dict[str, Any]:
    steps, miss = len(rows), [r for r in rows if r["miss"]]
    sc = [r for r in miss if r["scored"]]
    d = [r["d"] for r in sc]
    by_g: Dict[str, List[Dict[str, Any]]] = {}
    for r in sc:
        by_g.setdefault(r["group"], []).append(r)
    ci = None
    if len(by_g) > 1:
        rng = random.Random(seed)
        gl = list(by_g.values())
        vals = []
        for _ in range(reps):
            pool = [x["d"] for j in (rng.randrange(len(gl)) for _ in range(len(gl))) for x in gl[j]]
            v = c20(pool)
            if v is not None:
                vals.append(v)
        vals.sort()
        ci = [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None
    return {"choice_steps": steps, "misses": len(miss), "miss_rate": len(miss) / steps if steps else None, "misses_scored": len(sc), "scored_share": len(sc) / len(miss) if miss else None,
            "median_d": _pct(d, 0.5), "p90_d": _pct(d, 0.9), "mean_d": sum(d) / len(d) if d else None, "share_d_ge_1": sum(1 for x in d if x >= 1.0) / len(d) if d else None,
            "mean_d_raw": sum(r["d_raw"] for r in sc) / len(sc) if sc else None, "share_clipped": sum(1 for r in sc if r["d_raw"] > D_MAX) / len(sc) if sc else None,
            "C20": c20(d), "C20_ci": ci, "groups": len(by_g), "clear_label_share": sum(1 for r in rows if r["clear"]) / steps if steps else None}


def verdict(per_bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """NO-SIGNAL if the top 20 % of misses carry < 50 % of cost in both 51-100 and 101-256 (primary: the two gnn_eng checkpoints pooled). Not readable if a bucket has < 80 % of its misses scored."""
    cs = {b: per_bucket[b]["C20"] for b in BUCKETS}
    cov = {b: per_bucket[b]["scored_share"] for b in BUCKETS}
    if any(v is None for v in cs.values()) or any(v is None or v < MIN_SCORED for v in cov.values()):
        return {"verdict": "UNREADABLE", "C20": cs, "scored_share": cov}
    return {"verdict": "NO-SIGNAL" if all(c < NO_SIGNAL_C20 for c in cs.values()) else "SIGNAL-PRESENT", "C20": cs, "scored_share": cov}


def cmd_combine(a):
    reg = json.load(open(a.regret))
    tf = json.load(open(a.tf))
    rows = group_rows(reg, tf)
    res: Dict[str, Any] = {"groups": len(reg), "primary_runs": tf["runs"], "per_bucket": {}, "per_run": {}, "twin": {}}
    buckets = sorted({r["bucket"] for r in rows}, key=lambda b: (len(b), b))
    for b in buckets:
        res["per_bucket"][b] = summarize([r for r in rows if r["bucket"] == b])
        for run in tf["runs"]:
            res["per_run"][f"{b}|{run}"] = summarize([r for r in rows if r["bucket"] == b and r["run"] == run])
    res["verdict"] = verdict(res["per_bucket"]) if all(b in res["per_bucket"] for b in BUCKETS) else {"verdict": "UNREADABLE"}
    nonneg = [m[1] for r in reg.values() for row in r["moves"].values() for m in row]
    res["label_local_optimality"] = {"moves": len(nonneg), "improving_moves": sum(1 for x in nonneg if x < -1e-9), "improving_share": sum(1 for x in nonneg if x < -1e-9) / len(nonneg) if nonneg else None}
    if a.tf_twin:
        twin = json.load(open(a.tf_twin))
        trows = group_rows(reg, twin)
        for b in buckets:
            res["twin"][b] = summarize([r for r in trows if r["bucket"] == b])
    json.dump(res, open(a.out, "w"), indent=1)
    if a.md:
        f = lambda x, pct=True: "-" if x is None else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")
        L = ["# D0: cost of teacher-forced misses (Arm 1b -final.pt, ILS-labelled val groups; development)", "", f"VERDICT: {res['verdict']}", "",
             f"label local optimality: {res['label_local_optimality']}", "", "| bucket | runs | choice steps | miss rate | misses scored | median d | p90 d | mean d | share d>=1 | C20 [95% CI] | clear-label share |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for tag, d in [(k, v) for k, v in res["per_bucket"].items()] + [(k, v) for k, v in res["per_run"].items()] + [("twin " + k, v) for k, v in res["twin"].items()]:
            ci = d["C20_ci"]
            L.append(f"| {tag} | - | {d['choice_steps']} | {f(d['miss_rate'])} | {d['misses_scored']} ({f(d['scored_share'])}) | {f(d['median_d'], False)} | {f(d['p90_d'], False)} | {f(d['mean_d'], False)} | {f(d['share_d_ge_1'])} | "
                     f"{f(d['C20'])}{' [' + f(ci[0]) + ', ' + f(ci[1]) + ']' if ci else ''} | {f(d['clear_label_share'])} |")
        Path(a.md).write_text("\n".join(L) + "\n")
    print(json.dumps(res["verdict"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw"); d.add_argument("--src", required=True); d.add_argument("--out", required=True)
    r = sub.add_parser("regret"); r.add_argument("--shards", required=True); r.add_argument("--draw", required=True); r.add_argument("--out", required=True)
    t = sub.add_parser("tf"); t.add_argument("--shards", required=True); t.add_argument("--arm", required=True, choices=("gnn_eng", "twin_eng")); t.add_argument("--out", required=True)
    c = sub.add_parser("combine"); c.add_argument("--regret", required=True); c.add_argument("--tf", required=True); c.add_argument("--tf-twin", default=""); c.add_argument("--out", required=True); c.add_argument("--md", default="")
    a = ap.parse_args(argv)
    {"draw": cmd_draw, "regret": cmd_regret, "tf": cmd_tf, "combine": cmd_combine}[a.cmd](a)


if __name__ == "__main__":
    main()

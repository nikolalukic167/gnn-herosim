#!/usr/bin/env python3
"""Move-regret table for the expected move-regret lever (spec: docs/lineages/uncut_groups_s0_v1.md, REGISTERED-DESIGN 2026-10-15; step 2).

For every ILS-labelled group of the Arm 1 TRAIN split, the cost of moving each task from the label's placement to each other candidate, under the labeller's own plan cost S
(`_pg_plan_cost`, ib0; the single-task neighbours the ILS' ICM pass evaluates):

  D_c(t) = S(label with task t moved to c) - S(label);   d_c(t) = min(D_c(t) / (S(label) / n), D_MAX),   D_MAX = 10, d = 0 on the label's own candidate.

  uncut_move_regret_label.py time  --cache GRAPHS.pkl --out timing.json [--per-bucket 50]     200-group timing (stratified), projected CPU-h for the full train set
  uncut_move_regret_label.py label --cache GRAPHS.pkl --task K --stride S --out part_K.pkl     the table for groups K::S of the ILS-labelled train groups
  uncut_move_regret_label.py merge --parts DIR --out move_regret.pkl                           one table: {"header": {...}, "groups": {dataset_id: {"d": [[per candidate, graph order] per task], ...}}}

Train groups only, ILS-labelled only (placements.jsonl header `label_source == "ils"`). Asserted for every group: its id is not one of the 220 test placeholders of the Arm 1 split nor one of the
57 `accel_replica_v1` test ids, and its topology (`generation_provenance.json` seed) is not 16226, 16251-16270 or 19537-19548. A candidate the engine never scored gets d = D_MAX
(counted); a task the engine does not decide (pinned / queued) gets d = 0 on every candidate (counted). The engine pass is `uncut_d0_regret`'s, unchanged in what it computes."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts_cosim")):
    if p not in sys.path:
        sys.path.insert(0, p)

ALPHA = "inf"
D_MAX = 10.0
SPLIT = "experiments/uncut_arm1_split.json"
ACCEL_SPLIT = "experiments/accel_replica_v1_split.json"
CORPUS = "/share/nikola.lukic/uncut_arm1/corpus"
BUCKETS = (("le10", 1, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_256", 101, 256))


def bucket(n: int) -> str:
    return next(b for b, lo, hi in BUCKETS if lo <= n <= hi)


def header(gid: str) -> Dict[str, Any]:
    kind, ds = gid.split("/")
    with open(f"{CORPUS}/{'train' if 'train' in kind else 'heldout'}/{ds}/placements/placements.jsonl") as fh:
        return json.loads(fh.readline())


def table_row(candidates: List[Any], label_idx: int, moves: Dict[Any, float] | None, S0: float, n: int) -> tuple[List[float], int]:
    """d for one task's candidates in graph order. `moves` maps a candidate key (node, platform) to its D (None = the engine does not decide this task). Returns (d, n_unscored)."""
    if moves is None:
        return [0.0] * len(candidates), 0
    out, unscored = [], 0
    for i, c in enumerate(candidates):
        key = (int(c[0]), int(c[1]))
        if i == label_idx:
            out.append(0.0)
        elif key in moves:
            out.append(min(max(0.0, moves[key]) / (S0 / n), D_MAX))
        else:
            out.append(D_MAX)
            unscored += 1
    return out, unscored


def engine_table(gid: str, g: Any, ds: str) -> Dict[str, Any]:
    """The engine pass of uncut_d0_regret.cmd_regret for one group: S(label) and the S of every single-task move, keyed by dataset task index."""
    import uncut_ils as U
    import uncut_replay_diag as R

    n = int(g.n_tasks)
    label = [[int(v) for v in g.task_logit_to_placement[t][int(g.tied_optimal_logit_plans[R.ALPHA][0][t])]] for t in range(n)]
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
        evals = 0
        try:
            def S(plan):
                return float(sched._pg_plan_cost(batch_tasks, [couple[k] for k in plan], orch, memo, nodes)[0])
            S0 = S(start)
            moves = {}
            for i, tk in enumerate(batch_tasks):
                if i in pinned or int(tk.id) not in local_of:
                    continue
                row = {}
                for c in ckeys[i]:
                    if c == start[i]:
                        continue
                    row[c] = S(start[:i] + (c,) + start[i + 1:]) - S0
                    evals += 1
                moves[local_of[int(tk.id)]] = row
        finally:
            sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = saved
        return {"S0": S0, "moves": moves, "evals": evals}

    U._search_at_decision = table
    try:
        d = U.decision_pass({"dataset": ds, "sim_input": R.SIM_INPUT}, "cd_ib0", 0, 1, 1, None, check_arm=False)
    finally:
        U._search_at_decision = orig
    return {"S0": d["S0"], "moves": d["moves"], "evals": d["evals"]}


def label_group(gid: str, g: Any, banned: set, test_ids: set) -> Dict[str, Any]:
    import uncut_replay_diag as R
    from uncut_d0_regret import assert_not_test

    ds = R.dataset_dir(gid)
    seed = json.load(open(f"{ds}/generation_provenance.json")).get("seed")
    assert_not_test(gid, seed, sorted(test_ids))
    n = int(g.n_tasks)
    t0 = time.time()
    r = engine_table(gid, g, ds)
    d, unscored, undecided, n_cand, n_alt = [], 0, 0, 0, 0
    for t in range(n):
        plan_idx = int(g.tied_optimal_logit_plans[R.ALPHA][0][t])
        moves = r["moves"].get(t)
        row, u = table_row([tuple(c) for c in g.task_logit_to_placement[t]], plan_idx, moves, r["S0"], n)
        undecided += moves is None
        unscored += u
        n_cand += len(row)
        n_alt += len(row) - 1 if moves is not None else 0
        d.append(row)
    return {"d": d, "S0": r["S0"], "n": n, "unscored": unscored, "undecided_tasks": undecided, "candidates": n_cand, "alternatives_of_decided_tasks": n_alt, "engine_evals": r["evals"], "wall": time.time() - t0, "topology": seed}


def train_ils_groups() -> Dict[str, str]:
    split = json.load(open(REPO / SPLIT))
    out = {}
    for gid in split["train"]:
        h = header(gid)
        if h.get("label_source") == "ils":
            out[gid] = bucket(int(h["n_tasks"]))
    return out


def _forbidden():
    test_ids = set(json.load(open(REPO / SPLIT))["test"]) | set(json.load(open(REPO / ACCEL_SPLIT))["test"])
    return set(), test_ids


def _selected_graphs(cache: str, ids):
    graphs = pickle.load(open(cache, "rb"))
    want = set(ids)
    return [g for g in graphs if str(g.dataset_id) in want]


def cmd_time(a):
    ils = train_ils_groups()
    rng = random.Random("uncut-move-regret:v1:timing")
    sample = []
    for b, _, _ in BUCKETS:
        pool = sorted(g for g, bb in ils.items() if bb == b)
        sample += rng.sample(pool, min(a.per_bucket, len(pool)))
    graphs = {str(g.dataset_id): g for g in _selected_graphs(a.cache, sample)}
    banned, test_ids = _forbidden()
    rows = []
    for gid in sample:
        r = label_group(gid, graphs[gid], banned, test_ids)
        rows.append({"group": gid, "bucket": ils[gid], "n": r["n"], "wall": r["wall"], "evals": r["engine_evals"], "unscored": r["unscored"], "undecided_tasks": r["undecided_tasks"],
                     "candidates": r["candidates"], "alternatives_of_decided_tasks": r["alternatives_of_decided_tasks"]})
    per = {}
    for b, _, _ in BUCKETS:
        rb = [x for x in rows if x["bucket"] == b]
        total = sum(1 for g, bb in ils.items() if bb == b)
        per[b] = {"groups_in_train": total, "sampled": len(rb), "mean_wall_s": sum(x["wall"] for x in rb) / len(rb) if rb else None,
                  "projected_cpu_h": (total * sum(x["wall"] for x in rb) / len(rb) / 3600.0) if rb else 0.0}
    tot = lambda k: sum(x[k] for x in rows)
    shares = {"unscored_share_of_alternatives": tot("unscored") / max(1, tot("alternatives_of_decided_tasks")), "undecided_share_of_tasks": tot("undecided_tasks") / max(1, tot("n")),
              "unscored_share_of_candidates": tot("unscored") / max(1, tot("candidates")),
              "by_bucket": {b: {"unscored_share_of_alternatives": sum(x["unscored"] for x in rows if x["bucket"] == b) / max(1, sum(x["alternatives_of_decided_tasks"] for x in rows if x["bucket"] == b)),
                                "undecided_share_of_tasks": sum(x["undecided_tasks"] for x in rows if x["bucket"] == b) / max(1, sum(x["n"] for x in rows if x["bucket"] == b))} for b, _, _ in BUCKETS}}
    out = {"ils_labelled_train_groups": len(ils), "per_bucket": per, "shares": shares, "projected_cpu_h_total": sum(v["projected_cpu_h"] for v in per.values()), "rows": rows}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("ils_labelled_train_groups", "per_bucket", "shares", "projected_cpu_h_total")}, indent=1))


def cmd_label(a):
    ils = train_ils_groups()
    mine = sorted(ils)[a.task::a.stride]
    graphs = {str(g.dataset_id): g for g in _selected_graphs(a.cache, mine)}
    missing = [g for g in mine if g not in graphs]
    if missing:
        raise SystemExit(f"FAIL LOUD: {len(missing)} ILS-labelled train groups are not in the cache, e.g. {missing[:3]}")
    banned, test_ids = _forbidden()
    out: Dict[str, Any] = {}
    for gid in mine:
        out[gid] = label_group(gid, graphs[gid], banned, test_ids)
    pickle.dump(out, open(a.out, "wb"))
    print(f"label: {len(out)} groups, {sum(v['wall'] for v in out.values()) / 3600:.2f} CPU-h")


def cmd_merge(a):
    groups: Dict[str, Any] = {}
    for f in sorted(Path(a.parts).glob("part_*.pkl")):
        groups.update(pickle.load(open(f, "rb")))
    ils = train_ils_groups()
    if set(groups) != set(ils):
        raise SystemExit(f"FAIL LOUD: the parts cover {len(groups)} groups, the ILS-labelled train set has {len(ils)}")
    unscored = sum(v["unscored"] for v in groups.values())
    cands = sum(len(r) for v in groups.values() for r in v["d"])
    hdr = {"alpha_key": ALPHA, "d_max": D_MAX, "split": SPLIT, "n_groups": len(groups), "unscored_candidates": unscored, "candidates": cands,
           "undecided_tasks": sum(v["undecided_tasks"] for v in groups.values()), "by_bucket": {b: sum(1 for x in ils.values() if x == b) for b, _, _ in BUCKETS}}
    pickle.dump({"header": hdr, "groups": {g: {"d": v["d"], "S0": v["S0"], "n": v["n"]} for g, v in groups.items()}}, open(a.out, "wb"))
    print(json.dumps(hdr), hashlib.sha256(open(a.out, "rb").read()).hexdigest())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time"); t.add_argument("--cache", required=True); t.add_argument("--out", required=True); t.add_argument("--per-bucket", type=int, default=50)
    lb = sub.add_parser("label"); lb.add_argument("--cache", required=True); lb.add_argument("--task", type=int, required=True); lb.add_argument("--stride", type=int, required=True); lb.add_argument("--out", required=True)
    m = sub.add_parser("merge"); m.add_argument("--parts", required=True); m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"time": cmd_time, "label": cmd_label, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()

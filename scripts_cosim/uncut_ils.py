#!/usr/bin/env python3
"""uncut_groups_s0_v1 K1(b) tool and label engine: iterated local search on ib0's surrogate S for ONE captured batch state, then an exact replay of
the hand plan and of each new ILS incumbent on the same fidelity snapshot.

State = a live_audit snapshot record (HEROSIM_SNAPSHOT_FIDELITY=1) of a whole peer group (> 10 tasks, <= CAP = 256). The decision is replayed under the hand arm
(the arm's own scheduler code decides the batch at the snapshot instant; that plan is the hand plan), and at that decision the scheduler's own `_pg_plan_cost`
(the S that cd_exactS_ib0 scores with, in-batch stacking alpha 0, drain alpha 1) scores every plan the ILS visits. No second S exists in this file.

ILS (`ils`): start at the hand plan; best-improvement single moves per task (ICM sweeps) alternating with greedy alpha-expansion moves (per label, tasks ordered by
their single-move gain, every prefix scored, the best strictly improving prefix applied); on a local optimum a seeded kick (a few random reassignments, or k tasks
pulled onto one label) and a new descent; a kick is kept only when it beats the incumbent. S on every distinct plan evaluated, incumbents logged along the
trajectory with the evaluation index, ties broken by candidate order (node id, platform id), nothing depends on set order: two runs at one seed are byte-identical.

  uncut_ils.py run   JOBS.jsonl OUT.jsonl [--arm cd_expand] [--seed 0] [--max-evals N | --evals-per-task-cand 200] [--max-seconds T] [--replay all|final|none]
  uncut_ils.py read  ROWS.jsonl [--json OUT]       K1(b): mean (RTT(hand) - RTT(ILS)) / RTT(ILS), share within 1 %, p90, per bucket, S-regret beside it

A job: {"state_id", "bucket", "snapfile", "trigger_task_id", "cell_config", "sim_input", "cell_summary" (its `env` is the physics env of the capture), "time" (optional)}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CAP = 256
EPS = 1e-9
BUCKETS = (("11_50", 11, 50), ("51_100", 51, 100), ("101_226", 101, 226))
Key = Tuple[int, int]

# the hand arms a K1 hand plan can come from: (strategy, env of the arm's decision); the capture's own physics env is applied first
ARMS = {
    "cd": ("peer_greedy_network_cd_peer_greedy_network_cd", {}),
    "cd_expand": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXPANSION": "1"}),
    "cd_exactS": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_CD_EXPANSION": "1"}),
    "cd_exactS_ib0": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_CD_EXPANSION": "1", "HEROSIM_PG_BATCH_ALPHA": "0"}),
}


# ---- the search (pure: S is a callable, so it is testable without a simulator) ------------------------------------------------------------

def ils(cands: Sequence[Sequence[Key]], start: Sequence[Key], S: Callable[[Tuple[Key, ...]], float], *, seed: int, max_evals: int,
        max_seconds: Optional[float] = None, cap: int = CAP) -> Dict[str, Any]:
    """ILS on S over the product of `cands` (task i may take any key of cands[i], sorted by (node id, platform id); a task whose start key is not among them is pinned).
    Budget: `max_evals` DISTINCT plans scored (the start counts), or `max_seconds`, whichever ends first; the budget is checked before every evaluation.
    Returns {"best": plan, "best_S", "start_S", "trajectory": [{"eval", "S", "plan", "iter"}], "evals": [{"eval", "S", "iter", "plan"}], "n_evals", "stopped"}.
    A plan is a tuple of keys in task order. `evals` is every distinct plan scored, in order, with nothing deduplicated away beyond distinctness."""
    n = len(cands)
    if n > cap:
        raise ValueError(f"FAIL LOUD: a group of {n} tasks is above the label cap {cap}")
    if len(start) != n:
        raise ValueError("start plan length differs from the number of tasks")
    cs: List[List[Key]] = []
    for i in range(n):
        s = tuple(start[i])
        c = sorted({tuple(k) for k in cands[i]})
        cs.append(c if s in c else [s])
    rng = random.Random(f"uncut_ils:{seed}")
    t0 = time.time()
    cache: Dict[Tuple[Key, ...], float] = {}
    evals: List[Dict[str, Any]] = []
    traj: List[Dict[str, Any]] = []
    state = {"iter": 0, "stop": None}
    best: Dict[str, Any] = {"S": None, "plan": None}

    class _Stop(Exception):
        pass

    def score(plan: Tuple[Key, ...]) -> float:
        v = cache.get(plan)
        if v is not None:
            return v
        if len(cache) >= max_evals:
            state["stop"] = "evals"
            raise _Stop
        if max_seconds is not None and time.time() - t0 >= max_seconds:
            state["stop"] = "seconds"
            raise _Stop
        v = float(S(plan))
        cache[plan] = v
        evals.append({"eval": len(cache), "S": v, "iter": state["iter"], "plan": plan})
        if best["S"] is None or v < best["S"] - EPS:
            best["S"], best["plan"] = v, plan
            traj.append({"eval": len(cache), "S": v, "plan": plan, "iter": state["iter"]})
        return v

    movable = [i for i in range(n) if len(cs[i]) > 1]
    labels = sorted({k for c in cs for k in c})

    def icm(plan: List[Key], cur: float) -> Tuple[List[Key], float, bool]:
        changed = False
        order = list(movable)
        rng.shuffle(order)
        for i in order:
            pick, pv = None, cur
            for k in cs[i]:
                if k == plan[i]:
                    continue
                trial = plan[:i] + [k] + plan[i + 1:]
                v = score(tuple(trial))
                if v < pv - EPS:
                    pick, pv = k, v
            if pick is not None:
                plan[i], cur, changed = pick, pv, True
        return plan, cur, changed

    def expand(plan: List[Key], cur: float) -> Tuple[List[Key], float, bool]:
        changed = False
        for lab in labels:
            movers = [i for i in movable if plan[i] != lab and lab in cs[i]]
            if len(movers) < 2:
                continue            # a single mover is an ICM move
            gains = []
            for i in movers:
                trial = plan[:i] + [lab] + plan[i + 1:]
                gains.append((score(tuple(trial)), i))
            gains.sort()
            sub: List[int] = []
            pick, pv = None, cur
            for _g, i in gains:
                sub.append(i)
                if len(sub) < 2:
                    continue
                trial = list(plan)
                for j in sub:
                    trial[j] = lab
                v = score(tuple(trial))
                if v < pv - EPS:
                    pick, pv = list(sub), v
            if pick is not None:
                for j in pick:
                    plan[j] = lab
                cur, changed = pv, True
        return plan, cur, changed

    def descend(plan: List[Key], cur: float) -> Tuple[List[Key], float]:
        while True:
            plan, cur, c1 = icm(plan, cur)
            plan, cur, c2 = expand(plan, cur)
            if not (c1 or c2):
                return plan, cur

    def kick(plan: List[Key]) -> List[Key]:
        out = list(plan)
        if len(movable) == 0:
            return out
        if rng.random() < 0.5:
            m = rng.randint(2, min(5, max(2, len(movable))))
            for i in rng.sample(movable, min(m, len(movable))):
                alts = [k for k in cs[i] if k != out[i]]
                out[i] = alts[rng.randrange(len(alts))]
        else:
            lab = labels[rng.randrange(len(labels))]
            pool = [i for i in movable if lab in cs[i]]
            for i in rng.sample(pool, min(len(pool), rng.randint(2, 6))):
                out[i] = lab
        return out

    plan0 = tuple(tuple(k) for k in start)
    try:
        score(plan0)
        start_S = cache[plan0]
        cur_plan, cur = list(plan0), start_S
        cur_plan, cur = descend(cur_plan, cur)
        stale = 0
        while True:
            if not movable:
                state["stop"] = "no-moves"
                break
            state["iter"] += 1
            before = len(cache)
            trial = kick(best["plan"])
            tv = score(tuple(trial))
            trial, tv = descend(trial, tv)
            stale = 0 if len(cache) > before else stale + 1
            if stale >= 300:
                state["stop"] = "converged"      # 300 kicks in a row reached only plans already scored
                break
    except _Stop:
        pass
    if state["stop"] is None:
        state["stop"] = "done"
    return {"best": best["plan"], "best_S": best["S"], "start_S": cache[plan0], "trajectory": traj, "evals": evals, "n_evals": len(cache),
            "stopped": state["stop"], "iters": state["iter"], "wall": time.time() - t0}


def default_budget(cands: Sequence[Sequence[Key]], per: int = 200) -> int:
    """The registered budget: 200 x n x (mean candidates per task) evaluations per group."""
    n = len(cands)
    return int(per * n * (sum(len(c) for c in cands) / max(n, 1)))


# ---- the replay passes (datalab; the scheduler's own S and the fidelity replay) -------------------------------------------------------------

class _Done(Exception):
    def __init__(self, out):
        self.out = out


def _setup_env(arm: str, cell_summary: Optional[str]) -> Tuple[str, Dict[str, str]]:
    strategy, extra = ARMS[arm]
    os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
    if cell_summary:
        for k, v in (json.load(open(cell_summary)).get("env") or {}).items():
            if k.startswith("HEROSIM_") or k.startswith("GNN_"):
                os.environ[k] = str(v)
    for k in ("HEROSIM_PG_CD_EXACT", "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_BATCH_ALPHA", "HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_ORACLE", "HEROSIM_PG_ORACLE_SEARCH",
              "HEROSIM_PG_CD_EXACT_PRUNE", "HEROSIM_MAX_EVENTS", "GNN_CD_REFINE", "GNN_SLATE_NO_SPLIT", "GNN_SERVE_CANDIDATE_SLATE"):
        os.environ.pop(k, None)
    for k in [k for k in os.environ if k.startswith("HEROSIM_DECISION")]:
        os.environ.pop(k)                       # the decision-cost meter stays off the offline path
    os.environ.update(GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", PYTHONHASHSEED="0", SIM_FORCE_FULL_STATS="1")
    os.environ.update(extra)
    return strategy, extra


def _load_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    for line in open(job["snapfile"]):
        rec = json.loads(line)
        if int(rec["trigger_task_id"]) == int(job["trigger_task_id"]) and (job.get("time") is None or abs(float(rec["time"]) - float(job["time"])) < 1e-6):
            return rec
    raise RuntimeError(f"FAIL LOUD: no snapshot with trigger {job['trigger_task_id']} in {job['snapfile']}")


def decision_pass(job: Dict[str, Any], arm: str, seed: int, max_evals: Optional[int], per: int, max_seconds: Optional[float]) -> Dict[str, Any]:
    """Replay the snapshot under `arm`; at the batch decision score the hand plan with the scheduler's S and run the ILS from it. Returns the search result
    plus the hand plan, in snapshot batch order (gid ascending) as (node id, platform id)."""
    strategy, _ = _setup_env(arm, job.get("cell_summary"))
    from src.executesimulation import execute_simulation
    from src.placement import fidelity_replay, snapshot_fidelity
    from src.placement.live_snapshot_seed import build_live_snapshot_seed
    from src.policy.gnn import orchestrator as GO
    from src.policy.peer_greedy_network import scheduler as PG

    if job.get("dataset"):          # a corpus dataset (frontier_v1 states): its own spec, tasks in dataset order, so a placements.jsonl table indexes the plans
        ds = Path(job["dataset"])
        seed_ds = json.load(open(ds / "infrastructure.json"))["live_snapshot_seed"]
        spec = deepcopy(seed_ds["fidelity_replay"])
        if not Path(spec["sim_input"]).exists():
            spec["sim_input"] = job.get("sim_input") or str(REPO / "data" / "nofs-ids")
        snap, replicas_by_type = spec["snapshot"], seed_ds["replicas_by_type"]
        types = [next(iter(e["application"]["dag"])) for e in json.load(open(ds / "workload.json"))["events"]]
    else:
        snap = _load_snapshot(job)
        spec = {"snapshot": snap, "cell_config": job["cell_config"], "sim_input": job["sim_input"], "live_run_params": snapshot_fidelity.live_run_params()}
        replicas_by_type = build_live_snapshot_seed(snap)["replicas_by_type"]
        types = [r["fn"] for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
    fid = snap["fidelity"]
    t0 = float(snap["time"])
    offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)] for t, specs in replicas_by_type.items()}
    fr = fidelity_replay.FidelityReplay(spec, types, offered)
    local_of = {fr.batch_local[i]: i for i in range(len(fr.batch_local))}   # replay id -> index in snapshot batch order
    forced = dict(fr.queued_forced)
    orig_inf = PG.PeerGreedyNetworkBatchScheduler._prefix_inference
    orig_monitor = GO.GNNOrchestrator.monitor_process
    phase = (math.ceil(t0) - t0) % 1.0
    seen: Dict[str, Any] = {"decisions": 0}

    def patched(self, batch_tasks, system_state, queue_snapshot, temporal_state):
        seen["decisions"] += 1
        node_by_id = {nd.id: nd for nd in self.nodes.items}
        out, free = {}, []
        for i, t in enumerate(batch_tasks):
            f = forced.get(int(t.id))
            if f is not None:
                out[i] = (int(f[0]), int(f[1]))
                t.planned_node_name = node_by_id[int(f[0])].node_name
            else:
                free.append(i)
        if not out:
            res = orig_inf(self, batch_tasks, system_state, queue_snapshot, temporal_state)
        else:
            res = {}
            if free:
                sub = self._pg_decide([batch_tasks[i] for i in free], system_state)
                for k, i in enumerate(free):
                    res[i] = sub[k]
            res.update(out)
        if set(local_of) <= {int(t.id) for t in batch_tasks} and "out" not in seen:
            seen["out"] = _search_at_decision(self, batch_tasks, system_state, res, local_of, set(fr.queued_forced), seed, max_evals, per, max_seconds,
                                                int(job.get("exhaustive_cap") or 0))
            raise _Done(seen["out"])
        return res

    def monitor(self):
        if phase > 0:
            yield self.env.timeout(phase)
        yield self.env.timeout(0)
        yield from orig_monitor(self)

    PG.PeerGreedyNetworkBatchScheduler._prefix_inference = patched
    GO.GNNOrchestrator.monitor_process = monitor
    try:
        infra = deepcopy(fr.base_infra)
        seed_live = deepcopy(fr.seed)
        seed_live["fidelity"] = fid
        infra["live_snapshot_seed"] = seed_live
        infra["fast_forward_warmup"] = True
        infra["fast_forward_threshold"] = 1
        try:
            execute_simulation({"infrastructure": infra, "workload": deepcopy(fr.wl)}, fr.sim_inputs, scheduling_strategy=strategy,
                               cache_policy="fifo", task_priority="fifo", models=None, **fr.kw)
        except _Done as d:
            out = d.out
        else:
            raise RuntimeError("FAIL LOUD: the simulation ended without the batch being decided whole (no decision with the whole group)")
    finally:
        PG.PeerGreedyNetworkBatchScheduler._prefix_inference = orig_inf
        GO.GNNOrchestrator.monitor_process = orig_monitor
    out.update(t0=t0, types=types, spec=spec)
    return out


def _search_at_decision(sched, batch_tasks, system_state, res, local_of, queued, seed, max_evals, per, max_seconds, exhaustive_cap=0) -> Dict[str, Any]:
    """Runs inside the decision: res[i] = (node id, platform id) of batch_tasks[i] under the hand arm."""
    placements = {i: (int(res[i][0]), int(res[i][1])) for i in range(len(batch_tasks))}
    cands_o, couple, keys, plan_o, pinned = sched._pg_plan_objects(batch_tasks, system_state, placements)
    orch, nodes = sched._pg_orchestrator(), sched.nodes.items
    idx = [i for i, t in enumerate(batch_tasks)]
    for i, t in enumerate(batch_tasks):
        if int(t.id) in queued:
            pinned.add(i)
    ckeys = [[(int(a.id), int(b.id)) for a, b in cs] if i not in pinned else [placements[i]] for i, cs in enumerate(cands_o)]
    start = tuple(placements[i] for i in idx)
    saved = (sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha)
    sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = True, 1.0, 0.0     # ib0 (HEROSIM_PG_BATCH_ALPHA=0)
    memo: Dict[str, float] = {}
    n_calls = [0]

    def S(plan):
        n_calls[0] += 1
        return float(sched._pg_plan_cost(batch_tasks, [couple[k] for k in plan], orch, memo, nodes)[0])

    try:
        budget = max_evals if max_evals else default_budget(ckeys, per)
        t1 = time.time()
        r = ils(ckeys, start, S, seed=seed, max_evals=budget, max_seconds=max_seconds)
        wall_s = time.time() - t1
        ils_calls = n_calls[0]
        fresh = [float(sched._pg_plan_cost(batch_tasks, [couple[k] for k in r["best"]], orch, {}, nodes)[0]), float(sched._pg_plan_cost(batch_tasks, [couple[k] for k in start], orch, {}, nodes)[0])]
        if abs(fresh[0] - r["best_S"]) > 1e-6 or abs(fresh[1] - r["start_S"]) > 1e-6:
            raise RuntimeError("FAIL LOUD: S with a shared memo differs from S with a fresh memo")
        exh = None
        if exhaustive_cap:          # the S-optimum of the whole product, for states small enough to enumerate (the ILS reaches it or the tool is wrong)
            from itertools import product

            total = math.prod(len(c) for c in ckeys)
            if total <= exhaustive_cap:
                best_e, argm = None, None
                for plan in product(*ckeys):
                    v = S(plan)
                    if best_e is None or v < best_e - EPS:
                        best_e, argm = v, plan
                exh = {"plans": total, "s_opt": best_e, "ils_gap": r["best_S"] - best_e}
    finally:
        sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = saved
    order = sorted(((local_of[int(t.id)], i) for i, t in enumerate(batch_tasks) if int(t.id) in local_of))
    to_batch = lambda plan: [list(plan[i]) for _b, i in order]       # snapshot batch order (gid ascending)
    return {"n": len(batch_tasks), "n_batch": len(order), "n_pinned": len(pinned), "cands_mean": sum(len(c) for c in ckeys) / len(ckeys),
            "hand": to_batch(start), "start_S": r["start_S"], "best": to_batch(r["best"]), "best_S": r["best_S"], "n_evals": r["n_evals"], "budget": budget,
            "stopped": r["stopped"], "iters": r["iters"], "s_calls": ils_calls, "search_wall": wall_s, "exhaustive": exh,
            "trajectory": [{"eval": t["eval"], "S": t["S"], "iter": t["iter"], "plan": to_batch(t["plan"])} for t in r["trajectory"]],
            "evals_log": [(e["eval"], e["S"], e["iter"]) for e in r["evals"]]}


def replay_rtts(spec: Dict[str, Any], types: Sequence[str], plans: Sequence[Sequence[Sequence[int]]]) -> Tuple[List[float], float, List[float]]:
    """Fidelity replay of each plan (batch order = snapshot gid order) on the snapshot: the corpus label, sum over the batch of (doneTime - scheduledTime).
    The same code as plan_oracle._label_chunk."""
    from src.placement import plan_oracle

    spec = dict(spec)
    spec["queued_tasks"] = len(spec["snapshot"]["fidelity"]["queued"])
    spec["batch_tasks"] = len(spec["snapshot"]["fidelity"]["batch"])
    return plan_oracle._label_chunk(spec, list(types), [[tuple(k) for k in p] for p in plans])


# ---- the state driver and the reader -------------------------------------------------------------------------------------------------------

def bucket_of(n: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    raise ValueError(f"group of {n} tasks is outside the K1 buckets")


def run_state(job: Dict[str, Any], arm: str, seed: int, max_evals: Optional[int], per: int, max_seconds: Optional[float], replay: str) -> Dict[str, Any]:
    t0 = time.time()
    d = decision_pass(job, arm, seed, max_evals, per, max_seconds)
    row = {k: d[k] for k in ("n", "n_batch", "n_pinned", "cands_mean", "start_S", "best_S", "n_evals", "budget", "stopped", "iters", "s_calls", "search_wall",
                             "hand", "best", "t0", "exhaustive")}
    row.update(state_id=job.get("state_id"), bucket=job.get("bucket") or bucket_of(d["n_batch"]), arm=arm, seed=seed,
               s_per_call_ms=1000.0 * d["search_wall"] / max(d["s_calls"], 1), n_incumbents=len(d["trajectory"]),
               trajectory=[{"eval": t["eval"], "S": t["S"], "iter": t["iter"]} for t in d["trajectory"]], evals_log=d["evals_log"])
    if replay != "none":
        plans = [d["hand"]] + ([t["plan"] for t in d["trajectory"][1:]] if replay == "all" else [d["best"]] if d["best"] != d["hand"] else [])
        t1 = time.time()
        rt, setup, runs = replay_rtts(d["spec"], d["types"], plans)
        row.update(rtt_hand=rt[0], rtt_ils=rt[-1] if len(rt) > 1 else rt[0], rtt_incumbents=rt[1:], replay_setup=setup, replay_run_mean=sum(runs) / len(runs),
                   replay_wall=time.time() - t1, n_replays=len(rt))
        row["regret"] = (row["rtt_hand"] - row["rtt_ils"]) / row["rtt_ils"]
        row["s_regret"] = (row["start_S"] - row["best_S"]) / row["best_S"] if row["best_S"] else None
    row["wall"] = time.time() - t0
    return row


def read_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """K1(b): mean over states of (RTT(hand) - RTT(ILS)) / RTT(ILS) and the shares, pooled and per bucket; S-regret beside the replay. The replay rules."""
    ok = [r for r in rows if r.get("regret") is not None]
    if len(ok) != len(rows):
        raise ValueError(f"FAIL LOUD: {len(rows) - len(ok)} rows carry no replayed regret")

    def block(rs):
        if not rs:
            return {"n": 0}
        g = sorted(r["regret"] for r in rs)
        s = [r["s_regret"] for r in rs if r.get("s_regret") is not None]
        return {"n": len(rs), "mean_regret_pct": 100 * st.fmean(g), "median_pct": 100 * st.median(g), "p90_pct": 100 * g[min(len(g) - 1, int(0.9 * len(g)))],
                "share_within_1pct": sum(x <= 0.01 for x in g) / len(g), "share_ils_worse_than_hand": sum(x < -1e-9 for x in g) / len(g),
                "mean_s_regret_pct": 100 * st.fmean(s) if s else None,
                "share_s_and_replay_disagree": sum((r["s_regret"] or 0) > 1e-9 and r["regret"] < -1e-9 for r in rs) / len(rs),
                "kill_K1b_mean_ge_1pct": bool(st.fmean(g) >= 0.01)}

    out = {"pooled": block(ok)}
    for name, _lo, _hi in BUCKETS:
        out[name] = block([r for r in ok if r["bucket"] == name])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"); r.add_argument("out")
    r.add_argument("--arm", default="cd_expand", choices=sorted(ARMS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-evals", type=int, default=None); r.add_argument("--evals-per-task-cand", type=int, default=200)
    r.add_argument("--max-seconds", type=float, default=None); r.add_argument("--replay", default="all", choices=("all", "final", "none"))
    d = sub.add_parser("read")
    d.add_argument("rows"); d.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        with open(a.out, "a") as fh:
            for line in open(a.jobs):
                if line.strip():
                    job = json.loads(line)
                    row = run_state(job, a.arm, a.seed, a.max_evals, a.evals_per_task_cand, a.max_seconds, a.replay)
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n"); fh.flush()
                    print(job.get("state_id"), "n", row["n_batch"], "evals", row["n_evals"], "S", round(row["start_S"], 3), "->", round(row["best_S"], 3),
                          "regret", row.get("regret"), "wall", round(row["wall"], 1), flush=True)
        return 0
    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    res = read_rows(rows)
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

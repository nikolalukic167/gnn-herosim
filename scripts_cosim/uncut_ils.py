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
  uncut_ils.py make-jobs SEALED.json CELLS_DIR JOBS.jsonl --token <arm token> --inputs <inputs_group>
  uncut_ils.py make-labels JOBS.jsonl OUT_DIR --sealed k1_sealed_states.json (--k1-read-commit <hash> | --dry-run)   labels refuse until K1 is read
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
ARM_KEYS = ("HEROSIM_PG_CD_EXACT", "HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_BATCH_ALPHA", "HEROSIM_PG_DRAIN_ALPHA", "HEROSIM_PG_CD_EXACT_PRUNE")

# the hand arms a K1 hand plan can come from: (strategy, env of the arm's decision); the capture's own physics env is applied first
ARMS = {
    "cd": ("peer_greedy_network_cd_peer_greedy_network_cd", {}),
    "cd_expand": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXPANSION": "1"}),
    "cd_exactS": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXACT": "1", "HEROSIM_PG_CD_EXPANSION": "1"}),
    "cd_ib0": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_BATCH_ALPHA": "0"}),
    "cdx_ib0": ("peer_greedy_network_cd_peer_greedy_network_cd", {"HEROSIM_PG_CD_EXPANSION": "1", "HEROSIM_PG_BATCH_ALPHA": "0"}),
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
    state = {"iter": 0, "stop": None, "revisits": 0}
    best: Dict[str, Any] = {"S": None, "plan": None}

    class _Stop(Exception):
        pass

    def score(plan: Tuple[Key, ...]) -> float:
        v = cache.get(plan)
        if v is not None:
            state["revisits"] += 1
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
            "stopped": state["stop"], "iters": state["iter"], "revisits": state["revisits"], "wall": time.time() - t0}


def default_budget(cands: Sequence[Sequence[Key]], per: int = 200) -> int:
    """The registered budget: 200 x n x (mean candidates per task) evaluations per group."""
    n = len(cands)
    return int(per * n * (sum(len(c) for c in cands) / max(n, 1)))


# ---- the replay passes (datalab; the scheduler's own S and the fidelity replay) -------------------------------------------------------------

class _Done(Exception):
    def __init__(self, out):
        self.out = out


def _setup_env(arm: str, job: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    cell_summary = job.get("cell_summary")
    strategy, extra = ARMS[arm]
    os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
    if job.get("dataset"):          # the corpus' own physics (generation_provenance.json), as the rollout replay applies it
        prov = json.load(open(Path(job["dataset"]) / "generation_provenance.json"))
        for k, v in (prov.get("physics_env") or {}).items():
            if k.startswith("HEROSIM_") or k == "COSIM_AUTOSCALER_RECONCILE_INTERVAL":
                os.environ[k] = str(v)
    if cell_summary:
        cell_env = json.load(open(cell_summary)).get("env") or {}
        for k, v in cell_env.items():
            if k.startswith("HEROSIM_") or k.startswith("GNN_"):
                os.environ[k] = str(v)
        norm = lambda d: {k: str(v) for k, v in d.items() if k in ARM_KEYS and (str(v) not in ("0", "") or k == "HEROSIM_PG_BATCH_ALPHA")}
        have, want = norm(cell_env), norm(ARMS[arm][1])
        if have != want:
            raise RuntimeError(f"FAIL LOUD: --arm {arm} decides with {want}, the cell's own env says {have}")
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
    strategy, _ = _setup_env(arm, job)
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
                                                int(job.get("exhaustive_cap") or 0), bool(job.get("keep_plans")))
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
    out.update(t0=t0, types=types, spec=spec, snap=snap)
    return out


def _search_at_decision(sched, batch_tasks, system_state, res, local_of, queued, seed, max_evals, per, max_seconds, exhaustive_cap=0, keep_plans=False) -> Dict[str, Any]:
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
    slate_log10 = sum(math.log10(min(sched.pg_cd_exact_top_k, len(c))) for i, c in enumerate(ckeys) if i not in pinned)   # cd_exactS's slate product
    exact_cap_hit = bool(sched.pg_cd_exact and slate_log10 > math.log10(sched.pg_cd_exact_max_plans))
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
    return {"plan_space": math.prod(len(c) for c in ckeys), "n": len(batch_tasks), "n_batch": len(order), "n_pinned": len(pinned), "cands_mean": sum(len(c) for c in ckeys) / len(ckeys),
            "hand": to_batch(start), "start_S": r["start_S"], "best": to_batch(r["best"]), "best_S": r["best_S"], "n_evals": r["n_evals"], "budget": budget,
            "stopped": r["stopped"], "iters": r["iters"], "s_calls": ils_calls, "search_wall": wall_s, "exhaustive": exh,
            "slate_log10": slate_log10, "exact_cap_hit": exact_cap_hit,
            "trajectory": [{"eval": t["eval"], "S": t["S"], "iter": t["iter"], "plan": to_batch(t["plan"])} for t in r["trajectory"]],
            "evals_log": [(e["eval"], e["S"], e["iter"]) for e in r["evals"]], "revisits": r["revisits"],
            "evals_full": [(e["eval"], e["S"], e["iter"], to_batch(e["plan"])) for e in r["evals"]] if keep_plans else None}


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


def check_chosen(snap: Dict[str, Any], hand: Sequence[Sequence[int]], state_id: Any) -> bool:
    """The served plan the capture recorded (`chosen`: {"<gid>": [node id, platform id]} for every batch task) against the replayed re-decision, bit-equal, batch order
    = gid ascending. A mismatch stops the run: nothing is read until it is understood."""
    gids = [int(r["gid"]) for r in sorted(snap["fidelity"]["batch"], key=lambda r: int(r["gid"]))]
    ch = snap["chosen"]
    if not isinstance(ch, dict) or {int(k) for k in ch} != set(gids):
        raise ValueError(f"FAIL LOUD: state {state_id}: snapshot `chosen` is not a map over the batch's gids")
    live = [[int(ch[str(g)][0]), int(ch[str(g)][1])] for g in gids]
    if live != [[int(a), int(b)] for a, b in hand]:
        raise RuntimeError(f"FAIL LOUD: state {state_id}: the replayed re-decision differs from the served plan; "
                           f"{sum(a != list(map(int, b)) for a, b in zip(live, hand))} of {len(gids)} tasks differ")
    return True


def run_state(job: Dict[str, Any], arm: str, seed: int, max_evals: Optional[int], per: int, max_seconds: Optional[float], replay: str) -> Dict[str, Any]:
    t0 = time.time()
    d = decision_pass(job, arm, seed, max_evals, per, max_seconds)
    row = {k: d[k] for k in ("n", "n_batch", "n_pinned", "cands_mean", "start_S", "best_S", "n_evals", "budget", "stopped", "iters", "s_calls", "search_wall",
                             "hand", "best", "t0", "exhaustive", "slate_log10", "exact_cap_hit", "plan_space")}
    row["n_extra_captures"] = int(job.get("n_extra_captures") or 0)
    row["role"], row["parent"] = job.get("role", "first"), job.get("parent")
    row["time_bound"] = d["stopped"] == "seconds"          # the per-state --max-seconds bound, logged where it binds
    row["chosen_match"] = check_chosen(d["snap"], d["hand"], job.get("state_id")) if d["snap"].get("chosen") is not None else None
    row["revisits"], row["evals_full"] = d["revisits"], d["evals_full"]
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


def read_rows(rows: Sequence[Dict[str, Any]], missing: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """K1(b): mean over states of (RTT(hand) - RTT(ILS)) / RTT(ILS) and the shares, pooled and per bucket; S-regret beside the replay. The replay rules.
    The main set is the FIRST capture of every state that has no retry capture. States with a retry (n_extra_captures > 0: the first hand plan left tasks unplaced) are
    excluded from it, listed, and given a descriptive composite (first + retry batches replayed, summed). The mean is printed twice: captured states only, and with every
    missing id imputed at its bucket's maximum observed regret; NOT-SATISFIED if the first is < 1 % and the second >= 1 %."""
    ok = [r for r in rows if r.get("regret") is not None]
    if len(ok) != len(rows):
        raise ValueError(f"FAIL LOUD: {len(rows) - len(ok)} rows carry no replayed regret")
    firsts = [r for r in ok if r.get("role", "first") != "retry"]
    retries = [r for r in ok if r.get("role") == "retry"]
    main = [r for r in firsts if not r.get("n_extra_captures")]
    excluded = [r for r in firsts if r.get("n_extra_captures")]

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

    out = {"pooled": block(main)}
    for name, _lo, _hi in BUCKETS:
        out[name] = block([r for r in main if r["bucket"] == name])
    out["missing"] = {"total": len(missing), **{name: [m["state_id"] for m in missing if m["bucket"] == name] for name, _lo, _hi in BUCKETS}}
    if main:
        pooled_max = max(r["regret"] for r in main)
        imp = []
        for m in missing:
            same = [r["regret"] for r in main if r["bucket"] == m["bucket"]]
            imp.append(max(same) if same else pooled_max)
        mean_cap = st.fmean(r["regret"] for r in main)
        mean_imp = st.fmean([r["regret"] for r in main] + imp)
        out["mean_captured_pct"], out["mean_imputed_pct"] = 100 * mean_cap, 100 * mean_imp
        out["imputed_rule"] = "each missing id at its bucket's maximum observed regret among the main states (pooled maximum if the bucket has none)"
        out["NOT_SATISFIED"] = bool(mean_cap < 0.01 and mean_imp >= 0.01)
    by_parent: Dict[Any, List[Dict[str, Any]]] = {}
    for r in retries:
        by_parent.setdefault(r["parent"], []).append(r)
    comp = []
    for r in excluded:
        rs = by_parent.get(r["state_id"], [])
        if len(rs) == r["n_extra_captures"]:         # composite only when every capture of the state is present
            h, i = r["rtt_hand"] + sum(x["rtt_hand"] for x in rs), r["rtt_ils"] + sum(x["rtt_ils"] for x in rs)
            comp.append({"state_id": r["state_id"], "bucket": r["bucket"], "captures": 1 + len(rs), "composite_regret_pct": 100 * (h - i) / i})
    out["excluded_retry_states"] = {"count": len(excluded), "share_of_captured_states": len(excluded) / len(firsts) if firsts else None,
                                    "flag_gt_10pct": bool(firsts and len(excluded) / len(firsts) > 0.10),
                                    "by_bucket": {name: [r["state_id"] for r in excluded if r["bucket"] == name] for name, _lo, _hi in BUCKETS},
                                    "composite_note": "descriptive: the sum of RTT over the first batch and each retry batch, each replayed as its own snapshot; not in any mean",
                                    "composite": comp}
    return out


# ---- the label driver (dry-run only until K1 is read) ---------------------------------------------------------------------------------------

ILS_VERSION = "uncut_ils/1"
LABEL_FIELDS = ("topology", "window", "rung", "trigger_task_id")


def sealed_keys(sealed: Sequence[Dict[str, Any]]) -> set:
    return {(str(x["topology"]), str(x["window"]), str(x["rung"]), int(x["trigger_task_id"])) for x in sealed}


def label_gate(jobs: Sequence[Dict[str, Any]], sealed: Sequence[Dict[str, Any]], k1_read_commit: Optional[str], dry_run: bool, repo: Path = REPO) -> None:
    """Labels cannot start before K1 is read. Without --k1-read-commit only a --dry-run on identity (m) states (state ids `idm_*`) is allowed; with it, a real commit of
    this repository must be named. A sealed K1(b) state is never labelled, with or without the flag. Every job names (topology, window, rung, trigger_task_id)."""
    import re
    import subprocess

    if k1_read_commit:
        if not re.fullmatch(r"[0-9a-f]{7,40}", k1_read_commit):
            raise ValueError(f"FAIL LOUD: --k1-read-commit {k1_read_commit!r} is not a commit hash")
        if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", k1_read_commit + "^{commit}"], capture_output=True).returncode != 0:
            raise ValueError(f"FAIL LOUD: --k1-read-commit {k1_read_commit} is not a commit of {repo}")
    elif not dry_run:
        raise RuntimeError("FAIL LOUD: labels refuse to run on any corpus state until K1 is read: pass --k1-read-commit <hash of the commit that records the K1 read> "
                           "(or --dry-run on identity (m) states only)")
    ex = sealed_keys(sealed)
    for j in jobs:
        miss = [f for f in LABEL_FIELDS if j.get(f) is None]
        if miss:
            raise ValueError(f"FAIL LOUD: state {j.get('state_id')}: the job lacks {miss}")
        if (str(j["topology"]), str(j["window"]), str(j["rung"]), int(j["trigger_task_id"])) in ex:
            raise RuntimeError(f"FAIL LOUD: state {j.get('state_id')} is a sealed K1(b) state; it is never labelled")
        if not k1_read_commit and not str(j.get("state_id", "")).startswith("idm_"):
            raise RuntimeError(f"FAIL LOUD: state {j.get('state_id')} is not an identity (m) state; a dry run is for identity (m) states only")


def label_header(row: Dict[str, Any], job: Dict[str, Any], seed: int, k1_read_commit: Optional[str], dry_run: bool, sealed_file: Optional[str], sealed_sha: Optional[str],
                 code_commit: Optional[str]) -> Dict[str, Any]:
    """The mandatory first line of a sampled placements.jsonl (docs/notes/placements_jsonl_required.md, 'Sampled files above the plan cap')."""
    return {"header": True, "exhaustive": False, "plan_space_size": row["plan_space"],
            "n_tasks": row["n_batch"], "eval_budget": row["budget"], "evals_done": row["n_evals"], "search": ILS_VERSION, "ils_version": ILS_VERSION, "seed": seed,
            "s_pin": {"surrogate": "peer_greedy_network _pg_plan_cost", "batch_alpha": 0.0, "drain_alpha": 1.0, "arm": row["arm"], "code_commit": code_commit},
            "cap": CAP, "stopped": row["stopped"], "revisits_not_rescored": row["revisits"], "distinct_plans_only": True,
            "replayed_rows": row["n_incumbents"], "replayed_rows_are": "the hand plan (start) and each new ILS incumbent; null elsewhere; "
            "the top-8 by S are replayed only if K1(b) fires (not done here)", "top8_replayed": False, "state_id": row["state_id"], "bucket": row.get("bucket"),
            "sealed_exclusion": {"file": sealed_file, "sha256": sealed_sha}, "k1_read_commit": k1_read_commit, "dry_run": dry_run,
            "hand_plan_source": "the served plan (`chosen`), bit-equal to the re-decision" if row.get("chosen_match") else "replayed re-decision (the snapshot carries no `chosen`)"}


def label_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per ILS-evaluated plan: iteration index, S, placement_plan (batch order), and the replayed RTT on the hand plan and on new incumbents only."""
    rtt_at = {1: row.get("rtt_hand")}
    for t, v in zip(row["trajectory"][1:], row.get("rtt_incumbents") or []):
        rtt_at[t["eval"]] = v
    out = []
    for ev, sv, it, plan in row["evals_full"]:
        out.append({"eval": ev, "iter": it, "S": sv, "placement_plan": {str(i): [int(a), int(b)] for i, (a, b) in enumerate(plan)}, "rtt": rtt_at.get(ev),
                    "replayed": ev in rtt_at})
    return out


def write_labels(out_dir: Path, header: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Path:
    d = out_dir / str(header["state_id"]).replace("/", "_") / "placements"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "placements.jsonl"
    with open(f, "w") as fh:
        fh.write(json.dumps(header, separators=(",", ":")) + "\n")
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return f


def _is_missing(st_: Dict[str, Any], missing: Sequence[Any]) -> bool:
    for m in missing:
        if isinstance(m, str):
            if m == st_["state_id"]:
                return True
        elif all(str(m[k]) == str(st_[k]) for k in m if k in st_) and "trigger_task_id" in m:
            return True
    return False


def captures_of(snapfile: str, trigger: int) -> List[Dict[str, Any]]:
    """Every capture in the file led by `trigger`, in file order (capture order); the K1(b) state is the first."""
    out = []
    for line in open(snapfile):
        if line.strip():
            rec = json.loads(line)
            if int(rec["trigger_task_id"]) == int(trigger):
                out.append({"time": float(rec["time"]), "n_batch": len(rec["fidelity"]["batch"])})
    return out


def make_jobs(sealed: Sequence[Dict[str, Any]], cells: Path, token: str, inputs: Path, sim_input: str, seed: int = 0,
              missing: Sequence[Any] = ()) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(jobs, missing states). One job per sealed state from a hand-cell directory: cells/cc40s<topo>__<window><rung>__<token>_s<seed>.{snapshots.jsonl,summary.json}; the
    config is inputs/wf1_<rung>/cfg/cc40s<topo>.json. The state is the FIRST capture led by its trigger id (pinned by its time); later same-trigger captures (a sub-batch
    0.01 s after the whole batch) are dropped and counted in `n_extra_captures`. A state listed in `missing` (state ids, or {topology, window, rung, trigger_task_id}
    dicts: ids that were never a trigger at k = 1) is skipped and returned as missing with its bucket. A state the cell's own summary lists in `sealed_snapshots.missing` is skipped the same way. Any other missing file or trigger fails loud."""
    jobs, gone = [], []
    for st_ in sealed:
        if _is_missing(st_, missing):
            gone.append({k: st_[k] for k in ("state_id", "bucket", "group_size", "topology", "window", "rung", "trigger_task_id")})
            continue
        stem = cells / f"cc40s{st_['topology']}__{st_['window']}{st_['rung']}__{token}_s{seed}"
        cfg = inputs / f"wf1_{st_['rung']}" / "cfg" / f"cc40s{st_['topology']}.json"
        for f in (Path(str(stem) + ".snapshots.jsonl"), Path(str(stem) + ".summary.json"), cfg):
            if not f.exists():
                raise FileNotFoundError(f"FAIL LOUD: state {st_['state_id']}: {f} does not exist")
        caps = captures_of(str(stem) + ".snapshots.jsonl", st_["trigger_task_id"])
        cell_missing = [int(x) for x in ((json.load(open(str(stem) + ".summary.json")).get("sealed_snapshots") or {}).get("missing") or [])]
        if not caps and int(st_["trigger_task_id"]) in cell_missing:       # the cell's own record: this id was never a trigger there
            gone.append({k: st_[k] for k in ("state_id", "bucket", "group_size", "topology", "window", "rung", "trigger_task_id")})
            continue
        if not caps:
            raise FileNotFoundError(f"FAIL LOUD: state {st_['state_id']}: trigger {st_['trigger_task_id']} was never captured in {stem}.snapshots.jsonl and is not listed as missing")
        jobs.append({"state_id": st_["state_id"], "bucket": st_["bucket"], "group_size": st_["group_size"], "snapfile": str(stem) + ".snapshots.jsonl",
                     "trigger_task_id": st_["trigger_task_id"], "topology": st_["topology"], "window": st_["window"], "rung": st_["rung"], "time": caps[0]["time"], "n_extra_captures": len(caps) - 1, "first_capture_n": caps[0]["n_batch"],
                     "cell_summary": str(stem) + ".summary.json", "cell_config": str(cfg), "sim_input": sim_input, "role": "first"})
        for k, cap in enumerate(caps[1:], 1):       # descriptive only: a retry capture is its own batch, replayed for the composite and never in the main mean
            jobs.append({**jobs[-k], "state_id": f"{st_['state_id']}#r{k}", "role": "retry", "parent": st_["state_id"], "time": cap["time"],
                         "n_extra_captures": 0, "first_capture_n": cap["n_batch"]})
    return jobs, gone


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("jobs"); r.add_argument("out")
    r.add_argument("--arm", default="cd_expand", choices=sorted(ARMS)); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-evals", type=int, default=None); r.add_argument("--evals-per-task-cand", type=int, default=200)
    r.add_argument("--max-seconds", type=float, default=300.0, help="per-state bound on the ILS search; logged as time_bound where it binds"); r.add_argument("--replay", default="all", choices=("all", "final", "none"))
    lb = sub.add_parser("make-labels", help="the label driver: a headed, sampled placements.jsonl per state; refuses until --k1-read-commit (or --dry-run on identity (m) states)")
    lb.add_argument("jobs"); lb.add_argument("out_dir"); lb.add_argument("--sealed", required=True, help="k1_sealed_states.json: its states are never labelled")
    lb.add_argument("--k1-read-commit", default=None); lb.add_argument("--dry-run", action="store_true")
    lb.add_argument("--arm", default="cd_expand", choices=sorted(ARMS)); lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--max-evals", type=int, default=None); lb.add_argument("--evals-per-task-cand", type=int, default=200)
    lb.add_argument("--max-seconds", type=float, default=300.0)
    m = sub.add_parser("make-jobs")
    m.add_argument("sealed"); m.add_argument("cells"); m.add_argument("out"); m.add_argument("--token", required=True, help="the hand cell's arm token, e.g. cd_ib0_k100")
    m.add_argument("--inputs", required=True, help="gate inputs_group dir"); m.add_argument("--sim-input", default=str(REPO / "data" / "nofs-ids"))
    m.add_argument("--only", default=None, help="comma list of state ids (a shard)")
    m.add_argument("--missing", default=None, help="JSON list of sealed states that were never a trigger at k = 1 (state ids or {topology, window, rung, trigger_task_id})")
    d = sub.add_parser("read")
    d.add_argument("rows"); d.add_argument("--json", default=None); d.add_argument("--missing", default=None, help="the <jobs>.missing.json make-jobs wrote")
    a = ap.parse_args()
    if a.cmd == "make-labels":
        import hashlib
        import subprocess

        jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
        sealed = json.load(open(a.sealed))
        label_gate(jobs, sealed, a.k1_read_commit, a.dry_run)
        sha = hashlib.sha256(Path(a.sealed).read_bytes()).hexdigest()
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        for job in jobs:
            job = dict(job, keep_plans=True)
            row = run_state(job, a.arm, a.seed, a.max_evals, a.evals_per_task_cand, a.max_seconds, "all")
            f = write_labels(Path(a.out_dir), label_header(row, job, a.seed, a.k1_read_commit, a.dry_run, a.sealed, sha, head), label_rows(row))
            print(job["state_id"], "n", row["n_batch"], "evals", row["n_evals"], "replayed", row["n_replays"], "->", f, flush=True)
        return 0
    if a.cmd == "make-jobs":
        sealed = json.load(open(a.sealed))
        if a.only:
            keep = set(a.only.split(","))
            sealed = [x for x in sealed if x["state_id"] in keep]
        jobs, gone = make_jobs(sealed, Path(a.cells), a.token, Path(a.inputs), a.sim_input, missing=json.load(open(a.missing)) if a.missing else ())
        Path(a.out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
        Path(a.out + ".missing.json").write_text(json.dumps(gone, indent=1))
        print(f"{len(jobs)} jobs -> {a.out}; {len(gone)} missing -> {a.out}.missing.json; extra captures dropped: {sum(j['n_extra_captures'] for j in jobs)}")
        return 0
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
    res = read_rows(rows, json.load(open(a.missing)) if a.missing else ())
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

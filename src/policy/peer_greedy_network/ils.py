"""Iterated local search on a whole-plan score S, for one batch (uncut_groups_s0_v1). Pure: S is a callable, so it is testable without a simulator.

Shared by `scripts_cosim/uncut_ils.py` (the label engine) and the live arm `cd_ib0_ils` (`PeerGreedyNetworkCDScheduler._pg_ils`), so the searched
labels and the served search are one implementation.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CAP = 256
EPS = 1e-9
Key = Tuple[int, int]


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

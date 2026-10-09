"""r1_attribution_v1: the node's DECLARED candidate pruning, one definition for corpus build and for serving.

The node declares, before any label exists: each task keeps its top 5 candidates ranked by exact standalone cost; if the batch's
pruned plan space still exceeds 100,000 plans it is split into sub-batches of at most 4 tasks. The corpus builder
(scripts_cosim/make_warm_corpus.py --fidelity) and the serving mode (GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1,
GNNScheduler._prefix_inference) both call `slate` below on the same payload fields, so a dataset's candidate sets are the sets a
served batch sees.

Standalone cost is the peer-greedy / CD base score of one task with no placed partner, `_pg_choose`'s
`base = drain + cold + exec_s + lat` (src/policy/peer_greedy_network/scheduler.py:283; `exch` = 0 for a task alone), written over
the audit-snapshot candidate payload (`live_audit._candidate_payload`, the dict the corpus is built from):

    drain = queue_drain_seconds   platform_queue_drain_seconds(platform, ...)   scheduler.py:272, live_audit.py:213
    cold  = 0 if initialized else cold_start_time                              scheduling_cost.py:395-399 (the snapshot form of
                                                                                incoming_cold_start_time, scheduling_cost.py:71)
    exec  = execution_time        task.type["executionTime"][platform type]    scheduler.py:280 (xf = 1 at policy time scale 1)
    lat   = network_latency       network_latency_between(task.node_name, ...) scheduler.py:282, scheduling_cost.py:86

Ties are broken by (node_id, platform_id), as `_pg_choose` breaks them (scheduler.py:294).
"""
from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Tuple

RULE = "declared_pruning_v1"
TOP_K = 5
MAX_PLANS = 100_000
SUB_BATCH = 4
MAX_PLANS_ENV = "DECLARED_SLATE_MAX_PLANS"  # test hook: lowers the 100,000 so a small batch exercises sub-batching; never set in a run


def max_plans() -> int:
    raw = os.environ.get(MAX_PLANS_ENV, "").strip()
    return int(raw) if raw else MAX_PLANS


def standalone_cost(candidate: Mapping[str, Any]) -> float:
    cold = 0.0 if candidate.get("initialized", True) else float(candidate.get("cold_start_time", 0.0) or 0.0)
    return (float(candidate.get("queue_drain_seconds", 0.0) or 0.0) + cold
            + float(candidate.get("execution_time", 0.0) or 0.0) + float(candidate.get("network_latency", 0.0) or 0.0))


def prune_candidates(candidates: Sequence[Mapping[str, Any]], k: int = TOP_K) -> List[Mapping[str, Any]]:
    """The task's k cheapest candidates by standalone cost, deterministic tie-break, in rank order."""
    ranked = sorted(candidates, key=lambda c: (standalone_cost(c), int(c["node_id"]), int(c["platform_id"])))
    return ranked[:k]


def plan_space(sizes: Iterable[int]) -> int:
    n = 1
    for s in sizes:
        n *= int(s)
    return n


def sub_batches(task_ids: Sequence[int], pairs: Iterable[Tuple[int, int]], size: int = SUB_BATCH) -> List[List[int]]:
    """Consecutive runs of a breadth-first walk of the peer graph (neighbours in id order, a new walk from the smallest unvisited id),
    cut every `size` tasks; each group is returned in ascending id order. Related tasks tend to share a group; the walk and the cut
    are functions of the ids and the pairs alone, so build and serving cut the same batch the same way."""
    ids = sorted(int(t) for t in task_ids)
    adj: Dict[int, List[int]] = {t: [] for t in ids}
    for a, b in pairs:
        a, b = int(a), int(b)
        if a in adj and b in adj and a != b:
            adj[a].append(b)
            adj[b].append(a)
    order: List[int] = []
    seen = set()
    for start in ids:
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        while queue:
            t = queue.popleft()
            order.append(t)
            for n in sorted(set(adj[t])):
                if n not in seen:
                    seen.add(n)
                    queue.append(n)
    return [sorted(order[i:i + size]) for i in range(0, len(order), size)]


class Slate(NamedTuple):
    kept: List[List[Mapping[str, Any]]]   # per task (input order): the pruned candidates, in rank order
    full_sizes: List[int]                 # per task: candidates before pruning
    plans: int                            # product of the pruned sizes over the WHOLE batch
    groups: List[List[int]]               # task positions per decision unit: one group = the batch, several = sub-batches
    pruned: bool                          # some task had more than TOP_K candidates
    sub_batched: bool                     # the pruned plan space exceeded MAX_PLANS


def slate(tasks: Sequence[Mapping[str, Any]], pairs: Iterable[Tuple[int, int]] = ()) -> Slate:
    """`tasks`: [{task_id, candidates: [payload dicts with node_id, platform_id, ...]}] in any fixed order; `pairs`: the batch's
    peer pairs over task_id."""
    kept = [prune_candidates(t["candidates"]) for t in tasks]
    full = [len(t["candidates"]) for t in tasks]
    plans = plan_space(len(k) for k in kept)
    if plans <= max_plans():
        groups = [list(range(len(tasks)))]
    else:
        pos = {int(t["task_id"]): i for i, t in enumerate(tasks)}
        groups = [[pos[g] for g in grp] for grp in sub_batches(list(pos), pairs)]
    return Slate(kept, full, plans, groups, any(f > TOP_K for f in full), len(groups) > 1)


# ---- serving mode: recorded in the checkpoint's sidecar, refused on mismatch -------------------------------------------------
ENV = "GNN_SERVE_CANDIDATE_SLATE"


def serving_slate() -> str | None:
    """The candidate-slate rule this process serves under: RULE, or None (every reachable replica, as before this mode)."""
    raw = os.environ.get(ENV, "").strip()
    if raw in ("", "0", "none"):
        return None
    if raw != RULE:
        raise ValueError(f"FAIL LOUD: {ENV}={raw!r}; expected unset or {RULE!r}")
    return raw


def require_matching_slate(trained: str | None, *, what: str) -> None:
    """A checkpoint trained on a pruned slate must be served over it and a checkpoint trained without one must not be: the candidate
    set is part of what the weights were fitted on."""
    now = serving_slate()
    if (trained or None) != now:
        raise ValueError(
            f"{what}: trained with candidate_slate={trained!r} but this run serves with {ENV}={now!r}; export {ENV}="
            f"{trained or 'unset'} to match (the candidate set is part of the training distribution)")

"""plan_rerank_v1 Phase 1: the candidate pool's GNN quantities, one definition for the offline pool (train_near_rtt's
NEAR_RTT_EVAL_VARIANTS / NEAR_RTT_EVAL_EXTRA_PLANS dump) and for serving (src/policy/plan_rerank/live.py).

`score_fn(t, committed)` is the base GNN's per-step scorer (make_partial_state_score_fn) and `tl[t]` the task's candidate list in
logit order, both over the same graph. The decode order is masked_topo's (seq_decode.topological_task_order) and the greedy tie rule
is the registered one: the lowest (node, platform) among equal logits.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence, Tuple

Couple = Tuple[int, int]
ScoreFn = Callable[[int, Mapping[int, Couple]], Sequence[float]]


def _step_logits(score_fn: ScoreFn, t: int, prefix: Mapping[int, Couple]) -> List[float]:
    return [float(v) for v in score_fn(t, dict(prefix))]


def greedy_from(score_fn: ScoreFn, tl: Sequence[Sequence[Couple]], order: Sequence[int], prefix: Mapping[int, Couple],
                start: int, acc: float) -> Tuple[Dict[int, Couple], float]:
    prefix = dict(prefix)
    for t in order[start:]:
        lg = _step_logits(score_fn, t, prefix)
        i = min(range(len(lg)), key=lambda j: (-lg[j], tl[t][j]))
        prefix[t] = tl[t][i]
        acc += lg[i]
    return prefix, acc


def step0_branches(score_fn: ScoreFn, tl: Sequence[Sequence[Couple]], order: Sequence[int]) -> List[Tuple[Tuple[Couple, ...], float]]:
    """Per candidate of the first task in decode order (candidate order): the plan the greedy decode completes from it and the sum
    of the step logits along that plan under its own prefix."""
    n = len(tl)
    t0 = order[0]
    lg0 = _step_logits(score_fn, t0, {})
    out = []
    for i, c in enumerate(tl[t0]):
        prefix, acc = greedy_from(score_fn, tl, order, {t0: c}, 1, lg0[i])
        out.append((tuple(prefix[t] for t in range(n)), acc))
    return out


def teacher_forced_sum(score_fn: ScoreFn, tl: Sequence[Sequence[Couple]], order: Sequence[int], plan: Sequence[Couple]) -> float:
    """The summed step logits of `plan` along the decode order, each step under the plan's own prefix."""
    acc, prefix = 0.0, {}
    for t in order:
        acc += _step_logits(score_fn, t, prefix)[list(tl[t]).index(tuple(plan[t]))]
        prefix[t] = tuple(plan[t])
    return acc


def plan_score(score_fn: ScoreFn, tl: Sequence[Sequence[Couple]], plan: Sequence[Couple]) -> float:
    """gnn_selfsearch's plan score: the sum over tasks (index order) of the logit of the task's choice, every other task committed."""
    n = len(tl)
    total = 0.0
    for t in range(n):
        logits = score_fn(t, {j: tuple(plan[j]) for j in range(n) if j != t})
        total += float(logits[list(tl[t]).index(tuple(plan[t]))])
    return total

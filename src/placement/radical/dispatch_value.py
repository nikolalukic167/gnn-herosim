"""Counterfactual final-cost labels for feasible ready-set actions."""
import numpy as np

from src.placement.radical.dispatch_state import trace_priority


def trajectory_values(engine, problem, assignment, priority):
    realized, states = trace_priority(problem, assignment, priority)
    order = [state["target"] for state in states]
    baseline = engine.score_priority(problem, assignment, realized)[0]
    values = np.full((len(states), assignment.size), np.nan, dtype=np.float32)
    for step, state in enumerate(states):
        prefix = order[:step]
        used = set(prefix)
        for candidate in np.flatnonzero(state["feasible"]):
            remaining = [index for index in order if index not in used and index != candidate]
            proposed = [*prefix, int(candidate), *remaining]
            rank = np.empty(assignment.size, dtype=np.int64)
            rank[np.asarray(proposed)] = np.arange(assignment.size)
            values[step, candidate] = engine.score_priority(
                problem, assignment, rank.reshape(assignment.shape))[0]
        if values[step, order[step]] != baseline:
            raise ValueError("counterfactual continuation does not preserve baseline")
    return realized, states, values

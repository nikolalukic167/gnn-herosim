import numpy as np

from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_state import DYNAMIC_FEATURES, materialize, trace_priority
from src.placement.radical.environment import initial, problem


def test_priority_trace_preserves_behavior(tmp_path):
    physical = problem(132000)
    engine = DispatchNative(tmp_path)
    assignment = initial(physical, "affinity")
    priority = priorities(physical, assignment, "srpt")
    expected, _ = engine.score_priority(physical, assignment, priority)
    realized, states = trace_priority(physical, assignment, priority)
    actual, _ = engine.score_priority(physical, assignment, realized)
    assert actual == expected
    assert len(states) == assignment.size
    assert all(state["dynamic"].shape == (assignment.size, DYNAMIC_FEATURES)
               for state in states)
    assert all(state["feasible"][state["target"]] for state in states)


def test_materializer_rejects_infeasible_choice():
    physical = problem(132001)
    assignment = initial(physical, "affinity")
    def wrong(dynamic, feasible):
        return int(np.flatnonzero(~feasible)[0])
    try:
        materialize(physical, assignment, wrong)
    except ValueError as error:
        assert "infeasible" in str(error)
    else:
        raise AssertionError("invalid choice was accepted")

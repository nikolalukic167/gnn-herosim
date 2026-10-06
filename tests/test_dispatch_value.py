import numpy as np

from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_value import trajectory_values
from src.placement.radical.environment import initial, problem


def test_counterfactual_values_preserve_baseline_and_cover_candidates(tmp_path):
    physical = problem(135000, jobs=3, ops=3)
    assignment = initial(physical, "affinity")
    priority = priorities(physical, assignment, "srpt")
    engine = DispatchNative(tmp_path)
    realized, states, values = trajectory_values(engine, physical, assignment, priority)
    baseline = engine.score_priority(physical, assignment, realized)[0]
    assert values.shape == (assignment.size, assignment.size)
    for step, state in enumerate(states):
        assert np.array_equal(np.isfinite(values[step]), state["feasible"])
        assert values[step, state["target"]] == baseline
        assert np.isfinite(values[step]).any()

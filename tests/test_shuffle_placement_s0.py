import numpy as np
import pytest

from scripts_cosim.shuffle_placement_s0 import maxmin_rates, replay, route


def network():
    links = {f"{d}:{h}": 100.0 for d in ("in", "out", "memory") for h in range(4)}
    links.update({"core:0": 100.0, "core:1": 100.0})
    return {"capacities": links, "mapper_hosts": [0, 2, 1, 3], "speed": [1.0] * 4, "latency": 0.0}


def test_maxmin_reallocates_after_other_bottleneck():
    rates = maxmin_rates([("a", "b"), ("b",), ("c",)], {"a": 1.0, "b": 10.0, "c": 7.0})
    np.testing.assert_allclose(rates, [1.0, 9.0, 7.0])


def test_shared_receiver_bottleneck_and_release_time():
    trace = {"bytes": [[100], [100]], "ready_s": [0.0, 0.0], "reduce_s": [0.5]}
    assert replay([(0, trace, (1,))], network()) == pytest.approx([2.5])
    trace["ready_s"] = [0.0, 3.0]
    assert replay([(0, trace, (1,))], network()) == pytest.approx([4.5])


def test_ready_after_all_inputs_and_compute_serialization():
    trace = {"bytes": [[0, 0]], "ready_s": [2.0], "reduce_s": [3.0, 4.0]}
    assert replay([(0, trace, (1, 1))], network()) == pytest.approx([9.0])
    assert replay([(0, trace, (1, 3))], network()) == pytest.approx([6.0])


def test_overlapping_jobs_compete_and_are_deterministic():
    trace = {"bytes": [[100]], "ready_s": [0.0], "reduce_s": [0.0]}
    jobs = [(0, trace, (1,)), (0.5, trace, (1,))]
    assert replay(jobs, network()) == pytest.approx([1.5, 1.5])
    assert replay(jobs, network()) == replay(jobs, network())


def test_local_and_core_routes_are_distinct():
    assert route(0, 0) == ("memory:0",)
    assert route(0, 1) == ("out:0", "in:1")
    assert route(0, 2) == ("out:0", "in:2", "core:0")
    assert route(2, 0)[-1] == "core:1"


def test_link_capacity_conservation():
    paths = [("a", "b"), ("a",), ("b", "c"), ("c",)]
    caps = {"a": 10, "b": 7, "c": 13}
    rates = maxmin_rates(paths, caps)
    for link, cap in caps.items():
        assert sum(rate for path, rate in zip(paths, rates) if link in path) <= cap + 1e-10

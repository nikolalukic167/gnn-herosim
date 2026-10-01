import pytest

from scripts_cosim.shuffle_network_gate import flow_seconds, select
from test_shuffle_placement_s0 import network


def test_validation_flow_sharing():
    assert flow_seconds([("a",), ("a",)], [100, 100], {"a": 100}) == pytest.approx(2)
    assert flow_seconds([("a",), ("b",)], [100, 100], {"a": 100, "b": 100}) == pytest.approx(1)


def test_adaptive_choice_responds_to_observed_traffic():
    topo = network()
    topo["capacities"] = {k: 1000 if k.startswith("memory") else 100 for k in topo["capacities"]}
    trace = {"bytes": [[100, 100, 100, 100]] * 4, "ready_s": [0] * 4, "reduce_s": [0.01] * 4}
    snapshot = {"progress": {}, "completed": {}, "observed_at": 1.0}
    history = [{"id": "old", "trace": {**trace, "bytes": [[10000] * 4] * 4},
                "plan": [0] * 4, "data_ready": [0] * 4}]
    static, _ = select(trace, topo, history, snapshot, "static_route", 16)
    adaptive, active = select(trace, topo, history, snapshot, "adaptive_route", 16)
    assert active == 1
    assert adaptive != static
    assert sum(h == 0 for h in adaptive) < sum(h == 0 for h in static)


def test_completed_jobs_do_not_enter_forecast():
    trace = {"bytes": [[10] * 4] * 4, "ready_s": [0] * 4, "reduce_s": [0.01] * 4}
    history = [{"id": "old", "trace": trace, "plan": [0, 1, 2, 3], "data_ready": [0] * 4}]
    snapshot = {"observed_at": 10, "progress": {f"job:old:{m}:{r}": {"received": 10} for m in range(4) for r in range(4)},
                "completed": {f"job:old:{r}": {} for r in range(4)}}
    _, active = select(trace, network(), history, snapshot, "adaptive_search", 16)
    assert active == 0

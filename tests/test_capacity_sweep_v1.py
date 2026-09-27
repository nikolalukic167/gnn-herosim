"""capacity_sweep_v1 (docs/lineages/capacity_sweep_v1.md): the keep_alive flag, the drift statistic, the knee."""
import importlib
import os
import sys

import pytest

from src.placement.constants import KEEP_ALIVE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))


def _es():
    return importlib.import_module("src.executesimulation")


def test_keep_alive_default_is_bit_identical(monkeypatch):
    monkeypatch.delenv("HEROSIM_KEEP_ALIVE", raising=False)
    es = _es()
    v = es._resolve_keep_alive(1.0)
    assert v == KEEP_ALIVE and type(v) is type(KEEP_ALIVE)
    assert es._resolve_keep_alive(0.5) == KEEP_ALIVE * 0.5


def test_keep_alive_override_and_refusals(monkeypatch):
    es = _es()
    monkeypatch.setenv("HEROSIM_KEEP_ALIVE", "1e9")
    assert es._resolve_keep_alive(1.0) == 1e9
    assert es._resolve_keep_alive(2.0) == 2e9
    for bad in ("0", "-3", "nan", "x"):
        monkeypatch.setenv("HEROSIM_KEEP_ALIVE", bad)
        with pytest.raises(ValueError):
            es._resolve_keep_alive(1.0)


def test_keep_alive_is_in_provenance():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "executesimulation.py")).read()
    assert '"HEROSIM_KEEP_ALIVE",' in src


def test_queue_drift_quarters():
    import fresh_topo_burst_v1_gate as G
    rows = [{"taskId": i, "dispatchedTime": float(i), "queueTime": float(i // 25)} for i in range(100)]
    rows.append({"taskId": -1, "dispatchedTime": 0.0, "queueTime": 1e6})
    d = G.queue_drift(rows)
    assert d["quarter_mean_queue"] == [0.0, 1.0, 2.0, 3.0]
    assert d["last_over_first"] is None
    rows = [{"taskId": i, "dispatchedTime": float(i), "queueTime": 1.0 + (i >= 75)} for i in range(100)]
    assert G.queue_drift(rows)["last_over_first"] == 2.0
    assert G.queue_drift([]) is None


def test_capacity_phase_tasks():
    import fresh_topo_burst_v1_gate as G
    tasks = G.tasks_for("capacity", {"topologies": list(range(12))})
    assert len(tasks) == 12 * (3 * 4 * 3 + 4 * 2 + 4 * 4)
    for t in tasks:
        assert t["window"] in G.LADDER_RUNGS
    ka = [t for t in tasks if t["window"] in G.KA_WINDOWS]
    assert {t["kind"] for t in ka} == {"reactive", "cd", "selfpredict", "xs1load_selfref"}
    assert all(G.LADDER_RUNGS[w] == f"x10d{w[6]}" for w in G.KA_WINDOWS)


def test_knee_capacity_index_sign_test():
    import capacity_sweep_v1_read as R
    assert R.knee([0.5, 0.6, 0.7, 0.9, 0.95]) == 1.2
    assert R.knee([0.5, 0.9, 0.7, 0.7, 0.7]) == 1.0
    assert R.knee([0.85, 0.5, 0.5, 0.5, 0.5]) == 0.0
    assert R.knee([0.5] * 5) == 1.5
    assert R.knee([0.5, None, 0.5, 0.5, 0.5]) is None
    assert R.capacity_index([0.5, 0.7, 0.9, 0.95, 0.99]) == pytest.approx(1.15)
    assert R.capacity_index([0.5] * 5) == 1.5
    assert R.sign_test(8, 0) == pytest.approx(2 / 256)
    assert R.sign_test(0, 0) == 1.0

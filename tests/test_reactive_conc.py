"""reactive_conc (HEROSIM_KN_CONC=1): the least-connected key counts queue + compute-lock waiters + running. Off by default (the registered key)."""
import os
import sys
from types import SimpleNamespace

import pytest

from src.policy.knative_network.scheduler import KnativeScheduler


def _platform(pid, queue=0, waiters=0, running=0):
    return SimpleNamespace(id=pid, queue=SimpleNamespace(items=[0] * queue), initialized=SimpleNamespace(triggered=True),
                           compute_lock=SimpleNamespace(queue=[0] * waiters, users=[0] * running))


def _pick(monkeypatch, conc, platforms):
    if conc is None:
        monkeypatch.delenv("HEROSIM_KN_CONC", raising=False)
    else:
        monkeypatch.setenv("HEROSIM_KN_CONC", conc)
    s = object.__new__(KnativeScheduler)
    s._kn_conc = os.environ.get("HEROSIM_KN_CONC", "0") == "1"
    nodes = [SimpleNamespace(id=i, node_name=f"n{i}") for i in range(len(platforms))]
    reps = list(zip(nodes, platforms))
    s._get_valid_replicas = lambda replicas, task: sorted(replicas, key=lambda r: (r[0].id, r[1].id))
    state = SimpleNamespace(replicas={"cnn": reps})
    gen = s.placement(state, SimpleNamespace(type={"name": "cnn"}, id=0))
    try:
        next(gen)
    except StopIteration as stop:
        return stop.value[1].id
    raise AssertionError("placement did not return")


def test_flag_off_is_the_blind_key_and_ties_to_the_lowest_id(monkeypatch):
    # replica 0 holds 50 lock waiters, replica 1 none; the blind key sees both queues empty and takes the lowest id
    assert _pick(monkeypatch, None, [_platform(10, waiters=50), _platform(11)]) == 10


def test_flag_on_counts_waiters_and_running(monkeypatch):
    assert _pick(monkeypatch, "1", [_platform(10, waiters=50), _platform(11)]) == 11
    assert _pick(monkeypatch, "1", [_platform(10, running=1), _platform(11, queue=1, running=1)]) == 10
    assert _pick(monkeypatch, "1", [_platform(10, running=1), _platform(11, running=1)]) == 10   # tie: lowest identity


def test_bad_value_fails_loudly(monkeypatch):
    from src.placement.scheduler import Scheduler

    monkeypatch.setattr(Scheduler, "__init__", lambda self, *a, **k: None)
    monkeypatch.setenv("HEROSIM_KN_CONC", "yes")
    with pytest.raises(ValueError, match="HEROSIM_KN_CONC"):
        KnativeScheduler()
    monkeypatch.setenv("HEROSIM_KN_CONC", "1")
    assert KnativeScheduler()._kn_conc is True
    monkeypatch.delenv("HEROSIM_KN_CONC")
    assert KnativeScheduler()._kn_conc is False


def test_harness_serves_reactive_conc_and_the_flag_is_in_provenance():
    import inspect
    from src import executesimulation

    sys.path.insert(0, "scripts_cosim")
    import fresh_topo_burst_v1_gate as G

    assert G.RULE_POLICY["reactive_conc"] == "knative_network" and "reactive_conc" in G.R1A_DIAG and "reactive_conc" in G.R1A_ARMS
    assert '"HEROSIM_KN_CONC"' in inspect.getsource(executesimulation.build_run_provenance)
    src = open(G.__file__).read()
    assert 'if kind == "reactive_conc":\n            env["HEROSIM_KN_CONC"] = "1"' in src and '"HEROSIM_KN_CONC", "GNN_SLATE_NO_SPLIT"' in src

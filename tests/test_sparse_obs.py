"""sparse_obs_s0_v1: the observation flags degrade what the scheduler reads and nothing else; flag-off is the registered arithmetic."""
import math
from types import SimpleNamespace

import pytest

from src.placement.live_audit import platform_queue_drain_seconds
from src.placement.live_snapshot_seed import _approx_comm
from src.placement.scheduling_cost import incoming_cold_start_time, network_latency_between
from src.policy.peer_greedy_network import observation as O
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler

TYPES = {
    "cnn": {"name": "cnn", "executionTime": {"gpu": 2.0, "cpu": 8.0}, "coldStartDuration": {"gpu": 5.0, "cpu": 1.0},
            "stateSize": {"app": {"input": 100 * 1024 * 1024, "output": 0}}},
    "svm": {"name": "svm", "executionTime": {"gpu": 1.0, "cpu": 3.0}, "coldStartDuration": {"gpu": 4.0, "cpu": 1.0},
            "stateSize": {"app": {"input": 0, "output": 0}}},
}


def _task(i, name, node="client0"):
    return SimpleNamespace(id=i, type=TYPES[name], node_name=node)


def _platform(nodename, pid, ptype, queue=(), current=None, previous=None):
    node = SimpleNamespace(id=pid, node_name=nodename, network_map={"client0": 0.5})
    p = SimpleNamespace(id=pid, node=node, type={"shortName": ptype}, queue=SimpleNamespace(items=list(queue)), virtual_warmup_total_time=0.0,
                        current_task=current, previous_task=previous, inflight_service_end=None,
                        env=SimpleNamespace(now=0.0), initialized=SimpleNamespace(triggered=True))
    p._payload_transfer_time = lambda other, b: b / 1e8
    p.peer_link_latency = lambda other, context="": 0.01
    node.platforms = [p]
    return node, p


def _orch(table=None, tasks=()):
    return SimpleNamespace(peer_exchange=table or {}, task_by_id={t.id: t for t in tasks}, data=SimpleNamespace(task_types=TYPES))


def _sched(monkeypatch, **env):
    for k in O.OBS_ENVS + ("HEROSIM_PG_INFLIGHT", "HEROSIM_PG_COLD_TAIL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    s = object.__new__(PeerGreedyNetworkCDScheduler)
    s._pg_init()
    return s


def _scene():
    q = [_task(10, "cnn"), _task(11, "svm"), _task(12, "svm")]
    n1, p1 = _platform("n1", 1, "gpu", queue=q)
    n2, p2 = _platform("n2", 2, "cpu", queue=[])
    return n1, p1, n2, p2


def test_no_flag_means_no_observation(monkeypatch):
    s = _sched(monkeypatch)
    assert s._obs is None
    assert (s.obs_drain_degraded, s.obs_partner_hidden) == (0, 0)


def test_flag_off_matches_the_registered_terms(monkeypatch):
    s = _sched(monkeypatch)
    n1, p1, n2, p2 = _scene()
    orch = _orch()
    for p in (p1, p2):
        assert s._pg_drain(p, orch, {}, 1.0) == platform_queue_drain_seconds(p, orch, {}, exec_scale=1.0)
        assert s._pg_running(p) == 0.0
    t = _task(1, "cnn")
    assert s._pg_exec_table(t.type, "gpu") == 2.0 and s._pg_exec_table(t.type, "cpu") == 8.0
    node, plat, svc = s._pg_choose(t, [(n1, p1), (n2, p2)], orch, memo={}, committed_service={}, planned={}, nodes=[n1, n2])
    want = {}
    for n, p in ((n1, p1), (n2, p2)):
        want[n.node_name] = (platform_queue_drain_seconds(p, orch, {}, exec_scale=1.0) + incoming_cold_start_time(t, p)
                             + float(t.type["executionTime"][p.type["shortName"]]) + network_latency_between(t.node_name, n, [n1, n2]))
    assert node.node_name == min(want, key=lambda k: (want[k], k))


def test_queue_count_naive_and_ec(monkeypatch):
    n1, p1, n2, p2 = _scene()
    orch = _orch()
    naive = _sched(monkeypatch, HEROSIM_OBS_QUEUE="count", HEROSIM_OBS_ESTIMATOR="naive")
    svc = {k: sum(v["executionTime"].values()) / 2 + _approx_comm(v) for k, v in TYPES.items()}
    g = sum(svc.values()) / 2
    assert naive._pg_drain(p1, orch, {}, 1.0) == pytest.approx(3 * g)
    assert naive._pg_drain(p2, orch, {}, 1.0) == 0.0
    assert naive.obs_drain_degraded == 2 and naive.obs_drain_exact == 0
    ec = _sched(monkeypatch, HEROSIM_OBS_QUEUE="count", HEROSIM_OBS_ESTIMATOR="ec")
    assert ec._pg_drain(p1, orch, {}, 1.0) == pytest.approx(svc["cnn"] + 2 * svc["svm"])


def test_exec_noise_is_fixed_per_pair_and_seeded(monkeypatch):
    a = _sched(monkeypatch, HEROSIM_OBS_EXEC_SIGMA="0.3", HEROSIM_OBS_ESTIMATOR="naive", HEROSIM_OBS_SEED="7")
    b = _sched(monkeypatch, HEROSIM_OBS_EXEC_SIGMA="0.3", HEROSIM_OBS_ESTIMATOR="naive", HEROSIM_OBS_SEED="7")
    c = _sched(monkeypatch, HEROSIM_OBS_EXEC_SIGMA="0.3", HEROSIM_OBS_ESTIMATOR="naive", HEROSIM_OBS_SEED="8")
    cnn = TYPES["cnn"]
    x = a._pg_exec_table(cnn, "gpu")
    assert x == a._pg_exec_table(cnn, "gpu") == b._pg_exec_table(cnn, "gpu") != c._pg_exec_table(cnn, "gpu")
    assert x != 2.0 and x > 0 and a._pg_exec_table(cnn, "cpu") / 8.0 != x / 2.0
    ratios = [a._obs.exec_est(TYPES[t], p, 1.0) for t in TYPES for p in ("gpu", "cpu")]
    assert all(math.isfinite(r) and r > 0 for r in ratios)
    z = _sched(monkeypatch, HEROSIM_OBS_QUEUE="count", HEROSIM_OBS_ESTIMATOR="ec")
    assert z._pg_exec_table(cnn, "gpu") == 2.0  # no sigma: the table value


def test_exec_noise_reaches_the_exact_queue_walk(monkeypatch):
    n1, p1, n2, p2 = _scene()
    s = _sched(monkeypatch, HEROSIM_OBS_EXEC_SIGMA="0.3", HEROSIM_OBS_ESTIMATOR="naive")
    exact = platform_queue_drain_seconds(p1, _orch(), {}, exec_scale=1.0)
    noisy = s._pg_drain(p1, _orch(), {}, 1.0)
    assert noisy != exact and s.obs_drain_exact == 1 and s.obs_exec_degraded >= 3


def test_inflight_running(monkeypatch):
    n, p = _platform("n1", 1, "gpu", current=_task(5, "cnn"))
    naive = _sched(monkeypatch, HEROSIM_OBS_INFLIGHT="running", HEROSIM_OBS_ESTIMATOR="naive")
    assert naive._pg_running(p) == 0.0
    ec = _sched(monkeypatch, HEROSIM_OBS_INFLIGHT="running", HEROSIM_OBS_ESTIMATOR="ec")
    assert ec._pg_running(p) == pytest.approx(0.5 * (5.0 + _approx_comm(TYPES["cnn"])))
    idle_n, idle = _platform("n2", 2, "cpu")
    assert ec._pg_running(idle) == 0.0
    monkeypatch.setenv("HEROSIM_PG_INFLIGHT", "1")
    with pytest.raises(ValueError, match="HEROSIM_PG_INFLIGHT"):
        O.Observation.from_env()


def test_partners_arrival_hides_the_plan_and_ec_predicts(monkeypatch):
    t = _task(1, "cnn")
    partner = _task(2, "svm")
    second = _task(3, "svm")
    placed_node, placed_plat = _platform("nP", 9, "gpu")
    second.platform = placed_plat
    orch = _orch({1: {2: 1e8}, 2: {1: 1e8, 3: 5e7}}, [t, partner, second])
    off = _sched(monkeypatch)
    assert off._pg_peer_nodes(t, orch, {2: "nX"}) == [("nX", 1e8)]
    naive = _sched(monkeypatch, HEROSIM_OBS_PARTNERS="arrival", HEROSIM_OBS_ESTIMATOR="naive")
    assert naive._pg_peer_nodes(t, orch, {2: "nX"}) == [] and naive.obs_partner_hidden == 1 and naive.pg_partners_unknown == 1
    ec = _sched(monkeypatch, HEROSIM_OBS_PARTNERS="arrival", HEROSIM_OBS_ESTIMATOR="ec")
    assert ec._pg_peer_nodes(t, orch, {2: "nX"}) == [("nP", 1e8)] and ec.obs_partner_predicted == 1


def test_validation(monkeypatch):
    for k in O.OBS_ENVS:
        monkeypatch.delenv(k, raising=False)
    assert O.Observation.from_env() is None
    monkeypatch.setenv("HEROSIM_OBS_QUEUE", "count")
    with pytest.raises(ValueError, match="HEROSIM_OBS_ESTIMATOR"):
        O.Observation.from_env()
    monkeypatch.setenv("HEROSIM_OBS_ESTIMATOR", "ec")
    monkeypatch.setenv("HEROSIM_OBS_QUEUE", "length")
    with pytest.raises(ValueError, match="HEROSIM_OBS_QUEUE"):
        O.Observation.from_env()
    monkeypatch.delenv("HEROSIM_OBS_QUEUE")
    with pytest.raises(ValueError, match="no degradation"):
        O.Observation.from_env()


# ---- A0+ (HEROSIM_PG_COLD_TAIL) and W1 (HEROSIM_OBS_WARM=seconds) ----

def test_cold_flag_off_is_the_registered_term(monkeypatch):
    s = _sched(monkeypatch)
    n, p = _platform("n1", 1, "gpu", previous=_task(9, "cnn"))
    for t in (_task(1, "cnn"), _task(2, "svm")):
        assert s._pg_cold(t, p, _orch()) == incoming_cold_start_time(t, p)
    assert s.pg_cold_tail_calls == 0


def test_cold_tail_prices_against_the_queue_tail_and_drops_the_current_task_rule(monkeypatch):
    s = _sched(monkeypatch, HEROSIM_PG_COLD_TAIL="1")
    orch = _orch()
    svm, cnn = _task(2, "svm"), _task(1, "cnn")
    # previous_task cnn, empty queue: a cnn is warm, an svm cold (the table value)
    n, p = _platform("n1", 1, "gpu", previous=_task(9, "cnn"))
    assert s._pg_cold(cnn, p, orch) == 0.0 and s._pg_cold(svm, p, orch) == 4.0
    # queue tail svm behind a cnn previous_task: now the svm is warm and the cnn cold, though previous_task says cnn
    n, p = _platform("n1", 1, "gpu", queue=[_task(5, "svm")], previous=_task(9, "cnn"))
    assert s._pg_cold(svm, p, orch) == 0.0 and s._pg_cold(cnn, p, orch) == 5.0
    assert incoming_cold_start_time(svm, p) == 4.0   # the registered term reads previous_task
    # current_task set (computing) does not zero the cold of a different type
    n, p = _platform("n1", 1, "gpu", current=_task(6, "svm"), previous=_task(6, "svm"))
    assert s._pg_cold(cnn, p, orch) == 5.0 and incoming_cold_start_time(cnn, p) == 0.0
    # nothing ever ran: cold; an explicit in-batch predecessor overrides the tail
    n, p = _platform("n1", 1, "gpu")
    assert s._pg_cold(cnn, p, orch) == 5.0 and s._pg_cold(cnn, p, orch, "cnn") == 0.0 and s._pg_cold(cnn, p, orch, None) == 5.0
    assert s.pg_cold_tail_calls == 8


def test_w1_keeps_the_warm_bit_and_degrades_the_seconds(monkeypatch):
    n, p = _platform("n1", 1, "gpu", previous=_task(9, "cnn"))
    orch = _orch()
    cnn, svm = _task(1, "cnn"), _task(2, "svm")
    allv = [5.0, 1.0, 4.0, 1.0]
    naive = _sched(monkeypatch, HEROSIM_PG_COLD_TAIL="1", HEROSIM_OBS_WARM="seconds", HEROSIM_OBS_ESTIMATOR="naive")
    assert naive._pg_cold(cnn, p, orch) == 0.0                                  # the bit is exact
    assert naive._pg_cold(svm, p, orch) == pytest.approx(sum(allv) / 4)          # global mean of the cold table
    ec = _sched(monkeypatch, HEROSIM_PG_COLD_TAIL="1", HEROSIM_OBS_WARM="seconds", HEROSIM_OBS_ESTIMATOR="ec")
    assert ec._pg_cold(svm, p, orch) == pytest.approx((5.0 + 4.0) / 2)           # gpu: mean over task types
    assert ec.obs_cold_degraded == 1 and ec.obs_cold_exact == 0
    with pytest.raises(ValueError, match="HEROSIM_PG_COLD_TAIL"):
        _sched(monkeypatch, HEROSIM_OBS_WARM="seconds", HEROSIM_OBS_ESTIMATOR="ec")


def test_plan_cost_uses_the_in_batch_predecessor(monkeypatch):
    s = _sched(monkeypatch, HEROSIM_PG_COLD_TAIL="1")
    s.exchange_on = False
    n1, p1 = _platform("n1", 1, "gpu")
    n2, p2 = _platform("n2", 2, "gpu")
    orch = _orch()
    a, b = _task(1, "svm"), _task(2, "svm")
    stacked, _, scores_stacked = s._pg_plan_cost([a, b], [(n1, p1), (n1, p1)], orch, {}, [n1, n2])
    split, _, scores_split = s._pg_plan_cost([a, b], [(n1, p1), (n2, p2)], orch, {}, [n1, n2])
    # both tasks see the other's service when stacked (exec 1 s + the 0.002 s I/O approximation); the second svm also follows an svm, so it pays no cold (4 s)
    svc = 1.0 + _approx_comm(TYPES["svm"])
    assert scores_stacked[0] - scores_split[0] == pytest.approx(svc)
    assert scores_split[1] - scores_stacked[1] == pytest.approx(4.0 - svc)
    assert s.pg_cold_tail_calls == 4

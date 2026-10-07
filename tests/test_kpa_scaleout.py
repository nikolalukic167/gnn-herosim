"""kpa_scaleout_v1 (docs/lineages/kpa_scaleout_v1.md): HEROSIM_SCALEOUT=kpa vs the legacy default.

Pins:
- legacy is the default; an unknown mode fails loud; the target resolves to 100 (int) under legacy and
  0.7 (float) under kpa, and HEROSIM_QUEUE_LENGTH accepts a float;
- the stable/panic windows scale with HEROSIM_POLICY_TIME_SCALE exactly as keep-alive does;
- the KPA decision: stable sizing ceil(avg / 0.7), panic at >= 200 % of ready capacity, no scale-down in
  panic, panic left one stable window after the last over-threshold sample, the last replica held until
  a full stable window without traffic;
- in-flight concurrency counts queued + admitted (mid-transfer) + in service, with release-mode tasks
  counted once;
- both autoscaler families (knative_network, gnn) run the KPA reconcile in SimPy: load scale-ups are
  logged as "load", scheduler-driven ones as "reachability", the memory cap refuses rather than
  overcommits, and nothing kpa-specific appears under legacy.
"""
from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import simpy

from src.placement import scaleout
from src.placement.scaleout import KpaConfig, KpaScaler, platform_in_flight


def _es():
    return importlib.import_module("src.executesimulation")


# ---------------------------------------------------------------- flag, target, windows

def test_legacy_is_the_default_and_unknown_fails_loud(monkeypatch):
    monkeypatch.delenv("HEROSIM_SCALEOUT", raising=False)
    assert scaleout.scaleout_mode() == "legacy"
    monkeypatch.setenv("HEROSIM_SCALEOUT", "hpa")
    with pytest.raises(ValueError, match="HEROSIM_SCALEOUT"):
        scaleout.scaleout_mode()


def test_target_resolution(monkeypatch):
    es = _es()
    monkeypatch.delenv("HEROSIM_QUEUE_LENGTH", raising=False)
    monkeypatch.delenv("HEROSIM_SCALEOUT", raising=False)
    v = es._resolve_queue_length()
    assert v == 100 and type(v) is int
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    assert es._resolve_queue_length() == pytest.approx(0.7)
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "0.7")
    assert es._resolve_queue_length() == pytest.approx(0.7)
    monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", "100")
    v = es._resolve_queue_length()
    assert v == 100 and type(v) is int
    assert es._resolve_queue_length(30) == 30 and type(es._resolve_queue_length(30)) is int
    assert es._resolve_queue_length(2.5) == 2.5
    for bad in ("0", "-1", "nan", "inf", "x"):
        monkeypatch.setenv("HEROSIM_QUEUE_LENGTH", bad)
        with pytest.raises(ValueError):
            es._resolve_queue_length()


def test_windows_scale_with_policy_time_scale(monkeypatch):
    monkeypatch.delenv("HEROSIM_POLICY_TIME_SCALE", raising=False)
    cfg = KpaConfig.from_env(0.7)
    assert (cfg.stable_window, cfg.panic_window, cfg.panic_threshold) == (60.0, 6.0, 2.0)
    monkeypatch.setenv("HEROSIM_POLICY_TIME_SCALE", "0.5")
    cfg = KpaConfig.from_env(0.7)
    assert (cfg.stable_window, cfg.panic_window) == (30.0, 3.0)
    assert cfg.describe()["policy_time_scale"] == 0.5
    with pytest.raises(ValueError):
        KpaConfig.from_env(0)


def test_provenance_whitelists_the_flag(monkeypatch):
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    prov = _es().build_run_provenance({}, "knative_network")
    assert prov["env"]["HEROSIM_SCALEOUT"] == "kpa"


# ---------------------------------------------------------------- the decision

CFG = KpaConfig(target=0.7, stable_window=60.0, panic_window=6.0)


def _feed(scaler, fn, samples):
    for t, v in samples:
        scaler.observe(fn, t, v)


def test_stable_sizing_is_ceil_of_average_over_target():
    s = KpaScaler(CFG)
    _feed(s, "f", [(t, 1.4) for t in range(0, 61)])
    d = s.decide("f", 60.0, current=2, ready=2)
    assert not d.panicking and d.stable_avg == pytest.approx(1.4) and d.desired == 2
    _feed(s, "g", [(t, 1.5) for t in range(0, 61)])
    assert s.decide("g", 60.0, current=2, ready=2).desired == 3


def test_panic_enters_at_200_percent_and_never_scales_down():
    s = KpaScaler(CFG)
    _feed(s, "f", [(t, 0.0) for t in range(0, 60)])
    s.observe("f", 60.0, 1.39)  # panic avg 1.39/6 samples ... below threshold for 1 ready replica
    assert not s.decide("f", 60.0, current=1, ready=1).panicking
    for t in range(61, 64):
        s.observe("f", float(t), 6.0)
    d = s.decide("f", 63.0, current=1, ready=1)
    assert d.panicking and d.entered_panic
    assert d.desired == math.ceil(d.panic_avg / 0.7)
    peak = d.desired
    # demand collapses: still panicking, desired never drops below the panic peak
    for t in range(64, 120):
        s.observe("f", float(t), 0.0)
        d = s.decide("f", float(t), current=peak, ready=peak)
        if t - 63 < 60:
            assert d.panicking and d.desired >= peak and not d.entered_panic
    # one stable window after the last over-threshold sample, panic ends and the stable rule resumes
    s.observe("f", 124.0, 0.0)
    d = s.decide("f", 124.0, current=peak, ready=peak)
    assert not d.panicking and d.desired < peak


def test_last_replica_needs_a_full_quiet_stable_window():
    s = KpaScaler(CFG)
    s.observe("f", 0.0, 1.0)
    for t in range(1, 60):
        s.observe("f", float(t), 0.0)
        assert s.decide("f", float(t), current=1, ready=1).desired == 1
    s.observe("f", 60.0, 0.0)
    assert s.decide("f", 60.0, current=1, ready=1).desired == 0
    # a function that never saw traffic still waits for its window to be covered
    s2 = KpaScaler(CFG)
    s2.observe("g", 10.0, 0.0)
    assert s2.decide("g", 10.0, current=1, ready=1).desired == 1
    s2.observe("g", 70.0, 0.0)
    assert s2.decide("g", 70.0, current=1, ready=1).desired == 0


# ---------------------------------------------------------------- in-flight signal

class _Q:
    def __init__(self, n):
        self.items = [object()] * n


class FakeSignalPlatform:
    def __init__(self, queued=0, admitted=None, current=None, inflight=(), virtual=0):
        self.queue = _Q(queued)
        self.virtual_warmup_count = virtual
        self.admitted = admitted
        self.current_task = current
        self.inflight = list(inflight)

    def queue_length(self):
        return len(self.queue.items) + self.virtual_warmup_count


def test_in_flight_counts_queued_admitted_and_in_service_once():
    a, b, c = object(), object(), object()
    assert platform_in_flight(FakeSignalPlatform()) == 0
    assert platform_in_flight(FakeSignalPlatform(queued=3, virtual=2)) == 5
    assert platform_in_flight(FakeSignalPlatform(queued=1, admitted=a, current=b)) == 3
    # release mode: current_task is one of the in-flight tasks and is not counted twice
    assert platform_in_flight(FakeSignalPlatform(queued=1, current=b, inflight=(b, c))) == 3


# ---------------------------------------------------------------- SimPy, both families

class FakeNode:
    def __init__(self, env, nid, name, memory):
        self.id = nid
        self.node_name = name
        self.memory = memory
        self.available_memory = memory
        self.available_platforms = 0
        self.network_map = {}
        self.cache_hits = 0

    def __repr__(self):
        return self.node_name


class FakePlatform:
    def __init__(self, env, pid, node):
        self.env = env
        self.id = pid
        self.node = node
        self.type = {"shortName": "cpu", "name": "cpu"}
        self.queue = simpy.Store(env)
        self.virtual_warmup_count = 0
        self.admitted = None
        self.current_task = None
        self.inflight = []
        self.rendezvous_procs = {}
        self.idle_since = math.inf
        self.last_allocated = math.inf
        self.last_removed = math.inf
        self.initialized = env.event()
        self.storage_time = 0.0
        self.previous_task = None

    def queue_length(self):
        return len(self.queue.items)

    def __repr__(self):
        return f"P{self.id}@{self.node}"


FAMILIES = {
    "knative_network": "src.policy.knative_network.autoscaler",
    "gnn": "src.policy.gnn.autoscaler",
}


def _world(monkeypatch, family, *, servers=2, platforms_per_node=4, memory=3.0):
    module = importlib.import_module(FAMILIES[family])
    monkeypatch.setattr(module, "needs_image_pull", lambda *a, **k: False)
    monkeypatch.setattr(module, "image_pull_disk_hit", lambda *a, **k: False)
    env = simpy.Environment()
    nodes, available = [], {}
    pid = 0
    for i in range(servers):
        node = FakeNode(env, i, f"node{i}", memory)
        node.network_map = {f"node{j}": 0.001 for j in range(servers) if j != i}
        available[node] = set()
        for _ in range(platforms_per_node):
            available[node].add(FakePlatform(env, pid, node))
            pid += 1
        node.available_platforms = platforms_per_node
        nodes.append(node)
    task_type = {"name": "f", "platforms": ["cpu"], "memoryRequirements": {"cpu": 1.0}}
    data = SimpleNamespace(task_types={"f": task_type}, platform_types={"cpu": {"shortName": "cpu"}})
    policy = SimpleNamespace(queue_length=0.7, reconcile_interval=1, keep_alive=30)
    state = SimpleNamespace(
        replicas={"f": set()},
        available_resources=available,
        scheduler_state=SimpleNamespace(average_contention={"f": {}}, target_concurrencies={"f": {"cpu": 0.7}}),
    )
    mutex = simpy.Store(env)
    mutex.put(state)
    autoscaler = module.KnativeAutoscaler(env, mutex, data, policy)
    return env, autoscaler, state, nodes, task_type


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_runs_kpa_with_causes_memory_cap_and_scale_to_zero(monkeypatch, family):
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    monkeypatch.delenv("HEROSIM_POLICY_TIME_SCALE", raising=False)
    monkeypatch.delenv("HEROSIM_SERVER_ONLY_REPLICAS", raising=False)
    monkeypatch.delenv("HEROSIM_REPLICA_PLATFORM_TYPES", raising=False)
    env, autoscaler, state, nodes, task_type = _world(monkeypatch, family)
    assert autoscaler.kpa is not None and autoscaler.kpa.config.target == 0.7

    def scenario():
        # a task from node0 finds no replica: the scheduler path creates one (reachability)
        yield env.process(autoscaler.create_first_replica(state, task_type, source_node_name="node0"))
        assert len(state.replicas["f"]) == 1
        (_, first), = state.replicas["f"]
        # a burst of 10 queued tasks on it from t=5 to t=12
        yield env.timeout(5)
        for _ in range(10):
            first.queue.put(object())
        yield env.timeout(7)
        first.queue.items.clear()
        first.idle_since = env.now

    panic_ticks = set()
    decide = autoscaler.kpa.decide

    def recording_decide(fn, now, current, ready):
        d = decide(fn, now, current, ready)
        if d.panicking:
            panic_ticks.add(now)
        return d

    autoscaler.kpa.decide = recording_decide
    env.process(scenario())
    env.process(autoscaler.autoscaler_process())
    peak = []
    env.process(_watch(env, state, peak))
    env.run(until=400)

    stats = autoscaler.scaleout_summary()
    ups = [e for e in autoscaler.scale_events if e.get("action") == "up"]
    downs = [e for e in autoscaler.scale_events if e.get("action") == "down"]
    assert ups[0]["cause"] == "reachability"
    assert {e["cause"] for e in ups[1:]} == {"load"}
    assert stats["scale_ups_by_cause"] == {"load": len(ups) - 1, "reachability": 1}
    # 2 nodes x 3 GB / 1 GB per replica: six replicas at most although eight platforms are free
    assert max(peak) == 6
    assert stats["memory_cap_refusals"] > 0 and stats["memory_cap_blocked"] > 0
    assert all(n.available_memory >= 0 for n in nodes)
    assert stats["panic_entries"] >= 1
    # no scale-down while panicking, and panic outlasts the 7 s burst by a stable window
    assert panic_ticks and not {e["timestamp"] for e in downs} & panic_ticks
    assert max(panic_ticks) >= 5 + 60
    assert min(e["timestamp"] for e in downs) > max(panic_ticks)
    # back to zero only after a full quiet stable window, and every platform and GB returned
    assert state.replicas["f"] == set() and stats["scale_to_zero"] == 1
    assert all(n.available_memory == 3.0 for n in nodes)
    assert stats["scale_downs"] == len(downs) == len(ups)


def _watch(env, state, peak):
    while True:
        peak.append(len(state.replicas["f"]))
        yield env.timeout(0.5)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_legacy_carries_nothing_kpa(monkeypatch, family):
    monkeypatch.delenv("HEROSIM_SCALEOUT", raising=False)
    monkeypatch.delenv("HEROSIM_SERVER_ONLY_REPLICAS", raising=False)
    monkeypatch.delenv("HEROSIM_REPLICA_PLATFORM_TYPES", raising=False)
    env, autoscaler, state, nodes, task_type = _world(monkeypatch, family, memory=1.0, platforms_per_node=2)
    assert autoscaler.kpa is None and autoscaler.scaleout_summary() is None

    def scenario():
        yield env.process(autoscaler.create_first_replica(state, task_type, source_node_name="node0"))
        yield env.process(autoscaler.scale_up(2, state, "f", "cpu"))

    env.process(scenario())
    env.run(until=1)
    assert all("cause" not in e for e in autoscaler.scale_events)
    # legacy checks total, not free, node memory: it overcommits, as every run before 2026-10-07 did
    assert min(n.available_memory for n in nodes) < 0


# --- amendments A1-A3 (2026-10-07) ---------------------------------------------------------------


def test_a2_scale_down_at_most_halves_per_decision():
    s = KpaScaler(CFG)
    _feed(s, "f", [(t, 0.0) for t in range(0, 200)])
    assert s.decide("f", 199.0, current=8, ready=8).desired == 4
    assert s.decide("f", 199.0, current=3, ready=3).desired == 1
    # one ready replica: floor(1 / 2) = 0, so the quiet-window rule alone governs the last replica
    assert s.decide("f", 199.0, current=1, ready=1).desired == 0
    assert CFG.max_scale_down_rate == 2.0 and CFG.max_scale_up_rate == 1000.0


def test_a1_pending_tasks_count_until_scheduled_and_are_pruned():
    env = simpy.Environment()
    tasks = [SimpleNamespace(scheduled=env.event()) for _ in range(3)]
    pending = list(tasks)
    assert scaleout.pending_in_flight(pending) == 3
    tasks[0].scheduled.succeed()
    assert scaleout.pending_in_flight(pending) == 2 and len(pending) == 2
    assert scaleout.pending_in_flight(None) == 0


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a1_orchestrator_arrivals_feed_kpa_demand(monkeypatch, family):
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    monkeypatch.delenv("HEROSIM_POLICY_TIME_SCALE", raising=False)
    env, autoscaler, state, nodes, task_type = _world(monkeypatch, family)
    held = [SimpleNamespace(type={"name": "f"}, scheduled=env.event()) for _ in range(5)]
    for t in held:
        autoscaler.kpa_note_arrival(t)
    seen = []
    observe = autoscaler.kpa.observe
    autoscaler.kpa.observe = lambda fn, now, v: (seen.append(v), observe(fn, now, v))
    state.replicas["f"] = set()
    env.process(autoscaler.create_first_replica(state, task_type, source_node_name="node0"))
    env.process(autoscaler.autoscaler_process())
    env.run(until=3)
    assert seen and seen[-1] == 5  # no replica holds anything; all demand is the held batch


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a3_never_served_reachability_replica_is_not_idle(monkeypatch, family):
    monkeypatch.setenv("HEROSIM_SCALEOUT", "kpa")
    monkeypatch.delenv("HEROSIM_POLICY_TIME_SCALE", raising=False)
    env, autoscaler, state, nodes, task_type = _world(monkeypatch, family)

    def scenario():
        yield env.process(autoscaler.create_first_replica(state, task_type, source_node_name="node0"))
        (node, plat), = state.replicas["f"]
        yield plat.initialized.succeed() if not plat.initialized.triggered else env.timeout(0)
        plat.idle_since = env.now
        yield env.timeout(10)
        yield env.process(autoscaler._kpa_scale_down(1, state, "f"))
        assert len(state.replicas["f"]) == 1  # protected: it has not started a task
        plat.last_started = env.now
        yield env.process(autoscaler._kpa_scale_down(1, state, "f"))
        assert state.replicas["f"] == set()

    env.process(scenario())
    env.run(until=50)
    assert autoscaler.scaleout_summary()["scale_downs"] == 1

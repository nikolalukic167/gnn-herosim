"""Service observations must not change physics or legacy snapshot accounting."""
from types import SimpleNamespace as S
import json

import pytest
import simpy

from src.placement.availability import (
    AVAILABILITY, CONTRACT_ENV, LEGACY, ServiceState, begin_service, drain_contract,
    end_service, platform_availability, service_phase,
)
from src.placement.infrastructure import Platform
from src.placement.live_audit import platform_queue_drain_seconds


def fixture(*, cold=0., execution=10., input_s=0., output_s=0., ingress=0., virtual=0.):
    env = simpy.Environment()
    storage_type = {'remote': True, 'throughput': {'read': 1., 'write': 1.},
                    'latency': {'read': 0., 'write': 0.}}
    remote = S(type=storage_type, store_data=lambda task: True)
    local = S(type={**storage_type, 'remote': False}, store_data=lambda task: True)
    stores = simpy.FilterStore(env)
    stores.items.extend([local, remote])
    node = S(node_name='server', network_map={'client': ingress}, network={'bandwidth': 1.},
             storage=stores, ingress_pipe=None, fabric=None, compute_slots=None,
             contention_time=0., local_dependencies=0, orchestrator_ref=None)
    p = Platform.__new__(Platform)
    p.id = 1; p.type = {'shortName': 'cpu', 'name': 'cpu'}; p.node = node; p.env = env
    p.fast_forward_warmup = False
    p.virtual_warmup_count = int(virtual > 0)
    p.virtual_warmup_total_time = virtual
    p.virtual_warmup_task_type = 'fn' if virtual else None
    p.queue = simpy.Store(env)
    p.initialized = env.event(); p.initialized.succeed()
    p.previous_task = None; p.current_task = None; p.service_state = ServiceState()
    p.tasks_count = 0; p.load_time = 0.; p.storage_time = 0.
    task = S(id=0, type={'name': 'fn', 'executionTime': {'cpu': execution},
                         'coldStartDuration': {'cpu': cold},
                         'stateSize': {'app': {'input': input_s*1024*1024, 'output': output_s*1024*1024}}},
             application=S(type={'name': 'app'}), dependencies=[], node_name='client',
             arrived=env.event(), started=env.event(), done=env.event(), storage={}, is_internal=False)
    env.process(p.platform_process())
    return env, p, task


@pytest.fixture(autouse=True)
def clean_physics(monkeypatch):
    monkeypatch.delenv('HEROSIM_PEER_EXCHANGE', raising=False)
    monkeypatch.delenv(CONTRACT_ENV, raising=False)


def test_real_running_task_is_not_idle():
    env, p, task = fixture()
    assert platform_availability(p, None).known_work_s == 0
    p.queue.put(task)
    while p.service_state.phase != 'execution':
        env.step()
    assert platform_queue_drain_seconds(p, None) == 0
    assert platform_availability(p, None).current_service_s == 10
    env.run(until=4.)
    estimate = platform_availability(p, None)
    assert estimate.phase == 'execution'
    assert estimate.current_service_s == 6
    env.run(until=task.done)
    assert env.now == 10
    assert not platform_availability(p, None).busy


def test_real_timed_phases_decrease_and_preserve_completion():
    env, p, task = fixture(cold=3, execution=10, input_s=4, output_s=5, ingress=2)
    p.queue.put(task)
    checks = [(1., 'ingress_latency', 14.), (3., 'cold_start', 12.),
              (7., 'input_transfer', 12.), (12., 'execution', 7.),
              (21., 'output_transfer', 3.)]
    for time, phase, remaining in checks:
        env.run(until=time)
        estimate = platform_availability(p, None)
        assert estimate.phase == phase
        assert estimate.current_service_s == remaining
    env.run(until=task.done)
    assert env.now == 24


def test_real_compute_slot_wait_stays_unknown_until_released():
    env, p, task = fixture()
    p.node.compute_slots = simpy.Resource(env, capacity=1)
    held = p.node.compute_slots.request()
    p.queue.put(task)
    env.run(until=1.)
    a = platform_availability(p, None)
    assert a.unresolved == ('compute_slot',) and a.busy
    assert a.current_service_s == 10
    p.node.compute_slots.release(held)
    env.run(until=2.)
    assert not platform_availability(p, None).has_unresolved_wait
    assert platform_availability(p, None).current_service_s == 9
    env.run(until=task.done)
    assert env.now == 11


@pytest.mark.parametrize('kind', ['ingress', 'link'])
def test_real_network_resource_wait_and_transfer(kind):
    env, p, task = fixture(ingress=1., input_s=2.)
    resource = simpy.Resource(env, capacity=1)
    held = resource.request()
    if kind == 'ingress':
        p.node.ingress_pipe = resource
        p.node.ingress_bandwidth_mbps = 1.
        p.node.ingress_wait_total = 0.
        wait_phase, transfer_phase, reason = 'ingress_wait', 'ingress_transfer', 'ingress_pipe'
    else:
        p.node.fabric = S(hops=lambda a, b: [('link', 1.)], pipe=lambda k: resource, link_wait_total=0.)
        task.link_wait_time = 0.; task.link_transfer_time = 0.; task.link_hops = 0
        wait_phase, transfer_phase, reason = 'link_wait', 'link_transfer', 'network_link'
    p.queue.put(task)
    env.run(until=3.)
    a = platform_availability(p, None)
    assert a.phase == wait_phase and a.unresolved == (reason,)
    assert a.current_service_s == 12
    resource.release(held)
    env.run(until=4.)
    a = platform_availability(p, None)
    assert a.phase == transfer_phase and not a.has_unresolved_wait
    assert a.current_service_s == 11
    env.run(until=task.done)
    assert env.now == 17


def test_storage_wait_with_zero_known_work_is_still_busy():
    env, p, task = fixture()
    remote_request = p.node.storage.get(lambda s: s.type['remote'])
    local_request = p.node.storage.get(lambda s: not s.type['remote'])
    p.queue.put(task)
    env.run(until=2.)
    assert platform_availability(p, None).unresolved == ('input_storage',)
    p.node.storage.put(remote_request.value)
    env.run(until=13.)
    a = platform_availability(p, None)
    assert a.unresolved == ('output_storage',)
    assert a.busy and a.known_work_s == 0
    p.node.storage.put(local_request.value)
    env.run(until=task.done)
    assert env.now == 13


def test_real_peer_wait_cannot_be_confused_with_idle(monkeypatch):
    monkeypatch.setenv('HEROSIM_PEER_EXCHANGE', '1')
    env, p, task = fixture()
    ready = env.event()
    orch = S(peer_exchange={0: {1: 1.}}, task_by_id={0: task}, peer_ready_event=lambda tid: ready)
    p.node.orchestrator_ref = orch
    p.queue.put(task)
    env.run(until=2.)
    a = platform_availability(p, orch)
    assert a.phase == 'peer_rendezvous' and a.has_unresolved_wait and a.busy
    assert a.current_service_s == 10
    # Unrelated unarrived state is not an observation of this platform's work.
    orch.task_by_id[99] = S(platform=S(queue=S(items=['forbidden'])))
    assert platform_availability(p, orch) == a
    orch.task_by_id[1] = S(platform=S(node=S(node_name='unannounced'), queue=S(items=['future'])))
    assert platform_availability(p, orch) == a
    orch.task_by_id[1] = S(platform=p)
    ready.succeed()
    env.run(until=3.)
    assert not platform_availability(p, orch).has_unresolved_wait
    env.run(until=task.done)
    assert env.now == 12


def test_real_virtual_backlog_decreases_without_changing_legacy_field():
    env, p, task = fixture(virtual=12.)
    p.queue.put(task)
    env.run(until=5.)
    a = platform_availability(p, None)
    assert a.virtual_backlog_s == 7
    assert a.current_service_s == 0
    assert a.queued_work_s == pytest.approx(10.002)
    assert platform_queue_drain_seconds(p, None) == pytest.approx(22.002)
    env.run(until=task.done)
    assert env.now == 22
    assert platform_availability(p, None).known_work_s == 0


def test_virtual_slot_wait_is_not_counted_twice():
    env, p, _ = fixture(virtual=12.)
    p.node.compute_slots = simpy.Resource(env, capacity=1)
    held = p.node.compute_slots.request()
    env.run(until=5.)
    a = platform_availability(p, None)
    assert a.known_work_s == 12 and a.has_unresolved_wait
    p.node.compute_slots.release(held)
    env.run(until=8.)
    assert platform_availability(p, None).virtual_backlog_s == 9


def test_fast_forward_excludes_only_the_represented_warmup_tasks():
    class HashTask(S):
        __hash__ = object.__hash__
        __eq__ = object.__eq__
    env, p, task = fixture()
    warm = HashTask(**vars(task))
    p._warmup_tasks = [warm]
    p.queue.items.extend([warm, task])
    begin_service(p, virtual_seconds=12)
    p.service_state.pending.clear()
    service_phase(p, 'fast_forward_warmup', seconds=12)
    a = platform_availability(p, None)
    assert a.virtual_backlog_s == 12
    assert a.queued_work_s == pytest.approx(10.002)
    p.virtual_warmup_total_time = 7.
    a = platform_availability(p, None)
    assert a.virtual_backlog_s == 19
    assert a.queued_work_s == pytest.approx(10.002)


def test_contract_is_explicit_and_learned_arms_reject_v2(monkeypatch):
    from src.policy.peer_greedy_network.scheduler import _PeerGreedyCore
    assert drain_contract() == LEGACY
    monkeypatch.setenv(CONTRACT_ENV, 'typo')
    with pytest.raises(ValueError, match='unsupported'):
        drain_contract()
    monkeypatch.setenv(CONTRACT_ENV, AVAILABILITY)
    for label in ('peer_greedy_learned_network', 'peer_greedy_learned_network_batch'):
        scheduler = _PeerGreedyCore()
        scheduler._policy_label = label
        with pytest.raises(ValueError, match='rule-only'):
            scheduler._pg_init()


def test_rule_prefers_available_platform_and_handles_all_blocked(monkeypatch):
    from src.policy.peer_greedy_network.scheduler import _PeerGreedyCore
    _, busy, task = fixture()
    _, idle, _ = fixture()
    busy.node.id = 1; idle.node.id = 2; idle.id = 2
    for p in (busy, idle):
        p.previous_task = S(type={'name': 'fn'})
    begin_service(busy, task)
    service_phase(busy, 'execution', seconds=10.)
    candidates = [(p.node, p) for p in (busy, idle)]
    rule = _PeerGreedyCore(); rule.exchange_on = False
    rule._pg_init()
    def choose():
        return rule._pg_choose(task, candidates, None, memo={}, committed_service={}, planned={},
                               nodes=[busy.node, idle.node])[1]
    assert choose() is busy  # Legacy ties break by node identity.
    monkeypatch.setenv(CONTRACT_ENV, AVAILABILITY)
    rule._pg_init()
    assert choose() is idle
    service_phase(idle, 'peer_rendezvous', event=idle.env.event(), unresolved=('peer_rendezvous',))
    assert choose() is busy
    service_phase(busy, 'peer_rendezvous', event=busy.env.event(), unresolved=('peer_rendezvous',))
    assert choose() in (busy, idle)


def test_legacy_rule_reproduces_retained_task_timing_and_placements(monkeypatch, tmp_path):
    from scripts_cosim.peer_lookahead_trace import ROOT, run
    from scripts_cosim.peer_lookahead_live_probe import make_workload
    saved = ROOT / 'simulation_data/peer_lookahead_v1/complete_trace/read.json'
    if not saved.exists():
        pytest.skip('historical local artifact unavailable')
    reference = json.loads(saved.read_text())
    config = reference['config']
    source_path = ROOT / config['source_corpus'] / 'ds_00064' / 'optimal_result.json'
    if not source_path.exists():
        pytest.skip('source snapshot unavailable')
    monkeypatch.setenv('HEROSIM_PEER_EXCHANGE', '1')
    monkeypatch.setenv('HEROSIM_COSIM_KEEP_ALIVE', '1000000')
    monkeypatch.setenv('COSIM_SUPPRESS_SIM_PRINTS', '1')
    monkeypatch.setenv('SIM_FORCE_FULL_STATS', '1')
    monkeypatch.delenv('HEROSIM_DATA_LOCALITY', raising=False)
    monkeypatch.delenv('COSIM_AUTOSCALER_RECONCILE_INTERVAL', raising=False)
    source = json.loads(source_path.read_text())
    workload = make_workload(source, config, 1., 1, 16)
    result = run(source, config, workload, 'immediate', tmp_path / 'legacy.log')
    old = reference['regression']['immediate']
    assert result['total_rtt'] == old['total_rtt']
    assert result['pg_drain_contract'] == LEGACY
    fields = ('taskId', 'elapsedTime', 'queueTime', 'peerExchangeTime', 'peerRendezvousWait',
              'executionNode', 'executionPlatform', 'startedTime', 'arrivedTime', 'doneTime', 'scheduledTime')
    canonical = lambda rows: sorted([{k: r[k] for k in fields if k in r} for r in rows], key=lambda r: r['taskId'])
    assert canonical(result['tasks']) == canonical(old['tasks'])
    monkeypatch.setenv(CONTRACT_ENV, AVAILABILITY)
    updated = run(source, config, workload, 'immediate', tmp_path / 'v2.log')
    assert updated['pg_drain_contract'] == AVAILABILITY
    assert len(updated['tasks']) == 16

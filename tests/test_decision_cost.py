"""decision_cost_s0_v1: parsing, metering, flag-off no-op, and the scheduler server's FIFO queue."""
import pytest
import simpy

from src.placement import decision_cost as dc


def test_cost_ms_unset_zero_and_value(monkeypatch):
    monkeypatch.delenv(dc.ENV, raising=False)
    assert dc.cost_ms() == 0.0
    monkeypatch.setenv(dc.ENV, "0")
    assert dc.cost_ms() == 0.0
    monkeypatch.setenv(dc.ENV, "0.25")
    assert dc.cost_ms() == 0.25


@pytest.mark.parametrize("bad", ["abc", "-1"])
def test_cost_ms_fails_loud(monkeypatch, bad):
    monkeypatch.setenv(dc.ENV, bad)
    with pytest.raises(ValueError, match="FAIL LOUD"):
        dc.cost_ms()


def test_meter_counts_on_self_and_forwards_to_host():
    class Host:
        pass

    class Refiner:
        def __init__(self, host):
            self._host = host

    h = Host()
    r = Refiner(h)
    dc.meter(h, 3)
    dc.meter(r, 4)
    assert h.sched_plan_evals == 7
    assert not hasattr(r, "sched_plan_evals")


def test_server_is_fifo_single_capacity_and_books_waits():
    env = simpy.Environment()

    class S:
        pass

    s = S()
    s.env = env
    done = []

    def batch(name, arrive, service):
        yield env.timeout(arrive)
        srv = dc.server_of(s)
        t0 = env.now
        req = srv.request()
        yield req
        dc.book(s, "sched_queue_wait_seconds", env.now - t0)
        dc.book(s, "sched_queue_wait_max", env.now - t0, mx=True)
        yield env.timeout(service)
        srv.release(req)
        done.append((name, env.now))

    env.process(batch("a", 0, 5))
    env.process(batch("b", 1, 2))
    env.process(batch("c", 2, 1))
    env.run()
    assert done == [("a", 5), ("b", 7), ("c", 8)]
    assert s.sched_queue_wait_seconds == pytest.approx(4 + 5)
    assert s.sched_queue_wait_max == pytest.approx(5)
    assert dc.server_of(s) is dc.server_of(s)


def test_orchestrator_exports_the_books():
    import inspect
    from src.placement import orchestrator

    src = inspect.getsource(orchestrator)
    for name in dc.COUNTERS:
        assert f'"{name}"' in src


def test_felt_task_seconds_is_the_sum_of_tasks_times_the_batch_charge(monkeypatch):
    """sched_felt_task_seconds books, per charged batch, (tasks in the batch) x (the batch's charge): every task waits the whole charge."""
    from types import SimpleNamespace

    import pytest as _pytest
    from test_masked_topo_stale_placement import _prefix_batch_harness

    import src.placement.replica_seeding as rs

    monkeypatch.setenv("HEROSIM_DECISION_COST_MS", "2")       # 2 ms per plan evaluated
    monkeypatch.delenv("HEROSIM_DECISION_COST_MS_BY_BUCKET", raising=False)
    monkeypatch.delenv("GNN_SERVE_CORPUS_SLATE", raising=False)

    class Stop(Exception):
        pass

    def boom(*a, **k):
        raise Stop

    monkeypatch.setattr(rs, "start_deferred_cold_init", boom)
    env, sched, state, _task, _deferred = _prefix_batch_harness(evict_after_decode=False)
    sched.gnn_pure_decisions = 0
    sched.sched_plan_evals = 0
    base_inference = sched._prefix_inference
    plans = iter((500, 250))

    def inference(tasks, st, q, t):
        sched.sched_plan_evals += next(plans)                # this batch's metered plans
        return base_inference(tasks, st, q, t)

    sched._prefix_inference = inference
    for n_tasks in (3, 2):
        batch = [SimpleNamespace(id=100 * n_tasks + i, type={"name": "rf"}, planned_node_name=None) for i in range(n_tasks)]
        env.process(sched._process_task_batch_prefix(batch))
        try:
            env.run()
        except Stop:
            pass
        if state not in sched.mutex.items:                   # the stopped commit still holds the state; hand it back for the next batch
            sched.mutex.items.append(state)
    # batch 1: 3 tasks, 500 plans x 2 ms = 1.0 s; batch 2: 2 tasks, 250 plans x 2 ms = 0.5 s
    assert sched.sched_busy_seconds == _pytest.approx(1.5)
    assert sched.sched_felt_task_seconds == _pytest.approx(3 * 1.0 + 2 * 0.5)
    assert sched.sched_batches_charged == 2

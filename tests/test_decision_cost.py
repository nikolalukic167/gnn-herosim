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

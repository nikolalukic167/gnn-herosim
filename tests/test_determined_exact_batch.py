"""A fidelity replay schedules its batch at the snapshot instant: the determined scheduler's batch collection must not sleep a 1 ms
poll when the other events are put later in the same instant (ds_03200: the batch's link request fell behind a ghost's by 0.064 ms)."""
import simpy

from src.policy.determined.scheduler import DeterminedScheduler


class T:
    def __init__(self, i):
        self.id = i
        self.dependencies = []


def _scheduler(env, exact):
    s = object.__new__(DeterminedScheduler)
    s.env, s.tasks, s.batch_size, s.batch_timeout = env, simpy.FilterStore(env), 3, 0.02
    s.debug_enabled, s.exact_batch = False, exact
    return s


def _run(exact):
    env = simpy.Environment()
    s = _scheduler(env, exact)
    out = {}

    def gateway():
        for i in range(3):          # one put per step of the same instant, as the gateway does
            yield s.tasks.put(T(i))
            yield env.timeout(0)

    def collector():
        out["batch"] = yield from s._collect_task_batch()
        out["t"] = env.now

    env.process(collector())
    env.process(gateway())
    env.run(until=1)
    return out


def test_exact_batch_is_collected_at_the_same_instant():
    out = _run(True)
    assert [t.id for t in out["batch"]] == [0, 1, 2] and out["t"] == 0.0


def test_the_default_collection_still_polls_when_the_queue_is_momentarily_empty():
    out = _run(False)
    assert len(out["batch"]) == 3 and out["t"] > 0.0

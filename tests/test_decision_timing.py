"""Decision-cost measurement: the timing wrapper must be invisible to SimPy."""
import sys
from pathlib import Path

import simpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.placement.decision_timing import DecisionLog, timed_generator  # noqa: E402


def _run(wrap):
    env, trace, sink = simpy.Environment(), [], [0.0]
    ev = env.event()

    def inner():
        v = yield env.timeout(2, value="a")
        trace.append((env.now, v))
        try:
            yield ev
        except RuntimeError as e:
            trace.append((env.now, str(e)))
        yield env.timeout(1)
        return 42

    def failer():
        yield env.timeout(5)
        ev.fail(RuntimeError("boom"))

    def outer():
        r = yield env.process(timed_generator(inner(), sink) if wrap else inner())
        trace.append((env.now, r))

    env.process(failer())
    env.process(outer())
    env.run()
    return trace, env.now, sink[0]


def test_wrapper_yields_the_same_events_in_the_same_order():
    plain, wrapped = _run(False), _run(True)
    assert plain[0] == wrapped[0] == [(2, "a"), (5, "boom"), (6, 42)]
    assert plain[1] == wrapped[1] == 6
    assert wrapped[2] >= 0.0


def test_summary_weights_per_task_and_counts_moves():
    log = DecisionLog()
    log.add(1, 0.002)
    log.add(9, 0.090, refine_moves=4)  # 0.01 s per task over 9 tasks
    s = log.summary()
    assert s["calls"] == 2 and s["tasks"] == 10 and abs(s["total_s"] - 0.092) < 1e-12
    assert s["per_task_median_s"] == 0.01 and s["refine_moves_total"] == 4 and s["refine_moves_per_call_mean"] == 2.0
    assert DecisionLog().summary() == {"model_load_s": 0.0, "calls": 0} or DecisionLog().summary()["calls"] == 0

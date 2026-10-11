import pytest
from src.placement.snapshot_fidelity import ghost_order_key


def _g(tid, q, pop=-0.0143, stage="ingress", link="net"):
    return {"tid": tid, "q": q, "pop": pop, "stage": stage, "link_stage": link}


def test_ghosts_popped_at_the_same_instant_are_ordered_by_task_id_not_platform_name():
    # ds_03200: 1236 on node4:226 and 1238 on node4:218 were popped together; live, 1236 took the shared link first
    ghosts = [_g(1238, "node4:218"), _g(1236, "node4:226")]
    assert [g["tid"] for g in sorted(ghosts, key=ghost_order_key)] == [1236, 1238]


def test_pop_order_still_comes_first_and_hold_before_wait_before_the_rest():
    held = _g(5, "a:1", pop=-1.0, link="hold")
    waiting = _g(4, "a:2", pop=-2.0, link="wait")
    early = _g(9, "z:9", pop=-3.0)
    late = _g(1, "a:1", pop=-0.5)
    assert [g["tid"] for g in sorted([late, early, waiting, held], key=ghost_order_key)] == [5, 4, 9, 1]


def test_equal_net_end_ghosts_request_in_ghost_order_not_platform_start_order():
    import simpy

    from src.placement.snapshot_fidelity import GhostTask, _ingress

    env = simpy.Environment()
    who = {}
    order = []

    class Pipe:
        def request(self):
            order.append(who[env.active_process])
            return env.event().succeed()

        def release(self, req):
            pass

    class Fabric:
        def hops(self, src, dst):
            return [("L", 15.0)]

        def pipe(self, key):
            return Pipe()

    class Node:
        node_name = "n"
        fabric = Fabric()

    class Plat:
        env = None
        node = Node()

    Plat.env = env

    def mk(tid):
        g = GhostTask({"tid": tid, "src": "c", "fn": "dnn1", "link_stage": "net", "net_remaining": 0.05, "input_bytes": 150000.0,
                       "stage": "ingress"}, {})
        g.net_timer = env.timeout(0.05)   # created at apply time, in ghost order
        return g

    a, b = mk(1236), mk(1238)
    # the platform processes start in the opposite order (platform name: 1238's first)
    pb = env.process(_ingress(Plat(), b))
    pa = env.process(_ingress(Plat(), a))
    who[pb], who[pa] = 1238, 1236
    env.run(until=1)
    assert order[:2] == [1236, 1238]


def test_a_same_instant_ghost_is_due_exactly_when_a_batch_task_with_the_same_latency_is():
    from src.placement.snapshot_fidelity import net_due

    class N:
        network_map = {"client_node15": 0.045069}

    latency = N.network_map["client_node15"]
    rec = {"src": "client_node15", "pop": 0.0, "net_remaining": latency + 3e-17}   # (pop + latency) - now, rounded
    assert net_due(N(), rec) == latency
    rec2 = {"src": "client_node15", "pop": -0.0143, "net_remaining": latency - 0.0143}
    assert abs(net_due(N(), rec2) - rec2["net_remaining"]) < 1e-12
    assert net_due(N(), {"src": "elsewhere", "pop": 0.0, "net_remaining": 0.5}) == 0.5


def test_wait_stage_ghosts_are_ordered_by_request_time_then_task_id():
    # ds_16401: 36205/36206/36207 (node1) and 36201 (node4), all popped at -0.2404, request the link at different times because the
    # latency to their platform's node differs: 201 requests 1.5 ms later and queues behind them, whatever its task id
    ghosts = [(_g(36201, "node4:227", pop=-0.2404, link="wait"), -0.2404 + 0.0537),
              (_g(36205, "node1:203", pop=-0.2404, link="wait"), -0.2404 + 0.0522),
              (_g(36206, "node1:205", pop=-0.2404, link="wait"), -0.2404 + 0.0522),
              (_g(36207, "node1:211", pop=-0.2404, link="wait"), -0.2404 + 0.0522)]
    order = sorted(ghosts, key=lambda x: ghost_order_key(x[0], x[1]))
    assert [g["tid"] for g, _ in order] == [36205, 36206, 36207, 36201]


def test_variant_g_the_computing_task_then_lock_waiters_in_lock_request_order():
    # I11 heavy opening, 16201 t = 89.855 (node4:827, xavierDla cnn): 12 was computing (output blocked by a pull), 11 / 1114 / 1236
    # waited for the lock since their inputs ended. Pop order put 11 first, which took the lock ahead of 12 and overlapped one
    # execution with the pull hold; live, 12 held the lock and the waiters followed in io_end order.
    computing = {"tid": 12, "q": "node4:827", "pop": -14.6, "stage": "compute", "compute_remaining": 0.0}
    waiters = [{"tid": 11, "q": "node4:827", "pop": -14.7, "stage": "lock_wait", "order": -10.0},
               {"tid": 1236, "q": "node4:827", "pop": -2.0, "stage": "lock_wait", "order": -1.5},
               {"tid": 1114, "q": "node4:827", "pop": -9.0, "stage": "lock_wait", "order": -8.0}]
    reading = {"tid": 7, "q": "node4:827", "pop": -20.0, "stage": "input_io", "io_remaining": 0.4, "order": 0.4}
    order = [g["tid"] for g in sorted(waiters + [reading, computing], key=ghost_order_key)]
    assert order == [12, 11, 1114, 1236, 7]


def test_variant_g_inflight_ghosts_come_after_every_ingress_ghost():
    late_ingress = _g(99, "a:1", pop=-0.001)
    computing = {"tid": 1, "q": "a:1", "pop": -50.0, "stage": "compute", "compute_remaining": 1.0}
    assert [g["tid"] for g in sorted([computing, late_ingress], key=ghost_order_key)] == [99, 1]


def test_variant_g_replays_the_pull_hold_without_overlap():
    """End to end on the two resume coroutines: a computing task whose output waits for a pull and a lock-waiter on one replica.
    The waiter must execute after the pull ends, as live did."""
    import simpy
    from types import SimpleNamespace

    from src.placement.snapshot_fidelity import GhostTask, _serve

    env = simpy.Environment()
    storage = simpy.FilterStore(env)
    local = SimpleNamespace(type={"remote": False})
    storage.put(local)
    node = SimpleNamespace(storage=storage)
    plat = SimpleNamespace(env=env, node=node, compute_lock=simpy.Resource(env, capacity=1), inflight=[], current_task=None,
                           previous_task=None, idle_since=None)
    done = {}

    def pull():
        st = yield storage.get(lambda s: not s.type.get("remote"))
        yield env.timeout(0.766)
        yield storage.put(st)

    env.process(pull())
    tt = {"name": "cnn", "executionTime": {}, "coldStartDuration": {}}
    recs = [{"tid": 11, "q": "n:1", "pop": -14.7, "stage": "lock_wait", "order": -10.0, "fn": "cnn", "src": "c0", "exec": 0.5618, "output": 0.0},
            {"tid": 12, "q": "n:1", "pop": -14.6, "stage": "compute", "compute_remaining": 0.0, "fn": "cnn", "src": "c0", "exec": 0.5618, "output": 0.0}]
    for rec in sorted(recs, key=ghost_order_key):
        g = GhostTask(rec, tt)
        plat.inflight.append(g)

        def run(g=g):
            yield from _serve(plat, g)
            done[g.rec["tid"]] = env.now

        env.process(run())
    env.run()
    assert done[12] == pytest.approx(0.766) and done[11] == pytest.approx(0.766 + 0.5618)

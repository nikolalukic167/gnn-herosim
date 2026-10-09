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

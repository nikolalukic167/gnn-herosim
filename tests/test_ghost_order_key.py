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

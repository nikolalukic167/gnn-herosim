import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts_cosim"))


def test_the_bar_sensitivity_kind_is_cd_ib0_plus_the_inflight_flag_and_outside_the_default_grid():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    k = "cd_ib0_inflight"
    assert G.RULE_POLICY[k] == G.RULE_POLICY["cd_ib0"] == "peer_greedy_network_cd"
    assert G.ib0_alpha(k) == G.ib0_alpha("cd_ib0") == {"HEROSIM_PG_BATCH_ALPHA": "0"}
    assert G.check_alias(k) == "cd_inflight" and G.check_alias("cd_ib0") == "cd"
    assert k in G.R1A_DIAG and k in G.R1A_ARMS and k not in G.R1A_CLASSICAL
    assert G.split_dc(k + "_k100") == (k, 1.0)


def test_the_complete_variant_is_a_second_kind_with_cds_checks():
    from scripts_cosim import fresh_topo_burst_v1_gate as G
    k = "cd_ib0_inflightfull"
    assert G.RULE_POLICY[k] == "peer_greedy_network_cd" and G.ib0_alpha(k) == {"HEROSIM_PG_BATCH_ALPHA": "0"}
    assert G.check_alias(k) == "cd" and k in G.R1A_DIAG and k not in G.R1A_CLASSICAL
    assert G.split_dc(k + "_k100") == (k, 1.0)


def _task(exec_s, io_end=None, start=None, output=0):
    from types import SimpleNamespace
    return SimpleNamespace(type={"executionTime": {"t0": exec_s}, "stateSize": {}}, application=None, io_end_time=io_end, compute_start_time=start)


def _platform(inflight, now=100.0):
    from types import SimpleNamespace
    return SimpleNamespace(env=SimpleNamespace(now=now), type={"shortName": "t0"}, inflight=inflight)


def test_lock_waiters_make_a_platform_cost_more_than_an_idle_one():
    from src.placement.live_audit import inflight_full_seconds
    idle = _platform([])
    waiters = _platform([_task(2.0, io_end=90.0), _task(3.0, io_end=95.0), _task(4.0, io_end=98.0)])
    assert inflight_full_seconds(idle) == 0.0
    assert inflight_full_seconds(waiters) == 9.0 > inflight_full_seconds(idle)


def test_the_stages_of_an_inflight_task():
    from src.placement.live_audit import inflight_full_seconds
    running = _task(10.0, io_end=90.0, start=95.0)              # 95 + 10 - 100 = 5 s left
    finished_compute = _task(1.0, io_end=80.0, start=90.0)      # compute ended at 91, output estimate 0: nothing left
    waiting = _task(2.0, io_end=96.0)                           # owes its execution
    in_input = _task(3.0)                                       # owes execution + the approximate comm (0 without a state size)
    assert inflight_full_seconds(_platform([running, finished_compute, waiting, in_input])) == 5.0 + 0.0 + 2.0 + 3.0
    assert inflight_full_seconds(_platform([waiting]), exec_scale=2.0) == 4.0


def test_every_ib0_variant_gets_the_ib0_alpha_in_the_served_env():
    """The served env for a cd_ib0 cell sets HEROSIM_PG_BATCH_ALPHA where run_one builds it; the in-flight variants must pass the same branch
    (the cell check, which wants ib0_alpha(kind), failed the first smoke because this spot was missed)."""
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "scripts_cosim/fresh_topo_burst_v1_gate.py")).read()
    m = re.search(r'if kind in \("cd_ib0", ([^)]*)\):\n\s+env\.update\(ib0_alpha\(kind\)\)', src)
    assert m and '"cd_ib0_inflight"' in m.group(1) and '"cd_ib0_inflightfull"' in m.group(1)

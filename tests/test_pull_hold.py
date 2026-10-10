"""accel pull-hold control: the node's image-pull queue as one definition for the fidelity capture and for a policy
(snapshot_fidelity._node_pull_walk / node_pull_hold_seconds), the pull ledger behind HEROSIM_PULL_LEDGER, and the CD term
HEROSIM_PG_PULL_HOLD (max(0, hold - time to the output write), off by default)."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.placement import snapshot_fidelity as F
from src.placement import warmth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Plat:
    def __init__(self, pid, short, initialized):
        self.id = pid
        self.type = {"shortName": short}
        self.initialized = SimpleNamespace(triggered=initialized)


def pull_world(monkeypatch, now=100.0):
    """node0: one pull holding storage until now+5 (cnn on GPU), then two waiters: rf on CPU (new image, 2 GB at 100 MB/s)
    and a second rf call on another CPU platform (same image, cached by the first waiter: 0 s). node1: no pulls."""
    monkeypatch.setattr(warmth, "needs_image_pull", lambda physics, p, node, tt, active_storage=None: True)
    local = SimpleNamespace(type={"remote": False, "throughput": {"write": 100.0}, "latency": {"write": 0.5}}, functions_cache=[])
    remote = SimpleNamespace(type={"remote": True})
    gpu, cpu1, cpu2 = Plat(10, "xavierGpu", False), Plat(11, "xavierCpu", False), Plat(12, "xavierCpu", True)
    calls = [
        {"platform": gpu, "fn": "cnn", "init_start": now - 3.0, "end": now + 5.0, "done": False},
        {"platform": cpu1, "fn": "rf", "init_start": now - 2.0, "end": None, "done": False},
        {"platform": cpu2, "fn": "rf", "init_start": now - 1.0, "end": None, "done": False},
        {"platform": cpu2, "fn": "dnn1", "init_start": now - 9.0, "end": now - 4.0, "done": True},  # finished: ignored
    ]
    node0 = SimpleNamespace(node_name="node0", network={"bandwidth": 1000.0}, _fid_storages=[local, remote], _fid_pull_calls=calls)
    node1 = SimpleNamespace(node_name="node1", network={"bandwidth": 1000.0}, _fid_storages=[local, remote])
    sched = SimpleNamespace(env=SimpleNamespace(now=now, warmth_physics="node_disk_v2"), nodes=SimpleNamespace(items=[node0, node1]),
                            data=SimpleNamespace(task_types={"cnn": {"imageSize": {"xavierGpu": 1000.0}},
                                                             "rf": {"imageSize": {"xavierCpu": 2048.0}}}))
    return sched, node0, node1


def test_capture_and_helper_share_one_queue_walk(monkeypatch):
    sched, node0, node1 = pull_world(monkeypatch)
    platforms = {f"node0:{p}": {} for p in (10, 11, 12)}
    recs = F._capture_pulls(sched, platforms, 100.0)
    waiter_own = 2048.0 / (100.0 / 1024) + 0.5
    assert [(r["q"], r["rank"], r["estimated"]) for r in recs] == [("node0:10", 0, False), ("node0:11", 1, True), ("node0:12", 2, True)]
    assert [r["own"] for r in recs] == pytest.approx([5.0, waiter_own, 0.0])
    assert platforms["node0:10"]["pull_remaining"] == pytest.approx(5.0)            # ready when its own pull ends
    assert platforms["node0:11"]["pull_remaining"] == pytest.approx(5.0 + waiter_own)
    assert "pull_remaining" not in platforms["node0:12"]                              # already initialised
    hold = F.node_pull_hold_seconds(sched, node0, 100.0)
    assert hold == pytest.approx(sum(r["own"] for r in recs if r["node"] == "node0"))  # capture == helper
    assert F.node_pull_hold_seconds(sched, node1, 100.0) == 0.0


def test_a_finished_holder_still_listed_counts_from_now(monkeypatch):
    sched, node0, _ = pull_world(monkeypatch)
    node0._fid_pull_calls[0]["end"] = 99.0   # its end passed; the call is not marked done yet
    walk = F._node_pull_walk(sched, node0, 100.0)
    assert walk[0][1] == 0.0 and walk[0][3] == 100.0


# ---- the CD term -------------------------------------------------------------------------------------------------------
from src.policy.peer_greedy_network import scheduler as PG  # noqa: E402


def cd_shell(monkeypatch, pull_hold, holds):
    """A _PeerGreedyCore with the costs pinned: node A (cheap, base 1.0 s) and node B (base 2.0 s)."""
    s = object.__new__(PG.PeerGreedyNetworkBatchScheduler)
    s.exchange_on, s.pg_exec_oracle, s.pg_inflight, s.pg_ext_rate, s._pg_capture_path = False, False, False, None, None
    s.pg_exchange_scale, s.pg_decisions, s.pg_moved_by_exchange, s.pg_joined_partner = 1.0, 0, 0, 0
    s.pg_pull_hold, s.pg_pull_charged, s.pg_pull_seconds = pull_hold, 0, 0.0
    s.env = SimpleNamespace(now=50.0)
    drain = {"A": 0.5, "B": 1.5}
    monkeypatch.setattr(PG, "platform_queue_drain_seconds", lambda p, orch, memo, exec_scale=1.0: drain[p.node.node_name])
    monkeypatch.setattr(PG, "incoming_cold_start_time", lambda task, p: 0.0)
    monkeypatch.setattr(PG, "network_latency_between", lambda src, node, nodes: 0.2)
    calls = []

    def hold(sched, node, now):
        calls.append(node.node_name)
        return holds[node.node_name]

    monkeypatch.setattr(F, "node_pull_hold_seconds", hold)
    mk = lambda n: (SimpleNamespace(id={"A": 1, "B": 2}[n], node_name=n), None)
    cands = []
    for n in ("A", "B"):
        node, _ = mk(n)
        cands.append((node, SimpleNamespace(id=10 * node.id, node=node, type={"shortName": "xavierCpu"})))
    task = SimpleNamespace(id=7, type={"name": "rf", "executionTime": {"xavierCpu": 0.3}, "stateSize": {}}, node_name="client0")
    return s, task, cands, calls


def choose(s, task, cands, memo):
    node, platform, _svc = s._pg_choose(task, cands, None, memo=memo, committed_service={}, planned={}, nodes=[])
    return node.node_name


def test_term_off_never_reads_the_ledger(monkeypatch):
    s, task, cands, calls = cd_shell(monkeypatch, False, {"A": 30.0, "B": 0.0})
    memo = {}
    assert choose(s, task, cands, memo) == "A"
    assert calls == [] and memo == {} and (s.pg_pull_charged, s.pg_pull_seconds) == (0, 0.0)


def test_term_on_prices_the_wait_at_the_output_write(monkeypatch):
    s, task, cands, calls = cd_shell(monkeypatch, True, {"A": 30.0, "B": 0.0})
    memo = {}
    assert choose(s, task, cands, memo) == "B"
    # A: hold 30 - (drain 0.5 + cold 0 + lat 0.2 + exch 0 + exec 0.3) = 29.0 charged; B holds nothing
    assert (s.pg_pull_charged, s.pg_pull_seconds) == (1, pytest.approx(29.0))
    assert memo == {"pull_hold:A": 30.0, "pull_hold:B": 0.0}
    choose(s, task, cands, memo)                       # memoised per node within a batch
    assert calls == ["A", "B"]


def test_a_hold_shorter_than_the_time_to_output_costs_nothing(monkeypatch):
    s, task, cands, calls = cd_shell(monkeypatch, True, {"A": 0.9, "B": 0.0})   # A's write at 1.0 s > hold 0.9 s
    assert choose(s, task, cands, {}) == "A"
    assert (s.pg_pull_charged, s.pg_pull_seconds) == (0, 0.0)


def test_the_learned_scorer_refuses_the_term():
    s = object.__new__(PG.PeerGreedyLearnedNetworkScheduler)
    s.pg_pull_hold = True
    with pytest.raises(ValueError, match="hand rule only"):
        s._pg_choose(SimpleNamespace(type={}), [], None, memo={}, committed_service={}, planned={}, nodes=[])


# ---- flags -------------------------------------------------------------------------------------------------------------
def run_py(code, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("HEROSIM_PULL_LEDGER", "HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_PULL_HOLD")}
    e.update(env, PYTHONPATH=REPO, HEROSIM_PEER_EXCHANGE="1")
    return subprocess.run([sys.executable, "-c", code], env=e, cwd=REPO, capture_output=True, text=True)


def test_ledger_is_off_by_default_and_on_by_either_flag():
    code = "from src.placement.infrastructure import PULL_LEDGER, FIDELITY; print(PULL_LEDGER, FIDELITY)"
    assert run_py(code).stdout.split() == ["False", "False"]
    assert run_py(code, HEROSIM_PULL_LEDGER="1").stdout.split() == ["True", "False"]
    assert run_py(code, HEROSIM_SNAPSHOT_FIDELITY="1").stdout.split() == ["True", "True"]
    bad = run_py(code, HEROSIM_PULL_LEDGER="yes")
    assert bad.returncode != 0 and "HEROSIM_PULL_LEDGER" in bad.stderr


def test_the_term_needs_the_ledger():
    code = ("from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler as C\n"
            "s = object.__new__(C)\n"
            "s._pg_batched = True\n"
            "s._pg_init()\n"
            "print(s.pg_pull_hold)")
    off = run_py(code)
    assert off.returncode == 0 and off.stdout.split()[-1] == "False", off.stderr[-400:]
    no_ledger = run_py(code, HEROSIM_PG_PULL_HOLD="1")
    assert no_ledger.returncode != 0 and "HEROSIM_PULL_LEDGER=1" in no_ledger.stderr
    on = run_py(code, HEROSIM_PG_PULL_HOLD="1", HEROSIM_PULL_LEDGER="1")
    assert on.returncode == 0 and on.stdout.split()[-1] == "True", on.stderr[-400:]


def test_harness_keeps_cd_pull_out_of_the_default_grid():
    sys.path.insert(0, os.path.join(REPO, "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert "cd_pull" in G.R1A_ARMS and "cd_pull" not in G.R1A_CLASSICAL
    assert G.RULE_POLICY["cd_pull"] == G.RULE_POLICY["cd_ledger"] == "peer_greedy_network_cd"
    assert "cd_ledger" in G.R1A_ARMS and "cd_ledger" not in G.R1A_CLASSICAL

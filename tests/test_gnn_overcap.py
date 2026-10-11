"""gnn_overcap_v1 (HEROSIM_PG_CD_OVERCAP=gnn): in a batch above the exact search's plan cap the refine returns the plan the GNN seeded it with, unchanged,
instead of the expansion fallback; a batch within the cap is refined and searched exactly as cd_exactS does. The toy is test_cd_expansion's three-node case:
the seeded split pair costs 16 s more than the pair on C, and 3 x 3 = 9 plans is above a cap of 8."""
import os

import pytest

from test_cd_expansion import _shell
from src.policy.peer_greedy_network import scheduler as S

SEED = {0: (1, 11), 1: (2, 22)}


def _refiner_ov(monkeypatch, **kw):
    """The flag is read by every scheduler's init and refused by the CD shell the toy borrows its primitives from, so it is lifted while that shell is built."""
    flag = os.environ.pop(S.PG_CD_OVERCAP_ENV, None)
    try:
        s, tasks, state = _shell(monkeypatch, **kw)
    finally:
        if flag is not None:
            os.environ[S.PG_CD_OVERCAP_ENV] = flag
    from types import SimpleNamespace

    from src.policy.peer_greedy_network.scheduler import GnnCdRefiner

    host = SimpleNamespace(nodes=s.nodes, _get_valid_replicas=s._get_valid_replicas, _orchestrator=s._pg_orchestrator)
    return GnnCdRefiner(host), tasks, state


def _env(monkeypatch, cap, overcap):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXACT_MAX_PLANS_ENV, str(cap))
    if overcap:
        monkeypatch.setenv(S.PG_CD_OVERCAP_ENV, overcap)
    else:
        monkeypatch.delenv(S.PG_CD_OVERCAP_ENV, raising=False)


def test_over_cap_with_the_path_on_serves_the_seed_and_runs_no_search(monkeypatch):
    _env(monkeypatch, 8, "gnn")
    r, tasks, state = _refiner_ov(monkeypatch)
    plan, info = r.refine(tasks, state, SEED, passes=3)
    assert plan == SEED and info == {"moved": 0, "node_change": 0, "platform_only": 0, "unstack": 0}
    assert (r.pg_overcap_pred, r.pg_overcap_gnn_batches, r.pg_overcap_gnn_tasks) == (1, 1, 2)
    assert (r.pg_exact_fallbacks, r.pg_exact_batches, r.pg_expand_batches, r.pg_cd_passes) == (0, 0, 0, 0)


def test_over_cap_with_the_path_off_is_the_expansion_fallback_and_the_predicate_matches_it(monkeypatch):
    _env(monkeypatch, 8, None)
    r, tasks, state = _refiner_ov(monkeypatch)
    plan, info = r.refine(tasks, state, SEED, passes=3)
    assert plan == {0: (3, 33), 1: (3, 33)} and info["moved"] == 2
    assert (r.pg_overcap_pred, r.pg_exact_fallbacks, r.pg_overcap_gnn_batches, r.pg_expand_batches) == (1, 1, 0, 1)


def test_within_the_cap_the_path_changes_nothing(monkeypatch):
    out = []
    for flag in (None, "gnn"):
        _env(monkeypatch, 100_000, flag)
        r, tasks, state = _refiner_ov(monkeypatch)
        out.append((r.refine(tasks, state, SEED, passes=3), r.pg_exact_batches, r.pg_exact_plans, r.pg_overcap_pred, r.pg_overcap_gnn_batches))
    assert out[0] == out[1] == (({0: (3, 33), 1: (3, 33)}, {"moved": 2, "node_change": 2, "platform_only": 0, "unstack": 0}), 1, 9, 0, 0)


def test_a_single_task_batch_is_never_over_the_cap(monkeypatch):
    _env(monkeypatch, 1, "gnn")
    r, tasks, state = _refiner_ov(monkeypatch)
    plan, _info = r.refine(tasks[:1], state, {0: (1, 11)}, passes=3)
    assert r.pg_overcap_pred == 0 and r.pg_overcap_gnn_batches == 0 and plan == {0: (1, 11)}


def test_a_pinned_task_counts_one_in_the_product(monkeypatch):
    _env(monkeypatch, 3, "gnn")   # 3 x 1 = 3 plans is within a cap of 3 once one task is pinned; 3 x 3 = 9 is not
    r, tasks, state = _refiner_ov(monkeypatch)
    monkeypatch.setattr(r, "_pg_plan_objects", lambda bt, st, pl: ([[1, 2, 3], [1, 2, 3]], {}, [set(), set()], [], {1}))
    assert r._pg_overcap_batch(tasks, state, SEED) is False
    monkeypatch.setattr(r, "_pg_plan_objects", lambda bt, st, pl: ([[1, 2, 3], [1, 2, 3]], {}, [set(), set()], [], set()))
    assert r._pg_overcap_batch(tasks, state, SEED) is True


def test_bad_values_and_the_wrong_shell_fail_loudly(monkeypatch):
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkCDScheduler

    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_OVERCAP_ENV, "yes")
    with pytest.raises(ValueError, match="must be gnn"):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_CD_OVERCAP_ENV, "gnn")
    with pytest.raises(RuntimeError, match="gnn_cd_refine only"):
        _shell(monkeypatch, cls=PeerGreedyNetworkCDScheduler)
    monkeypatch.delenv(S.PG_CD_EXACT_ENV)
    with pytest.raises(RuntimeError, match="needs"):
        _refiner_ov(monkeypatch)


def test_counters_are_exported_and_mirrored():
    from src.placement import orchestrator

    for name in ("pg_overcap_pred", "pg_overcap_gnn_batches", "pg_overcap_gnn_tasks"):
        assert name in S.PEER_GREEDY_COUNTERS and name in S.PG_SEARCH_COUNTERS
        assert f'"{name}"' in open(orchestrator.__file__).read()


def test_harness_serves_the_overcap_arms():
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert {"ra_gnn_eng_ovg", "ra_twin_eng_ovg", "ra_gnn_eng_ovo"} <= set(G.R1A_ARMS) and not set(G.R1A_OVERCAP) & set(G.RA_KINDS)
    assert "_ovg" in G.SUFFIXES and "_ovo" in G.SUFFIXES
    src = open(G.__file__).read()
    assert 'if kind.endswith("_ovg"):\n            env["HEROSIM_PG_CD_OVERCAP"] = "gnn"' in src
    assert '"HEROSIM_PG_CD_EXACT_PRUNE", "HEROSIM_PG_CD_OVERCAP", ' in src   # scrubbed per cell

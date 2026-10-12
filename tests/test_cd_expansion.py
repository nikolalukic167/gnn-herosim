"""cd_expand (HEROSIM_PG_CD_EXPANSION=1): exact-move alpha-expansion inside the batched CD refine. Off by default: the registered
peer_greedy_network_cd never calls it. The toy below is the case single-task moves cannot solve: two partners that each sit on the
node that executes them for free, paying a 10 s exchange, while a third node runs both at 1 s with no exchange. Moving one task alone
costs more than it saves (ICM is stuck); moving the pair is the improving move."""
from types import SimpleNamespace

import pytest

from src.policy.peer_greedy_network import scheduler as S
from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkBatchScheduler, PeerGreedyNetworkCDScheduler

EXCHANGE_S = 10.0


def _platform(pid, short):
    return SimpleNamespace(id=pid, type={"shortName": short}, initialized=SimpleNamespace(triggered=True),
                           _payload_transfer_time=lambda peer_node_name, payload: EXCHANGE_S,
                           peer_link_latency=lambda peer_node_name, context="": 0.0)


def _shell(monkeypatch, cls=PeerGreedyNetworkCDScheduler, exec_times=None):
    """A CD scheduler over three one-platform nodes A, B, C; drain, cold start, latency and I/O are zero, so S is exec + exchange
    (+ in-batch stacking). Task 0 runs free on A, task 1 free on B, both at 1 s on C; the two are partners."""
    monkeypatch.setenv("HEROSIM_PEER_EXCHANGE", "1")
    monkeypatch.setattr(S, "platform_queue_drain_seconds", lambda platform, orch, memo, exec_scale=1.0: 0.0)
    monkeypatch.setattr(S, "incoming_cold_start_time", lambda task, platform: 0.0)
    monkeypatch.setattr(S, "network_latency_between", lambda src, node, nodes: 0.0)
    monkeypatch.setattr(S, "_approx_comm", lambda task_type: 0.0)
    nodes = [SimpleNamespace(id=1, node_name="A"), SimpleNamespace(id=2, node_name="B"), SimpleNamespace(id=3, node_name="C")]
    reps = [(nodes[0], _platform(11, "pa")), (nodes[1], _platform(22, "pb")), (nodes[2], _platform(33, "pc"))]
    exec_times = exec_times or [{"pa": 0.0, "pb": 100.0, "pc": 1.0}, {"pa": 100.0, "pb": 0.0, "pc": 1.0}]
    tasks = [SimpleNamespace(id=i, type={"name": "cnn", "executionTime": et}, node_name="client0") for i, et in enumerate(exec_times)]
    s = object.__new__(cls)
    s._pg_init()
    s._get_valid_replicas = lambda replicas, task: sorted(replicas, key=lambda r: (r[0].id, r[1].id))
    s._pg_orchestrator = lambda: SimpleNamespace(peer_exchange={0: {1: 1.0}, 1: {0: 1.0}}, task_by_id={})
    s.nodes = SimpleNamespace(items=nodes)
    s.prefix_pairs_in_batch = 0
    s.prefix_peers_outside_batch = 0
    state = SimpleNamespace(replicas={"cnn": reps})
    return s, tasks, state


def test_flag_off_is_the_registered_cd_and_never_expands(monkeypatch):
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV, raising=False)
    s, tasks, state = _shell(monkeypatch)

    def never(*a, **k):
        raise AssertionError("expansion ran with the flag off")

    monkeypatch.setattr(s, "_pg_expand", never)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}   # ICM is stuck on the split pair
    assert s.pg_cd_passes == 1 and s.pg_cd_moves == 0
    assert (s.pg_expand_batches, s.pg_expand_sweeps, s.pg_expand_moves, s.pg_expand_tasks_moved, s.pg_expand_evals,
            s.pg_expand_gain_seconds, s.pg_expand_labels_skipped) == (0, 0, 0, 0, 0, 0.0, 0)


def test_flag_on_moves_the_pair_icm_cannot(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    assert (s.pg_expand_batches, s.pg_expand_moves, s.pg_expand_tasks_moved) == (1, 1, 2)
    # the start plan, then sweep 1: labels A (1 subset), B (1), C (3: {0}, {1}, {0,1}); sweep 2 finds nothing (A: 3, B: 3, C: none)
    assert s.pg_expand_sweeps == 2 and s.pg_expand_evals == 1 + 5 + 6
    # S summed: split pair 10 + 10 = 20 -> both on C: (1 + stacking 1) x 2 = 4
    assert s.pg_expand_gain_seconds == pytest.approx(16.0)
    # the single-task passes ran again on the new plan and kept it
    assert s.pg_cd_passes == 2 and s.pg_cd_moves == 0
    # the partner books count decisions only, not the scored trial plans: the first pass + two refine passes
    assert s.pg_partners_known == 1 + 2 + 2
    # the batch books describe the new plan
    assert s.pg_expand_labels_skipped == 0


def test_flag_on_leaves_a_plan_without_an_improving_subset_untouched(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    # C is now worse than the split: 20 s each, so neither a single move nor the pair improves on 10 + 10
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 0.0, "pb": 100.0, "pc": 20.0}, {"pa": 100.0, "pb": 0.0, "pc": 20.0}])
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert (s.pg_expand_batches, s.pg_expand_sweeps, s.pg_expand_moves, s.pg_expand_tasks_moved) == (1, 1, 0, 0)
    assert s.pg_expand_gain_seconds == 0.0 and s.pg_cd_passes == 1


def test_a_label_shared_by_too_many_tasks_is_skipped_and_counted(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_MAX_MOVERS_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}   # the pair move to C needs 2 movers
    assert s.pg_expand_labels_skipped == 1 and s.pg_expand_moves == 0


def test_a_task_held_outside_its_candidates_is_pinned(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 0.0, "pb": 0.0, "pc": 1.0}, {"pa": 100.0, "pb": 0.0, "pc": 1.0}])
    # task 0 may only take A at refine time but was placed on B before: it is scored where it is and never moved; with both
    # free on B and co-located, no subset of task 1 improves, so the plan stands
    only_a = {(1, 11)}
    s._pg_allowed = {0: only_a, 1: {(1, 11), (2, 22), (3, 33)}}
    placements, planned, service_of, committed = {0: (2, 22), 1: (2, 22)}, {0: "B", 1: "B"}, {0: ("B:22", 0.0), 1: ("B:22", 0.0)}, {"B:22": 0.0}
    moved = s._pg_expand(tasks, state, s._pg_orchestrator(), {}, committed, planned, placements, service_of)
    assert moved == 0 and placements == {0: (2, 22), 1: (2, 22)} and s.pg_expand_batches == 1


def test_bad_values_and_the_wrong_flavour_fail_loudly(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "2")
    with pytest.raises(ValueError, match=S.PG_CD_EXPANSION_ENV):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_SWEEPS_ENV, "0")
    with pytest.raises(ValueError, match=S.PG_CD_EXPANSION_SWEEPS_ENV):
        _shell(monkeypatch)
    monkeypatch.delenv(S.PG_CD_EXPANSION_SWEEPS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="has no refine to extend"):
        _shell(monkeypatch, cls=PeerGreedyNetworkBatchScheduler)


def test_counters_are_exported():
    import inspect

    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in ("pg_expand_batches", "pg_expand_sweeps", "pg_expand_moves", "pg_expand_tasks_moved", "pg_expand_evals",
              "pg_expand_gain_seconds", "pg_expand_labels_skipped"):
        assert k in S.PEER_GREEDY_COUNTERS
        assert f'"{k}"' in exporter   # the orchestrator's schedulerCounters whitelist, which the gate summaries read


def test_harness_serves_cd_expand_as_a_named_diagnostic_arm():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert G.RULE_POLICY["cd_expand"] == "peer_greedy_network_cd"
    assert "cd_expand" in G.R1A_DIAG and "cd_expand" in G.R1A_ARMS
    src = open(G.__file__).read()
    assert 'if kind == "cd_expand":\n            env["HEROSIM_PG_CD_EXPANSION"] = "1"' in src
    assert '"HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "GNN_SLATE_NO_SPLIT", "GNN_SELF_SEARCH", *KEEPWARM_ENV' in src   # scrubbed per cell before the kind sets it


# ---- cdxapply: the CD refine of a seeded (learned) plan, with the expansion ----

def _refiner(monkeypatch, exec_times=None):
    """A GnnCdRefiner over the same three-node toy, held by a fake GNN host."""
    from src.policy.peer_greedy_network.scheduler import GnnCdRefiner

    s, tasks, state = _shell(monkeypatch, exec_times=exec_times)   # only for the patched primitives, nodes and replicas
    host = SimpleNamespace(nodes=s.nodes, _get_valid_replicas=s._get_valid_replicas, _orchestrator=s._pg_orchestrator)
    r = GnnCdRefiner(host)
    return r, tasks, state


def test_refiner_flag_off_runs_the_passes_only_and_never_expands(monkeypatch):
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV, raising=False)
    r, tasks, state = _refiner(monkeypatch)

    def never(*a, **k):
        raise AssertionError("expansion ran with the flag off")

    monkeypatch.setattr(r, "_pg_expand", never)
    plan, info = r.refine(tasks, state, {0: (1, 11), 1: (2, 22)}, passes=3)
    assert plan == {0: (1, 11), 1: (2, 22)} and info["moved"] == 0     # ICM is stuck on the split pair
    assert r.pg_expand_batches == 0 and r.pg_expand_moves == 0


def test_refiner_flag_on_moves_the_seeded_pair(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    r, tasks, state = _refiner(monkeypatch)
    plan, info = r.refine(tasks, state, {0: (1, 11), 1: (2, 22)}, passes=3)
    assert plan == {0: (3, 33), 1: (3, 33)}
    assert info["moved"] == 2 and info["node_change"] == 2
    assert (r.pg_expand_batches, r.pg_expand_moves, r.pg_expand_tasks_moved) == (1, 1, 2)
    assert r.pg_expand_gain_seconds == pytest.approx(16.0)


def test_refiner_flag_on_keeps_a_seed_it_cannot_improve(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    r, tasks, state = _refiner(monkeypatch, exec_times=[{"pa": 0.0, "pb": 100.0, "pc": 20.0}, {"pa": 100.0, "pb": 0.0, "pc": 20.0}])
    plan, info = r.refine(tasks, state, {0: (1, 11), 1: (2, 22)}, passes=3)
    assert plan == {0: (1, 11), 1: (2, 22)} and info["moved"] == 0
    assert (r.pg_expand_batches, r.pg_expand_moves) == (1, 0)


def test_gnn_host_mirrors_the_refiner_expansion_counters(monkeypatch):
    from src.policy.gnn.scheduler import GNNScheduler

    host = object.__new__(GNNScheduler)
    for k in ("cdr_batches", "cdr_batches_changed", "cdr_tasks", "cdr_moved", "cdr_node_change", "cdr_platform_only", "cdr_unstack"):
        setattr(host, k, 0)
    counters = {k: (9.5 if k.endswith("_seconds") else i + 1) for i, k in enumerate(S.PG_SEARCH_COUNTERS)}
    stub = SimpleNamespace(pg_cd_expansion=True, refine=lambda tasks, state, seed, passes: ({0: (3, 33)}, {"moved": 1, "node_change": 1, "platform_only": 0, "unstack": 0}),
                           **counters)
    host._cd_refiner = stub
    out = host._cd_refine([SimpleNamespace(id=0)], {0: (1, 11)}, SimpleNamespace(replicas={}), "apply")
    assert out == {0: (3, 33)} and host.cdr_batches == 1 and host.cdr_moved == 1
    assert {k: getattr(host, k) for k in S.PG_SEARCH_COUNTERS} == counters
    # with the flag off on the refiner, the host carries no expansion books at all
    host2 = object.__new__(GNNScheduler)
    for k in ("cdr_batches", "cdr_batches_changed", "cdr_tasks", "cdr_moved", "cdr_node_change", "cdr_platform_only", "cdr_unstack"):
        setattr(host2, k, 0)
    host2._cd_refiner = SimpleNamespace(pg_cd_expansion=False, refine=stub.refine, **counters)
    host2._cd_refine([SimpleNamespace(id=0)], {0: (1, 11)}, SimpleNamespace(replicas={}), "apply")
    assert not any(hasattr(host2, k) for k in S.PG_SEARCH_COUNTERS)


def test_harness_serves_cdxapply_as_a_named_learned_diagnostic():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert "_cdxapply" in G.SUFFIXES and G.R1A_CDX == ("ra_gnn_eng_cdxapply", "ra_gnn_eng_physmp_cdxapply")
    for k in G.R1A_CDX:
        assert k in G.R1A_ARMS and next((k[:-len(s)] for s in G.SUFFIXES if k.endswith(s)), k) in G.RA_KINDS
    src = open(G.__file__).read()
    assert 'if kind.endswith("_cdxapply"):\n            env.update(GNN_CD_REFINE="apply", HEROSIM_PG_CD_EXPANSION="1", GNN_SLATE_NO_SPLIT="1")' in src
    assert 'R1A_RANDOM + R1A_DIAG + R1A_NOSPLIT + R1A_SELFSEARCH + R1A_CDX)' in src   # out of the default grid
    assert 'expands = kind.endswith("_cdxapply") or kind == "cd_expand"' in src
    # the check reads the flag back from run_provenance, which records a whitelist of env keys
    import inspect

    from src import executesimulation

    assert '"HEROSIM_PG_CD_EXPANSION"' in inspect.getsource(executesimulation.build_run_provenance)
    assert 'kind.endswith(("_nosplit", "_cdxapply", "_selfsearch"))' in src


# ---- cd_exactS: exact search over S on the top-K slate ----

def test_exact_flag_off_is_the_registered_cd_and_never_searches(monkeypatch):
    monkeypatch.delenv(S.PG_CD_EXACT_ENV, raising=False)
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV, raising=False)
    s, tasks, state = _shell(monkeypatch)

    def never(*a, **k):
        raise AssertionError("exact search ran with the flag off")

    monkeypatch.setattr(s, "_pg_exact", never)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert all(getattr(s, k) == 0 for k in S.PG_EXACT_COUNTERS)


def test_exact_finds_the_pair_plan_icm_cannot(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}
    # 3 couples per task, all within the top-5 slate: 9 plans enumerated, no tie at the optimum, 2 tasks moved
    assert (s.pg_exact_batches, s.pg_exact_plans, s.pg_exact_ties, s.pg_exact_fallbacks, s.pg_exact_kept_pass, s.pg_exact_tasks_moved) == (1, 9, 0, 0, 0, 2)
    assert s.pg_exact_gain_seconds == pytest.approx(16.0)   # 20 s split -> 4 s co-located, as the expansion found
    assert s.pg_expand_batches == 0                        # no fallback
    assert s.pg_cd_passes == 2 and s.pg_cd_moves == 0      # the passes ran again on the new plan and kept it


def test_exact_keeps_the_pass_plan_when_nothing_in_the_slate_is_cheaper(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 0.0, "pb": 100.0, "pc": 20.0}, {"pa": 100.0, "pb": 0.0, "pc": 20.0}])
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert (s.pg_exact_batches, s.pg_exact_plans, s.pg_exact_kept_pass, s.pg_exact_tasks_moved) == (1, 9, 1, 0)
    assert s.pg_cd_passes == 1


def test_exact_ties_go_to_the_lowest_couple_order_and_are_counted(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    # A and C are equally good for the pair (both 1 s exec, exchange 0 when co-located); B is free for nobody
    s, tasks, state = _shell(monkeypatch, exec_times=[{"pa": 1.0, "pb": 100.0, "pc": 1.0}, {"pa": 1.0, "pb": 100.0, "pc": 1.0}])
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (1, 11)}   # the pass already co-locates on A; (C, C) ties it
    assert s.pg_exact_ties == 1 and s.pg_exact_kept_pass == 1


def test_exact_falls_back_to_the_expansion_above_the_plan_cap(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXPANSION_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXACT_MAX_PLANS_ENV, "8")   # 9 plans > 8
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (3, 33), 1: (3, 33)}   # the expansion finds the pair move
    assert s.pg_exact_fallbacks == 1 and s.pg_exact_batches == 0 and s.pg_expand_batches == 1 and s.pg_expand_moves == 1


def test_exact_above_the_cap_without_the_expansion_leaves_the_pass_plan(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.delenv(S.PG_CD_EXPANSION_ENV, raising=False)
    monkeypatch.setenv(S.PG_CD_EXACT_MAX_PLANS_ENV, "8")
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert s.pg_exact_fallbacks == 1 and s.pg_expand_batches == 0


def test_exact_top_k_narrows_the_slate(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXACT_TOP_K_ENV, "1")   # each task keeps only its cheapest standalone couple: A for 0, B for 1
    s, tasks, state = _shell(monkeypatch)
    assert s._pg_decide(tasks, state) == {0: (1, 11), 1: (2, 22)}
    assert s.pg_exact_plans == 1 and s.pg_exact_kept_pass == 1


def test_exact_in_the_seeded_refine(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    r, tasks, state = _refiner(monkeypatch)
    plan, info = r.refine(tasks, state, {0: (1, 11), 1: (2, 22)}, passes=3)
    assert plan == {0: (3, 33), 1: (3, 33)} and info["moved"] == 2
    assert (r.pg_exact_batches, r.pg_exact_plans, r.pg_exact_tasks_moved) == (1, 9, 2)


def test_exact_bad_values_and_wrong_flavour_fail_loudly(monkeypatch):
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "yes")
    with pytest.raises(ValueError, match=S.PG_CD_EXACT_ENV):
        _shell(monkeypatch)
    monkeypatch.setenv(S.PG_CD_EXACT_ENV, "1")
    monkeypatch.setenv(S.PG_CD_EXACT_MAX_PLANS_ENV, "0")
    with pytest.raises(ValueError, match=S.PG_CD_EXACT_MAX_PLANS_ENV):
        _shell(monkeypatch)
    monkeypatch.delenv(S.PG_CD_EXACT_MAX_PLANS_ENV, raising=False)
    with pytest.raises(RuntimeError, match="has no refine to extend"):
        _shell(monkeypatch, cls=PeerGreedyNetworkBatchScheduler)


def test_exact_counters_are_exported_and_the_flag_is_in_provenance():
    import inspect

    from src import executesimulation
    from src.placement import orchestrator

    exporter = inspect.getsource(orchestrator.Orchestrator._scheduler_counters)
    for k in S.PG_EXACT_COUNTERS:
        assert k in S.PEER_GREEDY_COUNTERS and f'"{k}"' in exporter
    prov = inspect.getsource(executesimulation.build_run_provenance)
    for flag in ("HEROSIM_PG_CD_EXACT", "HEROSIM_PG_CD_SWAP", "HEROSIM_PG_FUSE"):
        assert f'"{flag}"' in prov


def test_harness_serves_cd_exactS_as_a_named_diagnostic_arm():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_cosim"))
    import fresh_topo_burst_v1_gate as G

    assert G.RULE_POLICY["cd_exactS"] == "peer_greedy_network_cd"
    assert "cd_exactS" in G.R1A_DIAG and "cd_exactS" in G.R1A_ARMS
    src = open(G.__file__).read()
    assert 'if kind == "cd_exactS":\n            env.update(HEROSIM_PG_CD_EXACT="1", HEROSIM_PG_CD_EXPANSION="1")' in src
    assert '"HEROSIM_PG_CD_EXPANSION", "HEROSIM_PG_CD_EXACT", "GNN_SLATE_NO_SPLIT", "GNN_SELF_SEARCH", *KEEPWARM_ENV' in src   # scrubbed per cell
    assert 'if kind == "cd_exactS" and (out["env"].get("HEROSIM_PG_CD_EXACT") != "1"' in src

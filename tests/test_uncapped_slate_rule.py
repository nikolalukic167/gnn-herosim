"""declared_pruning_v1_uncapped: every candidate kept (same rank order), the plan cap and sub-batching unchanged; the cache accepts the one
mixed corpus {declared_pruning_v1, declared_pruning_v1_uncapped}; a checkpoint trained on it is served uncapped; the old rule is unchanged."""
import random
from pathlib import Path

import pytest

from src.placement import declared_slate as D

ROOT = Path(__file__).resolve().parents[1]


def cand(node, plat, drain=0.0):
    return {"node_id": node, "platform_id": plat, "queue_drain_seconds": drain, "execution_time": 1.0, "network_latency": 0.0,
            "initialized": True, "cold_start_time": 5.0}


def _batch(rng, n_tasks, max_c):
    return [{"task_id": i, "candidates": [cand(n, rng.randrange(3), drain=float(rng.randrange(4))) for n in range(rng.randrange(1, max_c + 1))]}
            for i in range(n_tasks)]


def test_old_rule_is_byte_identical_flag_off():
    rng = random.Random(7)
    for _ in range(40):
        tasks = _batch(rng, rng.randrange(1, 9), 12)
        pairs = [(a, b) for a in range(len(tasks)) for b in range(a + 1, len(tasks)) if rng.random() < 0.4]
        base = D.slate(tasks, pairs)
        assert D.slate(tasks, pairs, None) == base and D.slate(tasks, pairs, D.RULE) == base
        # the pre-change definition, written out
        kept = [D.prune_candidates(t["candidates"]) for t in tasks]
        assert base.kept == kept and base.full_sizes == [len(t["candidates"]) for t in tasks]
        assert base.plans == D.plan_space(len(k) for k in kept) and base.pruned == any(len(t["candidates"]) > D.TOP_K for t in tasks)


def test_uncapped_keeps_every_candidate_in_rank_order():
    tasks = [{"task_id": 0, "candidates": [cand(n, 0, drain=float(n % 3)) for n in range(9)]}, {"task_id": 1, "candidates": [cand(n, 0) for n in range(7)]}]
    s = D.slate(tasks, [(0, 1)], D.RULE_UNCAPPED)
    assert [len(k) for k in s.kept] == [9, 7] and s.plans == 63 and s.pruned is False and s.groups == [[0, 1]]
    assert s.kept[0] == D.prune_candidates(tasks[0]["candidates"], 9)
    assert s.kept[0][:5] == D.prune_candidates(tasks[0]["candidates"])         # the old slate is the head of the new one


def test_uncapped_sub_batches_on_the_full_product(monkeypatch):
    tasks = [{"task_id": i, "candidates": [cand(n, 0) for n in range(9)]} for i in range(6)]    # 5^6 = 15,625 pruned; 9^6 = 531,441 full
    assert not D.slate(tasks, [(0, 1)]).sub_batched
    assert D.slate(tasks, [(0, 1)], D.RULE_UNCAPPED).sub_batched
    assert all(len(g) <= D.SUB_BATCH for g in D.slate(tasks, [(0, 1)], D.RULE_UNCAPPED).groups)


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="unknown candidate slate rule"):
        D.slate([{"task_id": 0, "candidates": [cand(0, 0)]}], [], "declared_pruning_v2")


def test_serving_mode_accepts_the_uncapped_rule_and_still_refuses_mismatches(monkeypatch):
    monkeypatch.setenv(D.ENV, D.RULE_UNCAPPED)
    assert D.serving_slate() == D.RULE_UNCAPPED
    D.require_matching_slate(D.RULE_UNCAPPED, what="m")
    for trained in (None, D.RULE):
        with pytest.raises(ValueError, match="candidate_slate"):
            D.require_matching_slate(trained, what="m")
    monkeypatch.setenv(D.ENV, D.RULE)
    with pytest.raises(ValueError, match="candidate_slate"):
        D.require_matching_slate(D.RULE_UNCAPPED, what="m")
    D.require_matching_slate(D.RULE, what="m")
    monkeypatch.delenv(D.ENV)
    D.require_matching_slate(None, what="m")
    with pytest.raises(ValueError, match="candidate_slate"):
        D.require_matching_slate(D.RULE_UNCAPPED, what="m")
    monkeypatch.setenv(D.ENV, "declared_pruning_v2")
    with pytest.raises(ValueError, match="expected unset"):
        D.serving_slate()


def test_the_serving_decoder_passes_the_serving_rule_to_the_slate():
    src = (ROOT / "src" / "policy" / "gnn" / "scheduler.py").read_text()
    assert "declared_slate.slate(payloads, pairs, declared_slate.serving_slate())" in src


def _ds(*rules):
    return {f"ds_{i}": {"candidate_slate": r} for i, r in enumerate(rules)}


def test_cache_slate_rule_single_mixed_and_refused():
    from src.notebooks.prepare_graphs_cache import _candidate_slate_rules, _single_candidate_slate as one

    assert one(_ds(D.RULE, D.RULE)) == D.RULE and _candidate_slate_rules(_ds(D.RULE, D.RULE)) is None
    assert one(_ds(D.RULE_UNCAPPED)) == D.RULE_UNCAPPED
    assert one(_ds(None, None)) is None and one({}) is None
    mixed = _ds(D.RULE, D.RULE_UNCAPPED, D.RULE)
    assert one(mixed) == D.RULE_UNCAPPED and _candidate_slate_rules(mixed) == [D.RULE, D.RULE_UNCAPPED]
    for bad in (_ds(D.RULE, None), _ds(D.RULE_UNCAPPED, None), _ds(D.RULE, D.RULE_UNCAPPED, "x")):
        with pytest.raises(RuntimeError, match="different candidate slates"):
            one(bad)

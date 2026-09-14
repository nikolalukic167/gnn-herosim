"""Tests for the drainable_debug_v1 D1 one-step read, written before any dataset was cut.

D1 is the read that decides which of three hypotheses the lineage pursues, so the two ways it
could quietly lie both have tests here: reconstructing the reactive rule wrongly (which would
make the reactive arm look better or worse than it is on exactly the states being compared),
and treating a plan that is missing from the sweep as a dataset to skip (which would bias the
whole read toward states where the policy agreed with the candidate subsample).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_cosim.drainable_debug_one_step_read import (  # noqa: E402
    D1A_SHORTEST_QUEUE_MAX_PCT,
    D1B_CHECKPOINT_MIN_PCT,
    D1_MIN_DATASETS,
    index_rows,
    plan_cost,
    regret_pct,
    shortest_queue_plan,
    summarise,
    verdict,
)


def _cand(node_id, plat_id, depth, initialized=True):
    return {
        "node_id": node_id,
        "platform_id": plat_id,
        "queue_key": f"node{node_id}:{plat_id}",
        "queue_length": depth,
        "initialized": initialized,
    }


def _snapshot(tasks):
    return {"tasks": tasks}


def test_shortest_queue_takes_the_shallowest_candidate():
    snap = _snapshot([{"task_id": 5, "candidates": [_cand(0, 10, 7), _cand(1, 11, 2)]}])
    assert shortest_queue_plan(snap, {5: 0}) == {0: (1, 11)}


def test_shortest_queue_sees_its_own_previous_placement():
    """The live scheduler pulls one task at a time, so the second task sees the first one's
    enqueue. Scoring both against the frozen snapshot would put them on the same replica and
    flatter the reactive arm on exactly the comparison this read exists to make."""
    snap = _snapshot([
        {"task_id": 0, "candidates": [_cand(0, 10, 0), _cand(1, 11, 0)]},
        {"task_id": 1, "candidates": [_cand(0, 10, 0), _cand(1, 11, 0)]},
    ])
    plan = shortest_queue_plan(snap, {0: 0, 1: 1})
    # Both start empty, so task 0 takes the lower-id replica on the tie-break; task 1 must
    # then see that replica at depth 1 and take the other.
    assert plan[0] == (0, 10)
    assert plan[1] == (1, 11), "the second task must not stack onto the replica just filled"


def test_shortest_queue_prefers_an_initialized_replica_even_when_deeper():
    """`KnativeScheduler.placement` filters to initialized replicas first and only falls back
    to the full set when none is initialized."""
    snap = _snapshot([{"task_id": 0, "candidates": [_cand(0, 10, 5, True), _cand(1, 11, 0, False)]}])
    assert shortest_queue_plan(snap, {0: 0}) == {0: (0, 10)}


def test_shortest_queue_falls_back_when_nothing_is_initialized():
    snap = _snapshot([{"task_id": 0, "candidates": [_cand(0, 10, 5, False), _cand(1, 11, 0, False)]}])
    assert shortest_queue_plan(snap, {0: 0}) == {0: (1, 11)}


def test_shortest_queue_breaks_ties_by_node_then_platform_id():
    """The live rule's tie-break is `(len(queue), node.id, platform.id)`. A different
    tie-break is a different plan and would be scored as the reactive arm's."""
    snap = _snapshot([{"task_id": 0, "candidates": [_cand(3, 9, 0), _cand(1, 12, 0), _cand(1, 4, 0)]}])
    assert shortest_queue_plan(snap, {0: 0}) == {0: (1, 4)}


def test_shortest_queue_places_in_arrival_order_not_cosim_order():
    """Co-sim task ids regroup the batch by application; the live scheduler saw arrival order.
    Task 7 arrives after task 3 even though it may carry the lower co-sim id."""
    snap = _snapshot([
        {"task_id": 7, "candidates": [_cand(0, 10, 0), _cand(1, 11, 0)]},
        {"task_id": 3, "candidates": [_cand(0, 10, 0), _cand(1, 11, 0)]},
    ])
    plan = shortest_queue_plan(snap, {3: 1, 7: 0})
    # Trace id 3 arrives first, so it takes node0:10 even though its co-sim id is 1; trace id 7
    # is placed second and gets node1:11. Reading in co-sim order would swap these.
    assert plan[1] == (0, 10)
    assert plan[0] == (1, 11)


def test_a_snapshot_task_missing_from_the_dataset_fails_loud():
    snap = _snapshot([{"task_id": 9, "candidates": [_cand(0, 10, 0)]}])
    with pytest.raises(SystemExit) as exc:
        shortest_queue_plan(snap, {0: 0})
    assert "no co-sim id" in str(exc.value)


def test_a_plan_outside_the_sweep_fails_loud_rather_than_being_skipped():
    """Skipping such a dataset would keep only the states where the policy agreed with the
    candidate subsample -- the exact states on which it looks best."""
    rows = [({0: (0, 10)}, 100.0), ({0: (1, 11)}, 120.0)]
    index = index_rows(rows)
    assert plan_cost({0: (1, 11)}, index, "ds_0") == 120.0
    with pytest.raises(SystemExit) as exc:
        plan_cost({0: (2, 12)}, index, "ds_0")
    assert "not in the enumerated sweep" in str(exc.value)
    assert "force-candidates-from-plans" in str(exc.value)


def test_regret_is_relative_to_the_sweep_optimum():
    assert regret_pct(110.0, 100.0) == pytest.approx(10.0)
    assert regret_pct(100.0, 100.0) == pytest.approx(0.0)
    with pytest.raises(SystemExit):
        regret_pct(1.0, 0.0)


def _rows(jsq, gnn):
    return [{"dataset": f"ds_{i}", "shortest_queue": a, "gnn": b} for i, (a, b) in enumerate(zip(jsq, gnn))]


def test_h_env_fires_when_the_reactive_rule_is_already_near_optimal():
    n = D1_MIN_DATASETS
    summary = summarise(_rows([0.5] * n, [14.0] * n), ["gnn"])
    out = verdict(summary)
    assert out["d1a"] == "H-ENV-SUPPORTED"
    assert out["d1b"] == "H-GAP-SUPPORTED"


def test_h_myopia_fires_when_the_checkpoint_beats_the_reactive_rule_one_step():
    n = D1_MIN_DATASETS
    out = verdict(summarise(_rows([12.0] * n, [4.0] * n), ["gnn"]), "gnn")
    assert out["d1c"] == "H-MYOPIA-SUPPORTED"
    assert "d1b" not in out


def test_a_source_below_the_minimum_is_void():
    out = verdict(summarise(_rows([0.5] * 5, [14.0] * 5), ["gnn"]), "gnn")
    assert out["verdict"] == "VOID"


def test_nothing_fires_in_the_middle():
    n = D1_MIN_DATASETS
    out = verdict(summarise(_rows([6.0] * n, [7.0] * n), ["gnn"]), "gnn")
    assert out["verdict"] == "NONE-FIRED"


def test_the_bars_are_the_registered_ones():
    assert D1A_SHORTEST_QUEUE_MAX_PCT == 3.0
    assert D1B_CHECKPOINT_MIN_PCT == 10.0
    assert D1_MIN_DATASETS == 40


# --- the corpus-cutting side of the same problem ---------------------------------------------

from scripts_cosim.make_warm_corpus import (  # noqa: E402
    choose_candidates,
    reactive_plan_keys,
)


def _snap_for_cut(n_tasks=2, n_cands=4):
    return {
        "tasks": [
            {
                "task_id": i,
                "task_type": "dnn1",
                "candidates": [_cand(c, 10 + c, depth=c) for c in range(n_cands)],
            }
            for i in range(n_tasks)
        ]
    }


def test_reactive_plan_keys_matches_the_read_side_rule():
    """The cutter and the reader must agree on what the reactive plan IS, or the cutter
    forces in one replica while the reader scores another and every dataset fails loud."""
    snap = _snap_for_cut(n_tasks=3, n_cands=4)
    keys = reactive_plan_keys(snap)
    plan = shortest_queue_plan(snap, {i: i for i in range(3)})
    from_plan = {f"node{n}:{p}" for n, p in plan.values()}
    assert keys["dnn1"] == from_plan


def test_forced_keys_survive_the_candidate_draw():
    """The draw is for sweep size and can drop the replica the reactive rule uses; with the
    flag, it cannot."""
    import random

    snap = _snap_for_cut(n_tasks=2, n_cands=4)
    forced = reactive_plan_keys(snap)
    # target_combos=4 forces a small subset (2 candidates per task at most)
    subset, record = choose_candidates(
        snap, random.Random(0), target_combos=4, max_combos=100, force_keys=forced
    )
    for ttype, keys in forced.items():
        assert keys <= subset[ttype], f"{ttype}: forced keys {keys} not in draw {subset[ttype]}"
    assert record["forced_keys"] == {t: sorted(k) for t, k in forced.items()}


def test_without_the_flag_the_record_says_so():
    import random

    snap = _snap_for_cut(n_tasks=2, n_cands=4)
    _subset, record = choose_candidates(snap, random.Random(0), target_combos=4, max_combos=100)
    assert record["forced_keys"] is None


def test_decoded_file_provenance_blocks_are_not_read_as_datasets(tmp_path):
    """The decode driver writes a `_stats` block next to the plans; treating it as a dataset
    would put a bogus entry in the corpus map."""
    from scripts_cosim.drainable_debug_one_step_read import load_decoded

    p = tmp_path / "decoded.json"
    p.write_text(json.dumps({
        "_stats": {"gnn": {"decoded": 60, "infeasible": 0}},
        "ds_00000": {"gnn": {"0": [1, 11]}},
    }))
    out = load_decoded(p)
    assert set(out) == {"ds_00000"}
    assert out["ds_00000"]["gnn"] == {0: (1, 11)}

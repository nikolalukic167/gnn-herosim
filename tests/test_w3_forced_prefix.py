"""W3 label-prefix control: the forced-prefix scorer wrapper (scripts_cosim/uncut_w3_control.py)."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts_cosim"))
import uncut_w3_control as C  # noqa: E402
from src.policy.gnn.seq_decode import decode_masked_topo_placement  # noqa: E402

N, K = 7, 4
CANDS = {t: [(i, 0) for i in range(K)] for t in range(N)}
PARENTS = {t: ([t - 1] if t else []) for t in range(N)}


def make_scorer(seed):
    calls = []

    def sf(task, committed):
        calls.append((task, dict(committed)))
        rng = random.Random(f"{seed}:{task}:{sorted(committed.items())}")
        return [rng.random() for _ in range(K)]
    return sf, calls


def run(sf):
    return decode_masked_topo_placement([None] * N, CANDS, N, dag_parents=PARENTS, node_caps={}, demands={t: [0.0] * K for t in range(N)}, score_fn=sf, allow_replica_reuse=True, relax_on_stuck=True)


@pytest.mark.parametrize("seed", range(5))
def test_nothing_forced_reproduces_the_plain_decode(seed):
    label = [[(seed + t) % K, 0] for t in range(N)]
    plain = run(make_scorer(seed)[0])
    assert run(C.force_prefix_score_fn(make_scorer(seed)[0], CANDS, label, set())) == plain


@pytest.mark.parametrize("seed", range(5))
def test_all_forced_is_the_label_and_a_prefix_is_kept(seed):
    label = [[(3 * seed + t) % K, 0] for t in range(N)]
    lab = [tuple(x) for x in label]
    assert [tuple(x) for x in run(C.force_prefix_score_fn(make_scorer(seed)[0], CANDS, label, set(range(N))))] == lab
    sf, calls = make_scorer(seed)
    plan = run(C.force_prefix_score_fn(sf, CANDS, label, {0, 1, 2}))
    assert [tuple(x) for x in plan[:3]] == lab[:3]
    assert [c[0] for c in calls] == list(range(N))                    # the scorer was called at every step, in order
    assert all(calls[t][1].get(u) == lab[u] for t in (3, 4) for u in (0, 1, 2))      # later tasks see the label placements as committed


def test_missing_label_candidate_fails_loudly():
    sf, _ = make_scorer(0)
    with pytest.raises(RuntimeError, match="FAIL LOUD"):
        run(C.force_prefix_score_fn(sf, CANDS, [[9, 9]] * N, {0}))


def test_plan_comparison_ignores_list_versus_tuple_and_nothing_else():
    a = [(1, 2), (3, 4)]                       # S5's decode_group: a list of tuples
    b = [[1, 2], [3, 4]]                       # forced_decode: a list of lists
    assert a != b and C.same_plan(a, b)
    assert not C.same_plan(a, [[1, 2], [3, 5]])
    assert not C.same_plan(a, [[1, 2]])
    assert C.same_plan(None, None) and not C.same_plan(a, None)

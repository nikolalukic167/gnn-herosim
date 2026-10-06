import numpy as np
import pytest
from src.placement.radical.block_moves import scaled_problem, baseline
from src.placement.radical.joint_dispatch import JointDispatch
from src.placement.radical.dispatch_live import replay


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    return JointDispatch(tmp_path_factory.mktemp('joint'))


@pytest.mark.parametrize('mode', range(4))
@pytest.mark.parametrize('anneal', [False, True])
def test_joint_search_exact_and_nonmutating(engine, mode, anneal):
    b = scaled_problem(991001, 4, 4, 8)
    base, a, rank = baseline(engine, b)
    old_a, old_rank = a.copy(), rank.copy()
    cost, chosen, priority = engine.joint_search(b, a, rank, 96, mode, anneal)
    native, ends = engine.score_priority(b, chosen, priority)
    ref = replay(b, chosen, priority)
    assert cost == native == ref['objective'] and cost <= base
    assert np.array_equal(ends, ref['ends'])
    assert np.array_equal(a, old_a) and np.array_equal(rank, old_rank)
    again = engine.joint_search(b, a, rank, 96, mode, anneal)
    assert cost == again[0] and np.array_equal(chosen, again[1]) and np.array_equal(priority, again[2])
    if mode == 0:
        assert np.array_equal(priority, rank)
    if mode == 1:
        assert np.array_equal(chosen, a)


def test_joint_zero_and_bad_arguments(engine):
    b = scaled_problem(991002, 3, 3, 4)
    base, a, rank = baseline(engine, b)
    cost, chosen, priority = engine.joint_search(b, a, rank, 0)
    assert cost == base and np.array_equal(chosen, a) and np.array_equal(priority, rank)
    for kwargs in ({'steps': -1}, {'steps': 2, 'mode': 8}, {'steps': 2, 'seed': -1}):
        with pytest.raises(ValueError):
            engine.joint_search(b, a, rank, **kwargs)


def test_timed_joint_retains_feasible_best(engine):
    b = scaled_problem(991003, 4, 4, 8)
    base, a, rank = baseline(engine, b)
    cost, chosen, priority, count = engine.joint_timed(b, a, rank, .02)
    assert count > 0 and cost <= base
    assert cost == replay(b, chosen, priority)['objective']
    for seconds in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            engine.joint_timed(b, a, rank, seconds)

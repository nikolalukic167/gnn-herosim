import itertools
import numpy as np
import pytest
from src.placement.radical.block_moves import scaled_problem, baseline
from src.placement.radical.dispatch_live import replay
from src.placement.radical.region_dispatch import RegionDispatch, region_pool


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    return RegionDispatch(tmp_path_factory.mktemp('region_build'))


@pytest.mark.parametrize('nodes', [(0,), (0, 3), (0, 3, 5), (0, 1, 3, 5)])
def test_exact_region_matches_independent_enumeration(engine, nodes):
    b = scaled_problem(551, 2, 3, 4)
    base, a, r = baseline(engine, b)
    original_a, original_r = a.copy(), r.copy()
    choices = [np.flatnonzero(np.isfinite(b['p'].reshape(-1, 4)[i])) for i in nodes]
    optimum = base
    for hosts in itertools.product(*choices):
        for ranks in itertools.permutations(r.ravel()[list(nodes)]):
            aa, rr = a.copy(), r.copy()
            aa.ravel()[list(nodes)] = hosts
            rr.ravel()[list(nodes)] = ranks
            optimum = min(optimum, replay(b, aa, rr)['objective'])
    cost, aa, rr = engine.repair(b, a, r, nodes)
    assert cost == optimum == replay(b, aa, rr)['objective']
    assert np.array_equal(a, original_a) and np.array_equal(r, original_r)
    outside = sorted(set(range(a.size)) - set(nodes))
    assert np.array_equal(aa.ravel()[outside], a.ravel()[outside])
    assert np.array_equal(rr.ravel()[outside], r.ravel()[outside])


def test_region_validation(engine):
    b = scaled_problem(552, 2, 3, 4)
    _, a, r = baseline(engine, b)
    for nodes in ([], [0, 0], [6], [-1], [1.5], [0, 1, 2, 3, 4]):
        with pytest.raises(ValueError):
            engine.repair(b, a, r, nodes)


def test_region_pool_deterministic_and_unique(engine):
    b = scaled_problem(553, 4, 4, 4)
    _, a, r = baseline(engine, b)
    first = region_pool(b, a, r, 32)
    assert first == region_pool(b, a, r, 32)
    assert len(first) == len(set(first)) == 32
    assert all(len(set(region)) == 4 and min(region) >= 0 and max(region) < a.size for region in first)
    with pytest.raises(ValueError):
        region_pool(b, a, r, 2000)

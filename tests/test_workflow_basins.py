import numpy as np
import pytest
from src.placement.workflow_basins import BasinPortfolio
from src.placement.workflow_planning import instance,evaluate,route_greedy

@pytest.fixture(scope='module')
def portfolio(tmp_path_factory):return BasinPortfolio(tmp_path_factory.mktemp('basin_native'))

@pytest.mark.parametrize('seed',[0,2,8])
def test_initializers_and_costs_match_scalar(portfolio,seed):
    p=instance(seed,jobs=3,ops=4,machines=4);plans,costs=portfolio.starts(p)
    assert plans.shape==(12,3,4)
    for a,c in zip(plans,costs):assert c==pytest.approx(evaluate(p,a),abs=1e-8)
    for i,w in enumerate((0,.25,.5,1.)):
        state=route_greedy(p,w);a=np.zeros(p.shape[:2],dtype=int)
        for j,k,h in state[3]:a[j,k]=h
        assert np.array_equal(a,plans[i])


def test_refinement_is_deterministic_and_locally_optimal(portfolio):
    p=instance(10,jobs=3,ops=4,machines=4);plans,_=portfolio.starts(p)
    for a in plans:
        cost,b,_=portfolio.refine(p,a);cost2,c,_=portfolio.refine(p,a)
        assert cost==cost2 and np.array_equal(b,c)
        assert cost<=evaluate(p,a)
        assert portfolio.exact.rank(p,b)[0].min()>=cost


def test_top2_selector_scores_only_declared_starting_plans(portfolio):
    p=instance(11,jobs=3,ops=4,machines=4);plans,costs=portfolio.starts(p)
    expected=min(portfolio.refine(p,plans[i])[0] for i in np.argsort(costs,kind='stable')[:2])
    assert portfolio.select(p,'top2')[0]==expected


def test_native_refinement_matches_traced_python_loop(portfolio):
    p=instance(63020);plans,costs=portfolio.starts(p)
    for i,a in enumerate(plans):
        c,b,n=portfolio.refine(p,a)
        d,e,m=portfolio.refine(p,a,observer=lambda *_:None)
        assert c==d and n==m and np.array_equal(b,e)
        start_cost,start_plan=portfolio.start(p,i)
        assert start_cost==costs[i] and np.array_equal(start_plan,a)

import numpy as np
import pytest
from src.placement.radical.environment import problem,initial
from src.placement.radical.pairs import PairNative
from scripts_cosim.mixed_pair_budget import BudgetPairs

@pytest.fixture(scope='module')
def engine(tmp_path_factory):return BudgetPairs(tmp_path_factory.mktemp('budget'))

@pytest.mark.parametrize('method',['graph1','graph4','random2','random8'])
def test_vectorized_portfolio_preserves_original_order_and_result(engine,method):
    b=problem(37,jobs=3,ops=3)
    expected,a=PairNative.plan(engine,b,method);got,z=engine.plan(b,method)
    assert got==expected and np.array_equal(a,z)

@pytest.mark.parametrize('method',['load:anneal512','fastest:descent8','affinity:anneal1280'])
def test_control_does_declared_search_and_keeps_problem(engine,method):
    b=problem(37,jobs=3,ops=3);original=b['p'].copy();start,search=method.split(':');anneal=search.startswith('anneal')
    steps=int(search[6:] if anneal else search[7:])
    c,a=engine.search(b,initial(b,start),steps,anneal);d,z=engine.plan(b,method)
    assert c==d and np.array_equal(a,z) and np.array_equal(b['p'],original)

def test_budget_gate_uses_requested_budget_and_rejects_overrun():
    from scripts_cosim.mixed_pair_budget import summary
    rows=[{'methods':{'oracle':{'cost':90.},'control5':{'cost':100.,'time_ms':4.9},
                       'control10':{'cost':100.,'time_ms':7.}},'free_selector':{'time_ms':4.}} for _ in range(8)]
    report=summary(rows,{'5':'rule5','10':'rule10'},[5,10])
    assert report['5']['passes'] and report['10']['passes']
    for row in rows:row['methods']['control5']['time_ms']=5.1
    report=summary(rows,{'5':'rule5','10':'rule10'},[5,10])
    assert not report['5']['passes'] and report['10']['passes']

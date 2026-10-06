import numpy as np
import pytest
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag,chronological,baseline
from src.placement.radical.dag_reservation_live import replay


@pytest.fixture(scope='module')
def engine(tmp_path_factory):return BridgedDag(tmp_path_factory.mktemp('bridge_build'))


@pytest.mark.parametrize('seed',[149901,149902,149903])
def test_mapping_preserves_or_improves_schedule(engine,seed):
    b=problem(seed,4,8,4);a=b['p'].argmin(-1)
    rank=np.random.default_rng(seed).permutation(a.size).reshape(a.shape)
    old,starts,_=engine.plan(b,a,rank,0);mapped=chronological(rank,starts)
    cost,release,ends=engine.plan(b,a,mapped,1)
    assert cost<=old
    ref=replay(b,a,mapped,release)
    assert cost==ref['objective'] and np.array_equal(ends,ref['ends'])
    result=engine.search(b,a,rank,0,mode=1)
    assert result[0]<=old


@pytest.mark.parametrize('mode',[0,1,2])
def test_bridge_search_determinism_and_feasibility(engine,mode):
    b=problem(149904,4,8,4);cost,a,rank,_,_=baseline(engine,b,mode)
    result=engine.search(b,a,rank,128,mode=mode)
    other=engine.search(b,a,rank,128,mode=mode)
    assert result[0]==other[0]<=cost and result[3:]==other[3:]
    assert np.array_equal(result[1],other[1]) and np.array_equal(result[2],other[2])
    cost,starts,ends=engine.plan(b,result[1],result[2],result[3])
    ref=replay(b,result[1],result[2],starts if result[3] else np.zeros(a.shape))
    assert cost==result[0]==ref['objective'] and np.array_equal(ends,ref['ends'])

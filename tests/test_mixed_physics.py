import numpy as np
import pytest
from src.placement.radical.environment import Native,problem,initial
from src.placement.radical.mixed import MixedNative
from src.placement.radical.live import run

@pytest.fixture(scope='module')
def engines(tmp_path_factory):
    d=tmp_path_factory.mktemp('mixed');return Native(d),MixedNative(d)

@pytest.mark.parametrize('seed',[1,2,100000,102015])
def test_specialized_score_and_search_are_identical(engines,seed):
    old,new=engines;b=problem(seed);a=initial(b,'load')
    c,e=old.score(b,a,'mixed');d,f=new.score(b,a)
    assert c==d and np.array_equal(e,f)
    for method,steps,anneal in [('descent',6,False),('anneal128',128,True)]:
        c,a=old.plan(b,'mixed',method);d,z=new.search(b,initial(b,'load'),steps,anneal)
        assert c==d and np.array_equal(a,z)
        assert d==pytest.approx(run(b,z,'mixed')['objective'])

@pytest.mark.parametrize('key,value',[('types',99),('locks',-.5),('domains',2),('setup',-1)])
def test_bad_inputs_rejected_before_native_call(engines,key,value):
    _,new=engines;b=problem(3);b[key]=b[key].astype(float);b[key].flat[0]=value
    with pytest.raises(ValueError):new.score(b,initial(b,'fastest'))

@pytest.mark.parametrize('seed',[3,106006])
def test_actual_herosim_completion_parity(engines,seed):
    import io
    from scripts_cosim.workflow_proposal_gate import configure_environment
    from scripts_cosim.mixed_physics_live import run as herosim
    configure_environment();_,engine=engines;b=problem(seed);cost,a=engine.search(b,initial(b,'affinity'),256,True);expected=run(b,a,'mixed');actual=herosim(b,a,io.StringIO(),seed)
    assert actual['total_rtt']*1000==pytest.approx(cost,abs=1e-7)
    assert actual['mixed_execution']['contract']=='mixed_execution_v1'
    for task in actual['tasks']:
        j,k=map(int,task['taskType']['name'][2:].split('_'))
        assert task['doneTime']*1000==pytest.approx(expected['ends'][j][k],abs=1e-7)

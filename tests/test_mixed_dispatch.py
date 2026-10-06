import io
import numpy as np
import pytest
from src.placement.radical.environment import problem,initial
from src.placement.radical.dispatch import DispatchNative,priorities
from src.placement.radical.dispatch_live import replay,herosim
from scripts_cosim.workflow_proposal_gate import configure_environment

@pytest.fixture(scope='module')
def engine(tmp_path_factory):return DispatchNative(tmp_path_factory.mktemp('dispatch'))

@pytest.mark.parametrize('seed',[3,19,106006])
@pytest.mark.parametrize('rule',['spt','srpt','longest','type','lock_low','lock_high','graph_tail'])
def test_native_matches_independent_events(engine,seed,rule):
    b=problem(seed);a=initial(b,'affinity');r=priorities(b,a,rule);c,e=engine.score_priority(b,a,r);ref=replay(b,a,r)
    assert c==ref['objective'] and np.array_equal(e,ref['ends'])
    c,z=engine.search_priority(b,a,r,64);ref=replay(b,a,z)
    assert c==ref['objective'] and sorted(z.ravel())==list(range(a.size))
    c2,z2=engine.search_priority(b,a,r,64)
    assert c==c2 and np.array_equal(z,z2)

@pytest.mark.parametrize('seed',[3,106006])
def test_actual_priority_dispatch_and_legacy_restoration(engine,seed):
    from src.placement.radical import coordinator
    from scripts_cosim.mixed_physics_live import run
    configure_environment();b=problem(seed);c,plan=engine.plan(b,'srpt:64');a,r=plan;original=coordinator.MixedExecution
    actual=herosim(b,a,r,io.StringIO(),seed);ref=replay(b,a,r)
    assert coordinator.MixedExecution is original
    assert actual['mixed_execution']['dispatch_contract']=='mixed_ready_priority_v1'
    assert actual['total_rtt']*1000==pytest.approx(c)
    for task in actual['tasks']:
        j,k=map(int,task['taskType']['name'][2:].split('_'))
        assert task['doneTime']*1000==pytest.approx(ref['ends'][j,k])
    old=run(b,a,io.StringIO(),seed)
    assert 'dispatch_contract' not in old['mixed_execution']
    assert old['total_rtt']*1000==pytest.approx(engine.score(b,a)[0])

@pytest.mark.parametrize('bad',[np.zeros((3,3),int),np.arange(9).reshape(3,3).astype(float)])
def test_bad_ranks_reject_before_native(engine,bad):
    b=problem(1,jobs=3,ops=3)
    with pytest.raises(ValueError):engine.score_priority(b,initial(b,'affinity'),bad)

@pytest.fixture(scope='module')
def adaptive(tmp_path_factory):
    from src.placement.radical.dispatch_adaptive import AdaptiveDispatch
    return AdaptiveDispatch(tmp_path_factory.mktemp('adaptive'))

@pytest.mark.parametrize('mode',range(9))
@pytest.mark.parametrize('seed',[19,106006])
def test_adaptive_materialization_preserves_exact_schedule(adaptive,mode,seed):
    b=problem(seed);a=initial(b,'affinity');cost,r=adaptive.adaptive_plan(b,a,mode)
    c,ends=adaptive.score_priority(b,a,r);ref=replay(b,a,r)
    assert cost==c==ref['objective'] and np.array_equal(ends,ref['ends'])
    assert sorted(r.ravel())==list(range(a.size))
    from src.placement.radical.dispatch_adaptive_reference import adaptive_replay
    expected,order,completed=adaptive_replay(b,a,mode)
    assert cost==expected and np.array_equal(r,order) and np.array_equal(ends,completed)

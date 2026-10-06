import numpy as np
import pytest
from src.placement.radical.environment import Native,problem,initial,FLAGS
from src.placement.radical.live import run
from src.placement.workflow_planning import evaluate

@pytest.fixture(scope='module')
def native(tmp_path_factory):return Native(tmp_path_factory.mktemp('radical'))

@pytest.mark.parametrize('mechanism',list(FLAGS))
@pytest.mark.parametrize('seed',[1,7,99])
def test_native_matches_independent_simpy(native,mechanism,seed):
    b=problem(seed,jobs=3,ops=4);a=initial(b,'load');score,ends=native.score(b,a,mechanism);live=run(b,a,mechanism)
    assert score==pytest.approx(live['objective'],abs=1e-7)
    np.testing.assert_allclose(ends,live['ends'],atol=1e-7,rtol=0)
    assert len(live['events'])==24
    for j in range(3):
        starts={e['operation']:e['start'] for e in live['events'] if e['event']=='start' and e['job']==j}
        assert all(starts[k]>=ends[j,k-1] for k in range(1,4))

@pytest.mark.parametrize('seed',[4,18,21])
def test_physics_off_matches_existing_workflow_evaluator(native,seed):
    b=problem(seed);a=initial(b,'load');assert native.score(b,a,'none')[0]==pytest.approx(evaluate(b['p'],a))

@pytest.mark.parametrize('mechanism',list(FLAGS))
def test_search_deterministic_and_not_worse_than_start(native,mechanism):
    b=problem(51,jobs=3,ops=4);start=native.plan(b,mechanism,'load')[0]
    for method in ('descent','anneal128'):
        c,a=native.plan(b,mechanism,method);d,z=native.plan(b,mechanism,method)
        assert c==d and np.array_equal(a,z) and c<=start
        assert c==pytest.approx(run(b,a,mechanism)['objective'])

def test_atomic_locks_and_domain_caps_are_respected(native):
    b=problem(8);_,a=native.plan(b,'mixed','descent');live=run(b,a,'mixed');events=live['events'];active={}
    for e in events:
        if e['event']=='done':active.pop(e['host'])
        else:
            j,k=e['job'],e['operation']
            assert not any(int(b['locks'][j,k])&int(b['locks'][r['job'],r['operation']]) for r in active.values())
            active[e['host']]=e
            assert all(sum(bool(b['domains'][h,g]) for h in active)<=2 for g in range(3))
    assert not active

def test_invalid_plan_rejected_before_native(native):
    b=problem(1);a=initial(b,'load');a[0,0]=-1
    with pytest.raises(ValueError):native.score(b,a,'setup')

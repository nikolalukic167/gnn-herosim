import numpy as np
import pytest
from src.placement.workflow_planning import instance,initial,rollout,evaluate
from src.placement.workflow_exact import ExactWorkflow

@pytest.fixture(scope='module')
def exact(tmp_path_factory):return ExactWorkflow(tmp_path_factory.mktemp('native_workflow'))

@pytest.mark.parametrize('seed',[0,1,2,3])
def test_native_all_moves_match_independent_scalar_replay(exact,seed):
    p=instance(seed,jobs=3,ops=4,machines=4);cost,a=exact.greedy(p);state=rollout(initial(p),p)
    expected=np.zeros((3,4),dtype=int)
    for j,k,h in state[3]:expected[j,k]=h
    assert np.array_equal(a,expected);assert cost==pytest.approx(state[2],abs=1e-10)
    for base in [a,np.argsort(p,axis=-1)[...,1]]:
        scores,alternatives=exact.rank(p,base)
        for i,score in enumerate(scores):
            changed=base.copy();changed.ravel()[i]=alternatives.ravel()[i]
            assert score==pytest.approx(evaluate(p,changed),abs=1e-10)


def test_equal_release_and_duration_ties(exact):
    p=np.ones((3,3,3));p[:,:,2]=np.inf
    cost,a=exact.greedy(p)
    assert cost==evaluate(p,a)
    scores,alt=exact.rank(p,a)
    for i,score in enumerate(scores):
        b=a.copy();b.ravel()[i]=alt.ravel()[i];assert score==evaluate(p,b)


def test_descent_trace_preserves_output_and_is_locally_optimal(exact):
    p=instance(5);records=[];first=exact.descent(p);second=exact.descent(p,observer=lambda a,c:records.append((a,c)))
    assert first[0]==second[0] and np.array_equal(first[1],second[1])
    assert first[0]<=exact.greedy(p)[0]
    assert exact.rank(p,first[1])[0].min()>=first[0]
    assert all(abs(evaluate(p,a)-c)<1e-8 for a,c in records)


def test_native_wrapper_rejects_unsafe_inputs(exact):
    p=instance(5,jobs=2,ops=3,machines=3)
    with pytest.raises(ValueError):exact.rank(p,np.full((2,3),99))
    with pytest.raises(ValueError):exact.greedy(np.ones((2,3,3)))

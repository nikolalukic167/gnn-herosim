import itertools
import numpy as np
import pytest
from src.placement.radical.dag_reservation import DagReservation,problem,baseline,checked
from src.placement.radical.dag_reservation_live import replay
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.block_moves import scaled_problem
from scripts_cosim.dispatch_reservation_witness import fixture,certificate


@pytest.fixture(scope='module')
def engine(tmp_path_factory):return DagReservation(tmp_path_factory.mktemp('dag_build'))


@pytest.mark.parametrize('seed',[148801,148802,148803])
@pytest.mark.parametrize('mode',[0,1])
def test_native_replay_parity(engine,seed,mode):
    b=problem(seed,3,8,4);a=b['p'].argmin(-1);r=np.random.default_rng(seed).permutation(a.size).reshape(a.shape)
    cost,starts,ends=engine.plan(b,a,r,mode)
    release=starts if mode else np.zeros(a.shape)
    independent=replay(b,a,r,release)
    actual=engine.replay(b,a,r,release)
    assert cost==independent['objective']==actual[0]
    assert np.array_equal(starts,independent['starts']) and np.array_equal(ends,independent['ends'])
    assert np.array_equal(starts,actual[1]) and np.array_equal(ends,actual[2])


def test_chain_reduction_and_reservation_witness(engine,tmp_path):
    b=scaled_problem(148804,3,5,4)
    b['predecessors']=np.tile(np.array([0,1,2,4,8],dtype=np.uint64),(3,1))
    a=b['p'].argmin(-1);r=np.arange(a.size).reshape(a.shape)
    cost,_,ends=engine.plan(b,a,r,0)
    old=DispatchNative(tmp_path).score_priority(b,a,r)
    assert cost==old[0] and np.array_equal(ends,old[1])
    b=fixture();b['predecessors']=np.array([[0,1],[0,1]],dtype=np.uint64)
    best=np.inf
    for bits in itertools.product(range(2),repeat=4):
        for permutation in itertools.permutations(range(4)):
            cost,starts,ends=engine.plan(b,np.array(bits).reshape(2,2),np.array(permutation).reshape(2,2),1)
            best=min(best,cost)
    assert best==certificate(b)['best']['cost']==27


@pytest.mark.parametrize('mode',[0,1,2])
def test_search_reproducible_and_feasible(engine,mode):
    b=problem(148805,3,8,4)
    _,a,r,_,_=baseline(engine,b,mode)
    before_a,before_r=a.copy(),r.copy()
    first=engine.search(b,a,r,128,mode=mode)
    second=engine.search(b,a,r,128,mode=mode)
    assert first[0]==second[0] and first[3:]==second[3:]
    assert np.array_equal(first[1],second[1]) and np.array_equal(first[2],second[2])
    cost,starts,ends=engine.plan(b,first[1],first[2],first[3])
    ref=replay(b,first[1],first[2],starts if first[3] else np.zeros(a.shape))
    assert cost==first[0]==ref['objective']
    assert np.array_equal(ends,ref['ends'])
    assert np.array_equal(a,before_a) and np.array_equal(r,before_r)


def test_generator_and_validation(engine):
    b=problem(148806,4,8,4);a=b['p'].argmin(-1);r=np.arange(a.size).reshape(a.shape)
    assert np.array_equal(b['predecessors'],problem(148806,4,8,4)['predecessors'])
    assert any(int(p).bit_count()>1 for p in b['predecessors'].ravel())
    broken={**b,'predecessors':b['predecessors'].copy()};broken['predecessors'][0,2]=4
    with pytest.raises(ValueError):checked(broken,a,r)
    with pytest.raises(ValueError):engine.replay(b,a,r,np.full(a.shape,-1.))


def test_branches_execute_concurrently(engine):
    b=scaled_problem(148807,1,4,2)
    b['predecessors']=np.array([[0,1,1,6]],dtype=np.uint64)
    b['locks'][:]=0;b['types'][:]=0;b['p'][:]=3
    a=np.array([[0,0,1,0]]);r=np.arange(4).reshape(1,4)
    cost,starts,ends=engine.plan(b,a,r,0)
    assert starts.tolist()==[[0,3,3,6]] and cost==9
    assert replay(b,a,r,np.zeros(a.shape))['objective']==9


def test_live_diamond_uses_job_completion_not_task_rtt(engine,tmp_path):
    from src.placement.radical.dag_reservation_live import herosim
    from scripts_cosim.workflow_proposal_gate import configure_environment
    configure_environment()
    b=scaled_problem(148809,1,4,2)
    b['predecessors']=np.array([[0,1,1,6]],dtype=np.uint64)
    b['locks'][:]=0;b['types'][:]=0;b['p'][:]=3
    a=np.array([[0,0,1,0]]);rank=np.arange(4).reshape(1,4)
    cost,starts,ends=engine.plan(b,a,rank,1)
    with (tmp_path/'live.log').open('w') as log:actual=herosim(b,a,rank,starts,log)
    assert abs(actual['job_completion_sum']*1000-cost)<1e-9
    assert abs(actual['total_task_rtt']*1000-12)<1e-9
    assert cost==9
    for task in actual['tasks']:
        j,k=map(int,task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime']*1000-ends[j,k])<1e-9

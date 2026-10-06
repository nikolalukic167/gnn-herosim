import io
import numpy as np
import pytest
import torch
from src.placement.workflow_planning import instance,initial,rollout,beam,evaluate,route_greedy
from src.policy.workflow.model import features,WorkflowAssignmentNet
from scripts_cosim.workflow_qualification import assignment
from scripts_cosim.workflow_live import run

@pytest.mark.parametrize('seed',[60501,60502])
def test_complete_plan_score_and_eligibility(seed):
    p=instance(seed,jobs=3,ops=4,machines=3)
    for s in [rollout(initial(p),p),beam(p,4),route_greedy(p,.25)]:
        a=assignment(p,s)
        assert all(np.isfinite(p[j,k,h]) for j,k,h in s[3])
        assert len(s[3])==12
        assert s[2]==pytest.approx(evaluate(p,a))

def test_live_fifo_workflow_matches_full_prediction(monkeypatch):
    monkeypatch.setenv('HEROSIM_PEER_EXCHANGE','1');monkeypatch.setenv('HEROSIM_PG_DRAIN_CONTRACT','availability_v2');monkeypatch.setenv('SIM_FORCE_FULL_STATS','1')
    p=instance(60503,jobs=3,ops=4,machines=3)
    for s in [rollout(initial(p),p),beam(p,4)]:
        result=run(p/1000,assignment(p,s),io.StringIO())
        assert result['job_completion_sum']*1000==pytest.approx(s[2])
        assert result['total_rtt']==pytest.approx(result['job_completion_sum'])
        assert len(result['tasks'])==12

def test_mp_sees_other_operation_while_twin_is_pointwise():
    torch.manual_seed(0);p=instance(1,jobs=3,ops=4,machines=3);x,e=features(p)
    x=torch.from_numpy(x)[None];e=torch.from_numpy(e)[None]
    changed=x.clone();changed[0,0,1,0]+=.5
    off=WorkflowAssignmentNet(x.shape[-1],machines=3,mp=False).eval();on=WorkflowAssignmentNet(x.shape[-1],machines=3,mp=True).eval();on.load_state_dict(off.state_dict())
    with torch.no_grad():
        assert torch.equal(off(x,e)[0,0,0],off(changed,e)[0,0,0])
        assert not torch.allclose(on(x,e)[0,0,0],on(changed,e)[0,0,0])
        assert torch.isfinite(on(x,e)).all()


def test_exact_witness_requires_remote_structure():
    from itertools import product
    p=instance(1,jobs=3,ops=3,machines=4)
    q=p.copy();q[1,1],q[2,2]=p[2,2].copy(),p[1,1].copy()
    x,e=features(p);xx,ee=features(q)
    assert np.allclose(x[0,0],xx[0,0])
    roots=[]
    for problem in (p,q):
        best={}
        for labels in product(*[np.flatnonzero(np.isfinite(op)) for op in problem.reshape(-1,4)]):
            cost=evaluate(problem,np.array(labels).reshape(3,3))
            best[labels[0]]=min(cost,best.get(labels[0],float('inf')))
        roots.append(min(best,key=best.get))
    assert roots==[3,2]
    torch.manual_seed(0)
    model=WorkflowAssignmentNet(x.shape[-1],machines=4,mp=True).eval()
    with torch.no_grad():
        first=model(torch.tensor(x)[None],torch.tensor(e)[None])[0,0,0]
        second=model(torch.tensor(xx)[None],torch.tensor(ee)[None])[0,0,0]
    assert not torch.allclose(first,second)

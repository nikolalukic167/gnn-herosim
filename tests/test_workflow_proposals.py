import io
import numpy as np
import pytest
from src.placement.workflow_planning import instance,initial,rollout,evaluate,plan_observer
from src.placement.workflow_proposals import improve,hand_probabilities
from scripts_cosim.workflow_knative import run


def test_search_is_deterministic_valid_and_never_worsens_incumbent():
    p=instance(9,jobs=3,ops=4,machines=4);prob=hand_probabilities(p,4)
    first=improve(p,prob,32,111);second=improve(p,prob,32,111)
    assert first[0]==second[0] and np.array_equal(first[1],second[1])
    assert first[0]<=rollout(initial(p),p)[2]
    assert first[0]==pytest.approx(evaluate(p,first[1]))
    assert np.isfinite(np.take_along_axis(p,first[1][...,None],-1)).all()


def test_probability_contract_rejects_invalid_hosts():
    p=instance(3,jobs=2,ops=3,machines=4)
    with pytest.raises(ValueError,match='eligible hosts'):improve(p,np.ones_like(p)/4,2,1)


def test_observational_trace_preserves_search():
    p=instance(7,jobs=2,ops=3,machines=4);prob=hand_probabilities(p);records=[]
    before=improve(p,prob,8,12);token=plan_observer.set(lambda a,c:records.append((a.copy(),c)))
    try:after=improve(p,prob,8,12)
    finally:plan_observer.reset(token)
    assert len(records)==10
    assert before[0]==after[0] and np.array_equal(before[1],after[1])


def test_actual_knative_matches_replay_and_is_repeatable(monkeypatch):
    monkeypatch.setenv('SIM_FORCE_FULL_STATS','1');monkeypatch.setenv('HEROSIM_PG_DRAIN_CONTRACT','legacy_v0')
    p=instance(13,jobs=3,ops=4,machines=4)
    a=run(p/1000,io.StringIO(),13);b=run(p/1000,io.StringIO(),13)
    assert a['assignment']==b['assignment']
    assert a['total_rtt']==pytest.approx(b['total_rtt'])
    assert a['total_rtt']*1000==pytest.approx(evaluate(p,np.array(a['assignment'])))


def test_registered_gate_environment_supports_both_live_paths(monkeypatch):
    from scripts_cosim.workflow_proposal_gate import configure_environment
    from scripts_cosim.workflow_live import run as replay
    for key in ('HEROSIM_PEER_EXCHANGE','HEROSIM_PG_DRAIN_CONTRACT','SIM_FORCE_FULL_STATS'):
        monkeypatch.setenv(key,'invalid')
    configure_environment()
    p=instance(14,jobs=2,ops=3,machines=4)
    score,a=improve(p,hand_probabilities(p),4,14)
    assert replay(p/1000,a,io.StringIO(),14)['total_rtt']*1000==pytest.approx(score)
    assert len(run(p/1000,io.StringIO(),14)['tasks'])==6

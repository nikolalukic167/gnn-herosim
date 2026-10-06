"""Whole-trace control keeps declared groups local and preserves the earlier intervention."""
import json

import pytest

from scripts_cosim.peer_lookahead_live_probe import hand_controls, make_workload, simulate
from scripts_cosim.peer_lookahead_trace import ROOT, run, trace_workload


def test_repeated_trace_preserves_group_boundaries():
    cfg=json.loads((ROOT/'experiments/peer_lookahead_trace_v1.json').read_text())
    source={'config':{'workload':{'events':[{'node_name':'client0'}]}}}
    w=trace_workload(source,cfg,.1,1)
    assert len(w['events'])==len(w['peer_exchange'])==256
    assert all(i//8==j//8 for i,j,_ in w['peer_exchange'])
    assert {i for e in w['peer_exchange'] for i in e[:2]}==set(range(256))
    cfg['trace_tasks']=9
    with pytest.raises(ValueError,match='whole'):
        trace_workload(source,cfg,.1,0)


def test_real_engine_root_only_matches_probe_and_full_control_visits_every_peer_task(monkeypatch,tmp_path):
    cfg=json.loads((ROOT/'experiments/peer_lookahead_trace_v1.json').read_text())
    path=ROOT/cfg['source_corpus']/'ds_00064'/'optimal_result.json'
    if not path.exists():
        pytest.skip('source snapshot unavailable')
    for name in ('HEROSIM_PG_CAPTURE_PATH','HEROSIM_DATA_LOCALITY','COSIM_AUTOSCALER_RECONCILE_INTERVAL'):
        monkeypatch.delenv(name,raising=False)
    for name,value in {'HEROSIM_PEER_EXCHANGE':'1','SIM_FORCE_FULL_STATS':'1',
                       'COSIM_SUPPRESS_SIM_PRINTS':'1','HEROSIM_COSIM_KEEP_ALIVE':'1000000'}.items():
        monkeypatch.setenv(name,value)
    source=json.loads(path.read_text())
    w=make_workload(source,cfg,1.,1,16)
    _,f=simulate(source,cfg,w,tmp_path/'capture.log',capture=True)
    controls,_=hand_controls(f)
    forced,_=simulate(source,cfg,w,tmp_path/'forced.log',forced=f['candidates'][0][controls['peer_min_sum_2']])
    first=run(source,cfg,w,'group_first_two_hop',tmp_path/'first.log')
    assert first['total_rtt']==pytest.approx(forced['total_rtt'],rel=1e-10)
    assert [d['task_id'] for d in first['decisions'] if d['lookahead']]==[0]
    full=run(source,cfg,w,'two_hop',tmp_path/'full.log')
    assert [d['task_id'] for d in full['decisions'] if d['lookahead']]==list(range(8))
    assert sorted(d['task_id'] for d in full['decisions'])==list(range(16))
    assert full['decision_wall_total_s']>0
    assert full['rtt_plus_decision_wall_s']>=full['total_rtt']

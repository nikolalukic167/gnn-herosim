import json
import numpy as np
import pytest
from scripts_cosim import graph_size_screen as large
from scripts_cosim import graph_three_host_screen as small
from scripts_cosim import graph_decision_witness as w

@pytest.mark.parametrize('seed',[46200,46201,46209])
def test_solver_certificate_matches_exhaustive(seed):
    cfg=json.loads((w.ROOT/'experiments/graph_three_host_screen_v1.json').read_text())
    edges=small.make_edges(cfg,seed)
    plans,values=small.exhaustive(cfg,edges)
    result=large.exact_bound(cfg,edges,2)
    assert result['certified']
    assert result['incumbent']==pytest.approx(values.min())
    assert result['lower_bound']==pytest.approx(values.min())
    assert w.energy(cfg,edges,result['plan'])==pytest.approx(values.min())

def test_no_edge_objective_has_zero_certificate():
    cfg={'tasks':4,'hosts':3,'anchors':{'1':0,'2':1,'3':2},'payload_unit_bytes':1,'bandwidth_mbps':100,'peer_latency_s':.01}
    result=large.exact_bound(cfg,[],2)
    assert result['certified'] and result['incumbent']==0
    assert result['plan'][1:]==[0,1,2]

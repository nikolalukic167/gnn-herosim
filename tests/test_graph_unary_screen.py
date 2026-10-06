import itertools
import numpy as np
import pytest
from scripts_cosim import graph_unary_screen as u,graph_decision_witness as w,graph_size_screen as big,graph_three_host_screen as small

@pytest.fixture
def cfg():
 return {'tasks':6,'hosts':3,'anchors':{},'unary_seconds':[[0,2,8],[4,0,5],[2,5,0],[6,0,2],[0,4,1],[2,7,0]],'payload_unit_bytes':104857600,'bandwidth_mbps':100.,'peer_latency_s':.01,'execution_s':.1}

def test_unary_solver_and_enumeration(cfg):
 edges=[(0,1,1),(1,2,2),(2,3,1),(3,4,2),(4,5,1),(0,5,1)]
 plans,values=small.exhaustive(cfg,edges)
 assert np.allclose(values,[w.energy(cfg,edges,p) for p in plans])
 result=big.exact_bound(cfg,edges,2)
 assert result['certified'] and result['incumbent']==pytest.approx(values.min())

def test_expansion_no_improving_binary_move(cfg):
 edges=[(0,1,1),(1,2,2),(2,3,1),(3,4,2),(4,5,1),(0,5,1)]
 initial=np.argmin(cfg['unary_seconds'],axis=1).tolist();p=u.expand(cfg,edges,initial)
 assert w.energy(cfg,edges,p)<=w.energy(cfg,edges,initial)
 for alpha in range(3):
  for bits in itertools.product([False,True],repeat=6):
   q=[alpha if bit else h for bit,h in zip(bits,p)]
   assert w.energy(cfg,edges,q)>=w.energy(cfg,edges,p)-1e-8

def test_physical_execution_table_matches_unary(cfg):
 infra,inputs=u.unary_substrate(cfg)
 for i in range(6):
  for h,node in enumerate(infra['nodes'][1:]):
   key=f'hostcpu{h}'
   assert key in node['platforms']
   assert inputs['task_types'][f'w{i}']['executionTime'][key]==pytest.approx(.1+cfg['unary_seconds'][i][h])

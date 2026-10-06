"""The live probe preserves arrival semantics and measures a single intervention."""
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from scripts_cosim.peer_lookahead_live_probe import (
    ROOT, argmin_set, make_workload, min_sum, rank_stability, static_cd, static_cost, summarize_stats,
)


def test_horizons_preserve_full_peer_group_and_identical_prefix():
    config=json.loads((ROOT/'experiments/peer_lookahead_live_probe_v1.json').read_text())
    source={'config':{'workload':{'events':[{'node_name':'client0','qos':{'name':'medium'}}]}}}
    short=make_workload(source,config,.1,0,8)
    long=make_workload(source,config,.1,0,16)
    assert short['events']==long['events'][:8]
    assert short['peer_exchange']==long['peer_exchange']
    assert max(max(i,j) for i,j,_ in long['peer_exchange'])<8
    assert short['events'][7]['timestamp']>short['events'][0]['timestamp']
    with pytest.raises(ValueError,match='truncate'):
        make_workload(source,config,.1,0,7)
    assert make_workload(source,config,.1,0,16,exchange=False)==make_workload(source,config,.1,1,16,exchange=False)


def test_ragged_tree_messages_match_exhaustive_conditional_costs():
    costs=[np.array([0.,2.]),np.array([1.,0.,4.]),np.array([0.,3.])]
    pairs={(0,1):np.array([[0.,5.,2.],[3.,0.,1.]]),
           (1,2):np.array([[0.,4.],[5.,0.],[1.,3.]])}
    plans=list(product(range(2),range(3),range(2)))
    exact=np.array([min(static_cost(costs,pairs,p) for p in plans if p[0]==c) for c in range(2)])
    got=min_sum(costs,pairs,2)[0]
    assert np.allclose(got-got.min(),exact-exact.min())
    for p in plans:
        assert static_cost(costs,pairs,static_cd(costs,pairs,p))<=static_cost(costs,pairs,p)+1e-10


def test_rank_stability_accounts_for_ties_and_constant_costs():
    assert argmin_set([1.,1.,2.])==[0,1]
    assert rank_stability([1.,1.,2.],[2.,1.,3.])['common_best_candidate']
    assert not rank_stability([1.,2.],[2.,1.])['common_best_candidate']
    assert rank_stability([1.,1.],[1.,2.])['spearman'] is None


def test_stats_refuses_incomplete_runs_and_double_counted_rtt():
    row={'taskId':0,'elapsedTime':2.,'executionNode':'node0','executionPlatform':'1',
         'peerExchangeTime':.3,'peerRendezvousWait':.2}
    stats={'taskResults':[row],'total_rtt':2.,'totalPeerExchangeTime':.3,
           'totalPeerRendezvousWait':.2,'averageQueueTime':.5,'averageWaitTime':0.}
    assert summarize_stats(stats,1)['group_rtt']==2.
    with pytest.raises(RuntimeError,match='incomplete'):
        summarize_stats(stats,2)
    stats['total_rtt']=4.
    with pytest.raises(RuntimeError,match='sum of task RTTs'):
        summarize_stats(stats,1)

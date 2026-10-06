"""Serve fixed plans or actual Knative through HeROsim's opt-in mixed service."""
import copy,random
from contextlib import redirect_stdout,redirect_stderr
import numpy as np
from scripts_cosim.workflow_live import substrate
from scripts_cosim.peer_lookahead_live_probe import deadline
from src.placement.radical.coordinator import config_for

def run(b,assignment,log,seed=0):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    from src.policy.knative_network.scheduler import KnativeScheduler
    config,inputs=substrate(b['p']/1000);config['infrastructure']['mixed_execution']=config_for(b);decisions=[]
    cls=PeerGreedyNetworkScheduler if assignment is not None else KnativeScheduler;original=cls.placement
    def place(self,state,task):
        j,k=map(int,task.type['name'][2:].split('_'))
        if assignment is None:
            selected=yield from original(self,state,task);host=int(selected[0].node_name[4:])
        else:
            host=int(assignment[j,k]);matches=[(n,p) for n,p in self._get_valid_replicas(state.replicas[task.type['name']],task) if n.node_name==f'node{host}']
            if len(matches)!=1:raise ValueError('no unique eligible replica')
            selected=matches[0]
        decisions.append({'job':j,'operation':k,'host':host,'time':float(self.env.now)});return selected
    cls.placement=place;random.seed(seed);np.random.seed(seed)
    try:
        with redirect_stdout(log),redirect_stderr(log),deadline(20):result=execute_simulation(config,copy.deepcopy(inputs),'peer_greedy_network_peer_greedy_network' if assignment is not None else 'kn_network_kn_network',keep_alive=1000000,queue_length=100)
    finally:cls.placement=original
    stats=result['stats'];tasks=stats['taskResults'];j,o,_=b['p'].shape
    if len(tasks)!=j*o or len(decisions)!=j*o:raise RuntimeError('incomplete mixed live workload')
    plan=np.full((j,o),-1,dtype=int)
    for d in decisions:plan[d['job'],d['operation']]=d['host']
    return {'total_rtt':stats['total_rtt'],'tasks':tasks,'decisions':decisions,'assignment':plan.tolist(),'mixed_execution':stats['mixedExecution'],'config':config,'inputs':inputs}

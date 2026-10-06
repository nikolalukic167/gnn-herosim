"""Observe the actual KnativeNetwork policy on a declared workflow fixture."""
import copy,random
from contextlib import redirect_stdout,redirect_stderr
import numpy as np
from scripts_cosim.workflow_live import substrate
from scripts_cosim.peer_lookahead_live_probe import deadline


def run(p,log,seed):
    from src.executecosimulation import execute_simulation
    from src.policy.knative_network.scheduler import KnativeScheduler
    config,inputs=substrate(p);original=KnativeScheduler.placement;decisions=[]
    def observed(self,state,task):
        selected=yield from original(self,state,task)
        j,k=map(int,task.type['name'][2:].split('_'))
        decisions.append({'job':j,'operation':k,'host':int(selected[0].node_name[4:]),'time':float(self.env.now)})
        return selected
    random.seed(seed);np.random.seed(seed);KnativeScheduler.placement=observed
    try:
        with redirect_stdout(log),redirect_stderr(log),deadline(20):
            result=execute_simulation(config,copy.deepcopy(inputs),'kn_network_kn_network',keep_alive=1000000,queue_length=100)
    finally:KnativeScheduler.placement=original
    tasks=result['stats']['taskResults']
    if len(tasks)!=p.shape[0]*p.shape[1] or len(decisions)!=len(tasks):raise RuntimeError('incomplete actual Knative workflow')
    assignment=np.full(p.shape[:2],-1,dtype=int)
    for d in decisions:assignment[d['job'],d['operation']]=d['host']
    if np.any(assignment<0):raise RuntimeError('missing Knative operation placement')
    return {'assignment':assignment.tolist(),'total_rtt':result['stats']['total_rtt'],'tasks':tasks,'decisions':decisions}

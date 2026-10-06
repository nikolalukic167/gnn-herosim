"""Independent event replay and actual HeROsim adapter for irregular DAG plans."""
import copy
import random
from contextlib import redirect_stdout,redirect_stderr
import numpy as np
import simpy
from . import coordinator
from .dag_reservation import checked
from .reservation_dispatch import ReservationExecution
from scripts_cosim.workflow_live import substrate
from scripts_cosim.peer_lookahead_live_probe import deadline


def replay(b,assignment,rank,release):
    _,pred,a,r,_=checked(b,assignment,rank)
    release=np.asarray(release,dtype=float)
    if release.shape!=a.shape or not np.isfinite(release).all() or np.any(release<0):raise ValueError('invalid DAG reservations')
    jobs,ops=a.shape
    parents={(j,k):{(j,z) for z in range(k) if int(pred[j,k])&(1<<z)} for j in range(jobs) for k in range(ops)}
    todo=set(parents);done={};active={};last={};events=[];starts=np.empty(a.shape)
    env=simpy.Environment()
    def process():
        while len(done)<a.size:
            now=float(env.now)
            for host,rec in list(active.items()):
                if rec['end']<=now+1e-9:
                    node=(rec['job'],rec['operation']);done[node]=now;last[host]=int(b['types'][node]);del active[host]
                    events.append({'event':'done',**rec})
            if len(done)==a.size:break
            ready=sorted((node for node in todo if parents[node].issubset(done)),key=lambda node:int(r[node]))
            for node in ready:
                host=int(a[node]);lock=int(b['locks'][node])
                if release[node]>now+1e-9 or host in active:continue
                if any(lock&int(b['locks'][v['job'],v['operation']]) for v in active.values()):continue
                if any(b['domains'][host,g] and sum(b['domains'][h,g] for h in active)>=2 for g in range(b['domains'].shape[1])):continue
                setup=float(b['setup'][last[host],b['types'][node]]) if host in last else 0.
                rec={'job':node[0],'operation':node[1],'host':host,'start':now,'end':now+float(b['p'][node][host])+setup,'setup':setup}
                starts[node]=now;active[host]=rec;todo.remove(node);events.append({'event':'start',**rec})
            next_times=[v['end'] for v in active.values()]+[float(release[node]) for node in ready if node in todo and release[node]>now+1e-9]
            if not next_times:raise RuntimeError('DAG replay deadlock')
            yield env.timeout(min(next_times)-now)
    env.process(process());env.run()
    ends=np.array([[done[j,k] for k in range(ops)] for j in range(jobs)])
    return {'objective':float(ends[:,-1].sum()),'starts':starts,'ends':ends,'events':events}


def herosim(b,assignment,rank,release,log,arrivals=None):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    _,pred,a,r,_=checked(b,assignment,rank)
    release=np.asarray(release,dtype=float)
    if release.shape!=a.shape or not np.isfinite(release).all() or np.any(release<0):raise ValueError('invalid DAG reservations')
    config,inputs=substrate(b['p']/1000)
    jobs,ops=a.shape
    if arrivals is not None:
        arrivals=np.asarray(arrivals,dtype=float)
        if arrivals.shape != (jobs,) or not np.isfinite(arrivals).all() or np.any(arrivals < 0):
            raise ValueError('invalid job arrival times')
        for j,event in enumerate(config['workload']['events']):
            event['timestamp']=float(arrivals[j])/1000
    for j in range(jobs):
        inputs['application_types'][f'nofs-job{j}']['dag']={f'op{j}_{k}':[f'op{j}_{z}' for z in range(k) if int(pred[j,k])&(1<<z)] for k in range(ops)}
    config['infrastructure']['mixed_execution']=coordinator.config_for(b)
    old_execution,old_placement=coordinator.MixedExecution,PeerGreedyNetworkScheduler.placement
    decisions=[]
    class BoundDag(ReservationExecution):
        def __init__(self,env,settings):
            super().__init__(env,settings);self.priority=r.copy();self.not_before=release/1000;self.reservation_timers=set()
        def stats(self):
            return {**super().stats(),'dispatch_contract':'mixed_irregular_dag_reservation_v1','predecessors':pred.tolist()}
    def place(self,state,task):
        if False:yield
        j,k=map(int,task.type['name'][2:].split('_'));host=int(a[j,k])
        matches=[(n,p) for n,p in self._get_valid_replicas(state.replicas[task.type['name']],task) if n.node_name==f'node{host}']
        if len(matches)!=1:raise RuntimeError('DAG assignment has no unique eligible replica')
        decisions.append({'job':j,'operation':k,'host':host,'time':float(self.env.now)})
        return matches[0]
    coordinator.MixedExecution=BoundDag;PeerGreedyNetworkScheduler.placement=place
    random.seed(int(b['seed']));np.random.seed(int(b['seed']))
    try:
        with redirect_stdout(log),redirect_stderr(log),deadline(30):
            result=execute_simulation(config,copy.deepcopy(inputs),'peer_greedy_network_peer_greedy_network',keep_alive=1000000,queue_length=100)
    finally:
        coordinator.MixedExecution=old_execution;PeerGreedyNetworkScheduler.placement=old_placement
    stats=result['stats'];rows=stats['taskResults']
    if len(rows)!=a.size or len(decisions)!=a.size:raise RuntimeError('incomplete actual DAG execution')
    completions={}
    for task in rows:
        j,k=map(int,task['taskType']['name'][2:].split('_'))
        if k==ops-1:
            if j in completions:raise RuntimeError('duplicate DAG sink')
            completions[j]=task['doneTime']
    if set(completions)!=set(range(jobs)):raise RuntimeError('missing DAG sink')
    return {'job_completion_sum':sum(completions.values()),'job_completions':completions,
            'total_task_rtt':stats['total_rtt'],'tasks':rows,'decisions':decisions,
            'mixed_execution':stats['mixedExecution'],'config':config,'inputs':inputs}

"""Replay complete-workflow assignments in HeROsim with shared FIFO platforms."""
import copy,json,os,random
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts_cosim.peer_lookahead_live_probe import deadline

def substrate(p):
    jobs,ops,machines=p.shape
    inputs={k:json.loads((ROOT/'data/nofs-ids'/f).read_text()) for k,f in [('platform_types','platform-types.json'),('storage_types','storage-types.json'),('qos_types','qos-types.json')]}
    base=inputs['platform_types']['rpiCpu'];inputs['platform_types']={f'hostcpu{h}':{**base,'name':f'hostcpu{h}','shortName':f'hostcpu{h}'} for h in range(machines)}
    for storage in inputs['storage_types'].values():storage['latency']={'read':0.,'write':0.}
    inputs['task_types']={};inputs['application_types']={};replicas={}
    for j in range(jobs):
        app=f'nofs-job{j}';dag={}
        for k in range(ops):
            name=f'op{j}_{k}';dag[name]=[] if k==0 else [f'op{j}_{k-1}'];hs=np.flatnonzero(np.isfinite(p[j,k]))
            types=[f'hostcpu{h}' for h in hs]
            inputs['task_types'][name]={'name':name,'platforms':types,'memoryRequirements':{h:.01 for h in types},'coldStartDuration':{h:0. for h in types},'executionTime':{f'hostcpu{h}':float(p[j,k,h]) for h in hs},'energy':{h:0. for h in types},'imageSize':{h:0. for h in types},'stateSize':{app:{'input':0,'output':0}}}
            replicas[name]=[{'node_name':f'node{int(h)}','platform_id':int(h),'warm':True} for h in hs]
        inputs['application_types'][app]={'name':app,'dag':dag}
    node_names=['client_node0']+[f'node{h}' for h in range(machines)]
    nodes=[{'node_name':name,'memory':128,'platforms':[] if i==0 else [f'hostcpu{i-1}'],'storage':['flashCard','someRemote'],'type':'rpi','network_map':{other:0. for other in node_names if other!=name}} for i,name in enumerate(node_names)]
    # Private replicas retain per-type eligibility; the node's single compute slot supplies FIFO service.
    replicas={name:[] for name in inputs['task_types']};pid=0
    for h,node in enumerate(nodes[1:]):
        node['platforms']=[]
        for name,task in inputs['task_types'].items():
            if f'hostcpu{h}' in task['platforms']:
                node['platforms'].append(f'hostcpu{h}')
                replicas[name].append({'node_name':f'node{h}','platform_id':pid,'warm':True});pid+=1
    rc={name:{'per_client':0,'per_server':1} for name in replicas}
    infra={'compute_slots_per_node':1,'network':{'bandwidth':100.},'nodes':nodes,'warmth_physics':'node_disk_v2','preinitialize_platforms':True,'defer_cold_replica_init':False,'replicas':rc,'replica_plan':{'preinit_clients':[],'preinit_servers':node_names[1:],'preinit_task_types':[],'replicas_config':rc},'deterministic_replica_placements':replicas,'deterministic_queue_distributions':{name:{} for name in replicas}}
    workload={'rps':jobs,'duration':1,'events':[{'timestamp':0.,'application':inputs['application_types'][f'nofs-job{j}'],'qos':inputs['qos_types']['medium'],'node_name':'client_node0'} for j in range(jobs)],'peer_exchange':[]}
    return {'infrastructure':infra,'workload':workload},inputs

def run(p,assignment,log,seed=0):
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    config,inputs=substrate(p);original=PeerGreedyNetworkScheduler.placement;decisions=[]
    def place(self,state,task):
        if False:yield
        j,k=map(int,task.type['name'][2:].split('_'));host=int(assignment[j,k])
        matches=[(n,plat) for n,plat in self._get_valid_replicas(state.replicas[task.type['name']],task) if n.node_name==f'node{host}']
        if len(matches)!=1:raise RuntimeError('assignment has no unique replica')
        decisions.append({'job':j,'operation':k,'host':host,'time':float(self.env.now)})
        return matches[0]
    random.seed(seed);np.random.seed(seed);PeerGreedyNetworkScheduler.placement=place
    try:
        with redirect_stdout(log),redirect_stderr(log),deadline(20):
            result=execute_simulation(config,copy.deepcopy(inputs),'peer_greedy_network_peer_greedy_network',keep_alive=1000000,queue_length=100)
    finally:PeerGreedyNetworkScheduler.placement=original
    rows=result['stats']['taskResults'];jobs,ops,_=p.shape
    if len(rows)!=jobs*ops or len(decisions)!=jobs*ops:raise RuntimeError('incomplete live workflow')
    terminal=[r for r in rows if int(r['taskType']['name'].split('_')[1])==ops-1]
    if len(terminal)!=jobs:raise RuntimeError('missing completed jobs')
    return {'job_completion_sum':sum(r['doneTime'] for r in terminal),'total_rtt':result['stats']['total_rtt'],'tasks':rows,'decisions':decisions}

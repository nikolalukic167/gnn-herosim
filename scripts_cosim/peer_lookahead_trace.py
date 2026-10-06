#!/usr/bin/env python3
"""Serve hand lookahead throughout complete traces; no checkpoint or physics changes."""
from __future__ import annotations

import argparse
import copy
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts_cosim.peer_lookahead_live_probe import deadline, make_workload, min_sum, summarize_stats
from scripts_cosim.peer_lookahead_screen import cycle_edges, sha256


def trace_workload(source,config,gap,variant):
    n=config['trace_tasks']
    if n%8:
        raise ValueError('trace must contain whole eight-task groups')
    workload=make_workload(source,config,gap,variant,n)
    workload['peer_exchange']=[[offset+i,offset+j,config['payload_bytes']]
                              for offset in range(0,n,8) for i,j in cycle_edges(config['cycles'][variant])]
    return workload


def choose(self,state,task,manifest,group_ids,peers,types,arm):
    """Past placements are fixed; current/future candidates use CURRENT replicas."""
    from src.placement.live_audit import platform_queue_drain_seconds
    from src.placement.scheduling_cost import incoming_cold_start_time, network_latency_between

    orch=self._pg_orchestrator()
    costs,candidates=[],[]
    current=group_ids.index(int(task.id))
    memo={}
    fixed=set()
    for i,(tid,event) in enumerate(zip(group_ids,manifest)):
        known=orch.task_by_id.get(tid)
        placed=getattr(known,'platform',None)
        if tid!=int(task.id) and placed is not None:
            fixed.add(i)
            candidates.append([(placed.node,placed)])
            costs.append(np.zeros(1))
            continue
        name=next(iter(event['application']['dag']))
        proxy=SimpleNamespace(type=types[name],node_name=event['node_name'])
        valid=self._get_valid_replicas(state.replicas[name],proxy)
        ready=[r for r in valid if r[1].initialized.triggered]
        valid=ready or valid
        if not valid:
            raise RuntimeError(f'no current candidate forecast for declared task {tid}')
        candidates.append(valid)
        costs.append(np.array([platform_queue_drain_seconds(p,orch,memo)+
                               incoming_cold_start_time(proxy,p)+float(proxy.type['executionTime'][p.type['shortName']])+
                               network_latency_between(proxy.node_name,node,self.nodes.items)
                               for node,p in valid]))
    pairs={}
    index={tid:i for i,tid in enumerate(group_ids)}
    for ti,tj,payload in peers:
        i,j=index[ti],index[tj]
        if i in fixed and j in fixed:
            continue
        mat=np.empty((len(candidates[i]),len(candidates[j])))
        for a,(ni,pi) in enumerate(candidates[i]):
            for b,(nj,pj) in enumerate(candidates[j]):
                mat[a,b]=(self._pg_exchange_seconds(ni,pi,[(nj.node_name,payload)])+
                          self._pg_exchange_seconds(nj,pj,[(ni.node_name,payload)]))
        pairs[i,j]=mat
    if arm in ('peer_mass','committed_x2'):
        score=costs[current].copy()
        for (i,j),mat in pairs.items():
            if i==current and (arm=='peer_mass' or j in fixed):
                score+=mat.mean(1)
            elif j==current and (arm=='peer_mass' or i in fixed):
                score+=mat.mean(0)
    else:
        score=min_sum(costs,pairs,2)[current]
    choice=int(score.argmin())
    node,platform=candidates[current][choice]
    return node,platform,{'task_id':int(task.id),'group_start':group_ids[0],
                          'fixed_peers':len(fixed),'current_candidates':len(candidates[current]),
                          'choice':[int(node.id),int(platform.id)]}


def run(source,config,workload,arm,logpath):
    from src.placement.availability import LEGACY, drain_contract
    contract = drain_contract()
    if contract != LEGACY and arm not in ('immediate', 'reactive'):
        raise ValueError('closed lookahead controls require backlog_only_v1; availability_v2 is rule-only')
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    from src.policy.knative_network.scheduler import KnativeScheduler
    scheduler_class=KnativeScheduler if arm=='reactive' else PeerGreedyNetworkScheduler
    original=scheduler_class.placement
    edges_by_group={}
    for e in workload['peer_exchange']:
        start=(e[0]//8)*8
        if e[1]//8!=e[0]//8:
            raise ValueError('peer edge crosses announced group')
        edges_by_group.setdefault(start,[]).append(e)
    # Announcement is a copy of input metadata, never future Task objects.
    announced={}
    decisions=[]

    def wrapped(self,state,task):
        started=time.perf_counter()
        tid=int(task.id)
        group=(tid//8)*8
        if group in edges_by_group and tid==group:
            announced[group]=copy.deepcopy(workload['events'][group:group+8])
        use=arm not in ('immediate','reactive') and group in edges_by_group and (arm!='group_first_two_hop' or tid==group)
        if use:
            if group not in announced:
                raise RuntimeError('group manifest not announced')
            node,platform,row=choose(self,state,task,announced[group],list(range(group,group+8)),
                                     edges_by_group[group],source['sim_inputs']['task_types'],arm)
            result=(node,platform)
        else:
            result=yield from original(self,state,task)
            row={'task_id':tid,'choice':[int(result[0].id),int(result[1].id)]}
        row.update(lookahead=use,wall_seconds=time.perf_counter()-started)
        decisions.append(row)
        return result

    cfg=copy.deepcopy(source['config'])
    cfg['infrastructure'].pop('forced_placements',None)
    cfg['workload']=copy.deepcopy(workload)
    random.seed(config['seed']); np.random.seed(config['seed'])
    scheduler_class.placement=wrapped
    started=time.monotonic()
    try:
        with logpath.open('w') as log,redirect_stdout(log),redirect_stderr(log),deadline(config['run_timeout_s']):
            strategy='kn_network_kn_network' if arm=='reactive' else 'peer_greedy_network_peer_greedy_network'
            result=execute_simulation(cfg,copy.deepcopy(source['sim_inputs']),
                                      strategy,cache_policy='fifo',task_priority='fifo',
                                      keep_alive=1000000,queue_length=100)
        n=len(workload['events'])
        stats=summarize_stats(result['stats'],n)
        if sorted(r['task_id'] for r in decisions)!=list(range(n)):
            raise RuntimeError('not exactly one decision per task')
        placements={int(r['taskId']):r['executionNode'] for r in stats['tasks']}
        d=np.array([r['wall_seconds'] for r in decisions])
        stats.update(arm=arm,pg_drain_contract=result['stats'].get('schedulerCounters', {}).get('pg_drain_contract'),
                     wall_seconds=time.monotonic()-started,decisions=decisions,
                     decision_wall_total_s=float(d.sum()),decision_wall_median_ms=float(np.median(d)*1000),
                     decision_wall_p95_ms=float(np.percentile(d,95)*1000),
                     rtt_plus_decision_wall_s=stats['total_rtt']+float(d.sum()),
                     colocated_peer_pairs=sum(placements[i]==placements[j] for i,j,_ in workload['peer_exchange']),
                     peer_pairs=len(workload['peer_exchange']))
        quarter=max(1,n//4)
        stats['quarter_mean_rtt']=[float(np.mean([r['elapsedTime'] for r in stats['tasks']
                                               if q*quarter<=int(r['taskId'])<(q+1)*quarter])) for q in range(4)]
        return stats
    finally:
        scheduler_class.placement=original


def main():
    from src.placement.availability import LEGACY, drain_contract
    if drain_contract() != LEGACY:
        raise ValueError('closed lookahead comparison requires backlog_only_v1')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'experiments/peer_lookahead_trace_v1.json')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args(); cfg=json.loads(a.config.read_text())
    a.out_dir.mkdir(parents=True,exist_ok=True)
    logs=a.out_dir/'logs'; logs.mkdir(exist_ok=True)
    for k in list(os.environ):
        if k.startswith(('HEROSIM_FORCED_','HEROSIM_PG_','LIVE_AUDIT_','GNN_CAPTURE_','HEROSIM_MAX_')):
            os.environ.pop(k,None)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_COSIM_KEEP_ALIVE='1000000',
                      COSIM_SUPPRESS_SIM_PRINTS='1',SIM_FORCE_FULL_STATS='1')
    os.environ.pop('HEROSIM_DATA_LOCALITY',None)
    os.environ.pop('COSIM_AUTOSCALER_RECONCILE_INTERVAL',None)
    started=time.monotonic(); cells=[]
    for ds in cfg['sources'][:1] if a.smoke else cfg['sources']:
        path=ROOT/cfg['source_corpus']/ds/'optimal_result.json'; source=json.loads(path.read_text())
        for gap in cfg['arrival_gaps_s'][:1] if a.smoke else cfg['arrival_gaps_s']:
            for variant in range(2):
                w=trace_workload(source,cfg,gap,variant)
                cell={'source':ds,'source_sha256':sha256(path),'gap_s':gap,'variant':variant,'arms':{},'errors':{}}
                for arm in cfg['arms']:
                    tag=f'{ds}_g{gap}_v{variant}_{arm}'
                    try:
                        cell['arms'][arm]=run(source,cfg,w,arm,logs/f'{tag}.log')
                    except (RuntimeError,ValueError,KeyError,TimeoutError) as e:
                        cell['errors'][arm]=f'{type(e).__name__}: {e}'
                    print(tag,cell['errors'].get(arm,'completed'),flush=True)
                cells.append(cell)
                (a.out_dir/'partial.json').write_text(json.dumps(cells,indent=2,allow_nan=False)+'\n')
    # Replay the exact previously identified failure, including its peer-free suffix.
    regression_source=json.loads((ROOT/cfg['source_corpus']/'ds_00064'/'optimal_result.json').read_text())
    regression_w=make_workload(regression_source,cfg,1.,1,16)
    regression={arm:run(regression_source,cfg,regression_w,arm,logs/f'regression_{arm}.log') for arm in cfg['arms']}
    summary={}
    for arm in cfg['arms']:
        complete=[c for c in cells if not c['errors']]
        changes=[100*(c['arms'][arm]['total_rtt']/c['arms']['immediate']['total_rtt']-1) for c in complete]
        adjusted=[100*(c['arms'][arm]['rtt_plus_decision_wall_s']/c['arms']['immediate']['rtt_plus_decision_wall_s']-1) for c in complete]
        summary[arm]={'paired_median_rtt_change_pct':float(np.median(changes)) if changes else None,
                       'wins':sum(x<0 for x in changes),'n':len(changes),
                       'paired_median_rtt_plus_direct_overhead_change_pct':float(np.median(adjusted)) if adjusted else None}
    out={'config':cfg,'smoke':a.smoke,'config_sha256':sha256(a.config),'script_sha256':sha256(__file__),
         'dependency_hashes':{p:sha256(ROOT/p) for p in ['scripts_cosim/peer_lookahead_live_probe.py',
                             'src/policy/peer_greedy_network/scheduler.py','src/placement/infrastructure.py']},
         'wall_seconds':time.monotonic()-started,'cells':cells,'regression':regression,'summary':summary}
    (a.out_dir/'read.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if any(c['errors'] for c in cells):
        raise SystemExit(2)


if __name__=='__main__':
    main()

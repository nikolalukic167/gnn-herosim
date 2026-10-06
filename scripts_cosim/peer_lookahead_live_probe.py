#!/usr/bin/env python3
"""Actual-simulator partial-arrival probe with one forced choice and fixed continuation.

The scoped placement wrapper observes state at task 0 only. Future task metadata
comes from an explicit manifest; no future Task, queue or placement is consulted.
No production policy or checkpoint contract is changed.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_cosim.peer_lookahead_screen import cycle_edges, sha256


def make_workload(source, config, gap, variant, horizon, *, exchange=True):
    if horizon < len(config['task_types']):
        raise ValueError('horizon would truncate the peer group')
    base = source['config']['workload']['events'][0]
    events = []
    for i in range(horizon):
        task_type = config['task_types'][i % len(config['task_types'])]
        event = copy.deepcopy(base)
        event.update(timestamp=i * gap,
                     application={'name': f'nofs-{task_type}', 'dag': {task_type: []}})
        events.append(event)
    peers = [[i, j, config['payload_bytes']] for i, j in cycle_edges(config['cycles'][variant])] if exchange else []
    return {'rps': 1/gap, 'duration': horizon*gap, 'events': events, 'peer_exchange': peers}


def forecast(self, state, task, manifest, sim_inputs, peers):
    from src.placement.live_audit import platform_queue_drain_seconds
    from src.placement.live_snapshot_seed import _approx_comm
    from src.placement.scheduling_cost import incoming_cold_start_time, network_latency_between

    orch = self._pg_orchestrator()
    if set(orch.task_by_id) != {0}:
        raise RuntimeError('forecast is not at the first arrival')
    candidates, features, services = [], [], []
    for event in manifest:
        name = next(iter(event['application']['dag']))
        proxy = SimpleNamespace(type=sim_inputs['task_types'][name], node_name=event['node_name'])
        valid = self._get_valid_replicas(state.replicas[name], proxy)
        initialized = [r for r in valid if r[1].initialized.triggered]
        valid = initialized or valid
        if not valid:
            raise RuntimeError(f'no current candidate forecast for {name}')
        candidates.append(valid)
        rows, svc = [], []
        for node, platform in valid:
            ptype = platform.type['shortName']
            execution = float(proxy.type['executionTime'][ptype])
            rows.append([platform_queue_drain_seconds(platform, orch, {}),
                         incoming_cold_start_time(proxy, platform), execution,
                         network_latency_between(proxy.node_name, node, self.nodes.items)])
            svc.append(execution + _approx_comm(proxy.type))
        features.append(np.asarray(rows, dtype=float))
        services.append(svc)
    costs = [f.sum(1) for f in features]
    pair, all_pair = {}, {}
    edge_payloads = {(min(i,j), max(i,j)): b for i,j,b in peers}
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            x = np.zeros((len(candidates[i]), len(candidates[j])))
            sharing = np.zeros_like(x)
            payload = edge_payloads.get((i,j))
            for a, (ni, pi) in enumerate(candidates[i]):
                for b, (nj, pj) in enumerate(candidates[j]):
                    if payload is not None:
                        x[a,b] = self._pg_exchange_seconds(ni, pi, [(nj.node_name, payload)])
                        x[a,b] += self._pg_exchange_seconds(nj, pj, [(ni.node_name, payload)])
                    if (ni.id, pi.id) == (nj.id, pj.id):
                        sharing[a,b] = (services[i][a]+services[j][b])/2
            if payload is not None:
                pair[i,j] = x
            if np.any(x+sharing):
                all_pair[i,j] = x+sharing
    return {'costs': costs, 'peer': pair, 'full': all_pair,
            'candidates': [[[int(n.id), int(p.id)] for n,p in cs] for cs in candidates],
            'features': [f.tolist() for f in features],
            'time': float(self.env.now), 'observed_task_ids': sorted(orch.task_by_id)}


def min_sum(costs, pairs, rounds):
    neighbors = [[] for _ in costs]
    directed = {}
    for (i,j), mat in pairs.items():
        directed[i,j], directed[j,i] = mat, mat.T
        neighbors[i].append(j)
        neighbors[j].append(i)
    msg = {(i,j): np.zeros(len(costs[j])) for i,j in directed}
    for _ in range(rounds):
        new = {}
        for (i,j), mat in directed.items():
            cavity = costs[i].copy()
            for h in neighbors[i]:
                if h != j:
                    cavity += msg[h,i]
            v = (cavity[:,None]+mat).min(0)
            new[i,j] = v-v.min()
        msg = new
    result = [v.copy() for v in costs]
    for (i,j), v in msg.items():
        result[j] += v
    return result


def static_cost(costs, pairs, plan):
    return float(sum(costs[i][p] for i,p in enumerate(plan)) +
                 sum(mat[plan[i],plan[j]] for (i,j),mat in pairs.items()))


def static_greedy(costs, pairs, peer=None):
    plan = []
    for i,cost in enumerate(costs):
        score = cost.copy()
        for j,c in enumerate(plan):
            if (j,i) in pairs:
                score += pairs[j,i][c]
        if peer is not None:
            for j in range(i+1,len(costs)):
                if (i,j) in peer:
                    score += peer[i,j].mean(1)
        plan.append(int(score.argmin()))
    return plan


def static_cd(costs, pairs, initial):
    plan = list(initial)
    for _ in range(3):
        for i,cost in enumerate(costs):
            best, value = plan[i], static_cost(costs,pairs,plan)
            for c in range(len(cost)):
                trial = list(plan)
                trial[i] = c
                v = static_cost(costs,pairs,trial)
                if v < value-1e-10:
                    best,value = c,v
            plan[i] = best
    return plan


def hand_controls(f):
    costs, peer, pairs = f['costs'], f['peer'], f['full']
    plans = {'immediate': static_greedy(costs,pairs),
             'peer_mass': static_greedy(costs,pairs,peer)}
    beliefs = {f'peer_min_sum_{r}': min_sum(costs,peer,r) for r in (2,3,8)}
    beliefs['full_min_sum_8'] = min_sum(costs,pairs,8)
    plans.update({k: [int(v.argmin()) for v in b] for k,b in beliefs.items()})
    plans['cd3'] = static_cd(costs,pairs,plans['immediate'])
    plans['cd3_multistart'] = min((static_cd(costs,pairs,p) for p in list(plans.values())),
                                  key=lambda p: (static_cost(costs,pairs,p),tuple(p)))
    return {k: int(p[0]) for k,p in plans.items()}, {k:v[0].tolist() for k,v in beliefs.items()}


def summarize_stats(stats, expected):
    rows = stats.get('taskResults') or []
    if len(rows) != expected or {int(r['taskId']) for r in rows} != set(range(expected)):
        raise RuntimeError(f'incomplete run: expected {expected}, saw {len(rows)} tasks')
    total = sum(float(r['elapsedTime']) for r in rows)
    if not np.isclose(total, stats['total_rtt'], rtol=1e-9, atol=1e-9):
        raise RuntimeError('total_rtt does not equal sum of task RTTs')
    for stat, field in [('totalPeerExchangeTime','peerExchangeTime'),
                        ('totalPeerRendezvousWait','peerRendezvousWait')]:
        if not np.isclose(float(stats[stat]),sum(float(r[field]) for r in rows),rtol=1e-9,atol=1e-9):
            raise RuntimeError(f'{stat} does not reconcile with task rows')
    return {'total_rtt': total, 'num_tasks': expected,
            'group_rtt': sum(float(r['elapsedTime']) for r in rows if int(r['taskId']) < 8),
            'queue': float(stats['averageQueueTime'])*expected,
            'exchange': float(stats['totalPeerExchangeTime']),
            'rendezvous': float(stats['totalPeerRendezvousWait']),
            'wait': float(stats['averageWaitTime'])*expected,
            'root_placement': next([r['executionNode'],str(r['executionPlatform'])] for r in rows if int(r['taskId'])==0),
            'tasks': rows}


@contextmanager
def deadline(seconds):
    def expired(_signum,_frame):
        raise TimeoutError(f'simulator exceeded {seconds} wall seconds')
    previous = signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


def simulate(source, config, workload, log_path, forced=None, capture=False):
    from src.placement.availability import LEGACY, drain_contract
    if drain_contract() != LEGACY:
        raise ValueError('closed lookahead probe requires backlog_only_v1')
    from src.executecosimulation import execute_simulation
    from src.policy.peer_greedy_network.scheduler import PeerGreedyNetworkScheduler
    original = PeerGreedyNetworkScheduler.placement
    snapshots = []
    manifest = copy.deepcopy(workload['events'][:8])

    def wrapped(self,state,task):
        if int(task.id)==0 and capture:
            snapshots.append(forecast(self,state,task,manifest,source['sim_inputs'],workload['peer_exchange']))
        return (yield from original(self,state,task))

    cfg = copy.deepcopy(source['config'])
    cfg['infrastructure'].pop('forced_placements',None)
    if forced is not None:
        cfg['infrastructure']['forced_placements'] = {0: list(forced)}
    cfg['workload'] = copy.deepcopy(workload)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    PeerGreedyNetworkScheduler.placement = wrapped
    started=time.monotonic()
    try:
        with log_path.open('w') as log, redirect_stdout(log), redirect_stderr(log), deadline(config['run_timeout_s']):
            result = execute_simulation(cfg,copy.deepcopy(source['sim_inputs']),
                                        'peer_greedy_network_peer_greedy_network',
                                        cache_policy='fifo',task_priority='fifo',keep_alive=1000000,queue_length=100)
        summary = summarize_stats(result['stats'],len(workload['events']))
        summary['simulation_wall_seconds']=time.monotonic()-started
        placements={int(r['taskId']):r['executionNode'] for r in summary['tasks']}
        edges=workload['peer_exchange']
        summary['peer_pairs']=len(edges)
        summary['colocated_peer_pairs']=sum(placements[i]==placements[j] for i,j,_ in edges)
        if capture and len(snapshots)!=1:
            raise RuntimeError('missing or repeated root snapshot')
        return summary, snapshots[0] if snapshots else None
    finally:
        PeerGreedyNetworkScheduler.placement = original


def argmin_set(values):
    a = np.asarray(values)
    return np.flatnonzero(np.isclose(a,a.min(),rtol=1e-9,atol=1e-9)).tolist()


def rank_stability(a,b):
    from scipy.stats import spearmanr
    if np.allclose(a,a[0]) or np.allclose(b,b[0]):
        rho = None
    else:
        rho = float(spearmanr(a,b).statistic)
    return {'spearman':rho, 'common_best_candidate':bool(set(argmin_set(a))&set(argmin_set(b)))}


def run_probe(config, outdir, *, smoke=False):
    outdir.mkdir(parents=True,exist_ok=True)
    logs = outdir/'logs'
    logs.mkdir(exist_ok=True)
    # Explicit physics plus removal of optional hooks that would contaminate a replay.
    for k in list(os.environ):
        if k.startswith(('HEROSIM_FORCED_', 'HEROSIM_PG_', 'LIVE_AUDIT_', 'GNN_CAPTURE_', 'HEROSIM_MAX_')):
            os.environ.pop(k,None)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1', HEROSIM_COSIM_KEEP_ALIVE='1000000',
                      COSIM_SUPPRESS_SIM_PRINTS='1', SIM_FORCE_FULL_STATS='1')
    os.environ.pop('HEROSIM_DATA_LOCALITY',None)
    os.environ.pop('COSIM_AUTOSCALER_RECONCILE_INTERVAL',None)
    cells = []
    sources = config['sources'][:1] if smoke else config['sources']
    gaps = config['arrival_gaps_s'][:1] if smoke else config['arrival_gaps_s']
    for ds in sources:
        source_path = ROOT/config['source_corpus']/ds/'optimal_result.json'
        source = json.loads(source_path.read_text())
        for gap in gaps:
            cell = {'source':ds, 'source_sha256':sha256(source_path), 'gap_s':gap, 'variants':[]}
            for v in range(2):
                variant = {'variant':v, 'horizons':{}, 'errors':[]}
                for horizon in config['horizons_tasks']:
                    tag = f'{ds}_g{gap}_v{v}_h{horizon}'
                    workload = make_workload(source,config,gap,v,horizon)
                    (outdir/f'{tag}.workload.json').write_text(json.dumps(workload,indent=2)+'\n')
                    try:
                        baseline, f = simulate(source,config,workload,logs/f'{tag}_baseline.log',capture=True)
                        controls, columns = hand_controls(f)
                        if len(f['candidates'][0])<2:
                            raise RuntimeError('root has fewer than two candidates; headroom unreadable')
                        trials=[]
                        for c,placement in enumerate(f['candidates'][0]):
                            s,_ = simulate(source,config,workload,logs/f'{tag}_c{c}.log',forced=placement)
                            trials.append(s)
                        node_names = [n['node_name'] for n in source['config']['infrastructure']['nodes']]
                        bi = next(i for i,p in enumerate(f['candidates'][0]) if
                                  [node_names[p[0]],str(p[1])]==baseline['root_placement'])
                        if not np.isclose(baseline['total_rtt'],trials[bi]['total_rtt'],rtol=1e-9,atol=1e-9):
                            raise RuntimeError('forcing the baseline choice changes the replay')
                        if controls['immediate']!=bi:
                            raise RuntimeError('forecast immediate root differs from live greedy')
                        values=[s['total_rtt'] for s in trials]
                        variant['horizons'][str(horizon)] = {
                            'baseline':baseline,'trials':trials,'total_rtt_by_candidate':values,
                            'optimal_candidates':argmin_set(values), 'controls':controls,
                            'control_regret_pct':{k:100*(values[i]/min(values)-1) for k,i in controls.items()},
                            'forecast':{'candidates':f['candidates'],'features':f['features'],
                                        'time':f['time'],'observed_task_ids':f['observed_task_ids'],
                                        'root_hand_columns':columns}}
                    except (RuntimeError,TimeoutError,ValueError,KeyError) as e:
                        variant['errors'].append({'horizon':horizon,'error':str(e),'type':type(e).__name__})
                        print(f'{tag}: FAILED {type(e).__name__}: {e}',flush=True)
                        break
                cell['variants'].append(variant)
            # Both rewired workloads collapse to the exact same input when
            # exchange is absent. Execute that control, including every action.
            h=max(config['horizons_tasks'])
            off=make_workload(source,config,gap,0,h,exchange=False)
            if off!=make_workload(source,config,gap,1,h,exchange=False):
                raise RuntimeError('exchange-off workloads differ')
            try:
                baseline,f=simulate(source,config,off,logs/f'{ds}_g{gap}_off_baseline.log',capture=True)
                trials=[]
                for c,p in enumerate(f['candidates'][0]):
                    s,_=simulate(source,config,off,logs/f'{ds}_g{gap}_off_c{c}.log',forced=p)
                    if s['exchange']!=0 or s['rendezvous']!=0:
                        raise RuntimeError('exchange-off control charges peer physics')
                    trials.append(s)
                cell['exchange_off']={'horizon':h,'baseline':baseline,'trials':trials,
                                      'candidates':f['candidates'][0],
                                      'optimal_candidates':argmin_set([s['total_rtt'] for s in trials])}
            except (RuntimeError,TimeoutError,ValueError,KeyError) as e:
                cell['exchange_off']={'error':str(e),'type':type(e).__name__}
            cells.append(cell)
            (outdir/'partial.json').write_text(json.dumps(cells,indent=2,allow_nan=False)+'\n')
            print(f'completed {ds} gap={gap}: {[len(v["horizons"]) for v in cell["variants"]]} horizons',flush=True)
    return cells


def analyze(cells, config):
    complete, failed = [], []
    for cell in cells:
        if 'error' in cell['exchange_off'] or any(v['errors'] or len(v['horizons'])!=len(config['horizons_tasks']) for v in cell['variants']):
            failed.append({'source':cell['source'],'gap_s':cell['gap_s'],
                           'errors':[v['errors'] for v in cell['variants']], 'exchange_off_error':cell['exchange_off'].get('error')})
            continue
        a,b=cell['variants']
        by_h={}
        for h in map(str,config['horizons_tasks']):
            aa,bb=a['horizons'][h],b['horizons'][h]
            fa,fb=aa['forecast'],bb['forecast']
            if fa['candidates']!=fb['candidates'] or fa['features']!=fb['features']:
                raise RuntimeError('matched variants have different current-state candidate features')
            for name in ('peer_min_sum_2',):
                if not np.allclose(fa['root_hand_columns'][name],fb['root_hand_columns'][name],atol=1e-9,rtol=0):
                    raise RuntimeError('two-hop feature match failed')
            regrets=[100*(np.asarray(v['total_rtt_by_candidate'])/min(v['total_rtt_by_candidate'])-1) for v in (aa,bb)]
            by_h[h]={'disjoint_best_candidates':not bool(set(aa['optimal_candidates'])&set(bb['optimal_candidates'])),
                      'matched_blind_best_mean_regret_pct':float(np.mean(regrets,axis=0).min())}
        stable=[]
        for variant in cell['variants']:
            first=variant['horizons'][str(config['horizons_tasks'][0])]
            chosen=int(np.argmin(first['total_rtt_by_candidate']))
            for h,d in variant['horizons'].items():
                if d['forecast']!=first['forecast']:
                    raise RuntimeError('suffix altered the observation at task 0')
                # The eight-task rollout sees only the declared manifest. Its
                # zero regret on that same horizon is tautological, not a result.
                values=d['total_rtt_by_candidate']
                d['group_rollout_h8']={'candidate':chosen,'regret_pct':100*(values[chosen]/min(values)-1),
                                       'in_sample':int(h)==8,
                                       'planning_wall_seconds':sum(t['simulation_wall_seconds'] for t in first['trials'])}
                if int(h)>8:
                    stable.append({'horizon':int(h),**rank_stability(first['total_rtt_by_candidate'],values)})
        complete.append({'source':cell['source'],'gap_s':cell['gap_s'],'matched':by_h,'horizon_stability':stable})
    aggregate={}
    for h in map(str,config['horizons_tasks']):
        rows=[v['horizons'][h] for c in cells for v in c['variants']
              if 'error' not in c['exchange_off'] and all(not vv['errors'] and len(vv['horizons'])==len(config['horizons_tasks']) for vv in c['variants'])]
        if not rows:
            continue
        aggregate[h]={'variants':len(rows),'disjoint_pairs':sum(c['matched'][h]['disjoint_best_candidates'] for c in complete),
                       'controls':{},'baseline_rendezvous_median_s':float(np.median([r['baseline']['rendezvous'] for r in rows]))}
        for name in rows[0]['control_regret_pct']:
            regrets=[r['control_regret_pct'][name] for r in rows]
            aggregate[h]['controls'][name]={'median_regret_pct':float(np.median(regrets)),
                                            'max_regret_pct':float(max(regrets)),
                                            'above_5pct':sum(v>=config['material_regret_pct'] for v in regrets)}
        if int(h)>8:
            regrets=[r['group_rollout_h8']['regret_pct'] for r in rows]
            aggregate[h]['controls']['group_rollout_h8']={'median_regret_pct':float(np.median(regrets)),
                                                         'max_regret_pct':float(max(regrets)),
                                                         'above_5pct':sum(v>=config['material_regret_pct'] for v in regrets)}
    return {'complete_pairs':len(complete),'failed_pairs':failed,'pairs':complete,'by_horizon':aggregate}


def main():
    from src.placement.availability import LEGACY, drain_contract
    if drain_contract() != LEGACY:
        raise ValueError('closed lookahead probe requires backlog_only_v1')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'experiments/peer_lookahead_live_probe_v1.json')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    cfg=json.loads(a.config.read_text())
    started=time.monotonic()
    cells=run_probe(cfg,a.out_dir,smoke=a.smoke)
    summary=analyze(cells,cfg)
    dependencies=('src/executecosimulation.py','src/placement/simulation.py','src/placement/infrastructure.py',
                  'src/placement/orchestrator.py','src/policy/peer_greedy_network/scheduler.py',
                  'src/policy/knative_network/scheduler.py','src/placement/live_audit.py')
    result={'kind':cfg['kind'],'config':cfg,'config_sha256':sha256(a.config),
            'script_sha256':sha256(__file__),'dependency_hashes':{p:sha256(ROOT/p) for p in dependencies},
            'wall_seconds':time.monotonic()-started,'smoke':a.smoke,'cells':cells,'summary':summary}
    (a.out_dir/'read.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'wrote {a.out_dir / "read.json"}',flush=True)
    print(json.dumps(summary['by_horizon'],indent=2),flush=True)
    if summary['failed_pairs']:
        raise SystemExit(2)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Fresh-infrastructure validation with baseline-only load selection and fixed bars."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from scripts_cosim.peer_lookahead_screen import sha256
from scripts_cosim.peer_lookahead_trace import run, trace_workload


def infrastructure_fingerprint(source):
    """Physical infrastructure identity, independent of task placements/queue draws."""
    infra = source['config']['infrastructure']
    identity = {k: infra.get(k) for k in (
        'network', 'nodes', 'link_topology', 'compute_slots_per_node',
        'ingress_bandwidth_mbps', 'warmth_physics')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_sources(cfg):
    if cfg.get('source_fingerprint_contract') != 'physical_infrastructure_v2':
        raise ValueError('source manifest requires physical_infrastructure_v2 fingerprints')
    names = cfg['sources']
    entries = cfg['source_manifest']
    if len(set(names)) != len(names) or len(entries) != len(names):
        raise ValueError('duplicate sources or incomplete source manifest')
    manifest = {e['source']: e for e in entries}
    if len(manifest) != len(entries) or set(manifest) != set(names):
        raise ValueError('source manifest does not match configured sources')
    verified = {}
    seen = set()
    for name in names:
        path = ROOT / cfg['source_corpus'] / name / 'optimal_result.json'
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        expected = manifest[name]
        if file_hash != expected.get('source_sha256'):
            raise ValueError(f'{name}: source fingerprint mismatch')
        identity = infrastructure_fingerprint(json.loads(raw))
        if identity != expected.get('physical_infrastructure_sha256'):
            raise ValueError(f'{name}: source or infrastructure fingerprint mismatch')
        if identity in seen:
            raise ValueError(f'{name}: duplicate actual infrastructure')
        seen.add(identity)
        verified[name] = {'source_sha256': file_hash, 'physical_infrastructure_sha256': identity}
    return verified


def fresh_workload(source,config,gap,seed):
    """Two new traces, varying type order, cycle and spacing within each group."""
    w=trace_workload(source,config,gap,0)
    rng=random.Random(seed)
    timestamp=0.
    edges=[]
    for group in range(0,config['trace_tasks'],8):
        types=list(config['task_types']);rng.shuffle(types)
        cycle=list(config['cycles'][rng.randrange(len(config['cycles']))])
        for a,b in zip(cycle,cycle[1:]+cycle[:1]):
            edges.append([group+min(a,b),group+max(a,b),config['payload_bytes']])
        for i,name in enumerate(types):
            event=w['events'][group+i]
            event['application']={'name':f'nofs-{name}','dag':{name:[]}}
            event['timestamp']=timestamp
            timestamp+=gap*rng.uniform(.75,1.25)
    w['peer_exchange']=edges
    w['duration']=timestamp
    return w


def healthy(stats,bars):
    share=stats['queue']/stats['total_rtt']
    q=stats['quarter_mean_rtt']
    bound=q[1]*bars['quarter_growth_ratio_max']+bars['quarter_growth_absolute_allowance_s']
    return {'queue_share':share,'second_quarter_rtt':q[1],'last_quarter_rtt':q[3],
            'last_quarter_bound':bound,'pass':share<=bars['queue_share_max'] and q[3]<=bound}


def environment():
    for k in list(os.environ):
        if k.startswith(('HEROSIM_FORCED_','HEROSIM_PG_','LIVE_AUDIT_','GNN_CAPTURE_','HEROSIM_MAX_')):
            os.environ.pop(k,None)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_COSIM_KEEP_ALIVE='1000000',
                      COSIM_SUPPRESS_SIM_PRINTS='1',SIM_FORCE_FULL_STATS='1',OMP_NUM_THREADS='1')
    os.environ.pop('HEROSIM_DATA_LOCALITY',None)
    os.environ.pop('COSIM_AUTOSCALER_RECONCILE_INTERVAL',None)


def one_source(job):
    cfg,ds,outdir=job
    outdir=Path(outdir)/ds;outdir.mkdir(parents=True,exist_ok=True)
    environment()
    p=ROOT/cfg['source_corpus']/ds/'optimal_result.json'
    raw = p.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    expected = next(e for e in cfg['source_manifest'] if e['source'] == ds)
    if source_hash != expected['source_sha256']:
        raise ValueError(f'{ds}: source changed after preflight')
    source = json.loads(raw)
    if infrastructure_fingerprint(source) != expected['physical_infrastructure_sha256']:
        raise ValueError(f'{ds}: source changed after preflight')
    result={'source':ds,'source_sha256':source_hash,'load_screen':[],'traces':[],'error':None}
    try:
        selected=None
        for gap in cfg['arrival_gaps_s']:
            traces=[]
            for seed in cfg['workload_seeds']:
                w=fresh_workload(source,cfg,gap,seed)
                baseline={arm:run(source,cfg,w,arm,outdir/f'g{gap}_s{seed}_{arm}.log')
                          for arm in ['reactive','immediate']}
                health={arm:healthy(stats,cfg['acceptance']) for arm,stats in baseline.items()}
                traces.append({'seed':seed,'gap_s':gap,'arms':baseline,'health':health})
            passed=all(h['pass'] for t in traces for h in t['health'].values())
            result['load_screen'].append({'gap_s':gap,'pass':passed,
                                          'traces':[{'seed':t['seed'],'health':t['health']} for t in traces]})
            if passed:
                selected=traces
                break
        if selected is None:
            raise RuntimeError('no healthy load on fixed ladder; source remains inadmissible')
        for trace in selected:
            w=fresh_workload(source,cfg,trace['gap_s'],trace['seed'])
            (outdir/f's{trace["seed"]}.workload.json').write_text(json.dumps(w,indent=2)+'\n')
            for arm in ['peer_mass','group_first_two_hop']:
                trace['arms'][arm]=run(source,cfg,w,arm,outdir/f's{trace["seed"]}_{arm}.log')
            result['traces'].append(trace)
    except (RuntimeError,ValueError,KeyError,TimeoutError) as e:
        result['error']=f'{type(e).__name__}: {e}'
    (outdir/'read.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def evaluate(results,cfg):
    bars=cfg['acceptance']; valid=[r for r in results if not r['error']]
    outcome={'complete_infrastructures':len(valid),'required':bars['minimum_infrastructures'],
             'errors':[{'source':r['source'],'error':r['error']} for r in results if r['error']],
             'comparisons':{}}
    rng=np.random.default_rng(cfg['bootstrap_seed'])
    for control in ['immediate','peer_mass']:
        gains=[];flat=[]
        for source in valid:
            pairs=[]
            for t in source['traces']:
                a=t['arms']['group_first_two_hop']['total_rtt'];b=t['arms'][control]['total_rtt']
                gain=100*(1-a/b);pairs.append(gain)
                flat.append({'source':source['source'],'seed':t['seed'],'gain_pct':gain})
            gains.append(float(np.median(pairs)))
        if not gains:
            outcome['comparisons'][control]={'pass':False,'reason':'no readable infrastructure'}
            continue
        boot=np.median(rng.choice(gains,size=(20000,len(gains)),replace=True),axis=1)
        lower=float(np.quantile(boot,1-bars['bootstrap_confidence']))
        med=float(np.median(gains));share=sum(x>0 for x in gains)/len(gains)
        worst=min(flat,key=lambda v:v['gain_pct'])
        tests={'median_gain':med>=bars['median_gain_vs_each_control_pct'],
               'lower_confidence_bound':lower>=bars['bootstrap_lower_gain_pct'],
               'win_share':share>=bars['minimum_topology_win_share'],
               'regression_bound':worst['gain_pct']>=-bars['maximum_trace_regression_pct']}
        outcome['comparisons'][control]={'median_gain_pct':med,'bootstrap_one_sided_lower_gain_pct':lower,
                                         'infrastructure_win_share':share,'worst_trace':worst,
                                         'infrastructure_gains_pct':gains,'traces':flat,'checks':tests,'pass':all(tests.values())}
    enough=len(valid)>=bars['minimum_infrastructures'] and not outcome['errors']
    outcome['verdict']='ADVANCE' if enough and all(c['pass'] for c in outcome['comparisons'].values()) else 'DO-NOT-ADVANCE'
    outcome['reason']='fixed acceptance criteria; failure is not proof of model-class equivalence'
    return outcome


def main():
    from src.placement.availability import LEGACY, drain_contract
    if drain_contract() != LEGACY:
        raise ValueError('closed lookahead validation requires backlog_only_v1')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'experiments/peer_lookahead_validation_v1.json')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args();cfg=json.loads(a.config.read_text())
    verified_sources = verify_sources(cfg)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    frozen={'config':cfg,'config_sha256':sha256(a.config),'script_sha256':sha256(__file__),
            'verified_sources': verified_sources,
            'dependency_hashes':{p:sha256(ROOT/p) for p in ['scripts_cosim/peer_lookahead_trace.py',
              'scripts_cosim/peer_lookahead_live_probe.py','src/placement/infrastructure.py','src/policy/peer_greedy_network/scheduler.py']}}
    (a.out_dir/'protocol_before_run.json').write_text(json.dumps(frozen,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        fs={pool.submit(one_source,(cfg,ds,str(a.out_dir))):ds for ds in cfg['sources']}
        for f in as_completed(fs):
            r=f.result();results.append(r)
            print(r['source'],r['error'] or f'completed {len(r["traces"])} traces',flush=True)
    results.sort(key=lambda r:cfg['sources'].index(r['source']))
    summary=evaluate(results,cfg)
    out={**frozen,'workers':a.workers,'wall_seconds':time.monotonic()-started,'summary':summary,
         'source_results':[{'source':r['source'],'path':str(a.out_dir/r['source']/'read.json')} for r in results]}
    (a.out_dir/'read.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if summary['errors']:
        raise SystemExit(2)


if __name__=='__main__':
    main()

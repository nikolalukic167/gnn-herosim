"""Registered pair-selector ceiling and real-simulator gate; no learned policy claim."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,pack,serialized
from src.placement.radical.pairs import PairNative
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.evaluate_workflow import paired_summary
from scripts_cosim.audit_radical_physics import load_problem
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='experiments/mixed_pair_exploration_v1.json'
SOURCES=[PROTOCOL,'scripts_cosim/mixed_pair_exploration.py','src/placement/radical/pairs.py',
         'src/placement/radical/pairs.cpp','src/placement/radical/mixed.py','src/placement/radical/mixed.cpp',
         'src/placement/radical/kernel.cpp','src/placement/radical/environment.py',
         'scripts_cosim/mixed_physics_control_gate.py','scripts_cosim/evaluate_workflow.py']

def verify(out,pre,meta):
    if any(sha(ROOT/p)!=h for p,h in pre['sources'].items()):raise ValueError('source changed')
    seen=set()
    for seed,item in meta['inputs'].items():
        path=out/'inputs'/f'{seed}.json';b=load_problem(path)
        identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
        if sha(path)!=item['sha256'] or identity!=item['identity'] or identity in seen:
            raise ValueError('input identity/uniqueness failure')
        seen.add(identity)

def screen(out):
    if out.exists():raise ValueError('preserve existing experiment')
    out.mkdir(parents=True);(out/'inputs').mkdir()
    protocol=json.loads((ROOT/PROTOCOL).read_text());engine=PairNative(out/'build')
    pre={'protocol':protocol,'sources':{p:sha(ROOT/p) for p in SOURCES},'native':engine.provenance}
    save(out/'protocol_before_run.json',pre)
    prior=set()
    for path in out.parent.glob('*/source_validation.json'):
        d=json.loads(path.read_text())
        for row in d.get('inputs',{}).values():
            if isinstance(row,dict) and 'identity' in row:prior.add(row['identity'])
    meta={'status':'PASS','inputs':{},'prior_identities_checked':len(prior)}
    for key in ('calibration_seeds','fresh_seeds'):
        lo,hi=protocol[key]
        for seed in range(lo,hi+1):
            b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in prior:raise ValueError('reused physical input')
            prior.add(identity);path=out/'inputs'/f'{seed}.json';save(path,serialized(b))
            meta['inputs'][str(seed)]={'identity':identity,'sha256':sha(path)}
    save(out/'source_validation.json',meta);verify(out,pre,meta)
    methods=protocol['controls'];rows=[];lo,hi=protocol['calibration_seeds']
    for seed in range(lo,hi+1):
        b=load_problem(out/'inputs'/f'{seed}.json')
        rows.append({'seed':seed,'methods':{m:timed(lambda m=m:engine.plan(b,m),protocol['timing_repeats']) for m in methods}})
    metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in rows])),
                'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in rows],.95))} for m in methods}
    eligible=[m for m in methods if metrics[m]['p95_ms']<=protocol['budget_ms']]
    if not eligible:raise RuntimeError('no affordable control')
    selected=min(eligible,key=lambda m:metrics[m]['mean_cost'])
    save(out/'calibration.json',{'selected':selected,'metrics':metrics,'rows':rows})
    print('FROZEN',selected,metrics[selected],flush=True)
    rows=[];lo,hi=protocol['fresh_seeds']
    for seed in range(lo,hi+1):
        verify(out,pre,meta);b=load_problem(out/'inputs'/f'{seed}.json');cell=out/'test'/str(seed)
        (cell/'placements').mkdir(parents=True)
        control=timed(lambda:engine.plan(b,selected),protocol['timing_repeats'])
        cost,a=engine.incumbent(b);pairs=np.array(list(combinations(range(a.size),2)),dtype=np.int64)
        costs,plans=engine.candidates(b,a,pairs);idx=int(costs.argmin())
        oracle={'cost':float(costs[idx]),'plan':plans[idx].tolist(),'pair':pairs[idx].tolist()} if costs[idx]<cost else {'cost':cost,'plan':a.tolist(),'pair':None}
        methods_read={'control':control,'local':{'cost':cost,'plan':a.tolist()},'oracle':oracle}
        with (cell/'placements/placements.jsonl').open('w') as f:
            f.write(json.dumps({'pair':None,'placement_plan':a.tolist(),'objective':cost})+'\n')
            for pair,c,plan in zip(pairs,costs,plans):
                f.write(json.dumps({'pair':pair.tolist(),'placement_plan':plan.tolist(),'objective':float(c)})+'\n')
        row={'seed':seed,'methods':methods_read,'candidate_count':len(pairs),'improving_pairs':int((costs<cost).sum()),'candidate_sha256':sha(cell/'placements/placements.jsonl')}
        save(cell/'read.json',row);rows.append(row);print('SCREEN',seed,control['cost'],cost,oracle['cost'],flush=True)
    gain=paired_summary(np.array([[r['methods']['oracle']['cost'] for r in rows]]),np.array([[r['methods']['control']['cost'] for r in rows]]))
    p95=float(np.quantile([r['methods']['control']['time_ms'] for r in rows],.95))
    save(out/'read.json',{'selected':selected,'gain':gain,'control_p95_ms':p95,'headroom_passes':gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and p95<=2,
                          'candidate_count':sum(r['candidate_count'] for r in rows),'live_gate_pending':True,'cases':len(rows)})
    verify(out,pre,meta)
    print((out/'read.json').read_text(),flush=True)

def live(out):
    from scripts_cosim.mixed_physics_live import run
    from scripts_cosim.radical_physics_screen import sha
    from scripts_cosim.workflow_proposal_gate import configure_environment
    from src.placement.radical.live import run as independent
    pre=json.loads((out/'protocol_before_run.json').read_text());meta=json.loads((out/'source_validation.json').read_text())
    verify(out,pre,meta)
    target=out/'live'
    if target.exists():raise ValueError('preserve prior live gate')
    target.mkdir();configure_environment();rows=[];tasks=0
    sources={p:sha(ROOT/p) for p in ['scripts_cosim/mixed_physics_live.py','scripts_cosim/workflow_live.py',
             'src/placement/radical/live.py','src/placement/radical/coordinator.py','src/placement/infrastructure.py','src/placement/simulation.py','src/placement/executor.py']}
    save(target/'sources_before_run.json',sources)
    for cell in sorted((out/'test').iterdir()):
        verify(out,pre,meta)
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('live source changed')
        r=json.loads((cell/'read.json').read_text());seed=r['seed'];b=load_problem(out/'inputs'/f'{seed}.json')
        if sha(cell/'placements/placements.jsonl')!=r['candidate_sha256']:raise ValueError('changed candidates')
        dest=target/str(seed);dest.mkdir();costs={}
        with (dest/'simulation.log').open('w') as log,(dest/'results.jsonl').open('w') as f:
            for arm,d in r['methods'].items():
                a=np.array(d['plan']);stats=run(b,a,log,seed);ref=independent(b,a,'mixed')
                if abs(stats['total_rtt']*1000-d['cost'])>1e-6 or abs(ref['objective']-d['cost'])>1e-6:
                    raise ValueError('live objective mismatch')
                seen=set()
                for task in stats['tasks']:
                    j,k=map(int,task['taskType']['name'][2:].split('_'));seen.add((j,k))
                    if abs(task['doneTime']*1000-ref['ends'][j][k])>1e-6 or task['executionNode']!=f'node{a[j,k]}':
                        raise ValueError('task completion mismatch')
                if len(seen)!=a.size or len(stats['tasks'])!=a.size:raise ValueError('incomplete task coverage')
                tasks+=len(seen);costs[arm]=stats['total_rtt']*1000
                f.write(json.dumps({'arm':arm,'stats':stats},allow_nan=False)+'\n')
        rows.append({'seed':seed,'costs':costs,'sha256':sha(dest/'results.jsonl')});print('LIVE',seed,flush=True)
    report={'status':'PASS','runs':len(rows)*3,'tasks':tasks,'rows':rows,'sources':sources}
    save(out/'live_READ.json',report);print('LIVE PASS',report['runs'],tasks,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['screen','live']);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();(screen if args.mode=='screen' else live)(args.out)

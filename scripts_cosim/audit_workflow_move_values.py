"""Reconcile native-search evidence with scalar costs and actual simulator records."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from src.placement.workflow_planning import instance,evaluate
from scripts_cosim.workflow_live import substrate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    report=json.loads((root/'read.json').read_text());pre=json.loads((root/'protocol_before_run.json').read_text())
    for p,h in pre['sources'].items():
        if sha(p)!=h:raise ValueError(f'changed source: {p}')
    libs=list((root/'build').glob('*.so'))
    if len(libs)!=1 or sha(libs[0])!=pre['native']['library_sha256']:raise ValueError('compiled evaluator changed')
    for name,h in pre['frozen_checkpoints'].items():
        p=Path('models/workflow_amortized_v1')/name
        if sha(p.with_suffix('.pt'))!=h['weights_sha256'] or sha(p.with_suffix('.contract.json'))!=h['contract_sha256']:raise ValueError('model changed')
    lo,hi=pre['protocol']['fresh_live_seeds'];runs=tasks=scored=0;hashes={};identities=set();timing=[]
    for seed in range(lo,hi+1):
        cell=root/'test'/str(seed);r=json.loads((cell/'read.json').read_text());p=instance(seed);config,inputs=substrate(p/1000)
        if r['seed']!=seed or json.loads((cell/'input.json').read_text())!={'config':config,'inputs':inputs}:raise ValueError('source fixture mismatch')
        identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in identities:raise ValueError('duplicate actual source')
        identities.add(identity);arms=set()
        with (cell/'live.jsonl').open() as f:
            for line in f:
                row=json.loads(line);name=row['arm'];a=np.array(row['placement_plan']);s=row['stats'];records=s['tasks']
                if name in arms or len(records)!=80 or len(s['decisions'])!=80:raise ValueError('incomplete or duplicated live run')
                arms.add(name);expected=r['knative_rtt_ms'] if name=='knative_network' else r['methods'][name]['cost_ms']
                if abs(s['total_rtt']*1000-expected)>1e-6 or abs(evaluate(p,a)-expected)>1e-6:raise ValueError('live/scalar/native cost disagreement')
                if abs(sum(t['doneTime']-t['dispatchedTime'] for t in records)-s['total_rtt'])>1e-8:raise ValueError('task-time sum mismatch')
                ids=set()
                for t in records:
                    j,k=map(int,t['taskType']['name'][2:].split('_'));ids.add((j,k))
                    if t['executionNode']!=f'node{a[j,k]}' or abs(t['executionTime']*1000-p[j,k,a[j,k]])>1e-6:raise ValueError('executed task differs from source/plan')
                if len(ids)!=80:raise ValueError('duplicate task')
                for d in s['decisions']:
                    if d['host']!=a[d['job'],d['operation']]:raise ValueError('served choice mismatch')
                runs+=1;tasks+=80
        if arms!=set(r['methods'])|{'knative_network'}:raise ValueError('missing arm')
        count=0
        with (cell/'placements/placements.jsonl').open() as f:
            for line in f:
                row=json.loads(line)
                if abs(evaluate(p,np.array(row['placement_plan']))-row['rtt_ms'])>1e-7:raise ValueError('retained move cost mismatch')
                scored+=1;count+=1
        if count!=r['unique_scored_plans']:raise ValueError('missing move traces')
        timing.append(r['methods']['exact_descent']['planning_ms'])
        for name in ('read.json','input.json','live.jsonl','placements/placements.jsonl'):hashes[str((cell/name).relative_to(root))]=sha(cell/name)
    if runs!=report['live_runs'] or scored!=report['unique_scored_plans']:raise ValueError('aggregate coverage mismatch')
    return {'status':'PASS','cases':len(identities),'live_runs':runs,'completed_tasks':tasks,'independently_recomputed_proposals':scored,'descent_max_case_median_ms':max(timing),'descent_cases_over_2ms':sum(t>2 for t in timing),'artifacts_sha256':hashes,'report_sha256':sha(root/'read.json')}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args();r=audit(args.root);out=args.root/'AUDIT.json'
    if out.exists():raise ValueError('preserve previous audit')
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='artifacts_sha256'},indent=2))

"""Audit frozen search selections, actual task records, and greedy fallback usage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.placement.workflow_planning import instance,evaluate
from scripts_cosim.workflow_live import substrate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit(root):
    report=json.loads((root/'read.json').read_text());pre=json.loads((root/'protocol_before_run.json').read_text());selection=json.loads((root/'frozen_selection.json').read_text());retry=json.loads((root/'retry_amendment.json').read_text())
    if sha(Path(retry['original'])/'frozen_selection.json')!=sha(root/'frozen_selection.json') or sha(root/'frozen_selection.json')!=retry['frozen_selection_sha256']:raise ValueError('retry changed selected search settings')
    for path,h in pre['sources'].items():
        if sha(path)!=h:raise ValueError(f'changed source: {path}')
    for name,h in pre['frozen_checkpoints'].items():
        path=Path('models/workflow_amortized_v1')/name
        if sha(path.with_suffix('.pt'))!=h['weights_sha256'] or sha(path.with_suffix('.contract.json'))!=h['contract_sha256']:raise ValueError('checkpoint changed')
    rows=[];runs=0;task_count=0;seen=set();hashes={};identities=set()
    lo,hi=pre['protocol']['test_seeds']
    for seed in range(lo,hi+1):
        cell=root/'test'/str(seed);row=json.loads((cell/'read.json').read_text());p=instance(seed);config,inputs=substrate(p/1000)
        if json.loads((cell/'input.json').read_text())!={'config':config,'inputs':inputs}:raise ValueError('source fixture changed')
        identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in identities:raise ValueError('duplicate problem')
        identities.add(identity);seen.add(row['seed']);rows.append(row);checked=set()
        with (cell/'live.jsonl').open() as f:
            for line in f:
                r=json.loads(line);name=r['arm'];stats=r['stats'];a=np.array(r['placement_plan']);tasks=stats['tasks']
                expected=row['knative_rtt_ms'] if name=='knative_network' else {**row['learned'],**row['controls']}[name]['cost_ms']
                if name in checked or len(tasks)!=80 or abs(evaluate(p,a)-expected)>1e-6 or abs(stats['total_rtt']*1000-expected)>1e-6:raise ValueError('plan or live RTT mismatch')
                if abs(sum(t['doneTime']-t['dispatchedTime'] for t in tasks)-stats['total_rtt'])>1e-8:raise ValueError('task sum mismatch')
                operations=set()
                for t in tasks:
                    j,k=map(int,t['taskType']['name'][2:].split('_'));operations.add((j,k))
                    if abs(t['executionTime']*1000-p[j,k,a[j,k]])>1e-6 or t['executionNode']!=f'node{a[j,k]}':raise ValueError('executed assignment differs')
                if len(operations)!=80 or len(stats['decisions'])!=80:raise ValueError('incomplete operation or decision records')
                for d in stats['decisions']:
                    if d['host']!=a[d['job'],d['operation']]:raise ValueError('logged choice differs')
                checked.add(name);runs+=1;task_count+=len(tasks)
        if checked!=set(row['live_checked']):raise ValueError('missing live arm')
        with (cell/'placements/placements.jsonl').open() as f:
            for line in f:
                proposal=json.loads(line)
                if abs(evaluate(p,np.array(proposal['placement_plan']))-proposal['rtt_ms'])>1e-6:raise ValueError('searched proposal cost mismatch')
        for file in ('read.json','input.json','live.jsonl','placements/placements.jsonl'):hashes[str((cell/file).relative_to(root))]=sha(cell/file)
    if seen!=set(range(lo,hi+1)) or runs!=report['live_runs']:raise ValueError('incomplete gate')
    fallback={}
    for name in selection:
        key=f'{name}:{selection[name]["moves"]}';gains=[100*(1-r['learned'][key]['cost_ms']/r['controls']['ect']['cost_ms']) for r in rows]
        fallback[name]={'strictly_better_than_ect_cases':sum(g>1e-8 for g in gains),'median_gain_over_ect_pct':float(np.median(gains)),'mean_gain_over_ect_pct':float(np.mean(gains))}
    hand_gains=[100*(1-r['controls'][r['frontier']['2']]['cost_ms']/r['knative_rtt_ms']) for r in rows]
    return {'status':'PASS','cases':len(rows),'live_runs':runs,'completed_tasks':task_count,'proposal_costs_recomputed':True,'fallback':fallback,'best_hand_vs_knative_median_gain_pct':float(np.median(hand_gains)),'artifacts_sha256':hashes,'result_sha256':sha(root/'read.json')}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args();r=audit(args.root);out=args.root/'AUDIT.json'
    if out.exists():raise ValueError('preserve historical audit')
    out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='artifacts_sha256'},indent=2))

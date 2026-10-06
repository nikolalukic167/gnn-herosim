"""Reconcile retained live task records, source inputs and frozen models."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts_cosim.workflow_live import substrate


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root,cache,models):
    report=json.loads((root/'read.json').read_text());protocol=json.loads((root/'protocol_before_run.json').read_text())
    for name,h in protocol['sources'].items():
        if sha(name)!=h:raise ValueError(f'changed evaluation source: {name}')
    frozen=json.loads((root/'frozen_checkpoints.json').read_text())
    for name,contract in frozen.items():
        if sha(models/f'{name}.pt')!=contract['weights_sha256'] or sha(models/f'{name}.contract.json')!=contract['contract_sha256']:raise ValueError('frozen model changed')
    with np.load(cache/f"{protocol['split']}.npz") as z:cases={int(s):p for s,p in zip(z['seeds'],z['p'])}
    rows=list(root.glob('*/read.json'))
    if len(rows)!=report['cases'] or len(rows)!=len(cases):raise ValueError('incomplete gate')
    seen=set();runs=0;tasks=0;hashes={}
    for path in rows:
        row=json.loads(path.read_text());seed=row['seed']
        if seed in seen:raise ValueError('duplicate environment row')
        seen.add(seed);p=cases[seed];config,inputs=substrate(p/1000);actual=json.loads((path.parent/'input.json').read_text())
        if actual!={'config':config,'inputs':inputs}:raise ValueError('retained actual source differs from held-out problem')
        arms={**row['controls'],**row['learned']};checked=set();trace=path.parent/'placements/placements.jsonl'
        with trace.open() as f:
            for line in f:
                result=json.loads(line);name=result['arm'];stats=result['stats'];records=stats['tasks'];plan=np.array(result['placement_plan'])
                if name in checked or not np.array_equal(plan,arms[name]['assignment']):raise ValueError('duplicate/mismatched live plan')
                checked.add(name)
                if len(records)!=p.shape[0]*p.shape[1] or len(stats['decisions'])!=len(records):raise ValueError('incomplete task execution')
                if abs(sum(t['doneTime']-t['dispatchedTime'] for t in records)-result['rtt'])>1e-8:raise ValueError('task-time sum differs from reported RTT')
                if abs(result['rtt']*1000-arms[name]['cost_ms'])>1e-6:raise ValueError('live plan cost differs')
                operation_ids=set()
                for t in records:
                    j,k=map(int,t['taskType']['name'][2:].split('_'));operation_ids.add((j,k))
                    if abs(t['executionTime']*1000-p[j,k,plan[j,k]])>1e-6:raise ValueError('actual execution duration differs from source')
                if len(operation_ids)!=len(records):raise ValueError('duplicate completed operation')
                for d in stats['decisions']:
                    if d['host']!=plan[d['job'],d['operation']]:raise ValueError('served decision differs from plan')
                runs+=1;tasks+=len(records)
        if checked!=set(row['live_checked']) or not set(frozen).issubset(checked):raise ValueError('missing learned arm live replay')
        for file in (path,path.parent/'input.json',trace):hashes[str(file.relative_to(root))]=sha(file)
    if seen!=set(cases) or runs!=report['live_runs']:raise ValueError('gate coverage mismatch')
    return {'status':'PASS','cases':len(seen),'live_runs':runs,'completed_tasks':tasks,'artifacts_sha256':hashes,'frozen_models':frozen,'evaluation_report_sha256':sha(root/'read.json')}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);args=ap.parse_args();report=audit(args.root,args.cache,args.models)
    out=args.root/'AUDIT.json'
    if out.exists():raise ValueError('preserve previous audit')
    out.write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k not in ('artifacts_sha256','frozen_models')})

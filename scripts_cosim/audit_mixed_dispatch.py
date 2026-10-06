"""Re-read ordering gates, all retained plans, source identities and live coverage."""
import argparse,json
from pathlib import Path
import numpy as np
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import replay
from scripts_cosim.mixed_pair_exploration import verify
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import sha,save
from scripts_cosim.evaluate_workflow import paired_summary

def audit(root):
    pre=json.loads((root/'protocol_before_run.json').read_text());meta=json.loads((root/'source_validation.json').read_text());verify(root,pre,meta)
    engine=DispatchNative(root/'audit_build');old=json.loads((root/'AUDIT.json').read_text());report=json.loads((root/'read.json').read_text());rows=[];plans=runs=tasks=0
    lo,hi=pre['protocol']['fresh_seeds'];expected=set(range(lo,hi+1))
    if {r['seed'] for r in old['rows']}!=expected:raise ValueError('environment coverage mismatch')
    for receipt in old['rows']:
        seed=receipt['seed'];b=load_problem(root/'inputs'/f'{seed}.json');cell=root/'test'/str(seed);row=json.loads((cell/'read.json').read_text());path=cell/'placements/placements.jsonl'
        if sha(path)!=row['placements_sha256']:raise ValueError('plan hash mismatch')
        placements=[json.loads(line) for line in path.read_text().splitlines()]
        if len(placements)!=5:raise ValueError('missing retained control/reference plan')
        for plan in placements:
            a=np.array(plan['placement_plan']);priority=np.array(plan['priority']);c,ends=engine.score_priority(b,a,priority);independent=replay(b,a,priority)
            if c!=plan['objective'] or c!=independent['objective'] or not np.array_equal(ends,independent['ends']):raise ValueError('plan replay mismatch')
            plans+=1
        if row['methods']['reference']['cost']!=min(p['objective'] for p in placements if p['arm']!='srpt'):raise ValueError('incorrect reference selection')
        path=root/'live'/str(seed)/'results.jsonl'
        if sha(path)!=receipt['sha256']:raise ValueError('live hash mismatch')
        observed=set()
        for line in path.read_text().splitlines():
            r=json.loads(line);arm=r['arm'];observed.add(arm);stats=r['stats'];a,priority=np.array(row['methods'][arm]['plan']);independent=replay(b,a,priority)
            if abs(stats['total_rtt']*1000-independent['objective'])>1e-6:raise ValueError('live objective mismatch')
            ids=set()
            for task in stats['tasks']:
                j,k=map(int,task['taskType']['name'][2:].split('_'));ids.add((j,k))
                if task['executionNode']!=f'node{a[j,k]}' or abs(task['doneTime']*1000-independent['ends'][j,k])>1e-6:raise ValueError('live task mismatch')
            if len(ids)!=a.size or len(stats['tasks'])!=a.size:raise ValueError('task coverage failure')
            if stats['mixed_execution']['priority']!=priority.tolist():raise ValueError('priority contract mismatch')
            runs+=1;tasks+=a.size
        if observed!={'control','srpt','reference'}:raise ValueError('arm coverage mismatch')
        rows.append(row)
    gain=paired_summary(np.array([[r['methods']['reference']['cost'] for r in rows]]),np.array([[r['methods']['control']['cost'] for r in rows]]))
    if gain!=report['gain'] or float(np.quantile([r['methods']['control']['time_ms'] for r in rows],.95))!=report['control_p95_ms']:raise ValueError('statistics mismatch')
    if runs!=old['live_runs'] or tasks!=old['completed_operations']:raise ValueError('receipt count mismatch')
    save(root/'INDEPENDENT_AUDIT.json',{'status':'PASS','retained_plans':plans,'live_runs':runs,'completed_operations':tasks,'source_sha256':sha(Path(__file__)),'report_sha256':sha(root/'read.json')});print('PASS',root.name,plans,runs,tasks,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('roots',type=Path,nargs='+');args=ap.parse_args()
    for root in args.roots:audit(root)

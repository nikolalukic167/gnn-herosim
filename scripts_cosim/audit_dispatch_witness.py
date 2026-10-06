"""Independently replay every priority permutation in the small ordering witness."""
import argparse,json
from pathlib import Path
import numpy as np
from src.placement.radical.dispatch_live import replay
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import sha,save
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);root=ap.parse_args().root
    pre=json.loads((root/'protocol_before_run.json').read_text());read=json.loads((root/'read.json').read_text());meta=json.loads((root/'input_validation.json').read_text())
    if any(sha(ROOT/p)!=h for p,h in pre['sources'].items()):raise ValueError('witness source changed')
    plans=runs=tasks=0;flips=0
    for row in read['rows']:
        seed=row['seed'];mins=[]
        for variant in (0,1):
            path=root/f'{seed}_{variant}.json'
            if sha(path)!=meta['inputs'][f'{seed}_{variant}']['sha256']:raise ValueError('witness input changed')
            b=load_problem(path);best=[float('inf'),float('inf')];seen=set()
            for line in (root/str(seed)/f'placements_{variant}.jsonl').read_text().splitlines():
                r=json.loads(line);a=np.array(r['placement_plan']);priority=np.array(r['priority']);key=tuple(priority.ravel())
                if key in seen or sorted(key)!=list(range(6)):raise ValueError('invalid priority coverage')
                seen.add(key);first=int(priority[1,0]<priority[0,0]);actual=replay(b,a,priority)
                if first!=r['first_root'] or actual['objective']!=r['objective']:raise ValueError('independent objective mismatch')
                best[first]=min(best[first],actual['objective']);plans+=1
            if len(seen)!=720:raise ValueError('incomplete priority enumeration')
            if best!=row['conditional_minima'][variant]:raise ValueError('wrong conditional optimum')
            observed=set()
            for line in (root/str(seed)/f'live_{variant}.jsonl').read_text().splitlines():
                r=json.loads(line);first=r['first_root'];observed.add(first);stats=r['stats'];priority=np.array(r['priority']);a=np.array(stats['assignment']);expected=replay(b,a,priority)
                if abs(stats['total_rtt']*1000-best[first])>1e-6:raise ValueError('live optimum mismatch')
                events=stats['mixed_execution']['events'];start=next(e for e in events if e['event']=='start')
                if start['job']!=first:raise ValueError('wrong first execution')
                for task in stats['tasks']:
                    j,k=map(int,task['taskType']['name'][2:].split('_'))
                    if abs(task['doneTime']*1000-expected['ends'][j,k])>1e-6 or task['executionNode']!=f'node{a[j,k]}':raise ValueError('live operation mismatch')
                if len(stats['tasks'])!=6:raise ValueError('missing operation')
                runs+=1;tasks+=6
            if observed!={0,1}:raise ValueError('missing conditional live run')
            mins.append(best)
        flips+=int((mins[0][0]-mins[0][1])*(mins[1][0]-mins[1][1])<0)
    if plans!=read['enumerated_plans'] or runs!=read['live_runs'] or flips!=read['strict_flips']:raise ValueError('summary mismatch')
    save(root/'AUDIT.json',{'status':'PASS','independent_simpy_plan_replays':plans,'live_runs':runs,'operations':tasks,'strict_flips':flips,'report_sha256':sha(root/'read.json'),'audit_source_sha256':sha(Path(__file__))});print('AUDIT PASS',plans,runs,tasks,flips,flush=True)
if __name__=='__main__':main()

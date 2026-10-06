"""Validate the mixed-physics headroom gate in actual HeROsim."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.mixed_physics_live import run
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.live import run as independent
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--tag',default='live');args=ap.parse_args();root=args.root
 if (root/f'{args.tag}_READ.json').exists() or (root/args.tag).exists():raise ValueError('preserve earlier live gate')
 pre=json.loads((root/'protocol_before_run.json').read_text());meta=json.loads((root/'source_validation.json').read_text());sources={p:sha(ROOT/p) for p in ['scripts_cosim/mixed_physics_live_gate.py','scripts_cosim/mixed_physics_live.py','src/placement/radical/coordinator.py','src/placement/infrastructure.py','src/placement/simulation.py','src/placement/executor.py','src/executecosimulation.py']};save(root/f'{args.tag}_sources_before_run.json',sources);configure_environment();rows=[];runs=tasks=0
 for cell in sorted((root/'test').iterdir()):
  seed=int(cell.name);source=root/'inputs'/f'{seed}.json'
  if sha(source)!=meta['inputs'][str(seed)]['sha256'] or any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed')
  b=load_problem(source);r=json.loads((cell/'read.json').read_text());target=root/args.tag/str(seed);target.mkdir(parents=True);arms={**{label:np.array(x['plan']) for label,x in r['methods'].items()},'knative':None};reads={}
  with (target/'simulation.log').open('w') as log,(target/'results.jsonl').open('w') as output:
   for label,a in arms.items():
    live=run(b,a,log,seed);reference=independent(b,np.array(live['assignment']),'mixed')
    if abs(live['total_rtt']*1000-reference['objective'])>1e-6:raise ValueError(f'objective mismatch {seed} {label}')
    if label!='knative' and abs(live['total_rtt']*1000-r['methods'][label]['cost'])>1e-6:raise ValueError('search/live mismatch')
    ids=set()
    for task in live['tasks']:
     j,k=map(int,task['taskType']['name'][2:].split('_'));ids.add((j,k))
     if abs(task['doneTime']*1000-reference['ends'][j][k])>1e-6 or task['executionNode']!=f"node{live['assignment'][j][k]}":raise ValueError('task completion/placement mismatch')
    if len(ids)!=48 or abs(sum(t['doneTime']-t['dispatchedTime'] for t in live['tasks'])-live['total_rtt'])>1e-8:raise ValueError('incomplete task accounting')
    reads[label]=live['total_rtt']*1000;output.write(json.dumps({'arm':label,'stats':live},allow_nan=False)+'\n');runs+=1;tasks+=48
  rows.append({'seed':seed,'costs':reads,'sha256':sha(target/'results.jsonl')});print('HEROSIM',len(rows),'/32',flush=True)
 report={'status':'PASS','cases':len(rows),'live_runs':runs,'completed_tasks':tasks,'all_task_completion_times_match':True,'contract':'mixed_execution_v1','sources':sources,'rows':rows};save(root/f'{args.tag}_READ.json',report);print('PASS',runs,tasks,flush=True)
if __name__=='__main__':main()

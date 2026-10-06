"""Audit actual HeROsim model-gate decisions, physics and frozen checkpoints."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts_cosim.radical_physics_screen import sha,save
from scripts_cosim.audit_radical_physics import load_problem
from src.placement.radical.live import run as independent
from src.placement.radical.environment import pack
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--cache-dir',required=True,type=Path);ap.add_argument('--models',required=True,type=Path);args=ap.parse_args();root=args.root
 if (root/'AUDIT.json').exists():raise ValueError('preserve previous audit')
 report=json.loads((root/'read.json').read_text());pre=json.loads((root/'protocol_before_run.json').read_text());frozen=json.loads((root/'frozen_models_before_test.json').read_text());meta=json.loads((args.cache_dir/'METADATA.json').read_text())
 for p,h in pre['sources'].items():
  if sha(ROOT/p)!=h:raise ValueError('changed serving source')
 if sha(args.cache_dir/'METADATA.json')!=pre['data_metadata_sha256']:raise ValueError('changed dataset')
 for name,r in frozen.items():
  path=args.models/f'{name}.pt'
  if sha(path)!=r['weights_sha256'] or sha(path.with_suffix('.contract.json'))!=r['sidecar_sha256']:raise ValueError('changed checkpoint')
 identities=set();runs=tasks=0;hashes={};rows=[];setup=wait=0.
 for record in (r for r in meta['cases'] if r['split']=='test'):
  seed=record['seed'];cell=root/str(seed);source=args.cache_dir/f'ds_{seed}/problem.json';b=load_problem(source);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
  if identity!=record['identity'] or identity in identities or sha(source)!=record['source_sha256']:raise ValueError('source identity/fingerprint mismatch')
  identities.add(identity);r=json.loads((cell/'read.json').read_text());rows.append(r);arms=set()
  with (cell/'results.jsonl').open() as f:
   for line in f:
    result=json.loads(line);name=result['arm'];s=result['stats'];a=np.array(s['assignment']);expected=independent(b,a,'mixed');cost=r['knative_cost'] if name=='knative' else r['methods'][name]['cost']
    if name in arms or abs(cost-s['total_rtt']*1000)>1e-6 or abs(cost-expected['objective'])>1e-6:raise ValueError('live/reference/summary mismatch')
    arms.add(name);ops=set()
    if len(s['tasks'])!=48 or len(s['decisions'])!=48 or len(s['mixed_execution']['events'])!=96:raise ValueError('task/event coverage')
    executed={(e['job'],e['operation']):e for e in s['mixed_execution']['events'] if e['event']=='done'}
    for t in s['tasks']:
     j,k=map(int,t['taskType']['name'][2:].split('_'));ops.add((j,k));e=executed[j,k]
     if t['executionNode']!=f'node{a[j,k]}' or abs(t['doneTime']*1000-expected['ends'][j][k])>1e-6 or abs(t['executionTime']-(e['base']+e['setup']))>1e-9 or abs(e['base']*1000-b['p'][j,k,a[j,k]])>1e-9:raise ValueError('physical task mismatch')
    if len(ops)!=48 or abs(sum(t['doneTime']-t['dispatchedTime'] for t in s['tasks'])-s['total_rtt'])>1e-8:raise ValueError('RTT sum mismatch')
    for d in s['decisions']:
     if d['host']!=a[d['job'],d['operation']]:raise ValueError('served assignment mismatch')
    setup+=s['mixed_execution']['setup_total'];wait+=s['mixed_execution']['resource_wait_total'];runs+=1;tasks+=48
  if arms!=set(frozen)|{'hand_rule','guard_only','knative'}:raise ValueError('missing arm')
  for name in ('read.json','results.jsonl','placements/placements.jsonl'):hashes[str((cell/name).relative_to(root))]=sha(cell/name)
 if runs!=report['live_runs'] or tasks!=report['completed_tasks']:raise ValueError('aggregate coverage mismatch')
 def matrix(arm):return np.array([[r['methods'][f'{arm}_seed{s}']['cost'] for r in rows] for s in (201,202,203)])
 for name in ('mpoff','mlp_hand'):
  actual=paired_summary(matrix('gnn'),matrix(name))
  if actual!=report['comparisons'][name]:raise ValueError('paired summary mismatch')
 save(root/'AUDIT.json',{'status':'PASS','unique_test_environments':len(identities),'live_runs':runs,'completed_tasks':tasks,'setup_total_s':setup,'resource_wait_total_s':wait,'artifacts_sha256':hashes,'report_sha256':sha(root/'read.json')});print('AUDIT PASS',runs,tasks,flush=True)
if __name__=='__main__':main()

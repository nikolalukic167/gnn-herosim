"""Frozen matched-model gate through real HeROsim mixed-physics execution."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from src.policy.mixed.model import MixedNet,CONTRACT
from src.policy.mixed.train import load_split,serve,guarded
from src.placement.radical.mixed import MixedNative
from src.placement.radical.environment import initial
from scripts_cosim.mixed_physics_live import run
from src.placement.radical.live import run as independent
from scripts_cosim.workflow_proposal_gate import configure_environment
from scripts_cosim.radical_physics_screen import sha,save
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
 if out.exists():raise ValueError('preserve previous gate')
 out.mkdir(parents=True);torch.set_num_threads(1);engine=MixedNative(out/'build');models={};frozen={}
 for arm in ('gnn','mpoff','mlp_hand'):
  for seed in (201,202,203):
   name=f'{arm}_seed{seed}';path=args.models/f'{name}.pt';side=json.loads(path.with_suffix('.contract.json').read_text())
   if side['contract']!=CONTRACT or side['arm']!=arm or side['seed']!=seed or side['architecture']['mp']!=(arm=='gnn') or side['data_metadata_sha256']!=sha(args.cache_dir/'METADATA.json'):raise ValueError('checkpoint contract mismatch')
   for p,h in side['sources'].items():
    if sha(ROOT/p)!=h:raise ValueError('checkpoint source mismatch')
   model=MixedNet(**side['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval();models[name]=(model,arm=='mlp_hand');frozen[name]={'weights_sha256':sha(path),'sidecar_sha256':sha(path.with_suffix('.contract.json')),'selected_epoch':side['selected_epoch']}
 save(out/'frozen_models_before_test.json',frozen);data,problems,*_=load_split(args.cache_dir,'test');sources={p:sha(ROOT/p) for p in ['experiments/mixed_physics_learning_v1_protocol.json','experiments/mixed_physics_learning_v1_decode_ablation.json','scripts_cosim/evaluate_mixed_models.py','scripts_cosim/mixed_physics_live.py','src/placement/radical/coordinator.py','src/placement/infrastructure.py','src/placement/simulation.py','src/policy/mixed/train.py','src/policy/mixed/model.py']};save(out/'protocol_before_run.json',{'sources':sources,'data_metadata_sha256':sha(args.cache_dir/'METADATA.json'),'physics':'mixed_execution_v1','selection':'frozen validation-selected checkpoints','repeats':7,'native':engine.provenance})
 identities=[r['identity'] for r in json.loads((args.cache_dir/'METADATA.json').read_text())['cases']]
 if len(set(identities))!=len(identities):raise ValueError('reused source across splits')
 configure_environment();rows=[];runs=0
 for model,hand in models.values():
  for _ in range(5):serve(model,engine,problems[0],hand)
 for seed,b in zip(data['seeds'],problems):
  if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed before simulation')
  cell=out/str(seed);(cell/'placements').mkdir(parents=True);methods={}
  for name,item in [*models.items(),('hand_rule',None),('guard_only',None)]:
   timings=[];previous=None
   for _ in range(7):
    t=time.perf_counter()
    if name=='hand_rule':c,a=engine.search(b,initial(b,'affinity'),256,True)
    elif name=='guard_only':c,a=guarded(engine,b,initial(b,'fastest'))
    else:c,a=serve(item[0],engine,b,item[1])
    timings.append(1000*(time.perf_counter()-t))
    if previous is not None and (c!=previous[0] or not np.array_equal(a,previous[1])):raise ValueError('nondeterministic serving')
    previous=(c,a)
   methods[name]={'cost':c,'plan':a.tolist(),'time_ms':float(np.median(timings)),'timings_ms':timings}
  with (cell/'simulation.log').open('w') as log,(cell/'results.jsonl').open('w') as f,(cell/'placements/placements.jsonl').open('w') as traces:
   for name,r in methods.items():
    live=run(b,np.array(r['plan']),log,int(seed));ref=independent(b,np.array(r['plan']),'mixed')
    if abs(live['total_rtt']*1000-r['cost'])>1e-6:raise ValueError('served plan/live objective mismatch')
    for task in live['tasks']:
     j,k=map(int,task['taskType']['name'][2:].split('_'))
     if abs(task['doneTime']*1000-ref['ends'][j][k])>1e-6:raise ValueError('individual completion mismatch')
    f.write(json.dumps({'arm':name,'stats':live},allow_nan=False)+'\n');traces.write(json.dumps({'arm':name,'placement_plan':r['plan'],'rtt_ms':r['cost']})+'\n');runs+=1
   kn=run(b,None,log,int(seed));knref=independent(b,np.array(kn['assignment']),'mixed')
   if abs(kn['total_rtt']*1000-knref['objective'])>1e-6:raise ValueError('Knative replay mismatch')
   f.write(json.dumps({'arm':'knative','stats':kn},allow_nan=False)+'\n');runs+=1
  row={'seed':int(seed),'methods':methods,'knative_cost':kn['total_rtt']*1000};rows.append(row);save(cell/'read.json',row);print('MIXED LIVE',len(rows),'/64',flush=True)
 def values(arm):return np.array([[r['methods'][f'{arm}_seed{s}']['cost'] for r in rows] for s in (201,202,203)])
 gnn=values('gnn');rule=np.array([[r['methods']['hand_rule']['cost'] for r in rows]]);kn=np.array([[r['knative_cost'] for r in rows]]);comparisons={'mpoff':paired_summary(gnn,values('mpoff')),'mlp_hand':paired_summary(gnn,values('mlp_hand')),'hand_rule':paired_summary(gnn,np.repeat(rule,3,axis=0)),'knative':paired_summary(gnn,np.repeat(kn,3,axis=0)),'guard_only':paired_summary(gnn,np.repeat(np.array([[r['methods']['guard_only']['cost'] for r in rows]]),3,axis=0))};timing={name:{'median_ms':float(np.median([r['methods'][name]['time_ms'] for r in rows])),'p95_ms':float(np.quantile([r['methods'][name]['time_ms'] for r in rows],.95))} for name in rows[0]['methods']};passes=all(comparisons[k]['median_gain_pct']>=5 and comparisons[k]['hierarchical_median_ci95_pct'][0]>0 for k in ('mpoff','mlp_hand','hand_rule')) and all(timing[f'gnn_seed{s}']['p95_ms']<=2 for s in (201,202,203));report={'verdict':'PILOT_GNN_WIN' if passes else 'NO_CONTROL_SURVIVING_GNN_WIN','cases':len(rows),'live_runs':runs,'completed_tasks':runs*48,'comparisons':comparisons,'timing':timing,'mean_rtt_ms':{name:float(np.mean([r['methods'][name]['cost'] for r in rows])) for name in timing},'frozen':frozen,'sources':sources};save(out/'read.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('frozen','sources')},indent=2),flush=True)
if __name__=='__main__':main()

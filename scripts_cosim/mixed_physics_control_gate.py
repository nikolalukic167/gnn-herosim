"""Budget challenge for a specialized exact setup/lock/domain scorer."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,initial,serialized,pack
from src.placement.radical.mixed import MixedNative
from src.placement.workflow_exact import ExactWorkflow
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
def plan(engine,old,b,method):
 start,search=method.split(':');a=old.greedy(b['p'])[1] if start=='ect' else initial(b,start)
 if search=='direct':return engine.score(b,a)[0],a
 anneal=search.startswith('anneal');steps=int(search[6:] if anneal else search[7:]);return engine.search(b,a,steps,anneal)
def timed(fn,repeats):
 times=[];previous=None
 for _ in range(repeats):
  t=time.perf_counter();cost,a=fn();times.append(1000*(time.perf_counter()-t))
  if previous is not None and (cost!=previous[0] or not np.array_equal(a,previous[1])):raise ValueError('nondeterministic control')
  previous=(cost,a)
 return {'cost':cost,'plan':a.tolist(),'time_ms':float(np.median(times)),'timings_ms':times}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out
 if out.exists():raise ValueError('preserve previous evidence')
 out.mkdir(parents=True);protocol=json.loads((ROOT/'experiments/mixed_physics_v1_gate.json').read_text());engine=MixedNative(out/'build');old=ExactWorkflow(out/'old_build');sources={p:sha(ROOT/p) for p in ['experiments/mixed_physics_v1_gate.json','scripts_cosim/mixed_physics_control_gate.py','src/placement/radical/mixed.py','src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','src/placement/workflow_exact.py','src/placement/native/workflow.cpp']};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':engine.provenance})
 cases={};ids=set();metadata={};(out/'inputs').mkdir()
 priorids=set()
 for name in ('radical_physics_v1','radical_physics_v1_confirmation'):
  d=json.loads((out.parent/name/'source_validation.json').read_text());priorids.update(r['identity'] for r in d['inputs'].values())
 for key in ('calibration_seeds','fresh_seeds'):
  lo,hi=protocol[key]
  for seed in range(lo,hi+1):
   b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
   if identity in ids or identity in priorids:raise ValueError('reused physical problem')
   ids.add(identity);cases[seed]=b;p=out/'inputs'/f'{seed}.json';save(p,serialized(b));metadata[str(seed)]={'identity':identity,'sha256':sha(p)}
 save(out/'source_validation.json',{'status':'PASS','inputs':metadata,'unique_new':len(ids)})
 methods=[f'{a}:{s}' for a in protocol['starts'] for s in protocol['searches']];rows=[];lo,hi=protocol['calibration_seeds'];repeats=protocol['timing_repeats']
 for seed in range(lo,hi+1):
  b=cases[seed];rows.append({'seed':seed,'methods':{m:timed(lambda m=m:plan(engine,old,b,m),repeats) for m in methods}})
 metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in rows])),'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in rows],.95))} for m in methods};eligible=[m for m in methods if metrics[m]['p95_ms']<=protocol['budget_ms']]
 if not eligible:raise RuntimeError('no feasible hand control')
 selected=min(eligible,key=lambda m:metrics[m]['mean_cost']);save(out/'calibration.json',{'selected':selected,'metrics':metrics,'rows':rows});print('FROZEN',selected,metrics[selected],flush=True)
 rows=[];lo,hi=protocol['fresh_seeds']
 for seed in range(lo,hi+1):
  b=cases[seed];cell=out/'test'/str(seed);(cell/'placements').mkdir(parents=True);results={'control':timed(lambda:plan(engine,old,b,selected),repeats)}
  for start in ('fastest','load'):results[f'reference_{start}']=timed(lambda:plan(engine,old,b,f'{start}:anneal8192'),repeats)
  reference=min(results,key=lambda m:results[m]['cost']);r={'seed':seed,'methods':results,'reference':reference};save(cell/'read.json',r)
  with (cell/'placements/placements.jsonl').open('w') as f:
   for label,d in results.items():f.write(json.dumps({'arm':label,'placement_plan':d['plan'],'objective':d['cost']})+'\n')
  rows.append(r)
 gain=paired_summary(np.array([[r['methods'][r['reference']]['cost'] for r in rows]]),np.array([[r['methods']['control']['cost'] for r in rows]]));p95=float(np.quantile([r['methods']['control']['time_ms'] for r in rows],.95));passed=gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and p95<=2
 if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('changed scoring source')
 report={'selected':selected,'gain':gain,'control_p95_ms':p95,'headroom_passes':passed,'live_gate_pending':True,'cases':len(rows),'sources':sources};save(out/'read.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

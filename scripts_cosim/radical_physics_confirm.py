"""Fresh challenge using 24 deployable control combinations and a neutral arm."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from src.placement.radical.environment import Native,problem,initial,serialized,pack,FLAGS
from src.placement.radical.live import run
from src.placement.workflow_exact import ExactWorkflow
from scripts_cosim.radical_physics_screen import sha,save
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]

def plan(native,old,b,mechanism,start,search):
    a=old.greedy(b['p'])[1] if start=='ect' else initial(b,start)
    if search=='direct':return native.score(b,a,mechanism)[0],a
    q,a,dims=native.checked(b,a);anneal=search.startswith('anneal');steps=int(search[6:] if anneal else search[7:]);cost=native.lib.radical_search(q,a,*dims,FLAGS[mechanism],steps,int(anneal),77)
    if not np.isfinite(cost):raise RuntimeError('invalid search')
    return float(cost),a

def timed(fn):
    times=[];prior=None
    for _ in range(3):
        t=time.perf_counter();c,a=fn();times.append(1000*(time.perf_counter()-t))
        if prior is not None and (prior[0]!=c or not np.array_equal(prior[1],a)):raise RuntimeError('nondeterminism')
        prior=(c,a)
    return {'cost':c,'plan':a.tolist(),'time_ms':float(np.median(times)),'timings_ms':times}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve previous run')
    out.mkdir(parents=True);protocol=json.loads((ROOT/'experiments/radical_physics_v1_confirmation.json').read_text());native=Native(out/'build');old=ExactWorkflow(out/'old_build');sources={p:sha(ROOT/p) for p in ['experiments/radical_physics_v1_confirmation.json','scripts_cosim/radical_physics_confirm.py','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','src/placement/radical/live.py','src/placement/workflow_exact.py','src/placement/native/workflow.cpp']};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':native.provenance,'old_native':old.provenance})
    cases={};identities=set();files={};(out/'inputs').mkdir()
    for key in ('calibration_seeds','fresh_seeds'):
        lo,hi=protocol[key]
        for seed in range(lo,hi+1):
            b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in identities:raise ValueError('duplicate actual source')
            identities.add(identity);cases[seed]=b;path=out/'inputs'/f'{seed}.json';save(path,serialized(b));files[str(seed)]={'identity':identity,'sha256':sha(path)}
    prior=json.loads((out.parent/'radical_physics_v1/source_validation.json').read_text())['inputs'];oldids={v['identity'] for v in prior.values()}
    lo,hi=protocol['fresh_seeds']
    if any(files[str(seed)]['identity'] in oldids for seed in range(lo,hi+1)):raise ValueError('reused fresh source')
    save(out/'source_validation.json',{'status':'PASS','inputs':files,'unique_fresh':hi-lo+1})
    methods=[f'{start}:{search}' for start in protocol['starts'] for search in protocol['searches']];selected={}
    for mechanism in protocol['mechanisms']:
        directory=out/mechanism;directory.mkdir();rows=[];lo,hi=protocol['calibration_seeds']
        for seed in range(lo,hi+1):
            b=cases[seed];r={}
            for method in methods:
                start,search=method.split(':');r[method]=timed(lambda:plan(native,old,b,mechanism,start,search))
            rows.append({'seed':seed,'methods':r})
        metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in rows])),'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in rows],.95))} for m in methods};eligible=[m for m in methods if metrics[m]['p95_ms']<=protocol['budget_ms']]
        if not eligible:raise RuntimeError('no feasible control')
        selected[mechanism]=min(eligible,key=lambda m:metrics[m]['mean_cost']);save(directory/'calibration.json',{'selected':selected[mechanism],'metrics':metrics,'rows':rows});print('FROZEN',mechanism,selected[mechanism],metrics[selected[mechanism]],flush=True)
    save(out/'frozen_selectors.json',selected);summary={};count=0
    for mechanism,method in selected.items():
        start,search=method.split(':');rows=[];directory=out/mechanism;lo,hi=protocol['fresh_seeds']
        for seed in range(lo,hi+1):
            if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed')
            if sha(out/'inputs'/f'{seed}.json')!=files[str(seed)]['sha256']:raise ValueError('input changed')
            b=cases[seed];cell=directory/str(seed);(cell/'placements').mkdir(parents=True);results={'control':timed(lambda:plan(native,old,b,mechanism,start,search))}
            for init in ('load','fastest'):results[f'reference_{init}']=timed(lambda:plan(native,old,b,mechanism,init,'anneal8192'))
            reference=min(results,key=lambda m:results[m]['cost']);live={}
            for label in ('control','reference_load','reference_fastest'):
                a=np.array(results[label]['plan']);live[label]=run(b,a,mechanism);count+=1;cost,ends=native.score(b,a,mechanism)
                if abs(live[label]['objective']-results[label]['cost'])>1e-6 or not np.allclose(live[label]['ends'],ends,atol=1e-7,rtol=0):raise RuntimeError('native/independent replay mismatch')
            save(cell/'live.json',live)
            with (cell/'placements/placements.jsonl').open('w') as f:
                for label,r in results.items():f.write(json.dumps({'arm':label,'placement_plan':r['plan'],'objective':r['cost']})+'\n')
            row={'seed':seed,'methods':results,'reference':reference};save(cell/'read.json',row);rows.append(row)
        gain=paired_summary(np.array([[r['methods'][r['reference']]['cost'] for r in rows]]),np.array([[r['methods']['control']['cost'] for r in rows]]));p95=float(np.quantile([r['methods']['control']['time_ms'] for r in rows],.95));summary[mechanism]={'selected':method,'gain':gain,'control_p95_ms':p95,'passes':gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and p95<=2,'cases':len(rows)};save(directory/'read.json',summary[mechanism]);print('CONFIRMED',mechanism,json.dumps(summary[mechanism]),flush=True)
    save(out/'read.json',{'results':summary,'simpy_runs':count,'scope':'independent sandbox, not production HeROsim or GNN result','sources':sources})
if __name__=='__main__':main()

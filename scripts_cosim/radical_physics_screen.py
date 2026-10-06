"""Calibrate strong hand controls and screen eight isolated physics mechanisms."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from src.placement.radical.environment import Native,problem,serialized,pack,FLAGS
from src.placement.radical.live import run
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def timed(native,b,mechanism,method):
    times=[];expected=None
    for _ in range(3):
        start=time.perf_counter();c,a=native.plan(b,mechanism,method);times.append((time.perf_counter()-start)*1000)
        if expected is not None and (expected[0]!=c or not np.array_equal(expected[1],a)):raise RuntimeError('nondeterministic planner')
        expected=(c,a)
    return {'cost':c,'plan':a.tolist(),'time_ms':float(np.median(times)),'timings_ms':times}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve earlier experiment')
    out.mkdir(parents=True);protocol=json.loads((ROOT/'experiments/radical_physics_v1.json').read_text());native=Native(out/'build')
    paths=['experiments/radical_physics_v1.json','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','src/placement/radical/live.py','scripts_cosim/radical_physics_screen.py']
    sources={p:sha(ROOT/p) for p in paths};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':native.provenance})
    seeds=[]
    for key in ('calibration_seeds','screen_seeds'):
        lo,hi=protocol[key];seeds+=list(range(lo,hi+1))
    cases={s:problem(s,*protocol['shape']) for s in seeds};ids=set();files={};(out/'inputs').mkdir()
    for seed,b in cases.items():
        path=out/'inputs'/f'{seed}.json';save(path,serialized(b));identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
        if identity in ids:raise ValueError('duplicate actual environment')
        ids.add(identity);files[str(seed)]={'file_sha256':sha(path),'identity':identity}
    save(out/'source_validation.json',{'status':'PASS','unique_environments':len(ids),'paired_across_mechanisms':True,'inputs':files})
    summary={};runs=0;calibrations={}
    for mechanism in protocol['mechanisms']:
        directory=out/mechanism;directory.mkdir();cal=[];lo,hi=protocol['calibration_seeds']
        for seed in range(lo,hi+1):cal.append({'seed':seed,'methods':{m:timed(native,cases[seed],mechanism,m) for m in protocol['controls']}})
        metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in cal])),'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in cal],.95))} for m in protocol['controls']}
        eligible=[m for m in metrics if metrics[m]['p95_ms']<=protocol['planning_budget_ms']]
        if not eligible:raise RuntimeError('no control fits budget')
        selected=min(eligible,key=lambda m:metrics[m]['mean_cost']);save(directory/'calibration.json',{'selected':selected,'metrics':metrics,'rows':cal});calibrations[mechanism]=selected
        print('CALIBRATED',mechanism,selected,metrics,flush=True)
    # All selectors are fixed before opening any screen outcomes.
    save(out/'frozen_selectors.json',calibrations)
    for mechanism,selected in calibrations.items():
        directory=out/mechanism;rows=[];lo,hi=protocol['screen_seeds']
        for seed in range(lo,hi+1):
            for path,h in sources.items():
                if sha(ROOT/path)!=h:raise ValueError('changed source')
            if sha(out/'inputs'/f'{seed}.json')!=files[str(seed)]['file_sha256']:raise ValueError('changed input')
            b=cases[seed];cell=directory/str(seed);(cell/'placements').mkdir(parents=True)
            results={m:timed(native,b,mechanism,m) for m in protocol['controls']};results['anneal8192']=timed(native,b,mechanism,'anneal8192');teacher=min(results,key=lambda m:results[m]['cost'])
            with (cell/'placements/placements.jsonl').open('w') as f:
                for method,r in results.items():f.write(json.dumps({'method':method,'placement_plan':r['plan'],'objective':r['cost']})+'\n')
            live={}
            for label,method in (('control',selected),('reference',teacher),('fastest','fastest')):
                r=results[method];a=np.array(r['plan']);score,ends=native.score(b,a,mechanism);stats=run(b,a,mechanism);runs+=1
                if abs(stats['objective']-r['cost'])>1e-6 or not np.allclose(ends,stats['ends'],atol=1e-7,rtol=0):raise RuntimeError('native/SimPy mismatch')
                stats['method']=method;live[label]=stats
            # Turning the mechanism off is a counterfactual on the same placements, not a reoptimized oracle.
            off={label:run(b,np.array(results[method]['plan']),'none')['objective'] for label,method in (('control',selected),('reference',teacher))};runs+=2
            save(cell/'live.json',live);row={'seed':seed,'selected':selected,'teacher_method':teacher,'methods':results,'mechanism_off_same_plans':off};save(cell/'read.json',row);rows.append(row)
        control=np.array([[r['methods'][selected]['cost'] for r in rows]]);reference=np.array([[r['methods'][r['teacher_method']]['cost'] for r in rows]]);fast=np.array([[r['methods']['fastest']['cost'] for r in rows]])
        gain=paired_summary(reference,control);p95=float(np.quantile([r['methods'][selected]['time_ms'] for r in rows],.95));shortlist=gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and p95<=protocol['planning_budget_ms']
        summary[mechanism]={'selected_control':selected,'reference_gain':gain,'control_gain_vs_fastest':paired_summary(control,fast),'control_p95_ms':p95,'reference_anneal8192_p95_ms':float(np.quantile([r['methods']['anneal8192']['time_ms'] for r in rows],.95)),'shortlist':shortlist,'cases':len(rows)}
        save(directory/'read.json',summary[mechanism]);print('SCREEN',mechanism,json.dumps(summary[mechanism]),flush=True)
    save(out/'read.json',{'scope':'exploratory sandbox; independent SimPy validation, not main HeROsim integration','unique_screen_environments':16,'mechanisms':summary,'simpy_replays':runs,'training_runs':0,'sources':sources})
if __name__=='__main__':main()

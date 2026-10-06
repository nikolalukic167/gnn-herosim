"""Calibrated initializer selection, held-out headroom and mandatory live replay."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from src.placement.workflow_basins import BasinPortfolio
from src.placement.workflow_planning import instance,evaluate
from scripts_cosim.workflow_move_value_gate import timed,sha
from scripts_cosim.validate_workflow_data import validate
from scripts_cosim.workflow_proposal_gate import configure_environment
from scripts_cosim.workflow_live import run,substrate
from scripts_cosim.workflow_knative import run as knative_run
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'simulation_data/gnn_environment_search_v1'
def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def portfolio(engine,p,observer=None):
    plans,_=engine.starts(p)
    outcomes=[engine.refine(p,a,observer=observer) for a in plans]
    return min(outcomes,key=lambda x:x[0]),outcomes

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve existing evidence')
    out.mkdir(parents=True);protocol=json.loads((ROOT/'experiments/workflow_basin_v1.json').read_text());engine=BasinPortfolio(out/'build')
    names=['experiments/workflow_basin_v1.json','src/placement/workflow_basins.py','src/placement/native/workflow_starts.cpp','src/placement/native/workflow.cpp','src/placement/workflow_exact.py','src/placement/workflow_planning.py','scripts_cosim/workflow_basin_gate.py','scripts_cosim/workflow_live.py','scripts_cosim/workflow_knative.py','scripts_cosim/workflow_proposal_gate.py','src/placement/executor.py','src/placement/infrastructure.py','src/policy/knative_network/scheduler.py']
    sources={n:sha(ROOT/n) for n in names};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':engine.provenance})
    save(out/'parent_validation.json',validate(BASE/'workflow_corpus'))
    data=np.load(BASE/'workflow_corpus/validation.npz');problems={int(s):p for s,p in zip(data['seeds'],data['p'])};repeats=protocol['timing_repeats']
    lo,hi=protocol['calibration_seeds'];cal=[]
    for seed in range(lo,hi+1):
        p=problems[seed];_,results=portfolio(engine,p);cal.append({'seed':seed,'final_costs':[r[0] for r in results]})
    fixed=int(np.argmin(np.mean([r['final_costs'] for r in cal],axis=0)));rules=['fixed','min_initial','top2','ect']
    for row in cal:
        p=problems[row['seed']];row['methods']={rule:timed(lambda rule=rule:engine.select(p,rule,fixed),repeats) for rule in rules}
    metrics={rule:{'mean_rtt_ms':float(np.mean([r['methods'][rule]['cost_ms'] for r in cal])),'p95_ms':float(np.quantile([r['methods'][rule]['planning_ms'] for r in cal],.95))} for rule in rules}
    eligible=[r for r in rules if metrics[r]['p95_ms']<=protocol['budget_ms']]
    save(out/'calibration_candidates.json',{'fixed_index':fixed,'metrics':metrics,'rows':cal})
    if not eligible:raise RuntimeError('no hand selector meets registered budget')
    selected=min(eligible,key=lambda r:metrics[r]['mean_rtt_ms']);save(out/'calibration.json',{'fixed_index':fixed,'selected_rule':selected,'metrics':metrics,'rows':cal});print('FROZEN',selected,'fixed',fixed,metrics,flush=True)
    lo,hi=protocol['qualification_seeds'];qualification=[]
    for seed in range(lo,hi+1):
        p=problems[seed];oracle=timed(lambda:portfolio(engine,p)[0],repeats);hand=timed(lambda:engine.select(p,selected,fixed),repeats)
        qualification.append({'seed':seed,'oracle':oracle,'hand':hand})
    gain=paired_summary(np.array([[r['oracle']['cost_ms'] for r in qualification]]),np.array([[r['hand']['cost_ms'] for r in qualification]]))
    save(out/'qualification.json',{'gain':gain,'rows':qualification});print('HEADROOM',gain,flush=True)
    old_ids=set()
    for source in (BASE/'workflow_corpus').glob('ds_*/problem.json'):
        p=np.array(json.loads(source.read_text())['processing_ms'],dtype=float);p[np.isnan(p)]=np.inf;identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in old_ids:raise ValueError('duplicate actual parent source')
        old_ids.add(identity)
    for name in ('workflow_proposal_v1_retry','workflow_move_value_v1'):
        old_ids.update(json.loads((BASE/name/'source_validation.json').read_text())['new_ids'])
    lo,hi=protocol['fresh_live_seeds'];fresh={s:instance(s) for s in range(lo,hi+1)};ids=set()
    for p in fresh.values():
        identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in old_ids or identity in ids:raise ValueError('duplicate actual fresh infrastructure')
        ids.add(identity)
    save(out/'source_validation.json',{'status':'PASS','old_unique':len(old_ids),'new_unique':len(ids),'new_ids':sorted(ids)})
    configure_environment();test=[];live_runs=0
    for seed,p in fresh.items():
        for name,h in sources.items():
            if sha(ROOT/name)!=h:raise ValueError('source changed before simulation')
        cell=out/'test'/str(seed);(cell/'placements').mkdir(parents=True);config,inputs=substrate(p/1000);save(cell/'input.json',{'config':config,'inputs':inputs})
        records={'hand':timed(lambda:engine.select(p,selected,fixed),repeats),'exact_descent':timed(lambda:engine.select(p,'ect'),repeats),'oracle':timed(lambda:portfolio(engine,p)[0],repeats)}
        traces={}
        def observer(a,c):traces[np.asarray(a,dtype=np.int16).tobytes()]=(a.copy(),float(c))
        best,results=portfolio(engine,p,observer)
        if abs(best[0]-records['oracle']['cost_ms'])>1e-7:raise RuntimeError('tracing changes result')
        with (cell/'placements/placements.jsonl').open('w') as f:
            for a,c in traces.values():f.write(json.dumps({'placement_plan':a.tolist(),'rtt_ms':c})+'\n')
        with (cell/'simulation.log').open('w') as log,(cell/'live.jsonl').open('w') as f:
            for name,r in records.items():
                stats=run(p/1000,np.array(r['assignment']),log,seed);live_runs+=1
                if abs(stats['total_rtt']*1000-r['cost_ms'])>1e-6:raise RuntimeError('native/live mismatch')
                f.write(json.dumps({'arm':name,'placement_plan':r['assignment'],'stats':stats},allow_nan=False)+'\n')
            kn=knative_run(p/1000,log,seed);live_runs+=1
            if abs(evaluate(p,np.array(kn['assignment']))-kn['total_rtt']*1000)>1e-6:raise RuntimeError('Knative parity mismatch')
            f.write(json.dumps({'arm':'knative_network','placement_plan':kn['assignment'],'stats':kn},allow_nan=False)+'\n')
        row={'seed':seed,'methods':records,'knative_rtt_ms':kn['total_rtt']*1000,'portfolio_final_costs':[r[0] for r in results],'unique_scored_plans':len(traces)};save(cell/'read.json',row);test.append(row);print('live',len(test),'/32',flush=True)
    def compare(a,b):return paired_summary(np.array([[r['methods'][a]['cost_ms'] for r in test]]),np.array([[r['knative_rtt_ms'] if b=='knative' else r['methods'][b]['cost_ms'] for r in test]]))
    timing={name:{'median_ms':float(np.median([r['methods'][name]['planning_ms'] for r in test])),'p95_ms':float(np.quantile([r['methods'][name]['planning_ms'] for r in test],.95))} for name in records}
    report={'cases':len(test),'live_runs':live_runs,'fixed_index':fixed,'selected_rule':selected,'qualification_gain':gain,'live_oracle_gain_vs_hand':compare('oracle','hand'),'live_hand_gain_vs_knative':compare('hand','knative'),'timing':timing,'unique_scored_plans':sum(r['unique_scored_plans'] for r in test),'sources':sources,'native':engine.provenance}
    save(out/'read.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('sources','native')},indent=2),flush=True)
if __name__=='__main__':main()

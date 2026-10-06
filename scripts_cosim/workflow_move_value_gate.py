"""Exact move-value preflight followed by a mandatory fresh HeROsim gate."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from src.placement.workflow_exact import ExactWorkflow
from src.placement.workflow_planning import instance,evaluate,initial,rollout
from src.policy.workflow.model import WorkflowAssignmentNet
from src.policy.workflow.train import load_split
from scripts_cosim.workflow_proposal_gate import probabilities,configure_environment
from scripts_cosim.workflow_live import run,substrate
from scripts_cosim.workflow_knative import run as knative_run
from scripts_cosim.evaluate_workflow import paired_summary

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn,repeats):
    times=[];expected=None
    for _ in range(repeats):
        start=time.perf_counter();result=fn();times.append(1000*(time.perf_counter()-start))
        if expected is not None and (result[0]!=expected[0] or not np.array_equal(result[1],expected[1])):raise RuntimeError('nondeterministic native search')
        expected=result
    return {'cost_ms':float(result[0]),'assignment':result[1].tolist(),'planning_ms':float(np.median(times)),'timings_ms':times}


def compare_scalar(exact,p,a):
    costs,alternatives=exact.rank(p,a)
    for i,cost in enumerate(costs):
        b=a.copy();b.ravel()[i]=alternatives.ravel()[i]
        if abs(evaluate(p,b)-cost)>1e-7:raise RuntimeError('compiled/scalar move cost mismatch')


def guarded(exact,model,p):
    cost,a=exact.greedy(p);b=probabilities(model,p,False).argmax(-1);other=evaluate(p,b)
    return (other,b) if other<cost else (cost,a)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('preserve previous experiment')
    args.out.mkdir(parents=True);torch.set_num_threads(1)
    protocol=json.loads((ROOT/'experiments/workflow_move_value_v1.json').read_text());exact=ExactWorkflow(args.out/'build')
    names=['experiments/workflow_move_value_v1.json','src/placement/native/workflow.cpp','src/placement/workflow_exact.py','src/placement/workflow_planning.py','scripts_cosim/workflow_move_value_gate.py','scripts_cosim/workflow_live.py','scripts_cosim/workflow_knative.py','scripts_cosim/workflow_proposal_gate.py','scripts_cosim/evaluate_workflow.py','src/policy/workflow/model.py','src/placement/infrastructure.py','src/placement/executor.py','src/policy/knative_network/scheduler.py']
    sources={n:sha(ROOT/n) for n in names};frozen={};models={};weights=ROOT/'models/workflow_amortized_v1'
    for seed in (101,102,103):
        name=f'gnn_seed{seed}';path=weights/f'{name}.pt';side=json.loads(path.with_suffix('.contract.json').read_text())
        for source,h in side['sources'].items():
            if sha(ROOT/source)!=h:raise ValueError('frozen model source mismatch')
        if side['arm']!='gnn' or not side['architecture']['mp']:raise ValueError('not an MP-on checkpoint')
        model=WorkflowAssignmentNet(**side['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();models[name]=model
        frozen[name]={'weights_sha256':sha(path),'contract_sha256':sha(path.with_suffix('.contract.json'))}
    (args.out/'protocol_before_run.json').write_text(json.dumps({'protocol':protocol,'sources':sources,'native':exact.provenance,'frozen_checkpoints':frozen},indent=2)+'\n')
    cache=ROOT/'simulation_data/gnn_environment_search_v1/workflow_corpus';val,_,_,_=load_split(cache,'validation');lo,hi=protocol['validation_seeds'];rows=[]
    for i,seed in enumerate(val['seeds']):
        if not lo<=seed<=hi:continue
        p=val['p'][i];r=timed(lambda:exact.descent(p,1),protocol['timing_repeats']);_,a=exact.greedy(p);compare_scalar(exact,p,a)
        if abs(evaluate(p,np.array(r['assignment']))-r['cost_ms'])>1e-7:raise RuntimeError('one-move score mismatch')
        rows.append({'seed':int(seed),**r})
    if len(rows)!=hi-lo+1:raise ValueError('incomplete validation split')
    p95=float(np.quantile([r['planning_ms'] for r in rows],.95));stop=p95<=protocol['budget_ms']
    (args.out/'preflight.json').write_text(json.dumps({'rows':rows,'p95_ms':p95,'stop_move_value_training':stop},indent=2)+'\n');print('EXACT_PREFLIGHT',p95,'ms','STOP_TRAINING',stop,flush=True)
    if not stop:raise RuntimeError('exact preflight leaves a learning opening; follow registered training branch before any test evaluation')
    # Verify actual new problem identities before the first fresh simulation.
    old_ids=set()
    for source in cache.glob('ds_*/problem.json'):
        p=np.array(json.loads(source.read_text())['processing_ms'],dtype=float);p[np.isnan(p)]=np.inf;old_ids.add(hashlib.sha256(p.tobytes()).hexdigest())
    prior=ROOT/'simulation_data/gnn_environment_search_v1/workflow_proposal_v1_retry/source_validation.json'
    old_ids.update(json.loads(prior.read_text())['new_ids'])
    lo,hi=protocol['fresh_live_seeds'];problems={s:instance(s) for s in range(lo,hi+1)};identities=set()
    for p in problems.values():
        identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in old_ids or identity in identities:raise ValueError('duplicate/reused actual problem')
        identities.add(identity)
    (args.out/'source_validation.json').write_text(json.dumps({'status':'PASS','old_unique':len(old_ids),'new_unique':len(identities),'new_ids':sorted(identities)},indent=2)+'\n')
    for model in models.values():
        for _ in range(5):probabilities(model,val['p'][0],False)
    configure_environment();test=[];live_runs=0
    for seed,p in problems.items():
        for name,h in sources.items():
            if sha(ROOT/name)!=h:raise ValueError('source changed before live run')
        cell=args.out/'test'/str(seed);(cell/'placements').mkdir(parents=True)
        config,inputs=substrate(p/1000);(cell/'input.json').write_text(json.dumps({'config':config,'inputs':inputs},allow_nan=False)+'\n')
        records={};repeats=protocol['timing_repeats']
        records['ect']=timed(lambda:exact.greedy(p),repeats)
        records['exact_one_move']=timed(lambda:exact.descent(p,1),repeats)
        records['exact_descent']=timed(lambda:exact.descent(p,64),repeats)
        for name,model in models.items():records[name]=timed(lambda model=model:guarded(exact,model,p),repeats)
        traces={}
        def observer(a,c):traces[np.asarray(a,dtype=np.int16).tobytes()]=(a.copy(),float(c))
        traced=exact.descent(p,64,observer)
        if abs(traced[0]-records['exact_descent']['cost_ms'])>1e-7:raise RuntimeError('tracing changes exact descent')
        compare_scalar(exact,p,np.array(records['ect']['assignment']));compare_scalar(exact,p,np.array(records['exact_descent']['assignment']))
        with (cell/'placements/placements.jsonl').open('w') as out:
            for a,c in traces.values():out.write(json.dumps({'placement_plan':a.tolist(),'rtt_ms':c})+'\n')
        with (cell/'simulation.log').open('w') as log,(cell/'live.jsonl').open('w') as out:
            for name,r in records.items():
                stats=run(p/1000,np.array(r['assignment']),log,seed);live_runs+=1
                if abs(stats['total_rtt']*1000-r['cost_ms'])>1e-6:raise RuntimeError('native/model cost disagrees with HeROsim')
                out.write(json.dumps({'arm':name,'placement_plan':r['assignment'],'stats':stats},allow_nan=False)+'\n')
            kn=knative_run(p/1000,log,seed);live_runs+=1
            if abs(evaluate(p,np.array(kn['assignment']))-kn['total_rtt']*1000)>1e-6:raise RuntimeError('actual Knative plan parity mismatch')
            out.write(json.dumps({'arm':'knative_network','placement_plan':kn['assignment'],'stats':kn},allow_nan=False)+'\n')
        row={'seed':seed,'methods':records,'knative_rtt_ms':kn['total_rtt']*1000,'descent_rounds':traced[2],'unique_scored_plans':len(traces)};test.append(row);(cell/'read.json').write_text(json.dumps(row,indent=2)+'\n');print('live',len(test),'/32',flush=True)
    gnn=np.array([[r['methods'][name]['cost_ms'] for r in test] for name in models]);native=np.array([r['methods']['exact_descent']['cost_ms'] for r in test]);kn=np.array([r['knative_rtt_ms'] for r in test]);ect=np.array([r['methods']['ect']['cost_ms'] for r in test])
    timing={name:{'median_ms':float(np.median([r['methods'][name]['planning_ms'] for r in test])),'p95_ms':float(np.quantile([r['methods'][name]['planning_ms'] for r in test],.95))} for name in test[0]['methods']}
    report={'verdict':'EXACT_TARGET_AFFORDABLE_NO_MOVE_VALUE_TRAINING','cases':len(test),'live_runs':live_runs,'preflight_p95_ms':p95,'timing':timing,'native_gain_vs_gnn':paired_summary(np.tile(native,(3,1)),gnn),'native_gain_vs_knative':paired_summary(native[None],kn[None]),'native_gain_vs_ect':paired_summary(native[None],ect[None]),'mean_rtt_ms':{**{k:float(np.mean([r['methods'][k]['cost_ms'] for r in test])) for k in timing},'knative_network':float(kn.mean())},'max_descent_rounds':max(r['descent_rounds'] for r in test),'unique_scored_plans':sum(r['unique_scored_plans'] for r in test),'sources':sources,'native':exact.provenance,'frozen_checkpoints':frozen}
    (args.out/'read.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('sources','native','frozen_checkpoints')},indent=2))
if __name__=='__main__':main()

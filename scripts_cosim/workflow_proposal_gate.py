"""Validation-selected proposal search, fresh live gate, and actual Knative control."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from src.policy.workflow.model import WorkflowAssignmentNet,features,CONTRACT
from src.policy.workflow.train import load_split
from src.placement.workflow_planning import instance,evaluate,plan_observer
from src.placement.workflow_proposals import improve,hand_probabilities
from scripts_cosim.workflow_qualification import methods
from scripts_cosim.workflow_live import run,substrate
from scripts_cosim.workflow_knative import run as knative_run
from scripts_cosim.evaluate_workflow import paired_summary

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def probabilities(model,p,hand):
    x,e=features(p,hand)
    with torch.inference_mode():
        return model(torch.tensor(x)[None],torch.tensor(e)[None]).softmax(-1)[0].numpy()


def measured(fn,repeats,observer):
    timings=[];expected=None
    for _ in range(repeats):
        start=time.perf_counter();score,a=fn();timings.append((time.perf_counter()-start)*1000)
        if expected is not None and (score!=expected[0] or not np.array_equal(a,expected[1])):raise RuntimeError('non-deterministic search')
        expected=(score,a.copy())
    token=plan_observer.set(observer)
    try:score,a=fn()
    finally:plan_observer.reset(token)
    if score!=expected[0] or not np.array_equal(a,expected[1]):raise RuntimeError('observational trace changes search')
    return {'cost_ms':float(score),'planning_ms':float(np.median(timings)),'timings_ms':timings,'assignment':a.tolist()}


def read_case(p,seed,models,counts,repeats,selected=None):
    traces={}
    def observer(a,c):
        a=np.asarray(a,dtype=np.int16);key=a.tobytes()
        if key in traces and abs(traces[key][1]-c)>1e-7:raise RuntimeError('same plan has inconsistent cost')
        traces[key]=(a.copy(),float(c))
    learned={}
    for name,(model,hand) in models.items():
        ns=counts if selected is None else [selected[name]['moves']]
        for moves in ns:
            learned[f'{name}:{moves}']=measured(lambda model=model,hand=hand,moves=moves:improve(p,probabilities(model,p,hand),moves,seed),repeats,observer)
    hand={}
    for label,temp in [('uniform',None),('duration_softmax_4',4),('duration_softmax_12',12)]:
        for moves in counts:
            hand[f'{label}:{moves}']=measured(lambda temp=temp,moves=moves:improve(p,hand_probabilities(p,temp),moves,seed),repeats,observer)
    readings=[methods(p,teacher=False) for _ in range(repeats)]
    token=plan_observer.set(observer)
    try:traced=methods(p,teacher=False)
    finally:plan_observer.reset(token)
    for name,r in readings[0].items():
        if any(abs(x[name]['cost_ms']-r['cost_ms'])>1e-7 or x[name]['assignment']!=r['assignment'] for x in readings+[traced]):raise RuntimeError('control changed across repeats')
        r['timings_ms']=[x[name]['planning_ms'] for x in readings];r['planning_ms']=float(np.median(r['timings_ms']));r.pop('plan',None);hand[name]=r
    return {'seed':seed,'learned':learned,'controls':hand},traces


def write_case(root,row,p,traces):
    cell=root/str(row['seed']);(cell/'placements').mkdir(parents=True)
    config,inputs=substrate(p/1000)
    (cell/'input.json').write_text(json.dumps({'config':config,'inputs':inputs},allow_nan=False)+'\n')
    (cell/'read.json').write_text(json.dumps(row,indent=2)+'\n')
    with (cell/'placements/placements.jsonl').open('w') as out:
        for a,c in traces.values():out.write(json.dumps({'placement_plan':a.tolist(),'rtt_ms':c},allow_nan=False)+'\n')
    return cell


def configure_environment():
    for key in list(os.environ):
        if key.startswith(('HEROSIM_','GNN_','LIVE_AUDIT_','COSIM_')):os.environ.pop(key)
    os.environ.update(SIM_FORCE_FULL_STATS='1',HEROSIM_PEER_EXCHANGE='1',HEROSIM_PG_DRAIN_CONTRACT='availability_v2')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume-from',type=Path);args=ap.parse_args()
    if args.out.exists():raise ValueError('preserve historical experiment')
    args.out.mkdir(parents=True);torch.set_num_threads(1)
    protocol=json.loads((ROOT/'experiments/workflow_proposal_v1.json').read_text());cache=ROOT/'simulation_data/gnn_environment_search_v1/workflow_corpus';weights=ROOT/'models/workflow_amortized_v1'
    names=['experiments/workflow_proposal_v1.json','src/placement/workflow_proposals.py','scripts_cosim/workflow_proposal_gate.py','scripts_cosim/workflow_knative.py','scripts_cosim/workflow_live.py','src/policy/workflow/model.py','src/placement/workflow_planning.py','scripts_cosim/workflow_qualification.py','scripts_cosim/evaluate_workflow.py','src/placement/infrastructure.py','src/placement/executor.py','src/policy/knative_network/scheduler.py','src/placement/simulation.py']
    sources={p:sha(ROOT/p) for p in names};models={};frozen={}
    for arm in ('gnn','mpoff','mlp_hand'):
        for seed in protocol['training_draws']:
            name=f'{arm}_seed{seed}';path=weights/f'{name}.pt';side=path.with_suffix('.contract.json');contract=json.loads(side.read_text())
            if contract['contract']!=CONTRACT or contract['arm']!=arm or contract['architecture']['mp']!=(arm=='gnn'):raise ValueError('checkpoint contract mismatch')
            if contract['data_metadata_sha256']!=sha(cache/'METADATA.json'):raise ValueError('checkpoint data mismatch')
            for src,h in contract['sources'].items():
                if sha(ROOT/src)!=h:raise ValueError('checkpoint source mismatch')
            model=WorkflowAssignmentNet(**contract['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval();models[name]=(model,arm=='mlp_hand')
            frozen[name]={'weights_sha256':sha(path),'contract_sha256':sha(side)}
    (args.out/'protocol_before_run.json').write_text(json.dumps({'protocol':protocol,'sources':sources,'frozen_checkpoints':frozen},indent=2)+'\n')
    old_meta=json.loads((cache/'METADATA.json').read_text());old_ids=set()
    for case in old_meta['cases']:
        src=cache/f"ds_{case['seed']}/problem.json"
        if sha(src)!=case['source_sha256']:raise ValueError('old source bytes mismatch')
        p=np.array(json.loads(src.read_text())['processing_ms'],dtype=float);p[np.isnan(p)]=np.inf;identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in old_ids or identity!=case['instance_sha256']:raise ValueError('old source identity mismatch or duplicate')
        old_ids.add(identity)
    counts=protocol['proposal_counts'];repeats=protocol['timing_repeats'];budget=protocol['planning_budget_ms']
    if args.resume_from:
        prior=json.loads((args.resume_from/'protocol_before_run.json').read_text())
        if prior['frozen_checkpoints']!=frozen or prior['protocol']!=protocol:raise ValueError('retry cannot change frozen weights or protocol')
        selected=json.loads((args.resume_from/'frozen_selection.json').read_text())
        (args.out/'retry_amendment.json').write_text(json.dumps({'original':str(args.resume_from),'frozen_selection_sha256':sha(args.resume_from/'frozen_selection.json'),'reason':'first live call rejected missing HEROSIM_PEER_EXCHANGE; no completed live result; same selected settings retained'},indent=2)+'\n')
    else:
        val,_,_,_=load_split(cache,'validation');lo,hi=protocol['validation_seeds'];ids=[i for i,s in enumerate(val['seeds']) if lo<=s<=hi]
        if len(ids)!=hi-lo+1:raise ValueError('missing validation instances')
        for model,hand in models.values():
            for _ in range(5):probabilities(model,val['p'][ids[0]],hand)
        counts=protocol['proposal_counts'];repeats=protocol['timing_repeats'];budget=protocol['planning_budget_ms'];validation=[]
        for i in ids:
            p=val['p'][i];seed=int(val['seeds'][i]);row,trace=read_case(p,seed,models,counts,repeats);write_case(args.out/'validation',row,p,trace);validation.append(row)
            print('validation',len(validation),'/',len(ids),flush=True)
        selected={}
        for name in models:
            choices=[]
            for moves in counts:
                key=f'{name}:{moves}';choices.append({'moves':moves,'validation_mean_rtt_ms':float(np.mean([r['learned'][key]['cost_ms'] for r in validation])),'validation_p95_ms':float(np.quantile([r['learned'][key]['planning_ms'] for r in validation],.95))})
            valid=[c for c in choices if c['validation_p95_ms']<=budget]
            chosen=min(valid,key=lambda c:c['validation_mean_rtt_ms']) if valid else min(choices,key=lambda c:c['validation_p95_ms'])
            selected[name]={**chosen,'budget_qualified':bool(valid),'all_choices':choices}
    warm_p=instance(protocol['validation_seeds'][0])
    for model,hand in models.values():
        for _ in range(5):probabilities(model,warm_p,hand)
    (args.out/'frozen_selection.json').write_text(json.dumps(selected,indent=2)+'\n');print('SELECTION_FROZEN',json.dumps({k:{x:y for x,y in v.items() if x!='all_choices'} for k,v in selected.items()}),flush=True)
    # Fresh problems are constructed only after every decoding choice is frozen.
    lo,hi=protocol['test_seeds'];problems={s:instance(s) for s in range(lo,hi+1)};new_ids=set()
    for p in problems.values():
        identity=hashlib.sha256(p.tobytes()).hexdigest()
        if identity in old_ids or identity in new_ids:raise ValueError('duplicate or reused actual test problem')
        new_ids.add(identity)
    (args.out/'source_validation.json').write_text(json.dumps({'status':'PASS','old_unique':len(old_ids),'new_unique':len(new_ids),'new_ids':sorted(new_ids)},indent=2)+'\n')
    configure_environment()
    rows=[];live_count=0
    for seed,p in problems.items():
        for path,h in sources.items():
            if sha(ROOT/path)!=h:raise ValueError('source changed before live validation')
        row,trace=read_case(p,seed,models,counts,repeats,selected);cell=write_case(args.out/'test',row,p,trace)
        frontier={str(b):min((k for k,v in row['controls'].items() if v['planning_ms']<=b),key=lambda k:row['controls'][k]['cost_ms'],default=None) for b in (1,2,5,10)}
        if frontier['2'] is None:raise ValueError('no budget-compliant hand control')
        checked={**row['learned'],**{name:row['controls'][name] for name in set(['ect',frontier['2']])}}
        with (cell/'simulation.log').open('w') as log,(cell/'live.jsonl').open('w') as out:
            for name,r in checked.items():
                stat=run(p/1000,np.array(r['assignment']),log,seed);live_count+=1
                if abs(stat['total_rtt']*1000-r['cost_ms'])>1e-6 or abs(sum(t['doneTime']-t['dispatchedTime'] for t in stat['tasks'])-stat['total_rtt'])>1e-8:raise RuntimeError('live plan mismatch')
                out.write(json.dumps({'arm':name,'placement_plan':r['assignment'],'stats':stat},allow_nan=False)+'\n')
            kn=knative_run(p/1000,log,seed);live_count+=1
            if abs(evaluate(p,np.array(kn['assignment']))-kn['total_rtt']*1000)>1e-6:raise RuntimeError('actual Knative cost differs from reconstructed plan')
            out.write(json.dumps({'arm':'knative_network','placement_plan':kn['assignment'],'stats':kn},allow_nan=False)+'\n')
        row.update(frontier=frontier,knative_rtt_ms=kn['total_rtt']*1000,live_checked=list(checked)+['knative_network']);(cell/'read.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print('live',len(rows),'/',len(problems),'runs',live_count,flush=True)
    costs={arm:np.array([[r['learned'][f'{arm}_seed{s}:{selected[f"{arm}_seed{s}"]["moves"]}']['cost_ms'] for r in rows] for s in protocol['training_draws']]) for arm in ('gnn','mpoff','mlp_hand')}
    hand=np.array([r['controls'][r['frontier']['2']]['cost_ms'] for r in rows]);kn=np.array([r['knative_rtt_ms'] for r in rows])
    comparisons={k:paired_summary(costs['gnn'],v) for k,v in [('mpoff',costs['mpoff']),('mlp_hand',costs['mlp_hand']),('best_hand_2ms',np.tile(hand,(3,1))),('knative_network',np.tile(kn,(3,1)))]}
    timing={name:{'median_ms':float(np.median([r['learned'][f'{name}:{selected[name]["moves"]}']['planning_ms'] for r in rows])),'p95_ms':float(np.quantile([r['learned'][f'{name}:{selected[name]["moves"]}']['planning_ms'] for r in rows],.95))} for name in models}
    quality=all(comparisons[k]['median_gain_pct']>=5 and comparisons[k]['hierarchical_median_ci95_pct'][0]>0 for k in ('mpoff','mlp_hand','best_hand_2ms'));budget_pass=all(v['p95_ms']<=budget and selected[k]['budget_qualified'] for k,v in timing.items() if k.startswith('gnn_'))
    report={'verdict':'PILOT_PASS' if quality and budget_pass else 'PILOT_NEGATIVE','cases':len(rows),'live_runs':live_count,'comparisons':comparisons,'timing':timing,'quality_pass':quality,'budget_pass':budget_pass,'selected':selected,'mean_rtt_ms':{**{k:float(v.mean()) for k,v in costs.items()},'best_hand_2ms':float(hand.mean()),'knative_network':float(kn.mean())},'hand_frontier':{str(b):{'cases':sum(r['frontier'][str(b)] is not None for r in rows),'mean_rtt_ms':float(np.mean([r['controls'][r['frontier'][str(b)]]['cost_ms'] for r in rows if r['frontier'][str(b)] is not None]))} for b in (1,2,5,10)},'sources':sources,'frozen_checkpoints':frozen}
    (args.out/'read.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('selected','sources','frozen_checkpoints')},indent=2),flush=True)
if __name__=='__main__':main()

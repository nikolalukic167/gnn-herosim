"""Frozen-checkpoint workflow evaluation with timed controls and real-engine replay."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from src.policy.workflow.model import WorkflowAssignmentNet, CONTRACT, features
from src.policy.workflow.train import load_split, sha
from src.placement.workflow_planning import evaluate
from scripts_cosim.workflow_qualification import methods
from scripts_cosim.workflow_live import run, substrate


def infer(model, p, hand):
    x,e=features(p,hand=hand)
    with torch.inference_mode():
        return model(torch.from_numpy(x)[None],torch.from_numpy(e)[None]).argmax(-1)[0].numpy()


def paired_summary(gnn, control):
    # Resample environments and training draws, rather than treating all draws as new environments.
    gain=100*(1-gnn/control)
    rng=np.random.default_rng(921);boot=[]
    for _ in range(5000):
        draws=rng.integers(gnn.shape[0],size=gnn.shape[0])
        cases=rng.integers(gnn.shape[1],size=gnn.shape[1])
        boot.append(np.median(gain[np.ix_(draws,cases)].mean(0)))
    return {'median_gain_pct':float(np.median(gain.mean(0))),
            'mean_gain_pct':float(gain.mean()),
            'hierarchical_median_ci95_pct':np.quantile(boot,[.025,.975]).tolist(),
            'individual_training_draw_medians_pct':np.median(gain,axis=1).tolist()}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--split',choices=['validation','test'],default='test')
    ap.add_argument('--limit',type=int);ap.add_argument('--timing-repeats',type=int,default=5)
    args=ap.parse_args();torch.set_num_threads(1)
    if args.out.exists():raise ValueError('preserve previous evaluation')
    args.out.mkdir(parents=True)
    frozen={};models={}
    for arm in ('gnn','mpoff','mlp_hand'):
        for seed in (101,102,103):
            path=args.models/f'{arm}_seed{seed}.pt';side=json.loads(path.with_suffix('.contract.json').read_text())
            if side['contract']!=CONTRACT or side['arm']!=arm or side['seed']!=seed:raise ValueError('checkpoint contract mismatch')
            if side['data_metadata_sha256']!=sha(args.cache_dir/'METADATA.json'):raise ValueError('checkpoint data fingerprint mismatch')
            for source in ('src/policy/workflow/model.py','src/placement/workflow_planning.py'):
                if sha(source)!=side['sources'][source]:raise ValueError('serving source differs from training')
            model=WorkflowAssignmentNet(**side['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval()
            name=f'{arm}_seed{seed}';models[name]=(model,arm=='mlp_hand')
            frozen[name]={'weights_sha256':sha(path),'contract_sha256':sha(path.with_suffix('.contract.json')),'selected_epoch':side['selected_epoch']}
    # Freeze the selected artifacts before opening held-out data.
    (args.out/'frozen_checkpoints.json').write_text(json.dumps(frozen,indent=2)+'\n')
    data,_,_,_=load_split(args.cache_dir,args.split)
    ids=list(range(len(data['p'])))[:args.limit]
    identity=[hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for p in data['p']]
    if len(set(identity))!=len(identity):raise ValueError('duplicate actual test infrastructure')
    for k in list(os.environ):
        if k.startswith(('HEROSIM_','GNN_','LIVE_AUDIT_','COSIM_')):os.environ.pop(k)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_PG_DRAIN_CONTRACT='availability_v2',SIM_FORCE_FULL_STATS='1')
    sources={str(p):sha(p) for p in [Path(__file__),Path('scripts_cosim/workflow_live.py'),Path('scripts_cosim/workflow_qualification.py'),Path('src/placement/workflow_planning.py'),Path('src/policy/workflow/model.py'),Path('src/placement/infrastructure.py'),Path('src/placement/executor.py'),Path('src/placement/availability.py'),Path('src/executesimulation.py')]}
    (args.out/'protocol_before_run.json').write_text(json.dumps({'split':args.split,'cases':len(ids),'timing_repeats':args.timing_repeats,'sources':sources,'data_sha256':sha(args.cache_dir/f'{args.split}.npz'),'thread_count':1,'python':os.sys.version,'torch':torch.__version__,'numpy':np.__version__},indent=2)+'\n')
    for model,hand in models.values():
        for _ in range(10):infer(model,data['p'][ids[0]],hand)
    rows=[];live_runs=0
    for i in ids:
        p=data['p'][i];seed=int(data['seeds'][i]);cell=args.out/str(seed);cell.mkdir()
        # Repeated full invocations include feature construction, search, and decoding.
        reads=[methods(p) for _ in range(args.timing_repeats)]
        controls=reads[0]
        for name,r in controls.items():
            if any(abs(other[name]['cost_ms']-r['cost_ms'])>1e-7 for other in reads):raise ValueError('nondeterministic control')
            r['timings_ms']=[other[name]['planning_ms'] for other in reads]
            r['planning_ms']=float(np.median(r['timings_ms']));r.pop('plan',None)
        learned={}
        for name,(model,hand) in models.items():
            times=[];a=None
            for _ in range(args.timing_repeats):
                start=time.perf_counter();pred=infer(model,p,hand);times.append(1000*(time.perf_counter()-start))
                if a is not None and not np.array_equal(pred,a):raise ValueError('nondeterministic model inference')
                a=pred
            learned[name]={'cost_ms':evaluate(p,a),'planning_ms':float(np.median(times)),'timings_ms':times,'assignment':a.tolist()}
        frontier={}
        for budget in (1,2,5,10,50,100,1000):
            eligible={k:v for k,v in controls.items() if v['planning_ms']<=budget}
            frontier[str(budget)]=min(eligible,key=lambda k:eligible[k]['cost_ms']) if eligible else None
        if frontier['2'] is None:raise ValueError('no hand control fits registered budget')
        teacher=min(controls,key=lambda k:controls[k]['cost_ms'])
        check={**learned,**{name:controls[name] for name in set(['ect',frontier['2'],teacher])}}
        config,inputs=substrate(p/1000)
        (cell/'input.json').write_text(json.dumps({'config':config,'inputs':inputs},allow_nan=False)+'\n')
        (cell/'placements').mkdir()
        with (cell/'simulation.log').open('w') as log,(cell/'placements/placements.jsonl').open('w') as trace:
            for name,r in check.items():
                stat=run(p/1000,np.array(r['assignment']),log,seed);live_runs+=1
                if not np.isclose(stat['job_completion_sum']*1000,r['cost_ms'],rtol=0,atol=1e-6) or not np.isclose(stat['total_rtt'],stat['job_completion_sum'],rtol=0,atol=1e-8):raise ValueError(f'live mismatch {seed} {name}')
                trace.write(json.dumps({'arm':name,'placement_plan':r['assignment'],'rtt':stat['total_rtt'],'stats':stat},allow_nan=False)+'\n')
        row={'seed':seed,'learned':learned,'controls':controls,'frontier':frontier,'teacher':teacher,'live_checked':list(check)}
        (cell/'read.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print('evaluated',len(rows),'of',len(ids),'live_runs',live_runs,flush=True)
    costs={arm:np.array([[r['learned'][f'{arm}_seed{s}']['cost_ms'] for r in rows] for s in (101,102,103)]) for arm in ('gnn','mpoff','mlp_hand')}
    hand=np.array([r['controls'][r['frontier']['2']]['cost_ms'] for r in rows])
    comparisons={arm:paired_summary(costs['gnn'],c) for arm,c in [('mpoff',costs['mpoff']),('mlp_hand',costs['mlp_hand']),('best_hand_2ms',hand[None].repeat(3,axis=0))]}
    timing={name:{'median_ms':float(np.median([r['learned'][name]['planning_ms'] for r in rows])),'p95_ms':float(np.quantile([r['learned'][name]['planning_ms'] for r in rows],.95))} for name in models}
    # A quality win does not pass the registered budget if the GNN misses its latency target.
    quality=all(v['median_gain_pct']>=5 and v['hierarchical_median_ci95_pct'][0]>0 for v in comparisons.values())
    budget=all(v['p95_ms']<=2 for k,v in timing.items() if k.startswith('gnn_'))
    report={'comparisons':comparisons,'learned_timing':timing,'live_runs':live_runs,'cases':len(rows),'quality_pass':quality,'budget_p95_pass':budget,'verdict':'PILOT_PASS' if quality and budget else 'PILOT_NEGATIVE','mean_rtt_ms':{arm:float(c.mean()) for arm,c in costs.items()},'hand_frontier_mean_rtt_ms':{str(b):float(np.mean([r['controls'][r['frontier'][str(b)]]['cost_ms'] for r in rows if r['frontier'][str(b)] is not None])) for b in (1,2,5,10,50,100,1000)},'frontier_coverage':{str(b):sum(r['frontier'][str(b)] is not None for r in rows) for b in (1,2,5,10,50,100,1000)},'unrestricted_teacher_mean_rtt_ms':float(np.mean([r['controls'][r['teacher']]['cost_ms'] for r in rows])),'frozen_checkpoints':frozen,'sources':sources}
    (args.out/'read.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

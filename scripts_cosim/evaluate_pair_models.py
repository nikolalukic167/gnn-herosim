"""Frozen validation/timing read and complete real-HeROsim pair-selector live gate."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from src.policy.pair_selector.model import PairNet,CONTRACT
from src.policy.pair_selector.train import load_split,serve,predict
from scripts_cosim.mixed_pair_budget import BudgetPairs
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.radical_physics_screen import sha,save
from scripts_cosim.evaluate_workflow import paired_summary
from scripts_cosim.mixed_physics_live import run as herosim
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.live import run as independent
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);ap.add_argument('--models',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('preserve earlier gate')
    args.out.mkdir(parents=True);torch.set_num_threads(1);models={};frozen={};engine=BudgetPairs(args.out/'build')
    for arm in ('gnn','mpoff','mlp_hand'):
        for seed in (301,302,303):
            name=f'{arm}_seed{seed}';path=args.models/f'{name}.pt';side=json.loads(path.with_suffix('.contract.json').read_text())
            if side['contract']!=CONTRACT or side['arm']!=arm or side['seed']!=seed or side['physics']!='mixed_execution_v1':raise ValueError('checkpoint contract mismatch')
            if side['data_metadata_sha256']!=sha(args.cache_dir/'METADATA.json') or any(sha(ROOT/p)!=h for p,h in side['sources'].items()):raise ValueError('training source/data mismatch')
            model=PairNet(**side['architecture']);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval();models[name]=(model,arm=='mlp_hand')
            frozen[name]={'weights_sha256':sha(path),'contract_sha256':sha(path.with_suffix('.contract.json')),'selected_epoch':side['selected_epoch']}
    sources={p:sha(ROOT/p) for p in ['scripts_cosim/evaluate_pair_models.py','src/policy/pair_selector/train.py','src/policy/pair_selector/model.py',
             'scripts_cosim/mixed_physics_live.py','scripts_cosim/workflow_live.py','src/placement/radical/live.py','src/placement/radical/coordinator.py',
             'src/placement/infrastructure.py','src/placement/simulation.py','src/placement/executor.py']}
    save(args.out/'frozen_before_test.json',{'models':frozen,'sources':sources,'metadata_sha256':sha(args.cache_dir/'METADATA.json')})
    val,problems,_,_,_,_=load_split(args.cache_dir,'validation');validation={}
    for name,(model,hand) in models.items():
        _,_,x,adj,pair,_=load_split(args.cache_dir,'validation',hand);indices=predict(model,x,adj,pair)
        expected=val['costs'][np.arange(len(indices)),indices];reads=[]
        for index,b in enumerate(problems):
            read=timed(lambda:serve(model,engine,b,hand),7)
            if read['cost']!=expected[index]:raise ValueError('cached/live inference mismatch')
            reads.append(read)
        mean=float(expected.mean());p95=float(np.quantile([r['time_ms'] for r in reads],.95));gain=100*(1-mean/float(val['rule'].mean()))
        validation[name]={'mean_cost':mean,'gain_vs_rule_pct':gain,'p95_ms':p95,'passes':gain>=5 and p95<=5,'rows':reads}
        print('VALIDATION',name,mean,gain,p95,flush=True)
    save(args.out/'validation_READ.json',validation)
    data,problems,_,_,_,_=load_split(args.cache_dir,'test');configure_environment();rows=[];completed=0
    for i,b in enumerate(problems):
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('runtime source changed')
        for name,r in frozen.items():
            path=args.models/f'{name}.pt'
            if sha(path)!=r['weights_sha256'] or sha(path.with_suffix('.contract.json'))!=r['contract_sha256']:raise ValueError('checkpoint changed')
        seed=int(data['seeds'][i]);cell=args.out/'test'/str(seed);cell.mkdir(parents=True)
        methods={name:timed(lambda model=model,hand=hand:serve(model,engine,b,hand),7) for name,(model,hand) in models.items()}
        methods['graph1']=timed(lambda:engine.plan(b,'graph1'),7);methods['incumbent']=timed(lambda:engine.incumbent(b),7)
        best=json.loads((args.cache_dir/f'ds_{seed}/best.json').read_text());methods['oracle']={'cost':best['objective'],'plan':best['placement_plan'],'unpriced_oracle':True}
        with (cell/'simulation.log').open('w') as log,(cell/'results.jsonl').open('w') as output:
            for name,read in methods.items():
                a=np.array(read['plan']);stats=herosim(b,a,log,seed);reference=independent(b,a,'mixed')
                if abs(stats['total_rtt']*1000-read['cost'])>1e-6 or abs(reference['objective']-read['cost'])>1e-6:raise ValueError('live objective mismatch')
                seen=set()
                for task in stats['tasks']:
                    j,k=map(int,task['taskType']['name'][2:].split('_'));seen.add((j,k))
                    if abs(task['doneTime']*1000-reference['ends'][j][k])>1e-6 or task['executionNode']!=f'node{a[j,k]}':raise ValueError('task completion mismatch')
                if len(seen)!=a.size or len(stats['tasks'])!=a.size:raise ValueError('incomplete task accounting')
                output.write(json.dumps({'arm':name,'stats':stats},allow_nan=False)+'\n');completed+=a.size
        row={'seed':seed,'methods':methods,'live_sha256':sha(cell/'results.jsonl')};save(cell/'read.json',row);rows.append(row);print('LIVE',len(rows),'/32',flush=True)
    def matrix(arm):return np.array([[r['methods'][f'{arm}_seed{seed}']['cost'] for r in rows] for seed in (301,302,303)])
    gnn=matrix('gnn');comparisons={arm:paired_summary(gnn,matrix(arm)) for arm in ('mpoff','mlp_hand')}
    comparisons['graph1']=paired_summary(gnn,np.array([[r['methods']['graph1']['cost'] for r in rows]]))
    comparisons['incumbent']=paired_summary(gnn,np.array([[r['methods']['incumbent']['cost'] for r in rows]]))
    timings={name:float(np.quantile([r['methods'][name]['time_ms'] for r in rows],.95)) for name in [*models,'graph1','incumbent']}
    passed=all(comparisons[arm]['median_gain_pct']>=5 and comparisons[arm]['hierarchical_median_ci95_pct'][0]>0 for arm in ('mpoff','mlp_hand','graph1')) and all(timings[f'gnn_seed{s}']<=5 for s in (301,302,303))
    report={'cases':len(rows),'live_runs':len(rows)*12,'completed_operations':completed,'comparisons':comparisons,'p95_ms':timings,'passes':passed,
            'mean_costs':{name:float(np.mean([r['methods'][name]['cost'] for r in rows])) for name in rows[0]['methods']}}
    save(args.out/'read.json',report);print(json.dumps(report,indent=2),flush=True)
    audit(args.out,args.cache_dir,args.models)

def audit(out,cache,models):
    from scripts_cosim.audit_radical_physics import load_problem
    freeze=json.loads((out/'frozen_before_test.json').read_text());meta=json.loads((cache/'METADATA.json').read_text())
    if sha(cache/'METADATA.json')!=freeze['metadata_sha256'] or any(sha(ROOT/p)!=h for p,h in freeze['sources'].items()):raise ValueError('gate source changed')
    for name,r in freeze['models'].items():
        path=models/f'{name}.pt'
        if sha(path)!=r['weights_sha256'] or sha(path.with_suffix('.contract.json'))!=r['contract_sha256']:raise ValueError('model artifact changed')
    expected={r['seed']:r for r in meta['cases'] if r['split']=='test'};seen=set();runs=tasks=0
    for cell in sorted((out/'test').iterdir()):
        seed=int(cell.name);seen.add(seed);r=json.loads((cell/'read.json').read_text());path=cache/f'ds_{seed}/problem.json'
        if sha(path)!=expected[seed]['sha256'] or sha(cell/'results.jsonl')!=r['live_sha256']:raise ValueError('live data hash mismatch')
        b=load_problem(path);arms=set()
        for line in (cell/'results.jsonl').read_text().splitlines():
            result=json.loads(line);name=result['arm'];arms.add(name);stats=result['stats'];read=r['methods'][name];a=np.array(read['plan']);reference=independent(b,a,'mixed')
            if abs(stats['total_rtt']*1000-read['cost'])>1e-6 or abs(reference['objective']-read['cost'])>1e-6:raise ValueError('audit cost mismatch')
            ids=set()
            for task in stats['tasks']:
                j,k=map(int,task['taskType']['name'][2:].split('_'));ids.add((j,k))
                if task['executionNode']!=f'node{a[j,k]}' or abs(task['doneTime']*1000-reference['ends'][j][k])>1e-6:raise ValueError('audit completion mismatch')
            if len(ids)!=a.size or len(stats['tasks'])!=a.size:raise ValueError('audit coverage failure')
            tasks+=len(ids);runs+=1
        if arms!=set(r['methods']) or len(arms)!=12:raise ValueError('arm coverage mismatch')
    if seen!=set(expected):raise ValueError('test coverage mismatch')
    report=json.loads((out/'read.json').read_text())
    if runs!=report['live_runs'] or tasks!=report['completed_operations']:raise ValueError('report coverage mismatch')
    save(out/'AUDIT.json',{'status':'PASS','runs':runs,'completed_operations':tasks,'report_sha256':sha(out/'read.json'),'frozen_sha256':sha(out/'frozen_before_test.json')});print('AUDIT PASS',runs,tasks,flush=True)
if __name__=='__main__':main()

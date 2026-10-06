"""Registered priority-order screen, independent replay and actual HeROsim gate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,pack,serialized
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import replay,herosim
from scripts_cosim.mixed_pair_exploration import verify
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.evaluate_workflow import paired_summary
from scripts_cosim.workflow_proposal_gate import configure_environment
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['experiments/mixed_dispatch_v1.json','scripts_cosim/mixed_dispatch_gate.py','src/placement/radical/dispatch.py','src/placement/radical/dispatch.cpp','src/placement/radical/dispatch_live.py','src/placement/radical/coordinator.py','src/placement/radical/mixed.py','src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','src/placement/infrastructure.py','src/placement/simulation.py','src/placement/executor.py','scripts_cosim/mixed_physics_live.py','scripts_cosim/workflow_live.py','scripts_cosim/mixed_pair_exploration.py','scripts_cosim/mixed_physics_control_gate.py','scripts_cosim/evaluate_workflow.py']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve earlier results')
    (out/'inputs').mkdir(parents=True);protocol=json.loads((ROOT/SOURCES[0]).read_text());engine=DispatchNative(out/'build')
    pre={'protocol':protocol,'sources':{p:sha(ROOT/p) for p in SOURCES},'native':engine.provenance};save(out/'protocol_before_run.json',pre)
    prior=set()
    for path in out.parent.glob('*/source_validation.json'):
        for r in json.loads(path.read_text()).get('inputs',{}).values():
            if isinstance(r,dict) and 'identity' in r:prior.add(r['identity'])
    meta={'status':'PASS','prior_identities_checked':len(prior),'inputs':{}}
    for key in ('calibration_seeds','fresh_seeds'):
        lo,hi=protocol[key]
        for seed in range(lo,hi+1):
            b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in prior:raise ValueError('reused physical input')
            prior.add(identity);path=out/'inputs'/f'{seed}.json';save(path,serialized(b));meta['inputs'][str(seed)]={'identity':identity,'sha256':sha(path)}
    save(out/'source_validation.json',meta);verify(out,pre,meta)
    methods=[f'{start}:{n}' for start in protocol['starts'] for n in protocol['search_steps']];cal=[];lo,hi=protocol['calibration_seeds']
    for seed in range(lo,hi+1):
        b=load_problem(out/'inputs'/f'{seed}.json');cal.append({'seed':seed,'methods':{m:timed(lambda m=m:engine.plan(b,m),protocol['timing_repeats']) for m in methods}});print('CAL',seed,flush=True)
    metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in cal])),'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in cal],.95))} for m in methods}
    eligible=[m for m in methods if metrics[m]['p95_ms']<=protocol['calibration_fraction']*protocol['budget_ms']]
    if not eligible:raise RuntimeError('no affordable control')
    selected=min(eligible,key=lambda m:metrics[m]['mean_cost']);save(out/'calibration.json',{'selected':selected,'metrics':metrics,'rows':cal});print('FROZEN',selected,metrics[selected],flush=True)
    rows=[];lo,hi=protocol['fresh_seeds']
    for seed in range(lo,hi+1):
        verify(out,pre,meta);b=load_problem(out/'inputs'/f'{seed}.json');cell=out/'test'/str(seed);(cell/'placements').mkdir(parents=True)
        reads={'control':timed(lambda:engine.plan(b,selected),7),'srpt':timed(lambda:engine.plan(b,'srpt:0'),7)}
        refs={start:engine.plan(b,f'{start}:8192') for start in ('spt','srpt')}
        ref=min([ (reads['control']['cost'],np.array(reads['control']['plan'])),*refs.values()],key=lambda item:item[0]);reads['reference']={'cost':ref[0],'plan':ref[1].tolist(),'unpriced_reference':True}
        with (cell/'placements/placements.jsonl').open('w') as f:
            for name,r in reads.items():f.write(json.dumps({'arm':name,'placement_plan':r['plan'][0],'priority':r['plan'][1],'objective':r['cost']})+'\n')
            for name,(c,p) in refs.items():f.write(json.dumps({'arm':'reference_start_'+name,'placement_plan':p[0].tolist(),'priority':p[1].tolist(),'objective':c})+'\n')
        row={'seed':seed,'methods':reads,'placements_sha256':sha(cell/'placements/placements.jsonl')};save(cell/'read.json',row);rows.append(row);print('SCREEN',seed,{k:v['cost'] for k,v in reads.items()},flush=True)
    gain=paired_summary(np.array([[r['methods']['reference']['cost'] for r in rows]]),np.array([[r['methods']['control']['cost'] for r in rows]]));p95=float(np.quantile([r['methods']['control']['time_ms'] for r in rows],.95))
    report={'cases':len(rows),'selected':selected,'gain':gain,'control_p95_ms':p95,'passes':gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and p95<=5,'live_gate_pending':True}
    save(out/'read.json',report);print(json.dumps(report,indent=2),flush=True)
    configure_environment();tasks=0;runs=0;receipts=[]
    for row in rows:
        verify(out,pre,meta);seed=row['seed'];b=load_problem(out/'inputs'/f'{seed}.json');cell=out/'live'/str(seed);cell.mkdir(parents=True)
        if sha(out/'test'/str(seed)/'placements/placements.jsonl')!=row['placements_sha256']:raise ValueError('plans changed')
        with (cell/'simulation.log').open('w') as log,(cell/'results.jsonl').open('w') as output:
            for arm,r in row['methods'].items():
                a,priority=np.array(r['plan']);expected=replay(b,a,priority);actual=herosim(b,a,priority,log,seed);c,ends=engine.score_priority(b,a,priority)
                if c!=r['cost'] or c!=expected['objective'] or abs(actual['total_rtt']*1000-c)>1e-6 or not np.array_equal(ends,expected['ends']):raise ValueError('objective/replay mismatch')
                ids=set()
                for task in actual['tasks']:
                    j,k=map(int,task['taskType']['name'][2:].split('_'));ids.add((j,k))
                    if task['executionNode']!=f'node{a[j,k]}' or abs(task['doneTime']*1000-ends[j,k])>1e-6:raise ValueError('completion mismatch')
                if len(ids)!=a.size or len(actual['tasks'])!=a.size:raise ValueError('incomplete workload')
                if actual['mixed_execution']['dispatch_contract']!='mixed_ready_priority_v1':raise ValueError('dispatch contract not recorded')
                output.write(json.dumps({'arm':arm,'stats':actual},allow_nan=False)+'\n');tasks+=a.size;runs+=1
        receipts.append({'seed':seed,'sha256':sha(cell/'results.jsonl')});print('LIVE',seed,flush=True)
    verify(out,pre,meta)
    save(out/'AUDIT.json',{'status':'PASS','live_runs':runs,'completed_operations':tasks,'all_task_completions_match':True,'rows':receipts,'report_sha256':sha(out/'read.json')});print('AUDIT PASS',runs,tasks,flush=True)
if __name__=='__main__':main()

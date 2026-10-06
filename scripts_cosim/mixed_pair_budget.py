"""Budget-matched hand-search challenge to the frozen deep pair portfolio."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from src.placement.radical.pairs import PairNative,rank_pairs
from src.placement.radical.environment import problem,initial,pack,serialized
from scripts_cosim.mixed_pair_exploration import verify,live,SOURCES
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.evaluate_workflow import paired_summary
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='experiments/mixed_pair_budget_v1.json'

class BudgetPairs(PairNative):
    def plan(self,b,method):
        if ':' in method:
            start,search=method.split(':')
            anneal=search.startswith('anneal')
            steps=int(search[6:] if anneal else search[7:])
            return self.search(b,initial(b,start),steps,anneal)
        cost,a=self.incumbent(b)
        i,j=np.triu_indices(a.size,1)
        if method.startswith('graph'):
            locks=b['locks'].ravel();hosts=a.ravel();ops=a.shape[1]
            shared=np.zeros(len(i),dtype=int)
            for bit in range(6):shared+=((locks[i]&locks[j])>>bit)&1
            order=np.lexsort((j,i,-(hosts[i]==hosts[j]).astype(int),-(i//ops==j//ops).astype(int),-shared))
            count=int(method[5:])
        elif method.startswith('random'):
            order=np.random.default_rng(77).permutation(len(i));count=int(method[6:])
        else:raise ValueError('unknown control')
        pairs=np.column_stack((i[order[:count]],j[order[:count]]))
        values,plans=self.candidates(b,a,pairs);idx=int(values.argmin())
        return (float(values[idx]),plans[idx]) if values[idx]<cost else (cost,a)

def summary(rows,selected,budgets):
    result={}
    for budget in budgets:
        arm=f'control{budget}'
        gain=paired_summary(np.array([[r['methods']['oracle']['cost'] for r in rows]]),np.array([[r['methods'][arm]['cost'] for r in rows]]))
        control_p95=float(np.quantile([r['methods'][arm]['time_ms'] for r in rows],.95))
        oracle_p95=float(np.quantile([r['free_selector']['time_ms'] for r in rows],.95))
        result[str(budget)]={'selected':selected[str(budget)],'gain':gain,'control_p95_ms':control_p95,'free_selector_p95_ms':oracle_p95,
                            'passes':gain['median_gain_pct']>=5 and gain['hierarchical_median_ci95_pct'][0]>0 and max(control_p95,oracle_p95)<=budget}
    return result

def screen(out):
    if out.exists():raise ValueError('preserve prior artifacts')
    (out/'inputs').mkdir(parents=True)
    protocol=json.loads((ROOT/PROTOCOL).read_text());engine=BudgetPairs(out/'build')
    sources=[*SOURCES,PROTOCOL,'scripts_cosim/mixed_pair_budget.py']
    pre={'protocol':protocol,'sources':{p:sha(ROOT/p) for p in sources},'native':engine.provenance}
    save(out/'protocol_before_run.json',pre)
    prior=set()
    for path in out.parent.glob('*/source_validation.json'):
        for item in json.loads(path.read_text()).get('inputs',{}).values():
            if isinstance(item,dict) and 'identity' in item:prior.add(item['identity'])
    meta={'status':'PASS','prior_identities_checked':len(prior),'inputs':{}}
    for key in ('calibration_seeds','fresh_seeds'):
        lo,hi=protocol[key]
        for seed in range(lo,hi+1):
            b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in prior:raise ValueError('duplicate physical problem')
            prior.add(identity);path=out/'inputs'/f'{seed}.json';save(path,serialized(b))
            meta['inputs'][str(seed)]={'identity':identity,'sha256':sha(path)}
    save(out/'source_validation.json',meta);verify(out,pre,meta)
    methods=[f'{start}:anneal{n}' for start in protocol['starts'] for n in protocol['anneal_steps']]
    methods += [f'{start}:descent{n}' for start in protocol['starts'] for n in protocol['descent_rounds']]
    methods += protocol['pair_controls'];rows=[];lo,hi=protocol['calibration_seeds']
    for seed in range(lo,hi+1):
        b=load_problem(out/'inputs'/f'{seed}.json')
        rows.append({'seed':seed,'methods':{m:timed(lambda m=m:engine.plan(b,m),protocol['timing_repeats']) for m in methods}})
        print('CALIBRATE',seed,flush=True)
    metrics={m:{'mean_cost':float(np.mean([r['methods'][m]['cost'] for r in rows])),
                'p95_ms':float(np.quantile([r['methods'][m]['time_ms'] for r in rows],.95))} for m in methods}
    selected={}
    for budget in protocol['budgets_ms']:
        eligible=[m for m in methods if metrics[m]['p95_ms']<=budget]
        if not eligible:raise RuntimeError('no affordable control')
        selected[str(budget)]=min(eligible,key=lambda m:metrics[m]['mean_cost'])
    save(out/'calibration.json',{'selected':selected,'metrics':metrics,'rows':rows});print('FROZEN',selected,flush=True)
    rows=[];lo,hi=protocol['fresh_seeds']
    for seed in range(lo,hi+1):
        verify(out,pre,meta);b=load_problem(out/'inputs'/f'{seed}.json');cell=out/'test'/str(seed);(cell/'placements').mkdir(parents=True)
        reads={f'control{budget}':timed(lambda budget=budget:engine.plan(b,selected[str(budget)]),protocol['timing_repeats']) for budget in protocol['budgets_ms']}
        cost,a=engine.incumbent(b);pairs=np.array(list(combinations(range(a.size),2)),dtype=np.int64)
        values,plans=engine.candidates(b,a,pairs);idx=int(values.argmin())
        oracle={'cost':float(values[idx]),'plan':plans[idx].tolist(),'pair':pairs[idx].tolist()} if values[idx]<cost else {'cost':cost,'plan':a.tolist(),'pair':None}
        reads['oracle']=oracle
        def free_selector():
            c,z=engine.incumbent(b)
            if oracle['pair'] is None:return c,z
            scores,assignments=engine.candidates(b,z,np.array([oracle['pair']]))
            return (float(scores[0]),assignments[0]) if scores[0]<c else (c,z)
        receipt=timed(free_selector,protocol['timing_repeats'])
        if receipt['cost']!=oracle['cost'] or receipt['plan']!=oracle['plan']:raise ValueError('oracle recipe mismatch')
        with (cell/'placements/placements.jsonl').open('w') as f:
            f.write(json.dumps({'pair':None,'placement_plan':a.tolist(),'objective':cost})+'\n')
            for pair,c,z in zip(pairs,values,plans):f.write(json.dumps({'pair':pair.tolist(),'placement_plan':z.tolist(),'objective':float(c)})+'\n')
        row={'seed':seed,'methods':reads,'free_selector':receipt,'candidate_sha256':sha(cell/'placements/placements.jsonl')}
        save(cell/'read.json',row);rows.append(row);print('SCREEN',seed,{k:v['cost'] for k,v in reads.items()},flush=True)
    save(out/'read.json',{'cases':len(rows),'budgets':summary(rows,selected,protocol['budgets_ms']),'live_gate_pending':True})
    print((out/'read.json').read_text(),flush=True)

def audit(out):
    pre=json.loads((out/'protocol_before_run.json').read_text());meta=json.loads((out/'source_validation.json').read_text());verify(out,pre,meta)
    engine=BudgetPairs(out/'audit_build');report=json.loads((out/'live_READ.json').read_text());rows=[];count=0
    if report['status']!='PASS' or any(sha(ROOT/p)!=h for p,h in report['sources'].items()):raise ValueError('live provenance failure')
    for live_row in report['rows']:
        seed=live_row['seed'];cell=out/'test'/str(seed);row=json.loads((cell/'read.json').read_text());b=load_problem(out/'inputs'/f'{seed}.json')
        path=cell/'placements/placements.jsonl'
        if sha(path)!=row['candidate_sha256']:raise ValueError('candidate hash failure')
        records=[json.loads(line) for line in path.read_text().splitlines()]
        c,a=engine.incumbent(b)
        if records[0]['objective']!=c or records[0]['placement_plan']!=a.tolist():raise ValueError('incumbent mismatch')
        pairs=np.array([r['pair'] for r in records[1:]])
        expected=np.array(list(combinations(range(a.size),2)))
        if not np.array_equal(pairs,expected):raise ValueError('pair coverage failure')
        values,plans=engine.candidates(b,a,pairs)
        for value,plan,r in zip(values,plans,records[1:]):
            if value!=r['objective'] or plan.tolist()!=r['placement_plan']:raise ValueError('candidate mismatch')
            if engine.score(b,plan)[0]!=value:raise ValueError('score mismatch')
            count+=1
        if min(c,float(values.min()))!=row['methods']['oracle']['cost']:raise ValueError('incorrect ceiling')
        path=out/'live'/str(seed)/'results.jsonl'
        if sha(path)!=live_row['sha256']:raise ValueError('live artifact changed')
        actual=[json.loads(line) for line in path.read_text().splitlines()]
        if len(actual)!=3 or {r['arm'] for r in actual}!=set(row['methods']):raise ValueError('arm coverage failure')
        for r in actual:
            expected=row['methods'][r['arm']]
            if r['stats']['assignment']!=expected['plan'] or abs(r['stats']['total_rtt']*1000-expected['cost'])>1e-6:raise ValueError('live objective mismatch')
        rows.append(row)
    calibration=json.loads((out/'calibration.json').read_text());read=json.loads((out/'read.json').read_text())
    if summary(rows,calibration['selected'],pre['protocol']['budgets_ms'])!=read['budgets']:raise ValueError('statistics mismatch')
    save(out/'AUDIT.json',{'status':'PASS','candidate_plans':count,'live_runs':report['runs'],'completed_operations':report['tasks'],
                         'artifact_hashes':{p:sha(out/p) for p in ('read.json','live_READ.json','calibration.json','source_validation.json','protocol_before_run.json')}})
    print('AUDIT PASS',count,report['runs'],report['tasks'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['screen','live','audit']);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();{'screen':screen,'live':live,'audit':audit}[args.mode](args.out)

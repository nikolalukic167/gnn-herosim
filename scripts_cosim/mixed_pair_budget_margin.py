"""Precommitted timing-margin follow-up, preserving the original budget attempt."""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,pack,serialized
from scripts_cosim.mixed_pair_budget import BudgetPairs,summary,audit
from scripts_cosim.mixed_pair_exploration import verify,live
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.audit_radical_physics import load_problem
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='experiments/mixed_pair_budget_margin_v1.json'

def screen(out,calibration_root):
    if out.exists():raise ValueError('preserve prior evidence')
    original=json.loads((calibration_root/'protocol_before_run.json').read_text())
    verify(calibration_root,original,json.loads((calibration_root/'source_validation.json').read_text()))
    calibration=json.loads((calibration_root/'calibration.json').read_text())
    protocol=json.loads((ROOT/PROTOCOL).read_text());selected={}
    for budget in protocol['budgets_ms']:
        allowed=[m for m,d in calibration['metrics'].items() if d['p95_ms']<=protocol['calibration_fraction']*budget]
        if not allowed:raise ValueError('no control fits margin')
        selected[str(budget)]=min(allowed,key=lambda m:calibration['metrics'][m]['mean_cost'])
    (out/'inputs').mkdir(parents=True);engine=BudgetPairs(out/'build')
    sources={**original['sources'],PROTOCOL:sha(ROOT/PROTOCOL),'scripts_cosim/mixed_pair_budget_margin.py':sha(Path(__file__))}
    pre={'protocol':protocol,'sources':sources,'native':engine.provenance,'calibration_sha256':sha(calibration_root/'calibration.json')}
    save(out/'protocol_before_run.json',pre);save(out/'calibration.json',{**calibration,'selected':selected,'selection_fraction':protocol['calibration_fraction']})
    print('FROZEN',selected,flush=True)
    prior=set()
    for path in out.parent.glob('*/source_validation.json'):
        for d in json.loads(path.read_text()).get('inputs',{}).values():
            if isinstance(d,dict) and 'identity' in d:prior.add(d['identity'])
    meta={'status':'PASS','prior_identities_checked':len(prior),'inputs':{}}
    lo,hi=protocol['fresh_seeds']
    for seed in range(lo,hi+1):
        b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
        if identity in prior:raise ValueError('reused physical problem')
        prior.add(identity);path=out/'inputs'/f'{seed}.json';save(path,serialized(b))
        meta['inputs'][str(seed)]={'identity':identity,'sha256':sha(path)}
    save(out/'source_validation.json',meta);verify(out,pre,meta);rows=[]
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
        if receipt['cost']!=oracle['cost'] or receipt['plan']!=oracle['plan']:raise ValueError('oracle mismatch')
        with (cell/'placements/placements.jsonl').open('w') as f:
            f.write(json.dumps({'pair':None,'placement_plan':a.tolist(),'objective':cost})+'\n')
            for pair,c,z in zip(pairs,values,plans):f.write(json.dumps({'pair':pair.tolist(),'placement_plan':z.tolist(),'objective':float(c)})+'\n')
        row={'seed':seed,'methods':reads,'free_selector':receipt,'candidate_sha256':sha(cell/'placements/placements.jsonl')}
        save(cell/'read.json',row);rows.append(row);print('SCREEN',seed,flush=True)
    save(out/'read.json',{'cases':len(rows),'budgets':summary(rows,selected,protocol['budgets_ms']),'live_gate_pending':True})
    print((out/'read.json').read_text(),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['screen','live','audit']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--calibration-root',type=Path)
    args=ap.parse_args()
    if args.mode=='screen':screen(args.out,args.calibration_root)
    else:{'live':live,'audit':audit}[args.mode](args.out)

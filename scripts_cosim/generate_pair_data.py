"""Exhaustive small pair-selection corpus; all candidate plans and identities retained."""
import argparse,hashlib,json
from itertools import combinations
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,pack,serialized
from src.policy.pair_selector.model import features,CONTRACT
from scripts_cosim.mixed_pair_budget import BudgetPairs
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.audit_radical_physics import load_problem
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['scripts_cosim/generate_pair_data.py','src/policy/pair_selector/model.py','src/policy/mixed/model.py',
         'src/placement/radical/pairs.py','src/placement/radical/pairs.cpp','src/placement/radical/mixed.py',
         'src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','scripts_cosim/mixed_pair_budget.py']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve earlier corpus')
    out.mkdir(parents=True);engine=BudgetPairs(out/'build');protocol=json.loads((ROOT/'experiments/mixed_pair_learning_v1_protocol.json').read_text())
    sources={p:sha(ROOT/p) for p in SOURCES};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':engine.provenance})
    prior=set()
    for path in out.parent.glob('*/source_validation.json'):
        for r in json.loads(path.read_text()).get('inputs',{}).values():
            if isinstance(r,dict) and 'identity' in r:prior.add(r['identity'])
    cases=[];inputs={}
    for split in ('train','validation','test'):
        lo,hi=protocol[f'{split}_seeds']
        for seed in range(lo,hi+1):
            b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in prior:raise ValueError('duplicate physical input')
            prior.add(identity);cell=out/f'ds_{seed}';(cell/'placements').mkdir(parents=True);path=cell/'problem.json';save(path,serialized(b))
            r={'split':split,'seed':seed,'identity':identity,'sha256':sha(path)};cases.append(r);inputs[str(seed)]=r
    save(out/'source_validation.json',{'status':'PASS','inputs':inputs})
    files={};total=0
    for split in ('train','validation','test'):
        buffers={k:[] for k in ('seeds','x','xh','adj','pair','costs','rule')}
        for r in [r for r in cases if r['split']==split]:
            seed=r['seed'];cell=out/f'ds_{seed}';path=cell/'problem.json'
            if sha(path)!=r['sha256'] or any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed')
            b=load_problem(path);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity!=r['identity']:raise ValueError('physical identity mismatch')
            c,a=engine.incumbent(b);_,ends=engine.score(b,a);pairs=np.array(list(combinations(range(a.size),2)),dtype=np.int64)
            raw,plans=engine.candidates(b,a,pairs);costs=np.r_[c,np.minimum(raw,c)].astype(np.float32)
            x,adj,pair=features(b,a,ends);xh,_,_=features(b,a,ends,True)
            rule,ruleplan=engine.plan(b,'graph1')
            with (cell/'placements/placements.jsonl').open('w') as f:
                f.write(json.dumps({'pair':None,'placement_plan':a.tolist(),'objective':c,'served_objective':c})+'\n')
                for ids,value,plan in zip(pairs,raw,plans):f.write(json.dumps({'pair':ids.tolist(),'placement_plan':plan.tolist(),'objective':float(value),'served_objective':min(float(value),c)})+'\n')
            idx=int(costs.argmin());best=a if idx==0 else plans[idx-1]
            save(cell/'best.json',{'objective':float(costs[idx]),'placement_plan':best.tolist(),'candidate_index':idx,'global_optimality_certified':False})
            r['placements_sha256']=sha(cell/'placements/placements.jsonl');r['best_sha256']=sha(cell/'best.json');r['candidate_count']=len(costs)
            for k,v in {'seeds':seed,'x':x,'xh':xh,'adj':adj,'pair':pair,'costs':costs,'rule':rule}.items():buffers[k].append(v)
            total+=len(costs)
            if seed%8==7:print('DATA',split,seed,flush=True)
        path=out/f'{split}.npz';np.savez_compressed(path,**{k:np.array(v) for k,v in buffers.items()});files[path.name]=sha(path)
    save(out/'METADATA.json',{'contract':CONTRACT,'physics':'mixed_execution_v1','sources':sources,'files':files,'cases':cases,'candidate_count':total})
    validate(out)

def validate(out):
    meta=json.loads((out/'METADATA.json').read_text());engine=BudgetPairs(out/'validation_build');seen=set();count=0
    if any(sha(ROOT/p)!=h for p,h in meta['sources'].items()):raise ValueError('data code changed')
    for split in ('train','validation','test'):
        if sha(out/f'{split}.npz')!=meta['files'][f'{split}.npz']:raise ValueError('cache changed')
        with np.load(out/f'{split}.npz') as cache:
            for index,seed in enumerate(cache['seeds']):
                r=next(r for r in meta['cases'] if r['seed']==seed)
                if r['split']!=split:raise ValueError('split mismatch')
                cell=out/f'ds_{seed}';b=load_problem(cell/'problem.json');identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
                if identity in seen or identity!=r['identity'] or sha(cell/'problem.json')!=r['sha256']:raise ValueError('identity failure')
                seen.add(identity)
                if sha(cell/'placements/placements.jsonl')!=r['placements_sha256'] or sha(cell/'best.json')!=r['best_sha256']:raise ValueError('label artifact changed')
                c,a=engine.incumbent(b);_,ends=engine.score(b,a);x,adj,pair=features(b,a,ends);xh,_,_=features(b,a,ends,True)
                for k,v in {'x':x,'xh':xh,'adj':adj,'pair':pair}.items():
                    if not np.array_equal(cache[k][index],v):raise ValueError('feature mismatch')
                rows=[json.loads(line) for line in (cell/'placements/placements.jsonl').read_text().splitlines()]
                expected=np.array(list(combinations(range(a.size),2)))
                if len(rows)!=len(expected)+1 or not np.array_equal(expected,[r['pair'] for r in rows[1:]]):raise ValueError('candidate coverage failure')
                raw,plans=engine.candidates(b,a,expected);costs=np.r_[c,np.minimum(raw,c)]
                if not np.array_equal(costs,cache['costs'][index]) or rows[0]['placement_plan']!=a.tolist():raise ValueError('target mismatch')
                for value,plan,row in zip(raw,plans,rows[1:]):
                    if value!=row['objective'] or row['served_objective']!=min(c,value) or plan.tolist()!=row['placement_plan']:raise ValueError('placement mismatch')
                best=json.loads((cell/'best.json').read_text());bestplan=np.array(best['placement_plan'])
                if best['objective']!=float(costs.min()) or engine.score(b,bestplan)[0]!=float(costs.min()):raise ValueError('oracle mismatch')
                if engine.plan(b,'graph1')[0]!=cache['rule'][index]:raise ValueError('rule target mismatch')
                count+=len(rows)
    if count!=meta['candidate_count']:raise ValueError('candidate count mismatch')
    save(out/'VALIDATION.json',{'status':'PASS','unique_inputs':len(seen),'candidates':count,'metadata_sha256':sha(out/'METADATA.json')});print('VALIDATED',len(seen),count,flush=True)
if __name__=='__main__':main()

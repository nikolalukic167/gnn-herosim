"""Exhaustive local-observation ordering witness with full actual live replay."""
import argparse,copy,hashlib,io,json
from itertools import permutations
from pathlib import Path
import numpy as np
import torch
from src.placement.radical.environment import problem,initial,pack,serialized
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import replay,herosim
from src.policy.mixed.model import features
from src.policy.dispatch_priority.model import PriorityNet
from scripts_cosim.radical_physics_screen import save,sha
from scripts_cosim.workflow_proposal_gate import configure_environment
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    if out.exists():raise ValueError('preserve witness')
    out.mkdir(parents=True);engine=DispatchNative(out/'build');torch.set_num_threads(1);configure_environment()
    protocol=json.loads((ROOT/'experiments/mixed_dispatch_witness_v1.json').read_text())
    sources={p:sha(ROOT/p) for p in ['experiments/mixed_dispatch_witness_v1.json','scripts_cosim/mixed_dispatch_witness.py','src/policy/dispatch_priority/model.py','src/policy/mixed/model.py','src/placement/radical/dispatch.py','src/placement/radical/dispatch.cpp','src/placement/radical/dispatch_live.py','src/placement/radical/environment.py','src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/coordinator.py','src/placement/simulation.py','src/placement/infrastructure.py','scripts_cosim/mixed_physics_live.py','scripts_cosim/workflow_live.py']}
    save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':engine.provenance});cases=[];identities=set()
    lo,hi=protocol['seeds']
    for seed in range(lo,hi+1):
        first=problem(seed,jobs=2,ops=3);second=copy.deepcopy(first)
        for key in ('types','locks'):second[key][:,1:]=second[key][:,1:][:,::-1]
        variants=[]
        for v,b in enumerate((first,second)):
            identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
            if identity in identities:raise ValueError('duplicate physical witness')
            identities.add(identity);path=out/f'{seed}_{v}.json';save(path,serialized(b));variants.append({'b':b,'path':path,'sha256':sha(path),'identity':identity})
        cases.append((seed,variants))
    save(out/'input_validation.json',{'status':'PASS','unique_inputs':len(identities),'inputs':{f'{seed}_{v}':{k:d[k] for k in ('sha256','identity')} for seed,vs in cases for v,d in enumerate(vs)}})
    rows=[];runs=tasks=0
    for seed,variants in cases:
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed')
        b0,b1=[d['b'] for d in variants];a=initial(b0,'fastest');x0,e0,adj0=features(b0);x1,e1,adj1=features(b1)
        if not np.array_equal(x0[[0,3]],x1[[0,3]]):raise ValueError('root observations differ')
        torch.manual_seed(991);gnn=PriorityNet(x0.shape[-1],mp=True).eval();off=PriorityNet(x0.shape[-1],mp=False).eval()
        h0,_,_=features(b0,True);h1,_,_=features(b1,True);hand=PriorityNet(h0.shape[-1],mp=False).eval()
        differences={}
        with torch.inference_mode():
            for name,model,left,right in [('gnn',gnn,x0,x1),('mpoff',off,x0,x1),('hand_graph',hand,h0,h1)]:
                l=model(torch.tensor(left)[None],torch.tensor(adj0)[None])[0,[0,3]];r=model(torch.tensor(right)[None],torch.tensor(adj1)[None])[0,[0,3]]
                differences[name]=float(torch.max(torch.abs(l-r)))
        if differences['mpoff']!=0:raise ValueError('MP-OFF root changed')
        minima=[];cell=out/str(seed);cell.mkdir()
        for v,d in enumerate(variants):
            if sha(d['path'])!=d['sha256'] or hashlib.sha256(pack(d['b']).tobytes()).hexdigest()!=d['identity']:raise ValueError('input changed')
            b=d['b'];best=[(float('inf'),None),(float('inf'),None)]
            with (cell/f'placements_{v}.jsonl').open('w') as f:
                for perm in permutations(range(6)):
                    priority=np.array(perm,dtype=np.int64).reshape(2,3);root=int(priority[1,0]<priority[0,0]);c,_=engine.score_priority(b,a,priority)
                    f.write(json.dumps({'placement_plan':a.tolist(),'priority':priority.tolist(),'first_root':root,'objective':c})+'\n')
                    if c<best[root][0]:best[root]=(c,priority.copy())
            minima.append([c for c,_ in best])
            with (cell/f'live_{v}.jsonl').open('w') as output,(cell/f'simulation_{v}.log').open('w') as log:
                for root,(c,priority) in enumerate(best):
                    actual=herosim(b,a,priority,log,seed);reference=replay(b,a,priority)
                    if reference['objective']!=c or abs(actual['total_rtt']*1000-c)>1e-6:raise ValueError('live cost mismatch')
                    for task in actual['tasks']:
                        j,k=map(int,task['taskType']['name'][2:].split('_'))
                        if abs(task['doneTime']*1000-reference['ends'][j,k])>1e-6:raise ValueError('live completion mismatch')
                    if len(actual['tasks'])!=6:raise ValueError('incomplete witness')
                    output.write(json.dumps({'first_root':root,'priority':priority.tolist(),'stats':actual},allow_nan=False)+'\n');runs+=1;tasks+=6
        flip=(minima[0][0]-minima[0][1])*(minima[1][0]-minima[1][1])<0
        rows.append({'seed':seed,'conditional_minima':minima,'strict_flip':flip,'root_logit_differences':differences});print('WITNESS',seed,flip,minima,flush=True)
    result={'rows':rows,'strict_flips':sum(r['strict_flip'] for r in rows),'gnn_distinguishes_flips':sum(r['strict_flip'] and r['root_logit_differences']['gnn']>1e-7 for r in rows),'hand_distinguishes_flips':sum(r['strict_flip'] and r['root_logit_differences']['hand_graph']>1e-7 for r in rows),'enumerated_plans':len(rows)*2*720,'live_runs':runs,'completed_operations':tasks,'status':'PASS'}
    save(out/'read.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__':main()

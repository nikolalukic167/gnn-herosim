"""Generate fingerprinted heuristic-labeled workflow data with full searched-plan traces."""
import argparse,hashlib,json,os,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.placement.workflow_planning import instance,plan_observer,evaluate
from src.policy.workflow.model import features,CONTRACT
from scripts_cosim.workflow_qualification import methods


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_case(task):
    seed,split,out=task;out=Path(out);cell=out/f'ds_{seed}';(cell/'placements').mkdir(parents=True)
    p=instance(seed);seen=set();count=0
    with (cell/'placements/placements.jsonl').open('w') as trace:
        def record(a,cost):
            nonlocal count
            key=a.tobytes()
            if key in seen:return
            seen.add(key);count+=1
            trace.write(json.dumps({'placement_plan':a.tolist(),'rtt_ms':cost},separators=(',',':'),allow_nan=False)+'\n')
        token=plan_observer.set(record)
        try:read=methods(p)
        finally:plan_observer.reset(token)
    teacher=min(read,key=lambda k:read[k]['cost_ms']);labels=np.array(read[teacher]['assignment'],dtype=np.int64)
    if abs(evaluate(p,labels)-read[teacher]['cost_ms'])>1e-7:raise RuntimeError('teacher cost mismatch')
    problem={'seed':seed,'processing_ms':[[[float(v) if np.isfinite(v) else None for v in op] for op in job] for job in p]}
    (cell/'problem.json').write_text(json.dumps(problem,allow_nan=False)+'\n')
    (cell/'best.json').write_text(json.dumps({'kind':'best_observed_not_certified_optimum','teacher':teacher,'rtt_ms':read[teacher]['cost_ms'],'placement_plan':labels.tolist(),'unique_complete_search_plans':count,'split':split,'methods':read},allow_nan=False)+'\n')
    x,e=features(p);xh,_=features(p,True)
    return {'seed':seed,'split':split,'p':p,'labels':labels,'x':x,'xh':xh,'eligible':e,'teacher':read[teacher]['cost_ms'],'source_sha256':sha(cell/'problem.json'),'placement_sha256':sha(cell/'placements/placements.jsonl'),'instance_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'trace_count':count}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.out.exists():raise ValueError('preserve existing corpus; output must be new')
    args.out.mkdir(parents=True)
    protocol=json.loads((ROOT/'experiments/workflow_amortized_v1_protocol.json').read_text())
    sources={p:sha(ROOT/p) for p in ['scripts_cosim/generate_workflow_data.py','src/placement/workflow_planning.py','src/policy/workflow/model.py','scripts_cosim/workflow_qualification.py','experiments/workflow_amortized_v1_protocol.json']}
    (args.out/'protocol_before_run.json').write_text(json.dumps({'protocol':protocol,'sources':sources,'smoke':args.smoke},indent=2)+'\n')
    tasks=[]
    for split,key in [('train','train_seeds'),('validation','validation_seeds'),('test','sealed_test_seeds')]:
        lo,hi=protocol[key];seeds=list(range(lo,hi+1));seeds=seeds[:2] if args.smoke else seeds
        tasks += [(s,split,str(args.out)) for s in seeds]
    if len({s for s,_,_ in tasks})!=len(tasks):raise ValueError('overlapping splits')
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(make_case,tasks,chunksize=1):
            rows.append(row)
            if len(rows)%32==0 or args.smoke:print(f'{len(rows)}/{len(tasks)} cases in {time.monotonic()-start:.1f}s',flush=True)
    if len({r['instance_sha256'] for r in rows})!=len(rows):raise RuntimeError('duplicate source instances')
    manifest=[]
    for split in ('train','validation','test'):
        subset=[r for r in rows if r['split']==split]
        np.savez_compressed(args.out/f'{split}.npz',p=np.stack([r['p'] for r in subset]),labels=np.stack([r['labels'] for r in subset]),x=np.stack([r['x'] for r in subset]),xh=np.stack([r['xh'] for r in subset]),eligible=np.stack([r['eligible'] for r in subset]),teacher=np.array([r['teacher'] for r in subset]),seeds=np.array([r['seed'] for r in subset]))
    for r in rows:manifest.append({k:r[k] for k in ['seed','split','source_sha256','placement_sha256','instance_sha256','trace_count']})
    for p,h in sources.items():
        if sha(ROOT/p)!=h:raise RuntimeError('source changed during generation')
    meta={'contract':CONTRACT,'label':'best_observed_complete_workflow_RTT_not_optimal','units':'milliseconds','smoke':args.smoke,'sources':sources,'cases':manifest,'files':{f'{s}.npz':sha(args.out/f'{s}.npz') for s in ('train','validation','test')},'wall_s':time.monotonic()-start}
    (args.out/'METADATA.json').write_text(json.dumps(meta,indent=2)+'\n');print('complete',len(rows),flush=True)
if __name__=='__main__':main()

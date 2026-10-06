"""Pre-training fresh-seed quality/runtime gate and real-engine parity checks."""
import argparse,hashlib,io,json,os,time
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.placement.workflow_planning import instance,initial,rollout,beam,anneal,route_greedy,evaluate
from scripts_cosim.workflow_live import run,substrate

def assignment(p,state):
    out=np.zeros(p.shape[:2],dtype=int)
    for j,k,h in state[3]:out[j,k]=h
    return out

def methods(p,teacher=True):
    result={}
    for name,func in [('ect',lambda:rollout(initial(p),p))]+[(f'load_{weight}',lambda weight=weight:route_greedy(p,weight)) for weight in (.1,.25,.5,1.)]+[(f'beam{width}',lambda width=width:beam(p,width)) for width in (1,2,4,16,64)]+([('roll16',lambda:beam(p,16,True))] if teacher else []):
        st=time.perf_counter();state=func();a=assignment(p,state);ms=1000*(time.perf_counter()-st)
        if abs(state[2]-evaluate(p,a))>1e-7:raise RuntimeError('plan replay differs from search')
        result[name]={'cost_ms':state[2],'planning_ms':ms,'assignment':a.tolist(),'plan':state[3]}
    if teacher:
        start=min(result.values(),key=lambda r:r['cost_ms']);st=time.perf_counter();cost,a=anneal(p,start['plan'],0,2000)
        result['anneal']={'cost_ms':cost,'planning_ms':1000*(time.perf_counter()-st)+sum(r['planning_ms'] for r in result.values()),'assignment':a.tolist()}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('preserve previous gate')
    args.out.mkdir(parents=True);protocol=json.loads((ROOT/'experiments/workflow_amortized_v1_protocol.json').read_text())
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/workflow_amortized_v1_protocol.json','scripts_cosim/workflow_qualification.py','scripts_cosim/workflow_live.py','src/placement/workflow_planning.py','src/placement/infrastructure.py','src/placement/availability.py','src/policy/workflow/model.py']}
    (args.out/'protocol_before_run.json').write_text(json.dumps({'protocol':protocol,'sources':sources},indent=2)+'\n')
    for k in list(os.environ):
        if k.startswith(('HEROSIM_','GNN_','LIVE_AUDIT_','COSIM_')):os.environ.pop(k)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_PG_DRAIN_CONTRACT='availability_v2',SIM_FORCE_FULL_STATS='1')
    rows=[];runs=0
    for seed in protocol['gate_seeds']:
        p=instance(seed);read=methods(p)
        cheap={k:v for k,v in read.items() if v['planning_ms']<=protocol['planning_budget_ms']}
        if not cheap:raise RuntimeError('no baseline fits budget')
        baseline=min(cheap,key=lambda k:cheap[k]['cost_ms']);teacher=min(read,key=lambda k:read[k]['cost_ms'])
        cell=args.out/str(seed);(cell/'placements').mkdir(parents=True)
        config,inputs=substrate(p/1000)
        (cell/'input.json').write_text(json.dumps({'config':config,'inputs':inputs},indent=2,allow_nan=False)+'\n')
        checked={}
        with (cell/'simulation.log').open('w') as log,(cell/'placements/placements.jsonl').open('w') as out:
            for name in dict.fromkeys(['ect',baseline,teacher]):
                a=np.array(read[name]['assignment']);stat=run(p/1000,a,log,seed);runs+=1
                if abs(stat['job_completion_sum']*1000-read[name]['cost_ms'])>1e-6 or abs(stat['total_rtt']-stat['job_completion_sum'])>1e-8:raise RuntimeError('complete live objective mismatch')
                checked[name]=stat['job_completion_sum']*1000;out.write(json.dumps({'arm':name,'placement_plan':a.tolist(),'rtt':stat['total_rtt'],'stats':stat},allow_nan=False)+'\n')
        row={'seed':seed,'methods':read,'best_within_budget':baseline,'teacher':teacher,'gain_pct':100*(1-read[teacher]['cost_ms']/read[baseline]['cost_ms']),'live_checked':checked}
        (cell/'read.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
        print(seed,'gain',round(row['gain_pct'],2),'baseline',baseline,'teacher',teacher,flush=True)
    gains=np.array([r['gain_pct'] for r in rows]);rng=np.random.default_rng(1);lo=float(np.quantile(np.median(rng.choice(gains,(10000,len(gains))),axis=1),.025));median=float(np.median(gains))
    outcome={'rows':rows,'live_runs':runs,'median_gain_pct':median,'bootstrap_median_lower_pct':lo,'verdict':'ADVANCE_TO_PILOT' if median>=5 and lo>0 else 'DO_NOT_TRAIN','sources':sources}
    (args.out/'read.json').write_text(json.dumps(outcome,indent=2)+'\n');print(outcome['verdict'],median,lo,flush=True)
if __name__=='__main__':main()

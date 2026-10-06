"""Exhaustive paired witnesses for future-order setup and shared-resource effects."""
import argparse,copy,itertools,json,hashlib
from pathlib import Path
import numpy as np
from src.placement.radical.environment import Native,problem,serialized
from src.placement.radical.live import run
from scripts_cosim.radical_physics_screen import save,sha
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    if out.exists():raise ValueError('preserve earlier evidence')
    out.mkdir(parents=True);native=Native(out/'build');protocol=json.loads((ROOT/'experiments/radical_physics_v1_witness.json').read_text());paths=['experiments/radical_physics_v1_witness.json','scripts_cosim/radical_physics_witness.py','src/placement/radical/kernel.cpp','src/placement/radical/environment.py','src/placement/radical/live.py'];sources={p:sha(ROOT/p) for p in paths};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':native.provenance});report={};total=0
    for mechanism in protocol['mechanisms']:
        rows=[];lo,hi=protocol['seeds']
        for seed in range(lo,hi+1):
            b=problem(seed,*protocol['shape']);other=copy.deepcopy(b);indices=[(j,k) for j in range(3) for k in range(1,3)];perm=np.random.default_rng(seed+1).permutation(len(indices))
            for key in ('types','locks'):
                original=[b[key][j,k] for j,k in indices]
                for dst,source in zip(indices,perm):other[key][dst]=original[source]
                if not np.array_equal(b[key][:,0],other[key][:,0]) or sorted(b[key].ravel())!=sorted(other[key].ravel()):raise RuntimeError('local observation or histogram changed')
            if not np.array_equal(b['p'],other['p']):raise RuntimeError('processing changed')
            pair=[]
            for arm,case in (('original',b),('rewired',other)):
                cell=out/mechanism/str(seed)/arm;(cell/'placements').mkdir(parents=True);save(cell/'input.json',serialized(case));allowed=np.argsort(case['p'],axis=-1)[...,:2];roots=list(map(int,allowed[0,0]));best={h:None for h in roots}
                with (cell/'placements/placements.jsonl').open('w') as f:
                    for bits in itertools.product((0,1),repeat=9):
                        a=np.take_along_axis(allowed,np.array(bits).reshape(3,3,1),axis=-1)[...,0];cost,_=native.score(case,a,mechanism);total+=1;root=int(a[0,0]);f.write(json.dumps({'placement_plan':a.tolist(),'objective':cost})+'\n')
                        if best[root] is None or cost<best[root]['cost']:best[root]={'cost':cost,'plan':a.tolist()}
                live={}
                for h,r in best.items():
                    live[str(h)]=run(case,np.array(r['plan']),mechanism)
                    if abs(live[str(h)]['objective']-r['cost'])>1e-7:raise RuntimeError('exact/native live mismatch')
                save(cell/'live.json',live);choice=min(roots,key=lambda h:best[h]['cost']);gap=max(x['cost'] for x in best.values())/best[choice]['cost']-1
                r={'best_by_root':best,'choice':choice,'wrong_root_regret_pct':100*gap};save(cell/'read.json',r);pair.append(r)
            flip=pair[0]['choice']!=pair[1]['choice'] and min(r['wrong_root_regret_pct'] for r in pair)>1e-8
            rows.append({'seed':seed,'flips_unique_optimal_root':flip,'cases':pair})
        report[mechanism]={'pairs':len(rows),'strict_flips':sum(r['flips_unique_optimal_root'] for r in rows),'rows':rows};print(mechanism,'strict flips',report[mechanism]['strict_flips'],'/',len(rows),flush=True)
    save(out/'read.json',{'scope':'small constructive witness, solved exactly without learning','results':report,'enumerated_placements':total,'simpy_replays':len(protocol['mechanisms'])*16*4,'sources':sources})
if __name__=='__main__':main()

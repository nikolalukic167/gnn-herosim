"""CPU-only size screen with certified optimization bounds and live plan replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts_cosim import graph_decision_witness as w
from scripts_cosim import graph_three_host_screen as small


def exact_bound(cfg, edges, seconds):
    n, k = cfg['tasks'], cfg['hosts']
    count = n*k + len(edges)
    objective = np.zeros(count)
    if 'unary_seconds' in cfg:
        objective[:n*k] = np.asarray(cfg['unary_seconds']).ravel()
    lower = np.zeros(count); upper = np.ones(count)
    for i, domain in enumerate(w.domains_for(cfg)):
        for h in range(k):
            if h not in domain:
                upper[i*k+h] = 0
    # x assigns each task one host; z is one iff an edge is cut.
    matrix = lil_matrix((n + 2*k*len(edges), count))
    lo = np.full(matrix.shape[0], -np.inf); hi = np.zeros(matrix.shape[0])
    for i in range(n):
        matrix[i, i*k:(i+1)*k] = 1; lo[i] = hi[i] = 1
    row = n
    for e, (i,j,weight) in enumerate(edges):
        z = n*k+e; objective[z] = w.exchange_cost(cfg, weight)
        for h in range(k):
            for a,b in ((i,j),(j,i)):
                matrix[row,a*k+h] = 1; matrix[row,b*k+h] = -1; matrix[row,z] = -1
                row += 1
    result = milp(objective, integrality=np.r_[np.ones(n*k),np.zeros(len(edges))],
                  bounds=Bounds(lower,upper), constraints=LinearConstraint(matrix.tocsr(),lo,hi),
                  options={'time_limit':seconds,'mip_rel_gap':0.0})
    if result.status not in (0,1):
        raise RuntimeError(f'optimization failed: {result.message}')
    plan = None
    if result.x is not None:
        assignment = result.x[:n*k].reshape(n,k)
        if not np.allclose(assignment,np.round(assignment),atol=1e-6,rtol=0):
            raise RuntimeError('noninteger solver incumbent')
        plan = assignment.argmax(1).tolist()
        if any(h not in d for h,d in zip(plan,w.domains_for(cfg))):
            raise RuntimeError('infeasible solver incumbent')
        if abs(w.energy(cfg,edges,plan)-result.fun)>1e-6:
            raise RuntimeError('solver and physical objective disagree')
    bound = getattr(result,'mip_dual_bound',None)
    return {'status':int(result.status),'certified':result.status==0,
            'lower_bound':float(bound) if bound is not None and np.isfinite(bound) else None,
            'plan':plan,'incumbent':float(result.fun) if plan is not None else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=ROOT/'experiments/graph_size_screen_v1.json')
    ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args(); protocol_cfg=json.loads(args.config.read_text())
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('output must be empty')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for key in list(os.environ):
        if key.startswith(('HEROSIM_','GNN_','LIVE_AUDIT_','COSIM_')):
            os.environ.pop(key)
    os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_PG_DRAIN_CONTRACT='availability_v2',
                      SIM_FORCE_FULL_STATS='1',COSIM_SUPPRESS_SIM_PRINTS='1')
    paths=[Path(__file__),args.config,ROOT/'scripts_cosim/graph_three_host_screen.py',
           ROOT/'scripts_cosim/graph_decision_witness.py',ROOT/'scripts_cosim/peer_lookahead_live_probe.py',
           ROOT/'src/placement/infrastructure.py',ROOT/'src/placement/availability.py',
           ROOT/'src/policy/peer_greedy_network/scheduler.py',ROOT/'src/executecosimulation.py',ROOT/'src/eventgenerator.py']
    hashes={str(p.resolve().relative_to(ROOT)):w.digest(p) for p in paths}
    (args.out_dir/'protocol_before_run.json').write_text(json.dumps({'config':protocol_cfg,'sources':hashes},indent=2)+'\n')
    results=[]; runs=0; started=time.monotonic()
    for n in protocol_cfg['sizes']:
        for seed in protocol_cfg['graph_seeds']:
            if time.monotonic()-started>protocol_cfg['total_timeout_s']:
                raise TimeoutError('screen budget exceeded')
            cfg={**protocol_cfg,'tasks':n,'anchors':{str(n-3):0,str(n-2):1,str(n-1):2},
                 'edge_probability':protocol_cfg['extra_degree']/(n-1),'seed':seed}
            edges=small.make_edges(cfg,seed); controls={}; timings={}
            for name in ('immediate_paper','peer_mass','min_sum_3','min_sum_9'):
                t=time.perf_counter(); controls[name]=w.control_plan(cfg,edges,name)
                timings[name]=1000*(time.perf_counter()-t)
            starts=[controls['immediate_paper'],controls['peer_mass']]+[[d[0] if len(d)==1 else h for d in w.domains_for(cfg)] for h in range(3)]
            t=time.perf_counter(); controls['cd_without_mp']=small.coordinate_descent(cfg,edges,starts)
            timings['cd_without_mp']=1000*(time.perf_counter()-t)+timings['immediate_paper']+timings['peer_mass']
            t=time.perf_counter(); bound=exact_bound(cfg,edges,protocol_cfg['solver_timeout_s'])
            solver_ms=1000*(time.perf_counter()-t)
            if bound['plan'] is not None:
                controls['solver']=bound['plan']; timings['solver']=solver_ms
            cell=args.out_dir/f'n{n}_s{seed}'; (cell/'placements').mkdir(parents=True)
            infra,inputs=w.substrate(cfg)
            (cell/'input.json').write_text(json.dumps({'config':cfg,'edges':edges,'infrastructure':infra,'inputs':inputs},indent=2)+'\n')
            by_plan={}; rows=[]
            with (cell/'simulation.log').open('w') as log,(cell/'placements/placements.jsonl').open('w') as out:
                for p in controls.values():
                    key=tuple(p)
                    if key in by_plan: continue
                    stat=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],p,log);runs+=1
                    if abs(stat['exchange']-w.energy(cfg,edges,p))>1e-7 or stat['queue']>1e-7:
                        raise RuntimeError('live objective mismatch')
                    by_plan[key]=stat;row={'placement_plan':p,'stats':stat,'rtt':stat['total_rtt']};rows.append(row)
                    out.write(json.dumps(row,allow_nan=False)+'\n')
                rule=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],None,log,immediate=True);runs+=1
                off=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],controls['peer_mass'],log,exchange=False);runs+=1
            constants=[r['rtt']-r['stats']['exchange'] for r in rows]
            if max(constants)-min(constants)>1e-7 or off['exchange'] or off['rendezvous']:
                raise RuntimeError('live reduction or exchange-off check failed')
            constant=constants[0]
            if abs(rule['total_rtt']-rule['exchange']-constant)>1e-7 or rule['queue']>1e-7:
                raise RuntimeError('immediate live objective mismatch')
            read={'tasks':n,'seed':seed,'edges':edges,'bound':bound,'solver_wall_ms':solver_ms,
                  'constant_rtt':constant,'controls':{name:{'plan':p,'exchange':w.energy(cfg,edges,p),
                  'live_rtt':by_plan[tuple(p)]['total_rtt'],'wall_ms':timings[name]} for name,p in controls.items()},
                  'immediate_live':rule,'exchange_off':off,'live_unique_plans':len(rows),
                  'sweep_kind':'selected_control_plans_not_exhaustive',
                  'sweep_sha256':w.digest(cell/'placements/placements.jsonl')}
            (cell/'read.json').write_text(json.dumps(read,indent=2,allow_nan=False)+'\n');results.append(read)
            print(f'n={n} seed={seed} certified={bound["certified"]} solver={solver_ms:.1f}ms CD={read["controls"]["cd_without_mp"]["exchange"]:.2f} bound={bound["lower_bound"]}',flush=True)
    for path,h in hashes.items():
        if w.digest(ROOT/path)!=h: raise RuntimeError('source changed during run')
    (args.out_dir/'read.json').write_text(json.dumps({'config':protocol_cfg,'rows':results,'live_runs':runs,
        'wall_s':time.monotonic()-started,'sources':hashes},indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()

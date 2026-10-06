"""Live checks of exploratory winners plus four fixed fresh heterogeneous cases."""
import argparse,json,os,time
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts_cosim import graph_unary_screen as search, graph_decision_witness as w, graph_size_screen as big, graph_three_host_screen as small

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out-name',default='live');ap.add_argument('--start-index',type=int,default=0);ap.add_argument('--run-timeout',type=int,default=10);args=ap.parse_args();out=args.root/args.out_name
 if out.exists():raise ValueError('preserve outputs')
 out.mkdir()
 cases=[]
 for f in ('exploratory_24_48.json','exploratory_128.json'):
  rows=json.loads((args.root/f).read_text())['rows'];cases+=sorted(rows,key=lambda r:r['gain_pct'],reverse=True)[:2]
 base=cases[0]['cfg']
 for seed in range(50600,50604):
  cfg={**base,'seed':seed,'tasks':48,'anchors':{},'edge_probability':4/47}
  cfg['unary_seconds']=(np.random.default_rng(seed+1000).integers(0,101,(48,3))*.8).tolist()
  cases.append({'cfg':cfg,'edges':small.make_edges(cfg,seed)})
 cases=cases[args.start_index:]
 for case in cases:case['cfg']['run_timeout_s']=args.run_timeout
 paths=['scripts_cosim/'+name for name in ('graph_unary_live_check.py','graph_unary_screen.py','graph_decision_witness.py','graph_three_host_screen.py','graph_size_screen.py','peer_lookahead_live_probe.py')]+['src/placement/infrastructure.py','src/placement/availability.py','src/executecosimulation.py','src/eventgenerator.py','src/policy/peer_greedy_network/scheduler.py']
 hashes={p:w.digest(ROOT/p) for p in paths}
 (out/'protocol_before_run.json').write_text(json.dumps({'cases':cases,'source_sha256':hashes,'selection':'two largest exploratory gaps per size block plus four fixed fresh seeds'},indent=2)+'\n')
 for k in list(os.environ):
  if k.startswith(('HEROSIM_','GNN_','LIVE_AUDIT_','COSIM_')):os.environ.pop(k)
 os.environ.update(HEROSIM_PEER_EXCHANGE='1',HEROSIM_PG_DRAIN_CONTRACT='availability_v2',SIM_FORCE_FULL_STATS='1',COSIM_SUPPRESS_SIM_PRINTS='1')
 results=[];runs=0
 for row in cases:
  cfg,edges=row['cfg'],row['edges'];n=cfg['tasks'];seed=cfg['seed'];local=np.argmin(cfg['unary_seconds'],axis=1).tolist()
  t=time.perf_counter();cd=small.coordinate_descent(cfg,edges,[local]+[[h]*n for h in range(3)]);exp=search.expand(cfg,edges,cd);searchms=1000*(time.perf_counter()-t)
  t=time.perf_counter();bound=big.exact_bound(cfg,edges,1);solverms=1000*(time.perf_counter()-t)
  if not bound['certified']:raise RuntimeError('live certificate unresolved')
  constant=min(([h]*n for h in range(3)),key=lambda p:w.energy(cfg,edges,p))
  controls={'local_execution':local,'best_constant_host':constant,'cd':cd,'expansion':exp,'exact':bound['plan']}
  infra,inputs=search.unary_substrate(cfg);cell=out/f'n{n}_s{seed}';(cell/'placements').mkdir(parents=True)
  (cell/'input.json').write_text(json.dumps({'cfg':cfg,'edges':edges,'infrastructure':infra,'inputs':inputs},indent=2)+'\n')
  byplan={}
  with (cell/'simulation.log').open('w') as log,(cell/'placements/placements.jsonl').open('w') as sweep:
   for p in controls.values():
    if tuple(p) in byplan:continue
    stat=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],p,log);runs+=1
    execution=sum(cfg['unary_seconds'][i][h] for i,h in enumerate(p))
    if abs(sum(t['executionTime'] for t in stat['tasks'])-execution-n*cfg['execution_s'])>1e-6:raise RuntimeError('execution mismatch')
    if abs(stat['exchange']+execution-w.energy(cfg,edges,p))>1e-6 or stat['queue']>1e-6:raise RuntimeError('objective mismatch')
    byplan[tuple(p)]=stat;sweep.write(json.dumps({'placement_plan':p,'rtt':stat['total_rtt'],'stats':stat},allow_nan=False)+'\n')
   rule=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],None,log,immediate=True);runs+=1
   off=w.live(cfg,infra,inputs,edges,cfg['arrival_gap_s'],bound['plan'],log,exchange=False);runs+=1
  constants=[s['total_rtt']-w.energy(cfg,edges,p) for p,s in byplan.items()]
  if max(constants)-min(constants)>1e-6 or off['exchange'] or off['rendezvous']:raise RuntimeError('reduction failed')
  read={'tasks':n,'seed':seed,'bound':bound,'search_ms':searchms,'solver_ms':solverms,'controls':{name:{'plan':p,'rtt':byplan[tuple(p)]['total_rtt'],'objective':w.energy(cfg,edges,p)} for name,p in controls.items()},'immediate_live':rule,'exchange_off':off,'constant_rtt':constants[0],'sweep_sha256':w.digest(cell/'placements/placements.jsonl')}
  read['expansion_gap_pct']=100*(read['controls']['expansion']['rtt']/read['controls']['exact']['rtt']-1)
  results.append(read);(cell/'read.json').write_text(json.dumps(read,indent=2)+'\n')
  print(n,seed,'live residual %',read['expansion_gap_pct'],flush=True)
 for p,h in hashes.items():
  if w.digest(ROOT/p)!=h:raise RuntimeError('source changed')
 (out/'read.json').write_text(json.dumps({'rows':results,'live_runs':runs,'source_sha256':hashes},indent=2)+'\n')
if __name__=='__main__':main()

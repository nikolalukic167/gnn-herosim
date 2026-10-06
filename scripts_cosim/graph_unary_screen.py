"""Exploratory heterogeneous-execution versus peer-exchange screen; no training."""
import argparse,copy,json,time,itertools,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from scripts_cosim import graph_size_screen as big, graph_three_host_screen as small, graph_decision_witness as w

def expand(cfg,edges,initial):
 p=list(initial); unary=np.asarray(cfg['unary_seconds']);n=len(p);moves=0
 while True:
  improved=False
  for a in range(cfg['hosts']):
   d0=unary[np.arange(n),p].copy(); d1=unary[:,a].copy(); cap=np.zeros((n+2,n+2),dtype=np.int64)
   for i,j,wt in edges:
    weight=w.exchange_cost(cfg,wt); v00=weight*(p[i]!=p[j]);v01=weight*(p[i]!=a);v10=weight*(a!=p[j]); c=(v01+v10-v00)/2
    d1[i]+=v10-v00-c;d1[j]+=v01-v00-c
    cap[i,j]+=round(c*10000);cap[j,i]+=round(c*10000)
   for i in range(n):
    m=min(d0[i],d1[i]);cap[n,i]=round((d1[i]-m)*10000);cap[i,n+1]=round((d0[i]-m)*10000)
   flow=maximum_flow(csr_matrix(cap),n,n+1);res=cap-flow.flow.toarray();seen={n};stack=[n]
   while stack:
    for j in np.flatnonzero(res[stack.pop()]>0):
     if int(j) not in seen:seen.add(int(j));stack.append(int(j))
   q=[p[i] if i in seen else a for i in range(n)]
   if w.energy(cfg,edges,q)<w.energy(cfg,edges,p)-1e-7:p=q;improved=True;moves+=1
  if not improved:return p

def unary_substrate(cfg):
 infra,inputs=w.substrate(cfg)
 base=copy.deepcopy(inputs['platform_types'])
 inputs['platform_types']={}
 for h,node in enumerate(infra['nodes'][1:]):
  node['platforms']=[f'hostcpu{h}' for i in range(cfg['tasks'])]
 for i in range(cfg['tasks']):
  task=inputs['task_types'][f'w{i}']; names=[f'hostcpu{h}' for h in range(cfg['hosts'])]
  task['platforms']=names
  for key in ('memoryRequirements','coldStartDuration','executionTime','energy'):
   task[key]={name:(cfg['execution_s']+cfg['unary_seconds'][i][h] if key=='executionTime' else .01 if key=='memoryRequirements' else 0.) for h,name in enumerate(names)}
  for name in names: inputs['platform_types'][name]={**base[f'wcpu{i}'],'name':name,'shortName':name}
 return infra,inputs

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.exists():raise ValueError('preserve old output')
 base=json.loads(Path('experiments/graph_size_screen_v1.json').read_text()); rows=[];started=time.monotonic()
 for n,scale,seed in list(itertools.product([24,48],[.5,2,8],range(48400,48408)))+list(itertools.product([128],[4,8,16],range(49500,49504))):
  cfg={**base,'tasks':n,'anchors':{},'edge_probability':4/(n-1),'seed':seed}
  rng=np.random.default_rng(seed+1000);cfg['unary_seconds']=(rng.integers(0,101,(n,3))*scale*.1).tolist()
  edges=small.make_edges(cfg,seed)
  t=time.perf_counter(); local=np.argmin(cfg['unary_seconds'],axis=1).tolist();cd=small.coordinate_descent(cfg,edges,[local]+[[h]*n for h in range(3)]);cdms=(time.perf_counter()-t)*1000
  t=time.perf_counter();ex=expand(cfg,edges,cd);exms=cdms+(time.perf_counter()-t)*1000
  t=time.perf_counter();bound=big.exact_bound(cfg,edges,1.);solverms=(time.perf_counter()-t)*1000
  score=w.energy(cfg,edges,ex);best=bound['incumbent'];gain=100*(score-best)/score if best is not None else None
  row={'cfg':cfg,'edges':edges,'cd':cd,'expansion':ex,'cd_ms':cdms,'expansion_ms':exms,'bound':bound,'solver_ms':solverms,'gain_pct':gain};rows.append(row)
  print(n,scale,seed,'gap',round(gain,3) if gain is not None else None,'cert',bound['certified'],'times',round(exms,1),round(solverms,1),flush=True)
 args.out.write_text(json.dumps({'rows':rows,'wall_s':time.monotonic()-started},indent=2))

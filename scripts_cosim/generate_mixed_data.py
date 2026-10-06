"""Generate disjoint complete-workflow training data with full search traces."""
import argparse,ctypes,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from src.placement.radical.environment import problem,initial,serialized,pack
from src.placement.radical.mixed import MixedNative
from src.policy.mixed.model import features,CONTRACT
from scripts_cosim.radical_physics_screen import save,sha
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
 if out.exists():raise ValueError('preserve previous corpus')
 out.mkdir(parents=True);engine=MixedNative(out/'build');protocol=json.loads((ROOT/'experiments/mixed_physics_learning_v1_protocol.json').read_text());src=ROOT/'src/placement/radical/trace.cpp';library=out/'build/trace.so';subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(library)],check=True);lib=ctypes.CDLL(str(library.resolve()));f=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');a=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');i=ctypes.c_int;lib.mixed_trace.argtypes=[f,a,i,i,i,i,i,i,ctypes.c_uint64,ctypes.c_char_p];lib.mixed_trace.restype=ctypes.c_double
 sources={p:sha(ROOT/p) for p in ['experiments/mixed_physics_learning_v1_protocol.json','scripts_cosim/generate_mixed_data.py','src/policy/mixed/model.py','src/placement/radical/environment.py','src/placement/radical/mixed.py','src/placement/radical/mixed.cpp','src/placement/radical/kernel.cpp','src/placement/radical/trace.cpp']};meta={'contract':CONTRACT,'physics':'mixed_execution_v1','queue_feature_contract':'declared_manifest_no_queue','warmth_physics':'all_warm_zero_io','sources':sources,'cases':[],'files':{},'native':engine.provenance,'trace_library_sha256':sha(library)};save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources,'native':engine.provenance});identities=set()
 # Freeze and check every actual input identity before label generation.
 cases={}
 for split,(lo,hi) in protocol['data'].items():
  cases[split]=[]
  for seed in range(lo,hi+1):
   b=problem(seed);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
   if identity in identities:raise ValueError('duplicate physical environment')
   identities.add(identity);cell=out/f'ds_{seed}';(cell/'placements').mkdir(parents=True);save(cell/'problem.json',serialized(b));cases[split].append((seed,b,identity))
 save(out/'source_validation.json',{'status':'PASS','actual_unique':len(identities),'identities':sorted(identities)})
 for split,records in cases.items():
  cache={k:[] for k in ('seeds','x','xh','eligible','adj','labels','teacher')}
  for number,(seed,b,identity) in enumerate(records):
   cell=out/f'ds_{seed}';trace=cell/'placements/placements.jsonl';candidates=[]
   for start in ('fastest','load'):
    q,plan,dims=engine.checked(b,initial(b,start));cost=float(lib.mixed_trace(q,plan,*dims,8192,77,str(trace).encode()))
    if not np.isfinite(cost) or cost!=engine.score(b,plan)[0]:raise ValueError('teacher trace mismatch')
    candidates.append((cost,plan.copy()))
   control=engine.search(b,initial(b,'affinity'),256,True);candidates.append(control)
   with trace.open('a') as f:f.write(json.dumps({'placement_plan':control[1].tolist(),'rtt_ms':control[0]})+'\n')
   cost,label=min(candidates,key=lambda r:r[0]);save(cell/'best.json',{'placement_plan':label.tolist(),'rtt_ms':cost,'certified_optimal':False});x,e,adj=features(b);xh,_,_=features(b,True)
   for key,v in [('seeds',seed),('x',x),('xh',xh),('eligible',e),('adj',adj),('labels',label.reshape(-1)),('teacher',cost)]:cache[key].append(v)
   meta['cases'].append({'seed':seed,'split':split,'identity':identity,'source_sha256':sha(cell/'problem.json'),'trace_sha256':sha(trace),'trace_rows':16387,'best_sha256':sha(cell/'best.json')})
   if (number+1)%32==0:print(split,number+1,'/',len(records),flush=True)
  np.savez_compressed(out/f'{split}.npz',**{k:np.array(v) for k,v in cache.items()});meta['files'][f'{split}.npz']=sha(out/f'{split}.npz')
 if any(sha(ROOT/p)!=h for p,h in sources.items()):raise ValueError('source changed during generation')
 save(out/'METADATA.json',meta);print('COMPLETE',len(meta['cases']),flush=True)
if __name__=='__main__':main()

"""Validate physical inputs, cached graph features and all retained trace bounds."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import sha,save
from src.policy.mixed.model import features,CONTRACT
from src.placement.radical.environment import pack
from src.placement.radical.mixed import MixedNative
ROOT=Path(__file__).resolve().parents[1]
def validate(root):
 meta=json.loads((root/'METADATA.json').read_text());engine=MixedNative(root/'build');identities=set();count=0
 if meta['contract']!=CONTRACT:raise ValueError('wrong feature contract')
 for name,h in meta['sources'].items():
  if sha(ROOT/name)!=h:raise ValueError('source mismatch')
 for split in ('train','validation','test'):
  path=root/f'{split}.npz'
  if sha(path)!=meta['files'][path.name]:raise ValueError('cache mismatch')
  with np.load(path) as z:d={k:z[k].copy() for k in z.files}
  records={r['seed']:r for r in meta['cases'] if r['split']==split}
  if len(records)!=len(d['seeds']) or len(set(d['seeds']))!=len(records):raise ValueError('split coverage')
  for i,seed in enumerate(d['seeds']):
   cell=root/f'ds_{seed}';r=records[int(seed)];source=cell/'problem.json';trace=cell/'placements/placements.jsonl';bestfile=cell/'best.json'
   if sha(source)!=r['source_sha256'] or sha(trace)!=r['trace_sha256'] or sha(bestfile)!=r['best_sha256']:raise ValueError('actual artifact mismatch')
   b=load_problem(source);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
   if identity!=r['identity'] or identity in identities:raise ValueError('duplicate/mismatched physical source')
   identities.add(identity);x,e,adj=features(b);xh,_,_=features(b,True)
   for key,v in [('x',x),('xh',xh),('eligible',e),('adj',adj)]:
    if not np.array_equal(v,d[key][i]):raise ValueError('cached graph mismatch')
   best=json.loads(bestfile.read_text());a=np.array(best['placement_plan']);score=engine.score(b,a)[0]
   if score!=best['rtt_ms'] or score!=d['teacher'][i] or not np.array_equal(a.ravel(),d['labels'][i]):raise ValueError('label mismatch')
   minimum=float('inf');rows=0;found=False
   with trace.open() as f:
    for line in f:
     cost=float(line.rsplit(':',1)[1].strip('}\n '));minimum=min(minimum,cost);rows+=1
     if cost==score:
      candidate=json.loads(line)
      if np.array_equal(candidate['placement_plan'],a):found=True
     if rows in (1,8193,8194,16386,16387):
      candidate=json.loads(line)
      if engine.score(b,np.array(candidate['placement_plan']))[0]!=cost:raise ValueError('trace replay mismatch')
   if rows!=r['trace_rows'] or minimum!=score or not found:raise ValueError('incomplete/inconsistent search trace')
   count+=rows
 return {'status':'PASS','physical_unique':len(identities),'trace_rows_checked':count,'metadata_sha256':sha(root/'METADATA.json')}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();path=args.root/'VALIDATION.json'
 if path.exists():raise ValueError('preserve previous validation')
 r=validate(args.root);save(path,r);print(json.dumps(r),flush=True)

"""Test the proposed graph on exact witnesses with invariant engineered base rows."""
import argparse,copy,itertools,json
from pathlib import Path
import numpy as np
import torch
from src.placement.radical.environment import problem,serialized
from src.placement.radical.mixed import MixedNative
from src.policy.mixed.model import MixedNet,features
from scripts_cosim.radical_physics_screen import save,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
 if out.exists():raise ValueError('preserve previous witness')
 out.mkdir(parents=True);torch.set_num_threads(1);engine=MixedNative(out/'build');rows=[];save(out/'protocol_before_run.json',{'seeds':[109000,109015],'intervention':'reverse the two future type/lock requirements within every job, preserving each job histogram and all first-task base features','shape':[3,3,4],'model_seed':17,'trained':False,'sources':{p:sha(p) for p in ['src/policy/mixed/model.py','scripts_cosim/mixed_graph_witness.py']}})
 for seed in range(109000,109016):
  b=problem(seed,jobs=3,ops=3);other=copy.deepcopy(b)
  for key in ('types','locks'):other[key][:,1:]=b[key][:,1:][:,::-1]
  original=features(b);rewired=features(other)
  if not np.array_equal(original[0][0],rewired[0][0]):raise ValueError('base root observation changed')
  logits={}
  for arm in ('gnn','mpoff','mlp_hand'):
   data=[features(case,arm=='mlp_hand') for case in (b,other)];torch.manual_seed(17);model=MixedNet(data[0][0].shape[-1],mp=arm=='gnn').eval()
   with torch.inference_mode():values=[model(torch.tensor(x)[None],torch.tensor(e)[None],torch.tensor(a)[None])[0,0].numpy() for x,e,a in data]
   eligible=np.isfinite(b['p'][0,0]);logits[arm]=float(np.max(np.abs(values[0][eligible]-values[1][eligible])))
  bests=[]
  for name,case in (('original',b),('rewired',other)):
   cell=out/str(seed)/name;(cell/'placements').mkdir(parents=True);save(cell/'problem.json',serialized(case));allowed=np.argsort(case['p'],axis=-1)[...,:2];costs={int(h):float('inf') for h in allowed[0,0]}
   with (cell/'placements/placements.jsonl').open('w') as f:
    for bits in itertools.product((0,1),repeat=9):
     a=np.take_along_axis(allowed,np.array(bits).reshape(3,3,1),axis=-1)[...,0];cost=engine.score(case,a)[0];costs[int(a[0,0])]=min(costs[int(a[0,0])],cost);f.write(json.dumps({'placement_plan':a.tolist(),'rtt_ms':cost})+'\n')
   bests.append(costs)
  choices=[min(c,key=c.get) for c in bests];strict=choices[0]!=choices[1] and all(len(set(c.values()))==2 for c in bests);rows.append({'seed':seed,'root_costs':bests,'strict_flip':strict,'root_logit_difference':logits})
 report={'pairs':len(rows),'strict_flips':sum(r['strict_flip'] for r in rows),'gnn_distinguishes_all_flips':all(r['root_logit_difference']['gnn']>1e-7 for r in rows if r['strict_flip']),'mpoff_root_identical':all(r['root_logit_difference']['mpoff']==0 for r in rows),'hand_distinguishes_all_flips':all(r['root_logit_difference']['mlp_hand']>1e-7 for r in rows if r['strict_flip']),'rows':rows};save(out/'read.json',report);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()

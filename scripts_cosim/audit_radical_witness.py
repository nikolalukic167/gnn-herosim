import argparse,json
from pathlib import Path
import numpy as np
from scripts_cosim.audit_radical_physics import load_problem,digest
from src.placement.radical.live import run
ap=argparse.ArgumentParser(description='Independently replay every exhaustive witness placement');ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
if (root/'AUDIT.json').exists():raise ValueError('preserve previous audit')
pre=json.loads((root/'protocol_before_run.json').read_text());count=0;files={}
for path,h in pre['sources'].items():
 if digest(Path(path))!=h:raise ValueError('source changed')
for mechanism in pre['protocol']['mechanisms']:
 for cell in sorted((root/mechanism).glob('*/*')):
  b=load_problem(cell/'input.json');best={};plans=set()
  for line in (cell/'placements/placements.jsonl').read_text().splitlines():
   r=json.loads(line);a=np.array(r['placement_plan']);s=run(b,a,mechanism);count+=1;key=tuple(a.ravel())
   if key in plans or abs(s['objective']-r['objective'])>1e-7:raise ValueError('invalid sweep')
   plans.add(key);h=int(a[0,0]);best[h]=min(best.get(h,float('inf')),s['objective'])
  if len(plans)!=512:raise ValueError('incomplete enumeration')
  stored=json.loads((cell/'read.json').read_text())
  if any(abs(stored['best_by_root'][str(h)]['cost']-c)>1e-7 for h,c in best.items()):raise ValueError('root conditional minimum wrong')
  for name in ('input.json','read.json','live.json','placements/placements.jsonl'):files[str((cell/name).relative_to(root))]=digest(cell/name)
result={'status':'PASS','independent_simpy_recomputed_placements':count,'artifacts_sha256':files,'report_sha256':digest(root/'read.json')};(root/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',count,'exhaustive placements independently replayed')

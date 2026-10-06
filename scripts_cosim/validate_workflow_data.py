"""Validate actual workflow sources, traces and cached arrays before using a corpus."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from src.policy.workflow.model import CONTRACT, features
from src.placement.workflow_planning import evaluate

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(root):
    root=Path(root);meta=json.loads((root/'METADATA.json').read_text())
    if meta['contract']!=CONTRACT:raise ValueError('unknown workflow contract')
    for name,expected in meta['sources'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'source fingerprint mismatch: {name}')
    all_identity=set();all_seeds=set();counts={}
    for split in ('train','validation','test'):
        file=root/f'{split}.npz'
        if digest(file)!=meta['files'][file.name]:raise ValueError(f'cache fingerprint mismatch: {file.name}')
        with np.load(file) as archive:data={k:archive[k].copy() for k in archive.files}
        cases=[r for r in meta['cases'] if r['split']==split]
        if len(cases)!=len(data['seeds']):raise ValueError('manifest/cache case count differs')
        by_seed={int(seed):i for i,seed in enumerate(data['seeds'])}
        if len(by_seed)!=len(cases):raise ValueError('duplicate cached seed')
        for case in cases:
            seed=case['seed'];cell=root/f'ds_{seed}';source=cell/'problem.json';trace=cell/'placements/placements.jsonl'
            if digest(source)!=case['source_sha256'] or digest(trace)!=case['placement_sha256']:raise ValueError(f'actual artifact fingerprint mismatch: {seed}')
            problem=json.loads(source.read_text());p=np.array(problem['processing_ms'],dtype=float);p[np.isnan(p)]=np.inf
            identity=hashlib.sha256(p.tobytes()).hexdigest()
            if identity!=case['instance_sha256'] or identity in all_identity or seed in all_seeds:raise ValueError(f'duplicate or mismatched actual infrastructure: {seed}')
            if problem['seed']!=seed:raise ValueError('source seed differs from manifest')
            all_identity.add(identity);all_seeds.add(seed);i=by_seed[seed]
            if not np.array_equal(p,data['p'][i]):raise ValueError('cache differs from actual problem')
            x,e=features(p);xh,_=features(p,True)
            for name,actual in [('x',x),('xh',xh),('eligible',e)]:
                if not np.array_equal(actual,data[name][i]):raise ValueError(f'cached feature mismatch: {name}')
            best=json.loads((cell/'best.json').read_text());a=np.array(best['placement_plan'])
            if not np.array_equal(a,data['labels'][i]) or abs(evaluate(p,a)-data['teacher'][i])>1e-7 or abs(best['rtt_ms']-data['teacher'][i])>1e-7:raise ValueError('label/cost mismatch')
        counts[split]=len(cases)
    if sum(counts.values())!=len(meta['cases']):raise ValueError('unknown split in manifest')
    return {'status':'PASS','actual_source_identity_unique':len(all_identity),'counts':counts,'metadata_sha256':digest(root/'METADATA.json')}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('--receipt',type=Path,required=True);args=ap.parse_args()
    report=validate(args.root)
    if args.receipt.exists():raise ValueError('preserve previous receipt')
    args.receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

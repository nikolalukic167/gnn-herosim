"""Read retained inputs, plans and events; independently validate physics and coverage."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from src.placement.radical.environment import pack,FLAGS
from src.placement.radical.live import run

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_problem(path):
    raw=json.loads(path.read_text());b={k:np.array(v) if isinstance(v,list) else v for k,v in raw.items()};b['p']=np.array(raw['p'],float);b['p'][np.isnan(b['p'])]=np.inf;return b

def audit(root):
    pre=json.loads((root/'protocol_before_run.json').read_text());meta=json.loads((root/'source_validation.json').read_text());confirmation='fresh_seeds' in pre['protocol'];runs=tasks=0;hashes={};bindings={}
    for name,h in pre['sources'].items():
        if digest(Path(name))!=h:raise ValueError('changed source: '+name)
    ids=set();problems={}
    for seed,r in meta['inputs'].items():
        path=root/'inputs'/f'{seed}.json';b=load_problem(path);identity=hashlib.sha256(pack(b).tobytes()).hexdigest()
        if digest(path)!=r.get('file_sha256',r.get('sha256')) or identity!=r['identity'] or identity in ids:raise ValueError('source fingerprint/uniqueness failure')
        ids.add(identity);problems[int(seed)]=b
    results=json.loads((root/'read.json').read_text());lo,hi=pre['protocol']['fresh_seeds' if confirmation else 'screen_seeds']
    for mechanism in pre['protocol']['mechanisms']:
        bindings[mechanism]={'setup_duration':0.,'thermal_extra_duration':0.,'lock_blocks':0,'power_blocks':0,'maintenance_blocks':0,'reference_tardiness':0.}
        for seed in range(lo,hi+1):
            cell=root/mechanism/str(seed);b=problems[seed];r=json.loads((cell/'read.json').read_text());live=json.loads((cell/'live.json').read_text());expected={'control','reference_load','reference_fastest'} if confirmation else {'control','reference','fastest'}
            if set(live)!=expected:raise ValueError('missing live arm')
            for label,s in live.items():
                method=label if confirmation else s['method'];a=np.array(r['methods'][method]['plan']);replayed=run(b,a,mechanism)
                if abs(replayed['objective']-s['objective'])>1e-7 or replayed['events']!=s['events']:raise ValueError('nonreproducible retained events')
                if abs(s['objective']-r['methods'][method]['cost'])>1e-6:raise ValueError('native/live cost mismatch')
                starts=[e for e in s['events'] if e['event']=='start'];done=[e for e in s['events'] if e['event']=='done'];shape=b['p'].shape
                if len(starts)!=shape[0]*shape[1] or len(done)!=len(starts):raise ValueError('incomplete simulation')
                byid={(e['job'],e['operation']):e for e in starts}
                if len(byid)!=len(starts):raise ValueError('duplicate task')
                for e in starts:
                    j,k,h=e['job'],e['operation'],e['host']
                    if h!=a[j,k] or e['base']!=b['p'][j,k,h]:raise ValueError('wrong physical task')
                    if k and e['start']<byid[j,k-1]['end']-1e-9:raise ValueError('precedence violation')
                    if FLAGS[mechanism]&16:
                        lo_m,hi_m=b['maintenance'][h]
                        if e['start']<hi_m-1e-9 and e['end']>lo_m+1e-9:raise ValueError('maintenance overlap')
                if label==('reference' if not confirmation else r['reference']):
                    x=bindings[mechanism];x['setup_duration']+=sum(e['setup'] for e in starts);x['thermal_extra_duration']+=sum(e['end']-e['start']-e['base']-e['setup'] for e in starts);x['lock_blocks']+=s['blocking_observations']['locks'];x['power_blocks']+=s['blocking_observations']['power'];x['maintenance_blocks']+=s['blocking_observations']['maintenance'];x['reference_tardiness']+=s['tardiness']
                runs+=1;tasks+=len(starts)
            for name in ('read.json','live.json','placements/placements.jsonl'):hashes[str((cell/name).relative_to(root))]=digest(cell/name)
    expected_runs=results.get('simpy_runs',results.get('simpy_replays'))
    if not confirmation:expected_runs-=2*(hi-lo+1)*len(pre['protocol']['mechanisms'])
    if runs!=expected_runs:raise ValueError('coverage aggregate mismatch')
    return {'status':'PASS','retained_live_replays':runs,'task_records':tasks,'unique_inputs':len(ids),'binding_in_reference':bindings,'artifacts_sha256':hashes,'report_sha256':digest(root/'read.json')}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();r=audit(args.root);path=args.root/'AUDIT.json'
    if path.exists():raise ValueError('preserve existing audit')
    path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='artifacts_sha256'},indent=2))

"""Fresh matched-budget challenge after translating schedules between DAG decoders."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag,baseline
from src.placement.radical.environment import serialized,pack
from scripts_cosim.dag_reservation_screen import PLAN_KEYS,plan,live,verify_plan,verify_live,summarize
from scripts_cosim.mixed_joint_dispatch_screen import ROOT,read,save,sha
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL=ROOT/'experiments/dag_reservation_bridge_v1.json'


def base_plan(engine,b,mode=2):
    cost,a,rank,decoder,rule=baseline(engine,b,mode)
    return {**plan(engine,b,a,rank,decoder,cost),'rule':rule}


def control(engine,b,mode,temperature,budget_ms):
    tick=time.perf_counter();cost,a,rank,decoder,rule=baseline(engine,b,mode)
    remaining=budget_ms/1000-(time.perf_counter()-tick)-.006;count=0
    if remaining>.001:cost,a,rank,decoder,count=engine.search(b,a,rank,2**31-1,mode,temperature,77,remaining)
    result=plan(engine,b,a,rank,decoder,cost)
    return {**result,'rule':rule,'scores':count,'time_ms':(time.perf_counter()-tick)*1000}


def arms(engine,b,protocol,budget):
    result={}
    for name,(mode,temp) in protocol['hand_modes'].items():
        trials=[control(engine,b,mode,temp,budget) for _ in range(protocol['timing_repeats'])]
        median=sorted(trials,key=lambda r:r['cost'])[len(trials)//2]
        result[name]={**median,'repeats':trials,'timing_valid':all(r['time_ms']<=budget for r in trials)}
    return result


def run(out):
    out.mkdir(parents=True,exist_ok=False);protocol=read(PROTOCOL)
    old=read(ROOT/'simulation_data/gnn_environment_search_v1/dag_reservation_screen_v1/protocol_before_run.json')['sources']
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    new=[PROTOCOL,Path(__file__).resolve(),ROOT/'src/placement/radical/dag_reservation_bridge.cpp',ROOT/'src/placement/radical/dag_reservation_bridge.py']
    sources={**old,**{str(p.relative_to(ROOT)):sha(p) for p in new}}
    save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources})
    engine=BridgedDag(out/'build');save(out/'native.json',engine.provenance);configure_environment()
    runs=operations=0;identities=set()
    def case(seed,shape,parent):
        b=problem(seed,*shape);identity=hashlib.sha256(pack(b).tobytes()+b['predecessors'].tobytes()).hexdigest()
        assert identity not in identities;identities.add(identity)
        cell=out/parent/str(seed);cell.mkdir(parents=True)
        save(cell/'case.json',{'seed':seed,'shape':shape});save(cell/'input.json',serialized(b));return b,cell
    def serve(b,p,cell,name):
        nonlocal runs,operations
        counts=live(engine,b,p,cell,name);runs+=counts['runs'];operations+=counts['operations']
    for seed in (149900,149905):
        b,cell=case(seed,[2,8,4],'parity')
        for mode in (0,1,2):
            _,a,rank,_,_=baseline(engine,b,mode)
            cost,a,rank,decoder,count=engine.search(b,a,rank,256,mode,.001)
            serve(b,plan(engine,b,a,rank,decoder,cost),cell,f'mode{mode}')
    print('BRIDGED DAG PARITY PASS',flush=True)
    calibration=[];shape=protocol['shape'];cal_budget=protocol['budget_ms']*protocol['calibration_fraction']
    for seed in range(protocol['calibration_seeds'][0],protocol['calibration_seeds'][1]+1):
        b,cell=case(seed,shape,'calibration');row={'seed':seed,'arms':arms(engine,b,protocol,cal_budget)}
        save(cell/'read.json',row);calibration.append(row)
        print('BRIDGE CAL',seed,{n:p['cost'] for n,p in row['arms'].items()},flush=True)
    eligible=[n for n in protocol['hand_modes'] if all(r['arms'][n]['timing_valid'] for r in calibration)]
    if not eligible:raise RuntimeError('no timing-eligible bridge calibration control')
    selected=min(eligible,key=lambda n:np.mean([r['arms'][n]['cost'] for r in calibration]))
    save(out/'calibration.json',{'selected':selected,'eligible':eligible,'rows':calibration})
    print('BRIDGE FROZEN CONTROL',selected,flush=True)
    for row in calibration:serve(problem(row['seed'],*shape),row['arms'][selected],out/'calibration'/str(row['seed']),'control')
    rows=[]
    for seed in range(protocol['fresh_seeds'][0],protocol['fresh_seeds'][1]+1):
        b,cell=case(seed,shape,'fresh');base=base_plan(engine,b);hand=arms(engine,b,protocol,protocol['budget_ms'])
        starts={'control':hand[selected],'baseline':base};refs=[]
        for start,mode,temp in itertools.product(protocol['reference_starts'],protocol['reference_modes'],protocol['reference_temperatures']):
            source=starts[start]
            cost,a,rank,decoder,count=engine.search(b,np.array(source['assignment']),np.array(source['priority']),protocol['reference_steps'],mode,temp,protocol['reference_seed'])
            refs.append({**plan(engine,b,a,rank,decoder,cost),'start':start,'search_mode':mode,'temperature':temp,'steps':count})
        best=min([base,*hand.values(),*refs],key=lambda r:r['cost'])
        row={'seed':seed,'baseline':base,'arms':hand,'references':refs,'reference':{k:best[k] for k in PLAN_KEYS}}
        save(cell/'read.json',row);rows.append(row)
        print('BRIDGE FRESH',seed,'headroom_pct',round((hand[selected]['cost']-best['cost'])/hand[selected]['cost']*100,3),flush=True)
    report=summarize(rows,selected,protocol);save(out/'screen_before_live.json',report)
    for row in rows:
        b=problem(row['seed'],*shape);cell=out/'fresh'/str(row['seed']);(cell/'placements').mkdir()
        with (cell/'placements/placements.jsonl').open('w') as output:
            for name,p in [('baseline',row['baseline']),('control',row['arms'][selected]),('reference',row['reference'])]:
                serve(b,p,cell,name);output.write(json.dumps({'arm':name,**{k:p[k] for k in PLAN_KEYS}})+'\n')
        print('BRIDGE LIVE',row['seed'],flush=True)
    report.update({'shape':shape,'inputs':len(identities),'live_runs':runs,'live_operations':operations})
    save(out/'read.json',report)
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')})
    print(json.dumps(report,indent=2),flush=True)


def audit(out):
    pre=read(out/'protocol_before_run.json');protocol=pre['protocol']
    for name,digest in pre['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in read(out/'artifacts.json').items():assert sha(out/name)==digest,name
    engine=BridgedDag(out/'build');assert engine.provenance==read(out/'native.json')
    calibration=read(out/'calibration.json');selected=calibration['selected'];identities=set();rows=[];cal=[];runs=operations=proposals=0
    for case_path in sorted(out.glob('*/*/case.json')):
        meta=read(case_path);cell=case_path.parent;b=problem(meta['seed'],*meta['shape'])
        assert serialized(b)==read(cell/'input.json')
        identity=hashlib.sha256(pack(b).tobytes()+b['predecessors'].tobytes()).hexdigest();assert identity not in identities;identities.add(identity)
        for path in cell.glob('*_replay.json'):
            p=read(path);independent=verify_plan(engine,b,p);assert independent['events']==p['events']
            verify_live(b,p,read(path.with_name(path.name.replace('_replay.json','_live.json'))));runs+=1;operations+=np.size(p['assignment'])
        if not (cell/'read.json').exists():continue
        row=read(cell/'read.json');budget=protocol['budget_ms']*(protocol['calibration_fraction'] if cell.parent.name=='calibration' else 1)
        for arm in row['arms'].values():
            median=sorted(arm['repeats'],key=lambda r:r['cost'])[len(arm['repeats'])//2]
            assert all(arm[k]==median[k] for k in median)
            assert arm['timing_valid']==all(t['time_ms']<=budget for t in arm['repeats'])
            for trial in arm['repeats']:verify_plan(engine,b,trial)
        if cell.parent.name=='calibration':cal.append(row);continue
        rows.append(row);assert row['baseline']==base_plan(engine,b)
        starts={'control':row['arms'][selected],'baseline':row['baseline']}
        expected=list(itertools.product(protocol['reference_starts'],protocol['reference_modes'],protocol['reference_temperatures']))
        assert [(r['start'],r['search_mode'],r['temperature']) for r in row['references']]==expected
        for ref in row['references']:
            source=starts[ref['start']]
            cost,a,rank,decoder,count=engine.search(b,np.array(source['assignment']),np.array(source['priority']),protocol['reference_steps'],ref['search_mode'],ref['temperature'],protocol['reference_seed'])
            assert plan(engine,b,a,rank,decoder,cost)=={k:ref[k] for k in PLAN_KEYS}
            assert ref['steps']==count;proposals+=count;verify_plan(engine,b,ref)
        best=min([row['baseline'],*row['arms'].values(),*row['references']],key=lambda r:r['cost'])
        assert row['reference']=={k:best[k] for k in PLAN_KEYS}
        plans=[json.loads(line) for line in (cell/'placements/placements.jsonl').read_text().splitlines()]
        assert [p['arm'] for p in plans]==['baseline','control','reference']
        for p,target in zip(plans,[row['baseline'],row['arms'][selected],row['reference']]):
            served=read(cell/(p['arm']+'_replay.json'))
            for k in PLAN_KEYS:assert p[k]==target[k]==served[k]
        print('BRIDGE AUDIT',row['seed'],flush=True)
    assert calibration['rows']==sorted(cal,key=lambda r:r['seed'])
    eligible=[n for n in protocol['hand_modes'] if all(r['arms'][n]['timing_valid'] for r in cal)]
    assert calibration['eligible']==eligible and selected==min(eligible,key=lambda n:np.mean([r['arms'][n]['cost'] for r in cal]))
    assert {r['seed'] for r in cal}==set(range(protocol['calibration_seeds'][0],protocol['calibration_seeds'][1]+1))
    assert {r['seed'] for r in rows}==set(range(protocol['fresh_seeds'][0],protocol['fresh_seeds'][1]+1))
    report=read(out/'read.json');expected=summarize(sorted(rows,key=lambda r:r['seed']),selected,protocol)
    assert all(report[k]==v for k,v in expected.items())
    assert (report['live_runs'],report['live_operations'],report['inputs'])==(runs,operations,len(identities))==(58,13408,22)
    result={'status':'PASS','inputs':len(identities),'live_runs':runs,'live_operations':operations,'reference_proposals_reexecuted':proposals,'report_sha256':sha(out/'read.json')}
    save(out/'AUDIT.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();audit(args.out) if args.audit_only else run(args.out)

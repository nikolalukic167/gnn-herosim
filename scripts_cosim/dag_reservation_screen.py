"""Frozen CPU scale and headroom qualification for DAG reservation planning."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
from src.placement.radical.dag_reservation import DagReservation,problem,baseline
from src.placement.radical.dag_reservation_live import replay,herosim
from src.placement.radical.environment import serialized,pack
from src.placement.radical.block_moves import scaled_problem
from scripts_cosim.dispatch_reservation_witness import fixture,certificate
from scripts_cosim.mixed_joint_dispatch_screen import ROOT,read,save,sha,gains
from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL=ROOT/'experiments/dag_reservation_screen_v1.json'
PLAN_KEYS=('cost','assignment','priority','decoder','release','starts','ends')


def make_case(kind,seed,shape):
    if kind=='witness':
        b=fixture();b['predecessors']=np.array([[0,1],[0,1]],dtype=np.uint64)
    elif kind=='diamond':
        b=scaled_problem(seed,1,4,2);b['predecessors']=np.array([[0,1,1,6]],dtype=np.uint64)
        b['locks'][:]=0;b['types'][:]=0;b['p'][:]=3
    else:b=problem(seed,*shape)
    return b


def plan(engine,b,a,rank,decoder,cost=None):
    value,starts,ends=engine.plan(b,a,rank,decoder)
    if cost is not None and value!=cost:raise ValueError('search/plan mismatch')
    release=starts if decoder else np.zeros(a.shape)
    return {'cost':value,'assignment':a.tolist(),'priority':rank.tolist(),'decoder':decoder,
            'release':release.tolist(),'starts':starts.tolist(),'ends':ends.tolist()}


def base_plan(engine,b,mode=2):
    cost,a,rank,decoder,rule=baseline(engine,b,mode)
    return {**plan(engine,b,a,rank,decoder,cost),'rule':rule}


def control(engine,b,mode,temperature,budget_ms):
    tick=time.perf_counter()
    cost,a,rank,decoder,rule=baseline(engine,b,mode)
    remaining=budget_ms/1000-(time.perf_counter()-tick)-.004
    count=0
    if remaining>.001:
        cost,a,rank,decoder,count=engine.search(b,a,rank,2**31-1,mode,temperature,77,remaining)
    result=plan(engine,b,a,rank,decoder,cost)
    return {**result,'rule':rule,'scores':count,'time_ms':(time.perf_counter()-tick)*1000}


def arms(engine,b,protocol,budget_ms):
    result={}
    for name,(mode,temp) in protocol['hand_modes'].items():
        repeats=[control(engine,b,mode,temp,budget_ms) for _ in range(protocol['timing_repeats'])]
        median=sorted(repeats,key=lambda r:r['cost'])[len(repeats)//2]
        result[name]={**median,'repeats':repeats,'timing_valid':all(r['time_ms']<=budget_ms for r in repeats)}
    return result


def verify_plan(engine,b,p):
    a,rank,release=np.array(p['assignment']),np.array(p['priority']),np.array(p['release'])
    cost,starts,ends=engine.plan(b,a,rank,p['decoder'])
    native=engine.replay(b,a,rank,release);independent=replay(b,a,rank,release)
    assert cost==p['cost']==native[0]==independent['objective']
    for expected,actual,other,recorded in ((starts,native[1],independent['starts'],p['starts']),(ends,native[2],independent['ends'],p['ends'])):
        assert np.array_equal(expected,actual) and np.array_equal(expected,other) and np.array_equal(expected,recorded)
    assert np.array_equal(release,starts if p['decoder'] else np.zeros(a.shape))
    return independent


def verify_live(b,p,actual):
    assert abs(actual['job_completion_sum']*1000-p['cost'])<1e-6
    assert actual['mixed_execution']['dispatch_contract']=='mixed_irregular_dag_reservation_v1'
    assert actual['mixed_execution']['predecessors']==b['predecessors'].tolist()
    a,ends=np.array(p['assignment']),np.array(p['ends']);seen=set()
    for task in actual['tasks']:
        j,k=map(int,task['taskType']['name'][2:].split('_'))
        assert (j,k) not in seen
        seen.add((j,k))
        assert task['executionNode']==f'node{a[j,k]}' and abs(task['doneTime']*1000-ends[j,k])<1e-6
    assert len(seen)==a.size
    for j in range(a.shape[0]):
        dag=actual['config']['workload']['events'][j]['application']['dag']
        assert dag=={f'op{j}_{k}':[f'op{j}_{z}' for z in range(k) if int(b['predecessors'][j,k])&(1<<z)] for k in range(a.shape[1])}


def live(engine,b,p,cell,name):
    independent=verify_plan(engine,b,p)
    save(cell/(name+'_replay.json'),{**{k:p[k] for k in PLAN_KEYS},'events':independent['events']})
    with (cell/(name+'_live.log')).open('w') as log:
        actual=herosim(b,np.array(p['assignment']),np.array(p['priority']),np.array(p['release']),log)
    verify_live(b,p,actual);save(cell/(name+'_live.json'),actual)
    return {'runs':1,'operations':np.size(p['assignment'])}


def scale_summary(rows,protocol):
    result=[]
    for shape in protocol['shapes']:
        group=[r for r in rows if r['shape']==shape]
        times=[t for r in group for t in r['times_ms']]
        result.append({'shape':shape,'p95_ms':float(np.quantile(times,.95)),
                       'median_utilization':float(np.median([r['utilization'] for r in group]))})
    eligible=[r for r in result if r['p95_ms']<=protocol['scale_p95_limit_ms'] and r['median_utilization']>=protocol['minimum_median_utilization']]
    return {'rungs':result,'selected_shape':max(eligible,key=lambda r:np.prod(r['shape'][:2]))['shape'] if eligible else None}


def summarize(rows,selected,protocol):
    ref=np.array([r['reference']['cost'] for r in rows])
    costs={n:np.array([r['arms'][n]['cost'] for r in rows]) for n in protocol['hand_modes']}
    eligible=[n for n in costs if all(r['arms'][n]['timing_valid'] for r in rows)]
    primary=gains(ref,costs[selected]);timing=selected in eligible
    passed=timing and primary['median_pct']>=protocol['advance_median_headroom_pct'] and primary['ci95_pct'][0]>protocol['advance_bootstrap_lower_pct']
    envelope=gains(ref,np.min(np.stack([costs[n] for n in eligible]),axis=0)) if eligible else None
    return {'status':'HEADROOM-ONLY-PASS' if passed else 'RECIPE-NOT-QUALIFIED','selected':selected,'eligible':eligible,
            'headroom':primary,'eligible_envelope':envelope,'per_arm_headroom':{n:gains(ref,c) for n,c in costs.items()},
            'headroom_pass':bool(passed),'timing_pass':timing,'training_allowed':False}


def run(out):
    out.mkdir(parents=True,exist_ok=False);protocol=read(PROTOCOL)
    old=read(ROOT/'simulation_data/gnn_environment_search_v1/dispatch_reservation_witness_v1/protocol_before_run.json')['sources']
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    new=[PROTOCOL,Path(__file__).resolve(),ROOT/'src/placement/radical/dag_reservation.py',ROOT/'src/placement/radical/dag_reservation.cpp',ROOT/'src/placement/radical/dag_reservation_live.py']
    sources={**old,**{str(p.relative_to(ROOT)):sha(p) for p in new}}
    save(out/'protocol_before_run.json',{'protocol':protocol,'sources':sources})
    engine=DagReservation(out/'build');save(out/'native.json',engine.provenance);configure_environment()
    identities=set();runs=operations=0
    def case(kind,seed,shape,parent):
        b=make_case(kind,seed,shape);identity=hashlib.sha256(pack(b).tobytes()+np.asarray(b['predecessors'],dtype=np.uint64).tobytes()).hexdigest()
        assert identity not in identities
        identities.add(identity)
        cell=out/parent/str(seed);cell.mkdir(parents=True)
        save(cell/'case.json',{'kind':kind,'seed':seed,'shape':shape});save(cell/'input.json',serialized(b))
        return b,cell
    def serve(b,p,cell,name):
        nonlocal runs,operations
        counts=live(engine,b,p,cell,name);runs+=counts['runs'];operations+=counts['operations']
    for kind,seed,shape in [('witness',145900,[2,2,2]),('diamond',148900,[1,4,2]),('irregular',148901,[2,8,4])]:
        b,cell=case(kind,seed,shape,'parity')
        for mode in (0,1):
            p=base_plan(engine,b,mode);serve(b,p,cell,f'mode{mode}')
            if kind=='witness' and mode==1:assert p['cost']==certificate(b)['best']['cost']==27
            if kind=='diamond':assert p['cost']==9
    print('DAG THREE-WAY PARITY PASS',flush=True)
    scales=[]
    for shape,start_seed in zip(protocol['shapes'],protocol['scale_seed_starts']):
        for seed in range(start_seed,start_seed+protocol['scale_inputs_per_shape']):
            b,cell=case('irregular',seed,shape,'scale');times=[];p=None
            for _ in range(protocol['timing_repeats']):
                tick=time.perf_counter();_,a,rank,_,_=baseline(engine,b)
                cost,a,rank,decoder,count=engine.search(b,a,rank,protocol['scale_proxy_steps'],2,.001)
                p=plan(engine,b,a,rank,decoder,cost);times.append((time.perf_counter()-tick)*1000)
            starts,ends=np.array(p['starts']),np.array(p['ends'])
            row={'seed':seed,'shape':shape,'times_ms':times,'utilization':float((ends-starts).sum()/ends.max()/shape[2]),'plan':p}
            save(cell/'read.json',row);scales.append(row);serve(b,p,cell,'proxy')
            print('SCALE',seed,shape,'max_ms',round(max(times),2),'util',round(row['utilization'],3),flush=True)
    scale=scale_summary(scales,protocol);save(out/'scale.json',scale);shape=scale['selected_shape']
    print('SCALE SELECTION',scale,flush=True)
    if shape is None:
        report={'status':'SCALE-NOT-QUALIFIED','training_allowed':False}
    else:
        calibration=[];cal_budget=protocol['budget_ms']*protocol['calibration_fraction']
        for seed in range(protocol['calibration_seeds'][0],protocol['calibration_seeds'][1]+1):
            b,cell=case('irregular',seed,shape,'calibration');row={'seed':seed,'arms':arms(engine,b,protocol,cal_budget)}
            save(cell/'read.json',row);calibration.append(row)
            print('CAL',seed,{n:r['cost'] for n,r in row['arms'].items()},flush=True)
        eligible=[n for n in protocol['hand_modes'] if all(r['arms'][n]['timing_valid'] for r in calibration)]
        if not eligible:raise RuntimeError('no eligible calibration control; no silent fallback')
        selected=min(eligible,key=lambda n:np.mean([r['arms'][n]['cost'] for r in calibration]))
        save(out/'calibration.json',{'selected':selected,'eligible':eligible,'rows':calibration})
        print('FROZEN CONTROL',selected,flush=True)
        for row in calibration:
            b=problem(row['seed'],*shape);serve(b,row['arms'][selected],out/'calibration'/str(row['seed']),'control')
        rows=[]
        for seed in range(protocol['fresh_seeds'][0],protocol['fresh_seeds'][1]+1):
            b,cell=case('irregular',seed,shape,'fresh');base=base_plan(engine,b);hand=arms(engine,b,protocol,protocol['budget_ms'])
            refs=[];starts={'control':hand[selected],'baseline':base}
            for start,mode,temp in itertools.product(protocol['reference_starts'],protocol['reference_modes'],protocol['reference_temperatures']):
                source=starts[start]
                cost,a,rank,decoder,count=engine.search(b,np.array(source['assignment']),np.array(source['priority']),protocol['reference_steps'],mode,temp,protocol['reference_seed'])
                refs.append({**plan(engine,b,a,rank,decoder,cost),'start':start,'search_mode':mode,'temperature':temp,'steps':count})
            best=min([base,*hand.values(),*refs],key=lambda r:r['cost'])
            row={'seed':seed,'baseline':base,'arms':hand,'references':refs,'reference':{k:best[k] for k in PLAN_KEYS}}
            save(cell/'read.json',row);rows.append(row)
            print('FRESH',seed,'headroom_pct',round((hand[selected]['cost']-best['cost'])/hand[selected]['cost']*100,3),flush=True)
        report=summarize(rows,selected,protocol);save(out/'screen_before_live.json',report)
        for row in rows:
            b=problem(row['seed'],*shape);cell=out/'fresh'/str(row['seed']);(cell/'placements').mkdir()
            with (cell/'placements/placements.jsonl').open('w') as output:
                for name,p in [('baseline',row['baseline']),('control',row['arms'][selected]),('reference',row['reference'])]:
                    serve(b,p,cell,name);output.write(json.dumps({'arm':name,**{k:p[k] for k in PLAN_KEYS}})+'\n')
            print('DAG LIVE',row['seed'],flush=True)
    report.update({'live_runs':runs,'live_operations':operations,'inputs':len(identities),'shape':shape})
    save(out/'read.json',report)
    for name,digest in sources.items():assert sha(ROOT/name)==digest,name
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')})
    print(json.dumps(report,indent=2),flush=True)


def audit(out):
    pre=read(out/'protocol_before_run.json');protocol=pre['protocol']
    for name,digest in pre['sources'].items():assert sha(ROOT/name)==digest,name
    for name,digest in read(out/'artifacts.json').items():assert sha(out/name)==digest,name
    engine=DagReservation(out/'build');assert engine.provenance==read(out/'native.json')
    identities=set();runs=operations=proposals=0;scales=[];fresh=[];cal=[]
    scale=read(out/'scale.json');shape=scale['selected_shape']
    selected=read(out/'calibration.json')['selected'] if shape else None
    for case_path in sorted(out.glob('*/*/case.json')):
        metadata=read(case_path);cell=case_path.parent;b=make_case(**metadata)
        assert serialized(b)==read(cell/'input.json')
        identity=hashlib.sha256(pack(b).tobytes()+np.asarray(b['predecessors'],dtype=np.uint64).tobytes()).hexdigest()
        assert identity not in identities;identities.add(identity)
        for path in cell.glob('*_replay.json'):
            p=read(path);independent=verify_plan(engine,b,p);assert independent['events']==p['events']
            verify_live(b,p,read(path.with_name(path.name.replace('_replay.json','_live.json'))))
            runs+=1;operations+=np.size(p['assignment'])
        if not (cell/'read.json').exists():continue
        row=read(cell/'read.json')
        if cell.parent.name=='scale':
            scales.append(row);verify_plan(engine,b,row['plan'])
            _,a,rank,_,_=baseline(engine,b)
            cost,a,rank,decoder,count=engine.search(b,a,rank,protocol['scale_proxy_steps'],2,.001)
            assert plan(engine,b,a,rank,decoder,cost)==row['plan'];proposals+=count
            p=row['plan'];util=(np.array(p['ends'])-np.array(p['starts'])).sum()/np.max(p['ends'])/b['p'].shape[2]
            assert row['utilization']==util
        else:
            budget=protocol['budget_ms']*(protocol['calibration_fraction'] if cell.parent.name=='calibration' else 1)
            for arm in row['arms'].values():
                median=sorted(arm['repeats'],key=lambda r:r['cost'])[len(arm['repeats'])//2]
                assert all(arm[k]==median[k] for k in median)
                assert arm['timing_valid']==all(t['time_ms']<=budget for t in arm['repeats'])
                for trial in arm['repeats']:verify_plan(engine,b,trial)
            if cell.parent.name=='calibration':cal.append(row)
            else:
                fresh.append(row);assert row['baseline']==base_plan(engine,b)
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
                records=[json.loads(line) for line in (cell/'placements/placements.jsonl').read_text().splitlines()]
                assert [r['arm'] for r in records]==['baseline','control','reference']
                for p,expected_plan in zip(records,[row['baseline'],row['arms'][selected],row['reference']]):
                    for k in PLAN_KEYS:assert p[k]==expected_plan[k]
                    served=read(cell/(p['arm']+'_replay.json'))
                    for k in PLAN_KEYS:assert served[k]==p[k]
                print('DAG AUDIT',row['seed'],flush=True)
    assert scale==scale_summary(scales,protocol)
    report=read(out/'read.json')
    if shape:
        stored=read(out/'calibration.json');assert stored['rows']==sorted(cal,key=lambda r:r['seed'])
        eligible=[n for n in protocol['hand_modes'] if all(r['arms'][n]['timing_valid'] for r in cal)]
        assert stored['eligible']==eligible and selected==min(eligible,key=lambda n:np.mean([r['arms'][n]['cost'] for r in cal]))
        assert {r['seed'] for r in cal}==set(range(protocol['calibration_seeds'][0],protocol['calibration_seeds'][1]+1))
        assert {r['seed'] for r in fresh}==set(range(protocol['fresh_seeds'][0],protocol['fresh_seeds'][1]+1))
        expected=summarize(sorted(fresh,key=lambda r:r['seed']),selected,protocol)
        assert all(report[k]==v for k,v in expected.items())
        assert runs==66
    else:assert report['status']=='SCALE-NOT-QUALIFIED' and runs==14
    assert report['live_runs']==runs and report['live_operations']==operations and report['inputs']==len(identities)
    result={'status':'PASS','inputs':len(identities),'live_runs':runs,'live_operations':operations,'proposals_reexecuted':proposals,'report_sha256':sha(out/'read.json')}
    save(out/'AUDIT.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();audit(args.out) if args.audit_only else run(args.out)

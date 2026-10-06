"""Certify and live-check a constructed counterexample to non-delay-only search."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from src.placement.radical.block_moves import scaled_problem
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import replay as old_replay, herosim as old_live
from src.placement.radical.reservation_dispatch import replay, herosim
from src.placement.radical.environment import serialized
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL = ROOT/'experiments/dispatch_reservation_witness_v1.json'


def fixture():
    protocol = read(PROTOCOL)
    b = scaled_problem(protocol['seed'],2,2,2)
    b['p'] = np.array(protocol['durations'],dtype=float)
    b['types'][:] = 0
    b['locks'] = np.array([[1,1],[2,2]])
    return b


def certificate(b):
    assert b['p'].shape == (2,2,2) and np.isfinite(b['p']).all() and np.all(b['p'] > 0)
    assert np.all(b['types'] == b['types'][0,0]) and b['setup'][b['types'][0,0],b['types'][0,0]] == 0
    assert not any(int(x)&int(y) for x in b['locks'][0] for y in b['locks'][1])
    assert np.all((b['domains']==0)|(b['domains']==1)) and np.all(b['domains'].sum(0)<=2)
    best, schedules, acyclic = None, 0, 0
    for bits in itertools.product(range(2),repeat=4):
        a = np.array(bits).reshape(2,2)
        groups = [[i for i,h in enumerate(bits) if h==host] for host in range(2)]
        for sequences in itertools.product(*(itertools.permutations(g) for g in groups)):
            schedules += 1
            edges = {(0,1),(2,3)}
            for sequence in sequences:
                edges.update(zip(sequence,sequence[1:]))
            remaining, starts, ends = set(range(4)), np.zeros(4), np.zeros(4)
            while remaining:
                ready = sorted(i for i in remaining if not any(v==i and u in remaining for u,v in edges))
                if not ready:
                    break
                for i in ready:
                    starts[i] = max([ends[u] for u,v in edges if v==i]+[0.])
                    ends[i] = starts[i]+b['p'].reshape(4,2)[i,bits[i]]
                    remaining.remove(i)
            if remaining:
                continue
            acyclic += 1
            cost = float(ends[[1,3]].sum())
            if best is None or cost < best['cost']:
                best = {'cost':cost,'assignment':a.tolist(),'starts':starts.reshape(2,2).tolist(),
                        'ends':ends.reshape(2,2).tolist(),'host_sequences':[list(s) for s in sequences]}
    return {'best':best,'host_order_combinations':schedules,'acyclic_combinations':acyclic}


def check_live(actual, expected, assignment):
    assert abs(actual['total_rtt']*1000-expected['objective']) < 1e-6
    seen = set()
    for task in actual['tasks']:
        j,k = map(int,task['taskType']['name'][2:].split('_'))
        assert (j,k) not in seen
        seen.add((j,k))
        assert task['executionNode']==f'node{assignment[j,k]}'
        assert abs(task['doneTime']*1000-expected['ends'][j,k]) < 1e-6
    assert len(seen)==assignment.size


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    previous = read(ROOT/'simulation_data/gnn_environment_search_v1/mixed_region_repair_s0_v1/protocol_before_run.json')['sources']
    sources = {**previous,**{str(p.relative_to(ROOT)):sha(p) for p in (PROTOCOL,Path(__file__).resolve(),ROOT/'src/placement/radical/reservation_dispatch.py')}}
    for name,digest in previous.items():
        assert sha(ROOT/name)==digest,name
    save(out/'protocol_before_run.json',{'protocol':read(PROTOCOL),'sources':sources})
    b = fixture()
    save(out/'input.json',serialized(b))
    engine = DispatchNative(out/'build')
    save(out/'native.json',engine.provenance)
    (out/'placements').mkdir()
    best = None
    with (out/'placements/placements.jsonl').open('w') as output:
        for bits in itertools.product(range(2),repeat=4):
            a = np.array(bits).reshape(2,2)
            for permutation in itertools.permutations(range(4)):
                rank = np.array(permutation).reshape(2,2)
                cost,ends = engine.score_priority(b,a,rank)
                ref = old_replay(b,a,rank)
                assert ref['objective']==cost and np.array_equal(ref['ends'],ends)
                row = {'assignment':a.tolist(),'priority':rank.tolist(),'cost':cost}
                output.write(json.dumps(row)+'\n')
                if best is None or cost < best['cost']:
                    best = row
    cert = certificate(b)
    save(out/'certificate.json',cert)
    a,rank = np.array(best['assignment']),np.array(best['priority'])
    optimal_a = np.array(cert['best']['assignment'])
    cases = [('historical',a,rank,np.zeros((2,2)),True),
             ('zero_reservation',a,rank,np.zeros((2,2)),False),
             ('certified',optimal_a,np.arange(4).reshape(2,2),np.array(cert['best']['starts']),False),
             ('hand_reservation',np.array([[0,1],[1,0]]),np.arange(4).reshape(2,2),np.array([[6.,0.],[0.,0.]]),False)]
    configure_environment()
    costs = {}
    for name,a,rank,release,historical in cases:
        ref = old_replay(b,a,rank) if historical else replay(b,a,rank,release)
        with (out/(name+'_live.log')).open('w') as log:
            actual = old_live(b,a,rank,log,b['seed']) if historical else herosim(b,a,rank,release,log,b['seed'])
        check_live(actual,ref,a)
        save(out/(name+'_live.json'),actual)
        save(out/(name+'_replay.json'),{**ref,'ends':ref['ends'].tolist(),'assignment':a.tolist(),'priority':rank.tolist(),'not_before':release.tolist()})
        costs[name] = ref['objective']
    assert costs['historical']==costs['zero_reservation']==best['cost']
    assert costs['certified']==costs['hand_reservation']==cert['best']['cost']
    report = {'status':'ACTION-SPACE-INCOMPLETE','nondelay_plans':384,'certificate':cert,'costs':costs,
              'gain_pct':(best['cost']-cert['best']['cost'])/best['cost']*100,
              'live_runs':4,'live_operations':16,'training_allowed':False,'gnn_result':False}
    save(out/'read.json',report)
    for name,digest in sources.items():
        assert sha(ROOT/name)==digest,name
    save(out/'artifacts.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')})
    print(json.dumps(report,indent=2))


def audit(out):
    pre = read(out/'protocol_before_run.json')
    for name,digest in pre['sources'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in read(out/'artifacts.json').items():
        assert sha(out/name)==digest,name
    b = fixture()
    assert serialized(b)==read(out/'input.json')
    engine = DispatchNative(out/'build')
    assert engine.provenance==read(out/'native.json')
    plans = [json.loads(line) for line in (out/'placements/placements.jsonl').read_text().splitlines()]
    seen = set()
    for plan in plans:
        a,rank = np.array(plan['assignment']),np.array(plan['priority'])
        key = tuple(a.ravel()),tuple(rank.ravel())
        assert key not in seen
        seen.add(key)
        assert engine.score_priority(b,a,rank)[0]==old_replay(b,a,rank)['objective']==plan['cost']
    expected = set(itertools.product(itertools.product(range(2),repeat=4),itertools.permutations(range(4))))
    assert seen==expected and len(plans)==384
    assert certificate(b)==read(out/'certificate.json')
    report = read(out/'read.json')
    for name in ('historical','zero_reservation','certified','hand_reservation'):
        rec = read(out/(name+'_replay.json'))
        a,rank,release = np.array(rec['assignment']),np.array(rec['priority']),np.array(rec['not_before'])
        ref = old_replay(b,a,rank) if name=='historical' else replay(b,a,rank,release)
        assert ref['objective']==rec['objective']==report['costs'][name]
        assert np.array_equal(ref['ends'],rec['ends']) and ref['events']==rec['events']
        check_live(read(out/(name+'_live.json')),ref,a)
    assert report['costs']['historical']==min(p['cost'] for p in plans)
    assert report['costs']['certified']==certificate(b)['best']['cost']
    assert report['costs']['zero_reservation']==report['costs']['historical']
    assert report['costs']['hand_reservation']==report['costs']['certified']
    assert report['gain_pct']==(report['costs']['historical']-report['costs']['certified'])/report['costs']['historical']*100
    result = {'status':'PASS','nondelay_plans':len(plans),'live_runs':4,'live_operations':16,'report_sha256':sha(out/'read.json')}
    save(out/'AUDIT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--audit-only',action='store_true')
    args = parser.parse_args()
    audit(args.out) if args.audit_only else run(args.out)

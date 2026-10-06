"""Optimistic headroom and serving-cost screen for exact region selection."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from src.placement.radical.block_moves import scaled_problem
from src.placement.radical.environment import serialized
from src.placement.radical.dispatch_live import replay
from src.placement.radical.region_dispatch import RegionDispatch, region_pool
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha, control, record
from scripts_cosim.mixed_dispatch_block_screen import parity
from scripts_cosim.workflow_proposal_gate import configure_environment
from scripts_cosim.audit_radical_physics import load_problem

PROTOCOL = ROOT / 'experiments/mixed_region_repair_s0_v1.json'


def summarize(rows, protocol):
    improvements = [(r['control']['cost'] - min(r['control']['cost'], r['best_repair']['cost'])) / r['control']['cost'] * 100 for r in rows]
    paths = [r['start']['time_ms'] + r['pool_ms'] + t + protocol['future_selector_reserve_ms']
             for r in rows for t in r['winning_repair_ms']]
    timing = (max(paths) <= protocol['budget_ms']
              and all(t['time_ms'] <= protocol['budget_ms'] for r in rows for t in r['control_repeats'])
              and all(t['time_ms'] <= protocol['partial_start_budget_ms'] for r in rows for t in r['start_repeats']))
    passed = float(np.median(improvements)) >= protocol['advance_median_headroom_pct'] and timing
    return {'status': 'S0-PASS-ONLY' if passed else 'S0-NO-GO', 'training_allowed': False,
            'median_headroom_pct': float(np.median(improvements)), 'per_input_headroom_pct': improvements,
            'optimistic_path_max_ms': max(paths), 'timing_pass': timing,
            'live_runs': len(rows) * 3, 'live_operations': len(rows) * 3 * math.prod(protocol['shape'][:2]),
            'exact_candidates': len(rows) * protocol['regions_per_input'] * 384}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(PROTOCOL)
    previous = read(ROOT / 'simulation_data/gnn_environment_search_v1/mixed_joint_dispatch_screen_v1/protocol_before_run.json')['sources']
    for name, digest in previous.items():
        assert sha(ROOT / name) == digest, name
    sources = {**previous, **{str(p.relative_to(ROOT)): sha(p) for p in (PROTOCOL, Path(__file__).resolve(),
               ROOT / 'src/placement/radical/region_dispatch.py', ROOT / 'src/placement/radical/region_dispatch.cpp')}}
    save(out / 'protocol_before_run.json', {'protocol': protocol, 'sources': sources})
    engine = RegionDispatch(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    rows = []
    for seed in range(protocol['seeds'][0], protocol['seeds'][1] + 1):
        b = scaled_problem(seed, *protocol['shape'])
        cell = out / str(seed)
        cell.mkdir()
        save(cell / 'input.json', serialized(b))
        full = [control(engine, b, 3, False, protocol['budget_ms']) for _ in range(protocol['timing_repeats'])]
        partial = [control(engine, b, 3, False, protocol['partial_start_budget_ms']) for _ in range(protocol['timing_repeats'])]
        chosen, start = max(full, key=lambda x:x['cost']), max(partial, key=lambda x:x['cost'])
        a, rank = np.array(start['assignment']), np.array(start['priority'])
        tick = time.perf_counter()
        regions = region_pool(b, a, rank, protocol['regions_per_input'])
        pool_ms = (time.perf_counter() - tick) * 1000
        repairs = []
        for nodes in regions:
            tick = time.perf_counter()
            cost, aa, rr = engine.repair(b, a, rank, nodes)
            repairs.append(record(cost, aa, rr, region=list(nodes), time_ms=(time.perf_counter()-tick)*1000))
        best = min(repairs, key=lambda x:x['cost'])
        repeat_times = []
        for _ in range(protocol['timing_repeats']):
            tick = time.perf_counter()
            repeated = engine.repair(b, a, rank, best['region'])
            repeat_times.append((time.perf_counter()-tick)*1000)
            assert repeated[0] == best['cost']
            assert np.array_equal(repeated[1], best['assignment']) and np.array_equal(repeated[2], best['priority'])
        row = {'seed':seed, 'control_repeats':full, 'start_repeats':partial, 'control':chosen, 'start':start,
               'pool_ms':pool_ms, 'repairs':repairs, 'best_repair':best, 'winning_repair_ms':repeat_times}
        save(cell / 'read.json', row)
        rows.append(row)
        print('REGION', seed, 'headroom_pct', round((chosen['cost']-min(chosen['cost'],best['cost']))/chosen['cost']*100,3),
              'repair_ms',round(max(repeat_times),3),flush=True)
    for row in rows:
        b = scaled_problem(row['seed'], *protocol['shape'])
        cell = out / str(row['seed'])
        (cell / 'placements').mkdir()
        with (cell / 'placements/placements.jsonl').open('w') as output:
            for name in ('start', 'control', 'best_repair'):
                arm = row[name]
                parity(engine, b, np.array(arm['assignment']), np.array(arm['priority']), cell, name, live=True)
                output.write(json.dumps({'arm':name, **{k:arm[k] for k in ('cost','assignment','priority')}})+'\n')
        print('REGION LIVE',row['seed'],flush=True)
    report = summarize(rows, protocol)
    save(out / 'read.json', report)
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, name
    save(out / 'artifacts.json', {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json','.jsonl')})
    print(json.dumps(report,indent=2),flush=True)


def audit(out):
    pre = read(out / 'protocol_before_run.json')
    protocol = pre['protocol']
    for name, digest in pre['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(out / 'artifacts.json').items():
        assert sha(out / name) == digest, name
    engine = RegionDispatch(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    rows, runs, operations = [], 0, 0
    for seed in range(protocol['seeds'][0], protocol['seeds'][1]+1):
        cell = out / str(seed)
        b = load_problem(cell / 'input.json')
        assert serialized(scaled_problem(seed,*protocol['shape'])) == read(cell/'input.json')
        row = read(cell / 'read.json')
        rows.append(row)
        assert row['control'] == max(row['control_repeats'],key=lambda x:x['cost'])
        assert row['start'] == max(row['start_repeats'],key=lambda x:x['cost'])
        for trial in row['control_repeats'] + row['start_repeats']:
            assert replay(b, np.array(trial['assignment']), np.array(trial['priority']))['objective'] == trial['cost']
        a, rank = np.array(row['start']['assignment']), np.array(row['start']['priority'])
        assert [list(r) for r in region_pool(b,a,rank,protocol['regions_per_input'])] == [r['region'] for r in row['repairs']]
        for repair in row['repairs']:
            cost, aa, rr = engine.repair(b,a,rank,repair['region'])
            assert cost == repair['cost'] == replay(b,aa,rr)['objective']
            assert np.array_equal(aa,repair['assignment']) and np.array_equal(rr,repair['priority'])
        assert row['best_repair'] == min(row['repairs'],key=lambda x:x['cost'])
        plans = [json.loads(line) for line in (cell/'placements/placements.jsonl').read_text().splitlines()]
        assert [p['arm'] for p in plans] == ['start','control','best_repair']
        for plan in plans:
            name = plan['arm']
            for key in ('cost','assignment','priority'):
                assert plan[key] == row[name][key]
            rec = read(cell / (name+'_replay.json'))
            actual = read(cell / (name+'_live.json'))
            aa, rr = np.array(plan['assignment']), np.array(plan['priority'])
            independent = replay(b,aa,rr)
            assert rec['assignment'] == plan['assignment'] and rec['priority'] == plan['priority']
            assert rec['objective'] == independent['objective'] == plan['cost']
            assert rec['events'] == independent['events'] and np.array_equal(rec['ends'],independent['ends'])
            assert abs(actual['total_rtt']*1000-plan['cost'])<1e-6
            seen = set()
            for task in actual['tasks']:
                j,k = map(int,task['taskType']['name'][2:].split('_'))
                assert (j,k) not in seen
                seen.add((j,k))
                assert task['executionNode'] == f'node{aa[j,k]}'
                assert abs(task['doneTime']*1000-independent['ends'][j,k])<1e-6
            assert len(seen) == aa.size
            runs += 1
            operations += aa.size
        print('REGION AUDIT',seed,flush=True)
    report = read(out/'read.json')
    assert report == summarize(rows,protocol)
    assert (runs,operations)==(report['live_runs'],report['live_operations'])
    result = {'status':'PASS','inputs':len(rows),'live_runs':runs,'live_operations':operations,
              'exact_candidates_reexecuted':report['exact_candidates'],'report_sha256':sha(out/'read.json')}
    save(out/'AUDIT.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    audit(args.out) if args.audit_only else run(args.out)

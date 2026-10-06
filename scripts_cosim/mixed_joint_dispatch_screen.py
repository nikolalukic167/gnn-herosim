"""Joint placement/order opportunity screen against deadline-aware compiled controls."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from src.placement.radical.block_moves import scaled_problem, baseline
from src.placement.radical.environment import initial, serialized, pack
from src.placement.radical.dispatch import priorities
from src.placement.radical.dispatch_live import replay
from src.placement.radical.joint_dispatch import JointDispatch
from scripts_cosim.mixed_dispatch_block_screen import parity
from scripts_cosim.workflow_proposal_gate import configure_environment
from scripts_cosim.audit_radical_physics import load_problem

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/mixed_joint_dispatch_screen_v1.json'


def read(path):
    return json.loads(path.read_text())


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(cost, a, rank, **extra):
    return {'cost': cost, 'assignment': a.tolist(), 'priority': rank.tolist(), **extra}


def control(engine, b, mode, anneal, budget_ms):
    tick = time.perf_counter()
    cost, a, rank = baseline(engine, b)
    remaining = budget_ms / 1000 - (time.perf_counter() - tick) - .002
    scores = 0
    if remaining > .001:
        cost, a, rank, scores = engine.joint_timed(b, a, rank, remaining, mode=mode, anneal=anneal, seed=77)
    return record(cost, a, rank, time_ms=(time.perf_counter() - tick) * 1000, scores=scores)


def arm_read(engine, b, protocol, budget_ms):
    result = {}
    for name, (mode, anneal) in protocol['hand_modes'].items():
        repeats = [control(engine, b, mode, anneal, budget_ms) for _ in range(protocol['timing_repeats'])]
        worst = max(repeats, key=lambda r: r['cost'])
        result[name] = {**worst, 'repeats': repeats, 'all_within_budget': all(r['time_ms'] <= budget_ms for r in repeats)}
    return result


def gains(reference, control_costs):
    data = (control_costs - reference) / control_costs * 100
    rng = np.random.default_rng(773)
    boot = np.median(rng.choice(data, (10000, len(data)), replace=True), axis=1)
    return {'median_pct': float(np.median(data)), 'mean_pct': float(np.mean(data)),
            'ci95_pct': np.quantile(boot, [.025, .975]).tolist(), 'per_environment_pct': data.tolist()}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(PROTOCOL)
    old = read(ROOT / 'simulation_data/gnn_environment_search_v1/mixed_dispatch_block_screen_v1/protocol_before_run.json')['sources']
    for name, digest in old.items():
        if sha(ROOT / name) != digest:
            raise ValueError('historical source changed: ' + name)
    sources = {**old, **{str(p.relative_to(ROOT)): sha(p) for p in (PROTOCOL, Path(__file__).resolve(),
                ROOT / 'src/placement/radical/joint_dispatch.py', ROOT / 'src/placement/radical/joint_dispatch.cpp')}}
    save(out / 'protocol_before_run.json', {'protocol': protocol, 'sources': sources})
    engine = JointDispatch(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    identities = set()
    def case(seed, shape, parent):
        b = scaled_problem(seed, *shape)
        identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
        if identity in identities or not np.array_equal(pack(b), pack(scaled_problem(seed, *shape))):
            raise ValueError('input collision or nondeterminism')
        identities.add(identity)
        cell = out / parent / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        return b, cell
    for seed in (142900, 142901):
        b, cell = case(seed, [4,4,8], 'parity')
        _, a, rank = baseline(engine, b)
        for mode in (0, 1, 3):
            _, chosen, priority = engine.joint_search(b, a, rank, 128, mode=mode)
            parity(engine, b, chosen, priority, cell, f'mode{mode}', live=True)
    print('JOINT PARITY PASS', flush=True)
    calibration = []
    cal_budget = protocol['budget_ms'] * protocol['calibration_fraction']
    for seed in range(protocol['calibration_seeds'][0], protocol['calibration_seeds'][1] + 1):
        b, cell = case(seed, protocol['shape'], 'calibration')
        arms = arm_read(engine, b, protocol, cal_budget)
        row = {'seed': seed, 'arms': arms}
        calibration.append(row)
        save(cell / 'read.json', row)
        print('CAL', seed, {name: arm['cost'] for name, arm in arms.items()}, flush=True)
    eligible = [name for name in protocol['hand_modes'] if all(row['arms'][name]['all_within_budget'] for row in calibration)]
    if not eligible:
        raise RuntimeError('no timing-eligible calibration control')
    selected = min(eligible, key=lambda name: np.mean([row['arms'][name]['cost'] for row in calibration]))
    save(out / 'calibration.json', {'selected': selected, 'eligible': eligible, 'rows': calibration})
    print('FROZEN CONTROL', selected, flush=True)
    rows = []
    for seed in range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1] + 1):
        b, cell = case(seed, protocol['shape'], 'fresh')
        base = record(*baseline(engine, b))
        arms = arm_read(engine, b, protocol, protocol['budget_ms'])
        starts = {'selected_control': (np.array(arms[selected]['assignment']), np.array(arms[selected]['priority']))}
        fastest = initial(b, 'fastest')
        starts['fastest_srpt'] = (fastest, priorities(b, fastest, 'srpt'))
        references = []
        for start_name in protocol['reference_starts']:
            for search_seed in protocol['reference_seeds']:
                cost, a, rank = engine.joint_search(b, *starts[start_name], protocol['reference_steps_per_start'], mode=3, anneal=True, seed=search_seed)
                references.append(record(cost, a, rank, start=start_name, seed=search_seed,
                                         steps=protocol['reference_steps_per_start']))
        best = min([base, *arms.values(), *references], key=lambda r: r['cost'])
        row = {'seed': seed, 'baseline': base, 'arms': arms, 'references': references,
               'reference': {key: best[key] for key in ('cost','assignment','priority')}}
        save(cell / 'read.json', row)
        rows.append(row)
        print('SCREEN', seed, 'reference_gain_vs_control_pct', round((arms[selected]['cost'] - best['cost']) / arms[selected]['cost'] * 100, 3), flush=True)
    fresh_eligible = [name for name in protocol['hand_modes'] if all(row['arms'][name]['all_within_budget'] for row in rows)]
    costs = {name: np.array([row['arms'][name]['cost'] for row in rows]) for name in protocol['hand_modes']}
    ref = np.array([row['reference']['cost'] for row in rows])
    primary = gains(ref, costs[selected])
    envelope = gains(ref, np.min(np.stack([costs[name] for name in fresh_eligible]), axis=0)) if fresh_eligible else None
    timing = selected in fresh_eligible
    passes = timing and primary['median_pct'] >= protocol['advance_median_headroom_pct'] and primary['ci95_pct'][0] > protocol['advance_bootstrap_lower_pct']
    report = {'status': 'PENDING-LIVE', 'selected': selected, 'fresh_eligible': fresh_eligible,
              'reference_vs_selected': primary, 'reference_vs_eligible_envelope': envelope,
              'per_arm_reference_gains': {name: gains(ref, costs[name]) for name in costs},
              'joint_control_vs_ordering': gains(costs[selected], costs['ordering']),
              'headroom_pass': passes, 'selected_timing_pass': timing, 'training_allowed': False,
              'live_runs': 0, 'live_operations': 0}
    save(out / 'read.json', report)
    for row in rows:
        b = scaled_problem(row['seed'], *protocol['shape'])
        cell = out / 'fresh' / str(row['seed'])
        (cell / 'placements').mkdir()
        with (cell / 'placements/placements.jsonl').open('w') as output:
            for name, arm in [('baseline', row['baseline']), ('control', row['arms'][selected]), ('reference', row['reference'])]:
                parity(engine, b, np.array(arm['assignment']), np.array(arm['priority']), cell, name, live=True)
                output.write(json.dumps({'arm': name, **{k: arm[k] for k in ('cost','assignment','priority')}}) + '\n')
                report['live_runs'] += 1
                report['live_operations'] += np.prod(protocol['shape'][:2]).item()
        print('LIVE', row['seed'], flush=True)
    report['status'] = 'HEADROOM-ONLY-PASS' if passes else 'PRETRAINING-NO-GO'
    save(out / 'read.json', report)
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError('source changed: ' + name)
    save(out / 'artifacts.json', {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.json', '.jsonl')})
    print(json.dumps(report, indent=2), flush=True)


def audit(out):
    pre = read(out / 'protocol_before_run.json')
    protocol = pre['protocol']
    for name, digest in pre['sources'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in read(out / 'artifacts.json').items():
        assert sha(out / name) == digest, name
    engine = JointDispatch(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    identities, live_runs, live_ops, reference_proposals = set(), 0, 0, 0
    for input_path in sorted(out.glob('*/*/input.json')):
        b = load_problem(input_path)
        assert np.array_equal(pack(b), pack(scaled_problem(b['seed'], *b['p'].shape)))
        identity = hashlib.sha256(pack(b).tobytes()).hexdigest()
        assert identity not in identities
        identities.add(identity)
        cell = input_path.parent
        for path in cell.glob('*_replay.json'):
            rec = read(path)
            a, rank = np.array(rec['assignment']), np.array(rec['priority'])
            cost, ends = engine.score_priority(b, a, rank)
            independent = replay(b, a, rank)
            assert cost == rec['objective'] == independent['objective']
            assert np.array_equal(ends, rec['ends']) and np.array_equal(ends, independent['ends'])
            assert rec['events'] == independent['events']
            actual = read(path.with_name(path.name.replace('_replay.json','_live.json')))
            assert abs(actual['total_rtt'] * 1000 - cost) < 1e-6
            seen = set()
            for task in actual['tasks']:
                j, k = map(int, task['taskType']['name'][2:].split('_'))
                assert (j,k) not in seen
                seen.add((j,k))
                assert task['executionNode'] == f'node{a[j,k]}'
                assert abs(task['doneTime'] * 1000 - ends[j,k]) < 1e-6
            assert len(seen) == a.size
            live_runs += 1
            live_ops += a.size
        if not (cell / 'read.json').exists():
            continue
        row = read(cell / 'read.json')
        for arm in row['arms'].values():
            assert arm['cost'] == max(q['cost'] for q in arm['repeats'])
            for trial in arm['repeats']:
                a, rank = np.array(trial['assignment']), np.array(trial['priority'])
                assert engine.score_priority(b, a, rank)[0] == trial['cost'] == replay(b, a, rank)['objective']
            limit = protocol['budget_ms'] * (protocol['calibration_fraction'] if cell.parent.name == 'calibration' else 1)
            assert arm['all_within_budget'] == all(q['time_ms'] <= limit for q in arm['repeats'])
        if cell.parent.name == 'fresh':
            selected = read(out / 'calibration.json')['selected']
            fast = initial(b, 'fastest')
            starts = {'selected_control': (np.array(row['arms'][selected]['assignment']), np.array(row['arms'][selected]['priority'])),
                      'fastest_srpt': (fast, priorities(b, fast, 'srpt'))}
            assert row['baseline'] == record(*baseline(engine, b))
            assert len(row['references']) == len(protocol['reference_seeds']) * len(protocol['reference_starts'])
            for ref in row['references']:
                cost, a, rank = engine.joint_search(b, *starts[ref['start']], ref['steps'], mode=3, anneal=True, seed=ref['seed'])
                assert ref['steps'] == protocol['reference_steps_per_start']
                assert cost == ref['cost'] == replay(b, a, rank)['objective']
                assert np.array_equal(a, ref['assignment']) and np.array_equal(rank, ref['priority'])
                reference_proposals += ref['steps']
            best = min([row['baseline'], *row['arms'].values(), *row['references']], key=lambda r: r['cost'])
            assert row['reference'] == {k: best[k] for k in ('cost','assignment','priority')}
            plans = [json.loads(line) for line in (cell / 'placements/placements.jsonl').read_text().splitlines()]
            assert [p['arm'] for p in plans] == ['baseline','control','reference']
            for plan, expected in zip(plans, (row['baseline'],row['arms'][selected],row['reference'])):
                for key in ('cost','assignment','priority'):
                    assert plan[key] == expected[key]
                retained = read(cell / (plan['arm'] + '_replay.json'))
                assert retained['assignment'] == plan['assignment'] and retained['priority'] == plan['priority']
            print('AUDIT', row['seed'], flush=True)
    calibration = read(out / 'calibration.json')
    for row in calibration['rows']:
        assert row == read(out / 'calibration' / str(row['seed']) / 'read.json')
    expected_cal = set(range(protocol['calibration_seeds'][0], protocol['calibration_seeds'][1]+1))
    expected_fresh = set(range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1]+1))
    assert {int(p.name) for p in (out / 'calibration').iterdir()} == expected_cal
    assert {int(p.name) for p in (out / 'fresh').iterdir()} == expected_fresh
    eligible = [n for n in protocol['hand_modes'] if all(row['arms'][n]['all_within_budget'] for row in calibration['rows'])]
    selected = min(eligible, key=lambda n: np.mean([row['arms'][n]['cost'] for row in calibration['rows']]))
    assert calibration['eligible'] == eligible and calibration['selected'] == selected
    rows = [read(out / 'fresh' / str(seed) / 'read.json') for seed in sorted(expected_fresh)]
    report = read(out / 'read.json')
    ref = np.array([r['reference']['cost'] for r in rows])
    costs = {name: np.array([r['arms'][name]['cost'] for r in rows]) for name in protocol['hand_modes']}
    fresh_eligible = [n for n in protocol['hand_modes'] if all(r['arms'][n]['all_within_budget'] for r in rows)]
    assert report['selected'] == selected and report['fresh_eligible'] == fresh_eligible
    assert report['reference_vs_selected'] == gains(ref, costs[selected])
    envelope = gains(ref, np.min(np.stack([costs[n] for n in fresh_eligible]),axis=0)) if fresh_eligible else None
    assert report['reference_vs_eligible_envelope'] == envelope
    assert report['per_arm_reference_gains'] == {n:gains(ref,costs[n]) for n in costs}
    assert report['joint_control_vs_ordering'] == gains(costs[selected],costs['ordering'])
    passes = selected in fresh_eligible and gains(ref,costs[selected])['median_pct'] >= protocol['advance_median_headroom_pct'] and gains(ref,costs[selected])['ci95_pct'][0] > protocol['advance_bootstrap_lower_pct']
    assert report['headroom_pass'] == passes and report['selected_timing_pass'] == (selected in fresh_eligible)
    assert report['status'] == ('HEADROOM-ONLY-PASS' if passes else 'PRETRAINING-NO-GO')
    assert report['live_runs'] == 48 and report['live_operations'] == 12288
    assert live_runs == 54 and live_ops == 12384
    result = {'status':'PASS','unique_inputs':len(identities),'live_runs':live_runs,'live_operations':live_ops,
              'reference_proposals_reexecuted':reference_proposals,'report_sha256':sha(out/'read.json')}
    save(out / 'AUDIT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    audit(args.out) if args.audit_only else run(args.out)

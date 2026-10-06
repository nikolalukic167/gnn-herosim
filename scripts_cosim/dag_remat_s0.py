"""Registered CPU and actual-HeROsim screen for one-level DAG rematerialization."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from scripts_cosim.dag_memory_lifetime_s0 import RULES
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dag_memory import hand_retention
from src.placement.radical.dag_remat import problem, replay
from src.placement.radical.dag_remat_live import herosim
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.placement.radical.environment import pack, serialized

PROTOCOL = ROOT / 'experiments/dag_remat_s0_v1.json'


def identity(b):
    return hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes() +
                          b['output_size_mb'].tobytes() + b['host_memory_mb'].tobytes() +
                          b['memory_contract'].encode()).hexdigest()


def candidates(b, a):
    jobs, ops = a.shape
    output = []
    remat = []
    for j in range(jobs):
        for k in range(ops):
            children = [z for z in range(k + 1, ops) if int(b['predecessors'][j, z]) & (1 << k)]
            if not children:
                continue
            output.append((j, k))
            if all(np.isfinite(b['p'][j, k, int(a[j, z])]) for z in children):
                remat.append((j, k))
    return output, remat


def cheap_actions(b, a):
    _, eligible = candidates(b, a)
    action = np.zeros(a.shape, dtype=np.int8)
    for j, k in eligible:
        children = [z for z in range(k + 1, a.shape[1]) if int(b['predecessors'][j, z]) & (1 << k)]
        grand = [z for z in range(k) if int(b['predecessors'][j, k]) & (1 << z)]
        remat = np.mean([b['p'][j, k, int(a[j, z])] +
                         sum(b['output_size_mb'][j, g] * b['store_read_ms_per_mb'] for g in grand)
                         for z in children])
        action[j, k] = int(remat < b['output_size_mb'][j, k] * b['store_read_ms_per_mb'])
    return action


def record(b, a, rank, keep, recompute, result=None):
    result = replay(b, a, rank, keep, recompute) if result is None else result
    return {'cost': result['objective'], 'retain': np.asarray(keep, dtype=np.int8).tolist(),
            'recompute': np.asarray(recompute, dtype=np.int8).tolist(),
            'counts': result['counts'], 'starts': result['starts'].tolist(),
            'ends': result['ends'].tolist()}


def simple_controls(b, a, rank):
    zero = np.zeros(a.shape, dtype=np.int8)
    cheap = cheap_actions(b, a)
    result = {rule: record(b, a, rank, hand_retention(b, a, rule), zero) for rule in RULES}
    for rule in ('keep_all', 'spill_all', 'local_reuse'):
        result[f'{rule}_cheap_remat'] = record(b, a, rank, hand_retention(b, a, rule), cheap)
    return result


def make_case(engine, seed, shape):
    b = problem(seed, *shape)
    tick = time.perf_counter()
    _, a, rank, _, _ = baseline(engine, b, mode=0)
    simple = simple_controls(b, a, rank)
    start = min(simple, key=lambda rule: simple[rule]['cost'])
    return b, a, rank, simple, start, (time.perf_counter() - tick) * 1000


def search(b, a, rank, initial, mode, seed, seconds=None, steps=None):
    if mode not in ('greedy', 'anneal'):
        raise ValueError('unknown search mode')
    rng = np.random.default_rng(seed)
    tick = time.perf_counter()
    keep = np.asarray(initial['retain'], dtype=np.int8).copy()
    recompute = np.asarray(initial['recompute'], dtype=np.int8).copy()
    value = float(initial['cost'])
    best_keep, best_recompute, best_value = keep.copy(), recompute.copy(), value
    output, remat = candidates(b, a)
    ordered = [(0, j, k) for j, k in sorted(output, key=lambda x: (-b['output_size_mb'][x], x))]
    ordered += [(1, j, k) for j, k in sorted(remat, key=lambda x: (-b['output_size_mb'][x], x))]
    if not ordered:
        raise ValueError('no rematerialization moves')
    count = 0
    while steps is None or count < steps:
        if seconds is not None and time.perf_counter() - tick >= max(0., seconds - .012):
            break
        chosen = ordered[count % len(ordered)] if mode == 'greedy' else ordered[int(rng.integers(len(ordered)))]
        trial_keep, trial_recompute = keep.copy(), recompute.copy()
        (trial_keep if chosen[0] == 0 else trial_recompute)[chosen[1], chosen[2]] ^= 1
        outcome = replay(b, a, rank, trial_keep, trial_recompute)
        trial_cost = outcome['objective']
        if trial_cost < best_value:
            best_keep, best_recompute, best_value = trial_keep.copy(), trial_recompute.copy(), trial_cost
        accept = trial_cost < value
        if mode == 'anneal' and not accept:
            progress = count / max(1, steps) if steps is not None else min(1., (time.perf_counter() - tick) / seconds)
            temperature = max(.01, initial['cost'] * .01 * (.01 ** progress))
            accept = rng.random() < np.exp((value - trial_cost) / temperature)
        if accept:
            keep, recompute, value = trial_keep, trial_recompute, trial_cost
        count += 1
    return {**record(b, a, rank, best_keep, best_recompute),
            'scores': count, 'time_ms': (time.perf_counter() - tick) * 1000}


def live_case(b, a, rank, plan, cell, name):
    expected = replay(b, a, rank, plan['retain'], plan['recompute'])
    assert expected['objective'] == plan['cost'] and expected['counts'] == plan['counts']
    with (cell / f'{name}.log').open('w') as log:
        result = herosim(b, a, rank, plan['retain'], plan['recompute'], log)
    assert abs(result['job_completion_sum'] * 1000 - expected['objective']) < 1e-6
    assert result['mixed_execution']['counts'] == expected['counts']
    for task in result['tasks']:
        j, k = map(int, task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime'] * 1000 - expected['ends'][j, k]) < 1e-6
    save(cell / f'{name}_live.json', result)
    return a.size


def run(out, protocol_path=PROTOCOL):
    out.mkdir(parents=True, exist_ok=False)
    protocol_path = Path(protocol_path).resolve()
    protocol = read(protocol_path)
    sources = [protocol_path, Path(__file__).resolve(), ROOT / 'src/placement/radical/dag_remat.py',
               ROOT / 'src/placement/radical/dag_remat_live.py',
               ROOT / 'src/placement/radical/dag_memory.py',
               ROOT / 'src/placement/radical/dag_memory_live.py',
               ROOT / 'src/placement/radical/dag_reservation.cpp']
    save(out / 'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(path.relative_to(ROOT)): sha(path) for path in sources}})
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    identities = set()
    calibration = []
    for seed in range(protocol['calibration_seeds'][0], protocol['calibration_seeds'][1] + 1):
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, seed, protocol['shape'])
        key = identity(b)
        assert key not in identities
        identities.add(key)
        arms = {mode: search(b, a, rank, simple[initial_rule], mode, seed + 31,
                             seconds=max(0., protocol['budget_ms'] - prep_ms - 8) / 1000)
                for mode in protocol['hand_modes']}
        for arm in arms.values():
            arm['search_ms'] = arm['time_ms']
            arm['time_ms'] += prep_ms
        row = {'seed': seed, 'identity': key, 'assignment': a.tolist(), 'priority': rank.tolist(),
               'initial_rule': initial_rule, 'preparation_ms': prep_ms, 'simple': simple, 'arms': arms}
        cell = out / 'calibration' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        calibration.append(row)
        print('CAL', seed, initial_rule, {name: arm['cost'] for name, arm in arms.items()}, flush=True)
    eligible = [mode for mode in protocol['hand_modes']
                if all(row['arms'][mode]['time_ms'] <= protocol['budget_ms'] for row in calibration)]
    if not eligible:
        raise RuntimeError('no timing-eligible hand remat search')
    selected = min(eligible, key=lambda mode: np.mean([row['arms'][mode]['cost'] for row in calibration]))
    save(out / 'calibration.json', {'selected': selected, 'eligible': eligible, 'rows': calibration})
    print('SELECTED', selected, flush=True)
    rows = []
    for seed in range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1] + 1):
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, seed, protocol['shape'])
        key = identity(b)
        assert key not in identities
        identities.add(key)
        initial = simple[initial_rule]
        hand = search(b, a, rank, initial, selected, seed + 31,
                      seconds=max(0., protocol['budget_ms'] - prep_ms - 8) / 1000)
        hand['search_ms'] = hand['time_ms']
        hand['time_ms'] += prep_ms
        references = [search(b, a, rank, source, 'anneal', seed + 71 + offset,
                             steps=protocol['reference_steps'])
                      for offset, source in enumerate((initial, hand))]
        reference = min([initial, hand, *references], key=lambda plan: plan['cost'])
        row = {'seed': seed, 'identity': key, 'assignment': a.tolist(), 'priority': rank.tolist(),
               'initial_rule': initial_rule, 'preparation_ms': prep_ms, 'simple': simple,
               'hand': hand, 'references': references, 'reference': reference}
        cell = out / 'fresh' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        rows.append(row)
        print('FRESH', seed, 'gain_pct', round(100 * (hand['cost'] - reference['cost']) / hand['cost'], 3),
              'remat', reference['counts']['rematerializations'], flush=True)
    gains = np.array([100 * (row['hand']['cost'] - row['reference']['cost']) / row['hand']['cost'] for row in rows])
    capture = np.array([(row['simple'][row['initial_rule']]['cost'] - row['hand']['cost']) /
                        (row['simple'][row['initial_rule']]['cost'] - row['reference']['cost'])
                        if row['simple'][row['initial_rule']]['cost'] > row['reference']['cost'] else 1.
                        for row in rows])
    binding = np.array([row['reference']['counts']['binding_completions'] > 0 for row in rows])
    remat = np.array([row['reference']['counts']['rematerializations'] > 0 for row in rows])
    report = {'fresh_cases': len(rows), 'median_reference_headroom_pct': float(np.median(gains)),
              'median_hand_capture': float(np.median(capture)),
              'binding_fraction': float(binding.mean()), 'remat_fraction': float(remat.mean()),
              'headroom_pass': bool(np.median(gains) >= protocol['headroom_bar_pct']),
              'hardness_pass': bool(np.median(capture) < protocol['hand_capture_ceiling']),
              'binding_pass': bool(binding.mean() >= protocol['binding_fraction_bar']),
              'remat_pass': bool(remat.mean() >= protocol['remat_fraction_bar'])}
    save(out / 'read_before_live.json', report)
    live_runs = operations = 0
    seed = protocol['calibration_seeds'][0]
    row = read(out / 'calibration' / str(seed) / 'read.json')
    b = problem(seed, *protocol['shape'])
    operations += live_case(b, np.array(row['assignment']), np.array(row['priority']),
                            row['simple'][row['initial_rule']], out / 'calibration' / str(seed), 'initial')
    live_runs += 1
    for row in rows:
        b = problem(row['seed'], *protocol['shape'])
        a, rank = np.array(row['assignment']), np.array(row['priority'])
        cell = out / 'fresh' / str(row['seed'])
        for name, plan in [('initial', row['simple'][row['initial_rule']]),
                           ('hand', row['hand']), ('reference', row['reference'])]:
            operations += live_case(b, a, rank, plan, cell, name)
            live_runs += 1
        print('LIVE', row['seed'], flush=True)
    report.update(live_runs=live_runs, live_operations=operations, selected_hand=selected,
                  maximum_hand_time_ms=max(row['hand']['time_ms'] for row in rows))
    save(out / 'read.json', report)
    for name, digest in read(out / 'protocol_before_run.json')['sources'].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.out, args.protocol)

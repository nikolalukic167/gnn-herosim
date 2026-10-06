"""CPU and actual-HeROsim qualification for DAG output-memory lifetimes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.placement.radical.dag_memory import hand_retention, problem, replay
from src.placement.radical.dag_memory_live import herosim
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline
from src.placement.radical.environment import pack, serialized
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL = ROOT / 'experiments/dag_memory_lifetime_s0_v1_corrected.json'
RULES = ('spill_all', 'keep_all', 'reuse', 'local_reuse', 'benefit_per_mb')


def identity(b):
    return hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes() +
                          b['output_size_mb'].tobytes() + b['host_memory_mb'].tobytes() +
                          str((b['store_read_ms_per_mb'], b['peer_read_ms_per_mb'])).encode()).hexdigest()


def record(b, a, rank, keep, result=None):
    result = replay(b, a, rank, keep) if result is None else result
    return {'cost': result['objective'], 'retain': np.asarray(keep, dtype=np.int8).tolist(),
            'counts': result['counts'], 'starts': result['starts'].tolist(),
            'ends': result['ends'].tolist()}


def simple_controls(b, a, rank):
    return {rule: record(b, a, rank, hand_retention(b, a, rule)) for rule in RULES}


def search(b, a, rank, initial, mode, seed, seconds=None, steps=None):
    if mode not in ('greedy', 'anneal'):
        raise ValueError('unknown memory search')
    rng = np.random.default_rng(seed)
    tick = time.perf_counter()
    current = np.asarray(initial['retain'], dtype=np.int8).copy()
    value = float(initial['cost'])
    best, best_value = current.copy(), value
    candidates = [i for i in range(current.size) if any(int(b['predecessors'][i // current.shape[1], k]) &
                  (1 << (i % current.shape[1])) for k in range(i % current.shape[1] + 1, current.shape[1]))]
    order = sorted(candidates, key=lambda i: (-b['output_size_mb'].flat[i], i))
    count = 0
    while steps is None or count < steps:
        if seconds is not None and time.perf_counter() - tick >= seconds - .012:
            break
        if mode == 'greedy':
            node = order[count % len(order)]
            chosen = [node]
        else:
            chosen = [int(rng.choice(candidates))]
            if rng.random() < .25:
                second = int(rng.choice(candidates))
                if second != chosen[0]:
                    chosen.append(second)
        trial = current.copy()
        for node in chosen:
            trial.flat[node] ^= 1
        outcome = replay(b, a, rank, trial)
        cost = outcome['objective']
        if cost < best_value:
            best, best_value = trial.copy(), cost
        accept = cost < value
        if mode == 'anneal' and not accept:
            progress = count / max(1, steps) if steps is not None else min(1., (time.perf_counter() - tick) / seconds)
            temperature = max(.01, initial['cost'] * .01 * (.01 ** progress))
            accept = rng.random() < np.exp((value - cost) / temperature)
        if accept:
            current, value = trial, cost
        count += 1
    return {**record(b, a, rank, best), 'scores': count, 'time_ms': (time.perf_counter() - tick) * 1000}


def make_case(engine, seed, shape):
    b = problem(seed, *shape)
    tick = time.perf_counter()
    _, a, rank, _, _ = baseline(engine, b, mode=0)
    simple = simple_controls(b, a, rank)
    start = min(simple, key=lambda rule: simple[rule]['cost'])
    return b, a, rank, simple, start, (time.perf_counter() - tick) * 1000


def live_case(b, a, rank, p, cell, name):
    expected = replay(b, a, rank, p['retain'])
    assert expected['objective'] == p['cost'] and expected['counts'] == p['counts']
    with (cell / f'{name}.log').open('w') as log:
        result = herosim(b, a, rank, p['retain'], log)
    assert abs(result['job_completion_sum'] * 1000 - expected['objective']) < 1e-6
    assert result['mixed_execution']['counts'] == expected['counts']
    for task in result['tasks']:
        j, k = map(int, task['taskType']['name'][2:].split('_'))
        assert abs(task['doneTime'] * 1000 - expected['ends'][j, k]) < 1e-6
    save(cell / f'{name}_live.json', result)
    return a.size


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(PROTOCOL)
    sources = [PROTOCOL, Path(__file__).resolve(),
               ROOT / 'src/placement/radical/dag_memory.py',
               ROOT / 'src/placement/radical/dag_memory_live.py',
               ROOT / 'src/placement/radical/dag_reservation.py',
               ROOT / 'src/placement/radical/dag_reservation.cpp']
    save(out / 'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources}})
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    shape = protocol['shape']
    calibration = []
    identities = set()
    for seed in range(protocol['calibration_seeds'][0], protocol['calibration_seeds'][1] + 1):
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, seed, shape)
        key = identity(b)
        assert key not in identities
        identities.add(key)
        initial = simple[initial_rule]
        arms = {mode: search(b, a, rank, initial, mode, seed + 31,
                             seconds=max(0, protocol['budget_ms'] - prep_ms - 8) / 1000)
                for mode in protocol['hand_modes']}
        for arm in arms.values():
            arm['search_ms'] = arm['time_ms']
            arm['time_ms'] += prep_ms
        row = {'seed': seed, 'identity': key, 'assignment': a.tolist(), 'priority': rank.tolist(),
               'simple': simple, 'initial_rule': initial_rule, 'preparation_ms': prep_ms, 'arms': arms}
        cell = out / 'calibration' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        calibration.append(row)
        print('CAL', seed, initial_rule, {name: p['cost'] for name, p in arms.items()}, flush=True)
    eligible = [mode for mode in protocol['hand_modes']
                if all(row['arms'][mode]['time_ms'] <= protocol['budget_ms'] for row in calibration)]
    if not eligible:
        raise RuntimeError('no timing-eligible hand memory search')
    selected = min(eligible, key=lambda mode: np.mean([row['arms'][mode]['cost'] for row in calibration]))
    save(out / 'calibration.json', {'selected': selected, 'eligible': eligible, 'rows': calibration})
    print('SELECTED', selected, flush=True)
    rows = []
    for seed in range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1] + 1):
        b, a, rank, simple, initial_rule, prep_ms = make_case(engine, seed, shape)
        key = identity(b)
        assert key not in identities
        identities.add(key)
        initial = simple[initial_rule]
        hand = search(b, a, rank, initial, selected, seed + 31,
                      seconds=max(0, protocol['budget_ms'] - prep_ms - 8) / 1000)
        hand['search_ms'] = hand['time_ms']
        hand['time_ms'] += prep_ms
        references = [search(b, a, rank, source, 'anneal', seed + 71 + offset,
                             steps=protocol['reference_steps'])
                      for offset, source in enumerate((initial, hand))]
        reference = min([initial, hand, *references], key=lambda p: p['cost'])
        row = {'seed': seed, 'identity': key, 'assignment': a.tolist(), 'priority': rank.tolist(),
               'simple': simple, 'initial_rule': initial_rule, 'preparation_ms': prep_ms, 'hand': hand,
               'references': references, 'reference': reference}
        cell = out / 'fresh' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        rows.append(row)
        print('FRESH', seed, 'gain_pct', round(100 * (hand['cost'] - reference['cost']) / hand['cost'], 3),
              'binding', reference['counts']['binding_completions'], flush=True)
    gains = np.array([(r['hand']['cost'] - r['reference']['cost']) / r['hand']['cost'] for r in rows])
    capture = np.array([(r['simple'][r['initial_rule']]['cost'] - r['hand']['cost']) /
                        (r['simple'][r['initial_rule']]['cost'] - r['reference']['cost'])
                        if r['simple'][r['initial_rule']]['cost'] > r['reference']['cost'] else 1.
                        for r in rows])
    binding = np.array([r['reference']['counts']['binding_completions'] > 0 for r in rows])
    report = {'fresh_cases': len(rows), 'median_reference_headroom_pct': float(np.median(gains) * 100),
              'median_hand_capture': float(np.median(capture)), 'binding_fraction': float(binding.mean()),
              'headroom_pass': bool(np.median(gains) * 100 >= protocol['headroom_bar_pct']),
              'hardness_pass': bool(np.median(capture) < protocol['hand_capture_ceiling']),
              'binding_pass': bool(binding.mean() >= protocol['binding_fraction_bar'])}
    save(out / 'read_before_live.json', report)
    live_runs = operations = 0
    for seed in protocol['calibration_seeds']:
        b = problem(seed, *shape)
        row = read(out / 'calibration' / str(seed) / 'read.json')
        if seed == protocol['calibration_seeds'][0]:
            cell = out / 'calibration' / str(seed)
            operations += live_case(b, np.array(row['assignment']), np.array(row['priority']),
                                    row['simple']['keep_all'], cell, 'keep_all')
            live_runs += 1
    for row in rows:
        b = problem(row['seed'], *shape)
        a, rank = np.array(row['assignment']), np.array(row['priority'])
        cell = out / 'fresh' / str(row['seed'])
        for name, p in [('initial', row['simple'][row['initial_rule']]),
                        ('hand', row['hand']), ('reference', row['reference'])]:
            operations += live_case(b, a, rank, p, cell, name)
            live_runs += 1
        print('LIVE', row['seed'], flush=True)
    report.update({'live_runs': live_runs, 'live_operations': operations, 'selected_hand': selected})
    save(out / 'read.json', report)
    for name, digest in read(out / 'protocol_before_run.json')['sources'].items():
        assert sha(ROOT / name) == digest, name
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)

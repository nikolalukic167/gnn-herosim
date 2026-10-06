"""CPU qualification of broad block proposals from a strong 128-operation start."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline, chronological
from src.placement.radical.environment import pack, serialized
from scripts_cosim.dag_block_stepwise_s0 import scored_pool
from scripts_cosim.dag_reservation_screen import live, plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL = ROOT / 'experiments/dag_block_proposer_128_s0_v1.json'


def strong_start(engine, b, protocol):
    tick = time.perf_counter()
    cost, assignment, rank, decoder, rule = baseline(engine, b)
    baseline_ms = (time.perf_counter() - tick) * 1000
    remaining = max(0., (protocol['common_start_budget_ms'] - baseline_ms - 8) / 1000)
    cost, assignment, rank, decoder, count = engine.search(
        b, assignment.copy(), rank.copy(), 2**31 - 1, 2, .005, 77, remaining)
    result = plan(engine, b, assignment, rank, decoder, cost)
    if decoder == 0:
        mapped = chronological(rank, result['starts'])
        converted = plan(engine, b, assignment, mapped, 1)
        if converted['cost'] > result['cost']:
            raise RuntimeError('non-delay to reservation translation worsened strong start')
        result = min((result, converted), key=lambda p: p['cost'])
    result.update(rule=rule, scores=count, time_ms=(time.perf_counter()-tick)*1000)
    if result['time_ms'] > protocol['common_start_budget_ms']:
        raise RuntimeError('strong start exceeded budget')
    return result


def hand_proposer(engine, b, start, protocol):
    tick = time.perf_counter()
    assignment = np.asarray(start['assignment'], dtype=np.int64)
    rank = np.asarray(start['priority'], dtype=np.int64)
    cost, assignment, rank, decoder, count = engine.search(
        b, assignment.copy(), rank.copy(), protocol['post_start_exact_evaluation_cap'],
        2, .005, 79, protocol['post_start_budget_ms']/1000)
    result = plan(engine, b, assignment, rank, decoder, cost)
    if result['cost'] > start['cost']:
        raise RuntimeError('hand proposer worsened strong start')
    result.update(scores=count, time_ms=(time.perf_counter()-tick)*1000)
    if count > protocol['post_start_exact_evaluation_cap'] or result['time_ms'] > protocol['post_start_budget_ms']:
        raise RuntimeError('hand proposer exceeded matched budget')
    return result


def broad_reference(engine, b, start, protocol):
    current = start
    passes = []
    for _ in range(protocol['oracle_block_descent_rounds']):
        tick = time.perf_counter()
        candidate, records, pool_size = scored_pool(engine, b, current, protocol['candidate_limit_per_family'])
        passes.append({'source_cost': current['cost'], 'best_cost': candidate['cost'],
                       'pool_size': pool_size, 'evaluated': len(records),
                       'elapsed_ms': (time.perf_counter()-tick)*1000,
                       'candidate_costs': records})
        if candidate['cost'] >= current['cost']:
            break
        current = candidate
    return current, passes


def case(engine, seed, protocol):
    b = problem(seed, *protocol['shape'])
    identity = hashlib.sha256(pack(b).tobytes()+b['predecessors'].tobytes()).hexdigest()
    start = strong_start(engine, b, protocol)
    hand = hand_proposer(engine, b, start, protocol)
    oracle, passes = broad_reference(engine, b, start, protocol)
    if oracle['cost'] > start['cost']:
        raise RuntimeError('bounded oracle worsened strong start')
    row = {'seed': seed, 'shape': protocol['shape'], 'identity': identity,
           'strong_start': start, 'hand': hand, 'oracle': oracle, 'passes': passes}
    return b, row


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(PROTOCOL)
    sources = [PROTOCOL, Path(__file__).resolve(),
               ROOT/'scripts_cosim/dag_block_stepwise_s0.py',
               ROOT/'src/placement/radical/dag_block_moves.py',
               ROOT/'src/placement/radical/dag_reservation.py',
               ROOT/'src/placement/radical/dag_reservation_bridge.py',
               ROOT/'src/placement/radical/dag_reservation.cpp',
               ROOT/'src/placement/radical/dag_reservation_bridge.cpp']
    save(out/'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources}})
    configure_environment()
    engine = BridgedDag(out/'build')
    save(out/'native.json', engine.provenance)
    identities = set()
    for seed in range(protocol['opened_calibration_seeds'][0], protocol['opened_calibration_seeds'][1]+1):
        b, row = case(engine, seed, protocol)
        if row['identity'] in identities:
            raise RuntimeError('duplicate physical workflow')
        identities.add(row['identity'])
        cell = out/'calibration'/str(seed)
        cell.mkdir(parents=True)
        save(cell/'input.json', serialized(b))
        save(cell/'read.json', row)
        print('CAL', seed, 'start', row['strong_start']['cost'], 'oracle', row['oracle']['cost'], flush=True)
    rows = []
    for seed in range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1]+1):
        b, row = case(engine, seed, protocol)
        if row['identity'] in identities:
            raise RuntimeError('duplicate physical workflow')
        identities.add(row['identity'])
        cell = out/'fresh'/str(seed)
        cell.mkdir(parents=True)
        save(cell/'input.json', serialized(b))
        save(cell/'read.json', row)
        rows.append(row)
        gain = 100*(row['strong_start']['cost']-row['oracle']['cost'])/row['strong_start']['cost']
        print('FRESH', seed, 'oracle_gain_pct', round(gain, 3),
              'hand', row['hand']['cost'], 'pool', row['passes'][0]['pool_size'], flush=True)
    gains = [100*(r['strong_start']['cost']-r['oracle']['cost'])/r['strong_start']['cost'] for r in rows]
    captures = [(r['strong_start']['cost']-r['hand']['cost'])/(r['strong_start']['cost']-r['oracle']['cost'])
                if r['strong_start']['cost'] > r['oracle']['cost'] else 1.0 for r in rows]
    report = {'contract': 'dag_block_proposer_128_s0_v1', 'fresh_cases': len(rows),
              'shape': protocol['shape'], 'median_oracle_gain_pct': float(np.median(gains)),
              'median_hand_capture': float(np.median(captures)),
              'oracle_gain_pct': gains, 'hand_capture': captures,
              'headroom_pass': bool(np.median(gains) >= protocol['headroom_bar_pct']),
              'hardness_pass': bool(np.median(captures) < protocol['hand_capture_ceiling']),
              'total_exact_block_evaluations': sum(sum(p['evaluated'] for p in r['passes']) for r in rows)}
    save(out/'read_before_live.json', report)
    first = protocol['opened_calibration_seeds'][0]
    b = problem(first, *protocol['shape'])
    calibration_cell = out/'calibration'/str(first)
    calibration = read(calibration_cell/'read.json')
    live(engine, b, calibration['strong_start'], calibration_cell, 'strong_start')
    for row in rows:
        b = problem(row['seed'], *protocol['shape'])
        cell = out/'fresh'/str(row['seed'])
        for name in ('strong_start', 'hand', 'oracle'):
            live(engine, b, row[name], cell, name)
        print('LIVE', row['seed'], flush=True)
    report['live_runs'] = 1+3*len(rows)
    report['live_operations'] = report['live_runs']*protocol['shape'][0]*protocol['shape'][1]
    save(out/'read.json', report)
    for name, digest in read(out/'protocol_before_run.json')['sources'].items():
        if sha(ROOT/name) != digest:
            raise RuntimeError('source changed during gate: '+name)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)

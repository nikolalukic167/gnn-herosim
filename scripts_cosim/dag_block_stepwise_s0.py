"""CPU-only, fresh DAG block/reservation proposal qualification."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.placement.radical.dag_block_moves import apply_proposal, candidate_pool
from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag, baseline, chronological
from src.placement.radical.environment import pack, serialized
from scripts_cosim.dag_reservation_screen import live, plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment

PROTOCOL = ROOT / 'experiments/dag_block_stepwise_s0_v1_corrected.json'


def scored_pool(engine, b, incumbent, per_family, deadline=None):
    assignment = np.asarray(incumbent['assignment'], dtype=np.int64)
    rank = np.asarray(incumbent['priority'], dtype=np.int64)
    starts = np.asarray(incumbent['starts'])
    if incumbent['decoder'] == 0:
        rank = chronological(rank, starts)
        translated = plan(engine, b, assignment, rank, 1)
        if translated['cost'] > incumbent['cost']:
            raise RuntimeError('non-delay to reservation translation worsened schedule')
        incumbent = min((incumbent, translated), key=lambda item: item['cost'])
        starts = np.asarray(translated['starts'])
    pool = candidate_pool(b, assignment, rank, starts, per_family)
    best = incumbent
    records = []
    for proposal in pool:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        a, r = apply_proposal(b, assignment, rank, proposal)
        cost, _, _ = engine.plan(b, a, r, proposal.decoder)
        records.append({'family': proposal.family, 'operations': list(proposal.operations),
                        'anchor': proposal.anchor, 'decoder': proposal.decoder,
                        'flip_hosts': proposal.flip_hosts, 'cost': cost})
        if cost < best['cost']:
            best = plan(engine, b, a, r, proposal.decoder, cost)
    return best, records, len(pool)


def block_descent(engine, b, incumbent, per_family, rounds=3):
    path = []
    for _ in range(rounds):
        candidate, records, total = scored_pool(engine, b, incumbent, per_family)
        path.append({'evaluated': len(records), 'pool': total, 'cost': candidate['cost']})
        if candidate['cost'] >= incumbent['cost']:
            break
        incumbent = candidate
    return incumbent, path


def case(engine, seed, shape, protocol):
    b = problem(seed, *shape)
    identity = hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes()).hexdigest()
    tick = time.perf_counter()
    cost, a, rank, decoder, rule = baseline(engine, b)
    base = {**plan(engine, b, a, rank, decoder, cost), 'rule': rule}
    base_ms = (time.perf_counter() - tick) * 1000
    cost, a, rank, decoder, count = engine.search(
        b, a.copy(), rank.copy(), 2**31 - 1, 2, .005, 77,
        max(0, (protocol['budget_ms'] - base_ms - 8) / 1000))
    hand = {**plan(engine, b, a, rank, decoder, cost), 'scores': count,
            'time_ms': (time.perf_counter() - tick) * 1000}
    pool_tick = time.perf_counter()
    broad, candidates, pool_size = scored_pool(engine, b, base, protocol['candidate_limit_per_family'])
    broad_ms = (time.perf_counter() - pool_tick) * 1000
    budget_tick = time.perf_counter()
    deadline = budget_tick + max(0, (protocol['budget_ms'] - base_ms - 8) / 1000)
    budgeted, budget_records, _ = scored_pool(engine, b, base, protocol['candidate_limit_per_family'], deadline)
    budgeted['time_ms'] = base_ms + (time.perf_counter() - budget_tick) * 1000
    return b, {'seed': seed, 'shape': shape, 'identity': identity, 'baseline': base,
               'hand': hand, 'budgeted_block': budgeted, 'broad_block': broad,
               'candidate_costs': candidates, 'candidate_pool_size': pool_size,
               'budgeted_candidate_count': len(budget_records), 'baseline_ms': base_ms,
               'broad_pool_ms': broad_ms}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(PROTOCOL)
    source_paths = [PROTOCOL, Path(__file__).resolve(),
                    ROOT / 'src/placement/radical/dag_block_moves.py',
                    ROOT / 'src/placement/radical/dag_reservation.py',
                    ROOT / 'src/placement/radical/dag_reservation_bridge.py',
                    ROOT / 'src/placement/radical/dag_reservation.cpp',
                    ROOT / 'src/placement/radical/dag_reservation_bridge.cpp']
    save(out / 'protocol_before_run.json', {'protocol': protocol,
         'sources': {str(p.relative_to(ROOT)): sha(p) for p in source_paths}})
    engine = BridgedDag(out / 'build')
    save(out / 'native.json', engine.provenance)
    configure_environment()
    calibrations = []
    identities = set()
    for shape in protocol['shapes']:
        for seed in range(protocol['calibration_seeds'][0], protocol['calibration_seeds'][1] + 1):
            b, row = case(engine, seed, shape, protocol)
            assert row['identity'] not in identities
            identities.add(row['identity'])
            cell = out / 'calibration' / f'{shape[0]}x{shape[1]}_{seed}'
            cell.mkdir(parents=True)
            save(cell / 'input.json', serialized(b))
            save(cell / 'read.json', row)
            calibrations.append(row)
            print('CAL', shape, seed, row['baseline_ms'], row['broad_pool_ms'], flush=True)
    selected = max(protocol['shapes'], key=lambda x: x[0] * x[1])
    save(out / 'calibration.json', {'selected_shape': selected, 'rows': calibrations})
    rows = []
    for seed in range(protocol['fresh_seeds'][0], protocol['fresh_seeds'][1] + 1):
        b, row = case(engine, seed, selected, protocol)
        assert row['identity'] not in identities
        identities.add(row['identity'])
        descent, descent_path = block_descent(engine, b, row['broad_block'], protocol['candidate_limit_per_family'])
        row['block_descent'] = descent
        row['block_descent_path'] = descent_path
        refs = []
        for origin in ('baseline', 'hand', 'block_descent'):
            source = row[origin]
            c, a, rank, decoder, count = engine.search(
                b, np.asarray(source['assignment']), np.asarray(source['priority']),
                protocol['reference_steps'], 2, .005, 77)
            refs.append({**plan(engine, b, a, rank, decoder, c), 'origin': origin, 'scores': count})
        best = min([row['baseline'], row['hand'], row['budgeted_block'], row['broad_block'], descent, *refs],
                   key=lambda item: item['cost'])
        row['references'] = refs
        row['reference'] = best
        cell = out / 'fresh' / str(seed)
        cell.mkdir(parents=True)
        save(cell / 'input.json', serialized(b))
        save(cell / 'read.json', row)
        rows.append(row)
        print('FRESH', seed, 'headroom', round(100 * (row['hand']['cost'] - best['cost']) / row['hand']['cost'], 3), flush=True)
    gains = np.array([(r['hand']['cost'] - r['reference']['cost']) / r['hand']['cost'] for r in rows])
    base_gains = np.array([r['baseline']['cost'] - r['reference']['cost'] for r in rows])
    capture = np.array([(r['baseline']['cost'] - r['hand']['cost']) / gap if gap > 0 else 1.
                        for r, gap in zip(rows, base_gains)])
    report = {'shape': selected, 'fresh_cases': len(rows), 'median_reference_headroom_pct': float(np.median(gains) * 100),
              'median_hand_capture': float(np.median(capture)),
              'median_block_gain_over_hand_pct': float(np.median([(r['hand']['cost'] - min(r['hand']['cost'], r['broad_block']['cost'])) / r['hand']['cost'] * 100 for r in rows])),
              'headroom_pass': bool(np.median(gains) * 100 >= protocol['headroom_bar_pct']),
              'hardness_pass': bool(np.median(capture) < protocol['hand_capture_ceiling'])}
    save(out / 'read_before_live.json', report)
    for row in calibrations:
        b = problem(row['seed'], *row['shape'])
        cell = out / 'calibration' / f"{row['shape'][0]}x{row['shape'][1]}_{row['seed']}"
        if row['seed'] == protocol['calibration_seeds'][0]:
            live(engine, b, row['baseline'], cell, 'baseline')
    for row in rows:
        b = problem(row['seed'], *selected)
        cell = out / 'fresh' / str(row['seed'])
        for name in ('hand', 'broad_block', 'reference'):
            live(engine, b, row[name], cell, name)
    report['live_runs'] = len(protocol['shapes']) + 3 * len(rows)
    save(out / 'read.json', report)
    for path, digest in read(out / 'protocol_before_run.json')['sources'].items():
        assert sha(ROOT / path) == digest, path
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.out)

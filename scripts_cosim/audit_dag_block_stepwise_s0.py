"""Independent artifact, candidate-cost and live replay audit of the DAG block screen."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.placement.radical.dag_reservation import problem
from src.placement.radical.dag_reservation_bridge import BridgedDag
from src.placement.radical.environment import pack, serialized
from scripts_cosim.dag_block_stepwise_s0 import scored_pool
from scripts_cosim.dag_reservation_screen import verify_live, verify_plan
from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha


def audit(out):
    registration = read(out / 'protocol_before_run.json')
    protocol = registration['protocol']
    for name, digest in registration['sources'].items():
        assert sha(ROOT / name) == digest, name
    engine = BridgedDag(out / 'build')
    assert engine.provenance == read(out / 'native.json')
    calibration = read(out / 'calibration.json')
    assert calibration['selected_shape'] == protocol['shapes'][-1]
    seen = set()
    runs = operations = candidates = 0
    fresh = []
    for root in ('calibration', 'fresh'):
        for cell in sorted((out / root).iterdir()):
            row = read(cell / 'read.json')
            b = problem(row['seed'], *row['shape'])
            assert read(cell / 'input.json') == serialized(b)
            identity = hashlib.sha256(pack(b).tobytes() + b['predecessors'].tobytes()).hexdigest()
            assert row['identity'] == identity and identity not in seen
            seen.add(identity)
            for name in ('baseline', 'hand', 'budgeted_block', 'broad_block'):
                verify_plan(engine, b, row[name])
            broad, records, size = scored_pool(engine, b, row['baseline'], protocol['candidate_limit_per_family'])
            assert broad['cost'] == row['broad_block']['cost']
            assert records == row['candidate_costs'] and size == row['candidate_pool_size']
            candidates += len(records)
            if root == 'fresh':
                fresh.append(row)
                verify_plan(engine, b, row['block_descent'])
                verify_plan(engine, b, row['reference'])
                for reference in row['references']:
                    verify_plan(engine, b, reference)
                best = min([row['baseline'], row['hand'], row['budgeted_block'],
                            row['broad_block'], row['block_descent'], *row['references']],
                           key=lambda p: p['cost'])
                assert row['reference']['cost'] == best['cost']
            for name in ('baseline', 'hand', 'broad_block', 'reference'):
                path = cell / f'{name}_live.json'
                if path.exists():
                    p = row[name]
                    verify_live(b, p, read(path))
                    assert read(cell / f'{name}_replay.json')['cost'] == p['cost']
                    runs += 1
                    operations += b['p'].shape[0] * b['p'].shape[1]
    assert len(fresh) == protocol['fresh_seeds'][1] - protocol['fresh_seeds'][0] + 1
    gains = np.array([(r['hand']['cost'] - r['reference']['cost']) / r['hand']['cost'] for r in fresh])
    capture = np.array([(r['baseline']['cost'] - r['hand']['cost']) /
                        (r['baseline']['cost'] - r['reference']['cost'])
                        if r['baseline']['cost'] > r['reference']['cost'] else 1 for r in fresh])
    report = read(out / 'read.json')
    assert np.median(gains) * 100 == report['median_reference_headroom_pct']
    assert np.median(capture) == report['median_hand_capture']
    assert report['live_runs'] == runs == len(protocol['shapes']) + 3 * len(fresh)
    result = {'status': 'PASS', 'inputs': len(seen), 'candidate_costs_rechecked': candidates,
              'live_runs': runs, 'live_operations': operations, 'report_sha256': sha(out / 'read.json')}
    save(out / 'AUDIT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out)

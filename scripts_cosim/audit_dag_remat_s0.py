"""Independent read-back audit of the corrected rematerialization live gate."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts_cosim.mixed_joint_dispatch_screen import ROOT, read, save, sha
from src.placement.radical.dag_remat import problem, replay


def audit(root):
    receipt = read(root / 'protocol_before_run.json')
    protocol = receipt['protocol']
    for path, fingerprint in receipt['sources'].items():
        assert sha(ROOT / path) == fingerprint, path
    report = read(root / 'read.json')
    assert report['live_runs'] == 49 and report['live_operations'] == 3136
    seeds = [protocol['calibration_seeds'][0], *range(protocol['fresh_seeds'][0],
                                                     protocol['fresh_seeds'][1] + 1)]
    identities = set()
    runs = operations = no_op_checks = 0
    for seed in seeds:
        group = 'calibration' if seed == seeds[0] else 'fresh'
        cell = root / group / str(seed)
        row = read(cell / 'read.json')
        b = problem(seed, *protocol['shape'])
        from scripts_cosim.dag_remat_s0 import identity
        assert identity(b) == row['identity'] and row['identity'] not in identities
        identities.add(row['identity'])
        a = np.array(row['assignment'])
        rank = np.array(row['priority'])
        plans = [('initial', row['simple'][row['initial_rule']])]
        if group == 'fresh':
            plans += [('hand', row['hand']), ('reference', row['reference'])]
        for name, plan in plans:
            expected = replay(b, a, rank, plan['retain'], plan['recompute'])
            assert expected['objective'] == plan['cost'] and expected['counts'] == plan['counts']
            actual = read(cell / f'{name}_live.json')
            assert actual['mixed_execution']['counts'] == expected['counts']
            assert abs(actual['job_completion_sum'] * 1000 - expected['objective']) < 1e-6
            assert len(actual['tasks']) == a.size
            for task in actual['tasks']:
                j, k = map(int, task['taskType']['name'][2:].split('_'))
                assert abs(task['doneTime'] * 1000 - expected['ends'][j, k]) < 1e-6
            if expected['counts']['rematerializations'] == 0 and np.any(plan['recompute']):
                absent = replay(b, a, rank, plan['retain'], np.zeros(a.shape, dtype=np.int8))
                assert absent['objective'] == expected['objective']
                no_op_checks += 1
            runs += 1
            operations += a.size
    assert runs == report['live_runs'] and operations == report['live_operations']
    result = {'cases': len(seeds), 'live_runs_audited': runs, 'operations_audited': operations,
              'no_op_recompute_plan_checks': no_op_checks,
              'source_and_identity_fingerprints': 'pass'}
    save(root / 'AUDIT.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)

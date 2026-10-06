"""Audit and summarize the frozen 6x6-to-larger L2D transfer gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / 'docs/lineages/l2d_size_transfer_v1/gate_freeze_2026-09-23.json'
ARMS = ('gnn', 'mpoff', 'mlp_t1')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_target(size: int, path: Path, freeze: dict) -> dict:
    data = json.loads(path.read_text())
    check(data['contract'] == 'l2d_size_transfer_pilot_v1', f'{size}: wrong contract')
    check(data['source'] == 6 and data['target'] == size, f'{size}: wrong size')
    check(data['instances'] == 100 and data['seeds'] == 8, f'{size}: wrong scope')
    check(data['upstream_commit'] == freeze['upstream_commit'], f'{size}: wrong upstream')
    test_path = Path(data['test_path'])
    check(test_path.name == f'test_{size}x{size}_seed301.npy', f'{size}: wrong test set')
    check(sha256(test_path) == data['test_sha256'], f'{size}: test hash mismatch')
    check(set(data['arms']) == set(ARMS), f'{size}: wrong arms')
    check(data['checkpoint_protocol'] == 'best_val, source-size validation only', f'{size}: wrong checkpoint selection')
    by_arm = {}
    for arm in ARMS:
        rows = data['arms'][arm]
        check(len(rows) == 8 and [row['seed'] for row in rows] == list(range(8)),
              f'{size} {arm}: missing seed')
        for row in rows:
            source = Path(freeze['source_map'][f"{arm}:{row['seed']}"])
            checkpoint = Path(row['checkpoint_path'])
            sidecar = Path(str(checkpoint) + '.contract.json')
            check(checkpoint.resolve() == source, f'{size} {arm}: wrong checkpoint')
            check(sha256(checkpoint) == row['checkpoint_sha256'] and
                  sha256(sidecar) == row['sidecar_sha256'], f'{size} {arm}: checkpoint changed')
            contract = json.loads(sidecar.read_text())
            check(contract['n_j'] == contract['n_m'] == 6 and contract['arm'] == arm and
                  contract['study_seed'] == row['seed'] and contract['selected_update'] == row['selected_update'],
                  f'{size} {arm}: source contract mismatch')
            values = row['makespans']
            check(len(values) == 100 and all(math.isfinite(x) and x > 0 for x in values),
                  f'{size} {arm}: invalid per-instance makespans')
            check(math.isclose(sum(values) / 100, row['mean_makespan'], abs_tol=1e-9),
                  f'{size} {arm}: mean mismatch')
        by_arm[arm] = [row['mean_makespan'] for row in rows]
    pdr = data['fdd_mwkr']
    check(len(pdr['makespans']) == 100 and all(math.isfinite(x) and x > 0 for x in pdr['makespans']),
          f'{size}: invalid FDD/MWKR result')
    check(math.isclose(sum(pdr['makespans']) / 100, pdr['mean_makespan'], abs_tol=1e-9),
          f'{size}: FDD/MWKR mean mismatch')
    comparisons = {}
    for arm in ('mpoff', 'mlp_t1'):
        gains = [(by_arm[arm][i] - by_arm['gnn'][i]) / by_arm[arm][i] * 100 for i in range(8)]
        comparisons[arm] = {'paired_gains_pct': gains, 'median_gain_pct': median(gains),
                            'positive_seeds': sum(gain > 0 for gain in gains),
                            'pass': median(gains) >= 1 and sum(gain > 0 for gain in gains) >= 7}
    return {'target': size, 'result_sha256': sha256(path), 'test_sha256': data['test_sha256'],
            'means_by_arm': by_arm, 'median_by_arm': {arm: median(means) for arm, means in by_arm.items()},
            'fdd_mwkr_mean': pdr['mean_makespan'], 'comparisons': comparisons}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target10', type=Path, required=True)
    parser.add_argument('--target15', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(FREEZE.read_text())
    for name, info in freeze['artifacts'].items():
        path = Path(name)
        check(path.stat().st_size == info['bytes'] and sha256(path) == info['sha256'],
              f'frozen artifact changed: {path}')
    upstream = Path('/root/projects/L2D')
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--short'], text=True).strip()
    check(commit == freeze['upstream_commit'] and not dirty, 'upstream code changed')
    targets = {str(size): audit_target(size, path, freeze)
               for size, path in ((10, args.target10), (15, args.target15))}
    passed = all(result['pass'] for target in targets.values()
                 for result in target['comparisons'].values())
    report = {'contract': 'l2d_size_transfer_v1_audit', 'status': 'pass',
              'freeze_sha256': sha256(FREEZE), 'models_evaluated': 48,
              'instances_per_model': 100, 'registered_pass': passed, 'targets': targets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'audit': 'pass', 'registered_pass': passed,
                      'comparisons': {size: target['comparisons'] for size, target in targets.items()}},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
